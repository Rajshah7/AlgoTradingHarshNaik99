import numpy as np
import pandas as pd
from algoTrading.config import Config
from algoTrading.strategies.mark2_strategy import Mark2Strategy


class DojiStrategy(Mark2Strategy):
    """
    Python port of the Pine Script "Doji strategy W".

    Entry rules (evaluated on every candle):
      A Doji is detected when:
          abs(open - close) <= (high - low) * doji_size

      Optional volume filter: volume > EMA(volume, volume_ma_period)

      On a valid Doji candle two OCA stop-orders are placed:

        BUY  STOP at long_entry:
            prev candle is "long" AND range[prev] > range[doji]
                → long_entry = high[prev]
            otherwise
                → long_entry = max(high[doji], high[prev])

        SELL STOP at short_entry:
            prev candle is "long" AND range[prev] > range[doji]
                → short_entry = low[prev]
            otherwise
                → short_entry = min(low[doji], low[prev])

      Whichever stop is triggered first becomes the live trade; the other
      is cancelled (OCA logic handled by the engine via the `oca_group` column).

    Exit:
      Fixed price-unit SL / TP pulled from Config:
          long  : sl = entry_price - Config.STOP_LOSS
                  tp = entry_price + Config.TAKE_PROFIT
          short : sl = entry_price + Config.STOP_LOSS
                  tp = entry_price - Config.TAKE_PROFIT

    Output columns (same contract as Mark2 / MarkDollarSuperTrend)
    ───────────────────────────────────────────────────────────────
    signal       :  1 = long,  -1 = short,  0 = none
    sl           :  stop-loss  price
    tp           :  take-profit price
    lot          :  position size (Config.LOT_SIZE via Mark2Strategy)
    entry_price  :  stop-order trigger price (long_entry or short_entry)
    oca_group    :  integer; both OCA legs share the same id (engine cancels
                   the unfilled leg once the other is triggered)
    doji         :  True on the originating doji candle row(s)

    FIX NOTES (vs. original):
    ─────────────────────────
    1. Each doji now emits TWO rows (one per OCA leg) instead of one row
       with buried aux columns — this matches the engine contract expected
       by Mark2Strategy / MarkDollarSuperTrendStrategy.
    2. signal = -1 is correctly assigned to the SHORT leg (was always 1).
    3. lot is populated on BOTH legs (was 0 on the short leg).
    4. Removed short_entry / short_sl / short_tp aux columns — redundant
       once the short leg has its own proper row.
    5. Volume EMA column is only accessed when use_volume_filter is True,
       avoiding a KeyError when the filter is disabled.
    6. The output DataFrame index is reset so duplicate index values
       (two rows per doji candle timestamp) don't confuse downstream code.
    """

    _STRATEGY_KEY = "doji_strategy"

    def __init__(
        self,
        doji_size: float = 0.05,
        long_candle_ratio: float = 0.7,
        use_volume_filter: bool = False,
        volume_ma_period: int = 24,
    ):
        """
        Parameters
        ----------
        doji_size         : body / range threshold          (Pine default 0.05)
        long_candle_ratio : min body/range for a "long" candle (Pine default 0.7)
        use_volume_filter : gate signals on volume > EMA(volume, volume_ma_period)
        volume_ma_period  : EMA span for the volume filter  (Pine default 24)

        The following are read directly from Config:
            Config.STOP_LOSS       – price-unit SL distance from entry
            Config.TAKE_PROFIT     – price-unit TP distance from entry
            Config.LOT_SIZE        – position size  (via Mark2Strategy.__init__)
            Config.MAX_CANDLE_SIZE – skip doji candles larger than this (or None)
        """
        super().__init__()  # provides self.lot_size, self.rr, self.calc_tp, etc.

        self.doji_size         = doji_size
        self.long_candle_ratio = long_candle_ratio
        self.use_volume_filter = use_volume_filter
        self.volume_ma_period  = volume_ma_period

        self.stop_loss       = Config.STOP_LOSS
        self.take_profit     = Config.TAKE_PROFIT
        self.max_candle_size = getattr(Config, "MAX_CANDLE_SIZE", None)

    # ── Candle-classification helpers ─────────────────────────────────

    def _candle_too_big(self, row) -> bool:
        """True if the candle's high-low range exceeds Config.MAX_CANDLE_SIZE."""
        if self.max_candle_size is None:
            return False
        return (row["high"] - row["low"]) > self.max_candle_size

    def _is_doji(self, row) -> bool:
        """True when body size <= doji_size * full range (Pine: data condition)."""
        rng = row["high"] - row["low"]
        if rng == 0:
            return False
        return abs(row["open"] - row["close"]) <= rng * self.doji_size

    def _is_long_candle(self, row) -> bool:
        """True when body / range > long_candle_ratio (Pine: longcandle)."""
        rng = row["high"] - row["low"]
        if rng == 0:
            return False
        return (abs(row["open"] - row["close"]) / rng) > self.long_candle_ratio

    # ── Volume EMA preparation ────────────────────────────────────────

    def _prepare_volume_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds _vol_ema column only when the volume filter is active."""
        # FIX 5: only compute when needed — avoids KeyError if 'volume'
        # column is absent and filter is disabled.
        if self.use_volume_filter:
            df = df.copy()
            df["_vol_ema"] = (
                df["volume"].ewm(span=self.volume_ma_period, adjust=False).mean()
            )
        return df

    # ── Stop-order entry-price helpers ────────────────────────────────

    @staticmethod
    def _long_entry_price(doji_row, prev_row, prev_is_long: bool) -> float:
        """
        BUY-stop trigger (Pine: longDist).
        prev long AND prev range > doji range → prev high.
        Otherwise                             → max(doji high, prev high).
        """
        if prev_is_long and (
            (prev_row["high"] - prev_row["low"]) > (doji_row["high"] - doji_row["low"])
        ):
            return prev_row["high"]
        return max(doji_row["high"], prev_row["high"])

    @staticmethod
    def _short_entry_price(doji_row, prev_row, prev_is_long: bool) -> float:
        """
        SELL-stop trigger (Pine: shortDist).
        prev long AND prev range > doji range → prev low.
        Otherwise                             → min(doji low, prev low).
        """
        if prev_is_long and (
            (prev_row["high"] - prev_row["low"]) > (doji_row["high"] - doji_row["low"])
        ):
            return prev_row["low"]
        return min(doji_row["low"], prev_row["low"])

    # ── Signal-row builder ────────────────────────────────────────────

    def _build_signal_row(
        self,
        base_row: pd.Series,
        signal: int,          #  1 = long,  -1 = short
        entry: float,
        oca_id: int,
    ) -> dict:
        """
        Return a dict representing one OCA leg row.

        FIX 1 & 2: produces a complete, standalone row for each leg so the
        engine sees signal = 1 for the long leg and signal = -1 for the
        short leg — both with the same oca_group id.
        """
        if signal == 1:        # LONG leg
            sl = entry - self.stop_loss
            tp = entry + self.take_profit
        else:                  # SHORT leg  (signal == -1)
            sl = entry + self.stop_loss
            tp = entry - self.take_profit

        row_dict = base_row.to_dict()   # carry all OHLCV / timestamp columns
        row_dict.update(
            {
                "signal":      signal,
                "entry_price": entry,
                "sl":          sl,
                "tp":          tp,
                "lot":         self.lot_size,   # FIX 3: lot on both legs
                "oca_group":   oca_id,
                "doji":        True,
            }
        )
        return row_dict

    # ── Main signal generation ────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Walk candle-by-candle and emit TWO signal rows for each qualifying
        doji — one for the LONG OCA leg (signal = 1) and one for the SHORT
        OCA leg (signal = -1).  Both rows carry the same oca_group id so
        the engine can cancel the unfilled leg once the first stop triggers.

        Flow:
            Doji detected?
              └─ Volume filter passes?  (if enabled)
                   └─ Candle not too big?
                        ├─ Emit LONG  row: signal=1,  entry=longDist,
                        │                  sl=entry-SL, tp=entry+TP
                        └─ Emit SHORT row: signal=-1, entry=shortDist,
                                           sl=entry+SL, tp=entry-TP

        Returns a NEW DataFrame (reset index) containing only signal rows
        (non-doji candles are dropped).  Downstream engines filter on
        signal != 0 already, so dropping zero rows is safe and efficient.
        """
        df = df.copy()
        df = self._prepare_volume_ema(df)

        signal_rows: list[dict] = []
        oca_id = 0

        for i in range(1, len(df)):
            row      = df.iloc[i]
            prev_row = df.iloc[i - 1]

            # ── Doji detection ────────────────────────────────────────
            if not self._is_doji(row):
                continue

            # ── Volume filter ─────────────────────────────────────────
            # FIX 5: only check _vol_ema when filter is active
            if self.use_volume_filter and row["volume"] <= row["_vol_ema"]:
                continue

            # ── Candle-size guard ─────────────────────────────────────
            if self._candle_too_big(row):
                continue

            oca_id      += 1
            prev_is_long = self._is_long_candle(prev_row)
            long_e       = self._long_entry_price(row, prev_row, prev_is_long)
            short_e      = self._short_entry_price(row, prev_row, prev_is_long)

            # FIX 1 & 2: emit one proper row per OCA leg
            signal_rows.append(
                self._build_signal_row(row, signal=1,  entry=long_e,  oca_id=oca_id)
            )
            signal_rows.append(
                self._build_signal_row(row, signal=-1, entry=short_e, oca_id=oca_id)
            )

        if not signal_rows:
            # Return empty DataFrame with correct columns
            cols = list(df.columns) + [
                "signal", "entry_price", "sl", "tp", "lot", "oca_group", "doji"
            ]
            return pd.DataFrame(columns=cols)

        # FIX 6: reset_index so duplicate timestamps don't cause issues
        result = pd.DataFrame(signal_rows).reset_index(drop=True)

        # Ensure correct dtypes
        result["signal"]    = result["signal"].astype(int)
        result["oca_group"] = result["oca_group"].astype(int)
        result["lot"]       = result["lot"].astype(float)
        result["doji"]      = result["doji"].astype(bool)

        return result