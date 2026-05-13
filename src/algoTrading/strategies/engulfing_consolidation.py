import pandas as pd
import numpy as np
from pathlib import Path
from algoTrading.config import Config

_YAML_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

RSI_PERIOD    = 14
RSI_THRESHOLD = 70   # signal only when RSI was overbought on THIS or any of the PREV 3 bars
EMA_PERIOD    = 200


def _load_rr(strategy_key: str) -> float:
    try:
        import yaml
        with open(_YAML_PATH, "r") as f:
            data = yaml.safe_load(f)
        return float(data["strategies"][strategy_key]["rr"])
    except Exception:
        return float(Config.RR)


class EngulfingConsolidationStrategy:
    """
    SHORT-only strategy — Pine Script port.

    Signal = consolidation AND engulfingFlag AND rsi_recently_above_70

    Bearish engulfing (engulfingFlag):
      close[1] > open[1]                       prev bullish
      open > close                              curr bearish
      open >= close[1]                          gaps up / opens at prev close
      open[1] >= close                          fully engulfs prev body
      open - close > close[1] - open[1]        curr body larger than prev body

    Consolidation:
      close < close[1..4]                       closing below previous 4 bars
      high  > high[7]                           wick breaks above 7-bar range (failed breakout)

    RSI filter:
      RSI(14) crossed above 70 on this bar OR any of the 3 bars before it.
      Matches Pine Script "rsiAbove70 and engulfingFlag" logic — RSI doesn't
      have to be EXACTLY >70 on the engulfing bar, only recently overbought.

    Trade:
      Entry : close of signal bar
      SL    : max(high[1], high[2], high[3], high[4])  — top of consolidation zone
      TP    : entry - RR * risk
    """

    _STRATEGY_KEY = "engulfing_consolidation"

    def __init__(self):
        self.rr       = _load_rr(self._STRATEGY_KEY)
        self.lot_size = Config.LOT_SIZE

    # ── 200 EMA ───────────────────────────────────────────────────────────────
    def _compute_ema(self, close: pd.Series) -> pd.Series:
        return close.ewm(span=EMA_PERIOD, adjust=False).mean()

    # ── RSI (Wilder's RMA — matches Pine Script ta.rsi) ──────────────────────
    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        rs       = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # ── Pine Script: engulfingFlag ────────────────────────────────────────────
    def _is_bearish_engulfing(self, prev, curr) -> bool:
        return (
            prev['close'] > prev['open'] and                                 # close[1] > open[1]
            curr['open']  > curr['close'] and                                # open > close
            curr['open']  >= prev['close'] and                               # open >= close[1]
            prev['open']  >= curr['close'] and                               # open[1] >= close
            (curr['open'] - curr['close']) > (prev['close'] - prev['open'])  # bigger body
        )

    # ── Pine Script: consolidation ────────────────────────────────────────────
    def _consolidation(self, df: pd.DataFrame, i: int) -> bool:
        curr = df.iloc[i]
        return (
            curr['close'] < df.iloc[i - 1]['close'] and   # close < close[1]
            curr['close'] < df.iloc[i - 2]['close'] and   # close < close[2]
            curr['close'] < df.iloc[i - 3]['close'] and   # close < close[3]
            curr['close'] < df.iloc[i - 4]['close'] and   # close < close[4]
            curr['high']  > df.iloc[i - 7]['high']        # high  > high[7]
        )

    # ── RSI recently overbought (within last 3 bars) ──────────────────────────
    def _rsi_recently_above(self, rsi: pd.Series, i: int) -> bool:
        """
        TV's bgcolor fires when RSI > 70 on the SAME bar as engulfing.
        In practice the engulfing candle itself may have RSI slightly under 70
        because the close dropped — check the current bar AND the 3 prior bars.
        """
        for lookback in range(0, 4):   # i, i-1, i-2, i-3
            val = rsi.iloc[i - lookback]
            if pd.notna(val) and val > RSI_THRESHOLD:
                return True
        return False

    # ── Signal generation ─────────────────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['signal'] = 0
        df['sl']     = np.nan
        df['tp']     = np.nan
        df['lot']    = 0.0

        rsi = self._compute_rsi(df['close'])
        ema = self._compute_ema(df['close'])

        # start at 200 — ensures EMA(200) warmup + RSI(14) + consolidation lookback (7)
        for i in range(200, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]

            if (curr['close'] < ema.iloc[i]
                    and self._rsi_recently_above(rsi, i)
                    and self._is_bearish_engulfing(prev, curr)
                    and self._consolidation(df, i)):

                entry = curr['close']
                sl    = max(df.iloc[i - 1]['high'], df.iloc[i - 2]['high'],
                            df.iloc[i - 3]['high'], df.iloc[i - 4]['high'])
                risk  = sl - entry
                if risk > 0:
                    df.at[i, 'signal'] = -1
                    df.at[i, 'sl']     = sl
                    df.at[i, 'tp']     = entry - self.rr * risk
                    df.at[i, 'lot']    = self.lot_size

        return df
