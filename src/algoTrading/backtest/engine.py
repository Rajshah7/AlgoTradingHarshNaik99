import pandas as pd
from pathlib import Path
from algoTrading.config import Config

class BacktestEngine:

    def __init__(self, capital, risk_per_trade, symbol=""):
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.symbol = symbol

        self.position = 0
        self.entry_price = None
        self.position_size = 0
        self.sl = None
        self.tp = None

        self.trades           = []
        self._tp_mode         = Config.TP_MODE
        self._open_strategy   = ""

        self._daily_losses = 0
        self._current_day  = None

    def run(self, df, save=True):

        df = df.reset_index(drop=True)
        max_daily_losses = getattr(Config, 'MAX_DAILY_LOSSES', None)

        for i in range(len(df)):
            row = df.iloc[i]

            signal = row.get('signal', 0)
            close  = row['close']
            time   = row['time']

            # Reset daily loss counter on new calendar day
            bar_day = pd.Timestamp(time).date()
            if bar_day != self._current_day:
                self._current_day  = bar_day
                self._daily_losses = 0

            day_blocked = (max_daily_losses is not None
                           and self._daily_losses >= max_daily_losses)

            # =========================
            # LONG ENTRY
            # =========================
            if signal == 1 and self.position == 0 and not day_blocked:

                sl = row.get('sl')
                tp = row.get('tp')

                if pd.isna(sl) or pd.isna(tp):
                    continue

                risk_per_unit = close - sl

                if risk_per_unit <= 0:
                    continue

                risk_amount = self.capital * self.risk_per_trade
                self.position_size = risk_amount / risk_per_unit

                self.entry_price      = close
                self.sl               = sl
                self.tp               = tp
                self.position         = 1
                self._open_strategy   = row.get('_strategy', '')

                self.trades.append({
                    "symbol":   self.symbol,
                    "strategy": self._open_strategy,
                    "time":     time,
                    "type":     "BUY",
                    "entry_price": close,
                    "lot_size":    self.position_size,
                    "sl":          self.sl,
                    "tp":          self.tp,
                    "capital":     self.capital,
                })

            # =========================
            # LONG EXIT
            # =========================
            elif self.position == 1:

                exit_trade = False
                exit_reason = None

                # reverse engulfing on candle immediately after entry
                if row.get('reverse_exit', 0) == 1:
                    exit_trade = True
                    exit_reason = "REV"

                # candle CLOSE below SL
                elif close < self.sl:
                    exit_trade = True
                    exit_reason = "SL"

                elif close >= self.tp:
                    exit_trade = True
                    exit_reason = "TP"

                if exit_trade:

                    exit_price = close
                    profit = (exit_price - self.entry_price) * self.position_size

                    self.capital += profit
                    self.capital = max(self.capital, 0)

                    if exit_reason == "TP":
                        close_label = "ST" if self._tp_mode == "st" else ("R:R" if self._tp_mode == "rr" else "TP")
                    elif exit_reason == "REV":
                        close_label = "REV"
                    else:
                        close_label = "SL"

                    self.trades.append({
                        "symbol":      self.symbol,
                        "strategy":    self._open_strategy,
                        "time":        time,
                        "type":        "SELL",
                        "entry_price": self.entry_price,
                        "exit_price":  exit_price,
                        "sl":          self.sl,
                        "tp":          self.tp,
                        "lot_size":    self.position_size,
                        "profit":      profit,
                        "exit_reason": exit_reason,
                        "exit_label":  close_label,
                        "capital":     self.capital,
                    })

                    if profit < 0:
                        self._daily_losses += 1

                    self.position = 0
                    self.entry_price = None
                    self.sl = None
                    self.tp = None

            # =========================
            # SHORT ENTRY
            # =========================
            elif signal == -1 and self.position == 0 and not day_blocked:

                sl = row.get('sl')
                tp = row.get('tp')

                if pd.isna(sl) or pd.isna(tp):
                    continue

                risk_per_unit = sl - close  # SL is above entry for short

                if risk_per_unit <= 0:
                    continue

                risk_amount = self.capital * self.risk_per_trade
                self.position_size = risk_amount / risk_per_unit

                self.entry_price    = close
                self.sl             = sl
                self.tp             = tp
                self.position       = -1
                self._open_strategy = row.get('_strategy', '')

                self.trades.append({
                    "symbol":   self.symbol,
                    "strategy": self._open_strategy,
                    "time":     time,
                    "type":     "SHORT",
                    "entry_price": close,
                    "lot_size":    self.position_size,
                    "sl":          self.sl,
                    "tp":          self.tp,
                    "capital":     self.capital,
                })

            # =========================
            # SHORT EXIT
            # =========================
            elif self.position == -1:

                exit_trade = False
                exit_reason = None

                # reverse engulfing on candle immediately after entry
                if row.get('reverse_exit', 0) == 1:
                    exit_trade = True
                    exit_reason = "REV"

                # candle CLOSE above SL high
                elif close > self.sl:
                    exit_trade = True
                    exit_reason = "SL"

                # price reached target
                elif close <= self.tp:
                    exit_trade = True
                    exit_reason = "TP"

                if exit_trade:

                    exit_price = close
                    profit = (self.entry_price - exit_price) * self.position_size

                    self.capital += profit
                    self.capital = max(self.capital, 0)

                    if exit_reason == "TP":
                        close_label = "ST" if self._tp_mode == "st" else ("R:R" if self._tp_mode == "rr" else "TP")
                    elif exit_reason == "REV":
                        close_label = "REV"
                    else:
                        close_label = "SL"

                    self.trades.append({
                        "symbol":      self.symbol,
                        "strategy":    self._open_strategy,
                        "time":        time,
                        "type":        "COVER",
                        "entry_price": self.entry_price,
                        "exit_price":  exit_price,
                        "sl":          self.sl,
                        "tp":          self.tp,
                        "lot_size":    self.position_size,
                        "profit":      profit,
                        "exit_reason": exit_reason,
                        "exit_label":  close_label,
                        "capital":     self.capital,
                    })

                    if profit < 0:
                        self._daily_losses += 1

                    self.position = 0
                    self.entry_price = None
                    self.sl = None
                    self.tp = None

            # stop if account gone
            if self.capital <= 0:
                print("❌ Account blown")
                break

        if save:
            self.save_trades()
        return self.results()

    def save_trades(self):
        if not self.trades:
            print("⚠️ No trades to save")
            return

        df = pd.DataFrame(self.trades)

        base_dir = Path(__file__).resolve().parents[1]
        file_path = base_dir / "data" / "trade_data.csv"

        df.to_csv(file_path, index=False)
        print(f"✅ Trades saved to {file_path}")

    def results(self):
        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(self.capital, 2),
            "return (%)": round(((self.capital - self.initial_capital) / self.initial_capital) * 100, 2),
            "total_trades": len(self.trades)
        }