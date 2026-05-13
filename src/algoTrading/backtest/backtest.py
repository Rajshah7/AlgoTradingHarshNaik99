# =============================
# backtest/backtest.py
# =============================
import pandas as pd
from strategies.moving_average import MovingAverageStrategy
from broker.paper_broker import PaperBroker


def run_backtest(data_path):
    data = pd.read_csv(data_path)

    strategy = MovingAverageStrategy()
    broker = PaperBroker(capital=1000)

    data = strategy.generate_signals(data)

    for i in range(1, len(data)):
        signal = data['signal'].iloc[i]
        price = data['close'].iloc[i]

        if signal == 1:
            broker.buy(price, qty=1)
        elif signal == -1:
            broker.sell(price, qty=1)

    print("Final Capital:", broker.capital)
    print("Trades:")
    for t in broker.trade_log:
        print(t)
