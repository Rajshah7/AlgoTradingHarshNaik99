import MetaTrader5 as mt5
from config import Config

class MT5Broker:

    def __init__(self):
        self.symbol = Config.SYMBOL
        self.lot = Config.LOT_SIZE

    def place_order(self, action):
        tick = mt5.symbol_info_tick(self.symbol)

        if tick is None:
            print("❌ Symbol not found")
            return

        price = tick.ask if action == "BUY" else tick.bid

        sl = price - Config.STOP_LOSS * 0.0001 if action == "BUY" else price + Config.STOP_LOSS * 0.0001
        tp = price + Config.TAKE_PROFIT * 0.0001 if action == "BUY" else price - Config.TAKE_PROFIT * 0.0001

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot,
            "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "MT5 Python Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        print("Order Result:", result)