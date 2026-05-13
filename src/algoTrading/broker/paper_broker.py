# =============================
# broker/paper_broker.py
# =============================
class PaperBroker:
    def __init__(self, capital):
        self.capital = capital
        self.position = 0
        self.trade_log = []

    def buy(self, price, qty):
        cost = price * qty
        if self.capital >= cost:
            self.capital -= cost
            self.position += qty
            self.trade_log.append(f"BUY {qty} @ {price}")

    def sell(self, price, qty):
        if self.position >= qty:
            self.capital += price * qty
            self.position -= qty
            self.trade_log.append(f"SELL {qty} @ {price}")
