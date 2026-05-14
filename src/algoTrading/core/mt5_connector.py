import MetaTrader5 as mt5
from algoTrading.config import Config

def connect():
    login    = getattr(Config, 'MT5_LOGIN',    None)
    password = getattr(Config, 'MT5_PASSWORD', None)
    server   = getattr(Config, 'MT5_SERVER',   None)

    if login and password and server:
        ok = mt5.initialize(login=int(login), password=str(password), server=str(server))
        print(f"  Connecting as account {login} on {server} ...")
    else:
        ok = mt5.initialize()

    if not ok:
        print(f"❌ MT5 Initialization Failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info:
        print(f"✅ MT5 Connected — {info.name} | {info.server} | Balance: {info.balance} {info.currency}")
    else:
        print("✅ MT5 Connected")
    return True


def shutdown():
    mt5.shutdown()

def get_available_symbols(limit=20):

    # Fetch available symbols from MT5

    symbols = mt5.symbols_get()

    if symbols is None:
        print("❌ symbols_get() failed:", mt5.last_error())
        return []

    symbol_names = [s.name for s in symbols]

    print(f"Total Symbols: {len(symbol_names)}")

    return symbol_names[:limit]