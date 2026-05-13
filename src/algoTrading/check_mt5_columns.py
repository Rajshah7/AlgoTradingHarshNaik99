import MetaTrader5 as mt5
import pandas as pd

# ----------------------------
# CONNECT MT5
# ----------------------------
if not mt5.initialize():
    print("❌ MT5 Initialization Failed:", mt5.last_error())
    quit()

print("✅ MT5 Connected\n")

# ----------------------------
# FETCH DATA
# ----------------------------
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M15
bars = 10

rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

if rates is None:
    print("❌ Failed to fetch data:", mt5.last_error())
    mt5.shutdown()
    quit()

# ----------------------------
# CONVERT TO DATAFRAME
# ----------------------------
df = pd.DataFrame(rates)

# Convert time
df['time'] = pd.to_datetime(df['time'], unit='s')

# ----------------------------
# PRINT STRUCTURE
# ----------------------------
print("📊 Available Columns:\n")
print(df.columns.tolist())

print("\n📌 Sample Data:\n")
print(df.head())

# ----------------------------
# SHUTDOWN
# ----------------------------
mt5.shutdown()