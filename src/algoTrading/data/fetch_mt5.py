import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path
import os

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

def fetch_and_store(symbol, timeframe, bars, save_path):

    print(f"Fetching {bars} bars for {symbol}...")

    tf = TIMEFRAME_MAP.get(timeframe)

    if tf is None:
        raise ValueError("❌ Invalid timeframe")

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

    if rates is None:
        print("❌ Error:", mt5.last_error())
        raise Exception("Failed to fetch MT5 data")

    if len(rates) == 0:
        raise Exception("❌ No data returned")

    print(f"✅ Received {len(rates)} bars")

    # ----------------------------
    # CONVERT TO DATAFRAME (ALL COLUMNS)
    # ----------------------------
    df = pd.DataFrame(rates)

    # Convert time properly
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 🔥 KEEP ALL COLUMNS (NO DROP)
    # ['time','open','high','low','close','tick_volume','spread','real_volume']

    # ----------------------------
    # DELETE OLD FILE (if exists)
    # ----------------------------
    if os.path.exists(save_path):
        print(f"⚠️ Existing file found. Deleting: {save_path}")
        os.remove(save_path)

    # ----------------------------
    # SAVE CSV
    # ----------------------------
    df.to_csv(save_path, index=False)

    print(f"✅ Data saved to {save_path}")