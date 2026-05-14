"""
fetch_kaggle.py — Download XAUUSD historical data from Kaggle (2004-2024)
and optionally append recent bars from MT5 (2024 → now).

Usage:
    cd src
    python -m algoTrading.data.fetch_kaggle            # Kaggle only
    python -m algoTrading.data.fetch_kaggle --merge-mt5 # Kaggle + MT5 top-up

After running, set MODE = "csv" in config.py and run main_backtest normally.
"""

import sys
import shutil
import argparse
import pandas as pd
from pathlib import Path

BASE     = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"

DATASET = "novandraanugrah/xauusd-gold-price-historical-data-2004-2024"
SYMBOL  = "XAUUSD"

# Maps Config.TIMEFRAME strings to keywords we look for in Kaggle filenames
TF_KEYWORDS = {
    "M1":  ["1min", "1_min", "m1", "1m", "minute1", "1minute"],
    "M5":  ["5min", "5_min", "m5", "5m", "minute5", "5minute"],
    "M15": ["15min", "15_min", "m15", "15m"],
    "M30": ["30min", "30_min", "m30", "30m"],
    "H1":  ["1hour", "1h", "h1", "60min", "hourly"],
    "H4":  ["4hour", "4h", "h4", "240min"],
    "D1":  ["daily", "1day", "d1", "1d", "day"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_csv(csv_files: list[Path], timeframe: str) -> Path:
    """Return the CSV that best matches the requested timeframe."""
    keywords = TF_KEYWORDS.get(timeframe.upper(), [])
    for kw in keywords:
        for f in csv_files:
            if kw in f.name.lower():
                return f
    # Fallback: pick the largest file (usually the most granular / complete)
    return max(csv_files, key=lambda f: f.stat().st_size)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any Kaggle OHLCV DataFrame to the standard schema."""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Time column
    for candidate in ["datetime", "date", "time", "timestamp", "open time"]:
        if candidate in df.columns:
            df["time"] = pd.to_datetime(df[candidate], utc=False, errors="coerce")
            break
    else:
        raise ValueError(f"No time column found. Columns: {list(df.columns)}")

    # Price columns — handle possible alternate names
    rename = {
        "open price": "open", "high price": "high",
        "low price": "low",   "close price": "close",
        "price": "close",
    }
    df.rename(columns=rename, inplace=True)

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'. Found: {list(df.columns)}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Volume — optional for gold data
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1)
    elif "tick_volume" in df.columns:
        df["volume"] = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(1)
    else:
        df["volume"] = 1

    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["time"] = df["time"].dt.tz_localize(None)   # strip tz if present
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df


def _merge_mt5(df_kaggle: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Fetch bars from MT5 starting where Kaggle ends and append them."""
    try:
        import MetaTrader5 as mt5
        from algoTrading.core.mt5_connector import connect, shutdown
        from algoTrading.data.fetch_mt5 import TIMEFRAME_MAP
    except ImportError:
        print("  ⚠️  MetaTrader5 not installed — skipping MT5 merge")
        return df_kaggle

    kaggle_end = df_kaggle["time"].max()
    print(f"  Kaggle data ends at : {kaggle_end}")
    print(f"  Connecting to MT5 to fetch bars from {kaggle_end.date()} → now ...")

    if not connect():
        print("  ❌ MT5 connect failed — using Kaggle data only")
        return df_kaggle

    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        print(f"  ❌ Unknown timeframe '{timeframe}' — using Kaggle data only")
        shutdown()
        return df_kaggle

    # Fetch a generous window — duplicates will be dropped on merge
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, 500_000)
    shutdown()

    if rates is None or len(rates) == 0:
        print("  ❌ MT5 returned no data — using Kaggle data only")
        return df_kaggle

    df_mt5 = pd.DataFrame(rates)
    df_mt5["time"] = pd.to_datetime(df_mt5["time"], unit="s")
    df_mt5 = df_mt5.rename(columns={"tick_volume": "volume"})

    if "volume" not in df_mt5.columns:
        df_mt5["volume"] = 1

    df_mt5 = df_mt5[["time", "open", "high", "low", "close", "volume"]]
    df_mt5 = df_mt5[df_mt5["time"] > kaggle_end]

    if df_mt5.empty:
        print("  ℹ️  No new MT5 bars beyond Kaggle range")
        return df_kaggle

    print(f"  MT5 contributed {len(df_mt5):,} bars "
          f"({df_mt5['time'].min().date()} → {df_mt5['time'].max().date()})")

    combined = pd.concat([df_kaggle, df_mt5], ignore_index=True)
    combined = combined.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download XAUUSD Kaggle dataset")
    parser.add_argument(
        "--merge-mt5", action="store_true",
        help="Append MT5 bars from where Kaggle ends up to today"
    )
    parser.add_argument(
        "--timeframe", default=None,
        help="Override timeframe (default: reads Config.TIMEFRAME)"
    )
    args = parser.parse_args()

    from algoTrading.config import Config
    timeframe = args.timeframe or Config.TIMEFRAME

    # ── Step 1: Download ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Kaggle Dataset Download — XAUUSD {timeframe}")
    print(f"{'='*60}")

    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed. Run:  pip install kagglehub")
        sys.exit(1)

    print(f"Downloading: {DATASET}")
    path = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded to: {path}")

    # ── Step 2: Find the right CSV ────────────────────────────────────
    csv_files = sorted(path.glob("**/*.csv"))
    if not csv_files:
        print("❌ No CSV files found in downloaded dataset")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV file(s):")
    for f in csv_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}  ({size_mb:.1f} MB)")

    chosen = _pick_csv(csv_files, timeframe)
    print(f"\nUsing: {chosen.name}  (timeframe={timeframe})")

    # ── Step 3: Load + normalize ──────────────────────────────────────
    print("Loading and normalizing ...")
    raw = pd.read_csv(chosen)
    print(f"Raw shape: {raw.shape} | columns: {list(raw.columns)}")

    df = _normalize(raw)
    print(f"Normalized: {len(df):,} rows | "
          f"{df['time'].min().date()} → {df['time'].max().date()}")

    # ── Step 4: Optional MT5 merge ────────────────────────────────────
    if args.merge_mt5:
        print("\nMerging with MT5 for recent bars ...")
        df = _merge_mt5(df, timeframe)

    # ── Step 5: Save ──────────────────────────────────────────────────
    DATA_DIR.mkdir(exist_ok=True)

    out_csv  = DATA_DIR / f"ohlcv_{SYMBOL}.csv"
    samp_csv = DATA_DIR / "sample_data.csv"

    df.to_csv(out_csv,  index=False)
    shutil.copy2(out_csv, samp_csv)

    print(f"\n✅ Saved  : {out_csv}")
    print(f"✅ Copied : {samp_csv}")
    print(f"   Total rows : {len(df):,}")
    print(f"   Date range : {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"\nNext step → set  MODE = \"csv\"  in config.py, then run:")
    print(f"   python -m algoTrading.main_backtest")


if __name__ == "__main__":
    main()
