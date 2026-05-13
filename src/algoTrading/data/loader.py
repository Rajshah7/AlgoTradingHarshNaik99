import pandas as pd
from pathlib import Path


def load_csv(relative_path):
    """
    Load CSV safely and normalize columns for strategy use
    """

    base_dir = Path(__file__).resolve().parents[1]
    print(f"base_dir: {base_dir}")

    full_path = base_dir / relative_path

    if not full_path.exists():
        raise FileNotFoundError(f"❌ File not found: {full_path}")

    df = pd.read_csv(full_path)

    # -------------------------
    # Normalize column names
    # -------------------------
    df.columns = [c.lower() for c in df.columns]

    # -------------------------
    # Required OHLC check
    # -------------------------
    required_cols = {'time', 'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ Missing required columns: {required_cols}")

    # -------------------------
    # Convert types
    # -------------------------
    df['time'] = pd.to_datetime(df['time'])

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # -------------------------
    # 🔥 FIX: Volume normalization
    # -------------------------
    if 'volume' not in df.columns:
        if 'tick_volume' in df.columns:
            df['volume'] = pd.to_numeric(df['tick_volume'], errors='coerce')
        elif 'real_volume' in df.columns:
            df['volume'] = pd.to_numeric(df['real_volume'], errors='coerce')
        else:
            raise ValueError("❌ No volume column found")

    # -------------------------
    # Optional: keep spread if exists
    # -------------------------
    if 'spread' in df.columns:
        df['spread'] = pd.to_numeric(df['spread'], errors='coerce')

    # -------------------------
    # Drop bad rows
    # -------------------------
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

    # -------------------------
    # Sort by time (important)
    # -------------------------
    df = df.sort_values('time').reset_index(drop=True)

    return df