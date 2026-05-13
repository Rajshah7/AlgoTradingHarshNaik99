import json
import time
import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path

TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

CACHE_TTL = 3600  # 1 hour in seconds


def _meta_path(save_path: str) -> Path:
    return Path(save_path).with_suffix(".meta")


def _cache_valid(save_path: str, timeframe: str) -> bool:
    """Return True if the cached CSV is < 1 hour old and was built for the same timeframe."""
    csv_p  = Path(save_path)
    meta_p = _meta_path(save_path)

    if not csv_p.exists() or not meta_p.exists():
        return False

    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        return False

    if meta.get("timeframe") != timeframe:
        return False

    age_seconds = time.time() - meta.get("cached_at", 0)
    return age_seconds < CACHE_TTL


def fetch_and_store(symbol, timeframe, bars, save_path):

    # ── Cache hit: skip MT5 fetch ─────────────────────────────────
    if _cache_valid(save_path, timeframe):
        meta    = json.loads(_meta_path(save_path).read_text())
        age_min = int((time.time() - meta["cached_at"]) / 60)
        remaining = int((CACHE_TTL - (time.time() - meta["cached_at"])) / 60)
        print(f"✅ Cache hit — {symbol} {timeframe} | cached {age_min}m ago | refreshes in ~{remaining}m")
        return

    # ── Cache miss: fetch from MT5 ────────────────────────────────
    print(f"🔄 Fetching {bars} bars for {symbol} {timeframe} from MT5...")

    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        raise ValueError(f"❌ Invalid timeframe: {timeframe!r}")

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

    if rates is None:
        print("❌ Error:", mt5.last_error())
        raise Exception("Failed to fetch MT5 data")

    if len(rates) == 0:
        raise Exception("❌ No data returned")

    print(f"✅ Received {len(rates)} bars")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Overwrite CSV
    csv_p = Path(save_path)
    if csv_p.exists():
        csv_p.unlink()

    df.to_csv(save_path, index=False)
    print(f"✅ Data saved to {save_path}")

    # Write / update sidecar metadata
    _meta_path(save_path).write_text(json.dumps({
        "symbol":    symbol,
        "timeframe": timeframe,
        "bars":      len(df),
        "cached_at": time.time(),
    }))
    print(f"📦 Cache written — next refresh in 1h")
