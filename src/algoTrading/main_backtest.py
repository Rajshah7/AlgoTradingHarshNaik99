import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from algoTrading.core.mt5_connector import connect, shutdown
from algoTrading.dashboard import build_dashboard
from algoTrading.data.fetch_mt5 import fetch_and_store
from algoTrading.data.loader import load_csv
from algoTrading.backtest.engine import BacktestEngine
from algoTrading.backtest.metrics import analyze_trades
from algoTrading.config import Config

from algoTrading.strategies.moving_average import MovingAverageStrategy
from algoTrading.strategies.supertrend_strategy import SupertrendStrategy
from algoTrading.strategies.engulfing_strategy import EngulfingStrategy
from algoTrading.strategies.green_dollar import GreenDollarStrategy
from algoTrading.strategies.engulfing_consolidation import EngulfingConsolidationStrategy
from algoTrading.strategies.mark2_strategy import Mark2Strategy
from algoTrading.strategies.engulfing_reversal import EngulfingReversalStrategy
from algoTrading.strategies.mark_dollar_supertrend import MarkDollarSuperTrendStrategy
from algoTrading.strategies.rsi_engulfing_strategy import RSIEngulfingStrategy
from algoTrading.strategies.mark5_supertrend import Mark5SupertrendStrategy

BASE = Path(__file__).resolve().parent

STRATEGY_MAP = {
    "engulfing":               EngulfingStrategy,
    "green_dollar":            GreenDollarStrategy,
    "ma":                      MovingAverageStrategy,
    "supertrend":              SupertrendStrategy,
    "engulfing_consolidation": EngulfingConsolidationStrategy,
    "engulfing_reversal":      EngulfingReversalStrategy,
    "mark2":                   Mark2Strategy,
    "mark_dollar_supertrend":  MarkDollarSuperTrendStrategy,
    "rsi_engulfing":           RSIEngulfingStrategy,
    "mark5_supertrend":        Mark5SupertrendStrategy,
}


def get_strategy(name: str):
    cls = STRATEGY_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {list(STRATEGY_MAP)}")
    return cls()


def fetch_symbol(symbol: str, is_first: bool):
    """Fetch OHLCV from MT5, save to ohlcv_{symbol}.csv. First symbol also → sample_data.csv."""
    ohlcv_abs  = str(BASE / f"data/ohlcv_{symbol}.csv")
    sample_csv = BASE / "data" / "sample_data.csv"

    start_date = getattr(Config, 'START_DATE', None)
    from_date  = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None

    fetch_and_store(
        symbol=symbol,
        timeframe=Config.TIMEFRAME,
        bars=getattr(Config, 'BARS', 200000),
        save_path=ohlcv_abs,
        from_date=from_date,
    )
    if is_first:
        shutil.copy2(ohlcv_abs, str(sample_csv))
        print(f"  Chart data → sample_data.csv")


def _merge_signals(df_base: pd.DataFrame, sig_dfs: list, strategy_names: list) -> pd.DataFrame:
    """
    Merge signal DataFrames from multiple strategies into a single timeline.

    Rules:
      - OHLCV columns come from df_base (authoritative).
      - For each bar, the first strategy (in config order) with a non-zero
        signal wins — later strategies are skipped for that bar.
      - The winning strategy's sl, tp, lot, and reverse_exit are used.
      - A '_strategy' column records which strategy owns each signal bar.
    """
    merged = df_base.copy()
    merged['signal']       = 0
    merged['sl']           = np.nan
    merged['tp']           = np.nan
    merged['lot']          = 0.0
    merged['reverse_exit'] = 0
    merged['_strategy']    = ''

    for sig_df, name in zip(sig_dfs, strategy_names):
        # Only fill bars that haven't been claimed yet
        free = merged['signal'] == 0
        has_signal = sig_df['signal'] != 0
        mask = free & has_signal

        merged.loc[mask, 'signal']    = sig_df.loc[mask, 'signal']
        merged.loc[mask, 'sl']        = sig_df.loc[mask, 'sl']
        merged.loc[mask, 'tp']        = sig_df.loc[mask, 'tp']
        merged.loc[mask, 'lot']       = sig_df.loc[mask, 'lot']
        merged.loc[mask, '_strategy'] = name

        if 'reverse_exit' in sig_df.columns:
            rev_mask = mask & (sig_df['reverse_exit'] == 1)
            merged.loc[rev_mask, 'reverse_exit'] = 1

    return merged


def run_combined_symbol(symbol: str, strategy_names: list) -> list:
    """
    Run all strategies on one symbol with a single shared capital pool.
    Signals are merged chronologically — first strategy to fire on a bar wins.
    Returns trade rows, each tagged with the originating strategy name.
    """
    data_path = getattr(Config, 'DATA_PATH', None)
    ohlcv_rel = data_path if data_path else f"data/ohlcv_{symbol}.csv"
    df_base   = load_csv(ohlcv_rel)

    start_date = getattr(Config, 'START_DATE', None)
    end_date   = getattr(Config, 'END_DATE',   None)
    if start_date is not None:
        cutoff  = pd.Timestamp(start_date)
        df_base = df_base[df_base['time'] >= cutoff].reset_index(drop=True)
        print(f"    START_DATE={start_date} → {len(df_base):,} bars from {cutoff.date()} onwards")
    if end_date is not None:
        end_ts  = pd.Timestamp(end_date)
        df_base = df_base[df_base['time'] <= end_ts].reset_index(drop=True)
        print(f"    END_DATE={end_date} → {len(df_base):,} bars up to {end_ts.date()}")

    # Generate signals from every strategy independently
    sig_dfs = []
    total_signals = 0
    all_skips = []
    for name in strategy_names:
        sig_df = get_strategy(name).generate_signals(df_base)
        n = int((sig_df['signal'] != 0).sum())
        total_signals += n
        print(f"    [{name}] signals: {n}")
        sig_dfs.append(sig_df)
        if 'skip' in sig_df.columns:
            skipped = sig_df[sig_df['skip'] == 'SKIP'][['time']].copy()
            n_skip = len(skipped)
            if n_skip:
                print(f"    [{name}] skipped : {n_skip}")
            all_skips.append(skipped)

    # Save skip markers for dashboard chart
    if all_skips:
        skip_df = pd.concat(all_skips, ignore_index=True)
        skip_path = BASE / "data" / "skip_data.csv"
        skip_df.to_csv(skip_path, index=False)

    # Merge into one timeline
    merged = _merge_signals(df_base, sig_dfs, strategy_names)
    combined_signals = int((merged['signal'] != 0).sum())
    print(f"    Combined unique signals : {combined_signals}  (out of {total_signals} raw)")

    # Single engine — one capital account for all strategies
    engine = BacktestEngine(
        capital=Config.INITIAL_CAPITAL,
        risk_per_trade=Config.RISK_PER_TRADE,
        symbol=symbol,
    )
    result = engine.run(merged, save=False)

    wins   = sum(1 for t in engine.trades if t.get("type") in ("SELL", "COVER") and t.get("profit", 0) > 0)
    losses = sum(1 for t in engine.trades if t.get("type") in ("SELL", "COVER") and t.get("profit", 0) <= 0)
    pnl    = result["final_capital"] - Config.INITIAL_CAPITAL

    print(f"    Trades  : {result['total_trades']}  |  W {wins}  L {losses}")
    print(f"    P&L     : {'+' if pnl >= 0 else ''}{pnl:.2f}  ({result['return (%)']:+.2f}%)")
    print(f"    Capital : {Config.INITIAL_CAPITAL} → {result['final_capital']}")

    return engine.trades


def main():
    symbols    = [s.strip() for s in Config.SYMBOL.split(",")   if s.strip()]
    strategies = [s.strip() for s in Config.STRATEGY.split(",") if s.strip()]

    mode = getattr(Config, "MODE", "mt5").lower()
    csv_mode = mode in ("csv", "kaggle")

    print(f"\n{'='*60}")
    print(f"  BACKTEST  —  merged strategy execution")
    print(f"  Mode       : {'CSV / Kaggle (offline)' if csv_mode else 'MT5 (live fetch)'}")
    print(f"  Strategies : {', '.join(strategies)}")
    print(f"  Symbols    : {', '.join(symbols)}")
    print(f"  Timeframe  : {Config.TIMEFRAME}  |  Bars : {getattr(Config, 'BARS', 'N/A')}")
    print(f"  Capital    : ${Config.INITIAL_CAPITAL} per symbol  (shared across strategies)")
    print(f"{'='*60}")

    if csv_mode:
        # ── CSV / Kaggle mode — use existing files, no MT5 needed ──────
        data_path = getattr(Config, 'DATA_PATH', None)
        print(f"\n── CSV mode — using {'DATA_PATH' if data_path else 'ohlcv_{symbol}.csv'} ──")
        for symbol in symbols:
            csv_p = BASE / data_path if data_path else BASE / "data" / f"ohlcv_{symbol}.csv"
            if not csv_p.exists():
                hint = "check Config.DATA_PATH" if data_path else "run fetch_kaggle.py first"
                print(f"  ❌ {csv_p} not found — {hint}")
                return
            print(f"  ✅ {symbol} — {csv_p.name}")
    else:
        # ── MT5 mode — connect and fetch ───────────────────────────────
        if not connect():
            return
        print(f"\n── Fetching data ──────────────────────────────────────────")
        for i, symbol in enumerate(symbols):
            print(f"  {symbol}")
            fetch_symbol(symbol, is_first=(i == 0))
        shutdown()

    # ── Run each symbol with all strategies merged ─────────────────
    all_trades = []
    for symbol in symbols:
        print(f"\n── {symbol}  |  {Config.TIMEFRAME}  {'─'*40}")
        trades = run_combined_symbol(symbol, strategies)
        all_trades.extend(trades)

    # ── Save combined trade log (sorted by time) ───────────────────
    trade_path = BASE / "data" / "trade_data.csv"
    if not all_trades:
        print("\nNo trades executed.")
        return

    trade_df = pd.DataFrame(all_trades)
    trade_df['time'] = pd.to_datetime(trade_df['time'])
    trade_df = trade_df.sort_values('time').reset_index(drop=True)
    trade_df.to_csv(trade_path, index=False)
    print(f"\nSaved {len(all_trades)} trade rows → trade_data.csv  (sorted by time)")

    # ── Overall metrics ────────────────────────────────────────────
    metrics = analyze_trades()
    print("\n===== COMBINED PERFORMANCE METRICS =====")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # ── Dashboard ──────────────────────────────────────────────────
    build_dashboard()


if __name__ == "__main__":
    main()
