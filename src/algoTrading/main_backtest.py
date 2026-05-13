import shutil
import numpy as np
import pandas as pd
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
    fetch_and_store(
        symbol=symbol,
        timeframe=Config.TIMEFRAME,
        bars=Config.BARS,
        save_path=ohlcv_abs,
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
            # Only copy reverse_exit flags that belong to this strategy's signals
            rev_mask = mask & (sig_df['reverse_exit'] == 1)
            merged.loc[rev_mask, 'reverse_exit'] = 1

    return merged


def run_combined_symbol(symbol: str, strategy_names: list) -> list:
    """
    Run all strategies on one symbol with a single shared capital pool.
    Signals are merged chronologically — first strategy to fire on a bar wins.
    Returns trade rows, each tagged with the originating strategy name.
    """
    ohlcv_rel = f"data/ohlcv_{symbol}.csv"
    df_base   = load_csv(ohlcv_rel)

    # Generate signals from every strategy independently
    sig_dfs = []
    total_signals = 0
    for name in strategy_names:
        sig_df = get_strategy(name).generate_signals(df_base)
        n = int((sig_df['signal'] != 0).sum())
        total_signals += n
        print(f"    [{name}] signals: {n}")
        sig_dfs.append(sig_df)

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

    print(f"\n{'='*60}")
    print(f"  BACKTEST  —  merged strategy execution")
    print(f"  Strategies : {', '.join(strategies)}")
    print(f"  Symbols    : {', '.join(symbols)}")
    print(f"  Timeframe  : {Config.TIMEFRAME}  |  Bars : {Config.BARS}")
    print(f"  Capital    : ${Config.INITIAL_CAPITAL} per symbol  (shared across strategies)")
    print(f"{'='*60}")

    # ── Connect MT5 ────────────────────────────────────────────────
    if not connect():
        return

    # ── Fetch all symbols once ─────────────────────────────────────
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
