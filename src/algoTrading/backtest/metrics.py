import pandas as pd
import numpy as np
from pathlib import Path


def _calc_streaks(profits):
    max_win = max_loss = cur = 0
    cur_type = None
    for p in profits:
        if p > 0:
            cur = (cur + 1) if cur_type == 'win' else 1
            cur_type = 'win'
            max_win = max(max_win, cur)
        elif p < 0:
            cur = (cur + 1) if cur_type == 'loss' else 1
            cur_type = 'loss'
            max_loss = max(max_loss, cur)
    return max_win, max_loss, cur, cur_type or 'none'


def analyze_trades():

    base_dir  = Path(__file__).resolve().parents[1]
    file_path = base_dir / "data" / "trade_data.csv"

    if not file_path.exists():
        raise FileNotFoundError("❌ trade_data.csv not found")

    df = pd.read_csv(file_path)
    if df.empty:
        return {"error": "No data in trade file"}

    # ── Clean ────────────────────────────────────────────────────────
    df['profit']  = pd.to_numeric(df.get('profit'),  errors='coerce')
    df['capital'] = pd.to_numeric(df.get('capital'), errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['capital'])

    # ── Completed trades only ────────────────────────────────────────
    trades = df[df['type'].isin(['SELL', 'COVER'])].copy()
    trades = trades.dropna(subset=['profit', 'capital'])

    if trades.empty:
        return {"error": "No valid completed trades"}

    # ── Capital ──────────────────────────────────────────────────────
    starting_capital  = trades['capital'].iloc[0] - trades['profit'].iloc[0]
    ending_capital    = trades['capital'].iloc[-1]
    total_profit_val  = ending_capital - starting_capital
    profit_pct        = (total_profit_val / starting_capital) * 100

    # ── Trade metrics ────────────────────────────────────────────────
    profits_list = trades['profit'].tolist()
    total_trades = len(trades)

    wins   = trades[trades['profit'] > 0]
    losses = trades[trades['profit'] < 0]
    num_wins   = len(wins)
    num_losses = len(losses)
    win_rate   = (num_wins / total_trades) * 100 if total_trades else 0

    gross_win  = wins['profit'].sum()   if num_wins   else 0
    gross_loss = abs(losses['profit'].sum()) if num_losses else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0.0

    avg_win  = wins['profit'].mean()   if num_wins   else 0
    avg_loss = losses['profit'].mean() if num_losses else 0  # negative

    best_trade  = trades['profit'].max()
    worst_trade = trades['profit'].min()
    avg_profit  = trades['profit'].mean()

    # Expectancy: expected P&L per trade
    expectancy = (
        (win_rate / 100) * avg_win +
        ((1 - win_rate / 100) * avg_loss)
    ) if total_trades else 0

    # ── Drawdown ─────────────────────────────────────────────────────
    equity   = trades['capital']
    peak     = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    # Recovery factor: total profit / max drawdown amount
    max_dd_amount = (peak * abs(drawdown)).max()
    recovery_factor = total_profit_val / max_dd_amount if max_dd_amount > 0 else 0

    # ── Sharpe (trade-by-trade) ───────────────────────────────────────
    pnl_std = trades['profit'].std()
    sharpe  = (avg_profit / pnl_std) if pnl_std > 0 else 0

    # ── Streaks ──────────────────────────────────────────────────────
    max_ws, max_ls, cur_s, cur_t = _calc_streaks(profits_list)

    # ── Output ───────────────────────────────────────────────────────
    return {
        # Capital
        "starting_capital":  round(float(starting_capital), 2),
        "ending_capital":    round(float(ending_capital),   2),
        "total_profit":      round(float(total_profit_val), 2),
        "profit (%)":        round(float(profit_pct),       2),

        # Trade counts
        "total_trades":      int(total_trades),
        "wins":              int(num_wins),
        "losses":            int(num_losses),

        # Rates & ratios
        "win_rate (%)":      round(float(win_rate),       2),
        "profit_factor":     round(float(profit_factor),  2),
        "sharpe_ratio":      round(float(sharpe),         3),
        "recovery_factor":   round(float(recovery_factor),2),

        # Per-trade stats
        "avg_profit":        round(float(avg_profit),     2),
        "avg_win":           round(float(avg_win),        2),
        "avg_loss":          round(float(avg_loss),       2),
        "best_trade":        round(float(best_trade),     2),
        "worst_trade":       round(float(worst_trade),    2),
        "expectancy":        round(float(expectancy),     2),

        # Risk
        "max_drawdown (%)":  round(float(max_drawdown * 100), 2),

        # Streaks
        "max_win_streak":    int(max_ws),
        "max_loss_streak":   int(max_ls),
        "current_streak":    f"{cur_s} {cur_t}",
    }
