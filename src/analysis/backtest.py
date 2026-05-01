from __future__ import annotations

"""
Backtesting module: simulate betting strategies using recalibrated probabilities.
Computes cumulative PnL, Sharpe ratio, max drawdown, and Kelly-sized returns.
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit


def kelly_fraction(true_prob: float, market_price: float, direction: str = "yes") -> float:
    """
    Kelly criterion bet size as fraction of bankroll.
    direction='yes': buying YES shares at market_price
    direction='no':  buying NO shares at (1 - market_price)
    Returns 0 if no edge.
    """
    if direction == "yes":
        # odds b = (1 - market_price) / market_price (net profit per dollar risked)
        edge = true_prob - market_price
        if edge <= 0:
            return 0.0
        return edge / (1 - market_price)
    else:
        no_price = 1 - market_price
        edge = (1 - true_prob) - no_price
        if edge <= 0:
            return 0.0
        return edge / market_price


def run_backtest(
    df: pd.DataFrame,
    true_prob_col: str = "recalibrated_prob",
    market_prob_col: str = "predicted_prob",
    outcome_col: str = "outcome",
    strategy: str = "quarter_kelly",
    min_edge: float = 0.03,
    initial_bankroll: float = 1000.0,
    max_bet_fraction: float = 0.10,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate a betting strategy on the dataset.

    strategy options:
      'kelly'         — full Kelly sizing
      'quarter_kelly' — 25% Kelly (safer, common in practice)
      'half_kelly'    — 50% Kelly
      'flat'          — fixed 2% of initial bankroll per bet

    min_edge: minimum probability edge required to place a bet
    max_bet_fraction: cap bet at this fraction of current bankroll

    Returns (trade_log DataFrame, summary dict)
    """
    bankroll = initial_bankroll
    trades = []

    kelly_multiplier = {"kelly": 1.0, "quarter_kelly": 0.25, "half_kelly": 0.5}.get(strategy, 0.25)
    flat_bet = initial_bankroll * 0.02

    for _, row in df.iterrows():
        true_p = float(row[true_prob_col])
        market_p = float(row[market_prob_col])
        outcome = int(row[outcome_col])

        yes_edge = true_p - market_p
        no_edge = (1 - true_p) - (1 - market_p)

        if abs(yes_edge) >= abs(no_edge) and yes_edge >= min_edge:
            direction = "yes"
            edge = yes_edge
        elif no_edge >= min_edge:
            direction = "no"
            edge = no_edge
        else:
            continue

        if strategy == "flat":
            bet_size = min(flat_bet, bankroll * max_bet_fraction)
        else:
            kf = kelly_fraction(true_p, market_p, direction) * kelly_multiplier
            bet_size = min(kf * bankroll, bankroll * max_bet_fraction)

        bet_size = max(0.0, bet_size)
        if bet_size < 0.01:
            continue

        if direction == "yes":
            payout_if_win = bet_size * (1 - market_p) / market_p
            win = outcome == 1
        else:
            payout_if_win = bet_size * market_p / (1 - market_p)
            win = outcome == 0

        pnl = payout_if_win if win else -bet_size
        bankroll += pnl

        trades.append({
            "question": row.get("question", "")[:60],
            "category": row.get("category", ""),
            "direction": direction,
            "market_prob": round(market_p, 3),
            "true_prob": round(true_p, 3),
            "edge": round(edge, 3),
            "bet_size": round(bet_size, 2),
            "win": win,
            "pnl": round(pnl, 2),
            "bankroll": round(bankroll, 2),
        })

    trade_log = pd.DataFrame(trades)

    if trade_log.empty:
        return trade_log, {"error": "No bets placed — try lowering min_edge"}

    summary = _compute_summary(trade_log, initial_bankroll, strategy)
    return trade_log, summary


def _compute_summary(trades: pd.DataFrame, initial_bankroll: float, strategy: str) -> dict:
    n = len(trades)
    wins = trades["win"].sum()
    total_pnl = trades["pnl"].sum()
    final_bankroll = trades["bankroll"].iloc[-1]

    # Sharpe ratio (per-trade)
    returns = trades["pnl"] / initial_bankroll
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Max drawdown
    cumulative = trades["bankroll"].values
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_drawdown = float(drawdown.max())

    # ROI
    roi = (final_bankroll - initial_bankroll) / initial_bankroll

    return {
        "strategy": strategy,
        "n_bets": n,
        "win_rate": round(wins / n, 4),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 4),
        "final_bankroll": round(final_bankroll, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 4),
        "avg_edge": round(trades["edge"].mean(), 4),
        "avg_bet_pct": round((trades["bet_size"] / initial_bankroll).mean(), 4),
    }


def compare_strategies(
    df: pd.DataFrame,
    true_prob_col: str = "recalibrated_prob",
    market_prob_col: str = "predicted_prob",
    outcome_col: str = "outcome",
    min_edge: float = 0.03,
) -> pd.DataFrame:
    """Run all strategies and return a comparison DataFrame."""
    rows = []
    for strat in ["kelly", "half_kelly", "quarter_kelly", "flat"]:
        _, summary = run_backtest(df, true_prob_col, market_prob_col, outcome_col,
                                  strategy=strat, min_edge=min_edge)
        if "error" not in summary:
            rows.append(summary)
    return pd.DataFrame(rows).set_index("strategy")


def compute_cumulative_pnl(trade_log: pd.DataFrame, initial_bankroll: float = 1000.0) -> pd.DataFrame:
    """Return cumulative PnL series for plotting."""
    df = trade_log.copy().reset_index(drop=True)
    df["trade_num"] = range(1, len(df) + 1)
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["cumulative_roi"] = df["cumulative_pnl"] / initial_bankroll
    df["bankroll_normalized"] = df["bankroll"] / initial_bankroll
    return df
