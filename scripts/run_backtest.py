"""
Run recalibration + backtesting pipeline.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --strategy quarter_kelly --min-edge 0.05
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from loguru import logger

from src.pipeline.ingest import get_conn, load_analysis_df
from src.pipeline.features import build_features
from src.analysis.recalibration import (
    fit_isotonic, cross_validate_calibrators, recalibration_reliability_data,
)
from src.analysis.backtest import run_backtest, compare_strategies, compute_cumulative_pnl
from src.analysis.calibration import expected_calibration_error, brier_score, reliability_data

FIGURES = "outputs/figures"
os.makedirs(FIGURES, exist_ok=True)


def plot_recalibration_comparison(rel_df, save_path):
    methods = rel_df["method"].unique()
    colors = {
        "predicted_prob": "#adb5bd",
        "isotonic": "#0066cc",
        "platt": "#2a9d8f",
        "temperature": "#e63946",
    }
    labels = {
        "predicted_prob": "Uncalibrated (market prices)",
        "isotonic": "Isotonic regression",
        "platt": "Platt scaling",
        "temperature": "Temperature scaling",
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, alpha=0.4, label="Perfect")

    for method in ["predicted_prob", "isotonic", "platt", "temperature"]:
        sub = rel_df[rel_df["method"] == method]
        if sub.empty:
            continue
        lw = 1.5 if method == "predicted_prob" else 2.5
        ax.plot(sub["mean_pred"], sub["actual_rate"], "o-",
                color=colors[method], label=labels[method],
                linewidth=lw, markersize=6, alpha=0.85)

    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Actual resolution rate", fontsize=12)
    ax.set_title("Recalibration: Before vs. After\nReliability Diagrams", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_backtest(trade_log, summary, save_path):
    cum = compute_cumulative_pnl(trade_log)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Cumulative PnL
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(cum["trade_num"], cum["cumulative_pnl"], color="#0066cc", linewidth=2)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.fill_between(cum["trade_num"], cum["cumulative_pnl"], 0,
                     where=cum["cumulative_pnl"] >= 0, alpha=0.15, color="#2a9d8f")
    ax1.fill_between(cum["trade_num"], cum["cumulative_pnl"], 0,
                     where=cum["cumulative_pnl"] < 0, alpha=0.15, color="#e63946")
    ax1.set_xlabel("Trade #", fontsize=11)
    ax1.set_ylabel("Cumulative PnL ($)", fontsize=11)
    ax1.set_title(
        f"Backtesting: {summary['strategy'].replace('_', ' ').title()} Strategy\n"
        f"ROI={summary['roi']:+.1%}  |  Sharpe={summary['sharpe_ratio']:.2f}  |  "
        f"Max Drawdown={summary['max_drawdown']:.1%}  |  Win Rate={summary['win_rate']:.1%}",
        fontsize=12, fontweight="bold"
    )

    # 2. PnL distribution
    ax2 = fig.add_subplot(gs[1, 0])
    wins = trade_log.loc[trade_log["win"], "pnl"]
    losses = trade_log.loc[~trade_log["win"], "pnl"]
    ax2.hist(losses, bins=20, color="#e63946", alpha=0.7, label="Losses")
    ax2.hist(wins, bins=20, color="#2a9d8f", alpha=0.7, label="Wins")
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("PnL per trade ($)", fontsize=11)
    ax2.set_title("PnL Distribution", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    # 3. Edge vs PnL scatter
    ax3 = fig.add_subplot(gs[1, 1])
    colors_scatter = ["#2a9d8f" if w else "#e63946" for w in trade_log["win"]]
    ax3.scatter(trade_log["edge"], trade_log["pnl"], c=colors_scatter, alpha=0.5, s=20)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_xlabel("Edge (recalibrated − market price)", fontsize=11)
    ax3.set_ylabel("PnL ($)", fontsize=11)
    ax3.set_title("Edge vs. PnL", fontsize=11, fontweight="bold")

    plt.suptitle("", y=1.0)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="quarter_kelly",
                        choices=["kelly", "half_kelly", "quarter_kelly", "flat"])
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--db", default=os.getenv("DB_PATH", "data/markets.duckdb"))
    args = parser.parse_args()

    conn = get_conn(args.db)
    df = load_analysis_df(conn, hours_to_close=args.hours)
    df = build_features(df)

    if df.empty:
        logger.error("No data. Run collect_data.py first.")
        return

    logger.info(f"Loaded {len(df)} markets for recalibration + backtesting")

    # ── Recalibration ─────────────────────────────────────────────────────────
    logger.info("Cross-validating calibrators...")
    cv_df, cv_summary = cross_validate_calibrators(df)
    logger.info("\nCross-validated performance (mean ± std):")
    logger.info(cv_summary.to_string())

    logger.info("\nFitting calibrators on full dataset for visualization...")
    rel_df, temperature = recalibration_reliability_data(df)
    logger.info(f"Optimal temperature: {temperature:.3f}")

    plot_recalibration_comparison(rel_df, f"{FIGURES}/recalibration_comparison.png")
    logger.info("Saved recalibration_comparison.png")

    # ── Add recalibrated probs to df for backtesting ──────────────────────────
    from src.analysis.recalibration import fit_isotonic
    ir = fit_isotonic(df)
    df["recalibrated_prob"] = ir.predict(df["predicted_prob"].values)

    # ── Backtesting ───────────────────────────────────────────────────────────
    logger.info(f"\nRunning backtest: strategy={args.strategy}, min_edge={args.min_edge}")
    trade_log, summary = run_backtest(
        df,
        strategy=args.strategy,
        min_edge=args.min_edge,
    )

    if "error" in summary:
        logger.error(summary["error"])
        return

    logger.info(f"\n{'='*40}")
    logger.info(f"Strategy:      {summary['strategy']}")
    logger.info(f"Bets placed:   {summary['n_bets']}")
    logger.info(f"Win rate:      {summary['win_rate']:.1%}")
    logger.info(f"Total PnL:     ${summary['total_pnl']:+.2f}")
    logger.info(f"ROI:           {summary['roi']:+.1%}")
    logger.info(f"Sharpe ratio:  {summary['sharpe_ratio']:.3f}")
    logger.info(f"Max drawdown:  {summary['max_drawdown']:.1%}")
    logger.info(f"{'='*40}")

    plot_backtest(trade_log, summary, f"{FIGURES}/backtest_{args.strategy}.png")
    logger.info(f"Saved backtest_{args.strategy}.png")

    logger.info("\nComparing all strategies...")
    comparison = compare_strategies(df, min_edge=args.min_edge)
    logger.info("\n" + comparison.to_string())

    conn.close()


if __name__ == "__main__":
    main()
