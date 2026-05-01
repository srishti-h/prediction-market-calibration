from __future__ import annotations

"""
Publication-quality plots for calibration analysis.
All functions save to outputs/figures/ and return the figure path.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "polymarket": "#0066cc",
    "kalshi": "#e63946",
    "longshot": "#e63946",
    "contested": "#457b9d",
    "favorite": "#2a9d8f",
    "perfect": "#adb5bd",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def reliability_diagram(
    rel_df: pd.DataFrame,
    title: str = "Reliability Diagram",
    ece: float = None,
    brier: float = None,
    save_path: str = None,
) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], "--", color=PALETTE["perfect"], linewidth=1.5, label="Perfect calibration", zorder=1)

    ax.fill_between(
        rel_df["mean_pred"],
        rel_df["ci_low"],
        rel_df["ci_high"],
        alpha=0.2,
        color=PALETTE["polymarket"],
        label="95% CI",
    )
    ax.plot(
        rel_df["mean_pred"],
        rel_df["actual_rate"],
        "o-",
        color=PALETTE["polymarket"],
        linewidth=2,
        markersize=8,
        zorder=5,
        label="Market prices",
    )

    # size markers by count
    sizes = (rel_df["count"] / rel_df["count"].max()) * 200 + 30
    scatter = ax.scatter(
        rel_df["mean_pred"],
        rel_df["actual_rate"],
        s=sizes,
        color=PALETTE["polymarket"],
        zorder=6,
        alpha=0.8,
    )

    subtitle_parts = []
    if ece is not None:
        subtitle_parts.append(f"ECE = {ece:.3f}")
    if brier is not None:
        subtitle_parts.append(f"Brier = {brier:.3f}")
    subtitle = "  |  ".join(subtitle_parts)

    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Actual resolution rate", fontsize=12)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / "reliability_diagram.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def favorite_longshot_chart(fls_df: pd.DataFrame, save_path: str = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [PALETTE.get(t, "#457b9d") for t in fls_df["tier"]]
    bars = ax.bar(fls_df["tier"], fls_df["bias"], color=colors, alpha=0.85, width=0.5)

    ax.axhline(0, color="black", linewidth=0.8)

    for bar, (_, row) in zip(bars, fls_df.iterrows()):
        sig = "*" if row["significant"] else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.005 if bar.get_height() >= 0 else -0.015),
            f"{row['bias']:+.3f}{sig}\n(n={row['n']:,})",
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=10,
        )

    ax.set_xlabel("Market tier", fontsize=12)
    ax.set_ylabel("Bias (actual rate − predicted probability)", fontsize=12)
    ax.set_title(
        "Favorite-Longshot Bias\nPositive = market underestimates probability (* = p < 0.05)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / "favorite_longshot_bias.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def category_ece_chart(cat_df: pd.DataFrame, save_path: str = None) -> str:
    cat_df = cat_df.sort_values("ece")
    fig, ax = plt.subplots(figsize=(9, max(4, len(cat_df) * 0.6)))

    bars = ax.barh(cat_df["category"], cat_df["ece"], color=PALETTE["polymarket"], alpha=0.8)
    for bar, n in zip(bars, cat_df["n"]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"n={n:,}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Expected Calibration Error (lower = better)", fontsize=12)
    ax.set_title("Calibration by Market Category\nLower ECE = more efficiently priced", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / "category_ece.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def temporal_calibration_chart(temporal_df: pd.DataFrame, save_path: str = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        temporal_df["hours_to_close"],
        temporal_df["ece"],
        "o-",
        color=PALETTE["polymarket"],
        linewidth=2.5,
        markersize=9,
    )

    for _, row in temporal_df.iterrows():
        ax.annotate(
            f"ECE={row['ece']:.3f}\n(n={row['n']:,})",
            (row["hours_to_close"], row["ece"]),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=9,
        )

    ax.invert_xaxis()
    ax.set_xlabel("Hours before market close", fontsize=12)
    ax.set_ylabel("Expected Calibration Error", fontsize=12)
    ax.set_title("How Calibration Improves as Markets Approach Resolution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / "temporal_calibration.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def price_drift_chart(drift_df: pd.DataFrame, save_path: str = None) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))

    for resolution, sub in drift_df.groupby("resolution"):
        sub = sub.sort_values("hours_to_close", ascending=False)
        color = PALETTE["favorite"] if resolution == "YES" else PALETTE["longshot"]
        ax.plot(sub["hours_to_close"], sub["mean_price"], "o-", color=color, label=f"Resolved {resolution}", linewidth=2)
        ax.fill_between(
            sub["hours_to_close"],
            sub["mean_price"] - sub["std_price"] / np.sqrt(sub["n"].clip(1)),
            sub["mean_price"] + sub["std_price"] / np.sqrt(sub["n"].clip(1)),
            alpha=0.15,
            color=color,
        )

    ax.invert_xaxis()
    ax.set_xlabel("Hours before market close", fontsize=12)
    ax.set_ylabel("Mean YES price", fontsize=12)
    ax.set_title("Price Drift by Resolution Outcome\nMarkets should diverge as new information arrives", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / "price_drift.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def volume_calibration_scatter(df: pd.DataFrame, save_path: str = None) -> str:
    df = df.copy()
    df["abs_bias"] = abs(df["outcome"] - df["predicted_prob"])
    df["log_volume"] = np.log1p(df["volume"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["log_volume"], df["abs_bias"], alpha=0.15, s=10, color=PALETTE["polymarket"])

    # trend line
    valid = df[["log_volume", "abs_bias"]].dropna()
    z = np.polyfit(valid["log_volume"], valid["abs_bias"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid["log_volume"].min(), valid["log_volume"].max(), 200)
    ax.plot(x_line, p(x_line), color=PALETTE["longshot"], linewidth=2, label=f"Trend (slope={z[0]:.4f})")

    ax.set_xlabel("log(1 + volume)", fontsize=12)
    ax.set_ylabel("|predicted prob − actual outcome|", fontsize=12)
    ax.set_title("Liquidity vs. Calibration Error\nDo higher-volume markets price more accurately?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = save_path or str(FIGURES_DIR / "volume_calibration.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
