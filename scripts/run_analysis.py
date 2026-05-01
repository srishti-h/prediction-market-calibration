"""
Run full calibration analysis and generate all figures.

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --hours 1
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from src.pipeline.ingest import get_conn, load_analysis_df
from src.pipeline.features import build_features
from src.analysis import calibration, bias, efficiency
from src.viz import plots


def main():
    parser = argparse.ArgumentParser(description="Run prediction market analysis")
    parser.add_argument("--hours", type=int, default=24, help="Snapshot hours-to-close for primary analysis")
    parser.add_argument("--bins", type=int, default=10, help="Number of calibration bins")
    parser.add_argument("--db", default=os.getenv("DB_PATH", "data/markets.duckdb"))
    args = parser.parse_args()

    conn = get_conn(args.db)
    df = load_analysis_df(conn, hours_to_close=args.hours)

    if df.empty:
        logger.error("No data found. Run scripts/collect_data.py first.")
        return

    df = build_features(df, n_bins=args.bins)

    # ── Calibration ──────────────────────────────────────────────────────────
    logger.info("Computing calibration metrics...")
    ece = calibration.expected_calibration_error(df, n_bins=args.bins)
    bs = calibration.brier_score(df)
    ll = calibration.log_loss(df)
    rel = calibration.reliability_data(df, n_bins=args.bins)
    cat_cal = calibration.calibration_by_group(df, "category", n_bins=args.bins)
    temporal = calibration.temporal_calibration(conn, n_bins=args.bins)

    logger.info(f"Overall ECE: {ece:.4f}")
    logger.info(f"Brier Score: {bs:.4f}")
    logger.info(f"Log Loss:    {ll:.4f}")
    logger.info("\nCalibration by category:")
    logger.info(cat_cal.to_string(index=False))

    # ── Bias ─────────────────────────────────────────────────────────────────
    logger.info("\nFavorite-longshot analysis...")
    fls = bias.favorite_longshot_stats(df)
    logistic = bias.logistic_calibration_curve(df)
    vol_bias = bias.volume_bias_correlation(df)

    logger.info(fls.to_string(index=False))
    logger.info(f"\nLogistic slope: {logistic['slope']} ({logistic['interpretation']})")
    logger.info(f"Volume-bias correlation: r={vol_bias['pearson_r']}, p={vol_bias['p_value']}")
    logger.info(f"  → {vol_bias['interpretation']}")

    # ── Efficiency ───────────────────────────────────────────────────────────
    logger.info("\nEfficiency analysis...")
    auc_by_horizon = efficiency.early_vs_late_predictiveness(conn)
    cat_eff = efficiency.category_efficiency(conn)
    drift = efficiency.price_drift_by_resolution(conn)

    logger.info(f"AUC by horizon: {auc_by_horizon}")
    logger.info("\nCategory efficiency:")
    logger.info(cat_eff.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("\nGenerating figures...")

    plots.reliability_diagram(rel, title="Polymarket Calibration", ece=ece, brier=bs)
    plots.favorite_longshot_chart(fls)
    plots.category_ece_chart(cat_cal)

    if not temporal.empty:
        plots.temporal_calibration_chart(temporal)

    if not drift.empty:
        plots.price_drift_chart(drift)

    plots.volume_calibration_scatter(df)

    logger.info("Figures saved to outputs/figures/")

    # Save summary JSON for dashboard
    summary = {
        "ece": ece,
        "brier_score": bs,
        "log_loss": ll,
        "n_markets": len(df),
        "logistic_slope": logistic["slope"],
        "logistic_interpretation": logistic["interpretation"],
        "auc_by_horizon": auc_by_horizon,
        "volume_bias": vol_bias,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary saved to outputs/summary.json")
    conn.close()


if __name__ == "__main__":
    main()
