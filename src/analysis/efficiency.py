from __future__ import annotations

"""
Market efficiency analysis: price drift, information incorporation, cross-source comparison.
"""

import numpy as np
import pandas as pd
from scipy import stats


def price_drift_by_resolution(conn) -> pd.DataFrame:
    """
    Compare average price trajectories for markets that resolved YES vs NO.
    Tests whether early prices predict final resolution or if drift adds signal.
    """
    query = """
    SELECT
        m.id AS market_id,
        m.resolution,
        s.hours_to_close,
        s.price
    FROM markets m
    JOIN snapshots s ON m.id = s.market_id
    ORDER BY m.id, s.hours_to_close DESC
    """
    df = conn.execute(query).fetchdf()

    pivot = df.pivot_table(
        index=["market_id", "resolution"],
        columns="hours_to_close",
        values="price",
        aggfunc="first",
    ).reset_index()

    # compute mean price at each snapshot by resolution
    result = (
        df.groupby(["resolution", "hours_to_close"])["price"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    result.columns = ["resolution", "hours_to_close", "mean_price", "std_price", "n"]
    return result.sort_values(["resolution", "hours_to_close"], ascending=[True, False])


def early_vs_late_predictiveness(conn) -> dict:
    """
    Compare AUC of T-168h price vs T-1h price as binary classifier for resolution.
    Higher AUC at T-1h than T-168h = markets are incorporating new information.
    """
    from sklearn.metrics import roc_auc_score

    rows = {}
    for hours in [168, 72, 24, 1]:
        query = f"""
        SELECT s.price, CASE m.resolution WHEN 'YES' THEN 1 ELSE 0 END AS outcome
        FROM markets m JOIN snapshots s ON m.id = s.market_id
        WHERE s.hours_to_close = {hours}
        """
        df = conn.execute(query).fetchdf()
        if len(df) < 50:
            continue
        auc = roc_auc_score(df["outcome"], df["price"])
        rows[hours] = round(float(auc), 4)

    return rows


def category_efficiency(conn) -> pd.DataFrame:
    """
    ECE and Brier score by category, showing which market types
    are most efficiently priced.
    """
    from src.analysis.calibration import expected_calibration_error, brier_score
    from src.pipeline.ingest import load_analysis_df

    df = load_analysis_df(conn, hours_to_close=24)
    rows = []
    for category, sub in df.groupby("category"):
        if len(sub) < 30:
            continue
        rows.append({
            "category": category,
            "n": len(sub),
            "ece": round(expected_calibration_error(sub), 4),
            "brier_score": round(brier_score(sub), 4),
            "yes_rate": round(sub["outcome"].mean(), 3),
            "mean_predicted": round(sub["predicted_prob"].mean(), 3),
        })
    return pd.DataFrame(rows).sort_values("ece")


def price_momentum(conn) -> pd.DataFrame:
    """
    For markets with price data at multiple horizons:
    compute directional accuracy of early price moves.
    Does a T-168h → T-24h upward drift predict YES resolution?
    """
    query = """
    WITH s168 AS (SELECT market_id, price AS p168 FROM snapshots WHERE hours_to_close = 168),
         s24  AS (SELECT market_id, price AS p24  FROM snapshots WHERE hours_to_close = 24)
    SELECT
        m.id AS market_id,
        m.resolution,
        s168.p168,
        s24.p24,
        s24.p24 - s168.p168 AS drift,
        CASE m.resolution WHEN 'YES' THEN 1 ELSE 0 END AS outcome
    FROM markets m
    JOIN s168 ON m.id = s168.market_id
    JOIN s24  ON m.id = s24.market_id
    """
    df = conn.execute(query).fetchdf()
    if df.empty:
        return df

    # directional accuracy: did the drift correctly predict resolution direction?
    df["drift_direction"] = np.sign(df["drift"])
    df["resolution_direction"] = df["outcome"] * 2 - 1  # YES=+1, NO=-1
    df["drift_correct"] = (df["drift_direction"] == df["resolution_direction"]) & (df["drift_direction"] != 0)

    summary = {
        "total_drifting": int((df["drift_direction"] != 0).sum()),
        "drift_accuracy": round(df.loc[df["drift_direction"] != 0, "drift_correct"].mean(), 4),
        "mean_abs_drift": round(df["drift"].abs().mean(), 4),
    }

    return df, summary


def cross_source_comparison(conn) -> pd.DataFrame:
    """
    Compare ECE and Brier score between Polymarket and Kalshi markets.
    """
    from src.analysis.calibration import expected_calibration_error, brier_score
    from src.pipeline.ingest import load_analysis_df

    df = load_analysis_df(conn, hours_to_close=24)
    rows = []
    for source, sub in df.groupby("source"):
        if len(sub) < 30:
            continue
        rows.append({
            "source": source,
            "n": len(sub),
            "ece": round(expected_calibration_error(sub), 4),
            "brier_score": round(brier_score(sub), 4),
        })
    return pd.DataFrame(rows)
