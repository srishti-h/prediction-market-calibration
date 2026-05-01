"""
Calibration analysis: reliability diagrams, ECE, Brier score.
"""

import numpy as np
import pandas as pd
from scipy import stats


def brier_score(df: pd.DataFrame, prob_col: str = "predicted_prob", outcome_col: str = "outcome") -> float:
    return float(np.mean((df[prob_col] - df[outcome_col]) ** 2))


def expected_calibration_error(
    df: pd.DataFrame,
    n_bins: int = 10,
    prob_col: str = "predicted_prob",
    outcome_col: str = "outcome",
) -> float:
    """
    ECE: weighted average absolute deviation between predicted and actual rates.
    Lower is better. Perfect calibration = 0.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    total = len(df)
    ece = 0.0

    for i in range(n_bins):
        mask = (df[prob_col] >= bin_edges[i]) & (df[prob_col] < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (df[prob_col] >= bin_edges[i]) & (df[prob_col] <= bin_edges[i + 1])
        bucket = df[mask]
        if len(bucket) == 0:
            continue
        mean_pred = bucket[prob_col].mean()
        mean_actual = bucket[outcome_col].mean()
        ece += (len(bucket) / total) * abs(mean_pred - mean_actual)

    return float(ece)


def reliability_data(
    df: pd.DataFrame,
    n_bins: int = 10,
    prob_col: str = "predicted_prob",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    """
    Compute per-bin statistics for a reliability diagram.
    Returns DataFrame with columns: bin_center, mean_pred, actual_rate, count, ci_low, ci_high.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (df[prob_col] >= lo) & (df[prob_col] <= hi)
        bucket = df[mask]
        n = len(bucket)
        if n == 0:
            continue

        actual_rate = bucket[outcome_col].mean()
        mean_pred = bucket[prob_col].mean()
        se = np.sqrt(actual_rate * (1 - actual_rate) / max(n, 1))
        ci_low = max(0, actual_rate - 1.96 * se)
        ci_high = min(1, actual_rate + 1.96 * se)

        rows.append({
            "bin_center": (lo + hi) / 2,
            "mean_pred": mean_pred,
            "actual_rate": actual_rate,
            "count": n,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    return pd.DataFrame(rows)


def calibration_by_group(
    df: pd.DataFrame,
    group_col: str,
    n_bins: int = 10,
    prob_col: str = "predicted_prob",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    """
    Compute ECE and Brier score per group (e.g. category, source).
    """
    rows = []
    for group, sub in df.groupby(group_col):
        if len(sub) < 30:
            continue
        rows.append({
            group_col: group,
            "n": len(sub),
            "ece": expected_calibration_error(sub, n_bins, prob_col, outcome_col),
            "brier_score": brier_score(sub, prob_col, outcome_col),
            "yes_rate": sub[outcome_col].mean(),
            "mean_predicted": sub[prob_col].mean(),
        })
    return pd.DataFrame(rows).sort_values("ece")


def temporal_calibration(
    conn,
    hours_list: list[int] = [168, 72, 24, 1],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute ECE at each snapshot horizon to show how calibration evolves
    as markets approach resolution.
    """
    from src.pipeline.ingest import load_analysis_df

    rows = []
    for hours in hours_list:
        df = load_analysis_df(conn, hours_to_close=hours)
        if len(df) < 50:
            continue
        rows.append({
            "hours_to_close": hours,
            "n": len(df),
            "ece": expected_calibration_error(df, n_bins),
            "brier_score": brier_score(df),
        })
    return pd.DataFrame(rows).sort_values("hours_to_close", ascending=False)


def log_loss(df: pd.DataFrame, prob_col: str = "predicted_prob", outcome_col: str = "outcome") -> float:
    p = df[prob_col].clip(1e-7, 1 - 1e-7)
    y = df[outcome_col]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
