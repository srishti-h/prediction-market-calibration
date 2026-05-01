from __future__ import annotations

"""
Bias analysis: favorite-longshot bias, overconfidence, and systematic mispricing.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def favorite_longshot_stats(df: pd.DataFrame, prob_col: str = "predicted_prob", outcome_col: str = "outcome") -> pd.DataFrame:
    """
    Compare implied probability vs. actual resolution rate by confidence tier.
    The favorite-longshot bias manifests as:
      - longshots (<30%): actual rate HIGHER than implied (market overestimates difficulty)
      - favorites (>70%): actual rate LOWER than implied (market overestimates certainty)
    """
    tiers = {
        "longshot": df[prob_col] < 0.30,
        "contested": (df[prob_col] >= 0.30) & (df[prob_col] <= 0.70),
        "favorite": df[prob_col] > 0.70,
    }
    rows = []
    for tier, mask in tiers.items():
        sub = df[mask]
        if len(sub) < 10:
            continue
        mean_pred = sub[prob_col].mean()
        actual_rate = sub[outcome_col].mean()
        bias = actual_rate - mean_pred  # positive = underpredicted (good for bettor)
        se = np.sqrt(actual_rate * (1 - actual_rate) / len(sub))
        z = bias / se if se > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append({
            "tier": tier,
            "n": len(sub),
            "mean_predicted": round(mean_pred, 4),
            "actual_rate": round(actual_rate, 4),
            "bias": round(bias, 4),
            "z_score": round(z, 2),
            "p_value": round(p_val, 4),
            "significant": p_val < 0.05,
        })
    return pd.DataFrame(rows)


def logistic_calibration_curve(df: pd.DataFrame, prob_col: str = "predicted_prob", outcome_col: str = "outcome") -> dict:
    """
    Fit logistic regression: outcome ~ log_odds(predicted_prob).
    Perfect calibration: intercept=0, slope=1.
    Overconfidence: slope < 1 (extremes are too extreme).
    Underconfidence: slope > 1.
    """
    log_odds = np.log(df[prob_col].clip(0.001, 0.999) / (1 - df[prob_col].clip(0.001, 0.999)))
    X = sm.add_constant(log_odds)
    y = df[outcome_col]

    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    intercept, slope = result.params
    intercept_ci = result.conf_int().iloc[0].tolist()
    slope_ci = result.conf_int().iloc[1].tolist()

    return {
        "intercept": round(float(intercept), 4),
        "slope": round(float(slope), 4),
        "intercept_ci": [round(x, 4) for x in intercept_ci],
        "slope_ci": [round(x, 4) for x in slope_ci],
        "slope_p_value": round(float(result.pvalues.iloc[1]), 4),
        "interpretation": _interpret_slope(slope),
    }


def _interpret_slope(slope: float) -> str:
    if slope < 0.85:
        return "overconfident (extremes too extreme)"
    elif slope > 1.15:
        return "underconfident (extremes too moderate)"
    else:
        return "well-calibrated"


def overconfidence_by_bin(df: pd.DataFrame, prob_col: str = "predicted_prob", outcome_col: str = "outcome") -> pd.DataFrame:
    """
    For each decile bin, compute: mean predicted, actual rate, and signed bias.
    Positive bias = market underpredicts (good value for YES bettors).
    """
    bins = np.linspace(0, 1, 11)
    rows = []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (df[prob_col] >= lo) & (df[prob_col] <= hi)
        sub = df[mask]
        if len(sub) < 5:
            continue
        rows.append({
            "bin": f"{int(lo*100)}-{int(hi*100)}%",
            "bin_mid": (lo + hi) / 2,
            "n": len(sub),
            "mean_predicted": sub[prob_col].mean(),
            "actual_rate": sub[outcome_col].mean(),
            "bias": sub[outcome_col].mean() - sub[prob_col].mean(),
        })
    return pd.DataFrame(rows)


def volume_bias_correlation(df: pd.DataFrame, prob_col: str = "predicted_prob", outcome_col: str = "outcome") -> dict:
    """
    Test whether higher-volume markets exhibit less favorite-longshot bias.
    Hypothesis: professional traders correct mispricing in liquid markets.
    """
    df = df.copy()
    df["abs_bias"] = abs(df[outcome_col] - df[prob_col])
    df["log_volume"] = np.log1p(df["volume"])

    r, p = stats.pearsonr(df["log_volume"].dropna(), df.loc[df["log_volume"].notna(), "abs_bias"])
    return {
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
        "interpretation": "more liquid markets are better calibrated" if r < 0 and p < 0.05 else "no significant relationship",
    }
