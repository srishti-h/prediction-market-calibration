from __future__ import annotations

"""
Recalibration models: isotonic regression, Platt scaling, temperature scaling.
Shows how to correct systematic market mispricing.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_prob)) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


# ── Isotonic Regression ───────────────────────────────────────────────────────

def fit_isotonic(df: pd.DataFrame, prob_col: str = "predicted_prob", outcome_col: str = "outcome") -> IsotonicRegression:
    """
    Fit isotonic regression calibrator on the full dataset.
    Monotone constraint ensures higher predicted prob → higher calibrated prob.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(df[prob_col].values, df[outcome_col].values)
    return ir


# ── Platt Scaling ─────────────────────────────────────────────────────────────

def fit_platt(df: pd.DataFrame, prob_col: str = "predicted_prob", outcome_col: str = "outcome") -> LogisticRegression:
    """
    Platt scaling: fit logistic regression on log-odds of predicted probability.
    Equivalent to learning an affine transformation of the log-odds space.
    """
    log_odds = logit(df[prob_col].clip(1e-6, 1 - 1e-6)).values.reshape(-1, 1)
    lr = LogisticRegression(C=1e10)  # minimal regularization
    lr.fit(log_odds, df[outcome_col].values)
    return lr


def platt_predict(lr: LogisticRegression, probs: np.ndarray) -> np.ndarray:
    log_odds = logit(np.clip(probs, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    return lr.predict_proba(log_odds)[:, 1]


# ── Temperature Scaling ───────────────────────────────────────────────────────

def fit_temperature(df: pd.DataFrame, prob_col: str = "predicted_prob", outcome_col: str = "outcome") -> float:
    """
    Temperature scaling: divide log-odds by T before sigmoid.
    T > 1 → soften predictions (fix overconfidence).
    T < 1 → sharpen predictions (fix underconfidence).
    """
    log_odds = logit(df[prob_col].clip(1e-6, 1 - 1e-6)).values
    y = df[outcome_col].values

    def nll(T):
        calibrated = expit(log_odds / T)
        calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(calibrated) + (1 - y) * np.log(1 - calibrated))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def temperature_predict(T: float, probs: np.ndarray) -> np.ndarray:
    log_odds = logit(np.clip(probs, 1e-6, 1 - 1e-6))
    return expit(log_odds / T)


# ── Cross-Validated Comparison ────────────────────────────────────────────────

def cross_validate_calibrators(
    df: pd.DataFrame,
    prob_col: str = "predicted_prob",
    outcome_col: str = "outcome",
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    K-fold cross-validation comparing uncalibrated vs. three calibration methods.
    Returns DataFrame with ECE and Brier score per fold and method.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = df[prob_col].values
    y = df[outcome_col].values

    results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_df = pd.DataFrame({prob_col: X_train, outcome_col: y_train})

        # Uncalibrated
        results.append({
            "fold": fold, "method": "uncalibrated",
            "ece": _ece(y_test, X_test),
            "brier": brier_score_loss(y_test, X_test),
        })

        # Isotonic
        ir = fit_isotonic(train_df, prob_col, outcome_col)
        p_iso = ir.predict(X_test)
        results.append({
            "fold": fold, "method": "isotonic",
            "ece": _ece(y_test, p_iso),
            "brier": brier_score_loss(y_test, p_iso),
        })

        # Platt
        lr = fit_platt(train_df, prob_col, outcome_col)
        p_platt = platt_predict(lr, X_test)
        results.append({
            "fold": fold, "method": "platt_scaling",
            "ece": _ece(y_test, p_platt),
            "brier": brier_score_loss(y_test, p_platt),
        })

        # Temperature
        T = fit_temperature(train_df, prob_col, outcome_col)
        p_temp = temperature_predict(T, X_test)
        results.append({
            "fold": fold, "method": f"temperature (T={T:.2f})",
            "ece": _ece(y_test, p_temp),
            "brier": brier_score_loss(y_test, p_temp),
        })

    cv_df = pd.DataFrame(results)
    summary = (
        cv_df.groupby("method")[["ece", "brier"]]
        .agg(["mean", "std"])
        .round(4)
    )
    return cv_df, summary


def recalibration_reliability_data(
    df: pd.DataFrame,
    prob_col: str = "predicted_prob",
    outcome_col: str = "outcome",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Fit all calibrators on full data, return reliability data for each method.
    For visualization — use cross_validate for unbiased performance estimates.
    """
    ir = fit_isotonic(df, prob_col, outcome_col)
    lr = fit_platt(df, prob_col, outcome_col)
    T = fit_temperature(df, prob_col, outcome_col)

    df = df.copy()
    df["isotonic"] = ir.predict(df[prob_col].values)
    df["platt"] = platt_predict(lr, df[prob_col].values)
    df["temperature"] = temperature_predict(T, df[prob_col].values)

    rows = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for col in [prob_col, "isotonic", "platt", "temperature"]:
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (df[col] >= lo) & (df[col] <= hi)
            if mask.sum() < 3:
                continue
            rows.append({
                "method": col,
                "bin_center": (lo + hi) / 2,
                "mean_pred": df.loc[mask, col].mean(),
                "actual_rate": df.loc[mask, outcome_col].mean(),
                "count": int(mask.sum()),
            })

    return pd.DataFrame(rows), T
