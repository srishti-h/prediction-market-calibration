"""
Feature engineering on top of the raw analysis DataFrame.
Adds binned probabilities, log-odds, and market metadata features.
"""

import numpy as np
import pandas as pd


def add_prob_bins(df: pd.DataFrame, n_bins: int = 10, col: str = "predicted_prob") -> pd.DataFrame:
    """
    Add decile probability bins. Each bin label is the bin midpoint.
    """
    df = df.copy()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]
    df["prob_bin"] = pd.cut(
        df[col], bins=bin_edges, labels=bin_labels, include_lowest=True
    ).astype(float)
    return df


def add_log_odds(df: pd.DataFrame, col: str = "predicted_prob") -> pd.DataFrame:
    df = df.copy()
    p = df[col].clip(0.001, 0.999)
    df["log_odds"] = np.log(p / (1 - p))
    return df


def add_confidence_tier(df: pd.DataFrame, col: str = "predicted_prob") -> pd.DataFrame:
    """
    Label markets as favorite (>70%), longshot (<30%), or contested (30-70%).
    """
    df = df.copy()
    conditions = [
        df[col] > 0.70,
        df[col] < 0.30,
    ]
    choices = ["favorite", "longshot"]
    df["confidence_tier"] = np.select(conditions, choices, default="contested")
    return df


def add_log_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_volume"] = np.log1p(df["volume"])
    return df


def build_features(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    df = add_prob_bins(df, n_bins=n_bins)
    df = add_log_odds(df)
    df = add_confidence_tier(df)
    df = add_log_volume(df)
    return df
