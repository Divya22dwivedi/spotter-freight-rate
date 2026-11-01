"""Loading, cleaning and target-anomaly detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
WEIGHT_CAP = 47_500.0  # hard ceiling observed in the data (clipped values)
DIST_BINS = [0, 150, 300, 500, 800, 1200, 1800, 2500, 5000]


def load_raw():
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    return train, valid


def daily_market_index(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Mean market_index per day, from feature columns only (no labels)."""
    both = pd.concat([train[["date", "market_index"]], valid[["date", "market_index"]]])
    return both.groupby("date")["market_index"].mean()


def daily_quote_signal(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Median quote_signal per day (a smooth day-level 'quote index'), features only."""
    both = pd.concat([train[["date", "quote_signal"]], valid[["date", "quote_signal"]]])
    return both.groupby("date")["quote_signal"].median()


def clean_inputs(df: pd.DataFrame, daily_index: pd.Series) -> pd.DataFrame:
    """Fix input-data issues. Uses only feature columns (never the target).

    * weight < 0            -> sign flip: take abs() and keep a flag
    * weight == 47_500      -> clipped at a ceiling: keep a flag
    * weight missing        -> left as NaN (LightGBM handles it) + flag
    * market_index missing  -> filled with that day's mean index across all
                               loads (train + validation features, no labels)
    """
    out = df.copy()
    out["weight_neg"] = (out["weight"] < 0).astype(int)
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight_capped"] = (out["weight"].abs() >= WEIGHT_CAP).astype(int)
    out["weight"] = out["weight"].abs()
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out["date"].map(daily_index))
    return out


def rate_ratio(train: pd.DataFrame) -> pd.Series:
    """posted rate-per-mile divided by the median for its (distance bin x equipment)."""
    rpm = train["posted_rate"] / train["distance"]
    bins = pd.cut(train["distance"], DIST_BINS)
    ref = rpm.groupby([bins, train["equipment"]], observed=True).transform("median")
    return rpm / ref


def flag_target_anomalies(train: pd.DataFrame, lo: float = 0.6, hi: float = 1.8) -> pd.Series:
    """True for rows whose posted_rate is implausible for its distance/equipment.

    Clean rows sit within roughly +-40% of the group median; corrupted rows are
    far outside (>2x or <0.5x), so the [lo, hi] band lies in a clear gap.
    """
    r = rate_ratio(train)
    return (r < lo) | (r > hi)
