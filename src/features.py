"""Feature engineering (shared by training, validation and the December chart)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
T0 = pd.Timestamp("2025-01-01")


def build_features(df: pd.DataFrame, cities: list[str], daily_mi: pd.Series, daily_qs: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["distance"] = df["distance"]
    f["log_distance"] = np.log(df["distance"])
    f["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT)
    f["weight"] = df["weight"]
    f["weight_missing"] = df["weight_missing"]
    f["weight_capped"] = df["weight_capped"]
    # market_index / quote_signal each have a smooth DAY-level component (drives the rate
    # level) and a load-level wiggle around it. Day-level -> linear part, wiggle -> trees.
    mi_day = df["date"].map(daily_mi)
    qs_day = df["date"].map(daily_qs)
    f["log_mi_day"] = np.log(mi_day)
    f["qs_day"] = qs_day
    f["mi_dev"] = df["market_index"] - mi_day
    f["qs_dev"] = df["quote_signal"] - qs_day
    f["qs_dev_abs"] = f["qs_dev"].abs()

    # geography
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        f[c] = df[c]
    f["d_lat"] = df["delivery_lat"] - df["pickup_lat"]
    f["d_lon"] = df["delivery_lon"] - df["pickup_lon"]
    f["pickup"] = pd.Categorical(df["pickup"], categories=cities)
    f["delivery"] = pd.Categorical(df["delivery"], categories=cities)

    # Calendar effects go ONLY into the linear stage (never the trees): Nov-Dec are not in the
    # training data, and trees given date-like inputs would just memorise the (confounded) trend.
    d = df["date"]
    dtq = (d.dt.to_period("Q").dt.end_time.dt.normalize() - d).dt.days
    for k in range(1, 7):
        f[f"dow_{k}"] = (d.dt.dayofweek == k).astype(float)
    f["qend_decay"] = np.exp(-dtq / 7.0)  # quarter-end lift, decays over ~2 weeks
    # trend input for the linear component (not seen by the trees)
    f["t"] = (d - T0).dt.days / 300.0

    return f
