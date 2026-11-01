"""Model definitions and shared training helpers."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

import data
import features

SEED = 42

LGB_PARAMS = dict(
    objective="huber",
    alpha=0.15,            # huber delta on the log-rate-per-mile scale
    learning_rate=0.06,
    n_estimators=400,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=-1,
)

# stationary features seen by the trees (no market_index / no time index: those go in the linear part)
TREE_COLS = ["distance", "log_distance", "equipment", "weight", "weight_missing", "weight_capped",
             "mi_dev", "qs_dev", "qs_dev_abs", "pickup_lat", "pickup_lon", "delivery_lat",
             "delivery_lon", "d_lat", "d_lon"]
CITY_COLS = ["pickup", "delivery"]
CAL_COLS = [f"dow_{k}" for k in range(1, 7)] + ["qend_decay"]
LINEAR_COLS = ["log_mi_day", "qs_day", "t"] + CAL_COLS


class TrendGBM:
    """log(rate/mile) = GBM(stationary features) + b1*log(market_index) + b2*t.

    Trees cannot extrapolate a time trend into Nov-Dec, so the drift and the
    market-index effect live in a small linear model. The two parts are fit by
    backfitting (alternating) so neither absorbs the other's signal.
    """

    def __init__(self, tree_cols=TREE_COLS, linear_cols=LINEAR_COLS, params=None,
                 n_estimators=None, n_rounds=2):
        self.tree_cols, self.linear_cols = list(tree_cols), list(linear_cols)
        self.params, self.n_estimators, self.n_rounds = params, n_estimators, n_rounds

    def fit(self, F, y):
        from sklearn.linear_model import LinearRegression
        y = np.asarray(y)
        lin = np.zeros(len(y))
        for _ in range(self.n_rounds):
            self.gbm = fit_lgb(F[self.tree_cols], y - lin, self.params, self.n_estimators)
            resid = y - self.gbm.predict(F[self.tree_cols])
            self.lin = LinearRegression().fit(F[self.linear_cols], resid)
            lin = self.lin.predict(F[self.linear_cols])
        return self

    def predict(self, F):
        return self.gbm.predict(F[self.tree_cols]) + self.lin.predict(F[self.linear_cols])


def prep(train_raw, valid_raw):
    """Clean inputs; return cleaned train/valid, city list and the two daily signals."""
    daily = data.daily_market_index(train_raw, valid_raw)
    tr = data.clean_inputs(train_raw, daily)
    va = data.clean_inputs(valid_raw, daily)
    cities = sorted(set(tr.pickup) | set(tr.delivery) | set(va.pickup) | set(va.delivery))
    daily_qs = data.daily_quote_signal(train_raw, valid_raw)
    return tr, va, cities, daily, daily_qs


def fit_lgb(X, y, params=None, n_estimators=None):
    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    if n_estimators:
        p["n_estimators"] = n_estimators
    m = lgb.LGBMRegressor(**p)
    m.fit(X, y)
    return m


def to_rate(pred_log_rpm, distance):
    return np.exp(pred_log_rpm) * np.asarray(distance)


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    return dict(
        MAE=float(np.mean(np.abs(err))),
        MAPE=float(np.mean(np.abs(err) / y_true) * 100),
        RMSE=float(np.sqrt(np.mean(err ** 2))),
        MdAPE=float(np.median(np.abs(err) / y_true) * 100),
    )
