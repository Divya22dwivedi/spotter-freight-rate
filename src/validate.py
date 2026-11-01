"""Validation study.

Why not a random K-fold?  The real task is out-of-time (train Jan-Oct, predict
Nov-Dec) and 12% of validation loads touch 8 cities never seen in training.
So we use:

  A) expanding-window time folds: train on months < M, test on month M (M=7..10)
  B) time + new-city simulation: train Jan-Aug without loads touching 8 held-out
     cities, test on Sep-Oct (reported separately for unseen / seen cities), 2 repeats

Metrics are computed on rows whose label passes the anomaly filter ("clean")
and, for transparency, on all rows.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import data
import features
import modeling as M

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def baseline_predict(train, test):
    """Median rate-per-mile by (distance bin x equipment) x distance."""
    rpm = train.posted_rate / train.distance
    med = rpm.groupby([pd.cut(train.distance, data.DIST_BINS), train.equipment], observed=True).median()
    keys = zip(pd.cut(test.distance, data.DIST_BINS), test.equipment)
    return np.array([med.get(k, rpm.median()) for k in keys]) * test.distance.values


def gbm_no_trend(F, y, t_idx, v_idx):
    """Same trees but market_index and t are given to the trees (cannot extrapolate)."""
    cols = M.TREE_COLS + ["log_mi_day", "qs_day", "t"] + M.CAL_COLS
    m = M.fit_lgb(F.iloc[t_idx][cols], y[t_idx])
    return m.predict(F.iloc[v_idx][cols])


def main():
    tr_raw, va_raw = data.load_raw()
    tr, va, cities, daily, daily_qs = M.prep(tr_raw, va_raw)
    anom = data.flag_target_anomalies(tr_raw).values
    F = features.build_features(tr, cities, daily, daily_qs)
    y = np.log(tr.posted_rate / tr.distance).values
    month = tr.date.dt.month.values
    rate = tr.posted_rate.values
    dist = tr.distance.values

    rows, coefs = [], []
    for vm in [7, 8, 9, 10]:
        t_idx = np.where((month < vm) & ~anom)[0]
        v_idx = np.where(month == vm)[0]
        clean = ~anom[v_idx]
        preds = {
            "baseline (bin median)": baseline_predict(tr.iloc[t_idx], tr.iloc[v_idx]),
            "LightGBM, no trend term": M.to_rate(gbm_no_trend(F, y, t_idx, v_idx), dist[v_idx]),
        }
        m = M.TrendGBM().fit(F.iloc[t_idx], y[t_idx])
        preds["Trend + LightGBM (final)"] = M.to_rate(m.predict(F.iloc[v_idx]), dist[v_idx])
        coefs.append(dict(fold=vm, b_log_market_index=float(m.lin.coef_[0]), b_trend_per_300d=float(m.lin.coef_[1])))
        for name, p in preds.items():
            for subset, mask in [("clean", clean), ("all", np.ones(len(v_idx), bool))]:
                rows.append(dict(scheme="A_time", fold=vm, model=name, subset=subset,
                                 **M.metrics(rate[v_idx][mask], p[mask])))
        print("fold", vm, "done", flush=True)

    # --- B: unseen cities + time -------------------------------------------
    rng = np.random.default_rng(0)
    city_list = np.array(sorted(set(tr.pickup) | set(tr.delivery)))
    for rep in range(2):
        held = set(rng.choice(city_list, 8, replace=False))
        touches = (tr.pickup.isin(held) | tr.delivery.isin(held)).values
        t_idx = np.where((month <= 8) & ~anom & ~touches)[0]
        v_idx = np.where(month >= 9)[0]
        m = M.TrendGBM().fit(F.iloc[t_idx], y[t_idx])
        p = M.to_rate(m.predict(F.iloc[v_idx]), dist[v_idx])
        for lab, mask in [("unseen cities", touches[v_idx]), ("seen cities", ~touches[v_idx])]:
            mask = mask & ~anom[v_idx]
            rows.append(dict(scheme="B_unseen_city", fold=rep, model="Trend + LightGBM (final)", subset=lab,
                             **M.metrics(rate[v_idx][mask], p[mask])))
        print("rep", rep, "done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cv_results.csv", index=False)
    pd.DataFrame(coefs).to_csv(OUT / "cv_trend_coefficients.csv", index=False)

    a = res[(res.scheme == "A_time")]
    summ = {
        "A_time_clean": a[a.subset == "clean"].groupby("model")[["MAE", "MAPE", "RMSE", "MdAPE"]].mean().round(2).to_dict("index"),
        "A_time_all": a[a.subset == "all"].groupby("model")[["MAE", "MAPE", "RMSE", "MdAPE"]].mean().round(2).to_dict("index"),
        "B_unseen_city": res[res.scheme == "B_unseen_city"].groupby("subset")[["MAE", "MAPE", "RMSE", "MdAPE"]].mean().round(2).to_dict("index"),
        "A_by_fold_MAPE_clean": a[(a.subset == "clean")].pivot(index="fold", columns="model", values="MAPE").round(2).to_dict("index"),
    }
    (OUT / "cv_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
