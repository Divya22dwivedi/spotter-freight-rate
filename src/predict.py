"""Train the final model on ALL labelled data and write the submission files.

Outputs (repo root):
  validation_predictions.csv                 load_id,predicted_rate   (12,000 rows, template order)
  data/december_chart_inputs_completed.csv   the 31 December rows with predicted_rate filled
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import data
import features
import modeling as M

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 7, 2025]


def main():
    tr_raw, va_raw = data.load_raw()
    tr, va, cities, daily, daily_qs = M.prep(tr_raw, va_raw)

    # 1) clean the labels: drop rows whose posted_rate is implausible (~1.4%)
    anom = data.flag_target_anomalies(tr_raw).values
    print(f"dropping {anom.sum()} anomalous-label rows ({anom.mean():.2%}) from training")
    F_tr = features.build_features(tr, cities, daily, daily_qs)[~anom]
    y = np.log(tr.posted_rate / tr.distance).values[~anom]

    # 2) fit a small seed-ensemble of the Trend + LightGBM model
    models = [M.TrendGBM(params=dict(random_state=s)).fit(F_tr, y) for s in SEEDS]
    lin = models[0].lin
    print("linear stage:", dict(zip(M.LINEAR_COLS, np.round(lin.coef_, 4))))

    def predict_rate(F, dist):
        log_rpm = np.mean([m.predict(F) for m in models], axis=0)
        return M.to_rate(log_rpm, dist)

    # 3) validation predictions
    F_va = features.build_features(va, cities, daily, daily_qs)
    pred = predict_rate(F_va, va.distance.values)
    out = pd.DataFrame({"load_id": va.load_id, "predicted_rate": np.round(pred, 2)})
    template = pd.read_csv(data.DATA / "validation_predictions_template.csv")
    out = template[["load_id"]].merge(out, on="load_id", how="left")  # keep template order
    assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all() and len(out) == 12_000
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print("wrote validation_predictions.csv", out.shape, out.predicted_rate.describe().round(1).to_dict())

    # 4) December chart: fixed lane/equipment/weight, only the date changes.
    #    market_index / quote_signal are not given for the chart rows, so we use the *day-level*
    #    values observed in validation.csv for each date (features only, no labels);
    #    load-level wiggles (mi_dev, qs_dev) are set to 0.
    dec = pd.read_csv(data.DATA / "december_chart_inputs.csv", parse_dates=["date"])
    geo = tr.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    geo_d = tr.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    d = dec.copy()
    d["pickup_lat"], d["pickup_lon"] = geo.loc["Lexington"].values
    d["delivery_lat"], d["delivery_lon"] = geo_d.loc["Fort Wayne"].values
    d["market_index"] = d["date"].map(daily)
    d["quote_signal"] = d["date"].map(daily_qs)
    d["weight_neg"] = d["weight_missing"] = d["weight_capped"] = 0
    assert d.market_index.notna().all() and d.quote_signal.notna().all()
    F_dec = features.build_features(d, cities, daily, daily_qs)
    dec["predicted_rate"] = np.round(predict_rate(F_dec, d.distance.values), 2)
    dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
    dec.to_csv(ROOT / "data" / "december_chart_inputs_completed.csv", index=False)
    print(dec[["date", "predicted_rate"]].T.to_string(header=False))


if __name__ == "__main__":
    main()
