# Freight Rate Prediction (Spotter ML Engineer assessment)

Predicts `posted_rate` for 12,000 held-out loads (Nov-Dec 2025) from 48,000 labelled loads (Jan-Oct 2025),
plus the fixed 31-day December chart.

## Approach in one paragraph
Out-of-time problem, so validation is time-based (expanding window) plus a simulation of the 8 cities that
appear only in the validation set. Corrupted labels (1.4%) are removed from training. The model is
`log(rate/mile) = LightGBM(load-level features) + linear stage(day-level market index, day-level quote signal, time trend, weekday, quarter-end)`,
because trees cannot extrapolate a time trend into Nov-Dec. Details and results: `report.pdf`.

## Setup & run
```bash
python -m pip install -r requirements.txt
# put the 4 provided CSVs in data/ with these names:
#   train_test.csv, validation.csv, validation_predictions_template.csv, december_chart_inputs.csv

python src/validate.py        # time-based CV + unseen-city simulation  -> outputs/cv_*.csv, cv_summary.json (~4 min on 1 CPU)
python src/predict.py         # trains final model -> validation_predictions.csv, data/december_chart_inputs_completed.csv
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs_completed.csv   # -> scorer_results/candidate_december.png
python src/make_report.py     # -> report.pdf
```
(`run_all.sh` does all of the above.) Tested with Python 3.12, pandas 3, LightGBM 4.7.

## Layout
- `src/data.py` loading, input cleaning, label-anomaly filter
- `src/features.py` feature engineering (shared by train / validation / December chart)
- `src/modeling.py` `TrendGBM` (LightGBM + linear stage, backfitted) and metrics
- `src/validate.py` validation study; `src/predict.py` final fit and submission files
- `validation_predictions.csv` final predictions; `scorer_results/candidate_december.png` chart; `report.pdf` report

## Notes / assumptions
- December chart rows have no `market_index` / `quote_signal`: I use that date's day-level values observed in `validation.csv` (features only, no labels).
- Data files are git-ignored (assessment data); copy them into `data/` to reproduce.
