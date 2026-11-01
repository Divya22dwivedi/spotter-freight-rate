#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
python src/validate.py
python src/predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs_completed.csv
python src/make_report.py
