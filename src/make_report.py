"""Build the PDF report (figures + text) from outputs/ and scorer_results/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import data

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"; REP.mkdir(exist_ok=True)
TEAL = "#064A56"


def figures():
    tr, va = data.load_raw()
    # 1) label anomalies
    r = data.rate_ratio(tr)
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=150)
    ax.hist(np.log2(r), bins=120, color=TEAL)
    ax.axvline(np.log2(0.6), color="crimson", ls="--"); ax.axvline(np.log2(1.8), color="crimson", ls="--")
    ax.set_yscale("log"); ax.set_xlabel("log2( rate-per-mile / median for distance bin x equipment )"); ax.set_ylabel("loads (log)")
    ax.set_title("Corrupted posted_rate rows sit far outside a clean band (677 rows, 1.4%)", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(REP / "fig_anomalies.png"); plt.close(fig)
    # 2) daily market index (train + validation) with split marker
    both = pd.concat([tr[["date", "market_index"]], va[["date", "market_index"]]]).groupby("date").market_index.mean()
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=150)
    ax.plot(both.index, both.values, color=TEAL, lw=1.2)
    ax.axvline(pd.Timestamp("2025-11-01"), color="crimson", ls="--"); ax.text(pd.Timestamp("2025-11-03"), 1.3, "validation\n(Nov-Dec)", color="crimson", fontsize=8)
    ax.set_ylabel("daily mean market_index"); ax.set_title("Train is Jan-Oct, validation is Nov-Dec: an out-of-time problem", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(REP / "fig_timesplit.png"); plt.close(fig)
    # 3) CV by fold
    cv = pd.read_csv(ROOT / "outputs" / "cv_results.csv")
    a = cv[(cv.scheme == "A_time") & (cv.subset == "clean")].pivot(index="fold", columns="model", values="MAPE")
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=150)
    a.rename(index={7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}).plot.bar(ax=ax, color=["#9DAFB3", "#5FA8B5", TEAL], width=0.75, rot=0)
    ax.set_ylabel("MAPE %"); ax.set_xlabel("test month (trained on all earlier months)"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Expanding-window time validation", loc="left", fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(REP / "fig_cv.png"); plt.close(fig)


def build():
    figures()
    s = json.loads((ROOT / "outputs" / "cv_summary.json").read_text())
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13.5, spaceAfter=5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=18, textColor=colors.HexColor(TEAL))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor(TEAL), spaceBefore=8)
    P = lambda t, st=body: Paragraph(t, st)
    cl = s["A_time_clean"]; un = s["B_unseen_city"]
    def row(name, k): return [name, f'{cl[k]["MAE"]:.1f}', f'{cl[k]["MAPE"]:.2f}%', f'{cl[k]["MdAPE"]:.2f}%', f'{cl[k]["RMSE"]:.1f}']
    tbl = Table([["Model (time folds Jul-Oct, clean rows)", "MAE $", "MAPE", "MdAPE", "RMSE $"],
                 row("Baseline: median rate/mile by distance bin x equipment", "baseline (bin median)"),
                 row("LightGBM, trees see market index + time (no trend term)", "LightGBM, no trend term"),
                 row("Trend + LightGBM (final)", "Trend + LightGBM (final)")],
                colWidths=[8.3*cm, 1.9*cm, 1.9*cm, 1.9*cm, 1.9*cm])
    tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                             ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9DAFB3")),
                             ("FONTNAME", (0, 3), (-1, 3), "Helvetica-Bold"), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    img = lambda f, w=16*cm: Image(str(REP / f), width=w, height=w * 0.4)
    story = [P("Freight Rate Prediction: Validation and Split Approach", h1),
        P("Machine Learning Engineer assessment, Spotter. Candidate: Divya Dwivedi.", body),
        P("1. Problem framing", h2),
        P("<b>train_test.csv</b> covers 2025-01-01 to 2025-10-31 (48,000 loads); the 12,000 loads to predict in <b>validation.csv</b> are 2025-11-01 to 2025-12-31. "
          "This is a forecasting problem, so a random K-fold would leak the future and overstate accuracy. Two further facts shape the design: "
          "(a) the validation set contains <b>8 cities never seen in training</b> (Chicago, Charlotte, Laredo, San Diego, Knoxville, Jackson, Norfolk, Allentown; 12% of rows), and "
          "(b) the December chart rows contain no market_index or quote_signal."),
        img("fig_timesplit.png"),
        P("2. How the data was split and validated", h2),
        P("<b>Scheme A, expanding-window time folds.</b> For M = Jul, Aug, Sep, Oct: train on all months before M, test on month M. Every preprocessing step "
          "(label-anomaly filter, imputation, linear coefficients) is fit on the training months only."),
        P("<b>Scheme B, time plus new-city simulation.</b> Train on Jan-Aug with every load touching 8 randomly held-out cities removed (2 repeats); test on Sep-Oct, "
          "reporting unseen-city and seen-city rows separately. Result: MAPE "
          f'<b>{un["unseen cities"]["MAPE"]:.2f}%</b> on unseen cities vs <b>{un["seen cities"]["MAPE"]:.2f}%</b> on seen cities, so the model generalises to new cities (it uses coordinates, not city IDs).'),
        P("Metrics are computed on rows whose label passes the anomaly filter; the table in section 4 also lists all-rows numbers for transparency. "
          "No hyper-parameter tuning against the test months was done beyond the structural choices below, so the figures are not tuned to the folds."),
        P("3. Data-quality issues and handling", h2),
        P("<b>Corrupted labels:</b> 677 rows (1.4%) have a posted rate 2x-6x too high or 2x-6x too low versus the median for their distance bin and equipment. The clean band is narrow (0.8x-1.3x) and the gap is clear, so these rows are dropped from training only."),
        img("fig_anomalies.png"),
        P("<b>Weight:</b> negative values (sign flips; 292 train, 145 validation) take abs(); 47,500 is a clipping ceiling (1,191 train rows), flagged; missing (300 / 165) left as NaN with a flag. "
          "<b>market_index</b> missing (374 / 249) is filled with that day's mean index (features only, no labels). Distances are consistent with the coordinates (road/straight-line ratio about 1.18) and lane distances are stable, so no distance repair was needed. No duplicate IDs or rows."),
        P("4. Model and results", h2),
        P("Almost all of the signal is distance and equipment (rate per mile falls with distance; reefer and flatbed carry a premium). On top of that, the "
          "<b>day-level</b> market_index and quote_signal move the rate level, there is a steady upward drift (about 6% per 300 days), a weekly cycle (-0.3% Tue-Sat vs Mon) and a quarter-end lift. "
          "Tree models cannot extrapolate a drift into months they have never seen, so the final model is <b>log(rate/mile) = LightGBM(load-level features) + linear stage</b>, "
          "where the linear stage holds log market index (day-level), quote_signal (day-level), time, day-of-week and a decaying quarter-end term. "
          "The two parts are fit by backfitting; trees only see load-level features (distance, equipment, weight, coordinates, and each load's deviation from the day's index), "
          "never date-like inputs. The final model averages 3 seeds, is trained on all clean labelled rows, and uses a Huber loss on the log scale."),
        tbl, Spacer(1, 6), img("fig_cv.png"),
        P(f'On all rows including corrupted labels the final model has MAPE {s["A_time_all"]["Trend + LightGBM (final)"]["MAPE"]:.2f}% and MdAPE {s["A_time_all"]["Trend + LightGBM (final)"]["MdAPE"]:.2f}% '
          "(corrupted rows are unpredictable by construction). <b>Known weakness:</b> the August fold (4.3% MAPE) is worse than the simple baseline: with only Jan-Jul history, the market and trend effects are weakly identified. "
          "The Sep and Oct folds, which are closest to the Nov-Dec test period in training length, score 1.7% and 1.4%."),
        P("5. Fixed December chart", h2),
        P("The 31 December rows fix lane (Lexington to Fort Wayne), 360 miles, Dry Van and 32,000 lb. Because market_index and quote_signal are not provided for them, "
          "I use that date's day-level market_index and quote_signal as observed in validation.csv (feature columns only, no labels) and set the load-level deviations to zero. "
          "The chart shows a weekly cycle (Thursday peak, Sunday/Monday trough) and a rise toward quarter end. Holiday effects cannot be learned from Jan-Oct data and are not modelled."),
        Image(str(ROOT / "scorer_results" / "candidate_december.png"), width=16*cm, height=16*cm * 0.47),
        P("<i>Chart produced by the provided score.py.</i>")]
    SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                      title="Freight Rate Prediction - Validation Report", author="Divya Dwivedi").build(story)


if __name__ == "__main__":
    build()
