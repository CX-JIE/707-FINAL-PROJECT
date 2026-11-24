# A Simple, Interpretable Model versus a Black Box for Predicting In-Hospital Mortality in ICU Patients

707 Final Project · **Team:** Junjie Cai, Yifei Chen, Jiahui Hu, Yuyao Lei

We compare a seven-predictor logistic GAM with XGBoost on a broad first-24-hour
feature set for predicting in-hospital death of ICU patients. A shallow decision
tree, LASSO/Ridge logistic regression, XGBoost on the same 7 and 15 predictors,
and the clinical scores SAPS-I and SOFA are also included. Models are evaluated
with stratified 5-fold nested cross-validation on set A and tested on the
held-out sets B and C. The evaluation covers AUROC, AUPRC, Brier score,
calibration, threshold metrics, decision curves, paired bootstrap differences,
and SHAP/PDP explanations.

**Headline (set A, out-of-fold):** AUROC 0.736 for GAM-7 vs 0.826 for XGB-full.
The paired difference is 0.091 (95% bootstrap CI 0.072–0.110).

## Deliverables

| File | Content |
|:---|:---|
| [`Final_Report.md`](Final_Report.md) | Final report (renders on GitHub; figures from `results/`) |
| `Final_Report.html` | Same report as one self-contained HTML file (figures embedded) |
| [`Proposal_finpro.md`](Proposal_finpro.md) | Project proposal |
| [`code/`](code/) | Analysis scripts |
| [`results/`](results/) | All tables (CSV/JSON) and figures (PNG) used in the report |

## Data

The project uses the **PhysioNet/Computing in Cardiology Challenge 2012**
dataset, v1.0.0: <https://physionet.org/content/challenge-2012/1.0.0/>.
It is distributed under the **Open Data Commons Attribution License v1.0
(ODC-BY 1.0)**. The data are **not** included in this repository. To download
them into `data/`:

```bash
mkdir -p data && cd data
for f in set-a.tar.gz set-b.tar.gz set-c.tar.gz Outcomes-a.txt Outcomes-b.txt Outcomes-c.txt; do
  curl -LO "https://physionet.org/files/challenge-2012/1.0.0/$f"
done
cd ..
```

If you use the data, please cite:

- Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting in-hospital mortality
  of ICU patients: the PhysioNet/Computing in Cardiology Challenge 2012.
  *Computing in Cardiology*. 2012;39:245–248.
- Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of
  a new research resource for complex physiologic signals. *Circulation*.
  2000;101(23):e215–e220.

## How to reproduce

```bash
conda env create -f environment.yml
conda activate icu707
bash run_all.sh                    # features -> data quality -> models -> supplement -> report
```

`run_all.sh` extracts the archives into `data_raw/` and builds features in
`data_derived/`. It writes all results to `results/` and finally generates
`Final_Report.md` and `Final_Report.html`. The full model run (nested CV with
2,000 bootstrap resamples) takes a while. To regenerate only the report from
existing results:

```bash
python code/07_final_report.py              # report only
python code/07_final_report.py --with-code  # report + analysis source as an appendix
```

| Script | Purpose | Main outputs |
|:---|:---|:---|
| `code/featuresets.py` | Definitions of the small-7, core-15, and full feature sets | — |
| `code/01_build_features.py` | Parse raw records and build first-24 h summaries (mean/min/max/last/count) | `data_derived/features_{a,b,c}.csv` |
| `code/02_data_quality.py` | Cohort summary, missingness, distributions, redundancy | `results/cohort_summary.json`, `*missingness*.csv`, figures |
| `code/03_models.py` | Nested CV for all models, clinical scores, paired bootstrap, B/C test, SHAP/PDP | `results/performance_*.csv`, `paired_bootstrap.csv`, figures |
| `code/05_supplement.py` | Ridge-15 and model-complexity summary | `results/ridge_coefficients.csv`, `model_complexity.csv` |
| `code/07_final_report.py` | Final report from saved results | `Final_Report.md`, `Final_Report.html` |
| `code/04_report.py`, `code/06_report_zh.py` | Quick-look HTML summaries (English / Chinese) | `report.html`, `report_zh.html` |

Seeds are fixed (outer folds 707, inner folds 708). Package versions used for
the reported run are recorded in `results/run_info.json`.

## Team contributions

| Member | Responsibility |
|:---|:---|
| Junjie Cai | Data access and license, cohort construction, missingness, data provenance |
| Yifei Chen | GAM, shallow decision tree, LASSO; coefficient and partial-effect plots |
| Jiahui Hu | XGBoost and clinical-score benchmarks; feature importance and post hoc explanations |
| Yuyao Lei | Shared validation framework, consistent splits, discrimination/calibration/decision-curve comparisons |
