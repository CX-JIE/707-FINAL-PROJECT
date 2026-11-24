"""Generate the final report (Final_Report.md, plus Final_Report.html if the
`markdown` package is installed) from the saved analysis outputs in results/.

Usage (from the project root, after `bash run_all.sh`):
    python code/07_final_report.py              # report only
    python code/07_final_report.py --with-code  # also append the analysis source

The script only reads saved results, data_derived/ and data/Outcomes-*.txt;
it never refits a model. Narrative text lives in this file; every number in
the report is computed from the result files at generation time.
"""
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
include_code = "--with-code" in sys.argv

required = ["performance_setA_cv.csv", "paired_bootstrap.csv", "run_info.json", "threshold_metrics.csv",
            "performance_external_sets.csv", "selected_hyperparameters.json", "model_complexity.csv"]
missing = [f for f in required if not (R / f).exists()]
if missing:
    sys.exit(f"Missing analysis outputs {missing}. Run `bash run_all.sh` first.")

OUT = []


def emit(text):
    """Append static Markdown text."""
    OUT.append(text.strip("\n"))


def md(text):
    """Append Markdown text generated from results."""
    OUT.append(text.strip("\n"))


def csv(name):
    return pd.read_csv(R / name)


def table(df, digits=3):
    """Render a DataFrame as a Markdown pipe table."""
    names = {"GAM-7 + NA indicators": "GAM-7 + NA/vent", "XGB-full (native NA)": "XGB-full native NA"}
    out = df.astype(object).map(lambda value: "—" if pd.isna(value) else
                                f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else
                                names.get(str(value), str(value)))
    cols = [str(c) for c in out.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(":---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(v.replace("|", "\\|") for v in row) + " |" for row in out.values.tolist()]
    md("\n".join(lines))


# ------------------------------------------------------------------ shared inputs
perf = csv("performance_setA_cv.csv")
pv = perf.set_index("model")
boot = csv("paired_bootstrap.csv")
ri = json.loads((R / "run_info.json").read_text())
if ri["quick"]:
    sys.exit("Quick/test model results cannot be used for the final report.")
cohort = []
for st in "abc":
    x = pd.read_csv(ROOT / f"data_derived/features_{st}.csv")
    ytab = pd.read_csv(ROOT / f"data/Outcomes-{st}.txt")
    assert x.RecordID.is_unique and ytab.RecordID.is_unique
    assert set(x.RecordID) == set(ytab.RecordID)
    cohort.append({"Set": st.upper(), "Records": len(x), "Deaths": int(ytab["In-hospital_death"].sum()),
                   "Mortality (%)": 100*ytab["In-hospital_death"].mean(), "Median age": x.Age.median()})
A = pd.read_csv(ROOT / "data_derived/features_a.csv")
primary = boot.set_index("comparison").loc["XGB-full minus GAM-7"]
sizes = json.loads((R / "featureset_sizes.json").read_text())
main_models = ["GAM-7", "XGB-full", "XGB-7", "Tree-7", "LASSO-15", "SAPS-I", "SOFA"]
main_perf = perf.set_index("model").loc[main_models].reset_index()

# ------------------------------------------------------------------ report body
emit(r'''
# A Simple, Interpretable Model versus a Black Box for Predicting In-Hospital Mortality in ICU Patients

**707 Final Project — Final Report**

**Team:** Junjie Cai, Yifei Chen, Jiahui Hu, Yuyao Lei

**Contents:** [Summary](#summary) · [Data](#data) · [Models and validation](#models) ·
[Results](#results) · [Interpretation](#interpretation) · [Limitations and conclusions](#conclusions) ·
[Reproducibility](#reproducibility) · [References](#references) · [AI statement](#ai-statement) ·
[Supplementary materials](#sec-supplement)
''')

emit(r'''
<a id="summary"></a>

# Summary

We compared a seven-predictor logistic generalized additive model (GAM-7) with
XGBoost using a broader first-24-hour feature set (XGB-full), with additional
interpretable models and clinical-score benchmarks. The outcome was in-hospital
death. Model development used nested cross-validation in set A; sets B and C
were held out from training and tuning.
''')

md(f"Set A contained **{ri['n']:,} records and {ri['deaths']} deaths ({ri['mortality_pct']:.2f}%)**. "
   f"The pooled out-of-fold AUROC was **{pv.loc['GAM-7','AUROC']:.3f} for GAM-7** and "
   f"**{pv.loc['XGB-full','AUROC']:.3f} for XGB-full**. Their paired difference was "
   f"{primary.dAUROC:.3f} (conditional bootstrap 95% CI {primary.dAUROC_lo:.3f} to {primary.dAUROC_hi:.3f}). "
   f"Brier scores were {pv.loc['GAM-7','Brier']:.3f} and {pv.loc['XGB-full','Brier']:.3f}, respectively.")

emit(r'''
The tested compact GAM sacrificed discrimination for a simpler additive
representation. This is a comparison of these fitted specifications, not proof
that every interpretable model must perform worse. No equivalence margin was
specified, so the study does not establish equivalence or noninferiority.
Post hoc explanations help describe XGBoost's predictions; neither these
explanations nor the decision curves establish clinical benefit or causal effects.

<a id="data"></a>

# Data, predictors, and quality checks

## Source and eligible cohort

We used PhysioNet/Computing in Cardiology Challenge 2012 v1.0.0, including
`set-a`, `set-b`, and `set-c` and their corresponding outcome files [1,2,3,5]. The unit of
analysis was an ICU stay record. The source cohort required an ICU stay of at
least 48 hours; therefore, this analysis does **not** represent all ICU admissions,
including those dying or discharged before that eligibility threshold [3]. Although
the source describes adult admissions, a few recorded ages are below 18; the
analysis retains the supplied cohort and treats this as a data-quality limitation.
''')

md("**Table 1. Cohort characteristics and mortality by PhysioNet data set.**")
table(pd.DataFrame(cohort), 2)

emit(r'''
No record was excluded solely because predictors were missing. Outcome linkage
was checked one-to-one by RecordID, with binary outcome coding and matching
record counts. Sets B and C are **same-source held-out test sets**, not independent
geographic or prospective external validation cohorts.

## Feature construction

Predictors use measurements between 00:00 and 24:00 inclusive. Every raw record
file was checked to be in chronological order, so first/last summaries follow
the recorded time sequence. Negative sentinel values for static descriptors and
weight, and values outside the broad plausibility ranges listed in
`code/01_build_features.py`, are set to missing; FiO2 percentages are converted
to fractions. These screens do not guarantee removal of every erroneous measurement.

The small-7 features are age, minimum GCS, mean heart rate, mean combined MAP,
mean respiratory rate, minimum SaO2, and maximum BUN. Core-15 adds minimum PaO2,
maximum creatinine, maximum WBC, minimum platelets, minimum hematocrit, maximum
glucose, maximum bilirubin, and mean temperature. Combined MAP uses invasive
measurements when available, otherwise non-invasive measurements.

The seven-variable set follows the proposal (Section 3.2): age represents
demographics; GCS represents neurological status; heart rate and MAP represent
hemodynamics; respiratory rate and SaO2 represent respiratory status/oxygenation;
and BUN represents renal-related physiology [1,3]. This is a compact, domain-based
design choice, not the first seven variables selected by a statistical ranking.
The code performs no outcome-based selection of these seven variables. However,
the proposal provides no formal optimization of this subset or contemporaneous
selection audit beyond its stated rationale. We therefore do not claim it is
the optimal seven-variable set. High respiratory-variable missingness is an
important weakness of this choice.

The fifteen-variable set incorporated additional physiological and laboratory
measurements, including PaO₂ to further characterize respiratory status and
oxygenation; creatinine and bilirubin to assess renal and hepatic function;
WBC, platelets, and hematocrit to capture inflammatory and hematological
status; and glucose and temperature to reflect metabolic and physiological
conditions [1,3]. Together, these variables provided a broader range of clinical
information, allowing exploratory investigation of the potential benefits of
incorporating additional predictors through comparison with an intermediate
feature set.
''')

full_cols = [c for c in A if c not in {"RecordID", "last_obs_min", "n_obs_24h"} and not c.endswith("_first")]
constant = A[full_cols].fillna(A[full_cols].median()).nunique().le(1)
md(f"The full set contains **{len(full_cols)} input columns**: four static descriptors, "
   "mean/minimum/maximum/last/count for 37 time-series variables, three combined-MAP summaries, "
   f"and total urine output. In set A, {int(constant.sum())} of these columns are constant after median "
   f"imputation ({', '.join(constant[constant].index)}). Thus, {len(full_cols)} is an input-column count, "
   "not a count of independent informative predictors. RecordID, the all-record last timestamp, "
   "the overall observation count, and the unused first-value summaries are excluded from the model inputs.")

emit(r'''
The proposal established the main variable groups and some aggregation examples,
but did not document every final aggregation or sensitivity analysis. The
last/count summaries and the additional missingness/ventilation sensitivity are
reported as implementation choices, without claiming they were all fixed before
outcomes were examined. The proposal is submitted unchanged rather than
retrospectively rewritten to match the implementation.

## Missingness and residual artifacts
''')

miss = csv("core15_missingness_setA.csv")
md("**Table 2. Missingness and distribution summaries for the 15 core predictors in set A.**")
table(miss[["label", "missing_pct", "median", "q1", "q3"]].rename(columns={
    "label": "Predictor", "missing_pct": "Missing (%)", "median": "Median", "q1": "Q1", "q3": "Q3"}), 1)
vent = A.MechVent_max.eq(1)
rr_missing = A.RespRate_mean.isna()
md(f"Respiratory rate is missing for {rr_missing.mean()*100:.1f}% of set A, "
   f"including {(rr_missing & vent).sum():,} of {vent.sum():,} records with documented ventilation. "
   f"However, {(rr_missing & ~vent).sum():,} records without documented ventilation also lack respiratory rate. "
   "Missing respiratory rate is therefore not equivalent to ventilation. A median-valued "
   "predictor may be an observed value or an imputed value; it cannot identify ventilation by itself.")
map_flag = A.MAP_mean.gt(200) & A.NIMAP_mean.between(40, 120)
md(f"A residual artifact audit flags {int(map_flag.sum())} set-A records with mean invasive MAP "
   "above 200 mmHg while mean non-invasive MAP is between 40 and 120 mmHg. These are descriptive "
   "discordance flags, not validated exclusion rules. They remain in the primary analysis because "
   "the source does not identify which values should be corrected. Results may be sensitive to this "
   "choice; no robustness claim is made without an additional cleaning sensitivity analysis.")

emit(r'''
![Availability of measurements in the first 24 hours. Dark bars identify variables represented in core-15.](results/fig_missingness_all.png)

Missingness may encode treatment and measurement practices as well as patient
condition. The sensitivity labeled `GAM-7 + NA indicators` in result files adds
two missingness indicators **and a ventilation predictor** (10 inputs in total).
It is abbreviated GAM-7 + NA/vent here. Its performance change cannot be
attributed exclusively to missingness handling. Correlated physiological
variables and repeated summaries remain in the full feature set; they can share
or redistribute post hoc importance.

<a id="models"></a>

# Models and validation

## Model roles and figure selection

GAM-7, Tree-7, and XGB-7 use **exactly the same seven constructed predictors**.
LASSO-15, Ridge-15, and XGB-15 use **exactly the same core-15 predictors**.
XGB-full uses the full first-24-hour input set. The primary GAM-7 versus XGB-full
comparison intentionally changes both representation and model family; the
GAM-7 versus XGB-7 comparison holds the input set constant. There is no claim
that all models use 15 predictors.

**Table 3. Roles of the prespecified, supplementary, exploratory, and sensitivity models.**

| Role | Models | Purpose |
|:---|:---|:---|
| Primary | GAM-7, XGB-full | Main compact/additive versus broader boosted-model comparison |
| Clinical benchmarks | SAPS-I, SOFA | Supplied clinical scores |
| Supplementary | XGB-7 | Compare model families using the same seven inputs |
| Exploratory | Tree-7, LASSO-15, Ridge-15, XGB-15 | Simple alternatives, penalty comparison, intermediate feature set |
| Sensitivity | XGB-full native NA, GAM-7 + NA/vent | Alternative missing-value handling/additional indicators |

Ridge-15 assesses whether the core-15 logistic-regression results change with L2
rather than L1 regularization. XGB-15 provides an intermediate feature-set
comparison between XGB-7 and XGB-full. Neither is a separate primary hypothesis.

For readability, the main ROC, calibration, and decision-curve figures show
GAM-7, XGB-full, XGB-7, LASSO-15, Tree-7, and the two clinical benchmarks.
**Ridge-15, XGB-15, and the sensitivity models are omitted from those figures,
but included in the complete performance tables in [Appendix S1](#sec-supp-performance).** Thus, some exploratory
models are plotted and others are not; omission is not because XGBoost is a
black box. SHAP was computed for XGB-full and XGB-7, not XGB-15. XGB-15 was
evaluated in set-A cross-validation but was not refitted for B/C testing.

## Fitting and tuning

- GAM-7 uses penalized cubic splines, 10 basis functions per continuous
  predictor [6]. A common smoothing parameter is selected from 11 log-spaced values
  between 0.01 and 1,000 using inner-fold log-loss. A candidate whose fit raises
  an error receives a missing score and cannot be selected (scikit-learn default);
  convergence warnings were suppressed during fitting and were not logged.
- Tree-7 tunes maximum depth (2, 3, 4) and minimum leaf size (25, 50, 100).
- LASSO-15 and Ridge-15 use fold-internal standardization and tune inverse
  regularization strength C over 13 and 16 values, respectively.
- XGBoost uses histogram-based boosting and 25 randomly sampled configurations
  per inner search, spanning learning rate, number of trees, maximum depth,
  child weight, row/column sampling, L2 penalty, and split penalty. The maximum
  searched tree count is 600. Median imputation is primary; native missing-value
  handling is a separate sensitivity model [7].

Set A uses stratified five-fold outer cross-validation with the same outer
folds for every model (seed 707). Each outer training set has an inner stratified
five-fold search (seed 708). Imputation, scaling, and tuning occur only within
the relevant training folds [8]. After evaluation, selected models are tuned on all
of set A and refitted for interpretation and B/C prediction; these fitted
interpretation models are distinct from the outer-fold fits used for performance.

Selected configurations for each outer fold are reported in
[Appendix S4](#sec-supp-tuning).

All primary fitted models, including XGB-full, use training-fold **median
imputation**. For a shared predictor and identical training split, independently
fitted imputers therefore produce the same replacement value. LASSO and Ridge
add standardization after imputation. Native XGBoost missing-value routing is
restricted to the explicitly labeled sensitivity model. Consequently, native
versus median handling is not mixed into the primary model comparison, although
the broader feature set can still contain additional missingness information
through measurement counts.

## Clinical scores and metrics

SAPS-I and SOFA are taken from the supplied outcomes files, with negative values
treated as missing. Their **raw scores** are used consistently for AUROC,
including paired bootstrap comparisons [9,10]. For AUPRC (average precision), Brier
score, calibration, thresholds, and decision curves, scores are converted to
probabilities by logistic recalibration fitted only on outer training folds.
The supplied score windows were not independently reconstructed from raw
measurements; their temporal comparability to our first-24-hour predictors is
therefore uncertain.

Calibration-in-the-large is an intercept estimated with slope fixed at one;
calibration slope is estimated with a free intercept. Thresholds are the
corresponding outer training-fold mortality prevalence and 0.30. They are
illustrative evaluation thresholds, not validated treatment rules.

Paired percentile bootstrap intervals use 2,000 patient resamples of fixed
out-of-fold predictions, using the same eligible records for both models in
each comparison. They do not capture retraining, tuning, or fold-assignment
uncertainty. Exploratory comparisons are not adjusted for multiplicity.

<a id="results"></a>

# Results

## Main set-A performance

The main tables show the same five fitted models and two clinical benchmarks
as the main figures. [Appendix S1](#sec-supp-performance) reports all eleven
models, including Ridge-15, XGB-15, and both sensitivity analyses.
''')

md("**Table 4. Main set-A out-of-fold predictive performance.**")
table(main_perf[["model", "n", "AUROC", "AUPRC", "Brier"]].rename(columns={"model":"Model", "n":"N", "AUPRC":"AP"}))

emit(r'''
Higher AUROC and AP indicate better discrimination/ranking; lower Brier indicates
better probability accuracy. Model values use all set-A records; each clinical
score uses its available-score subset. Cross-row differences involving clinical
scores should therefore be read from the **paired comparisons below**, not
calculated by subtracting unmatched full-cohort estimates.
''')

fv = csv("fold_variability.csv")
cal = main_perf[["model", "cal_intercept", "cal_slope"]].merge(fv[["model", "AUROC_sd"]], on="model")
md("**Table 5. Calibration and outer-fold AUROC variability for the main models.**")
table(cal.rename(columns={"model":"Model", "cal_intercept":"Cal. intercept", "cal_slope":"Cal. slope", "AUROC_sd":"Fold AUROC SD"}))

emit(r'''
![Out-of-fold ROC and decile calibration. The clinical-score curves use available-score subsets, with N stated in the ROC legend. These are descriptive overlays; formal paired comparisons use matched records.](results/fig_roc_calibration.png)

## Paired performance differences

Selected primary, same-input, tree, and clinical-benchmark comparisons are shown
below. [Appendix S2](#sec-supp-paired) includes all exploratory and sensitivity
comparisons, with the same sign conventions and conditional-bootstrap limitations.
''')

bt = boot.copy()
bt["AUROC difference (95% CI)"] = bt.apply(lambda r: f"{r.dAUROC:.3f} ({r.dAUROC_lo:.3f}, {r.dAUROC_hi:.3f})", axis=1)
bt["Brier difference (95% CI)"] = bt.apply(lambda r: f"{r.dBrier:.4f} ({r.dBrier_lo:.4f}, {r.dBrier_hi:.4f})", axis=1)
# Separate tables keep the uncertainty intervals legible.
main_pairs = ["XGB-full minus GAM-7", "XGB-7 minus GAM-7", "XGB-full minus XGB-7", "XGB-full minus Tree-7"]
main_bt = bt[bt.comparison.isin(main_pairs) | bt.comparison.str.endswith(("minus SAPS-I", "minus SOFA"))]
md("**Table 6. Selected paired AUROC differences with 95% bootstrap intervals.**")
table(main_bt[["comparison", "n", "AUROC difference (95% CI)"]].rename(columns={"comparison":"First minus second", "n":"N"}))
md("**Table 7. Selected paired Brier-score differences with 95% bootstrap intervals.**")
table(main_bt[["comparison", "Brier difference (95% CI)"]].rename(columns={"comparison":"First minus second"}))

emit(r'''
Positive AUROC differences favor the first model; negative Brier differences
favor the first model. Interval interpretation is conditional on the fitted
out-of-fold predictions and should not be treated as a fully refitted
cross-validation uncertainty analysis.
''')

md(f"With seven inputs, XGB-7 achieved AUROC {pv.loc['XGB-7','AUROC']:.3f}; "
   f"XGB-15 achieved {pv.loc['XGB-15','AUROC']:.3f}; and XGB-full achieved "
   f"{pv.loc['XGB-full','AUROC']:.3f}. These ordered feature-set comparisons suggest that "
   "the broader input representation was useful under the implemented tuning procedure. "
   "They do not isolate individual features, monitoring frequency, or interactions: "
   "feature content and fitted hyperparameters change together. The XGB-7 versus GAM-7 "
   "difference compares model specifications on the same inputs, not a pure estimate "
   "of interaction benefit. These AUROC differences must not be interpreted as a causal percentage decomposition.")
md(f"Ridge-15 and LASSO-15 yielded AUROCs of {pv.loc['Ridge-15','AUROC']:.3f} and "
   f"{pv.loc['LASSO-15','AUROC']:.3f}. Their closeness is descriptive, not a formal equivalence result. "
   f"The expanded GAM-7 + NA/vent sensitivity achieved {pv.loc['GAM-7 + NA indicators','AUROC']:.3f}; "
   "it includes information absent from GAM-7 and should not be called a pure imputation comparison.")

emit(r'''
## Threshold performance and decision curves

The main tables summarize the seven plotted models. Full confusion counts and
all four threshold metrics, including specificity and NPV, appear for all eleven
models in [Appendix S3](#sec-supp-thresholds).
''')

thr = csv("threshold_metrics.csv")
for part, (threshold, rows) in zip("ab", thr[thr.model.isin(main_models)].groupby("threshold", sort=False)):
    md(f"**Table 8{part}. Threshold classification results for the main models (threshold: {threshold}).**")
    table(rows[["model", "TP", "FP", "FN", "TN", "sens", "PPV"]].rename(columns={"model":"Model", "sens":"Sensitivity"}))

emit(r'''
![Decision curves with consistent denominators. Left: all records, five main fitted models. Right: records with both clinical scores available, all seven plotted models. Each panel has its own matched-cohort treat-all reference.](results/fig_decision_curves.png)

Net benefit weights false positives by threshold odds, $t/(1-t)$. Within a panel,
all curves use the same records and prevalence. Panels should not be treated as
the same target population. The threshold range 0.05–0.50 is illustrative; no
specific intervention, treatment effect, or resource constraint was modeled.
Net-benefit differences represent weighted true-positive equivalents, not
observed additional lives saved.

## Same-source held-out sets B and C
''')

ext = csv("performance_external_sets.csv")
for part, (st, rows) in zip("ab", ext.groupby("set")):
    out = rows[["model", "n", "AUROC", "AUPRC", "Brier"]].rename(columns={"model":"Model", "n":"N", "AUPRC":"AP"})
    md(f"**Table 9{part}. Same-source held-out predictive performance — set {st}.**")
    table(out.astype(object).where(out.notna(), "Not evaluated"))

emit(r'''
Only raw-score AUROC was evaluated for the clinical benchmarks on B/C; missing
probability metrics are not zeros. XGB-15 and the two sensitivity models were
not included in this held-out evaluation. B/C were not used for tuning, and
higher held-out AUROC than cross-validated AUROC would not prove absence of
overfitting: population composition and sampling variation can also differ.

<a id="interpretation"></a>

# Interpretation

## Additive effects and model complexity
''')

md("**Table 10. Model inputs and fitted complexity.**")
table(csv("model_complexity.csv")[["model", "predictors", "size"]].rename(columns={"model":"Model", "predictors":"Inputs", "size":"Fitted complexity"}))

emit(r'''
Tree rules, additive smooths, and linear coefficients are directly inspectable
representations. They are not automatically clinically reliable. XGBoost requires
post hoc summaries; its interpretability is not equivalent to that of a small
decision tree or an additive model.

![GAM partial effects and binned XGBoost SHAP dependence, fitted on all of set A. Curves are centered for display; GAM smooths and SHAP values are different mathematical quantities. GAM bands are approximate conditional intervals and do not include model-selection uncertainty.](results/fig_partial_effects.png)

These curves describe fitted associations on the development data. GCS,
respiratory rate, and oxygen saturation may also reflect treatment or recording
practices. Sedation, surgical recovery, and clinical measurement decisions were
not separately measured or tested, so they cannot be established as mechanisms
from these plots. Last-value summaries describe recent observed state, not an
explicit trajectory or rate of recovery.

## Linear coefficients and tree rules
''')

lc = csv("lasso_coefficients.csv").merge(csv("ridge_coefficients.csv")[["feature", "coef_per_SD"]], on="feature", suffixes=("_L1", "_L2"))
md("**Table 11. LASSO and Ridge coefficients and LASSO selection frequency.**")
table(lc[["feature", "coef_per_SD_L1", "coef_per_SD_L2", "selected_in_outer_folds_of_5"]].rename(columns={
    "feature":"Feature", "coef_per_SD_L1":"LASSO / SD", "coef_per_SD_L2":"Ridge / SD", "selected_in_outer_folds_of_5":"LASSO folds / 5"}))

emit(r'''
Coefficients are conditional log-odds changes per training-set standard deviation
after imputation and scaling. LASSO and Ridge impose linear effects; they do not
estimate U-shaped relationships unless those relationships are explicitly encoded.
Selection frequency is descriptive across the five outer training fits, not a
validated stability-selection probability.
''')

md("```text\n" + (R / "tree_rules.txt").read_text() + "\n```")

emit(r'''
Tree thresholds apply after median imputation. Leaf values printed here are
training-sample class weights/counts, not held-out performance estimates.

## XGBoost and SHAP

TreeExplainer was used for XGB-full and XGB-7, with model-output contributions
on the log-odds scale [4,11]. Explanations are computed on the full set-A refits and are
descriptive in-sample analyses, not out-of-fold explanation validation. XGB-15
has performance estimates but **no SHAP analysis in this project**.

![XGB-full SHAP summary for the 20 largest mean absolute contributions. Color denotes feature value, and horizontal position denotes contribution to fitted log-odds relative to the explainer baseline.](results/fig_shap_summary_xgb_full.png)
''')

imp = csv("xgb_full_shap_importance.csv")
md("The highest-ranked full-model inputs by mean absolute SHAP are " + ", ".join(imp.feature.head(6)) + ". "
   "These rankings describe attribution under the fitted model and explanation method. "
   "Correlated summaries can share attribution; a high rank does not establish a causal effect "
   "or prove that removing the feature would reduce predictive performance.")
md("**Table 12. Ten highest-ranked XGB-full predictors by mean absolute SHAP.**")
table(imp.head(10)[["feature", "mean_abs_SHAP"]].rename(columns={"feature":"Feature", "mean_abs_SHAP":"Mean absolute SHAP"}), 4)

emit(r'''
Measurement counts can carry information about data availability and clinical
practice. An ablation study would be needed to attribute performance improvement
specifically to count features, last values, or any other feature family.

![XGB-full partial dependence and individual conditional expectation curves for the seven compact-model predictors.](results/fig_pdp_xgb_full.png)

Partial dependence replaces one input while leaving other inputs unchanged.
Because this feature set contains correlated summaries (for example minimum,
maximum, mean, and last values of the same variable), this can create unrealistic
combinations. These curves are model summaries, not simulated clinical interventions.

## XGB-7 interactions
''')

md("**Table 13. Leading pairwise SHAP interactions for XGB-7.**")
table(csv("xgb7_shap_interactions.csv").head(10).rename(columns={"var1":"Feature 1", "var2":"Feature 2", "mean_abs_interaction":"Mean absolute pair contribution"}), 4)

emit(r'''
The pair statistic is twice the mean absolute off-diagonal SHAP interaction
value, counting both symmetric entries. It is an attribution-magnitude summary,
not a percentage of model performance. Absolute main-effect and total-SHAP
magnitudes cannot be divided into a valid percentage of signal explained:
signed contributions can cancel, and absolute values are not additive. The
project therefore makes no claim that an additive model captures a particular
percentage of the full signal.

<a id="conclusions"></a>

# Limitations and conclusions

The comparison is limited by a single retrospective data source, a cohort
selected to have ICU stays of at least 48 hours, substantial missingness,
residual measurement artifacts, and uncertainty about supplied clinical-score
windows. No alternate MAP-cleaning analysis or feature-family ablation was
performed. Native missing-value handling and added indicators address only
part of these issues.

The tuning search is finite. Selecting the maximum searched number of trees
indicates a search boundary, not evidence that increasing it must improve
performance or strengthen the conclusion. Other compact model specifications,
feature selections, and tuning ranges could give different comparisons.
Bootstrap intervals condition on existing predictions; repeated nested
cross-validation or full refitting would be needed to assess additional
training variability. Multiple exploratory comparisons and in-sample
explanation plots should be interpreted accordingly.

Within the tested specifications, XGB-full offers better discrimination and
probability accuracy than GAM-7, while GAM-7 offers a simpler additive
representation. The seven-input and core-15 XGBoost comparisons provide
context for this trade-off, without identifying a causal source of the gain.
Neither predictive performance nor post hoc interpretability establishes that
a model is ready for clinical deployment.

<a id="reproducibility"></a>

# Reproducibility and project organization

The analysis is a sequence of numbered Python scripts in `code/`, run in the
order 01 → 02 → 03 → 05 by `run_all.sh`:

| Script | Purpose |
|:---|:---|
| `01_build_features.py` | Parse raw records and build first-24-hour features for sets A, B, and C |
| `02_data_quality.py` | Cohort summary, missingness, distributions, redundancy |
| `03_models.py` | Nested cross-validation, paired bootstrap, B/C evaluation, SHAP/PDP |
| `05_supplement.py` | Ridge-15, model-complexity summary |
| `featuresets.py` | Shared definitions of the small-7, core-15, and full feature sets |

The Python environment is specified in `environment.yml` (conda environment
`icu707`). With the PhysioNet files in `data/` (see `README.md`):

- `bash run_all.sh`: extract data, build features, fit and evaluate all models,
  and write every table and figure to `results/`.
- `python code/07_final_report.py`: generate this report (`Final_Report.md`,
  and a self-contained `Final_Report.html` when the `markdown` package is
  installed) from `results/`. It only reads saved results; it does not refit models.
- `python code/07_final_report.py --with-code`: additionally append the
  analysis source code as an appendix.

`run_all.sh` runs the report step automatically at the end. All paths are
relative to the project root. `code/04_report.py` and
`code/06_report_zh.py` produce quick-look HTML summaries and are not part of
this report.
''')

md("**Recorded modeling versions:** " + "; ".join(f"{k} {v}" for k, v in ri["versions"].items()) + ".")

emit(r'''
<a id="references"></a>

# References

1. Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting in-hospital mortality
   of ICU patients: the PhysioNet/Computing in Cardiology Challenge 2012.
   *Computing in Cardiology*. 2012;39:245–248.
2. Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of
   a new research resource for complex physiologic signals. *Circulation*.
   2000;101(23):e215–e220.
3. PhysioNet. [Predicting Mortality of ICU Patients: PhysioNet/Computing in
   Cardiology Challenge 2012, version 1.0.0](https://physionet.org/content/challenge-2012/1.0.0/).
   Data distributed under the Open Data Commons Attribution License v1.0.
4. SHAP documentation. [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
   Consulted for SHAP and interaction-value interpretation.
5. Pollard T, et al. PhysioNet as a global platform for biomedical research.
   *Nature Health*. 2026;1:792–795.
   [doi:10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).
   Current standard PhysioNet citation requested by the dataset page.
6. Hastie T, Tibshirani R. Generalized additive models. *Statistical Science*. 1986;1(3):297–310.
7. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016:785–794.
8. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7(1):91.
9. le Gall J-R, et al. A simplified acute physiology score for ICU patients. *Critical Care Medicine*. 1984;12:975–7.
10. Vincent JL, et al. The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. *Intensive Care Medicine*. 1996;22(7):707–10.
11. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. In: *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. 2017:4765–4774.

<a id="ai-statement"></a>

# AI Statement

AI tools were used to support proposal organization, code development and debugging, and the preparation of the final report and presentation. Our team established the research question, study design, predictor sets, modeling approaches, validation strategy, and evaluation metrics. AI assistance was primarily used to refine the organization of the proposal, facilitate programming, improve the clarity of written materials, and synthesize key findings for presentation. The team was also responsible for defining the analytical objectives and workflow, reviewing the implementation, and interpreting the final results.

The use of AI offered several practical advantages. In particular, AI-assisted programming improved efficiency by helping identify potential sources of coding errors, suggesting alternative debugging strategies, and accelerating the development and refinement of analytical code. AI also facilitated the communication of complex results by helping condense the detailed report into a more focused presentation narrative. Nevertheless, the project also highlighted important limitations of AI assistance. AI-generated code and explanations were not always fully consistent with the intended analytical workflow and could contain errors or rely on assumptions that were inappropriate for the collaborative computational environment. In addition, AI could oversimplify methodological concepts, overlook important analytical details, or produce interpretations that were not fully supported by the numerical results.

Overall, AI tools helped reduce the time required for structural organization, programming and debugging, and scientific writing. This allowed our team to devote greater attention to evaluating analytical decisions, understanding the underlying statistical methods, and critically interpreting the results. The team retained responsibility for all methodological decisions, analyses, and conclusions presented in this report.


<a id="sec-supplement"></a>

# Supplementary Materials / Appendix

These materials are part of the report. Tables are generated from the same
result files as the main text; no additional models are fitted here. S1-S3
provide complete results, while S4 documents the selected tuning configurations.
The optional analysis-code listing follows these materials only when the report
is generated with `--with-code`.

<a id="sec-supp-performance"></a>

## S1. Complete model performance

**Table S1. Set-A out-of-fold performance for all models.** AUROC for SAPS-I and
SOFA uses raw scores; AP and Brier use fold-trained recalibrated probabilities.
Clinical-score sample sizes differ, so comparisons with these scores should use
the matched-record differences in S2. Lower Brier is better.
''')

table(perf[["model", "n", "AUROC", "AUPRC", "Brier"]].rename(columns={"model":"Model", "n":"N", "AUPRC":"AP"}), 4)

emit(r'''
**Table S2. Calibration and variation across outer folds.** Calibration-in-the-large
fixes the slope at one; calibration slope is estimated with a free intercept.
Fold standard deviations describe variation, not confidence intervals.
''')

supp_cal = perf[["model", "cal_intercept", "cal_slope"]].merge(fv[["model", "AUROC_sd", "Brier_sd"]], on="model", validate="one_to_one")
table(supp_cal.rename(columns={"model":"Model", "cal_intercept":"Cal. intercept", "cal_slope":"Cal. slope", "AUROC_sd":"AUROC SD", "Brier_sd":"Brier SD"}), 4)

emit(r'''
The native-NA XGBoost sensitivity changes missing-value handling relative to
median-imputed XGB-full. The GAM-7 + NA/vent sensitivity adds two missingness
indicators and ventilation; it does not isolate imputation alone. Neither
sensitivity, nor XGB-15, was evaluated on B/C. Ridge-15 was evaluated on B/C.
SHAP coverage remains XGB-full and XGB-7 only.


<a id="sec-supp-paired"></a>

## S2. Complete paired comparisons

**Table S3. AUROC differences with conditional 95% bootstrap intervals.** Each
row uses the same records for both models and 2,000 jointly resampled record
sets. Positive differences favor the first model. These intervals condition on
existing out-of-fold predictions and do not incorporate refitting or retuning.
Exploratory intervals have not been adjusted for multiple comparisons.
''')

supp_bt = boot.copy()
supp_bt["Difference (95% CI)"] = supp_bt.apply(lambda r: f"{r.dAUROC:.4f} ({r.dAUROC_lo:.4f}, {r.dAUROC_hi:.4f})", axis=1)
table(supp_bt[["comparison", "n", "Difference (95% CI)"]].rename(columns={"comparison":"First minus second", "n":"N"}))

emit(r'''
**Table S4. Brier-score differences with conditional 95% bootstrap intervals.**
Negative differences favor the first model. The matched sample size for each
comparison is the same as in Table S3. Five decimal places preserve small
differences between the two regularized logistic models.
''')

supp_bt["Difference (95% CI)"] = supp_bt.apply(lambda r: f"{r.dBrier:.5f} ({r.dBrier_lo:.5f}, {r.dBrier_hi:.5f})", axis=1)
table(supp_bt[["comparison", "Difference (95% CI)"]].rename(columns={"comparison":"First minus second"}))

emit(r'''
<a id="sec-supp-thresholds"></a>

## S3. Complete threshold results

The prevalence threshold is calculated using each record's outer training fold,
not the held-out fold. The second threshold is fixed at 0.30. These are illustrative
cutoffs, not validated clinical intervention rules. Counts and probability metrics
use the available-score subset for each clinical benchmark.
''')

for number, (threshold, rows) in enumerate(thr.groupby("threshold", sort=False), start=5):
    md(f"**Table S{number}. All-model threshold performance: {threshold}.**")
    table(rows[["model", "TP", "FP", "FN", "TN"]].rename(columns={"model":"Model"}))
    md("**Table continued: sensitivity, specificity, positive predictive value (PPV), and negative predictive value (NPV).**")
    table(rows[["model", "sens", "spec", "PPV", "NPV"]].rename(columns={"model":"Model", "sens":"Sensitivity", "spec":"Specificity"}), 4)

emit(r'''
<a id="sec-supp-tuning"></a>

## S4. Tuning selections and numerical checks

The tables below list the configuration selected separately within each outer
training fold. Fold labels 1-5 correspond to stored fold indices 0-4. Selection
uses five-fold inner cross-validation and negative log-loss; all models share
the outer folds. These are **outer-fold selections**, not the parameters of the
separate full-set-A refits used for interpretation and B/C prediction.

The search ranges are described in the Methods. For XGBoost, the tables report
the chosen configuration among 25 sampled configurations, not an exhaustive
search of all parameter combinations.
''')

selected = json.loads((R / "selected_hyperparameters.json").read_text())
simple_rows = []
for model, fits in selected.items():
    assert len(fits) == 5, f"Expected five outer-fold selections: {model}"
    if model.startswith("XGB"):
        continue
    for parameter in fits[0]:
        row = {"Model": model, "Parameter": parameter.split("__")[-1]}
        row.update({f"Fold {i+1}": f"{values[parameter]:.6g}" for i, values in enumerate(fits)})
        simple_rows.append(row)
md("**Table S7. Selected parameters for interpretable and regularized models.** C is inverse regularization strength; lam is the GAM smoothing penalty.")
table(pd.DataFrame(simple_rows), 4)
for number, model in enumerate(["XGB-7", "XGB-15", "XGB-full", "XGB-full (native NA)"], start=8):
    md(f"**Table S{number}. Selected parameters: {model}.**")
    rows = []
    for parameter in selected[model][0]:
        row = {"Parameter": parameter.split("__")[-1]}
        row.update({f"Fold {i+1}": f"{values[parameter]:.6g}" for i, values in enumerate(selected[model])})
        rows.append(row)
    table(pd.DataFrame(rows), 3)

emit(r'''
**Numerical checks.** A tuning candidate whose fit raised an error received a
missing cross-validation score and could not be selected (scikit-learn default).
Convergence warnings were suppressed during fitting and candidate-level scores
were not saved, so the number of failed candidates is not reported. Every
selected model produced finite out-of-fold predictions. No new tuning or
sensitivity analysis is introduced by this appendix.
''')

if include_code:
    md("# Additional appendix: complete analysis source\n\n"
       "The code below is read directly from the analysis scripts; it is not a separately maintained copy.")
    for name in ["featuresets", "01_build_features", "02_data_quality", "03_models", "05_supplement"]:
        source = (ROOT / f"code/{name}.py").read_text()
        md(f"## {name}.py\n\n```python\n{source.rstrip()}\n```")


# ------------------------------------------------------------------ write outputs
text = "\n\n".join(OUT) + "\n"
(ROOT / "Final_Report.md").write_text(text)
print("Wrote Final_Report.md")
try:
    import markdown
except ImportError:
    print("Python package `markdown` not installed: skipped Final_Report.html")
else:
    def inline_img(m):
        data = base64.b64encode((ROOT / m.group(2)).read_bytes()).decode()
        return f'<img alt="{m.group(1)}" src="data:image/png;base64,{data}">'
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    body = re.sub(r'<img alt="([^"]*)" src="(results/[^"]+\.png)"\s*/?>', inline_img, body)
    css = ("body{max-width:960px;margin:2rem auto;padding:0 1rem;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
           "line-height:1.55;color:#222}table{border-collapse:collapse;margin:1rem 0;font-size:.9rem}"
           "th,td{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}th{background:#f3f3f3}"
           "img{max-width:100%}pre{background:#f6f8fa;padding:.8rem;overflow-x:auto;font-size:.8rem}"
           "code{font-size:.9em}h1,h2{border-bottom:1px solid #eee;padding-bottom:.2rem}")
    (ROOT / "Final_Report.html").write_text(
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Final Report — ICU mortality: interpretable model vs black box</title>"
        f"<style>{css}</style></head><body>\n{body}\n</body></html>\n")
    print("Wrote Final_Report.html (self-contained, images embedded)")
