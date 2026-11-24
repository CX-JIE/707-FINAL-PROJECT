"""Build a self-contained HTML report from the results folder.
Usage: python 04_report.py <results_dir> <out_html>
Model sections are filled in automatically when 03_models.py outputs exist.
"""
import sys, os, json, base64, html
import pandas as pd

R, OUT = sys.argv[1], sys.argv[2]
J = lambda f: json.load(open(os.path.join(R, f)))
C = lambda f: pd.read_csv(os.path.join(R, f))
has = lambda f: os.path.exists(os.path.join(R, f))


def img(f, w="100%"):
    b = base64.b64encode(open(os.path.join(R, f), "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="max-width:{w}">'


def tab(df, fmt=3):
    return df.to_html(index=False, float_format=lambda x: f"{x:.{fmt}f}", border=0, classes="t")


coh, fs = J("cohort_summary.json"), J("featureset_sizes.json")
miss = C("core15_missingness_setA.csv")
vent = C("ventilation_vs_missingness_setA.csv")
pat = C("small7_missing_patterns_setA.csv")
red = C("redundant_pairs_setA.csv")
allm = C("all_variables_missingness_setA.csv")
MODELS_DONE = has("performance_setA_cv.csv")

css = """body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1050px;margin:32px auto;padding:0 20px;color:#1f2933;line-height:1.6}
h1{font-size:26px;margin-bottom:4px}h2{border-bottom:2px solid #e4e7eb;padding-bottom:4px;margin-top:38px}h3{margin-top:24px}
.t{border-collapse:collapse;font-size:13px;margin:10px 0}.t th,.t td{padding:4px 9px;border-bottom:1px solid #e4e7eb;text-align:right}.t th{background:#f5f7fa}
.t td:first-child,.t th:first-child{text-align:left}.box{background:#fff8e6;border-left:4px solid #f0b429;padding:10px 14px;margin:14px 0}
.ok{background:#effcf6;border-left:4px solid #27ab83;padding:10px 14px;margin:14px 0}.meta{color:#616e7c;font-size:14px}code{background:#f5f7fa;padding:1px 4px}"""

h = [f"<html><head><meta charset='utf-8'><title>707 Final Project Report</title><style>{css}</style></head><body>"]
h.append("<h1>Interpretable model vs black box: predicting in-hospital mortality in ICU patients</h1>")
h.append("<p class='meta'>707 Final Project · PhysioNet/CinC Challenge 2012 · Analysis follows Proposal v4 (15 Oct 2026)</p>")

if MODELS_DONE and has("summary.html"):
    h.append(open(os.path.join(R, "summary.html")).read())
if not MODELS_DONE:
    h.append("""<div class='box'><b>Status: data preparation and quality checks are done. The modeling pipeline is written and tested, but has not been run on real labels.</b><br>
The <code>data/</code> folder holds only the set-a/b/c record archives. It does not contain the PhysioNet outcome files <code>Outcomes-a.txt</code> (and <code>-b</code>, <code>-c</code>).
Those files are the only source of the in-hospital mortality label and of the SAPS-I and SOFA scores. Without them no model can be trained or evaluated, so this version of the report has no model results.
Once the outcome files are placed in <code>data/</code>, create the environment with <code>conda env create -f environment.yml</code>, then run <code>conda activate icu707 && bash run_all.sh</code>. It regenerates this report with every model section filled in.</div>""")

# 1 data
h.append("<h2>1. Data and cohort (Proposal §2)</h2>")
h.append("<p><b>Data source:</b> PhysioNet/Computing in Cardiology Challenge 2012, <i>Predicting Mortality of ICU Patients</i>, v1.0.0 (https://physionet.org/content/challenge-2012/1.0.0/). "
         "<b>License:</b> Open Data Commons Attribution License v1.0 (ODC-BY 1.0), which allows use and redistribution with attribution. "
         "<b>Required citations:</b> Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting in-hospital mortality of ICU patients: the PhysioNet/Computing in Cardiology Challenge 2012. <i>Computing in Cardiology</i> 2012;39:245–248; "
         "and Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. <i>Circulation</i> 2000;101(23):e215–e220. "
         "Files used: set-a/b/c (per-patient time series) and Outcomes-a/b/c.txt (in-hospital death, SAPS-I, SOFA). The outcome coding was checked: In-hospital_death is binary 0/1 with no missing values, and −1 in SAPS-I/SOFA means missing.</p>")
h.append("<p>Unit of analysis: one ICU stay record. Only measurements with timestamp ≤ 24:00 (first 24 h after ICU admission) are used. "
         "Physiologically implausible values (e.g. HR/MAP ≤ 0, Temp &lt; 25 °C, weight = −1) are set to missing. FiO₂ recorded as a percentage is rescaled to a fraction. "
         "Following the proposal, no patient is excluded for missing predictors.</p>")
cdf = pd.DataFrame([{"set": s.upper(), "records": v["n_records"], "no observations in first 24 h": v["n_no_obs_24h"],
                     "median observations in 24 h": v["median_obs_24h"], "median age": v["age_median"], "female %": round(v["female_pct"], 1),
                     "ICU type 1/2/3/4 (%)": " / ".join(str(v["icutype_pct"].get(k, v["icutype_pct"].get(str(k)))) for k in ["1", "2", "3", "4"])} for s, v in coh.items()])
h.append(tab(cdf, 1))
h.append("<p class='meta'>ICU types: 1 = coronary care, 2 = cardiac surgery recovery, 3 = medical, 4 = surgical. Set A is the development set; sets B and C are held out (external evaluation only if their outcome files are available).</p>")
if MODELS_DONE:
    ri = J("run_info.json")
    h.append(f"<div class='ok'>Set A records with a valid outcome: <b>{ri['n']}</b> (all 4,000 eligible; none excluded). In-hospital deaths: <b>{ri['deaths']}</b> ({ri['mortality_pct']:.1f}%); survivors: <b>{ri['n'] - ri['deaths']}</b>. Set B: 568 deaths / 3,432 survivors (14.2%); set C: 585 deaths / 3,415 survivors (14.6%).</div>")

# 2 features
h.append("<h2>2. Feature construction and data quality (Proposal §3)</h2>")
h.append(f"<p>Three feature sets: <b>small-7</b> (GAM / decision tree / XGB-7), <b>core-15</b> (LASSO / XGB-15) and <b>full</b> ({fs['n_full_features']} features: 4 static descriptors, mean/min/max/last/count for all 37 time-series variables, plus combined MAP and 24-h urine total). "
         "MAP is taken from the invasive measurement when present, otherwise the non-invasive one (MAPc). Summary statistics were fixed in advance as stated in the proposal (GCS min, SaO₂ min, HR mean, BUN max, and so on).</p>")
h.append("<h3>2.1 Missingness and distributions of the 15 core predictors (set A)</h3>")
h.append(tab(miss.drop(columns=["feature"]), 2))
h.append(img("fig_core15_distributions.png"))
h.append("<h3>2.2 Key finding: respiratory rate and SaO₂ are structurally missing</h3>")
h.append(tab(vent, 1))
h.append("""<p>Respiratory rate is missing for <b>100%</b> of patients on mechanical ventilation (it is recorded only for spontaneously breathing patients). SaO₂ is missing for 62% of set A overall.
These values are not missing at random: whether a value was recorded is itself a marker of illness severity. Proposal §3.2 requires this to be documented before models are evaluated. <b>Decided before any outcome was seen</b>:
(1) the primary analysis follows the protocol and keeps all 7 prespecified predictors with training-fold median imputation;
(2) a sensitivity analysis, <i>GAM-7 + NA indicators</i>, adds three binary variables: RespRate not recorded, SaO₂ not recorded, and any mechanical ventilation in the first 24 h.
This shows how much the imputation choice affects the compact model's performance.</p>""")
h.append("<p>Top 10 missingness patterns across the small-7 set (M = missing; column order Age, GCS, HR, MAP, RR, SaO₂, BUN):</p>" + tab(pat, 0))
h.append("<h3>2.3 Variable availability (all variables)</h3>" + img("fig_missingness_all.png", "75%"))
h.append("<h3>2.4 Redundancy (|Spearman ρ| ≥ 0.5)</h3>" + tab(red, 3))
h.append("<p>The blood-pressure variables are strongly correlated with one another (invasive and non-invasive systolic, diastolic and MAP), and so are BUN and creatinine (ρ≈0.75) and SaO₂ and PaO₂ (ρ≈0.69, but only 1,483 patients have both). "
         "The GAM uses only one of each such pair, which avoids collinearity. In XGB-full these correlations spread importance across related variables, so the post hoc explanations need to be read with care (Proposal §6.4).</p>")

# 3 methods
h.append("<h2>3. Models and validation (Proposal §4–5)</h2>")
h.append("""<ul>
<li><b>Logistic GAM (small-7)</b>: <code>pygam.LogisticGAM</code>. Each continuous predictor gets a penalized cubic regression spline s() with 10 basis functions. The smoothing parameter λ is chosen by inner 5-fold CV on log-loss over 11 grid points from 0.01 to 1000.</li>
<li><b>Shallow decision tree (small-7)</b>: scikit-learn CART, depth ∈ {2,3,4}, minimum leaf size ∈ {25,50,100}, chosen by inner CV.</li>
<li><b>LASSO / ridge logistic (core-15)</b>: standardized inputs; L1 (LASSO, 13 grid points) or L2 (ridge, 16 grid points) penalty; C chosen by inner CV. Ridge is fitted by <code>code/05_supplement.py</code> using the same outer folds and the same fold-internal preprocessing.</li>
<li><b>XGBoost</b>: <code>xgboost.XGBClassifier</code> (hist), with full / 7 / 15 predictors. Inner 5-fold random search (25 configurations) over learning_rate, n_estimators, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda and gamma. The primary model uses median-imputed inputs; a sensitivity analysis uses XGBoost's native missing-value handling (Proposal §3.5).</li>
<li><b>SAPS-I / SOFA</b>: taken directly from the PhysioNet outcome files (−1 treated as missing). AUROC uses the raw score. For calibration and Brier score, each score is converted to a probability by logistic recalibration fitted within the training folds.</li>
<li><b>Validation</b>: stratified 5-fold nested CV, with the same outer folds for every model (<code>outer_folds_setA.csv</code>, seed 707). All imputation and tuning happen inside the training folds.</li>
<li><b>Metrics</b>: AUROC, AUPRC, Brier score, calibration intercept and slope, decile calibration plots, confusion matrices at two thresholds (the training-fold mortality rate, about 0.14, and a fixed 0.30), decision curves over thresholds 0.05–0.50, and per-fold variability. Paired bootstrap (2,000 resamples of patients) for AUROC and Brier differences with 95% CIs.</li>
<li><b>Post hoc explanation (Proposal §6.2–6.3)</b>: <code>shap.TreeExplainer</code> SHAP values (summary plot and global importance), XGBoost gain importance, partial dependence / ICE, and SHAP interaction values (XGB-7). The SHAP dependence curves are compared against pyGAM partial effects with 95% CIs, the decision-tree rules and the LASSO coefficients.</li></ul>""")

if MODELS_DONE:
    perf, boot, thr, fv = C("performance_setA_cv.csv"), C("paired_bootstrap.csv"), C("threshold_metrics.csv"), C("fold_variability.csv")
    h.append("<h2>4. Results</h2><h3>4.1 Cross-validated performance (set A, out-of-fold)</h3>" + tab(perf.merge(fv[["model", "AUROC_sd"]], on="model"), 3))
    h.append(img("fig_roc_calibration.png"))
    h.append("<h3>4.2 Paired bootstrap differences (95% CI)</h3>" + tab(boot, 4))
    h.append("<p class='meta'>ΔAUROC &gt; 0 means the first model discriminates better; ΔBrier &lt; 0 means the first model's probabilities are more accurate.</p>")
    h.append("<h3>4.3 Threshold metrics</h3>" + tab(thr[["model", "threshold", "TP", "FP", "FN", "TN", "sens", "spec", "PPV", "NPV"]], 3))
    h.append("<h3>4.4 Decision curve analysis</h3>" + img("fig_decision_curves.png", "70%"))
    if has("performance_external_sets.csv"):
        h.append("<h3>4.5 Held-out sets B/C (models refit on all of set A)</h3>" + tab(C("performance_external_sets.csv"), 3))
    if has("model_complexity.csv"):
        h.append("<h3>4.6 Relative model complexity (Proposal §5.2)</h3>" + tab(C("model_complexity.csv"), 3))
    h.append("<h2>5. Interpretability</h2><h3>5.1 GAM partial effects (95% CI) vs XGBoost SHAP dependence</h3>" + img("fig_partial_effects.png"))
    ts = J("tree_size.json")
    h.append(f"<h3>5.2 Decision tree rules (depth {ts['depth']}, {ts['n_leaves']} leaves)</h3><pre style='font-size:12px'>{html.escape(open(os.path.join(R, 'tree_rules.txt')).read())}</pre>")
    lc = C("lasso_coefficients.csv").merge(C("ridge_coefficients.csv")[["feature", "coef_per_SD"]].rename(columns={"coef_per_SD": "ridge_coef_per_SD"}), on="feature") if has("ridge_coefficients.csv") else C("lasso_coefficients.csv")
    h.append("<h3>5.3 LASSO and ridge coefficients (per SD; LASSO selection frequency across 5 outer folds)</h3>" + tab(lc, 3))
    gc = J("gam_complexity.json")
    h.append(f"<p class='meta'>GAM-7 effective degrees of freedom = {gc['GAM-7_edof']:.1f} (λ = {gc['GAM-7_lam']:.3g}).</p>")
    h.append("<h3>5.4 XGB-full: SHAP</h3>" + img("fig_shap_summary_xgb_full.png", "70%") + img("fig_xgb_importance.png", "60%"))
    h.append(tab(C("xgb_full_shap_importance.csv").head(20), 4))
    h.append("<h3>5.5 XGB-full: partial dependence / ICE</h3>" + img("fig_pdp_xgb_full.png"))
    h.append("<h3>5.6 XGB-7: SHAP interactions (structure an additive GAM cannot represent)</h3>" + tab(C("xgb7_shap_interactions.csv").head(10), 4) + tab(C("xgb7_shap_main_vs_total.csv"), 4))
    if has("interpretation.html"):
        h.append(open(os.path.join(R, "interpretation.html")).read())
    v = ri.get("versions", {})
    h.append(f"<p class='meta'>Software: xgboost {v.get('xgboost')}, pygam {v.get('pygam')}, shap {v.get('shap')}, scikit-learn {v.get('sklearn')}; inner random search {ri['n_iter_xgb']} configurations; bootstrap {ri['n_boot']} resamples.</p>")
else:
    h.append("<h2>4. Results and interpretability</h2><div class='box'>Pending the outcome files (see Status at the top). Once they are available, this section will contain, in order: the cross-validated performance table with fold-level variability; ROC and calibration plots; "
             "paired-bootstrap ΔAUROC/ΔBrier for XGB-full vs GAM-7, XGB-7 vs GAM-7 and the other pairs; threshold confusion matrices; decision curves; results on held-out sets B/C; "
             "pyGAM partial effects compared with XGBoost SHAP dependence; the decision-tree rules; LASSO coefficients; XGBoost SHAP importance, PDP/ICE and SHAP interactions.</div>")

h.append("<h2>7. Limitations</h2><ul><li>Post hoc explanations describe how a model makes predictions. They are not causal effects.</li>"
         "<li>Respiratory rate and SaO₂ are structurally missing. Median imputation hides the fact that a value was not recorded, which may understate how well the compact model could do (see the sensitivity analysis).</li>"

         "<li>In the XGBoost random search, the median selected number of trees equals the top of the grid (600), so performance might improve slightly with more trees. This would only strengthen the conclusion that XGB-full is ahead.</li>"
         "<li>Single-source data (PhysioNet 2012, with its patient mix and measurement frequency), so generalizability is limited. No evidence of clinical benefit is claimed.</li></ul>")
h.append("<h2>8. Reproducibility</h2><p><code>bash run_all.sh</code> runs the full pipeline: <code>code/01_build_features.py</code> → <code>02_data_quality.py</code> → <code>03_models.py</code> → <code>05_supplement.py</code> (ridge, complexity table) → <code>04_report.py</code>. "
         "Outputs go to <code>data_derived/</code> and <code>results/</code>. Environment: <code>conda env create -f environment.yml</code> (numpy, pandas, scikit-learn, matplotlib, xgboost, pygam, shap). Nothing was committed or pushed to git.</p>")
h.append("</body></html>")
open(OUT, "w").write("\n".join(h))
print("wrote", OUT, "models_done=", MODELS_DONE)
