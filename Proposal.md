# A Simple, Interpretable Model versus a Black Box for Predicting In-Hospital Mortality in ICU Patients

> **707 Final Project — Proposal** · **Team:** Junjie Cai (jc1206@duke.edu), Yifei Chen (yc772@duke.edu), Jiahui Hu (jh1029@duke.edu), Yuyao Lei (yl1216@duke.edu)· **Repository:** [CX-JIE/707-FINAL-PROJECT](https://github.com/CX-JIE/707-FINAL-PROJECT) · **Date:** 15 October 2026

---

## Project Snapshot

| Item | Plan |
| :--- | :--- |
| **Objective** | Compare a simple, interpretable model with a black-box model for predicting **in-hospital mortality** in ICU patients |
| **Data** | PhysioNet/CinC Challenge 2012 (ODC-BY license) |
| **Unit of analysis** | One ICU patient record |
| **Prediction window** | First 24 hours of ICU admission |
| **Outcome** | In-hospital death (binary) |
| **Core feature set** | 15 prespecified, clinically interpretable predictors |
| **Full feature set** | All eligible measurements and patient characteristics from the first 24 hours |
| **Models compared** | Logistic GAM · shallow decision tree · LASSO/ridge logistic regression · gradient boosting (XGBoost) · SAPS-I and SOFA |
| **Validation** | Stratified 5-fold cross-validation at the patient-record level, with all preprocessing and tuning performed inside the training folds |
| **Primary metrics** | AUROC, calibration plots and Brier score, threshold-based metrics, decision curve analysis |

---

## 1. Research Question and Motivation

### 1.1 Research question

> Can a simple, interpretable model achieve predictive performance **comparable to** that of a more complex machine-learning model for in-hospital mortality, using only data collected during the first 24 hours of ICU admission?

### 1.2 Motivation

Early mortality prediction may help clinicians identify high-risk patients. However, complex models can be difficult to interpret and difficult to implement in practice. This project therefore evaluates the trade-off between three considerations:

1. **Predictive performance** — how well the model discriminates between patients who die and patients who survive, and how accurate its predicted probabilities are.
2. **Interpretability** — how easily a clinician can understand and communicate how the model reaches its predictions.
3. **Potential clinical usefulness** — whether differences between models would plausibly matter at realistic decision thresholds.

Rather than assuming that a more complex model is better, we investigate whether a model using a small number of clinically meaningful predictors can achieve comparable performance.

### 1.3 Scope of the project

| In scope | Out of scope |
| :--- | :--- |
| Fitting and comparing predictive models on a public ICU dataset | Estimating causal effects of predictors on mortality |
| Evaluating discrimination, calibration, classification, and potential clinical utility | Prospective deployment, clinical trials, or claims of improved patient outcomes |
| Post hoc explanation of the black-box model | Treating post hoc explanations as evidence of causal risk factors |

---

## 2. Data Source, Cohort, and Outcome

### 2.1 Data source and license

We will use the publicly available **PhysioNet/CinC Challenge 2012** dataset, which contains ICU patient records with time-stamped physiological measurements, clinical variables, and outcome information.

- **Data source:** <https://physionet.org/content/challenge-2012/1.0.0/>
- **License:** Open Data Commons Attribution License v1.0 (ODC-BY), subject to verification of the applicable dataset terms before analysis.

### 2.2 Unit of analysis and cohort definition

The unit of analysis will be **one ICU patient record**. A record will be included if it meets both of the following criteria:

- a valid in-hospital mortality outcome is available; and
- there is sufficient information to establish the prediction window.

Patients will **not** be excluded solely because some predictor measurements are missing.

### 2.3 Outcome definition

The primary outcome will be **in-hospital death**, coded as a binary variable using the dataset's documented outcome definition.

### 2.4 Prediction window

Predictors will be restricted to information available during the **first 24 hours of the ICU stay**. Measurements recorded after this window will not be used to construct predictors.

### 2.5 Pre-analysis verification

Before finalizing the cohort, we will:

1. download and inspect the data;
2. verify the outcome coding and variable definitions; and
3. report the numbers of eligible records, deaths, and survivors.

---

## 3. Predictors and Data Preprocessing

We will construct **two feature sets**: a small, interpretable core set and a full set of eligible measurements.

### 3.1 The core feature set: 15 prespecified predictors

The interpretable feature set consists of 15 candidate predictors, prespecified on the basis of their clinical relevance and interpretability.

| # | Predictor | Clinical domain |
| :-: | :--- | :--- |
| 1 | Age | Demographic characteristics |
| 2 | Glasgow Coma Scale (GCS) | Neurological status |
| 3 | Heart rate (HR) | Cardiovascular and hemodynamic status |
| 4 | Mean arterial pressure (MAP) | Cardiovascular and hemodynamic status |
| 5 | Respiratory rate | Respiratory status and oxygenation |
| 6 | Oxygen saturation (SaO₂) | Respiratory status and oxygenation |
| 7 | Blood urea nitrogen (BUN) | Renal function |
| 8 | Arterial oxygen partial pressure (PaO₂) | Respiratory status and oxygenation |
| 9 | Creatinine | Renal function |
| 10 | White blood cell count (WBC) | Inflammation and hematological status |
| 11 | Platelet count | Inflammation and hematological status |
| 12 | Hematocrit (HCT) | Inflammation and hematological status |
| 13 | Glucose | Metabolic status |
| 14 | Bilirubin | Hepatic function |
| 15 | Temperature | Additional physiological status |

These predictors cover several clinically relevant dimensions of critical illness and can be interpreted in terms of patient characteristics, physiological status, and organ function.

Two points about this set are worth noting:

- **Availability and coding** will be verified against the downloaded data before modeling.
- **Inclusion is based on clinical rationale**, not on an assumption that every variable will independently improve prediction.

### 3.2 Summarizing repeated measurements

For repeatedly measured variables, we will derive **prespecified summaries** from measurements collected during the first 24 hours of ICU admission. Depending on the variable, these may include the mean, the minimum, or the maximum. For example:

| Variable (example) | Candidate summary |
| :--- | :--- |
| Glasgow Coma Scale (GCS) | Minimum |
| Oxygen saturation (SaO₂) | Minimum |
| Heart rate (HR) | Mean |
| Blood urea nitrogen (BUN) | Maximum |

Before finalizing the summaries, we will inspect variable definitions, units, data completeness, and distributions.

### 3.3 The full feature set

The full feature set will include the eligible measurements and patient characteristics available in the dataset during the first 24 hours, **beyond** the 15-variable core set. Potential additional features include:

- electrolytes;
- liver enzymes;
- albumin;
- lactate;
- blood gas measurements;
- blood pressure measurements obtained using different methods; and
- other laboratory or physiological measurements.

### 3.4 Missingness and redundancy

We will examine missingness and redundancy before including additional variables. For example, MAP and systolic or diastolic blood pressure provide related but non-identical information, while PaO₂ and SaO₂ represent different aspects of oxygenation. Variables with substantial missingness or overlapping information will be evaluated carefully rather than included automatically.

### 3.5 Why we compare two feature sets

We will compare models using the core predictor set with models using an expanded feature set to assess whether additional clinical information improves predictive performance.

Because both the **number of predictors** and the **model's flexibility** may change in this comparison, any performance difference will not be attributed to model complexity alone.

---

## 4. Models

### 4.1 Overview of the model comparison

We will compare four types of prediction approaches, plus two clinical score baselines.

| Model | Role | Predictors | Form of interpretability |
| :--- | :--- | :--- | :--- |
| **Logistic GAM** | Interpretable candidate | The first seven of 15 core predictors | Smooth, additive partial effects |
| **Shallow decision tree** | Interpretable candidate | The first seven of 15 core predictors | A compact set of explicit decision rules |
| **LASSO / ridge logistic regression** | Regularized statistical comparison | Eligible predictor set | Sparse or shrunk coefficients |
| **Gradient boosting (XGBoost)** | Black-box benchmark | Full eligible feature set | None directly; post hoc explanations only |
| **SAPS-I, SOFA** | Clinical reference points | Predefined score definitions | Established clinical severity scores |

Abbreviations: GAM = generalized additive model; LASSO = least absolute shrinkage and selection operator; XGBoost = extreme gradient boosting; SAPS-I = Simplified Acute Physiology Score I; SOFA = Sequential Organ Failure Assessment.

### 4.2 Interpretable models: logistic GAM and shallow decision tree

**Logistic generalized additive model (GAM).** The GAM will allow selected continuous predictors to have nonlinear associations with mortality through smooth functions, while preserving interpretable partial-effect plots.

**Shallow decision tree.** The tree will use a restricted depth to produce a small set of explicit decision rules that can be inspected and communicated clinically.

We will compare their predictive performance and interpretability, recognizing that the GAM captures **smooth additive relationships** whereas the tree captures **threshold-based interactions**. These two models therefore represent two complementary forms of interpretability:

- smooth, additive effects (GAM); and
- explicit decision rules (tree).

### 4.3 Regularized model: LASSO or ridge

LASSO or ridge logistic regression will be fitted on the eligible predictor set, with the regularization parameter selected **within the training data**.

### 4.4 Black-box model: gradient boosting

A gradient boosting algorithm such as **XGBoost** will be fitted on the full eligible feature set, with hyperparameters tuned **within the training data**. Gradient boosting represents the more flexible, less directly interpretable approach.

### 4.5 Clinical score baselines: SAPS-I and SOFA

SAPS-I and SOFA will be evaluated as **existing clinical severity-score benchmarks**, where the required score values are available and their definitions are verified. They provide clinically recognizable reference points rather than being treated as newly fitted machine-learning models.

### 4.6 Rules for a fair comparison

1. **Same predictors first.** To make the primary comparison informative, the GAM, the shallow decision tree, and gradient boosting will be evaluated using the same 15 prespecified predictors.
2. **Then expand.** We will then examine whether an expanded feature set improves the performance of gradient boosting.
3. **No information leakage.** All preprocessing and model tuning will be conducted within the training data.
4. **Same folds.** All fitted models will be evaluated on the same outer folds.

---

## 5. Validation and Evaluation

### 5.1 Validation design

We will use **stratified five-fold cross-validation at the patient-record level**. The procedure is:

1. Split records into five stratified folds, keeping the death/survival ratio similar across folds.
2. Within each training fold, fit all preprocessing steps and select hyperparameters.
3. Where computationally feasible, use an **inner cross-validation loop** for tuning, to reduce optimism in model comparisons.
4. Evaluate every fitted model on the same held-out fold, so that all models are compared on identical data.

### 5.2 Evaluation metrics

| Dimension | Metrics | Purpose |
| :--- | :--- | :--- |
| **Discrimination and calibration** | AUROC; calibration plots and the Brier score; AUPRC may also be reported | Assess discrimination and the accuracy of predicted mortality probabilities; AUPRC adds information under class imbalance |
| **Threshold-based classification** | Confusion matrices at prespecified or training-data-selected thresholds; sensitivity, specificity, positive predictive value, negative predictive value | Show which types of classification errors each model makes at a particular threshold, complementing AUROC |
| **Potential clinical utility** | Decision curve analysis over a prespecified range of plausible risk thresholds | Exploratory assessment of net benefit, not evidence of improved patient outcomes |
| **Stability and interpretability** | Variation in performance across validation folds; GAM partial-effect plots; LASSO coefficients; a summary of relative model complexity; tree depth and number of terminal nodes | Show whether performance is stable, and whether it can be achieved with a compact set of decision rules |

### 5.3 Principles of interpretation

- We will report cross-validated performance with variability or uncertainty estimates where feasible.
- We will emphasize the **trade-off** between predictive performance and interpretability rather than identifying a single model as universally best.
- Differences in AUROC will **not** be interpreted in isolation; calibration and decision curves will also be considered when assessing whether a performance difference could matter in practice.

---

## 6. Expected Findings and Limitations

### 6.1 Expected findings

We expect that the more flexible gradient boosting model may achieve better predictive performance than the simpler interpretable models, by capturing nonlinear relationships and interactions among predictors.

However, the performance gain may be **modest**. In that case, a GAM or a shallow decision tree could offer a more transparent alternative with comparable predictive performance. We will evaluate this trade-off using discrimination, calibration, threshold-based classification metrics, and, if feasible, decision curve analysis — rather than assuming that the most complex model will perform best.

### 6.2 Explaining the black box after the fact

To better understand the behavior of gradient boosting, we will use post hoc explanation methods, depending on feasibility:

- feature importance;
- SHAP values; and
- partial dependence or related effect plots.

These methods may help us to identify influential predictors, to examine how individual predictors contribute to patient-level predictions, and to explore patterns learned by the complex model that may not be captured by the simpler models. We will compare these patterns with the effects represented by the GAM and with the decision rules learned by the shallow decision tree.

### 6.3 Limitations

1. **Post hoc explanations are not causal.** They describe how a fitted model generates predictions; they do not establish causal effects or prove that a predictor is an independent clinical risk factor.
2. **Correlated predictors are hard to attribute.** Correlated predictors may distribute or obscure apparent feature contributions, and partial dependence plots may be difficult to interpret when combinations of predictor values are uncommon in the data. Feature importance and SHAP values may also vary across fitted models and validation samples.
3. **Generalizability is limited.** The dataset's missingness patterns, measurement frequency, and patient population may limit how far the findings extend.
4. **No evidence of clinical benefit.** This project evaluates predictive performance using observational data and does not establish that applying the models would improve clinical outcomes.

### 6.4 Overall aim

Overall, we aim to determine:

- whether a relatively simple and interpretable model can achieve **competitive predictive performance**; and
- whether post hoc explanations provide **useful, but appropriately limited, insight** into a more complex model.

---

## 7. Reproducibility and Team Contributions

### 7.1 Repository and data handling

The project will be maintained in a shared GitHub repository: <https://github.com/CX-JIE/707-FINAL-PROJECT>.

The repository will include:

- the proposal;
- data-download or data-preparation scripts;
- the analysis code; and
- documentation of the dataset source, license, and required citation.

Large data files will **not** be committed unless redistribution is permitted and the files are appropriately sized. Each team member will contribute through their own GitHub account, and each member will submit an individual contribution statement.

### 7.2 Team roles

| Member | Responsibility | Concrete tasks |
| :--- | :--- | :--- |
| **Junjie Cai** | Data and cohort construction | Verify data access and license terms; inspect outcome coding; construct the patient-level cohort; summarize missingness; document data provenance |
| **Yifei Chen** | Interpretable and regularized models | Implement the GAM, shallow decision tree, and LASSO logistic regression; conduct appropriate model tuning; prepare coefficient and partial-effect visualizations |
| **Jiahui Hu** | Black-box modeling and evaluation | Implement gradient boosting and the clinical-score benchmarks; prepare feature-importance and post hoc model-explanation visualizations|
| **Yuyao Lei** |Shared validation and comparative evaluation|Coordinate the shared validation framework; ensure consistent data splits and evaluation procedures across models; produce discrimination, calibration, and decision curve comparisons; summarize comparative results|

### 7.3 Shared standards

All members will:

- review the analysis plan;
- cross-check the results;
- contribute to the interpretation, project proposal, final report and in-class presentation; and
- document their individual contributions.

The team will use a **shared set of validation folds** and **consistent outcome definitions** to ensure that model comparisons are reproducible and fair.

---

## References

1. PhysioNet. Predicting Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012. Dataset documentation and files. https://physionet.org/content/challenge-2012/1.0.0/
2. Silva, I., Moody, G., Scott, D. J., Celi, L. A., & Mark, R. G. (2012). Predicting in-hospital mortality of ICU patients: The PhysioNet/Computing in cardiology challenge 2012 2012 Computing in Cardiology, Krakow, Poland. 

---

## Appendix: Abbreviations

| Abbreviation | Meaning |
| :--- | :--- |
| AUPRC | Area under the precision–recall curve |
| AUROC | Area under the receiver operating characteristic curve |
| BUN | Blood urea nitrogen |
| GAM | Generalized additive model |
| GCS | Glasgow Coma Scale |
| HCT | Hematocrit |
| ICU | Intensive care unit |
| LASSO | Least absolute shrinkage and selection operator |
| MAP | Mean arterial pressure |
| NPV | Negative predictive value |
| PaO₂ | Arterial oxygen partial pressure |
| PPV | Positive predictive value |
| SaO₂ | Oxygen saturation |
| SAPS-I | Simplified Acute Physiology Score I |
| SHAP | SHapley Additive exPlanations |
| SOFA | Sequential Organ Failure Assessment |
| WBC | White blood cell count |
| XGBoost | Extreme gradient boosting |
|              ||




