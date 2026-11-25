"""Supplement to 03_models.py (Proposal Snapshot "LASSO/ridge"; §5.2 complexity summary).

- Ridge (L2) logistic regression on the core-15 set, same outer folds / inner-CV tuning /
  fold-internal median imputation as all other models; results appended to the main result files.
- Relative model-complexity table.
Usage: python 05_supplement.py <derived_dir> <outcomes_dir> <results_dir>
Requires only numpy/pandas/scikit-learn.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from featuresets import CORE15

D, O, R = sys.argv[1], sys.argv[2], sys.argv[3]
SEED, NAME = 707, "Ridge-15"
C15 = list(CORE15)


def load(s):
    p = os.path.join(O, f"Outcomes-{s}.txt")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(os.path.join(D, f"features_{s}.csv")).merge(pd.read_csv(p), on="RecordID")
    return df[df["In-hospital_death"].isin([0, 1])].reset_index(drop=True)


def ridge():
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l2", max_iter=5000))])
    return GridSearchCV(pipe, {"lr__C": list(np.logspace(-4, 1, 16))}, scoring="neg_log_loss",
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1))


def cal_int_slope(yy, pp):
    pp = np.clip(pp, 1e-6, 1 - 1e-6); lp = np.log(pp / (1 - pp))
    slope = LogisticRegression(C=1e6).fit(lp[:, None], yy).coef_[0, 0]
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp))); a += (yy - q).sum() / (q * (1 - q)).sum()
    return a, slope


def conf(yy, pred):
    tp = int((pred & (yy == 1)).sum()); fp = int((pred & (yy == 0)).sum())
    fn = int((~pred & (yy == 1)).sum()); tn = int((~pred & (yy == 0)).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                PPV=tp / max(tp + fp, 1), NPV=tn / max(tn + fn, 1))


def upsert(fname, new, key="model"):
    f = os.path.join(R, fname)
    old = pd.read_csv(f)
    old = old[~old[key].astype(str).str.startswith(NAME)]
    pd.concat([old, new], ignore_index=True).to_csv(f, index=False)


A = load("a")
folds_df = pd.read_csv(os.path.join(R, "outer_folds_setA.csv"))
A = A.merge(folds_df, on="RecordID")
assert len(A) == len(folds_df)
y, folds = A["In-hospital_death"].values.astype(int), A["fold"].values

oof = pd.read_csv(os.path.join(R, "oof_predictions_setA.csv"))
assert (oof.RecordID.values == A.RecordID.values).all()
p = np.zeros(len(A)); chosen = []
for k in range(5):
    tr, te = folds != k, folds == k
    est = ridge().fit(A.loc[tr, C15], y[tr])
    p[te] = est.predict_proba(A.loc[te, C15])[:, 1]
    chosen.append(float(est.best_params_["lr__C"]))
oof[NAME] = p
oof.to_csv(os.path.join(R, "oof_predictions_setA.csv"), index=False)
hp = json.load(open(os.path.join(R, "selected_hyperparameters.json"))); hp[NAME] = [{"lr__C": c} for c in chosen]
json.dump(hp, open(os.path.join(R, "selected_hyperparameters.json"), "w"), indent=1)

ci, cs = cal_int_slope(y, p)
upsert("performance_setA_cv.csv", pd.DataFrame([dict(model=NAME, group="exploratory", n_features=15, n=len(y), AUROC=roc_auc_score(y, p),
                                                     AUPRC=average_precision_score(y, p), Brier=brier_score_loss(y, p), cal_intercept=ci, cal_slope=cs)]))
fr = pd.DataFrame([dict(model=NAME, fold=k, AUROC=roc_auc_score(y[folds == k], p[folds == k]), Brier=brier_score_loss(y[folds == k], p[folds == k])) for k in range(5)])
upsert("performance_by_fold.csv", fr)
upsert("fold_variability.csv", pd.DataFrame([dict(model=NAME, AUROC_sd=fr.AUROC.std(), AUROC_min=fr.AUROC.min(), AUROC_max=fr.AUROC.max(), Brier_sd=fr.Brier.std())]))
tprev = np.array([y[folds != k].mean() for k in folds])
t1 = conf(y, p >= tprev); t1.update(model=NAME, threshold="training-fold prevalence (~0.14)")
t2 = conf(y, p >= 0.30); t2.update(model=NAME, threshold="0.30")
upsert("threshold_metrics.csv", pd.DataFrame([t1, t2]))

dca = pd.read_csv(os.path.join(R, "decision_curves.csv")); n = len(y)
dca[NAME] = [((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in dca.threshold]
dca.to_csv(os.path.join(R, "decision_curves.csv"), index=False)

rng = np.random.default_rng(SEED); rows = []
for other in ["GAM-7", "LASSO-15"]:
    q = oof[other].values; bb = []
    for _ in range(2000):
        ii = rng.choice(n, n, replace=True)
        bb.append((roc_auc_score(y[ii], p[ii]) - roc_auc_score(y[ii], q[ii]), brier_score_loss(y[ii], p[ii]) - brier_score_loss(y[ii], q[ii])))
    bb = np.array(bb)
    rows.append(dict(comparison=f"{NAME} minus {other}", n=n, dAUROC=roc_auc_score(y, p) - roc_auc_score(y, q), dAUROC_lo=np.percentile(bb[:, 0], 2.5),
                     dAUROC_hi=np.percentile(bb[:, 0], 97.5), dBrier=brier_score_loss(y, p) - brier_score_loss(y, q),
                     dBrier_lo=np.percentile(bb[:, 1], 2.5), dBrier_hi=np.percentile(bb[:, 1], 97.5)))
upsert("paired_bootstrap.csv", pd.DataFrame(rows), key="comparison")

final = ridge().fit(A[C15], y).best_estimator_
coef = pd.DataFrame({"feature": C15, "label": [CORE15[c] for c in C15], "coef_per_SD": final.named_steps["lr"].coef_[0]})
coef.sort_values("coef_per_SD", key=abs, ascending=False).to_csv(os.path.join(R, "ridge_coefficients.csv"), index=False)
ext = []
for s in ["b", "c"]:
    T = load(s)
    if T is None:
        continue
    yt = T["In-hospital_death"].values.astype(int); pt = final.predict_proba(T[C15])[:, 1]; a_, b_ = cal_int_slope(yt, pt)
    ext.append(dict(set=s.upper(), model=NAME, n=len(yt), AUROC=roc_auc_score(yt, pt), AUPRC=average_precision_score(yt, pt),
                    Brier=brier_score_loss(yt, pt), cal_intercept=a_, cal_slope=b_))
if ext:
    upsert("performance_external_sets.csv", pd.DataFrame(ext))

# ---------------- relative model complexity (§5.2)
gc = json.load(open(os.path.join(R, "gam_complexity.json"))); ts = json.load(open(os.path.join(R, "tree_size.json")))
las = pd.read_csv(os.path.join(R, "lasso_coefficients.csv"))
perf = pd.read_csv(os.path.join(R, "performance_setA_cv.csv")).set_index("model")


def xgb_desc(m):
    ps = hp[m]
    ne = [d["xgb__n_estimators"] for d in ps]; md = [d["xgb__max_depth"] for d in ps]
    return f"{int(np.median(ne))} trees × depth {int(np.median(md))} (median over folds; ≤ {int(np.median(ne)) * 2 ** int(np.median(md)):,} leaves)"


cx = [
    ("Tree-7", 7, f"depth {ts['depth']}, {ts['n_leaves']} leaves = {ts['n_leaves']} if-then rules", "Explicit rules; can be printed on one card"),
    ("GAM-7", 7, f"7 smooth terms, effective df = {gc['GAM-7_edof']:.1f}", "One curve per predictor (additive)"),
    ("LASSO-15", 15, f"{int((las.coef_per_SD.abs() > 1e-8).sum())} non-zero coefficients", "One coefficient per predictor (linear)"),
    ("Ridge-15", 15, "15 shrunken coefficients", "One coefficient per predictor (linear)"),
    ("XGB-7", 7, xgb_desc("XGB-7"), "Post hoc only (SHAP / PDP)"),
    ("XGB-15", 15, xgb_desc("XGB-15"), "Post hoc only (SHAP / PDP)"),
    ("XGB-full", 193, xgb_desc("XGB-full"), "Post hoc only (SHAP / PDP)"),
    ("SAPS-I", "14 variables", "Fixed additive point score", "Established clinical score"),
    ("SOFA", "6 organ systems", "Fixed additive point score", "Established clinical score"),
]
pd.DataFrame([dict(model=m, predictors=k, size=s, interpretability=i, AUROC=perf.loc[m, "AUROC"]) for m, k, s, i in cx]).to_csv(
    os.path.join(R, "model_complexity.csv"), index=False)
print(f"{NAME}: AUROC={roc_auc_score(y, p):.4f} Brier={brier_score_loss(y, p):.4f} C={chosen}")
print(pd.DataFrame(rows).round(4).to_string()); print(pd.DataFrame(ext).round(3).to_string()); print(coef.round(3).to_string())
print(pd.read_csv(os.path.join(R, "model_complexity.csv")).to_string())
