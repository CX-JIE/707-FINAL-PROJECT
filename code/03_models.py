"""Nested cross-validated model comparison — strictly as in Proposal v4 §4–§6.

Models: logistic GAM (pyGAM), shallow decision tree, LASSO logistic regression,
XGBoost (full / 7 / 15 predictors), SAPS-I and SOFA. Post hoc: SHAP (TreeExplainer),
feature importance, partial dependence.

Usage:
  python 03_models.py <derived_dir> <outcomes_dir> <results_dir> [--quick]

<outcomes_dir> must contain Outcomes-a.txt (PhysioNet 2012 format:
RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death).
If Outcomes-b.txt / Outcomes-c.txt are present, the models refit on all of set A
are also evaluated on those sets as held-out tests (Proposal §2.2).

Requires: numpy pandas scikit-learn matplotlib xgboost pygam shap
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, roc_curve
from xgboost import XGBClassifier
from pygam import LogisticGAM, s, l
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from featuresets import SMALL7, CORE15, full_features

warnings.filterwarnings("ignore")
D, O, R = sys.argv[1], sys.argv[2], sys.argv[3]
QUICK = "--quick" in sys.argv
os.makedirs(R, exist_ok=True)
SEED = 707
N_ITER_XGB = 6 if QUICK else 25
N_BOOT = 200 if QUICK else 2000
N_JOBS = int(os.environ.get("N_JOBS", "-1"))


def load(set_):
    X = pd.read_csv(os.path.join(D, f"features_{set_}.csv"))
    p = os.path.join(O, f"Outcomes-{set_}.txt")
    if not os.path.exists(p):
        return None
    y = pd.read_csv(p)
    df = X.merge(y, on="RecordID", how="inner")
    df = df[df["In-hospital_death"].isin([0, 1])]
    for sc in ["SAPS-I", "SOFA"]:
        df.loc[df[sc] < 0, sc] = np.nan
    df["RespRate_missing"] = df["RespRate_mean"].isna().astype(float)
    df["SaO2_missing"] = df["SaO2_min"].isna().astype(float)
    df["MechVent_any"] = (df["MechVent_count"] > 0).astype(float)
    return df.reset_index(drop=True)


A = load("a")
assert A is not None, "Outcomes-a.txt not found"
y = A["In-hospital_death"].values.astype(int)
S7, C15 = list(SMALL7), list(CORE15)
FULL = full_features(pd.read_csv(os.path.join(D, "features_a.csv"), nrows=1).columns)
# Sensitivity analysis prespecified before outcomes were seen (see report §2.2)
S7I = S7 + ["RespRate_missing", "SaO2_missing", "MechVent_any"]
BIN7I = ("RespRate_missing", "SaO2_missing", "MechVent_any")


def inner():
    return StratifiedKFold(5, shuffle=True, random_state=SEED + 1)


# ---------------------------------------------------------------- pyGAM wrapper
class GAMClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around pygam.LogisticGAM.
    Continuous predictors -> penalised cubic regression splines s(); binary -> linear l().
    The smoothing parameter lam is tuned by inner CV (GridSearchCV) per Proposal §5.1."""

    def __init__(self, lam=1.0, n_splines=10, binary_idx=()):
        self.lam, self.n_splines, self.binary_idx = lam, n_splines, binary_idx

    def _terms(self, p):
        t = None
        for j in range(p):
            term = l(j) if j in self.binary_idx else s(j, n_splines=self.n_splines, spline_order=3)
            t = term if t is None else t + term
        return t

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.array([0, 1])
        self.gam_ = LogisticGAM(self._terms(X.shape[1]), lam=self.lam, max_iter=200).fit(X, y)
        return self

    def predict_proba(self, X):
        p = np.clip(self.gam_.predict_mu(np.asarray(X, float)), 1e-8, 1 - 1e-8)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def decision_function(self, X):
        p = self.predict_proba(X)[:, 1]
        return np.log(p / (1 - p))


# ---------------------------------------------------------------- model factory
def gam_pipe(cols, binary=()):
    bidx = tuple(i for i, c in enumerate(cols) if c in binary)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("gam", GAMClassifier(binary_idx=bidx))])
    return GridSearchCV(pipe, {"gam__lam": list(np.logspace(-2, 3, 11))}, scoring="neg_log_loss", cv=inner(), n_jobs=N_JOBS)


def tree_pipe():
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("dt", DecisionTreeClassifier(random_state=SEED))])
    return GridSearchCV(pipe, {"dt__max_depth": [2, 3, 4], "dt__min_samples_leaf": [25, 50, 100]}, scoring="neg_log_loss", cv=inner())


def lasso_pipe():
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000))])
    return GridSearchCV(pipe, {"lr__C": list(np.logspace(-3, 1, 13))}, scoring="neg_log_loss", cv=inner())


def xgb_pipe(impute=True):
    steps = ([("imp", SimpleImputer(strategy="median"))] if impute else []) + \
            [("xgb", XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                                   random_state=SEED, n_jobs=1, verbosity=0))]
    grid = {"xgb__learning_rate": [0.02, 0.05, 0.1], "xgb__n_estimators": [100, 200, 400, 600],
            "xgb__max_depth": [2, 3, 4, 6], "xgb__min_child_weight": [1, 5, 10],
            "xgb__subsample": [0.7, 0.85, 1.0], "xgb__colsample_bytree": [0.5, 0.8, 1.0],
            "xgb__reg_lambda": [1.0, 5.0, 10.0], "xgb__gamma": [0.0, 1.0]}
    return RandomizedSearchCV(Pipeline(steps), grid, n_iter=N_ITER_XGB, scoring="neg_log_loss",
                              cv=inner(), random_state=SEED, n_jobs=N_JOBS)


MODELS = {  # name: (factory, columns, analysis tier per Proposal §4.6)
    "GAM-7": (lambda: gam_pipe(S7), S7, "primary"),
    "XGB-full": (lambda: xgb_pipe(True), FULL, "primary"),
    "XGB-7": (lambda: xgb_pipe(True), S7, "supplementary"),
    "Tree-7": (tree_pipe, S7, "exploratory"),
    "LASSO-15": (lasso_pipe, C15, "exploratory"),
    "XGB-15": (lambda: xgb_pipe(True), C15, "exploratory"),
    "XGB-full (native NA)": (lambda: xgb_pipe(False), FULL, "sensitivity"),
    "GAM-7 + NA indicators": (lambda: gam_pipe(S7I, BIN7I), S7I, "sensitivity"),
}
SCORES = ["SAPS-I", "SOFA"]

# ---------------------------------------------------------------- outer CV
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds = np.zeros(len(A), int)
for k, (_, te) in enumerate(outer.split(A, y)):
    folds[te] = k
pd.DataFrame({"RecordID": A.RecordID, "fold": folds}).to_csv(os.path.join(R, "outer_folds_setA.csv"), index=False)

oof = pd.DataFrame({"RecordID": A.RecordID, "y": y, "fold": folds})
thr_train, chosen = {}, {m: [] for m in MODELS}
for name, (fac, cols, grp) in MODELS.items():
    p = np.zeros(len(A)); thr_train[name] = []
    for k in range(5):
        tr, te = folds != k, folds == k
        est = fac().fit(A.loc[tr, cols], y[tr])
        p[te] = est.predict_proba(A.loc[te, cols])[:, 1]
        chosen[name].append({kk: (v.item() if isinstance(v, np.generic) else v) for kk, v in est.best_params_.items()})
        thr_train[name].append(float(y[tr].mean()))
    oof[name] = p
    print(f"{name:24s} AUROC={roc_auc_score(y, p):.3f}", flush=True)

# Clinical scores: AUROC on the raw score; probabilities by logistic recalibration on training folds
for sc in SCORES:
    p = np.full(len(A), np.nan); thr_train[sc] = []
    for k in range(5):
        tr, te = (folds != k) & A[sc].notna().values, (folds == k) & A[sc].notna().values
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(A.loc[tr, [sc]], y[tr])
        p[te] = lr.predict_proba(A.loc[te, [sc]])[:, 1]
        thr_train[sc].append(float(y[folds != k].mean()))
    oof[sc] = p
    ok = ~np.isnan(p)
    print(f"{sc:24s} AUROC={roc_auc_score(y[ok], A.loc[ok, sc]):.3f} (n={ok.sum()})")
oof.to_csv(os.path.join(R, "oof_predictions_setA.csv"), index=False)
json.dump(chosen, open(os.path.join(R, "selected_hyperparameters.json"), "w"), indent=1)

ALL = list(MODELS) + SCORES
GROUP = {**{m: MODELS[m][2] for m in MODELS}, "SAPS-I": "primary (clinical)", "SOFA": "primary (clinical)"}
NFEAT = {m: len(MODELS[m][1]) for m in MODELS}; NFEAT.update({"SAPS-I": "score", "SOFA": "score"})


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


rows, trows, frows = [], [], []
for m in ALL:
    ok = oof[m].notna().values; yy, pp = y[ok], oof[m].values[ok]
    auc = roc_auc_score(yy, A.loc[ok, m].values if m in SCORES else pp)
    ci, cs = cal_int_slope(yy, pp)
    rows.append(dict(model=m, group=GROUP[m], n_features=NFEAT[m], n=int(ok.sum()), AUROC=auc,
                     AUPRC=average_precision_score(yy, pp), Brier=brier_score_loss(yy, pp), cal_intercept=ci, cal_slope=cs))
    for k in range(5):
        f = ok & (folds == k)
        frows.append(dict(model=m, fold=k, AUROC=roc_auc_score(y[f], oof[m].values[f]), Brier=brier_score_loss(y[f], oof[m].values[f])))
    tprev = np.array([thr_train[m][k] for k in folds])[ok]
    d = conf(yy, pp >= tprev); d.update(model=m, threshold="training-fold prevalence (~0.14)"); trows.append(d)
    d = conf(yy, pp >= 0.30); d.update(model=m, threshold="0.30"); trows.append(d)
perf = pd.DataFrame(rows); perf.to_csv(os.path.join(R, "performance_setA_cv.csv"), index=False)
fperf = pd.DataFrame(frows); fperf.to_csv(os.path.join(R, "performance_by_fold.csv"), index=False)
fperf.groupby("model").agg(AUROC_sd=("AUROC", "std"), AUROC_min=("AUROC", "min"), AUROC_max=("AUROC", "max"),
                           Brier_sd=("Brier", "std")).reset_index().to_csv(os.path.join(R, "fold_variability.csv"), index=False)
pd.DataFrame(trows).to_csv(os.path.join(R, "threshold_metrics.csv"), index=False)

# ---------------------------------------------------------------- paired bootstrap (§5.3)
rng = np.random.default_rng(SEED)
comps = [("XGB-full", "GAM-7"), ("XGB-7", "GAM-7"), ("XGB-15", "GAM-7"), ("LASSO-15", "GAM-7"), ("Tree-7", "GAM-7"),
         ("XGB-full", "XGB-7"), ("XGB-full (native NA)", "XGB-full"), ("GAM-7 + NA indicators", "GAM-7"),
         ("GAM-7", "SAPS-I"), ("GAM-7", "SOFA"), ("XGB-full", "SAPS-I"), ("XGB-full", "SOFA")]
brows = []
for c in comps:
    ii0 = np.where(oof[list(c)].notna().all(axis=1))[0]
    p1, p2 = oof[c[0]].values, oof[c[1]].values
    bb = []
    for _ in range(N_BOOT):
        ii = rng.choice(ii0, len(ii0), replace=True); yy = y[ii]
        bb.append((roc_auc_score(yy, p1[ii]) - roc_auc_score(yy, p2[ii]), brier_score_loss(yy, p1[ii]) - brier_score_loss(yy, p2[ii])))
    bb = np.array(bb)
    brows.append(dict(comparison=f"{c[0]} minus {c[1]}", n=len(ii0),
                      dAUROC=roc_auc_score(y[ii0], p1[ii0]) - roc_auc_score(y[ii0], p2[ii0]),
                      dAUROC_lo=np.percentile(bb[:, 0], 2.5), dAUROC_hi=np.percentile(bb[:, 0], 97.5),
                      dBrier=brier_score_loss(y[ii0], p1[ii0]) - brier_score_loss(y[ii0], p2[ii0]),
                      dBrier_lo=np.percentile(bb[:, 1], 2.5), dBrier_hi=np.percentile(bb[:, 1], 97.5)))
pd.DataFrame(brows).to_csv(os.path.join(R, "paired_bootstrap.csv"), index=False)

# ---------------------------------------------------------------- decision curves
ts = np.arange(0.05, 0.51, 0.01); prev = y.mean()
dca = {"threshold": ts, "Treat all": prev - (1 - prev) * ts / (1 - ts), "Treat none": np.zeros_like(ts)}
CURVE_MODELS = ["GAM-7", "XGB-full", "XGB-7", "LASSO-15", "Tree-7", "SAPS-I", "SOFA"]
for m in CURVE_MODELS:
    ok = oof[m].notna().values; yy, pp = y[ok], oof[m].values[ok]; n = len(yy)
    dca[m] = [((pp >= t) & (yy == 1)).sum() / n - ((pp >= t) & (yy == 0)).sum() / n * t / (1 - t) for t in ts]
dca = pd.DataFrame(dca); dca.to_csv(os.path.join(R, "decision_curves.csv"), index=False)

COL = {"GAM-7": "#1b6ca8", "XGB-full": "#d1495b", "XGB-7": "#edae49", "LASSO-15": "#00798c", "Tree-7": "#6a4c93",
       "SAPS-I": "#555555", "SOFA": "#999999", "XGB-15": "#f08a4b"}
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for m in CURVE_MODELS:
    ok = oof[m].notna().values; sc_ = A.loc[ok, m].values if m in SCORES else oof[m].values[ok]
    fpr, tpr, _ = roc_curve(y[ok], sc_)
    ax[0].plot(fpr, tpr, color=COL[m], lw=1.6, label=f"{m} ({roc_auc_score(y[ok], sc_):.3f})")
    g = pd.DataFrame({"p": oof[m].values[ok], "y": y[ok]})
    g["q"] = pd.qcut(g.p, 10, duplicates="drop"); g = g.groupby("q", observed=True).mean()
    ax[1].plot(g.p, g.y, "o-", color=COL[m], ms=4, lw=1.2, label=m)
ax[0].plot([0, 1], [0, 1], ":", c="k", lw=.8); ax[0].set(xlabel="1 − specificity", ylabel="Sensitivity", title="ROC (out-of-fold, set A)")
ax[1].plot([0, .8], [0, .8], ":", c="k", lw=.8); ax[1].set(xlabel="Mean predicted risk (decile)", ylabel="Observed mortality", title="Calibration (deciles)")
for a_ in ax: a_.legend(fontsize=8, frameon=False); a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_roc_calibration.png"), dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.8))
for m in ["Treat all"] + CURVE_MODELS:
    ax.plot(ts, dca[m], color=COL.get(m, "#bbbbbb"), lw=1 if m == "Treat all" else 1.6, ls="--" if m == "Treat all" else "-", label=m)
ax.axhline(0, c="k", lw=.8); ax.set_ylim(-0.02, prev + 0.02)
ax.set(xlabel="Risk threshold", ylabel="Net benefit", title="Decision curves (out-of-fold, set A)")
ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_decision_curves.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- final fits on all of set A (interpretation)
final = {m: MODELS[m][0]().fit(A[MODELS[m][1]], y).best_estimator_ for m in ["GAM-7", "Tree-7", "LASSO-15", "XGB-full", "XGB-7"]}


def shap_for(m):
    pipe = final[m]; cols = MODELS[m][1]
    Xt = pd.DataFrame(pipe[:-1].transform(A[cols]) if len(pipe) > 1 else A[cols].values, columns=cols)
    sv = shap.TreeExplainer(pipe[-1]).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    return Xt, np.asarray(sv)


Xf, SVf = shap_for("XGB-full")
X7, SV7 = shap_for("XGB-7")

# SHAP global importance (XGB-full) + built-in XGBoost gain importance
imp = pd.DataFrame({"feature": FULL, "mean_abs_SHAP": np.abs(SVf).mean(0),
                    "xgb_gain": [final["XGB-full"][-1].get_booster().get_score(importance_type="gain").get(f"f{i}", 0.0) for i in range(len(FULL))]})
# booster feature names may be real column names depending on input type
gain = final["XGB-full"][-1].get_booster().get_score(importance_type="gain")
if any(k in FULL for k in gain):
    imp["xgb_gain"] = [gain.get(f, 0.0) for f in FULL]
imp = imp.sort_values("mean_abs_SHAP", ascending=False)
imp.to_csv(os.path.join(R, "xgb_full_shap_importance.csv"), index=False)

plt.figure()
shap.summary_plot(SVf, Xf, max_display=20, show=False, plot_size=(8, 7))
plt.title("XGB-full: SHAP summary (top 20)")
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_shap_summary_xgb_full.png"), dpi=150); plt.close("all")

top = imp.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(top.feature, top.mean_abs_SHAP, color=["#1b6ca8" if f in C15 else "#d1495b" for f in top.feature])
ax.set(xlabel="mean |SHAP| (log-odds)", title="XGB-full: global SHAP importance\n(blue = core-15 feature, red = other)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_xgb_importance.png"), dpi=150); plt.close()

# GAM partial effects (pyGAM, 95% CI) vs XGB SHAP dependence vs XGB partial dependence
gam = final["GAM-7"][-1].gam_
fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for ax, c in zip(axes.flat, S7):
    j = S7.index(c)
    XX = gam.generate_X_grid(term=j, n=100)
    pdep, ci = gam.partial_dependence(term=j, X=XX, width=0.95)
    xg = XX[:, j]
    lo, hi = np.nanpercentile(A[c], [1, 99])
    keep = (xg >= lo) & (xg <= hi)
    off = pdep[keep].mean()
    ax.fill_between(xg[keep], ci[keep, 0] - off, ci[keep, 1] - off, color=COL["GAM-7"], alpha=.15)
    ax.plot(xg[keep], pdep[keep] - off, color=COL["GAM-7"], lw=2.2, label="GAM-7 partial effect (95% CI)")
    for m, Xs, SV in [("XGB-7", X7, SV7), ("XGB-full", Xf, SVf)]:
        jj = MODELS[m][1].index(c)
        xv, sv = Xs[c].values, SV[:, jj]
        msk = (xv >= lo) & (xv <= hi)
        bins = np.unique(np.quantile(xv[msk], np.linspace(0, 1, 21)))
        b = pd.DataFrame({"x": xv[msk], "s": sv[msk], "bin": pd.cut(xv[msk], bins, include_lowest=True)}).groupby("bin", observed=True).mean()
        ax.plot(b.x, b.s - b.s.mean(), "o--", ms=3, lw=1.2, color=COL[m], label=f"{m} mean SHAP")
    ax.set_title(SMALL7[c], fontsize=10); ax.set_ylabel("log-odds (centred)", fontsize=8)
    ax2 = ax.twinx(); ax2.hist(A[c].dropna().clip(lo, hi), bins=30, color="#cccccc", alpha=.5); ax2.set_yticks([])
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False); ax.spines[["top"]].set_visible(False)
axes.flat[0].legend(fontsize=7, frameon=False); axes.flat[-1].axis("off")
plt.suptitle("GAM partial effects vs XGBoost SHAP dependence (models fit on all of set A)")
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_partial_effects.png"), dpi=140); plt.close()

# Partial dependence plots for XGB-full (core-7 features), probability scale
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
Xs = A[FULL].sample(min(1000, len(A)), random_state=SEED)
for ax, c in zip(axes.flat, S7):
    # manual PDP/ICE (avoids sklearn<->xgboost estimator-tag incompatibilities)
    gx = np.linspace(*np.nanpercentile(A[c], [2, 98]), 40)
    ice = np.empty((len(Xs), len(gx)))
    for g_i, gv in enumerate(gx):
        Xm = Xs.copy(); Xm[c] = gv
        ice[:, g_i] = final["XGB-full"].predict_proba(Xm)[:, 1]
    for i in range(0, ice.shape[0], 20):
        ax.plot(gx, ice[i], color="#d1495b", alpha=.08, lw=.7)
    ax.plot(gx, ice.mean(0), color="#8c2f39", lw=2.2)
    ax.set_title(SMALL7[c], fontsize=10); ax.set_ylabel("Predicted risk", fontsize=8); ax.spines[["top", "right"]].set_visible(False)
axes.flat[-1].axis("off")
plt.suptitle("XGB-full: partial dependence (bold) and ICE curves (5% sample) for the 7 GAM predictors")
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_pdp_xgb_full.png"), dpi=130); plt.close()

# SHAP interaction strength (XGB-7): which pairs interact (not capturable by additive GAM)
inter = shap.TreeExplainer(final["XGB-7"][-1]).shap_interaction_values(X7)
inter = inter[1] if isinstance(inter, list) else inter
M = np.abs(inter).mean(0)
pairs = [dict(var1=S7[i], var2=S7[j], mean_abs_interaction=2 * M[i, j]) for i in range(7) for j in range(i + 1, 7)]
pd.DataFrame(pairs).sort_values("mean_abs_interaction", ascending=False).to_csv(os.path.join(R, "xgb7_shap_interactions.csv"), index=False)
pd.DataFrame({"feature": S7, "mean_abs_main_effect": np.diag(M), "mean_abs_total_SHAP": np.abs(SV7).mean(0)}).to_csv(
    os.path.join(R, "xgb7_shap_main_vs_total.csv"), index=False)

# Decision tree rules
tree = final["Tree-7"].named_steps["dt"]
open(os.path.join(R, "tree_rules.txt"), "w").write(export_text(tree, feature_names=S7, show_weights=True))
json.dump({"depth": int(tree.get_depth()), "n_leaves": int(tree.get_n_leaves())}, open(os.path.join(R, "tree_size.json"), "w"))

# LASSO coefficients + selection frequency across outer folds
las = final["LASSO-15"].named_steps["lr"]
coef = pd.DataFrame({"feature": C15, "label": [CORE15[c] for c in C15], "coef_per_SD": las.coef_[0]})
selfreq = np.zeros(len(C15))
for k in range(5):
    e = lasso_pipe().fit(A.loc[folds != k, C15], y[folds != k]).best_estimator_
    selfreq += np.abs(e.named_steps["lr"].coef_[0]) > 1e-8
coef["selected_in_outer_folds_of_5"] = selfreq.astype(int)
coef.sort_values("coef_per_SD", key=abs, ascending=False).to_csv(os.path.join(R, "lasso_coefficients.csv"), index=False)

# GAM effective degrees of freedom (complexity summary, §5.2)
gstats = final["GAM-7"][-1].gam_.statistics_
json.dump({"GAM-7_edof": float(gstats["edof"]), "GAM-7_lam": float(final["GAM-7"].get_params()["gam__lam"])},
          open(os.path.join(R, "gam_complexity.json"), "w"))

# ---------------------------------------------------------------- held-out sets B / C
ext = []
for st in ["b", "c"]:
    T = load(st)
    if T is None:
        continue
    yt = T["In-hospital_death"].values.astype(int)
    for m in ["GAM-7", "XGB-full", "XGB-7", "LASSO-15", "Tree-7"]:
        pt = final[m].predict_proba(T[MODELS[m][1]])[:, 1]
        ci, cs = cal_int_slope(yt, pt)
        ext.append(dict(set=st.upper(), model=m, n=len(yt), AUROC=roc_auc_score(yt, pt), AUPRC=average_precision_score(yt, pt),
                        Brier=brier_score_loss(yt, pt), cal_intercept=ci, cal_slope=cs))
    for sc in SCORES:
        ok = T[sc].notna().values
        ext.append(dict(set=st.upper(), model=sc, n=int(ok.sum()), AUROC=roc_auc_score(yt[ok], T.loc[ok, sc])))
if ext:
    pd.DataFrame(ext).to_csv(os.path.join(R, "performance_external_sets.csv"), index=False)

import xgboost, pygam, sklearn
json.dump({"n": int(len(A)), "deaths": int(y.sum()), "mortality_pct": float(y.mean() * 100), "quick": QUICK, "n_boot": N_BOOT,
           "n_iter_xgb": N_ITER_XGB, "versions": {"xgboost": xgboost.__version__, "pygam": getattr(pygam, "__version__", "?"),
                                                    "shap": shap.__version__, "sklearn": sklearn.__version__}},
          open(os.path.join(R, "run_info.json"), "w"), indent=1)
print(perf.round(3).to_string()); print(pd.DataFrame(brows).round(4).to_string())
