"""Cohort + data-quality summary (Proposal §2.5, §3.5). Does NOT require outcomes.
Usage: python 02_data_quality.py <derived_dir> <results_dir>
"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from featuresets import SMALL7, CORE15, full_features

D, R = sys.argv[1], sys.argv[2]
os.makedirs(R, exist_ok=True)
summ = {}
dfs = {s: pd.read_csv(os.path.join(D, f"features_{s}.csv")) for s in "abc"}
for s, df in dfs.items():
    summ[s] = {"n_records": int(len(df)),
               "n_no_obs_24h": int((df.n_obs_24h == 0).sum()),
               "median_obs_24h": float(df.n_obs_24h.median()),
               "age_median": float(df.Age.median()),
               "female_pct": float((df.Gender == 0).mean() * 100),
               "icutype_pct": {int(k): round(v * 100, 1) for k, v in df.ICUType.value_counts(normalize=True).sort_index().items()}}
json.dump(summ, open(os.path.join(R, "cohort_summary.json"), "w"), indent=2)

a = dfs["a"]
# Missingness & distribution of core predictors (set A)
rows = []
for c, lab in CORE15.items():
    x = a[c]
    rows.append({"feature": c, "label": lab, "in_small7": c in SMALL7,
                 "missing_pct": round(x.isna().mean() * 100, 1),
                 "median": round(x.median(), 2), "q1": round(x.quantile(.25), 2),
                 "q3": round(x.quantile(.75), 2), "min": round(x.min(), 2), "max": round(x.max(), 2)})
miss = pd.DataFrame(rows)
miss.to_csv(os.path.join(R, "core15_missingness_setA.csv"), index=False)

# Missingness for all variables (any measurement in first 24 h)
base = sorted({c.rsplit("_", 1)[0] for c in a.columns if c.endswith("_count")})
allm = pd.DataFrame({"variable": base,
                     "pct_patients_with_no_measurement": [round((a[f"{v}_count"] == 0).mean() * 100, 1) for v in base],
                     "median_n_measurements_if_any": [float(a.loc[a[f"{v}_count"] > 0, f"{v}_count"].median()) if (a[f"{v}_count"] > 0).any() else 0 for v in base]}
                    ).sort_values("pct_patients_with_no_measurement")
allm.to_csv(os.path.join(R, "all_variables_missingness_setA.csv"), index=False)

# Missingness pattern among the 7 predictors
pat = a[list(SMALL7)].isna()
patc = pat.apply(lambda r: "".join("M" if v else "." for v in r), axis=1).value_counts().head(10)
patdf = pd.DataFrame({"pattern (" + " ".join(SMALL7) + ")": patc.index, "n": patc.values})
patdf.to_csv(os.path.join(R, "small7_missing_patterns_setA.csv"), index=False)
# Ventilation vs RespRate / SaO2 missingness
vent = a["MechVent_count"] > 0
vtab = pd.DataFrame({"mech_vent_24h": [False, True],
                     "n": [int((~vent).sum()), int(vent.sum())],
                     "RespRate_missing_pct": [round(a.loc[~vent, "RespRate_mean"].isna().mean() * 100, 1), round(a.loc[vent, "RespRate_mean"].isna().mean() * 100, 1)],
                     "SaO2_missing_pct": [round(a.loc[~vent, "SaO2_min"].isna().mean() * 100, 1), round(a.loc[vent, "SaO2_min"].isna().mean() * 100, 1)],
                     "PaO2_missing_pct": [round(a.loc[~vent, "PaO2_min"].isna().mean() * 100, 1), round(a.loc[vent, "PaO2_min"].isna().mean() * 100, 1)]})
vtab.to_csv(os.path.join(R, "ventilation_vs_missingness_setA.csv"), index=False)

# Redundancy: Spearman correlations among core + related
rel = list(CORE15) + ["SysABP_mean", "NISysABP_mean", "DiasABP_mean", "NIMAP_mean", "MAP_mean", "FiO2_max", "Lactate_max"]
corr = a[rel].corr(method="spearman")
corr.round(3).to_csv(os.path.join(R, "spearman_corr_setA.csv"))
pairs = []
for i, c1 in enumerate(rel):
    for c2 in rel[i + 1:]:
        r = corr.loc[c1, c2]
        if abs(r) >= 0.5:
            n = int(a[[c1, c2]].dropna().shape[0])
            pairs.append({"var1": c1, "var2": c2, "spearman": round(r, 3), "n_pairwise": n})
pd.DataFrame(pairs).sort_values("spearman", key=abs, ascending=False).to_csv(os.path.join(R, "redundant_pairs_setA.csv"), index=False)

nfull = len(full_features(a.columns))
json.dump({"n_full_features": nfull, "n_core15": 15, "n_small7": 7}, open(os.path.join(R, "featureset_sizes.json"), "w"))

# Figures
fig, ax = plt.subplots(figsize=(7, 8))
am = allm.sort_values("pct_patients_with_no_measurement", ascending=True)
cols = ["#2a6f97" if v in {c.split("_")[0] for c in CORE15} or v == "MAPc" else "#a9bcd0" for v in am.variable]
ax.barh(am.variable, am.pct_patients_with_no_measurement, color=cols)
ax.set_xlabel("% of set-A patients with no measurement in first 24 h")
ax.set_title("Variable availability, first 24 h (dark = core-15 variable)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_missingness_all.png"), dpi=150); plt.close()

fig, axes = plt.subplots(3, 5, figsize=(15, 8))
for ax, (c, lab) in zip(axes.flat, CORE15.items()):
    x = a[c].dropna()
    lo, hi = x.quantile([.005, .995])
    ax.hist(x.clip(lo, hi), bins=40, color="#2a6f97" if c in SMALL7 else "#7a9cbf")
    ax.set_title(f"{lab}\nmissing {a[c].isna().mean()*100:.0f}%", fontsize=9)
    ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
plt.suptitle("Distributions of the 15 core predictors, set A (dark = small-7 set; 0.5–99.5% clipped)")
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_core15_distributions.png"), dpi=130); plt.close()
print(json.dumps(summ, indent=1)); print(miss.to_string()); print(vtab.to_string()); print(patdf.to_string()); print(pd.DataFrame(pairs).to_string()); print("full features", nfull)
