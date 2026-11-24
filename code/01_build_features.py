"""Build patient-level feature tables from PhysioNet/CinC 2012 records.

Usage: python 01_build_features.py <raw_dir_with_set-a_set-b_set-c> <out_dir>

Only measurements with Time <= 24:00 (first 24 h of the ICU stay) are used.
Outputs (per set): features_<set>.csv  (one row per RecordID)
"""
import sys, os, glob
import numpy as np
import pandas as pd

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

STATIC = ["Age", "Gender", "Height", "ICUType"]
TS_VARS = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
           "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR",
           "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP",
           "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
           "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight"]

# Physiologically implausible values -> treated as missing (documented in report)
PLAUSIBLE = {"HR": (1, 300), "MAP": (1, 300), "NIMAP": (1, 300), "SysABP": (1, 300),
             "NISysABP": (1, 300), "DiasABP": (1, 250), "NIDiasABP": (1, 250),
             "Temp": (25, 45), "pH": (6.5, 8.0), "Weight": (20, 300),
             "Height": (100, 230), "RespRate": (1, 80), "SaO2": (20, 100),
             "PaO2": (10, 700), "FiO2": (0.21, 1.0), "Glucose": (10, 1500),
             "HCT": (5, 70), "Urine": (0, 5000)}


def read_record(path):
    d = pd.read_csv(path)
    d = d.dropna(subset=["Parameter"])
    hh = d["Time"].str.split(":", expand=True).astype(int)
    d["t"] = hh[0] * 60 + hh[1]
    rid = int(d.loc[d.Parameter == "RecordID", "Value"].iloc[0])
    st = d[d.t == 0].set_index("Parameter")["Value"]
    row = {"RecordID": rid}
    for s in STATIC:
        v = st.get(s, np.nan)
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        row[s] = np.nan if (pd.isna(v) or v < 0) else float(v)
    ts = d[(~d.Parameter.isin(STATIC + ["RecordID"])) & (d.t <= 24 * 60)].copy()
    # Weight at t=0 is a descriptor (-1 = missing); drop -1 everywhere
    ts = ts[~((ts.Parameter == "Weight") & (ts.Value <= 0))]
    # FiO2 sometimes recorded in percent
    m = (ts.Parameter == "FiO2") & (ts.Value > 1)
    ts.loc[m, "Value"] = ts.loc[m, "Value"] / 100
    for p, (lo, hi) in PLAUSIBLE.items():
        bad = (ts.Parameter == p) & ((ts.Value < lo) | (ts.Value > hi))
        ts = ts[~bad]
    if not np.isnan(row["Height"]) and not (100 <= row["Height"] <= 230):
        row["Height"] = np.nan
    g = ts.groupby("Parameter")["Value"]
    agg = g.agg(["mean", "min", "max", "last", "first", "count"])
    for v in TS_VARS:
        if v in agg.index:
            a = agg.loc[v]
            for k in ["mean", "min", "max", "last", "first", "count"]:
                row[f"{v}_{k}"] = float(a[k])
        else:
            for k in ["mean", "min", "max", "last", "first"]:
                row[f"{v}_{k}"] = np.nan
            row[f"{v}_count"] = 0.0
    # Combined MAP: invasive if available, else non-invasive
    for k in ["mean", "min", "max"]:
        row[f"MAPc_{k}"] = row[f"MAP_{k}"] if not np.isnan(row[f"MAP_{k}"]) else row[f"NIMAP_{k}"]
    row["Urine_sum"] = float(ts.loc[ts.Parameter == "Urine", "Value"].sum()) if row["Urine_count"] > 0 else np.nan
    row["last_obs_min"] = int(d["t"].max())
    row["n_obs_24h"] = int(len(ts))
    return row


for s in ["a", "b", "c"]:
    files = sorted(glob.glob(os.path.join(RAW, f"set-{s}", "*.txt")))
    rows = [read_record(f) for f in files]
    df = pd.DataFrame(rows).sort_values("RecordID")
    df.to_csv(os.path.join(OUT, f"features_{s}.csv"), index=False)
    print(s, df.shape)
