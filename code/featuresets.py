"""Prespecified feature sets (Proposal §3)."""

# 7-predictor small set (GAM, tree, XGB-7) with prespecified 24 h summaries
SMALL7 = {
    "Age": "Age",
    "GCS_min": "GCS (min)",
    "HR_mean": "Heart rate (mean)",
    "MAPc_mean": "MAP (mean; invasive, else non-invasive)",
    "RespRate_mean": "Respiratory rate (mean)",
    "SaO2_min": "SaO2 (min)",
    "BUN_max": "BUN (max)",
}

# 15-predictor core set (LASSO, XGB-15)
CORE15 = dict(SMALL7)
CORE15.update({
    "PaO2_min": "PaO2 (min)",
    "Creatinine_max": "Creatinine (max)",
    "WBC_max": "WBC (max)",
    "Platelets_min": "Platelets (min)",
    "HCT_min": "Hematocrit (min)",
    "Glucose_max": "Glucose (max)",
    "Bilirubin_max": "Bilirubin (max)",
    "Temp_mean": "Temperature (mean)",
})


def full_features(columns):
    """All eligible first-24h features: static descriptors + mean/min/max/last/count
    of every time-series variable + derived (combined MAP, urine total)."""
    drop = {"RecordID", "last_obs_min", "n_obs_24h"}
    keep = [c for c in columns if c not in drop and not c.endswith("_first")]
    return keep
