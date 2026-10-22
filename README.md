# 707 Final Project

Compare a simple, interpretable model with a black-box model for predicting **in-hospital mortality** in ICU patients, using only data from the first 24 hours of admission.

**Team:** Junjie Cai, Yifei Chen, Jiahui Hu, Yuyao Lei

## Overview

The project uses the public [PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) dataset (ODC-BY). Each record is one ICU patient. The outcome is in-hospital death.

Models include a logistic GAM, a shallow decision tree, LASSO and ridge logistic regression, XGBoost, and the clinical scores SAPS-I and SOFA. Performance is compared with stratified 5-fold cross-validation, using AUROC, calibration, Brier score, threshold-based metrics, and decision curve analysis.

The full plan is in [Proposal.md](Proposal.md).
