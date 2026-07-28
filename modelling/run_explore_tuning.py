"""Run the ad-hoc exploratory nested-CV tuning pass for the non-winning
Stage A models, so their tuned-vs-default improvement can be compared against
the official ExtraTrees/Screening1/W120/H1 winner from 02d_nested_tuning_cv.ipynb.

Replaces four separate copy-pasted scripts (explore_lightgbm_tuning.py,
explore_randomforest_tuning.py, explore_xgboost_tuning.py,
explore_xgboost_screening2_tuning.py) with one config table -- see
RINGKASAN_REVISI_METODOLOGI.md #4 for the same table alongside results.
"""
from explore_lib import run_exploratory_tuning

CONFIGS = [
    dict(model_name="RandomForest", cov_group="All_Covariates", window=20, horizon=1,
         out_prefix="explore_randomforest_allcov_w20h1"),
    dict(model_name="XGBoost", cov_group="Screening2", window=20, horizon=1,
         out_prefix="explore_xgboost_screening2_w20h1"),
    dict(model_name="XGBoost", cov_group="USDIDR", window=20, horizon=1,
         out_prefix="explore_xgboost_usdidr_w20h1"),
    dict(model_name="LightGBM", cov_group="Screening1", window=20, horizon=1,
         out_prefix="explore_lightgbm_screening1_w20h1"),
]

if __name__ == "__main__":
    for cfg in CONFIGS:
        run_exploratory_tuning(**cfg)
