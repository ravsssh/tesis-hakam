"""Shared runner for ad-hoc exploratory nested-CV tuning outside the official
Stage A/B pipeline (see 02d_nested_tuning_cv.ipynb for the official winner).

Consolidates what used to be four near-identical ~140-line scripts
(explore_lightgbm_tuning.py, explore_randomforest_tuning.py,
explore_xgboost_tuning.py, explore_xgboost_screening2_tuning.py) into a single
function, parameterized by (model, covariate group, window, horizon). See
run_explore_tuning.py for the config table that replaces those four files.
"""
import glob
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import optuna
import pandas as pd
from optuna.samplers import TPESampler

import cv_lib as cv

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_exploratory_tuning(model_name, cov_group, window, horizon, out_prefix,
                            n_outer_folds=5, n_inner_folds=3,
                            n_trials_per_outer_fold=40, n_trials_production=50,
                            df_merged=None, df_stage_a=None):
    """Same nested-CV Optuna mechanism as 02d_nested_tuning_cv.ipynb (Stage B),
    scoped to an arbitrary (model, covariate group, window, horizon) instead of
    the Stage A winner. Purely exploratory -- writes to out_prefix-tagged files,
    never touches the official nested_cv_outer_fold_results.csv /
    optuna_tuning_results.joblib / final_config.joblib.
    """
    if df_merged is None:
        df_merged = joblib.load(sorted(glob.glob("saved_models/df_merged_*.joblib"), reverse=True)[0])
    if df_stage_a is None:
        df_stage_a = pd.read_csv("stage_a_fold_results.csv")

    cov_vars = cv.SCENARIO_COVARIATES[cov_group]
    embargo = horizon
    out_folds_csv = f"{out_prefix}_folds.csv"
    out_summary_csv = f"{out_prefix}_summary.csv"
    out_tuning_joblib = f"saved_models/{out_prefix}_tuning.joblib"

    target_ts, cov_ts = cv.to_series(df_merged, "IHSG", cov_vars if cov_vars else None)
    n = len(target_ts)
    outer_folds = cv.expanding_window_folds(n, n_folds=n_outer_folds, embargo=embargo)
    print(f"[{out_prefix}] n={n}, {len(outer_folds)} outer folds")

    df_default = df_stage_a[
        (df_stage_a["Model"] == model_name) &
        (df_stage_a["Covariates"] == cov_group) &
        (df_stage_a["Window"] == window) &
        (df_stage_a["Horizon"] == horizon)
    ].sort_values("Fold").reset_index(drop=True)

    assert len(df_default) == len(outer_folds), (
        f"[{out_prefix}] Stage A has {len(df_default)} fold rows, rebuilt {len(outer_folds)} "
        "outer folds -- mismatch"
    )
    for f, (_, row) in zip(outer_folds, df_default.iterrows()):
        rebuilt = str(target_ts[f["test_start"]].start_time().date())
        assert rebuilt == row["test_start_date"], (
            f"[{out_prefix}] fold {f['fold']} date mismatch: {rebuilt} vs {row['test_start_date']}"
        )
    print(f"[{out_prefix}] outer folds match Stage A's fold boundaries -- paired comparison valid.")

    if os.path.exists(out_folds_csv):
        df_resume = pd.read_csv(out_folds_csv)
        outer_results = df_resume.to_dict("records")
        done_folds = set(df_resume["Fold"])
        print(f"[{out_prefix}] resuming: {len(outer_results)} outer folds already done: {sorted(done_folds)}")
    else:
        outer_results = []
        done_folds = set()

    for f in outer_folds:
        if f["fold"] in done_folds:
            print(f"[{out_prefix}] outer fold {f['fold']} ... skipped (already in checkpoint)")
            continue

        print(f"\n[{out_prefix}] --- outer fold {f['fold']} "
              f"(train_end={f['train_end']}, test=[{f['test_start']}:{f['test_end']}]) ---")

        objective, inner_folds = cv.make_inner_cv_objective(
            model_name, target_ts, cov_ts, window, horizon,
            outer_train_end=f["train_end"], n_inner_folds=n_inner_folds, embargo=embargo,
        )
        print(f"  inner folds: {len(inner_folds)} (bounded by outer_train_end={f['train_end']})")

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42),
                                     study_name=f"{out_prefix}_outer{f['fold']}")
        study.optimize(objective, n_trials=n_trials_per_outer_fold, show_progress_bar=False)
        print(f"  best inner-CV MAPE: {study.best_value:.4f}  params={study.best_params}")

        tuned_metrics = cv.run_fold(model_name, study.best_params, target_ts, cov_ts, window, horizon,
                                     f["train_end"], f["test_start"], f["test_end"])
        default_row = df_default[df_default["Fold"] == f["fold"]].iloc[0]

        outer_results.append({
            "Fold": f["fold"],
            "train_end_date": tuned_metrics["train_end_date"],
            "test_start_date": tuned_metrics["test_start_date"],
            "test_end_date": tuned_metrics["test_end_date"],
            "regime_flags": tuned_metrics["regime_flags"],
            "default_mape": default_row["mape"], "tuned_mape": tuned_metrics["mape"],
            "default_da": default_row["da"], "tuned_da": tuned_metrics["da"],
            "default_mase": default_row["mase"], "tuned_mase": tuned_metrics["mase"],
            "best_params": study.best_params,
        })
        improvement = (default_row["mape"] - tuned_metrics["mape"]) / default_row["mape"] * 100
        print(f"  outer test: default MAPE={default_row['mape']:.4f}%  tuned MAPE={tuned_metrics['mape']:.4f}%  "
              f"improvement={improvement:.2f}%"
              + (f"  [REGIME: {tuned_metrics['regime_flags']}]" if tuned_metrics["regime_flags"] else ""))

        pd.DataFrame(outer_results).to_csv(out_folds_csv, index=False)
        print(f"  -> checkpoint saved ({len(outer_results)} outer folds)")

    df_outer = pd.DataFrame(outer_results)
    df_outer.to_csv(out_folds_csv, index=False)
    print(f"[{out_prefix}] saved: {out_folds_csv} ({len(df_outer)} rows)")

    df_outer["improvement_pct"] = (df_outer["default_mape"] - df_outer["tuned_mape"]) / df_outer["default_mape"] * 100
    summary = {
        "model": model_name, "covariates": cov_group, "window": window, "horizon": horizon,
        "default_mape_mean": df_outer["default_mape"].mean(), "default_mape_std": df_outer["default_mape"].std(),
        "tuned_mape_mean": df_outer["tuned_mape"].mean(), "tuned_mape_std": df_outer["tuned_mape"].std(),
        "mean_improvement_pct": df_outer["improvement_pct"].mean(),
        "n_outer_folds": len(df_outer),
        "n_folds_with_regime_flag": df_outer["regime_flags"].notna().sum(),
    }
    pd.DataFrame([summary]).to_csv(out_summary_csv, index=False)
    print(f"[{out_prefix}] mean improvement (default -> tuned): {summary['mean_improvement_pct']:.2f}%")
    print(f"[{out_prefix}] saved: {out_summary_csv}")

    if not os.path.exists(out_tuning_joblib):
        final_train_end = outer_folds[-1]["train_end"]
        prod_objective, prod_inner_folds = cv.make_inner_cv_objective(
            model_name, target_ts, cov_ts, window, horizon,
            outer_train_end=final_train_end, n_inner_folds=n_inner_folds, embargo=embargo,
        )
        print(f"[{out_prefix}] production tuning: {len(prod_inner_folds)} inner folds "
              f"bounded by train_end={final_train_end}")
        prod_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42),
                                          study_name=f"{out_prefix}_production")
        prod_study.optimize(prod_objective, n_trials=n_trials_production, show_progress_bar=False)
        print(f"[{out_prefix}] production best inner-CV MAPE: {prod_study.best_value:.4f}")
        print(f"[{out_prefix}] production best params: {prod_study.best_params}")
        joblib.dump({model_name: {"best_params": prod_study.best_params, "best_value": prod_study.best_value}},
                    out_tuning_joblib)
        print(f"[{out_prefix}] saved: {out_tuning_joblib}")

    print(f"[{out_prefix}] DONE\n")
    return summary
