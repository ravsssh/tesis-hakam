"""Shared walk-forward CV, transform, and model-building helpers.

Consolidates logic that used to be copy-pasted (with small drifts) across
02_phase1_screening.ipynb, 02a_phase1_screening_v2.ipynb, 02b_default_params.ipynb,
03_shap_analysis.ipynb and 04_visualization.ipynb, and extends it with an
expanding-window CV fold helper that supports an embargo gap between training
and evaluation, used by 02c_model_selection_cv.ipynb (Stage A) and
02d_nested_tuning_cv.ipynb (Stage B).
"""

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.models import RandomForestModel, XGBModel, LightGBMModel
from darts.dataprocessing.transformers import Scaler, Diff
from darts.utils.missing_values import fill_missing_values
from sklearn.ensemble import ExtraTreesRegressor

try:
    from darts.models import SKLearnModel
except ImportError:
    from darts.models import RegressionModel as SKLearnModel


# --------------------------------------------------------------------------
# Variable/scenario registries (verbatim from 00_data_preprocessing.ipynb /
# 02b_default_params.ipynb so downstream notebooks don't drift from them)
# --------------------------------------------------------------------------

LEVEL_VARS = ["M2", "USDIDR", "Coal", "Copper", "Nickel", "Silver", "Tin", "STI", "Gold", "WTI", "GDP"]
RATE_VARS = ["BI_Rate", "CPI", "NPL_Ratio", "US_Treasury_10Y"]

WINDOWS = [20, 120]
HORIZONS = [1, 5, 20]

DEFAULT_CONFIGS = {
    "RandomForest": {
        "n_estimators": 300,
        "max_depth": 5,
        "max_features": "sqrt",
        "max_samples": 0.7,
    },
    "ExtraTrees": {
        "n_estimators": 300,
        "max_depth": 5,
        "max_features": "sqrt",
    },
    "XGBoost": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
    },
    "LightGBM": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
    },
}

MODEL_NAMES = ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM"]

SINGLE_COVARIATES = {
    # Macro (7)
    "BI_Rate": ["BI_Rate"],
    "CPI": ["CPI"],
    "M2": ["M2"],
    "NPL_Ratio": ["NPL_Ratio"],
    "USDIDR": ["USDIDR"],
    "GDP": ["GDP"],
    "US_Treasury_10Y": ["US_Treasury_10Y"],
    # Commodity (7)
    "Coal": ["Coal"],
    "Copper": ["Copper"],
    "Nickel": ["Nickel"],
    "Silver": ["Silver"],
    "Tin": ["Tin"],
    "Gold": ["Gold"],
    "WTI": ["WTI"],
    # Regional (1)
    "STI": ["STI"],
}

GROUP_COVARIATES = {
    "Baseline": [],
    "Screening1": ["Silver", "WTI", "Gold", "STI", "Coal", "Tin", "NPL_Ratio"],
    "Screening2": ["Silver", "WTI", "Gold", "STI", "Coal", "Tin", "NPL_Ratio", "CPI", "USDIDR", "Nickel"],
    "All_Commodity_STI": ["Coal", "Copper", "Nickel", "Silver", "Tin", "Gold", "WTI", "STI"],
    "All_Macro_no_UST": ["BI_Rate", "CPI", "M2", "NPL_Ratio", "USDIDR", "GDP"],
    "All_Covariates": [
        "BI_Rate", "CPI", "M2", "NPL_Ratio", "USDIDR", "GDP",
        "Coal", "Copper", "Nickel", "Silver", "Tin", "Gold", "WTI", "STI",
        "US_Treasury_10Y",
    ],
}

# Stage A model-selection grid: 1 baseline (no covariates) + 15 single covariates
# + 5 non-baseline groups = 21 scenarios. Deliberately built from GROUP_COVARIATES /
# SINGLE_COVARIATES above rather than duplicating "no covariates" under two names
# (the old pipeline had both a "None" single scenario and a "Baseline" group scenario
# for the same experiment).
SCENARIO_COVARIATES = {"Baseline": []}
SCENARIO_COVARIATES.update(SINGLE_COVARIATES)
SCENARIO_COVARIATES.update({k: v for k, v in GROUP_COVARIATES.items() if k != "Baseline"})

assert len(SCENARIO_COVARIATES) == 21, f"expected 21 scenarios, got {len(SCENARIO_COVARIATES)}"

# Placeholder regime windows for outlier-fold flagging — adjust if your thesis
# committee wants different boundaries.
REGIME_WINDOWS = {
    "covid_crash": (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-30")),
    "rate_hike_cycle_2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}


# --------------------------------------------------------------------------
# Expanding-window CV folds with embargo
# --------------------------------------------------------------------------

def expanding_window_folds(n, n_folds=5, min_train_fraction=0.4, test_fraction=0.15, embargo=0):
    """Expanding-window walk-forward fold boundaries, index-based.

    Extends the fold formula already used in the (superseded)
    02_phase1_screening.ipynb's train_and_evaluate (train_end = min_train + fold*step,
    test_end = min(train_end+test_size, n)) by inserting an embargo gap between the
    end of training and the start of evaluation: test_start = train_end + embargo.
    The training boundary itself (train_end) is untouched by the embargo -- only
    where *evaluation* begins shifts. Folds where the embargo/test window no longer
    fits are dropped, not padded.

    Returns a list of dicts: {"fold", "train_end", "test_start", "test_end"}, all
    as integer indices into the underlying TimeSeries.
    """
    min_train = int(n * min_train_fraction)
    test_size = int(n * test_fraction)
    available = n - min_train - test_size - embargo
    step = max(1, available // max(1, n_folds - 1))

    folds = []
    for fold in range(n_folds):
        train_end = min_train + fold * step
        test_start = train_end + embargo
        test_end = min(test_start + test_size, n)
        if test_start >= test_end or test_end > n:
            break
        folds.append({"fold": fold, "train_end": train_end, "test_start": test_start, "test_end": test_end})

    for f in folds:
        assert f["test_start"] >= f["train_end"] + embargo
        assert f["test_start"] < f["test_end"] <= n

    return folds


def assert_embargo_gap(target_ts, train_end, test_start):
    """Runtime check that the embargo produced an actual date gap, not just an
    index gap (guards against off-by-one index math silently collapsing it)."""
    if train_end == test_start:
        return
    train_last_date = target_ts[train_end - 1].end_time()
    test_first_date = target_ts[test_start].start_time()
    assert test_first_date > train_last_date, (
        f"embargo did not create a date gap: train ends {train_last_date}, "
        f"test starts {test_first_date}"
    )


def flag_regime_overlap(test_start_date, test_end_date, regime_windows=None):
    """Return the list of regime names whose window overlaps [test_start_date, test_end_date)."""
    regime_windows = regime_windows or REGIME_WINDOWS
    flags = []
    for name, (start, end) in regime_windows.items():
        if test_start_date < end and test_end_date > start:
            flags.append(name)
    return flags


# --------------------------------------------------------------------------
# Series construction / transforms (train-only fit, verbatim pattern reused
# from 02a/02b, parameterized by an explicit train_end index)
# --------------------------------------------------------------------------

def to_series(df, target_col, covariates=None):
    target = TimeSeries.from_dataframe(
        df, time_col="date", value_cols=target_col,
        fill_missing_dates=True, freq="B",
    )
    target = fill_missing_values(target)
    cov = None
    if covariates:
        cov = TimeSeries.from_dataframe(
            df, time_col="date", value_cols=covariates,
            fill_missing_dates=True, freq="B",
        )
        cov = fill_missing_values(cov)
    return target, cov


def fit_target_transform(target_ts, train_end, eval_end):
    """Fit log->Diff(1)->Scaler on target_ts[:train_end] only, apply to
    target_ts[:eval_end]. Returns (train_scaled, eval_scaled, scaler)."""
    train_ts = target_ts[:train_end]
    eval_ts = target_ts[:eval_end]

    train_log = train_ts.map(np.log)
    eval_log = eval_ts.map(np.log)

    differencer = Diff(lags=1)
    train_log_diff = differencer.fit_transform(train_log)
    assert train_log.end_time() == target_ts[train_end - 1].end_time(), "differencer fit past train_end"
    eval_log_diff = differencer.transform(eval_log)

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_log_diff)
    eval_scaled = scaler.transform(eval_log_diff)
    return train_scaled, eval_scaled, scaler


def fit_covariate_transform(cov_ts, train_end, eval_end):
    """Fit log/Diff/Scaler on cov_ts[:train_end] only, apply to cov_ts[:eval_end]."""
    if cov_ts is None:
        return None

    train_cov = cov_ts[:train_end]
    eval_cov = cov_ts[:eval_end]
    cov_cols = cov_ts.components.tolist()
    level_cols = [c for c in cov_cols if c in LEVEL_VARS]
    rate_cols = [c for c in cov_cols if c in RATE_VARS]

    parts_train, parts_eval = [], []
    if level_cols:
        d = Diff(lags=1)
        parts_train.append(d.fit_transform(train_cov[level_cols].map(np.log)))
        parts_eval.append(d.transform(eval_cov[level_cols].map(np.log)))
    if rate_cols:
        d = Diff(lags=1)
        parts_train.append(d.fit_transform(train_cov[rate_cols]))
        parts_eval.append(d.transform(eval_cov[rate_cols]))

    ct, ce = parts_train[0], parts_eval[0]
    for pt, pe in zip(parts_train[1:], parts_eval[1:]):
        ct = ct.stack(pt)
        ce = ce.stack(pe)

    cov_scaler = Scaler()
    cov_scaler.fit(ct)
    return cov_scaler.transform(ce)


# --------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------

def build_model(model_name, params, window, horizon, has_covariates):
    """Generic model factory taking an explicit params dict -- works for
    DEFAULT_CONFIGS, an Optuna trial's suggested params, or final tuned params."""
    common = {
        "lags": window,
        "lags_past_covariates": window if has_covariates else None,
        "output_chunk_length": horizon,
    }
    if model_name == "RandomForest":
        return RandomForestModel(**common, random_state=42, n_jobs=-1, **params)
    elif model_name == "ExtraTrees":
        return SKLearnModel(**common, model=ExtraTreesRegressor(random_state=42, n_jobs=-1, **params))
    elif model_name == "XGBoost":
        return XGBModel(**common, random_state=42, n_jobs=-1, **params)
    elif model_name == "LightGBM":
        return LightGBMModel(**common, random_state=42, n_jobs=-1, verbose=-1, **params)
    raise ValueError(f"Unknown model: {model_name}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def inverse_and_metrics(forecast_list, full_ts, scaler, train_end):
    """Inverse-transform scaled log-diff forecasts back to IHSG price level
    and compute mape/mae/rmse/r2/da/mase/hit_large. Verbatim logic from
    02a_phase1_screening_v2.ipynb's inverse_and_metrics, parameterized by
    train_end instead of split_idx."""
    full_log = full_ts.map(np.log)
    all_dates, all_prices = [], []
    for chunk_scaled in forecast_list:
        chunk_diff = scaler.inverse_transform(chunk_scaled)
        dates = chunk_diff.time_index
        vals = chunk_diff.values().flatten()
        idx = full_ts.get_index_at_point(dates[0])
        if idx == 0:
            continue
        anchor = full_log[idx - 1].values()[0][0]
        log_prices = anchor + np.cumsum(vals)
        all_dates.extend(dates)
        all_prices.extend(np.exp(log_prices))

    pred_df = pd.DataFrame({"date": pd.to_datetime(all_dates), "predicted": all_prices})
    actual_df = full_ts.to_dataframe().reset_index()
    actual_df.columns = ["date", "actual"]
    eval_df = pd.merge(actual_df, pred_df, on="date", how="inner")
    y_true = eval_df["actual"].values
    y_pred = eval_df["predicted"].values

    return compute_metrics(y_true, y_pred, full_ts[:train_end].values().flatten())


def compute_metrics(y_true, y_pred, y_train):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    a_dir = np.diff(y_true)
    p_dir = y_pred[1:] - y_true[:-1]
    da = np.mean((a_dir > 0) == (p_dir > 0)) * 100 if len(a_dir) > 0 else np.nan
    naive_mae = np.mean(np.abs(np.diff(y_train))) if len(y_train) > 1 else np.nan
    mase = mae / naive_mae if (not np.isnan(naive_mae) and naive_mae > 0) else np.nan
    if len(y_true) > 2 and len(a_dir) > 0:
        ret = np.diff(y_true) / y_true[:-1]
        sigma = np.std(ret)
        lm = np.abs(ret) > sigma
        hit_large = np.mean((a_dir[lm] > 0) == (p_dir[lm] > 0)) * 100 if lm.sum() > 0 else np.nan
    else:
        hit_large = np.nan
    return {
        "mape": round(mape, 4), "mae": round(mae, 4), "rmse": round(rmse, 4),
        "r2": round(r2, 4), "da": round(da, 2),
        "mase": round(mase, 4) if not np.isnan(mase) else np.nan,
        "hit_large": round(hit_large, 2) if not np.isnan(hit_large) else np.nan,
    }


# --------------------------------------------------------------------------
# Single-fold runner
# --------------------------------------------------------------------------

def run_fold(model_name, params, target_ts, cov_ts, window, horizon, train_end, test_start, test_end):
    """Fit on target_ts[:train_end] only, evaluate on [test_start:test_end)
    (test_start = train_end + embargo). Returns a metrics dict plus fold date
    range and regime flags."""
    assert_embargo_gap(target_ts, train_end, test_start)

    train_scaled, eval_scaled, scaler = fit_target_transform(target_ts, train_end, test_end)
    eval_cov_scaled = fit_covariate_transform(cov_ts, train_end, test_end) if cov_ts is not None else None

    # past_covariates passed to .fit() only needs to *cover* train_scaled's time
    # range for lag lookups -- Darts does not use covariate values beyond what
    # train_scaled's own index requires, so passing the full (train-fit) eval_cov_scaled
    # here is the same pattern already used in 02a/02b, not a leak.
    model = build_model(model_name, params, window, horizon, has_covariates=cov_ts is not None)
    model.fit(train_scaled, past_covariates=eval_cov_scaled)

    test_start_time = target_ts[test_start].start_time()
    forecast_list = model.historical_forecasts(
        series=eval_scaled,
        past_covariates=eval_cov_scaled,
        start=test_start_time,
        forecast_horizon=horizon,
        stride=horizon,
        retrain=False,
        last_points_only=False,
        verbose=False,
    )
    if isinstance(forecast_list, TimeSeries):
        forecast_list = [forecast_list]

    metrics = inverse_and_metrics(forecast_list, target_ts[:test_end], scaler, train_end)
    test_end_time = target_ts[test_end - 1].end_time()
    metrics["train_end_date"] = str(target_ts[train_end - 1].end_time().date())
    metrics["test_start_date"] = str(test_start_time.date())
    metrics["test_end_date"] = str(test_end_time.date())
    metrics["regime_flags"] = ",".join(flag_regime_overlap(test_start_time, test_end_time)) or None

    del model
    return metrics


if __name__ == "__main__":
    # Quick self-check of the pure-index fold logic (no darts/data needed).
    folds = expanding_window_folds(2408, n_folds=5, embargo=20)
    assert len(folds) >= 1
    for f in folds:
        assert f["test_start"] >= f["train_end"] + 20
        assert f["test_start"] < f["test_end"] <= 2408
    print(f"expanding_window_folds self-check OK: {folds}")

    overlap = flag_regime_overlap(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-01"))
    assert overlap == ["covid_crash"]
    no_overlap = flag_regime_overlap(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-01"))
    assert no_overlap == []
    partial = flag_regime_overlap(pd.Timestamp("2020-06-15"), pd.Timestamp("2020-07-15"))
    assert partial == ["covid_crash"]
    print("flag_regime_overlap self-check OK")

    assert len(SCENARIO_COVARIATES) == 21
    print(f"SCENARIO_COVARIATES self-check OK: {len(SCENARIO_COVARIATES)} scenarios")
