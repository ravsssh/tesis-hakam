# tesis-hakam

Multi-horizon Jakarta Composite Index (JCI/IHSG) forecasting with four tree-based ensemble models — Random Forest, Extra Trees, XGBoost, and LightGBM — tuned via Optuna Bayesian optimization and interpreted with SHAP.

---

## Overview

This repo holds the modeling pipeline for a thesis/paper on predicting JCI daily closing price using macroeconomic, commodity, and regional-index covariates. The approach screens 15 candidate covariates individually, builds covariate groups from the screening survivors, and evaluates all four models across two lookback windows (20, 120 business days) and three forecast horizons (1, 5, 20 days). Hyperparameters are tuned with Optuna/TPE rather than grid search, and the best configuration is interpreted with SHAP to attribute predictions back to individual covariates.

The repo evolved from an earlier single-model, dual-frequency design (`modelling/model-1-level/`, `modelling/model-2/`) through a multi-model, single-80/20-split framework, to the current **walk-forward CV** design in `modelling/00_*.ipynb`, `modelling/cv_lib.py`, and `modelling/02c_*.ipynb` through `modelling/04_*.ipynb`. Paper review flagged two methodological issues in the prior design — look-ahead bias in the macro-variable merge, and a single fixed train/test split — both addressed in the current pipeline (see [Methodology Revision](#methodology-revision) below). The `model-1-*` / `model-2*` / `model2-klasifikasi/` folders are early exploratory iterations kept for reference; `modelling/legacy_pre_revision/` holds the notebooks superseded by the walk-forward CV revision.

---

## Repository Structure

```
tesis-hakam/
├── modelling/
│   ├── 00_data_preprocessing.ipynb      # Load, merge (with publication-lag shift), ADF test, transform, scale → df_merged
│   ├── cv_lib.py                        # Shared walk-forward CV, transform, model-factory, metrics helpers
│   ├── 02c_model_selection_cv.ipynb     # Stage A: default-params × walk-forward CV over full grid → winning config
│   ├── 02d_nested_tuning_cv.ipynb       # Stage B: nested-CV Optuna tuning scoped to the winning config only
│   ├── 02e_visualization_cv.ipynb       # Stage A/B heatmaps, window/DA comparison, default-vs-tuned per fold
│   ├── 03_shap_analysis.ipynb           # SHAP on the winning config's final out-of-sample fold
│   ├── 04_visualization.ipynb           # Winning config's predictions per outer CV fold, publication figure
│   ├── legacy_pre_revision/             # Superseded: 01/02/02a/02b/06 (single 80/20 split, cross-scenario tuning objective)
│   ├── NOTEBOOK_REVISION_SPEC.md        # Spec for the 4-model / multi-horizon revision
│   ├── revised_experiment_framework.md  # Phase 0–3 experiment design, search spaces
│   ├── ringkasan_notebook.md            # Write-up of the earlier RF-only pipeline
│   ├── stage_a_fold_results.csv         # Stage A: per Model×Covariates×Window×Horizon×Fold results
│   ├── stage_a_summary.csv              # Stage A: mean/std per combo
│   ├── nested_cv_outer_fold_results.csv # Stage B: per-fold default vs tuned MAPE/DA/MASE, regime flags
│   ├── nested_cv_default_vs_tuned_summary.csv # Stage B: aggregated improvement %
│   ├── descriptive_stats_transformed.csv# Descriptive stats on transformed variables
│   ├── saved_models/                    # joblib-cached merged data, winning/final config, Optuna results
│   ├── model-1-level/, model-2/, ...    # earlier single-model exploratory notebooks
│   └── plot_*.png, *_shap_*.png         # generated figures (predictions, SHAP, heatmaps)
├── dataset/
│   ├── ihsg.csv                         # Target: JCI daily close
│   ├── bi_interest_rate.csv, cpi.csv, m2.csv, npl_ratio.csv, gdp.csv  # domestic macro
│   ├── usd_idr.csv, ustressury10y.csv   # USD/IDR, US 10Y yield
│   ├── Coal.csv, Copper.csv, Nickel.csv, Silver.csv, Tin.csv, Gold.csv, wti.csv  # commodities
│   ├── STI.csv                          # Straits Times Index (regional)
│   └── publikasi bi/                    # raw Bank Indonesia monthly bulletin tables (.xls)
├── draft/                               # thesis chapters, paper outline, experiment notes
├── Paper/                               # submitted paper manuscript
├── reference/                           # literature PDFs
└── setup_env.sh                         # conda environment bootstrap
```

---

## Dataset

| Attribute | Detail |
|---|---|
| Target | IHSG (JCI) daily closing price → log-difference |
| Period | 2 Jan 2015 – 31 Jan 2025 (~2,443 business days) |
| Split | 80% train (Jan 2015 – ~Jan 2023) / 20% test (~Jan 2023 – Jan 2025), temporal, no shuffling |
| Windows | 20 business days (~1 month), 120 business days (~6 months) |
| Horizons | H1 (next-day), H5 (1 week), H20 (1 month) |
| Covariates | 15 total — 7 macro, 7 commodity, 1 regional (see below) |

| Variable | Category | Frequency | Transformation |
|---|---|---|---|
| BI_Rate | Macro | Monthly → ffill daily | First difference |
| CPI | Macro | Monthly → ffill daily | First difference |
| M2 | Macro | Monthly → ffill daily | Log-difference |
| NPL_Ratio | Macro | Monthly → ffill daily | First difference |
| GDP | Macro | Quarterly → ffill daily | Log-difference |
| USDIDR | Macro | Daily | Log-difference |
| US_Treasury_10Y | Macro (global) | Daily | First difference |
| Coal, Copper, Nickel, Silver, Tin, Gold | Commodity | Daily | Log-difference |
| WTI | Commodity (energy) | Daily | Log-difference |
| STI | Regional | Daily | Log-difference |

All 15 variables are non-stationary at level (ADF p > 0.05) and stationary after the transformation above (ADF p < 0.05). Monthly/quarterly series are merged to daily frequency via `merge_asof` (backward fill). `MinMaxScaler` is fit on the training split only and applied to the full series to avoid leakage.

---

## Models

| Model | Library | Wrapper |
|---|---|---|
| Random Forest | scikit-learn | Darts `RandomForestModel` |
| Extra Trees | scikit-learn | Darts `SKLearnModel(model=ExtraTreesRegressor(...))` — no native Darts class |
| XGBoost | xgboost | Darts `XGBModel` |
| LightGBM | lightgbm | Darts `LightGBMModel` |

Each model is wrapped through [Darts](https://github.com/unit8co/darts) so target and covariate lags share one fit/predict interface. Hyperparameters are tuned with Optuna (TPE sampler), but — per the revised design (see [Methodology Revision](#methodology-revision)) — tuning is now scoped to the single winning (model, window, horizon, covariate-group) configuration selected by Stage A, not run separately across all 4 models. For the current winner (ExtraTrees), nested-CV tuning improved mean MAPE by only 0.07% over default hyperparameters (0.636% → 0.635%) — consistent with the pre-revision finding that bagging models (RF, ET) are largely insensitive to tuning, versus boosting models (XGBoost, LightGBM) which benefited more under the old per-model tuning pass.

---

## Experiment Design

1. **Stage A — Model/config selection (default hyperparameters)**: all 4 models × 2 windows × 3 horizons × 21 covariate scenarios (1 baseline + 15 single covariates + 5 groups) evaluated with untuned default hyperparameters, using 5-fold expanding-window walk-forward CV with an embargo (`02c_model_selection_cv.ipynb`). Apples-to-apples across models since nothing is tuned yet — selects the single best (model, window, horizon, covariate-group) configuration by mean CV MAPE.
2. **Stage B — Scoped Bayesian tuning**: Optuna/TPE runs *only* on the Stage A winner, with a nested-CV objective — an inner 3-fold walk-forward search (bounded strictly to each outer fold's own training range) picks hyperparameters, evaluated once out-of-sample per outer fold (`02d_nested_tuning_cv.ipynb`). This replaces the old design's single hyperparameter set chosen by averaging MAPE across all 6 window×horizon combinations — the tuning objective is now scoped to one already-selected scenario, not averaged across unrelated ones. A final production tuning pass produces the hyperparameters used downstream.
3. **Interpretation**: SHAP (`TreeExplainer`) on the winning configuration's final out-of-sample CV fold, plus Stage A/B visualization (`03_shap_analysis.ipynb`, `04_visualization.ipynb`, `02e_visualization_cv.ipynb`).

All evaluation uses expanding-window walk-forward CV with an embargo gap between training and evaluation (never a single fixed split, never k-fold random splits) to respect time order and avoid look-ahead bias. See [Methodology Revision](#methodology-revision) for why this replaced the prior single-80/20-split design.

---

## Methodology Revision

Paper review surfaced two issues in the original pipeline (now in `modelling/legacy_pre_revision/`), both addressed in the current design:

1. **Macro-variable look-ahead bias**: `BI_Rate`, `CPI`, `M2`, `NPL_Ratio`, `GDP` were merged via `merge_asof(direction="backward")` keyed on period-*reference* dates (e.g. GDP dated at quarter-start) rather than *publication* dates — giving the model up to ~9 weeks of foresight on GDP/M2 specifically. Fixed in `00_data_preprocessing.ipynb` by shifting each series by its real publication lag before merging, verified against actual BPS/BI release-calendar dates: GDP +124 days, M2 +51 days, NPL_Ratio +51 days (placeholder pending an OJK-calendar check), CPI/BI_Rate +0 days (already released within days of the reference date).
2. **Single 80/20 split + cross-scenario-averaged tuning objective**: the old pipeline picked one hyperparameter set per model by averaging MAPE across all 6 window×horizon combinations, then evaluated every covariate scenario on one fixed train/test split. Replaced by the two-stage Stage A/Stage B design above — default-hyperparameter walk-forward CV selects the configuration, then Optuna tunes only that configuration with a nested-CV objective.

---

## Results

Under the revised walk-forward CV pipeline (`02c_model_selection_cv.ipynb` + `02d_nested_tuning_cv.ipynb`), the winning configuration — selected by mean CV MAPE across the full 4-model × 21-covariate-scenario × 2-window × 3-horizon grid, default hyperparameters, 5-fold expanding-window CV with embargo — is:

| Model | Covariate group | Window | Horizon | Mean CV MAPE (%, default params) | Mean CV MAPE (%, tuned) |
|---|---|---|---|---|---|
| ExtraTrees | Screening1 (Silver, WTI, Gold, STI, Coal, Tin, NPL_Ratio) | 120 | H1 (next-day) | 0.6360 (± 0.150) | 0.6353 (± 0.149) |

This replaces the pre-revision pipeline's LightGBM/Screening1/W120/H1 pick — the change reflects the corrected methodology (walk-forward CV + untuned model-selection stage instead of a single 80/20 split with pre-tuned hyperparameters) rather than a change in the underlying signal; Screening1 remains the best-performing covariate group in both versions.

Per-outer-fold results for the winning configuration (`nested_cv_outer_fold_results.csv`), default vs. nested-CV-tuned hyperparameters:

| Fold | Test period | Default MAPE (%) | Tuned MAPE (%) | Default DA (%) | Tuned DA (%) | Regime flag |
|---|---|---|---|---|---|---|
| 0 | 2019-02-15 – 2020-08-12 | 0.8084 | 0.8077 | 49.48 | 53.09 | COVID crash |
| 1 | 2020-03-30 – 2021-09-23 | 0.7826 | 0.7788 | 50.52 | 54.90 | COVID crash |
| 2 | 2021-05-11 – 2022-11-04 | 0.5697 | 0.5711 | 52.58 | 51.80 | 2022 rate-hike cycle |
| 3 | 2022-06-22 – 2023-12-18 | 0.4734 | 0.4749 | 55.67 | 51.80 | 2022 rate-hike cycle |
| 4 | 2023-08-03 – 2025-01-28 | 0.5459 | 0.5441 | 53.87 | 52.58 | — |

4 of 5 folds overlap a flagged regime window (COVID crash or the 2022 rate-hike cycle) — MAPE is markedly higher in the two COVID-crash folds (~0.78–0.81%) than the post-2021 folds (~0.47–0.57%), so the overall mean is pulled up by that period rather than being uniform across time. Tuning's effect is inconsistent fold-to-fold (helps in folds 0, 1, 4; slightly hurts in folds 2, 3) and averages out to a negligible net improvement — see [Models](#models) for why this matches the pre-revision finding for bagging models.

Mean CV MAPE per model across the full Stage A grid (`stage_a_summary.csv`), by window, default hyperparameters:

| Model | W20 mean MAPE (%) | W120 mean MAPE (%) |
|---|---|---|
| ExtraTrees | 1.2314 | 1.2328 |
| RandomForest | 1.2354 | 1.2388 |
| LightGBM | 1.3741 | 1.3582 |
| XGBoost | 1.3685 | 1.3682 |

Bagging models (RF, ET) outperform boosting models (XGBoost, LightGBM) at default hyperparameters here — the opposite ranking from the pre-revision, Optuna-tuned comparison, since boosting models need tuning to be competitive (see [Models](#models)) and Stage A deliberately evaluates everyone untuned. Directional accuracy stays close to 50–51% across all models, averaged across the full grid (`da_mean` in `stage_a_summary.csv`), consistent with JCI behaving close to a random walk day-to-day at this level of aggregation.

---

## Explainability

SHAP (`TreeExplainer`) is computed on the **final outer CV fold's out-of-sample test set** (2023-08-03 – 2025-01-28) for the winning configuration (ExtraTrees, Screening1 group, W120, H1) using the production-tuned hyperparameters, not on training data, so the attributions match what the model actually used to produce its reported predictions.

| Rank | Variable | Mean \|SHAP\| | Category |
|---|---|---|---|
| 1 | NPL_Ratio | 0.000161 | Macro/Rate |
| 2 | Tin | 0.000070 | Commodity |
| 3 | Silver | 0.000064 | Commodity |
| 4 | STI | 0.000056 | Regional |
| 5 | IHSG (own lag) | 0.000037 | Autoregressive |
| 6 | Gold | 0.000036 | Commodity |
| 7 | WTI | 0.000025 | Commodity |
| 8 | Coal | 0.000023 | Commodity |

Same qualitative story as the pre-revision result: NPL_Ratio dominates, commodity prices (tin, silver, gold, WTI, coal) and the regional STI index carry most of the remaining attribution, and IHSG's own lag contributes relatively little — external covariates carry information not already present in the index's own history. This held up across the methodology revision (different model — ExtraTrees vs. LightGBM — different test window, same top variable and same broad ranking), which is some evidence the finding isn't an artifact of the pre-revision leakage/split issues. Note SHAP values here are computed on log-returns, so they reflect each variable's contribution to the predicted daily return, not to the price level.

---

## Setup

```bash
git clone https://github.com/ravsssh/tesis-hakam.git
cd tesis-hakam
bash setup_env.sh
conda activate tesis_hakam
```

`setup_env.sh` creates a `tesis_hakam` conda environment (Python 3.11) with numpy, pandas, scikit-learn, statsmodels, Darts (without torch), xgboost, lightgbm, optuna, shap, and Jupyter, and registers the kernel as "Python (tesis_hakam)".

Run notebooks in order:

| Step | Notebook | Output |
|---|---|---|
| 1 | `00_data_preprocessing.ipynb` | `df_merged` joblib, publication-lag-corrected (used by all downstream notebooks) |
| 2 | `02c_model_selection_cv.ipynb` | `stage_a_fold_results.csv`, `stage_a_summary.csv`, `saved_models/winning_config.joblib` (~30–45 min, 2,520 fits — includes a smoke test first) |
| 3 | `02d_nested_tuning_cv.ipynb` | `nested_cv_outer_fold_results.csv`, `nested_cv_default_vs_tuned_summary.csv`, `saved_models/optuna_tuning_results.joblib`, `saved_models/final_config.joblib` |
| 4 | `03_shap_analysis.ipynb` | `shap_variable_importance_*.csv`, SHAP plots (on the final out-of-sample CV fold) |
| 5 | `04_visualization.ipynb` | Winning-config predictions per outer CV fold, publication figure |
| 6 | `02e_visualization_cv.ipynb` | Stage A/B heatmaps, window/DA comparison, default-vs-tuned-per-fold plot |

`modelling/legacy_pre_revision/` holds the superseded `01`/`02`/`02a`/`02b`/`06` notebooks (single 80/20 split, cross-scenario-averaged tuning objective) — kept for the paper's methodology/limitations discussion, not part of the current run order.

---

## Dependencies

| Category | Packages |
|---|---|
| ML | scikit-learn, xgboost, lightgbm |
| Time series | darts |
| Optimization | optuna |
| Explainability | shap |
| Data | pandas, numpy==1.26.4, scipy, statsmodels |
| Notebooks | jupyter, ipykernel, nbconvert, nbformat |

---

## Citation

The accompanying manuscript is in `Paper/Paper Hakam.docx`. Citation details (venue, year, DOI) will be added once published.
