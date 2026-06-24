# tesis-hakam

Multi-horizon Jakarta Composite Index (JCI/IHSG) forecasting with four tree-based ensemble models — Random Forest, Extra Trees, XGBoost, and LightGBM — tuned via Optuna Bayesian optimization and interpreted with SHAP.

---

## Overview

This repo holds the modeling pipeline for a thesis/paper on predicting JCI daily closing price using macroeconomic, commodity, and regional-index covariates. The approach screens 15 candidate covariates individually, builds covariate groups from the screening survivors, and evaluates all four models across two lookback windows (20, 120 business days) and three forecast horizons (1, 5, 20 days). Hyperparameters are tuned with Optuna/TPE rather than grid search, and the best configuration is interpreted with SHAP to attribute predictions back to individual covariates.

The repo evolved from an earlier single-model, dual-frequency design (`modelling/model-1-level/`, `modelling/model-2/`) into the current multi-model, multi-horizon framework in `modelling/00_*.ipynb` through `modelling/04_*.ipynb`. The notebooks below `modelling/02b_*` and the numbered series are the ones that feed the current paper; the `model-1-*` / `model-2*` / `model2-klasifikasi/` folders are earlier exploratory iterations kept for reference.

---

## Repository Structure

```
tesis-hakam/
├── modelling/
│   ├── 00_data_preprocessing.ipynb      # Load, merge, ADF test, transform, scale → df_merged
│   ├── 01_hyperparameter_tuning.ipynb   # Optuna/TPE tuning for 4 models (30 trials each)
│   ├── 02_phase1_screening.ipynb        # Single-covariate screening, 5-fold expanding CV
│   ├── 02a_phase1_screening_v2.ipynb    # Screening + groups, 80/20 split (384 experiments)
│   ├── 02b_default_params.ipynb         # Default vs Optuna hyperparameter comparison
│   ├── 03_shap_analysis.ipynb           # SHAP on the best configuration
│   ├── 04_visualization.ipynb           # Top-3 model prediction plots, publication figure
│   ├── 06_visualization.ipynb           # Phase 1/2 multi-model comparison (heatmaps, DA)
│   ├── NOTEBOOK_REVISION_SPEC.md        # Spec for the 4-model / multi-horizon revision
│   ├── revised_experiment_framework.md  # Phase 0–3 experiment design, search spaces
│   ├── ringkasan_notebook.md            # Write-up of the earlier RF-only pipeline
│   ├── phase1a_screening_results.csv    # Single-covariate screening results
│   ├── phase1a_group_results.csv        # Covariate-group results (Phase 1)
│   ├── phase2_group_results.csv         # Covariate-group results (Phase 2)
│   ├── optuna_tuning_summary.csv        # Best hyperparameters per model
│   ├── descriptive_stats_transformed.csv# Descriptive stats on transformed variables
│   ├── saved_models/                    # joblib-cached merged data, Optuna studies, configs
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

Each model is wrapped through [Darts](https://github.com/unit8co/darts) so target and covariate lags share one fit/predict interface. Hyperparameters are tuned with Optuna (TPE sampler, `MedianPruner`, 30 trials/model, objective = mean MAPE across window × horizon combinations). Tuning improved boosting models the most (XGBoost +9.9%, LightGBM +13.3% MAPE reduction vs. default) while bagging models (RF, ET) were largely insensitive to tuning.

---

## Experiment Design

1. **Phase 0 — Hyperparameter tuning**: Optuna/TPE per model, baseline covariates, W120, 30 trials each (`01_hyperparameter_tuning.ipynb`).
2. **Phase 1 — Single-covariate screening**: each of the 15 covariates evaluated alone vs. a no-covariate baseline, across 4 models × 2 windows × 3 horizons (`02a_phase1_screening_v2.ipynb`). A covariate passes if it improves MAPE ≥0.3% or DA ≥1.0pp relative to baseline.
3. **Phase 2 — Group configuration**: covariates that pass screening are combined into 6 groups (Baseline, Screening1, Screening2, All_Commodity_STI, All_Macro_no_UST, All_Covariates) and re-evaluated across the same model × window × horizon grid.
4. **Phase 3 — Interpretation**: SHAP (`TreeExplainer`) on the best configuration's test-set predictions, plus cross-model and cross-horizon comparison (`03_shap_analysis.ipynb`, `06_visualization.ipynb`).

All evaluation uses a temporal 80/20 split (Phase 1/2 in `02a`/`02b`) or 5-fold expanding-window CV (`02`), never k-fold random splits, to respect time order.

---

## Results

Best configuration per horizon (lowest MAPE, Optuna-tuned hyperparameters, group experiments):

| Model | Horizon | Covariate group | Window | MAPE (%) | DA (%) |
|---|---|---|---|---|---|
| LightGBM | H1 (next-day) | Screening1 | 120 | 0.5007 | 55.89 |
| XGBoost | H5 (1 week) | Screening2 | 20 | 0.8133 | 47.71 |
| XGBoost | H20 (1 month) | Screening1 | 120 | 1.3238 | 48.36 |

Average MAPE by group across all models/scenarios — Screening1 (Silver, WTI, Gold, STI, Coal, Tin, NPL_Ratio) is the best-performing group, and using all 15 covariates does not beat it:

| Group | Avg MAPE (%) |
|---|---|
| Screening1 | 0.9047 |
| All_Commodity_STI | 0.9071 |
| Screening2 | 0.9080 |
| Baseline | 0.9169 |
| All_Macro_no_UST | 0.9185 |
| All_Covariates | 0.9201 |

Optuna tuning vs. default hyperparameters, averaged over all non-baseline scenarios:

| Model | Default MAPE (%) | Optuna MAPE (%) | Improvement |
|---|---|---|---|
| LightGBM | 1.0438 | 0.9047 | +13.33% |
| XGBoost | 1.0075 | 0.9080 | +9.87% |
| Extra Trees | 0.9145 | 0.9153 | -0.09% |
| Random Forest | 0.9169 | 0.9223 | -0.59% |

Directional accuracy stays close to 50% across all horizons (52–56% at best), indicating the models are far better at estimating return magnitude than direction — consistent with JCI behaving close to a random walk day-to-day. MAPE rises with horizon (≈0.50% at H1, ≈0.84% at H5, ≈1.39% at H20), as expected for multi-step forecasting.

---

## Explainability

SHAP (`TreeExplainer`) is computed on the out-of-sample test set for the best configuration (LightGBM, Screening1 group, W120, H1), not on training data, so the attributions match what the model actually used to produce its reported predictions.

| Rank | Variable | Mean \|SHAP\| | Category |
|---|---|---|---|
| 1 | NPL_Ratio | 0.000142 | Macro |
| 2 | Silver | 0.000125 | Commodity |
| 3 | STI | 0.000064 | Regional |
| 4 | Gold | 0.000064 | Commodity |
| 5 | WTI | 0.000041 | Commodity |
| 6 | Coal | 0.000040 | Commodity |
| 7 | Tin | 0.000029 | Commodity |
| 8 | IHSG (own lag) | 0.000024 | Autoregressive |

Commodity prices (silver, gold, WTI) dominate the attribution, consistent with Indonesia's exposure as a commodity exporter; IHSG's own lag contributes the least, suggesting the external covariates carry information not already present in the index's own history. Note SHAP values here are computed on log-returns, so they reflect each variable's contribution to the predicted daily return, not to the price level.

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
| 1 | `00_data_preprocessing.ipynb` | `df_merged` joblib (used by all downstream notebooks) |
| 2 | `01_hyperparameter_tuning.ipynb` | `optuna_tuning_results.joblib`, `optuna_tuning_summary.csv` |
| 3 | `02a_phase1_screening_v2.ipynb` | `phase1a_screening_results.csv`, `phase1a_group_results.csv` |
| 4 | `02b_default_params.ipynb` | Default vs. Optuna comparison tables |
| 5 | `03_shap_analysis.ipynb` | `shap_variable_importance_*.csv`, SHAP plots |
| 6 | `04_visualization.ipynb` | Top-3 model prediction plots, publication figure |
| 7 | `06_visualization.ipynb` | Multi-model comparison heatmaps |

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
