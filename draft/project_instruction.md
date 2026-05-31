# THESIS PROJECT CONTEXT

## Research Overview
- **Title:** Multi-Horizon Jakarta Composite Index Forecasting Using Tree-Based Ensemble and Bayesian Hyperparameter Optimization
- **Student:** Hakam
- **Focus:** Multi-horizon JCI prediction using 4 tree-based ensemble models, 6 covariate group configurations, and Bayesian hyperparameter optimization (Optuna) vs default parameters

---

## Model Design

**Framework:**
- Target: IHSG (Jakarta Composite Index) daily closing price (log-differenced)
- Period: 2 January 2015 – 31 January 2025 (2.443 business days)
- Train/Test split: 80% train (~1.954 days), 20% test (~489 days, ~Jan 2023 – Jan 2025)
- Evaluation: 80/20 temporal split (tidak ada CV — single split)
- Lag features: Darts library, scikit-learn/xgboost/lightgbm backend

**4 Algorithms:**
| Model | Type | Library |
|---|---|---|
| Random Forest | Bagging | scikit-learn (via Darts) |
| Extra Trees | Bagging (random splits) | scikit-learn (via Darts SKLearnModel) |
| XGBoost | Gradient Boosting | xgboost (via Darts) |
| LightGBM | Light Gradient Boosting | lightgbm (via Darts) |

**2 Window Scenarios:**
- W20: 20 business days lookback (≈ 1 month)
- W120: 120 business days lookback (≈ 6 months)

**3 Forecast Horizons:**
- H1: 1 business day ahead (next-day)
- H5: 5 business days ahead (≈ 1 week)
- H20: 20 business days ahead (≈ 1 month)

**6 Covariate Group Configurations:**
| Group | Variables | N |
|---|---|---|
| Baseline | — (IHSG lags only) | 0 |
| Screening1 | Silver, WTI, Gold, STI, Coal, Tin, NPL_Ratio | 7 |
| Screening2 | Screening1 + CPI, USDIDR, Nickel | 10 |
| All_Commodity_STI | Coal, Copper, Nickel, Silver, Tin, Gold, WTI, STI | 8 |
| All_Macro_no_UST | BI_Rate, CPI, M2, NPL_Ratio, USDIDR, GDP | 6 |
| All_Covariates | All 15 variables | 15 |

**Total Group Experiments: 6 × 4 × 2 × 3 = 144**

---

## Variables (15 Covariates, 3 Categories)

| Variabel | Kategori | Frekuensi | Transformasi |
|---|---|---|---|
| BI_Rate | Macro | Bulanan | first-diff |
| CPI | Macro | Bulanan | first-diff |
| M2 | Macro | Bulanan | log-diff |
| NPL_Ratio | Macro | Bulanan | first-diff |
| GDP | Macro | Kuartalan | log-diff |
| USDIDR | Macro | Harian | log-diff |
| Coal | Commodity | Harian | log-diff |
| Copper | Commodity | Harian | log-diff |
| Nickel | Commodity | Harian | log-diff |
| Silver | Commodity | Harian | log-diff |
| Tin | Commodity | Harian | log-diff |
| Gold | Commodity | Harian | log-diff |
| WTI | Commodity | Harian | log-diff |
| STI | Regional | Harian | log-diff |

---

## Preprocessing Pipeline
1. ADF test → semua 16 variabel non-stationary at level
2. Transformasi: log-diff untuk level vars, first-diff untuk rate vars
3. MinMaxScaler (Darts Scaler) — fit on training set only
4. Temporal split: 80% train, 20% test, no shuffling

---

## Hyperparameter Optimization (Notebook 01)

**Method:** Optuna dengan TPE (Tree-structured Parzen Estimator) sampler  
**Trials:** 30 per model (120 total)  
**Objective:** Minimize avg MAPE across all window × horizon combinations  
**Seed:** 42

**Best Parameters (Optuna):**
| Parameter | RandomForest | ExtraTrees | XGBoost | LightGBM |
|---|---|---|---|---|
| n_estimators | 800 | 200 | 600 | 600 |
| max_depth | 3 | 5 | 11 | 10 |
| max_features | 0.7 | sqrt | — | — |
| max_samples | 0.8 | — | — | — |
| min_samples_split | 5 | 9 | — | — |
| min_samples_leaf | 6 | 3 | — | — |
| learning_rate | — | — | 0.0295 | 0.0117 |
| subsample | — | — | 0.8256 | 0.5853 |
| colsample_bytree | — | — | 0.3920 | 0.3456 |
| reg_alpha | — | — | 4.7230 | 3.4671 |
| reg_lambda | — | — | 0.0035 | 4.9056 |
| min_child_weight | — | — | 8 | — |
| num_leaves | — | — | — | 83 |
| min_child_samples | — | — | — | 42 |

---

## Key Results

### Hyperparameter Optimization (Notebook 02b)
| Model | Default MAPE | Optuna MAPE | Improvement |
|---|---|---|---|
| RandomForest | 0.9169% | 0.9223% | -0.59% |
| ExtraTrees | 0.9145% | 0.9153% | -0.09% |
| XGBoost | 1.0075% | 0.9080% | **+9.87%** |
| LightGBM | 1.0438% | 0.9047% | **+13.33%** |
| **Overall** | **0.9739%** | **0.9117%** | **+6.38%** |

### Group Experiments — Best MAPE per Group (Optuna)
| Group | Avg MAPE | vs Baseline |
|---|---|---|
| Screening1 | **0.9047%** | terbaik |
| All_Commodity_STI | 0.9071% | |
| Screening2 | 0.9080% | |
| Baseline | 0.9169% | referensi |
| All_Macro_no_UST | 0.9185% | |
| All_Covariates | 0.9201% | |

### Best MAPE per Model (Optuna, group experiments)
| Model | Avg MAPE |
|---|---|
| LightGBM | **0.9047%** |
| XGBoost | 0.9080% |
| ExtraTrees | 0.9153% |
| RandomForest | 0.9223% |

### Multi-Horizon Results
| Horizon | Avg MAPE | Avg DA | Avg MASE |
|---|---|---|---|
| H1 (next-day) | **0.5056%** | 52.66% | 0.9714 |
| H5 (1 week) | 0.8445% | 48.12% | 1.6225 |
| H20 (1 month) | 1.3876% | 48.80% | 2.6774 |

### Top 5 Konfigurasi Terbaik (Overall)
| Rank | Model | Group | W | H | MAPE | DA |
|---|---|---|---|---|---|---|
| #1 | LightGBM | Screening1 | 120 | 1 | **0.5007%** | 55.89% |
| #2 | XGBoost | Screening1 | 120 | 1 | 0.5012% | 51.90% |
| #3 | XGBoost | Screening1 | 20 | 1 | 0.5017% | 54.37% |
| #4 | LightGBM | Screening1 | 20 | 1 | 0.5026% | 54.75% |
| #5 | LightGBM | All_Macro_no_UST | 20 | 1 | 0.5026% | 53.80% |

### Covariate Screening Summary (Notebook 02a)
Top covariates by pass rate (24 combos = 4 model × 2 window × 3 horizon):
| Covariate | Pass Rate | Avg MAPE Impr | Tier |
|---|---|---|---|
| Silver | 66.7% | +0.637% | ✅ Kuat |
| WTI | 54.2% | +0.190% | ✅ Kuat |
| Gold | 45.8% | +0.402% | 🟡 Moderat |
| STI | 37.5% | +0.317% | 🟡 Moderat |
| Coal, NPL_Ratio, Tin | 25–29% | +0.1% | 🟡 Moderat |

---

## Analysis Completed
- ✅ Data preprocessing + ADF stationarity testing (NB 00)
- ✅ Bayesian hyperparameter optimization — Optuna TPE, 30 trials × 4 models (NB 01)
- ✅ Single covariate screening — 384 experiments (NB 02a)
- ✅ Group covariate experiments — 144 experiments, Optuna params (NB 02a)
- ✅ Default vs Optuna comparison — 144 experiments, default params (NB 02b)
- ✅ SHAP analysis — best config from screening (NB 03, partial)
- ⏳ Visualization — actual vs predicted line graphs (NB 04, in progress)

---

## Notebook Pipeline
```
NB 00 — Data Preprocessing
  └─ Output: df_merged.joblib, adf_*.csv

NB 01 — Hyperparameter Tuning (Optuna)
  └─ Output: optuna_tuning_results.joblib

NB 02a — Single Covariate Screening + Group Experiments (Optuna)
  └─ Output: phase1a_screening_results.csv, phase1a_screening_analysis.csv
             phase1a_group_results.csv, top3_group_config.joblib

NB 02b — Group Experiments (Default params — for comparison)
  └─ Output: phase1b_default_results.csv

NB 03 — SHAP Analysis
  └─ Output: shap_variable_importance_*.csv, plot_shap_*.png

NB 04 — Visualization (Actual vs Predicted)
  └─ Output: plot_top3_predictions_*.png
```

---

## Writing Guidelines
- Language: Academic English (for paper), Academic Indonesian (for thesis)
- Use Scopus-indexed references, prefer 2020–2025
- Avoid generic claims without citations
- **Paper title focus:** multi-horizon, tree-based ensemble, Bayesian optimization — NOT screening detail, NOT SHAP
- **Thesis:** follows Bab 1–5 structure (see tesis-existing.md)
- Key claim to support: Optuna improves MAPE by +6.38% overall, especially boosting models

## Research Questions (Paper)
1. How do different covariate group configurations affect multi-horizon JCI prediction accuracy across four tree-based ensemble models?
2. To what extent does Bayesian hyperparameter optimization (Optuna/TPE) improve prediction performance compared to default parameters across different models and forecast horizons?
3. Which model-covariate-window-horizon combination achieves the best predictive accuracy for JCI forecasting?

## Key Narrative Points
- Commodity variables (Silver, WTI, Gold) outperform macro variables for JCI prediction → Indonesia as commodity exporter
- Screening1 (7 vars) beats All_Covariates (15 vars) → parsimony principle, fewer covariates can be better
- LightGBM and XGBoost benefit most from Optuna (+13.3%, +9.9%) → boosting models more sensitive to hyperparameters
- Directional accuracy ~50% across all horizons → JCI direction difficult to predict, consistent with semi-strong EMH
- MAPE degrades significantly with horizon: 0.50% (H1) → 0.84% (H5) → 1.39% (H20)
- Optimization most impactful at long horizons (H20 improvement +9.21%)
