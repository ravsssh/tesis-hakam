# REVISED EXPERIMENT FRAMEWORK
## Prediksi IHSG Menggunakan Ensemble Learning Berbasis Pohon Keputusan dengan Variabel Makroekonomi dan Komoditas

**Last updated:** March 27, 2026  
**Status:** Post-bimbingan revision — incorporating advisor feedback

---
## 1. MODEL ARCHITECTURE

### 1.1 Empat Model Tree-Based Ensemble

| Model | Kategori | Library | Referensi Utama |
|-------|----------|---------|-----------------|
| **Random Forest** | Bagging | Darts → scikit-learn | Breiman (2001) |
| **Extra Trees** | Bagging | Darts → scikit-learn | Geurts et al. (2006) |
| **XGBoost** | Boosting | xgboost / Darts | Chen & Guestrin (2016) |
| **LightGBM** | Boosting | lightgbm / Darts | Ke et al. (2017) |

### 1.2 Justifikasi Pemilihan Model

**Bagging vs Boosting comparison:**
- **Bagging** (RF, Extra Trees): Membangun pohon secara paralel & independen, mengurangi variance melalui averaging. Extra Trees menambahkan randomisasi pada threshold split.
- **Boosting** (XGBoost, LightGBM): Membangun pohon secara sekuensial, setiap pohon memperbaiki error pohon sebelumnya. Mengurangi bias secara iteratif.

**Mengapa keempat model ini:**
Uji coba dan membandingkan bagaimana berbagai machine learning berbasis pohon keputusan bekerja untuk prediksi timeseries ihsg

---

## 2. EXPERIMENT DESIGN

### 2.1 Variables (UPDATED — added 3 new covariates)

**Target:** IHSG daily closing price (log-differenced)

**15 Covariates:**
| # | Variable | Category | Frequency | Transformation |
|---|----------|----------|-----------|----------------|
| 1 | BI_Rate | Macro | Monthly → daily | First difference |
| 2 | CPI | Macro | Monthly → daily | First difference |
| 3 | M2 | Macro | Monthly → daily | Log-difference |
| 4 | NPL_Ratio | Macro | Monthly → daily | First difference |
| 5 | USDIDR | Macro | Daily | Log-difference |
| 6 | GDP | Macro | Quarterly → daily | Log-difference |
| 7 | US_Treasury_10Y | Macro (global) | Daily | First difference |
| 8 | Coal | Commodity | Daily | Log-difference |
| 9 | Copper | Commodity | Daily | Log-difference |
| 10 | Nickel | Commodity | Daily | Log-difference |
| 11 | Silver | Commodity | Daily | Log-difference |
| 12 | Tin | Commodity | Daily | Log-difference |
| 13 | Gold | Commodity | Daily | Log-difference |
| 14 | WTI_Oil | Commodity (energy) | Daily | Log-difference |
| 15 | STI | Regional | Daily | Log-difference |



### 2.2 Window Scenarios

| Scenario | Window (Lookback) | Horizon | Keterangan |
|----------|-------------------|---------|------------|
| W30_H1 | 30 hari | 1 hari | Short-term momentum |
| W120_H1 | 120 hari | 39 hari | Long-term, siklus makro |

### 2.3 Two-Phase Experiment Design

```
PHASE 0: Hyperparameter Tuning (Optuna + TPE)
    → 4 tuning sessions (1 per model)
    → Output: optimal hyperparameters per model

PHASE 1: Single Covariate Screening
    → 16 configs (1 baseline + 15 single covariates)
    → × 2 windows × 4 models = 128 experiments
    → Criterion: MAPE improvement threshold vs baseline
    → Output: list of "significant" covariates per model/window

PHASE 2: Group Configuration (Bottom-Up)
    → Groups built from Phase 1 survivors
    → × 2 windows × 4 models
    → Output: best model × covariate × window combination

PHASE 3: Analysis & Interpretation
    → SHAP analysis on best configuration(s)
    → Cross-model comparison (bagging vs boosting)
    → Covariate contribution analysis
```

---

## 4. PHASE 0: HYPERPARAMETER TUNING WITH OPTUNA

### 4.1 Tuning Strategy

- **Method:** Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Sessions:** 4 (one per model)
- **Tuning configuration:** Baseline (None covariates), W120_H1
- **Objective:** Minimize mean MAPE across 5-fold expanding window CV
- **Trials:** 100 per model (adjustable based on compute)
- **Pruning:** Optuna MedianPruner (early stop unpromising trials)

### 4.2 Search Spaces

**Random Forest:**
```python
{
    'n_estimators':     trial.suggest_int('n_estimators', 100, 1000, step=100),
    'max_depth':        trial.suggest_int('max_depth', 3, 20),
    'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
    'max_samples':      trial.suggest_float('max_samples', 0.5, 0.9, step=0.1),
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
}
```

**Extra Trees:**
```python
{
    'n_estimators':     trial.suggest_int('n_estimators', 100, 1000, step=100),
    'max_depth':        trial.suggest_int('max_depth', 3, 20),
    'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
    # Note: Extra Trees doesn't use max_samples (no bootstrap by default)
}
```

**XGBoost:**
```python
{
    'n_estimators':     trial.suggest_int('n_estimators', 100, 1000, step=100),
    'max_depth':        trial.suggest_int('max_depth', 3, 15),
    'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
    'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
}
```

**LightGBM:**
```python
{
    'n_estimators':     trial.suggest_int('n_estimators', 100, 1000, step=100),
    'max_depth':        trial.suggest_int('max_depth', 3, 15),
    'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'num_leaves':       trial.suggest_int('num_leaves', 15, 127),
    'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
    'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
    'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
}
```

### 4.3 Output Phase 0

Tabel best hyperparameters per model → digunakan fixed untuk semua experiments di Phase 1 & 2.

---

## 5. PHASE 1: SINGLE COVARIATE SCREENING

### 5.1 Experiment Matrix

| Config | Covariates | × Windows | × Models | = Experiments |
|--------|-----------|-----------|----------|---------------|
| Baseline (None) | 0 | 2 | 4 | 8 |
| BI_Rate only | 1 | 2 | 4 | 8 |
| CPI only | 1 | 2 | 4 | 8 |
| M2 only | 1 | 2 | 4 | 8 |
| NPL_Ratio only | 1 | 2 | 4 | 8 |
| USDIDR only | 1 | 2 | 4 | 8 |
| **GDP only** ⭐ | 1 | 2 | 4 | 8 |
| **US_Treasury_10Y only** ⭐ | 1 | 2 | 4 | 8 |
| Coal only | 1 | 2 | 4 | 8 |
| Copper only | 1 | 2 | 4 | 8 |
| Nickel only | 1 | 2 | 4 | 8 |
| Silver only | 1 | 2 | 4 | 8 |
| Tin only | 1 | 2 | 4 | 8 |
| Gold only | 1 | 2 | 4 | 8 |
| **WTI_Oil only** ⭐ | 1 | 2 | 4 | 8 |
| STI only | 1 | 2 | 4 | 8 |
| **TOTAL** | | | | **128** |

### 5.2 Screening Criterion

**MAPE Improvement Threshold:**

```
Relative Improvement (%) = (MAPE_baseline - MAPE_covariate) / MAPE_baseline × 100
```

**Threshold options (to be determined empirically):**
- Conservative: ≥ 0.5% relative improvement
- Moderate: ≥ 0.3% relative improvement  
- Liberal: any improvement (MAPE_covariate < MAPE_baseline)

**Recommendation:** Start with **≥ 0.3% relative improvement** as the threshold. Given the previous experiment range (MAPE 0.6592–0.6661%), this corresponds to roughly ≥0.002 percentage point absolute improvement. Adjust after seeing Phase 1 results.

**Important:** Screening is done **per model per window**. A covariate may be significant for RF but not for XGBoost, or for W120 but not W30.

### 5.3 Output Phase 1

**Screening Results Table (example format):**

| Covariate | RF_W30 | RF_W120 | ET_W30 | ET_W120 | XGB_W30 | XGB_W120 | LGBM_W30 | LGBM_W120 |
|-----------|--------|---------|--------|---------|---------|----------|----------|-----------|
| BI_Rate | ✓/✗ | ✓/✗ | ... | ... | ... | ... | ... | ... |
| CPI | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Decision for Phase 2:** Use covariates that pass screening in **majority** of model-window combinations (≥ 5 out of 8), OR use a **union** approach (any covariate that passes in at least 1 combination).

---

## 6. PHASE 2: GROUP CONFIGURATION (BOTTOM-UP)

### 6.1 Group Building Strategy

Based on Phase 1 results, build groups **bottom-up**:

**Option A — Category-informed bottom-up:**
If significant covariates span multiple categories, form groups by category:
- **Significant_Macro**: all macro covariates that passed screening
- **Significant_Commodity**: all commodity covariates that passed screening
- **Significant_Regional**: STI if it passed screening
- **Significant_All**: union of all significant covariates

**Option B — Incremental addition:**
1. Start with single best covariate
2. Add next best covariate, check if MAPE improves
3. Continue until adding more covariates stops improving (greedy forward selection)

**Recommendation:** Option A is simpler and more interpretable for thesis. Option B is more rigorous but computationally expensive. Start with Option A.

### 6.2 Estimated Experiment Count

Assuming Phase 1 yields ~3-4 group configurations:
- ~4 groups × 2 windows × 4 models = **32 experiments**
- Plus Phase 1 baseline (8) already computed
- **Phase 2 total: ~32 new experiments**

### 6.3 Output Phase 2

- Best model × covariate × window combination
- Ranking table across all configurations
- Statistical comparison between bagging vs boosting performance

---

## 7. PHASE 3: ANALYSIS & INTERPRETATION

### 7.1 SHAP Analysis
- Applied to **best overall configuration** and **best per-model configurations**
- Feature importance aggregated by category (Macro / Commodity / Regional / IHSG lags)
- SHAP summary plots, dependence plots, force plots

### 7.2 Cross-Model Comparison
- Bagging (RF, ET) vs Boosting (XGB, LGBM) aggregate performance
- Model ranking by MAPE, RMSE, R², DA
- Sensitivity analysis: which model benefits most from covariates?

### 7.3 Covariate Contribution Analysis
- Which covariates consistently improve predictions across all models?
- Are there model-specific covariate preferences?
- Economic interpretation of significant covariates

---

## 8. TOTAL EXPERIMENT COUNT

| Phase | Experiments | Description |
|-------|-------------|-------------|
| Phase 0 | 4 tuning sessions × ~100 trials × 5 folds | Hyperparameter optimization |
| Phase 1 | 128 | Single covariate screening (16 configs × 2 windows × 4 models) |
| Phase 2 | ~32 (estimated) | Group configurations from screening |
| Phase 3 | Analysis only | SHAP, comparison, interpretation |
| **Total experiments** | **~136** | **(excluding tuning trials)** |

---

## 9. VALIDATION METHOD (unchanged)

**5-Fold Expanding Window Cross-Validation:**
- Fold 1: Train ~40% data, Test ~15%
- Fold 2–5: Training expands, test size ~15% per fold
- Metrics reported as **mean ± std** across 5 folds

**Metrics:**
- MAPE (primary), RMSE, MAE, R², Directional Accuracy

---

## 10. REVISED RESEARCH QUESTIONS (DRAFT)

1. Bagaimana performa empat algoritma ensemble learning berbasis pohon keputusan (Random Forest, Extra Trees, XGBoost, LightGBM) dalam memprediksi IHSG harian, dan model mana yang menghasilkan akurasi tertinggi?

2. Variabel kovariat mana yang secara signifikan memperbaiki prediksi baseline berdasarkan screening individual, dan bagaimana kombinasi kovariat optimal dibangun secara bottom-up dari hasil screening tersebut?

3. Bagaimana interpretasi ekonomi dari kontribusi variabel prediktor berdasarkan analisis SHAP, dan apakah terdapat perbedaan pola feature importance antara model bagging dan boosting?

---

## 11. REVISED RESEARCH OBJECTIVES (DRAFT)

1. Membangun dan membandingkan empat model ensemble learning berbasis pohon keputusan (Random Forest, Extra Trees, XGBoost, LightGBM) untuk prediksi IHSG harian dengan hyperparameter yang dioptimasi menggunakan Bayesian Optimization (TPE).

2. Mengidentifikasi variabel kovariat yang signifikan melalui screening individual, membangun konfigurasi kovariat optimal secara bottom-up, dan mengevaluasi performa seluruh skenario eksperimen menggunakan expanding window cross-validation.

3. Menginterpretasi kontribusi setiap variabel prediktor terhadap prediksi IHSG menggunakan SHAP analysis dan menganalisis perbedaan pola feature importance antara paradigma bagging dan boosting.

---

## 12. IMPLEMENTATION NOTES

### 12.1 Darts Library Compatibility
- RF dan Extra Trees: native support via `RandomForest` / custom wrapper
- XGBoost: `XGBModel` in Darts (check version compatibility)
- LightGBM: `LightGBMModel` in Darts

### 12.2 Optuna Integration
```python
import optuna
from optuna.samplers import TPESampler

sampler = TPESampler(seed=42)
study = optuna.create_study(
    direction='minimize',  # minimize MAPE
    sampler=sampler,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(objective, n_trials=100)
```

### 12.3 Compute Estimates
- Phase 0: ~4 × 100 trials × 5 folds = ~2,000 model fits (longest phase)
- Phase 1: 104 × 5 folds = 520 model fits
- Phase 2: ~32 × 5 folds = 160 model fits
- **Total: ~2,680 model fits** (excluding Phase 0 optimization)

### 12.4 Suggested Execution Order
1. ✅ Data preprocessing (sudah selesai — reuse pipeline existing)
2. 🔄 Phase 0: Tune RF → Tune ET → Tune XGB → Tune LGBM
3. 🔄 Phase 1: Run all 104 screening experiments
4. 📊 Analyze Phase 1 → determine significant covariates
5. 🔄 Phase 2: Run group experiments
6. 📊 Phase 3: SHAP + comparison analysis
7. ✍️ Write up results

---

## 13. TITLE OPTIONS

1. **"Prediksi IHSG Menggunakan Ensemble Learning Berbasis Pohon Keputusan dengan Variabel Makroekonomi dan Komoditas"**
   - Pro: Elegant, academic, covers all 4 models
   - Con: Doesn't mention specific models

2. **"Prediksi IHSG Menggunakan Model Bagging dan Boosting dengan Variabel Makroekonomi dan Komoditas"**
   - Pro: Highlights the bagging vs boosting comparison
   - Con: Less specific

3. **"Perbandingan Random Forest, Extra Trees, XGBoost, dan LightGBM untuk Prediksi IHSG dengan Variabel Makroekonomi dan Komoditas"**
   - Pro: Very explicit
   - Con: Long title

4. **"Prediksi IHSG Menggunakan Ensemble Tree-Based Models dengan Seleksi Kovariat dan Bayesian Optimization"**
   - Pro: Highlights methodology (covariate selection + Optuna)
   - Con: Doesn't mention specific variable categories

**Recommendation:** Option 1 or 3 — discuss with advisor.


### Current Model Instantiation
```python
from darts.models import RandomForestModel

model = RandomForestModel(
    lags=window,
    lags_past_covariates=window if cov_vars else None,
    output_chunk_length=HORIZON,
    n_estimators=500,
    max_depth=5,
    max_features='sqrt',
    max_samples=0.7,
    random_state=42,
    n_jobs=-1,
)
```
### Change 1: ADD 3 NEW MODELS (Extra Trees, XGBoost, LightGBM)

**Darts model classes to use:**
```python
from darts.models import RandomForestModel, XGBModel, LightGBMModel, SKLearnModel
from sklearn.ensemble import ExtraTreesRegressor
```

**Important — Extra Trees in Darts:**
Darts does NOT have a dedicated `ExtraTreesModel` class. Extra Trees is used via the generic `SKLearnModel` wrapper, which can wrap any scikit-learn-like regression model. Pattern:

```python
from darts.models import SKLearnModel
from sklearn.ensemble import ExtraTreesRegressor

et_model = SKLearnModel(
    lags=window,
    lags_past_covariates=window if cov_vars else None,
    output_chunk_length=HORIZON,
    model=ExtraTreesRegressor(
        n_estimators=500,
        max_depth=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    ),
)
```

**Notes on Extra Trees behavior:**
- `ExtraTreesRegressor` defaults to `bootstrap=False` (unlike RF which uses `bootstrap=True`)
- Because of this, `max_samples` parameter is ignored unless you explicitly set `bootstrap=True`
- The randomization in ET comes from random threshold splits, not from bootstrap sampling
- This is the philosophical difference between RF (bagging via bootstrap) and ET (bagging via random splits)

**Darts version compatibility:**
- Darts ≥ 0.30: use `SKLearnModel`
- Darts < 0.30: use `RegressionModel` (older name, same functionality)
- Check with: `import darts; print(darts.__version__)`

**Model config dict (placeholder hyperparameters — will be replaced by Optuna results):**
```python
MODEL_CONFIGS = {
    'RandomForest': {
        'class': RandomForestModel,  # native Darts wrapper
        'params': {
            'n_estimators': 500,
            'max_depth': 5,
            'max_features': 'sqrt',
            'max_samples': 0.7,
            'random_state': 42,
            'n_jobs': -1,
        }
    },
    'ExtraTrees': {
        'class': SKLearnModel,
        'sklearn_model': ExtraTreesRegressor,  # passed to SKLearnModel(model=...)
        'params': {
            'n_estimators': 500,
            'max_depth': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            # Note: bootstrap=False by default, so max_samples is ignored
        }
    },
    'XGBoost': {
        'class': XGBModel,
        'params': {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
        }
    },
    'LightGBM': {
        'class': LightGBMModel,
        'params': {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,  # suppress LightGBM warnings
        }
    },
}
```
