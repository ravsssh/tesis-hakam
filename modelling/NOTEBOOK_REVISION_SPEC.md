# NOTEBOOK REVISION SPEC — For Claude Code (VSCode)
# Prediksi IHSG: Multi-Model Tree-Based Ensemble Framework

> **Purpose:** Instruksi coding untuk jupyter notebook eksperimen prediksi timeseries IHSG menggunakan variabel macroeconomics, commoddity price dan regional index dengan menggunakan tree-based ensemble (extra tree, xgboost, random forest dan lightgbm) dengan hyperparameter tuning menggunakan optuna

---

## CONTEXT: APA YANG SUDAH ADA DI NOTEBOOK

### Current Pipeline (Jika ada tahap yang miss atau missinterpretasi kasih tahu)
1. **Data loading** — CSV dari GitHub, merge macro (monthly) + daily via `merge_asof`
2. **ADF test** — semua variabel non-stationary → transformasi
3. **Transformasi:**
   - LEVEL_VARS → `diff(log(x))`: M2, USDIDR, Coal, Copper, Nickel, Silver, Tin, Gold, STI, WTI, GDP
   - RATE_VARS → `diff(x)`: BI_Rate, CPI, NPL_Ratio, US_Treasury_10Y
   - Target IHSG → `diff(log(x))`
4. **Scaling** — MinMaxScaler fit on train only, transform on full data
5. **Darts TimeSeries** conversion — target + past_covariates
6. **Train/Test Split** — temporal split, train 80% awal, test 20% akhir
7. **Model** — `RandomForestModel` from Darts
8. **Metrics** — MAPE, RMSE, MAE, R², DA (single evaluation on test set)
9. **SHAP** — feature importance from RF
10. **Visualization** — prediction plots, residuals, covariate comparison

### Current Experiment Loop Structure (approximate)
> **Catatan:** Loop di bawah ini adalah struktur dari studi awal (RF only, 7 covariate sets, W30/W60/W120).
> Revisi ini **mengganti** struktur tersebut dengan two-phase design di bawah. Window W60 tidak digunakan di revision — hanya W30 dan W120.

```python
COVARIATE_SETS = {
    'None': [],
    'Macro': ['BI_Rate', 'CPI', 'M2', 'NPL_Ratio', 'USDIDR'],
    'Commodity': ['Coal', 'Copper', 'Nickel', 'Silver', 'Tin', 'Gold'],
    'Regional': ['STI'],
    'Macro_Regional': ['BI_Rate', 'CPI', 'M2', 'NPL_Ratio', 'USDIDR', 'STI'],
    'Commodity_Regional': ['Coal', 'Copper', 'Nickel', 'Silver', 'Tin', 'Gold', 'STI'],
    'Full': ['BI_Rate', 'CPI', 'M2', 'NPL_Ratio', 'USDIDR', 'Coal', 'Copper', 'Nickel', 'Silver', 'Tin', 'Gold', 'STI'],
}
WINDOWS = [30, 60, 120]  # NOTE: revision hanya pakai [30, 120]
HORIZON = 1

for cov_name, cov_vars in COVARIATE_SETS.items():
    for window in WINDOWS:
        # build model, fit on train set, evaluate on test set
```

### Machine learning model used

**Darts model classes to use:**
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

### Change 2: REPLACE COVARIATE SETS WITH TWO-PHASE DESIGN

**Phase 1 — Single Covariate Screening:**
```python
# REPLACE the old COVARIATE_SETS dict with this:
SINGLE_COVARIATES = {
    'None':            [],           # baseline
    # Macro Domestic (existing 5)
    'BI_Rate':         ['BI_Rate'],
    'CPI':             ['CPI'],
    'M2':              ['M2'],
    'NPL_Ratio':       ['NPL_Ratio'],
    'USDIDR':          ['USDIDR'],
    # Macro Domestic (NEW)
    'GDP':             ['GDP'],
    # Macro Global (NEW)
    'US_Treasury_10Y': ['US_Treasury_10Y'],
    # Commodity (existing 6)
    'Coal':            ['Coal'],
    'Copper':          ['Copper'],
    'Nickel':          ['Nickel'],
    'Silver':          ['Silver'],
    'Tin':             ['Tin'],
    'Gold':            ['Gold'],
    # Commodity (NEW - energy)
    'WTI':             ['WTI'],
    # Regional (existing 1)
    'STI':             ['STI'],
}
# Total: 16 configs × 2 windows × 4 models = 128 experiments
```

**Variable categories (15 covariates total):**
- **Macro** (7): BI_Rate, CPI, M2, NPL_Ratio, USDIDR, GDP, US_Treasury_10Y
- **Commodity** (7): Coal, Copper, Nickel, Silver, Tin, Gold, WTI
- **Regional** (1): STI

**Notes on new variables:**
- **GDP**: Quarterly data from BPS via Bloomberg. Forward-fill ke daily (~63 trading days per value). Expect GDP mungkin gagal screening karena staleness — ini bisa jadi finding menarik untuk diskusi.
- **US_Treasury_10Y**: Daily yield data, treated as rate variable → first difference.
- **WTI**: Daily crude oil price, treated as level variable → log-difference. Brent tidak dipakai karena sangat berkorelasi dengan WTI (>0.95).

**Phase 2 — Group Configuration (run AFTER Phase 1 analysis):**
```python
# This dict will be populated AFTER analyzing Phase 1 results.
# Example structure (actual groups depend on screening results):
GROUP_COVARIATES = {
    'Sig_Macro':     [...],  # macro vars that passed screening
    'Sig_Commodity': [...],  # commodity vars that passed screening
    'Sig_All':       [...],  # all significant covariates combined
    # possibly more groups based on results
}
```


### OPTUNA HYPERPARAMETER TUNING (Separate Notebook/Section)

**Create a NEW notebook or section: `01_hyperparameter_tuning.ipynb`**

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

def create_objective(model_name, target_ts, window=120):
    """Create Optuna objective function for a given model."""
    
    def objective(trial):
        # Define search space per model
        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'max_samples': trial.suggest_float('max_samples', 0.5, 0.9, step=0.1),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            model = RandomForestModel(
                lags=window,
                output_chunk_length=1,
                random_state=42,
                n_jobs=-1,
                **params,
            )
        
        elif model_name == 'ExtraTrees':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            # Extra Trees via SKLearnModel wrapper (no native Darts class)
            model = SKLearnModel(
                lags=window,
                output_chunk_length=1,
                model=ExtraTreesRegressor(
                    random_state=42,
                    n_jobs=-1,
                    **params,
                ),
            )
        
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            model = XGBModel(
                lags=window,
                output_chunk_length=1,
                random_state=42,
                n_jobs=-1,
                **params,
            )
        
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            }
            model = LightGBMModel(
                lags=window,
                output_chunk_length=1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                **params,
            )
        
        # Fit on train set, evaluate on test set
        train_ts, test_ts = target_ts[:int(len(target_ts) * 0.8)], target_ts[int(len(target_ts) * 0.8):]
        model.fit(train_ts)
        pred = model.predict(len(test_ts))
        from darts.metrics import mape as mape_metric
        return mape_metric(test_ts, pred)  # minimize MAPE
    
    return objective

# Run tuning for each model
TUNING_RESULTS = {}
for model_name in ['RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM']:
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'IHSG_{model_name}',
    )
    objective = create_objective(model_name, target_ts, window=120)
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    TUNING_RESULTS[model_name] = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study,
    }
    print(f"\n{model_name}: Best MAPE = {study.best_value:.4f}%")
    print(f"  Best params: {study.best_params}")

# Save results
import joblib
joblib.dump(TUNING_RESULTS, 'optuna_tuning_results.joblib')
```

### Change 5: REVISE MAIN EXPERIMENT LOOP

**Replace the old single-model loop with multi-model loop:**
```python
# After loading Optuna results:
TUNING_RESULTS = joblib.load('optuna_tuning_results.joblib')

# Phase 1 experiment loop
results = []

for model_name, model_cfg in MODEL_CONFIGS.items():
    # Use Optuna-tuned hyperparameters
    best_params = TUNING_RESULTS[model_name]['best_params']
    
    for cov_name, cov_vars in SINGLE_COVARIATES.items():
        for window in WINDOWS:
            print(f"Running: {model_name} | {cov_name} | W{window}_H1")
            
            # Build model with tuned params
            model = build_model(
                model_name=model_name,
                params=best_params,
                window=window,
                horizon=HORIZON,
                has_covariates=(len(cov_vars) > 0),
            )
            
            # Temporal train/test split (80/20)
            split_idx = int(len(target_ts) * 0.8)
            train_ts = target_ts[:split_idx]
            test_ts  = target_ts[split_idx:]
            
            cov_ts = covariate_ts[cov_vars] if cov_vars else None
            train_cov = cov_ts[:split_idx] if cov_ts is not None else None
            
            model.fit(train_ts, past_covariates=train_cov)
            pred = model.predict(len(test_ts), past_covariates=cov_ts)
            
            metrics = calculate_metrics(test_ts, pred)
            
            # Collect results
            results.append({
                'Model': model_name,
                'Covariates': cov_name,
                'Window': window,
                'MAPE': metrics['mape'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R2': metrics['r2'],
                'DA': metrics['da'],
            })

df_results = pd.DataFrame(results)
df_results.to_csv('phase1_screening_results.csv', index=False)
```

### Change 6: ADD SCREENING ANALYSIS (after Phase 1)

> **Catatan:** `phase1_screening_results.csv` yang sudah ada memiliki kolom `MAPE_mean` (format lama dengan k-fold).
> Fungsi di bawah membaca kolom tersebut. Jika di masa depan format CSV di-update ke kolom `MAPE` (tanpa k-fold),
> ganti semua `MAPE_mean` di bawah menjadi `MAPE`.

```python
def analyze_screening(df_results, threshold_pct=0.3):
    """
    Identify covariates that improve MAPE by >= threshold_pct% 
    relative to baseline (None), for each model × window combo.
    
    Membaca kolom 'MAPE_mean' dari phase1_screening_results.csv (format existing).
    """
    # Normalisasi kolom: support kedua format (MAPE_mean dari k-fold, atau MAPE dari simple split)
    mape_col = 'MAPE_mean' if 'MAPE_mean' in df_results.columns else 'MAPE'
    
    screening = []
    
    for model_name in df_results['Model'].unique():
        for window in df_results['Window'].unique():
            mask = (df_results['Model'] == model_name) & (df_results['Window'] == window)
            subset = df_results[mask]
            
            baseline_mape = subset[subset['Covariates'] == 'None'][mape_col].values[0]
            
            for _, row in subset.iterrows():
                if row['Covariates'] == 'None':
                    continue
                
                improvement = (baseline_mape - row[mape_col]) / baseline_mape * 100
                passed = improvement >= threshold_pct
                
                screening.append({
                    'Model': model_name,
                    'Window': window,
                    'Covariate': row['Covariates'],
                    'MAPE': row[mape_col],
                    'Baseline_MAPE': baseline_mape,
                    'Improvement_pct': improvement,
                    'Passed': passed,
                })
    
    df_screening = pd.DataFrame(screening)
    return df_screening

df_screening = analyze_screening(df_results, threshold_pct=0.3)

# Summary: which covariates passed in how many model×window combos
pass_counts = df_screening[df_screening['Passed']].groupby('Covariate').size()
print("Covariate pass counts (out of 8 model×window combos):")
print(pass_counts.sort_values(ascending=False))

# Build Phase 2 groups from survivors
significant_macro = [v for v in ['BI_Rate','CPI','M2','NPL_Ratio','USDIDR','GDP','US_Treasury_10Y'] 
                     if v in pass_counts.index]
significant_commodity = [v for v in ['Coal','Copper','Nickel','Silver','Tin','Gold','WTI'] 
                         if v in pass_counts.index]
significant_regional = ['STI'] if 'STI' in pass_counts.index else []
significant_all = significant_macro + significant_commodity + significant_regional

GROUP_COVARIATES = {}
if significant_macro:
    GROUP_COVARIATES['Sig_Macro'] = significant_macro
if significant_commodity:
    GROUP_COVARIATES['Sig_Commodity'] = significant_commodity
if significant_regional:
    GROUP_COVARIATES['Sig_Regional'] = significant_regional
if len(significant_all) > 1:
    GROUP_COVARIATES['Sig_All'] = significant_all

print(f"\nPhase 2 groups: {GROUP_COVARIATES}")
```

### Change 7: SHAP ANALYSIS UPDATES

**Extend SHAP to cover all 4 models (not just RF):**
```python
# SHAP works differently for tree-based models:
# - RF, ExtraTrees: use shap.TreeExplainer (native)
# - XGBoost: use shap.TreeExplainer (native XGBoost support)
# - LightGBM: use shap.TreeExplainer (native LightGBM support)
# All 4 models support TreeExplainer, so the approach is unified.

import shap

def get_shap_values(model, X_test):
    """Extract SHAP values from trained Darts model."""
    # Access underlying sklearn/xgb/lgb model
    underlying_model = model.model  # Darts stores the sklearn model here
    # For Darts RegressionModel, it may be model.model or model.model.estimator
    
    explainer = shap.TreeExplainer(underlying_model)
    shap_values = explainer.shap_values(X_test)
    return shap_values, explainer
```

---

## IMPLEMENTATION STATUS

Status per notebook dan file output per tanggal revisi spec ini:

| Notebook | Status | Keterangan |
|----------|--------|------------|
| `00_data_preprocessing.ipynb` | ✅ DONE | Data load, merge, ADF, transformasi, scaling selesai |
| `01_hyperparameter_tuning.ipynb` | ⚠️ EXISTS | File ada, perlu cek apakah Optuna sudah di-run atau masih template |
| `02_phase1_screening.ipynb` | ✅ DONE | 128 eksperimen (4 model × 16 covariate × 2 window) selesai |
| `03_screening_analysis.ipynb` | ❌ TODO | Analisis Phase 1 → design Phase 2 groups |
| `04_phase2_groups.ipynb` | ❌ TODO | Group experiments dari survivors Phase 1 |
| `05_shap_analysis.ipynb` | ❌ TODO | SHAP multi-model (4 model) |
| `06_visualization.ipynb` | ❌ TODO | Multi-model comparison plots |

| File Output | Status | Keterangan |
|-------------|--------|------------|
| `phase1_screening_results.csv` | ✅ DONE | 128 baris, semua 4 model |
| `phase1_screening_analysis.csv` | ✅ DONE | 120 baris (4 model × 15 covariate × 2 window) |
| `optuna_tuning_results.joblib` | ⚠️ CEK | Perlu verifikasi apakah sudah ada |
| `phase2_group_results.csv` | ❌ TODO | Menunggu Phase 2 |
| `shap_feature_importance_*.csv` | ❌ TODO | Menunggu SHAP analysis |

---

## NOTEBOOK STRUCTURE RECOMMENDATION

Split into separate notebooks for manageability:

```
notebooks/
├── 00_data_preprocessing.ipynb          # Data load, merge, ADF, transform, scale (DONE ✅)
├── 01_hyperparameter_tuning.ipynb       # Optuna tuning 4 models (CEK ⚠️)
├── 02_phase1_screening.ipynb            # 128 single covariate experiments (DONE ✅)
├── 03_screening_analysis.ipynb          # TODO: Analyze Phase 1, design Phase 2 groups
├── 04_phase2_groups.ipynb               # TODO: Group experiments from Phase 1 survivors
├── 05_shap_analysis.ipynb               # TODO: SHAP extended to all 4 models
├── 06_visualization.ipynb               # TODO: Multi-model comparison plots
└── utils/
    ├── models.py                        # build_model() function for all 4 models
    ├── evaluation.py                    # evaluate_model(), calculate_metrics()
    └── screening.py                     # analyze_screening()
```

---

## HELPER FUNCTIONS TO CREATE

### `build_model()` — unified model factory
```python
def build_model(model_name, params, window, horizon=1, has_covariates=True):
    """
    Build a Darts model with given hyperparameters.
    
    Args:
        model_name: 'RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM'
        params: dict of hyperparameters (from Optuna)
        window: int, lookback window (lags)
        horizon: int, forecast horizon (always 1)
        has_covariates: bool, whether to set lags_past_covariates
    
    Returns:
        Darts model instance
    """
    common = {
        'lags': window,
        'lags_past_covariates': window if has_covariates else None,
        'output_chunk_length': horizon,
    }
    
    if model_name == 'RandomForest':
        return RandomForestModel(**common, **params, random_state=42, n_jobs=-1)
    elif model_name == 'ExtraTrees':
        # ExtraTrees has no native Darts wrapper — use SKLearnModel
        return SKLearnModel(
            **common,
            model=ExtraTreesRegressor(random_state=42, n_jobs=-1, **params),
        )
    elif model_name == 'XGBoost':
        return XGBModel(**common, **params, random_state=42)
    elif model_name == 'LightGBM':
        return LightGBMModel(**common, **params, random_state=42, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
```

**Important parameter notes:**
- `RandomForestModel` accepts `n_jobs` directly as a Darts model parameter
- `XGBModel` and `LightGBMModel` pass `n_jobs` via `**kwargs` to underlying xgboost/lightgbm
- `SKLearnModel` does NOT accept `n_jobs` directly — it must be passed to the underlying sklearn model (`ExtraTreesRegressor(n_jobs=-1, ...)`)
- Same applies to `random_state`: pass to the underlying sklearn model when using `SKLearnModel`

### `calculate_metrics()` — helper untuk hitung semua metrik dari prediksi Darts
```python
import numpy as np
from darts.metrics import mape, rmse, mae, r2_score

def calculate_metrics(actual_ts, pred_ts):
    """
    Hitung semua metrik dari dua Darts TimeSeries.
    
    Returns:
        dict: {'mape': float, 'rmse': float, 'mae': float, 'r2': float, 'da': float}
    """
    actual = actual_ts.values().flatten()
    pred   = pred_ts.values().flatten()
    
    # Directional Accuracy: sign(pred) == sign(actual) untuk log-return target
    da = np.mean(np.sign(pred) == np.sign(actual)) * 100
    
    return {
        'mape': mape(actual_ts, pred_ts),
        'rmse': rmse(actual_ts, pred_ts),
        'mae':  mae(actual_ts, pred_ts),
        'r2':   r2_score(actual_ts, pred_ts),
        'da':   da,
    }
```

---

## CRITICAL CONSTRAINTS

1. **DO NOT change data preprocessing** — ADF, transformasi, scaling pipeline sudah benar
2. **Evaluasi: temporal 80/20 split** — train = 80% awal, test = 20% akhir, tidak ada k-fold
3. **Hyperparameters are PLACEHOLDERS** — notebook harus bisa swap in Optuna results easily
4. **Random state = 42 everywhere** — untuk reprodusibilitas
5. **Save ALL results to CSV/joblib** — setiap phase harus punya output file
6. **Progress logging** — print statement di setiap experiment karena total ~136 runs
7. **Memory management** — `del model` dan `gc.collect()` setelah setiap experiment
8. **Error handling** — wrap experiment loop dalam try/except, log failures, continue

---

## OUTPUT FILES EXPECTED

| Phase | File | Content |
|-------|------|---------|
| Phase 0 | `optuna_tuning_results.joblib` | Best params per model + study objects |
| Phase 0 | `optuna_tuning_summary.csv` | Summary table of best params |
| Phase 1 | `phase1_screening_results.csv` | 128 rows (4 model × 16 covariate × 2 window), kolom: MAPE, RMSE, MAE, R2, DA |
| Phase 1 | `phase1_screening_analysis.csv` | Screening pass/fail per covariate |
| Phase 2 | `phase2_group_results.csv` | ~32 rows, group experiment results |
| Final | `all_experiment_results.csv` | Combined Phase 1 + Phase 2 |
| SHAP | `shap_feature_importance_*.csv` | Per-model SHAP values |

---

## DARTS LIBRARY NOTES

### Class hierarchy (Darts ≥ 0.30)

| Model | Darts class | Notes |
|-------|-------------|-------|
| Random Forest | `RandomForestModel` | Native Darts wrapper, accepts `n_estimators`, `max_depth` directly |
| Extra Trees | `SKLearnModel` + `ExtraTreesRegressor` | **No native wrapper.** Use `SKLearnModel(model=ExtraTreesRegressor(...))` |
| XGBoost | `XGBModel` | Native Darts wrapper, kwargs passed to xgboost |
| LightGBM | `LightGBMModel` | Native Darts wrapper, kwargs passed to lightgbm |

### Verified Darts API signatures (from official docs)

```python
# Random Forest
RandomForestModel(
    lags=None, lags_past_covariates=None, lags_future_covariates=None,
    output_chunk_length=1, output_chunk_shift=0, add_encoders=None,
    n_estimators=100, max_depth=None,
    multi_models=True, use_static_covariates=True, random_state=None,
    **kwargs  # passed to sklearn RandomForestRegressor
)

# XGBoost
XGBModel(
    lags=None, lags_past_covariates=None, lags_future_covariates=None,
    output_chunk_length=1, output_chunk_shift=0, add_encoders=None,
    likelihood=None, quantiles=None, random_state=None,
    multi_models=True, use_static_covariates=True,
    **kwargs  # passed to xgboost.XGBRegressor
)

# LightGBM
LightGBMModel(
    lags=None, lags_past_covariates=None, lags_future_covariates=None,
    output_chunk_length=1, output_chunk_shift=0, add_encoders=None,
    likelihood=None, quantiles=None, random_state=None,
    multi_models=True, use_static_covariates=True,
    categorical_past_covariates=None, categorical_future_covariates=None,
    categorical_static_covariates=None,
    **kwargs  # passed to lightgbm.LGBMRegressor
)

# SKLearnModel (for Extra Trees and any other sklearn-like model)
SKLearnModel(
    lags=None, lags_past_covariates=None, lags_future_covariates=None,
    output_chunk_length=1, output_chunk_shift=0, add_encoders=None,
    model=None,  # ← pass ExtraTreesRegressor() instance here
    multi_models=True, use_static_covariates=True,
)
```

### Key behavioral notes

- **Hyperparameters for RF**: Things like `max_features`, `max_samples`, `min_samples_split` are passed via `**kwargs` to `RandomForestRegressor` — they're NOT explicit parameters of `RandomForestModel`
- **Hyperparameters for ET**: All sklearn params go inside `ExtraTreesRegressor(...)`, NOT to `SKLearnModel` itself
- **`random_state` handling**:
  - `RandomForestModel`, `XGBModel`, `LightGBMModel`: pass directly as kwarg
  - `SKLearnModel`: pass to underlying model `ExtraTreesRegressor(random_state=42)`
- **`n_jobs` handling**: same pattern as `random_state`
- **Extra Trees `bootstrap=False` by default**: `max_samples` is ignored unless you set `bootstrap=True`. Don't include `max_samples` in ET's Optuna search space.

### Package requirements

- `darts >= 0.30` (for `SKLearnModel` name; older versions use `RegressionModel`)
- `xgboost` (for `XGBModel`)
- `lightgbm` (for `LightGBMModel`)
- `scikit-learn` (for `RandomForestRegressor`, `ExtraTreesRegressor`)

### SHAP integration

- For `RandomForestModel`, `XGBModel`, `LightGBMModel`: access underlying model via `model.model` 
- For `SKLearnModel` wrapping ExtraTrees: access via `model.model` (returns the `ExtraTreesRegressor` instance, possibly wrapped in `MultiOutputRegressor` if multi-output)
- All 4 models work with `shap.TreeExplainer`

---

## QUICK REFERENCE: EXPERIMENT MATRIX

### Phase 1 (128 experiments)
```
Models:     [RandomForest, ExtraTrees, XGBoost, LightGBM]  → 4
Covariates: [None,
             BI_Rate, CPI, M2, NPL_Ratio, USDIDR, GDP, US_Treasury_10Y,  (7 macro)
             Coal, Copper, Nickel, Silver, Tin, Gold, WTI,               (7 commodity)
             STI]                                                         (1 regional)
                                                            → 16
Windows:    [30, 120]                                       → 2
Total:      4 × 16 × 2 = 128
```

### Phase 2 (~32 experiments, estimated)
```
Models:     [RandomForest, ExtraTrees, XGBoost, LightGBM]  → 4
Covariates: [Sig_Macro, Sig_Commodity, Sig_All, ...]       → ~4 (TBD from screening)
Windows:    [30, 120]                                       → 2
Total:      4 × ~4 × 2 = ~32
```

### Grand Total: ~160 experiments

---

## NEXT STEPS (Phase 2 ke Atas)

Urutan pengerjaan yang tersisa setelah Phase 1 selesai:

1. **Verifikasi Optuna** (`01_hyperparameter_tuning.ipynb`)
   - Cek apakah `optuna_tuning_results.joblib` sudah ada di folder
   - Kalau belum: jalankan Optuna tuning dulu (100 trials × 4 model = 400 trials, ~berjam-jam)
   - Kalau sudah: load saja dan lanjut ke Phase 2

2. **Design Phase 2 Groups** (`03_screening_analysis.ipynb`)
   - Load `phase1_screening_analysis.csv` yang sudah ada
   - Identifikasi covariate yang konsisten "Passed=True" di mayoritas model×window combo
   - Susun `GROUP_COVARIATES` dict (Sig_Macro, Sig_Commodity, Sig_All, dst.)
   - Threshold 0.3% improvement di MAPE relatif terhadap baseline

3. **Phase 2 Group Experiments** (`04_phase2_groups.ipynb`)
   - ~32 eksperimen: 4 model × ~4 grup × 2 window
   - Gunakan Optuna-tuned hyperparameters dari step 1
   - Save ke `phase2_group_results.csv`
   - Identifikasi best config (model + covariate group + window) untuk SHAP analysis

4. **SHAP Analysis** (`05_shap_analysis.ipynb`)
   - Gunakan best config dari Phase 2 sebagai model utama
   - Jalankan SHAP pada data test set (20% akhir) — lihat Methodological Notes
   - Plot: summary plot, beeswarm, bar chart per model
   - Save ke `shap_feature_importance_[model]_[timestamp].csv`

5. **Visualization Komparatif** (`06_visualization.ipynb`)
   - Heatmap MAPE per model × covariate group
   - Bar chart ranking model (mean MAPE ± std across best configs)
   - SHAP comparison antar model
   - Directional accuracy comparison

---

## DATA LOADING NOTES (NEW VARIABLES)

3 new variables harus di-merge ke dataset existing:

```python
# WTI Crude Oil — daily price (level)
# US Treasury 10Y — daily yield (rate, percentage)
# GDP Indonesia — quarterly value (level), forward-fill ke daily

# Source: Bloomberg Terminal (sama dengan dataset existing)

# Frekuensi merge:
# - WTI: daily, langsung merge
# - US_Treasury_10Y: daily, langsung merge
# - GDP: quarterly → forward-fill via merge_asof (direction='backward')
#   Sama treatment seperti BI_Rate, CPI, M2, NPL_Ratio yang monthly

# Catatan stale data:
# - GDP rilis quarterly → 1 nilai bertahan ~63 trading days
# - Ini lebih stale dari macro lain (monthly ~22 days)
# - Jangan kaget kalau GDP gagal screening
```

---

## METHODOLOGICAL NOTES & SARAN

Catatan metodologis berikut untuk memperkuat kaidah penelitian. Bukan keharusan semua diimplementasi,
tapi minimal yang bertanda **[PENTING]** perlu dibahas di tesis (methodology atau limitations section).

### a) Naive Benchmark **[PENTING]**

Tanpa naive benchmark, klaim "MAPE 0.66%" tidak punya konteks yang kuat. Reviewer tesis hampir pasti akan menanyakan ini.

```python
# Naive model 1: Random Walk — prediksi besok = hari ini (log-return = 0)
# Untuk log-differenced target, ini berarti prediksi = 0 (no change)
naive_mape = mean(abs(actual_log_returns)) / mean(abs(actual_log_returns)) * 100  # = 100% jika prediksi selalu 0

# Naive model 2: Mean historical return
# prediksi = mean(training log-returns)

# Cara mudah via Darts:
from darts.models import NaiveMean, NaiveSeasonal
naive_model = NaiveMean()  # predicts the training mean
```

Jika model RF/XGB/dll mengalahkan naive model dengan signifikan → bukti bahwa model benar-benar "belajar" sesuatu.
Jika tidak → perlu diskusi di limitations.

### b) GDP Publication Lag **[PENTING — berdampak pada validitas]**

GDP Indonesia dirilis BPS **~2 bulan setelah akhir kuartal**:
- Q1 (Jan–Mar) → rilis ~Mei
- Q2 (Apr–Jun) → rilis ~Agustus
- dst.

Jika `merge_asof` menggunakan tanggal kuartal (misal Jan 1, Apr 1) tanpa di-offset, model akan melihat GDP Q1 sejak Januari — padahal data itu baru tersedia di Mei. Ini adalah **look-ahead bias**.

**Saran implementasi:**
```python
# Offset GDP 60 hari ke depan sebelum merge
gdp_df['Date'] = gdp_df['Date'] + pd.DateOffset(days=60)
# Kemudian merge_asof seperti biasa
```

Kalau GDP tetap gagal screening setelah di-offset → metodologi semakin valid dan hasilnya lebih credible.
Kalau tidak sempat implementasi → wajib dicantumkan sebagai limitasi di tesis.

### c) SHAP Harus Dihitung pada Data Test (Out-of-Sample)

SHAP dari training data = "apa yang dipelajari model dari data historis".
SHAP dari test data = "apa yang digunakan model untuk prediksi aktual".

Untuk explainability yang valid dalam konteks prediksi:

```python
# Jangan:
shap_values = explainer.shap_values(X_train)  # bias ke training patterns

# Lakukan:
# Hitung SHAP pada X_test (20% akhir data)
shap_values = explainer.shap_values(X_test)
```

Ini memastikan SHAP values konsisten dengan performa model yang dilaporkan.

### d) Uji Statistik untuk Perbandingan Model **[DISARANKAN]**

Range MAPE antar model di Phase 1 kemungkinan < 0.01 percentage point. Perbedaan sekecil itu
**tidak bisa diklaim signifikan tanpa uji statistik**.

**Diebold-Mariano Test** cocok untuk membandingkan forecast accuracy dua model dari error time series:
```python
# pip install arch
from arch.tests.utility import diebold_mariano

# e1 = per-timestep forecast errors model A (pada test set)
# e2 = per-timestep forecast errors model B (pada test set)
e1 = actual_values - pred_model_a
e2 = actual_values - pred_model_b
dm_stat, dm_pvalue = diebold_mariano(e1, e2, power=2)  # power=2 → MSE-based
```

### e) Metrik Utama: MAE vs MAPE

MAPE sudah valid di sini karena dihitung pada level IHSG (bukan log-return). Namun untuk tesis:
- **MAPE**: lebih intuitive untuk pembaca umum ("error rata-rata 0.66%")
- **MAE**: lebih robust terhadap outlier, satuannya jelas (poin IHSG)
- **DA (Directional Accuracy)**: metrik paling relevan untuk investor — apakah prediksi arah naik/turun benar?

**Rekomendasi**: Gunakan MAPE sebagai primary ranking metric (seperti sekarang), tapi highlight DA
lebih prominently di tesis karena itulah yang paling actionable untuk keputusan investasi.

Catatan DA saat ini (50–54%) = sedikit di atas random guess (50%). Ini **bukan kelemahan yang perlu disembunyikan** —
justru ini menunjukkan bahwa model mampu mengestimasi magnitude (R²=0.97, MAPE rendah) tapi pasar sebagian
besar mengikuti random walk untuk arah pergerakan harian.
