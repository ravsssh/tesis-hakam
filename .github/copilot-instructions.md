# Copilot Instructions for IHSG Thesis Research Project

## AI Agent Role
You are a **quantitative analyst (quant)** expert in:
- Capital markets and stock market analysis
- Machine learning for financial time series forecasting
- Indonesian macroeconomics and monetary policy

## Project Overview
This is a **financial/economic research thesis project** predicting the Indonesian Stock Exchange Composite Index (IHSG/^JKSE) using machine learning with:
- Regional stock indices (STI - Straits Times Index)
- Commodity prices (Gold/XAUUSD, Nickel, Copper, Coal, Silver, Tin)
- Indonesian macroeconomic indicators (BI Rate, Money Supply M2 YoY, Inflation YoY, USD/IDR exchange rate, NPL ratio)

## Research Models
### Model 1: IHSG with Macroeconomic Variables (Monthly)
- **Target**: IHSG monthly closing price
- **Covariates**: Inflation_YoY, M2_YoY, USDIDR, BI_Rate, NPL_Ratio
- **Data file**: `dataset/model1.csv` (monthly, January 2015 - January 2025, 121 observations)
- **Framework**: `darts.models.RandomForestModel` with Optuna hyperparameter tuning
- **Benchmark**: XGBoost model for comparison
- **Performance**: MAPE 2.44%, RMSE 234.10, MAE 175.94, R² 0.23
- **Best Hyperparameters**:
  - lags: 6, lags_past_covariates: 1
  - n_estimators: 100, max_depth: 7
  - min_samples_split: 10, min_samples_leaf: 4
  - Scaler: MinMaxScaler
- **Notebook**: `model-1.ipynb`

### Model 2: IHSG with Regional Index & Commodity Prices (Daily)
- **Target**: IHSG daily closing price
- **Covariates**: STI (Straits Times Index), Coal, Copper, Silver, Tin, Nickel
- **Data files**:
  - `dataset/ihsg_daily.csv` (IHSG daily data)
  - `dataset/STI.csv`, `dataset/Coal.csv`, `dataset/Copper.csv`, `dataset/Silver.csv`, `dataset/Tin.csv`, `dataset/Nickel.csv`
- **Framework**: `darts.models.RandomForestModel` with Optuna tuning
- **Benchmark**: XGBoost model for comparison
- **Critical Fix**: Uses only actual trading days (no forward-filling)
- **Notebook**: `model-2.ipynb`

## Key Findings from Model 1

### Model Performance
- **Random Forest outperforms XGBoost** (4/4 metrics):
  - Random Forest: MAPE 2.44%, RMSE 234.10, MAE 175.94, R² 0.23
  - XGBoost: MAPE 2.45%, RMSE 239.86, MAE 177.43, R² 0.19

### Feature Importance (SHAP Analysis)
Top contributors to IHSG prediction:
1. **IHSG historical lags**: 82.16% (autoregressive component dominant)
2. **NPL Ratio**: 9.08% (credit quality indicator)
3. **Inflation YoY**: 3.76%
4. **USDIDR**: 2.30%
5. **BI Rate**: 1.41%
6. **M2 YoY**: 1.28%

**Key Insight**: Past IHSG values are the strongest predictor (autoregressive behavior), but NPL Ratio shows significant importance among macroeconomic factors, suggesting credit quality is a key leading indicator for stock market performance.

### Hyperparameter Tuning Results
- **Method**: Optuna (TPE sampler) with 100 trials (~1% of search space)
- **Best configuration**:
  - Short target lag (6 months) + minimal covariate lag (1 month)
  - Moderate ensemble (100 estimators) with deeper trees (max_depth=7)
  - No bootstrap, larger min_samples for regularization
- **Validation MAPE**: 5.94% (tuning phase) → **Test MAPE**: 2.44% (final evaluation)

## Directory Structure
- `model-1.ipynb` - Model 1: IHSG with macroeconomic variables (monthly)
- `model-2.ipynb` - Model 2: IHSG with regional index & commodity prices (daily)
- `dataset/model1.csv` - Monthly IHSG + macroeconomic data (Model 1)
- `dataset/ihsg_daily.csv` - Daily IHSG data (Model 2)
- `dataset/STI.csv`, `dataset/Coal.csv`, `dataset/Copper.csv`, `dataset/Silver.csv`, `dataset/Tin.csv`, `dataset/Nickel.csv` - Model 2 covariates
- `Thesis/Dataset model 1(Stock and Commodity)/` - Stock index and commodity correlation analysis
- `Thesis/Dataset model 2(Macroeconomics ID)/` - Macroeconomic factors impact on IHSG
- `Thesis/macrodataset/` - Raw macroeconomic data from Bank Indonesia
- `Thesis/Code/` - Time series analysis code using `darts` library

## Darts Time Series Forecasting

### RandomForestModel Pattern (As Implemented in Model 1 & 2)
```python
from darts import TimeSeries
from darts.models import RandomForestModel, XGBModel
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import optuna
from optuna.samplers import TPESampler

# 1. Convert pandas to darts TimeSeries with proper frequency
df_ts = df_clean.set_index('Date')

# MODEL 1 (Monthly): Safe to use asfreq
df_ts = df_ts.asfreq('ME')  # Month-End frequency for monthly data

# MODEL 2 (Daily): DO NOT use asfreq or ffill
# df_ts = df.set_index('Date')  # Use as-is, no reindexing

# MODEL 1: Monthly macroeconomic data
target_series = TimeSeries.from_dataframe(
    df_ts,
    value_cols='IHSG',
    fill_missing_dates=True,  # OK for monthly
    freq='ME'
)

covariates = TimeSeries.from_dataframe(
    df_ts,
    value_cols=['Inflation_YoY', 'M2_YoY', 'USDIDR', 'BI_Rate', 'NPL_Ratio'],
    fill_missing_dates=True,  # OK for monthly
    freq='ME'
)

# MODEL 2: Daily multi-market data (CORRECTED)
# target_series = TimeSeries.from_dataframe(
#     df_ts,
#     value_cols='IHSG',
#     fill_missing_dates=False,  # Only actual trading days (no filling gaps)
#     freq='D'  # Calendar day frequency (handles irregular gaps)
# )
#
# covariates = TimeSeries.from_dataframe(
#     df_ts,
#     value_cols=['STI', 'Coal', 'Copper', 'Silver', 'Tin', 'Nickel'],
#     fill_missing_dates=False,  # Only actual trading days (no filling gaps)
#     freq='D'  # Calendar day frequency (handles irregular gaps)
# )

# 2. Scale data (MinMaxScaler for better performance)
scaler_target = Scaler(scaler=MinMaxScaler())
scaler_cov = Scaler(scaler=MinMaxScaler())

# Train/test split (80/20)
TRAIN_RATIO = 0.8
split_point = int(len(target_series) * TRAIN_RATIO)

train_target = target_series[:split_point]
test_target = target_series[split_point:]
train_cov = covariates[:split_point]
test_cov = covariates[split_point:]

# Fit scalers on training data only (avoid data leakage)
train_target_scaled = scaler_target.fit_transform(train_target)
test_target_scaled = scaler_target.transform(test_target)
train_cov_scaled = scaler_cov.fit_transform(train_cov)
test_cov_scaled = scaler_cov.transform(test_cov)

# Full scaled series for backtesting
target_scaled = scaler_target.transform(target_series)
cov_scaled = scaler_cov.transform(covariates)

# 3. Hyperparameter Tuning with Optuna (100 trials)
OUTPUT_CHUNK_LENGTH = 1  # 1-step-ahead forecast

def objective(trial):
    lags = trial.suggest_categorical('lags', [1, 3, 6, 12])
    lags_past_covariates = trial.suggest_categorical('lags_past_covariates', [1, 3, 6, 12])
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 200])
    max_depth = trial.suggest_categorical('max_depth', [3, 5, 7, None])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10])
    min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2, 4])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    try:
        model = RandomForestModel(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            output_chunk_length=OUTPUT_CHUNK_LENGTH,
            n_estimators=n_estimators,
            max_depth=max_depth,
            multi_models=True,
            random_state=42,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            n_jobs=-1
        )
        model.fit(train_target_scaled, past_covariates=cov_scaled)

        # Use historical_forecasts for 1-step-ahead predictions
        backtest_pred = model.historical_forecasts(
            series=target_scaled,
            past_covariates=cov_scaled,
            start=test_target_scaled.start_time(),
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False
        )

        mape_score = mape(test_target_scaled, backtest_pred)
        return mape_score
    except:
        return float('inf')

study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

# 4. Train final model with best parameters
best_params = study.best_trial.params
final_model = RandomForestModel(
    lags=best_params['lags'],
    lags_past_covariates=best_params['lags_past_covariates'],
    output_chunk_length=OUTPUT_CHUNK_LENGTH,
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    multi_models=True,
    random_state=42,
    max_features=best_params.get('max_features'),
    min_samples_split=best_params.get('min_samples_split', 2),
    min_samples_leaf=best_params.get('min_samples_leaf', 1),
    bootstrap=best_params.get('bootstrap', True),
    n_jobs=-1
)
final_model.fit(train_target_scaled, past_covariates=cov_scaled)

# 5. Generate 1-step-ahead predictions using historical_forecasts
predictions_scaled = final_model.historical_forecasts(
    series=target_scaled,
    past_covariates=cov_scaled,
    start=test_target_scaled.start_time(),
    forecast_horizon=1,  # 1-step-ahead
    stride=1,            # Move 1 step each time
    retrain=False,
    verbose=True
)

# Inverse transform to original scale
predictions = scaler_target.inverse_transform(predictions_scaled)
test_actual = scaler_target.inverse_transform(test_target_scaled)
```

### Model Evaluation
```python
from darts.metrics import mape, rmse, mae, r2_score

# Calculate evaluation metrics
mape_score = mape(test_actual, predictions)
rmse_score = rmse(test_actual, predictions)
mae_score = mae(test_actual, predictions)
r2 = r2_score(test_actual, predictions)

print("="*60)
print("MODEL EVALUATION METRICS")
print("="*60)
print(f"MAPE (Mean Absolute Percentage Error): {mape_score:.4f}%")
print(f"RMSE (Root Mean Square Error):         {rmse_score:.4f}")
print(f"MAE (Mean Absolute Error):             {mae_score:.4f}")
print(f"R² Score:                              {r2:.4f}")
```

### XGBoost Benchmark Model
Both Model 1 and Model 2 include XGBoost as a benchmark:

```python
from darts.models import XGBModel

# XGBoost with Optuna tuning (similar structure to RandomForest)
def xgb_objective(trial):
    lags = trial.suggest_categorical('lags', [1, 3, 6, 12])
    lags_past_covariates = trial.suggest_categorical('lags_past_covariates', [1, 3, 6, 12])
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 200, 300])
    max_depth = trial.suggest_categorical('max_depth', [3, 5, 7, 10])
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2])
    subsample = trial.suggest_categorical('subsample', [0.6, 0.8, 1.0])
    colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0])

    try:
        model = XGBModel(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            output_chunk_length=OUTPUT_CHUNK_LENGTH,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            verbosity=0
        )
        model.fit(train_target_scaled, past_covariates=cov_scaled)

        backtest_pred = model.historical_forecasts(
            series=target_scaled,
            past_covariates=cov_scaled,
            start=test_target_scaled.start_time(),
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False
        )

        return mape(test_target_scaled, backtest_pred)
    except:
        return float('inf')

xgb_study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=100)
```

### SHAP Feature Importance Analysis
Both models use SHAP for interpretability:

```python
import shap
from sklearn.ensemble import RandomForestRegressor

# Extract underlying sklearn model
rf_estimator = final_model.model.estimators_[0]

# Create lagged feature matrix
def create_lagged_features(target_df, cov_df, target_lags, cov_lags):
    features = pd.DataFrame(index=target_df.index[max(target_lags, cov_lags):])

    # Target lags
    for lag in range(1, target_lags + 1):
        features[f'IHSG_lag{lag}'] = target_df['IHSG'].shift(lag).values[max(target_lags, cov_lags):]

    # Covariate lags
    for col in cov_df.columns:
        for lag in range(1, cov_lags + 1):
            features[f'{col}_lag{lag}'] = cov_df[col].shift(lag).values[max(target_lags, cov_lags):]

    return features.dropna()

X_features = create_lagged_features(target_df, cov_df, lags, lags_cov)
y_target = target_df['IHSG'].values[max(lags, lags_cov) + 1:]
X_train = X_features.iloc[:-1].values

# Train RF for SHAP
rf_shap = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    random_state=42,
    n_jobs=-1
)
rf_shap.fit(X_train, y_target)

# SHAP analysis
explainer = shap.TreeExplainer(rf_shap)
shap_values = explainer.shap_values(X_train)

# Visualizations
shap.summary_plot(shap_values, X_train_df, plot_type="bar")  # Bar plot
shap.summary_plot(shap_values, X_train_df)  # Beeswarm plot
```

## Data Conventions

### CSV Format (Indonesian Locale)
Data from Indonesian sources uses locale-specific formatting:
```python
# Decimal: comma (,) | Thousand separator: period (.)
df[col] = df[col].str.replace('.', '', regex=False)  # Remove thousand sep
df[col] = df[col].str.replace(',', '.', regex=False)  # Convert decimal
df[col] = df[col].astype(float)

# Percentage columns: remove % and divide by 100
df['Perubahan%'] = df['Perubahan%'].str.replace('%', '').str.replace(',', '.').astype(float) / 100.0
```

### Date Parsing
- Indonesian format: `%d/%m/%Y` (e.g., "31/01/2025")
- Month-Year format: `%b-%y` (e.g., "Jan-25")
- US format from yfinance: `parse_dates=['Date']` with default parser

### Column Naming (Indonesian/English)
| Indonesian | English |
|------------|---------|
| Tanggal | Date |
| Terakhir | Close |
| Pembukaan | Open |
| Tertinggi | High |
| Terendah | Low |
| Perubahan% | Change% |

## Key Libraries & Patterns

### Data Sources
```python
import yfinance as yf  # Stock data: ^JKSE (IHSG), ^STI
import pandas_datareader as pdr  # FRED data for US Treasury yields
```

### Time Series Analysis
- Use `darts` library for time series forecasting
- Standardize with `sklearn.preprocessing.StandardScaler` for multi-variable comparison
- Use `MinMaxScaler` for normalization in visualizations

### Visualization Style (Model 1 & 2 Standard)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Thesis color palette
COLORS = {
    'IHSG': '#2E7D32',
    'IHSG_light': '#4CAF50',
    'IHSG_dark': '#1B5E20',
    'Inflation': '#D32F2F',
    'BI_Rate': '#F57C00',
    'M2_YoY': '#1976D2',
    'USDIDR': '#7B1FA2',
    'NPL_Ratio': '#C2185B',
    'train': '#2E7D32',
    'actual': '#1976D2',
    'predicted': '#D32F2F',
    'error': '#9E9E9E',
    'split_line': '#616161',
}

COLOR_PALETTE = ['#2E7D32', '#D32F2F', '#F57C00', '#1976D2', '#7B1FA2', '#C2185B']

# Publication-quality matplotlib settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100          # Display DPI
plt.rcParams['savefig.dpi'] = 300         # Export DPI (publication quality)
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Line and grid settings
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.edgecolor'] = '0.8'

# Always save with 300 DPI and add source
# plt.savefig('filename.png', dpi=300, bbox_inches='tight')
# Add footer: "Source: Author's calculation, 2025"
```

### Date Alignment Pattern
When correlating multiple time series, always align dates first:
```python
common_dates = set(df1['Date']).intersection(set(df2['Date']))
df1_aligned = df1[df1['Date'].isin(common_dates)].sort_values('Date')
```

## Data Period
- Primary analysis period: **2015-01-01 to 2025-01-01** (10 years)
- **Model 1**: Monthly data (121 observations, January 2015 - January 2025)
- **Model 2**: Daily data for stock/commodity correlation
- Train/Test split: **80/20** (consistent across both models)

## Important Notes & Best Practices

### Data Leakage Prevention
**CRITICAL**: Always fit scalers on training data only, then transform both train and test:
```python
# ✅ CORRECT - Fit on train, transform on train and test
train_target_scaled = scaler_target.fit_transform(train_target)
test_target_scaled = scaler_target.transform(test_target)  # No fit!

# ❌ WRONG - This leaks test data information
target_scaled = scaler_target.fit_transform(target_series)
train, test = target_scaled.split_after(0.8)
```

### Frequency Settings for TimeSeries
- **Monthly data**: Use `freq='ME'` (Month-End) with `asfreq('ME')` safe for monthly data
- **Daily data with irregular gaps**: Use `freq='D'` with `fill_missing_dates=False`
  - ⚠️ **DO NOT** use `freq=None` - will fail with ValueError on irregular data
  - ⚠️ **DO NOT** use `asfreq('B')` + `ffill()` - creates identical predictions (see critical issue below)

### 1-Step-Ahead vs Multi-Step Forecasting
Both models use **1-step-ahead forecasting** with `historical_forecasts`:
- `forecast_horizon=1`: Predict next month/day only
- `stride=1`: Move forward one step at a time
- `retrain=False`: Use trained model without retraining (faster, realistic deployment)

### Optuna vs GridSearch
- **Optuna (preferred)**: Smart sampling with TPE, explores ~1% of search space efficiently
- **GridSearch**: Exhaustive search, slower but comprehensive
- Both models use Optuna with 100 trials

### SHAP Analysis Requirements
- Extract underlying sklearn model: `final_model.model.estimators_[0]`
- Reconstruct lagged features manually for SHAP input
- Aggregate SHAP values across lags to get variable-level importance

### ⚠️ CRITICAL: Forward-Filling Issue in Daily Data (Model 2)

**Problem:** Using `asfreq('B')` + `ffill()` on daily financial data causes **identical predictions** for consecutive days.

**Root Cause:**
```python
# ❌ WRONG - Creates artificial data points
df_ts = df.asfreq('B')  # Adds missing business days (weekends, holidays)
df_ts = df_ts.ffill()   # Forward fills: all covariates = last known values
```

**What Happens:**
- Dec 22 (Fri): Real data → Model predicts based on actual STI, Coal, Copper, etc.
- Dec 25-29 (Mon-Fri): No trading data → `ffill()` copies Dec 22 values for **all** covariates
- Model sees **identical input features** → produces **identical predictions** (e.g., all = 7151.29)
- Results: 50+ consecutive days with same prediction value

**Impact:**
- Artificially inflated metrics (MAPE appears better than reality)
- Predictions have no value (just repeating last prediction)
- Violates fundamental assumption: each prediction should use unique information

**Solution:**
```python
# ✅ CORRECT - Use only actual trading days
df_ts = df.set_index('Date')
# DO NOT reindex or forward fill

target_series = TimeSeries.from_dataframe(
    df_ts,
    value_cols='IHSG',
    fill_missing_dates=False,  # Only use actual dates (DON'T fill gaps)
    freq='D'  # Calendar day frequency (handles irregular gaps)
)

# NOTE: Use freq='D' not freq=None
# freq=None fails on irregular data (different market holidays cause ValueError)
# freq='D' tells darts the intended frequency, but with fill_missing_dates=False
# it still only uses actual dates without filling gaps
```

**Key Takeaway:** For multi-market daily data (IHSG + STI + commodities), only predict on days where **all markets traded**. This is:
- More realistic (you can only trade when markets are open)
- Methodologically sound (no artificial data)
- Produces meaningful metrics (true forecasting ability)

## Notebooks Workflow (Model 1 & 2 Standard Pipeline)

### 1. Import Libraries
```python
import pandas as pd, numpy as np
from darts import TimeSeries
from darts.models import RandomForestModel, XGBModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae, r2_score
from sklearn.preprocessing import MinMaxScaler
import optuna, shap
import matplotlib.pyplot as plt, seaborn as sns
```

### 2. Data Loading & Preprocessing
- Load CSV data (monthly or daily)
- Parse dates: `pd.to_datetime(df['Date'], format='%d/%m/%Y')`
- Clean percentage columns: `df[col].str.replace('%', '').astype(float)`
- Sort by date: `df.sort_values('Date').reset_index(drop=True)`
- Rename columns for consistency

### 3. Exploratory Data Analysis (EDA)
- Time series plots for IHSG and covariates
- Correlation matrix with thesis color palette
- Descriptive statistics

### 4. Create Darts TimeSeries Objects
- Set date index with frequency (`freq='ME'` or `'D'`)
- Create target and covariate TimeSeries
- Verify frequency and date range

### 5. Train/Test Split & Scaling
- 80/20 split
- MinMaxScaler on train data (fit), then transform test
- Create full scaled series for backtesting

### 6. Hyperparameter Tuning with Optuna
- 100 trials with TPE sampler
- Search space: lags, n_estimators, max_depth, sklearn params
- Optimize for MAPE using historical_forecasts

### 7. Train Final Model
- Use best parameters from Optuna
- Fit on training data with past covariates

### 8. Generate 1-Step-Ahead Predictions
- Use `historical_forecasts()` with `forecast_horizon=1`
- Inverse transform to original scale

### 9. Model Evaluation
- Calculate MAPE, RMSE, MAE, R²
- Visualize actual vs predicted
- Analyze prediction errors

### 10. Feature Importance (SHAP)
- Reconstruct lagged features
- Train standalone sklearn RandomForest
- Generate SHAP values and plots

### 11. Benchmark with XGBoost
- Same Optuna tuning process
- Compare performance metrics (MAPE, RMSE, MAE, R²)
- Visualization: RF vs XGB comparison

### 12. Results Summary
- Print final metrics and best hyperparameters
- Save predictions to CSV
- Export publication-quality plots (300 DPI)
