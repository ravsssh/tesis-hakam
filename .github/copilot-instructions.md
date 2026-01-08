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
- **Target**: IHSG closing price
- **Covariates**: Money Supply M2 YoY, USD/IDR, Inflation Rate YoY, BI Interest Rate, NPL Ratio
- **Data file**: `data1.csv` (monthly, 2015-2025)
- **Framework**: `darts.models.forecasting.RandomForestModel`

### Model 2: IHSG with Commodity Prices (Daily) - TBD
- **Target**: IHSG daily close
- **Covariates**: Gold, Nickel, Copper, Coal, Silver, Tin prices

## Directory Structure
- `data1.csv` - Monthly IHSG + macroeconomic data (Model 1)
- `Thesis/Dataset model 1(Stock and Commodity)/` - Stock index and commodity correlation analysis
- `Thesis/Dataset model 2(Macroeconomics ID)/` - Macroeconomic factors impact on IHSG
- `Thesis/macrodataset/` - Raw macroeconomic data from Bank Indonesia
- `Thesis/Code/` - Time series analysis code using `darts` library
- Root CSVs (`Coal.csv`, `Nickel.csv`, etc.) - Raw commodity price data

## Darts Time Series Forecasting

### RandomForestModel Pattern
```python
from darts import TimeSeries
from darts.models import RandomForestModel
from darts.dataprocessing.transformers import Scaler

# Convert pandas to darts TimeSeries
target_series = TimeSeries.from_dataframe(df, time_col='Date', value_cols='IHSG')
covariates = TimeSeries.from_dataframe(df, time_col='Date', 
    value_cols=['Money Supply M2 YoY', 'USDIDR', 'Inflation Rate YoY', 'BI Interest Rate', 'npl_ratio'])

# Scale data
scaler_target = Scaler()
scaler_cov = Scaler()
target_scaled = scaler_target.fit_transform(target_series)
cov_scaled = scaler_cov.fit_transform(covariates)

# RandomForest with past covariates (macroeconomic variables are past-known)
model = RandomForestModel(
    lags=12,                      # Use 12 months of target history
    lags_past_covariates=12,      # Use 12 months of covariate history
    output_chunk_length=1,        # Predict 1 step ahead
    n_estimators=200,
    random_state=42
)

# Train/test split
train_target, test_target = target_scaled.split_after(0.8)
train_cov, test_cov = cov_scaled.split_after(0.8)

# Fit and predict
model.fit(train_target, past_covariates=train_cov)
forecast = model.predict(n=len(test_target), past_covariates=cov_scaled)

# Inverse transform predictions
forecast_original = scaler_target.inverse_transform(forecast)
```

### Model Evaluation
```python
from darts.metrics import mape, rmse, mae

# Calculate metrics
print(f"MAPE: {mape(test_target, forecast):.2f}%")
print(f"RMSE: {rmse(test_target, forecast):.4f}")
print(f"MAE: {mae(test_target, forecast):.4f}")

# Backtest for rolling window evaluation
backtest_results = model.backtest(
    series=target_scaled,
    past_covariates=cov_scaled,
    start=0.7,           # Start validation at 70% of data
    forecast_horizon=1,
    stride=1,
    retrain=True,
    metric=mape
)
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

### Visualization Style
```python
# Thesis-consistent color palette
COLORS = {
    'IHSG': '#1f77b4', 'Inflation': '#d62728', 'BI_Rate': '#ff7f0e',
    'Credit': '#2ca02c', 'Money_Supply': '#17becf', 'LEI': '#62447e',
    'USD_IDR': '#8c564b', 'NPL': '#e377c2', 'Net_Buy': '#2ca02c'
}
# Always add: add_footnote(ax, "Source: Author's calculation, 2025")
```

### Date Alignment Pattern
When correlating multiple time series, always align dates first:
```python
common_dates = set(df1['Date']).intersection(set(df2['Date']))
df1_aligned = df1[df1['Date'].isin(common_dates)].sort_values('Date')
```

## Data Period
- Primary analysis period: **2015-01-01 to 2025-01-01** (10 years)
- Monthly aggregation for macroeconomic analysis
- Daily data for stock/commodity correlation

## Notebooks Workflow
1. Download data via `yfinance` → Save to CSV
2. Clean Indonesian numeric formats
3. Align dates across datasets
4. Standardize/normalize for comparison
5. Correlation analysis with `scipy.stats.pearsonr`
6. Visualization with consistent thesis styling
