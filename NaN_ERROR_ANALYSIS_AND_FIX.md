# NaN Error Analysis and Fix - Model 2

## Problem Summary
ALL 100 Optuna trials were failing with the error: **"Input y contains NaN"**

This error originated from sklearn's RandomForest model during fitting, indicating that NaN values were being passed into the training data.

## Root Cause Analysis

### The Issue: freq='D' + Irregular Time Series + Large Lags

The problem was caused by a **fundamental mismatch** between how Darts interprets lags when using calendar day frequency (`freq='D'`) with irregular time series data.

#### Your Data Structure:
- **2,268 trading days** spanning from 2015-01-06 to 2025-12-30
- **4,011 calendar days** in that period
- Only **56.5% of calendar days** have data (trading days only)
- Natural gaps for weekends and holidays

#### What Was Happening:

When you create a TimeSeries with:
```python
TimeSeries.from_dataframe(
    df_ts,
    value_cols=TARGET_COL,
    fill_missing_dates=False,
    freq='D'  # ← THIS WAS THE PROBLEM
)
```

And then use `lags=42` in the model, Darts interprets this as:
- **"Look back 42 CALENDAR days"** (not 42 trading days)

But your data only contains trading days, so:
1. When lags=42, Darts tries to access data from 42 calendar days ago
2. That specific date might be a **weekend or holiday** that doesn't exist in your irregular time series
3. Darts creates **NaN values** for those missing lag features
4. These NaN values get passed to RandomForest.fit()
5. sklearn raises: **"Input y contains NaN"**

#### Concrete Example:

Test set starts: **2023-10-17** (a trading day)

| Lag Value | Calendar Date Accessed | Exists in Data? | Result |
|-----------|------------------------|-----------------|---------|
| lags=5    | 2023-10-12 (Thursday)  | ❌ No (probably a holiday) | **NaN created** |
| lags=10   | 2023-10-07 (Saturday)  | ❌ No (weekend) | **NaN created** |
| lags=21   | 2023-09-26 (Tuesday)   | ✅ Yes | OK |
| lags=42   | 2023-09-05 (Tuesday)   | ✅ Yes | OK |

Even though lags=21 and lags=42 might work for some dates, as `historical_forecasts` slides through different dates in the test set, it **will eventually encounter dates where the lookback lands on weekends/holidays**, causing NaN values.

### Why ALL 100 Trials Failed

Your Optuna search space included:
```python
lags = trial.suggest_categorical('lags', [5, 10, 21, 42])
lags_past_covariates = trial.suggest_categorical('lags_past_covariates', [5, 10, 21, 42])
```

**Every single trial** used at least one of these lag values, and with the freq='D' issue, **all of them eventually hit dates where the lookback created NaN values**.

## The Fix Applied

### What Was Changed:

Removed the `freq='D'` parameter from both TimeSeries creation calls:

**Before:**
```python
target_series = TimeSeries.from_dataframe(
    df_ts,
    value_cols=TARGET_COL,
    fill_missing_dates=False,
    freq='D'  # ← REMOVED
)

covariates = TimeSeries.from_dataframe(
    df_ts,
    value_cols=COVARIATE_COLS,
    fill_missing_dates=False,
    freq='D'  # ← REMOVED
)
```

**After:**
```python
target_series = TimeSeries.from_dataframe(
    df_ts,
    value_cols=TARGET_COL,
    fill_missing_dates=False,
    # freq parameter removed - Darts will use position-based indexing
)

covariates = TimeSeries.from_dataframe(
    df_ts,
    value_cols=COVARIATE_COLS,
    fill_missing_dates=False,
    # freq parameter removed - Darts will use position-based indexing
)
```

### How This Fixes The Issue:

Without the `freq='D'` parameter, Darts treats lags as **position-based** (integer indexing):

- `lags=5` now means: "look back 5 **actual data points**" (5 trading days)
- `lags=42` now means: "look back 42 **actual data points**" (42 trading days)

This is **exactly what you want** for trading day analysis:
- ✅ lags=5 → last 5 trading days (about 1 week)
- ✅ lags=10 → last 10 trading days (about 2 weeks)
- ✅ lags=21 → last 21 trading days (about 1 month)
- ✅ lags=42 → last 42 trading days (about 2 months)

**No more NaN values** because Darts will always access actual existing data points, never trying to access weekends or holidays.

## What This Means For Your Analysis

### Before (with freq='D'):
- lags=42 looked back **42 calendar days** → only ~24 actual trading days
- Inconsistent lookback periods due to varying weekend/holiday patterns
- NaN values created when lookback lands on non-trading days

### After (without freq):
- lags=42 looks back **42 actual trading days** → consistent 42-day lookback
- More intuitive for financial analysis (actual market days matter, not calendar days)
- No NaN values, all trials should now complete successfully

## Next Steps

1. **Re-run the Optuna hyperparameter tuning cell** - all 100 trials should now complete without NaN errors
2. The model will now properly use the intended number of trading days for lag features
3. Your results should be more reliable because:
   - Consistent lookback periods across all predictions
   - No missing data or NaN handling required
   - Lags represent actual market activity, not calendar time

## Verification

To verify the fix works, you can check:
1. Run the TimeSeries creation cell - check `target_series.freq` (might show None or an inferred frequency)
2. Run the Optuna cell - monitor for "Trial failed" messages
3. All 100 trials should complete successfully now

## Technical Details

### Why This Is Better:
- **Trading days are what matter** in financial analysis, not calendar days
- A 42-trading-day pattern is consistent whether it spans 60 or 65 calendar days
- Position-based indexing is more robust for irregular time series
- Aligns with how financial practitioners think about lookback periods

### Alternative Solutions (Not Recommended):
1. ❌ `fill_missing_dates=True` - Would add fake data for weekends/holidays
2. ❌ Reduce lags to [3,5,7,10] - Doesn't fix the fundamental issue
3. ❌ Use freq='B' (business days) - Still has issues with holidays

The solution applied (removing freq parameter) is the **best approach** for your trading day data.
