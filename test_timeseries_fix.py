#!/usr/bin/env python3
"""
Test script to verify Model 2 TimeSeries creation fix
Run this locally to check if the freq='D' + fill_missing_dates=False works
"""

import pandas as pd
from darts import TimeSeries

print("="*70)
print("MODEL 2 TIMESERIES CREATION TEST")
print("="*70)

# Load merged data
df = pd.read_csv('model2_merged_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"\n✓ Loaded data: {len(df)} rows")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

# Set index
df_ts = df.set_index('Date')

# Check for date gaps
date_diffs = df_ts.index.to_series().diff()
print(f"\n✓ Date gap analysis:")
print(f"  Min gap: {date_diffs.min()}")
print(f"  Max gap: {date_diffs.max()}")
unique_gaps = sorted(date_diffs.dropna().unique())
print(f"  Unique gaps (first 10): {unique_gaps[:10]}")

# Test TimeSeries creation with the fix
print("\n" + "="*70)
print("TESTING: freq='D' + fill_missing_dates=False")
print("="*70)

try:
    target_series = TimeSeries.from_dataframe(
        df_ts,
        value_cols='IHSG',
        fill_missing_dates=False,
        freq='D'
    )

    covariates = TimeSeries.from_dataframe(
        df_ts,
        value_cols=['STI', 'Coal', 'Copper', 'Silver', 'Tin', 'Nickel'],
        fill_missing_dates=False,
        freq='D'
    )

    print("\n✅ SUCCESS! TimeSeries created without errors")
    print(f"\nTarget Series (IHSG):")
    print(f"  - Start: {target_series.start_time()}")
    print(f"  - End: {target_series.end_time()}")
    print(f"  - Length: {len(target_series)} trading days")
    print(f"  - Frequency: {target_series.freq}")

    print(f"\nCovariates:")
    print(f"  - Components: {covariates.components.tolist()}")
    print(f"  - Length: {len(covariates)} trading days")

    print("\n" + "="*70)
    print("✅ FIX VERIFIED: No ValueError, TimeSeries created successfully!")
    print("="*70)

except ValueError as e:
    print(f"\n❌ FAILED: {e}")
    print("\nThe fix didn't work. Please check:")
    print("  1. Is 'darts' installed? (pip install darts)")
    print("  2. Is the CSV file present?")
    print("  3. Are there any data issues?")

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("If successful, you can now run the full model-2.ipynb notebook!")
print("="*70)
