#!/usr/bin/env python3
"""
Downsample the 30-day trend data from 1-second to 1-minute resolution.
This reduces the dataset from 2.59M rows to 43.2K rows.
"""

import pandas as pd

# Read the original data
df = pd.read_csv('30day_upward_trend.csv')
print(f"Original data: {len(df)} rows")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Resample to 1-minute intervals (take mean of each minute)
df_resampled = df.set_index('timestamp').resample('1min').mean().reset_index()

# Round values
df_resampled['value'] = df_resampled['value'].round(3)

# Save
output_file = '30day_upward_trend_1min.csv'
df_resampled.to_csv(output_file, index=False)

print(f"Resampled data: {len(df_resampled)} rows")
print(f"Saved to: {output_file}")
print(f"\nReduction: {len(df) / len(df_resampled):.1f}x smaller")
print(f"\nNow with horizon=1440 (1 day), you'll only need 1440 predictions")
print(f"instead of 86,400 predictions!")
