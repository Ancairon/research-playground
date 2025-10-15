#!/usr/bin/env python3
"""
Regular scoring script that runs all models with full output and timing.
Includes all features from the silent version but with verbose output.
"""

from extra_trees_minimal import main as etmain
from forecasting.xgboost.xgboost_whole import main as xgbmain
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models-old'))
from arima_model import main as arimamain
from hw_model import main as hwmain
from multi_rf_model import main as multirf_main
import pandas as pd
import time

print("Starting model comparison...")
start_time = time.time()

# Common parameters
csv_file = '5000secs_pattern_rpi.csv'
common_params = {
    'csv': csv_file,
    'ip': None,
    'context': None,
    'dimension': None,
    'date_col': 'timestamp',
    'target_col': "value",
    'horizon': 10,
    'random_state': 42,
    'window': 1000,
    'retrain_threshold': 5,
    'max_errors': 1,
    'quiet': False  # Key parameter for full output
}

# Model-specific parameters
et_params = {**common_params, 'test_size': 0.3}
xgb_params = {**common_params}  # XGBoost doesn't use test_size
arima_params = {**common_params, 'test_size': 0.3}
hw_params = {**common_params, 'test_size': 0.3}
multirf_params = {**common_params, 'test_size': 0.3}

print("Running ExtraTrees model...")
et_start = time.time()
et_result = etmain(
    max_lag=10,
    n_estimators=222,
    **et_params
)
et_time = time.time() - et_start
et_result['model'] = 'ExtraTrees'
print(f"✓ ExtraTrees completed in {et_time:.1f}s")

print("\nRunning XGBoost model...")
xgb_start = time.time()
xgb_result = xgbmain(**xgb_params)
xgb_time = time.time() - xgb_start
xgb_result['model'] = 'XGBoost'
print(f"✓ XGBoost completed in {xgb_time:.1f}s")

print("\nRunning ARIMA model...")
arima_start = time.time()
arima_result = arimamain(**arima_params)
arima_time = time.time() - arima_start
arima_result['model'] = 'ARIMA'
print(f"✓ ARIMA completed in {arima_time:.1f}s")

print("\nRunning Holt-Winters model...")
hw_start = time.time()
hw_result = hwmain(**hw_params)
hw_time = time.time() - hw_start
hw_result['model'] = 'Holt-Winters'
print(f"✓ Holt-Winters completed in {hw_time:.1f}s")

print("\nRunning Multi-RF model...")
multirf_start = time.time()
multirf_result = multirf_main(
    lag=10,
    **multirf_params
)
multirf_time = time.time() - multirf_start
multirf_result['model'] = 'Multi-RF'
print(f"✓ Multi-RF completed in {multirf_time:.1f}s")

# Combine results
score = pd.DataFrame([et_result, xgb_result, arima_result, hw_result, multirf_result])

# Round numerical columns to 3 decimal places for cleaner output
numerical_columns = ['mean_avg_err', 'max_avg_err', 'min_avg_err', 'p80_avg_err', 
                    'p95_avg_err', 'p99_avg_err', 'mbe', 'pbias_pct', 
                    'avg_training_time', 'avg_inference_time']
for col in numerical_columns:
    if col in score.columns:
        score[col] = score[col].round(3)

score.to_csv('results.csv', index=False)

end_time = time.time()
total_time = end_time - start_time

print(f"\nModel comparison completed in {total_time:.1f} seconds")
print(f"Results saved to 'results.csv'")

print("\nModel Timing Summary:")
print("=" * 50)
print(f"ExtraTrees:    {et_time:.1f}s")
print(f"XGBoost:       {xgb_time:.1f}s") 
print(f"ARIMA:         {arima_time:.1f}s")
print(f"Holt-Winters:  {hw_time:.1f}s")
print(f"Multi-RF:      {multirf_time:.1f}s")
print(f"Total:         {total_time:.1f}s")
print("=" * 50)

print("\nFinal Results Summary:")
print("=" * 60)
# Display with rounded values for readability
display_cols = ['model', 'mean_avg_err', 'max_avg_err', 'min_avg_err', 'avg_training_time', 'avg_inference_time']
print(score[display_cols].to_string(index=False, float_format='%.3f'))
print("=" * 60)