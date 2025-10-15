#!/usr/bin/env python3
"""
Minimal, cleaned silent scoring orchestrator.

This version removes previous parallel/benchmarking remnants and keeps a
serial execution flow. It runs the models in cheapest-first order and
saves the combined `results_silent.csv` after each model completes so
progress is visible.

Behavior notes:
- Does not change model internals or their retraining logic.
- Imports model modules lazily (inside the run block) to avoid any module
  level side-effects during startup.
"""

from __future__ import annotations

import os
import sys
import time
import pandas as pd
import inspect

# Ensure models-old is on the path for legacy models
sys.path.append(os.path.join(os.path.dirname(__file__), 'models-old'))

print("Starting silent model comparison...")
start_time = time.time()

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--dry-run', action='store_true', help='Validate imports and parameter passing without running training')
args, _ = parser.parse_known_args()
DRY_RUN = bool(args.dry_run)

# Common parameters (these match the original defaults used by the models)
csv_file = '2400secs_pattern.csv'
common_params = {
    'csv': csv_file,
    'ip': None,
    'context': None,
    'dimension': None,
    'date_col': 'timestamp',
    'target_col': 'value',
    'horizon': 10,
    'random_state': 42,
    'window': 100,
    'test_size': 0.3,
    'retrain_scale': 0.05,
    'retrain_min': 1.0,
    'retrain_threshold': None,
    'max_errors': 1,
    'quiet': True,
}

# Compute a conservative dynamic retrain threshold using recent values (best-effort)
def compute_dynamic_retrain(params: dict) -> float:
    try:
        df = pd.read_csv(params['csv'])
        date_col = params.get('date_col', 'timestamp')
        if date_col in df.columns:
            # try to normalize to datetimes
            try:
                if pd.api.types.is_numeric_dtype(df[date_col].dtype):
                    df[date_col] = pd.to_datetime(df[date_col], unit='s')
                else:
                    df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            except Exception:
                # best-effort: ignore if conversion fails
                pass
        feed = df[[params['target_col']]].rename(columns={params['target_col']: 'value'})
        recent_len = min(params.get('window', 50), len(feed))
        recent_vals = feed['value'].iloc[-recent_len:]
        base_mean = float(recent_vals.mean()) if len(recent_vals) else 0.0
        scale = float(params.get('retrain_scale', 0.05))
        floor = float(params.get('retrain_min', 1.0))
        return max(floor, scale * abs(base_mean))
    except Exception:
        return float(params.get('retrain_min', 1.0))

dynamic_retrain = compute_dynamic_retrain(common_params)

# Per-model params
et_params = {**common_params, 'test_size': 0.3, 'retrain_threshold': dynamic_retrain}
xgb_params = {**common_params, 'retrain_threshold': dynamic_retrain}
arima_params = {**common_params, 'test_size': 0.3, 'retrain_threshold': dynamic_retrain}
hw_params = {**common_params, 'test_size': 0.3, 'retrain_threshold': dynamic_retrain}
multirf_params = {**common_params, 'test_size': 0.3, 'retrain_threshold': dynamic_retrain}

# collect results and write after each model
results_list = []

# --- Non-invasive CSV caching -------------------------------------------------
# Read the CSV once and monkeypatch pandas.read_csv so subsequent reads for the
# same path return a copy of the cached DataFrame. This is safe because we only
# alter pandas at runtime for the duration of this script and restore it later.
_orig_read_csv = None
_cached_df = None
_cached_path = common_params.get('csv')
try:
    _orig_read_csv = pd.read_csv
    if _cached_path and os.path.exists(_cached_path):
        # read once using the original function
        _cached_df = _orig_read_csv(_cached_path)

    def _patched_read_csv(path, *args, **kwargs):
        # If the caller requests the cached path, return a copy of the cached df
        try:
            if isinstance(path, str) and _cached_df is not None and os.path.abspath(path) == os.path.abspath(_cached_path):
                return _cached_df.copy()
        except Exception as exc:
            # fall back to original reader if something unexpected occurs
            # keep the exception for debugging if needed
            _ = exc
        return _orig_read_csv(path, *args, **kwargs)

    # apply the patch to the pandas module object so other imports see it
    pd.read_csv = _patched_read_csv
    # also ensure the sys.modules entry points to the same patched module
    if 'pandas' in sys.modules:
        sys.modules['pandas'].read_csv = _patched_read_csv
except Exception as exc:
    # on any error, leave pandas untouched; models will fall back to normal reads
    _orig_read_csv = None
    _cached_df = None
# -----------------------------------------------------------------------------

def _save_partial_results(results: list[dict]) -> None:
    df = pd.DataFrame(results)
    if 'model' in df.columns:
        cols = ['model'] + [c for c in df.columns if c != 'model']
        df = df[cols]
    # round a set of expected numeric columns if present
    numerical_columns = [
        'mean_avg_err', 'max_avg_err', 'min_avg_err', 'p80_avg_err',
        'p95_avg_err', 'p99_avg_err', 'mbe', 'pbias_pct',
        'avg_training_time', 'avg_inference_time',
    ]
    for col in numerical_columns:
        if col in df.columns:
            df[col] = df[col].round(3)
    df.to_csv('results_silent.csv', index=False)


try:
    # 5) XGBoost (heavier)
    print("Running XGBoost model (last, heavier)...")
    t0 = time.time()
    try:
        from forecasting.xgboost.xgboost_whole import main as xgbmain
        xgb_sig = inspect.signature(xgbmain)
        xgb_kwargs = {k: v for k, v in xgb_params.items() if k in xgb_sig.parameters}
        # ensure dynamic_retrain and retrain params are present if supported
        for k in ('dynamic_retrain', 'retrain_scale', 'retrain_min', 'retrain_use_mad', 'retrain_consec', 'retrain_cooldown'):
            if k in xgb_sig.parameters and k in common_params:
                xgb_kwargs[k] = common_params[k]
        if DRY_RUN:
            print('DRY RUN: XGBoost will be called with:', xgb_kwargs)
            xgb_result = {'model': 'XGBoost', 'note': 'dry-run'}
        else:
            xgb_result = xgbmain(**xgb_kwargs)
        xgb_time = time.time() - t0
        xgb_result['model'] = 'XGBoost'
        print(f"✓ XGBoost completed in {xgb_time:.1f}s")
    except ValueError as ve:
        # Common PyCaret error when dataset is too small for requested folds
        err_msg = str(ve)
        print(f"XGBoost skipped: {err_msg}")
        xgb_time = time.time() - t0
        xgb_result = {'model': 'XGBoost', 'error': err_msg}
    except Exception as e:
        err_msg = str(e)
        print(f"XGBoost failed: {err_msg}")
        xgb_time = time.time() - t0
        xgb_result = {'model': 'XGBoost', 'error': err_msg}

    results_list.append(xgb_result)
    pd.DataFrame([xgb_result]).to_csv('results_xgboost.csv', index=False)
    _save_partial_results(results_list)

    # 1) Holt-Winters (cheap)
    def _prepare_kwargs_for(fn, params: dict, extra: dict | None = None) -> dict:
        """Return a filtered kwargs dict suitable for calling fn.

        If the function accepts **kwargs, return params merged with extra.
        Otherwise only include parameter names that appear in the function signature.
        """
        sig = inspect.signature(fn)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        out = {}
        if accepts_var_kw:
            out.update(params)
        else:
            for k, v in params.items():
                if k in sig.parameters:
                    out[k] = v
        if extra:
            out.update(extra)
        return out

    print("Running Holt-Winters model (cheap)...")
    t0 = time.time()
    from hw_model import main as hwmain
    # Explicitly build hw kwargs from allowed signature keys
    hw_sig = inspect.signature(hwmain)
    hw_kwargs = {k: v for k, v in hw_params.items() if k in hw_sig.parameters}
    if DRY_RUN:
        print('DRY RUN: Holt-Winters will be called with:', hw_kwargs)
        hw_result = {'model': 'Holt-Winters', 'note': 'dry-run'}
    else:
        hw_result = hwmain(**hw_kwargs)
    hw_time = time.time() - t0
    hw_result['model'] = 'Holt-Winters'
    results_list.append(hw_result)
    pd.DataFrame([hw_result]).to_csv('results_hw.csv', index=False)
    _save_partial_results(results_list)
    print(f"✓ Holt-Winters completed in {hw_time:.1f}s")

    # 2) ARIMA (cheap)
    print("Running ARIMA model (cheap)...")
    t0 = time.time()
    from arima_model import main as arimamain
    arima_sig = inspect.signature(arimamain)
    arima_kwargs = {k: v for k, v in arima_params.items() if k in arima_sig.parameters}
    if DRY_RUN:
        print('DRY RUN: ARIMA will be called with:', arima_kwargs)
        arima_result = {'model': 'ARIMA', 'note': 'dry-run'}
    else:
        arima_result = arimamain(**arima_kwargs)
    arima_time = time.time() - t0
    arima_result['model'] = 'ARIMA'
    results_list.append(arima_result)
    pd.DataFrame([arima_result]).to_csv('results_arima.csv', index=False)
    _save_partial_results(results_list)
    print(f"✓ ARIMA completed in {arima_time:.1f}s")

    # 3) ExtraTrees
    print("Running ExtraTrees model...")
    t0 = time.time()
    from extra_trees_minimal import main as etmain
    et_sig = inspect.signature(etmain)
    et_kwargs = {k: v for k, v in et_params.items() if k in et_sig.parameters}
    # add explicit extras expected by ExtraTrees
    et_kwargs.update({'max_lag': 10, 'n_estimators': 222})
    if DRY_RUN:
        print('DRY RUN: ExtraTrees will be called with:', et_kwargs)
        et_result = {'model': 'ExtraTrees', 'note': 'dry-run'}
    else:
        et_result = etmain(**et_kwargs)
    et_time = time.time() - t0
    et_result['model'] = 'ExtraTrees'
    results_list.append(et_result)
    pd.DataFrame([et_result]).to_csv('results_extratrees.csv', index=False)
    _save_partial_results(results_list)
    print(f"✓ ExtraTrees completed in {et_time:.1f}s")

    # 4) Multi-RF
    print("Running Multi-RF model...")
    t0 = time.time()
    from multi_rf_model import main as multirf_main
    mr_sig = inspect.signature(multirf_main)
    multirf_kwargs = {k: v for k, v in multirf_params.items() if k in mr_sig.parameters}
    multirf_kwargs.update({'lag': 10})
    if DRY_RUN:
        print('DRY RUN: Multi-RF will be called with:', multirf_kwargs)
        multirf_result = {'model': 'Multi-RF', 'note': 'dry-run'}
    else:
        multirf_result = multirf_main(**multirf_kwargs)
    multirf_time = time.time() - t0
    multirf_result['model'] = 'Multi-RF'
    results_list.append(multirf_result)
    pd.DataFrame([multirf_result]).to_csv('results_multirf.csv', index=False)
    _save_partial_results(results_list)
    print(f"✓ Multi-RF completed in {multirf_time:.1f}s")

    

except Exception as exc:
    print(f"Error running models: {exc}")
    # write whatever we have so far
    try:
        _save_partial_results(results_list)
    except Exception:
        pass
    raise

# Final combined CSV is already updated incrementally; print summary
end_time = time.time()
total_time = end_time - start_time
print(f"\nSilent comparison completed in {total_time:.1f} seconds")
print(f"Results saved to 'results_silent.csv'")

print("\nModel Timing Summary:")
print("=" * 50)
for name, t in [
    ("Holt-Winters", locals().get('hw_time', float('nan'))),
    ("ARIMA", locals().get('arima_time', float('nan'))),
    ("ExtraTrees", locals().get('et_time', float('nan'))),
    ("Multi-RF", locals().get('multirf_time', float('nan'))),
    ("XGBoost", locals().get('xgb_time', float('nan'))),
]:
    print(f"{name:14s}: {t:.1f}s")
print(f"Total:         {total_time:.1f}s")
print("=" * 50)

print("\nFinal Results Summary:")
print("=" * 60)

# restore pandas.read_csv if we patched it earlier
try:
    if _orig_read_csv is not None:
        pd.read_csv = _orig_read_csv
except Exception:
    pass
try:
    score = pd.DataFrame(results_list)
    display_cols = ['model', 'mean_avg_err', 'max_avg_err', 'min_avg_err', 'avg_training_time', 'avg_inference_time']
    # print only columns that exist
    display_cols = [c for c in display_cols if c in score.columns]
    print(score[display_cols].to_string(index=False, float_format='%.3f'))
except Exception:
    print(results_list)
print("=" * 60)
