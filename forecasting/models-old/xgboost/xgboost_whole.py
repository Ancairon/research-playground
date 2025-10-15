# xgb_unified_forecaster.py
import argparse
import time
from collections import deque

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
import os

from pycaret.time_series import TSForecastingExperiment
import io
import contextlib
import logging
from typing import Optional

import xgboost
# patch __version__ if missing
if not hasattr(xgboost, "__version__"):
    try:
        import importlib.metadata as importlib_metadata
        xgboost.__version__ = importlib_metadata.version("xgboost")
    except Exception:
        xgboost.__version__ = "1.6.0"


def getDataFromAPI(ip, context, dimension, seconds_back):
    url = (
        f"http://{ip}:19999/api/v3/data?"
        f"contexts={context}&dimensions={dimension}&"
        f"after=-{seconds_back}&before=0&points={seconds_back}&"
        f"group=average&format=json&options=seconds,jsonwrap"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    rows = resp.json()['result']['data']
    df = pd.DataFrame(rows, columns=['timestamp', 'value'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.asfreq('s').ffill()
    return df


def _safe_mean(x): return float(np.mean(x)) if len(x) else float('nan')
def _safe_max(x): return float(np.max(x)) if len(x) else float('nan')
def _safe_min(x): return float(np.min(x)) if len(x) else float('nan')


def _safe_pct(x, q): return float(
    np.percentile(x, q)) if len(x) else float('nan')


def compute_threshold_from_errors(errors, retrain_scale, retrain_min, use_mad=True):
    """Compute retrain threshold from a sequence of absolute errors.

    Returns a float threshold (same units as errors).
    Uses MAD -> sigma (1.4826 * MAD) when use_mad is True, otherwise uses std.
    """
    arr = np.asarray(errors, dtype=float)
    if arr.size == 0:
        return float(retrain_min)
    if use_mad:
        med = float(np.nanmedian(arr))
        mad = float(np.nanmedian(np.abs(arr - med)))
        sigma = 1.4826 * mad
    else:
        sigma = float(np.nanstd(arr))
    thresh = max(float(retrain_min), float(retrain_scale) * abs(sigma))
    return thresh


def _build_xgb_pipeline(series: pd.Series, horizon: int, session_id: int):
    y = series.copy()
    y.index = pd.PeriodIndex(y.index, freq='S')
    exp = TSForecastingExperiment()
    # PyCaret prints verbose experiment info during setup/create/finalize.
    # Suppress pycaret logger and stdout/stderr while building the pipeline.
    class _suppress_pycaret:
        def __init__(self, logger_name: Optional[str] = 'pycaret'):
            self.logger_name = logger_name
            self._old_level = None
            self._stream = io.StringIO()
        def __enter__(self):
            logger = logging.getLogger(self.logger_name)
            self._old_level = logger.level
            logger.setLevel(logging.ERROR)
            self._out = contextlib.redirect_stdout(self._stream)
            self._err = contextlib.redirect_stderr(self._stream)
            self._out.__enter__()
            self._err.__enter__()
        def __exit__(self, exc_type, exc, tb):
            self._err.__exit__(exc_type, exc, tb)
            self._out.__exit__(exc_type, exc, tb)
            logging.getLogger(self.logger_name).setLevel(self._old_level)

    with _suppress_pycaret():
        exp.setup(
            data=y,
            fh=horizon,
            seasonal_period=60,
            session_id=session_id,
            numeric_imputation_target="ffill"
        )
        xgb_pipe = exp.create_model('xgboost_cds_dt')
        final_xgb = exp.finalize_model(xgb_pipe)
    return exp, final_xgb


def main(
        csv=None,                      # Path to CSV file for offline simulation (None = live Netdata)
        ip="localhost",                # Netdata server IP/hostname
        context="system.net",          # Netdata context (metric category, e.g., system.cpu, system.net)
        dimension="received",          # Netdata dimension (specific metric, e.g., user, received, sent)
        horizon=5,                     # Number of steps ahead to forecast
        random_state=42,               # Random seed for reproducibility
        window=300,                    # Training window size (number of historical points)
        test_size=0.3,                 # Test split ratio (only used for CSV mode)
        ylim_min=0.0,                  # Chart Y-axis minimum (ignored if ylim_auto=True)
        ylim_max=100.0,                # Chart Y-axis maximum (ignored if ylim_auto=True)
        ylim_auto=True,                # Auto-scale Y-axis based on data range
        target_col="value",            # CSV column name for target values
        date_col="timestamp",          # CSV column name for timestamps
        dynamic_retrain=True,          # Enable dynamic threshold computation from recent errors
        retrain_scale=3.0,             # Multiplier for MAD/mean to compute dynamic threshold (3.0 for MAPE)
        retrain_min=50.0,              # Minimum threshold as percentage (e.g., 20.0 = 20% MAPE)
        retrain_use_mad=True,          # Use MAD (robust, outlier-resistant) instead of std for threshold
        retrain_consec=2,              # Number of consecutive threshold violations to trigger retrain
        retrain_cooldown=5,            # Minimum steps between retrains (cooldown period)
        quiet=False,                   # Suppress console output (prints)
        print_min_validations: int = 3,  # Minimum successful horizon validations before printing
        prediction_smoothing: int = 5, # Number of recent predictions to average (1=no smoothing, 3=default)
        prediction_interval: float = 2.0,  # Seconds between predictions (1.0=every second, 2.0=every 2 seconds)
        live_server=True,              # Enable live demo server (browser visualization)
        live_server_port=5000          # Port for live demo server
):
    """
    XGBoost live forecasting with dynamic MAPE-based retraining.
    
    Key Parameter Groups:
    ---------------------
    Data Source:    csv, ip, context, dimension, window
    Model Config:   horizon, random_state, test_size, prediction_smoothing, prediction_interval
    Retraining:     dynamic_retrain, retrain_scale, retrain_min, retrain_use_mad, 
                    retrain_consec, retrain_cooldown
    Visualization:  ylim_auto, ylim_min, ylim_max, live_server, live_server_port
    
    MAD (Median Absolute Deviation) Explanation:
    --------------------------------------------
    When retrain_use_mad=True, the threshold is computed using MAD instead of std:
    - MAD = median(|errors - median(errors)|)
    - sigma = 1.4826 * MAD (converts MAD to equivalent std)
    - threshold = max(retrain_min, retrain_scale * sigma)
    - More robust to outliers than std (standard deviation)
    - Prevents one bad prediction from inflating threshold
    - Recommended for production use
    
    IMPORTANT - retrain_scale values:
    - For MAPE (percentage errors): use 2.0-5.0 (default 3.0)
      * Example: MAD=15%, sigma=22%, threshold=3.0*22%=66%
    - For absolute errors (KB/s, etc): use 10.0-20.0
      * Example: MAD=10 KB/s, sigma=15, threshold=10.0*15=150 KB/s
    - Too high → retraining never triggers (threshold=500%+)
    - Too low → retraining triggers too often (unstable model)
    
    WARNING: When retrain_use_mad=False, std of percentage errors can be very large
    (e.g., std of [10%, 15%, 20%] ≈ 5%, so threshold = 20.0 * 5% = 100%+), making
    retraining nearly impossible. Always use MAD unless you have a specific reason.
    
    Prediction Smoothing:
    --------------------
    When prediction_smoothing > 1, predictions are averaged across the last N forecasts:
    - Reduces flip-flopping between spike/dip predictions
    - Smooths out noise while maintaining responsiveness
    - Default: 3 (average last 3 predictions)
    - Set to 1 to disable smoothing (use raw predictions)
    
    Prediction Interval:
    -------------------
    Controls how often predictions are made:
    - prediction_interval=1.0: Predict every second (very responsive, but reactive)
    - prediction_interval=2.0: Predict every 2 seconds (balanced, recommended)
    - prediction_interval=5.0: Predict every 5 seconds (very stable, minimal CPU)
    
    Understanding Reactive vs Predictive Behavior:
    ----------------------------------------------
    XGBoost is fundamentally REACTIVE - it predicts based on recent patterns, not future events.
    
    For spiky/periodic patterns:
    - Need sufficient window to see multiple cycles (e.g., 15s spike cycle needs 300s+ window to see 20 cycles)
    - Larger window = better pattern recognition
    - More smoothing = less flip-flopping but slower response
    - Slower prediction_interval = more stable forecasts
    
    Pattern-specific recommendations:
    - Smooth patterns (sine, daily): window=120, smoothing=3, interval=1.0
    - Spiky patterns: window=300-600, smoothing=5-7, interval=2.0-3.0
    - Step changes: window=180-300, smoothing=3-5, interval=2.0
    - Trends: window=300+, smoothing=5, interval=2.0, aggressive retraining (retrain_scale=2.0)
    
    Recommended Settings:
    --------------------
    - Dynamic retraining (MAPE): retrain_scale=2.0-5.0, retrain_min=20.0-50.0%, retrain_use_mad=True
    - Dynamic retraining (absolute): retrain_scale=10.0-20.0, retrain_min=50-100 (units), retrain_use_mad=True
    - Fast retraining: retrain_consec=2, retrain_cooldown=5-10
    - Smooth patterns: window=120-180, prediction_smoothing=3, prediction_interval=1.0
    - Spiky patterns: window=300-600, prediction_smoothing=5-7, prediction_interval=2.0-3.0
    - Auto Y-axis: ylim_auto=True (ignores ylim_min/max)
    """
    # Minimum number of successful horizon validations required before printing.
    # This is a regular function parameter (default 3) to match the rest of the
    # main() configuration options.
    
    # Start prediction server if live_server is enabled
    if live_server:
        import subprocess
        import threading
        import webbrowser
        
        # Build server command
        server_cmd = [
            'python3', 'prediction_server.py',
            '--port', str(live_server_port),
            '--open-demo',
            '--ip', str(ip or 'localhost'),
            '--context', str(context or 'system.cpu'),
            '--dimension', str(dimension or 'user')
        ]
        # Only pass y-axis limits if NOT using auto mode
        if not ylim_auto:
            if ylim_min is not None:
                server_cmd.extend(['--ymin', str(ylim_min)])
            if ylim_max is not None:
                server_cmd.extend(['--ymax', str(ylim_max)])
        
        # Start server in background
        print(f"[XGB] Starting prediction server on port {live_server_port}...")
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__)
        )

        time.sleep(2)
        
        print(f"[XGB] Prediction server started. Demo should open in browser.")
        print(f"[XGB] Live mode enabled - matplotlib plotting disabled (server handles visualization)")
        print(f"[XGB] XGBoost output will continue to print below...")
        print("")

    if csv:
        df_csv = pd.read_csv(csv)

        if np.issubdtype(df_csv[date_col].dtype, np.number):
            df_csv[date_col] = pd.to_datetime(df_csv[date_col], unit='s')
        else:
            df_csv[date_col] = pd.to_datetime(df_csv[date_col])
        df_csv.set_index(date_col, inplace=True)

        feed_df = df_csv[[target_col]].rename(columns={target_col: 'value'})
        window = min(window, len(feed_df))
        # Use the same train_size convention as other models so scoring is comparable
        train_size = int(window * (1 - test_size))
        init_df = feed_df.iloc[:train_size].copy()
        feed_ptr = window  # simulation pointer
    else:
        init_df = getDataFromAPI(ip, context, dimension, window)

    # Guard: ensure we have some initial training rows
    if init_df is None or len(init_df) == 0:
        print(f"[XGB] No initial data available for training.\n  csv={csv} ip={ip} context={context} dimension={dimension} window={window}")
        return {
            "mean_avg_err": float('nan'),
            "max_avg_err": float('nan'),
            "min_avg_err": float('nan'),
            "p80_avg_err": float('nan'),
            "p95_avg_err": float('nan'),
            "p99_avg_err": float('nan'),
            "mbe": float('nan'),
            "pbias_pct": float('nan'),
            "avg_training_time": float('nan'),
            "avg_inference_time": float('nan')
        }

    start_train = time.time()
    exp, pipeline = _build_xgb_pipeline(
        init_df['value'], horizon, random_state)
    initial_training_time = time.time() - start_train
    if not quiet:
        print(f"[XGB] Initial pipeline trained in {initial_training_time:.3f}s")

    # Only setup matplotlib if NOT in live_server mode
    # (live_server provides browser-based visualization instead)
    if not live_server:
        plt.ion()
        fig, ax = plt.subplots()
        # Initialize dynamic y-limits (they will only expand when new extremes appear)
        if ylim_auto:
            try:
                init_vals = init_df['value'].astype(float).values
                if len(init_vals) > 0 and np.isfinite(init_vals).any():
                    init_min = float(np.nanmin(init_vals))
                    init_max = float(np.nanmax(init_vals))
                    pad = (init_max - init_min) * 0.05 if init_max > init_min else 1.0
                    # Use data-driven limits for auto mode. If user explicitly set
                    # ylim_min/ylim_max (non-default), we still prefer the data range
                    # for autoscale; the CLI flags are treated as suggestions only.
                    current_ylim_min = init_min - pad
                    current_ylim_max = init_max + pad
                else:
                    current_ylim_min = ylim_min
                    current_ylim_max = ylim_max
            except Exception:
                current_ylim_min = ylim_min
                current_ylim_max = ylim_max
        else:
            current_ylim_min = ylim_min
            current_ylim_max = ylim_max

    # Error tracking for metrics and dynamic retraining
    errors = deque(maxlen=10)          # Small window for rolling MAPE average display
    recent_errors = deque(maxlen=max(50, retrain_consec * 5))  # Larger window for robust threshold computation
    result_avg_err = []                # List of window-averaged errors for final stats
    signed_errors = []                 # Pointwise signed errors y_true - y_pred
    y_true_list = []
    y_pred_list = []
    
    # Store predictions to validate later when actual values arrive
    # Each entry: (creation_step, last_actual_time, [pred_values])
    pending_validations = deque(maxlen=100)
    
    # Prediction smoothing: store recent raw predictions to average
    # Each entry is a list of prediction values [pred1, pred2, ..., predH]
    recent_predictions = deque(maxlen=max(1, int(prediction_smoothing)))
    
    holdback = 0
    # retrain control state
    consec_count = 0
    last_retrain_step = -9999
    step = 0

    # prepare retrain log CSV
    retrain_log_path = os.path.join(os.getcwd(), 'xgb_retrain_log.csv')
    write_header = not os.path.exists(retrain_log_path)
    try:
        retrain_log_fh = open(retrain_log_path, 'a', newline='')
        retrain_writer = csv.DictWriter(retrain_log_fh, fieldnames=[
            'step', 'timestamp', 'pred_timestamp', 'actual', 'forecast', 'abs_error', 'threshold',
            'consec_count', 'last_retrain_step', 'retrain_triggered'
        ])
        if write_header:
            retrain_writer.writeheader()
    except Exception:
        retrain_log_fh = None
        retrain_writer = None
    
    # Timing tracking
    training_times = [initial_training_time]
    inference_times = []

    try:
        while True:
            if csv:
                if feed_ptr > len(feed_df):
                    print("End of simulation feed.")
                    break
                live_df = feed_df.iloc[:feed_ptr].copy()
                feed_ptr += 1
            else:
                live_df = getDataFromAPI(ip, context, dimension, window)

            ts = live_df['value'].copy()
            ts.index = pd.PeriodIndex(ts.index, freq='S')

            # update history
            pipeline.update(ts)

            # forecast
            start_inf = time.time()
            fh = list(range(1, horizon + 1))
            fc = pipeline.predict(fh=fh)
            fc.index = fc.index.to_timestamp(freq='S')
            raw_preds = fc.values.tolist()
            inf_time = time.time() - start_inf
            inference_times.append(inf_time)
            
            # Apply prediction smoothing if enabled
            if raw_preds:
                raw_pred_values = [float(raw_preds[i]) for i in range(len(raw_preds))]
                recent_predictions.append(raw_pred_values)
                
                # Compute smoothed predictions (average of recent predictions)
                if len(recent_predictions) >= 1 and prediction_smoothing > 1:
                    # Average across the recent predictions for each horizon step
                    smoothed = []
                    for i in range(horizon):
                        values_at_step_i = [pred_list[i] for pred_list in recent_predictions if i < len(pred_list)]
                        if values_at_step_i:
                            smoothed.append(float(np.mean(values_at_step_i)))
                        else:
                            smoothed.append(raw_pred_values[i] if i < len(raw_pred_values) else 0.0)
                    preds = smoothed
                else:
                    # No smoothing or not enough predictions yet
                    preds = raw_pred_values
            else:
                preds = []
            
            # Send predictions to live demo server if enabled
            if 'live_server' in locals() and live_server:
                try:
                    requests.post(
                        f'http://localhost:{live_server_port}/predictions',
                        json={
                            'predictions': preds,
                            'context': context,
                            'dimension': dimension,
                            'timestamp': live_df.index[-1].isoformat()
                        },
                        timeout=0.5
                    )
                except Exception:
                    pass  # Silently continue if server unavailable

            # Store smoothed predictions for validation after horizon seconds
            if preds:
                try:
                    # Store: (creation_step, last_actual_timestamp, smoothed_prediction_values)
                    last_actual_ts = live_df.index[-1]
                    pred_values = [float(preds[i]) for i in range(len(preds))]
                    pending_validations.append((step, last_actual_ts, pred_values))
                except Exception:
                    pass  # Continue if prediction storage fails
            
            step += 1

            # Validate ONE pending prediction per step (oldest one that's ready)
            # This happens AFTER storing so we validate previous predictions
            validated_count = 0
            if pending_validations:
                creation_step, last_actual_ts, pred_values = pending_validations[0]
                
                # Check if enough time has passed (horizon seconds since prediction)
                steps_elapsed = step - creation_step
                
                if steps_elapsed >= horizon:
                    # Remove from queue - we'll validate it now
                    pending_validations.popleft()
                    
                    # Try to get the actual values that correspond to this prediction
                    horizon_errors = []
                    horizon_actuals = []
                    horizon_preds = []
                    
                    for i, pred_val in enumerate(pred_values):
                        # The i-th prediction was for (i+1) seconds after last_actual_ts
                        # Time elapsed since prediction was made: steps_elapsed seconds
                        # So the predicted time is now (steps_elapsed - (i+1)) seconds ago from current time
                        offset_from_now = steps_elapsed - (i + 1)
                        
                        # Look for actual value at this offset from the end of live_df
                        y_true_now = None
                        try:
                            # Get current timestamp
                            current_ts = live_df.index[-1]
                            target_ts = current_ts - pd.Timedelta(seconds=offset_from_now)
                            target_ts_floor = target_ts.floor('S')
                            
                            # Find matching timestamp in live_df
                            idx_secs = live_df.index.floor('S')
                            matches = live_df.loc[idx_secs == target_ts_floor]
                            if len(matches) > 0:
                                y_true_now = float(matches['value'].iloc[-1])
                        except Exception:
                            y_true_now = None
                        
                        if y_true_now is None:
                            continue
                        
                        # Calculate error for this prediction
                        err_abs = abs(y_true_now - float(pred_val))
                        err_relative = (err_abs / max(abs(y_true_now), 1e-6)) * 100.0
                        
                        horizon_errors.append(err_relative)
                        horizon_actuals.append(y_true_now)
                        horizon_preds.append(pred_val)
                    
                    # Only process if we validated the full horizon
                    if len(horizon_errors) == horizon:
                        validated_count += 1
                        
                        # Mean error across all horizon steps
                        mean_horizon_error = float(np.mean(horizon_errors))
                        
                        errors.append(mean_horizon_error)
                        recent_errors.append(mean_horizon_error)

                        # Compute threshold now so we can show the per-validation
                        # numbers that are used to decide retraining.
                        threshold = None
                        if dynamic_retrain:
                            try:
                                threshold = compute_threshold_from_errors(
                                    recent_errors, retrain_scale, retrain_min, retrain_use_mad
                                )
                            except Exception:
                                threshold = float(retrain_min)

                        # Print per-validation details so it's clear which validation
                        # caused the retrain decision (shows mean_horizon_error,
                        # computed threshold, and the current consec_count).
                        if not quiet:
                            try:
                                current_ts = live_df.index[-1]
                                ts_str = current_ts.strftime('%Y-%m-%d %H:%M:%S')
                            except Exception:
                                ts_str = ''
                            thr_str = f"{threshold:.2f}%" if threshold is not None else "N/A"
                            print(
                                f"[{ts_str}] VALIDATION step={step} mean_horizon_error={mean_horizon_error:.2f}% "
                                f"threshold={thr_str} (retrain_min={retrain_min})"
                            )
                        
                        # Store for final statistics
                        for actual, pred in zip(horizon_actuals, horizon_preds):
                            signed_errors.append(actual - pred)
                            y_true_list.append(actual)
                            y_pred_list.append(pred)
                        
                        # Store rolling average for final statistics (no print - VALIDATION print is sufficient)
                        if len(errors) >= int(print_min_validations):
                            avg_err = float(np.mean(errors))
                            result_avg_err.append(avg_err)
                        
                        # Update consecutive counter for retraining using the
                        # threshold we computed above.
                        if threshold is not None:
                            if mean_horizon_error > threshold:
                                consec_count += 1
                            else:
                                consec_count = 0
                            
                            # Check if we can retrain
                            can_retrain = (
                                consec_count >= retrain_consec and
                                (step - last_retrain_step) >= retrain_cooldown and
                                len(recent_errors) >= retrain_consec
                            )
                            
                            # Log this validation
                            if retrain_writer is not None:
                                try:
                                    mean_actual = float(np.mean(horizon_actuals))
                                    mean_pred = float(np.mean(horizon_preds))
                                    
                                    retrain_writer.writerow({
                                        'step': step,
                                        'timestamp': live_df.index[-1],
                                        'pred_timestamp': creation_step,
                                        'mean_actual': mean_actual,
                                        'mean_forecast': mean_pred,
                                        'mean_horizon_error_pct': float(mean_horizon_error),
                                        'threshold': float(threshold),
                                        'consec_count': int(consec_count),
                                        'last_retrain_step': int(last_retrain_step),
                                        'retrain_triggered': bool(can_retrain)
                                    })
                                    if retrain_log_fh:
                                        retrain_log_fh.flush()
                                except Exception:
                                    pass
                            
                            # Perform retrain if triggered
                            if can_retrain:
                                # perform retrain and update state
                                try:
                                    start_t = time.time()
                                    exp, pipeline = _build_xgb_pipeline(
                                        live_df['value'], horizon, random_state
                                    )
                                    retrain_time = time.time() - start_t
                                    training_times.append(retrain_time)
                                    last_retrain_step = step
                                    consec_count = 0
                                    if not quiet:
                                        print(f"[XGB] Retrained at step {step} (time={retrain_time:.3f}s), threshold={threshold:.1f}% relative error")
                                except Exception as e:
                                    if not quiet:
                                        print(f"[XGB] Retrain failed (continuing): {e}")
                                finally:
                                    errors.clear()
                                    recent_errors.clear()

            # Plotting only if not in live_server mode (browser handles viz)
            if not live_server:
                recent = ts.to_timestamp(freq='S').iloc[-10:]
                ax.clear()
                ax.plot(recent.index, recent.values, 'b-', label='Actual')
                ax.plot(fc.index,      fc.values,    'r--o',
                        label=f'Forecast ({horizon}s)')

                all_times = recent.index.union(fc.index)
                ax.set_xlim(all_times.min(), all_times.max())
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                fig.autofmt_xdate()

                if ylim_auto:
                    try:
                        actual_vals = recent.values.astype(float)
                        try:
                            pred_vals = np.array(fc.values, dtype=float).ravel()
                        except Exception:
                            pred_vals = np.array(list(fc.values), dtype=float).ravel()
                        all_vals = np.concatenate([actual_vals, pred_vals])
                        if np.isfinite(all_vals).any():
                            minv = float(np.nanmin(all_vals))
                            maxv = float(np.nanmax(all_vals))
                            if maxv > minv:
                                pad = (maxv - minv) * 0.05
                            else:
                                pad = 1.0
                            # expand only when new extremes exceed current limits
                            if minv - pad < current_ylim_min:
                                current_ylim_min = minv - pad
                            if maxv + pad > current_ylim_max:
                                current_ylim_max = maxv + pad
                            ax.set_ylim(current_ylim_min, current_ylim_max)
                        else:
                            ax.set_ylim(current_ylim_min, current_ylim_max)
                    except Exception:
                        ax.set_ylim(current_ylim_min, current_ylim_max)
                else:
                    ax.set_ylim(current_ylim_min, current_ylim_max)
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()

            # Sleep for configured prediction interval (only in live mode, not CSV simulation)
            if not csv:
                time.sleep(prediction_interval)

    except KeyboardInterrupt:
        if not quiet:
            print("\nLive forecasting stopped by user.")

    avg_mean = _safe_mean(result_avg_err)
    avg_max = _safe_max(result_avg_err)
    avg_min = _safe_min(result_avg_err)
    p80 = _safe_pct(result_avg_err, 80)
    p95 = _safe_pct(result_avg_err, 95)
    p99 = _safe_pct(result_avg_err, 99)

    y_true_arr = np.array(y_true_list, dtype=float)
    y_pred_arr = np.array(y_pred_list, dtype=float)
    mbe = float(np.mean(y_true_arr - y_pred_arr)
                ) if len(y_true_arr) else float('nan')
    pbias = (
        float(100.0 * np.sum(y_true_arr - y_pred_arr) / np.sum(y_true_arr))
        if len(y_true_arr) and np.sum(y_true_arr) != 0 else float('nan')
    )

    # Calculate average timing metrics
    avg_training_time = _safe_mean(training_times)
    avg_inference_time = _safe_mean(inference_times)

    if not quiet:
        print("AVERAGE", avg_mean, "% MAX", avg_max, "% MIN", avg_min, "% (MAPE)")
        print(f"P80 {p80}%  P95 {p95}%  P99 {p99}% (MAPE)")
        print(f"MBE {mbe}  PBIAS% {pbias}")
        print(f"Avg Training Time: {avg_training_time:.3f}s")
        print(f"Avg Inference Time: {avg_inference_time:.3f}s")

    return {
        "mean_avg_err": avg_mean,
        "max_avg_err": avg_max,
        "min_avg_err": avg_min,
        "p80_avg_err": p80,
        "p95_avg_err": p95,
        "p99_avg_err": p99,
        "mbe": mbe,
        "pbias_pct": pbias,
        "avg_training_time": avg_training_time,
        "avg_inference_time": avg_inference_time
    }


if __name__ == '__main__':

    main(
        
    )
