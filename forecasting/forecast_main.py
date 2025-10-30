"""
Universal time series forecasting main entry point.

Supports multiple models: XGBoost, Prophet, LSTM/GRU
Model selection via command-line argument.
"""

from universal_forecaster import UniversalForecaster
from models import create_model, list_available_models, get_model_info
import requests
import pandas as pd
import argparse
import time
import sys
import os
import signal
import psutil
import datetime as dt
import numpy as np

# Add models directory to path
sys.path.insert(0, os.path.dirname(__file__))


def get_data_from_api(ip, context, dimension, seconds_back):
    """Fetch data from Netdata API."""
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
    df = df.sort_index()
    # print(df.index)
    return df


def kill_old_prediction_servers():
    """Kill any existing prediction_server.py processes."""
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and 'prediction_server.py' in ' '.join(cmdline):
                # Found a prediction server
                print(
                    f"[Server] Killing old prediction server (PID {proc.info['pid']})")
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except psutil.TimeoutExpired:
                    proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if killed_count > 0:
        print(f"[Server] Killed {killed_count} old server(s)\n")
        time.sleep(1)  # Give ports time to release

    return killed_count


def main():
    parser = argparse.ArgumentParser(
        description="Universal time series forecasting with multiple model support"
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=list_available_models(),
        help='Forecasting model to use'
    )

    # Data source
    parser.add_argument('--ip', type=str, default='localhost',
                        help='Netdata server IP')
    parser.add_argument('--context', type=str,
                        default='ip.tcppackets', help='Netdata context')
    parser.add_argument('--dimension', type=str,
                        default='received', help='Netdata dimension')

    # Model parameters
    parser.add_argument('--horizon', type=int, default=5,
                        help='Forecast horizon (steps ahead)')
    parser.add_argument('--window', type=int, default=25,
                        help='Inference window size (for predictions)')
    parser.add_argument('--train-window', type=int, default=100,
                        help='Initial training window size (defaults to --window)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')

    # Forecasting parameters
    parser.add_argument('--prediction-smoothing', type=int,
                        default=2, help='Number of predictions to average')
    parser.add_argument('--prediction-interval', type=float,
                        default=5.0, help='Seconds between predictions')

    # Retraining parameters
    parser.add_argument('--retrain-scale', type=float,
                        default=3.0, help='MAD multiplier for threshold')
    parser.add_argument('--retrain-min', type=float,
                        default=30.0, help='Minimum retrain threshold (%%)')
    parser.add_argument('--retrain-consec', type=int, default=3,
                        help='Consecutive violations to retrain')
    parser.add_argument('--retrain-cooldown', type=int,
                        default=0, help='Min steps between retrains')
    parser.add_argument('--no-mad', action='store_true',
                        help='Use std instead of MAD')

    # Backoff parameters
    parser.add_argument('--retrain-rapid-seconds', type=int, default=10,
                        help='Wall-clock seconds - retrains faster than this trigger backoff')
    parser.add_argument('--backoff-long-seconds', type=int,
                        default=15, help='Base seconds for backoff window')
    parser.add_argument('--backoff-max-retrains', type=int, default=5,
                        help='Max retrains during backoff before extension')
    parser.add_argument('--backoff-clear-consecutive-ok', type=int, default=3,
                        help='Consecutive OK suppressed validations needed to clear backoff')

    # Output
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    parser.add_argument('--print-min-validations', type=int,
                        default=3, help='Min validations before printing')

    # Server
    parser.add_argument('--no-live-server', action='store_true',
                        help='Disable live visualization server')
    parser.add_argument('--live-server-port', type=int,
                        default=5000, help='Live server port')

    args = parser.parse_args()

    # Print model info
    if not args.quiet:
        train_window = args.train_window if args.train_window is not None else args.window
        model_info = get_model_info(args.model)
        print(f"\n{'='*70}")
        print("Universal Time Series Forecaster")
        print(f"Model: {model_info['class']}")
        print(f"{model_info['description']}")
        print(
            f"Train Window: {train_window}s | Inference Window: {args.window}s")
        print(f"{'='*70}\n")

    # Create model with appropriate parameters
    model_kwargs = {
        'horizon': args.horizon,
        'random_state': args.random_state
    }

    # if args.model == 'prophet':
    #     model_kwargs['seasonality_mode'] = 'additive'
    #     if args.custom_seasonality:
    #         # Parse format: name:period:fourier_order
    #         parts = args.custom_seasonality.split(':')
    #         if len(parts) == 3:
    #             model_kwargs['custom_seasonalities'] = [{
    #                 'name': parts[0],
    #                 'period': int(parts[1]),
    #                 'fourier_order': int(parts[2])
    #             }]
    # elif args.model == 'lstm':
    #     model_kwargs['lookback'] = args.lookback
    #     model_kwargs['hidden_size'] = args.hidden_size
    #     model_kwargs['epochs'] = args.epochs

    model = create_model(args.model, **model_kwargs)

    # Determine training window (use train_window if specified, otherwise window)
    train_window = args.train_window if args.train_window is not None else args.window

    # Create universal forecaster
    forecaster = UniversalForecaster(
        model=model,
        window=args.window,
        train_window=train_window,
        prediction_smoothing=args.prediction_smoothing,
        retrain_scale=args.retrain_scale,
        retrain_min=args.retrain_min,
        retrain_use_mad=not args.no_mad,
        retrain_consec=args.retrain_consec,
        retrain_cooldown=args.retrain_cooldown,
        retrain_rapid_seconds=args.retrain_rapid_seconds,
        backoff_long_seconds=args.backoff_long_seconds,
        backoff_max_retrains=args.backoff_max_retrains,
        backoff_clear_consecutive_ok=args.backoff_clear_consecutive_ok,
        print_min_validations=args.print_min_validations,
        quiet=args.quiet
    )

    init_df = get_data_from_api(
        args.ip, args.context, args.dimension, train_window)['value']

    # Initial training
    forecaster.train_initial(init_df)

    # Start live server if enabled
    if not args.no_live_server:
        # Kill any old prediction servers first
        kill_old_prediction_servers()

        import subprocess
        server_cmd = [
            'python3', 'prediction_server.py',
            '--port', str(args.live_server_port),
            '--ip', args.ip,
            '--context', args.context,
            '--dimension', args.dimension
        ]
        try:
            subprocess.Popen(
                server_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(__file__)
            )
            time.sleep(2)
            if not args.quiet:
                print(
                    f"[Server] Live visualization started on port {args.live_server_port}")
                print(
                    f"[Server] Monitoring: {args.context} -> {args.dimension}")
                print(
                    f"[Server] Web UI: http://localhost:{args.live_server_port}/\n")
        except Exception as e:
            if not args.quiet:
                print(f"[Server] Could not start: {e}\n")

    # Main forecasting loop
    next_iteration_time = time.time()
    try:
        while True:

            # Get current data
            live_df = get_data_from_api(
                args.ip, args.context, args.dimension, args.window)

            live_df_value = live_df['value']
            # Forecast
            predictions = forecaster.forecast_step(live_df_value)

            pred_timestamps = np.arange(
                live_df.index[-1] + dt.timedelta(seconds=1),
                live_df.index[-1] + dt.timedelta(seconds=args.horizon+1), dtype='datetime64[s]')

            pred_df = pd.DataFrame({'timestamp':pred_timestamps, 'prediction':predictions})
            pred_df.set_index('timestamp', inplace=True)
            pred_df = pred_df.sort_index()
            

            # Validate all ready predictions in chronological order
            # This MUST happen before sending to UI because validation can extend/clear backoff
            while True:
                validated = forecaster.validate_predictions(live_df_value)
                if validated is None:
                    break  # No more ready predictions

            # Send to live server AFTER validation (so backoff state is up-to-date)
        
            # Use only _backoff_active for backoff state
            in_backoff = getattr(forecaster, '_backoff_active', False)
            current_value = float(
                live_df_value.iloc[-1]) if len(live_df_value) > 0 else 0
            try:
                # Ensure we always send the full horizon of predictions
                horizon = args.horizon
                pred_values = list(pred_df['prediction'])
                pred_times = list(pred_df.index)
                if len(pred_values) < horizon:
                    # Pad with NaN or last value if not enough predictions
                    pad_value = float('nan') if not pred_values else float(pred_values[-1])
                    for _ in range(horizon - len(pred_values)):
                        pred_values.append(pad_value)
                        pred_times.append(pred_times[-1] + dt.timedelta(seconds=1))
                pred_payload = [
                    {'timestamp': ts.isoformat(), 'value': float(val)}
                    for ts, val in zip(pred_times, pred_values)
                ]
                print(f"[SEND] Sending {len(pred_payload)} predictions: " + ', '.join([f"{p['timestamp']}" for p in pred_payload]))
                requests.post(
                    f'http://localhost:{args.live_server_port}/predictions',
                    json={
                        'predictions': pred_payload,
                        'context': args.context,
                        'dimension': args.dimension,
                        'prediction_interval': args.prediction_interval,
                        'timestamp': live_df_value.index[-1].isoformat(),
                        'backoff': in_backoff,
                        'actual': current_value
                    },
                    # Give the local prediction server a bit more time to respond
                    timeout=2.0
                )
            except Exception as e:
                print(e)
                pass


            next_iteration_time += args.prediction_interval
            sleep_time = next_iteration_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're behind schedule, skip ahead to next interval
                next_iteration_time = time.time()

    except KeyboardInterrupt:
        if not args.quiet:
            print("\nForecasting stopped by user.")

    # Print statistics
    stats = forecaster.get_statistics()
    if not args.quiet:
        print(f"\n{'='*70}")
        print(f"Final Statistics ({model.get_model_name()})")
        print(f"{'='*70}")
        print(
            f"MAPE - Mean: {stats['mean_avg_err']:.2f}%  Max: {stats['max_avg_err']:.2f}%  Min: {stats['min_avg_err']:.2f}%")
        print(
            f"MAPE - P80: {stats['p80_avg_err']:.2f}%  P95: {stats['p95_avg_err']:.2f}%  P99: {stats['p99_avg_err']:.2f}%")
        print(f"MBE: {stats['mbe']:.2f}  PBIAS: {stats['pbias_pct']:.2f}%")
        print(
            f"Baseline - Mean: {stats['baseline_mean']:.2f}  Std: {stats['baseline_std']:.2f}")
        print(
            f"Baseline Deviation - Mean: {stats['mean_baseline_deviation']:.2f}σ  Max: {stats['max_baseline_deviation']:.2f}σ")
        print(f"Avg Training Time: {stats['avg_training_time']:.3f}s")
        print(f"Avg Inference Time: {stats['avg_inference_time']:.6f}s")
        print(f"Total Retrains: {stats['total_retrains']}")
        print(f"{'='*70}\n")

    return stats


if __name__ == '__main__':
    main()
