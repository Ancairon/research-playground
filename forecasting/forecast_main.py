"""
Universal time series forecasting main entry point.

Supports multiple models: XGBoost, LSTM/GRU
Model selection via command-line argument.
Config files supported via --config flag.
"""

from universal_forecaster import UniversalForecaster
from models import create_model, list_available_models, get_model_info
from config_cache import ConfigFingerprint, load_cached_results, save_cache
from shared_prediction import evaluate_predictions, single_shot_evaluation
import requests
import subprocess
import pandas as pd
import argparse
import time
import sys
import os
import psutil
import datetime as dt
import numpy as np
import yaml
import json

# Add models directory to path
sys.path.insert(0, os.path.dirname(__file__))


def get_data_from_api(ip, context, dimension, seconds_back):
    """Fetch data from Netdata API."""
    url = (
        f"http://{ip}:19999/api/v3/data?"
        f"contexts={context}&dimensions={dimension}&"
        f"after=-{seconds_back}&before=0&points={seconds_back}&"
        f"group=average&gtime=0&tier=0&format=json&options=seconds,jsonwrap"
    )
    r = requests.get(url, timeout=30)
    labels = r.json()['result']['labels']
    data = r.json()['result']['data']

    records = []
    for row in data:
        ts = row[0]
        val = row[1]
        records.append({'timestamp': ts, 'value': val})

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)
    return df


class CSVDataSource:
    """Data source that replays CSV data, emulating Netdata API behavior."""
    
    def __init__(self, csv_path, train_window):
        """
        Args:
            csv_path: Path to CSV file (timestamp,value format)
            train_window: Initial training window size - will start replay after this
        """
        self.df = pd.read_csv(csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True)
        self.df.set_index('timestamp', inplace=True)
        self.df = self.df.sort_index()
        
        # Start position: after train_window
        if len(self.df) < train_window:
            raise ValueError(f"CSV has only {len(self.df)} rows, need at least {train_window} for training")
        
        self.current_index = train_window  # Start after training data
        self.total_rows = len(self.df)
        
        # For simulation: map CSV timestamps to "current" time
        # This allows validation to work properly
        self.start_time = pd.Timestamp.now(tz='UTC')
        self.csv_start_time = self.df.index[train_window]
        
        print(f"[CSV Mode] Loaded {self.total_rows} rows from {csv_path}")
        print(f"[CSV Mode] Starting replay from row {self.current_index} (after training window)")
        print(f"[CSV Mode] CSV time range: {self.df.index[0]} to {self.df.index[-1]}")
        print(f"[CSV Mode] Simulated current time starts at: {self.start_time}")
    
    def _csv_to_current_time(self, csv_timestamp):
        """Convert CSV timestamp to current simulation time."""
        elapsed = (csv_timestamp - self.csv_start_time).total_seconds()
        return self.start_time + pd.Timedelta(seconds=elapsed)
    
    def get_current_value(self):
        """Get current value and advance index."""
        if self.current_index >= self.total_rows:
            return None  # End of data
        
        value = self.df.iloc[self.current_index]['value']
        self.current_index += 1
        return value
    
    def get_current_timestamp(self):
        """Get timestamp of current position (mapped to current time)."""
        if self.current_index >= self.total_rows:
            return None
        csv_ts = self.df.index[self.current_index]
        return self._csv_to_current_time(csv_ts)
    
    def get_history(self, seconds_back):
        """Get historical data up to current position (with timestamps mapped to current time)."""
        # Get data from (current_index - seconds_back) to current_index
        start_idx = max(0, self.current_index - seconds_back)
        end_idx = self.current_index
        
        history_df = self.df.iloc[start_idx:end_idx].copy()
        
        # Remap timestamps to current simulation time
        history_df.index = [self._csv_to_current_time(ts) for ts in history_df.index]
        
        return history_df
    
    def is_finished(self):
        """Check if we've reached the end of the CSV data."""
        return self.current_index >= self.total_rows


def kill_old_prediction_servers():
    """Kill any old prediction_server.py processes to avoid port conflicts."""
    try:
        current_pid = os.getpid()
        for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            try:
                pid = proc.info["pid"]
                if pid == current_pid:
                    continue
                cmdline = proc.info.get("cmdline") or []
                if any("prediction_server.py" in part for part in cmdline):
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except ImportError:
        # psutil not available; skip process cleanup
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Universal time series forecasting with multiple model support"
    )

    # Config file support
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML/JSON config file (overrides all other arguments)'
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
    
    # LSTM/GRU/LSTM-Attention shared parameters
    parser.add_argument('--lookback', type=int, default=60,
                        help='LSTM/GRU/LSTM-Attn: Number of past timesteps to use as input')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='LSTM/GRU/LSTM-Attn/TFT: Hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='LSTM/GRU/LSTM-Attn/TFT: Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='LSTM/GRU/LSTM-Attn/TFT: Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='LSTM/GRU/LSTM-Attn/N-BEATS/TFT: Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='LSTM/GRU/LSTM-Attn/N-BEATS/TFT: Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='LSTM/GRU/LSTM-Attn/N-BEATS/TFT: Training batch size')
    parser.add_argument('--scaler-type', type=str, default='standard', choices=['standard', 'none'],
                        help='LSTM/GRU/LSTM-Attn: Scaler type. standard=StandardScaler (better for drift), none=no scaling (raw values)')
    parser.add_argument('--bias-correction', type=bool, default=True,
                        help='LSTM/GRU/LSTM-Attn: Enable automatic bias correction to fix systematic over/under-prediction')
    parser.add_argument('--use-differencing', type=bool, default=False,
                        help='LSTM/GRU/LSTM-Attn: Learn changes instead of absolute values to eliminate level-shift drift')
    
    # N-BEATS specific parameters
    parser.add_argument('--num-stacks', type=int, default=2,
                        help='N-BEATS: Number of stacks')
    parser.add_argument('--num-blocks', type=int, default=3,
                        help='N-BEATS: Number of blocks per stack')
    parser.add_argument('--theta-size', type=int, default=8,
                        help='N-BEATS: Theta dimension for basis expansion')
    
    # TFT specific parameters
    parser.add_argument('--num-heads', type=int, default=4,
                        help='TFT: Number of attention heads')
    
    
    # Random Forest / Extra Trees parameters
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='RF/ET: Number of trees in the forest')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='RF/ET: Maximum depth of trees (None = unlimited)')
    parser.add_argument('--min-samples-split', type=int, default=2,
                        help='RF/ET: Minimum samples to split a node')
    parser.add_argument('--min-samples-leaf', type=int, default=1,
                        help='RF/ET: Minimum samples in a leaf node')
    parser.add_argument('--max-features', type=str, default='sqrt',
                        help='RF/ET: Number of features for splits (sqrt, log2, or int)')

    # Forecasting parameters
    parser.add_argument('--prediction-smoothing', type=int,
                        default=1, help='Number of predictions to average')
    parser.add_argument('--prediction-interval', type=float,
                        default=1.0, help='Seconds between predictions')

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
    parser.add_argument('--backoff-clear-consecutive-ok', type=int, default=5,
                        help='Consecutive OK suppressed validations needed to clear backoff')

    # Error-spike retrain parameters
    parser.add_argument('--retrain-error-scale', type=float, default=3.0,
                        help='Error spike threshold multiplier (retrain if error > scale * MAD/std)')
    parser.add_argument('--retrain-error-min', type=float, default=70.0,
                        help='Minimum error threshold for spike retrain (MAPE %%)')

    # Training safety
    parser.add_argument('--max-train-loss', type=float, default=1000.0,
                        help='Maximum acceptable training loss (avg per-epoch MSE). Exceeding this will abort training')

    # Output
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    parser.add_argument('--print-min-validations', type=int,
                        default=3, help='Min validations before printing')
    parser.add_argument('--aggregation-method', type=str,
                        choices=['mean', 'median', 'last', 'weighted'],
                        default='weighted', help='Aggregation method for overlapping predictions')
    parser.add_argument('--aggregation-weight-tau', type=float,
                        default=1.0, help='Tau (in steps) for recency weighting when using weighted aggregation')

    # Server
    parser.add_argument('--no-live-server', action='store_true',
                        help='Disable live visualization server')
    parser.add_argument('--live-server-port', type=int,
                        default=5000, help='Live server port')
    
    # CSV Mode
    parser.add_argument('--csv-file', type=str,
                        help='CSV filename in csv/ folder (enables CSV replay mode)')
    parser.add_argument('--csv-fast', action='store_true',
                        help='Fast replay mode - process CSV data as fast as possible (no sleep)')
    
    # Single-shot mode
    parser.add_argument('--single-shot', action='store_true',
                        help='Single-shot mode: train once, predict once, evaluate, and exit')
    parser.add_argument('--single-shot-skip-live-server', action='store_true',
                        help='Skip live server in single-shot mode (auto-enabled if --single-shot)')
    parser.add_argument('--ignore-cache', action='store_true',
                        help='Ignore cached results and force recomputation')

    # Parse command line args first
    args = parser.parse_args()
    
    # Load config file if specified
    if args.config:
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        # Determine file type and load
        _, ext = os.path.splitext(config_path)
        with open(config_path, 'r') as f:
            if ext in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif ext == '.json':
                config = json.load(f)
            else:
                print(f"Error: Unsupported config file format: {ext}")
                print("Supported formats: .yaml, .yml, .json")
                sys.exit(1)
        
        # If csv-file is not specified in config, infer it from config filename
        if 'csv-file' not in config and 'csv_file' not in config:
            # Extract basename without extension from config path
            config_basename = os.path.splitext(os.path.basename(config_path))[0]
            # Try to find a matching CSV file
            inferred_csv = f"{config_basename}.csv"
            config['csv-file'] = inferred_csv
            if not args.quiet:
                print(f"[Config] Inferred CSV file from config name: {inferred_csv}")
        
        # Override args with config values
        # Map config keys to argument names
        key_mapping = {
            'server-port': 'live_server_port',
            'live-server': 'no_live_server',  # Note: inverted logic
            'lookback': 'window',  # Translate lookback to window for backwards compatibility
        }
        
        for key, value in config.items():
            # Use mapping if available, otherwise convert hyphens to underscores
            if key in key_mapping:
                attr_name = key_mapping[key]
                # Handle inverted boolean logic for live-server
                if key == 'live-server':
                    value = not value  # invert because arg is 'no_live_server'
            else:
                attr_name = key.replace('-', '_')
            
            if hasattr(args, attr_name):
                setattr(args, attr_name, value)
            else:
                print(f"Warning: Unknown config parameter '{key}' - ignoring")
        
        print(f"\n[Config] Loaded configuration from: {config_path}")

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

    # Add model-specific parameters
    if args.model in ['lstm', 'gru', 'lstm-attention', 'lstm-attn']:
        model_kwargs['lookback'] = args.window
        model_kwargs['hidden_size'] = args.hidden_size
        model_kwargs['num_layers'] = args.num_layers
        model_kwargs['dropout'] = args.dropout
        model_kwargs['learning_rate'] = args.learning_rate
        model_kwargs['epochs'] = args.epochs
        model_kwargs['batch_size'] = args.batch_size
        model_kwargs['scaler_type'] = args.scaler_type
        model_kwargs['bias_correction'] = args.bias_correction
        model_kwargs['use_differencing'] = args.use_differencing
    elif args.model == 'nbeats':
        model_kwargs['lookback'] = args.window
        model_kwargs['num_stacks'] = args.num_stacks
        model_kwargs['num_blocks'] = args.num_blocks
        model_kwargs['theta_size'] = args.theta_size
        model_kwargs['hidden_size'] = args.hidden_size
        model_kwargs['learning_rate'] = args.learning_rate
        model_kwargs['epochs'] = args.epochs
        model_kwargs['batch_size'] = args.batch_size
    elif args.model == 'tft':
        model_kwargs['lookback'] = args.window
        model_kwargs['hidden_size'] = args.hidden_size
        model_kwargs['num_heads'] = args.num_heads
        model_kwargs['num_layers'] = args.num_layers
        model_kwargs['dropout'] = args.dropout
        model_kwargs['learning_rate'] = args.learning_rate
        model_kwargs['epochs'] = args.epochs
        model_kwargs['batch_size'] = args.batch_size

    elif args.model in ['randomforest', 'rf', 'extratrees', 'et']:
        model_kwargs['n_estimators'] = args.n_estimators
        model_kwargs['max_depth'] = args.max_depth
        model_kwargs['min_samples_split'] = args.min_samples_split
        model_kwargs['min_samples_leaf'] = args.min_samples_leaf
        model_kwargs['max_features'] = args.max_features

    # Build fingerprint & hash for caching (after all args are finalized)
    fingerprint = ConfigFingerprint.from_args(args)
    config_hash = fingerprint.hash()
    readable_name = fingerprint.get_readable_name()
    if not args.quiet:
        print(f"[Cache] Config name: {readable_name}")
        print(f"[Cache] Config hash: {config_hash[:16]}...")

    model = create_model(args.model, **model_kwargs)

    # Determine training window (use train_window if specified, otherwise window)
    train_window = args.train_window if args.train_window is not None else args.window

    # Initialize CSV data source if in CSV mode
    csv_source = None
    if args.csv_file:
        csv_path = os.path.join(os.path.dirname(__file__), 'csv', args.csv_file)
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
        csv_source = CSVDataSource(csv_path, train_window)
        
        if args.csv_fast:
            print("[CSV Mode] Fast replay enabled - processing as fast as possible")
        else:
            print("[CSV Mode] Normal replay - respecting original timestamps")

    # Create history fetcher based on mode
    if csv_source:
        history_fetcher = lambda seconds: csv_source.get_history(seconds)['value']
    else:
        history_fetcher = lambda seconds: get_data_from_api(args.ip, args.context, args.dimension, seconds)['value']

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
        backoff_error_scale=args.retrain_error_scale,
        backoff_error_min=args.retrain_error_min,
        print_min_validations=args.print_min_validations,
        quiet=args.quiet,
        history_fetcher=history_fetcher,
        max_train_loss=args.max_train_loss
    )
    # Apply aggregation preferences from CLI
    forecaster.aggregation_method = args.aggregation_method
    forecaster.aggregation_weight_tau = args.aggregation_weight_tau

    # === SINGLE-SHOT MODE ===
    if args.single_shot:
        # Check cache before doing any heavy work (unless --ignore-cache is set)
        if not args.ignore_cache:
            readable_name = fingerprint.get_readable_name()
            cached = load_cached_results(config_hash, readable_name)
            if cached is not None:
                if not args.quiet:
                    print("\n[Cache] Hit - using cached results for this configuration.")
                    print(f"[Cache] Name: {readable_name}")
                    print(f"[Cache] Hash: {config_hash[:16]}...")

                # Print cached summary
                print("\n[Single-Shot] CACHED RESULTS:")
                print(f"  Prediction Timestamp: {cached.get('prediction_timestamp')}")
                print(f"  Horizon: {cached.get('horizon')} steps")
                mean_mape = cached.get('mean_mape')
                if mean_mape is not None:
                    print(f"  Mean MAPE: {mean_mape:.2f}%")
                print(f"  Min Error: {cached.get('min_error')}%")
                print(f"  Max Error: {cached.get('max_error')}%")
                print(f"  Std Dev: {cached.get('std_error')}")

                # Start prediction server with cached results if not skipped
                if not args.single_shot_skip_live_server and not args.no_live_server:
                    kill_old_prediction_servers()
                    server_cmd = [
                        'python3', 'prediction_server.py',
                        '--port', str(args.live_server_port),
                        '--ip', args.ip if not csv_source else 'csv-mode',
                        '--context', args.context if not csv_source else args.csv_file,
                        '--dimension', args.dimension if not csv_source else f"model={args.model}",
                        '--horizon', str(args.horizon)
                    ]
                    try:
                        subprocess.Popen(
                            server_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            cwd=os.path.dirname(__file__)
                        )
                        time.sleep(2)
                        server_started = True
                        if not args.quiet:
                            print(f"\n[Server] Visualization server started on port {args.live_server_port}")
                    except Exception as e:
                        if not args.quiet:
                            print(f"[Server] Could not start: {e}")
                        server_started = False

                    # Send cached data to visualization server
                    if server_started:
                        try:
                            requests.post(
                                f'http://localhost:{args.live_server_port}/predictions',
                                json={
                                    'predictions': cached.get('predictions', []),
                                    'context': args.context if not csv_source else args.csv_file,
                                    'dimension': args.dimension if not csv_source else f"model={args.model}",
                                    'prediction_interval': args.prediction_interval,
                                    'timestamp': cached.get('prediction_timestamp'),
                                    'backoff': False,
                                    'actual': 0.0,  # Not available from cache
                                    'csv_mode': csv_source is not None,
                                    'historical_data': cached.get('historical_data', []),
                                    'future_actuals': cached.get('future_actuals', []),
                                    'single_shot': True,
                                    'mean_error': cached.get('mean_mape', 0.0)
                                },
                                timeout=2.0
                            )
                            if not args.quiet:
                                print(f"[Cache] Visualization updated with cached results")
                        except Exception as e:
                            if not args.quiet:
                                print(f"[Cache] Could not update visualization: {e}")

                    # Keep server running
                    if server_started:
                        print(f"\n[Single-Shot] Visualization server is running.")
                        print(f"[Single-Shot] View results at: http://localhost:{args.live_server_port}/")
                        print(f"[Single-Shot] Press Ctrl+C to exit.")
                        
                        try:
                            while True:
                                time.sleep(1)
                        except KeyboardInterrupt:
                            print(f"\n[Single-Shot] Shutting down...")
                
                # Exit after showing cached results
                sys.exit(0)

        # Cache miss or ignored: proceed with normal training flow
        
        # Start live server for visualization (unless explicitly disabled)
        server_started = False
        if not args.single_shot_skip_live_server and not args.no_live_server:
            kill_old_prediction_servers()
            server_cmd = [
                'python3', 'prediction_server.py',
                '--port', str(args.live_server_port),
                '--ip', args.ip if not csv_source else 'csv-mode',
                '--context', args.context if not csv_source else args.csv_file,
                '--dimension', args.dimension if not csv_source else f"model={args.model}",
                '--horizon', str(args.horizon)
            ]
            try:
                subprocess.Popen(
                    server_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=os.path.dirname(__file__)
                )
                time.sleep(2)
                server_started = True
                if not args.quiet:
                    print(f"[Server] Live visualization started on port {args.live_server_port}")
                    print(f"[Server] Web UI: http://localhost:{args.live_server_port}/\n")
            except Exception as e:
                if not args.quiet:
                    print(f"[Server] Could not start: {e}\n")
        
        # Use shared evaluation logic for consistent predictions
        if csv_source:
            # CSV mode: Use END-aligned evaluation
            # Pass the pandas Series with its datetime index, not the numpy values
            full_data = csv_source.df['value']
            
            result = single_shot_evaluation(
                forecaster=forecaster,
                data=full_data,
                train_window=train_window,
                lookback=args.window,
                horizon=args.horizon,
                verbose=not args.quiet
            )
            
            # Check for errors from evaluation
            if 'error' in result:
                print(f"\n❌ Evaluation failed: {result['error']}")
                print(f"   MAPE: {result['mape']}")
                return
            
            predictions = result['predictions']
            actual_values = result['actuals']
            errors = result['errors']
            mean_error = result['mape']
            train_time = result['train_time']
            inference_time = result.get('inference_time', 0.0)
            
            # Get timestamps for predictions from the CSV
            # Prediction indices are at the END: [len(data) - horizon : len(data)]
            prediction_indices = list(range(len(full_data) - args.horizon, len(full_data)))
            actual_timestamps = [csv_source.df.index[i] for i in prediction_indices]
            pred_timestamps = [csv_source._csv_to_current_time(ts) for ts in actual_timestamps]
            
            # Get the timestamp just before predictions start (last training point)
            prediction_position = len(full_data) - args.horizon
            prediction_timestamp = csv_source._csv_to_current_time(csv_source.df.index[prediction_position - 1])
            
            # Historical data: only the training window (not entire CSV)
            # Training window ends at prediction_position and starts train_window points before
            train_start_idx = prediction_position - train_window
            if train_start_idx < 0:
                train_start_idx = 0
            
            hist_df = csv_source.df.iloc[train_start_idx:prediction_position].copy()
            hist_df.index = [csv_source._csv_to_current_time(ts) for ts in hist_df.index]
            
            historical_data = [
                {
                    'timestamp': ts.isoformat(),
                    'value': float(val)
                }
                for ts, val in zip(hist_df.index, hist_df['value'].values)
            ]
            
            if not args.quiet:
                print(f"[Single-Shot] Sending {len(historical_data)} historical data points (training window) to visualization")
        else:
            # Live API mode: Sequential training and prediction
            init_df = get_data_from_api(args.ip, args.context, args.dimension, train_window)['value']
            
            # Initial training with timing
            train_start = time.time()
            forecaster.train_initial(init_df)
            train_time = time.time() - train_start
            
            if not args.quiet:
                print("\n[Single-Shot] Mode enabled - making one prediction and evaluating")
                print(f"[Single-Shot] Training completed in {train_time:.2f}s")
            
            # Get current data for prediction
            prediction_df = get_data_from_api(args.ip, args.context, args.dimension, args.window)
            prediction_timestamp = prediction_df.index[-1]
            
            # Make ONE prediction with timing
            inference_start = time.time()
            predictions = forecaster.forecast_step(prediction_df['value'])
            inference_time = time.time() - inference_start
            
            if not args.quiet:
                print(f"[Single-Shot] Prediction made at {prediction_timestamp}")
                print(f"[Single-Shot] Inference completed in {inference_time:.4f}s")
                print(f"[Single-Shot] Horizon: {args.horizon} steps")
                print(f"[Single-Shot] Predictions: {[f'{p:.3f}' for p in predictions]}")
            
            # Historical data for visualization
            historical_data = [
                {
                    'timestamp': ts.isoformat(),
                    'value': float(val)
                }
                for ts, val in zip(prediction_df.index, prediction_df['value'].values)
            ]
            
            # Wait for actual data to arrive
            actual_values = []
            actual_timestamps = []
            if not args.quiet:
                print(f"[Single-Shot] Waiting {args.horizon * args.prediction_interval} seconds for actual data...")
            
            for step in range(args.horizon):
                time.sleep(args.prediction_interval)
                step_df = get_data_from_api(args.ip, args.context, args.dimension, 1)
                actual_values.append(float(step_df['value'].iloc[-1]))
                actual_timestamps.append(step_df.index[-1])
            
            # Generate prediction timestamps
            if len(prediction_df) >= 2:
                data_interval = int((prediction_df.index[-1] - prediction_df.index[-2]).total_seconds())
            else:
                data_interval = int(args.prediction_interval)
            
            pred_timestamps = pd.date_range(
                start=prediction_timestamp + pd.Timedelta(seconds=data_interval),
                periods=args.horizon,
                freq=f'{data_interval}s'
            )
            
            # Calculate errors
            eval_metrics = evaluate_predictions(
                predictions=predictions,
                actuals=actual_values,
                verbose=not args.quiet
            )
            errors = eval_metrics['errors']
            mean_error = eval_metrics['mape']
        
        if not args.quiet:
            print(f"[Single-Shot] Sample prediction timestamps: {pred_timestamps[:3] if hasattr(pred_timestamps, '__getitem__') else list(pred_timestamps)[:3]}")
        
        # Build prediction payload for visualization
        pred_payload = []
        for i, (ts, pred_val) in enumerate(zip(pred_timestamps, predictions)):
            # Add uncertainty bands based on recent errors (if available)
            if hasattr(forecaster, 'recent_errors') and len(forecaster.recent_errors) > 0:
                recent_avg_error = np.mean(list(forecaster.recent_errors)[-10:])
                uncertainty_pct = min(recent_avg_error, 50.0) / 100.0
            else:
                uncertainty_pct = 0.20  # default 20%
            
            pred_payload.append({
                'timestamp': ts.isoformat(),
                'value': float(pred_val),
                'min': float(pred_val * (1 - uncertainty_pct)),
                'max': float(pred_val * (1 + uncertainty_pct)),
                'actual': float(actual_values[i]) if i < len(actual_values) else None
            })
        
        if not args.quiet:
            for i, (pred, actual, error) in enumerate(zip(predictions, actual_values, errors)):
                print(f"[Single-Shot] Step {i+1}/{args.horizon}: Pred={pred:.3f}, Actual={actual:.3f}, MAPE={error:.2f}%")
        
        # Send to visualization server if running
        if server_started:
            try:
                # Prepare future actual data for visualization
                future_actuals = [
                    {
                        'timestamp': ts.isoformat(),
                        'value': float(val)
                    }
                    for ts, val in zip(pred_timestamps, actual_values)  # Use pred_timestamps for alignment
                ]
                
                # In CSV mode, override context/dimension to show run-specific info
                if csv_source is not None:
                    ctx = args.csv_file or 'csv-mode'
                    dim = f"model={args.model}"
                else:
                    ctx = args.context
                    dim = args.dimension

                # Get the last historical value for the visualization
                if csv_source is not None:
                    last_actual = float(historical_data[-1]['value']) if historical_data else 0.0
                else:
                    last_actual = float(prediction_df['value'].iloc[-1])
                
                requests.post(
                    f'http://localhost:{args.live_server_port}/predictions',
                    json={
                        'predictions': pred_payload,
                        'context': ctx,
                        'dimension': dim,
                        'prediction_interval': args.prediction_interval,
                        'timestamp': prediction_timestamp.isoformat(),
                        'backoff': False,
                        'actual': last_actual,
                        'csv_mode': csv_source is not None,
                        'historical_data': historical_data,
                        'future_actuals': future_actuals,
                        'single_shot': True,
                        'mean_error': mean_error
                    },
                    timeout=2.0
                )
                if not args.quiet:
                    print(f"\n[Single-Shot] Visualization updated - check http://localhost:{args.live_server_port}/")
            except Exception as e:
                if not args.quiet:
                    print(f"[Single-Shot] Could not update visualization: {e}")
        
        # Save results to cache
        results_payload = {
            "prediction_timestamp": prediction_timestamp.isoformat(),
            "horizon": args.horizon,
            "mean_mape": float(mean_error),
            "min_error": float(min(errors)),
            "max_error": float(max(errors)),
            "std_error": float(np.std(errors)),
            "predictions": pred_payload,
            "historical_data": historical_data,
            "future_actuals": future_actuals,
        }
        save_cache(config_hash, fingerprint, results_payload)
        if not args.quiet:
            print(f"[Cache] Saved results to cache (hash: {config_hash[:16]}...)")
        
        print(f"\n[Single-Shot] RESULTS:")
        print(f"  Prediction Timestamp: {prediction_timestamp}")
        print(f"  Horizon: {args.horizon} steps")
        print(f"  Mean MAPE: {mean_error:.2f}%")
        print(f"  Min Error: {min(errors):.2f}%")
        print(f"  Max Error: {max(errors):.2f}%")
        print(f"  Std Dev: {np.std(errors):.2f}%")
        
        # Format timing outputs
        train_time_str = f"{train_time:.2f}s" if train_time < 60 else f"{train_time/60:.2f}m"
        inference_time_str = f"{inference_time*1000:.2f}ms" if inference_time < 1 else f"{inference_time:.2f}s"
        print(f"  Train Time: {train_time_str}")
        print(f"  Inference Time: {inference_time_str}")

        if server_started:
            print(f"\n[Single-Shot] Visualization server is running.")
            print(f"[Single-Shot] View results at: http://localhost:{args.live_server_port}/")
            print(f"[Single-Shot] Press Ctrl+C to exit.")
            
            # Keep running to allow user to view visualization
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"\n[Single-Shot] Shutting down...")
        else:
            print(f"\n[Single-Shot] Complete. Exiting.")
        
        sys.exit(0)

    # Start live server if enabled
    if not args.no_live_server:
        # Kill any old prediction servers first
        kill_old_prediction_servers()

        server_cmd = [
            'python3', 'prediction_server.py',
            '--port', str(args.live_server_port),
            '--ip', args.ip,
            '--context', args.context,
            '--dimension', args.dimension,
            '--horizon', str(args.horizon)
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
    csv_last_timestamp = None
    
    try:
        while True:
            # Check if CSV mode has finished
            if csv_source and csv_source.is_finished():
                print("\n[CSV Mode] Reached end of CSV data. Exiting.")
                break

            # Get current data based on mode
            if csv_source:
                # CSV Mode
                live_df = csv_source.get_history(args.window)
                current_timestamp = csv_source.get_current_timestamp()
                
                # Advance to next row for next iteration
                csv_source.current_index += 1
                
                # Calculate sleep time for normal replay (use prediction_interval)
                if not args.csv_fast:
                    # In normal mode, use the configured prediction_interval
                    # (Don't use actual CSV timestamp differences as they could be from months ago)
                    time.sleep(args.prediction_interval)
                # else: fast mode - no sleep
                
                csv_last_timestamp = current_timestamp
                
            else:
                # Live API Mode
                live_df = get_data_from_api(
                    args.ip, args.context, args.dimension, args.window)

            live_df_value = live_df['value']
            # Forecast
            predictions = forecaster.forecast_step(live_df_value)
            
            if not args.quiet and csv_source:
                # Compact single-line status for CSV mode
                backoff_status = "BACKOFF" if getattr(forecaster, '_backoff_active', False) else "ACTIVE"
                latest_err = getattr(forecaster, '_latest_validation_error', None)
                err_display = f"{latest_err:.1f}%" if latest_err is not None else "N/A"
                threshold = getattr(forecaster, '_current_threshold', args.retrain_error_min)
                print(f"\r[CSV {csv_source.current_index:04d}/{csv_source.total_rows}] {backoff_status:8s} | Err: {err_display:6s} / Thr: {threshold:.1f}% | Pred: {predictions[0]:.2f}", end='', flush=True)

            pred_timestamps = np.arange(
                live_df.index[-1] + dt.timedelta(seconds=1),
                live_df.index[-1] + dt.timedelta(seconds=args.horizon+1), dtype='datetime64[s]')

            # Use aggregated predictions for UI display if available
            aggregated_predictions = []
            for ts in pred_timestamps:
                ts_pd = pd.Timestamp(ts)
                # If this target timestamp has been finalized (all contributors have arrived),
                # send the finalized aggregated value (historical final value)
                if hasattr(forecaster, 'finalized_predictions') and ts_pd in forecaster.finalized_predictions:
                    try:
                        agg_val, n_contrib, finalized_step = forecaster.finalized_predictions[ts_pd]
                        aggregated_predictions.append(float(agg_val))
                        continue
                    except Exception:
                        pass

                # Otherwise, for non-final (future) targets show the most recent contributor
                if hasattr(forecaster, 'pending_predictions') and ts_pd in forecaster.pending_predictions:
                    pred_vals = [pred_val for _, pred_val in forecaster.pending_predictions[ts_pd]]
                    if pred_vals:
                        # Use the most recent prediction for visualization of future targets
                        aggregated_predictions.append(float(pred_vals[-1]))
                        continue

                # Fallback to the current prediction array
                idx = len(aggregated_predictions)
                aggregated_predictions.append(predictions[idx] if idx < len(predictions) else 0.0)

            pred_df = pd.DataFrame({'timestamp':pred_timestamps, 'prediction':aggregated_predictions})
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
                pred_payload = []
                # For each target timestamp, include:
                # - 'agg': finalized aggregated value (only present if finalized)
                # - 'agg_finalized': bool whether agg is final
                # - 'contributors': list of contributor values (may be empty)
                # - 'value': compatibility field (agg if finalized else most recent contributor or NaN)
                for ts, val in zip(pred_times, pred_values):
                    ts_pd = pd.Timestamp(ts)
                    agg = None
                    agg_finalized = False
                    contributors = []

                    if hasattr(forecaster, 'pending_predictions') and ts_pd in forecaster.pending_predictions:
                        contributors = [pv for _, pv in forecaster.pending_predictions[ts_pd]]

                    if hasattr(forecaster, 'finalized_predictions') and ts_pd in forecaster.finalized_predictions:
                        try:
                            agg_val, n_contrib, finalized_step = forecaster.finalized_predictions[ts_pd]
                            agg = float(agg_val)
                            agg_finalized = True
                        except Exception:
                            agg = None

                    # Determine a compatibility 'value' field: use agg if finalized, else most recent contributor or NaN
                    # sanitize numeric values to JSON-friendly types (no NaN/Inf, use null)
                    def _sanitize_num(x):
                        try:
                            xv = float(x)
                        except Exception:
                            return None
                        # Use numpy.isfinite for broad compatibility
                        if not np.isfinite(xv):
                            return None
                        return float(xv)

                    contributors_s = [_sanitize_num(c) for c in contributors]
                    agg_s = _sanitize_num(agg) if agg is not None else None
                    val_s = _sanitize_num(val)  # Sanitize the raw prediction value

                    if agg_finalized:
                        out_value = agg_s
                    else:
                        # pick most recent non-null contributor if available
                        out_value = None
                        for c in reversed(contributors_s):
                            if c is not None:
                                out_value = c
                                break
                        
                        # If no contributors, use the raw prediction value
                        if out_value is None:
                            out_value = val_s

                    # compute min/max from sanitized contributors (if any)
                    contrib_vals = [c for c in contributors_s if c is not None]
                    
                    # If we have multiple contributors, use their range
                    if len(contrib_vals) > 1:
                        min_v = float(min(contrib_vals))
                        max_v = float(max(contrib_vals))
                    # If only one contributor or raw prediction, add synthetic uncertainty based on recent errors
                    elif contrib_vals or val_s is not None:
                        base_val = contrib_vals[0] if contrib_vals else val_s
                        # Use recent error to estimate uncertainty (default to 20% if no errors)
                        if hasattr(forecaster, 'recent_errors') and len(forecaster.recent_errors) > 0:
                            recent_avg_error = np.mean(list(forecaster.recent_errors)[-10:])  # last 10 errors
                            uncertainty_pct = min(recent_avg_error, 50.0) / 100.0  # cap at 50%
                        else:
                            uncertainty_pct = 0.20  # default 20% uncertainty
                        
                        min_v = float(base_val * (1 - uncertainty_pct))
                        max_v = float(base_val * (1 + uncertainty_pct))
                    else:
                        min_v = None
                        max_v = None

                    pred_payload.append({
                        'timestamp': ts.isoformat(),
                        'value': out_value,
                        'agg': agg_s,
                        'agg_finalized': bool(agg_finalized),
                        'contributors': contributors_s,
                        'min': min_v,
                        'max': max_v
                    })
                # print(f"[SEND] Sending {len(pred_payload)} predictions: " + ', '.join([f"{p['timestamp']}" for p in pred_payload]))
                
                # Prepare historical data for CSV mode
                historical_data = None
                if csv_source:
                    # Send recent historical data (last 60 points for chart)
                    hist_window = min(60, len(live_df_value))
                    hist_df = live_df_value.tail(hist_window)
                    historical_data = [
                        {
                            'timestamp': ts.isoformat(),
                            'value': float(val)
                        }
                        for ts, val in zip(hist_df.index, hist_df.values)
                    ]
                
                requests.post(
                    f'http://localhost:{args.live_server_port}/predictions',
                    json={
                        'predictions': pred_payload,
                        'context': args.context,
                        'dimension': args.dimension,
                        'prediction_interval': args.prediction_interval,
                        'timestamp': live_df_value.index[-1].isoformat(),
                        'backoff': in_backoff,
                        'actual': current_value,
                        'csv_mode': csv_source is not None,
                        'historical_data': historical_data
                    },
                    # Give the local prediction server a bit more time to respond
                    timeout=2.0
                )
            except Exception as e:
                print(e)
                pass

            # Handle timing based on mode
            if csv_source:
                # CSV Mode timing
                if args.csv_fast:
                    # Fast mode - no sleep, process as fast as possible
                    pass
                else:
                    # Normal mode - sleep handled earlier based on CSV timestamps
                    pass
            else:
                # Live API mode - respect prediction interval
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
