"""
Production LSTM-Attention Forecasting System

Complete forecasting pipeline including:
- PyTorch LSTM with attention mechanism
- Hyperparameter tuning (staged search)
- Training and inference
"""

import time
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import warnings
from itertools import product
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

from smoothing import apply_smoothing


def sanitize_for_json(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif isinstance(obj, bool):
        return bool(obj)
    elif obj is None or isinstance(obj, (int, float, str)):
        return obj
    else:
        return str(obj)


def detect_granularity(data: pd.Series) -> Optional[int]:
    """
    Detect the time granularity (in seconds) of a time series with timestamp index.
    
    Since granularity is stable and consistent in the data, we calculate it from
    the first few intervals and return a single stable value.
    
    Returns:
        Granularity in seconds, or None if timestamps are not available.
    """
    if not hasattr(data, 'index') or not hasattr(data.index, 'to_pydatetime'):
        return None
    
    try:
        timestamps = data.index.to_pydatetime()
        if len(timestamps) < 2:
            return None
        
        # Calculate granularity from first interval (data is stable)
        granularity = int((timestamps[1] - timestamps[0]).total_seconds())
        return max(1, granularity)  # At least 1 second
    except Exception:
        return None


def calculate_future_timestamps(last_timestamp, granularity: int, horizon: int, as_unix: bool = False):
    """
    Calculate future timestamps based on last timestamp and granularity.
    
    Args:
        last_timestamp: Last observed timestamp (datetime, ISO string, or Unix timestamp)
        granularity: Time granularity in seconds
        horizon: Number of future points to predict
        as_unix: If True, return Unix timestamps (integers); otherwise ISO 8601 strings
    
    Returns:
        List of timestamps (Unix integers if as_unix=True, ISO 8601 strings otherwise)
    """
    if isinstance(last_timestamp, str):
        last_timestamp = pd.to_datetime(last_timestamp)
    elif isinstance(last_timestamp, (int, float)):
        # Unix timestamp
        last_timestamp = pd.to_datetime(last_timestamp, unit='s')
    
    future_timestamps = []
    for i in range(1, horizon + 1):
        future_ts = last_timestamp + pd.Timedelta(seconds=granularity * i)
        if as_unix:
            future_timestamps.append(int(future_ts.timestamp()))
        else:
            future_timestamps.append(future_ts.isoformat())
    
    return future_timestamps


def evaluate_model_config(
    model,  # LSTMAttentionModel
    data: pd.Series,
    train_window: int,
    lookback: int,
    horizon: int,
    smoothing_method: Optional[str] = None,
    smoothing_window: int = 3,
    smoothing_alpha: float = 0.2,
    epochs: int = 100,
    verbose: bool = False,
    use_unix_timestamps: bool = False
) -> Dict[str, Any]:
    """
    Train and evaluate a model configuration on time series data.

    Predicts the last `horizon` points using `train_window` history.
    Returns predictions, actuals, metrics, and timestamps.
    
    Args:
        use_unix_timestamps: If True, return Unix timestamps (integers); otherwise ISO 8601 strings
    Returns predictions, actuals, metrics (sMAPE, RMSE, MBE), and timing.
    """
    # Data validation
    min_required = train_window + horizon
    if len(data) < min_required:
        return {'error': f'Insufficient data: need {min_required}, have {len(data)}', 'mape': float('inf')}

    min_train_window = lookback + horizon
    if train_window < min_train_window:
        return {'error': f'train_window {train_window} < minimum {min_train_window}', 'mape': float('inf')}

    # Calculate indices (predict LAST horizon points)
    data_end = len(data)
    prediction_start = data_end - horizon
    train_start = prediction_start - train_window
    train_end = prediction_start

    if train_start < 0:
        return {'error': f'Not enough data: need {train_window + horizon}, have {len(data)}', 'mape': float('inf')}

    # Get training data
    train_data = data.iloc[train_start:train_end]

    # Apply smoothing if configured
    try:
        train_data_smoothed = apply_smoothing(
            train_data, method=smoothing_method, window=smoothing_window, alpha=smoothing_alpha) if smoothing_method else train_data
    except Exception:
        train_data_smoothed = train_data

    # Capture train data for visualization (raw + smoothed)
    train_data_raw_list = list(train_data.values) if hasattr(
        train_data, 'values') else list(train_data)
    if isinstance(train_data_smoothed, pd.Series):
        smoothed_train_list = list(train_data_smoothed.values)
    else:
        smoothed_train_list = list(train_data_smoothed)

    # Train model
    start_time = time.time()
    try:
        model.train(train_data_smoothed, epochs=epochs, quiet=not verbose)
    except Exception as e:
        return {'error': f'Training failed: {str(e)}', 'mape': float('inf')}
    train_time = time.time() - start_time

    # Get prediction input (last lookback points)
    prediction_input = train_data.iloc[-lookback:]

    # Apply smoothing to prediction input if configured
    try:
        prediction_input_smoothed = apply_smoothing(
            prediction_input, method=smoothing_method, window=smoothing_window, alpha=smoothing_alpha) if smoothing_method else prediction_input
    except Exception:
        prediction_input_smoothed = prediction_input

    # Update model with live data if supported
    if model.supports_online_learning():
        try:
            model.update(prediction_input_smoothed)
        except Exception:
            pass

    # Make predictions
    inference_start = time.time()
    try:
        predictions = model.predict()
        if not predictions:
            return {'error': 'No predictions generated', 'mape': float('inf')}
        predictions = [max(-1e9, min(1e9, float(p)))
                       if np.isfinite(p) else 0.0 for p in predictions]
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}', 'mape': float('inf')}
    inference_time = time.time() - inference_start

    # Get actual values
    actuals = [float(data.iloc[prediction_start + i]) for i in range(horizon)]
    
    # Extract actual timestamps from the data index (if available)
    actual_timestamps = []
    if hasattr(data, 'index') and hasattr(data.index, 'to_pydatetime'):
        try:
            if use_unix_timestamps:
                actual_timestamps = [int(data.index[prediction_start + i].timestamp()) for i in range(horizon)]
            else:
                actual_timestamps = [data.index[prediction_start + i].isoformat() for i in range(horizon)]
        except Exception:
            actual_timestamps = []

    # Apply smoothing to actuals for evaluation if configured
    actuals_orig = list(actuals)
    try:
        actuals_eval = apply_smoothing(actuals, method=smoothing_method, window=smoothing_window,
                                       alpha=smoothing_alpha) if smoothing_method else actuals
        if isinstance(actuals_eval, pd.Series):
            actuals_eval = list(actuals_eval.values)
    except Exception:
        actuals_eval = actuals

    # Calculate metrics (sMAPE)
    errors = []
    for actual, pred in zip(actuals_eval, predictions):
        err_abs = abs(actual - pred)
        denominator = (abs(actual) + abs(pred)) / 2.0
        mape_val = 0.0 if denominator < 1e-6 else min(
            (err_abs / denominator) * 100.0, 1000.0)
        errors.append(mape_val)

    mape = np.mean(errors) if errors else float('inf')
    mbe = np.mean([actual - pred for actual,
                  pred in zip(actuals_eval, predictions)])
    rmse = np.sqrt(np.mean(
        [(actual - pred)**2 for actual, pred in zip(actuals_eval, predictions)]))

    result = {
        'predictions': predictions,
        'prediction_timestamps': actual_timestamps,
        'actuals': actuals_orig,
        'actual_timestamps': actual_timestamps,
        'smoothed_actuals': actuals_eval,
        'train_data': train_data_raw_list,
        'smoothed_train_data': smoothed_train_list,
        'train_time': train_time,
        'inference_time': inference_time,
        'mape': mape,
        'mbe': mbe,
        'rmse': rmse,
        'errors': errors,
        'train_window': train_window,
        'smoothing_applied': smoothing_method is not None
    }

    return sanitize_for_json(result)

# PYTORCH MODEL COMPONENTS


class TimeSeriesDataset(Dataset):
    """Sliding window dataset for time series."""

    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return max(0, len(self.data) - self.lookback - self.horizon + 1)

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback:idx + self.lookback + self.horizon]
        X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
        return torch.from_numpy(X_np), torch.from_numpy(y_np)


class Attention(nn.Module):
    """Scaled dot-product attention for LSTM hidden states."""

    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        energy = torch.tanh(self.attn(hidden_states))
        attention = self.v(energy).squeeze(-1)
        attention_weights = torch.softmax(attention, dim=1)
        context = (attention_weights.unsqueeze(-1) * hidden_states).sum(dim=1)
        return context, attention_weights


class LSTMAttentionNetwork(nn.Module):
    """LSTM with attention mechanism for multi-step forecasting."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int, dropout: float = 0.2):
        super(LSTMAttentionNetwork, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = Attention(hidden_size)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, horizon)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            predictions: (batch, horizon)
        """
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        predictions = self.decoder(context)
        return predictions


class LSTMAttentionModel:
    """
    LSTM with Attention for time series forecasting.

    Args:
        horizon: Number of steps to predict
        lookback: Number of historical steps to use
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        learning_rate: Optimizer learning rate
        epochs: Training epochs
        batch_size: Training batch size
        scaler_type: 'standard', 'robust', or 'none'
        use_differencing: Learn changes instead of absolute values
    """

    def __init__(
        self,
        horizon: int = 5,
        random_state: int = 42,
        lookback: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        scaler_type: str = 'none',
        use_differencing: bool = False,
        **kwargs
    ):
        self.horizon = horizon
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        self.use_differencing = use_differencing

        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'none':
            self.scaler = None
        else:
            raise ValueError(
                f"Invalid scaler_type: {scaler_type}. Must be 'standard', 'robust' or 'none'")

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.last_data = None
        self.last_data_raw = None
        self.last_value = None

        # if not torch.cuda.is_available() and (hidden_size >= 128 or lookback >= 180):
        #     warnings.warn(
        #         "Training on CPU - large model may be slow. Consider using GPU.")

        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self, data: pd.Series, **kwargs) -> float:
        """Train LSTM with Attention."""
        start_time = time.time()
        max_train_loss = kwargs.get('max_train_loss', None)
        max_train_seconds = kwargs.get('max_train_seconds', None)

        try:
            import os
            os.environ.setdefault('OMP_NUM_THREADS', '1')
            os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
            os.environ.setdefault('MKL_NUM_THREADS', '1')
            torch.set_num_threads(1)
        except Exception:
            pass

        if len(data) < self.lookback + self.horizon:
            raise ValueError(
                f"Insufficient data for training. Need at least {self.lookback + self.horizon} points, "
                f"got {len(data)}"
            )

        # Apply differencing if enabled
        if self.use_differencing:
            self.last_value = data.values[-1]
            diffs = np.diff(data.values)
            if diffs.size == 0:
                pad = 0.0
                data_values = np.array([pad], dtype=np.float32)
            else:
                pad = float(diffs[0])
                data_values = np.concatenate([[pad], diffs])
        else:
            data_values = data.values

        # Scale data
        if self.scaler is None:
            data_scaled = data_values
            self.last_data_raw = data.values[-self.lookback:]
        else:
            data_scaled = self.scaler.fit_transform(
                data_values.reshape(-1, 1)).flatten()

        # Create dataset
        dataset = TimeSeriesDataset(data_scaled, self.lookback, self.horizon)

        if len(dataset) == 0:
            raise ValueError("No training samples created")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )

        # Create model
        self.model = LSTMAttentionNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            horizon=self.horizon,
            dropout=self.dropout
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # Early stopping
        best_loss = float('inf')
        patience = 7
        patience_counter = 0

        # Training loop
        self.model.train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                num_batches = 0
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.unsqueeze(-1).to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)

                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches
                scheduler.step(avg_loss)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(
                        f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

                if max_train_loss is not None:
                    try:
                        if float(max_train_loss) is not None and avg_loss > float(max_train_loss):
                            print(
                                f"  Aborting training: avg_loss={avg_loss:.6f} > max_train_loss={max_train_loss}")
                            best_loss = min(best_loss, avg_loss)
                            break
                    except Exception:
                        pass

                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                        break

                if max_train_seconds is not None:
                    elapsed = time.time() - start_time
                    if elapsed > float(max_train_seconds):
                        print(
                            f"  Aborting training: elapsed {elapsed:.1f}s > max_train_seconds={max_train_seconds}s")
                        break

        self.is_trained = True
        self.last_data = data_scaled[-self.lookback:]

        training_time = time.time() - start_time
        try:
            self.last_train_loss = float(best_loss)
        except Exception:
            self.last_train_loss = float('nan')
        return training_time

    def predict(self, **kwargs) -> List[float]:
        """Generate predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        if self.last_data is None or len(self.last_data) < self.lookback:
            raise RuntimeError("Insufficient data for prediction")

        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(
                self.last_data[-self.lookback:]).unsqueeze(0).unsqueeze(-1).to(self.device)
            predictions_scaled = self.model(X).cpu().numpy().flatten()

        # Inverse transform
        if self.scaler is None:
            predictions = predictions_scaled
        else:
            predictions = self.scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)).flatten()

        # Reverse differencing
        if self.use_differencing:
            absolute_predictions = np.zeros(len(predictions))
            current_value = self.last_value
            for i in range(len(predictions)):
                current_value = current_value + predictions[i]
                absolute_predictions[i] = current_value
            predictions = absolute_predictions

        return predictions.tolist()

    def update(self, data: pd.Series, **kwargs):
        """Update last_data window for prediction."""
        if not self.is_trained:
            warnings.warn("Model not trained yet")
            return

        try:
            if self.use_differencing:
                self.last_value = data.values[-1]
                data_values = np.diff(data.values)
                if len(data_values) == 0:
                    return
            else:
                data_values = data.values

            if self.scaler is None:
                data_scaled = data_values
                self.last_data_raw = np.concatenate(
                    [self.last_data_raw, data.values])[-self.lookback:]
            else:
                data_scaled = self.scaler.transform(
                    data_values.reshape(-1, 1)).flatten()

            if self.last_data is None:
                self.last_data = data_scaled[-self.lookback:]
            else:
                self.last_data = np.concatenate(
                    [self.last_data, data_scaled])[-self.lookback:]
        except Exception as e:
            warnings.warn(f"Update failed: {e}")

    def get_model_name(self) -> str:
        return "LSTM+Attention"

    def supports_online_learning(self) -> bool:
        return False

# HYPERPARAMETER TUNING


def tune_lstm_attention(data: pd.Series, horizon: int) -> dict:
    """
    Staged hyperparameter search: 30-epoch probe on ~35% configs, then 150-epoch
    full training on top 5 performers. Searches lookback, hidden_size, learning_rate,
    batch_size, differencing, scaling, and smoothing options.

    Returns: Best config dict with tuned hyperparameters
    """
    data_size = len(data)
    max_lookback = data_size // 3

    config_epochs = 150
    probe_epochs = 30
    config_dropout = 0.2

    def filter_lookbacks(values):
        filtered = [v for v in values if v <=
                    max_lookback and (v + horizon) < data_size]
        if not filtered:
            safe_lookback = min(max_lookback, max(
                60, (data_size - horizon) // 2))
            filtered = [safe_lookback]
        return filtered

    def _format_smoothing_config(val):
        if val is None:
            return "None"
        if isinstance(val, (list, tuple)) and len(val) == 3:
            method, window, alpha = val
            if method is None:
                return "None (no smoothing)"
            elif method in ('moving_average', 'ma'):
                return f"moving_average(window={window})"
            elif method in ('ewma', 'exponential'):
                return f"ewma(alpha={alpha})"
            else:
                return str(val)
        return str(val)

    def _is_similar(a, b, min_matches=None):
        keys = set(a.keys()) & set(b.keys())
        if min_matches is None:
            min_matches = max(1, len(keys) // 2)
        matches = sum(1 for k in keys if a.get(k) == b.get(k))
        return matches >= min_matches

    # Build search space
    base_lookbacks = [60, 120]
    min_lookback = max(30, horizon // 10)
    max_lookback_search = min(1000, max_lookback, data_size // 4)

    adaptive_lookbacks = [min_lookback, max_lookback_search]
    if max_lookback_search > min_lookback * 3:
        adaptive_lookbacks.append((min_lookback + max_lookback_search) // 2)

    all_lookbacks = sorted(set(base_lookbacks + adaptive_lookbacks))
    lookback_values = filter_lookbacks(all_lookbacks)

    search_space = {
        'lookback': lookback_values,
        'hidden_size': [64, 128],
        'num_layers': [2],
        'dropout': [config_dropout],
        'learning_rate': [0.001, 0.0005],
        'epochs': [config_epochs],
        'batch_size': [128, 256],
        'use-differencing': [False, True],
        'scaler-type': ['none', 'standard', 'robust'],
        'smoothing-config': [
            (None, None, None),
            ('moving_average', 3, None),
            ('moving_average', 5, None),
            ('ewma', None, 0.5),
        ],
    }

    param_names = list(search_space.keys())
    param_values = [search_space[k] for k in param_names]
    all_configs = [dict(zip(param_names, v)) for v in product(*param_values)]

    total_possible = len(all_configs)

    phase1_size = max(15, min(50, int(total_possible * 0.35)))
    random.seed(42)
    phase1_configs = random.sample(
        all_configs, min(phase1_size, total_possible))

    print(f"\n{'='*70}")
    print("LSTM-Attention Hyperparameter Tuning")
    print("Mode: STAGED SEARCH (Fast Probe → Full Training on Winners)")
    print(
        f"Phase 1: {len(phase1_configs)} configs with {probe_epochs} epochs (from {total_possible} total)")
    print(f"Phase 2: Top configs retrained with {config_epochs} epochs")
    print(f"{'='*70}\n")

    # PHASE 1
    print(f"{'='*70}")
    print(
        f"PHASE 1: FAST PROBE - Testing {len(phase1_configs)} configs with {probe_epochs} epochs each")
    print(f"{'='*70}\n")

    results = []
    bad_configs = []
    best_mape_so_far = float('inf')
    skipped = 0

    for i, config in enumerate(phase1_configs, 1):
        if any(_is_similar(config, bc) for bc in bad_configs):
            print(
                f"\n[{i}/{len(phase1_configs)}] Skipping configuration similar to poor-performing region")
            continue

        eval_config = config.copy()
        eval_config['epochs'] = probe_epochs

        print(f"\n[{i}/{len(phase1_configs)}] Testing configuration:")
        for key, val in eval_config.items():
            if key == 'smoothing-config':
                print(f"  {key}: {_format_smoothing_config(val)}")
            else:
                print(f"  {key}: {val}")

        lookback = config['lookback']
        train_window = lookback + horizon

        if len(data) < train_window + horizon:
            print("  ⏭️  SKIPPING: Not enough data")
            skipped += 1
            continue

        smoothing_method = None
        smoothing_window = 3
        smoothing_alpha = 0.2
        if 'smoothing-config' in eval_config:
            sm_cfg = eval_config['smoothing-config']
            if sm_cfg is not None and isinstance(sm_cfg, (list, tuple)) and len(sm_cfg) == 3:
                smoothing_method = sm_cfg[0]
                if sm_cfg[1] is not None:
                    smoothing_window = sm_cfg[1]
                if sm_cfg[2] is not None:
                    smoothing_alpha = sm_cfg[2]

        model_kwargs = {
            'horizon': horizon,
            'lookback': lookback,
            'random_state': 42,
        }
        for key, value in eval_config.items():
            if key not in ['lookback', 'smoothing-config']:
                model_kwargs[key.replace('-', '_')] = value

        try:
            model = LSTMAttentionModel(**model_kwargs)
            result = evaluate_model_config(
                model=model,
                data=data,
                train_window=train_window,
                lookback=lookback,
                horizon=horizon,
                smoothing_method=smoothing_method,
                smoothing_window=smoothing_window,
                smoothing_alpha=smoothing_alpha,
                epochs=30,
                verbose=True
            )

            if 'error' in result:
                print(f"  ❌ ERROR: {result['error']}")
                bad_configs.append(config)
                results.append({'config': config, 'original_config': config, 'mape': float(
                    'inf'), 'error': result['error']})
                continue

            mape = result.get('mape', float('inf'))
            rmse = result.get('rmse', float('nan'))
            train_time = result.get('train_time', 0.0)

            print(
                f"  ✓ MAPE: {mape:.2f}% | RMSE: {rmse:.2f} | Train: {train_time:.1f}s")

            if mape < best_mape_so_far:
                print(f"  🌟 NEW BEST! (Previous: {best_mape_so_far:.2f}%)")
                best_mape_so_far = mape

            if mape >= 500.0 or not np.isfinite(mape):
                bad_configs.append(config)

            results.append({
                'config': eval_config,
                'original_config': config,
                'mape': mape,
                'rmse': rmse,
                'train_time': train_time,
            })
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            bad_configs.append(config)
            continue

    if skipped == len(phase1_configs):
        return ("All configurations were skipped due to insufficient data vs horizon.")

    # PHASE 2
    valid_results = [r for r in results if 'error' not in r and r.get(
        'mape', float('inf')) < float('inf') and r.get('mape', float('inf')) < 50.0]

    valid_results.sort(key=lambda x: x['mape'])
    top_n = min(5, len(valid_results))

    print(f"\n{'='*70}")
    print(
        f"PHASE 2: FULL TRAINING - Retraining top {top_n} configs with {config_epochs} epochs")
    print(f"{'='*70}\n")

    phase2_results = []

    for idx in range(top_n):
        probe_result = valid_results[idx]
        original_config = probe_result.get(
            'original_config', probe_result['config'])
        full_config = original_config.copy()
        full_config['epochs'] = config_epochs

        print(
            f"\n[Full Train {idx+1}/{top_n}] Probe MAPE was {probe_result['mape']:.2f}%")
        print(f"  Retraining with {config_epochs} epochs:")
        for key, val in full_config.items():
            if key == 'smoothing-config':
                print(f"    {key}: {_format_smoothing_config(val)}")
            else:
                print(f"    {key}: {val}")

        lookback = full_config['lookback']
        epochs = full_config['epochs']
        train_window = lookback + horizon

        smoothing_method = None
        smoothing_window = 3
        smoothing_alpha = 0.2
        if 'smoothing-config' in full_config:
            sm_cfg = full_config['smoothing-config']
            if sm_cfg is not None and isinstance(sm_cfg, (list, tuple)) and len(sm_cfg) == 3:
                smoothing_method = sm_cfg[0]
                if sm_cfg[1] is not None:
                    smoothing_window = sm_cfg[1]
                if sm_cfg[2] is not None:
                    smoothing_alpha = sm_cfg[2]

        model_kwargs = {
            'horizon': horizon,
            'lookback': lookback,
            'random_state': 42,
        }
        for key, value in full_config.items():
            if key not in ['lookback', 'smoothing-config']:
                model_kwargs[key.replace('-', '_')] = value

        try:
            model = LSTMAttentionModel(**model_kwargs)
            result = evaluate_model_config(
                model=model,
                data=data,
                train_window=train_window,
                lookback=lookback,
                horizon=horizon,
                smoothing_method=smoothing_method,
                smoothing_window=smoothing_window,
                smoothing_alpha=smoothing_alpha,
                epochs=epochs,
                verbose=True
            )

            if 'error' in result:
                print(f"  ❌ ERROR: {result['error']}")
                continue

            mape = result.get('mape', float('inf'))
            rmse = result.get('rmse', float('nan'))
            train_time = result.get('train_time', 0.0)

            improvement = probe_result['mape'] - mape
            print(
                f"  ✓ MAPE: {mape:.2f}% | RMSE: {rmse:.2f} | Train: {train_time:.1f}s")
            if improvement > 0:
                print(f"    📈 Improved {improvement:.2f}% from probe")
            elif improvement < 0:
                print(
                    f"    📉 Degraded {-improvement:.2f}% from probe (probe was better)")

            if mape < best_mape_so_far:
                print(f"  🌟 NEW BEST! (Previous: {best_mape_so_far:.2f}%)")
                best_mape_so_far = mape

            phase2_results.append({
                'config': full_config,
                'original_config': original_config,
                'mape': mape,
                'rmse': rmse,
                'train_time': train_time,
                'probe_mape': probe_result['mape'],
            })
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            continue

    if phase2_results:
        phase2_results.sort(key=lambda x: x['mape'])
        best_result = phase2_results[0]
        best_config = best_result['config']
        best_mape = best_result['mape']
    else:
        best_config = valid_results[0]['original_config'].copy()
        best_config['epochs'] = config_epochs
        best_mape = valid_results[0]['mape']
        print("\n⚠️  Phase 2 failed, using best from Phase 1")

    print(f"\n{'='*70}")
    print(
        f"✓ Best config found: lookback={best_config['lookback']}, MAPE={best_mape:.2f}%")
    print(f"{'='*70}\n")

    if best_mape < 50.0:
        return best_config
    else:
        return "No suitable config found with MAPE < 50%."


def tune_and_forecast(data: pd.Series, horizon: int, evaluation: bool = True, use_unix_timestamps: bool = False) -> Dict[str, Any]:
    """
    Complete pipeline: tune hyperparameters, train model, generate predictions.

    If evaluation=True: predict last `horizon` points and return metrics.
    If evaluation=False: train on all data and forecast future `horizon` points.
    
    Args:
        use_unix_timestamps: If True, return Unix timestamps (integers); otherwise ISO 8601 strings

    Returns: {predictions, actuals?, metrics?, train_data, smoothed_train_data?, ...}
    """
    best_config = tune_lstm_attention(data, horizon)

    if type(best_config) is str:
        return {'error': best_config}

    print(best_config)

    print(f"\n{'='*60}")
    print("BEST CONFIG FOUND:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    lookback = int(best_config['lookback'])
    train_window = int(lookback + horizon)

    smoothing_method = None
    smoothing_window = 3
    smoothing_alpha = 0.2
    if 'smoothing-config' in best_config:
        sm_cfg = best_config['smoothing-config']
        if sm_cfg is not None and isinstance(sm_cfg, (list, tuple)) and len(sm_cfg) == 3:
            smoothing_method = sm_cfg[0]
            if sm_cfg[1] is not None:
                smoothing_window = sm_cfg[1]
            if sm_cfg[2] is not None:
                smoothing_alpha = sm_cfg[2]

    config_epochs = best_config.get('epochs', 150)
    model_kwargs = {
        'horizon': horizon,
        'lookback': lookback,
        'random_state': 42,
        'epochs': config_epochs,
    }
    for key, value in best_config.items():
        if key not in ['lookback', 'smoothing-config', 'epochs']:
            model_kwargs[key.replace('-', '_')] = value

    if evaluation:
        model = LSTMAttentionModel(**model_kwargs)
        results = evaluate_model_config(
            model=model,
            data=data,
            train_window=train_window,
            lookback=lookback,
            horizon=horizon,
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_alpha=smoothing_alpha,
            epochs=config_epochs,
            verbose=True,
            use_unix_timestamps=use_unix_timestamps
        )
    else:
        train_data = data.iloc[-train_window:] if len(
            data) > train_window else data

        model = LSTMAttentionModel(**model_kwargs)

        train_data_for_training = train_data
        smoothed_train = None
        if smoothing_method:
            try:
                smoothed = apply_smoothing(
                    train_data.sort_index(),
                    method=smoothing_method,
                    window=smoothing_window,
                    alpha=smoothing_alpha,
                )
                if isinstance(smoothed, list):
                    smoothed = pd.Series(smoothed, index=train_data.index)
                train_data_for_training = smoothed
                smoothed_train = list(smoothed.values) if hasattr(
                    smoothed, 'values') else list(smoothed)
                print(
                    f"Training ON SMOOTHED series using method={smoothing_method}")
            except Exception as e:
                print(f"Warning: Smoothing failed, using raw data: {e}")

        print(
            f"Training on last {len(train_data)} points for future prediction...")
        model.train(train_data_for_training, epochs=config_epochs, quiet=False)

        prediction_input = train_data.iloc[-lookback:]

        # Apply smoothing to prediction input if configured
        try:
            prediction_input_smoothed = apply_smoothing(
                prediction_input, method=smoothing_method, window=smoothing_window, alpha=smoothing_alpha) if smoothing_method else prediction_input
        except Exception:
            prediction_input_smoothed = prediction_input

        # Update model if supported
        if model.supports_online_learning():
            try:
                model.update(prediction_input_smoothed)
            except Exception:
                pass

        predictions = model.predict()
        if predictions:
            predictions = [max(-1e9, min(1e9, float(p)))
                           if np.isfinite(p) else 0.0 for p in predictions]

        # Detect granularity and calculate future timestamps
        granularity = detect_granularity(data)
        prediction_timestamps = []
        if granularity is not None and hasattr(data, 'index') and len(data.index) > 0:
            try:
                last_timestamp = data.index[-1]
                prediction_timestamps = calculate_future_timestamps(
                    last_timestamp, granularity, horizon, as_unix=use_unix_timestamps
                )
            except Exception:
                prediction_timestamps = []

        results = {
            'predictions': predictions,
            'prediction_timestamps': prediction_timestamps,
            'granularity': granularity if granularity else None,
            'train_window': train_window,
            'train_data': list(train_data.values) if hasattr(train_data, 'values') else list(train_data),
            'smoothed_train_data': smoothed_train,
            'smoothing_applied': smoothing_method is not None,
        }

    return sanitize_for_json(results)
