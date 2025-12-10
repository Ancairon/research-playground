"""
Universal forecaster - minimal production version.

Handles:
- LSTM-Attention hyperparameter tuning (staged search)
- Single-shot forecasting with smoothing support
"""

import time
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from models.lstm_attention_model import LSTMAttentionModel
from shared_prediction import single_shot_evaluation
from train_utils import train_model
from smoothing import apply_smoothing


class UniversalForecaster:
    """
    Minimal forecasting engine for production use.

    Only includes train_initial() and forecast_step() - 
    no dynamic retraining, backoff, or validation features.
    """

    def __init__(
        self,
        model: LSTMAttentionModel,
        window: int = 30,
        train_window: int = 300,
        smoothing_method: Optional[str] = None,
        smoothing_window: int = 3,
        smoothing_alpha: float = 0.2,
        quiet: bool = False,
    ):
        """
        Initialize forecaster.

        Args:
            model: LSTM-Attention model instance
            window: Inference window size (lookback)
            train_window: Training window size
            smoothing_method: 'moving_average', 'ewma', or None
            smoothing_window: Window for moving average
            smoothing_alpha: Alpha for EWMA
            quiet: Suppress console output
        """
        self.model = model
        self.window = window
        self.train_window = train_window if train_window is not None else window
        self.smoothing_method = smoothing_method
        self.smoothing_window = smoothing_window
        self.smoothing_alpha = smoothing_alpha
        self.quiet = quiet

        # Baseline tracking
        self.baseline_mean = None
        self.baseline_std = None

        # Timing tracking
        self.training_times = []
        self.inference_times = []

    def train_initial(self, data: pd.Series) -> float:
        """
        Train model on initial data.

        Args:
            data: Training data series

        Returns:
            Training time in seconds
        """
        # Apply smoothing if configured
        try:
            if self.smoothing_method:
                data_for_training = apply_smoothing(
                    data,
                    method=self.smoothing_method,
                    window=self.smoothing_window,
                    alpha=self.smoothing_alpha,
                )
            else:
                data_for_training = data
        except Exception:
            data_for_training = data

        # Train model
        res = train_model(self.model, data_for_training, quiet=True)
        train_time = res.get('train_time', 0.0)
        self.training_times.append(train_time)

        # Calculate baseline statistics
        try:
            self.baseline_mean = float(
                res.get('baseline_mean', float(data_for_training.mean())))
            self.baseline_std = float(
                res.get('baseline_std', float(data_for_training.std())))
        except Exception:
            self.baseline_mean = float(data.mean())
            self.baseline_std = float(data.std())

        if not self.quiet:
            print(f"[{self.model.get_model_name()}] Init train t={train_time:.3f}s")
            print(
                f"[Baseline] Mean: {self.baseline_mean:.2f}, Std: {self.baseline_std:.2f}")

        return train_time

    def forecast_step(self, live_data: pd.Series) -> List[float]:
        """
        Generate predictions for horizon steps.

        Args:
            live_data: Current data window (lookback points)

        Returns:
            List of predictions for each horizon step
        """
        # Apply smoothing to input if configured
        try:
            if self.smoothing_method:
                smoothed_live = apply_smoothing(
                    live_data.sort_index(),
                    method=self.smoothing_method,
                    window=self.smoothing_window,
                    alpha=self.smoothing_alpha
                )
            else:
                smoothed_live = live_data
        except Exception:
            smoothed_live = live_data

        # Update model with new data (online learning if supported)
        if self.model.supports_online_learning():
            self.model.update(smoothed_live)

        # Generate prediction
        start_inf = time.time()
        raw_preds = self.model.predict()
        inf_time = time.time() - start_inf
        self.inference_times.append(inf_time)

        # Validate predictions (check for NaN, inf, or extreme values)
        if raw_preds:
            raw_preds = [
                max(-1e9, min(1e9, float(p))) if np.isfinite(p) else 0.0
                for p in raw_preds
            ]
            return [float(p) for p in raw_preds]

        return []


def tune_lstm_attention(data: pd.Series, horizon: int) -> dict:
    """
    Tune LSTM-Attention hyperparameters using staged search.

    IDENTICAL to the original LSTMAttentionTuner's 'auto' mode:
    - Phase 1: Fast probe with 30 epochs on sampled configs (35%, max 50)
    - Phase 2: Full training with 150 epochs on top 5 performers

    Args:
        data: Time series data
        horizon: Number of future steps to predict

    Returns:
        Best hyperparameter configuration dict
    """
    from itertools import product
    import random

    data_size = len(data)
    max_lookback = data_size // 3

    # Fixed config values
    config_epochs = 150  # Full training epochs
    probe_epochs = 30    # Fast probe epochs
    config_dropout = 0.2

    def filter_lookbacks(values):
        """Filter lookback values to fit within data constraints."""
        filtered = [v for v in values if v <=
                    max_lookback and (v + horizon) < data_size]
        if not filtered:
            safe_lookback = min(max_lookback, max(
                60, (data_size - horizon) // 2))
            filtered = [safe_lookback]
        return filtered

    def _format_smoothing_config(val):
        """Format smoothing-config tuple for display."""
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
        """Check if two configs are similar (for skipping bad regions)."""
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

    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = [search_space[k] for k in param_names]
    all_configs = [dict(zip(param_names, v)) for v in product(*param_values)]

    total_possible = len(all_configs)

    # Sample configs: 35% capped at 50 (identical to original)
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

    # PHASE 1: Fast probe
    print(f"{'='*70}")
    print(
        f"PHASE 1: FAST PROBE - Testing {len(phase1_configs)} configs with {probe_epochs} epochs each")
    print(f"{'='*70}\n")

    results = []
    bad_configs = []
    best_mape_so_far = float('inf')

    for i, config in enumerate(phase1_configs, 1):
        # Skip configs similar to bad ones
        if any(_is_similar(config, bc) for bc in bad_configs):
            print(
                f"\n[{i}/{len(phase1_configs)}] Skipping configuration similar to poor-performing region")
            continue

        # Override epochs for probe phase
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
            continue

        # Parse smoothing config
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

        # Build model kwargs (convert hyphenated keys to underscored)
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
            forecaster = UniversalForecaster(
                model,
                window=lookback,
                train_window=train_window,
                quiet=True,
                smoothing_method=smoothing_method,
                smoothing_window=smoothing_window,
                smoothing_alpha=smoothing_alpha
            )
            result = single_shot_evaluation(
                forecaster, data, train_window, lookback, horizon, verbose=True)

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

    # PHASE 2: Full training on top performers
    valid_results = [r for r in results if 'error' not in r and r.get(
        'mape', float('inf')) < float('inf')]

    if not valid_results:
        safe_lookback = min(60, len(data) // 4)
        best_config = {
            'lookback': safe_lookback,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': config_epochs,
            'batch_size': 128,
            'use-differencing': False,
            'scaler-type': 'standard',
        }
        print(
            f"\n⚠️  No valid config found in Phase 1, using fallback: lookback={safe_lookback}")
        return best_config

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
        train_window = lookback + horizon

        # Parse smoothing config
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

        # Build model kwargs
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
            forecaster = UniversalForecaster(
                model,
                window=lookback,
                train_window=train_window,
                quiet=True,
                smoothing_method=smoothing_method,
                smoothing_window=smoothing_window,
                smoothing_alpha=smoothing_alpha
            )
            result = single_shot_evaluation(
                forecaster, data, train_window, lookback, horizon, verbose=True)

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

    # Final result
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

    return best_config


def tune_and_forecast(data: pd.Series, horizon: int, evaluation: bool = True) -> Dict[str, Any]:
    """
    Tune hyperparameters and perform forecast.

    Args:
        data: Time series data
        horizon: Number of future steps to predict
        evaluation: If True, evaluate against held-out actuals

    Returns:
        Dict with predictions, metrics (if evaluation), and metadata
    """
    best_config = tune_lstm_attention(data, horizon)

    print(f"\n{'='*60}")
    print("BEST CONFIG FOUND:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    lookback = best_config['lookback']
    train_window = lookback + horizon

    # Parse smoothing config
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

    # Build model kwargs (convert hyphenated keys to underscored)
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
        # Final evaluation with best config
        model = LSTMAttentionModel(**model_kwargs)
        forecaster = UniversalForecaster(
            model,
            window=lookback,
            train_window=train_window,
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_alpha=smoothing_alpha
        )
        results = single_shot_evaluation(
            forecaster, data, train_window, lookback, horizon, verbose=True)
    else:
        # No evaluation - train on last train_window and predict future
        train_data = data.iloc[-train_window:] if len(
            data) > train_window else data

        model = LSTMAttentionModel(**model_kwargs)
        forecaster = UniversalForecaster(
            model,
            window=lookback,
            train_window=train_window,
            smoothing_method=smoothing_method,
            smoothing_window=smoothing_window,
            smoothing_alpha=smoothing_alpha
        )

        # Apply smoothing if configured
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
        forecaster.train_initial(train_data_for_training)

        # Get last lookback points for prediction
        prediction_input = train_data.iloc[-lookback:]
        predictions = forecaster.forecast_step(prediction_input)

        results = {
            'predictions': predictions,
            'train_window': train_window,
            'train_data': list(train_data.values) if hasattr(train_data, 'values') else list(train_data),
            'smoothed_train_data': smoothed_train,
            'smoothing_applied': smoothing_method is not None,
        }

    return results
