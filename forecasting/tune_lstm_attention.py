#!/usr/bin/env python3
"""
LSTM with Attention Hyperparameter Tuning System

Automatically tests different LSTM-Attention configurations to find optimal parameters
for your time series data. Uses single-shot evaluation to measure accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yaml
import json
import time
import signal
from datetime import datetime
from itertools import product
import argparse
import random

from models import create_model
from universal_forecaster import UniversalForecaster
from shared_prediction import single_shot_evaluation


class LSTMAttentionTuner:
    """Hyperparameter tuner for LSTM with Attention models."""
    
    def __init__(self, csv_file, horizon, train_window, inference_window, max_lookback=None, max_training=None, extra_train=None, max_train_loss=None, extra_train_provided=False, max_time_per_config=None):
        """
        Initialize tuner.
        
        Args:
            csv_file: Path to CSV file
            horizon: Prediction horizon
            train_window: Training window size (default, not used if extra_train is set)
            inference_window: Inference window size (for backwards compatibility)
            max_lookback: Maximum lookback to search (default: data_size // 3)
            max_training: Maximum train window to search (enables train window tuning)
            extra_train: Extra training data beyond lookback+horizon (enforces relationship)
            max_time_per_config: Maximum seconds allowed per config (default: None = no limit)
        """
        self.df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
        self.horizon = horizon
        self.train_window = train_window
        self.inference_window = inference_window
        self.csv_file = csv_file
        self.original_config = {}  # Will be populated with original config values
        self.extra_train = extra_train  # Store extra_train for train_window calculation
        # Whether the extra_train value was explicitly provided by the CLI (if True, keep it fixed)
        self.extra_train_provided = bool(extra_train_provided)
        # Optional training abort threshold used to skip clearly bad configs early
        self.max_train_loss = max_train_loss
        # Maximum time (seconds) allowed per configuration evaluation
        self.max_time_per_config = max_time_per_config
        # Track training times for estimation
        self._training_times = []
        
        # Set max_lookback
        data_size = len(self.df)
        if max_lookback is not None:
            self.max_lookback = max_lookback
        else:
            self.max_lookback = data_size // 3
        
        # Set max_training (for train_window tuning)
        if max_training is not None:
            self.max_training = min(max_training, data_size - horizon)
        else:
            self.max_training = None  # Will not tune train_window
        
        print(f"Loaded {data_size} data points from {csv_file}")
        print(f"Max lookback constrained to: {self.max_lookback} (available data: {data_size})")
        if self.max_training:
            print(f"Train window will be tuned up to: {self.max_training}")
        print(f"Will predict {self.horizon} steps ahead")
        if self.max_time_per_config:
            print(f"Max time per config: {self.max_time_per_config}s")
        
    def _estimate_training_time(self, config) -> float:
        """
        Estimate training time for a configuration based on complexity factors.
        
        Training time scales roughly with:
        - Number of training samples (train_window - lookback)
        - Number of epochs
        - Hidden size (quadratic)
        - Number of layers
        - Batch size (inverse - smaller batches = more iterations)
        
        Returns estimated seconds, or 0 if cannot estimate.
        """
        if not self._training_times:
            return 0  # No data to estimate from yet
        
        # Get config parameters
        lookback = config.get('lookback', config.get('window', 60))
        hidden_size = config.get('hidden_size', config.get('hidden-size', 64))
        num_layers = config.get('num_layers', config.get('num-layers', 2))
        epochs = config.get('epochs', 30)
        batch_size = config.get('batch_size', config.get('batch-size', 128))
        
        # Calculate train_window for this config
        extra_train = config.get('extra-train', config.get('extra_train', self.extra_train or 0))
        if isinstance(extra_train, float) and 0 < extra_train <= 1:
            extra_train = int(extra_train * len(self.df))
        train_window = lookback + self.horizon + int(extra_train or 0)
        
        # Complexity score (relative)
        num_samples = max(1, train_window - lookback)
        complexity = (num_samples * epochs * (hidden_size ** 1.5) * num_layers) / batch_size
        
        # Get average time per complexity unit from historical data
        if len(self._training_times) >= 1:
            avg_time_per_complexity = np.mean([t['time'] / max(1, t['complexity']) 
                                               for t in self._training_times])
            estimated = complexity * avg_time_per_complexity
            return estimated
        
        return 0
    
    def _record_training_time(self, config, actual_time: float):
        """Record actual training time for future estimation."""
        lookback = config.get('lookback', config.get('window', 60))
        hidden_size = config.get('hidden_size', config.get('hidden-size', 64))
        num_layers = config.get('num_layers', config.get('num-layers', 2))
        epochs = config.get('epochs', 30)
        batch_size = config.get('batch_size', config.get('batch-size', 128))
        
        extra_train = config.get('extra-train', config.get('extra_train', self.extra_train or 0))
        if isinstance(extra_train, float) and 0 < extra_train <= 1:
            extra_train = int(extra_train * len(self.df))
        train_window = lookback + self.horizon + int(extra_train or 0)
        
        num_samples = max(1, train_window - lookback)
        complexity = (num_samples * epochs * (hidden_size ** 1.5) * num_layers) / batch_size
        
        self._training_times.append({
            'time': actual_time,
            'complexity': complexity,
            'config': config.copy()
        })
        
        # Keep only last 20 records for estimation
        if len(self._training_times) > 20:
            self._training_times = self._training_times[-20:]
        
    def define_search_space(self, search_type='quick'):
        """
        Define hyperparameter search space.
        
        Args:
            search_type: 'quick', 'balanced', 'exhaustive', or 'auto'
        """
        data_size = len(self.df)
        
        # Use fixed sensible defaults for static parameters (do not respect YAML)
        # The tuner intentionally ignores most YAML defaults so the search
        # explores the full parameter ranges; the original config is only
        # evaluated as a baseline but does not influence the search space.
        config_epochs = 150
        config_batch_size = 128
        config_dropout = 0.2

        def filter_lookbacks(values):
            """Filter lookback values to fit within data and max_lookback constraints."""
            filtered = [v for v in values if v <= self.max_lookback and (v + self.horizon) < data_size]
            if not filtered:
                # If all values filtered out, use a safe default
                safe_lookback = min(self.max_lookback, max(60, (data_size - self.horizon) // 2))
                filtered = [safe_lookback]
            return filtered

        search_space = {}
        search_space['extra-train'] = [0, 0.0005, 0.001]
        
        if search_type == 'auto':
            # Adaptive auto-tuning: adjusts search space based on data size and horizon
            # Generate adaptive lookback range
            min_lookback = max(30, self.horizon // 5)  # At least 1/5 of horizon
            max_lookback_search = min(self.max_lookback, data_size // 3)
            
            # Create 5 evenly spaced lookback values
            if max_lookback_search > min_lookback * 3:
                lookback_values = [
                    min_lookback,
                    min_lookback * 2,
                    (min_lookback + max_lookback_search) // 2,
                    int(max_lookback_search * 0.75),
                    max_lookback_search
                ]
            else:
                lookback_values = [min_lookback, (min_lookback + max_lookback_search) // 2, max_lookback_search]
            
            lookback_values = filter_lookbacks(sorted(set(lookback_values)))
            
            # Scale model complexity with data
            if data_size < 2000:
                hidden_sizes = [32]
            elif data_size < 5000:
                hidden_sizes = [32, 64]
            else:
                hidden_sizes = [32,64]
            
            num_layers = [2] if data_size < 5000 or self.horizon < 1000 else [2, 3]
            dropout_values = [config_dropout]  # Use config value
            epochs = [config_epochs]  # Use config value
            batch_sizes = [128, 256]  # Search common batch sizes
            
            search_space = {
                'lookback': lookback_values,
                'hidden_size': hidden_sizes,
                'num_layers': num_layers,
                'dropout': dropout_values,
                'learning_rate': [0.001, 0.0005],  # Search both common values
                'epochs': epochs,
                'batch_size': batch_sizes,  # Search batch sizes
                'use-differencing': [False, True],  # Try both differencing modes
                'scaler-type': ['none', 'standard', 'robust'],
            }
            # Include extra-train in the search space when it's not fixed by CLI
            if self.extra_train_provided:
                # Keep a single fixed value
                search_space['extra-train'] = [self.extra_train]
            
            return search_space
        
        elif search_type == 'quick':
            # Ultra-fast: test 2-3 lookback values
            base_lookbacks = [60, 120]
            
            search_space = {
                'lookback': filter_lookbacks(base_lookbacks),
                'hidden_size': [64, 128],
                'num_layers': [2],
                'dropout': [config_dropout],
                'learning_rate': [0.001],  # Quick mode: just one learning rate
                'epochs': [config_epochs],
                'batch_size': [128],  # Quick: just one batch size
                'use-differencing': [False, True],
                'scaler-type': ['none', 'standard', 'robust'],
            }
            if self.extra_train_provided:
                search_space['extra-train'] = [self.extra_train]
            
            return search_space
        
        elif search_type == 'balanced':
            # Balanced: generate lookback values that scale to max_lookback
            # Start with base values, then add logarithmically spaced values up to max_lookback
            base_lookbacks = [60, 120, 240, 480]
            
            # Add more values scaling up to max_lookback
            if self.max_lookback > 480:
                # Add intermediate values between 480 and max_lookback
                current = 800
                while current < self.max_lookback:
                    base_lookbacks.append(current)
                    current = int(current * 1.5)  # Geometric progression
                
                # Always include max_lookback itself
                if base_lookbacks[-1] != self.max_lookback:
                    base_lookbacks.append(self.max_lookback)
            
            search_space = {
                'lookback': filter_lookbacks(sorted(set(base_lookbacks))),
                'hidden_size': [64, 128],
                'num_layers': [2],
                'dropout': [config_dropout],
                'learning_rate': [0.001, 0.0005],  # Balanced: test 2 learning rates
                'epochs': [config_epochs],
                'batch_size': [64, 128],  # Balanced: test 2 batch sizes
                'use-differencing': [False, True],
                'scaler-type': ['none', 'standard', 'robust'],
            }
            if self.extra_train_provided:
                search_space['extra-train'] = [self.extra_train]
            
            return search_space
        
        else:  # exhaustive
            # Comprehensive search: generate more lookback values that scale to max_lookback
            base_lookbacks = [30, 60, 120, 240, 480]
            
            # Add more values scaling up to max_lookback
            if self.max_lookback > 480:
                # Add intermediate values between 480 and max_lookback
                current = 800
                while current < self.max_lookback:
                    base_lookbacks.append(current)
                    current = int(current * 1.4)  # Slightly denser than balanced mode
                
                # Always include max_lookback itself
                if base_lookbacks[-1] != self.max_lookback:
                    base_lookbacks.append(self.max_lookback)
            
            search_space = {
                'lookback': filter_lookbacks(sorted(set(base_lookbacks))),
                'hidden_size': [32, 64, 128, 256],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3],  # Exhaustive tries multiple dropout values
                'learning_rate': [0.001, 0.0005, 0.0001],  # Exhaustive: test 3 learning rates
                'epochs': [config_epochs],  # Use config epochs with early stopping
                'batch_size': [32, 64, 128, 256],  # Exhaustive: search all common batch sizes
                'use-differencing': [False, True],
                'scaler-type': ['none', 'standard', 'robust'],
            }
            if self.extra_train_provided:
                search_space['extra-train'] = [self.extra_train]
            
            return search_space
    
    def evaluate_config(self, config, verbose=False):
        """
        Evaluate a single configuration using UniversalForecaster.
        
        Args:
            config: Dictionary of hyperparameters
            verbose: Whether to print progress
            
        Returns:
            Dictionary with metrics (MAPE, MBE, train_time, etc.)
        """
        try:
            # Check estimated time and skip if too long
            if self.max_time_per_config and len(self._training_times) >= 2:
                estimated_time = self._estimate_training_time(config)
                if estimated_time > self.max_time_per_config * 1.5:  # 50% margin
                    if verbose:
                        print(f"  ⏭️  SKIPPING: Estimated time {estimated_time:.0f}s > limit {self.max_time_per_config}s")
                    return {
                        'error': f'Skipped: estimated time {estimated_time:.0f}s exceeds limit',
                        'mape': float('inf'),
                        'config': config,
                        'skipped': True,
                    }
            
            # Get lookback from config (support 'window' as alias for backwards compatibility)
            lookback = config.get('lookback', config.get('window', 60))
            
            # Determine extra_train: prefer config-specified 'extra-train' if present (from search space),
            # otherwise fall back to the tuner-level extra_train (CLI-provided), and finally to config train_window.
            def _to_abs_extra(v):
                # Convert fractional extra-train (0-1) to absolute count using dataframe size
                if v is None:
                    return None
                try:
                    fv = float(v)
                except Exception:
                    return None
                data_len = len(self.df)
                if 0 < fv <= 1:
                    return int(max(1, fv * data_len))
                return int(fv)

            extra_from_config = None
            if 'extra-train' in config:
                extra_from_config = config.get('extra-train')
            elif 'extra_train' in config:
                extra_from_config = config.get('extra_train')

            if extra_from_config is not None:
                et_abs = _to_abs_extra(extra_from_config)
                train_window = lookback + self.horizon + et_abs
                if verbose:
                    print(f"  Calculated train_window: {train_window} = {lookback} (lookback) + {self.horizon} (horizon) + {et_abs} (extra_train from config)")
            elif self.extra_train is not None:
                et_abs = _to_abs_extra(self.extra_train)
                train_window = lookback + self.horizon + et_abs
                if verbose:
                    print(f"  Calculated train_window: {train_window} = {lookback} (lookback) + {self.horizon} (horizon) + {et_abs} (extra_train CLI)")
            else:
                # Fallback to config value if extra_train not set (backwards compatibility)
                train_window = config.get('train_window', config.get('train-window', lookback + self.horizon))
            
            # Validate train_window is sufficient
            min_train_window = lookback + self.horizon
            if train_window < min_train_window:
                if verbose:
                    print(f"  WARNING: train_window {train_window} < minimum {min_train_window}, using minimum")
                train_window = min_train_window
            
            # Prepare config for model - use forecast_main's parameter names
            model_kwargs = {
                'horizon': self.horizon,
                'lookback': lookback,
                'random_state': 42,
            }
            
            # Do NOT copy preserved parameters from the YAML into model kwargs.
            # The tuner intentionally ignores most YAML defaults so that the
            # search explores the configured search_space. The original config
            # is only evaluated as a baseline earlier, but it should not force
            # model parameter values during the search.
            
            # Add all config parameters (from tuning search space)
            for key, value in config.items():
                if key not in ['lookback', 'window', 'train_window', 'train-window', 'extra-train', 'extra_train']:  # Skip these - they're for tuner, not model
                    # Convert hyphenated keys to underscores
                    if isinstance(key, str) and '-' in key:
                        key = key.replace('-', '_')
                    model_kwargs[key] = value
            
            if verbose:
                print(f"  Model parameters: {model_kwargs}")
            
            # Create model first (same as forecast_main.py)
            model = create_model('lstm-attention', **model_kwargs)
            
            # Create UniversalForecaster with the model (same as forecast_main.py)
            forecaster = UniversalForecaster(
                model=model,
                window=lookback,
                train_window=train_window,
                max_train_loss=self.max_train_loss,
                # Allow UniversalForecaster (and underlying train helper) to enforce
                # a maximum per-config training time when provided by the tuner.
                max_train_seconds=self.max_time_per_config
            )
            
            # Use shared single-shot evaluation logic
            eval_start = time.time()
            result = single_shot_evaluation(
                forecaster=forecaster,
                data=self.df['value'],
                train_window=train_window,
                lookback=lookback,
                horizon=self.horizon,
                verbose=verbose
            )
            eval_time = time.time() - eval_start
            
            # Record training time for future estimation
            self._record_training_time(config, result.get('train_time', eval_time))
            
            # Add config to result
            result['config'] = config
            
            # Store only first 5 predictions/actuals to save memory
            if 'predictions' in result:
                result['predictions'] = result['predictions'][:5]
            if 'actuals' in result:
                result['actuals'] = result['actuals'][:5]
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'mape': float('inf'),
                'config': config,
            }
    
    def run_tuning(self, search_space, max_time_per_config=300, use_adaptive=True, initial_best_mape=None):
        """
        Run hyperparameter tuning with two-phase adaptive search.
        
        Phase 1 (Exploration): Sample the search space to find promising regions
        Phase 2 (Exploitation): Refine the best configurations with nearby parameter values
        
        Args:
            search_space: Dictionary of parameter ranges
            max_time_per_config: Skip configs that would take longer than this
            use_adaptive: Enable two-phase adaptive search (default: True)
            
        Returns:
            List of results, sorted by MAPE
        """
        # Generate all possible combinations
        param_names = list(search_space.keys())
        param_values = [search_space[k] for k in param_names]
        all_configs = [dict(zip(param_names, v)) for v in product(*param_values)]
        
        total_possible = len(all_configs)
        
        # For adaptive mode, sample Phase 1 (sparse exploration)
        # For exhaustive mode, test everything
        if use_adaptive:
            # Sample ~25-33% of the space, at least 10 configs, at most 50
            phase1_size = max(10, min(50, total_possible // 3))
            import random
            random.seed(42)  # Reproducibility
            phase1_configs = random.sample(all_configs, min(phase1_size, total_possible))
        else:
            # Exhaustive: test everything
            phase1_configs = all_configs
        
        print(f"\n{'='*70}")
        print(f"LSTM-Attention Hyperparameter Tuning")
        if use_adaptive:
            print(f"Mode: ADAPTIVE (Sparse Exploration → Dense Refinement)")
            print(f"Phase 1 configurations: {len(phase1_configs)} (sampled from {total_possible})")
        else:
            print(f"Mode: EXHAUSTIVE")
            print(f"Phase 1 configurations: {len(phase1_configs)}")
        print(f"{'='*70}\n")
        
        results = []
        # Initialize best_mape_so_far from caller-provided original MAPE when available
        try:
            best_mape_so_far = float(initial_best_mape) if initial_best_mape is not None else float('inf')
        except Exception:
            best_mape_so_far = float('inf')
        
        # PHASE 1: EXPLORATION - Sample parameter space
        print(f"{'='*70}")
        if use_adaptive:
            print("PHASE 1: SPARSE EXPLORATION - Sampling parameter space")
        else:
            print("PHASE 1: EXHAUSTIVE SEARCH - Testing all configurations")
        print(f"{'='*70}\n")
        
        bad_configs = []

        def _is_similar(a, b, min_matches=None):
            keys = set(a.keys()) & set(b.keys())
            if min_matches is None:
                min_matches = max(1, len(keys) // 2)
            matches = sum(1 for k in keys if a.get(k) == b.get(k))
            return matches >= min_matches

        for i, config in enumerate(phase1_configs, 1):
            if any(_is_similar(config, bc) for bc in bad_configs):
                print(f"\n[{i}/{len(phase1_configs)}] Skipping configuration similar to poor-performing region")
                continue

            print(f"\n[{i}/{len(phase1_configs)}] Testing configuration:")
            for key, val in config.items():
                print(f"  {key}: {val}")

            result = self.evaluate_config(config, verbose=True)

            if 'error' in result:
                print(f"  ❌ ERROR: {result['error']}")
                bad_configs.append(config)
                results.append(result)
                continue

            print(f"  ✓ MAPE: {result['mape']:.2f}% | "
                  f"RMSE: {result['rmse']:.2f} | "
                  f"Train: {result['train_time']:.1f}s")

            # Track best result for progress indication
            if result['mape'] < best_mape_so_far:
                print(f"  🌟 NEW BEST! (Previous: {best_mape_so_far:.2f}%)")
                best_mape_so_far = result['mape']

            try:
                mape_val = float(result.get('mape', float('inf')))
            except Exception:
                mape_val = float('inf')
            if mape_val >= 500.0 or not np.isfinite(mape_val):
                bad_configs.append(config)

            results.append(result)
        
        # PHASE 2: EXPLOITATION - Refine top performers
        if use_adaptive and len(results) > 0:
            # Sort by MAPE to identify top performers
            results.sort(key=lambda x: x['mape'])
            
            # Take top 3 configs (or fewer if not enough results)
            top_n = min(3, len([r for r in results if 'error' not in r]))
            if top_n > 0:
                print(f"\n{'='*70}")
                print(f"PHASE 2: EXPLOITATION - Refining top {top_n} configurations")
                print(f"{'='*70}\n")
                
                refinement_configs = []
                for idx in range(top_n):
                    best_config = results[idx]['config']
                    print(f"\nRefining config #{idx+1} (MAPE: {results[idx]['mape']:.2f}%):")
                    for key, val in best_config.items():
                        print(f"  {key}: {val}")
                    
                    # Generate refinements around this config
                    refinements = self._generate_refinements(best_config, search_space)
                    refinement_configs.extend(refinements)
                
                # Remove duplicates (configs we already tested)
                tested_configs = {self._config_to_key(r['config']) for r in results}
                new_refinements = [c for c in refinement_configs 
                                 if self._config_to_key(c) not in tested_configs]
                
                print(f"\nTesting {len(new_refinements)} refinement configurations...")
                
                for i, config in enumerate(new_refinements, 1):
                    print(f"\n[Refinement {i}/{len(new_refinements)}]:")
                    for key, val in config.items():
                        print(f"  {key}: {val}")
                    
                    result = self.evaluate_config(config, verbose=True)
                    results.append(result)
                    
                    if 'error' in result:
                        print(f"  ❌ ERROR: {result['error']}")
                    else:
                        print(f"  ✓ MAPE: {result['mape']:.2f}% | "
                              f"RMSE: {result['rmse']:.2f} | "
                              f"Train: {result['train_time']:.1f}s")
                        
                        if result['mape'] < best_mape_so_far:
                            print(f"  🌟 NEW BEST! (Previous: {best_mape_so_far:.2f}%)")
                            best_mape_so_far = result['mape']
        
        # Sort by MAPE
        results.sort(key=lambda x: x['mape'])
        
        return results
    
    def _config_to_key(self, config):
        """Convert config dict to hashable key for deduplication."""
        return tuple(sorted(config.items()))
    
    def _generate_refinements(self, base_config, search_space):
        """
        Generate refinement configurations around a good base config.
        
        For numeric parameters, try values between the base and its neighbors.
        For categorical parameters, keep them fixed.
        """
        refinements = []
        
        # For each parameter, try intermediate values
        for param, base_value in base_config.items():
            if param not in search_space:
                continue
                
            possible_values = search_space[param]
            
            # Skip if only one value or not numeric
            if len(possible_values) <= 1:
                continue
            
            # Find base_value position in search space
            try:
                base_idx = possible_values.index(base_value)
            except ValueError:
                continue
            
            # Generate intermediate values
            new_values = []
            
            # Try value between base and next larger (if exists)
            if base_idx < len(possible_values) - 1:
                next_val = possible_values[base_idx + 1]
                if isinstance(base_value, (int, float)) and isinstance(next_val, (int, float)):
                    # For lookback, try midpoint
                    if param == 'lookback':
                        mid = (base_value + next_val) // 2
                        if mid != base_value and mid != next_val:
                            new_values.append(mid)
                    # For learning_rate, try log-scale midpoint
                    elif param == 'learning_rate':
                        import math
                        log_mid = math.exp((math.log(base_value) + math.log(next_val)) / 2)
                        if log_mid != base_value and log_mid != next_val:
                            new_values.append(log_mid)
            
            # Try value between base and previous smaller (if exists)
            if base_idx > 0:
                prev_val = possible_values[base_idx - 1]
                if isinstance(base_value, (int, float)) and isinstance(prev_val, (int, float)):
                    if param == 'lookback':
                        mid = (base_value + prev_val) // 2
                        if mid != base_value and mid != prev_val and mid not in new_values:
                            new_values.append(mid)
                    elif param == 'learning_rate':
                        import math
                        log_mid = math.exp((math.log(base_value) + math.log(prev_val)) / 2)
                        if log_mid != base_value and log_mid != prev_val and log_mid not in new_values:
                            new_values.append(log_mid)
            
            # Create new configs with refined values
            for new_val in new_values:
                new_config = base_config.copy()
                new_config[param] = new_val
                refinements.append(new_config)
        
        return refinements
    
    def save_results(self, results, output_file):
        """Save tuning results to file."""
        # Prepare data for saving
        data_to_save = {
            'timestamp': datetime.now().isoformat(),
            'csv_file': self.csv_file,
            'horizon': self.horizon,
            'train_window': self.train_window,
            'window': self.inference_window,
            'total_configs': len(results),
            'results': results
        }
        
        # Determine format from extension
        _, ext = os.path.splitext(output_file)
        
        if ext == '.yaml' or ext == '.yml':
            with open(output_file, 'w') as f:
                yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False)
        else:  # JSON
            with open(output_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def print_summary(self, results, top_n=5):
        """Print summary of top configurations."""
        print(f"\n{'='*70}")
        print(f"TOP {top_n} CONFIGURATIONS (by MAPE)")
        print(f"{'='*70}\n")
        
        for i, result in enumerate(results[:top_n], 1):
            if 'error' in result:
                continue
            
            is_original = result['config'].get('_original', False)
            marker = " [ORIGINAL CONFIG]" if is_original else ""
                
            print(f"#{i} - MAPE: {result['mape']:.2f}%{marker}")
            print(f"   Config:")
            for key, val in result['config'].items():
                if key != '_original':  # Don't print the marker in config
                    print(f"     {key}: {val}")
            print(f"   Metrics:")
            print(f"     RMSE: {result['rmse']:.2f}")
            print(f"     MBE: {result['mbe']:.2f}")
            print(f"     Train Time: {result['train_time']:.1f}s")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='LSTM-Attention Hyperparameter Tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick tuning with fractional parameters (20% horizon, 30% extra training)
  python tune_lstm_attention.py --csv data.csv --horizon 0.2 --extra-train 0.3 --mode quick
  
  # Auto tuning with absolute values (predict 1000 points, 2000 extra training)
  python tune_lstm_attention.py --csv data.csv --horizon 1000 --extra-train 2000 --mode auto
  
  # Load from config file and override horizon
  python tune_lstm_attention.py --config config.yaml --horizon 0.15 --mode balanced
  
Note:
  - Values 0-1 are treated as fractions of dataset size
  - Values >1 are treated as absolute point counts
  - train_window = lookback + horizon + extra_train (tuner searches for optimal lookback)
        """
    )
    
    parser.add_argument('--csv', '--csv-file', dest='csv_file', 
                        help='Path to CSV file with time series data')
    parser.add_argument('--config', help='Path to YAML config file (alternative to --csv)')
    parser.add_argument('--horizon', type=float, default=0.2,
                        help='Forecast horizon - absolute value or fraction (0-1) of dataset size (default: 0.2 = 20%% of data)')
    parser.add_argument('--extra-train', type=float, default=None,
                        help='Extra training data beyond lookback+horizon - absolute value or fraction (0-1) of dataset size (default: None = use tuner search space). Total train_window = lookback + horizon + extra_train')
    parser.add_argument('--mode', choices=['quick', 'balanced', 'exhaustive', 'auto'],
                        default='auto',
                        help='Tuning mode (default: auto)')
    parser.add_argument('--output', default=None,
                        help='Output file for results (default: same as config file or lstm_attention_tuning_results.yaml)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save the full tuning results to a file. If not set, only the best config YAML may be saved.')
    parser.add_argument('--overwrite-config', action='store_true',
                        help='Overwrite the original config file if a better configuration is found (default: create new file)')
    parser.add_argument('--max-train-loss', type=float, default=None,
                        help='Optional max_train_loss to abort training early for poor configs')
    parser.add_argument('--max-time', type=int, default=120,
                        help='Maximum seconds per config before skipping (default: 120s). Set to 0 to disable.')
    
    args = parser.parse_args()
    
    # Load config if specified
    original_config = {}
    if args.config:
        with open(args.config, 'r') as f:
            original_config = yaml.safe_load(f)
        
        # Store original MAPE if it exists (for comparison later)
        original_mape = None
        if '# MAPE' in original_config:
            mape_str = str(original_config['# MAPE']).replace('%', '').strip()
            try:
                original_mape = float(mape_str)
            except ValueError:
                pass
        
        # If csv-file is not specified in config, infer it from config filename
        if 'csv-file' not in original_config and 'csv_file' not in original_config:
            # Extract basename without extension from config path
            config_basename = os.path.splitext(os.path.basename(args.config))[0]
            # Infer CSV filename (csv directory is at repo root)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            inferred_csv = os.path.join(repo_root, 'csv', f"{config_basename}.csv")
            original_config['csv-file'] = inferred_csv
            print(f"[Config] Inferred CSV file from config name: {inferred_csv}")
        
        # If output not specified, decide based on --overwrite-config flag
        if args.output is None:
            if args.overwrite_config:
                # Overwrite original config if better is found
                args.output = args.config
                print(f"[Config] Will overwrite original config: {args.output} (if improvement found)")
            else:
                # Generate output filename: config_name_tuning_results.yaml
                config_basename = os.path.splitext(os.path.basename(args.config))[0]
                args.output = f"{config_basename}_tuning_results.yaml"
                print(f"[Config] Will save results to: {args.output} (NOT overwriting original config)")
                print(f"[Config] Use --overwrite-config to overwrite the original config file")
        
        # Override with config values, but prefer explicit CLI arguments when provided.
        # Detect whether the user passed a specific CLI flag (so CLI wins over config file).
        def _cli_provided(flag_name):
            # Look for --flag_name in sys.argv (simple heuristic that covers common usage)
            return any(arg.startswith(f"--{flag_name}") for arg in sys.argv[1:])

        if 'csv-file' in original_config and not _cli_provided('csv') and not _cli_provided('csv-file'):
            args.csv_file = original_config['csv-file']
        if 'horizon' in original_config and not _cli_provided('horizon'):
            args.horizon = original_config['horizon']
        if 'train-window' in original_config and not _cli_provided('train-window') and not _cli_provided('train_window'):
            args.train_window = original_config['train-window']
        if 'window' in original_config and not _cli_provided('window'):
            args.window = original_config['window']
        if 'max-lookback' in original_config and args.max_lookback is None and not _cli_provided('max-lookback'):
            args.max_lookback = original_config['max-lookback']
        
        # Report preserved parameters
        # Note: 'use-differencing' is intentionally NOT preserved here so it
        # remains tunable even when present in the config.
        preserved_params = {}
        for key in ['scaler-type', 'bias-correction']:
            if key in original_config:
                preserved_params[key] = original_config[key]
        
        if preserved_params:
            print(f"[Config] Preserved parameters (not tuned): {preserved_params}")
        
        print(f"Loaded configuration from: {args.config}")
    
    # Load CSV to get data size for fraction calculations
    df = pd.read_csv(args.csv_file, parse_dates=['timestamp'], index_col='timestamp')
    data_size = len(df)
    
    # Helper function to convert fraction to absolute value
    def to_absolute(value, data_size, param_name):
        """Convert fraction (0-1) to absolute value, or keep absolute if >1."""
        if value is None:
            return None
        if 0 < value <= 1:
            absolute = int(value * data_size)
            print(f"[Param] {param_name}: {value} (fraction) → {absolute} (absolute, data_size={data_size})")
            return absolute
        else:
            return int(value)
    
    # Convert fractions to absolute values (only when provided)
    args.horizon = to_absolute(args.horizon, data_size, 'horizon')
    if args.extra_train is not None:
        args.extra_train = to_absolute(args.extra_train, data_size, 'extra_train')
    else:
        # Leave as None to indicate 'not provided' so tuner may use search-space defaults
        args.extra_train = None
    
    # Calculate max_lookback 
    # We cap it at data_size/3 for sanity (to leave room for horizon and avoid overfitting)
    args.max_lookback = data_size // 3
    print(f"[Param] max_lookback: {args.max_lookback} (data_size/3={data_size//3})")
    
    # Calculate train_window range for tuning
    # Minimum: lookback + horizon + extra_train (where lookback is small)
    # Maximum: max_lookback + horizon + extra_train (where lookback is at its max)
    # Treat None extra_train as 0 for range calculations (it means "not provided")
    args.max_training = args.max_lookback + args.horizon + (args.extra_train or 0)
    print(f"[Param] max_training: {args.max_training} (max_lookback + horizon + extra_train)")
    
    # For the initial tuner setup, use a reasonable default train_window
    # If extra_train is None (not provided), treat it as 0 for this initial heuristic
    args.train_window = args.horizon + (args.extra_train or 0) + (args.max_lookback // 2)
    args.window = args.max_lookback  # Inference window = max lookback
    
    print(f"\n[Tuning Strategy]")
    print(f"  Horizon: {args.horizon} points (what to predict)")
    print(f"  Extra training: {args.extra_train} points (beyond lookback+horizon)")
    print(f"  Max lookback: {args.max_lookback} points (tuner will search 0-{args.max_lookback})")
    print(f"  Train window range: {args.horizon} to {args.max_training} points")
    print(f"  Formula: train_window = lookback + horizon + extra_train\n")
    
    # Set default output if still not specified
    if args.output is None:
        args.output = 'lstm_attention_tuning_results.yaml'
    
    if not args.csv_file:
        print("ERROR: Must specify --csv or --config")
        sys.exit(1)
    
    # Initialize tuner
    tuner = LSTMAttentionTuner(
        csv_file=args.csv_file,
        horizon=args.horizon,
        train_window=args.train_window,
        inference_window=args.window,
        max_lookback=args.max_lookback,
        max_training=args.max_training,
        extra_train=args.extra_train,
        max_train_loss=args.max_train_loss,
        max_time_per_config=args.max_time if args.max_time > 0 else None
    )
    
    # Store original config for later use
    tuner.original_config = original_config
    
    # Evaluate original config first as baseline (if it exists and has required params).
    # Be permissive about which keys the original config uses (lookback/window/_-variants).
    original_result = None
    if original_config:
        # Extract likely tuning/model keys from the original config and pass them
        # through for evaluation. Support both hyphenated and underscored keys.
        possible_keys = [
            'lookback', 'window', 'hidden-size', 'hidden_size', 'num-layers', 'num_layers',
            'dropout', 'learning-rate', 'learning_rate', 'epochs', 'batch-size', 'batch_size',
            'use-differencing', 'use_differencing', 'learning-rate'
        ]
        original_eval_config = {}
        for key in possible_keys:
            if key in original_config:
                # Keep original key naming (evaluate_config handles hyphens)
                original_eval_config[key] = original_config[key]

        # Check if we have the minimum required parameters for a valid baseline
        # These must be ACTUALLY present in config, not inferred
        has_lookback = any(k in original_config for k in ['lookback', 'window'])
        has_model_params = any(k in original_config for k in ['hidden-size', 'hidden_size', 'num-layers', 'num_layers'])
        
        if has_lookback and has_model_params:
            print("\n" + "="*70)
            print("EVALUATING ORIGINAL CONFIG (Baseline)")
            print("="*70)
            print(f"Original config parameters: {original_eval_config}")
            original_result = tuner.evaluate_config(original_eval_config, verbose=True)

            if 'error' not in original_result:
                print(f"\n✓ Original config baseline:")
                print(f"  MAPE: {original_result['mape']:.2f}%")
                print(f"  RMSE: {original_result['rmse']:.2f}")
                print(f"  Train Time: {original_result['train_time']:.1f}s")
                # Update original_mape with actual measured value
                original_mape = original_result['mape']
            else:
                print(f"\n✗ Original config evaluation failed: {original_result.get('error')}")
            print("="*70 + "\n")
        else:
            missing = []
            if not has_lookback:
                missing.append("lookback/window")
            if not has_model_params:
                missing.append("hidden-size/num-layers")
            found_keys = [k for k in possible_keys if k in original_config]
            print(f"\n[Baseline] Skipping - config missing required model params: {', '.join(missing)}")
            if found_keys:
                print(f"[Baseline] Found: {found_keys}")
            else:
                print(f"[Baseline] No tunable model params found in config")
            print()
    
    # Define search space
    search_space = tuner.define_search_space(args.mode)
    
    print(f"\nSearch mode: {args.mode}")
    print("Parameter ranges:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")
    
    # Run tuning (seed with original MAPE if available so we don't incorrectly
    # treat worse configs as "NEW BEST" during phase 1)
    results = tuner.run_tuning(search_space, initial_best_mape=original_mape)
    
    # Add original config result to results list for comparison (if it exists)
    if original_result is not None and 'error' not in original_result:
        # Mark it as the original config
        original_result['config']['_original'] = True
        results.append(original_result)
        # Re-sort results by MAPE
        results.sort(key=lambda x: x.get('mape', float('inf')))
    
    # Print summary
    tuner.print_summary(results, top_n=5)
    
    # Save results (only if user explicitly requested saving or provided --output)
    # Detect whether --output was supplied on the command line so we don't save
    # results by default when the user didn't ask for them.
    output_was_provided = any(a.startswith('--output') for a in sys.argv[1:])
    results_output = args.output
    if args.config and args.overwrite_config and os.path.abspath(args.output) == os.path.abspath(args.config):
        config_basename = os.path.splitext(os.path.basename(args.config))[0]
        results_output = f"{config_basename}_tuning_results.yaml"
        # Only inform about results file when we will actually write it
        if args.save_results or output_was_provided:
            print(f"[Config] --overwrite-config requested; writing tuning RESULTS to separate file: {results_output}")

    if args.save_results or output_was_provided:
        tuner.save_results(results, results_output)
    else:
        print("Skipping saving tuning results file (use --save-results or --output to write).")

    # Ensure best_config_file exists to avoid UnboundLocalError in error paths
    best_config_file = None
    
    # Save best config as ready-to-use YAML (only if we have at least one valid result)
    # Find best non-error result
    best = None
    for r in results:
        if r and 'error' not in r:
            best = r
            break
    if best is not None:
        best_mape = best['mape']
        is_best_original = best['config'].get('_original', False)
        
        # Check if we should save (don't save if original is best, or if we didn't improve)
        should_save = True
        
        if is_best_original:
            # Original config is the best - don't overwrite
            should_save = False
            print(f"\n{'='*70}")
            print(f"✓ Original config is ALREADY OPTIMAL!")
            print(f"   Original MAPE: {best_mape:.2f}%")
            print(f"   No tuned config beat the original")
            print(f"   Config file will NOT be overwritten")
            print(f"{'='*70}\n")
        elif original_mape is not None:
            if best_mape >= original_mape:
                should_save = False
                print(f"\n{'='*70}")
                print(f"⚠️  Tuning did NOT improve over original config")
                print(f"   Original MAPE: {original_mape:.2f}%")
                print(f"   Best new MAPE: {best_mape:.2f}%")
                print(f"   Config file will NOT be overwritten")
                print(f"{'='*70}\n")
            else:
                print(f"\n{'='*70}")
                print(f"✅ Tuning IMPROVED over original config!")
                print(f"   Original MAPE: {original_mape:.2f}%")
                print(f"   Best new MAPE: {best_mape:.2f}%")
                print(f"   Improvement: {original_mape - best_mape:.2f}% reduction")
                print(f"   Config file will be updated")
                print(f"{'='*70}\n")
        
        if should_save:
            # Calculate the actual train_window based on best config using the formula
            best_lookback = best['config']['lookback']
            if args.extra_train is not None:
                best_train_window = best_lookback + args.horizon + args.extra_train
            else:
                best_train_window = best_lookback + args.horizon
            
            # Safety check: If output file is the same as input config, only overwrite if we improved
            best_config_file = args.output
            if args.config and os.path.abspath(args.output) == os.path.abspath(args.config):
                if original_mape is None or best_mape < original_mape:
                    print(f"[Config] Overwriting original config with improved results")
                else:
                    print(f"[Config] ERROR: Refusing to overwrite config - no improvement!")
                    should_save = False
            
            if not should_save:
                # Nothing to save, exit early (but avoid leaving best_config_file undefined)
                print("No config was saved.")
                return
            
            best_config = {
                '# LSTM-Attention Best Configuration': None,
                '# Generated': datetime.now().isoformat(),
                '# MAPE': f"{best['mape']:.2f}%",
                '# RMSE': f"{best['rmse']:.2f}",
                '# Train Time': f"{best['train_time']:.1f}s",
                'model': 'lstm-attention',
                'single-shot': True,
                'train-window': best_train_window,
                'horizon': args.horizon,
            }
            
            # Add best hyperparameters
            for key, val in best['config'].items():
                # Skip internal markers
                if key == '_original':
                    continue
                # Convert underscores to hyphens for YAML consistency
                yaml_key = key.replace('_', '-')
                best_config[yaml_key] = val
            
            # Preserve original config parameters that weren't tuned
            preserve_keys = ['scaler-type', 'bias-correction', 'use-differencing', 'prediction-smoothing']
            if hasattr(tuner, 'original_config'):
                for key in preserve_keys:
                    if key in tuner.original_config and key not in best_config:
                        best_config[key] = tuner.original_config[key]
            
            # Add server config
            best_config['server-port'] = 5000
            
            with open(best_config_file, 'w') as f:
                # Write comments manually since yaml.dump doesn't preserve them well
                f.write(f"# LSTM-Attention Best Configuration\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# MAPE: {best['mape']:.2f}%\n")
                f.write(f"# RMSE: {best['rmse']:.2f}\n")
                f.write(f"# Train Time: {best['train_time']:.1f}s\n\n")
                
                # Write the actual config
                config_to_write = {k: v for k, v in best_config.items() if not k.startswith('#')}
                yaml.dump(config_to_write, f, default_flow_style=False, sort_keys=False)
            
            print(f"\n✓ Best config saved to: {best_config_file}")
        
        # Print best config summary
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION:")
        print(f"{'='*70}")
        print(f"\nMAPE: {best['mape']:.2f}%")
        print(f"RMSE: {best['rmse']:.2f}")
        print(f"Train Time: {best['train_time']:.1f}s")
        print("\nParameters:")
        for key, val in best['config'].items():
            print(f"  {key}: {val}")
        print()
        # If a config file was written, print how to run it; otherwise, note that
        # the best configuration was not saved to disk.
        try:
            if should_save:
                print(f"You can now run: python forecast_main.py --config {best_config_file}")
            else:
                print("Best configuration not saved to disk. To save it, re-run with --overwrite-config or specify --output <file>")
        except Exception:
            # Be defensive: avoid crashes if best_config_file isn't defined
            print("Best configuration computed; provide --output or --overwrite-config to save it to a file.")
        print()


if __name__ == '__main__':
    main()
