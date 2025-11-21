#!/usr/bin/env python3
"""
LSTM Hyperparameter Tuning System

Automatically tests different LSTM configurations to find optimal parameters
for your time series data. Uses single-shot evaluation to measure accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
from itertools import product
import argparse

from models import create_model
from universal_forecaster import UniversalForecaster
from shared_prediction import single_shot_evaluation
from train_utils import train_model


class LSTMTuner:
    """Hyperparameter tuner for LSTM models."""
    
    def __init__(self, csv_file, horizon, train_window, inference_window, max_lookback=None, max_train_loss=None):
        """
        Initialize tuner.
        
        Args:
            csv_file: Path to CSV data file
            horizon: Forecast horizon (fixed)
            train_window: Training window size (fixed)
            inference_window: Inference window size (not used in tuning, kept for API compatibility)
            max_lookback: Maximum lookback value to search (None = auto)
        """
        self.csv_file = csv_file
        self.horizon = horizon
        self.train_window = train_window
        # For tuning, we use train_window as both training and inference window
        # This simplifies the logic and matches single-shot behavior
        self.inference_window = train_window
        # Optional training abort threshold used to skip clearly bad configs early
        self.max_train_loss = max_train_loss
        
        # Load data
        self.df = pd.read_csv(csv_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.set_index('timestamp', inplace=True)
        
        # Set max_lookback based on available data
        data_size = len(self.df)
        if max_lookback is None:
            # Default: use at most 1/3 of available data for lookback
            # This leaves 1/3 for training buffer and 1/3 for validation
            self.max_lookback = max(60, min(600, data_size // 3))
        else:
            self.max_lookback = max_lookback
        
        print(f"Loaded {data_size} data points from {csv_file}")
        print(f"Max lookback constrained to: {self.max_lookback} (available data: {data_size})")
        print(f"Will predict {self.horizon} steps ahead")
        
    def define_search_space(self, search_type='quick'):
        """
        Define hyperparameter search space.
        
        Args:
            search_type: 'quick', 'balanced', 'exhaustive', or 'auto'
        """
        if search_type == 'auto':
            # Smart auto-tuning based on problem characteristics
            data_size = len(self.df)
            max_lookback = min(500, self.train_window // 4)
            
            if self.horizon <= 100:
                lookback_values = [30, 60]
                num_layers = [1, 2]
            elif self.horizon <= 1000:
                lookback_values = [60, 120, 180]
                num_layers = [2]
            else:
                lookback_values = [120, 180, 240, 300]
                num_layers = [2, 3]
            
            lookback_values = [lb for lb in lookback_values if lb <= max_lookback]
            if not lookback_values:
                lookback_values = [min(120, max_lookback)]
            
            # Scale model size with data availability
            if data_size < 2000:
                hidden_sizes = [32, 64]
            elif data_size < 5000:
                hidden_sizes = [64, 128] if self.horizon <= 1000 else [128, 256]
            else:
                hidden_sizes = [128, 256] if self.horizon <= 1000 else [256, 512]
            
            return {
                'lookback': lookback_values,
                'hidden_size': hidden_sizes,
                'num_layers': num_layers,
                'dropout': [0.2, 0.3] if data_size < 5000 else [0.3, 0.4],
                'learning_rate': [0.001],
                'epochs': [100],  # Will early stop
                'batch_size': [256 if data_size >= 10000 else 128 if data_size >= 5000 else 64],
            }
        
        elif search_type == 'quick':
            # Fast exploration (4 configurations)
            return {
                'lookback': [60, 120],
                'hidden_size': [64],
                'num_layers': [2],
                'dropout': [0.2],
                'learning_rate': [0.001],
                'epochs': [50],
                'batch_size': [32],
            }
        
        elif search_type == 'balanced':
            # Medium exploration - generate lookback values that scale to max_lookback
            base_lookbacks = [60, 120, 240, 480]
            
            # Add more values scaling up to max_lookback
            if self.max_lookback > 480:
                current = 800
                while current < self.max_lookback:
                    base_lookbacks.append(current)
                    current = int(current * 1.5)  # Geometric progression
                
                # Always include max_lookback itself
                if base_lookbacks[-1] != self.max_lookback:
                    base_lookbacks.append(self.max_lookback)
            
            return {
                'lookback': sorted(set(base_lookbacks)),
                'hidden_size': [64, 128],
                'num_layers': [2, 3],
                'dropout': [0.2],
                'learning_rate': [0.001, 0.0005],
                'epochs': [50],
                'batch_size': [32],
            }
        
        else:  # exhaustive
            # Comprehensive search - generate more lookback values that scale to max_lookback
            base_lookbacks = [30, 60, 120, 240, 480]
            
            # Add more values scaling up to max_lookback
            if self.max_lookback > 480:
                current = 800
                while current < self.max_lookback:
                    base_lookbacks.append(current)
                    current = int(current * 1.4)  # Slightly denser than balanced
                
                # Always include max_lookback itself
                if base_lookbacks[-1] != self.max_lookback:
                    base_lookbacks.append(self.max_lookback)
            
            return {
                'lookback': sorted(set(base_lookbacks)),
                'hidden_size': [32, 64, 128, 256],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3],
                'learning_rate': [0.0005, 0.001],
                'epochs': [50],
                'batch_size': [32],
            }
    
    def evaluate_config(self, config, verbose=True):
        """
        Evaluate a single LSTM configuration.
        
        Args:
            config: Dictionary of hyperparameters
            verbose: Whether to print progress
            
        Returns:
            Dictionary with metrics (MAPE, MBE, train_time, etc.)
        """
        try:
            lookback = config.get('lookback', config.get('window', 60))
            train_window = lookback + self.horizon
            min_required = train_window
            if len(self.df) < min_required:
                return {
                    'error': f'Insufficient data: need {min_required}, have {len(self.df)}',
                    'mape': float('inf')
                }
            if lookback >= train_window:
                return {
                    'error': f'Lookback {lookback} >= train_window {train_window}',
                    'mape': float('inf')
                }
            config_for_model = dict(config)
            config_for_model['lookback'] = lookback
            model = create_model(
                'lstm',
                horizon=self.horizon,
                random_state=42,
                **config_for_model
            )
            train_data = self.df['value'].iloc[:train_window]
            if verbose:
                print(f"  Training with {len(train_data)} points...", end=' ')
            # Use centralized training helper for consistent behavior
            res = train_model(model, train_data, quiet=True, max_train_loss=self.max_train_loss)
            train_time = res.get('train_time', 0.0)
            if verbose:
                print(f"{train_time:.2f}s")
            if verbose:
                print(f"  Predicting...", end=' ')
            predictions = model.predict()
            if verbose:
                print("done")
            
            # Collect actual values for evaluation (points right after training data)
            actuals = []
            for i in range(self.horizon):
                actual_idx = train_window + i  # Use local train_window, not self.train_window
                if actual_idx >= len(self.df):
                    return {
                        'error': f'Insufficient data: need index {actual_idx}, have {len(self.df)}',
                        'mape': float('inf')
                    }
                actuals.append(float(self.df['value'].iloc[actual_idx]))
            
            # Calculate metrics using shared evaluation logic
            if len(actuals) != self.horizon:
                return {
                    'error': 'Could not collect all actuals',
                    'mape': float('inf')
                }
            
            from shared_prediction import evaluate_predictions
            eval_metrics = evaluate_predictions(
                predictions=predictions,
                actuals=actuals,
                verbose=False  # tune_lstm doesn't print evaluation details
            )
            
            return {
                'mape': eval_metrics['mape'],
                'mbe': eval_metrics['mbe'],
                'rmse': eval_metrics['rmse'],
                'train_time': train_time,
                'config': config,
                'predictions': predictions[:5],  # Store first 5 for inspection
                'actuals': actuals[:5],
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'mape': float('inf'),
                'config': config,
            }
    
    def run_tuning(self, search_space, max_time_per_config=300, use_adaptive=True):
        """
        Run hyperparameter tuning with optional two-phase adaptive search.
        
        Phase 1 (Exploration): Test all configurations to find promising regions
        Phase 2 (Exploitation): Refine the best configurations with nearby parameter values
        
        Args:
            search_space: Dictionary of parameter ranges
            max_time_per_config: Skip configs that would take longer than this (seconds)
            use_adaptive: Enable two-phase adaptive search (default: True)
            
        Returns:
            List of results sorted by MAPE
        """
        # Generate all combinations
        param_names = sorted(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        all_configs = [dict(zip(param_names, values)) for values in product(*param_values)]
        
        print(f"\n{'='*70}")
        print(f"LSTM Hyperparameter Tuning")
        if use_adaptive:
            print(f"Mode: ADAPTIVE (Exploration → Exploitation)")
        else:
            print(f"Mode: EXHAUSTIVE")
        print(f"Phase 1 configurations: {len(all_configs)}")
        print(f"{'='*70}\n")
        
        results = []
        best_mape_so_far = float('inf')
        
        # PHASE 1: EXPLORATION
        print(f"{'='*70}")
        print("PHASE 1: EXPLORATION - Sweeping entire parameter space")
        print(f"{'='*70}\n")
        
        bad_configs = []  # configs that performed extremely poorly

        def _is_similar(a, b, min_matches=None):
            """Return True if configs a and b share at least min_matches identical param values."""
            keys = set(a.keys()) & set(b.keys())
            if min_matches is None:
                min_matches = max(1, len(keys) // 2)
            matches = sum(1 for k in keys if a.get(k) == b.get(k))
            return matches >= min_matches

        for i, config in enumerate(all_configs, 1):
            # Skip configs that are similar to previously identified bad regions
            skip_due_to_bad = any(_is_similar(config, bc) for bc in bad_configs)
            if skip_due_to_bad:
                print(f"\n[{i}/{len(all_configs)}] Skipping configuration similar to poor-performing region")
                continue
            print(f"\n[{i}/{len(all_configs)}] Testing configuration:")
            for key, value in sorted(config.items()):
                print(f"  {key}: {value}")
            
            # Estimate time (rough heuristic)
            estimated_time = (self.train_window / 500) * (config['epochs'] / 20) * 5.45
            if estimated_time > max_time_per_config:
                print(f"  ⏭️  Skipping (estimated time: {estimated_time:.1f}s > {max_time_per_config}s)")
                continue
            
            result = self.evaluate_config(config, verbose=True)
            
            if 'error' not in result:
                print(f"  ✓ MAPE: {result['mape']:.2f}%, RMSE: {result['rmse']:.2f}, Time: {result['train_time']:.2f}s")
                
                # Track best result for progress indication
                if result['mape'] < best_mape_so_far:
                    print(f"  🌟 NEW BEST! (Previous: {best_mape_so_far:.2f}%)")
                    best_mape_so_far = result['mape']
            else:
                print(f"  ✗ Error: {result['error']}")
                # Mark this config as bad so we avoid exploring similar parameter combos
                bad_configs.append(config)
                results.append(result)
                continue

            # If MAPE is absurdly high (or infinite), treat as poor region
            try:
                mape_val = float(result.get('mape', float('inf')))
            except Exception:
                mape_val = float('inf')
            if mape_val >= 500.0 or not np.isfinite(mape_val):
                bad_configs.append(config)

            results.append(result)
        
        # PHASE 2: EXPLOITATION
        if use_adaptive and len(results) > 0:
            results.sort(key=lambda x: x['mape'])
            
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
                    
                    refinements = self._generate_refinements(best_config, search_space)
                    refinement_configs.extend(refinements)
                
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
        """Generate refinement configurations around a good base config."""
        refinements = []
        
        for param, base_value in base_config.items():
            if param not in search_space:
                continue
                
            possible_values = search_space[param]
            
            if len(possible_values) <= 1:
                continue
            
            try:
                base_idx = possible_values.index(base_value)
            except ValueError:
                continue
            
            new_values = []
            
            # Try value between base and next larger (if exists)
            if base_idx < len(possible_values) - 1:
                next_val = possible_values[base_idx + 1]
                if isinstance(base_value, (int, float)) and isinstance(next_val, (int, float)):
                    if param == 'lookback':
                        mid = (base_value + next_val) // 2
                        if mid != base_value and mid != next_val:
                            new_values.append(mid)
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
    
    def print_results(self, results, top_n=10):
        """Print top N results."""
        print("\n" + "="*100)
        print(f"TOP {top_n} CONFIGURATIONS (by MAPE)")
        print("="*100)
        
        valid_results = [r for r in results if 'error' not in r]
        
        for i, result in enumerate(valid_results[:top_n], 1):
            config = result['config']
            print(f"\n#{i} - MAPE: {result['mape']:.2f}% | RMSE: {result['rmse']:.2f} | Time: {result['train_time']:.2f}s")
            print(f"  Config: lookback={config['lookback']}, hidden={config['hidden_size']}, "
                  f"layers={config['num_layers']}, dropout={config['dropout']}, "
                  f"lr={config['learning_rate']}, epochs={config['epochs']}, batch={config['batch_size']}")
            print(f"  First 5 predictions: {[f'{p:.2f}' for p in result['predictions']]}")
            print(f"  First 5 actuals:     {[f'{a:.2f}' for a in result['actuals']]}")
    
    def save_results(self, results, output_file='tuning_results.json'):
        """Save results to JSON file."""
        # Convert to serializable format
        serializable = []
        for r in results:
            r_copy = r.copy()
            if 'config' in r_copy:
                # Flatten config
                config = r_copy.pop('config')
                for k, v in config.items():
                    r_copy[f'param_{k}'] = v
            serializable.append(r_copy)
        
        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
    
    def create_best_config_yaml(self, results, output_file='config_lstm_best.yaml', original_mape=None):
        """Create YAML config file with best parameters (only if better than original)."""
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            print("No valid results to create config from")
            return
        
        best = valid_results[0]
        best_mape = best['mape']
        
        # Check if we should save (only if we beat original or no original exists)
        should_save = True
        if original_mape is not None:
            if best_mape >= original_mape:
                should_save = False
                print(f"\n{'='*70}")
                print(f"⚠️  Tuning did NOT improve over original config")
                print(f"   Original MAPE: {original_mape:.2f}%")
                print(f"   Best new MAPE: {best_mape:.2f}%")
                print(f"   Config file will NOT be overwritten")
                print(f"{'='*70}\n")
                return
            else:
                print(f"\n{'='*70}")
                print(f"✅ Tuning IMPROVED over original config!")
                print(f"   Original MAPE: {original_mape:.2f}%")
                print(f"   Best new MAPE: {best_mape:.2f}%")
                print(f"   Improvement: {original_mape - best_mape:.2f}% reduction")
                print(f"   Config file will be updated")
                print(f"{'='*70}\n")
        
        config = best['config']
        
        # Calculate actual train_window based on best lookback
        best_lookback = config['lookback']
        best_train_window = best_lookback + self.horizon
        
        yaml_config = {
            'model': 'lstm',
            'single-shot': True,
            'horizon': self.horizon,
            'train-window': best_train_window,
            
            # Best LSTM parameters
            'lookback': config['lookback'],
            'hidden-size': config['hidden_size'],
            'num-layers': config['num_layers'],
            'dropout': config['dropout'],
            'learning-rate': config['learning_rate'],
            'epochs': config['epochs'],
            'batch-size': config['batch_size'],
            
            # Server
            'server-port': 5000,
        }
        
        with open(output_file, 'w') as f:
            f.write(f"# LSTM Best Configuration\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# MAPE: {best['mape']:.2f}%\n")
            f.write(f"# RMSE: {best['rmse']:.2f}\n")
            f.write(f"# Train Time: {best['train_time']:.1f}s\n\n")
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✓ Best config saved to: {output_file}")
        
        # Print best config summary
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION:")
        print(f"{'='*70}")
        print(f"MAPE: {best['mape']:.2f}%")
        print(f"RMSE: {best['rmse']:.2f}")
        print(f"Train Time: {best['train_time']:.1f}s")
        print(f"\nYou can now run: python forecast_main.py --config {output_file}")



def main():
    parser = argparse.ArgumentParser(description='LSTM Hyperparameter Tuning')
    parser.add_argument('--config', type=str,
                        help='YAML config file to load base settings from')
    parser.add_argument('--csv-file', type=str,
                        help='Path to CSV data file')
    parser.add_argument('--horizon', type=int,
                        help='Forecast horizon')
    parser.add_argument('--train-window', type=int,
                        help='Training window size')
    parser.add_argument('--inference-window', type=int,
                        help='Inference window size (defaults to train-window if not specified)')
    parser.add_argument('--search', type=str, default='auto',
                        choices=['auto', 'quick', 'balanced', 'exhaustive'],
                        help='Search strategy (default: auto - intelligently adapts to your data)')
    parser.add_argument('--max-time', type=int, default=300,
                        help='Max seconds per config (default: 300)')
    parser.add_argument('--max-train-loss', type=float, default=None,
                        help='Optional max_train_loss to abort training early for poor configs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (default: same as config file or tuning_results.json)')
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    config = {}
    original_mape = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store original MAPE if it exists (for comparison later)
        if '# MAPE' in config:
            mape_str = str(config['# MAPE']).replace('%', '').strip()
            try:
                original_mape = float(mape_str)
            except ValueError:
                pass
        
        # If csv-file is not specified in config, infer it from config filename
        if 'csv-file' not in config and 'csv_file' not in config:
            # Extract basename without extension from config path
            config_basename = os.path.splitext(os.path.basename(args.config))[0]
            # Infer CSV filename
            inferred_csv = f"csv/{config_basename}.csv"
            config['csv-file'] = inferred_csv
            print(f"[Config] Inferred CSV file from config name: {inferred_csv}")
        
        # If output not specified, use the config file as output
        if args.output is None:
            args.output = args.config
            print(f"[Config] Will save results to: {args.output}")
        
        print(f"Loaded base configuration from {args.config}")
    
    # Set default output if still not specified
    if args.output is None:
        args.output = 'tuning_results.json'
    
    # Override with command-line arguments (command line takes precedence)
    csv_file = args.csv_file or config.get('csv-file')
    horizon = args.horizon or config.get('horizon', 50)
    train_window = args.train_window or config.get('train-window', 2000)
    inference_window = args.inference_window or config.get('window', train_window)
    
    if not csv_file:
        print("ERROR: --csv-file is required (or csv-file in config)")
        return
    
    # Resolve relative path
    if not csv_file.startswith('/'):
        csv_file = os.path.join('csv', csv_file) if not csv_file.startswith('csv/') else csv_file
    
    print("="*100)
    print("LSTM HYPERPARAMETER TUNING")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  CSV File: {csv_file}")
    print(f"  Horizon: {horizon}")
    print(f"  Train Window: {train_window}")
    print(f"  Inference Window: {inference_window}")
    print(f"  Search Strategy: {args.search}")
    print(f"  Max Time per Config: {args.max_time}s")
    
    # Create tuner
    tuner = LSTMTuner(
        csv_file=csv_file,
        horizon=horizon,
        train_window=train_window,
        inference_window=inference_window,
        max_lookback=args.max_lookback,
        max_train_loss=args.max_train_loss
    )
    
    # Define search space
    search_space = tuner.define_search_space(args.search)
    
    if args.search == 'auto':
        print(f"\n🤖 Auto mode detected:")
        print(f"  • Lookback values: {search_space['lookback']}")
        print(f"  • Hidden sizes: {search_space['hidden_size']}")
        print(f"  • Network layers: {search_space['num_layers']}")
        print(f"  • Epochs: {search_space['epochs']}")
        print(f"  • Total configs: {len([1 for _ in product(*[search_space[k] for k in sorted(search_space.keys())])])}")
    
    # Run tuning
    results = tuner.run_tuning(search_space, max_time_per_config=args.max_time)
    
    # Print top results
    tuner.print_results(results, top_n=10)
    
    # Save results
    tuner.save_results(results, args.output)
    
    # Create best config (only if we beat the original)
    tuner.create_best_config_yaml(results, output_file=args.output, original_mape=original_mape)
    
    print("\n" + "="*100)
    print("TUNING COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
