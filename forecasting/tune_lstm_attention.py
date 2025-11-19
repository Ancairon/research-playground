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
from datetime import datetime
from itertools import product
import argparse
import random

from models import create_model
from universal_forecaster import UniversalForecaster
from shared_prediction import single_shot_evaluation


class LSTMAttentionTuner:
    """Hyperparameter tuner for LSTM with Attention models."""
    
    def __init__(self, csv_file, horizon, train_window, inference_window, max_lookback=None):
        """
        Initialize tuner.
        
        Args:
            csv_file: Path to CSV file
            horizon: Prediction horizon
            train_window: Training window size
            inference_window: Inference window size (for backwards compatibility)
            max_lookback: Maximum lookback to search (default: data_size // 3)
        """
        self.df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
        self.horizon = horizon
        self.train_window = train_window
        self.inference_window = inference_window
        self.csv_file = csv_file
        self.original_config = {}  # Will be populated with original config values
        
        # Set max_lookback
        data_size = len(self.df)
        if max_lookback is not None:
            self.max_lookback = max_lookback
        else:
            self.max_lookback = data_size // 3
        
        print(f"Loaded {data_size} data points from {csv_file}")
        print(f"Max lookback constrained to: {self.max_lookback} (available data: {data_size})")
        print(f"Will predict {self.horizon} steps ahead")
        
    def define_search_space(self, search_type='quick'):
        """
        Define hyperparameter search space.
        
        Args:
            search_type: 'quick', 'balanced', 'exhaustive', or 'auto'
        """
        data_size = len(self.df)
        
        # Get defaults from original config if available
        def get_config_value(key, default):
            """Get value from original config with fallback to default."""
            # Try hyphenated key first (YAML format)
            hyphen_key = key.replace('_', '-')
            if hyphen_key in self.original_config:
                return self.original_config[hyphen_key]
            # Try underscore key (Python format)
            if key in self.original_config:
                return self.original_config[key]
            return default
        
        # Get static parameters from config (not tuned)
        config_epochs = get_config_value('epochs', 50)
        config_batch_size = get_config_value('batch_size', 128)
        config_dropout = get_config_value('dropout', 0.2)
        
        # Helper function to filter lookback values based on constraints
        def filter_lookbacks(values):
            """Filter lookback values to fit within data and max_lookback constraints."""
            filtered = [v for v in values if v <= self.max_lookback and (v + self.horizon) < data_size]
            if not filtered:
                # If all values filtered out, use a safe default
                safe_lookback = min(self.max_lookback, max(60, (data_size - self.horizon) // 2))
                filtered = [safe_lookback]
            return filtered
        
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
                hidden_sizes = [64]
            elif data_size < 5000:
                hidden_sizes = [64, 128]
            else:
                hidden_sizes = [64, 128, 256]
            
            num_layers = [2] if data_size < 5000 or self.horizon < 1000 else [2, 3]
            dropout_values = [config_dropout]  # Use config value
            epochs = [config_epochs]  # Use config value
            
            return {
                'lookback': lookback_values,
                'hidden_size': hidden_sizes,
                'num_layers': num_layers,
                'dropout': dropout_values,
                'learning_rate': [0.001, 0.0005],  # Search both common values
                'epochs': epochs,
                'batch_size': [config_batch_size],  # Use config value
            }
        
        elif search_type == 'quick':
            # Ultra-fast: test 2-3 lookback values
            base_lookbacks = [60, 120]
            return {
                'lookback': filter_lookbacks(base_lookbacks),
                'hidden_size': [64, 128],
                'num_layers': [2],
                'dropout': [config_dropout],
                'learning_rate': [0.001],  # Quick mode: just one learning rate
                'epochs': [config_epochs],
                'batch_size': [config_batch_size],
            }
        
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
            
            return {
                'lookback': filter_lookbacks(sorted(set(base_lookbacks))),
                'hidden_size': [64, 128],
                'num_layers': [2],
                'dropout': [config_dropout],
                'learning_rate': [0.001, 0.0005],  # Balanced: test 2 learning rates
                'epochs': [config_epochs],
                'batch_size': [config_batch_size],
            }
        
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
            
            return {
                'lookback': filter_lookbacks(sorted(set(base_lookbacks))),
                'hidden_size': [32, 64, 128, 256],
                'num_layers': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3],  # Exhaustive tries multiple dropout values
                'learning_rate': [0.001, 0.0005, 0.0001],  # Exhaustive: test 3 learning rates
                'epochs': [config_epochs],  # Use config epochs with early stopping
                'batch_size': [32, 64],  # Exhaustive tries multiple batch sizes
            }
    
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
            # Get lookback from config (support 'window' as alias for backwards compatibility)
            lookback = config.get('lookback', config.get('window', 60))
            train_window = lookback + self.horizon
            
            # Prepare config for model - use forecast_main's parameter names
            model_kwargs = {
                'horizon': self.horizon,
                'lookback': lookback,
                'random_state': 42,
            }
            
            # Add preserved parameters from original config (not tuned but important)
            if hasattr(self, 'original_config') and self.original_config:
                preserve_params = ['scaler-type', 'scaler_type', 'bias-correction', 'bias_correction', 
                                   'use-differencing', 'use_differencing']
                for key in preserve_params:
                    if key in self.original_config:
                        # Convert to underscore format for model
                        model_key = key.replace('-', '_')
                        model_kwargs[model_key] = self.original_config[key]
            
            # Add all config parameters (from tuning search space)
            for key, value in config.items():
                if key not in ['lookback', 'window']:  # Skip lookback, we already set it
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
                train_window=train_window
            )
            
            # Use shared single-shot evaluation logic
            result = single_shot_evaluation(
                forecaster=forecaster,
                data=self.df['value'],
                train_window=train_window,
                lookback=lookback,
                horizon=self.horizon,
                verbose=verbose
            )
            
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
    
    def run_tuning(self, search_space, max_time_per_config=300, use_adaptive=True):
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
        best_mape_so_far = float('inf')
        
        # PHASE 1: EXPLORATION - Sample parameter space
        print(f"{'='*70}")
        if use_adaptive:
            print("PHASE 1: SPARSE EXPLORATION - Sampling parameter space")
        else:
            print("PHASE 1: EXHAUSTIVE SEARCH - Testing all configurations")
        print(f"{'='*70}\n")
        
        for i, config in enumerate(phase1_configs, 1):
            print(f"\n[{i}/{len(phase1_configs)}] Testing configuration:")
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
                
                # Track best result for progress indication
                if result['mape'] < best_mape_so_far:
                    print(f"  🌟 NEW BEST! (Previous: {best_mape_so_far:.2f}%)")
                    best_mape_so_far = result['mape']
        
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
                
            print(f"#{i} - MAPE: {result['mape']:.2f}%")
            print(f"   Config:")
            for key, val in result['config'].items():
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
  # Quick tuning (few combinations, fast)
  python tune_lstm_attention.py --csv data.csv --mode quick
  
  # Auto tuning (intelligent parameter selection)
  python tune_lstm_attention.py --csv data.csv --mode auto
  
  # Balanced tuning (moderate search)
  python tune_lstm_attention.py --csv data.csv --mode balanced
  
  # Load from config file
  python tune_lstm_attention.py --config config_lstm_attention_test.yaml --mode auto
        """
    )
    
    parser.add_argument('--csv', '--csv-file', dest='csv_file', 
                        help='Path to CSV file with time series data')
    parser.add_argument('--config', help='Path to YAML config file (alternative to --csv)')
    parser.add_argument('--horizon', type=int, default=3000,
                        help='Forecast horizon (default: 3000)')
    parser.add_argument('--train-window', type=int, default=10000,
                        help='Training window size (default: 10000)')
    parser.add_argument('--window', type=int, default=None,
                        help='Inference window size (default: same as train-window)')
    parser.add_argument('--max-lookback', type=int, default=None,
                        help='Maximum lookback to search (default: data_size/3, caps search space)')
    parser.add_argument('--mode', choices=['quick', 'balanced', 'exhaustive', 'auto'],
                        default='auto',
                        help='Tuning mode (default: auto)')
    parser.add_argument('--output', default=None,
                        help='Output file for results (default: same as config file or lstm_attention_tuning_results.yaml)')
    parser.add_argument('--overwrite-config', action='store_true',
                        help='Overwrite the original config file if a better configuration is found (default: create new file)')
    
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
            # Infer CSV filename
            inferred_csv = f"csv/{config_basename}.csv"
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
        
        # Override with config values
        if 'csv-file' in original_config:
            args.csv_file = original_config['csv-file']
        if 'horizon' in original_config:
            args.horizon = original_config['horizon']
        if 'train-window' in original_config:
            args.train_window = original_config['train-window']
        if 'window' in original_config:
            args.window = original_config['window']
        if 'max-lookback' in original_config and args.max_lookback is None:
            args.max_lookback = original_config['max-lookback']
        
        # Report preserved parameters
        preserved_params = {}
        for key in ['scaler-type', 'use-differencing', 'bias-correction']:
            if key in original_config:
                preserved_params[key] = original_config[key]
        
        if preserved_params:
            print(f"[Config] Preserved parameters (not tuned): {preserved_params}")
        
        print(f"Loaded configuration from: {args.config}")
    
    # Set default window if not specified
    if args.window is None:
        args.window = args.train_window
    
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
        max_lookback=args.max_lookback
    )
    
    # Store original config for later use
    tuner.original_config = original_config
    
    # Define search space
    search_space = tuner.define_search_space(args.mode)
    
    print(f"\nSearch mode: {args.mode}")
    print("Parameter ranges:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")
    
    # Run tuning
    results = tuner.run_tuning(search_space)
    
    # Print summary
    tuner.print_summary(results, top_n=5)
    
    # Save results
    tuner.save_results(results, args.output)
    
    # Save best config as ready-to-use YAML (only if we beat the original)
    if results and 'error' not in results[0]:
        best = results[0]
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
            else:
                print(f"\n{'='*70}")
                print(f"✅ Tuning IMPROVED over original config!")
                print(f"   Original MAPE: {original_mape:.2f}%")
                print(f"   Best new MAPE: {best_mape:.2f}%")
                print(f"   Improvement: {original_mape - best_mape:.2f}% reduction")
                print(f"   Config file will be updated")
                print(f"{'='*70}\n")
        
        if should_save:
            # Calculate the actual train_window based on best config
            best_lookback = best['config']['lookback']
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
        
        # Print best config
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION:")
        print(f"{'='*70}")
        print(f"\nMAPE: {best['mape']:.2f}%")
        print(f"RMSE: {best['rmse']:.2f}")
        print(f"Train Time: {best['train_time']:.1f}s")
        print("\nParameters:")
        for key, val in best['config'].items():
            print(f"  {key}: {val}")
        print(f"\nYou can now run: python forecast_main.py --config {best_config_file}")
        print()


if __name__ == '__main__':
    main()
