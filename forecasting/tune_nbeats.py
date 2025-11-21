#!/usr/bin/env python3
"""
N-BEATS Hyperparameter Tuning System

Automatically tests different N-BEATS configurations to find optimal parameters
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
from shared_prediction import evaluate_predictions
from train_utils import train_model


class NBEATSTuner:
    """Hyperparameter tuner for N-BEATS models."""
    
    def __init__(self, csv_file, horizon, train_window, inference_window, max_lookback=None, max_train_loss=None):
        """
        Initialize tuner.
        
        Args:
            csv_file: Path to CSV data file
            horizon: Forecast horizon (fixed)
            train_window: Training window size (fixed)
            inference_window: Inference window size (lookback for predictions)
            max_lookback: Maximum lookback value to search (None = auto)
        """
        self.csv_file = csv_file
        self.horizon = horizon
        self.train_window = train_window
        self.inference_window = inference_window
        # Optional training abort threshold used to skip clearly bad configs early
        self.max_train_loss = max_train_loss
        
        # Load data (handle relative paths)
        if not os.path.isabs(csv_file):
            csv_file = os.path.join(os.path.dirname(__file__), csv_file)
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
            # N-BEATS benefits from longer lookback for trend/seasonality decomposition
            max_lookback = min(800, self.train_window // 4)
            
            if self.horizon <= 100:
                lookback_values = [60, 120]
                num_stacks = [2]
                num_blocks = [2]
            elif self.horizon <= 1000:
                lookback_values = [120, 180, 240]
                num_stacks = [2]
                num_blocks = [3]
            else:
                # Long-term: need more lookback for pattern decomposition
                lookback_values = [180, 240, 360, 480]
                num_stacks = [2, 3]
                num_blocks = [3, 4]
            
            lookback_values = [lb for lb in lookback_values if lb <= max_lookback]
            if not lookback_values:
                lookback_values = [min(180, max_lookback)]
            
            # Scale model size with data availability
            if data_size < 2000:
                hidden_sizes = [128]
            elif data_size < 5000:
                hidden_sizes = [256]
            else:
                hidden_sizes = [256, 512] if self.horizon > 1000 else [256]
            
            return {
                'lookback': lookback_values,
                'num_stacks': num_stacks,
                'num_blocks': num_blocks,
                'theta_size': [8, 16] if data_size >= 5000 else [8],
                'hidden_size': hidden_sizes,
                'learning_rate': [0.001, 0.0005],
                'epochs': [100],  # Will early stop
                'batch_size': [256 if data_size >= 10000 else 128 if data_size >= 5000 else 64],
            }
        
        elif search_type == 'quick':
            # Fast exploration (2 configurations)
            return {
                'lookback': [120, 180],
                'num_stacks': [2],
                'num_blocks': [3],
                'theta_size': [8],
                'hidden_size': [256],
                'learning_rate': [0.001],
                'epochs': [50],
                'batch_size': [32],
            }
        
        elif search_type == 'balanced':
            # Medium exploration - generate lookback values that scale to max_lookback
            base_lookbacks = [120, 240, 480]
            
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
                'num_stacks': [2, 3],
                'num_blocks': [3, 4],
                'theta_size': [8, 16],
                'hidden_size': [256],
                'learning_rate': [0.001, 0.0005],
                'epochs': [50],
                'batch_size': [32],
            }
        
        else:  # exhaustive
            # Comprehensive search - generate more lookback values that scale to max_lookback
            base_lookbacks = [90, 120, 180, 240, 480]
            
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
                'num_stacks': [2, 3],
                'num_blocks': [2, 3, 4],
                'theta_size': [4, 8, 16],
                'hidden_size': [128, 256, 512],
                'learning_rate': [0.001, 0.0005],
                'epochs': [50],
                'batch_size': [32],
            }
    
    def evaluate_config(self, config, verbose=False):
        """
        Evaluate a single configuration.
        
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
                'nbeats',
                horizon=self.horizon,
                random_state=42,
                **config_for_model
            )
            train_data = self.df['value'].iloc[:train_window]
            if verbose:
                print(f"  Training with {len(train_data)} points...", end=' ')
            # Use centralized training helper
            res = train_model(model, train_data, quiet=True, max_train_loss=self.max_train_loss)
            train_time = res.get('train_time', 0.0)
            if verbose:
                print(f"{train_time:.2f}s")
            if verbose:
                pass
            # Make prediction
            if verbose:
                print(f"  Predicting...", end=' ')
            predictions = model.predict()
            if verbose:
                print("done")
            
            # Collect actual values
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
            
            eval_metrics = evaluate_predictions(
                predictions=predictions,
                actuals=actuals,
                verbose=False  # tune_nbeats doesn't print evaluation details
            )
            
            return {
                'mape': eval_metrics['mape'],
                'mbe': eval_metrics['mbe'],
                'rmse': eval_metrics['rmse'],
                'train_time': train_time,
                'config': config,
                'predictions': predictions[:5],
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
            max_time_per_config: Skip configs that would take longer than this
            use_adaptive: Enable two-phase adaptive search (default: True)
            
        Returns:
            List of results, sorted by MAPE
        """
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = [search_space[k] for k in param_names]
        all_configs = [dict(zip(param_names, v)) for v in product(*param_values)]
        
        total_configs = len(all_configs)
        print(f"\n{'='*70}")
        print(f"N-BEATS Hyperparameter Tuning")
        if use_adaptive:
            print(f"Mode: ADAPTIVE (Exploration → Exploitation)")
        else:
            print(f"Mode: EXHAUSTIVE")
        print(f"Phase 1 configurations: {total_configs}")
        print(f"{'='*70}\n")
        
        results = []
        best_mape_so_far = float('inf')
        
        # PHASE 1: EXPLORATION
        print(f"{'='*70}")
        print("PHASE 1: EXPLORATION - Sweeping entire parameter space")
        print(f"{'='*70}\n")
        
        bad_configs = []

        def _is_similar(a, b, min_matches=None):
            keys = set(a.keys()) & set(b.keys())
            if min_matches is None:
                min_matches = max(1, len(keys) // 2)
            matches = sum(1 for k in keys if a.get(k) == b.get(k))
            return matches >= min_matches

        for i, config in enumerate(all_configs, 1):
            if any(_is_similar(config, bc) for bc in bad_configs):
                print(f"\n[{i}/{total_configs}] Skipping configuration similar to poor-performing region")
                continue

            print(f"\n[{i}/{total_configs}] Testing configuration:")
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

            # Mark extremely poor configurations as bad
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
    
    def save_results(self, results, output_file):
        """Save tuning results to file."""
        # Prepare data for saving
        data_to_save = {
            'timestamp': datetime.now().isoformat(),
            'csv_file': self.csv_file,
            'horizon': self.horizon,
            'train_window': self.train_window,
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
        description='N-BEATS Hyperparameter Tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick tuning (few combinations, fast)
  python tune_nbeats.py --csv data.csv --mode quick
  
  # Auto tuning (intelligent parameter selection)
  python tune_nbeats.py --csv data.csv --mode auto
  
  # Balanced tuning (moderate search)
  python tune_nbeats.py --csv data.csv --mode balanced
  
  # Load from config file
  python tune_nbeats.py --config config_nbeats_test.yaml --mode auto
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
    parser.add_argument('--mode', choices=['quick', 'balanced', 'exhaustive', 'auto'],
                        default='auto',
                        help='Tuning mode (default: auto)')
    parser.add_argument('--max-train-loss', type=float, default=None,
                        help='Optional max_train_loss to abort training early for poor configs')
    parser.add_argument('--output', default=None,
                        help='Output file for results (default: same as config file or nbeats_tuning_results.yaml)')
    
    args = parser.parse_args()
    
    # Load config if specified
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
        
        # Override with config values
        if 'csv-file' in config:
            args.csv_file = config['csv-file']
        if 'horizon' in config:
            args.horizon = config['horizon']
        if 'train-window' in config:
            args.train_window = config['train-window']
        if 'window' in config:
            args.window = config['window']
        
        print(f"Loaded configuration from: {args.config}")
    
    # Set default window if not specified
    if args.window is None:
        args.window = args.train_window
    
    # Set default output if still not specified
    if args.output is None:
        args.output = 'nbeats_tuning_results.yaml'
    
    if not args.csv_file:
        print("ERROR: Must specify --csv or --config")
        sys.exit(1)
    
    # Initialize tuner
    tuner = NBEATSTuner(
        csv_file=args.csv_file,
        horizon=args.horizon,
        train_window=args.train_window,
        inference_window=args.window,
        max_lookback=args.max_lookback,
        max_train_loss=args.max_train_loss
    )
    
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
            # Calculate actual train_window based on best lookback
            best_lookback = best['config']['lookback']
            best_train_window = best_lookback + args.horizon
            
            # Use the original config file
            best_config_file = args.output
            best_config = {
                '# N-BEATS Best Configuration': None,
                '# Generated': datetime.now().isoformat(),
                '# MAPE': f"{best['mape']:.2f}%",
                '# RMSE': f"{best['rmse']:.2f}",
                '# Train Time': f"{best['train_time']:.1f}s",
                'model': 'nbeats',
                'single-shot': True,
                'train-window': best_train_window,
                'horizon': args.horizon,
            }
            
            # Add best hyperparameters
            for key, val in best['config'].items():
                # Convert underscores to hyphens for YAML consistency
                yaml_key = key.replace('_', '-')
                best_config[yaml_key] = val
            
            # Add server config
            best_config['server-port'] = 5000
            
            with open(best_config_file, 'w') as f:
                # Write comments manually since yaml.dump doesn't preserve them well
                f.write(f"# N-BEATS Best Configuration\n")
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
