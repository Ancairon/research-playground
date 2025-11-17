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
from config_cache import ConfigFingerprint, load_cached_results, save_cache


class NBEATSTuner:
    """Hyperparameter tuner for N-BEATS models."""
    
    def __init__(self, csv_file, horizon, train_window, inference_window):
        """
        Initialize tuner.
        
        Args:
            csv_file: Path to CSV data file
            horizon: Forecast horizon (fixed)
            train_window: Training window size (fixed)
            inference_window: Inference window size (not used in tuning)
        """
        self.csv_file = csv_file
        self.horizon = horizon
        self.train_window = train_window
        self.inference_window = train_window
        
        # Load data (handle relative paths)
        if not os.path.isabs(csv_file):
            csv_file = os.path.join(os.path.dirname(__file__), csv_file)
        self.df = pd.read_csv(csv_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.set_index('timestamp', inplace=True)
        
        print(f"Loaded {len(self.df)} data points from {csv_file}")
        print(f"Will use {self.train_window} points for training, then predict {self.horizon} steps ahead")
        
    def define_search_space(self, search_type='quick'):
        """
        Define hyperparameter search space.
        
        Args:
            search_type: 'quick', 'balanced', 'exhaustive', or 'auto'
        """
        if search_type == 'auto':
            # Intelligent auto-tuning based on data size and horizon
            data_size = len(self.df)
            
            # Lookback: N-BEATS benefits from longer lookback
            max_lookback = min(self.horizon * 3, self.train_window // 2)
            lookback_values = [
                max(60, self.horizon // 5),   # Small
                max(120, self.horizon // 2),  # Medium
                max(180, self.horizon),       # Large (at least horizon length)
            ]
            lookback_values = [lb for lb in lookback_values if lb <= max_lookback]
            if not lookback_values:
                lookback_values = [120]
            
            # Stack configuration
            if self.horizon < 500:
                num_stacks = [2]
                num_blocks = [2, 3]
            elif self.horizon < 2000:
                num_stacks = [2, 3]
                num_blocks = [3, 4]
            else:
                num_stacks = [2, 3]
                num_blocks = [3, 4]
            
            # Theta size (basis expansion degree)
            theta_sizes = [4, 8, 16]
            
            # Hidden size
            if self.horizon < 1000:
                hidden_sizes = [128, 256]
            else:
                hidden_sizes = [256, 512]
            
            # Epochs
            if data_size > 5000:
                epochs = [30, 50]
            else:
                epochs = [40, 60]
            
            return {
                'lookback': lookback_values,
                'num_stacks': num_stacks,
                'num_blocks': num_blocks,
                'theta_size': theta_sizes,
                'hidden_size': hidden_sizes,
                'learning_rate': [0.001, 0.0005],
                'epochs': epochs,
                'batch_size': [32, 64],
            }
        
        elif search_type == 'quick':
            return {
                'lookback': [120, 180],
                'num_stacks': [2],
                'num_blocks': [3],
                'theta_size': [8],
                'hidden_size': [256],
                'learning_rate': [0.001],
                'epochs': [30],
                'batch_size': [32],
            }
        
        elif search_type == 'balanced':
            return {
                'lookback': [90, 120, 180],
                'num_stacks': [2, 3],
                'num_blocks': [2, 3, 4],
                'theta_size': [4, 8, 16],
                'hidden_size': [128, 256],
                'learning_rate': [0.001, 0.0005],
                'epochs': [30, 50],
                'batch_size': [32, 64],
            }
        
        else:  # exhaustive
            return {
                'lookback': [60, 90, 120, 180, 240],
                'num_stacks': [2, 3, 4],
                'num_blocks': [2, 3, 4, 5],
                'theta_size': [4, 8, 16, 32],
                'hidden_size': [128, 256, 512],
                'learning_rate': [0.001, 0.0005, 0.0001],
                'epochs': [30, 50, 80],
                'batch_size': [16, 32, 64],
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
            # Check if we have enough data
            min_required = self.train_window + self.horizon
            if len(self.df) < min_required:
                return {
                    'error': f'Insufficient data: need {min_required}, have {len(self.df)}',
                    'mape': float('inf')
                }
            
            # Check if lookback is too large for training window
            if config['lookback'] >= self.train_window:
                return {
                    'error': f'Lookback {config["lookback"]} >= train_window {self.train_window}',
                    'mape': float('inf')
                }
            
            # Create model
            model = create_model(
                'nbeats',
                horizon=self.horizon,
                random_state=42,
                **config
            )
            
            # Get training data
            train_data = self.df['value'].iloc[:self.train_window]
            
            # Train
            if verbose:
                print(f"  Training with {len(train_data)} points...", end=' ')
            train_time = model.train(train_data)
            if verbose:
                print(f"{train_time:.2f}s")
            
            # Make prediction
            if verbose:
                print(f"  Predicting...", end=' ')
            predictions = model.predict()
            if verbose:
                print("done")
            
            # Collect actual values
            actuals = []
            for i in range(self.horizon):
                actual_idx = self.train_window + i
                if actual_idx >= len(self.df):
                    return {
                        'error': f'Insufficient data: need index {actual_idx}, have {len(self.df)}',
                        'mape': float('inf')
                    }
                actuals.append(self.df['value'].iloc[actual_idx])
            
            # Calculate metrics
            if len(actuals) != self.horizon:
                return {
                    'error': 'Could not collect all actuals',
                    'mape': float('inf')
                }
            
            # Calculate sMAPE (symmetric MAPE) - same as forecaster
            errors = []
            for actual, pred in zip(actuals, predictions):
                err_abs = abs(actual - pred)
                denominator = (abs(actual) + abs(pred)) / 2.0
                
                if denominator < 1e-6:
                    mape_val = 0.0
                else:
                    mape_val = (err_abs / denominator) * 100.0
                    mape_val = min(mape_val, 1000.0)  # Cap at 1000%
                
                errors.append(mape_val)
            
            mape = np.mean(errors) if errors else float('inf')
            
            # MBE
            mbe = np.mean([actual - pred for actual, pred in zip(actuals, predictions)])
            
            # RMSE
            rmse = np.sqrt(np.mean([(actual - pred)**2 for actual, pred in zip(actuals, predictions)]))
            
            return {
                'mape': mape,
                'mbe': mbe,
                'rmse': rmse,
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
    
    def run_tuning(self, search_space, max_time_per_config=300):
        """
        Run hyperparameter tuning.
        
        Args:
            search_space: Dictionary of parameter ranges
            max_time_per_config: Skip configs that would take longer than this
            
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
        print(f"Total configurations to test: {total_configs}")
        print(f"{'='*70}\n")
        
        results = []
        best_mape_so_far = float('inf')
        
        for i, config in enumerate(all_configs, 1):
            print(f"\n[{i}/{total_configs}] Testing configuration:")
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
        
        # Sort by MAPE
        results.sort(key=lambda x: x['mape'])
        
        return results
    
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
    parser.add_argument('--mode', choices=['quick', 'balanced', 'exhaustive', 'auto'],
                        default='auto',
                        help='Tuning mode (default: auto)')
    parser.add_argument('--output', default=None,
                        help='Output file for results (default: same as config file or nbeats_tuning_results.yaml)')
    
    args = parser.parse_args()
    
    # Load config if specified
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
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
        
        print(f"Loaded configuration from: {args.config}")
    
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
        inference_window=25  # Not used in tuning
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
    
    # Save best config as ready-to-use YAML (overwrite the original config file)
    if results and 'error' not in results[0]:
        best = results[0]
        
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
            'train-window': args.train_window,
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
        
        # Save to cache system
        class TempArgs:
            def __init__(self):
                self.model = 'nbeats'
                self.csv_file = args.csv_file
                self.ip = None
                self.context = None
                self.dimension = None
                self.window = args.train_window
                self.train_window = args.train_window
                self.horizon = args.horizon
                self.lookback = best['config']['lookback']
                self.hidden_size = best['config'].get('hidden_size', 256)
                self.num_layers = best['config'].get('num_layers', 4)
                self.dropout = best['config'].get('dropout', 0.1)
                self.learning_rate = best['config']['learning_rate']
                self.epochs = best['config']['epochs']
                self.batch_size = best['config']['batch_size']
                self.prediction_smoothing = 0
                self.aggregation_method = 'weighted'
                self.aggregation_weight_tau = 300.0
        
        temp_args = TempArgs()
        fingerprint = ConfigFingerprint.from_args(temp_args)
        config_hash = fingerprint.hash()
        
        cache_results = {
            "tuning_mode": True,
            "best_mape": float(best['mape']),
            "best_rmse": float(best['rmse']),
            "train_time": float(best['train_time']),
            "config": best['config'],
        }
        save_cache(config_hash, fingerprint, cache_results)
        print(f"✓ Results cached (hash: {config_hash[:16]}...)")
        
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
