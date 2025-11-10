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


class LSTMTuner:
    """Hyperparameter tuner for LSTM models."""
    
    def __init__(self, csv_file, horizon, train_window, inference_window):
        """
        Initialize tuner.
        
        Args:
            csv_file: Path to CSV data file
            horizon: Forecast horizon (fixed)
            train_window: Training window size (fixed)
            inference_window: Inference window size (not used in tuning, kept for API compatibility)
        """
        self.csv_file = csv_file
        self.horizon = horizon
        self.train_window = train_window
        # For tuning, we use train_window as both training and inference window
        # This simplifies the logic and matches single-shot behavior
        self.inference_window = train_window
        
        # Load data
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
            # Adapts parameter ranges to the problem
            data_size = len(self.df)
            
            # Lookback: between 10% and 100% of horizon, capped by data size
            max_lookback = min(self.horizon * 2, self.train_window // 2)
            lookback_values = [
                max(30, self.horizon // 10),  # Small: 10% of horizon
                max(60, self.horizon // 5),   # Medium: 20% of horizon
                max(120, self.horizon // 2),  # Large: 50% of horizon
            ]
            lookback_values = [lb for lb in lookback_values if lb <= max_lookback]
            if not lookback_values:
                lookback_values = [60]
            
            # Hidden size: scale with complexity
            if self.horizon < 100:
                hidden_sizes = [32, 64]
            elif self.horizon < 1000:
                hidden_sizes = [32, 64, 128]
            else:
                hidden_sizes = [64, 128, 256]
            
            # Layers: more layers for complex patterns
            if self.horizon < 500:
                num_layers = [1, 2]
            else:
                num_layers = [2, 3]
            
            # Epochs: balance training time vs accuracy
            if data_size > 5000:
                epochs = [30, 50]
            else:
                epochs = [40, 60]
            
            return {
                'lookback': lookback_values,
                'hidden_size': hidden_sizes,
                'num_layers': num_layers,
                'dropout': [0.1, 0.2],
                'learning_rate': [0.001, 0.002],
                'epochs': epochs,
                'batch_size': [32],
            }
        elif search_type == 'quick':
            # Fast exploration - 12 configurations
            return {
                'lookback': [60, 120],
                'hidden_size': [32, 64],
                'num_layers': [1, 2],
                'dropout': [0.1, 0.2],
                'learning_rate': [0.001],
                'epochs': [20, 30],
                'batch_size': [32],
            }
        elif search_type == 'balanced':
            # Medium exploration - ~100 configurations
            return {
                'lookback': [60, 120, 180],
                'hidden_size': [32, 64, 128],
                'num_layers': [1, 2, 3],
                'dropout': [0.1, 0.2, 0.3],
                'learning_rate': [0.0005, 0.001, 0.002],
                'epochs': [20, 40],
                'batch_size': [16, 32],
            }
        elif search_type == 'exhaustive':
            # Full exploration - 1000+ configurations
            return {
                'lookback': [30, 60, 120, 180, 240],
                'hidden_size': [16, 32, 64, 128, 256],
                'num_layers': [1, 2, 3, 4],
                'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
                'learning_rate': [0.0001, 0.0005, 0.001, 0.002, 0.005],
                'epochs': [10, 20, 40, 60, 100],
                'batch_size': [8, 16, 32, 64],
            }
        else:
            raise ValueError(f"Unknown search_type: {search_type}")
    
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
                'lstm',
                horizon=self.horizon,
                random_state=42,
                **config
            )
            
            # Get training data (first train_window points)
            train_data = self.df['value'].iloc[:self.train_window]
            
            # Train
            if verbose:
                print(f"  Training with {len(train_data)} points...", end=' ')
            train_time = model.train(train_data)
            if verbose:
                print(f"{train_time:.2f}s")
            
            # Make prediction directly (no need for forecaster wrapper in tuning)
            if verbose:
                print(f"  Predicting...", end=' ')
            predictions = model.predict()
            if verbose:
                print("done")
            
            # Collect actual values for evaluation (points right after training data)
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
            
            # MAPE
            errors = []
            for actual, pred in zip(actuals, predictions):
                if abs(actual) > 1e-6:
                    errors.append(abs(actual - pred) / abs(actual) * 100.0)
            
            mape = np.mean(errors) if errors else float('inf')
            mape = min(mape, 1000.0)  # Cap at 1000%
            
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
                'predictions': predictions[:5],  # Store first 5 for inspection
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
            max_time_per_config: Skip configs that would take longer than this (seconds)
            
        Returns:
            List of results sorted by MAPE
        """
        # Generate all combinations
        param_names = sorted(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        all_configs = [dict(zip(param_names, values)) for values in product(*param_values)]
        
        print(f"\nTotal configurations to test: {len(all_configs)}")
        print(f"Search space: {search_space}\n")
        
        results = []
        
        for i, config in enumerate(all_configs, 1):
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
            else:
                print(f"  ✗ Error: {result['error']}")
            
            results.append(result)
        
        # Sort by MAPE
        results.sort(key=lambda x: x['mape'])
        
        return results
    
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
    
    def create_best_config_yaml(self, results, output_file='config_lstm_best.yaml'):
        """Create YAML config file with best parameters."""
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            print("No valid results to create config from")
            return
        
        best = valid_results[0]
        config = best['config']
        
        yaml_config = {
            'csv-file': self.csv_file,
            'column': 'value',
            'model': 'lstm',
            'single-shot': True,
            'horizon': self.horizon,
            'train-window': self.train_window,
            
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
        print(f"You can now run: python forecast_main.py --config {output_file}")



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
    parser.add_argument('--output', type=str, default='tuning_results.json',
                        help='Output file for results (default: tuning_results.json)')
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded base configuration from {args.config}")
    
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
        inference_window=inference_window
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
    
    # Create best config
    tuner.create_best_config_yaml(results)
    
    print("\n" + "="*100)
    print("TUNING COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
