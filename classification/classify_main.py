#!/usr/bin/env python3
"""
Time Series Classification Main Entry Point

Usage:
    # Train from training config (recommended)
    python classify_main.py --train-from-config
    
    # Train with custom config path
    python classify_main.py --train-from-config --training-config ./train/training_config.yaml
    
    # Classify a time series from a config file (uses saved model)
    python classify_main.py --config ../configs/192.168.1.123_temp_pi.yaml
    
    # Classify with custom model
    python classify_main.py --config ../configs/my_config.yaml --model xgboost
    
    # Train on multiple labeled datasets (legacy mode)
    python classify_main.py --train --labels class1:config1.yaml,class2:config2.yaml
    
    # Classify and auto-assign class name if new
    python classify_main.py --config ../configs/new_data.yaml --new-class-name "temperature_pattern"
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.classifier import TimeSeriesClassifier
from classification.models import list_available_models
import yaml
from pathlib import Path


def visualize_csv_from_config(config_path: str, result=None):
    """
    Visualize time series data from a config file's CSV.
    
    Args:
        config_path: Path to YAML config file
        result: Optional ClassificationResult to show in title
    """
    # Load config to get CSV path
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    csv_path = config.get('csv')
    if not csv_path:
        # Infer CSV path from config filename (same name, .csv extension)
        config_basename = os.path.basename(config_path)
        csv_filename = os.path.splitext(config_basename)[0] + '.csv'
        # Look in csv/ directory at repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        csv_path = os.path.join(repo_root, 'csv', csv_filename)
    
    # Resolve CSV path relative to config if not absolute
    if not os.path.isabs(csv_path):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        csv_path = os.path.join(config_dir, csv_path)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Read CSV
    df = pd.read_csv(csv_path)
    column = 'value'
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"CSV File: {csv_path}")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")
    
    # Calculate average interval
    time_diffs = df['timestamp'].diff().dropna()
    avg_interval = time_diffs.mean()
    print(f"Average interval: {avg_interval}")
    
    print(f"\nValue statistics:")
    print(f"  Mean: {df[column].mean():.4f}")
    print(f"  Std:  {df[column].std():.4f}")
    print(f"  Min:  {df[column].min():.4f}")
    print(f"  Max:  {df[column].max():.4f}")
    print(f"  NaN:  {df[column].isna().sum()}")
    print(f"{'='*60}\n")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot data
    ax.plot(df['timestamp'], df[column], linewidth=1, alpha=0.8)
    
    # Formatting
    title = f"Time Series: {Path(csv_path).stem}"
    if result:
        title += f"\nClass: {result.predicted_class} ({result.confidence:.1%} confidence)"
        if result.is_new_class:
            title += " [NEW]"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel(column.capitalize(), fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    
    # Add statistics text box
    stats_text = (
        f"Points: {len(df)}\n"
        f"Mean: {df[column].mean():.4f}\n"
        f"Std: {df[column].std():.4f}\n"
        f"Range: [{df[column].min():.4f}, {df[column].max():.4f}]"
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9,
            family='monospace')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Time Series Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from training config file (classification/train/training_config.yaml)
  python classify_main.py --train-from-config
  
  # Train with custom training config
  python classify_main.py --train-from-config --training-config ./my_training_config.yaml
  
  # Classify a single time series (uses saved model)
  python classify_main.py --config ../configs/192.168.1.123_temp_pi.yaml
  
  # Classify with a specific model
  python classify_main.py --config ../configs/my_config.yaml --model xgboost
  
  # Train on labeled data (legacy mode - inline labels)
  python classify_main.py --train \\
    --labels "temp_pattern:../configs/192.168.1.123_temp_pi.yaml,disk_pattern:../configs/wg-disk-smooth10.yaml"
  
  # Set confidence threshold
  python classify_main.py --config ../configs/my_config.yaml --confidence 0.7
  
  # Specify class name for new patterns
  python classify_main.py --config ../configs/my_config.yaml --new-class-name "my_custom_class"
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file to classify'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Training mode - train on labeled data (legacy: use --labels)'
    )
    parser.add_argument(
        '--train-from-config',
        action='store_true',
        help='Train from training_config.yaml in train/ folder'
    )
    parser.add_argument(
        '--training-config',
        type=str,
        help='Path to training config YAML (default: classification/train/training_config.yaml)'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        help='Directory containing training CSV files (default: classification/train/)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        help='Comma-separated list of label:config_path pairs for training (legacy mode)'
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default='randomforest',
        choices=list_available_models(),
        help='Classification model to use (default: randomforest)'
    )
    
    # Classification parameters
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.60,
        help='Confidence threshold for classification (default: 0.60)'
    )
    parser.add_argument(
        '--new-class-name',
        type=str,
        help='Name for new class if detected (default: config filename)'
    )
    parser.add_argument(
        '--no-auto-retrain',
        action='store_true',
        help='Disable automatic retraining when new class is detected'
    )
    parser.add_argument(
        '--skip-new-class',
        action='store_true',
        help='Skip adding new class if detected (just report classification result)'
    )
    
    # Output options
    parser.add_argument(
        '--class-key',
        type=str,
        default='class',
        help='Key name for storing class in config (default: class)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output messages'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show classifier information and exit'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the CSV time series data'
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = TimeSeriesClassifier(
        model_name=args.model,
        confidence_threshold=args.confidence,
        random_state=42,
    )
    
    # Handle --info flag
    if args.info:
        info = classifier.get_info()
        print("\n" + "=" * 60)
        print("Time Series Classifier Information")
        print("=" * 60)
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("=" * 60 + "\n")
        return 0
    
    # Handle training from config mode (recommended)
    if args.train_from_config:
        if not args.quiet:
            print(f"\n{'=' * 60}")
            print("Training from Configuration")
            print(f"{'=' * 60}")
        
        try:
            train_time = classifier.train_from_config(
                training_config_path=args.training_config,
                train_dir=args.train_dir,
                save=True
            )
            
            if not args.quiet:
                print(f"\n{'=' * 60}")
                print(f"Training complete in {train_time:.2f}s")
                print(f"Classes: {classifier.classes_}")
                print(f"Model saved to: {classifier.model_path}")
                print(f"{'=' * 60}\n")
            
            return 0
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return 1
    
    # Handle legacy training mode (inline labels)
    if args.train:
        if not args.labels:
            print("Error: --labels required for training mode")
            print("Format: --labels 'class1:config1.yaml,class2:config2.yaml'")
            print("\nTip: Use --train-from-config instead for config-based training")
            return 1
        
        # Parse labels
        training_data = {}
        for item in args.labels.split(','):
            if ':' not in item:
                print(f"Error: Invalid label format: {item}")
                print("Format should be: label:config_path")
                return 1
            
            label, config_path = item.strip().split(':', 1)
            config_path = config_path.strip()
            
            # Resolve path
            if not os.path.isabs(config_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                repo_root = os.path.dirname(script_dir)
                config_path = os.path.join(repo_root, 'configs', config_path)
            
            if not os.path.exists(config_path):
                print(f"Error: Config file not found: {config_path}")
                return 1
            
            # Load data
            data, _ = classifier.load_from_config(config_path)
            training_data[label.strip()] = data
            
            if not args.quiet:
                print(f"  Loaded: {label} from {config_path}")
        
        # Train
        if not args.quiet:
            print(f"\nTraining {args.model} classifier on {len(training_data)} classes...")
        
        train_time = classifier.train(training_data)
        
        if not args.quiet:
            print(f"Training complete in {train_time:.2f}s")
            print(f"Classes: {classifier.classes_}")
        
        return 0
    
    # Handle classification mode
    if args.config:
        config_path = args.config
        
        # Resolve path
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            config_path = os.path.join(repo_root, 'configs', config_path)
        
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            return 1
        
        if not args.quiet:
            print(f"\n{'=' * 60}")
            print("Time Series Classification")
            print(f"{'=' * 60}")
            print(f"Config: {config_path}")
            print(f"Model: {args.model}")
            print(f"Confidence threshold: {args.confidence:.0%}")
            print(f"{'=' * 60}\n")
        
        # Classify
        result = classifier.classify_and_update_csv(
            config_path=config_path,
            class_key=args.class_key,
            new_class_name=args.new_class_name,
            auto_retrain=not args.no_auto_retrain,
            skip_new_class=args.skip_new_class,
        )
        
        if not args.quiet:
            print(f"\n{'=' * 60}")
            print("Classification Result")
            print(f"{'=' * 60}")
            print(f"  Predicted Class: {result.predicted_class}")
            print(f"  Confidence: {result.confidence:.2%}")
            print(f"  Is New Class: {result.is_new_class}")
            print("\n  All Probabilities:")
            for cls, prob in sorted(result.all_probabilities.items(), key=lambda x: -x[1]):
                print(f"    {cls}: {prob:.2%}")
            print(f"{'=' * 60}\n")
        
        # Visualize if requested
        if args.visualize:
            visualize_csv_from_config(config_path, result)
        
        return 0
    
    # No action specified
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
