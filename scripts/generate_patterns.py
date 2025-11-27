#!/usr/bin/env python3
"""
Generate synthetic time series patterns for classification training.

Creates CSV files with various patterns (sine, gaussian, trend, etc.) 
that can be used to train the time series classifier.

Usage:
    # Generate all default patterns
    python generate_patterns.py --output-dir ../classification/train
    
    # Generate specific patterns
    python generate_patterns.py --patterns sine gaussian trend --output-dir ./data
    
    # Custom parameters
    python generate_patterns.py --duration 7200 --interval 1 --noise 0.1
"""

import argparse
import os
import sys
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Optional


# Pattern generators
def generate_sine_wave(
    duration: int,
    interval: float = 1.0,
    amplitude: float = 50.0,
    period: float = 60.0,
    baseline: float = 100.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a sine wave pattern.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        amplitude: Wave amplitude
        period: Wave period in seconds
        baseline: Baseline value (center of oscillation)
        noise_level: Noise as fraction of amplitude (0.0-1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    t = np.arange(n_points) * interval
    
    # Generate sine wave
    values = baseline + amplitude * np.sin(2 * np.pi * t / period)
    
    # Add noise
    noise = np.random.normal(0, amplitude * noise_level, n_points)
    values = values + noise
    
    # Create timestamps
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_gaussian_noise(
    duration: int,
    interval: float = 1.0,
    mean: float = 100.0,
    std: float = 20.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate gaussian (white) noise pattern.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        mean: Mean value
        std: Standard deviation
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    values = np.random.normal(mean, std, n_points)
    values = np.maximum(values, 0)  # Ensure non-negative
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_linear_trend(
    duration: int,
    interval: float = 1.0,
    start_value: float = 50.0,
    end_value: float = 150.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a linear trend pattern.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        start_value: Starting value
        end_value: Ending value
        noise_level: Noise as fraction of range (0.0-1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    t = np.linspace(0, 1, n_points)
    
    # Linear interpolation
    values = start_value + (end_value - start_value) * t
    
    # Add noise
    value_range = abs(end_value - start_value)
    noise = np.random.normal(0, value_range * noise_level, n_points)
    values = values + noise
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_exponential_growth(
    duration: int,
    interval: float = 1.0,
    start_value: float = 10.0,
    growth_rate: float = 0.001,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate an exponential growth pattern.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        start_value: Starting value
        growth_rate: Exponential growth rate per second
        noise_level: Noise as fraction of value (0.0-1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    t = np.arange(n_points) * interval
    
    # Exponential growth
    values = start_value * np.exp(growth_rate * t)
    
    # Add multiplicative noise
    noise = np.random.normal(1, noise_level, n_points)
    values = values * noise
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_step_function(
    duration: int,
    interval: float = 1.0,
    levels: List[float] = [50, 100, 75, 125],
    step_duration: Optional[float] = None,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a step function pattern.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        levels: List of step levels
        step_duration: Duration of each step (default: duration / len(levels))
        noise_level: Noise as fraction of mean level (0.0-1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    
    if step_duration is None:
        step_duration = duration / len(levels)
    
    values = np.zeros(n_points)
    for i in range(n_points):
        t = i * interval
        level_idx = min(int(t / step_duration), len(levels) - 1)
        values[i] = levels[level_idx]
    
    # Add noise
    mean_level = np.mean(levels)
    noise = np.random.normal(0, mean_level * noise_level, n_points)
    values = values + noise
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_sawtooth(
    duration: int,
    interval: float = 1.0,
    min_value: float = 50.0,
    max_value: float = 150.0,
    period: float = 120.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a sawtooth wave pattern.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        min_value: Minimum value
        max_value: Maximum value
        period: Wave period in seconds
        noise_level: Noise as fraction of range (0.0-1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    t = np.arange(n_points) * interval
    
    # Sawtooth: linear ramp from 0 to 1 over each period
    phase = (t % period) / period
    values = min_value + (max_value - min_value) * phase
    
    # Add noise
    value_range = max_value - min_value
    noise = np.random.normal(0, value_range * noise_level, n_points)
    values = values + noise
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_square_wave(
    duration: int,
    interval: float = 1.0,
    low_value: float = 50.0,
    high_value: float = 150.0,
    period: float = 60.0,
    duty_cycle: float = 0.5,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a square wave pattern.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        low_value: Low state value
        high_value: High state value
        period: Wave period in seconds
        duty_cycle: Fraction of period in high state (0.0-1.0)
        noise_level: Noise as fraction of range (0.0-1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    t = np.arange(n_points) * interval
    
    # Square wave
    phase = (t % period) / period
    values = np.where(phase < duty_cycle, high_value, low_value)
    
    # Add noise
    value_range = high_value - low_value
    noise = np.random.normal(0, value_range * noise_level, n_points)
    values = values + noise
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_spiky(
    duration: int,
    interval: float = 1.0,
    baseline: float = 50.0,
    spike_height: float = 100.0,
    spike_probability: float = 0.05,
    spike_decay: float = 0.8,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a spiky pattern with random spikes on a baseline.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        baseline: Baseline value
        spike_height: Maximum spike height above baseline
        spike_probability: Probability of spike at each point
        spike_decay: Decay factor for spike (0.0-1.0)
        noise_level: Noise as fraction of baseline (0.0-1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    n_points = int(duration / interval)
    values = np.ones(n_points) * baseline
    
    # Generate spikes
    current_spike = 0
    for i in range(n_points):
        if random.random() < spike_probability:
            current_spike = spike_height * random.uniform(0.5, 1.0)
        
        values[i] = baseline + current_spike
        current_spike *= spike_decay
    
    # Add noise
    noise = np.random.normal(0, baseline * noise_level, n_points)
    values = values + noise
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_seasonal(
    duration: int,
    interval: float = 1.0,
    daily_amplitude: float = 30.0,
    weekly_amplitude: float = 20.0,
    baseline: float = 100.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a seasonal pattern with daily and weekly components.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        daily_amplitude: Amplitude of daily cycle
        weekly_amplitude: Amplitude of weekly cycle
        baseline: Baseline value
        noise_level: Noise as fraction of baseline (0.0-1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    t = np.arange(n_points) * interval
    
    # Daily cycle (period = 86400 seconds, but scaled for demo)
    daily_period = 300  # 5 minutes for demo
    daily = daily_amplitude * np.sin(2 * np.pi * t / daily_period)
    
    # Weekly cycle (longer period)
    weekly_period = 1800  # 30 minutes for demo
    weekly = weekly_amplitude * np.sin(2 * np.pi * t / weekly_period)
    
    values = baseline + daily + weekly
    
    # Add noise
    noise = np.random.normal(0, baseline * noise_level, n_points)
    values = values + noise
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_random_walk(
    duration: int,
    interval: float = 1.0,
    start_value: float = 100.0,
    step_std: float = 2.0,
    drift: float = 0.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a random walk pattern.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        start_value: Starting value
        step_std: Standard deviation of each step
        drift: Drift per step (positive = upward trend)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    steps = np.random.normal(drift, step_std, n_points)
    values = start_value + np.cumsum(steps)
    values = np.maximum(values, 0)  # Keep non-negative
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


def generate_constant(
    duration: int,
    interval: float = 1.0,
    value: float = 100.0,
    noise_level: float = 0.02,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a constant/flat pattern with minimal noise.
    
    Args:
        duration: Total duration in seconds
        interval: Sampling interval in seconds
        value: Constant value
        noise_level: Noise as fraction of value (0.0-1.0)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_points = int(duration / interval)
    values = np.ones(n_points) * value
    
    # Add minimal noise
    noise = np.random.normal(0, value * noise_level, n_points)
    values = values + noise
    
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i * interval) for i in range(n_points)]
    
    return pd.DataFrame({'timestamp': timestamps, 'value': values})


# Pattern registry
PATTERNS: Dict[str, Callable] = {
    'sine': generate_sine_wave,
    'gaussian': generate_gaussian_noise,
    'trend_up': lambda **kw: generate_linear_trend(start_value=50, end_value=150, **kw),
    'trend_down': lambda **kw: generate_linear_trend(start_value=150, end_value=50, **kw),
    'exponential': generate_exponential_growth,
    'step': generate_step_function,
    'sawtooth': generate_sawtooth,
    'square': generate_square_wave,
    'spiky': generate_spiky,
    'seasonal': generate_seasonal,
    'random_walk': generate_random_walk,
    'constant': generate_constant,
}


def generate_all_patterns(
    output_dir: str,
    duration: int = 3600,
    interval: float = 1.0,
    noise_level: float = 0.05,
    seed: int = 42,
    patterns: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Generate all patterns and save to CSV files.
    
    Args:
        output_dir: Directory to save CSV files
        duration: Duration of each pattern in seconds
        interval: Sampling interval in seconds
        noise_level: Default noise level
        seed: Random seed for reproducibility
        patterns: List of pattern names to generate (default: all)
        
    Returns:
        Dict mapping pattern names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if patterns is None:
        patterns = list(PATTERNS.keys())
    
    generated_files = {}
    
    for i, pattern_name in enumerate(patterns):
        if pattern_name not in PATTERNS:
            print(f"Warning: Unknown pattern '{pattern_name}', skipping")
            continue
        
        generator = PATTERNS[pattern_name]
        
        # Use different seed for each pattern for variety
        pattern_seed = seed + i * 1000
        
        print(f"Generating: {pattern_name}...", end=" ")
        
        try:
            # Generate with common parameters
            df = generator(
                duration=duration,
                interval=interval,
                noise_level=noise_level,
                seed=pattern_seed
            )
            
            # Save to CSV
            filename = f"{pattern_name}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            
            generated_files[pattern_name] = filepath
            print(f"✓ {len(df)} points -> {filename}")
            
        except TypeError as e:
            # Some generators don't accept all parameters
            df = generator(duration=duration, interval=interval, seed=pattern_seed)
            filename = f"{pattern_name}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            generated_files[pattern_name] = filepath
            print(f"✓ {len(df)} points -> {filename}")
    
    return generated_files


def generate_training_config(output_dir: str, generated_files: Dict[str, str]) -> str:
    """
    Generate a training_config.yaml file for the classifier.
    
    Args:
        output_dir: Directory containing the CSV files (used for reference only)
        generated_files: Dict mapping pattern names to file paths
        
    Returns:
        Path to the generated config file
    """
    # Config goes in classification/train/, not in csv/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    config_path = os.path.join(repo_root, 'classification', 'train', 'training_config.yaml')
    
    lines = [
        "# Auto-generated training configuration",
        "# Generated by generate_patterns.py",
        "#",
        "# Each pattern type is a separate class for classification.",
        "",
        "classes:",
    ]
    
    for pattern_name, filepath in sorted(generated_files.items()):
        filename = os.path.basename(filepath)
        lines.append(f"  {pattern_name}:")
        lines.append(f"    - {filename}")
    
    with open(config_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic time series patterns for classification training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available patterns:
  sine        - Sinusoidal wave
  gaussian    - Gaussian (white) noise
  trend_up    - Linear upward trend
  trend_down  - Linear downward trend
  exponential - Exponential growth
  step        - Step function
  sawtooth    - Sawtooth wave
  square      - Square wave
  spiky       - Random spikes on baseline
  seasonal    - Seasonal pattern (daily + weekly)
  random_walk - Random walk / Brownian motion
  constant    - Flat/constant with minimal noise

Examples:
  # Generate all patterns to classification/train/
  python generate_patterns.py
  
  # Generate specific patterns
  python generate_patterns.py --patterns sine gaussian trend_up
  
  # Custom parameters
  python generate_patterns.py --duration 7200 --noise 0.1 --output-dir ./data
  
  # Generate and update training config
  python generate_patterns.py --update-config
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for CSV files (default: ../classification/train/)'
    )
    
    parser.add_argument(
        '--patterns',
        nargs='+',
        choices=list(PATTERNS.keys()),
        default=None,
        help='Patterns to generate (default: all)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Duration in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Sampling interval in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--noise',
        type=float,
        default=0.05,
        help='Noise level as fraction (default: 0.05 = 5%%)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--update-config',
        action='store_true',
        help='Update training_config.yaml with generated patterns'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available patterns and exit'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable patterns:")
        print("=" * 40)
        for name in sorted(PATTERNS.keys()):
            print(f"  {name}")
        print("=" * 40)
        return 0
    
    # Set default output directory
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        args.output_dir = os.path.join(repo_root, 'csv')
    
    print(f"\n{'=' * 60}")
    print("Pattern Generator for Classification Training")
    print(f"{'=' * 60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Duration: {args.duration}s ({args.duration/60:.1f} minutes)")
    print(f"Interval: {args.interval}s")
    print(f"Noise level: {args.noise * 100:.1f}%")
    print(f"Seed: {args.seed}")
    print(f"{'=' * 60}\n")
    
    # Generate patterns
    generated = generate_all_patterns(
        output_dir=args.output_dir,
        duration=args.duration,
        interval=args.interval,
        noise_level=args.noise,
        seed=args.seed,
        patterns=args.patterns
    )
    
    print(f"\n✓ Generated {len(generated)} pattern files")
    
    # Update training config if requested
    if args.update_config:
        config_path = generate_training_config(args.output_dir, generated)
        print(f"✓ Updated training config: {config_path}")
    
    print(f"\nTo train the classifier:")
    print(f"  cd classification")
    print(f"  python classify_main.py --train-from-config")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
