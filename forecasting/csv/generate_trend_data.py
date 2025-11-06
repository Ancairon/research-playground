#!/usr/bin/env python3
"""
Generate time series data with highly predictable pattern.
Supports different granularities: second, minute, or hour.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

def generate_data(days, granularity, pattern='predictable'):
    """
    Generate time series data.
    
    Args:
        days: Number of days of data to generate
        granularity: 'second', 'minute', or 'hour'
        pattern: 'predictable' or 'realistic'
    """
    # Calculate points per day based on granularity
    if granularity == 'second':
        points_per_day = 86400  # 24 * 60 * 60
        interval_seconds = 1
    elif granularity == 'minute':
        points_per_day = 1440   # 24 * 60
        interval_seconds = 60
    elif granularity == 'hour':
        points_per_day = 24
        interval_seconds = 3600
    else:
        raise ValueError(f"Invalid granularity: {granularity}. Use 'second', 'minute', or 'hour'")
    
    total_points = days * points_per_day
    
    # Start date
    start_date = datetime(2025, 8, 1, 0, 0, 0)
    
    # Generate timestamps
    timestamps = [start_date + timedelta(seconds=i * interval_seconds) for i in range(total_points)]
    
    # Generate values based on pattern type
    if pattern == 'predictable':
        values = _generate_predictable_pattern(total_points, interval_seconds)
        pattern_desc = "Highly predictable with clean periodic patterns"
    elif pattern == 'realistic':
        values = _generate_realistic_pattern(total_points, interval_seconds)
        pattern_desc = "Realistic with noise, trends, and anomalies"
    else:
        raise ValueError(f"Invalid pattern: {pattern}. Use 'predictable' or 'realistic'")
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    # Round values to 3 decimal places
    df['value'] = df['value'].round(3)
    
    return df, points_per_day, pattern_desc


def _generate_predictable_pattern(total_points, interval_seconds):
    """Generate highly predictable patterns."""
    values = []
    
    for i in range(total_points):
        # Time in seconds since start
        t = i * interval_seconds
        
        # === BASE VALUE ===
        base = 50.0
        
        # === PATTERN 1: 60-second cycle (very predictable) ===
        pattern_60s = 10 * np.sin(2 * np.pi * t / 60)
        
        # === PATTERN 2: 5-minute cycle ===
        pattern_5min = 5 * np.sin(2 * np.pi * t / 300)
        
        # === PATTERN 3: Hourly cycle ===
        pattern_1h = 8 * np.cos(2 * np.pi * t / 3600)
        
        # === PATTERN 4: Daily cycle ===
        seconds_per_day = 86400
        hour_of_day = (t % seconds_per_day) / 3600
        if 6 <= hour_of_day < 22:  # Daytime
            daily_boost = 15
        else:  # Nighttime
            daily_boost = -10
        
        # === MINIMAL NOISE ===
        noise = np.random.normal(0, 0.5)
        
        # === COMBINE ALL COMPONENTS ===
        value = base + pattern_60s + pattern_5min + pattern_1h + daily_boost + noise
        
        # Ensure non-negative
        value = max(1.0, value)
        
        values.append(value)
    
    return values


def _generate_realistic_pattern(total_points, interval_seconds):
    """Generate realistic patterns with noise, trends, and anomalies."""
    values = []
    
    # Random walk component for trend drift
    trend = 0
    trend_momentum = 0
    
    # Anomaly tracking
    last_anomaly_time = -10000
    
    for i in range(total_points):
        # Time in seconds since start
        t = i * interval_seconds
        
        # === BASE VALUE WITH SLOW DRIFT ===
        # Base drifts up/down slowly over time
        base = 50.0 + trend
        
        # Random walk for trend (with mean reversion)
        trend_momentum = 0.95 * trend_momentum + np.random.normal(0, 0.02)
        trend += trend_momentum
        # Mean reversion - pull trend back toward 0
        trend *= 0.999
        
        # === MULTIPLE OVERLAPPING CYCLES (with phase drift) ===
        # Short cycle (1-2 minutes, slightly irregular)
        short_period = 90 + 30 * np.sin(t / 1000)  # Period varies 60-120s
        pattern_short = 8 * np.sin(2 * np.pi * t / short_period)
        
        # Medium cycle (10-15 minutes, with harmonics)
        medium_period = 720
        pattern_medium = 6 * np.sin(2 * np.pi * t / medium_period)
        pattern_medium += 2 * np.sin(4 * np.pi * t / medium_period)  # 2nd harmonic
        
        # Long cycle (hourly, with irregularity)
        long_period = 3600 + 300 * np.sin(t / 5000)  # Period varies
        pattern_long = 10 * np.cos(2 * np.pi * t / long_period)
        
        # === DAILY PATTERN (with weekend effect) ===
        seconds_per_day = 86400
        day_of_week = (i * interval_seconds // seconds_per_day) % 7
        hour_of_day = (t % seconds_per_day) / 3600
        
        # Business hours effect (stronger on weekdays)
        is_weekend = day_of_week >= 5
        if 9 <= hour_of_day < 17:  # Business hours
            daily_boost = 20 if not is_weekend else 8
        elif 6 <= hour_of_day < 9 or 17 <= hour_of_day < 22:  # Morning/evening
            daily_boost = 10 if not is_weekend else 5
        else:  # Night
            daily_boost = -15
        
        # === REALISTIC NOISE (heteroscedastic - varies with value) ===
        # Noise level increases with value (multiplicative noise)
        current_value_estimate = base + pattern_short + pattern_medium + pattern_long + daily_boost
        noise_level = 1.5 + 0.02 * abs(current_value_estimate)
        noise = np.random.normal(0, noise_level)
        
        # === OCCASIONAL SPIKES/ANOMALIES ===
        spike = 0
        if i - last_anomaly_time > 1000:  # At least 1000 points since last anomaly
            if np.random.random() < 0.001:  # 0.1% chance per point
                spike_type = np.random.choice(['spike', 'dip', 'burst'])
                if spike_type == 'spike':
                    spike = np.random.uniform(20, 50)
                elif spike_type == 'dip':
                    spike = np.random.uniform(-30, -15)
                else:  # burst
                    spike = np.random.uniform(10, 25)
                last_anomaly_time = i
        
        # === AUTOCORRELATED NOISE (AR(1) process) ===
        if i > 0:
            # Current noise slightly correlated with previous
            prev_noise_component = 0.3 * (values[-1] - current_value_estimate)
            noise += prev_noise_component
        
        # === COMBINE ALL COMPONENTS ===
        value = base + pattern_short + pattern_medium + pattern_long + daily_boost + noise + spike
        
        # Ensure non-negative with realistic floor
        value = max(5.0, value)
        
        values.append(value)
    
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate time series data with various patterns')
    parser.add_argument('--days', type=int, default=14, help='Number of days of data (default: 14)')
    parser.add_argument('--granularity', type=str, default='second', 
                        choices=['second', 'minute', 'hour'],
                        help='Time granularity: second, minute, or hour (default: second)')
    parser.add_argument('--pattern', type=str, default='predictable',
                        choices=['predictable', 'realistic'],
                        help='Pattern type: predictable (easy) or realistic (challenging) (default: predictable)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated based on parameters)')
    
    args = parser.parse_args()
    
    # Generate data
    df, points_per_day, pattern_desc = generate_data(args.days, args.granularity, args.pattern)
    
    # Auto-generate filename if not provided
    if args.output is None:
        output_file = f'{args.days}day_{args.pattern}_{args.granularity}.csv'
    else:
        output_file = args.output
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df):,} data points ({args.days} days)")
    print(f"Granularity: {args.granularity} ({points_per_day:,} points per day)")
    print(f"Pattern: {args.pattern} - {pattern_desc}")
    print(f"Saved to: {output_file}")
    print(f"\nData summary:")
    print(f"  Start date: {df['timestamp'].iloc[0]}")
    print(f"  End date:   {df['timestamp'].iloc[-1]}")
    print(f"  Mean value: {df['value'].mean():.3f}")
    print(f"  Min value:  {df['value'].min():.3f}")
    print(f"  Max value:  {df['value'].max():.3f}")
    print(f"  Std dev:    {df['value'].std():.3f}")
    
    if args.pattern == 'predictable':
        print(f"\nPattern components:")
        print(f"  - 60-second cycle (±10)")
        print(f"  - 5-minute cycle (±5)")
        print(f"  - 1-hour cycle (±8)")
        print(f"  - Daily day/night pattern (±15/-10)")
        print(f"  - Minimal noise (σ=0.5)")
    else:
        print(f"\nPattern components:")
        print(f"  - Irregular short cycles (1-2 min, varying period)")
        print(f"  - Medium cycles (10-15 min, with harmonics)")
        print(f"  - Long cycles (hourly, irregular)")
        print(f"  - Daily patterns (with weekend effects)")
        print(f"  - Trend drift (random walk with mean reversion)")
        print(f"  - Heteroscedastic noise (σ varies with value)")
        print(f"  - Autocorrelated noise (AR(1) process)")
        print(f"  - Random anomalies (spikes, dips, bursts)")
    
    print(f"\nUsage examples:")
    print(f"  # Predictable pattern")
    print(f"  python3 generate_trend_data.py --days 7 --granularity minute --pattern predictable")
    print(f"  # Realistic pattern")
    print(f"  python3 generate_trend_data.py --days 14 --granularity second --pattern realistic")
    print(f"  # Custom output")
    print(f"  python3 generate_trend_data.py --days 30 --pattern realistic --output my_data.csv")
