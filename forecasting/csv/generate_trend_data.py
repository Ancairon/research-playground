#!/usr/bin/env python3
"""
Generate realistic time series data with 5 difficulty levels.
Each level progressively adds complexity and realism.

Difficulty Levels:
1. EASY: Clean periodic patterns, minimal noise
2. MODERATE: Multiple cycles with some noise and trend
3. CHALLENGING: Irregular patterns, heteroscedastic noise, anomalies
4. HARD: Regime changes, non-stationary patterns, complex dependencies
5. EXTREME: All of the above plus external shocks, breaks, and chaos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

def generate_data(days, granularity, difficulty=1):
    """
    Generate time series data.
    
    Args:
        days: Number of days of data to generate
        granularity: 'second', 'minute', or 'hour'
        difficulty: 1 (easy) to 5 (extreme)
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
    
    # Generate values based on difficulty level
    if difficulty == 1:
        values = _generate_easy_pattern(total_points, interval_seconds)
        difficulty_desc = "EASY: Clean periodic patterns, minimal noise"
    elif difficulty == 2:
        values = _generate_moderate_pattern(total_points, interval_seconds)
        difficulty_desc = "MODERATE: Multiple cycles with noise and trend"
    elif difficulty == 3:
        values = _generate_challenging_pattern(total_points, interval_seconds)
        difficulty_desc = "CHALLENGING: Irregular patterns, heteroscedastic noise, anomalies"
    elif difficulty == 4:
        values = _generate_hard_pattern(total_points, interval_seconds)
        difficulty_desc = "HARD: Regime changes, non-stationary patterns, complex dependencies"
    elif difficulty == 5:
        values = _generate_extreme_pattern(total_points, interval_seconds)
        difficulty_desc = "EXTREME: External shocks, structural breaks, chaotic components"
    else:
        raise ValueError(f"Invalid difficulty: {difficulty}. Use 1 (easy) to 5 (extreme)")
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    # Round values to 3 decimal places
    df['value'] = df['value'].round(3)
    
    return df, points_per_day, difficulty_desc


# === DIFFICULTY LEVEL 1: EASY ===
def _generate_easy_pattern(total_points, interval_seconds):
    """
    Clean, highly predictable patterns.
    - Regular sine/cosine waves at fixed frequencies
    - Minimal noise
    - Clear day/night cycle
    - Easy to forecast
    """
    values = []
    
    for i in range(total_points):
        t = i * interval_seconds
        
        # Base value
        base = 50.0
        
        # Simple periodic patterns
        pattern_1min = 10 * np.sin(2 * np.pi * t / 60)        # 1-minute cycle
        pattern_5min = 5 * np.sin(2 * np.pi * t / 300)        # 5-minute cycle
        pattern_1hour = 8 * np.cos(2 * np.pi * t / 3600)      # 1-hour cycle
        
        # Simple day/night effect
        hour_of_day = (t % 86400) / 3600
        if 6 <= hour_of_day < 22:  # Day
            daily_effect = 15
        else:  # Night
            daily_effect = -10
        
        # Very small noise
        noise = np.random.normal(0, 0.5)
        
        value = base + pattern_1min + pattern_5min + pattern_1hour + daily_effect + noise
        values.append(max(1.0, value))
    
    return values


# === DIFFICULTY LEVEL 2: MODERATE ===
def _generate_moderate_pattern(total_points, interval_seconds):
    """
    Moderate complexity with realistic elements.
    - Multiple overlapping cycles
    - Gradual trend drift
    - Moderate noise
    - Business hour patterns
    """
    values = []
    trend = 0
    
    for i in range(total_points):
        t = i * interval_seconds
        
        # Base with slow drift
        base = 50.0 + trend
        trend += np.random.normal(0, 0.01)  # Random walk
        trend *= 0.998  # Mean reversion
        
        # Multiple cycles
        pattern_short = 8 * np.sin(2 * np.pi * t / 90)        # 1.5-minute cycle
        pattern_medium = 6 * np.sin(2 * np.pi * t / 600)      # 10-minute cycle
        pattern_long = 10 * np.cos(2 * np.pi * t / 3600)      # 1-hour cycle
        
        # Business hours effect
        hour_of_day = (t % 86400) / 3600
        day_of_week = (i * interval_seconds // 86400) % 7
        is_weekend = day_of_week >= 5
        
        if 9 <= hour_of_day < 17:  # Business hours
            daily_effect = 20 if not is_weekend else 5
        elif 6 <= hour_of_day < 9 or 17 <= hour_of_day < 22:
            daily_effect = 8 if not is_weekend else 2
        else:
            daily_effect = -12
        
        # Moderate noise
        noise = np.random.normal(0, 2.0)
        
        value = base + pattern_short + pattern_medium + pattern_long + daily_effect + noise
        values.append(max(3.0, value))
    
    return values


# === DIFFICULTY LEVEL 3: CHALLENGING ===
def _generate_challenging_pattern(total_points, interval_seconds):
    """
    Challenging with real-world complexity.
    - Irregular cycle periods
    - Heteroscedastic noise (variance changes)
    - Random anomalies
    - Autocorrelated noise
    """
    values = []
    trend = 0
    trend_momentum = 0
    last_anomaly = -10000
    
    for i in range(total_points):
        t = i * interval_seconds
        
        # Base with trend momentum
        base = 50.0 + trend
        trend_momentum = 0.95 * trend_momentum + np.random.normal(0, 0.02)
        trend += trend_momentum
        trend *= 0.999
        
        # Irregular cycles (periods vary)
        short_period = 90 + 30 * np.sin(t / 1000)
        pattern_short = 8 * np.sin(2 * np.pi * t / short_period)
        
        pattern_medium = 6 * np.sin(2 * np.pi * t / 720)
        pattern_medium += 2 * np.sin(4 * np.pi * t / 720)  # Harmonic
        
        long_period = 3600 + 300 * np.sin(t / 5000)
        pattern_long = 10 * np.cos(2 * np.pi * t / long_period)
        
        # Complex daily pattern
        hour_of_day = (t % 86400) / 3600
        day_of_week = (i * interval_seconds // 86400) % 7
        is_weekend = day_of_week >= 5
        
        if 9 <= hour_of_day < 17:
            daily_effect = 20 if not is_weekend else 8
        elif 6 <= hour_of_day < 9 or 17 <= hour_of_day < 22:
            daily_effect = 10 if not is_weekend else 5
        else:
            daily_effect = -15
        
        # Heteroscedastic noise (depends on current value)
        current_estimate = base + pattern_short + pattern_medium + pattern_long + daily_effect
        noise_level = 1.5 + 0.02 * abs(current_estimate)
        noise = np.random.normal(0, noise_level)
        
        # Autocorrelated noise
        if i > 0:
            noise += 0.3 * (values[-1] - current_estimate)
        
        # Random anomalies
        spike = 0
        if i - last_anomaly > 1000 and np.random.random() < 0.001:
            spike_type = np.random.choice(['spike', 'dip', 'burst'])
            if spike_type == 'spike':
                spike = np.random.uniform(20, 50)
            elif spike_type == 'dip':
                spike = np.random.uniform(-30, -15)
            else:
                spike = np.random.uniform(10, 25)
            last_anomaly = i
        
        value = base + pattern_short + pattern_medium + pattern_long + daily_effect + noise + spike
        values.append(max(5.0, value))
    
    return values


# === DIFFICULTY LEVEL 4: HARD ===
def _generate_hard_pattern(total_points, interval_seconds):
    """
    Hard: Multiple overlapping patterns with dynamic parameters - recognizable but challenging.
    - 3 overlapping cycles with time-varying frequencies and amplitudes
    - Regime switching (volatility and pattern strength changes)
    - Non-stationary trends
    - Pattern phase shifts and modulation
    - Complex but still predictable with the right model
    """
    values = []
    t = 0
    trend = 0.0
    regime_state = 0  # 0=normal, 1=high volatility, 2=low activity
    regime_counter = 0
    
    # Pattern parameters that evolve over time
    freq_1 = 0.005  # Fast cycle
    freq_2 = 0.002  # Medium cycle
    freq_3 = 0.0005  # Slow cycle
    
    amp_1 = 10.0
    amp_2 = 15.0
    amp_3 = 8.0
    
    phase_1 = 0.0
    phase_2 = 0.0
    phase_3 = 0.0
    
    volatility_regime = 1.0
    trend_strength = 0.02
    
    for i in range(total_points):
        t = i * interval_seconds
        
        # Regime switching (stays in regime for ~1000-3000 points)
        regime_counter += 1
        if regime_counter > np.random.randint(1000, 3000):
            regime_state = np.random.choice([0, 1, 2])
            regime_counter = 0
            
            if regime_state == 0:  # Normal
                volatility_regime = 1.0
                trend_strength = 0.02
            elif regime_state == 1:  # High volatility - patterns get stronger
                volatility_regime = 2.5
                trend_strength = 0.04
                # Amplify patterns
                amp_1 = 15.0
                amp_2 = 20.0
                amp_3 = 12.0
            else:  # Low activity - patterns get weaker
                volatility_regime = 0.3
                trend_strength = 0.005
                # Dampen patterns
                amp_1 = 5.0
                amp_2 = 8.0
                amp_3 = 4.0
        
        # Slowly modulate frequencies (periods change over time)
        freq_1_mod = freq_1 * (1 + 0.3 * np.sin(2 * np.pi * i / 5000))
        freq_2_mod = freq_2 * (1 + 0.2 * np.cos(2 * np.pi * i / 8000))
        freq_3_mod = freq_3 * (1 + 0.15 * np.sin(2 * np.pi * i / 12000))
        
        # Slowly modulate amplitudes
        amp_1_mod = amp_1 * (1 + 0.2 * np.cos(2 * np.pi * i / 6000))
        amp_2_mod = amp_2 * (1 + 0.15 * np.sin(2 * np.pi * i / 9000))
        amp_3_mod = amp_3 * (1 + 0.1 * np.cos(2 * np.pi * i / 15000))
        
        # Generate overlapping patterns
        pattern_1 = amp_1_mod * np.sin(2 * np.pi * freq_1_mod * i + phase_1)
        pattern_2 = amp_2_mod * np.sin(2 * np.pi * freq_2_mod * i + phase_2)
        pattern_3 = amp_3_mod * np.cos(2 * np.pi * freq_3_mod * i + phase_3)
        
        # Occasional phase shifts
        if i % 2000 == 0 and i > 0:
            phase_1 += np.random.uniform(-np.pi/4, np.pi/4)
            phase_2 += np.random.uniform(-np.pi/6, np.pi/6)
        
        # Non-stationary trend
        base = 50.0 + trend
        trend += np.random.normal(0, trend_strength)
        trend *= 0.997  # Weak mean reversion
        
        # Calendar effects
        hour_of_day = (t % 86400) / 3600
        day_of_week = (i * interval_seconds // 86400) % 7
        day_of_month = ((i * interval_seconds // 86400) % 30) + 1
        
        is_weekend = day_of_week >= 5
        is_month_end = day_of_month >= 28
        
        if 9 <= hour_of_day < 17:
            daily_effect = 15 if not is_weekend else 5
            if is_month_end:
                daily_effect *= 1.3  # Month-end bump
        elif 6 <= hour_of_day < 9 or 17 <= hour_of_day < 22:
            daily_effect = 8 if not is_weekend else 3
        else:
            daily_effect = -8
        
        # Long-memory noise with autocorrelation
        noise_base = np.random.normal(0, 3.0 * volatility_regime)
        if i >= 5:
            # Add dependencies on past values (ARMA-like)
            memory_effect = sum([0.3 ** j * (values[-j] - 50) for j in range(1, 6)]) / 10
            noise_base += memory_effect
        
        # Combine all components
        value = base + pattern_1 + pattern_2 + pattern_3 + daily_effect + noise_base
        
        # Keep in reasonable range
        value = np.clip(value, 5.0, 200.0)
        values.append(value)
    
    return values


# === DIFFICULTY LEVEL 5: EXTREME ===
def _generate_extreme_pattern(total_points, interval_seconds):
    """
    Extreme difficulty with all complexities.
    - Everything from level 4
    - Structural breaks (permanent level shifts)
    - External shocks (large temporary disturbances)
    - Chaotic components
    - Multiplicative effects
    - All kept bounded to reasonable values
    """
    values = []
    trend = 0.0
    volatility_regime = 1.0
    trend_strength = 0.02
    regime_state = 0
    regime_counter = 0
    structural_level_shift = 0.0
    chaos_state = 0.1
    shock_effect = 0.0  # Current shock effect (decays over time)
    shock_decay_rate = 1.0
    
    for i in range(total_points):
        t = i * interval_seconds
        
        # Structural breaks (rare but permanent shifts)
        if np.random.random() < 0.0002:  # Very rare - ~3 per 10 days
            shift = np.random.normal(0, 15)
            structural_level_shift += shift
            structural_level_shift = np.clip(structural_level_shift, -40, 40)
            if abs(shift) > 5:
                print(f"Structural break at point {i}: shift = {shift:.1f}")
        
        # Regime switching (volatility states)
        regime_counter += 1
        if regime_counter > np.random.randint(800, 3000):
            regime_state = np.random.choice([0, 1, 2, 3], p=[0.5, 0.25, 0.15, 0.1])
            regime_counter = 0
            
            if regime_state == 0:  # Normal
                volatility_regime = 1.0
                trend_strength = 0.02
            elif regime_state == 1:  # High volatility
                volatility_regime = 2.5
                trend_strength = 0.04
            elif regime_state == 2:  # Low activity
                volatility_regime = 0.3
                trend_strength = 0.005
            else:  # Chaotic
                volatility_regime = 3.5
                trend_strength = 0.06
        
        # External shocks (rare, large, decaying)
        if np.random.random() < 0.0008:  # Rare - ~10 per 10 days
            shock_effect = np.random.normal(0, 30)
            shock_decay_rate = 0.95 + np.random.uniform(0, 0.04)  # Decay between 0.95-0.99
            if abs(shock_effect) > 15:
                print(f"External shock at point {i}: magnitude = {shock_effect:.1f}")
        
        # Decay existing shock
        shock_effect *= shock_decay_rate
        if abs(shock_effect) < 0.5:
            shock_effect = 0.0
        
        # Non-stationary trend (bounded)
        base = 50.0 + trend + structural_level_shift
        trend += np.random.normal(0, trend_strength)
        trend = np.clip(trend, -30, 30)  # Keep bounded
        trend *= 0.992  # Very weak mean reversion
        
        # Chaotic component (logistic map - naturally bounded 0-1)
        chaos_state = 3.9 * chaos_state * (1 - chaos_state)
        chaos_component = 8 * (chaos_state - 0.5)  # Scale to -4 to +4
        
        # Complex overlapping cycles
        cycle_phase = t / 1000
        pattern_1 = 10 * np.sin(2 * np.pi * t / (100 + 50 * np.sin(cycle_phase)))
        pattern_2 = 8 * np.sin(2 * np.pi * t / (600 + 200 * np.cos(cycle_phase / 2)))
        pattern_3 = 12 * np.cos(2 * np.pi * t / 3600) * (1 + 0.3 * np.sin(cycle_phase / 3))
        pattern_4 = 6 * np.sin(2 * np.pi * t / 1800)  # Extra cycle
        
        # Multiplicative interaction (bounded)
        interaction = np.clip(0.15 * pattern_1 * pattern_2 / 20, -10, 10)
        
        # Complex calendar effects
        hour_of_day = (t % 86400) / 3600
        day_of_week = (i * interval_seconds // 86400) % 7
        day_of_month = ((i * interval_seconds // 86400) % 30) + 1
        week_of_month = (day_of_month - 1) // 7 + 1
        
        is_weekend = day_of_week >= 5
        is_monday = day_of_week == 0
        is_friday = day_of_week == 4
        is_month_end = day_of_month >= 28
        is_first_week = week_of_month == 1
        
        # Complex business hour effects
        if 9 <= hour_of_day < 17:
            daily_effect = 20 if not is_weekend else 5
            if is_month_end:
                daily_effect *= 1.5
            if is_monday:
                daily_effect *= 1.2
            if is_friday and is_first_week:
                daily_effect *= 0.8
        elif 6 <= hour_of_day < 9:
            daily_effect = 10 if not is_weekend else 2
        elif 17 <= hour_of_day < 22:
            daily_effect = 6 if not is_weekend else 3
        else:
            daily_effect = -12
        
        # Long-memory noise with regime dependence
        noise_base = np.random.normal(0, 3.5 * volatility_regime)
        if i >= 10:
            # Longer memory in extreme mode
            memory_weights = [0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]
            memory_effect = sum([w * (values[-j-1] - 50) for j, w in enumerate(memory_weights)]) / 8
            noise_base += memory_effect
        
        # Heavy-tailed noise (occasional extreme values)
        if np.random.random() < 0.015:  # 1.5% chance
            noise_base *= np.random.uniform(2.0, 4.0)
        
        # State-dependent heteroscedasticity
        current_estimate = base + pattern_1 + pattern_2 + pattern_3 + pattern_4
        noise_multiplier = 1 + 0.03 * abs(current_estimate - 50) / 50
        noise = noise_base * noise_multiplier
        
        # Combine all components
        value = (base + pattern_1 + pattern_2 + pattern_3 + pattern_4 + 
                interaction + chaos_component + daily_effect + shock_effect + noise)
        
        # Final bounding to keep in reasonable range
        value = np.clip(value, 5.0, 200.0)
        values.append(value)
    
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate realistic time series data with 5 difficulty levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Difficulty Levels:
  1 (EASY)        - Clean periodic patterns, minimal noise
  2 (MODERATE)    - Multiple cycles with noise and trend drift
  3 (CHALLENGING) - Irregular patterns, heteroscedastic noise, anomalies
  4 (HARD)        - Regime changes, non-stationary, complex dependencies
  5 (EXTREME)     - Structural breaks, external shocks, chaotic components

Trend Options:
  --trend up      - Add upward trend over time
  --trend down    - Add downward trend over time
  --trend none    - No additional trend (default)

Examples:
  # Easy pattern for testing
  python generate_trend_data.py --days 7 --difficulty 1
  
  # Moderate difficulty with upward trend (noisy but trending up)
  python generate_trend_data.py --days 14 --difficulty 2 --trend up --granularity minute
  
  # Medium difficulty with strong downward trend
  python generate_trend_data.py --days 10 --difficulty 2 --trend down --trend-strength 0.02
  
  # Extreme challenge for stress testing
  python generate_trend_data.py --days 30 --difficulty 5 --output extreme_challenge.csv
        """
    )
    parser.add_argument('--days', type=int, default=14, 
                        help='Number of days of data (default: 14)')
    parser.add_argument('--granularity', type=str, default='minute', 
                        choices=['second', 'minute', 'hour'],
                        help='Time granularity (default: minute)')
    parser.add_argument('--difficulty', type=int, default=2, choices=[1, 2, 3, 4, 5],
                        help='Difficulty level 1-5 (default: 2)')
    parser.add_argument('--trend', type=str, default=None, choices=['up', 'down', 'none'],
                        help='Add overall trend: up, down, or none (default: none)')
    parser.add_argument('--trend-strength', type=float, default=0.005,
                        help='Trend strength as additive change per point (default: 0.005)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Generate data
    df, points_per_day, difficulty_desc = generate_data(args.days, args.granularity, args.difficulty)
    
    # Apply trend if requested
    if args.trend and args.trend != 'none':
        print(f"\nApplying {args.trend}ward trend (strength: {args.trend_strength})...")
        
        # Calculate trend component
        total_points = len(df)
        trend_multiplier = 1.0 if args.trend == 'up' else -1.0
        
        # Additive linear trend - simpler and more predictable
        original_mean = df['value'].mean()
        original_std = df['value'].std()
        
        # Create smooth trend line
        trend_values = np.linspace(0, trend_multiplier * args.trend_strength * total_points, total_points)
        
        # Add the trend
        df['value'] = df['value'] + trend_values
        
        # For downtrend, shift up if we're getting too low
        if args.trend == 'down':
            min_val = df['value'].min()
            if min_val < 10:
                shift = 15 - min_val
                df['value'] = df['value'] + shift
        
        # Ensure no values below 5
        df['value'] = df['value'].clip(lower=5.0)

        df['value'] = df['value'].clip(lower=5.0)
    
    # Auto-generate filename if not provided
    if args.output is None:
        difficulty_names = ['', 'easy', 'moderate', 'challenging', 'hard', 'extreme']
        trend_suffix = f'_{args.trend}trend' if args.trend and args.trend != 'none' else ''
        output_file = f'{args.days}day_{difficulty_names[args.difficulty]}_{args.granularity}{trend_suffix}.csv'
    else:
        output_file = args.output
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Generated {len(df):,} data points ({args.days} days)")
    print(f"{'='*70}")
    print(f"Granularity: {args.granularity} ({points_per_day:,} points per day)")
    print(f"Difficulty:  Level {args.difficulty} - {difficulty_desc}")
    if args.trend and args.trend != 'none':
        print(f"Trend:       {args.trend.upper()}ward (strength: {args.trend_strength})")
    print(f"Output file: {output_file}")
    print(f"\nData range:")
    print(f"  Start:  {df['timestamp'].iloc[0]}")
    print(f"  End:    {df['timestamp'].iloc[-1]}")
    print(f"\nValue statistics:")
    print(f"  Mean:   {df['value'].mean():.3f}")
    print(f"  Std:    {df['value'].std():.3f}")
    print(f"  Min:    {df['value'].min():.3f}")
    print(f"  Max:    {df['value'].max():.3f}")
    print(f"  Median: {df['value'].median():.3f}")
    
    # Difficulty-specific info
    if args.difficulty == 1:
        print(f"\nLevel 1 (EASY) characteristics:")
        print(f"  ✓ Regular sine/cosine waves")
        print(f"  ✓ Fixed frequencies (1min, 5min, 1hour)")
        print(f"  ✓ Simple day/night cycle")
        print(f"  ✓ Minimal noise (σ=0.5)")
    elif args.difficulty == 2:
        print(f"\nLevel 2 (MODERATE) characteristics:")
        print(f"  ✓ Multiple overlapping cycles")
        print(f"  ✓ Gradual trend drift")
        print(f"  ✓ Business hour patterns")
        print(f"  ✓ Weekend effects")
        print(f"  ✓ Moderate noise (σ=2.0)")
    elif args.difficulty == 3:
        print(f"\nLevel 3 (CHALLENGING) characteristics:")
        print(f"  ✓ Irregular cycle periods")
        print(f"  ✓ Harmonic components")
        print(f"  ✓ Heteroscedastic noise")
        print(f"  ✓ Autocorrelated noise (AR(1))")
        print(f"  ✓ Random anomalies (spikes, dips)")
    elif args.difficulty == 4:
        print(f"\nLevel 4 (HARD) characteristics:")
        print(f"  ✓ Multiple regime states")
        print(f"  ✓ Non-stationary patterns")
        print(f"  ✓ Long-memory dependencies")
        print(f"  ✓ Month-end effects")
        print(f"  ✓ Dynamic volatility")
    elif args.difficulty == 5:
        print(f"\nLevel 5 (EXTREME) characteristics:")
        print(f"  ✓ Everything from level 4")
        print(f"  ✓ Structural breaks (permanent shifts)")
        print(f"  ✓ External shocks (rare, large)")
        print(f"  ✓ Chaotic components (logistic map)")
        print(f"  ✓ Multiplicative interactions")
        print(f"  ✓ Heavy-tailed noise")
        print(f"  ✓ Complex calendar effects")
    
    print(f"\n{'='*70}")
    print(f"Use this data with forecast_main.py:")
    print(f"  python forecast_main.py --csv-file {output_file} --model lstm")
    print(f"{'='*70}\n")
