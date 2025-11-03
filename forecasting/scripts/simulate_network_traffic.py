#!/usr/bin/env python3
"""
Simulate patterned network traffic for forecasting testing.
Creates realistic traffic patterns: daily cycles, bursts, trends.
"""

import argparse
import time
import math
import random
import subprocess
import signal
import sys
from collections import deque

# Global RNG and noise control. Settable from simulate_traffic (seed / noise scale).
RNG = random.Random()
NOISE_SCALE = 1.0
def generate_traffic_pattern(t, pattern='daily'):
    """
    Generate traffic value based on time and pattern.
    Returns bandwidth in KB/s

    Five patterns from easy to hard for forecasting models.
    ALL patterns are periodic and learnable - difficulty is in complexity.
    
    1. simple_sine - Pure sine wave (EASIEST)
    2. step_pattern - Regular high/low steps with smooth transitions
    3. sawtooth_burst - Sawtooth with periodic bursts
    4. multi_period - Multiple overlapping periodicities
    5. compound_wave - Complex but fully deterministic pattern (HARDEST)
    """
    
    # PATTERN 1: SIMPLE SINE (EASIEST)
    # Pure sine wave - smooth, predictable, single frequency
    # Period: 30 seconds, fully periodic
    if pattern == 'simple_sine':
        base = 120 + 80 * math.sin(t * 2 * math.pi / 30)
        noise = RNG.gauss(0, 3) * NOISE_SCALE
        return max(10, base + noise)
    
    # PATTERN 2: STEP PATTERN (EASY-MEDIUM)
    # Two-level pattern with smooth sine transitions
    # Period: 40 seconds (20s high, 20s low)
    if pattern == 'step_pattern':
        cycle = t % 40
        if cycle < 20:
            # High state with slight variation
            base = 170 + 15 * math.sin(cycle * math.pi / 10)
        else:
            # Low state with slight variation
            base = 70 + 10 * math.sin((cycle - 20) * math.pi / 10)
        noise = RNG.gauss(0, 5) * NOISE_SCALE
        return max(10, base + noise)
    
    # PATTERN 3: SAWTOOTH BURST (MEDIUM)
    # Sawtooth baseline + regular predictable bursts
    # Sawtooth period: 25s, Burst period: 15s
    if pattern == 'sawtooth_burst':
        # Sawtooth component (25-second period)
        cycle_25 = t % 25
        sawtooth = 60 + (100 * cycle_25 / 25)
        
        # Periodic burst every 15 seconds (2-second duration)
        cycle_15 = t % 15
        if cycle_15 < 2:
            burst = 80 * math.exp(-cycle_15 / 0.8)
        else:
            burst = 0
        
        base = sawtooth + burst
        noise = RNG.gauss(0, 6) * NOISE_SCALE
        return max(10, base + noise)
    
    # PATTERN 4: MULTI-PERIOD (MEDIUM-HARD)
    # Three distinct periodic components that interact
    # Periods: 12s (fast), 30s (medium), 60s (slow)
    if pattern == 'multi_period':
        # Fast oscillation (12-second period)
        fast = 35 * math.sin(t * 2 * math.pi / 12)
        
        # Medium oscillation (30-second period)
        medium = 50 * math.sin(t * 2 * math.pi / 30)
        
        # Slow oscillation (60-second period)
        slow = 30 * math.sin(t * 2 * math.pi / 60)
        
        base = 120 + fast + medium + slow
        noise = RNG.gauss(0, 7) * NOISE_SCALE
        return max(10, base + noise)
    
    # PATTERN 5: COMPOUND WAVE (HARDEST)
    # Complex deterministic pattern with multiple interacting periodicities
    # Combines: baseline wave, two burst cycles, and amplitude modulation
    # Main period: 60s, but with substructure at 20s, 15s, and 8s
    if pattern == 'compound_wave':
        # Primary slow wave (60-second period)
        primary = 120 + 40 * math.sin(t * 2 * math.pi / 60)
        
        # Secondary wave (20-second period) - modulates amplitude
        secondary = 30 * math.sin(t * 2 * math.pi / 20)
        
        # Fast oscillation (8-second period)
        fast = 25 * math.sin(t * 2 * math.pi / 8)
        
        # Periodic burst pattern (every 15 seconds, 3-second duration)
        cycle_15 = t % 15
        if cycle_15 < 3:
            # Burst strength varies with position in 60s cycle
            burst_mod = 0.7 + 0.3 * math.sin(t * 2 * math.pi / 60)
            burst = 60 * burst_mod * math.exp(-cycle_15 / 1.2)
        else:
            burst = 0
        
        # Amplitude modulation (every 45 seconds - interacts with main period)
        amp_mod = 1.0 + 0.3 * math.sin(t * 2 * math.pi / 45)
        
        # Combine all components with amplitude modulation
        base = (primary + secondary + fast) * amp_mod + burst
        
        # Moderate noise - pattern is complex but deterministic
        noise = RNG.gauss(0, 8) * NOISE_SCALE
        return max(10, base + noise)
    
    # DEFAULT: Fall back to simple sine
    base = 120 + 80 * math.sin(t * 2 * math.pi / 30)
    noise = RNG.gauss(0, 3) * NOISE_SCALE
    return max(10, base + noise)


def simulate_traffic(pattern='daily', duration=3600, interval=1, verbose=True):
    """
    Simulate network traffic by generating download activity.
    
    Args:
        pattern: Traffic pattern type
        duration: How long to run (seconds)
        interval: Update interval (seconds)
        verbose: Print traffic levels
    """
    print(f"Starting network traffic simulation...")
    print(f"Pattern: {pattern}")
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"Interval: {interval}s")
    print("-" * 60)
    
    start_time = time.time()
    step = 0
    
    # For actual traffic generation
    processes = []
    
    def cleanup(signum=None, frame=None):
        """Clean up running processes"""
        print("\n\nStopping traffic simulation...")
        for p in processes:
            try:
                p.terminate()
            except:
                pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            
            # Generate traffic level
            target_kbps = generate_traffic_pattern(elapsed, pattern)
            
            # Clean up finished processes more aggressively
            new_processes = []
            for p in processes:
                if p.poll() is None:  # Still running
                    new_processes.append(p)
                else:
                    try:
                        p.terminate()  # Force cleanup
                        p.wait(timeout=0.1)
                    except:
                        pass
            processes = new_processes
            
            # Kill oldest process if we have too many stuck ones
            if len(processes) >= 2:
                try:
                    oldest = processes[0]
                    oldest.terminate()
                    oldest.wait(timeout=0.1)
                    processes.pop(0)
                except:
                    pass
            
            # Limit concurrent downloads to prevent resource exhaustion
            max_concurrent_downloads = 10
            
            if len(processes) < max_concurrent_downloads and target_kbps > 10:
                # Generate actual network traffic using curl/wget to download from fast servers
                # Using curl with VERY short timeout to prevent pileup
                # Rotate between multiple servers to avoid per-server connection limits
                servers = [
                    'http://speedtest.tele2.net/1MB.zip',
                    'http://ipv4.download.thinkbroadband.com/1MB.zip',
                    'http://proof.ovh.net/files/1Mb.dat',
                ]
                server_url = servers[step % len(servers)]
                
                cmd = [
                    'curl',
                    '-s',  # Silent
                    '--max-time', '1',  # Hard 1-second total timeout
                    '--connect-timeout', '1',  # 1-second connection timeout
                    '--limit-rate', f'{int(target_kbps)}K',  # Rate limit
                    '-o', '/dev/null',  # Discard output
                    server_url
                ]
                
                try:
                    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    processes.append(p)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not start download: {e}")
            
            if verbose:
                print(f"[{step:4d}] t={elapsed:6.1f}s | Target: {target_kbps:6.1f} KB/s | Active downloads: {len(processes)} (limit: {max_concurrent_downloads})")
            
            step += 1
            time.sleep(interval)
        
        cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate patterned network traffic for forecasting tests"
    )
    
    parser.add_argument(
        '--pattern',
        choices=['simple_sine', 'step_pattern', 'sawtooth_burst', 'multi_period', 'compound_wave'],
        default='simple_sine',
        help='Traffic pattern to simulate (difficulty: easy → hard)\n'
             'simple_sine: Pure sine wave (30s period) - EASIEST\n'
             'step_pattern: High/low levels with smooth transitions (40s cycle)\n'
             'sawtooth_burst: Ramp pattern + periodic bursts (25s + 15s)\n'
             'multi_period: Three overlapping waves (12s/30s/60s periods)\n'
             'compound_wave: Complex multi-component pattern - HARDEST'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Simulation duration in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='Update interval in seconds (default: 1)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   Network Traffic Simulator (Forecasting Test Patterns)     ║
╚══════════════════════════════════════════════════════════════╝

=== DIFFICULTY PROGRESSION (EASY → HARD) ===

1. simple_sine (EASIEST)
   • Pure sine wave, 30-second period
   • Smooth, predictable, single frequency
   • Range: 40-200 KB/s
   • Perfect for: Testing basic model capability
   
2. square_wave (EASY-MEDIUM)
   • Regular high/low transitions, 40-second cycle
   • Sharp state changes, no gradual transitions
   • Range: 60-180 KB/s
   • Tests: Step detection, handling discontinuities
   
3. sawtooth (MEDIUM)
   • Linear ramp up, sharp drop, 25-second cycle
   • Predictable trend with sudden reset
   • Range: 60-200 KB/s
   • Tests: Trend detection, handling sudden changes
   
4. multi_frequency (MEDIUM-HARD)
   • Multiple overlapping sine waves (10s, 25s, 60s periods)
   • Complex interference patterns
   • Range: 30-250 KB/s
   • Tests: Frequency decomposition, pattern separation
   
5. burst_chaos (HARDEST)
   • Irregular bursts with multiple periodicities
   • 4 different burst sources (12s, 18s, 27s, 45s)
   • State-dependent behavior, unpredictable spikes
   • Range: 80-450 KB/s
   • Tests: Multi-scale learning, burst prediction, chaos handling

=== USAGE ===
  Start easy:  python simulate_network_traffic.py --pattern simple_sine
  Challenge:   python simulate_network_traffic.py --pattern burst_chaos
  
Press Ctrl+C to stop
""")
    
    simulate_traffic(
        pattern=args.pattern,
        duration=args.duration,
        interval=args.interval,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
