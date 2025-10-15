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

def generate_traffic_pattern(t, pattern='daily'):
    """
    Generate traffic value based on time and pattern.
    Returns bandwidth in KB/s
    Fast cycles for testing (patterns visible within seconds/minutes)
    """
    if pattern == 'daily':
        # Smooth sine wave: 30 second cycle (2 full cycles per minute)
        # Good for testing smoothing and prediction stability
        base = 100 + 80 * math.sin(t * 2 * math.pi / 30)  # 30s period
        noise = random.gauss(0, 8)  # Low noise for smooth predictions
        return max(10, base + noise)
    
    elif pattern == 'business_hours':
        # Step pattern: 20s high, 10s low (repeats every 30s)
        # Tests model's ability to predict step changes
        second = t % 30
        if second < 20:
            base = 150 + 20 * math.sin(second * math.pi / 10)
        else:
            base = 40 + 10 * random.random()
        
        noise = random.gauss(0, 10)
        return max(10, base + noise)
    
    elif pattern == 'spiky':
        # Sharp spikes every 15 seconds
        # Tests retraining on sudden changes and prediction smoothing
        second = t % 15
        if second < 2:  # 2-second spike every 15s
            base = 250 + 50 * random.random()
        else:
            base = 50 + 15 * math.sin(t * 2 * math.pi / 15)
        
        noise = random.gauss(0, 10)
        return max(10, base + noise)
    
    elif pattern == 'sawtooth':
        # Linear ramp up, sharp drop (20 second cycle)
        # Tests model's ability to track trends and predict drops
        cycle = (t % 20) / 20  # Position in 20s cycle
        if cycle < 0.75:  # Ramp up for 15 seconds
            base = 50 + 120 * (cycle / 0.75)
        else:  # Sharp drop for 5 seconds
            base = 50 + 20 * (1 - (cycle - 0.75) / 0.25)
        
        noise = random.gauss(0, 8)
        return max(10, base + noise)
    
    elif pattern == 'trend':
        # Upward trend with cyclical component (60 second cycle, then reset)
        # Tests model retraining as baseline shifts
        second_of_minute = t % 60
        base = 60 + (second_of_minute) * 2  # Increases 2 KB/s per second
        
        # Add 10-second cycle on top of trend
        cycle = 25 * math.sin(t * 2 * math.pi / 10)
        
        noise = random.gauss(0, 12)
        return max(10, base + cycle + noise)
    
    else:  # 'sine' or default
        # Fast sine wave: 20 second period (3 cycles per minute)
        # Clean predictable pattern for baseline testing
        base = 100 + 60 * math.sin(t * 2 * math.pi / 20)
        noise = random.gauss(0, 5)  # Very low noise
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
            
            # Clean up finished processes
            processes = [p for p in processes if p.poll() is None]
            
            # Generate actual network traffic using curl/wget to download from fast servers
            # Calculate bytes to download this interval
            bytes_to_download = int(target_kbps * 1024 * interval)
            
            if bytes_to_download > 0:
                # Download from fast.com or similar - using /dev/null endpoint
                # Using curl with rate limiting
                cmd = [
                    'curl',
                    '-s',  # Silent
                    '-m', str(interval + 1),  # Timeout
                    '--limit-rate', f'{int(target_kbps)}K',  # Rate limit
                    '-o', '/dev/null',  # Discard output
                    'http://speedtest.tele2.net/1MB.zip'  # Test file
                ]
                
                try:
                    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    processes.append(p)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not start download: {e}")
            
            if verbose:
                print(f"[{step:4d}] t={elapsed:6.1f}s | Target: {target_kbps:6.1f} KB/s | Active downloads: {len(processes)}")
            
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
        choices=['daily', 'business_hours', 'spiky', 'sawtooth', 'trend', 'sine'],
        default='daily',
        help='Traffic pattern to simulate'
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
╔══════════════════════════════════════════════════════════╗
║    Network Traffic Simulator (Fast Patterns)            ║
╚══════════════════════════════════════════════════════════╝

Pattern Descriptions (fast cycles for testing):
  daily          - Smooth sine: 30s cycle (2 per min) | Low noise
                   Range: 20-180 KB/s | Good for: smoothing tests
  
  business_hours - Step pattern: 20s high, 10s low (30s cycle)
                   Range: 40-170 KB/s | Good for: step change prediction
  
  spiky          - Sharp spikes every 15s (2s duration)
                   Range: 50-300 KB/s | Good for: retraining triggers
  
  sawtooth       - Linear ramp: 15s up, 5s drop (20s cycle)
                   Range: 50-170 KB/s | Good for: trend tracking
  
  trend          - Upward trend + 10s sine (resets every 60s)
                   Range: 60-200+ KB/s | Good for: model retraining
  
  sine           - Fast clean sine: 20s period (3 per min)
                   Range: 40-160 KB/s | Good for: baseline testing

All patterns optimized for 1-5 minute testing runs.
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
