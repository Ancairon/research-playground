#!/usr/bin/env python3
"""
Simulate random/unpatterned network traffic for forecasting testing.
Creates unpredictable traffic with no discernible pattern.
"""

import argparse
import time
import random
import subprocess
import signal
import sys

def generate_random_traffic(t):
    """
    Generate completely random traffic value.
    Returns bandwidth in KB/s
    """
    # Multiple random components make it unpredictable
    
    # Random walk
    if not hasattr(generate_random_traffic, 'last_value'):
        generate_random_traffic.last_value = 100
    
    change = random.gauss(0, 30)
    generate_random_traffic.last_value += change
    generate_random_traffic.last_value = max(5, min(500, generate_random_traffic.last_value))
    
    # Add random jumps
    if random.random() < 0.1:  # 10% chance of random jump
        generate_random_traffic.last_value = random.uniform(10, 400)
    
    # Add noise
    noise = random.gauss(0, 20)
    
    return max(5, generate_random_traffic.last_value + noise)


def simulate_random_traffic(duration=3600, interval=1, mode='chaotic', verbose=True):
    """
    Simulate unpatterned network traffic.
    
    Args:
        duration: How long to run (seconds)
        interval: Update interval (seconds)
        mode: Type of randomness (chaotic, jumpy, noisy, uniform)
        verbose: Print traffic levels
    """
    print(f"Starting random network traffic simulation...")
    print(f"Mode: {mode}")
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"Interval: {interval}s")
    print("-" * 60)
    
    start_time = time.time()
    step = 0
    processes = []
    last_value = 100
    
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
            
            # Generate traffic level based on mode
            if mode == 'chaotic':
                # Random walk with jumps
                change = random.gauss(0, 30)
                last_value += change
                if random.random() < 0.15:  # 15% chance of jump
                    last_value = random.uniform(10, 400)
                last_value = max(5, min(500, last_value))
                target_kbps = last_value + random.gauss(0, 20)
                
            elif mode == 'jumpy':
                # Frequent random jumps
                if random.random() < 0.3:  # 30% chance of new value
                    last_value = random.uniform(20, 300)
                target_kbps = last_value + random.gauss(0, 10)
                
            elif mode == 'noisy':
                # High noise random walk
                change = random.gauss(0, 50)
                last_value += change
                last_value = max(10, min(400, last_value))
                target_kbps = last_value + random.gauss(0, 40)
                
            elif mode == 'uniform':
                # Pure uniform random
                target_kbps = random.uniform(10, 300)
                last_value = target_kbps
                
            else:  # 'white_noise'
                # Gaussian white noise around mean
                target_kbps = random.gauss(150, 80)
            
            target_kbps = max(5, target_kbps)
            
            # Clean up finished processes
            processes = [p for p in processes if p.poll() is None]
            
            # Generate actual network traffic
            bytes_to_download = int(target_kbps * 1024 * interval)
            
            if bytes_to_download > 0:
                cmd = [
                    'curl',
                    '-s',
                    '-m', str(interval + 1),
                    '--limit-rate', f'{int(target_kbps)}K',
                    '-o', '/dev/null',
                    'http://speedtest.tele2.net/1MB.zip'
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
        description="Simulate random/unpatterned network traffic for forecasting tests"
    )
    
    parser.add_argument(
        '--mode',
        choices=['chaotic', 'jumpy', 'noisy', 'uniform', 'white_noise'],
        default='chaotic',
        help='Type of randomness'
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
║    Network Traffic Simulator (Random/Unpatterned)       ║
╚══════════════════════════════════════════════════════════╝

Randomness Modes:
  chaotic     - Random walk with occasional jumps (default)
  jumpy       - Frequent random level changes
  noisy       - High variance random walk
  uniform     - Uniformly distributed random values
  white_noise - Gaussian noise around mean

⚠️  This traffic has NO predictable pattern
   Use for testing forecasting model limitations

Press Ctrl+C to stop
""")
    
    simulate_random_traffic(
        duration=args.duration,
        interval=args.interval,
        mode=args.mode,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
