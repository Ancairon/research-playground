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

    Realistic patterns based on actual production scenarios.
    Patterns 1-5 are periodic and learnable - difficulty increases with complexity.
    
    1. heartbeat - Health check every 10s (spike to 150, quiet at 50 KB/s) - EASIEST
    2. api_requests - API polling: 15s primary + 8s secondary bursts + 60s baseline wave
    3. batch_job - Batch processing every 15s: 1.5s ramp-up, 4s processing, 1.5s ramp-down, 8s idle
    4. database_backup - 45s backup cycle (5s index, 10s data transfer, 5s verify, 25s idle) + 20s query load + 18s large queries
    5. web_traffic - Multi-source: 60s user wave, 25s cron, 12s CDN, 8s analytics, 35s crawlers, 120s trend
    6. chaos - Completely unpredictable random traffic with no learnable pattern - ULTIMATE CHALLENGE
    """
    
    # PATTERN 1: HEARTBEAT (EASIEST)
    # Scenario: Health check or monitoring probe sending data every 10 seconds
    # Small spike followed by quiet period - very predictable
    if pattern == 'heartbeat':
        cycle = t % 10  # 10-second heartbeat interval
        
        if cycle < 0.5:
            # Quick spike during heartbeat transmission
            base = 150 - 100 * (cycle / 0.5)
        else:
            # Quiet baseline with minimal activity
            base = 50 + 5 * math.sin(cycle * 2 * math.pi / 10)
        
        noise = RNG.gauss(0, 3) * NOISE_SCALE
        return max(10, base + noise)
    
    # PATTERN 2: API_REQUESTS (EASY-MEDIUM)
    # Scenario: Multiple clients polling API endpoints at different intervals
    # Main polling: every 15s, Secondary polling: every 8s, Background: slow variation
    if pattern == 'api_requests':
        # Baseline activity - ongoing connections
        baseline = 80 + 15 * math.sin(t * 2 * math.pi / 60)
        
        # Primary API polling (15-second interval)
        cycle_15 = t % 15
        if cycle_15 < 1.5:
            primary = 60 * math.exp(-cycle_15 / 0.5)
        else:
            primary = 0
        
        # Secondary faster polling (8-second interval, smaller)
        cycle_8 = t % 8
        if cycle_8 < 0.8:
            secondary = 35 * (1 - cycle_8 / 0.8)
        else:
            secondary = 0
        
        base = baseline + primary + secondary
        noise = RNG.gauss(0, 5) * NOISE_SCALE
        return max(10, base + noise)
    
    # PATTERN 3: BATCH_JOB (MEDIUM)
    # Scenario: Scheduled batch processing (e.g., log aggregation, data export)
    # Runs every 15 seconds, takes ~6 seconds with ramp-up and processing phases
    if pattern == 'batch_job':
        cycle = t % 15  # 15-second batch interval
        
        if cycle < 1.5:
            # Ramp-up phase: preparing data
            base = 100 + 120 * (cycle / 1.5)
        elif cycle < 5.5:
            # Processing phase: high sustained load
            processing_t = cycle - 1.5
            base = 220 + 50 * math.sin(processing_t * math.pi / 2)
        elif cycle < 7:
            # Ramp-down phase: finalizing and cleanup
            base = 220 - 130 * ((cycle - 5.5) / 1.5)
        else:
            # Idle phase: minimal background activity
            idle_t = cycle - 7
            base = 90 + 15 * math.sin(idle_t * 2 * math.pi / 8)
        
        noise = RNG.gauss(0, 8) * NOISE_SCALE
        return max(10, base + noise)
    
    # PATTERN 4: DATABASE_BACKUP (MEDIUM-HARD)
    # Scenario: Incremental database backup running every 45 seconds
    # Progressive load: starts slow (indexes), peaks (data), tapers (verification)
    # Interference from ongoing queries creates secondary pattern
    if pattern == 'database_backup':
        cycle = t % 45  # 45-second backup cycle
        
        # Backup activity profile
        if cycle < 5:
            # Initial phase: reading indexes and metadata
            backup = 70 + 40 * (cycle / 5)
        elif cycle < 15:
            # Heavy phase: transferring main data (peaks in middle)
            heavy_t = cycle - 5
            backup = 110 + 70 * math.sin(heavy_t * math.pi / 10)
        elif cycle < 20:
            # Verification phase: tapering off
            backup = 110 - 60 * ((cycle - 15) / 5)
        else:
            # Post-backup: light activity
            backup = 50
        
        # Ongoing query load (20-second period, independent of backup)
        query_load = 25 * math.sin(t * 2 * math.pi / 20)
        
        # Occasional large query every 18 seconds
        cycle_18 = t % 18
        if cycle_18 < 2:
            large_query = 40 * math.exp(-cycle_18 / 0.7)
        else:
            large_query = 0
        
        base = backup + query_load + large_query
        noise = RNG.gauss(0, 7) * NOISE_SCALE
        return max(10, base + noise)
    
    # PATTERN 5: WEB_TRAFFIC (HARDEST PREDICTABLE)
    # Scenario: Realistic web server with multiple overlapping patterns
    # - Base traffic oscillation (simulates user activity waves)
    # - Regular cron jobs every 25 seconds
    # - CDN cache refresh every 12 seconds
    # - Analytics beacon collection every 8 seconds
    # - Occasional crawler bots every 35 seconds
    if pattern == 'web_traffic':
        # Base traffic: organic user activity (60-second wave)
        base_traffic = 100 + 40 * math.sin(t * 2 * math.pi / 60)
        
        # User activity micro-pattern (15-second variation)
        user_micro = 20 * math.sin(t * 2 * math.pi / 15)
        
        # Cron jobs: scheduled tasks every 25 seconds (3s duration)
        cycle_25 = t % 25
        if cycle_25 < 3:
            cron = 50 * (1 - cycle_25 / 3) * math.exp(-cycle_25 / 1.5)
        else:
            cron = 0
        
        # CDN cache refresh: every 12 seconds (quick burst)
        cycle_12 = t % 12
        if cycle_12 < 0.8:
            cdn = 35 * (1 - cycle_12 / 0.8)
        else:
            cdn = 0
        
        # Analytics beacons: every 8 seconds (sustained mini-burst)
        cycle_8 = t % 8
        if cycle_8 < 1.5:
            analytics = 25 * math.exp(-cycle_8 / 0.6)
        else:
            analytics = 0
        
        # Crawler bots: every 35 seconds (longer sustained load)
        cycle_35 = t % 35
        if cycle_35 < 4:
            crawler = 60 * math.exp(-cycle_35 / 2.0)
        else:
            crawler = 0
        
        # Long-term trend (simulates daily pattern at small scale)
        # 120-second slow oscillation
        trend = 15 * math.sin(t * 2 * math.pi / 120)
        
        base = base_traffic + user_micro + cron + cdn + analytics + crawler + trend
        noise = RNG.gauss(0, 8) * NOISE_SCALE
        return max(10, base + noise)
    
    # PATTERN 6: CHAOS (UNPREDICTABLE)
    # Scenario: Truly chaotic, unpredictable traffic
    # Every second is completely random with no learnable pattern
    # Uses multiple random components that change independently
    # This is the ultimate stress test - models should fail to predict accurately
    if pattern == 'chaos':
        # Use time as seed to get different random values each second
        # But combine with multiple random sources for maximum chaos
        
        # Random baseline that changes completely every second
        RNG.seed(int(t * 7919))  # Prime number for better randomization
        baseline = RNG.uniform(40, 180)
        
        # Random spikes - high probability, highly variable
        RNG.seed(int(t * 9973))
        if RNG.random() < 0.25:  # 25% chance of spike
            spike = RNG.uniform(0, 150)
        else:
            spike = 0
        
        # Random drops - can suddenly go very low
        RNG.seed(int(t * 10007))
        if RNG.random() < 0.15:  # 15% chance of drop
            drop = -RNG.uniform(30, 100)
        else:
            drop = 0
        
        # Multiple random oscillations with random periods and amplitudes
        RNG.seed(int(t * 10009) + 1)
        osc1_period = RNG.uniform(3, 20)
        osc1_amp = RNG.uniform(10, 40)
        osc1 = osc1_amp * math.sin(t * 2 * math.pi / osc1_period)
        
        RNG.seed(int(t * 10009) + 2)
        osc2_period = RNG.uniform(5, 30)
        osc2_amp = RNG.uniform(5, 30)
        osc2 = osc2_amp * math.cos(t * 2 * math.pi / osc2_period)
        
        # Random walks - cumulative randomness
        walk_val = 0
        RNG.seed(int(t * 10007) + 3)
        for i in range(int(t) % 10):  # Changes based on time
            walk_val += RNG.uniform(-5, 5)
        
        # Sudden random jumps
        RNG.seed(int(t * 10007) + 4)
        if RNG.random() < 0.10:  # 10% chance
            jump = RNG.choice([-80, -60, -40, 40, 60, 80])
        else:
            jump = 0
        
        # Random bursts of activity
        RNG.seed(int(t * 10007) + 5)
        burst_duration = int(RNG.uniform(1, 5))
        if int(t) % 10 < burst_duration:
            burst = RNG.uniform(20, 80)
        else:
            burst = 0
        
        # Chaotic jitter
        RNG.seed(int(t * 10007) + 6)
        jitter = RNG.uniform(-30, 30)
        
        base = baseline + spike + drop + osc1 + osc2 + walk_val + jump + burst + jitter
        
        # Extreme noise
        RNG.seed(int(t * 10007) + 7)
        noise = RNG.gauss(0, 25) * NOISE_SCALE
        
        return max(10, base + noise)
    
    # DEFAULT: Fall back to heartbeat
    cycle = t % 10
    if cycle < 0.5:
        base = 150 - 100 * (cycle / 0.5)
    else:
        base = 50 + 5 * math.sin(cycle * 2 * math.pi / 10)
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
            
            # Clean up finished processes
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
            
            # Limit concurrent downloads to prevent resource exhaustion
            max_concurrent_downloads = 10
            
            # Kill oldest processes if we exceed the limit (stuck processes)
            while len(processes) >= max_concurrent_downloads:
                try:
                    oldest = processes[0]
                    oldest.terminate()
                    oldest.wait(timeout=0.1)
                    processes.pop(0)
                except:
                    processes.pop(0)  # Remove from list even if termination fails
            
            # Calculate how many downloads we need based on target bandwidth
            # Each download gets a share of the bandwidth
            # Higher bandwidth = more concurrent downloads
            if target_kbps > 10:
                # Scale downloads based on bandwidth: 1 download per ~15 KB/s
                # This gives better visibility of patterns
                desired_downloads = min(max(1, int(target_kbps / 15)), max_concurrent_downloads)
                
                # Start new downloads to reach desired level
                downloads_to_start = max(0, desired_downloads - len(processes))
                
                servers = [
                    'http://speedtest.tele2.net/1MB.zip',
                    'http://ipv4.download.thinkbroadband.com/1MB.zip',
                    'http://proof.ovh.net/files/1Mb.dat',
                ]
                
                for i in range(downloads_to_start):
                    if len(processes) >= max_concurrent_downloads:
                        break
                    
                    server_url = servers[(step + i) % len(servers)]
                    
                    # Distribute bandwidth across all downloads
                    bandwidth_per_download = int(target_kbps / desired_downloads)
                    
                    cmd = [
                        'curl',
                        '-s',  # Silent
                        '--max-time', '2',  # 2-second total timeout
                        '--connect-timeout', '1',  # 1-second connection timeout
                        '--limit-rate', f'{bandwidth_per_download}K',  # Rate limit per download
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
        choices=['heartbeat', 'api_requests', 'batch_job', 'database_backup', 'web_traffic', 'chaos'],
        default='heartbeat',
        help='Traffic pattern to simulate:\n'
             '  heartbeat: Health check every 10s (spike: 150 KB/s, quiet: 50 KB/s) - EASIEST\n'
             '  api_requests: API polling (15s + 8s cycles) on 60s baseline wave\n'
             '  batch_job: Every 15s - 1.5s ramp-up, 4s processing, 1.5s ramp-down, 8s idle\n'
             '  database_backup: Every 45s backup + 20s query cycles + 18s large queries\n'
             '  web_traffic: Multi-source (60s users, 25s cron, 12s CDN, 8s analytics, 35s bots, 120s trend)\n'
             '  chaos: Completely random - no learnable pattern - ULTIMATE CHALLENGE'
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

=== REALISTIC SCENARIO PATTERNS (EASY → HARD) ===

1. heartbeat (EASIEST)
   • Health check probe every 10 seconds
   • Sharp spike (150 KB/s) then quiet (50 KB/s)
   • Range: 45-155 KB/s
   • Tests: Simple periodic spike detection

2. api_requests (EASY-MEDIUM)
   • Multiple API polling cycles (15s primary, 8s secondary)
   • Baseline wave (60s period) with overlapping bursts
   • Range: 60-180 KB/s
   • Tests: Multi-period detection, overlapping patterns

3. batch_job (MEDIUM)
   • Scheduled processing every 15 seconds
   • Phases: 1.5s ramp-up → 4s processing → 1.5s ramp-down → 8s idle
   • Range: 75-270 KB/s
   • Tests: Phase detection, state transitions

4. database_backup (MEDIUM-HARD)
   • 45s backup cycle (index → transfer → verify)
   • Overlapping: 20s query load + 18s large query bursts
   • Range: 40-230 KB/s
   • Tests: Complex overlapping periodic patterns

5. web_traffic (HARD)
   • Realistic multi-source traffic simulation
   • 60s user wave, 25s cron jobs, 12s CDN refresh, 8s analytics, 35s crawler bots
   • Long-term 120s trend modulation
   • Range: 50-280 KB/s
   • Tests: Multiple overlapping periodicities at different scales

6. chaos (ULTIMATE CHALLENGE)
   • Completely unpredictable random traffic
   • Random spikes, drops, bursts, walks - no learnable pattern
   • Range: 10-300+ KB/s
   • Tests: Model behavior under true randomness (should fail)

=== USAGE ===
  Start easy:  python simulate_network_traffic.py --pattern heartbeat
  Challenge:   python simulate_network_traffic.py --pattern web_traffic
  Stress test: python simulate_network_traffic.py --pattern chaos
  
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
