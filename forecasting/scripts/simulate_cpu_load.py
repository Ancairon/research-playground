#!/usr/bin/env python3
"""
Simulate CPU load patterns for forecasting testing.
Creates predictable CPU usage patterns that Netdata can measure.
"""

import argparse
import time
import math
import multiprocessing
import signal
import sys

# Global flag for worker processes
should_run = multiprocessing.Value('i', 1)


def cpu_work(intensity, duration=1.0):
    """
    Perform CPU-intensive work for a given duration.
    
    Args:
        intensity: 0.0-1.0, percentage of time to use CPU
        duration: How long to work (seconds)
    """
    start = time.time()
    while time.time() - start < duration:
        if intensity > 0:
            # Do CPU-intensive work
            work_time = duration * intensity
            work_start = time.time()
            
            # Busy loop with calculations
            x = 0
            while time.time() - work_start < work_time and time.time() - start < duration:
                x += math.sqrt(x + 1) * math.sin(x) * math.cos(x)
            
            # Sleep for remaining time
            sleep_time = duration * (1 - intensity)
            if sleep_time > 0:
                time.sleep(min(sleep_time, duration - (time.time() - start)))
        else:
            time.sleep(duration)


def worker_process(intensity_queue, worker_id):
    """Worker process that consumes CPU based on queue intensity."""
    while should_run.value:
        try:
            intensity = intensity_queue.get(timeout=0.1)
            cpu_work(intensity, duration=1.0)
        except:
            time.sleep(0.1)


def generate_cpu_pattern(t, pattern='web_server'):
    """
    Generate CPU usage percentage (0-100) based on time and pattern.
    
    Args:
        t: Time in seconds since start
        pattern: Pattern name
        
    Returns:
        CPU usage percentage (0-100)
    """
    if pattern == 'web_server':
        # Web server: baseline + periodic request bursts every 15s
        cycle_15s = t % 15
        base = 20 + 10 * math.sin(t * 2 * math.pi / 60)
        
        if cycle_15s < 3:  # 3-second burst
            burst = 50 * math.exp(-cycle_15s / 1.5)
            base += burst
        
        return min(100, max(5, base))
    
    elif pattern == 'api_service':
        # API with regular polling + batch jobs
        cycle_10s = t % 10
        if cycle_10s < 1.5:
            base = 60
        else:
            base = 15 + 5 * math.sin(t * 2 * math.pi / 10)
        
        # Batch job every 30s
        cycle_30s = t % 30
        if 28 < cycle_30s < 30:
            base += 30
        
        return min(100, max(5, base))
    
    elif pattern == 'database_backup':
        # Continuous queries + periodic backup
        cycle_45s = t % 45
        base = 25 + 10 * math.sin(t * 2 * math.pi / 5)
        
        # Backup: heavy CPU for 10s every 45s
        if cycle_45s < 10:
            backup_load = 60 * (1 - cycle_45s / 10)
            base += backup_load
        
        return min(100, max(5, base))
    
    elif pattern == 'batch_processing':
        # Message queue: accumulate then process
        cycle_20s = t % 20
        
        if 15 < cycle_20s < 18:
            # Batch processing window
            base = 80 - 20 * (cycle_20s - 15)
        else:
            # Idle/accumulation
            base = 10 + 5 * (cycle_20s / 20)
        
        return min(100, max(5, base))
    
    elif pattern == 'monitoring':
        # Metrics scraping at different intervals
        cycle_5s = t % 5
        fast_load = 25 if cycle_5s < 0.5 else 5
        
        cycle_30s = t % 30
        slow_load = 50 if cycle_30s < 2 else 10
        
        base = fast_load + slow_load
        return min(100, max(5, base))
    
    elif pattern == 'video_encoding':
        # Video chunk encoding every 3s
        cycle_3s = t % 3
        
        if cycle_3s < 1:
            base = 95
        else:
            base = 15 + 10 * math.sin(t * 2 * math.pi / 3)
        
        return min(100, max(5, base))
    
    else:  # 'sine' or default
        base = 30 + 25 * math.sin(t * 2 * math.pi / 20)
        return min(100, max(5, base))


def simulate_cpu(pattern='web_server', duration=3600, num_cores=None):
    """
    Simulate CPU load patterns.
    
    Args:
        pattern: CPU pattern type
        duration: How long to run (seconds)
        num_cores: Number of CPU cores to use (default: all)
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    
    print(f"\n{'='*60}")
    print(f"CPU Load Simulator")
    print(f"{'='*60}")
    print(f"Pattern: {pattern}")
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"CPU Cores: {num_cores}")
    print(f"{'='*60}\n")
    
    # Create worker processes
    workers = []
    intensity_queues = []
    
    for i in range(num_cores):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker_process, args=(q, i))
        p.start()
        workers.append(p)
        intensity_queues.append(q)
    
    def cleanup(signum=None, frame=None):
        """Clean up worker processes"""
        print("\n\nStopping CPU simulation...")
        should_run.value = 0
        time.sleep(0.5)
        
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    start_time = time.time()
    step = 0
    
    try:
        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            
            # Get target CPU usage
            target_cpu = generate_cpu_pattern(elapsed, pattern)
            intensity = target_cpu / 100.0
            
            # Send intensity to all workers
            for q in intensity_queues:
                try:
                    q.put_nowait(intensity)
                except:
                    pass
            
            print(f"[{step:4d}] t={elapsed:6.1f}s | Target CPU: {target_cpu:5.1f}%")
            
            step += 1
            time.sleep(1)
        
        cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate CPU load patterns for forecasting tests"
    )
    
    parser.add_argument(
        '--pattern',
        choices=['web_server', 'api_service', 'database_backup', 'batch_processing',
                 'monitoring', 'video_encoding', 'sine'],
        default='web_server',
        help='CPU load pattern to simulate'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Simulation duration in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--cores',
        type=int,
        default=None,
        help='Number of CPU cores to use (default: all available)'
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         CPU Load Simulator (Production Patterns)        ║
╚══════════════════════════════════════════════════════════╝

=== PATTERNS ===
  web_server        - Baseline + request bursts every 15s
                      Range: 5-70% | Like: nginx, Apache
  
  api_service       - API polling (10s) + batch jobs (30s)
                      Range: 15-90% | Like: microservice API
  
  database_backup   - Continuous queries + backup every 45s
                      Range: 15-85% | Like: PostgreSQL, MySQL
  
  batch_processing  - Queue batch processing every 20s
                      Range: 10-80% | Like: message queue consumer
  
  monitoring        - Fast metrics (5s) + slow metrics (30s)
                      Range: 5-75% | Like: Prometheus scraping
  
  video_encoding    - Video chunk encoding every 3s
                      Range: 15-95% | Like: FFmpeg processing
  
  sine              - Smooth sine wave (baseline testing)
                      Range: 5-55% | Clean predictable pattern

Monitor with Netdata: system.cpu context, user dimension
Press Ctrl+C to stop
""")
    
    simulate_cpu(
        pattern=args.pattern,
        duration=args.duration,
        num_cores=args.cores
    )


if __name__ == '__main__':
    main()
