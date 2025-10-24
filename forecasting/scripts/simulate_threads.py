#!/usr/bin/env python3
"""
Simulate thread usage patterns for forecasting testing.
Creates predictable thread patterns that Netdata can measure.
Process-isolated and safe - won't affect system.
"""

import argparse
import time
import math
import signal
import sys
import threading
import random

# Global thread management
active_threads = []
should_exit = False
thread_should_exit = {}  # Per-thread exit signals


def worker_thread(thread_id):
    """
    Worker thread that does light work to stay alive.
    Exits when signaled via thread_should_exit[thread_id].
    
    Args:
        thread_id: Thread identifier
    """
    global should_exit, thread_should_exit
    
    while not should_exit and not thread_should_exit.get(thread_id, False):
        # Light CPU work to keep thread alive
        _ = sum(i * i for i in range(100))
        time.sleep(0.1)
    
    # Cleanup
    thread_should_exit.pop(thread_id, None)


def generate_thread_pattern(t, pattern='web_server', noise_level=0.15):
    """
    Generate target number of threads based on time and pattern.
    
    Args:
        t: Time in seconds since start
        pattern: Pattern name
        noise_level: Random noise as fraction of base value
        
    Returns:
        Target number of threads with noise
    """
    if pattern == 'web_server':
        # Web server: worker pool + request bursts every 15s
        cycle_15s = t % 15
        base = 10000 + 5 * math.sin(t * 2 * math.pi / 60)
        
        if cycle_15s < 3:  # Request burst - spawn workers
            burst = 40 * math.exp(-cycle_15s / 1.5)
            base += burst
        
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    elif pattern == 'api_service':
        # API with thread pool + batch jobs
        cycle_10s = t % 10
        if cycle_10s < 1.5:
            base = 30  # Handling requests
        else:
            base = 8 + 3 * math.sin(t * 2 * math.pi / 10)
        
        # Batch job every 30s
        cycle_30s = t % 30
        if 28 < cycle_30s < 30:
            base += 20
        
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    elif pattern == 'worker_pool':
        # Worker pool: grows during peak, shrinks during idle
        cycle_45s = t % 45
        
        if cycle_45s < 20:
            # Peak - scaling up workers
            base = 15 + 25 * (cycle_45s / 20)
        elif cycle_45s < 30:
            # Hold at peak
            base = 40
        else:
            # Scale down workers
            base = 15 + 25 * (1 - (cycle_45s - 30) / 15)
        
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    elif pattern == 'microservices':
        # Multiple service components with different thread patterns
        # Service A: every 8s
        cycle_8s = t % 8
        threads_a = 15 if cycle_8s < 1 else 5
        
        # Service B: every 12s
        cycle_12s = t % 12
        threads_b = 12 if cycle_12s < 1.5 else 4
        
        # Service C: every 6s (background tasks)
        cycle_6s = t % 6
        threads_c = 8 if cycle_6s < 0.5 else 3
        
        base = threads_a + threads_b + threads_c
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    elif pattern == 'async_tasks':
        # Async task queue: bursts of parallel tasks
        cycle_20s = t % 20
        
        if cycle_20s < 5:
            # Task burst
            base = 50 - 8 * cycle_20s
        else:
            # Idle with few background threads
            base = 8 + 2 * math.sin(t * 2 * math.pi / 20)
        
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    elif pattern == 'batch_processing':
        # Batch job: threads accumulate then process
        cycle_25s = t % 25
        
        if cycle_25s < 18:
            # Processing with many threads
            base = 10 + 30 * (cycle_25s / 18)
        else:
            # Cleanup phase
            base = 10
        
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    else:  # 'sine' or default
        base = 20 + 15 * math.sin(t * 2 * math.pi / 20)
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))


def manage_threads(target_count):
    """
    Adjust active threads to reach target count.
    
    Args:
        target_count: Desired number of threads
    """
    global active_threads, thread_should_exit
    
    # Remove dead threads
    active_threads = [(tid, t) for tid, t in active_threads if t.is_alive()]
    
    current_count = len(active_threads)
    
    if current_count < target_count:
        # Create more threads
        needed = target_count - current_count
        for i in range(needed):
            thread_id = f"worker_{time.time()}_{i}"
            thread_should_exit[thread_id] = False
            thread = threading.Thread(
                target=worker_thread,
                args=(thread_id,),
                daemon=True
            )
            thread.start()
            active_threads.append((thread_id, thread))
    
    elif current_count > target_count:
        # Signal excess threads to exit
        to_stop = current_count - target_count
        for i in range(to_stop):
            if active_threads:
                thread_id, thread = active_threads.pop()
                thread_should_exit[thread_id] = True


def simulate_threads(pattern='web_server', duration=3600, noise_level=0.15):
    """
    Simulate thread usage patterns.
    
    Args:
        pattern: Thread pattern type
        duration: How long to run (seconds)
        noise_level: Random noise level (0.0-1.0)
    """
    global active_threads, should_exit, thread_should_exit
    
    print(f"\n{'='*60}")
    print(f"Thread Usage Simulator")
    print(f"{'='*60}")
    print(f"Pattern: {pattern}")
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"Noise Level: {noise_level*100:.1f}%")
    print(f"{'='*60}\n")
    
    def cleanup(signum=None, frame=None):
        """Clean up threads"""
        global should_exit
        print("\n\nStopping thread simulation...")
        print(f"Signaling {len(active_threads)} threads to exit...")
        
        should_exit = True
        
        # Signal all threads to exit
        for thread_id, thread in active_threads:
            thread_should_exit[thread_id] = True
        
        time.sleep(0.5)  # Give threads time to exit gracefully
        
        print("Threads stopped.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    start_time = time.time()
    step = 0
    
    try:
        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            
            # Get target thread count with noise
            target_count = generate_thread_pattern(elapsed, pattern, noise_level)
            
            # Manage threads
            manage_threads(target_count)
            
            # Count actual threads (including main thread)
            actual_count = threading.active_count()
            
            print(f"[{step:4d}] t={elapsed:6.1f}s | Target: {target_count:4d} | Actual: {actual_count:4d} threads")
            
            step += 1
            time.sleep(1)
        
        cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate thread usage patterns for forecasting tests"
    )
    
    parser.add_argument(
        '--pattern',
        choices=['web_server', 'api_service', 'worker_pool', 'microservices',
                 'async_tasks', 'batch_processing', 'sine'],
        default='web_server',
        help='Thread pattern to simulate'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Simulation duration in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--noise',
        type=float,
        default=0.15,
        help='Noise level as fraction (0.0-1.0, default: 0.15 = 15%% random variation)'
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║      Thread Usage Simulator (Production Patterns)       ║
╚══════════════════════════════════════════════════════════╝

=== PATTERNS ===
  web_server        - Worker pool + request bursts every 15s
                      Range: 10-50 threads + {args.noise*100:.0f}% noise
  
  api_service       - Thread pool + batch jobs (10s, 30s cycles)
                      Range: 8-50 threads + {args.noise*100:.0f}% noise
  
  worker_pool       - Dynamic pool scaling (45s cycle)
                      Range: 15-40 threads + {args.noise*100:.0f}% noise
  
  microservices     - Multiple services (6s, 8s, 12s intervals)
                      Range: 12-35 threads + {args.noise*100:.0f}% noise
  
  async_tasks       - Async task queue bursts (20s cycle)
                      Range: 8-50 threads + {args.noise*100:.0f}% noise
  
  batch_processing  - Batch jobs with thread scaling (25s cycle)
                      Range: 10-40 threads + {args.noise*100:.0f}% noise
  
  sine              - Smooth sine wave (baseline testing)
                      Range: 5-35 threads + {args.noise*100:.0f}% noise

=== NOISE ===
  Random variation: {args.noise*100:.0f}% of base value
  Use --noise 0.0 for perfect patterns
  Use --noise 0.3 for challenging forecasting (30% variation)

Monitor with Netdata - Check these metrics:
  - apps.threads (application threads by process)
  - system.processes (total threads/processes)
  - Look for Python process threads

✓ Safe - process-isolated, won't affect system
✓ Threads do light work to stay alive
✓ Graceful cleanup on exit
Press Ctrl+C to stop
""")
    
    simulate_threads(
        pattern=args.pattern,
        duration=args.duration,
        noise_level=args.noise
    )


if __name__ == '__main__':
    main()
