#!/usr/bin/env python3
"""
Simulate disk I/O patterns for forecasting testing.
Creates predictable disk read/write patterns that Netdata can measure.
"""

import argparse
import time
import math
import os
import tempfile
import signal
import sys

def generate_io_pattern(t, pattern='web_server'):
    """
    Generate disk I/O rate in MB/s based on time and pattern.
    
    Args:
        t: Time in seconds since start
        pattern: Pattern name
        
    Returns:
        I/O rate in MB/s
    """
    if pattern == 'web_server':
        # Log writes + periodic cache flush
        cycle_15s = t % 15
        base = 10 + 5 * math.sin(t * 2 * math.pi / 60)
        
        if cycle_15s < 2:  # Cache flush
            burst = 100 * math.exp(-cycle_15s / 1.0)
            base += burst
        
        return max(5, base)
    
    elif pattern == 'database':
        # Continuous writes + checkpoint every 30s
        cycle_30s = t % 30
        base = 20 + 10 * math.sin(t * 2 * math.pi / 10)
        
        if 25 < cycle_30s < 30:  # Checkpoint
            base += 150
        
        return max(5, base)
    
    elif pattern == 'backup':
        # Large sequential writes every 45s
        cycle_45s = t % 45
        
        if cycle_45s < 15:
            base = 200 - 10 * cycle_45s
        else:
            base = 10
        
        return max(5, base)
    
    elif pattern == 'log_rotation':
        # Regular small writes + rotation every 20s
        cycle_20s = t % 20
        
        if 18 < cycle_20s < 20:
            base = 150
        else:
            base = 15 + 5 * (cycle_20s / 20)
        
        return max(5, base)
    
    elif pattern == 'cache_sync':
        # Cache synchronization every 10s
        cycle_10s = t % 10
        
        if cycle_10s < 1:
            base = 200
        else:
            base = 10 + 5 * math.sin(t * 2 * math.pi / 10)
        
        return max(5, base)
    
    else:  # 'sine' or default
        base = 20 + 15 * math.sin(t * 2 * math.pi / 20)
        return max(5, base)


def perform_io(mb_per_sec, duration=1.0, temp_dir=None):
    """
    Perform disk I/O at specified rate.
    
    Args:
        mb_per_sec: Target MB/s
        duration: How long to write (seconds)
        temp_dir: Temporary directory for test files
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    total_bytes = int(mb_per_sec * 1024 * 1024 * duration)
    chunk_size = 1024 * 1024  # 1MB chunks
    
    # Create temporary file
    temp_file = os.path.join(temp_dir, f"disk_sim_{os.getpid()}.tmp")
    
    try:
        written = 0
        start = time.time()
        
        with open(temp_file, 'wb') as f:
            while written < total_bytes and (time.time() - start) < duration:
                chunk = os.urandom(min(chunk_size, total_bytes - written))
                f.write(chunk)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
                written += len(chunk)
        
        # Clean up
        os.remove(temp_file)
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


def simulate_disk_io(pattern='web_server', duration=3600, temp_dir=None):
    """
    Simulate disk I/O patterns.
    
    Args:
        pattern: I/O pattern type
        duration: How long to run (seconds)
        temp_dir: Directory for temporary files
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    print(f"\n{'='*60}")
    print(f"Disk I/O Simulator")
    print(f"{'='*60}")
    print(f"Pattern: {pattern}")
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"Temp Dir: {temp_dir}")
    print(f"{'='*60}\n")
    
    def cleanup(signum=None, frame=None):
        """Clean up"""
        print("\n\nStopping disk I/O simulation...")
        
        # Clean up any leftover temp files
        for f in os.listdir(temp_dir):
            if f.startswith('disk_sim_'):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except:
                    pass
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    start_time = time.time()
    step = 0
    
    try:
        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            
            # Get target I/O rate
            target_mbps = generate_io_pattern(elapsed, pattern)
            
            # Perform I/O
            io_start = time.time()
            perform_io(target_mbps, duration=1.0, temp_dir=temp_dir)
            io_duration = time.time() - io_start
            
            # Sleep remainder of interval
            sleep_time = 1.0 - io_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            print(f"[{step:4d}] t={elapsed:6.1f}s | Target I/O: {target_mbps:6.1f} MB/s | Actual duration: {io_duration:.2f}s")
            
            step += 1
        
        cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate disk I/O patterns for forecasting tests"
    )
    
    parser.add_argument(
        '--pattern',
        choices=['web_server', 'database', 'backup', 'log_rotation', 'cache_sync', 'sine'],
        default='web_server',
        help='Disk I/O pattern to simulate'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Simulation duration in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=None,
        help='Directory for temporary files (default: system temp)'
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       Disk I/O Simulator (Production Patterns)          ║
╚══════════════════════════════════════════════════════════╝

=== PATTERNS ===
  web_server      - Log writes + cache flush every 15s
                    Range: 5-110 MB/s
  
  database        - Continuous writes + checkpoint every 30s
                    Range: 10-170 MB/s
  
  backup          - Large sequential writes every 45s
                    Range: 10-200 MB/s
  
  log_rotation    - Regular writes + rotation every 20s
                    Range: 15-150 MB/s
  
  cache_sync      - Cache sync every 10s
                    Range: 10-200 MB/s
  
  sine            - Smooth sine wave (baseline testing)
                    Range: 5-45 MB/s

Monitor with Netdata: disk_ops.* or disk_backlog.* context
⚠️  Warning: This will generate actual disk writes!
Press Ctrl+C to stop
""")
    
    simulate_disk_io(
        pattern=args.pattern,
        duration=args.duration,
        temp_dir=args.temp_dir
    )


if __name__ == '__main__':
    main()
