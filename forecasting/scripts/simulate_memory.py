#!/usr/bin/env python3
"""
Simulate memory usage patterns for forecasting testing.
Creates predictable memory allocation patterns that Netdata can measure.
"""

import argparse
import time
import math
import signal
import sys

# Global memory holders
memory_blocks = []


def generate_memory_pattern(t, pattern='web_server'):
    """
    Generate memory usage in MB based on time and pattern.
    
    Args:
        t: Time in seconds since start
        pattern: Pattern name
        
    Returns:
        Target memory usage in MB
    """
    if pattern == 'web_server':
        # Request processing: allocate memory every 15s
        cycle_15s = t % 15
        base = 100 + 50 * math.sin(t * 2 * math.pi / 60)
        
        if cycle_15s < 3:  # Request burst
            burst = 300 * math.exp(-cycle_15s / 1.5)
            base += burst
        
        return max(50, base)
    
    elif pattern == 'cache_service':
        # Cache filling up every 30s
        cycle_30s = t % 30
        base = 100 + 20 * math.sin(t * 2 * math.pi / 10)
        
        if cycle_30s < 5:  # Cache fill
            base += 200 * (5 - cycle_30s) / 5
        
        return max(50, base)
    
    elif pattern == 'batch_job':
        # Batch processing: memory grows then released
        cycle_20s = t % 20
        
        if cycle_20s < 15:
            # Memory accumulates during processing
            base = 100 + 400 * (cycle_20s / 15)
        else:
            # Memory released after batch
            base = 100
        
        return max(50, base)
    
    elif pattern == 'memory_leak':
        # Gradual memory increase until reset
        cycle_60s = t % 60
        base = 100 + 300 * (cycle_60s / 60)
        
        return max(50, base)
    
    elif pattern == 'gc_cycles':
        # Garbage collection: sawtooth pattern
        cycle_10s = t % 10
        
        if cycle_10s < 8:
            # Memory grows
            base = 100 + 200 * (cycle_10s / 8)
        else:
            # GC happens, memory drops
            base = 100
        
        return max(50, base)
    
    else:  # 'sine' or default
        base = 200 + 150 * math.sin(t * 2 * math.pi / 20)
        return max(50, base)


def allocate_memory(target_mb):
    """
    Allocate or free memory to reach target usage.
    
    Args:
        target_mb: Target memory in MB
    """
    global memory_blocks
    
    # Calculate current memory usage
    current_mb = sum(len(block) for block in memory_blocks) / (1024 * 1024)
    
    if current_mb < target_mb:
        # Allocate more memory
        needed = int((target_mb - current_mb) * 1024 * 1024)
        if needed > 0:
            # Allocate in blocks and fill with data to ensure it's actually allocated
            block = bytearray(needed)
            for i in range(0, len(block), 1024):
                block[i] = i % 256  # Touch memory to force allocation
            memory_blocks.append(block)
    
    elif current_mb > target_mb:
        # Free memory
        to_free = current_mb - target_mb
        freed = 0
        
        while memory_blocks and freed < to_free:
            block = memory_blocks.pop()
            freed += len(block) / (1024 * 1024)
            del block


def simulate_memory(pattern='web_server', duration=3600):
    """
    Simulate memory usage patterns.
    
    Args:
        pattern: Memory pattern type
        duration: How long to run (seconds)
    """
    global memory_blocks
    
    print(f"\n{'='*60}")
    print(f"Memory Usage Simulator")
    print(f"{'='*60}")
    print(f"Pattern: {pattern}")
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"{'='*60}\n")
    
    def cleanup(signum=None, frame=None):
        """Clean up memory"""
        print("\n\nStopping memory simulation...")
        print("Freeing allocated memory...")
        memory_blocks.clear()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    start_time = time.time()
    step = 0
    
    try:
        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            
            # Get target memory usage
            target_mb = generate_memory_pattern(elapsed, pattern)
            
            # Allocate/free memory
            allocate_memory(target_mb)
            
            # Calculate actual usage
            actual_mb = sum(len(block) for block in memory_blocks) / (1024 * 1024)
            
            print(f"[{step:4d}] t={elapsed:6.1f}s | Target: {target_mb:6.1f} MB | Actual: {actual_mb:6.1f} MB | Blocks: {len(memory_blocks)}")
            
            step += 1
            time.sleep(1)
        
        cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate memory usage patterns for forecasting tests"
    )
    
    parser.add_argument(
        '--pattern',
        choices=['web_server', 'cache_service', 'batch_job', 'memory_leak', 'gc_cycles', 'sine'],
        default='batch_job',
        help='Memory usage pattern to simulate'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Simulation duration in seconds (default: 3600 = 1 hour)'
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       Memory Usage Simulator (Production Patterns)      ║
╚══════════════════════════════════════════════════════════╝

=== PATTERNS ===
  web_server      - Request bursts every 15s
                    Range: 100-450 MB | Spikes during traffic
  
  cache_service   - Cache filling every 30s
                    Range: 100-320 MB | Periodic cache loads
  
  batch_job       - Memory grows during processing, then drops
                    Range: 100-500 MB | 20s cycle (15s up, 5s down)
  
  memory_leak     - Gradual increase until reset
                    Range: 100-400 MB | 60s cycle (simulates leak)
  
  gc_cycles       - Garbage collection sawtooth
                    Range: 100-300 MB | 10s cycle (8s up, 2s GC)
  
  sine            - Smooth sine wave (baseline testing)
                    Range: 50-350 MB | 20s period

Monitor with Netdata:
  Context: system.ram or mem.available
  Dimension: used or available (inverse)

⚠️  This script will allocate real memory!
   Make sure you have enough RAM available.
Press Ctrl+C to stop
""")
    
    simulate_memory(
        pattern=args.pattern,
        duration=args.duration
    )


if __name__ == '__main__':
    main()
