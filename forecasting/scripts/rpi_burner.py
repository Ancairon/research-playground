#!/usr/bin/env python3
"""
Standalone Random CPU Pattern Generator
Creates realistic CPU load patterns for metrics forecasting.
Copy-paste friendly for Raspberry Pi deployment.
"""
import multiprocessing as mp
import time
import math
import random
import sys

def burn_cpu():
    """
    Tight loop doing a mix of expensive ops:
      - math.sin/cos calls
      - random numbers
      - large integer multiplications
    """
    x = 1
    while True:
        # mix of FP & int work
        r = random.random()
        _ = math.sin(r) * math.cos(r)  # two trig calls
        x = x * 1234567  # big integer multiply
        x ^= x << 13     # some bit-twiddling
        x ^= x >> 7
        # keep x within some bound
        x = x & ((1 << 64) - 1)


def random_cpu_pattern():
    """
    Create random CPU load patterns within an hour cycle.
    - Random burst duration (30-300 seconds)
    - Random quiet period (60-600 seconds)
    - Random number of processes (2-4 for RPi)
    """
    print("Starting random CPU pattern generator for realistic metrics...")
    print("This will create varying CPU loads for forecasting data")
    
    start_time = time.time()
    cycle_duration = 3600  # 1 hour cycle
    
    while time.time() - start_time < cycle_duration:
        # Random burst parameters (adjusted for RPi)
        num_processes = random.randint(1, 4)  # Fewer processes for RPi
        burst_duration = random.randint(30, 300)  # 30 seconds to 5 minutes
        quiet_duration = random.randint(60, 600)   # 1 to 10 minutes
        
        print("\n--- Starting CPU burst ---")
        print(f"Processes: {num_processes}, Duration: {burst_duration}s")
        
        # Start CPU burners
        processes = []
        try:
            for i in range(num_processes):
                p = mp.Process(target=burn_cpu)
                p.start()
                processes.append(p)
                print(f"Started CPU burner #{i+1}")
                # Small random delay between process starts
                time.sleep(random.uniform(0.5, 2.0))
            
            # Run the burst
            time.sleep(burst_duration)
            
        finally:
            # Clean up processes
            for p in processes:
                p.terminate()
                p.join(timeout=1)
            print(f"Terminated {num_processes} CPU burners")
        
        # Quiet period
        print(f"--- Quiet period for {quiet_duration}s ---")
        time.sleep(quiet_duration)
        
        remaining_time = cycle_duration - (time.time() - start_time)
        if remaining_time > 0:
            print(f"Time remaining in cycle: {remaining_time:.0f}s")


def run_single_cycle():
    """Run one hour of random CPU patterns."""
    try:
        random_cpu_pattern()
        print("\nHour cycle completed!")
        return True
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return False


def run_continuous():
    """Run continuous random CPU patterns (multiple hour cycles)."""
    cycle_count = 0
    
    print("Starting continuous random CPU pattern generator...")
    print("Each cycle = 1 hour of random CPU bursts and quiet periods")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            cycle_count += 1
            print(f"\n=== Starting pattern cycle #{cycle_count} ===")
            
            if not run_single_cycle():
                break
                
            print(f"Completed cycle #{cycle_count}")
            print("Starting next cycle in 10 seconds...")
            time.sleep(10)  # Brief pause between cycles
            
    except KeyboardInterrupt:
        print(f"\nStopped after {cycle_count} cycles.")


def main():
    """Main function with command line options."""
    if len(sys.argv) > 1 and sys.argv[1] == "continuous":
        run_continuous()
    else:
        print("Running single 1-hour cycle...")
        print("Use 'python3 rpi_burner.py continuous' for multiple cycles")
        run_single_cycle()


if __name__ == "__main__":
    main()
