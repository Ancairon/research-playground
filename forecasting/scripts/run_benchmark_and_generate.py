"""
Run a local stress benchmark (CPU burner) for a specified duration, then call
`scripts/generatecsv.py` to fetch the last N seconds of Netdata into a CSV.

Usage:
  python3 scripts/run_benchmark_and_generate.py --duration 1800 --fetch-seconds 2400

This will run the burner for 30 minutes, then fetch the last 40 minutes (2400s)
from Netdata using the existing `generatecsv` logic.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import time
import random
from pathlib import Path

# we'll import the generatecsv module after adjusting sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))


def start_burners(workers: int = 4):
    procs = []
    for i in range(workers):
        p = mp.Process(target=_run_burn)
        p.start()
        procs.append(p)
    return procs


def _run_burn():
    # reuse burner.py's burn_cpu function by importing it
    try:
        from burner import burn_cpu
        burn_cpu()
    except Exception:
        # fallback: busy loop
        x = 1
        while True:
            x = x * 1234567


def stop_burners(procs):
    for p in procs:
        p.terminate()
        p.join(timeout=1)


def main(argv=None):
    p = argparse.ArgumentParser(description='Run benchmark then generate CSV')
    p.add_argument('--duration', type=int, default=None, help='Benchmark duration in seconds (if provided, runs continuously for this many seconds)')
    p.add_argument('--workers', type=int, default=4, help='Number of burner processes')
    p.add_argument('--fetch-seconds', type=int, default=2400, help='Seconds to fetch after benchmark (e.g., 2400 = 40 min)')
    p.add_argument('--active', type=int, default=None, help='Active burner seconds per cycle (pattern mode)')
    p.add_argument('--idle', type=int, default=None, help='Idle seconds between active bursts (pattern mode)')
    p.add_argument('--cycles', type=int, default=None, help='Number of cycles to run in pattern mode')
    p.add_argument('--jitter', type=float, default=0.1, help='Fractional jitter to apply to active/idle durations (0.1 = ±10%)')
    p.add_argument('--worker-jitter', type=float, default=0.0, help='Fractional jitter to apply to worker count per cycle (0.1 = ±10%)')
    p.add_argument('--netdata-ip', default='192.168.1.27', help='Netdata IP/host for generatecsv')
    p.add_argument('--context', default='system.cpu', help='Netdata context/context id')
    p.add_argument('--dimension', default='user', help='Netdata dimension')
    p.add_argument('--out', default=None, help='Output CSV path for generatecsv')
    args = p.parse_args(argv)

    print(f"Starting benchmark: workers={args.workers}")
    procs = []
    try:
        # Pattern mode if active/idle/cycles provided
        if args.active and args.idle and args.cycles:
            print(f"Pattern mode: active={args.active}s idle={args.idle}s cycles={args.cycles} jitter={args.jitter}")
            for c in range(args.cycles):
                # apply jitter to durations
                j = max(0.0, min(1.0, args.jitter))
                active_delta = random.uniform(-j, j)
                idle_delta = random.uniform(-j, j)
                active_sec = max(1, int(args.active * (1.0 + active_delta)))
                idle_sec = max(0, int(args.idle * (1.0 + idle_delta)))
                # optionally jitter worker count
                wj = max(0.0, min(1.0, args.worker_jitter)) if hasattr(args, 'worker_jitter') else 0.0
                worker_delta = random.uniform(-wj, wj) if wj > 0 else 0.0
                workers_this_cycle = max(1, int(args.workers * (1.0 + worker_delta)))

                print(f"Cycle {c+1}/{args.cycles}: START active period ({active_sec}s) workers={workers_this_cycle}")
                # start burners
                procs = []
                for _ in range(workers_this_cycle):
                    proc = mp.Process(target=_run_burn)
                    proc.start()
                    procs.append(proc)
                # run active period
                end_t = time.time() + active_sec
                while time.time() < end_t:
                    remaining = int(end_t - time.time())
                    print(f" active {remaining}s remaining", end='\r')
                    time.sleep(2)
                # stop burners for idle
                print('\nActive period complete; entering idle period')
                stop_burners(procs)
                procs = []
                idle_end = time.time() + idle_sec
                while time.time() < idle_end:
                    remaining = int(idle_end - time.time())
                    print(f" idle {remaining}s remaining", end='\r')
                    time.sleep(2)
                print('\nIdle period complete')
        elif args.duration:
            # continuous duration mode for backward compatibility
            print(f"Continuous mode: duration={args.duration}s")
            # start burners
            procs = []
            for i in range(args.workers):
                proc = mp.Process(target=_run_burn)
                proc.start()
                procs.append(proc)
                print(f"Started burner #{i+1}")
                time.sleep(0.5)

            # wait for duration
            end_t = time.time() + args.duration
            while time.time() < end_t:
                remaining = int(end_t - time.time())
                print(f"benchmark running, {remaining}s remaining", end='\r')
                time.sleep(5)
        else:
            print("No benchmark duration or pattern specified; exiting.")
            return 1
    finally:
        print('\nStopping burners (cleanup)...')
        stop_burners(procs)

    # now call generatecsv to fetch the last `fetch_seconds`
    print(f"Fetching last {args.fetch_seconds}s from Netdata via generatecsv...")
    try:
        from generatecsv import getDataFromAPI
        out = args.out
        getDataFromAPI(args.netdata_ip, args.context, args.dimension, args.fetch_seconds, out_file=out)
        print("CSV generation complete.")
    except Exception as e:
        print(f"Failed to call generatecsv: {e}")
        return 2

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
