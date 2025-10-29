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
    
    These patterns simulate REAL production traffic scenarios with 
    predictable structure that forecasting models should learn.
    """
    if pattern == 'web_server':
        # Realistic web server: baseline + periodic request bursts
        # 60s cycle with regular traffic spikes every 15s (page loads)
        cycle_15s = t % 15
        
        # Baseline traffic (background requests)
        base = 80 + 20 * math.sin(t * 2 * math.pi / 60)
        
        # Regular page load bursts every 15 seconds
        if cycle_15s < 3:  # 3-second burst for page load + assets
            burst = 150 * math.exp(-cycle_15s / 1.5)  # Exponential decay
            base += burst
        
        noise = random.gauss(0, 8)
        return max(10, base + noise)
    
    elif pattern == 'api_service':
        # RESTful API: regular polling + scheduled jobs
        # Clients poll every 10s, batch jobs every 30s
        
        # Baseline API polling (every 10s)
        cycle_10s = t % 10
        if cycle_10s < 1.5:
            base = 180 + 40 * random.random()  # API response burst
        else:
            base = 60 + 10 * math.sin(t * 2 * math.pi / 10)
        
        # Scheduled batch job every 30s (heavier load)
        cycle_30s = t % 30
        if 28 < cycle_30s < 30:
            base += 200  # Batch processing spike
        
        noise = random.gauss(0, 12)
        return max(10, base + noise)
    
    elif pattern == 'database_backup':
        # Database: continuous queries + periodic backups
        # Light queries + heavy backup every 45s
        cycle_45s = t % 45
        
        # Continuous query traffic
        base = 100 + 30 * math.sin(t * 2 * math.pi / 5)
        
        # Backup window: 10s of heavy I/O every 45s
        if cycle_45s < 10:
            backup_load = 250 * (1 - cycle_45s / 10)  # Decreasing load
            base += backup_load
        
        noise = random.gauss(0, 10)
        return max(10, base + noise)
    
    elif pattern == 'cdn_cache':
        # CDN cache: cache hits (low) + cache misses (high)
        # 20s cycle: mostly cached, periodic cache refreshes
        cycle_20s = t % 20
        
        if cycle_20s < 2:
            # Cache miss/refresh burst
            base = 300 + 50 * random.random()
        else:
            # Cached content (low bandwidth)
            base = 40 + 20 * math.sin(t * 2 * math.pi / 20)
        
        noise = random.gauss(0, 2)
        return max(10, base + noise)
    
    elif pattern == 'microservices':
        # Microservices mesh: service-to-service calls
        # Multiple services communicating on different schedules
        
        # Service A: every 8s
        cycle_8s = t % 8
        if cycle_8s < 1:
            traffic_a = 120
        else:
            traffic_a = 30
        
        # Service B: every 12s
        cycle_12s = t % 12
        if cycle_12s < 1.5:
            traffic_b = 100
        else:
            traffic_b = 25
        
        # Service C: every 6s (health checks)
        cycle_6s = t % 6
        if cycle_6s < 0.5:
            traffic_c = 60
        else:
            traffic_c = 15
        
        base = traffic_a + traffic_b + traffic_c
        noise = random.gauss(0, 10)
        return max(10, base + noise)
    
    elif pattern == 'streaming_video':
        # Video streaming: chunk downloads every 2-4s
        # Adaptive bitrate with buffer management
        cycle_3s = t % 3
        
        if cycle_3s < 0.8:
            # Video chunk download
            base = 400 + 100 * random.random()
        else:
            # Buffer processing/idle
            base = 50 + 20 * math.sin(t * 2 * math.pi / 3)
        
        noise = random.gauss(0, 15)
        return max(10, base + noise)
    
    elif pattern == 'iot_sensors':
        # IoT sensors: synchronized reporting intervals
        # All sensors report every 15s, staggered by 5s
        
        total = 0
        # Sensor batch 1: every 15s at t=0
        if (t % 15) < 1:
            total += 120
        
        # Sensor batch 2: every 15s at t=5
        if ((t - 5) % 15) < 1:
            total += 100
        
        # Sensor batch 3: every 15s at t=10
        if ((t - 10) % 15) < 1:
            total += 110
        
        # Baseline telemetry
        total += 40 + 10 * math.sin(t * 2 * math.pi / 15)
        
        noise = random.gauss(0, 8)
        return max(10, total + noise)
    
    elif pattern == 'message_queue':
        # Message queue: batched processing
        # Messages accumulate, then batch processed
        cycle_20s = t % 20
        
        if 15 < cycle_20s < 18:
            # Batch processing window (3s)
            base = 350 - 50 * (cycle_20s - 15)  # Decreasing as queue drains
        else:
            # Message accumulation (low traffic)
            base = 60 + 15 * (cycle_20s / 20) * 100
        
        noise = random.gauss(0, 12)
        return max(10, base + noise)
    
    elif pattern == 'monitoring_system':
        # Monitoring/metrics: regular scrapes
        # Multiple exporters scraped at different intervals
        
        # Fast metrics (every 5s)
        cycle_5s = t % 5
        if cycle_5s < 0.5:
            fast_metrics = 80
        else:
            fast_metrics = 20
        
        # Slow metrics (every 30s)
        cycle_30s = t % 30
        if cycle_30s < 2:
            slow_metrics = 200
        else:
            slow_metrics = 30
        
        base = fast_metrics + slow_metrics
        noise = random.gauss(0, 8)
        return max(10, base + noise)
    
    elif pattern == 'business_hours':
        # Legacy pattern kept for compatibility
        # Step pattern: 20s high, 10s low (repeats every 30s)
        second = t % 30
        if second < 20:
            base = 150 + 20 * math.sin(second * math.pi / 10)
        else:
            base = 40 + 10 * random.random()
        
        noise = random.gauss(0, 10)
        return max(10, base + noise)
    
    else:  # 'sine' or default
        # Baseline smooth pattern for testing
        base = 100 + 60 * math.sin(t * 2 * math.pi / 20)
        noise = random.gauss(0, 5)
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
        choices=['web_server', 'api_service', 'database_backup', 'cdn_cache', 
                 'microservices', 'streaming_video', 'iot_sensors', 'message_queue',
                 'monitoring_system', 'business_hours', 'sine'],
        default='web_server',
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
║   Network Traffic Simulator (Production Patterns)       ║
╚══════════════════════════════════════════════════════════╝

=== APPLICATION PATTERNS ===
  web_server      - HTTP server with regular page loads
                    60s cycle, bursts every 15s (3s duration)
                    Range: 60-250 KB/s | Like: nginx, Apache
  
  api_service     - RESTful API with polling + scheduled jobs
                    10s polling + 30s batch jobs
                    Range: 60-380 KB/s | Like: microservice API
  
  database_backup - DB queries + periodic backups
                    Continuous queries + 10s backup every 45s
                    Range: 70-350 KB/s | Like: PostgreSQL, MySQL

=== INFRASTRUCTURE PATTERNS ===
  cdn_cache       - CDN with cache hits/misses
                    20s cycle: 2s cache refresh bursts
                    Range: 40-350 KB/s | Like: Cloudflare, Akamai
  
  microservices   - Service mesh communication
                    Multiple services (6s, 8s, 12s intervals)
                    Range: 70-280 KB/s | Like: Kubernetes services
  
  message_queue   - Queue with batched processing
                    Messages accumulate, batch process every 20s
                    Range: 60-350 KB/s | Like: RabbitMQ, Kafka

=== STREAMING & IOT PATTERNS ===
  streaming_video - Video chunk downloads
                    3s cycle: 0.8s chunk downloads
                    Range: 50-500 KB/s | Like: HLS, DASH streaming
  
  iot_sensors     - Synchronized sensor reporting
                    3 batches every 15s (staggered by 5s)
                    Range: 40-370 KB/s | Like: industrial IoT
  
  monitoring_system - Metrics collection
                     5s fast metrics + 30s slow metrics
                     Range: 50-300 KB/s | Like: Prometheus scraping

=== BASELINE PATTERNS ===
  business_hours  - Step pattern (legacy compatibility)
                    20s high, 10s low (30s cycle)
                    Range: 40-170 KB/s
  
  sine            - Smooth sine wave (baseline testing)
                    20s period, low noise
                    Range: 40-160 KB/s

All patterns simulate REAL production workloads with predictable structure.
Models should learn these patterns for accurate forecasting.
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
