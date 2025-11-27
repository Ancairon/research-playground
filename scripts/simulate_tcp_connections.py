#!/usr/bin/env python3
"""
Simulate TCP connection patterns for forecasting testing.
Creates predictable connection patterns that Netdata can measure.
"""

import argparse
import time
import math
import socket
import signal
import sys
import random

# Global connection pool
active_connections = []


def generate_connection_pattern(t, pattern='web_server', noise_level=0.15):
    """
    Generate target number of concurrent TCP connections based on time and pattern.
    
    Args:
        t: Time in seconds since start
        pattern: Pattern name
        noise_level: Random noise as fraction of base value (default: 0.15 = 15%)
        
    Returns:
        Target number of connections with noise added
    """
    if pattern == 'web_server':
        # Web server: baseline + periodic traffic bursts every 15s
        cycle_15s = t % 15
        base = 20 + 10 * math.sin(t * 2 * math.pi / 60)
        
        if cycle_15s < 3:  # Request burst
            burst = 80 * math.exp(-cycle_15s / 1.5)
            base += burst
        
        # Add noise
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
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
            base += 40
        
        # Add noise
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    elif pattern == 'load_balancer':
        # Load balancer: continuous connections + health checks
        cycle_5s = t % 5
        base = 30 + 10 * math.sin(t * 2 * math.pi / 20)
        
        # Health check burst every 5s
        if cycle_5s < 0.5:
            base += 20
        
        # Add noise
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    elif pattern == 'database_pool':
        # Connection pool: grows during peak, shrinks during idle
        cycle_45s = t % 45
        
        if cycle_45s < 20:
            # Peak usage
            base = 50 + 30 * (cycle_45s / 20)
        elif cycle_45s < 30:
            # Hold connections
            base = 80
        else:
            # Release connections
            base = 50 - 30 * ((cycle_45s - 30) / 15)
        
        # Add noise
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    elif pattern == 'microservices':
        # Service mesh: multiple services communicating
        # Service A: every 8s
        cycle_8s = t % 8
        conns_a = 30 if cycle_8s < 1 else 10
        
        # Service B: every 12s
        cycle_12s = t % 12
        conns_b = 25 if cycle_12s < 1.5 else 8
        
        # Service C: every 6s (health checks)
        cycle_6s = t % 6
        conns_c = 15 if cycle_6s < 0.5 else 5
        
        base = conns_a + conns_b + conns_c
        # Add noise
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    elif pattern == 'batch_processing':
        # Batch job: connections accumulate then process
        cycle_20s = t % 20
        
        if 15 < cycle_20s < 18:
            # Processing window
            base = 100 - 20 * (cycle_20s - 15)
        else:
            # Accumulation
            base = 10 + 5 * (cycle_20s / 20)
        
        # Add noise
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))
    
    else:  # 'sine' or default
        base = 30 + 25 * math.sin(t * 2 * math.pi / 20)
        # Add noise
        noise = base * noise_level * (random.random() * 2 - 1)
        return int(max(5, base + noise))


def create_connection(host='localhost', port=80):
    """
    Create a TCP connection.
    
    Args:
        host: Target host
        port: Target port
        
    Returns:
        Socket object or None if failed
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        sock.connect((host, port))
        return sock
    except:
        return None


def manage_connections(target_count, host='localhost', port=80):
    """
    Adjust active connections to reach target count.
    
    Args:
        target_count: Desired number of connections
        host: Target host
        port: Target port
    """
    global active_connections
    
    # Remove dead connections
    active_connections = [s for s in active_connections if s.fileno() != -1]
    
    current_count = len(active_connections)
    
    if current_count < target_count:
        # Open more connections
        needed = target_count - current_count
        for _ in range(needed):
            sock = create_connection(host, port)
            if sock:
                active_connections.append(sock)
    
    elif current_count > target_count:
        # Close excess connections
        to_close = current_count - target_count
        for _ in range(to_close):
            if active_connections:
                sock = active_connections.pop()
                try:
                    sock.close()
                except:
                    pass


def simulate_connections(pattern='web_server', duration=3600, host='localhost', port=80, noise_level=0.15):
    """
    Simulate TCP connection patterns.
    
    Args:
        pattern: Connection pattern type
        duration: How long to run (seconds)
        host: Target host for connections
        port: Target port for connections
        noise_level: Random noise level (0.0-1.0, default 0.15 = 15%)
    """
    global active_connections
    
    print(f"\n{'='*60}")
    print(f"TCP Connection Simulator")
    print(f"{'='*60}")
    print(f"Pattern: {pattern}")
    print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"Target: {host}:{port}")
    print(f"Noise Level: {noise_level*100:.1f}%")
    print(f"{'='*60}\n")
    
    def cleanup(signum=None, frame=None):
        """Clean up connections"""
        print("\n\nStopping TCP simulation...")
        print(f"Closing {len(active_connections)} connections...")
        
        for sock in active_connections:
            try:
                sock.close()
            except:
                pass
        
        active_connections.clear()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    start_time = time.time()
    step = 0
    
    try:
        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time
            
            # Get target connection count with noise
            target_count = generate_connection_pattern(elapsed, pattern, noise_level)
            
            # Manage connections
            manage_connections(target_count, host, port)
            
            # Count actual connections
            actual_count = len(active_connections)
            
            print(f"[{step:4d}] t={elapsed:6.1f}s | Target: {target_count:4d} | Actual: {actual_count:4d} connections")
            
            step += 1
            time.sleep(1)
        
        cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate TCP connection patterns for forecasting tests"
    )
    
    parser.add_argument(
        '--pattern',
        choices=['web_server', 'api_service', 'load_balancer', 'database_pool',
                 'microservices', 'batch_processing', 'sine'],
        default='web_server',
        help='Connection pattern to simulate'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Simulation duration in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Target host for connections (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=80,
        help='Target port for connections (default: 80)'
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
║     TCP Connection Simulator (Production Patterns)      ║
╚══════════════════════════════════════════════════════════╝

=== PATTERNS ===
  web_server        - Traffic bursts every 15s
                      Range: 20-100 connections + {args.noise*100:.0f}% noise
  
  api_service       - Polling (10s) + batch jobs (30s)
                      Range: 15-100 connections + {args.noise*100:.0f}% noise
  
  load_balancer     - Continuous + health checks every 5s
                      Range: 30-60 connections + {args.noise*100:.0f}% noise
  
  database_pool     - Connection pool growth/shrink (45s cycle)
                      Range: 20-80 connections + {args.noise*100:.0f}% noise
  
  microservices     - Service mesh (6s, 8s, 12s intervals)
                      Range: 23-70 connections + {args.noise*100:.0f}% noise
  
  batch_processing  - Batch processing every 20s
                      Range: 10-100 connections + {args.noise*100:.0f}% noise
  
  sine              - Smooth sine wave (baseline testing)
                      Range: 5-55 connections + {args.noise*100:.0f}% noise

=== NOISE ===
  Random variation: {args.noise*100:.0f}% of base value
  Use --noise 0.0 for perfect patterns
  Use --noise 0.3 for challenging forecasting (30% variation)

Monitor with Netdata - Check these metrics:
  - ipv4.tcpsock (TCP sockets)
  - ipv4.tcpopens (TCP active opens)
  - netdata.tcp_connects (if monitoring netdata)
  - System TCP metrics

⚠️  Make sure target port is accessible!
   Default: localhost:80 (adjust with --host and --port)
Press Ctrl+C to stop
""")
    
    # Test connection first
    print("Testing connection...")
    test_sock = create_connection(args.host, args.port)
    if test_sock:
        test_sock.close()
        print(f"✓ Successfully connected to {args.host}:{args.port}\n")
    else:
        print(f"✗ Cannot connect to {args.host}:{args.port}")
        print(f"  Try different --host or --port, or start a local server:")
        print(f"  python3 -m http.server {args.port}\n")
        sys.exit(1)
    
    simulate_connections(
        pattern=args.pattern,
        duration=args.duration,
        host=args.host,
        port=args.port,
        noise_level=args.noise
    )


if __name__ == '__main__':
    main()
