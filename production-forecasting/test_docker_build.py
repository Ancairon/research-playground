#!/usr/bin/env python3
"""
Test script for production forecasting Docker build.

Tests:
1. Docker image builds successfully
2. Container starts and listens on port 5000
3. /forecast endpoint accepts requests and returns valid JSON
4. /forecast returns predictions and metrics
"""

import subprocess
import json
import time
import sys
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings for self-signed cert
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

TEST_IMAGE_NAME = "forecasting-service:test"
CONTAINER_NAME = "forecasting-test"
BASE_URL = "https://localhost:5000"


def run_command(cmd, description=""):
    """Run shell command and return output."""
    print(f">> {description or ' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAILED] {result.stderr}")
        return False
    return result


def test_build():
    """Test Docker image build."""
    print("\n" + "="*70)
    print("TEST 1: Docker Build")
    print("="*70)
    
    result = run_command(
        ["docker", "build", "-t", TEST_IMAGE_NAME, "."],
        "Building Docker image..."
    )
    if not result:
        return False
    
    print(f"[OK] Image built: {TEST_IMAGE_NAME}")
    return True


def test_container_start():
    """Test container startup."""
    print("\n" + "="*70)
    print("TEST 2: Container Startup")
    print("="*70)
    
    # Remove old container if exists
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], 
                   capture_output=True)
    
    # Start container
    print(f"➜ Starting container {CONTAINER_NAME}...")
    result = subprocess.run(
        ["docker", "run", "-d", "--name", CONTAINER_NAME, 
         "-p", "5000:5000", TEST_IMAGE_NAME],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[FAILED] Failed to start container: {result.stderr}")
        return False
    
    container_id = result.stdout.strip()
    print(f"[OK] Container started: {container_id[:12]}")
    
    # Wait for service to be ready
    print(">> Waiting for service to be ready...")
    max_attempts = 30
    for attempt in range(max_attempts):
        time.sleep(1)
        try:
            response = requests.get(f"{BASE_URL}/", verify=False, timeout=2)
            print(f"[OK] Service is ready (attempt {attempt+1})")
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"  Attempt {attempt+1}/{max_attempts}: {type(e).__name__}...", end="\r")
            else:
                print(f"[FAILED] Service failed to start after {max_attempts}s: {e}")
                # Print logs for debugging
                logs = subprocess.run(
                    ["docker", "logs", CONTAINER_NAME],
                    capture_output=True, text=True
                )
                print("\nContainer logs:")
                print(logs.stdout[-500:] if len(logs.stdout) > 500 else logs.stdout)
                return False
    
    return False


def test_forecast_simple():
    """Test /forecast endpoint with simple array data."""
    print("\n" + "="*70)
    print("TEST 3: Forecast Endpoint (Simple Array)")
    print("="*70)
    
    data = {
        "horizon": 5,
        "data": list(range(1, 51)),  # 50 points
        "evaluation": False
    }
    
    print(f">> POST {BASE_URL}/forecast")
    print(f"  Payload: horizon={data['horizon']}, data_points={len(data['data'])}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/forecast",
            json=data,
            verify=False,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"[FAILED] Got status {response.status_code}: {response.text}")
            return False
        
        result = response.json()
        
        # Validate response structure
        if "predictions" not in result:
            print(f"[FAILED] Missing 'predictions' in response: {result.keys()}")
            return False
        
        predictions = result["predictions"]
        if len(predictions) != data["horizon"]:
            print(f"[FAILED] Expected {data['horizon']} predictions, got {len(predictions)}")
            return False
        
        print(f"[OK] Response valid")
        print(f"  Predictions: {[f'{p:.6f}' for p in predictions]}")
        return True
        
    except Exception as e:
        print(f"[FAILED] Request failed: {e}")
        return False


def test_forecast_with_evaluation():
    """Test /forecast with evaluation=true to get metrics."""
    print("\n" + "="*70)
    print("TEST 4: Forecast Endpoint (With Evaluation)")
    print("="*70)
    
    data = {
        "horizon": 5,
        "data": list(range(1, 51)),  # 50 points
        "evaluation": True
    }
    
    print(f">> POST {BASE_URL}/forecast with evaluation=true")
    
    try:
        response = requests.post(
            f"{BASE_URL}/forecast",
            json=data,
            verify=False,
            timeout=120
        )
        
        if response.status_code != 200:
            print(f"[FAILED] Got status {response.status_code}: {response.text}")
            return False
        
        result = response.json()
        
        # Validate response structure
        required_keys = ["predictions", "actuals", "metrics"]
        for key in required_keys:
            if key not in result:
                print(f"[FAILED] Missing '{key}' in response")
                return False
        
        # Validate metrics
        metrics = result.get("metrics", {})
        required_metrics = ["mape", "rmse", "mbe"]
        for metric in required_metrics:
            if metric not in metrics:
                print(f"[FAILED] Missing metric '{metric}'")
                return False
        
        print(f"[OK] Response valid with metrics")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MBE:  {metrics['mbe']:.6f}")
        return True
        
    except Exception as e:
        print(f"[FAILED] Request failed: {e}")
        return False


def test_netdata_format():
    """Test /forecast with Netdata format data."""
    print("\n" + "="*70)
    print("TEST 5: Forecast Endpoint (Netdata Format)")
    print("="*70)
    
    # Netdata format: [[timestamp, [value, ...]], ...]
    data = {
        "horizon": 5,
        "data": [
            [1765387008, [0.5, 0, 0]],
            [1765120896, [0.6, 0, 0]],
            [1764854784, [0.55, 0, 0]],
            [1764588672, [0.65, 0, 0]],
            [1764322560, [0.7, 0, 0]],
            [1764056448, [0.75, 0, 0]],
            [1763790336, [0.8, 0, 0]],
            [1763524224, [0.85, 0, 0]],
            [1763258112, [0.9, 0, 0]],
            [1762992000, [0.95, 0, 0]],
            [1762725888, [1.0, 0, 0]],
            [1762459776, [1.05, 0, 0]],
            [1762193664, [1.1, 0, 0]],
            [1761927552, [1.15, 0, 0]],
            [1761661440, [1.2, 0, 0]],
            [1761395328, [1.25, 0, 0]],
            [1761129216, [1.3, 0, 0]],
            [1760863104, [1.35, 0, 0]],
            [1760596992, [1.4, 0, 0]],
            [1760330880, [1.45, 0, 0]],
            [1760064768, [1.5, 0, 0]],
            [1759798656, [1.55, 0, 0]],
            [1759532544, [1.6, 0, 0]],
            [1759266432, [1.65, 0, 0]],
        ],
        "evaluation": False
    }
    
    print(f">> POST {BASE_URL}/forecast with Netdata format")
    print(f"  Data points: {len(data['data'])}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/forecast",
            json=data,
            verify=False,
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"[FAILED] Got status {response.status_code}: {response.text}")
            return False
        
        result = response.json()
        
        if "predictions" not in result:
            print(f"[FAILED] Missing 'predictions' in response")
            return False
        
        predictions = result["predictions"]
        if len(predictions) != data["horizon"]:
            print(f"[FAILED] Expected {data['horizon']} predictions, got {len(predictions)}")
            return False
        
        print(f"[OK] Netdata format handled correctly")
        print(f"  Predictions: {[f'{p:.6f}' for p in predictions]}")
        return True
        
    except Exception as e:
        print(f"[FAILED] Request failed: {e}")
        return False


def cleanup():
    """Clean up Docker container."""
    print("\n" + "="*70)
    print("Cleanup")
    print("="*70)
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], 
                   capture_output=True)
    print(f"[OK] Container {CONTAINER_NAME} removed")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("FORECASTING SERVICE BUILD TEST")
    print("="*70)
    
    tests = [
        ("Build", test_build),
        ("Container Start", test_container_start),
        ("Forecast (Simple)", test_forecast_simple),
        ("Forecast (Evaluation)", test_forecast_with_evaluation),
        ("Forecast (Netdata)", test_netdata_format),
    ]
    
    results = []
    
    try:
        for name, test_func in tests:
            try:
                passed = test_func()
                results.append((name, passed))
            except Exception as e:
                print(f"\n[FAILED] EXCEPTION in {name}: {e}")
                results.append((name, False))
    finally:
        cleanup()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
