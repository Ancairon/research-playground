#!/usr/bin/env python3
"""
Mock ML Prediction Server for Netdata Charts

Simple Flask server that returns mock predictions.
Replace this with your actual ML model.

Usage:
    python3 mock-ml-server.py

Then configure inject-predictions.js:
    mlEndpoint: 'http://localhost:1234/predict'
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import math

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive prediction request and return mock predictions.
    
    Expected request body:
    {
        "chart": "system.cpu",
        "dimension": "user",
        "lastValue": 45.2,
        "horizonSeconds": 300,
        "points": 60
    }
    
    Returns:
    {
        "predictions": [45.3, 45.5, 45.7, ...]
    }
    """
    data = request.json
    
    chart = data.get('chart', '')
    dimension = data.get('dimension', '')
    last_value = data.get('lastValue', 50.0)
    horizon_seconds = data.get('horizonSeconds', 300)
    points = data.get('points', 60)
    
    print(f"[Predict] chart={chart}, dimension={dimension}, "
          f"lastValue={last_value}, points={points}")
    
    # Generate mock predictions
    predictions = generate_mock_predictions(
        last_value, 
        points, 
        chart, 
        dimension
    )
    
    return jsonify({
        "predictions": predictions,
        "chart": chart,
        "dimension": dimension,
        "model": "mock-v1"
    })

def generate_mock_predictions(last_value, count, chart, dimension):
    """
    Generate mock predictions based on chart/dimension.
    Replace this with your actual ML model.
    """
    predictions = []
    
    # Different mock patterns for different dimensions
    if 'cpu' in chart.lower():
        # CPU: slight upward trend with noise
        for i in range(count):
            trend = 0.02 * i  # Slight increase
            noise = random.uniform(-1, 1)
            value = last_value + trend + noise
            value = max(0, min(100, value))  # Clamp to 0-100
            predictions.append(round(value, 2))
            
    elif 'ram' in chart.lower():
        # RAM: gradual increase
        for i in range(count):
            trend = 0.05 * i
            noise = random.uniform(-0.5, 0.5)
            value = last_value + trend + noise
            value = max(0, min(100, value))
            predictions.append(round(value, 2))
            
    elif 'load' in chart.lower():
        # Load: oscillating pattern
        for i in range(count):
            wave = 2 * math.sin(i / 10)
            noise = random.uniform(-0.2, 0.2)
            value = last_value + wave + noise
            value = max(0, value)
            predictions.append(round(value, 3))
            
    else:
        # Default: repeat last value with small variation
        for i in range(count):
            noise = random.uniform(-0.5, 0.5)
            value = last_value + noise
            predictions.append(round(value, 2))
    
    return predictions

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "mock-ml-predictions"})

if __name__ == '__main__':
    print("=" * 60)
    print("Mock ML Prediction Server Starting")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST http://localhost:1234/predict - Get predictions")
    print("  GET  http://localhost:1234/health  - Health check")
    print("\nReplace generate_mock_predictions() with your ML model.")
    print("=" * 60)
    print()
    
    # Install flask-cors if needed:
    # pip install flask flask-cors
    
    app.run(host='0.0.0.0', port=1234, debug=True)
