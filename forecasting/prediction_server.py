#!/usr/bin/env python3
"""
Simple prediction server that receives predictions from xgboost_whole.py
and serves them to the live demo page.
"""

import argparse
import webbrowser
import threading
import os
from flask import Flask, request, jsonify, send_from_directory, redirect
from collections import deque
from datetime import datetime

app = Flask(__name__)

# Store server configuration for demo page
server_config = {
    'ip': 'localhost',
    'context': 'system.cpu',
    'dimension': 'user'
}

# Store latest predictions
latest_predictions = {
    'predictions': [],
    'timestamp': None,
    'metadata': {}
}

# Store prediction history (last 100)
prediction_history = deque(maxlen=100)

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


@app.route('/predictions', methods=['POST'])
def receive_predictions():
    """Receive predictions from xgboost_whole.py"""
    global latest_predictions, prediction_history
    
    try:
        data = request.json
        latest_predictions = {
            'predictions': data.get('predictions', []),
            'timestamp': datetime.now().isoformat(),
            'backoff': data.get('backoff', False),
            'actual': data.get('actual', 0),
            'csv_mode': data.get('csv_mode', False),
            'historical_data': data.get('historical_data'),
            'smoothed_historical': data.get('smoothed_historical'),
            'future_actuals': data.get('future_actuals'),  # NEW: actual future data for single-shot
            'single_shot': data.get('single_shot', False),  # NEW: single-shot mode flag
            'mean_error': data.get('mean_error'),  # NEW: overall error for single-shot
            'metadata': {
                'context': data.get('context'),
                'dimension': data.get('dimension'),
                'horizon': len(data.get('predictions', [])),
                'prediction_interval': data.get('prediction_interval')
            }
        }
        
        # Add to history
        prediction_history.append(latest_predictions.copy())
        
        status = "BACKOFF" if latest_predictions['backoff'] else "OK"
        csv_indicator = "[CSV]" if latest_predictions['csv_mode'] else "[LIVE]"
        mode_indicator = "[SINGLE-SHOT]" if latest_predictions['single_shot'] else ""
        print(f"{csv_indicator}{mode_indicator} [{latest_predictions['timestamp']}] {status} - Received {len(latest_predictions['predictions'])} predictions")
        
        return jsonify({'status': 'ok', 'received': len(latest_predictions['predictions'])})
    
    except Exception as e:
        print(f"Error receiving predictions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def get_predictions():
    """Serve latest predictions to demo page (compatible with existing demo)"""
    global latest_predictions
    
    try:
        return jsonify({
            'predictions': latest_predictions.get('predictions', []),
            'timestamp': latest_predictions.get('timestamp'),
            'backoff': latest_predictions.get('backoff', False),
            'actual': latest_predictions.get('actual', 0),
            'csv_mode': latest_predictions.get('csv_mode', False),
            'historical_data': latest_predictions.get('historical_data'),
            'smoothed_historical': latest_predictions.get('smoothed_historical'),
            'future_actuals': latest_predictions.get('future_actuals'),  # NEW
            'single_shot': latest_predictions.get('single_shot', False),  # NEW
            'mean_error': latest_predictions.get('mean_error'),  # NEW
            'metadata': latest_predictions.get('metadata', {})
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """Check server status"""
    return jsonify({
        'status': 'running',
        'has_predictions': len(latest_predictions['predictions']) > 0,
        'last_update': latest_predictions['timestamp'],
        'history_size': len(prediction_history)
    })


@app.route('/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    return jsonify({
        'history': list(prediction_history)
    })


@app.route('/')
def index():
    """Serve the demo page with current configuration"""
    return redirect('/demo')


@app.route('/demo')
def demo():
    """Serve the live predictions demo page"""
    demo_dir = os.path.join(os.path.dirname(__file__), 'demo')
    return send_from_directory(demo_dir, 'live-predictions.html')


@app.route('/config')
def get_config():
    """Get server configuration for demo page"""
    global server_config
    return jsonify(server_config)


def main(port=5000, open_demo=False, config=None):
    """Start the prediction server"""
    
    global server_config
    if config:
        server_config.update(config)
    
    if open_demo:
        # Build demo URL with pre-configured parameters
        demo_path = os.path.join(os.path.dirname(__file__), 'demo', 'live-predictions.html')
        demo_url = f"file://{os.path.abspath(demo_path)}"
        
        if config:
            demo_url += f"?netdata=http://{config.get('ip', 'localhost')}:19999"
            demo_url += f"&context={config.get('context', 'system.cpu')}"
            demo_url += f"&dimension={config.get('dimension', 'user')}"
            demo_url += f"&mlServer=http://localhost:{port}/predict"
            if config.get('ymin') is not None:
                demo_url += f"&ymin={config['ymin']}"
            if config.get('ymax') is not None:
                demo_url += f"&ymax={config['ymax']}"
        
        print(f"\nOpening demo at: {demo_url}\n")
        threading.Timer(1.0, lambda: webbrowser.open(demo_url)).start()
    
    print("=" * 60)
    print("Prediction Server Running")
    print("=" * 60)
    print(f"Web UI: http://localhost:{port}/")
    print(f"Monitoring: {server_config['context']} -> {server_config['dimension']}")
    print(f"Receive predictions: POST http://localhost:{port}/predictions")
    print(f"Get predictions: POST http://localhost:{port}/predict")
    print(f"Status: GET http://localhost:{port}/status")
    print(f"History: GET http://localhost:{port}/history")
    print("=" * 60)
    print()
    
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediction Server for Live Demo")
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--open-demo', action='store_true', help='Open demo page')
    parser.add_argument('--ip', default='localhost', help='Netdata IP for demo config')
    parser.add_argument('--context', default='system.cpu', help='Context for demo config')
    parser.add_argument('--dimension', default='user', help='Dimension for demo config')
    parser.add_argument('--horizon', type=int, default=10, help='Prediction horizon for demo config')
    parser.add_argument('--ymin', type=float, help='Y-axis min for demo')
    parser.add_argument('--ymax', type=float, help='Y-axis max for demo')
    
    args = parser.parse_args()
    
    config = {
        'ip': args.ip,
        'context': args.context,
        'dimension': args.dimension,
        'horizon': args.horizon,
        'ymin': args.ymin,
        'ymax': args.ymax
    }
    
    main(port=args.port, open_demo=args.open_demo, config=config)
