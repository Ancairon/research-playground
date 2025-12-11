"""
Production HTTPS endpoint for forecasting.

Accepts POST /forecast with JSON payload containing horizon and data, runs LSTM-Attention tuner and forecaster, returns predictions.
"""

import os
import ssl
import pandas as pd
from flask import Flask, request, jsonify
from forecasting import tune_and_forecast
import subprocess
from io import StringIO
from flask import request
from visualize_helper import generate_visualization_html


app = Flask(__name__)


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        horizon = data.get('horizon')
        if not isinstance(horizon, int) or horizon <= 0:
            return jsonify({'error': 'Invalid horizon: must be positive integer'}), 400

        invoke_helper = data.get('invoke_helper', False)
        evaluation = data.get('evaluation', False)

        # Parse data
        if 'data' in data:
            time_series = parse_data(data['data'])
        elif 'csv_data' in data:
            time_series = parse_csv_data(data['csv_data'])
        elif 'csv_path' in data:
            time_series = pd.read_csv(data['csv_path'], parse_dates=[
                                      'timestamp'], index_col='timestamp')['value']
        else:
            return jsonify({'error': 'Must provide "data" array, "csv_data" string, or "csv_path"'}), 400

        if len(time_series) < horizon + 10:  # minimum data
            return jsonify({'error': 'Insufficient data for forecasting'}), 400

        # Run tuning and forecasting
        results = tune_and_forecast(time_series, horizon, evaluation)

        response = {
            'predictions': results['predictions']
        }

        if evaluation:
            response['metrics'] = {
                'mape': results['mape'],
                'rmse': results['rmse'],
                'mbe': results['mbe']
            }
            response['actuals'] = results['actuals']

        if invoke_helper:
            metrics = response.get('metrics') if evaluation else None
            actuals = results.get('actuals') if evaluation else None
            # lookback + horizon used for training
            train_window = results.get('train_window')
            raw_train = results.get('train_data') if evaluation else None
            smoothed_train = results.get('smoothed_train_data') if results.get(
                'smoothing_applied') else None
            smoothed_actuals = results.get('smoothed_actuals') if results.get(
                'smoothing_applied') else None
            viz_url = run_visualization_helper(
                time_series, results['predictions'], horizon, metrics, actuals, train_window,
                smoothed_train, smoothed_actuals, raw_train
            )
            response['visualization_url'] = viz_url

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def parse_data(data):
    """Parse data from various formats into pd.Series."""
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and 'timestamp' in data[0] and 'value' in data[0]:
            # CSV format: [{"timestamp": str, "value": float}, ...]
            timestamps = [pd.to_datetime(item['timestamp']) for item in data]
            values = [item['value'] for item in data]
            series = pd.Series(values, index=timestamps)
        elif data and isinstance(data[0], list) and len(data[0]) == 2:
            # Netdata format: [[timestamp, [val1, val2, val3]], ...]
            timestamps = [item[0] for item in data]
            values = [item[1][0] if isinstance(
                item[1], list) else item[1] for item in data]
            series = pd.Series(
                values, index=pd.to_datetime(timestamps, unit='s'))
        else:
            # Simple array: assume sequential timestamps
            series = pd.Series(data)
    else:
        raise ValueError('Data must be array')
    return series


def parse_csv_data(csv_string):
    """Parse CSV string into pd.Series."""
    df = pd.read_csv(StringIO(csv_string), parse_dates=[
                     'timestamp'], index_col='timestamp')
    return df['value']


# Global storage for last visualization HTML
_last_visualization_html = None


def run_visualization_helper(data, predictions, horizon, metrics=None, actuals=None, train_window=None,
                             smoothed_train_data=None, smoothed_actuals=None, raw_train_data=None):
    """Generate visualization HTML and store it for serving."""
    global _last_visualization_html

    # Use raw_train_data from results if available (already correctly sliced)
    # Otherwise fall back to slicing full data
    if raw_train_data is not None:
        display_data = raw_train_data
    else:
        full_data = data.tolist() if hasattr(data, 'tolist') else list(data)
        if train_window and train_window < len(full_data):
            display_data = full_data[-train_window:]
        else:
            display_data = full_data

    # Get timestamps if available
    timestamps = None
    if hasattr(data, 'index') and hasattr(data.index, 'to_pydatetime'):
        try:
            all_ts = [ts.isoformat() for ts in data.index.to_pydatetime()]
            if train_window and train_window < len(all_ts):
                timestamps = all_ts[-train_window:]
            else:
                timestamps = all_ts
        except Exception:
            timestamps = None

    # Generate HTML with smoothing data if available
    _last_visualization_html = generate_visualization_html(
        historical_values=display_data,
        predictions=predictions,
        metrics=metrics,
        historical_timestamps=timestamps,
        actuals=actuals,
        smoothed_historical=smoothed_train_data,
        smoothed_actuals=smoothed_actuals,
        title="Forecast Results"
    )

    # Return full URL using request context
    return f"{request.url_root.rstrip('/')}/visualization.html"


@app.route('/visualization.html')
def serve_viz():
    """Serve the last generated visualization."""
    global _last_visualization_html
    if _last_visualization_html:
        return _last_visualization_html, 200, {'Content-Type': 'text/html'}
    return "No visualization available. Run /forecast with invoke_helper=true first.", 404


if __name__ == '__main__':
    # SSL context for HTTPS
    context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    # For local testing, use self-signed cert
    cert_file = 'cert.pem'
    key_file = 'key.pem'
    if not os.path.exists(cert_file):
        # Generate self-signed cert
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:4096', '-keyout', key_file,
                       '-out', cert_file, '-days', '365', '-nodes', '-subj', '/CN=localhost'], check=True)
    context.load_cert_chain(cert_file, key_file)

    app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=True)
