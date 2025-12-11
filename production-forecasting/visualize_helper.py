"""
Visualization helper for forecasting results.

Generates interactive Chart.js HTML visualization from data and predictions.

Handles timestamps:
1. If response has timestamps - use them directly
2. If no timestamps in response but original data has timestamps - infer granularity and generate
3. If original data has no timestamps - use index-based X axis (point 0, 1, 2, ...)
"""

import json
from datetime import datetime, timedelta


def infer_granularity_seconds(timestamps):
    """
    Infer the data granularity from a list of timestamps.
    Returns the median interval in seconds.
    """
    if len(timestamps) < 2:
        return 60  # default to 1 minute

    intervals = []
    for i in range(1, min(len(timestamps), 20)):  # sample first 20 intervals
        t1 = timestamps[i-1]
        t2 = timestamps[i]

        # Parse timestamps if strings
        if isinstance(t1, str):
            t1 = datetime.fromisoformat(t1.replace('Z', '+00:00'))
        if isinstance(t2, str):
            t2 = datetime.fromisoformat(t2.replace('Z', '+00:00'))

        if hasattr(t1, 'timestamp') and hasattr(t2, 'timestamp'):
            diff = abs(t2.timestamp() - t1.timestamp())
            intervals.append(diff)

    if not intervals:
        return 60

    # Return median interval
    intervals.sort()
    return intervals[len(intervals) // 2]


def generate_visualization_html(
    historical_values,
    predictions,
    metrics=None,
    historical_timestamps=None,
    prediction_timestamps=None,
    actuals=None,
    smoothed_historical=None,
    smoothed_actuals=None,
    title="Time Series Forecast"
):
    """
    Generate an interactive Chart.js HTML visualization.

    Args:
        historical_values: list of float - the raw historical data values (train window)
        predictions: list of float - the predicted values
        metrics: dict with 'mape', 'rmse', 'mbe' (optional)
        historical_timestamps: list of timestamps for historical data (optional)
        prediction_timestamps: list of timestamps for predictions (optional)
        actuals: list of float - raw actual values for the prediction period (optional)
        smoothed_historical: list of float - smoothed historical data (optional, when smoothing applied)
        smoothed_actuals: list of float - smoothed actual values (optional, when smoothing applied)
        title: chart title

    Returns:
        HTML string ready to serve
    """

    # Determine if we have timestamps
    has_timestamps = historical_timestamps is not None and len(
        historical_timestamps) > 0

    if has_timestamps:
        # Use real timestamps
        granularity = infer_granularity_seconds(historical_timestamps)

        # Parse historical timestamps to JS-friendly format
        hist_labels = []
        for ts in historical_timestamps:
            if isinstance(ts, str):
                hist_labels.append(ts)
            elif hasattr(ts, 'isoformat'):
                hist_labels.append(ts.isoformat())
            else:
                hist_labels.append(str(ts))

        # Generate prediction timestamps if not provided
        if prediction_timestamps and len(prediction_timestamps) == len(predictions):
            pred_labels = []
            for ts in prediction_timestamps:
                if isinstance(ts, str):
                    pred_labels.append(ts)
                elif hasattr(ts, 'isoformat'):
                    pred_labels.append(ts.isoformat())
                else:
                    pred_labels.append(str(ts))
        else:
            # Infer prediction timestamps from last historical + granularity
            last_ts = historical_timestamps[-1]
            if isinstance(last_ts, str):
                last_dt = datetime.fromisoformat(
                    last_ts.replace('Z', '+00:00'))
            else:
                last_dt = last_ts

            pred_labels = []
            for i in range(len(predictions)):
                next_dt = last_dt + timedelta(seconds=granularity * (i + 1))
                pred_labels.append(next_dt.isoformat())

        # Combine labels for chart
        all_labels = hist_labels + pred_labels
        x_axis_type = 'time'

    else:
        # No timestamps - use index-based labels
        hist_labels = list(range(len(historical_values)))
        pred_labels = list(range(len(historical_values), len(
            historical_values) + len(predictions)))
        all_labels = hist_labels + pred_labels
        x_axis_type = 'linear'

    # Determine if smoothing was applied
    has_smoothing = smoothed_historical is not None and len(
        smoothed_historical) > 0
    has_actuals = actuals is not None and len(actuals) > 0
    has_smoothed_actuals = smoothed_actuals is not None and len(
        smoothed_actuals) > 0

    # Prepare data for Chart.js
    # Raw Historical: values for historical points, null for prediction points
    hist_data = list(historical_values) + [None] * len(predictions)

    # Smoothed Historical (if available)
    smoothed_hist_data = (list(smoothed_historical) +
                          [None] * len(predictions)) if has_smoothing else None

    # Predictions: null for historical points, values for prediction points
    pred_data = [None] * len(historical_values) + list(predictions)

    # Raw Actuals: null for historical points, actual values for prediction points (if provided)
    actuals_data = [None] * len(historical_values) + (list(actuals)
                                                      if has_actuals else [None] * len(predictions))

    # Smoothed Actuals (if available)
    smoothed_actuals_data = ([None] * len(historical_values) +
                             list(smoothed_actuals)) if has_smoothed_actuals else None

    # Build metrics display
    metrics_html = ""
    if metrics:
        metrics_html = f"""
        <div class="metrics">
            <span>MAPE: {metrics.get('mape', 'N/A'):.2f}%</span>
            <span>RMSE: {metrics.get('rmse', 'N/A'):.4f}</span>
            <span>MBE: {metrics.get('mbe', 'N/A'):.4f}</span>
        </div>
        """

    # Build datasets
    datasets = []

    # Raw Historical (light blue, thinner if smoothing present)
    if has_smoothing:
        datasets.append(f"""{{
                        label: 'Historical (raw)',
                        data: {json.dumps(hist_data)},
                        borderColor: 'rgba(77, 171, 247, 0.4)',
                        backgroundColor: 'rgba(77, 171, 247, 0.05)',
                        borderWidth: 1,
                        pointRadius: 0,
                        pointHoverRadius: 3,
                        fill: false,
                        tension: 0.1,
                        spanGaps: false
                    }}""")
        # Smoothed Historical (solid blue)
        datasets.append(f"""{{
                        label: 'Historical (smoothed)',
                        data: {json.dumps(smoothed_hist_data)},
                        borderColor: '#4dabf7',
                        backgroundColor: 'rgba(77, 171, 247, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: false,
                        tension: 0.1,
                        spanGaps: false
                    }}""")
    else:
        datasets.append(f"""{{
                        label: 'Historical',
                        data: {json.dumps(hist_data)},
                        borderColor: '#4dabf7',
                        backgroundColor: 'rgba(77, 171, 247, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: false,
                        tension: 0.1,
                        spanGaps: false
                    }}""")

    # Actuals (green)
    if has_actuals:
        if has_smoothed_actuals:
            # Raw actuals (light green)
            datasets.append(f"""{{
                        label: 'Actuals (raw)',
                        data: {json.dumps(actuals_data)},
                        borderColor: 'rgba(81, 207, 102, 0.4)',
                        backgroundColor: 'rgba(81, 207, 102, 0.05)',
                        borderWidth: 1,
                        pointRadius: 1,
                        pointHoverRadius: 3,
                        fill: false,
                        tension: 0.1,
                        spanGaps: false
                    }}""")
            # Smoothed actuals (solid green)
            datasets.append(f"""{{
                        label: 'Actuals (smoothed)',
                        data: {json.dumps(smoothed_actuals_data)},
                        borderColor: '#51cf66',
                        backgroundColor: 'rgba(81, 207, 102, 0.1)',
                        borderWidth: 2,
                        pointRadius: 2,
                        pointHoverRadius: 4,
                        fill: false,
                        tension: 0.1,
                        spanGaps: false
                    }}""")
        else:
            datasets.append(f"""{{
                        label: 'Actuals',
                        data: {json.dumps(actuals_data)},
                        borderColor: '#51cf66',
                        backgroundColor: 'rgba(81, 207, 102, 0.1)',
                        borderWidth: 2,
                        pointRadius: 2,
                        pointHoverRadius: 4,
                        fill: false,
                        tension: 0.1,
                        spanGaps: false
                    }}""")

    # Predictions (red dashed)
    datasets.append(f"""{{
                        label: 'Predictions',
                        data: {json.dumps(pred_data)},
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 2,
                        pointHoverRadius: 4,
                        fill: false,
                        tension: 0.1,
                        spanGaps: false
                    }}""")

    datasets_js = ",\n                    ".join(datasets)

    # Legend info
    legend_parts = [
        f"<strong>Historical:</strong> {len(historical_values)} points"]
    if has_smoothing:
        legend_parts[-1] += " (raw + smoothed)"
    legend_parts.append(
        f"<strong>Predictions:</strong> {len(predictions)} points")
    if has_actuals:
        legend_parts.append(f"<strong>Actuals:</strong> {len(actuals)} points")
        if has_smoothed_actuals:
            legend_parts[-1] += " (raw + smoothed)"
    legend_info = " &nbsp;|&nbsp; ".join(legend_parts)

    # Generate HTML with Chart.js
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background-color: #0c0c0e;
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #ffffff;
            margin-bottom: 20px;
            font-size: 24px;
            font-weight: 500;
        }}
        .metrics {{
            display: flex;
            gap: 30px;
            margin-bottom: 20px;
            padding: 15px 20px;
            background: #1a1a1d;
            border-radius: 8px;
            border: 1px solid #333;
        }}
        .metrics span {{
            font-size: 14px;
            color: #b0b0b0;
        }}
        .chart-container {{
            background: #1a1a1d;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #333;
        }}
        canvas {{
            width: 100% !important;
            height: 500px !important;
        }}
        .legend-info {{
            margin-top: 15px;
            padding: 10px;
            background: #0c0c0e;
            border-radius: 4px;
            font-size: 12px;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        {metrics_html}
        <div class="chart-container">
            <canvas id="forecastChart"></canvas>
        </div>
        <div class="legend-info">
            {legend_info}
        </div>
    </div>

    <script>
        const labels = {json.dumps(all_labels)};
        const xAxisType = '{x_axis_type}';

        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {datasets_js}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{
                            color: '#e0e0e0',
                            usePointStyle: true,
                            padding: 20
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: '#333',
                        borderWidth: 1
                    }}
                }},
                scales: {{
                    x: {{
                        type: xAxisType,
                        display: true,
                        title: {{
                            display: true,
                            text: xAxisType === 'time' ? 'Time' : 'Index',
                            color: '#888'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.1)'
                        }},
                        ticks: {{
                            color: '#888',
                            maxTicksLimit: 10
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Value',
                            color: '#888'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.1)'
                        }},
                        ticks: {{
                            color: '#888'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    return html


def generate_visualization_from_response(response_data, original_data=None, original_timestamps=None):
    """
    Generate visualization from endpoint response.

    Args:
        response_data: dict from /forecast endpoint containing:
            - predictions: list of predicted values
            - actuals: list of actual values used
            - metrics: dict with mape, rmse, mbe
        original_data: optional, original input data (list of values)
        original_timestamps: optional, timestamps for the original data

    Returns:
        HTML string
    """
    predictions = response_data.get('predictions', [])
    actuals = response_data.get('actuals', [])
    metrics = response_data.get('metrics', {})

    # Use actuals if available, otherwise use original_data
    historical = actuals if actuals else (original_data or [])

    return generate_visualization_html(
        historical_values=historical,
        predictions=predictions,
        metrics=metrics,
        historical_timestamps=original_timestamps,
        prediction_timestamps=None,  # Will be inferred from historical
        title="Forecast Results"
    )


# Standalone test
if __name__ == '__main__':
    # Test with dummy data
    historical = [10, 12, 11, 13, 14, 15, 14, 16, 17, 18]
    predictions = [19, 20, 21, 22, 23]
    metrics = {'mape': 5.2, 'rmse': 1.3, 'mbe': 0.5}

    # Test without timestamps (index-based)
    html = generate_visualization_html(historical, predictions, metrics)
    with open('test_viz_no_ts.html', 'w') as f:
        f.write(html)
    print("Generated test_viz_no_ts.html (index-based)")

    # Test with timestamps
    from datetime import datetime, timedelta
    base = datetime.now()
    timestamps = [base + timedelta(minutes=i) for i in range(len(historical))]
    html = generate_visualization_html(
        historical, predictions, metrics, timestamps)
    with open('test_viz_with_ts.html', 'w') as f:
        f.write(html)
    print("Generated test_viz_with_ts.html (time-based)")
