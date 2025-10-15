# Live XGBoost Forecasting Demo

This demo provides a live visualization of Netdata metrics with XGBoost predictions.

## Quick Start

### Option 1: Auto-configured Demo (Recommended)

Run the XGBoost server which automatically opens the demo with matching configuration:

```bash
python xgboost_live_demo.py --open-demo
```

This will:
1. Start a Flask server on http://localhost:5000
2. Fetch data from Netdata (default: localhost, system.cpu.user)
3. Train an XGBoost model
4. Open the demo page in your browser with pre-configured settings
5. Provide live predictions via API

### Option 2: Manual Configuration

1. Start the XGBoost prediction server:
```bash
python xgboost_live_demo.py --ip localhost --context system.cpu --dimension user --horizon 10
```

2. Open `live-predictions.html` in your browser

3. Configure settings (or use defaults)

4. Click "Start Live Updates"

## Configuration Options

### XGBoost Server (`xgboost_live_demo.py`)

```bash
python xgboost_live_demo.py [options]

Options:
  --ip IP                 Netdata IP address (default: localhost)
  --context CONTEXT       Netdata context (default: system.cpu)
  --dimension DIMENSION   Netdata dimension (default: user)
  --horizon N             Forecast horizon in steps (default: 10)
  --window N              History window in seconds (default: 600)
  --port PORT             Flask server port (default: 5000)
  --open-demo             Automatically open demo page
  --update-interval N     Data refresh interval in seconds (default: 30)
  --ymin VALUE            Chart Y-axis minimum
  --ymax VALUE            Chart Y-axis maximum
```

### Demo Page (`live-predictions.html`)

#### URL Parameters (auto-configured when using --open-demo):
- `netdata` - Netdata URL (e.g., http://localhost:19999)
- `context` - Chart context (e.g., system.cpu)
- `dimension` - Dimension to plot (e.g., user)
- `horizon` - Prediction horizon
- `mlServer` - ML server URL (e.g., http://localhost:5000/predict)
- `ymin` / `ymax` - Y-axis limits
- `points` - History points to display
- `autostart=true` - Auto-start live updates

#### Manual Controls:
- **Netdata URL**: Netdata server endpoint
- **Chart (context)**: Metric context (e.g., system.cpu, system.load, etc.)
- **Dimension**: Specific dimension to plot
- **History Points**: Number of historical points to display
- **Prediction Horizon**: Number of future steps to predict
- **ML Server URL**: XGBoost prediction API endpoint
- **Update Interval**: Refresh frequency in milliseconds
- **Y-axis Min/Max**: Fixed axis limits (leave blank for auto)

## Examples

### CPU Usage Monitoring
```bash
python xgboost_live_demo.py \
  --ip localhost \
  --context system.cpu \
  --dimension user \
  --horizon 15 \
  --ymin 0 \
  --ymax 100 \
  --open-demo
```

### Network Traffic
```bash
python xgboost_live_demo.py \
  --ip localhost \
  --context net.eth0 \
  --dimension received \
  --horizon 20 \
  --window 1800 \
  --open-demo
```

### Memory Usage
```bash
python xgboost_live_demo.py \
  --ip localhost \
  --context system.ram \
  --dimension used \
  --horizon 10 \
  --open-demo
```

## API Endpoints

When the XGBoost server is running:

- **POST** `/predict` - Get predictions
  ```json
  {
    "data": [1.2, 3.4, 5.6, ...],
    "horizon": 10
  }
  ```

- **POST** `/retrain` - Force model retrain with fresh data

- **GET** `/status` - Check server status

## Features

- ✅ Real-time Netdata metric visualization
- ✅ Live XGBoost predictions with dashed line
- ✅ Configurable Y-axis limits
- ✅ Auto-updating chart
- ✅ Background data refresh
- ✅ Automatic model retraining
- ✅ Dark theme UI
- ✅ URL parameter configuration
- ✅ Fallback to static predictions if server unavailable

## Requirements

```bash
pip install flask pandas requests pycaret xgboost matplotlib
```

## Troubleshooting

**Chart doesn't update:**
- Check console (F12) for errors
- Verify Netdata is accessible at configured URL
- Ensure XGBoost server is running (http://localhost:5000/status)

**Predictions are flat:**
- XGBoost server may not be running - check http://localhost:5000/status
- Fallback mode shows static predictions (repeats last value)

**Model training errors:**
- Ensure sufficient data points (>100 recommended)
- Check PyCaret installation: `pip install pycaret[time_series]`
