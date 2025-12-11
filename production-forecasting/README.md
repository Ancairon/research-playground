# Production Forecasting Service

Time series forecasting API using PyTorch LSTM-Attention with custom automated hyperparameter tuning.

## Components

- **endpoint_server.py** - Flask HTTPS API with `/forecast` and `/visualization.html` endpoints
- **forecasting.py** - Complete LSTM-Attention forecasting pipeline (model + tuning)
- **smoothing.py** - Moving average and EWMA utilities
- **visualize_helper.py** - Chart.js HTML generation for results (used for testing)

## Quick Start

### 1. Start the server

```bash
cd production-forecasting
python3 endpoint_server.py
```

Server runs on `https://localhost:5000` (self-signed cert auto-generated).

### 2. Send forecast requests

**Simple array:**

```bash
curl -k -X POST https://localhost:5000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "horizon": 5,
    "data": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
             21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
             41,42,43,44,45,46,47,48,49,50],
    "evaluation": true,
    "invoke_helper": true
  }'
```

**From JSON file:**

```bash
curl -k -X POST https://localhost:5000/forecast \
  -H "Content-Type: application/json" \
  -d @csv/10.10.40.104_ram_used.csv.json
```

**Netdata format (newest→oldest, auto-sorted):**

```bash
curl -k -X POST https://localhost:5000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "horizon": 5,
    "data": [
      [1765387008, [0.0253978, 0, 0]],
      [1765120896, [0.0264692, 0, 0]],
      [1764854784, [0.0245871, 0, 0]]
    ],
    "evaluation": true,
    "invoke_helper": true
  }'
```

## Request Format

### Required fields

- `horizon` (int): Number of future steps to predict
- `data` (array): Time series data in one of these formats:
  - Simple array: `[1, 2, 3, ...]`
  - CSV format: `[{"timestamp": "2025-01-01T00:00:00Z", "value": 1.5}, ...]`
  - Netdata format: `[[timestamp_unix, [value, ...]], ...]` (uses first value, auto-sorts ascending)

### Optional fields

- `evaluation` (bool, default: false): If true, predicts last `horizon` points and returns metrics along with actuals for comparison
- `invoke_helper` (bool, default: false): If true, generates visualization HTML at `/visualization.html`

## Response Format

```json
{
  "predictions": [10.5, 11.2, 12.1, 13.0, 14.3],
  "metrics": {
    "mape": 2.45,
    "rmse": 0.32,
    "mbe": -0.05
  },
  "actuals": [10.3, 11.0, 12.3, 13.1, 14.0],
  "visualization_url": "https://localhost:5000/visualization.html"
}
```

- **predictions**: Forecasted values
- **metrics** (if `evaluation=true`):
  - `mape`: Symmetric Mean Absolute Percentage Error
  - `rmse`: Root Mean Squared Error  
  - `mbe`: Mean Bias Error
- **actuals** (if `evaluation=true`): Ground truth values
- **visualization_url** (if `invoke_helper=true`): URL to interactive Chart.js plot

## Forecasting Pipeline

### 1. Hyperparameter Tuning (Staged Search)

#### Phase 1: Fast probe (30 epochs)

- Samples ~35% of config space (max 50 configs)
- Tests: lookback, hidden_size, learning_rate, batch_size, dropout, differencing, scaling, smoothing

#### Phase 2: Full training (150 epochs)

- Retrains top 5 configs from Phase 1
- Returns best performer

### 2. Model Architecture

- **LSTM**: 2-layer recurrent network with attention mechanism
- **Attention**: Learns which historical timesteps matter most
- **Decoder**: 3-layer feedforward network (hidden → hidden/2 → horizon)
- **Preprocessing**: Optional scaling (standard/robust) and differencing
- **Smoothing**: Optional moving average or EWMA on training data

### 3. Training Strategy

- **Early stopping**: Stops if no improvement for 7 epochs
- **Learning rate scheduling**: Reduces LR on plateau
- **Gradient clipping**: Prevents exploding gradients
- **Reproducibility**: Fixed random seeds (seed=42 (duh :D))

## Visualization

When `invoke_helper=true`, the system generates an interactive Chart.js plot showing:

- **Historical** (blue): Training window data
- **Actuals (raw)** (cyan): Ground truth from evaluation period
- **Actuals (smoothed)** (green): Smoothed ground truth (if smoothing enabled)
- **Predictions** (red): Model forecasts

Access at: `https://localhost:5000/visualization.html`

**Note**: Visualization uses a global variable, so only the most recent forecast is stored (single-user testing only).
