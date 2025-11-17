# Universal Time Series Forecasting Framework

This is a modular, model-agnostic forecasting system with adaptive retraining and backoff control.

## Architecture

```bash
forecasting/
├── forecast_main.py           # CLI entry point
├── universal_forecaster.py    # Core forecasting engine
├── prediction_server.py       # Live visualization server
├── models/
│   ├── __init__.py           # Model factory & registry
│   ├── base_model.py         # Abstract base interface
│   ├── xgboost_model.py      # XGBoost implementation
│   └── lstm_model.py         # LSTM/GRU implementation
└── demo/
    ├── live-predictions.html  # Web UI for visualization
    └── inject-predictions.js  # Client-side charting
```

## Components

### forecast_main.py

- CLI argument parsing
- Data acquisition (Netdata API or pre-made CSV)
- Model initialization and configuration (can hardcode default values, so you can run just forecast.py)
  
  > Helps with testing, as you can have two terminal tabs, and alter the defaults in the editor, and not in the command with the arrows
- Main prediction loop with timing control
- Live server management (auto-kills old instances, so you don't have a zillion servers running)
- Final statistics reporting (once you hit ctrl+c, it reports main metrics)

### universal_forecaster.py

Core forecasting engine with:

- **Prediction smoothing**: Rolling average of last N predictions
- **Deferred validation**: Wait for full horizon before measuring error (so we can measure the full prediction against the full real thing)
- **Adaptive thresholds**: MAD/std-based error thresholds with baseline deviation adjustment (this helps with metrics being spiky, or a bit unstable, so that we don't retrain all the time)
- **Dynamic retraining**: Trigger retraining on consecutive threshold violations
- **Backoff mechanism**: Suppress predictions when model is unstable (if there is rapid retraining detected, don't give predictions, we are probably looking at something that is beginning to be event-based rather than a pattern based behavior)
  - **Hidden predictions**: Generate and validate suppressed predictions during backoff
- **Baseline tracking**: Monitor data drift from training distribution
- **Metrics**: MAPE (capped at 1000%), MBE, PBIAS, percentiles

### prediction_server.py

Flask-based WebSocket server for live visualization:

- Serves live HTML dashboard
- Pushes predictions and actuals to browser
- Handles backoff state for UI coloring (purple=normal, red=backoff)
- CORS enabled for development

### Models (inside `models/`)

plug and play model implementations following `BaseTimeSeriesModel` interface:

- **XGBoost**: Fast gradient boosting, supports online learning
- **LSTM**: PyTorch recurrent neural network for sequences

All models expose:

- `train(data)`: Train on data window
- `predict()`: Generate horizon-step predictions
- `update(data)`: Incremental learning (if supported)
- `supports_online_learning()`: Capability flag

(prophet was really bad, name suggested otherwise :smile:, LSTM was not a good performer initially, but I haven't tested it much)

**This has mainly been developed for XGBoost, so some stuff might not be present in the other two models, they were made to have a bit of variety**

| Model       | Best For                   | Online Learning | Speed     | Predictive |
|-------------|----------------------------|-----------------|-----------|------------|
| **XGBoost** | General regression         | ✅ Yes           | ⚡⚡⚡ Fast  | ❌ Reactive |
| **LSTM**    | Complex sequences          | ❌ No            | ⚡ Slow    | ✅ Yes      |

## Key Features

### Adaptive Retraining

- **MAD-based thresholds**: Robust to outliers (1.4826 × MAD ≈ std)
- **Baseline deviation adjustment**: Lowers threshold when data drifts from training distribution
- **Consecutive violations**: Requires N consecutive bad predictions to retrain
- **Cooldown period** (best be 0 though, as waiting for retrain affects the backoff retrain threshold): Prevents over-retraining

### Backoff Control

When rapid retraining is detected (retrains within `--retrain-rapid-seconds`, default 10s):

- **Suppresses predictions**: Blocks user-facing predictions for a specific time period
- **Hidden validation**: Continues generating/validating predictions in background
- **Adaptive extension**: Extends backoff if hidden predictions still fail
- **Auto-clear**: Exits backoff after N consecutive OK validations

(here we could also make the background evaluation on a longer interval, but that would make the model less responsive to re-emerging from suppression)

### Baseline Deviation Tracking

Monitors z-score of current data vs training baseline:

```python
deviation = |current_mean - baseline_mean| / baseline_std
```

If deviation > 2.0σ, threshold is lowered (more sensitive retraining):

```python
adjustment = max(0.7, 1.0 - (deviation - 2.0) × 0.1)
threshold_adjusted = base_threshold × adjustment
```

### Error Calculation

Symmetric MAPE (robust to near-zero values):

```python
error = |actual - pred| / ((|actual| + |pred|) / 2) × 100%
```

- Capped at 1000% to prevent overflow
- Skips validation when both actual and predicted near zero

## Quick Start

### Basic Usage

```bash
# XGBoost (default)
python3 forecast_main.py
```

### Common Options

```bash
# Data source
--csv data.csv              # Offline CSV simulation
--ip localhost              # Netdata server IP
--context system.net        # Metric context
--dimension received        # Metric dimension

# Model configuration
--model {xgboost,lstm}  # Model selection
--horizon 5                     # Forecast horizon
--window 300                    # Training window size
--random-state 42               # Random seed

# Forecasting behavior
--prediction-smoothing 5        # Average last N predictions
--prediction-interval 2.0       # Seconds between predictions

# Retraining (dynamic)
--retrain-scale 3.0            # MAD multiplier for threshold
--retrain-min 50.0             # Minimum threshold (%)
--retrain-consec 2             # Consecutive violations to retrain
--retrain-cooldown 5           # Min steps between retrains
--no-mad                       # Use std instead of MAD (not recommended)

# Output
--quiet                        # Suppress console output
--no-live-server              # Disable visualization server
```
