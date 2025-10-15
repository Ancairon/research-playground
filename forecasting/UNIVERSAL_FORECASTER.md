# Universal Time Series Forecasting Framework

Modular, model-agnostic forecasting system with support for multiple ML models.

## Architecture

```
forecasting/
├── forecast_main.py           # Main entry point (CLI)
├── universal_forecaster.py    # Model-agnostic forecasting engine
├── models/
│   ├── __init__.py           # Model factory
│   ├── base_model.py         # Abstract base class
│   ├── xgboost_model.py      # XGBoost implementation
│   ├── prophet_model.py      # Prophet implementation
│   └── lstm_model.py         # LSTM/GRU implementation
└── xgboost_whole.py          # Legacy (deprecated)
```

## Supported Models

| Model | Best For | Online Learning | Speed | Predictive |
|-------|----------|-----------------|-------|------------|
| **XGBoost** | General regression | ✅ Yes | ⚡⚡⚡ Fast | ❌ Reactive |
| **Prophet** | Periodic/seasonal patterns | ❌ No | ⚡⚡ Medium | ✅ Yes |
| **LSTM** | Complex sequences | ❌ No | ⚡ Slow | ✅ Yes |

## Quick Start

### Basic Usage

```bash
# XGBoost (default)
python3 forecast_main.py

# Prophet for spiky patterns
python3 forecast_main.py --model prophet --custom-seasonality spike:15:5

# LSTM for complex patterns
python3 forecast_main.py --model lstm --lookback 120 --epochs 100
```

### Common Options

```bash
# Data source
--csv data.csv              # Offline CSV simulation
--ip localhost              # Netdata server IP
--context system.net        # Metric context
--dimension received        # Metric dimension

# Model configuration
--model {xgboost,prophet,lstm}  # Model selection
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

## Model-Specific Parameters

### XGBoost
```bash
python3 forecast_main.py \
  --model xgboost \
  --seasonal-period 60 \
  --window 300 \
  --prediction-smoothing 5
```

### Prophet
```bash
# For 15-second spike cycle
python3 forecast_main.py \
  --model prophet \
  --custom-seasonality spike_cycle:15:5 \
  --window 300 \
  --prediction-smoothing 3

# For multiple seasonalities, run prophet with code
```

### LSTM
```bash
python3 forecast_main.py \
  --model lstm \
  --lookback 120 \
  --hidden-size 64 \
  --epochs 100 \
  --window 600 \
  --prediction-smoothing 7
```

## Pattern-Specific Recommendations

### Smooth Patterns (sine wave, daily cycle)
```bash
python3 forecast_main.py \
  --model prophet \
  --window 180 \
  --prediction-smoothing 3 \
  --prediction-interval 1.0
```

### Spiky Patterns (regular bursts)
```bash
python3 forecast_main.py \
  --model prophet \
  --custom-seasonality spike:15:5 \
  --window 600 \
  --prediction-smoothing 7 \
  --prediction-interval 2.0
```

### Step Changes (business hours pattern)
```bash
python3 forecast_main.py \
  --model xgboost \
  --window 300 \
  --prediction-smoothing 5 \
  --prediction-interval 2.0 \
  --retrain-scale 2.0
```

### Trends (increasing baseline)
```bash
python3 forecast_main.py \
  --model lstm \
  --lookback 120 \
  --window 600 \
  --prediction-smoothing 5 \
  --retrain-scale 2.0
```

## Adding New Models

1. Create new file in `models/` (e.g., `arima_model.py`)
2. Implement `BaseTimeSeriesModel` interface:
   ```python
   from models.base_model import BaseTimeSeriesModel
   
   class ARIMAModel(BaseTimeSeriesModel):
       def train(self, data, **kwargs): ...
       def predict(self, **kwargs): ...
       def update(self, data, **kwargs): ...
       def get_model_name(self): ...
       def get_model_params(self): ...
   ```
3. Register in `models/__init__.py`:
   ```python
   from models.arima_model import ARIMAModel
   
   MODEL_REGISTRY = {
       'xgboost': XGBoostModel,
       'prophet': ProphetModel,
       'lstm': LSTMModel,
       'arima': ARIMAModel,  # Add here
   }
   ```
4. Use it:
   ```bash
   python3 forecast_main.py --model arima
   ```

## Universal Forecaster Features

The `UniversalForecaster` class handles all model-agnostic logic:

- ✅ **Prediction smoothing** - ensemble averaging of recent predictions
- ✅ **Deferred validation** - validate predictions after horizon steps
- ✅ **Error tracking** - MAPE, MBE, PBIAS metrics
- ✅ **Dynamic retraining** - MAD-based threshold with consecutive violations
- ✅ **Logging** - CSV logs for retrain decisions
- ✅ **Statistics** - comprehensive final statistics

All models benefit from these features automatically!

## Understanding Retraining

### MAD-Based Thresholds

```
threshold = max(retrain_min, retrain_scale × MAD × 1.4826)

For MAPE errors:
- retrain_scale = 2.0-5.0 (default 3.0)
- retrain_min = 20.0-50.0%

Example:
errors = [40%, 45%, 38%, 42%]
MAD ≈ 3%
threshold = max(50%, 3.0 × 3% × 1.4826) = max(50%, 13.3%) = 50%
```

### Retrain Triggers

Retraining occurs when **all** conditions are met:
1. `mean_horizon_error > threshold` for `retrain_consec` consecutive validations
2. At least `retrain_cooldown` steps since last retrain
3. At least `retrain_consec` errors collected

## Output

### Console Output
```
[2025-10-15 07:00:26] VALIDATION step=39 mean_horizon_error=21.02% threshold=50.00% (retrain_min=50.0)
[Prophet (Meta)] Retrained at step 45 (time=2.341s), threshold=65.3%
```

### Retrain Log
CSV file: `{model_name}_retrain_log.csv`
```
step,timestamp,pred_timestamp,mean_horizon_error_pct,threshold,consec_count,last_retrain_step,retrain_triggered
39,2025-10-15 07:00:26,36,21.02,50.0,0,-9999,False
45,2025-10-15 07:00:32,42,67.31,65.3,2,39,True
```

### Final Statistics
```
==============================================================
Final Statistics (Prophet (Meta))
==============================================================
MAPE - Mean: 18.45%  Max: 67.31%  Min: 5.22%
MAPE - P80: 23.11%  P95: 45.67%  P99: 62.88%
MBE: -2.34  PBIAS: -1.23%
Avg Training Time: 2.134s
Avg Inference Time: 0.012345s
Total Retrains: 3
==============================================================
```

## Dependencies

```bash
# Core
pip install pandas numpy requests

# XGBoost
pip install pycaret[time_series] xgboost

# Prophet
pip install prophet

# LSTM
pip install torch scikit-learn
```

## Migration from Legacy

Old code:
```bash
python3 xgboost_whole.py
```

New equivalent:
```bash
python3 forecast_main.py --model xgboost
```

The new system is **fully backward compatible** in functionality but provides:
- ✅ Multiple model support
- ✅ Cleaner architecture
- ✅ Easier to extend
- ✅ Better code organization
