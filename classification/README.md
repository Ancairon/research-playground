# Time Series Classification Module

This module provides time series classification capabilities with automatic detection of new patterns and retraining support.

## Features

- **Interchangeable Models**: Support for Random Forest, XGBoost, and easily extendable to other classifiers
- **Automatic Feature Extraction**: Statistical and shape-based features extracted from time series
- **New Pattern Detection**: Classifies with confidence threshold; low-confidence predictions indicate potential new classes
- **Automatic Retraining**: When new patterns are detected, the model can automatically retrain with the new class
- **Config Integration**: Works with YAML config files from the forecasting module
- **Training Config**: Train from a YAML config file that maps CSVs to class labels

## Training Setup

### 1. Prepare Training Data

Place your training CSV files in the `classification/train/` folder.

### 2. Configure Training Classes

Edit `classification/train/training_config.yaml`:

```yaml
classes:
  temperature_stable:
    - temp_sample1.csv
    - temp_sample2.csv
  disk_usage_growing:
    - disk_sample1.csv
  network_bursty:
    - network_sample1.csv
    - network_sample2.csv
```

Each class can have one or more CSV files as training samples.

### 3. Train the Model

```bash
cd classification
python classify_main.py --train-from-config
```

The trained model is saved automatically to `classification/data/`.

## Usage

### Command Line

```bash
# TRAINING:
# Train from training config (recommended)
python classify_main.py --train-from-config

# Train with custom config path
python classify_main.py --train-from-config --training-config ./my_config.yaml --train-dir ./my_data/

# INFERENCE:
# Classify a single time series from config (uses saved model)
python classify_main.py --config ../configs/192.168.1.123_temp_pi.yaml

# Use a different model type
python classify_main.py --config ../configs/my_config.yaml --model xgboost

# Set custom confidence threshold (default: 60%)
python classify_main.py --config ../configs/my_config.yaml --confidence 0.7

# Specify class name for new patterns
python classify_main.py --config ../configs/my_config.yaml --new-class-name "my_custom_class"

# Show classifier info
python classify_main.py --info

# LEGACY TRAINING (inline labels):
python classify_main.py --train \
    --labels "temp_pattern:../configs/192.168.1.123_temp_pi.yaml,disk_pattern:../configs/wg-disk-smooth10.yaml"
```

### Python API

```python
from classification import TimeSeriesClassifier

# Option 1: Train from config file (recommended)
classifier = TimeSeriesClassifier(model_name='randomforest')
classifier.train_from_config()  # Uses classification/train/training_config.yaml

# Option 2: Load pre-trained model for inference
classifier = TimeSeriesClassifier.load()  # Loads from classification/data/

# Classify a new time series
result = classifier.classify(new_series)
print(f"Class: {result.predicted_class}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Is new pattern: {result.is_new_class}")

# Classify from config and update the config file
result = classifier.classify_and_update_csv(
    config_path='configs/my_config.yaml',
    class_key='class',  # Key to store class in config
    auto_retrain=True,  # Automatically retrain if new pattern
)
```

## Classification Flow

1. **Load Data**: Time series is loaded from CSV (path from config or inferred from config filename)
2. **Extract Features**: Statistical features are extracted from the time series:
   - Basic statistics (mean, std, min, max, median, IQR)
   - Distribution features (skewness, kurtosis)
   - Trend features (slope, R², change %)
   - Volatility features (diff stats, sign changes)
   - Autocorrelation features (lags 1, 5, 10)
   - Complexity features (turning points, entropy)
   - Shape features (half comparisons, peak positions)

3. **Classify**: The model predicts the class and confidence
4. **New Pattern Detection**: If confidence < threshold (default 60%), mark as new class
5. **Retrain (Optional)**: If new pattern detected and `auto_retrain=True`, add as new class and retrain
6. **Update Config**: Write the predicted class to the config file

## Available Models

| Model | Description |
|-------|-------------|
| `randomforest` / `rf` | Random Forest - robust, handles non-linear relationships |
| `xgboost` / `xgb` | XGBoost gradient boosting - high performance |

## Adding New Models

1. Create a new model file in `classification/models/`
2. Inherit from `BaseClassificationModel`
3. Implement required methods: `train`, `predict`, `predict_proba`, `get_model_name`, `get_model_params`
4. Register in `classification/models/__init__.py`

```python
from .base_model import BaseClassificationModel

class MyNewModel(BaseClassificationModel):
    def train(self, X, y, **kwargs):
        # Training logic
        pass
    
    def predict(self, X, **kwargs):
        # Prediction logic
        pass
    
    def predict_proba(self, X, **kwargs):
        # Probability prediction
        pass
    
    def get_model_name(self):
        return "My New Model"
    
    def get_model_params(self):
        return {...}
```

## Data Storage

The classifier stores its data in `classification/data/`:
- `classifier_model.joblib` - Trained model
- `classes.json` - List of known classes
- `training_data.npz` - Training features and labels (for retraining)

## Config File Format

The classifier uses YAML config files compatible with the forecasting module:

```yaml
model: lstm-attention
csv-file: 192.168.1.123_temp_pi.csv  # Optional - inferred from config name
train-window: 3650
horizon: 2000
# ... other forecasting params ...

# Added by classifier:
class: temperature_pattern  # Classification result
```
