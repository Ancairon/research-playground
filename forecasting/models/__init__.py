"""
Model factory - creates forecasting models by name.
"""

from typing import Dict, Any, Type
from .base_model import BaseTimeSeriesModel

# Core models (always available)
from .lstm_model import LSTMModel
from .random_forest_model import RandomForestModel
from .extra_trees_model import ExtraTreesModel
from .gru_model import GRUModel
from .lstm_attention_model import LSTMAttentionModel
from .nbeats_model import NBEATSModel
from .tft_model import TFTModel

# Optional models (may have dependency issues)
try:
    from .xgboost_model import XGBoostModel
    _XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: XGBoostModel not available (pycaret import failed): {e}")
    XGBoostModel = None
    _XGBOOST_AVAILABLE = False

try:
    from .timesfm_model import TimesFMModel
    _TIMESFM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TimesFMModel not available: {e}")
    TimesFMModel = None
    _TIMESFM_AVAILABLE = False


# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseTimeSeriesModel]] = {
    'lstm': LSTMModel,
    'randomforest': RandomForestModel,
    'rf': RandomForestModel,  # Alias
    'extratrees': ExtraTreesModel,
    'et': ExtraTreesModel,  # Alias
    'gru': GRUModel,
    'lstm-attention': LSTMAttentionModel,
    'lstm-attn': LSTMAttentionModel,  # Alias
    'nbeats': NBEATSModel,
    'tft': TFTModel,
}

# Add optional models if available
if _XGBOOST_AVAILABLE:
    MODEL_REGISTRY['xgboost'] = XGBoostModel

if _TIMESFM_AVAILABLE:
    MODEL_REGISTRY['timesfm'] = TimesFMModel
    MODEL_REGISTRY['times-fm'] = TimesFMModel


def create_model(model_name: str, **kwargs) -> BaseTimeSeriesModel:
    """
    Create a forecasting model by name.
    
    Args:
        model_name: Name of the model ('xgboost', 'lstm')
        **kwargs: Model-specific parameters
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_name_lower]
    return model_class(**kwargs)


def list_available_models() -> list:
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dict with model class and docstring
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        return None
    
    model_class = MODEL_REGISTRY[model_name_lower]
    return {
        'name': model_name_lower,
        'class': model_class.__name__,
        'description': model_class.__doc__.strip() if model_class.__doc__ else "No description"
    }
