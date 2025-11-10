"""
Model factory - creates forecasting models by name.
"""

from typing import Dict, Any, Type
from models.base_model import BaseTimeSeriesModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.random_forest_model import RandomForestModel
from models.extra_trees_model import ExtraTreesModel
from models.gru_model import GRUModel
from models.lstm_attention_model import LSTMAttentionModel
from models.nbeats_model import NBEATSModel
from models.tft_model import TFTModel


# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseTimeSeriesModel]] = {
    'xgboost': XGBoostModel,
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
