"""
Model factory - creates classification models by name.
"""

from typing import Dict, Any, Type, List

from .base_model import BaseClassificationModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel


# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseClassificationModel]] = {
    'randomforest': RandomForestModel,
    'rf': RandomForestModel,  # Alias
    'xgboost': XGBoostModel,
    'xgb': XGBoostModel,  # Alias
}


# Model information for help/documentation
MODEL_INFO: Dict[str, Dict[str, str]] = {
    'randomforest': {
        'class': 'RandomForestModel',
        'description': 'Random Forest ensemble classifier - robust, handles non-linear relationships well',
    },
    'xgboost': {
        'class': 'XGBoostModel',
        'description': 'XGBoost gradient boosting - high performance, handles imbalanced data',
    },
}


def create_model(model_name: str, **kwargs) -> BaseClassificationModel:
    """
    Create a classification model by name.
    
    Args:
        model_name: Name of the model ('randomforest', 'rf', 'xgboost', 'xgb')
        **kwargs: Model-specific parameters
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        available = ', '.join(sorted(set(MODEL_REGISTRY.keys())))
        raise ValueError(f"Unknown model: {model_name}. Available models: {available}")
    
    model_class = MODEL_REGISTRY[model_name_lower]
    return model_class(**kwargs)


def list_available_models() -> List[str]:
    """
    List all available model names (excluding aliases).
    
    Returns:
        List of model names
    """
    # Return unique models (without aliases)
    unique_models = []
    seen_classes = set()
    for name, cls in MODEL_REGISTRY.items():
        if cls not in seen_classes:
            unique_models.append(name)
            seen_classes.add(cls)
    return sorted(unique_models)


def get_model_info(model_name: str) -> Dict[str, str]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dict with 'class' and 'description' keys
    """
    model_name_lower = model_name.lower()
    
    # Handle aliases
    if model_name_lower == 'rf':
        model_name_lower = 'randomforest'
    elif model_name_lower == 'xgb':
        model_name_lower = 'xgboost'
    
    if model_name_lower not in MODEL_INFO:
        return {'class': 'Unknown', 'description': 'No information available'}
    
    return MODEL_INFO[model_name_lower]
