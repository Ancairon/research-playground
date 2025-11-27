"""
Time Series Classification Module

Classifies time series data into known categories or identifies new patterns.
"""

from .models import create_model, list_available_models, get_model_info
from .classifier import TimeSeriesClassifier

__all__ = [
    'create_model',
    'list_available_models', 
    'get_model_info',
    'TimeSeriesClassifier',
]
