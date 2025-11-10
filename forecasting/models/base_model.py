"""
Base interface for time series forecasting models.
All models must implement this interface to work with the universal forecaster.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Dict, Any


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series forecasting models.
    
    Any model (XGBoost, LSTM, etc.) must implement these methods
    to be compatible with the universal forecaster framework.
    """
    
    def __init__(self, horizon: int, random_state: int = 42, **kwargs):
        """
        Initialize the model.
        
        Args:
            horizon: Number of steps ahead to forecast
            random_state: Random seed for reproducibility
            **kwargs: Model-specific parameters
        """
        self.horizon = horizon
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, data: pd.Series, **kwargs) -> float:
        """
        Train (or retrain) the model on the provided data.
        
        Args:
            data: Time series data (pd.Series with datetime index)
            **kwargs: Model-specific training parameters
            
        Returns:
            Training time in seconds
        """
        pass
    
    @abstractmethod
    def predict(self, **kwargs) -> List[float]:
        """
        Generate predictions for the next `horizon` steps.
        
        Args:
            **kwargs: Model-specific prediction parameters
            
        Returns:
            List of predicted values (length = horizon)
        """
        pass
    
    @abstractmethod
    def update(self, data: pd.Series, **kwargs):
        """
        Update the model with new data (online learning if supported).
        
        Some models (XGBoost with PyCaret) support incremental updates.
        Others (LSTM) may need full retraining.
        
        Args:
            data: New time series data
            **kwargs: Model-specific update parameters
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model's display name."""
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Return current model parameters for logging/debugging."""
        pass
    
    def supports_online_learning(self) -> bool:
        """
        Indicate whether the model supports online learning (incremental updates).
        
        Returns:
            True if update() does incremental learning, False if it requires full retrain
        """
        return False
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return feature importance scores if available.
        
        Returns:
            Dict of feature names -> importance scores, or None if not supported
        """
        return None
