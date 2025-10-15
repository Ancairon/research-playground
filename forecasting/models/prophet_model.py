"""
Prophet forecasting model implementation.
Uses Meta's Prophet for seasonal/periodic pattern prediction.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings

from prophet import Prophet

from models.base_model import BaseTimeSeriesModel


class ProphetModel(BaseTimeSeriesModel):
    """
    Prophet time series forecasting model.
    
    Excellent for data with strong seasonal patterns and trends.
    Does NOT support online learning - requires full retraining.
    """
    
    def __init__(
        self,
        horizon: int = 5,
        random_state: int = 42,
        seasonality_mode: str = 'additive',
        custom_seasonalities: List[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize Prophet model.
        
        Args:
            horizon: Forecast horizon (steps ahead)
            random_state: Random seed
            seasonality_mode: 'additive' or 'multiplicative'
            custom_seasonalities: List of custom seasonality dicts, e.g.:
                [{'name': 'spike_cycle', 'period': 15, 'fourier_order': 5}]
            **kwargs: Additional Prophet parameters
        """
        super().__init__(horizon, random_state)
        self.seasonality_mode = seasonality_mode
        self.custom_seasonalities = custom_seasonalities or []
        self.prophet_kwargs = kwargs
        self.last_timestamp = None
        
    def train(self, data: pd.Series, **kwargs) -> float:
        """Train Prophet model."""
        start_time = time.time()
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        # Suppress Prophet warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create model
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                daily_seasonality=False,
                weekly_seasonality=False,
                yearly_seasonality=False,
                **self.prophet_kwargs
            )
            
            # Add custom seasonalities
            for seasonality in self.custom_seasonalities:
                self.model.add_seasonality(**seasonality)
            
            # Fit model
            self.model.fit(df)
        
        self.last_timestamp = data.index[-1]
        self.is_trained = True
        training_time = time.time() - start_time
        return training_time
    
    def predict(self, **kwargs) -> List[float]:
        """Generate predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        # Create future dataframe
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=self.last_timestamp + pd.Timedelta(seconds=1),
                periods=self.horizon,
                freq='S'
            )
        })
        
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)
        
        return forecast['yhat'].tolist()
    
    def update(self, data: pd.Series, **kwargs):
        """
        Update last timestamp for prediction.
        
        Note: Prophet does NOT support online learning.
        To incorporate new data, you must call train() again.
        """
        self.last_timestamp = data.index[-1]
    
    def get_model_name(self) -> str:
        return "Prophet (Meta)"
    
    def get_model_params(self) -> Dict[str, Any]:
        params = {
            "horizon": self.horizon,
            "seasonality_mode": self.seasonality_mode,
            "random_state": self.random_state,
            "online_learning": False
        }
        if self.custom_seasonalities:
            params["custom_seasonalities"] = self.custom_seasonalities
        return params
    
    def supports_online_learning(self) -> bool:
        return False
