"""
Random Forest forecasting model implementation.
Uses scikit-learn's RandomForestRegressor for time series prediction.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings

from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseTimeSeriesModel


class RandomForestModel(BaseTimeSeriesModel):
    """
    Random Forest time series forecasting model.
    
    Creates lag features and uses Random Forest for prediction.
    Good for capturing non-linear patterns with ensemble learning.
    Does NOT support online learning - requires full retraining.
    """
    
    def __init__(
        self,
        horizon: int = 5,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Initialize Random Forest model.
        
        Args:
            horizon: Forecast horizon (steps ahead)
            random_state: Random seed
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            max_features: Number of features to consider for splits
            n_jobs: Number of parallel jobs (-1 = all CPUs)
            **kwargs: Additional RandomForestRegressor parameters
        """
        super().__init__(horizon, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.rf_kwargs = kwargs
        
        # Will be set during training
        self.lag_window = None
        self.last_values = None
        
    def _create_lag_features(self, data: pd.Series) -> tuple:
        """
        Create lag features for training.
        
        Returns:
            (X, y) tuple of features and targets
        """
        # Determine lag window based on data size
        # Use at least horizon * 2, but cap at 100
        self.lag_window = min(max(self.horizon * 2, 20), 100, len(data) // 3)
        
        # Create lagged features
        features = []
        targets = []
        
        values = data.values
        for i in range(self.lag_window + self.horizon - 1, len(values)):
            # Features: lag_window points before the target
            X_row = values[i - self.lag_window - self.horizon + 1 : i - self.horizon + 1]
            # Target: value at horizon steps ahead
            y_val = values[i]
            
            features.append(X_row)
            targets.append(y_val)
        
        return np.array(features), np.array(targets)
    
    def train(self, data: pd.Series, **kwargs) -> float:
        """Train Random Forest model."""
        start_time = time.time()
        
        # Validate data size
        min_required = self.horizon * 3
        if len(data) < min_required:
            raise ValueError(
                f"Insufficient data for training. Need at least {min_required} points, "
                f"got {len(data)}"
            )
        
        # Create lag features
        X, y = self._create_lag_features(data)
        
        if len(X) == 0:
            raise ValueError("No training samples created. Data may be too short.")
        
        # Create and train model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.rf_kwargs
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        
        # Store last values for prediction
        self.last_values = data.values[-self.lag_window:]
        
        self.is_trained = True
        training_time = time.time() - start_time
        return training_time
    
    def predict(self, **kwargs) -> List[float]:
        """Generate predictions."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")
        
        if self.last_values is None or len(self.last_values) < self.lag_window:
            raise RuntimeError(
                f"Insufficient data for prediction. Need {self.lag_window} points, "
                f"got {len(self.last_values) if self.last_values is not None else 0}"
            )
        
        predictions = []
        current_values = self.last_values.copy()
        
        # Iteratively predict each step in the horizon
        for _ in range(self.horizon):
            # Take last lag_window values as features
            X = current_values[-self.lag_window:].reshape(1, -1)
            
            # Predict next value
            pred = self.model.predict(X)[0]
            predictions.append(pred)
            
            # Append prediction to current values for next iteration
            current_values = np.append(current_values, pred)
        
        return predictions
    
    def update(self, data: pd.Series, **kwargs):
        """
        Update last values window for prediction.
        
        Note: Random Forest does NOT support online learning.
        To incorporate new patterns, you must call train() again.
        """
        if len(data) > 0:
            # Update the last values window
            new_values = data.values
            if self.last_values is None:
                self.last_values = new_values[-self.lag_window:]
            else:
                # Concatenate and keep last lag_window values
                self.last_values = np.concatenate([self.last_values, new_values])[-self.lag_window:]
    
    def get_model_name(self) -> str:
        return "Random Forest"
    
    def get_model_params(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "lag_window": self.lag_window,
            "random_state": self.random_state,
            "online_learning": False
        }
    
    def supports_online_learning(self) -> bool:
        return False
