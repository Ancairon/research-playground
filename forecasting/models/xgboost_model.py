"""
XGBoost forecasting model implementation.
Wraps PyCaret's XGBoost time series forecaster.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
import io
import contextlib

from pycaret.time_series import TSForecastingExperiment
import xgboost

from models.base_model import BaseTimeSeriesModel

# Patch XGBoost version if missing
if not hasattr(xgboost, "__version__"):
    try:
        import importlib.metadata as importlib_metadata
        xgboost.__version__ = importlib_metadata.version("xgboost")
    except Exception:
        xgboost.__version__ = "1.6.0"


class XGBoostModel(BaseTimeSeriesModel):
    """
    XGBoost time series forecasting model using PyCaret.
    
    Supports online learning through PyCaret's update mechanism.
    """
    
    def __init__(self, horizon: int = 5, random_state: int = 42, **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            horizon: Forecast horizon (steps ahead)
            random_state: Random seed
            **kwargs: Additional PyCaret setup parameters
        """
        super().__init__(horizon, random_state)
        self.exp = None
        self.pipeline = None
        self.setup_kwargs = kwargs
        
    def train(self, data: pd.Series, **kwargs) -> float:
        """Train XGBoost model using PyCaret."""
        start_time = time.time()
        
        # Prepare data
        y = data.copy()
        y.index = pd.PeriodIndex(y.index, freq='S')
        
        # Suppress PyCaret verbose output
        with self._suppress_pycaret():
            self.exp = TSForecastingExperiment()
            self.exp.setup(
                data=y,
                fh=self.horizon,
                session_id=self.random_state,
                numeric_imputation_target="ffill",
                **self.setup_kwargs
            )
            xgb_pipe = self.exp.create_model('xgboost_cds_dt')
            self.pipeline = self.exp.finalize_model(xgb_pipe)
        
        self.is_trained = True
        training_time = time.time() - start_time
        return training_time
    
    def predict(self, **kwargs) -> List[float]:
        """Generate predictions."""
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model must be trained before prediction")
        
        fh = list(range(1, self.horizon + 1))
        fc = self.pipeline.predict(fh=fh)
        fc.index = fc.index.to_timestamp(freq='S')
        
        return fc.values.tolist()
    
    def update(self, data: pd.Series, **kwargs):
        """Update model with new data (online learning)."""
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model must be trained before update")
        
        ts = data.copy()
        ts.index = pd.PeriodIndex(ts.index, freq='S')
        self.pipeline.update(ts)
    
    def get_model_name(self) -> str:
        return "XGBoost (PyCaret)"
    
    def get_model_params(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "random_state": self.random_state,
            "online_learning": True
        }
    
    def supports_online_learning(self) -> bool:
        return True
    
    class _suppress_pycaret:
        """Context manager to suppress PyCaret verbose output."""
        def __init__(self, logger_name: str = 'pycaret'):
            self.logger_name = logger_name
            self._old_level = None
            self._stream = io.StringIO()
        
        def __enter__(self):
            logger = logging.getLogger(self.logger_name)
            self._old_level = logger.level
            logger.setLevel(logging.ERROR)
            self._out = contextlib.redirect_stdout(self._stream)
            self._err = contextlib.redirect_stderr(self._stream)
            self._out.__enter__()
            self._err.__enter__()
        
        def __exit__(self, exc_type, exc, tb):
            self._err.__exit__(exc_type, exc, tb)
            self._out.__exit__(exc_type, exc, tb)
            logging.getLogger(self.logger_name).setLevel(self._old_level)
