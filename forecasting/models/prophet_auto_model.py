"""
Prophet with automatic seasonality detection.
Uses FFT (Fast Fourier Transform) to detect dominant frequencies.
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings
from scipy import signal
from scipy.fft import fft, fftfreq

from prophet import Prophet

from models.base_model import BaseTimeSeriesModel


def detect_seasonalities(data: pd.Series, top_n: int = 3, min_period: int = 5, max_period: int = 300) -> List[Dict[str, Any]]:
    """
    Auto-detect seasonalities using FFT (Fourier Transform).
    
    Args:
        data: Time series data
        top_n: Number of top seasonalities to detect
        min_period: Minimum period to consider (seconds)
        max_period: Maximum period to consider (seconds)
        
    Returns:
        List of detected seasonalities with period and fourier_order
    """
    # Remove trend (detrend)
    values = data.values
    detrended = signal.detrend(values)
    
    # Compute FFT
    n = len(detrended)
    fft_vals = fft(detrended)
    fft_freq = fftfreq(n, d=1.0)  # Assuming 1-second sampling
    
    # Get power spectrum (magnitude)
    power = np.abs(fft_vals[:n//2])  # Only positive frequencies
    freqs = fft_freq[:n//2]
    
    # Filter by period range
    periods = 1.0 / freqs[1:]  # Skip DC component (freq=0)
    power = power[1:]
    
    valid_mask = (periods >= min_period) & (periods <= max_period)
    valid_periods = periods[valid_mask]
    valid_power = power[valid_mask]
    
    if len(valid_periods) == 0:
        return []
    
    # Find top N peaks
    top_indices = np.argsort(valid_power)[-top_n:][::-1]
    
    detected = []
    for i, idx in enumerate(top_indices):
        period = valid_periods[idx]
        strength = valid_power[idx]
        
        # Determine fourier_order based on signal strength and period
        # Shorter periods or stronger signals get higher fourier_order
        if strength > np.mean(valid_power) * 5:
            fourier_order = 7
        elif strength > np.mean(valid_power) * 3:
            fourier_order = 5
        elif strength > np.mean(valid_power) * 2:
            fourier_order = 3
        else:
            fourier_order = 2
        
        detected.append({
            'name': f'auto_seasonality_{i+1}',
            'period': float(period),
            'fourier_order': fourier_order,
            'strength': float(strength)
        })
    
    return detected


class ProphetAutoModel(BaseTimeSeriesModel):
    """
    Prophet with automatic seasonality detection.
    
    Analyzes training data to detect periodic patterns automatically.
    Good for unknown/changing patterns.
    """
    
    def __init__(
        self,
        horizon: int = 5,
        random_state: int = 42,
        seasonality_mode: str = 'additive',
        auto_detect: bool = True,
        top_seasonalities: int = 2,
        min_period: int = 10,
        max_period: int = 300,
        **kwargs
    ):
        """
        Initialize Prophet with auto-detection.
        
        Args:
            horizon: Forecast horizon
            random_state: Random seed
            seasonality_mode: 'additive' or 'multiplicative'
            auto_detect: Enable automatic seasonality detection
            top_seasonalities: Number of top patterns to detect
            min_period: Minimum period to look for (seconds)
            max_period: Maximum period to look for (seconds)
            **kwargs: Additional Prophet parameters
        """
        super().__init__(horizon, random_state)
        self.seasonality_mode = seasonality_mode
        self.auto_detect = auto_detect
        self.top_seasonalities = top_seasonalities
        self.min_period = min_period
        self.max_period = max_period
        self.prophet_kwargs = kwargs
        self.last_timestamp = None
        self.detected_seasonalities = []
        
    def train(self, data: pd.Series, **kwargs) -> float:
        """Train Prophet with auto-detected seasonalities."""
        start_time = time.time()
        
        # Detect seasonalities if enabled
        if self.auto_detect:
            self.detected_seasonalities = detect_seasonalities(
                data,
                top_n=self.top_seasonalities,
                min_period=self.min_period,
                max_period=self.max_period
            )
        
        # Prepare data
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        # Suppress warnings
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
            
            # Add detected seasonalities
            for seasonality in self.detected_seasonalities:
                self.model.add_seasonality(
                    name=seasonality['name'],
                    period=seasonality['period'],
                    fourier_order=seasonality['fourier_order']
                )
            
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
        
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=self.last_timestamp + pd.Timedelta(seconds=1),
                periods=self.horizon,
                freq='S'
            )
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future)
        
        return forecast['yhat'].tolist()
    
    def update(self, data: pd.Series, **kwargs):
        """Update last timestamp."""
        self.last_timestamp = data.index[-1]
    
    def get_model_name(self) -> str:
        if self.auto_detect and self.detected_seasonalities:
            periods = [f"{s['period']:.1f}s" for s in self.detected_seasonalities]
            return f"Prophet (Auto: {', '.join(periods)})"
        return "Prophet (Auto)"
    
    def get_model_params(self) -> Dict[str, Any]:
        params = {
            "horizon": self.horizon,
            "seasonality_mode": self.seasonality_mode,
            "auto_detect": self.auto_detect,
            "random_state": self.random_state,
            "online_learning": False
        }
        if self.detected_seasonalities:
            params["detected_seasonalities"] = self.detected_seasonalities
        return params
    
    def supports_online_learning(self) -> bool:
        return False
