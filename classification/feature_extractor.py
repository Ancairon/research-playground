"""
Feature Extractor for Time Series Classification

Extracts statistical and shape-based features from time series data
for use in classification models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats


class TimeSeriesFeatureExtractor:
    """
    Extracts features from time series data for classification.
    
    Features include:
    - Statistical measures (mean, std, min, max, etc.)
    - Shape features (trend, seasonality indicators)
    - Distribution features (skewness, kurtosis)
    - Complexity features (entropy, autocorrelation)
    """
    
    def __init__(self, include_advanced: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            include_advanced: Whether to include computationally expensive features
        """
        self.include_advanced = include_advanced
        self._feature_names: Optional[List[str]] = None
    
    @property
    def feature_names(self) -> List[str]:
        """Get the list of feature names."""
        if self._feature_names is None:
            # Generate a dummy extraction to get feature names
            dummy = pd.Series(np.random.randn(100))
            features = self.extract(dummy)
            self._feature_names = list(features.keys())
        return self._feature_names
    
    def extract(self, data: pd.Series) -> Dict[str, float]:
        """
        Extract all features from a time series.
        
        Args:
            data: Time series data (pd.Series)
            
        Returns:
            Dictionary mapping feature names to values
        """
        values = data.values.astype(float)
        features = {}
        
        # Basic statistics
        features.update(self._extract_basic_stats(values))
        
        # Distribution features
        features.update(self._extract_distribution_features(values))
        
        # Trend features
        features.update(self._extract_trend_features(values))
        
        # Volatility features
        features.update(self._extract_volatility_features(values))
        
        # Autocorrelation features
        features.update(self._extract_autocorrelation_features(values))
        
        if self.include_advanced:
            # Complexity features
            features.update(self._extract_complexity_features(values))
            
            # Shape features
            features.update(self._extract_shape_features(values))
        
        return features
    
    def _extract_basic_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features."""
        features = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'range': float(np.max(values) - np.min(values)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'length': float(len(values)),
        }
        
        # Coefficient of variation (handle zero mean)
        if features['mean'] != 0:
            features['cv'] = features['std'] / abs(features['mean'])
        else:
            features['cv'] = 0.0
        
        return features
    
    def _extract_distribution_features(self, values: np.ndarray) -> Dict[str, float]:
        """Extract distribution-based features."""
        features = {}
        
        # Skewness and kurtosis
        if len(values) > 2:
            try:
                features['skewness'] = float(stats.skew(values))
                features['kurtosis'] = float(stats.kurtosis(values))
            except (ValueError, FloatingPointError):
                features['skewness'] = 0.0
                features['kurtosis'] = 0.0
        else:
            features['skewness'] = 0.0
            features['kurtosis'] = 0.0
        
        # Percentage above/below mean
        mean = np.mean(values)
        features['pct_above_mean'] = float(np.sum(values > mean) / len(values))
        features['pct_below_mean'] = float(np.sum(values < mean) / len(values))
        
        return features
    
    def _extract_trend_features(self, values: np.ndarray) -> Dict[str, float]:
        """Extract trend-related features."""
        features = {}
        n = len(values)
        
        # Linear trend (slope and intercept)
        if n > 1:
            x = np.arange(n)
            try:
                slope, intercept, r_value, _, _ = stats.linregress(x, values)
                features['trend_slope'] = float(slope)
                features['trend_intercept'] = float(intercept)
                features['trend_r_squared'] = float(r_value ** 2)
            except (ValueError, FloatingPointError):
                features['trend_slope'] = 0.0
                features['trend_intercept'] = float(values[0]) if n > 0 else 0.0
                features['trend_r_squared'] = 0.0
        else:
            features['trend_slope'] = 0.0
            features['trend_intercept'] = float(values[0]) if n > 0 else 0.0
            features['trend_r_squared'] = 0.0
        
        # First and last value comparison
        if n > 1:
            features['first_value'] = float(values[0])
            features['last_value'] = float(values[-1])
            features['change_total'] = float(values[-1] - values[0])
            features['change_pct'] = float(
                (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else 0
            )
        else:
            features['first_value'] = float(values[0]) if n > 0 else 0.0
            features['last_value'] = float(values[0]) if n > 0 else 0.0
            features['change_total'] = 0.0
            features['change_pct'] = 0.0
        
        return features
    
    def _extract_volatility_features(self, values: np.ndarray) -> Dict[str, float]:
        """Extract volatility-related features."""
        features = {}
        n = len(values)
        
        # Returns/differences
        if n > 1:
            diffs = np.diff(values)
            features['diff_mean'] = float(np.mean(diffs))
            features['diff_std'] = float(np.std(diffs))
            features['diff_max'] = float(np.max(np.abs(diffs)))
            
            # Count of sign changes
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            features['sign_changes'] = float(sign_changes)
            features['sign_change_rate'] = float(sign_changes / (n - 2)) if n > 2 else 0.0
        else:
            features['diff_mean'] = 0.0
            features['diff_std'] = 0.0
            features['diff_max'] = 0.0
            features['sign_changes'] = 0.0
            features['sign_change_rate'] = 0.0
        
        # Rolling volatility (std of rolling window)
        if n >= 10:
            rolling_std = pd.Series(values).rolling(window=min(10, n//5)).std()
            features['rolling_std_mean'] = float(rolling_std.mean())
            features['rolling_std_std'] = float(rolling_std.std())
        else:
            features['rolling_std_mean'] = features.get('std', float(np.std(values)))
            features['rolling_std_std'] = 0.0
        
        return features
    
    def _extract_autocorrelation_features(self, values: np.ndarray) -> Dict[str, float]:
        """Extract autocorrelation features."""
        features = {}
        n = len(values)
        
        # Autocorrelation at different lags
        lags = [1, 5, 10]
        for lag in lags:
            if n > lag + 1:
                try:
                    ac = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    features[f'autocorr_lag{lag}'] = float(ac) if not np.isnan(ac) else 0.0
                except (ValueError, FloatingPointError):
                    features[f'autocorr_lag{lag}'] = 0.0
            else:
                features[f'autocorr_lag{lag}'] = 0.0
        
        return features
    
    def _extract_complexity_features(self, values: np.ndarray) -> Dict[str, float]:
        """Extract complexity-based features."""
        features = {}
        n = len(values)
        
        # Approximate entropy (simplified)
        if n > 2:
            # Number of local maxima/minima
            if n > 2:
                diffs = np.diff(values)
                local_max = np.sum((diffs[:-1] > 0) & (diffs[1:] < 0))
                local_min = np.sum((diffs[:-1] < 0) & (diffs[1:] > 0))
                features['n_local_max'] = float(local_max)
                features['n_local_min'] = float(local_min)
                features['n_turning_points'] = float(local_max + local_min)
                features['turning_point_rate'] = float((local_max + local_min) / (n - 2))
            else:
                features['n_local_max'] = 0.0
                features['n_local_min'] = 0.0
                features['n_turning_points'] = 0.0
                features['turning_point_rate'] = 0.0
        else:
            features['n_local_max'] = 0.0
            features['n_local_min'] = 0.0
            features['n_turning_points'] = 0.0
            features['turning_point_rate'] = 0.0
        
        # Binned entropy
        if n > 5:
            try:
                hist, _ = np.histogram(values, bins=min(10, n // 5))
                hist = hist / hist.sum()  # Normalize
                hist = hist[hist > 0]  # Remove zeros
                features['binned_entropy'] = float(-np.sum(hist * np.log2(hist)))
            except (ValueError, FloatingPointError):
                features['binned_entropy'] = 0.0
        else:
            features['binned_entropy'] = 0.0
        
        return features
    
    def _extract_shape_features(self, values: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features."""
        features = {}
        n = len(values)
        
        # Split into segments and compare
        if n >= 4:
            mid = n // 2
            first_half = values[:mid]
            second_half = values[mid:]
            
            features['mean_first_half'] = float(np.mean(first_half))
            features['mean_second_half'] = float(np.mean(second_half))
            features['std_first_half'] = float(np.std(first_half))
            features['std_second_half'] = float(np.std(second_half))
            
            # Change in behavior
            features['mean_shift'] = features['mean_second_half'] - features['mean_first_half']
            features['std_ratio'] = (
                features['std_second_half'] / features['std_first_half']
                if features['std_first_half'] > 0 else 1.0
            )
        else:
            features['mean_first_half'] = float(np.mean(values))
            features['mean_second_half'] = float(np.mean(values))
            features['std_first_half'] = float(np.std(values))
            features['std_second_half'] = float(np.std(values))
            features['mean_shift'] = 0.0
            features['std_ratio'] = 1.0
        
        # Peak features
        if n > 0:
            peak_idx = np.argmax(values)
            trough_idx = np.argmin(values)
            features['peak_position'] = float(peak_idx / n)
            features['trough_position'] = float(trough_idx / n)
        else:
            features['peak_position'] = 0.5
            features['trough_position'] = 0.5
        
        return features
    
    def extract_batch(self, data_list: List[pd.Series]) -> np.ndarray:
        """
        Extract features from multiple time series.
        
        Args:
            data_list: List of time series data
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features_list = []
        for data in data_list:
            features = self.extract(data)
            features_list.append(list(features.values()))
        
        return np.array(features_list)
