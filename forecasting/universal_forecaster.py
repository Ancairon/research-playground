"""
Universal forecaster - model-agnostic forecasting framework.

Handles:
- Prediction smoothing
- Deferred validation
- Error tracking (Symmetric MAPE with capping)
- Dynamic retraining (MAD-based thresholds)
- Metrics logging

Error Calculation:
- Uses Symmetric MAPE: (|actual - pred| / ((|actual| + |pred|) / 2)) * 100
- More robust to near-zero values than standard MAPE
- Errors capped at 1000% to prevent overflow from extreme outliers
- Skips validation points where both actual and predicted are near zero

Works with any model implementing BaseTimeSeriesModel interface.
"""

import time
import pandas as pd
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Any

from models.base_model import BaseTimeSeriesModel


def compute_threshold_from_errors(
    errors: List[float],
    retrain_scale: float,
    retrain_min: float,
    use_mad: bool = True
) -> float:
    """
    Compute retrain threshold from error history.
    
    Args:
        errors: List of historical errors (percentage values for MAPE)
        retrain_scale: Multiplier for MAD/std
        retrain_min: Minimum threshold value
        use_mad: Use MAD (robust) vs std (sensitive to outliers)
        
    Returns:
        Threshold value (same units as errors)
    """
    arr = np.asarray(errors, dtype=float)
    if arr.size == 0:
        return float(retrain_min)
    
    if use_mad:
        med = float(np.nanmedian(arr))
        mad = float(np.nanmedian(np.abs(arr - med)))
        sigma = 1.4826 * mad
    else:
        sigma = float(np.nanstd(arr))
    
    thresh = max(float(retrain_min), float(retrain_scale) * abs(sigma))
    return thresh


class UniversalForecaster:
    """
    Universal forecasting engine.
    
    Model-agnostic - works with any BaseTimeSeriesModel implementation.
    """
    
    def __init__(
        self,
        model: BaseTimeSeriesModel,
        window: int = 30,
        train_window: int = 300,
        prediction_smoothing: int = 3,
        dynamic_retrain: bool = True,
        retrain_scale: float = 3.0,
        retrain_min: float = 50.0,
        retrain_use_mad: bool = True,
        retrain_consec: int = 2,
        retrain_cooldown: int = 5,
        print_min_validations: int = 3,
        quiet: bool = False
    ):
        """
        Initialize universal forecaster.
        
        Args:
            model: Forecasting model (must implement BaseTimeSeriesModel)
            window: Inference window size (historical points for predictions)
            train_window: Initial training window size (defaults to window if None)
            prediction_smoothing: Number of predictions to average (1=no smoothing)
            dynamic_retrain: Enable dynamic threshold-based retraining
            retrain_scale: Multiplier for MAD/std threshold calculation
            retrain_min: Minimum retraining threshold
            retrain_use_mad: Use MAD (robust) vs std
            retrain_consec: Consecutive violations to trigger retrain
            retrain_cooldown: Minimum steps between retrains
            print_min_validations: Min validations before printing
            quiet: Suppress console output
        """
        self.model = model
        self.window = window
        self.train_window = train_window if train_window is not None else window
        self.prediction_smoothing = prediction_smoothing
        self.dynamic_retrain = dynamic_retrain
        self.retrain_scale = retrain_scale
        self.retrain_min = retrain_min
        self.retrain_use_mad = retrain_use_mad
        self.retrain_consec = retrain_consec
        self.retrain_cooldown = retrain_cooldown
        self.print_min_validations = print_min_validations
        self.quiet = quiet
        
        # State tracking
        self.step = 0
        self.last_retrain_step = -9999
        self.consec_count = 0
        
        # Error tracking
        self.errors = deque(maxlen=10)  # Rolling window for display
        self.recent_errors = deque(maxlen=max(50, retrain_consec * 5))  # For threshold computation
        self.result_avg_err = []
        self.signed_errors = []
        self.y_true_list = []
        self.y_pred_list = []
        
        # Prediction tracking
        self.pending_validations = deque(maxlen=100)
        self.recent_predictions = deque(maxlen=max(1, int(prediction_smoothing)))
        
        # Timing tracking
        self.training_times = []
        self.inference_times = []
    
    def train_initial(self, data: pd.Series) -> float:
        """
        Train model on initial data.
        
        Args:
            data: Initial training data
            
        Returns:
            Training time in seconds
        """
        train_time = self.model.train(data)
        self.training_times.append(train_time)
        
        if not self.quiet:
            print(f"[{self.model.get_model_name()}] Initial training completed in {train_time:.3f}s")
        
        return train_time
    
    def forecast_step(self, live_data: pd.Series) -> List[float]:
        """
        Perform one forecasting step.
        
        Args:
            live_data: Current live data window
            
        Returns:
            Smoothed predictions (list of floats)
        """
        # Update model with new data (online learning if supported)
        if self.model.supports_online_learning():
            self.model.update(live_data)
        
        # Generate prediction
        start_inf = time.time()
        raw_preds = self.model.predict()
        inf_time = time.time() - start_inf
        self.inference_times.append(inf_time)
        
        # Validate predictions (check for NaN, inf, or extreme values)
        if raw_preds:
            raw_preds = [
                max(-1e9, min(1e9, float(p))) if np.isfinite(p) else 0.0
                for p in raw_preds
            ]
        
        # Apply prediction smoothing
        if raw_preds:
            raw_pred_values = [float(p) for p in raw_preds]
            self.recent_predictions.append(raw_pred_values)
            
            # Compute smoothed predictions
            if len(self.recent_predictions) >= 1 and self.prediction_smoothing > 1:
                smoothed = []
                for i in range(self.model.horizon):
                    values_at_step_i = [
                        pred_list[i] for pred_list in self.recent_predictions
                        if i < len(pred_list)
                    ]
                    if values_at_step_i:
                        smoothed.append(float(np.mean(values_at_step_i)))
                    else:
                        smoothed.append(raw_pred_values[i] if i < len(raw_pred_values) else 0.0)
                preds = smoothed
            else:
                preds = raw_pred_values
        else:
            preds = []
        
        # Store for validation
        if preds:
            last_actual_ts = live_data.index[-1]
            pred_values = [float(p) for p in preds]
            self.pending_validations.append((self.step, last_actual_ts, pred_values))
        
        self.step += 1
        return preds
    
    def validate_predictions(self, live_data: pd.Series) -> Optional[float]:
        """
        Validate oldest pending prediction if ready.
        
        Args:
            live_data: Current live data
            
        Returns:
            Mean horizon error if validation occurred, None otherwise
        """
        if not self.pending_validations:
            return None
        
        creation_step, last_actual_ts, pred_values = self.pending_validations[0]
        steps_elapsed = self.step - creation_step
        
        if steps_elapsed < self.model.horizon:
            return None
        
        # Remove from queue
        self.pending_validations.popleft()
        
        # Validate all horizon steps
        horizon_errors = []
        horizon_actuals = []
        horizon_preds = []
        
        for i, pred_val in enumerate(pred_values):
            offset_from_now = steps_elapsed - (i + 1)
            
            # Find actual value
            try:
                current_ts = live_data.index[-1]
                target_ts = current_ts - pd.Timedelta(seconds=offset_from_now)
                target_ts_floor = target_ts.floor('S')
                
                idx_secs = live_data.index.floor('S')
                matches = live_data.loc[idx_secs == target_ts_floor]
                if len(matches) > 0:
                    y_true_now = float(matches.iloc[-1])
                else:
                    continue
            except Exception:
                continue
            
            # Calculate error with safeguards
            # Use symmetric MAPE to handle near-zero values better
            # Support negative metrics by using absolute values only in denominator
            err_abs = abs(y_true_now - float(pred_val))
            denominator = (abs(y_true_now) + abs(float(pred_val))) / 2.0
            
            # If both values are near zero, skip this validation point
            if denominator < 1e-6:
                continue
            
            # Symmetric MAPE - works with negative metrics
            # Denominator uses abs() to ensure positive scaling, numerator is absolute error
            err_relative = (err_abs / denominator) * 100.0
            
            # Cap extreme errors at 1000% to prevent overflow
            # (errors > 1000% indicate catastrophic prediction failure anyway)
            err_relative = min(err_relative, 1000.0)
            
            horizon_errors.append(err_relative)
            horizon_actuals.append(y_true_now)
            horizon_preds.append(pred_val)
        
        # Only process if validated full horizon
        if len(horizon_errors) != self.model.horizon:
            return None
        
        mean_horizon_error = float(np.mean(horizon_errors))
        
        # Debug: warn about extreme errors
        if mean_horizon_error > 500.0 and not self.quiet:
            print(
                f"[WARNING] Extreme error detected: {mean_horizon_error:.2f}% "
                f"(actual={np.mean(horizon_actuals):.2f}, pred={np.mean(horizon_preds):.2f})"
            )
        
        self.errors.append(mean_horizon_error)
        self.recent_errors.append(mean_horizon_error)
        
        # Compute threshold
        threshold = None
        if self.dynamic_retrain:
            try:
                threshold = compute_threshold_from_errors(
                    self.recent_errors, self.retrain_scale, self.retrain_min, self.retrain_use_mad
                )
            except Exception:
                threshold = float(self.retrain_min)
        
        # Print validation details
        if not self.quiet:
            try:
                ts_str = live_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ts_str = ''
            thr_str = f"{threshold:.2f}%" if threshold is not None else "N/A"
            print(
                f"[{ts_str}] VALIDATION step={self.step} mean_horizon_error={mean_horizon_error:.2f}% "
                f"threshold={thr_str} (retrain_min={self.retrain_min})"
            )
        
        # Store for final statistics
        for actual, pred in zip(horizon_actuals, horizon_preds):
            self.signed_errors.append(actual - pred)
            self.y_true_list.append(actual)
            self.y_pred_list.append(pred)
        
        # Store rolling average
        if len(self.errors) >= self.print_min_validations:
            avg_err = float(np.mean(self.errors))
            self.result_avg_err.append(avg_err)
        
        # Check for retrain
        if threshold is not None:
            self._check_retrain(live_data, mean_horizon_error, threshold, creation_step)
        
        return mean_horizon_error
    
    def _check_retrain(self, live_data: pd.Series, error: float, threshold: float, pred_step: int):
        """Check if retraining should be triggered."""
        # Update consecutive counter
        if error > threshold:
            self.consec_count += 1
        else:
            self.consec_count = 0
        
        # Check retrain conditions
        can_retrain = (
            self.consec_count >= self.retrain_consec and
            (self.step - self.last_retrain_step) >= self.retrain_cooldown and
            len(self.recent_errors) >= self.retrain_consec
        )
        
        # Perform retrain
        if can_retrain:
            try:
                retrain_time = self.model.train(live_data)
                self.training_times.append(retrain_time)
                self.last_retrain_step = self.step
                self.consec_count = 0
                
                if not self.quiet:
                    print(
                        f"[{self.model.get_model_name()}] Retrained at step {self.step} "
                        f"(time={retrain_time:.3f}s), threshold={threshold:.1f}%"
                    )
            except Exception as e:
                if not self.quiet:
                    print(f"[{self.model.get_model_name()}] Retrain failed: {e}")
            finally:
                self.errors.clear()
                self.recent_errors.clear()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get final forecasting statistics."""
        def safe_mean(x):
            return float(np.mean(x)) if len(x) else float('nan')
        
        def safe_percentile(x, q):
            return float(np.percentile(x, q)) if len(x) else float('nan')
        
        y_true_arr = np.array(self.y_true_list, dtype=float)
        y_pred_arr = np.array(self.y_pred_list, dtype=float)
        
        mbe = float(np.mean(y_true_arr - y_pred_arr)) if len(y_true_arr) else float('nan')
        pbias = (
            float(100.0 * np.sum(y_true_arr - y_pred_arr) / np.sum(y_true_arr))
            if len(y_true_arr) and np.sum(y_true_arr) != 0 else float('nan')
        )
        
        return {
            "mean_avg_err": safe_mean(self.result_avg_err),
            "max_avg_err": float(np.max(self.result_avg_err)) if len(self.result_avg_err) else float('nan'),
            "min_avg_err": float(np.min(self.result_avg_err)) if len(self.result_avg_err) else float('nan'),
            "p80_avg_err": safe_percentile(self.result_avg_err, 80),
            "p95_avg_err": safe_percentile(self.result_avg_err, 95),
            "p99_avg_err": safe_percentile(self.result_avg_err, 99),
            "mbe": mbe,
            "pbias_pct": pbias,
            "avg_training_time": safe_mean(self.training_times),
            "avg_inference_time": safe_mean(self.inference_times),
            "total_retrains": len(self.training_times) - 1  # Exclude initial train
        }
