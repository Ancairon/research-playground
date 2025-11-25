"""
Shared prediction and evaluation logic.

This module contains common code for making predictions and evaluating them,
ensuring that forecast_main.py and tune_lstm_attention.py use identical logic.

Key principle: When enough data exists, predictions should be made on the
LAST horizon points of the CSV, with train_window growing backward.
This ensures consistent evaluation across different configurations.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


def single_shot_evaluation(
    forecaster,
    data: pd.Series,
    train_window: int,
    lookback: int,
    horizon: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform a complete single-shot evaluation: train, predict, and evaluate.
    
    IMPORTANT: This function uses the END of the data for predictions to ensure
    that different train_window sizes (from different lookback values) all predict
    the same final horizon points. The train_window grows BACKWARD into history.
    
    Data layout:
    [----------history----------][train_window][horizon]
                                 ^train here   ^predict here (END)
    
    Args:
        forecaster: UniversalForecaster instance with a model
        data: Full time series data (pandas Series)
        train_window: Size of training window
        lookback: Number of past timesteps for prediction
        horizon: Number of steps to predict ahead
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with:
        - predictions: List of predicted values
        - actuals: List of actual values
        - train_time: Training time in seconds
        - mape: Mean Absolute Percentage Error
        - mbe: Mean Bias Error
        - rmse: Root Mean Squared Error
        - errors: List of individual percentage errors
        - error: Error message (if validation failed)
    """
    # Accept numpy arrays or pandas Series: convert to pandas Series for .iloc handling
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        data = pd.Series(data)
    elif isinstance(data, pd.DataFrame):
        # If a DataFrame was passed, try to use first column / values
        data = data.iloc[:, 0]

    # Validate data requirements
    min_required = train_window + horizon
    if len(data) < min_required:
        # Provide richer diagnostics to help debug unexpected sizing issues
        err_msg = (f'Insufficient data: need {min_required} (train_window={train_window} + horizon={horizon}), '
                   f'have {len(data)}; lookback={lookback}')
        if verbose:
            print(f"[DEBUG] {err_msg}")
        return {
            'error': err_msg,
            'mape': float('inf')
        }
    
    # Real constraint: train_window must have enough data to create samples
    # Each sample needs lookback + horizon points, so we need at least that much
    min_train_window = lookback + horizon
    if train_window < min_train_window:
        return {
            'error': f'train_window {train_window} < minimum required {min_train_window} (lookback {lookback} + horizon {horizon})',
            'mape': float('inf')
        }
    
    # Calculate indices for END-aligned prediction
    # We want to predict the last `horizon` points of the CSV
    # So training ends at: len(data) - horizon
    # And training starts at: len(data) - horizon - train_window
    data_end = len(data)
    prediction_start = data_end - horizon  # Where predictions start
    train_start = prediction_start - train_window  # Where training starts
    train_end = prediction_start  # Where training ends
    
    if train_start < 0:
        return {
            'error': f'Not enough data: need {train_window + horizon}, have {len(data)}',
            'mape': float('inf')
        }
    
    # Get training data (ending right before the prediction window)
    train_data = data.iloc[train_start:train_end]
    
    if verbose:
        print(f"\n=== Single-Shot Evaluation (END-aligned) ===")
        print(f"Data: {len(data)} points total")
        print(f"Training: rows [{train_start}, {train_end}) = {len(train_data)} points")
        print(f"Prediction input: last {lookback} points of training")
        print(f"Evaluation: rows [{prediction_start}, {data_end}) = {horizon} points (END of CSV)")
    
    # Train the model
    if verbose:
        print(f"  Training with {len(train_data)} points...")
        print(f"  First train value: {train_data.iloc[0]:.3f}, Last train value: {train_data.iloc[-1]:.3f}")
    
    start_time = time.time()
    forecaster.train_initial(train_data)
    train_time = time.time() - start_time
    
    if verbose:
        print(f"  Trained in {train_time:.2f}s")
        print(f"  Predicting {horizon} steps...")
    
    # Get the lookback window for prediction (last `lookback` points of training data)
    prediction_input = train_data.iloc[-lookback:]
    
    # Make prediction and measure inference time
    try:
        inference_start = time.time()
        predictions = forecaster.forecast_step(prediction_input)
        inference_time = time.time() - inference_start
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'mape': float('inf')
        }
    
    if verbose:
        print(f"  Got {len(predictions)} predictions")
        print(f"  First pred: {predictions[0]:.3f}, Last pred: {predictions[-1]:.3f}")
    
    # Collect actual values (the last `horizon` points of the CSV)
    actuals = []
    for i in range(horizon):
        actual_idx = prediction_start + i
        if actual_idx >= len(data):
            return {
                'error': f'Insufficient data: need index {actual_idx}, have {len(data)}',
                'mape': float('inf')
            }
        actuals.append(float(data.iloc[actual_idx]))
    
    # Evaluate predictions
    if verbose:
        print(f"  Actual values: rows [{prediction_start}, {data_end})")
        print(f"  First actual: {actuals[0]:.3f}, Last actual: {actuals[-1]:.3f}")
    
    # Calculate sMAPE (symmetric MAPE)
    errors = []
    for actual, pred in zip(actuals, predictions):
        err_abs = abs(actual - pred)
        denominator = (abs(actual) + abs(pred)) / 2.0
        
        if denominator < 1e-6:
            mape_val = 0.0
        else:
            mape_val = (err_abs / denominator) * 100.0
            mape_val = min(mape_val, 1000.0)  # Cap at 1000%
        
        errors.append(mape_val)
    
    mape = np.mean(errors) if errors else float('inf')
    
    # MBE (Mean Bias Error)
    mbe = np.mean([actual - pred for actual, pred in zip(actuals, predictions)])
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean([(actual - pred)**2 for actual, pred in zip(actuals, predictions)]))
    
    if verbose:
        print(f"  MAPE: {mape:.2f}%")
        print(f"  MBE: {mbe:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"=== Evaluation Complete ===\n")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'train_time': train_time,
        'inference_time': inference_time,
        'mape': mape,
        'mbe': mbe,
        'rmse': rmse,
        'errors': errors,
    }


def evaluate_predictions(
    predictions: List[float],
    actuals: List[float],
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate predictions against actual values using standard metrics.
    
    This is the core evaluation logic shared between forecast_main.py and
    tune_lstm_attention.py to ensure consistent metric calculation.
    
    Args:
        predictions: List of predicted values
        actuals: List of actual values
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with metrics:
        - mape: Mean Absolute Percentage Error (symmetric MAPE)
        - mbe: Mean Bias Error
        - rmse: Root Mean Squared Error
        - errors: List of individual percentage errors
    """
    if len(predictions) != len(actuals):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(actuals)} actuals")
    
    if verbose:
        print(f"  Evaluating {len(actuals)} predictions...")
        print(f"  First actual: {actuals[0]:.3f}, Last actual: {actuals[-1]:.3f}")
        print(f"  First pred: {predictions[0]:.3f}, Last pred: {predictions[-1]:.3f}")
    
    # Calculate sMAPE (symmetric MAPE) - prevents division by zero and asymmetry issues
    errors = []
    for actual, pred in zip(actuals, predictions):
        err_abs = abs(actual - pred)
        denominator = (abs(actual) + abs(pred)) / 2.0
        
        if denominator < 1e-6:
            mape_val = 0.0
        else:
            mape_val = (err_abs / denominator) * 100.0
            mape_val = min(mape_val, 1000.0)  # Cap at 1000%
        
        errors.append(mape_val)
    
    mape = np.mean(errors) if errors else float('inf')
    
    # MBE (Mean Bias Error) - shows systematic over/under-prediction
    mbe = np.mean([actual - pred for actual, pred in zip(actuals, predictions)])
    
    # RMSE (Root Mean Squared Error) - penalizes large errors
    rmse = np.sqrt(np.mean([(actual - pred)**2 for actual, pred in zip(actuals, predictions)]))
    
    if verbose:
        print(f"  MAPE: {mape:.2f}%")
        print(f"  MBE: {mbe:.3f}")
        print(f"  RMSE: {rmse:.3f}")
    
    return {
        'mape': mape,
        'mbe': mbe,
        'rmse': rmse,
        'errors': errors,
    }
