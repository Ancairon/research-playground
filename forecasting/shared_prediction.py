"""
Shared prediction and evaluation logic.

This module contains common code for making predictions and evaluating them,
ensuring that forecast_main.py and tune_lstm_attention.py use identical logic.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


def train_and_predict(
    forecaster,
    train_data: pd.Series,
    lookback: int,
    horizon: int,
    verbose: bool = False
) -> Tuple[List[float], float]:
    """
    Train a forecaster and make a single prediction.
    
    This is the core prediction logic shared between forecast_main.py (single-shot mode)
    and tune_lstm_attention.py (config evaluation).
    
    Args:
        forecaster: UniversalForecaster instance with a model
        train_data: Training data (pandas Series with 'value')
        lookback: Number of past timesteps to use for prediction
        horizon: Number of steps to predict ahead
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (predictions, train_time)
        - predictions: List of predicted values (length = horizon)
        - train_time: Time taken to train in seconds
    """
    if verbose:
        print(f"  Training with {len(train_data)} points...")
        print(f"  First train value: {train_data.iloc[0]:.3f}, Last train value: {train_data.iloc[-1]:.3f}")
    
    # Train the model
    start_time = time.time()
    forecaster.train_initial(train_data)
    train_time = time.time() - start_time
    
    if verbose:
        print(f"  Trained in {train_time:.2f}s")
        print(f"  Predicting {horizon} steps...")
    
    # Get the lookback window for prediction (last `lookback` points of training data)
    prediction_input = train_data.iloc[-lookback:]
    
    # Make prediction
    predictions = forecaster.forecast_step(prediction_input)
    
    if verbose:
        print(f"  Got {len(predictions)} predictions")
        print(f"  First pred: {predictions[0]:.3f}, Last pred: {predictions[-1]:.3f}")
    
    return predictions, train_time


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
    
    This function combines training, prediction, and evaluation into one operation,
    ensuring that both forecast_main.py and tune_lstm_attention.py use identical logic.
    
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
    # Validate data requirements
    min_required = train_window + horizon
    if len(data) < min_required:
        return {
            'error': f'Insufficient data: need {min_required}, have {len(data)}',
            'mape': float('inf')
        }
    
    if lookback >= train_window:
        return {
            'error': f'Lookback {lookback} >= train_window {train_window}',
            'mape': float('inf')
        }
    
    # Get training data (rows [0, train_window))
    train_data = data.iloc[:train_window]
    
    if verbose:
        print(f"\n=== Single-Shot Evaluation ===")
        print(f"Data: {len(data)} points total")
        print(f"Training: rows [0, {train_window})")
        print(f"Evaluation: rows [{train_window}, {train_window + horizon})")
    
    # Train and predict
    try:
        predictions, train_time = train_and_predict(
            forecaster=forecaster,
            train_data=train_data,
            lookback=lookback,
            horizon=horizon,
            verbose=verbose
        )
    except Exception as e:
        return {
            'error': f'Training/prediction failed: {str(e)}',
            'mape': float('inf')
        }
    
    # Collect actual values (rows [train_window, train_window + horizon))
    actuals = []
    for i in range(horizon):
        actual_idx = train_window + i
        if actual_idx >= len(data):
            return {
                'error': f'Insufficient data: need index {actual_idx}, have {len(data)}',
                'mape': float('inf')
            }
        actuals.append(float(data.iloc[actual_idx]))
    
    # Evaluate predictions
    try:
        metrics = evaluate_predictions(
            predictions=predictions,
            actuals=actuals,
            verbose=verbose
        )
    except Exception as e:
        return {
            'error': f'Evaluation failed: {str(e)}',
            'mape': float('inf')
        }
    
    # Combine results
    result = {
        'predictions': predictions,
        'actuals': actuals,
        'train_time': train_time,
        **metrics  # Include mape, mbe, rmse, errors
    }
    
    if verbose:
        print(f"=== Evaluation Complete ===\n")
    
    return result
