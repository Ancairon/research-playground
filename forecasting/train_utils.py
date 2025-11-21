"""
Centralized training utilities.

Provides a single wrapper around model.train(...) so callers (tuners and
the forecaster) use the exact same training flow and get consistent
returned information (training time and basic baseline statistics).

Do not change model implementations here; this is a small shim to
standardize usage across the project.
"""
from typing import Tuple, Dict
import time
import numpy as np


class TrainingFailed(Exception):
    """Raised when training completes but the resulting loss is invalid/too large."""
    pass


def train_model(model, series, quiet: bool = False, max_train_loss: float = 1000.0) -> Dict[str, float]:
    """
    Train a model on the provided series and return training metadata.

    Args:
        model: an object implementing train(series) -> train_time (float)
        series: pd.Series or array-like of training values
        quiet: suppress printing inside this helper (caller may still print)

    Returns:
        dict with keys: 'train_time', 'baseline_mean', 'baseline_std'
    """
    # Call model.train and measure/propagate the return value
    # Some models return the training time; others may return None and
    # set internal timing. Handle both cases.
    try:
        # Pass max_train_loss into model.train so models can early-abort during
        # the epoch loop if they observe an obviously-absurd loss. Models in
        # this repo accept **kwargs in their train(...) signature, so this is
        # a safe, non-breaking way to give them the threshold to check.
        result = model.train(series, max_train_loss=max_train_loss)
    except Exception:
        # Re-raise - callers handle exceptions
        raise

    # If model.train returned a numeric training time, use it. Otherwise,
    # try to get a .training_time attribute from the model (some models
    # may expose it), else set to 0.0
    train_time = 0.0
    if isinstance(result, (int, float)):
        train_time = float(result)
    else:
        train_time = float(getattr(model, 'last_train_time', 0.0) or getattr(model, 'training_time', 0.0) or 0.0)

    # Compute baseline mean/std from the provided series where possible
    try:
        import pandas as pd
        if hasattr(series, 'mean'):
            baseline_mean = float(series.mean())
            baseline_std = float(series.std())
        else:
            arr = np.asarray(series, dtype=float)
            baseline_mean = float(np.mean(arr)) if arr.size > 0 else float('nan')
            baseline_std = float(np.std(arr)) if arr.size > 0 else float('nan')
    except Exception:
        baseline_mean = float('nan')
        baseline_std = float('nan')

    # Detect absurdly high training loss if model exposes it
    # Models that perform iterative training should set `self.last_train_loss`.
    last_loss = getattr(model, 'last_train_loss', None)
    if last_loss is not None:
        try:
            last_loss = float(last_loss)
        except Exception:
            last_loss = float('nan')

        # Default heuristic: abort if loss is not finite or extremely large
        if not np.isfinite(last_loss):
            raise TrainingFailed(f"Training produced non-finite loss: {last_loss}")
        if max_train_loss is not None:
            if last_loss > float(max_train_loss):
                raise TrainingFailed(f"Training loss {last_loss} exceeds max_train_loss={max_train_loss}")
        else:
            # Conservative default: treat loss > 1e8 as absurd (very large MSE)
            if last_loss > 1e8:
                raise TrainingFailed(f"Training produced absurdly large loss: {last_loss}")

    if not quiet:
        try:
            name = getattr(model, 'get_model_name', lambda: getattr(model, '__class__', type(model)).__name__)()
        except Exception:
            name = getattr(model, '__class__', type(model)).__name__
        print(f"[{name}] train completed t={train_time:.3f}s mean={baseline_mean:.3f} std={baseline_std:.3f}")

    return {
        'train_time': train_time,
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
    }
