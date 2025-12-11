"""
Simple smoothing utilities for forecasting pipeline.

Provides a single `apply_smoothing` helper used by the forecaster and evaluation code. Supported methods:

- 'moving_average' / 'ma': simple trailing moving average
- 'ewma' / 'exponential': exponential weighted moving average

The functions accept pandas Series or lists and always return the same type/length as the input (Series -> Series, list -> list of floats).
"""
from typing import Union, List, Optional
import pandas as pd


def _to_series(x: Union[pd.Series, List[float]]) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.astype(float)
    return pd.Series(list(x), dtype=float)


def apply_smoothing(
    data: Union[pd.Series, List[float]],
    method: Optional[str] = None,
    window: int = 3,
    alpha: float = 0.2,
) -> Union[pd.Series, List[float]]:
    """
    Smooth a sequence using the requested method.

    Args:
        data: pandas Series or list of numeric values
        method: None or 'moving_average'|'ma'|'ewma'|'exponential'
        window: window size for moving average (int >=1)
        alpha: smoothing factor for EWMA (0-1)

    Returns:
        Smoothed data with same type as input.
    """
    if method is None:
        # No smoothing requested - return input unchanged
        return data

    s = _to_series(data)

    method_lower = (method or '').lower()
    if method_lower in ('moving_average', 'ma'):
        w = max(1, int(window))
        # trailing moving average, min_periods=1 to keep same length
        out = s.rolling(window=w, min_periods=1).mean()
    elif method_lower in ('ewma', 'exponential', 'ewm'):
        a = float(alpha) if alpha is not None else 0.2
        a = min(max(a, 0.0), 1.0)
        out = s.ewm(alpha=a, adjust=False).mean()
    else:
        # Unknown method - return input unchanged
        return data

    # Return same type as input
    if isinstance(data, pd.Series):
        # Preserve the original index
        out.index = data.index
        return out
    else:
        return list(out.astype(float).values)
