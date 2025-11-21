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
from typing import Optional, List, Dict

from models.base_model import BaseTimeSeriesModel
from train_utils import train_model


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


def _format_timestamp_local(timestamp_seconds: float) -> str:
    """Format Unix timestamp to local HH:MM:SS string, matching system timezone."""
    import time as time_module
    if timestamp_seconds == float('inf') or timestamp_seconds > 1e10:
        return 'indefinite'
    try:
        local_time = time_module.localtime(timestamp_seconds)
        return time_module.strftime('%H:%M:%S', local_time)
    except (OverflowError, ValueError):
        return 'indefinite'


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
        retrain_cooldown: int = 0,
        # Backoff behaviour when retraining happens too rapidly
        backoff_long_seconds: int = 30,
        # If retrains happen faster than this (seconds), treat as "rapid" and apply backoff
        retrain_rapid_seconds: int = 2,
        # Max retrains during backoff before forcing extension
        backoff_max_retrains: int = 2,
        # Consecutive OK suppressed validations needed to clear backoff
        backoff_clear_consecutive_ok: int = 3,
        # Error-threshold retrain: trigger retrain if error exceeds this multiplier of baseline std/MAD
        backoff_error_scale: float = 5.0,
        # Error-threshold retrain: minimum error threshold (MAPE %)
        backoff_error_min: float = 70.0,
        print_min_validations: int = 3,
        quiet: bool = False,
        max_train_loss: float = None,
        history_fetcher=None
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
            backoff_long_seconds: Base seconds for backoff window and retrain scheduling
            retrain_rapid_seconds: Wall-clock seconds - retrains faster than this trigger backoff
            backoff_max_retrains: Max retrains during backoff before forcing extension
            backoff_clear_consecutive_ok: Consecutive OK suppressed validations needed to clear backoff
            backoff_error_scale: Multiplier for error-threshold retrain (trigger if error > scale * std/MAD)
            backoff_error_min: Minimum error threshold for retrain (MAPE %)
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
        self.backoff_long_seconds = backoff_long_seconds
        self.retrain_rapid_seconds = retrain_rapid_seconds
        self.backoff_max_retrains = backoff_max_retrains
        self.backoff_clear_consecutive_ok = backoff_clear_consecutive_ok
        self.backoff_error_scale = backoff_error_scale
        self.backoff_error_min = backoff_error_min
        self.print_min_validations = print_min_validations
        self.quiet = quiet
        # Threshold passed down to train_model to allow early-abort during model.train
        self.max_train_loss = max_train_loss
        # Callable to fetch historical training data: history_fetcher(seconds_back) -> pd.Series
        self.history_fetcher = history_fetcher
        # Aggregation method for multiple predictions per target timestamp.
        # Options: 'mean' (default), 'median', 'last' (most recent), 'weighted' (recency-weighted)
        self.aggregation_method = 'mean'
        # Tau (in steps) for exponential recency weighting when aggregation_method=='weighted'
        self.aggregation_weight_tau = 2.0
        
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
        
        # Baseline tracking (for adaptive threshold adjustment)
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_deviation_history = deque(maxlen=50)
        
        # Prediction tracking
        self.pending_validations = deque(maxlen=100)
        self.pending_predictions = {}  # target_ts -> list of (step, pred_val)
        # Finalized aggregated predictions (target_ts -> (agg_val, n_contributors, finalized_step))
        self.finalized_predictions = {}
        # Set of target timestamps that have been finalized (aggregation done once)
        self.finalized_targets = set()
        # Track which finalized timestamps have been printed with their final actual value
        self.finalized_printed = set()
        self.recent_predictions = deque(maxlen=max(1, int(prediction_smoothing)))

    # Hidden/backoff tracking (predictions that are not visible to users during backoff)
        # Each hidden pending validation: (creation_step, creation_ts, pred_values, attempt_type)
        self.hidden_pending_validations = deque(maxlen=100)
        # If a retrain is scheduled during backoff, this stores the timestamp to execute it
        self.backoff_retrain_scheduled_time = None
        # Track wall-clock time of last retrain to detect rapid retraining
        self.last_retrain_time = 0.0
        # Internal backoff bookkeeping
        # Collected mean_horizon_errors for suppressed predictions during the active backoff
        self._suppressed_errors = []
        # Consecutive OK suppressed validations (used to decide when to clear backoff)
        self._consecutive_ok_suppressed = 0
        self._consecutive_error_suppressed = 0
        # Number of retrains executed while the current backoff is active
        self._retrain_count_since_backoff_start = 0
        # Whether a backoff window is currently active (set via _start_backoff)
        self._backoff_active = False
        # Backoff escalation counter (increases on hidden validation failures)
        self.backoff_fail_count = 0
        # Whether we've already announced the current backoff (avoid repeated prints)
        self._backoff_announced = False

        # No file logging by default; backoff events will be printed (and can be silenced with quiet=True)
        
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
        # Use centralized training helper for consistent behavior across callers
        res = train_model(self.model, data, quiet=True, max_train_loss=self.max_train_loss)
        train_time = res.get('train_time', 0.0)
        self.training_times.append(train_time)

        # Calculate baseline statistics from training data (from helper)
        try:
            self.baseline_mean = float(res.get('baseline_mean', float(data.mean())))
            self.baseline_std = float(res.get('baseline_std', float(data.std())))
        except Exception:
            self.baseline_mean = float(data.mean())
            self.baseline_std = float(data.std())

        if not self.quiet:
            print(f"[{self.model.get_model_name()}] Init train t={train_time:.3f}s")
            print(f"[Baseline] Mean: {self.baseline_mean:.2f}, Std: {self.baseline_std:.2f}")

        return train_time

    def _get_training_series(self, live_data: pd.Series) -> pd.Series:
        """
        Obtain a training series to use for retraining.

        Priority:
        - If a history_fetcher callable was provided at init, call it with
          self.train_window and expect a pd.Series (or DataFrame) back.
        - Otherwise, fall back to using the provided live_data (last window).

        The returned object will be a pd.Series of numeric values. If the
        fetcher returns a DataFrame, try to extract a sensible column ("value"
        or the first column).
        """
        # Try history_fetcher if provided
        if callable(getattr(self, 'history_fetcher', None)):
            try:
                res = self.history_fetcher(int(self.train_window))
                # If DataFrame, try to extract a sensible column
                if isinstance(res, pd.DataFrame):
                    if 'value' in res.columns:
                        ser = res['value']
                    else:
                        ser = res.iloc[:, 0]
                else:
                    ser = res

                # Ensure it's a Series
                if isinstance(ser, pd.Series):
                    # If it's shorter than requested, fall back to live_data
                    if len(ser) >= int(self.train_window):
                        return ser.sort_index()
                    else:
                        # fall through to fallback
                        pass
            except Exception:
                # Any failure: fall back to live_data below
                pass

        # Fallback: use the most recent data available in live_data
        try:
            # If live_data is longer than train_window, take the tail
            if len(live_data) >= int(self.train_window):
                return live_data.tail(int(self.train_window)).sort_index()
            else:
                return live_data.sort_index()
        except Exception:
            # As a last resort, return live_data as-is
            return live_data
    
    def forecast_step(self, live_data: pd.Series) -> List[float]:
        """
        Perform one forecasting step.
        
        Args:
            live_data: Current live data window
            
        Returns:
            Smoothed predictions (list of floats)
        """
        # If we're currently in a backoff window, suppress user-facing predictions
        if self._backoff_active:
            # Allow model update (online learning) but do not call predict or append pending validations
            if self.model.supports_online_learning():
                try:
                    self.model.update(live_data)
                except Exception as e:
                    self._append_backoff_log(self.model.get_model_name(), "online_update_failed", f"err={e}")
            if not self.quiet and not self._backoff_announced:
                self._append_backoff_log(self.model.get_model_name(), "suppress_predictions", "indefinite")
                self._backoff_announced = True
            try:
                raw_preds = self.model.predict()
                if raw_preds:
                    raw_preds = [max(-1e9, min(1e9, float(p))) if np.isfinite(p) else 0.0 for p in raw_preds]
                else:
                    raw_preds = []
                # Normalize creation timestamp to whole seconds for consistent matching/validation
                creation_ts = live_data.index[-1].floor('S')
                # Keep last hidden prediction for this creation timestamp
                try:
                    self.hidden_pending_validations = deque([h for h in self.hidden_pending_validations if h[1] != creation_ts], maxlen=self.hidden_pending_validations.maxlen)
                except Exception:
                    newdq = deque(maxlen=100)
                    for h in list(self.hidden_pending_validations):
                        if h[1] != creation_ts:
                            newdq.append(h)
                    self.hidden_pending_validations = newdq
                # Also store contributors for aggregation so suppressed predictions participate in final aggregation
                try:
                    if raw_preds:
                        pred_values = [float(p) for p in raw_preds]
                        target_timestamps = [creation_ts + pd.Timedelta(seconds=i+1) for i in range(len(pred_values))]
                        for i, (target_ts, pred_val) in enumerate(zip(target_timestamps, pred_values)):
                            if target_ts not in self.pending_predictions:
                                self.pending_predictions[target_ts] = []
                            self.pending_predictions[target_ts].append((self.step, pred_val))
                            if len(self.pending_predictions[target_ts]) > 10:
                                self.pending_predictions[target_ts] = self.pending_predictions[target_ts][-10:]

                            # Finalize if enough contributors
                            required = getattr(self.model, 'horizon', None)
                            if required is not None and len(self.pending_predictions[target_ts]) >= int(required) and target_ts not in self.finalized_targets:
                                try:
                                    agg_val, n_contrib = self._aggregate_for_target(self.pending_predictions[target_ts])
                                    self.finalized_predictions[target_ts] = (float(agg_val), int(n_contrib), self.step)
                                    self.finalized_targets.add(target_ts)
                                except Exception:
                                    pass
                except Exception:
                    pass

                self.hidden_pending_validations.append((self.step, creation_ts, raw_preds, "suppressed"))
                self.step += 1
                return raw_preds
            except Exception as e:
                self._append_backoff_log(self.model.get_model_name(), "suppressed_prediction_failed", f"err={e}")
                self.step += 1
                return []

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
        
        # Store for validation - aggregate multiple predictions for same future timestamps
        if preds:
            # Normalize creation timestamp to whole seconds for consistent matching/validation
            last_actual_ts = live_data.index[-1].floor('S')
            pred_values = [float(p) for p in preds]
            
            # Store predictions by their target timestamps (when they will be validated)
            # Each prediction covers horizon steps ahead
            target_timestamps = [last_actual_ts + pd.Timedelta(seconds=i+1) for i in range(len(pred_values))]
            
            for i, (target_ts, pred_val) in enumerate(zip(target_timestamps, pred_values)):
                # Initialize list for this target timestamp if not exists
                if target_ts not in self.pending_predictions:
                    self.pending_predictions[target_ts] = []
                self.pending_predictions[target_ts].append((self.step, pred_val))
                
                # Keep only recent predictions (limit memory)
                if len(self.pending_predictions[target_ts]) > 10:  # Keep last 10 predictions
                    self.pending_predictions[target_ts] = self.pending_predictions[target_ts][-10:]
                
                # If we now have as many contributors as the horizon, finalize aggregation once
                try:
                    required = getattr(self.model, 'horizon', None)
                except Exception:
                    required = None
                if required is not None and len(self.pending_predictions[target_ts]) >= int(required) and target_ts not in self.finalized_targets:
                    try:
                        agg_val, n_contrib = self._aggregate_for_target(self.pending_predictions[target_ts])
                        # Record finalized aggregated value (do not remove contributors - keep them for future visualization if needed)
                        self.finalized_predictions[target_ts] = (float(agg_val), int(n_contrib), self.step)
                        self.finalized_targets.add(target_ts)
                        # Do not print here; wait until actual value is available at validation time
                    except Exception:
                        # Guard against aggregation failures - ignore and continue
                        pass
            
            # Also store the full prediction set for UI display (keep only latest)
            # Remove any existing pending validation with same creation timestamp
            try:
                self.pending_validations = deque([pv for pv in self.pending_validations if pv[1] != last_actual_ts], maxlen=self.pending_validations.maxlen)
            except Exception:
                # Fallback: clear same-timestamp entries by rebuilding
                newdq = deque(maxlen=100)
                for pv in list(self.pending_validations):
                    if pv[1] != last_actual_ts:
                        newdq.append(pv)
                self.pending_validations = newdq
            self.pending_validations.append((self.step, last_actual_ts, pred_values))

            # Print predictions and their corresponding UTC seconds
            try:
                pred_times = [last_actual_ts + pd.Timedelta(seconds=i+1) for i in range(len(pred_values))]
                pred_times_str = [t.strftime('%H:%M:%S') for t in pred_times]
                # print(f"[PREDICT] step={self.step} preds=" + ', '.join(f"{v:.3f}@{t}" for v, t in zip(pred_values, pred_times_str)))
            except Exception as e:
                print(f"[PREDICT] print error: {e}")
        
        self.step += 1
        return preds

    def _append_backoff_log(self, model_name: str, action: str, details: str) -> None:
        """Print a backoff-related event (unless quiet).

        Kept as a method for one-place substitution if you later want to hook a real logger.
        """
        if self.quiet:
            return
        # Use _format_timestamp_local for consistency with other BOFF timestamps
        ts = _format_timestamp_local(time.time())
        print(f"[{ts}] BOFF {action}: {details}")

    def _start_backoff(self, until_ts: float) -> None:
        """Begin a backoff window that blocks user-facing predictions indefinitely until enough consecutive OKs are observed.

        Initializes suppressed-error accumulation and retrain counter.
        Clears pending visible validations to prevent them from printing during backoff.
        """
    # No longer use user_prediction_block_until for time-based blocking
        self._backoff_active = True
        self._suppressed_errors = []
        self._retrain_count_since_backoff_start = 0
        self._backoff_announced = False
        self.pending_validations.clear()
        if not self.quiet:
            print(f"\n{'='*70}")
            print(f"[BACKOFF] Entering prediction suppression mode (indefinite)")
            print(f"[BACKOFF] Predictions will be generated but suppressed until enough consecutive OKs are observed.")
            print(f"[BACKOFF] Reason: Rapid retraining detected (model unstable)")
            print(f"{'='*70}\n")

    def _maybe_finish_backoff(self, threshold: Optional[float]) -> None:
        """Evaluate backoff conditions and decide whether to clear backoff or keep it active.

        Decision rule: if we have enough consecutive OK suppressed validations, clear backoff.
        Otherwise keep backoff active indefinitely (no time-based expiration).
        
        Note: Backoff does NOT expire on its own - it only clears when conditions are met.
        """
        if not getattr(self, '_backoff_active', False):
            return

        # Only clear backoff if enough consecutive OKs
        required_ok = getattr(self, 'backoff_clear_consecutive_ok', 3)
        if self._consecutive_ok_suppressed >= required_ok:
            self.backoff_fail_count = 0
            # No longer use user_prediction_block_until for time-based blocking
            self._backoff_active = False
            if not self.quiet:
                print(f"\n{'='*70}")
                print(f"[BACKOFF] Cleared - Resuming normal predictions")
                print(f"[BACKOFF] Reason: {self._consecutive_ok_suppressed} consecutive OK validations (required: {required_ok})")
                print(f"{'='*70}\n")
            self._suppressed_errors = []
            self._consecutive_ok_suppressed = 0
        # Always clear pending visible predictions during backoff
        self.pending_validations.clear()

    def _validate_hidden_predictions(self, live_data: pd.Series, threshold: Optional[float]) -> None:
        """Validate ONE hidden pending prediction if ready and log results.

        If a short attempt fails (error > threshold), schedule a hidden retrain after
        `backoff_long_seconds`.
        
        Changed to validate one at a time instead of batch processing to avoid confusing
        timestamp dumps where all SUPP lines print together.
        """
        if not self.hidden_pending_validations:
            return

        # Check only the OLDEST prediction (like visible validations)
        creation_step, creation_ts, pred_values, attempt_type = self.hidden_pending_validations[0]
        steps_elapsed = self.step - creation_step
        
        # Use same readiness logic as visible validations
        if steps_elapsed < self.model.horizon:
            return  # Not ready yet

        # Remove from queue (we're processing it now)
        self.hidden_pending_validations.popleft()

        # Attempt validation using aggregated predictions
        horizon_errors = []
        horizon_actuals = []
        horizon_preds = []
        
        # Calculate target timestamps for this prediction set
        target_timestamps = [creation_ts + pd.Timedelta(seconds=i+1) for i in range(len(pred_values))]
        
        for i, target_ts in enumerate(target_timestamps):
            # Get aggregated prediction for this target timestamp
            if target_ts in self.pending_predictions and self.pending_predictions[target_ts]:
                aggregated_pred, contrib_n = self._aggregate_for_target(self.pending_predictions[target_ts])
            else:
                # Fallback to original prediction if no aggregated data
                aggregated_pred = pred_values[i]
            
            offset_from_now = steps_elapsed - (i + 1)
            try:
                current_ts = live_data.index[-1]
                target_ts_actual = current_ts - pd.Timedelta(seconds=offset_from_now)
                target_ts_floor = target_ts_actual.floor('S')
                idx_secs = live_data.index.floor('S')
                matches = live_data.loc[idx_secs == target_ts_floor]
                if len(matches) > 0:
                    y_true_now = float(matches.iloc[-1])
                else:
                    continue
            except Exception as e:
                # Log and continue
                self._append_backoff_log(self.model.get_model_name(), "hidden_validate_lookup_err", f"err={e}")
                continue

            # If this target was finalized and not yet printed with its final actual, print now
            if target_ts in self.finalized_predictions and target_ts not in self.finalized_printed and not self.quiet:
                try:
                    agg_val, n_contrib, finalized_step = self.finalized_predictions[target_ts]
                    print(f"[AGG-RESULT] target={target_ts} agg={agg_val:.3f} actual={y_true_now:.3f} n={n_contrib}")
                except Exception:
                    pass
                self.finalized_printed.add(target_ts)

            err_abs = abs(y_true_now - aggregated_pred)
            denominator = (abs(y_true_now) + abs(aggregated_pred)) / 2.0
            if denominator < 1e-6:
                continue
            
            # Calculate MAPE: symmetric mean absolute percentage error
            err_relative = (err_abs / denominator) * 100.0
            err_relative = min(err_relative, 1000.0)

            horizon_errors.append(err_relative)
            horizon_actuals.append(y_true_now)
            horizon_preds.append(aggregated_pred)

        if len(horizon_errors) != self.model.horizon:
            # Incomplete validation - discard this prediction
            return

        # Calculate final MAPE: mean of percentage errors
        mean_horizon_error = float(np.mean(horizon_errors))
        
        # Store for later validation (log happens in SUPP print or backoff decision)
        try:
            self._suppressed_errors.append(mean_horizon_error)
        except Exception as e:
            self._append_backoff_log(self.model.get_model_name(), "suppressed_append_failed", f"err={e}")

        if attempt_type == "short" and threshold is not None:
            # For short attempts, still schedule a retrain if error > threshold, but
            # do not immediately extend the user-visible block here. The retrain when
            # executed will increment the retrain counter used by end-of-backoff logic.
            if mean_horizon_error > threshold:
                # Shortened: bad short validation triggers extended backoff
                self.backoff_fail_count += 1
                multiplier = 2 ** (max(0, self.backoff_fail_count - 1))
                now = time.time()
                proposed = now + float(self.backoff_long_seconds) * multiplier
                max_cap = now + 30.0
                scheduled = min(proposed, max_cap)
                self.backoff_retrain_scheduled_time = scheduled
                sched_str = _format_timestamp_local(scheduled)
                self._append_backoff_log(
                    self.model.get_model_name(),
                    "schedule_retrain_after_backoff",
                    f"retrain_at={sched_str}, lvl={self.backoff_fail_count}, mult={multiplier}"
                )
                self._backoff_announced = False
            # else: Short validation passed, no log needed
        elif attempt_type == "suppressed" and threshold is not None:
            # Print the same validation-style summary we print for visible validations,
            # but DO NOT register these suppressed predictions into the forecaster's
            # visible error lists (they are only used for aggregate backoff decisions).
            if not self.quiet:
                try:
                    # Use stored creation timestamp (when prediction was made)
                    import time as time_module
                    ts_unix = creation_ts.timestamp()
                    ts_str = time_module.strftime('%H:%M:%S', time_module.localtime(ts_unix))
                except Exception as e:
                    self._append_backoff_log(self.model.get_model_name(), "timestamp_format_err", f"err={e}")
                    ts_str = ''
                thr_str = f"{threshold:.3f}" if threshold is not None else "N/A"
                print(f"[{ts_str}] [B] SUPP s={creation_step} err={mean_horizon_error:.3f}% thr={thr_str}%")
                if mean_horizon_error > 500.0:
                    print(f"[WARN] Extreme suppressed err={mean_horizon_error:.3f}% s={creation_step}")

            # Track consecutive OKs for backoff clearing logic (no separate BOFF log - redundant with SUPP line)
            if mean_horizon_error <= threshold:
                self._consecutive_ok_suppressed = getattr(self, '_consecutive_ok_suppressed', 0) + 1
                
                # Check if we've reached the required consecutive OKs to clear backoff immediately
                required_ok = getattr(self, 'backoff_clear_consecutive_ok', 3)
                if not self.quiet:
                    print(f"[DEBUG] Consecutive OK: {self._consecutive_ok_suppressed}/{required_ok}, backoff_active={getattr(self, '_backoff_active', False)}")
                
                if self._consecutive_ok_suppressed >= required_ok and getattr(self, '_backoff_active', False):
                    # Clear backoff immediately
                    self.backoff_fail_count = 0
                    self._backoff_active = False
                    
                    # Print clear resume message
                    if not self.quiet:
                        print(f"\n{'='*70}")
                        print(f"[BACKOFF] Cleared - Resuming normal predictions")
                        print(f"[BACKOFF] Reason: {self._consecutive_ok_suppressed} consecutive OK validations")
                        print(f"{'='*70}\n")
                    
                    self._append_backoff_log(
                        self.model.get_model_name(), 
                        'backoff_cleared_by_consecutive_ok',
                        f'ok_consec={self._consecutive_ok_suppressed}/{required_ok}'
                    )
                    self._suppressed_errors = []
                    self._consecutive_ok_suppressed = 0
            else:
                # Reset consecutive OK counter on any bad validation
                self._consecutive_ok_suppressed = 0

    def _maybe_execute_scheduled_retrain(self, live_data: pd.Series) -> None:
        """If a hidden retrain was scheduled and the scheduled step has been reached, execute it."""
        if self.backoff_retrain_scheduled_time is None:
            return
        if time.time() < self.backoff_retrain_scheduled_time:
            return
        try:
            retrain_series = self._get_training_series(live_data)
            res = train_model(self.model, retrain_series, quiet=True, max_train_loss=self.max_train_loss)
            retrain_time = res.get('train_time', 0.0)
            self.training_times.append(retrain_time)
            self.last_retrain_step = self.step

            # Update baseline statistics with new training data (from helper)
            try:
                self.baseline_mean = float(res.get('baseline_mean', float(retrain_series.mean())))
                self.baseline_std = float(res.get('baseline_std', float(retrain_series.std())))
            except Exception:
                # Fallback to live_data if computing stats on retrain_series fails
                self.baseline_mean = float(live_data.mean())
                self.baseline_std = float(live_data.std())

            # If we are in an active backoff, count this retrain for evaluation
            if getattr(self, '_backoff_active', False):
                try:
                    self._retrain_count_since_backoff_start += 1
                except Exception as e:
                    self._append_backoff_log(self.model.get_model_name(), "retrain_count_increment_failed", f"err={e}")

            # Reset escalation and announce
            self.backoff_retrain_scheduled_time = None
            self.consec_count = 0
            self.backoff_fail_count = 0
            self._backoff_announced = False
            self._append_backoff_log(self.model.get_model_name(), "hidden_retrain_executed",
                                     f"step={self.step}, time={retrain_time:.3f}s, baseline_mean={self.baseline_mean:.2f}")

            # Immediately generate a silent prediction (hidden) after retrain so we can validate it later
            try:
                raw_preds = self.model.predict()
                if raw_preds:
                    raw_preds = [max(-1e9, min(1e9, float(p))) if np.isfinite(p) else 0.0 for p in raw_preds]
                else:
                    raw_preds = []
                # store as hidden pending validation with creation timestamp and attempt type 'post_retrain'
                # Use negative step to avoid collision with visible predictions
                # Normalize creation timestamp to whole seconds for consistent matching/validation
                creation_ts = live_data.index[-1].floor('S')
                # Keep last hidden prediction for this creation timestamp (post retrain)
                try:
                    self.hidden_pending_validations = deque([h for h in self.hidden_pending_validations if h[1] != creation_ts], maxlen=self.hidden_pending_validations.maxlen)
                except Exception:
                    newdq = deque(maxlen=100)
                    for h in list(self.hidden_pending_validations):
                        if h[1] != creation_ts:
                            newdq.append(h)
                    self.hidden_pending_validations = newdq
                self.hidden_pending_validations.append((-self.step, creation_ts, raw_preds, "post_retrain"))
                self._append_backoff_log(self.model.get_model_name(), "hidden_pred_post_retrain",
                                         f"created_at_step={self.step}")
            except Exception as e:
                self._append_backoff_log(self.model.get_model_name(), "hidden_pred_post_retrain_failed", f"err={e}")
        except Exception as e:
            self._append_backoff_log(self.model.get_model_name(), "hidden_retrain_failed", f"err={e}")
            self.backoff_retrain_scheduled_time = None

    def _aggregate_for_target(self, preds_with_steps: list):
        """Aggregate a list of (step, pred_val) tuples into a single prediction.

        Returns (aggregated_value, n_contributors).
        """
        if not preds_with_steps:
            return (None, 0)

        # Extract in chronological order (they are appended in order)
        preds = [pv for _, pv in preds_with_steps]
        n = len(preds)
        method = getattr(self, 'aggregation_method', 'mean')

        if method == 'mean':
            return (float(np.mean(preds)), n)
        elif method == 'median':
            return (float(np.median(preds)), n)
        elif method == 'last':
            return (float(preds[-1]), n)
        elif method == 'weighted':
            # Recency-weighted: weight = exp(-age / tau), where age in steps = self.step - pred_step
            tau = max(1e-6, getattr(self, 'aggregation_weight_tau', 2.0))
            weights = []
            values = []
            for step, val in preds_with_steps:
                age = max(0, self.step - step)
                w = np.exp(-float(age) / float(tau))
                weights.append(w)
                values.append(float(val))
            weights = np.array(weights, dtype=float)
            values = np.array(values, dtype=float)
            if weights.sum() == 0:
                return (float(np.mean(values)), n)
            return (float(np.sum(weights * values) / np.sum(weights)), n)
        else:
            # Unknown method, fallback to mean
            return (float(np.mean(preds)), n)
    
    def _compute_error_backoff_threshold(self) -> float:
        """
        Compute error threshold for backoff triggering.
        
        Returns threshold in same units as error metric (% for MAPE, absolute for RMSE).
        If error exceeds this threshold, backoff is triggered to prevent bad predictions.
        """
        if len(self.recent_errors) == 0:
            # No history - use minimum threshold
            return float(self.backoff_error_min)
        
        arr = np.asarray(self.recent_errors, dtype=float)
        
        # Use MAD/std similar to retrain threshold, but with different scale
        if self.retrain_use_mad:
            med = float(np.nanmedian(arr))
            mad = float(np.nanmedian(np.abs(arr - med)))
            sigma = 1.4826 * mad
        else:
            sigma = float(np.nanstd(arr))
        
        # Use backoff_error_scale and backoff_error_min
        threshold = max(
            float(self.backoff_error_min),
            float(self.backoff_error_scale) * abs(sigma)
        )
        
        return threshold
    
    def validate_predictions(self, live_data: pd.Series) -> Optional[float]:
        """
        Validate oldest pending prediction (visible or hidden) in chronological order.
        
        Args:
            live_data: Current live data
            
        Returns:
            Mean horizon error if validation occurred, None otherwise
        """
        # Check both queues and pick the oldest one by creation timestamp
        visible_ready = None
        hidden_ready = None

        # Check visible queue (but skip if we're currently in backoff)
        if self.pending_validations and not self._backoff_active:
            creation_step, creation_ts, pred_values = self.pending_validations[0]
            steps_elapsed = self.step - creation_step
            if steps_elapsed >= self.model.horizon:
                visible_ready = (creation_step, creation_ts, pred_values, "visible")

        # Check hidden queue
        if self.hidden_pending_validations:
            creation_step, creation_ts, pred_values, attempt_type = self.hidden_pending_validations[0]
            steps_elapsed = self.step - creation_step
            if steps_elapsed >= self.model.horizon:
                hidden_ready = (creation_step, creation_ts, pred_values, attempt_type)

        # If nothing is ready, return None
        if visible_ready is None and hidden_ready is None:
            return None

        # Pick the oldest one by timestamp
        if visible_ready is not None and hidden_ready is not None:
            # Both ready - pick older timestamp
            if visible_ready[1] <= hidden_ready[1]:
                self.pending_validations.popleft()
                return self._validate_one_prediction(live_data, *visible_ready)
            else:
                self.hidden_pending_validations.popleft()
                return self._validate_one_prediction(live_data, *hidden_ready)
        elif visible_ready is not None:
            self.pending_validations.popleft()
            return self._validate_one_prediction(live_data, *visible_ready)
        else:  # hidden_ready is not None
            self.hidden_pending_validations.popleft()
            return self._validate_one_prediction(live_data, *hidden_ready)
    
    def _validate_one_prediction(self, live_data: pd.Series, creation_step: int, creation_ts, 
                                  pred_values: list, pred_type: str) -> Optional[float]:
        """
        Validate a single prediction set using aggregated predictions for each target timestamp.
        
        Args:
            live_data: Current live data
            creation_step: Step when prediction was created
            creation_ts: Timestamp when prediction was created
            pred_values: Original prediction values (for backward compatibility)
            pred_type: Type of prediction ("visible", "suppressed", "short", "post_retrain")
            
        Returns:
            Mean horizon error if validation succeeded, None otherwise
        """
        steps_elapsed = self.step - creation_step
        
        # Validate all horizon steps using aggregated predictions
        horizon_errors = []
        horizon_actuals = []
        horizon_preds = []
        
        for i in range(self.model.horizon):
            # Calculate target timestamp for this horizon step
            target_ts = creation_ts + pd.Timedelta(seconds=i+1)
            
            # Get aggregated prediction for this target timestamp
            if target_ts in self.pending_predictions and self.pending_predictions[target_ts]:
                aggregated_pred, contrib_n = self._aggregate_for_target(self.pending_predictions[target_ts])
                # Do not print intermediate aggregations here; final aggregation with actual will be printed once at validation time
            else:
                # Fallback to original prediction if no aggregated data
                if i < len(pred_values):
                    aggregated_pred = pred_values[i]
                else:
                    continue
            
            # Find actual value
            try:
                current_ts = live_data.index[-1]
                offset_from_now = steps_elapsed - (i + 1)
                target_ts_actual = current_ts - pd.Timedelta(seconds=offset_from_now)
                target_ts_floor = target_ts_actual.floor('S')
                
                idx_secs = live_data.index.floor('S')
                matches = live_data.loc[idx_secs == target_ts_floor]
                if len(matches) > 0:
                    y_true_now = float(matches.iloc[-1])
                else:
                    continue
            except Exception as e:
                if pred_type != "visible":
                    self._append_backoff_log(self.model.get_model_name(), "validation_lookup_failed", f"err={e}")
                continue

            # If this target was finalized and not yet printed with its final actual, print now
            if target_ts in self.finalized_predictions and target_ts not in self.finalized_printed and not self.quiet:
                try:
                    agg_val, n_contrib, finalized_step = self.finalized_predictions[target_ts]
                    print(f"[AGG-RESULT] creation={creation_ts} target={target_ts} agg={agg_val:.3f} actual={y_true_now:.3f} n={n_contrib}")
                except Exception:
                    pass
                self.finalized_printed.add(target_ts)
            
            # Calculate error with safeguards
            err_abs = abs(y_true_now - aggregated_pred)
            denominator = (abs(y_true_now) + abs(aggregated_pred)) / 2.0
            
            if denominator < 1e-6:
                continue
            
            # Calculate MAPE: symmetric mean absolute percentage error
            err_relative = (err_abs / denominator) * 100.0
            err_relative = min(err_relative, 1000.0)
            
            horizon_errors.append(err_relative)
            horizon_actuals.append(y_true_now)
            horizon_preds.append(aggregated_pred)
        
        # Only process if validated full horizon
        if len(horizon_errors) != self.model.horizon:
            return None
        
        # Calculate final MAPE: mean of percentage errors
        mean_horizon_error = float(np.mean(horizon_errors))
        
        # Store latest error for external display (e.g., CSV progress)
        self._latest_validation_error = mean_horizon_error
        
        # Clean up validated predictions from memory
        current_ts = live_data.index[-1].floor('S')
        # Remove predictions that are too old (more than horizon steps in the past)
        cutoff_ts = current_ts - pd.Timedelta(seconds=self.model.horizon * 2)
        self.pending_predictions = {
            ts: preds for ts, preds in self.pending_predictions.items() 
            if ts > cutoff_ts
        }
        
        # Handle based on prediction type
        if pred_type == "visible":
            return self._handle_visible_validation(live_data, creation_ts, mean_horizon_error, 
                                                   horizon_errors, horizon_actuals, horizon_preds, creation_step)
        else:
            return self._handle_hidden_validation(live_data, creation_ts, creation_step, mean_horizon_error, 
                                                  pred_type, horizon_actuals, horizon_preds)
    
    def _handle_visible_validation(self, live_data: pd.Series, creation_ts, mean_horizon_error: float,
                                   horizon_errors: list, horizon_actuals: list, horizon_preds: list,
                                   creation_step: int) -> float:
        """Handle a visible prediction validation result."""
        # Debug: warn about extreme errors
        if mean_horizon_error > 500.0 and not self.quiet:
            print(f"[WARN] Extreme err={mean_horizon_error:.3f}% act={np.mean(horizon_actuals):.3f} pred={np.mean(horizon_preds):.3f}")
        
        self.errors.append(mean_horizon_error)
        self.recent_errors.append(mean_horizon_error)
        
        # Calculate baseline deviation if baseline exists
        baseline_deviation = 0.0
        if self.baseline_mean is not None and self.baseline_std is not None and self.baseline_std > 1e-6:
            # Calculate z-score: how many std deviations from training baseline
            current_actual = float(np.mean(horizon_actuals))
            baseline_deviation = abs(current_actual - self.baseline_mean) / self.baseline_std
            self.baseline_deviation_history.append(baseline_deviation)
        
        # Compute threshold with baseline adjustment
        threshold = None
        baseline_adjustment_info = None  # Store info for later printing
        if self.dynamic_retrain:
            try:
                base_threshold = compute_threshold_from_errors(
                    self.recent_errors, self.retrain_scale, self.retrain_min, self.retrain_use_mad
                )
                
                # Adjust threshold based on baseline deviation
                # If current values deviate significantly from training baseline,
                # lower the threshold (more sensitive to errors)
                if baseline_deviation > 2.0:  # More than 2 std deviations from baseline
                    # Reduce threshold by up to 30% based on deviation magnitude
                    adjustment_factor = max(0.7, 1.0 - (baseline_deviation - 2.0) * 0.1)
                    threshold = base_threshold * adjustment_factor
                    
                    # Store for later printing (only if we're actually going to print VAL line)
                    if not self.quiet and len(self.baseline_deviation_history) % 10 == 0:
                        baseline_adjustment_info = (baseline_deviation, base_threshold, threshold, adjustment_factor)
                else:
                    threshold = base_threshold
                    
            except Exception as e:
                self._append_backoff_log(self.model.get_model_name(), "threshold_compute_failed", f"err={e}")
                threshold = float(self.retrain_min)

        # Validate any hidden predictions that might be pending and possibly schedule hidden retrains
        try:
            self._validate_hidden_predictions(live_data, threshold)
        except Exception as e:
            # Must not affect visible validation
            self._append_backoff_log(self.model.get_model_name(), "hidden_validation_error", f"err={e}")

        # Execute any scheduled hidden retrain if its time has arrived
        try:
            self._maybe_execute_scheduled_retrain(live_data)
        except Exception as e:
            self._append_backoff_log(self.model.get_model_name(), "scheduled_retrain_error", f"err={e}")
        # After hidden retrain handling, if a backoff window existed and has expired, evaluate it
        try:
            self._maybe_finish_backoff(threshold)
        except Exception as e:
            self._append_backoff_log(self.model.get_model_name(), "finish_backoff_error", f"err={e}")
        
        # Check for error-threshold-based retrain BEFORE normal retrain check
        # If error spikes beyond threshold, trigger immediate retrain
        if threshold is not None and not getattr(self, '_backoff_active', False):
            try:
                # Compute error threshold for retrain triggering
                error_retrain_threshold = self._compute_error_backoff_threshold()
                
                # If error exceeds threshold, trigger immediate retrain
                if mean_horizon_error > error_retrain_threshold:
                    # Check if we can retrain (respect cooldown)
                    can_retrain = (self.step - self.last_retrain_step) >= self.retrain_cooldown
                    
                    if can_retrain:
                        if not self.quiet:
                            print(f"\n{'='*70}")
                            print(f"[RETRAIN] Error spike detected!")
                            print(f"[RETRAIN] Error: {mean_horizon_error:.1f}% > Threshold: {error_retrain_threshold:.1f}%")
                            print(f"[RETRAIN] Triggering immediate retrain")
                            print(f"{'='*70}\n")
                        
                        # Execute immediate retrain
                        try:
                            retrain_series = self._get_training_series(live_data)
                            res = train_model(self.model, retrain_series, quiet=True, max_train_loss=self.max_train_loss)
                            retrain_time = res.get('train_time', 0.0)
                            self.training_times.append(retrain_time)
                            self.last_retrain_step = self.step
                            prev_retrain_time = self.last_retrain_time
                            self.last_retrain_time = time.time()
                            self.consec_count = 0  # Reset consecutive error counter

                            # Update baseline statistics
                            try:
                                self.baseline_mean = float(res.get('baseline_mean', float(retrain_series.mean())))
                                self.baseline_std = float(res.get('baseline_std', float(retrain_series.std())))
                            except Exception:
                                self.baseline_mean = float(live_data.mean())
                                self.baseline_std = float(live_data.std())

                            if not self.quiet:
                                print(f"[{self.model.get_model_name()}] Retrain s={self.step} t={retrain_time:.3f}s (error spike)")
                                print(f"[Baseline] Updated - Mean: {self.baseline_mean:.2f}, Std: {self.baseline_std:.2f}\n")

                            self._append_backoff_log(
                                self.model.get_model_name(), 
                                "error_spike_retrain",
                                f"err={mean_horizon_error:.1f}, thr={error_retrain_threshold:.1f}, time={retrain_time:.3f}s"
                            )

                            # Clear error history after retrain
                            self.errors.clear()
                            self.recent_errors.clear()

                            # Check if this was a rapid retrain (could trigger backoff)
                            now = time.time()
                            rapid_retrain_detected = (now - prev_retrain_time) < float(self.retrain_rapid_seconds)
                            if rapid_retrain_detected:
                                elapsed = now - prev_retrain_time
                                self._start_backoff(time.time() + float(self.backoff_long_seconds))
                                self._append_backoff_log(self.model.get_model_name(), "rapid_retrain_backoff",
                                                         f"elapsed={elapsed:.1f}s, block=indefinite")
                            
                        except Exception as e:
                            if not self.quiet:
                                print(f"[{self.model.get_model_name()}] Retrain failed: {e}")
                            self._append_backoff_log(self.model.get_model_name(), "error_spike_retrain_failed", f"err={e}")
                    else:
                        # Can't retrain yet due to cooldown
                        if not self.quiet:
                            cooldown_remaining = self.retrain_cooldown - (self.step - self.last_retrain_step)
                            print(f"[WARN] Error spike ({mean_horizon_error:.1f}% > {error_retrain_threshold:.1f}%) but retrain on cooldown ({cooldown_remaining} steps remaining)")
                    
            except Exception as e:
                self._append_backoff_log(self.model.get_model_name(), "error_retrain_check_failed", f"err={e}")
        
        # Check if backoff was activated/extended during _maybe_finish_backoff - if so, don't print VAL
        in_backoff_now = time.time() < getattr(self, 'user_prediction_block_until', 0.0)
        
        # Print validation details only if not in backoff
        if not self.quiet and not in_backoff_now:
            # First print baseline adjustment info if available
            if baseline_adjustment_info is not None:
                baseline_deviation, base_threshold, threshold, adjustment_factor = baseline_adjustment_info
                print(f"[Baseline] Deviation: {baseline_deviation:.2f}σ, "
                      f"Threshold: {base_threshold:.1f}% → {threshold:.1f}% "
                      f"(adjustment: {adjustment_factor:.2f}x)")
            
            # Then print validation line
            try:
                # Use current time (when validation happens) for real-time heartbeat
                import time as time_module
                ts_str = time_module.strftime('%H:%M:%S', time_module.localtime())
            except Exception as e:
                self._append_backoff_log(self.model.get_model_name(), "timestamp_format_failed", f"err={e}")
                ts_str = ''
            thr_str = f"{threshold:.3f}" if threshold is not None else "N/A"
            error_unit = "" if getattr(self, 'use_rmse', False) else "%"
            print(f"[{ts_str}] [P] VAL s={self.step} err={mean_horizon_error:.3f}{error_unit} thr={thr_str}{error_unit}")
        
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
    
    def _handle_hidden_validation(self, live_data: pd.Series, creation_ts, creation_step: int,
                                   mean_horizon_error: float, attempt_type: str,
                                   horizon_actuals: list, horizon_preds: list) -> Optional[float]:
        """Handle a hidden prediction validation result."""
        # Store error for backoff evaluation
        try:
            self._suppressed_errors.append(mean_horizon_error)
        except Exception as e:
            self._append_backoff_log(self.model.get_model_name(), "suppressed_append_failed", f"err={e}")
        
        # Compute threshold for hidden validation checks
        threshold = None
        if self.dynamic_retrain and len(self.recent_errors) > 0:
            try:
                threshold = compute_threshold_from_errors(
                    self.recent_errors, self.retrain_scale, self.retrain_min, self.retrain_use_mad
                )
            except Exception as e:
                threshold = float(self.retrain_min)
        else:
            threshold = float(self.retrain_min)
        
        
        if attempt_type == "suppressed" or attempt_type == "post_retrain":
            # Print validation summary for suppressed predictions
            if not self.quiet:
                try:
                    # Use current time (when validation happens) for real-time heartbeat
                    import time as time_module
                    ts_str = time_module.strftime('%H:%M:%S', time_module.localtime())
                except Exception as e:
                    self._append_backoff_log(self.model.get_model_name(), "timestamp_format_err", f"err={e}")
                    ts_str = ''
                thr_str = f"{threshold:.3f}" if threshold is not None else "N/A"
                # Don't print step number for hidden predictions to avoid confusion with visible predictions
                type_label = "SUPP" if attempt_type == "suppressed" else "POST"
                print(f"[{ts_str}] [B] {type_label} err={mean_horizon_error:.3f}% thr={thr_str}%")
                if mean_horizon_error > 500.0:
                    print(f"[WARN] Extreme suppressed err={mean_horizon_error:.3f}%")
            
            # Track consecutive OKs/errors for backoff clearing logic
            if threshold is not None:
                if mean_horizon_error <= threshold:
                    self._consecutive_ok_suppressed = getattr(self, '_consecutive_ok_suppressed', 0) + 1
                    
                    # Debug logging to trace backoff clearing
                    required_ok = getattr(self, 'backoff_clear_consecutive_ok', 3)
                    if not self.quiet:
                        print(f"[DEBUG] Consecutive OK: {self._consecutive_ok_suppressed}/{required_ok}, backoff_active={getattr(self, '_backoff_active', False)}")
                    
                    # Check if we've reached the required consecutive OKs to clear backoff immediately
                    if self._consecutive_ok_suppressed >= required_ok and getattr(self, '_backoff_active', False):
                        # Clear backoff immediately
                        self.backoff_fail_count = 0
                        # No longer use user_prediction_block_until for time-based blocking
                        self._backoff_active = False
                        
                        # Print clear resume message
                        if not self.quiet:
                            print(f"\n{'='*70}")
                            print(f"[BACKOFF] Cleared - Resuming normal predictions")
                            print(f"[BACKOFF] Reason: {self._consecutive_ok_suppressed} consecutive OK validations")
                            print(f"{'='*70}\n")
                        
                        self._append_backoff_log(
                            self.model.get_model_name(), 
                            'backoff_cleared_by_consecutive_ok',
                            f'ok_consec={self._consecutive_ok_suppressed}/{required_ok}'
                        )
                        self._suppressed_errors = []
                        self._consecutive_ok_suppressed = 0
                    
                    # Reset consecutive error counter on success
                    self._consecutive_error_suppressed = 0
                else:
                    # Reset consecutive OK counter on error
                    self._consecutive_ok_suppressed = 0
                    # Track consecutive errors during backoff
                    self._consecutive_error_suppressed += 1
                    
                    # If we get multiple consecutive bad suppressed predictions, trigger retrain immediately
                    # Use same threshold as normal retraining (retrain_consec)
                    if self._consecutive_error_suppressed >= self.retrain_consec:
                        try:
                            retrain_series = self._get_training_series(live_data)
                            res = train_model(self.model, retrain_series, quiet=True, max_train_loss=self.max_train_loss)
                            retrain_time = res.get('train_time', 0.0)
                            self.training_times.append(retrain_time)
                            self.last_retrain_step = self.step
                            prev_retrain_time = self.last_retrain_time
                            self.last_retrain_time = time.time()
                            try:
                                self.baseline_mean = float(res.get('baseline_mean', float(retrain_series.mean())))
                                self.baseline_std = float(res.get('baseline_std', float(retrain_series.std())))
                            except Exception:
                                self.baseline_mean = float(live_data.mean())
                                self.baseline_std = float(live_data.std())
                            if not self.quiet:
                                print(f"[SUPP] Retrain s={self.step} t={retrain_time:.3f}s (backoff mode)")
                                print(f"[Baseline] Updated - Mean: {self.baseline_mean:.2f}, Std: {self.baseline_std:.2f}")
                        except Exception:
                            if not self.quiet:
                                print(f"[SUPP] Retrain failed")
                        finally:
                            self.errors.clear()
                            self.recent_errors.clear()
                            self._consecutive_error_suppressed = 0  # Reset after retrain
        
        return mean_horizon_error
    
    def _check_retrain(self, live_data: pd.Series, error: float, threshold: float, _pred_step: int):
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
            now = time.time()
            # Check if this retrain is happening too soon after the last one
            rapid_retrain_detected = (now - self.last_retrain_time) < float(self.retrain_rapid_seconds)
            
            try:
                retrain_series = self._get_training_series(live_data)
                res = train_model(self.model, retrain_series, quiet=True, max_train_loss=self.max_train_loss)
                retrain_time = res.get('train_time', 0.0)
                self.training_times.append(retrain_time)
                self.last_retrain_step = self.step
                prev_retrain_time = self.last_retrain_time
                self.last_retrain_time = time.time()
                self.consec_count = 0

                # Update baseline statistics with new training data
                try:
                    self.baseline_mean = float(res.get('baseline_mean', float(retrain_series.mean())))
                    self.baseline_std = float(res.get('baseline_std', float(retrain_series.std())))
                except Exception:
                    self.baseline_mean = float(live_data.mean())
                    self.baseline_std = float(live_data.std())

                if not self.quiet:
                    print(f"[{self.model.get_model_name()}] Retrain s={self.step} t={retrain_time:.3f}s thr={threshold:.3f}%")
                    print(f"[Baseline] Updated - Mean: {self.baseline_mean:.2f}, Std: {self.baseline_std:.2f}")
                
                # After retrain completes, check if it was rapid and trigger backoff
                if rapid_retrain_detected:
                    elapsed = now - prev_retrain_time
                    # Initialize backoff bookkeeping
                    self._start_backoff(time.time() + float(self.backoff_long_seconds))
                    self._append_backoff_log(self.model.get_model_name(), "rapid_retrain_backoff",
                                             f"elapsed={elapsed:.1f}s, block=indefinite")
                    
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
            "total_retrains": len(self.training_times) - 1,  # Exclude initial train
            "baseline_mean": self.baseline_mean if self.baseline_mean is not None else float('nan'),
            "baseline_std": self.baseline_std if self.baseline_std is not None else float('nan'),
            "mean_baseline_deviation": safe_mean(self.baseline_deviation_history),
            "max_baseline_deviation": float(np.max(self.baseline_deviation_history)) if len(self.baseline_deviation_history) else float('nan')
        }
