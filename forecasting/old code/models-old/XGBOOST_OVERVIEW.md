# XGBoost Forecaster — Consolidated Overview

This document consolidates the implementation notes, validation timeline, printing rules,
and retraining/MAD explanation for the XGBoost live forecaster into a single reference.

## Summary

- Validation is deferred: predictions are stored and validated once actuals exist.
- One pending prediction is validated per main loop iteration (prevents duplicate prints).
- Error printing is configurable via the `print_min_validations` parameter (default 3).
- Retraining threshold uses MAD (median absolute deviation) by default; configurable via `retrain_use_mad`.

## What changed (high level)

- Before: code printed 1-step errors and sometimes printed multiple times per second.
- Now: full-horizon validation (e.g., horizon=5) — compute MAPE across the full horizon.
- Predictions are queued in `pending_validations` and validated only after horizon seconds have elapsed.
- Only one pending prediction is validated per step, preventing duplicate prints.

## Key concepts

- pending_validations: deque of (creation_step, last_actual_ts, [pred1..predH])
- horizon: number of steps ahead to forecast (h)
- validation timing: prediction at step N is validated when current step >= N + horizon (due to step increment, validation occurs around N+6 when horizon=5)
- print_min_validations: minimum successful horizon validations before printing the rolling average (default 3)

## Validation algorithm (concise)

1. When forecasting at step N, append (N, last_actual_ts, preds) to `pending_validations`.
2. Increment `step`.
3. If the oldest pending entry has elapsed >= horizon, pop it and validate:
   - For each prediction i (0..h-1), compute the target timestamp as current_ts - (steps_elapsed - (i+1)) seconds (look backward in live_df).
   - If all h actuals are found, compute the per-step absolute percentage errors and mean them → mean_horizon_error.
   - Append mean_horizon_error to `errors` and `recent_errors` deques.
4. If len(errors) >= print_min_validations, print the rolling average of `errors`.
5. Compute retraining threshold from `recent_errors` using MAD (or std) and retrain if conditions (consec_count, retrain_cooldown) are met.

## Timeline (example, horizon=5)

- T0..T4: predictions are produced and queued (no validation yet).
- T5: first validation completes (prediction from step 0 validated).
- T6: second validation (step 1).
- T7: third validation (step 2) → if `print_min_validations` == 3, the first print appears here.
- T8+: one validation per step; after the initial warmup, one print per step if threshold satisfied.

Note: because the implementation increments `step` after storing predictions, validation occurs one iteration later than a naive N+h computation; that's the observed step N+6 behavior when horizon=5.

## Printing policy

- Purpose: avoid noisy/unstable early prints. Historically the code used a hardcoded `3`.
- Current behavior: use `print_min_validations` function parameter (default 3) — consistent with other parameters.
- To change printing sensitivity, pass `print_min_validations` to `main(...)`.

Example (python API):

```python
# default
main(..., print_min_validations=3)

# be more conservative
main(..., print_min_validations=5)
```

CLI: the code does not add a CLI flag for this by default; if you want one, I can add `--print-min-validations` to the script.

## Prediction Smoothing

**Problem**: When predicting every second with a small window, predictions can flip-flop dramatically (spike→dip→spike), reducing user trust and potentially triggering unnecessary retrains.

**Solution**: Ensemble averaging of recent predictions (default: last 3 predictions).

### How it works

1. **Raw prediction**: XGBoost generates a raw forecast at each step (e.g., [v1, v2, v3, v4, v5] for horizon=5)
2. **Storage**: Raw predictions are stored in `recent_predictions` deque (maxlen=`prediction_smoothing`)
3. **Smoothing**: For each horizon step i, average all values at position i across recent predictions
4. **Usage**: Smoothed predictions are used for:
   - Validation (error computation)
   - Retraining decisions
   - Display to user
   - Live server updates

### Configuration

- `prediction_smoothing=1`: No smoothing (use raw predictions)
- `prediction_smoothing=3`: Default (average last 3 predictions) — good balance
- `prediction_smoothing=5`: More aggressive smoothing — slower to react but very stable

### Example

```
Step 100: raw=[10, 12, 14, 16, 18]  → stored in recent_predictions
Step 101: raw=[8, 10, 12, 14, 16]   → stored in recent_predictions
Step 102: raw=[9, 11, 13, 15, 17]   → stored in recent_predictions

Smoothed prediction at step 102:
  [mean([10,8,9]), mean([12,10,11]), mean([14,12,13]), mean([16,14,15]), mean([18,16,17])]
  = [9.0, 11.0, 13.0, 15.0, 17.0]
```

### Trade-offs

- **Pro**: Reduces noise, prevents flip-flopping, more stable user experience
- **Pro**: Validation errors are based on what users actually see
- **Con**: Slightly slower to react to genuine trend changes
- **Con**: Requires more memory (stores N recent prediction arrays)

### Recommended settings

- **High noise data**: `prediction_smoothing=5`, `window=120+`
- **Balanced (default)**: `prediction_smoothing=3`, `window=120`
- **Fast response**: `prediction_smoothing=1`, `window=60` (no smoothing)

## Retraining and MAD

- Retraining threshold is computed from recent validation errors.
- By default `retrain_use_mad=True`, which computes MAD-based sigma = 1.4826 * MAD and multiplies by `retrain_scale`.
- `retrain_min` acts as a floor; threshold = max(retrain_min, retrain_scale * sigma).

**Why MAD (not std)?**
- MAD is robust to spikes/outliers (one large error won't inflate threshold)
- When `retrain_use_mad=False`, the code uses standard deviation (std) instead
- **Problem with std**: If errors are percentage values (e.g., 10%, 15%, 20%), the std can be very large
  - Example: std([10%, 15%, 20%]) ≈ 5%
  - With retrain_scale=20.0: threshold = 20.0 × 5% = 100%+
  - Result: threshold becomes 25000%+ (unusable), retraining never triggers
- **Always use MAD** unless you have a very specific reason not to

**CRITICAL: retrain_scale depends on error type**

The `retrain_scale` parameter must be tuned based on whether you're using percentage errors (MAPE) or absolute errors:

| Error Type | retrain_scale | Example Calculation |
|------------|---------------|---------------------|
| **MAPE (%)** | **2.0 - 5.0** | MAD=15% → sigma=22% → threshold=3.0×22%=**66%** |
| Absolute (KB/s, etc) | 10.0 - 20.0 | MAD=10 → sigma=15 → threshold=10.0×15=**150** |

**What happens with wrong retrain_scale:**
- **Too high (20.0 for MAPE)**: Threshold becomes 300-500%+, retraining never happens
  - Example: errors=40-85%, threshold=300-500%, model never retrains
- **Too low (1.0)**: Threshold too sensitive, retrains constantly (model instability)

**Formula breakdown:**
```
recent_errors = [84%, 70%, 43%, 33%, 41%, ...]
MAD = median(|errors - median(errors)|)  # ≈ 15-20% typically
sigma = 1.4826 * MAD                      # ≈ 22-30%
threshold = max(retrain_min, retrain_scale * sigma)

With retrain_scale=20.0: threshold = max(50%, 20×25%) = max(50%, 500%) = 500%  ❌ Too high!
With retrain_scale=3.0:  threshold = max(50%, 3×25%)  = max(50%, 75%)  = 75%   ✅ Reasonable
```

**Recommended values:**
- MAPE errors: `retrain_scale=3.0`, `retrain_min=20.0-50.0%`
- Absolute errors: `retrain_scale=15.0`, `retrain_min=50-100` (in data units)

**Important**: Validation errors are computed from **smoothed predictions** (when `prediction_smoothing > 1`), not raw XGBoost outputs. This ensures retraining decisions are based on what users see.

## Prediction Interval

Controls how often predictions are generated:
- `prediction_interval=1.0`: Predict every second (default, most responsive)
- `prediction_interval=2.0`: Predict every 2 seconds (more stable, less CPU)
- `prediction_interval=5.0`: Predict every 5 seconds (very stable, minimal CPU)

Works with prediction smoothing:
- With `prediction_smoothing=3` and `prediction_interval=2.0`, you average the last 3 predictions made at 2-second intervals (6 seconds of history)
- Slower intervals → more stability but less responsiveness to sudden changes
- Only applies to live mode (not CSV simulation, which runs as fast as possible)

## Reactive vs Predictive Behavior

**Important limitation**: XGBoost is fundamentally **reactive**, not predictive of future unknown events.

### Why Predictions Feel Reactive

XGBoost learns from recent history. It cannot predict:
- Future spikes that haven't occurred yet
- Sudden pattern changes without historical examples
- Events outside the training window

For spiky patterns (e.g., spike every 15s):
- With `window=120` (2 min): Model sees ~8 cycles → struggles to learn pattern
- With `window=300` (5 min): Model sees ~20 cycles → better pattern recognition
- With `window=600` (10 min): Model sees ~40 cycles → good pattern learning

### Improving Pattern Recognition

| Pattern Type | window | smoothing | interval | retrain_scale | Why |
|--------------|--------|-----------|----------|---------------|-----|
| **Smooth (sine)** | 120-180 | 3 | 1.0 | 3.0 | Easy to predict, needs less history |
| **Spiky** | 300-600 | 5-7 | 2.0-3.0 | 2.0-3.0 | Needs many cycles, heavy smoothing |
| **Step changes** | 180-300 | 3-5 | 2.0 | 3.0 | Moderate history, moderate smoothing |
| **Trends** | 300+ | 5 | 2.0 | 2.0 | Long window to track baseline shift |

### What to Expect

✅ **XGBoost can predict:**
- Continuation of current trends
- Regular periodic patterns (after seeing many cycles)
- Gradual changes following recent behavior

❌ **XGBoost cannot predict:**
- Sudden unprecedented spikes/dips
- Pattern changes it hasn't seen before
- Events in the distant future (horizon >> pattern cycle)

**Bottom line**: If predictions feel reactive, increase `window` (see more pattern cycles), increase `prediction_smoothing` (reduce noise), and slow down `prediction_interval` (fewer rapid updates).

## Where the key logic lives

- `forecasting/xgboost_whole.py`
  - `pending_validations` queue
  - `print_min_validations` parameter in `main()`
  - error-printing inside the validation block (printing only when `len(errors) >= print_min_validations`)
  - `compute_threshold_from_errors()` implements MAD/mean thresholding

## Notes for maintainers

- Keep `print_min_validations` as a function parameter to match other config options.
- If you prefer CLI control, I can add an argparse flag and wire it to `main()` defaults.
- Tests you can run locally:

```bash
python3 forecasting/xgboost_whole.py            # default behavior
python3 forecasting/xgboost_whole.py --csv data.csv --quiet
```

## Archive / migration

The following files were merged into this overview and can be removed or archived:

- `HORIZON_ERROR_IMPLEMENTATION.md`
- `PRINT_FREQUENCY_FIX.md`
- `VALIDATION_FIX.md`
- `TIMELINE_EXPLANATION.md`
- `MAD_EXPLANATION.md`
- `FINAL_IMPLEMENTATION.md`

If you'd rather keep these as separate historical artifacts, I can stop short of deleting them and only add this consolidated overview.

---

If this consolidation looks good, I will remove the listed legacy files (or move them to an `archive/` folder instead) — tell me which you prefer.
