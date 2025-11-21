# multi_rf_model.py
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def _safe_mean(arr):
    return float(np.mean(arr)) if len(arr) else float('nan')

def _safe_max(arr):
    return float(np.max(arr)) if len(arr) else float('nan')

def _safe_min(arr):
    return float(np.min(arr)) if len(arr) else float('nan')

def _safe_pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float('nan')


def getDataFromAPI(ip, context, dimension, seconds_back):
    """
    Fetch per-second data from Netdata API v3 and return a DataFrame
    with a datetime index and a single 'value' column.
    """
    url = (
        f"http://{ip}:19999/api/v3/data?"
        f"contexts={context}&dimensions={dimension}&"
        f"after=-{seconds_back}&before=0&points={seconds_back}&"
        f"group=average&gtime=0&tier=0&format=json&options=seconds,jsonwrap"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    rows = resp.json()['result']['data']
    df = pd.DataFrame(rows, columns=['timestamp', 'value'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    return df.asfreq('s').ffill()


def train_rf_multi(df, lag, horizon):
    """
    Train multi-output Random Forest model on lag features predicting horizon targets.
    """
    data = pd.DataFrame(index=df.index)
    
    # Create lag features
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = df['value'].shift(i)
    
    # Create target features (future values)
    for h in range(1, horizon + 1):
        data[f't+{h}'] = df['value'].shift(-h)
    
    # Remove NaN rows
    data = data.dropna()
    
    if len(data) == 0:
        raise ValueError("No valid data after creating lag features")
    
    X = data[[f'lag_{i}' for i in range(1, lag + 1)]].values
    y = data[[f't+{h}' for h in range(1, horizon + 1)]].values
    
    base = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X, y)
    return model


def forecast_multi(df, model, lag, horizon):
    """
    Generate forecasts using the trained multi-output model.
    """
    try:
        recent = df['value'].iloc[-lag:].values
        if len(recent) < lag:
            # Pad with last value if needed
            recent = np.pad(recent, (lag - len(recent), 0), mode='edge')
        
        X_pred = recent[::-1].reshape(1, -1)  # Reverse for proper lag order
        preds = model.predict(X_pred)[0]  # Get predictions for all horizons
        
        # Create timestamps for predictions
        last_time = df.index[-1]
        future_times = [last_time + pd.Timedelta(seconds=i) for i in range(1, horizon + 1)]
        
        return preds, future_times
    except Exception as e:
        print(f"Forecast failed: {e}, using last value")
        last_val = df['value'].iloc[-1]
        last_time = df.index[-1]
        future_times = [last_time + pd.Timedelta(seconds=i) for i in range(1, horizon + 1)]
        return [last_val] * horizon, future_times


def main(
        csv=None,
        ip=None,
        context=None,
        dimension=None,
        lag=10,
        horizon=10,
        test_size=0.3,
        random_state=42,
        window=3600,
        target_col="value",
        date_col="timestamp",
        retrain_threshold=5,
        max_errors=1,
        quiet=False
):
    np.random.seed(random_state)
    
    if csv:
        df = pd.read_csv(csv)
        date, target = date_col, target_col
        if np.issubdtype(df[date].dtype, np.number):
            df[date] = pd.to_datetime(df[date], unit='s')
        else:
            df[date] = pd.to_datetime(df[date])
        df.set_index(date, inplace=True)

        window = min(window, len(df))
        feed_df = df[[target]].rename(columns={target: 'value'})
        
        # Use train_test_split like other models but start feed_ptr from window
        train_size = int(window * (1 - test_size))
        feed_ptr = window  # Start from window like other models
        
        # Initial training on training portion only
        initial_data = feed_df[:train_size]
    else:
        df = getDataFromAPI(ip, context, dimension, window)
        df = df.rename(columns={'value': target_col})
        initial_data = df

    # Initial training
    start_train = time.time()
    model = train_rf_multi(initial_data, lag, horizon)
    train_time = time.time() - start_train
    if not quiet:
        print(f"Initial Multi-RF training time: {train_time:.3f}s")

    # Initialize plotting only if not quiet
    if not quiet:
        plt.ion()
        fig, ax = plt.subplots()
    errors = deque(maxlen=max_errors)
    prev_pred = None
    holdback = 0
    result_avg_err = []
    signed_errors = []
    y_true_list = []
    y_pred_list = []
    
    # Timing tracking
    training_times = [train_time]
    inference_times = []

    while True:
        if csv:
            if feed_ptr > len(feed_df):
                if not quiet:
                    print("End of simulation feed.")
                break
            live_df = feed_df.iloc[:feed_ptr]
            feed_ptr += 1
        else:
            live_df = getDataFromAPI(ip, context, dimension, window)

        if prev_pred is not None:
            y_true_now = live_df['value'].iloc[-1]
            err_signed = y_true_now - prev_pred
            signed_errors.append(err_signed)
            y_true_list.append(y_true_now)
            y_pred_list.append(prev_pred)
            err = abs(err_signed)
            errors.append(err)
            
            if retrain_threshold is not None and len(errors) == errors.maxlen:
                avg_err = np.mean(errors)
                if not quiet:
                    print("AVERAGE_ERROR", avg_err, holdback)
                result_avg_err.append(avg_err)
                if avg_err > retrain_threshold:
                    if not quiet:
                        print(f"\n\nRetraining Multi-RF model, avg err={avg_err:.4f}")
                    holdback += 1
                    if holdback == 5:
                        holdback = 0
                        if not quiet:  
                            print("Major anomaly Holdback for 5 seconds before retrain")
                        time.sleep(5)
                    start_retrain = time.time()
                    try:
                        model = train_rf_multi(live_df, lag, horizon)
                        retrain_time = time.time() - start_retrain
                        training_times.append(retrain_time)
                        if not quiet:
                            print(f"Retraining time: {retrain_time:.3f}s\n")
                        time.sleep(1)
                        errors.clear()
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Retraining failed: {e}")
                else:
                    if not quiet:
                        print("zeroing the holdback")
                    holdback = 0

        # Make forecast
        start_inf = time.time()
        preds, future_times = forecast_multi(live_df, model, lag, horizon)
        inf_time = time.time() - start_inf
        inference_times.append(inf_time)
        if not quiet:
            print(f"Inference time: {inf_time:.3f}s")
        prev_pred = preds[0] if len(preds) > 0 else None

        # Plotting only if not quiet
        if not quiet:
            recent = live_df['value'].iloc[-20:]
            times = live_df.index[-20:]

            ax.clear()
            ax.plot(times, recent, 'b-', label='Actual')
            ax.plot(future_times, preds, 'r--o', label='Multi-RF Forecast')
            ax.set_xlim(times[0], future_times[-1])
            ax.set_ylim(0, 100)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()
            if not quiet:  # Only sleep in interactive mode
                time.sleep(1)

    # Calculate final metrics
    avg_mean = _safe_mean(result_avg_err)
    avg_max = _safe_max(result_avg_err)
    avg_min = _safe_min(result_avg_err)
    p80 = _safe_pct(result_avg_err, 80)
    p95 = _safe_pct(result_avg_err, 95)
    p99 = _safe_pct(result_avg_err, 99)

    y_true_arr = np.array(y_true_list, dtype=float)
    y_pred_arr = np.array(y_pred_list, dtype=float)
    mbe = float(np.mean(y_true_arr - y_pred_arr)
                ) if len(y_true_arr) else float('nan')
    pbias = (
        float(100.0 * np.sum(y_true_arr - y_pred_arr) / np.sum(y_true_arr))
        if len(y_true_arr) and np.sum(y_true_arr) != 0 else float('nan')
    )

    # Calculate average timing metrics
    avg_training_time = _safe_mean(training_times)
    avg_inference_time = _safe_mean(inference_times)

    if not quiet:
        print("AVERAGE", avg_mean, "MAX", avg_max, "MIN", avg_min)
        print(f"P80 {p80}  P95 {p95}  P99 {p99}")
        print(f"MBE {mbe}  PBIAS% {pbias}")
        print(f"Avg Training Time: {avg_training_time:.3f}s")
        print(f"Avg Inference Time: {avg_inference_time:.3f}s")

    return {
        "mean_avg_err": avg_mean,
        "max_avg_err": avg_max,
        "min_avg_err": avg_min,
        "p80_avg_err": p80,
        "p95_avg_err": p95,
        "p99_avg_err": p99,
        "mbe": mbe,
        "pbias_pct": pbias,
        "avg_training_time": avg_training_time,
        "avg_inference_time": avg_inference_time
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='CSV file to use instead of API')
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--context', default='system.cpu')
    parser.add_argument('--dimension', default='user')
    parser.add_argument('--lag', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--window', type=int, default=3600)
    parser.add_argument('--target_col', default='value')
    parser.add_argument('--date_col', default='timestamp')
    parser.add_argument('--retrain_threshold', type=float, default=5)
    parser.add_argument('--max_errors', type=int, default=1)
    parser.add_argument('--quiet', action='store_true', help='Suppress output for faster execution')
    
    args = parser.parse_args()
    
    main(
        csv=args.csv,
        ip=args.ip,
        context=args.context,
        dimension=args.dimension,
        lag=args.lag,
        horizon=args.horizon,
        test_size=args.test_size,
        random_state=args.random_state,
        window=args.window,
        target_col=args.target_col,
        date_col=args.date_col,
        retrain_threshold=args.retrain_threshold,
        max_errors=args.max_errors,
        quiet=args.quiet
    )
