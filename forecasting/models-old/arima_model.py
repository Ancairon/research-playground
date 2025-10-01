# arima_model.py
import argparse
import pandas as pd
import numpy as np
from collections import deque
from statsmodels.tsa.arima.model import ARIMA
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description="ARIMA TS forecaster with live plotting and adaptive retraining"
)
parser.add_argument('--csv', help='CSV file path', default=None)
parser.add_argument('--date_col',        default='timestamp',
                    help='Timestamp column name (or index if CSV)')
parser.add_argument('--target_col',      default='value',
                    help='Value column name (or index if CSV)')
parser.add_argument('--order_p',         type=int, default=1,
                    help='ARIMA p parameter (autoregressive order)')
parser.add_argument('--order_d',         type=int, default=1,
                    help='ARIMA d parameter (differencing order)')
parser.add_argument('--order_q',         type=int, default=1,
                    help='ARIMA q parameter (moving average order)')
parser.add_argument('--horizon',         type=int, default=1,
                    help='Forecast horizon (steps ahead)')
parser.add_argument('--test_size',       type=float, default=0.3,
                    help='Test set fraction for simulation')
parser.add_argument('--random_state',    type=int, default=42,
                    help='Random seed')
parser.add_argument('--ip',              default='localhost',
                    help='Netdata server IP')
parser.add_argument('--context',         default='system.cpu',
                    help='Netdata chart context')
parser.add_argument('--dimension',       default=None,
                    help='Netdata dimension key; defaults to target_col')
parser.add_argument('--window',          type=int, default=None,
                    help='Seconds of history to fetch')
parser.add_argument('--retrain_threshold', type=float, default=None,
                    help='Error threshold to trigger retrain')
parser.add_argument('--max_errors',      type=int, default=1,
                    help='Errors to allow before retrain')
args = parser.parse_args()

if args.dimension is None:
    args.dimension = args.target_col
if args.window is None:
    args.window = 3600  # Default 1 hour


def _safe_mean(x): return float(np.mean(x)) if len(x) else float('nan')
def _safe_max(x): return float(np.max(x)) if len(x) else float('nan')
def _safe_min(x): return float(np.min(x)) if len(x) else float('nan')


def _safe_pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float('nan')


def getDataFromAPI(ip, context, dimension, seconds_back):
    url = (
        f"http://{ip}:19999/api/v3/data?"
        f"contexts={context}&dimensions={dimension}&"
        f"after=-{seconds_back}&before=0&points={seconds_back}&"
        f"group=average&format=json&options=seconds,jsonwrap"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    rows = resp.json()['result']['data']
    df = pd.DataFrame(rows, columns=['timestamp', 'value'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    return df.asfreq('s').ffill()


def train_arima_model(series, order=(1, 1, 1)):
    """Train ARIMA model on the given series."""
    try:
        model = ARIMA(series, order=order).fit()
        return model
    except Exception as e:
        print(f"ARIMA training failed: {e}, using simpler model")
        # Fallback to simpler model
        try:
            return ARIMA(series, order=(1, 0, 0)).fit()
        except:
            return ARIMA(series, order=(0, 1, 0)).fit()


def main(
        csv=None,
        ip=None,
        context=None,
        dimension=None,
        order=(1, 1, 1),
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
    order_p, order_d, order_q = order
    
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
        
        # Use train_test_split like ExtraTrees but start feed_ptr from window
        train_size = int(window * (1 - test_size))
        feed_ptr = window  # Start from window like ExtraTrees, not train_size
        
        # Initial training on training portion only
        initial_data = feed_df['value'][:train_size]
    else:
        df = getDataFromAPI(ip, context, dimension, window)
        df = df.rename(columns={'value': target_col})
        initial_data = df[target_col]

    # Initial training
    start_train = time.time()
    model = train_arima_model(initial_data, order=(order_p, order_d, order_q))
    train_time = time.time() - start_train
    if not quiet:
        print(f"Initial ARIMA training time: {train_time:.3f}s")

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
                        print(f"\n\nRetraining ARIMA model, avg err={avg_err:.4f}")
                    holdback += 1
                    if holdback == 5:
                        holdback = 0
                        if not quiet:
                            print("Major anomaly Holdback for 5 seconds before retrain")
                        time.sleep(5)
                    start_retrain = time.time()
                    try:
                        model = train_arima_model(live_df['value'], order=(order_p, order_d, order_q))
                        retrain_time = time.time() - start_retrain
                        training_times.append(retrain_time)
                        if not quiet:
                            print(f"Retraining time: {retrain_time:.3f}s\n")
                        time.sleep(1)
                        errors.clear()
                    except Exception as e:
                        if not quiet:
                            print(f"Retraining failed: {e}")
                else:
                    if not quiet:
                        print("zeroing the holdback")
                    holdback = 0

        # Make forecast
        start_inf = time.time()
        try:
            # Update model with latest data point if possible
            latest_point = live_df['value'].iloc[-1:]
            model = model.append(latest_point, refit=False)
            
            # Forecast
            fc_result = model.forecast(steps=horizon)
            if hasattr(fc_result, 'values'):
                preds = fc_result.values.tolist()
            else:
                preds = fc_result.tolist() if hasattr(fc_result, 'tolist') else [float(fc_result)]
                
        except Exception as e:
            if not quiet:
                print(f"Forecast failed: {e}, using last value")
            preds = [live_df['value'].iloc[-1]] * horizon
            
        inf_time = time.time() - start_inf
        inference_times.append(inf_time)
        if not quiet:
            print(f"Inference time: {inf_time:.3f}s")
        prev_pred = preds[0] if preds else None

        # Plotting only if not quiet
        if not quiet:
            recent = live_df['value'].iloc[-20:]
            times = live_df.index[-20:]
            future = [
                live_df.index[-1] + pd.Timedelta(seconds=i+1)
                for i in range(horizon)
            ]

            ax.clear()
            ax.plot(times,  recent, 'b-', label='Actual')
            ax.plot(future, preds,   'r--o', label='ARIMA Forecast')
            ax.set_xlim(times[0], future[-1])
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
    main(
        csv=args.csv,
        ip=args.ip,
        context=args.context,
        dimension=args.dimension,
        order_p=args.order_p,
        order_d=args.order_d,
        order_q=args.order_q,
        horizon=args.horizon,
        test_size=args.test_size,
        random_state=args.random_state,
        window=args.window,
        target_col=args.target_col,
        date_col=args.date_col,
        retrain_threshold=args.retrain_threshold,
        max_errors=args.max_errors
    )
