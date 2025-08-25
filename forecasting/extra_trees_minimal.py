import argparse
import pandas as pd
import numpy as np
from collections import deque
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

parser = argparse.ArgumentParser(
    description="Minimal ExtraTrees TS forecaster with lag features, live plotting, and adaptive retraining"
)
parser.add_argument('--csv', help='CSV file path', default=None)
parser.add_argument('--date_col',        default='ts',
                    help='Timestamp column name (or index if CSV)')
parser.add_argument('--target_col',      default='value',
                    help='Value column name (or index if CSV)')
parser.add_argument('--max_lag',         type=int, default=10,
                    help='Number of lag features')
parser.add_argument('--horizon',         type=int, default=1,
                    help='Forecast horizon (steps ahead)')
parser.add_argument('--test_size',       type=float, default=0.2,
                    help='Test set fraction')
parser.add_argument('--n_estimators',    type=int, default=100,
                    help='ExtraTrees n_estimators')
parser.add_argument('--random_state',    type=int, default=42,
                    help='Random seed')
parser.add_argument('--ip',              default='localhost',
                    help='Netdata server IP')
parser.add_argument('--context',         default='system.cpu',
                    help='Netdata chart context')
parser.add_argument('--dimension',       default=None,
                    help='Netdata dimension key; defaults to target_col')
parser.add_argument('--window',          type=int, default=None,
                    help='Seconds of history to fetch (>= max_lag)')
parser.add_argument('--retrain_threshold', type=float, default=None,
                    help='Error threshold to trigger retrain')
parser.add_argument('--max_errors',      type=int, default=1,
                    help='Errors to allow before retrain')
args = parser.parse_args()

if args.dimension is None:
    args.dimension = args.target_col
if args.window is None or args.window < args.max_lag:
    args.window = args.max_lag


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


def main(
        csv=None,
        ip=None,
        context=None,
        dimension=None,
        max_lag=10,
        horizon=10,
        test_size=.3,
        n_estimators=222,
        random_state=42,
        window=3600,
        target_col="value",
        date_col="timestamp",
        retrain_threshold=5,
        max_errors=1
):
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
        feed_ptr = window
    else:

        df = getDataFromAPI(ip, context, dimension, window)
        df = df.rename(columns={'value': target_col})

    max_lag, horizon = max_lag, horizon
    for lag in range(1, max_lag + horizon):
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    df.dropna(inplace=True)

    X_all = df[[f'lag_{i}' for i in range(1, max_lag+1)]]
    y_all = df[target_col].shift(- (horizon - 1))
    data = pd.concat([X_all, y_all.rename('target')], axis=1).dropna()
    X, y = data[[f'lag_{i}' for i in range(1, max_lag+1)]], data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"Initial training time: {train_time:.3f}s")

    plt.ion()
    fig, ax = plt.subplots()
    errors = deque(maxlen=max_errors)
    prev_pred = None
    holdback = 0
    result_avg_err = []
    signed_errors = []
    y_true_list = []
    y_pred_list = []

    while True:
        if csv:
            if feed_ptr > len(feed_df):
                print("End of simulation feed.")
                break
            live_df = feed_df.iloc[:feed_ptr]
            feed_ptr += 1
        else:
            live_df = getDataFromAPI(
                ip, context, dimension, window
            )

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
                print("AVERAGE_ERROR", avg_err, holdback)
                result_avg_err.append(avg_err)
                if avg_err > retrain_threshold:
                    print(f"\n\nRetraining model, avg err={avg_err:.4f}")
                    holdback += 1
                    if holdback == 5:
                        holdback = 0
                        print("Major anomaly Holdback for 5 seconds before retrain")
                        time.sleep(5)
                    start_retrain = time.time()
                    df2 = live_df.copy()
                    for lag in range(1, max_lag + horizon):
                        df2[f'lag_{lag}'] = df2['value'].shift(lag)
                    df2.dropna(inplace=True)
                    XA = df2[[f'lag_{i}' for i in range(1, max_lag+1)]]
                    yA = df2['value'].shift(-(horizon-1)).dropna()
                    data2 = pd.concat(
                        [XA, yA.rename('target')], axis=1).dropna()
                    model.fit(
                        data2[[f'lag_{i}' for i in range(1, max_lag+1)]],
                        data2['target']
                    )
                    retrain_time = time.time() - start_retrain
                    print(f"Retraining time: {retrain_time:.3f}s\n")
                    time.sleep(1)
                    errors.clear()
                else:
                    print("zeroing the holdback")
                    holdback = 0

        start_inf = time.time()
        hist_vals = deque(live_df['value'].iloc[-max_lag:], maxlen=max_lag)
        preds = []
        for _ in range(horizon):
            feat = list(hist_vals)[::-1]
            Xf = pd.DataFrame(
                [feat], columns=[f'lag_{i}' for i in range(1, max_lag+1)]
            )
            yhat = model.predict(Xf)[0]
            preds.append(yhat)
            hist_vals.append(yhat)
        inf_time = time.time() - start_inf
        print(f"Inference time: {inf_time:.3f}s")
        prev_pred = preds[0] if preds else None

        recent = live_df['value'].iloc[-max_lag:]
        times = live_df.index[-max_lag:]
        future = [
            live_df.index[-1] + pd.Timedelta(seconds=i+1)
            for i in range(horizon)
        ]

        ax.clear()
        ax.plot(times,  recent, 'b-', label='Actual')
        ax.plot(future, preds,   'r--o', label='Forecast')
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
        time.sleep(1)

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

    print("AVERAGE", avg_mean, "MAX", avg_max, "MIN", avg_min)
    print(f"P80 {p80}  P95 {p95}  P99 {p99}")
    print(f"MBE {mbe}  PBIAS% {pbias}")

    return {
        "mean_avg_err": avg_mean,
        "max_avg_err": avg_max,
        "min_avg_err": avg_min,
        "p80_avg_err": p80,
        "p95_avg_err": p95,
        "p99_avg_err": p99,
        "mbe": mbe,
        "pbias_pct": pbias
    }


if __name__ == '__main__':
    main(
        csv=args.csv,
        ip=args.ip,
        context=args.context,
        dimension=args.dimension,
        max_lag=args.max_lag,
        horizon=args.horizon,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        window=args.window,
        target_col=args.target_col,
        date_col=args.date_col,
        retrain_threshold=args.retrain_threshold,
        max_errors=args.max_errors
    )
