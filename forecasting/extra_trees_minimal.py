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

# -------------------------
# Script Arguments
# -------------------------
parser = argparse.ArgumentParser(
    description="Minimal ExtraTrees TS forecaster with lag features, live plotting, and adaptive retraining"
)
parser.add_argument('--csv',              required=True, help='CSV file path')
parser.add_argument('--date_col',         default='ts',
                    help='Timestamp column name')
parser.add_argument('--target_col',       default='value',
                    help='Value column name')
parser.add_argument('--max_lag',          type=int,
                    default=10,   help='Number of lag features')
parser.add_argument('--horizon',          type=int, default=1,
                    help='Forecast horizon (steps ahead)')
parser.add_argument('--test_size',        type=float,
                    default=0.2, help='Test set fraction')
parser.add_argument('--n_estimators',     type=int,
                    default=100,  help='ExtraTrees n_estimators')
parser.add_argument('--random_state',     type=int,
                    default=42,   help='Random seed')
parser.add_argument('--ip',               default='localhost',
                    help='Netdata server IP')
parser.add_argument('--context',          default='system.cpu',
                    help='Netdata chart context')
parser.add_argument('--dimension',        default=None,
                    help='Netdata dimension key; defaults to target_col')
parser.add_argument('--window',           type=int, default=None,
                    help='Seconds of history to fetch (>= max_lag)')
parser.add_argument('--retrain_threshold', type=float, default=None,
                    help='Average one-step error threshold to trigger retrain')
parser.add_argument('--max_errors', type=int, default=1,
                    help='errors to allow before gathering average.')
args = parser.parse_args()

# Set defaults
if args.dimension is None:
    args.dimension = args.target_col
if args.window is None or args.window < args.max_lag:
    args.window = args.max_lag

# Utility to fetch live data


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


# 1) Load CSV and index
df = pd.read_csv(args.csv)
date, target = args.date_col, args.target_col
if np.issubdtype(df[date].dtype, np.number):
    df[date] = pd.to_datetime(df[date], unit='s')
else:
    df[date] = pd.to_datetime(df[date])
df.set_index(date, inplace=True)

# 2) Generate lag features for horizon forecasting
max_lag, horizon = args.max_lag, args.horizon
for lag in range(1, max_lag + horizon):
    df[f'lag_{lag}'] = df[target].shift(lag)
df.dropna(inplace=True)

# 3) Align X and y
X_all = df[[f'lag_{i}' for i in range(1, max_lag+1)]]
y_all = df[target].shift(- (horizon - 1))
data = pd.concat([X_all, y_all.rename('target')], axis=1).dropna()
X, y = data[[f'lag_{i}' for i in range(1, max_lag+1)]], data['target']

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, shuffle=False
)

# 5) Fit model with timing
model = ExtraTreesRegressor(
    n_estimators=args.n_estimators,
    random_state=args.random_state,
    n_jobs=-1
)
start_train = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_train
print(f"Initial training time: {train_time:.3f}s")

# 6) Evaluate metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
smape = np.mean(2 * np.abs(y_pred - y_test) /
                (np.abs(y_test)+np.abs(y_pred))) * 100
naive_err = np.mean(np.abs(y_train[1:].values - y_train[:-1].values))
mase = mae / naive_err
rmsse = np.sqrt(np.mean(((y_test-y_pred)/naive_err)**2))
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%")
print(f"MASE: {mase:.4f}, RMSSE: {rmsse:.4f}, R2: {r2:.4f}")

# 7) Live streaming forecast + adaptive retrain + live plotting
plt.ion()
fig, ax = plt.subplots()
errors = deque(maxlen=args.max_errors)
prev_pred = None
holdback = 0
while True:
    # Fetch live data
    live_df = getDataFromAPI(args.ip, args.context,
                             args.dimension, args.window)

    # One-step error and retraining
    if prev_pred is not None:
        err = abs(live_df['value'].iloc[-1] - prev_pred)
        errors.append(err)
        if args.retrain_threshold is not None and len(errors) == errors.maxlen:
            avg_err = np.mean(errors)
            print("AVERAGE_ERROR", avg_err, holdback)
            if avg_err > args.retrain_threshold:
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
                data2 = pd.concat([XA, yA.rename('target')], axis=1).dropna()
                model.fit(
                    data2[[f'lag_{i}' for i in range(1, max_lag+1)]], data2['target'])
                retrain_time = time.time() - start_retrain
                print(f"Retraining time: {retrain_time:.3f}s\n\n")
                # pause briefly after retraining before next inference
                time.sleep(1)  # adjust delay as needed
                errors.clear()
            if avg_err < args.retrain_threshold:
                print("zeroing the holdback")
                holdback = 0

    # Forecast with timing
    start_inf = time.time()
    hist_vals = deque(live_df['value'].iloc[-max_lag:], maxlen=max_lag)
    preds = []
    for _ in range(horizon):
        feat = list(hist_vals)[::-1]
        Xf = pd.DataFrame(
            [feat], columns=[f'lag_{i}' for i in range(1, max_lag+1)])
        yhat = model.predict(Xf)[0]
        preds.append(yhat)
        hist_vals.append(yhat)
    inf_time = time.time() - start_inf
    print(f"Inference time: {inf_time:.3f}s")
    prev_pred = preds[0] if preds else None

    # Plot only recent actuals and forecast
    recent = live_df['value'].iloc[-max_lag:]
    times = live_df.index[-max_lag:]
    future = [live_df.index[-1] +
              pd.Timedelta(seconds=i+1) for i in range(horizon)]

    ax.clear()
    ax.plot(times, recent, 'b-', label='Actual')
    ax.plot(future, preds, 'r-', label='Forecast')
    ax.set_xlim(times[0], future[-1])
    ax.set_ylim(0, 100)  # fix y-axis limits as per user
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)
