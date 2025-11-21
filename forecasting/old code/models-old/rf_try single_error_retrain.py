# rf_forecast.py

import argparse
import time

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


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
    data = resp.json()['result']['data']
    df = pd.DataFrame(data, columns=['timestamp', 'value'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


def train_rf_multi(df, lag, horizon):
    """
    From df['value'], build lag_1..lag_lag features and t+1..t+horizon targets.
    Returns a MultiOutputRegressor(RandomForest) fitted on non-NaN rows.
    """
    data = pd.DataFrame(index=df.index)
    for i in range(1, lag+1):
        data[f'lag_{i}'] = df['value'].shift(i)
    for h in range(1, horizon+1):
        data[f't+{h}'] = df['value'].shift(-h)
    data = data.dropna()
    data = data.asfreq('s').ffill()
    X = data[[f'lag_{i}' for i in range(1, lag+1)]].values
    y = data[[f't+{h}' for h in range(1, horizon+1)]].values
    base = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X, y)
    return model


def forecast_multi(df, model, lag, horizon):
    """
    Given the full df, take the last `lag` raw values, reverse them into
    a feature vector, and predict the next `horizon` steps.
    Returns a pd.Series indexed at future timestamps.
    """
    recent = df['value'].iloc[-lag:].values[::-1].reshape(1, -1)
    preds = model.predict(recent)[0]  # shape = (horizon,)
    last_t = df.index[-1]
    times = [last_t + pd.Timedelta(seconds=i) for i in range(1, horizon+1)]
    return pd.Series(preds, index=times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ip',        default='localhost')
    p.add_argument('--context',   default='system.cpu')
    p.add_argument('--dimension', default='user')
    p.add_argument('--window',    type=int, default=300,
                   help='seconds of history to keep')
    p.add_argument('--lag',       type=int, default=10,
                   help='number of lag seconds as features')
    p.add_argument('--horizon',   type=int, default=3,
                   help='how many steps ahead to forecast')
    p.add_argument('--sleep',     type=float, default=1.0,
                   help='seconds between updates')
    args = p.parse_args()

    df = getDataFromAPI(args.ip, args.context, args.dimension, args.window)
    model = train_rf_multi(df, args.lag, args.horizon)
    plt.ion()
    fig, ax = plt.subplots()
    prev_pred1 = None
    retrain = False
    prev_pred_ts = None
    try:
        while True:
            new = getDataFromAPI(args.ip, args.context, args.dimension, 1)
            df = pd.concat([df, new]).iloc[-args.window:]
            df = df[~df.index.duplicated(keep='last')]
            if retrain:
                print("retrain")
                model = train_rf_multi(df, args.lag, args.horizon)

            fc = forecast_multi(df, model, args.lag, args.horizon)

            actual1 = df['value'].iloc[-1]
            pred1 = fc.iloc[0]
            pred_ts = fc.index[0]

            if prev_pred1 is not None:
                actual_at_pred = df['value'].asof(prev_pred_ts)
                err1 = abs(prev_pred1 - actual_at_pred)
                retrain = err1 > 25
                print(f"{prev_pred_ts:%H:%M:%S}  "
                      f"pred={prev_pred1:.3f}  "
                      f"actual={actual_at_pred:.3f}  "
                      f"err={err1:.3f}  "
                      f"{'RETRAIN' if retrain else ''}")

            prev_pred1 = pred1
            prev_pred_ts = pred_ts

            roll_start = df.index[-1] - pd.Timedelta(seconds=10)
            df_roll = df.loc[roll_start:, 'value']

            ax.clear()
            ax.plot(df_roll.index, df_roll.values, 'b-', label='Actual')
            ax.plot(fc.index,      fc.values,      'r--o',
                    label=f'Forecast ({args.horizon}s)')
            ax.set_xlim(df_roll.index.min(), fc.index[-1])
            ax.set_ylim(0, 100)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == '__main__':
    main()
