# main.py
import pandas as pd
import requests
import time
import argparse
from arima_model import train_model as arima_train, forecast as arima_forecast
from hw_model import train_model as hw_train, forecast as hw_forecast
import matplotlib.pyplot as plt

def getDataFromAPI(ip, context, dimension, seconds_back):
    """Fetch per-second data from Netdata API v3 and return a DataFrame indexed by timestamp."""
    url = (
        f"http://{ip}:19999/api/v3/data?"
        f"contexts={context}&dimensions={dimension}&"
        f"after=-{seconds_back}&before=0&points={seconds_back}&"
        f"group=average&gtime=0&tier=0&format=json&options=seconds,jsonwrap"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    rows = resp.json().get('result', {}).get('data', [])
    df = pd.DataFrame([{'timestamp': ts, 'value': val} for ts, val in rows])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    # ensure a regular per-second index; drop duplicates if any
    df = df[~df.index.duplicated(keep='last')]
    df = df.asfreq('s')
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Live multi-step forecasting of per-second Netdata metrics')
    parser.add_argument('--ip',        default='localhost')
    parser.add_argument('--context',   default='system.cpu')
    parser.add_argument('--dimension', default='user')
    parser.add_argument('--window',    type=int,   default=3600,
                        help='History window length in seconds')
    parser.add_argument('--horizon',   type=int,   default=1,
                        help='How many steps ahead to forecast')
    parser.add_argument('--sleep',     type=float, default=1.0,
                        help='Seconds between iterations')
    parser.add_argument('--model',     choices=['arima','hw'], default='hw',
                        help='Which forecasting model to use')
    args = parser.parse_args()

    if args.model == 'arima':
        train = arima_train
        forecast = arima_forecast
    else:
        train = hw_train
        forecast = hw_forecast

    df    = getDataFromAPI(args.ip, args.context, args.dimension, args.window)
    model = train(df)            # initial model on full history

    plt.ion()
    fig, ax = plt.subplots()

    try:
        while True:
            H = args.horizon

            # back‐cast: train on df up to t–H and forecast the last H points
            df_train = df.iloc[:-H]               # all but last H samples
            fc       = forecast(df_train, model, H)
            # These fc.index values = the timestamps of df.iloc[-H:]

            # actuals from those same last H timestamps
            actuals = df['value'].iloc[-H:]

            errors = (fc - actuals).abs()
            print("\nTime     Forecast   Actual   AbsError")
            for t in fc.index:
                print(f"{t:%H:%M:%S}   {fc[t]:8.4f}   {actuals[t]:8.4f}   {errors[t]:8.4f}")

            # append the newest live point to df, drop oldest to maintain window
            live_pt = getDataFromAPI(args.ip, args.context, args.dimension, 1)
            df = pd.concat([df, live_pt]).iloc[-args.window:]
            df = df[~df.index.duplicated(keep='last')]

            # retrain model on the updated full history
            model = train(df)

            # plot last 10 s of actuals + the back‐cast
            roll_start = df.index[-1] - pd.Timedelta(seconds=10)
            df_roll    = df[df.index >= roll_start]

            ax.clear()
            ax.plot(df_roll.index, df_roll['value'], label='Actual', color='blue')
            h_times  = df_roll.index[-H:]
            h_values = df_roll['value'].iloc[-H:]
            ax.scatter(
                h_times,
                h_values,
                color='orange',
                edgecolor='k',
                s=80,
                zorder=5,
                label=f'Actual (last {H}s)'
            )
            ax.plot(fc.index, fc.values, 'r--o', label='Forecast')
            ax.legend()
            ax.set_xlim(df_roll.index.min(), fc.index[-1])
            y_min = min(df_roll['value'].min(), fc.min())
            y_max = max(df_roll['value'].max(), fc.max())
            ax.set_ylim(0,100)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            fig.autofmt_xdate()
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting.")
if __name__ == '__main__':
    main()
