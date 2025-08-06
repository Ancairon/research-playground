import argparse
import time

import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pycaret.time_series import TSForecastingExperiment

def getDataFromAPI(ip, context, dimension, seconds_back):
    # … exactly as before …
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
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.asfreq('s').ffill()
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Live forecasting with full refit")
    parser.add_argument('--ip',         default='localhost')
    parser.add_argument('--context',    default='system.cpu')
    parser.add_argument('--dimension',  default='user')
    parser.add_argument('--window',     type=int, default=300)
    parser.add_argument('--horizon',    type=int, default=10)
    parser.add_argument('--model_path', default='multi_step_forecast_model')
    parser.add_argument('--sleep',      type=float, default=1.0)
    args = parser.parse_args()

    # 1) load the saved pipeline
    exp      = TSForecastingExperiment()
    pipeline = exp.load_model(args.model_path)

    # 2) prepare live plot
    plt.ion()
    fig, ax = plt.subplots()

    try:
        while True:
            # fetch latest data
            df = getDataFromAPI(
                args.ip, args.context, args.dimension, args.window
            )
            ts = df['value']

            # *** FULL REFIT ***
            pipeline.fit(ts)

            # forecast
            fh = list(range(1, args.horizon + 1))
            fc = pipeline.predict(fh=fh)

            # print & plot as before…
            print("\nForecast:")
            for ts_ts, yhat in fc.items():
                print(f"  {ts_ts.strftime('%H:%M:%S')} → {yhat:.4f}")

            recent = ts.iloc[-10:]
            ax.clear()
            ax.plot(recent.index,  recent.values, 'b-', label='Actual')
            ax.plot(fc.index,      fc.values,    'r--o',
                    label=f'Forecast ({args.horizon}s)')
            all_times = recent.index.union(fc.index)
            ax.set_xlim(all_times.min(), all_times.max())
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()
            ymin, ymax = recent.min(), recent.max()
            ax.set_ylim(0, 100)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\nLive forecasting stopped by user.")

if __name__ == '__main__':
    main()
