import argparse
import time

import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pycaret.time_series import TSForecastingExperiment

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
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.asfreq('s').ffill()
    return df

def main():
    parser = argparse.ArgumentParser("Live XGB forecasting")
    parser.add_argument('--ip',         default='localhost')
    parser.add_argument('--context',    default='system.cpu')
    parser.add_argument('--dimension',  default='user')
    parser.add_argument('--window',     type=int,   default=3600)
    parser.add_argument('--horizon',    type=int,   default=10)
    parser.add_argument('--model_path', default='xgb_forecast_model')
    parser.add_argument('--sleep',      type=float, default=1.0)
    args = parser.parse_args()

    # load your saved XGB pipeline
    exp      = TSForecastingExperiment()
    pipeline = exp.load_model(args.model_path)

    plt.ion()
    fig, ax = plt.subplots()

    try:
        while True:
            # 1) fetch the latest window of raw data
            df = getDataFromAPI(
                args.ip, args.context, args.dimension, args.window
            )
            ts = df['value']

            # 2) convert to an sktime-compatible PeriodIndex at 1s freq
            ts = ts.copy()
            ts.index = pd.PeriodIndex(ts.index, freq='S')

            # 3) update the pipeline (this will refit under the hood for XGB)
            pipeline.update(ts)

            # 4) forecast next H steps
            fh = list(range(1, args.horizon + 1))
            fc = pipeline.predict(fh=fh)   # fc has a PeriodIndex

            # 5) bring forecast back to timestamps for printing/plotting
            fc.index = fc.index.to_timestamp(freq='S')

            # --- print forecast ---
            print("\nForecast:")
            for t, yhat in fc.items():
                print(f"  {t.strftime('%H:%M:%S')} → {yhat:.4f}")

            # --- plot last 10s of actual + forecast ---
            recent = ts.to_timestamp(freq='S').iloc[-10:]
            ax.clear()
            ax.plot(recent.index, recent.values, 'b-', label='Actual')
            ax.plot(fc.index,      fc.values,    'r--o',
                    label=f'Forecast ({args.horizon}s)')

            all_times = recent.index.union(fc.index)
            ax.set_xlim(all_times.min(), all_times.max())
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()

            ymin, ymax = recent.min(), recent.max()
            ax.set_ylim(0,100)
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
