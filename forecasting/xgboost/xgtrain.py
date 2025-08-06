# train_xgb_model.py

import argparse
import pandas as pd
import requests

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
    p = argparse.ArgumentParser("Train & save an XGBoost TS pipeline")
    p.add_argument('--ip',        default='localhost')
    p.add_argument('--context',   default='system.cpu')
    p.add_argument('--dimension', default='user')
    p.add_argument('--seconds',   type=int, default=6*3600,
                   help='History length (s)')
    p.add_argument('--horizon',   type=int, default=10)
    p.add_argument('--save_path', default='xgb_forecast_model')
    args = p.parse_args()

    # 1) Pull down your training series
    df = getDataFromAPI(args.ip, args.context, args.dimension, args.seconds)
    y  = df['value']
    y.index = pd.PeriodIndex(y.index, freq='S')  # PyCaret wants a PeriodIndex

    # 2) Setup & train
    exp = TSForecastingExperiment()
    exp.setup(
        data=y,
        fh=args.horizon,
        seasonal_period=60,
        session_id=42
    )

    # 3) Create the built-in XGBoost reduction model
    xgb_pipe = exp.create_model('xgboost_cds_dt')
    final_xgb = exp.finalize_model(xgb_pipe)

    # 4) Save it
    exp.save_model(final_xgb, args.save_path)
    print(f"Saved XGB pipeline to '{args.save_path}.pkl'")

if __name__ == '__main__':
    main()
