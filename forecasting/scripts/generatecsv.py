from matplotlib.font_manager import json_dump
import pandas as pd
import requests
import os


IP = '192.168.1.27'                # Netdata server hostname/IP
CONTEXT = 'system.cpu'            # Single context/chart to fetch
DIMENSION = 'user'              # Dimension key within the context
TIME_WINDOW = 5000              # Seconds of history to fetch for training
MODEL_OUT = 'rf_model.pkl'      # Path to save/load the trained model


def getDataFromAPI(ip, context, dimension, seconds_back, out_file: str | None = None):
    """
    Fetches raw JSON from Netdata API v3 using contexts and dimensions parameters,
    and returns a DataFrame with a datetime index and a 'value' column.
    """
    url = (
        f"http://{ip}:19999/api/v3/data?"
        f"contexts={context}&"
        f"dimensions={dimension}&"
        f"after=-{seconds_back}&before=0&"
        f"points={seconds_back}&group=average&"
        f"gtime=0&tier=0&format=json&options=seconds,jsonwrap"
    )
    r = requests.get(url, timeout=30)

    labels = r.json()['result']['labels']
    data = r.json()['result']['data']

    print(labels)

    records = []
    for row in data:
        ts = row[0]
        val = row[1]  # single dimension
        records.append({'timestamp': ts, 'value': val})

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    print(df)
    df = df.sort_values(by='timestamp', ascending=True)
    # df.to_csv(f"{context}_{dimension}.csv")
    # allow caller to override output filename; default to <seconds>secs_pattern_rpi.csv
    if out_file is None:
        out_file = f"{seconds_back}secs_pattern_rpi.csv"
    df.to_csv(out_file)
    return df


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='Fetch Netdata context/dimension into CSV')
    p.add_argument('--ip', default=IP, help='Netdata server IP/host')
    p.add_argument('--context', default=CONTEXT, help='Netdata context/chart id')
    p.add_argument('--dimension', default=DIMENSION, help='Dimension to extract')
    p.add_argument('--seconds', type=int, default=TIME_WINDOW, help='Seconds of history to fetch')
    p.add_argument('--out', default=None, help='Output CSV path')
    args = p.parse_args(argv)
    return getDataFromAPI(args.ip, args.context, args.dimension, args.seconds, out_file=args.out)


if __name__ == '__main__':
    # when run as script, call main()
    main()
