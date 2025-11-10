from matplotlib.font_manager import json_dump
import pandas as pd
import requests
import os


def getDataFromAPI(ip, context, dimension, points, out_file: str | None = None):
    """
    Fetches raw JSON from Netdata API v3 using contexts and dimensions parameters,
    and returns a DataFrame with a datetime index and a 'value' column.
    """
    url = (
        f"http://{ip}:19999/api/v3/data?"
        f"contexts={context}&"
        f"dimensions={dimension}&"
        f"after=-{points}&before=0&"
        f"points={points}&group=average&"
        f"gtime=0&format=json&options=minutes,jsonwrap"
    )
    r = requests.get(url, timeout=300)

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
        out_file = f"{points}secs_pattern_rpi.csv"
    df.to_csv(out_file)
    return df


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='Fetch Netdata context/dimension into CSV')
    p.add_argument('--ip', default='192.168.1.27', help='Netdata server IP/host')
    p.add_argument('--context', default='system.cpu', help='Netdata context/chart id')
    p.add_argument('--dimension', default='user', help='Dimension to extract')
    p.add_argument('--points', type=int, default=5000, help='points of history to fetch')
    p.add_argument('--out', default=None, help='Output CSV path')
    args = p.parse_args(argv)
    return getDataFromAPI(args.ip, args.context, args.dimension, args.points, out_file=args.out)


if __name__ == '__main__':
    # when run as script, call main()
    main()
