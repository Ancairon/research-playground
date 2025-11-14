from matplotlib.font_manager import json_dump
import pandas as pd
import requests
import os


def getDataFromAPI(ip, context, dimension, points, out_file: str | None = None, smooth_window: int | None = None):
    """
    Fetches raw JSON from Netdata API v3 using contexts and dimensions parameters,
    and returns a DataFrame with a datetime index and a 'value' column.
    
    Args:
        smooth_window: If provided, applies rolling average smoothing with this window size
    """
    url = (
                f"http://{ip}:19999/api/v3/data?"
                f"contexts={context}&"
                f"dimensions={dimension}&"
                f"after=-{60*60*24*7*4*3}&before=0&"
                f"points={points}&"
                f"options=seconds,jsonwrap&"
                f"format=json2"
    )
    r = requests.get(url, timeout=30)

    labels = r.json()['result']['labels']
    data = r.json()['result']['data']

    print(labels)

    records = []
    for row in data:
        ts = row[0]
        val = row[1][0]  # single dimension
        records.append({'timestamp': ts, 'value': val})

    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    print(df)
    df = df.sort_values(by='timestamp', ascending=True)
    
    # Check for missing values and fill them
    missing_count = df['value'].isna().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Filling with interpolation...")
        # First try linear interpolation (uses before and after points)
        df['value'] = df['value'].interpolate(method='linear', limit_direction='both')
        # If any NaN remain at edges, use forward/backward fill
        df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')
        print(f"Missing values filled successfully.")
    else:
        print("No missing values found.")
    
    # Apply smoothing if requested
    if smooth_window and smooth_window > 1:
        print(f"Applying rolling average smoothing (window={smooth_window})...")
        original_mean = df['value'].mean()
        df['value'] = df['value'].rolling(window=smooth_window, center=True, min_periods=1).mean()
        smoothed_mean = df['value'].mean()
        print(f"  Original mean: {original_mean:.4f}, Smoothed mean: {smoothed_mean:.4f}")
        print(f"Smoothing applied successfully.")
    
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
    p.add_argument('--smooth', type=int, default=None, help='Apply rolling average smoothing with this window size')
    args = p.parse_args(argv)
    return getDataFromAPI(args.ip, args.context, args.dimension, args.points, out_file=args.out, smooth_window=args.smooth)


if __name__ == '__main__':
    # when run as script, call main()
    main()
