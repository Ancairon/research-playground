from matplotlib.font_manager import json_dump
import pandas as pd
import requests
import os
from dotenv import load_dotenv


def getDataFromAPI(ip, context, dimension, points, out_file: str | None = None, smooth_window: int | None = None, bearer_token: str | None = None, use_https: bool = False):
    """
    Fetches raw JSON from Netdata API v3 using contexts and dimensions parameters,
    and returns a DataFrame with a datetime index and a 'value' column.
    
    Args:
        smooth_window: If provided, applies rolling average smoothing with this window size
        bearer_token: If provided, adds Authorization header with Bearer token
        use_https: If True, use https:// instead of http://
    """
    protocol = "https" if use_https else "http"
    port = "" if use_https else ":19999"
    url = (
                f"{protocol}://{ip}{port}/api/v3/data?"
                f"contexts={context}&"
                f"dimensions={dimension}&"
                f"after=-{60*60*24*7*4*11}&before=0&"
                f"points={points}&"
                f"options=seconds,jsonwrap&"
                f"format=json2"
    )
    print(url)
    
    headers = {}
    if bearer_token:
        headers['X-Netdata-auth'] = f'Bearer {bearer_token}'
        # Mask token for security in debug output
        masked_token = bearer_token[:4] + '...' + bearer_token[-4:] if len(bearer_token) > 8 else '***'
        print(f"Using Bearer token authentication (token: {masked_token}, length: {len(bearer_token)})")
    
    try:
        
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()  # Raise exception for 4xx/5xx status codes
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.text[:500]}")  # Print first 500 chars of response
        raise
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")
        raise
    except requests.exceptions.Timeout as e:
        print(f"Timeout Error: {e}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        raise

    try:
        json_response = r.json()
        labels = json_response['result']['labels']
        data = json_response['result']['data']
    except (KeyError, ValueError) as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response content: {r.text[:500]}")
        raise

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
    
    # Load .env file for bearer token
    load_dotenv()
    
    p = argparse.ArgumentParser(description='Fetch Netdata context/dimension into CSV')
    p.add_argument('--ip', default='192.168.1.27', help='Netdata server IP/host')
    p.add_argument('--context', default='system.cpu', help='Netdata context/chart id')
    p.add_argument('--dimension', default='user', help='Dimension to extract')
    p.add_argument('--points', type=int, default=5000, help='points of history to fetch')
    p.add_argument('--out', default=None, help='Output CSV path')
    p.add_argument('--smooth', type=int, default=None, help='Apply rolling average smoothing with this window size')
    p.add_argument('--authentication-bearer', action='store_true', 
                   help='Use Bearer token authentication from BEARER_TOKEN in .env file')
    p.add_argument('--https', action='store_true',
                   help='Use HTTPS instead of HTTP (no port suffix)')
    args = p.parse_args(argv)
    
    # Get bearer token from .env if --authentication-bearer is used
    bearer_token = None
    if args.authentication_bearer:
        bearer_token = os.environ.get('BEARER_TOKEN')
        if not bearer_token:
            print("Warning: --authentication-bearer specified but BEARER_TOKEN not found in .env file")
    
    return getDataFromAPI(args.ip, args.context, args.dimension, args.points, 
                          out_file=args.out, smooth_window=args.smooth,
                          bearer_token=bearer_token, use_https=args.https)


if __name__ == '__main__':
    # when run as script, call main()
    main()
