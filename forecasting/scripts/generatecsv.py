from matplotlib.font_manager import json_dump
import pandas as pd
import requests
import os


IP = '192.168.1.27'                # Netdata server hostname/IP
CONTEXT = 'system.cpu'            # Single context/chart to fetch
DIMENSION = 'user'              # Dimension key within the context
TIME_WINDOW = 5000              # Seconds of history to fetch for training
MODEL_OUT = 'rf_model.pkl'      # Path to save/load the trained model


def getDataFromAPI(ip, context, dimension, seconds_back):
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
    df.to_csv(f"5000secs_pattern_rpi.csv")
    return df


getDataFromAPI(IP, CONTEXT, DIMENSION, TIME_WINDOW)
