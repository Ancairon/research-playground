#!/usr/bin/env python3
"""
Helper script to convert CSV to JSON payload for forecasting endpoint.

Usage: python3 csv_to_payload.py <csv_file>

Creates <csv_file>.json with data array containing timestamps and values.
"""

import sys
import pandas as pd
import json


def csv_to_payload(csv_file):
    df = pd.read_csv(csv_file, parse_dates=['timestamp'])

    data = []
    for _, row in df.iterrows():
        data.append({
            "timestamp": row['timestamp'].isoformat(),
            "value": row['value']
        })

    payload = {
        "data": data
    }

    output_file = csv_file + '.json'
    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Created {output_file}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 csv_to_payload.py <csv_file>")
        sys.exit(1)

    csv_to_payload(sys.argv[1])
