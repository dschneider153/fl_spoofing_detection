import csv
import os
import pandas as pd
import numpy as np

csv_path = os.path.join('data', 'MBO', 'csv', 'output.csv')
df = pd.read_csv(csv_path, sep=",", skiprows=1, dtype={
        "ts_recv": "string",
        "ts_event": "string",
        "publisher_id": "int32",
        "instrument_id": "int32",
        "action": "category",
        "side": "category",
        "price": "float64",
        "size": "int32",
        "channel_id": "int32",
        "order_id": "int64",
        "flags": "int32",
        "ts_in_delta": "int64",
        "sequence": "int64",
        "symbol": "category",
    })

def extract_anchor_events(df):
    suspect_size_threshold = (df["size"].mean()) * 2
    anchor_events = df[df["size"] >= suspect_size_threshold]
    return anchor_events