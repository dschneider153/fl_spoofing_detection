import csv
import os
import pandas as pd
import numpy as np

csv_path = os.path.join('data', 'MBO', 'csv', 'output.csv')
mbo = pd.read_csv(csv_path, sep=",", skiprows=1, dtype={
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

mbo = mbo.sort_values("ts_event")
mbo["ts_recv"] = pd.to_datetime(mbo["ts_recv"], utc=True)
mbo["ts_event"] = pd.to_datetime(mbo["ts_event"], utc=True)

def extract_anchor_events(df):
    suspect_size_threshold = (df["size"].mean()) * 2
    anchor_events = df[df["size"] >= suspect_size_threshold]
    return anchor_events

anchor_events = extract_anchor_events(mbo)
print(anchor_events)

#Prequesite for following function
mbo = mbo.set_index("ts_event")

#Fuction that helps find the end of one anchor event(cancel/execute)
def find_order_end_ts(df, order_id, t_add):
    events = df[
        (df["order_id"] == order_id) &
        (df.index > t_add)
    ]

    end_events = events[events["action"].isin(["C", "F", "T"])]
    if end_events.empty:
        return None
    
    return end_events.index[0]

PRE_MS = pd.Timedelta(milliseconds=500)
Post_MS = pd.Timedelta(milliseconds=500)

# Feature extraction script
rows = []
debug_stats = {"no_end_found": 0, "cancel": 0, "fill": 0, "trade": 0}

for idx, anchor in anchor_events.iterrows():
    order_id = anchor["order_id"]
    t_add = anchor["ts_event"]

    t_end = find_order_end_ts(mbo, order_id, t_add)
    if t_end is None:
        debug_stats["no_end_found"] += 1
        continue

    end_event = mbo[
        (mbo["order_id"] == order_id) &
        (mbo.index == t_end)
    ].iloc[0]
    
    end_action = end_event["action"]
    
    # Tracking all different kinds of 
    if end_action == "C":
        debug_stats["cancel"] += 1
    elif end_action == "F":
        debug_stats["fill"] += 1
    elif end_action == "T":
        debug_stats["trade"] += 1

    pre_start = t_add - PRE_MS
    post_end = t_end + Post_MS

    pre = mbo.loc[pre_start:t_add]
    post = mbo.loc[t_end:post_end]

    features = {
        "order_id": order_id,
        "initial_size": anchor["size"],
        "lifetime_ms": (t_end - t_add).total_seconds() * 1000,
        "end_action": end_action,

        "pre_add_count": (pre["action"] == "A").sum(),
        "pre_cancel_count": (pre["action"] == "C").sum(),
        "post_cancel_count": (post["action"] == "C").sum(),

        "ended_with_cancel": int(end_action == "C"),
        "ended_with_fill": int(end_action == "F"),
        "ended_with_trade": int(end_action == "T"),  # Add this
    }

    rows.append(features)

print("Debug stats:", debug_stats)
features_df = pd.DataFrame(rows)
print("\nEnd action distribution:")
print(features_df["end_action"].value_counts())
print(features_df)