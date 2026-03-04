import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_csv(path, index_col=None):
    df = pd.read_csv(path, skiprows=1, header=0)
    # Remove duplicate header rows by checking first column
    first_col = df.columns[0]
    df = df[df[first_col] != first_col].reset_index(drop=True)
    skip_cols = {"ts_recv", "ts_event", "action", "side", "symbol"}
    for col in df.columns:
        if col not in skip_cols:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= len(converted) * 0.9:
                df[col] = converted
    return df

mbo_csv_path = os.path.join('data', 'MBO', 'csv', 'combined_output_january.csv')
mbo = clean_csv(mbo_csv_path)
mbo = mbo.astype({
        "ts_recv": "string",
        "ts_event": "string",
        "publisher_id": "int64",
        "instrument_id": "int64",
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

mbp10_csv_path = os.path.join('data', 'MBP-10', 'csv', 'combined_output_january.csv')
mbp10 = clean_csv(mbp10_csv_path)

mbo = mbo.sort_values("ts_event")
mbo["ts_recv"] = pd.to_datetime(mbo["ts_recv"], utc=True)
mbo["ts_event"] = pd.to_datetime(mbo["ts_event"], utc=True)

def extract_anchor_events(df):
    suspect_size_threshold = (df["size"].mean()) * 2
    anchor_events = df[df["size"] >= suspect_size_threshold]
    return anchor_events
anchor_events = extract_anchor_events(mbo)

mbp10 = mbp10.sort_values("ts_event")
mbp10["ts_recv"] = pd.to_datetime(mbp10["ts_recv"], utc=True)
mbp10["ts_event"] = pd.to_datetime(mbp10["ts_event"], utc=True)
mbp10 = mbp10.rename(columns={"ts_event": "ts_book"})
mbp10 = mbp10.iloc[10:-10].copy()
mbp10 = mbp10.dropna(subset=["bid_px_00", "ask_px_00"]).copy()
# Mid Price
mbp10["midprice"] = (mbp10["bid_px_00"] + mbp10["ask_px_00"]) / 2
# Imbalance
mbp10["imbalance"] = ((mbp10["bid_sz_00"] - mbp10["ask_sz_00"]) / (mbp10["bid_sz_00"] + mbp10["ask_sz_00"]).replace(0, np.nan))
# Spread
mbp10["spread"] = (mbp10["ask_px_00"] - mbp10["bid_px_00"])

max_book_lag = pd.Timedelta(milliseconds=100)
anchor_events_with_mbp = pd.merge_asof(anchor_events, mbp10, left_on="ts_event", right_on="ts_book", by="instrument_id", direction='backward', suffixes=("", "_book"))
anchor_events_with_mbp = anchor_events_with_mbp[anchor_events_with_mbp["ts_event"] - anchor_events_with_mbp["ts_book"] <= max_book_lag]

# Required for following function
mbo = mbo.set_index("ts_event")
mbp10 = mbp10.set_index("ts_book")

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
POST_MS = pd.Timedelta(milliseconds=500)

def get_midprice_at(df, t):
    idx = df.index.searchsorted(t)
    idx = min(idx, len(df) - 1)
    return df.iloc[idx]["midprice"]

def get_spread_at(df, t):
    idx = df.index.searchsorted(t)
    idx = min(idx, len(df) - 1)
    return df.iloc[idx]["spread"]

tick_size = (mbo["price"].sort_values().diff().dropna().loc[lambda x: x > 0].min())

# Feature extraction script
# First step: Vectorized feature extraction
# Distance ticks
anchor_events_with_mbp["distance_ticks"] = np.where(
    anchor_events_with_mbp["side"] == "B",   # Buy order
    (anchor_events_with_mbp["bid_px_00"] - anchor_events_with_mbp["price"]) / tick_size,
    (anchor_events_with_mbp["price"] - anchor_events_with_mbp["ask_px_00"]) / tick_size
)
# Spread normalized distance
spread_ticks = (
    ((anchor_events_with_mbp["ask_px_00"] - anchor_events_with_mbp["bid_px_00"]) / tick_size).clip(lower=1)
)
anchor_events_with_mbp["spread_normalized_distance"] = (
    anchor_events_with_mbp["distance_ticks"] / spread_ticks
)
# Book Depth
bid_px = anchor_events_with_mbp[[f"bid_px_0{i}" for i in range(10)]].values
bid_sz = anchor_events_with_mbp[[f"bid_sz_0{i}" for i in range(10)]].values
ask_px = anchor_events_with_mbp[[f"ask_px_0{i}" for i in range(10)]].values
ask_sz = anchor_events_with_mbp[[f"ask_sz_0{i}" for i in range(10)]].values
anchor_price = anchor_events_with_mbp["price"].values
side = anchor_events_with_mbp["side"].values
buy_mask = bid_px >= anchor_price[:,None]
sell_mask = ask_px >= anchor_price[:,None]
depth_buy = (buy_mask * bid_sz).sum(axis=1)
depth_sell = (sell_mask * ask_sz).sum(axis=1)
anchor_events_with_mbp["depth_volume"] = np.where (
    side == "B",
    depth_buy,
    depth_sell
)
anchor_events_with_mbp["depth_ratio"] = (
    anchor_events_with_mbp["depth_volume"] / anchor_events_with_mbp["size"]
)
# Second step: Feature loop
rows = []
for idx, anchor in anchor_events_with_mbp.iterrows():

    order_id = anchor["order_id"]
    t_add = anchor["ts_event"]
    anchor_price = anchor["price"]
    side = anchor["side"]
    t_end = find_order_end_ts(mbo, order_id, t_add)
    if t_end is None:
        continue
    end_event = mbo[
        (mbo["order_id"] == order_id) &
        (mbo.index == t_end)
    ].iloc[0]
    end_action = end_event["action"]
    if anchor["side"] == "B":
        side_sign = 1
    else:
        side_sign = -1

    pre_start = t_add - PRE_MS
    post_end = t_end + POST_MS
    pre_mbo = mbo.loc[pre_start:t_add]
    post_mbo = mbo.loc[t_end:post_end]
    
    # For relative:
    size_fivesec_window = mbo.loc[t_add - pd.Timedelta(seconds=5):t_add]
    size_fivesec_window = size_fivesec_window[size_fivesec_window["order_id"] != order_id]
    same_side_window = size_fivesec_window[size_fivesec_window["side"] == anchor["side"]]
    baseline = same_side_window["size"].median()
    relative_size = anchor["size"]/max(baseline, 1)

    best_bid = anchor["bid_px_00"]
    best_ask = anchor["ask_px_00"]
    distance_ticks = anchor["distance_ticks"]
    spread_normalized_distance = anchor["spread_normalized_distance"]
    depth_volume = anchor["depth_volume"]
    depth_ratio = anchor["depth_ratio"]
    
    midprice_at_start = get_midprice_at(mbp10, t_add)
    midprice_at_end = get_midprice_at(mbp10, t_end)
    midprice_after_50ms = get_midprice_at(mbp10, t_end + pd.Timedelta(milliseconds=50))
    midprice_after_200ms = get_midprice_at(mbp10, t_end + pd.Timedelta(milliseconds=200))
    midprice_after_1000ms = get_midprice_at(mbp10, t_end + pd.Timedelta(milliseconds=1000))
    midprice_change_during = midprice_at_end - midprice_at_start
    midprice_change_50ms = midprice_after_50ms - midprice_at_end
    midprice_change_200ms = midprice_after_200ms - midprice_at_end
    midprice_change_1000ms = midprice_after_1000ms - midprice_at_end

    t_after_forimbalance = t_end + pd.Timedelta(milliseconds=200)
    imb_after_row = mbp10.loc[:t_after_forimbalance].iloc[-1]
    imbalance_shift = imb_after_row["imbalance"] - anchor["imbalance"]

    features = {
        # General information on the anchor
        "order_id": order_id,
        "initial_size": anchor["size"],
        "anchor_price": anchor_price,
        "side": side,
        # First set of features: Lifecycle/Core Spoofing Features
        "relative_size": relative_size,
        "log_lifetime": np.log1p((t_end - t_add).total_seconds() * 1000),
        "ended_with_cancel": int(end_action == "C"),
        # Second set of features: Position of the anchor
        "best_bid": best_bid,
        "best_ask": best_ask,
        "distance_ticks": distance_ticks,
        "spread_normalized_distance": spread_normalized_distance,
        "depth_volume": depth_volume,
        "depth_ratio": depth_ratio,
        # Third set of features: Impact of the anchor
        "midprice_change_during": midprice_change_during,
        "midprice_change_50ms": midprice_change_50ms,
        "midprice_change_200ms": midprice_change_200ms,
        "midprice_change_1000ms": midprice_change_1000ms,
        "price_reversion": (midprice_change_during * midprice_change_1000ms),
        "signed_impact": (side_sign * (midprice_after_1000ms - midprice_at_end) / tick_size), 
        "post_trade_count": (post_mbo.shape[0]),
        "order_book_imbalance_shift": imbalance_shift,
        "spread_change_ticks": abs(get_spread_at(mbp10, t_end + POST_MS) - get_spread_at(mbp10, t_end)) / tick_size,
        # Fourth set of features: Context
        "pre_add_count": (pre_mbo["action"] == "A").sum(),
        "pre_cancel_count": (pre_mbo["action"] == "C").sum(),
        "post_cancel_count": (post_mbo["action"] == "C").sum(),
    }

    rows.append(features)

features_df = pd.DataFrame(rows)

# Clean up Dataframe
features_df = features_df[features_df["distance_ticks"] <= 200]

print(features_df)
features_df.to_csv('features/features_january.csv')

# Sanity check for relative sizes
'''print(features_df["relative_size"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))
ax = features_df["relative_size"].hist(bins=20)
fig = ax.get_figure()
fig.savefig('features/relativehisto.pdf')'''

# Sanity check for position features
'''print(tick_size)
print((anchor_events_with_mbp["ts_event"] - anchor_events_with_mbp["ts_book"]).describe())
print(features_df.nlargest(10, "distance_ticks"))
print(features_df["distance_ticks"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))
row = features_df.nlargest(1, "distance_ticks").iloc[0]
print(row[[
    "order_id",
    "side",
    "anchor_price",
    "best_bid",
    "best_ask",
    "distance_ticks"
]])
print(abs(anchor_price - row["best_bid"]) / tick_size)
ax = features_df["distance_ticks"].hist(bins=50)
fig = ax.get_figure()
fig.savefig('features/distancehisto.pdf')
print(features_df[["distance_ticks", "spread_normalized_distance", "depth_volume", "depth_ratio"]].describe())'''

#Sanity check for impact features
'''print(mbp10[["ts_book", "bid_px_00", "ask_px_00", "midprice"]].head(20))
print(mbp10["midprice"].describe())
print(mbp10[["bid_px_00", "ask_px_00", "midprice"]].isna().sum())
print(features_df["midprice_change_50ms"].describe())
print(features_df["midprice_change_200ms"].describe())
print(features_df["midprice_change_1000ms"].describe())
print(features_df[features_df["midprice_change_50ms"] == features_df["midprice_change_50ms"].min()])
print(features_df["signed_impact"].describe())
print(features_df["post_trade_count"].describe())
print(features_df["order_book_imbalance_shift"].describe())
print(features_df["order_book_imbalance_shift"].abs().quantile([0.5, 0.75, 0.9, 0.99]))
print(features_df["spread_change_ticks"].describe())
print(features_df["spread_change_ticks"].abs().quantile([0.5, 0.75, 0.9, 0.99]))
print(features_df.nlargest(3, "spread_change_ticks")[["order_id","best_bid", "best_ask", "spread_change_ticks"]])
print(features_df["midprice_change_during"].describe())'''