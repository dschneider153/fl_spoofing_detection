import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Time constants
PRE_MS = pd.Timedelta(milliseconds=500)
POST_MS = pd.Timedelta(milliseconds=500)

def clean_csv(path, index_col=None):
    df = pd.read_csv(path, header=0)
    # Remove duplicate header rows by checking first column with whitespace checking
    first_col = df.columns[0]
    df = df[df[first_col] != first_col].reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    skip_cols = {"ts_recv", "ts_event", "action", "side", "symbol"}
    for col in df.columns:
        if col not in skip_cols:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= len(converted) * 0.9:
                df[col] = converted
    return df

mbo_csv_path = os.path.join('data', 'MBO', 'csv', 'combined_output_january')
mbo = clean_csv(mbo_csv_path)

if(
    (mbo["price"].sort_values().diff().dropna().loc[lambda x: x > 0].min()) <= 0
):
    TICK_SIZE = 0.005
else:
    TICK_SIZE = (mbo["price"].sort_values().diff().dropna().loc[lambda x: x > 0].min())



mbp10_csv_path = os.path.join('data', 'NVIDIA_TEST', 'nvidia_mbp-10.csv')
mbp10 = clean_csv(mbp10_csv_path)

mbo = mbo.sort_values("ts_event")
mbo["ts_recv"] = pd.to_datetime(mbo["ts_recv"], utc=True)
mbo["ts_event"] = pd.to_datetime(mbo["ts_event"], utc=True)

def extract_anchor_events(df):
    suspect_size_threshold = df["size"].mean() * 2
    anchor_events = df[
        (df["size"] >= suspect_size_threshold) & 
        (df["action"].isin(["A", "T"]))
    ]
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

MAX_BOOK_LAG = pd.Timedelta(milliseconds=100)
anchor_events_with_mbp = pd.merge_asof(
    anchor_events, 
    mbp10, 
    left_on="ts_event", 
    right_on="ts_book", 
    by="instrument_id", 
    direction='backward', 
    suffixes=("", "_book")
)
anchor_events_with_mbp = anchor_events_with_mbp[anchor_events_with_mbp["ts_event"] - anchor_events_with_mbp["ts_book"] <= MAX_BOOK_LAG]

mbo = mbo.set_index("ts_event").sort_index()
mbp10 = mbp10.set_index("ts_book").sort_index()

# Core features without relative size
# Finds end event for each action by looking for the first order with an ending event (C,F,T)
end_events = mbo[mbo["action"].isin(["C", "F", "T"])].reset_index()[["ts_event", "order_id", "action"]]
end_events = end_events.rename(columns={"ts_event": "t_end"})
anchor_reset = anchor_events_with_mbp.reset_index()[["ts_event", "order_id"]].copy()
merged_ends = pd.merge_asof(
    anchor_reset,
    end_events,
    left_on="ts_event",
    right_on="t_end",
    by="order_id",
    direction="forward" 
)
merged_ends = merged_ends.dropna(subset=["t_end"])

merged_ends = merged_ends.rename(columns={"action": "end_action"})
anchor_events_with_mbp = anchor_events_with_mbp.reset_index().merge(
    merged_ends[["order_id", "ts_event", "t_end", "end_action"]],
    on=["order_id", "ts_event"],
    how="inner"
).set_index("ts_event")
# Lifetime 
lifetimes_ms = (anchor_events_with_mbp["t_end"] - anchor_events_with_mbp.index).dt.total_seconds() * 1000
anchor_events_with_mbp["log_lifetime"] = np.log1p(lifetimes_ms)
# Ended with cancel?
anchor_events_with_mbp["ended_with_cancel"] = (anchor_events_with_mbp["end_action"] == "C").astype(int)

# Assembling position features
# Distance ticks
anchor_events_with_mbp["distance_ticks"] = np.where(
    anchor_events_with_mbp["side"] == "B",   # Buy order
    (anchor_events_with_mbp["bid_px_00"] - anchor_events_with_mbp["price"]) / TICK_SIZE,
    (anchor_events_with_mbp["price"] - anchor_events_with_mbp["ask_px_00"]) / TICK_SIZE
)
# Spread normalized distance
spread_ticks = (
    ((anchor_events_with_mbp["ask_px_00"] - anchor_events_with_mbp["bid_px_00"]) / TICK_SIZE).clip(lower=1)
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
# Depth Ratio
anchor_events_with_mbp["depth_ratio"] = (
    anchor_events_with_mbp["depth_volume"] / anchor_events_with_mbp["size"]
)

# Function for finding midprice change
mbp10_index = mbp10.index.tz_localize(None).values

def vectorized_midprice(timestamps):
    if hasattr(timestamps, 'values'):
        timestamps = timestamps.tz_localize(None).values if hasattr(timestamps, 'tz_localize') else timestamps.values
    idx = mbp10_index.searchsorted(timestamps)
    idx = np.clip(idx, 0, len(mbp10) - 1)

    # Logic for removing time jumps between sessions
    matched_times = mbp10_index[idx]
    time_diff = matched_times - timestamps
    valid = np.abs(time_diff) <= 100_000_000
    midprice = np.full(len(timestamps), np.nan)
    midprice[valid] = mbp10["midprice"].values[idx[valid]]

    return mbp10["midprice"].values[idx]

# Function for finding spread change by taking each timestamp, getting the index through searchsorted and then the corresponding value
def vectorized_spread(timestamps):
    if hasattr(timestamps, 'values'):
        timestamps = timestamps.tz_localize(None).values if hasattr(timestamps, 'tz_localize') else timestamps.values
    idx = mbp10_index.searchsorted(timestamps)
    idx = np.clip(idx, 0, len(mbp10) - 1)

    # Logic for removing time jumps 
    matched_times = mbp10_index[idx]
    time_diff = matched_times - timestamps
    valid = np.abs(time_diff) <= 100_000_000
    spread = np.full(len(timestamps), np.nan)
    spread[valid] = mbp10["spread"].values[idx[valid]]

    return mbp10["spread"].values[idx]

t_add_arr = anchor_events_with_mbp.tz_localize(None).index
t_end_arr = anchor_events_with_mbp["t_end"].dt.tz_localize(None).values
# Vectorized imbalance lookup

# Assembling the impact features
# Midprice changes
ms50   = np.timedelta64(50, 'ms')
ms200  = np.timedelta64(200, 'ms')
ms1000 = np.timedelta64(1000, 'ms')
POST_MS_np = np.timedelta64(500, 'ms')

anchor_events_with_mbp["midprice_change_during"] = vectorized_midprice(t_end_arr) - vectorized_midprice(t_add_arr)
anchor_events_with_mbp["midprice_change_50ms"] = vectorized_midprice(t_end_arr + ms50) - vectorized_midprice(t_end_arr)
anchor_events_with_mbp["midprice_change_200ms"] = vectorized_midprice(t_end_arr + ms200) - vectorized_midprice(t_end_arr)
anchor_events_with_mbp["midprice_change_1000ms"] = vectorized_midprice(t_end_arr + ms1000) - vectorized_midprice(t_end_arr)
anchor_events_with_mbp["price_reversion"] = anchor_events_with_mbp["midprice_change_during"] * anchor_events_with_mbp["midprice_change_1000ms"]
# Spread change normalized on ticks
anchor_events_with_mbp["spread_change_ticks"] = np.abs((vectorized_spread(t_end_arr + POST_MS_np) - vectorized_spread(t_end_arr))) / TICK_SIZE
# Imbalance lookup
imb_times = t_end_arr + ms200
imb_idx = np.clip(mbp10_index.searchsorted(imb_times), 0, len(mbp10) - 1)
anchor_events_with_mbp["imbalance_after"] = mbp10["imbalance"].values[imb_idx]
# Order book imbalance shift
anchor_events_with_mbp["order_book_imbalance_shift"] = anchor_events_with_mbp["imbalance_after"] - anchor_events_with_mbp["imbalance"]
# Signed impact
midprice_at_end = vectorized_midprice(t_end_arr)
midprice_after_1000ms = vectorized_midprice(t_end_arr + ms1000)
side_sign = np.where(anchor_events_with_mbp["side"] == "B", 1, -1)
anchor_events_with_mbp["signed_impact_after"] = side_sign * ((midprice_after_1000ms - midprice_at_end) / TICK_SIZE)
# Signed impact for midprice change during for prive attraction
anchor_events_with_mbp["signed_impact_during"] = side_sign * ((anchor_events_with_mbp["midprice_change_during"]) / TICK_SIZE)
# Normalized attraction
anchor_events_with_mbp["attraction_ratio"] = np.where(
    anchor_events_with_mbp["distance_ticks"].abs() > 5, # less than 5 ticks distance cant produce any meaningful attraction
    anchor_events_with_mbp["signed_impact_during"] / anchor_events_with_mbp["distance_ticks"].abs(),
    np.nan
)

# Relative size (core feature)
mbo_adds = mbo[mbo["action"] == "A"].reset_index()[["ts_event", "order_id", "side", "size"]]

for side in ["B", "A", "N"]:
    # Creates a side mask, since relative size is defined by all orders on the SAME SIDE
    side_mask = mbo_adds["side"] == side
    side_adds = mbo_adds[side_mask].copy()
    if side_adds.empty:
        continue
    # Now applies the mask and calculates the rolling mean
    side_adds = side_adds.set_index("ts_event")
    side_adds["rolling_baseline"] = (
        side_adds["size"]
        .rolling("5s", closed="left") 
        .mean()
        .fillna(1.0)
    )
    mbo_adds.loc[side_mask, "rolling_baseline"] = side_adds["rolling_baseline"].values

anchor_reset2 = anchor_events_with_mbp.reset_index()[["ts_event", "order_id", "side"]].copy()
# Adds the baseline to each event 
baseline_lookup = pd.merge_asof(
    anchor_reset2,
    mbo_adds[["ts_event", "side", "rolling_baseline"]].sort_values("ts_event"),
    on="ts_event",
    by="side",
    direction="backward"
)
# Adds the baseline lookup to the main dataframe
anchor_events_with_mbp["baseline_median"] = baseline_lookup["rolling_baseline"].values
# Calculates the relative size with the baseline lookup
anchor_events_with_mbp["relative_size"] = (
    anchor_events_with_mbp["size"] / anchor_events_with_mbp["baseline_median"].clip(lower=1)
)

# Action counts
mbo_index = mbo.index.values
mbo_action = mbo["action"].values

# Replaced the count function with a vectorized
def count_actions_in_window(t_start_arr, t_end_arr, target_action):
    mask = mbo_action == target_action
    masked_index = mbo_index[mask]
    counts = (
        np.searchsorted(masked_index, t_end_arr, side='right') - 
        np.searchsorted(masked_index, t_start_arr, side='left')
    )
    return counts

pre_starts = (anchor_events_with_mbp.index - PRE_MS).values
t_adds = anchor_events_with_mbp.index.values
t_ends = anchor_events_with_mbp["t_end"].values
post_ends = (anchor_events_with_mbp["t_end"] + POST_MS).values

anchor_events_with_mbp["pre_add_count"] = count_actions_in_window(pre_starts, t_adds, "A")
anchor_events_with_mbp["pre_cancel_count"] = count_actions_in_window(pre_starts, t_adds, "C")
anchor_events_with_mbp["post_cancel_count"] = count_actions_in_window(t_ends, post_ends, "C")
anchor_events_with_mbp["post_trade_count"] = count_actions_in_window(t_ends, post_ends, "A")

features_df = anchor_events_with_mbp[[
    "order_id", "size", "price", "side",
    "relative_size", "log_lifetime", "ended_with_cancel",
    "bid_px_00", "ask_px_00", "distance_ticks", "spread_normalized_distance",
    "depth_volume", "depth_ratio",
    "midprice_change_during", "midprice_change_50ms", "midprice_change_200ms",
    "midprice_change_1000ms", "price_reversion", "signed_impact_during", "signed_impact_after", "attraction_ratio",
    "post_trade_count", "order_book_imbalance_shift", "spread_change_ticks",
    "pre_add_count", "pre_cancel_count", "post_cancel_count",
]].rename(columns={"size": "initial_size", "price": "anchor_price", "bid_px_00": "best_bid", "ask_px_00": "best_ask"}).copy()

# Post feature filtering, with filtering out extreme data outliers
features_df = features_df[features_df["side"].isin(["B", "A"])]
features_df = features_df[features_df["distance_ticks"] <= 1000]
features_df = features_df[features_df["spread_change_ticks"] <= 100]
features_df.to_csv('features/features_january.csv')

# Sanity checks
'''
# Sanity check for relative sizes
print(features_df["relative_size"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))
ax = features_df["relative_size"].hist(bins=20)
fig = ax.get_figure()
fig.savefig('features/relativehisto.pdf')

# Sanity check for position features
print(TICK_SIZE)
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
print(abs(anchor_price - row["best_bid"]) / TICK_SIZE)
ax = features_df["distance_ticks"].hist(bins=50)
fig = ax.get_figure()
fig.savefig('features/distancehisto.pdf')
print(features_df[["distance_ticks", "spread_normalized_distance", "depth_volume", "depth_ratio"]].describe())

# Sanity check for impact features
print(mbp10[["ts_book", "bid_px_00", "ask_px_00", "midprice"]].head(20))
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