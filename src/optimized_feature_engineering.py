import os
import warnings

import numpy as np
import pandas as pd

# Time constants
PRE_MS = pd.Timedelta(milliseconds=500)
POST_MS = pd.Timedelta(milliseconds=500)
SKIP_COLS = {"ts_recv", "ts_event", "action", "side", "symbol"}
 
MAX_BOOK_LAG = pd.Timedelta(milliseconds=100)
MAX_LOOKUP_LAG_NS = 100_000_000  # 100 ms in nanoseconds
 
MS50 = np.timedelta64(50, "ms")
MS200 = np.timedelta64(200, "ms")
MS1000 = np.timedelta64(1000, "ms")
POST_MS_NP = np.timedelta64(500, "ms")

# Paths
MBO_CSV_PATH = os.path.join("data", "MBO", "csv", "output.csv")
MBP10_CSV_PATH = os.path.join("data", "MBP-10", "csv", "output.csv")
OUTPUT_PATH = os.path.join("features", "features_january.csv")

FEATURE_COLUMNS = [
    "order_id",
    "size",
    "price",
    "side",
    "relative_size",
    "log_lifetime",
    "ended_with_cancel",
    "bid_px_00",
    "ask_px_00",
    "distance_ticks",
    "spread_normalized_distance",
    "depth_volume",
    "depth_ratio",
    "midprice_change_during",
    "midprice_change_50ms",
    "midprice_change_200ms",
    "midprice_change_1000ms",
    "price_reversion",
    "signed_impact_during",
    "signed_impact_after",
    "attraction_ratio",
    "post_trade_count",
    "order_book_imbalance_shift",
    "spread_change_ticks",
    "pre_add_count",
    "pre_cancel_count",
    "post_cancel_count",
]

FEATURE_RENAMES = {
    "size": "initial_size",
    "price": "anchor_price",
    "bid_px_00": "best_bid",
    "ask_px_00": "best_ask",
}

def clean_csv(source, numeric_threshold=0.9, warn_below=0.999):

    # Checks if input is source or DF
    df = (
        source.copy()
        if isinstance(source, pd.DataFrame)
        else pd.read_csv(source, header=0)
    )

    # Normalize column names
    df.columns = df.columns.astype(str).str.strip()

    # Remove unnamed column files
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Remove duplicate header columns
    first_col = df.columns[0]
    is_header_row = df[first_col].astype(str).str.strip() == first_col
    df = df[~is_header_row].reset_index(drop=True)

    # Columns cleaning
    for col in df.columns:
        # Skips predefined columns
        if col in SKIP_COLS:
            continue
        # Tries to convert all columns to numerical columns
        converted = pd.to_numeric(df[col], errors="coerce")
        # Checks all columns if they are numerical
        ratio = converted.notna().mean() if len(converted) else 0.0
        # If ratio smaller than numeric_threshold column was a text
        if ratio >= numeric_threshold:
            # At small percentage it will parse as numeric, but with warning
            if ratio < warn_below:
                lost = int((~converted.notna()).sum())
                warnings.warn(
                    f"Spalte {col!r}: {lost} von {len(converted)} Werten "
                    f"({1 - ratio:.2%}) nicht numerisch parsbar, werden NaN.",
                    stacklevel=2,
                )
            # Collects all converted numeric values to the df
            df[col] = converted

    return df

# Tick size infering
def infer_tick_size(mbo):
    
    positive_diffs = mbo["price"].sort_values().diff().dropna().loc[lambda x: x > 0]
    min_diff = positive_diffs.min()
    if min_diff <= 0:
        return 0.005
    return min_diff

# Prepares MBO data by sorting and parsing timestamps
def prepare_mbo(source):
    
    mbo = clean_csv(source)
    mbo = mbo.sort_values("ts_event")
    mbo["ts_recv"] = pd.to_datetime(mbo["ts_recv"], utc=True)
    mbo["ts_event"] = pd.to_datetime(mbo["ts_event"], utc=True)
    return mbo

def prepare_mbp10(source):
    mbp10 = clean_csv(source)
    mbp10 = mbp10.sort_values("ts_event")
    mbp10["ts_recv"] = pd.to_datetime(mbp10["ts_recv"], utc=True)
    mbp10["ts_event"] = pd.to_datetime(mbp10["ts_event"], utc=True)
    mbp10 = mbp10.rename(columns={"ts_event": "ts_book"})
    mbp10 = mbp10.iloc[10:-10].copy()
    mbp10 = mbp10.dropna(subset=["bid_px_00", "ask_px_00"]).copy()

    # Mid Price
    mbp10["midprice"] = (mbp10["bid_px_00"] + mbp10["ask_px_00"]) / 2
    # Imbalance
    mbp10["imbalance"] = (mbp10["bid_sz_00"] - mbp10["ask_sz_00"]) / (
        mbp10["bid_sz_00"] + mbp10["ask_sz_00"]
    ).replace(0, np.nan)
    # Spread
    mbp10["spread"] = mbp10["ask_px_00"] - mbp10["bid_px_00"]
    return mbp10

# Anchor extraction
def extract_anchor_events(df):
    suspect_size_threshold = df["size"].mean() * 2
    anchor_events = df[
        (df["size"] >= suspect_size_threshold) & (df["action"].isin(["A", "T"]))
    ]
    return anchor_events

# Merges anchors with mbp
def merge_anchors_with_book(anchor_events, mbp10, max_book_lag=MAX_BOOK_LAG):
    anchor_events_with_mbp = pd.merge_asof(
        anchor_events,
        mbp10,
        left_on="ts_event",
        right_on="ts_book",
        by="instrument_id",
        direction="backward",
        suffixes=("", "_book"),
    )
    anchor_events_with_mbp = anchor_events_with_mbp[
        anchor_events_with_mbp["ts_event"] - anchor_events_with_mbp["ts_book"]
        <= max_book_lag
    ]
    return anchor_events_with_mbp


# Core features without relative size
# Finds end event for each action by looking for the first order with an ending event (C,F,T)
def add_lifetime_features(anchor_events_with_mbp, mbo):
    
    end_events = mbo[mbo["action"].isin(["C", "F", "T"])].reset_index()[
        ["ts_event", "order_id", "action"]
    ]
    end_events = end_events.rename(columns={"ts_event": "t_end"})
 
    anchor_reset = anchor_events_with_mbp.reset_index()[["ts_event", "order_id"]].copy()
    merged_ends = pd.merge_asof(
        anchor_reset,
        end_events,
        left_on="ts_event",
        right_on="t_end",
        by="order_id",
        direction="forward",
    )
    merged_ends = merged_ends.dropna(subset=["t_end"])
    merged_ends = merged_ends.rename(columns={"action": "end_action"})
 
    anchor_events_with_mbp = (
        anchor_events_with_mbp.reset_index()
        .merge(
            merged_ends[["order_id", "ts_event", "t_end", "end_action"]],
            on=["order_id", "ts_event"],
            how="inner",
        )
        .set_index("ts_event")
    )
 
    # Lifetime
    lifetimes_ms = (
        anchor_events_with_mbp["t_end"] - anchor_events_with_mbp.index
    ).dt.total_seconds() * 1000
    anchor_events_with_mbp["log_lifetime"] = np.log1p(lifetimes_ms)
    # Ended with cancel?
    anchor_events_with_mbp["ended_with_cancel"] = (
        anchor_events_with_mbp["end_action"] == "C"
    ).astype(int)
    return anchor_events_with_mbp
 

# Assembling position features
def add_position_features(anchor_events_with_mbp, tick_size):
    # Distance ticks
    anchor_events_with_mbp["distance_ticks"] = np.where(
        anchor_events_with_mbp["side"] == "B",  # Buy order
        (anchor_events_with_mbp["bid_px_00"] - anchor_events_with_mbp["price"])
        / tick_size,
        (anchor_events_with_mbp["price"] - anchor_events_with_mbp["ask_px_00"])
        / tick_size,
    )
    # Spread normalized distance
    spread_ticks = (
        (anchor_events_with_mbp["ask_px_00"] - anchor_events_with_mbp["bid_px_00"])
        / tick_size
    ).clip(lower=1)
    anchor_events_with_mbp["spread_normalized_distance"] = (
        anchor_events_with_mbp["distance_ticks"] / spread_ticks
    )
    # Book Depth
    bid_px = anchor_events_with_mbp[[f"bid_px_0{i}" for i in range(10)]].values
    bid_sz = anchor_events_with_mbp[[f"bid_sz_0{i}" for i in range(10)]].values
    ask_px = anchor_events_with_mbp[[f"ask_px_0{i}" for i in range(10)]].values
    ask_sz = anchor_events_with_mbp[[f"ask_sz_0{i}" for i in range(10)]].values
    anchor_price = anchor_events_with_mbp["price"].values
    side_values = anchor_events_with_mbp["side"].values
    buy_mask = bid_px >= anchor_price[:, None]
    sell_mask = ask_px >= anchor_price[:, None]
    depth_buy = (buy_mask * bid_sz).sum(axis=1)
    depth_sell = (sell_mask * ask_sz).sum(axis=1)
    anchor_events_with_mbp["depth_volume"] = np.where(
        side_values == "B", depth_buy, depth_sell
    )
    # Depth Ratio
    anchor_events_with_mbp["depth_ratio"] = (
        anchor_events_with_mbp["depth_volume"] / anchor_events_with_mbp["size"]
    )
    return anchor_events_with_mbp

# Turns an Index/Series into a tz-naive numpy datetime64 array 
def to_naive_values(timestamps):
    if hasattr(timestamps, "values"):
        return (
            timestamps.tz_localize(None).values
            if hasattr(timestamps, "tz_localize")
            else timestamps.values
        )
    return timestamps

# Generic function to help find midprice, spread and imbalance vectorized
def vectorized_book_value(timestamps, book_index, values, drop_stale=False):
    """Takes each timestamp, gets the index through searchsorted and then the
    corresponding book value.
 
    drop_stale=True sets values to NaN when the matched book row is further away
    than MAX_LOOKUP_LAG_NS (time jumps between sessions). 
    """
    timestamps = to_naive_values(timestamps)
    idx = book_index.searchsorted(timestamps)
    idx = np.clip(idx, 0, len(book_index) - 1)
    matched = values[idx]
 
    if not drop_stale:
        return matched
 
    # Logic for removing time jumps between sessions
    time_diff = book_index[idx] - timestamps
    stale = np.abs(time_diff) > MAX_LOOKUP_LAG_NS
    result = matched.astype(float).copy()
    result[stale] = np.nan
    return result


# Assembling the impact features
def add_impact_features(anchor_events_with_mbp, mbp10, tick_size, drop_stale=False):
    """Assembling the impact features. `mbp10` is expected to be indexed by ts_book."""
    book_index = mbp10.index.tz_localize(None).values
    midprice_values = mbp10["midprice"].values
    spread_values = mbp10["spread"].values
    imbalance_values = mbp10["imbalance"].values
 
    def midprice_at(timestamps):
        return vectorized_book_value(
            timestamps, book_index, midprice_values, drop_stale
        )
 
    def spread_at(timestamps):
        return vectorized_book_value(timestamps, book_index, spread_values, drop_stale)
 
    t_add_arr = anchor_events_with_mbp.tz_localize(None).index
    t_end_arr = anchor_events_with_mbp["t_end"].dt.tz_localize(None).values
 
    # Midprice changes
    anchor_events_with_mbp["midprice_change_during"] = midprice_at(
        t_end_arr
    ) - midprice_at(t_add_arr)
    anchor_events_with_mbp["midprice_change_50ms"] = midprice_at(
        t_end_arr + MS50
    ) - midprice_at(t_end_arr)
    anchor_events_with_mbp["midprice_change_200ms"] = midprice_at(
        t_end_arr + MS200
    ) - midprice_at(t_end_arr)
    anchor_events_with_mbp["midprice_change_1000ms"] = midprice_at(
        t_end_arr + MS1000
    ) - midprice_at(t_end_arr)
    anchor_events_with_mbp["price_reversion"] = (
        anchor_events_with_mbp["midprice_change_during"]
        * anchor_events_with_mbp["midprice_change_1000ms"]
    )
    # Spread change normalized on ticks
    anchor_events_with_mbp["spread_change_ticks"] = (
        np.abs(spread_at(t_end_arr + POST_MS_NP) - spread_at(t_end_arr)) / tick_size
    )
    # Imbalance lookup
    anchor_events_with_mbp["imbalance_after"] = vectorized_book_value(
        t_end_arr + MS200, book_index, imbalance_values, drop_stale
    )
    # Order book imbalance shift
    anchor_events_with_mbp["order_book_imbalance_shift"] = (
        anchor_events_with_mbp["imbalance_after"] - anchor_events_with_mbp["imbalance"]
    )
    # Signed impact
    midprice_at_end = midprice_at(t_end_arr)
    midprice_after_1000ms = midprice_at(t_end_arr + MS1000)
    side_sign = np.where(anchor_events_with_mbp["side"] == "B", 1, -1)
    anchor_events_with_mbp["signed_impact_after"] = side_sign * (
        (midprice_after_1000ms - midprice_at_end) / tick_size
    )
    # Signed impact for midprice change during for price attraction
    anchor_events_with_mbp["signed_impact_during"] = side_sign * (
        (anchor_events_with_mbp["midprice_change_during"]) / tick_size
    )
    # Normalized attraction
    anchor_events_with_mbp["attraction_ratio"] = np.where(
        anchor_events_with_mbp["distance_ticks"].abs()
        > 5,  # less than 5 ticks distance cant produce any meaningful attraction
        anchor_events_with_mbp["signed_impact_during"]
        / anchor_events_with_mbp["distance_ticks"].abs(),
        np.nan,
    )
    return anchor_events_with_mbp

# Generic funcitons to compute rolling baselines for counting base features
def compute_rolling_baselines(mbo, window="5s"):
    """Rolling mean of the add sizes per side. Relative size is defined by all
    orders on the SAME SIDE. `mbo` is expected to be indexed by ts_event.
    """
    mbo_adds = mbo[mbo["action"] == "A"].reset_index()[
        ["ts_event", "order_id", "side", "size"]
    ]
 
    for order_side in ["B", "A", "N"]:
        # Creates a side mask
        side_mask = mbo_adds["side"] == order_side
        side_adds = mbo_adds[side_mask].copy()
        if side_adds.empty:
            continue
        # Now applies the mask and calculates the rolling mean
        side_adds = side_adds.set_index("ts_event")
        side_adds["rolling_baseline"] = (
            side_adds["size"].rolling(window, closed="left").mean().fillna(1.0)
        )
        mbo_adds.loc[side_mask, "rolling_baseline"] = side_adds[
            "rolling_baseline"
        ].values
 
    return mbo_adds
 
# Relative size (core feature)
def add_relative_size(anchor_events_with_mbp, mbo_adds):

    anchor_reset2 = anchor_events_with_mbp.reset_index()[
        ["ts_event", "order_id", "side"]
    ].copy()
    # Adds the baseline to each event
    baseline_lookup = pd.merge_asof(
        anchor_reset2,
        mbo_adds[["ts_event", "side", "rolling_baseline"]].sort_values("ts_event"),
        on="ts_event",
        by="side",
        direction="backward",
    )
    # Adds the baseline lookup to the main dataframe
    anchor_events_with_mbp["baseline_median"] = baseline_lookup[
        "rolling_baseline"
    ].values
    # Calculates the relative size with the baseline lookup
    anchor_events_with_mbp["relative_size"] = anchor_events_with_mbp[
        "size"
    ] / anchor_events_with_mbp["baseline_median"].clip(lower=1)
    return anchor_events_with_mbp
 
 
def count_actions_in_window(
    t_start_arr, t_end_arr, mbo_index, mbo_action, target_action
):
    """Vectorized action counting inside [t_start, t_end]."""
    mask = mbo_action == target_action
    masked_index = mbo_index[mask]
    counts = np.searchsorted(masked_index, t_end_arr, side="right") - np.searchsorted(
        masked_index, t_start_arr, side="left"
    )
    return counts
 
 
def add_action_counts(anchor_events_with_mbp, mbo):
    """Action counts. `mbo` is expected to be indexed by ts_event."""
    mbo_index = mbo.index.values
    mbo_action = mbo["action"].values
 
    pre_starts = (anchor_events_with_mbp.index - PRE_MS).values
    t_adds = anchor_events_with_mbp.index.values
    t_ends = anchor_events_with_mbp["t_end"].values
    post_ends = (anchor_events_with_mbp["t_end"] + POST_MS).values
 
    anchor_events_with_mbp["pre_add_count"] = count_actions_in_window(
        pre_starts, t_adds, mbo_index, mbo_action, "A"
    )
    anchor_events_with_mbp["pre_cancel_count"] = count_actions_in_window(
        pre_starts, t_adds, mbo_index, mbo_action, "C"
    )
    anchor_events_with_mbp["post_cancel_count"] = count_actions_in_window(
        t_ends, post_ends, mbo_index, mbo_action, "C"
    )
    anchor_events_with_mbp["post_trade_count"] = count_actions_in_window(
        t_ends, post_ends, mbo_index, mbo_action, "A"
    )
    return anchor_events_with_mbp
 
# Feature engineering functions
def build_features_df(anchor_events_with_mbp):
    return (
        anchor_events_with_mbp[FEATURE_COLUMNS].rename(columns=FEATURE_RENAMES).copy()
    )
 
# Post feature filtering
def filter_features(features_df):
    features_df = features_df[features_df["side"].isin(["B", "A"])]
    features_df = features_df[features_df["distance_ticks"] <= 1000]
    features_df = features_df[features_df["spread_change_ticks"] <= 100]
    return features_df
 
# Whole feature pipeline
def build_feature_pipeline(mbo_source=MBO_CSV_PATH, mbp10_source=MBP10_CSV_PATH):
    mbo = prepare_mbo(mbo_source)
    mbp10 = prepare_mbp10(mbp10_source)
 
    tick_size = infer_tick_size(mbo)
 
    anchor_events = extract_anchor_events(mbo)
    anchor_events_with_mbp = merge_anchors_with_book(anchor_events, mbp10)
 
    mbo = mbo.set_index("ts_event").sort_index()
    mbp10 = mbp10.set_index("ts_book").sort_index()
 
    anchor_events_with_mbp = add_lifetime_features(anchor_events_with_mbp, mbo)
    anchor_events_with_mbp = add_position_features(anchor_events_with_mbp, tick_size)
    anchor_events_with_mbp = add_impact_features(
        anchor_events_with_mbp, mbp10, tick_size
    )
 
    mbo_adds = compute_rolling_baselines(mbo)
    anchor_events_with_mbp = add_relative_size(anchor_events_with_mbp, mbo_adds)
    anchor_events_with_mbp = add_action_counts(anchor_events_with_mbp, mbo)
 
    features_df = build_features_df(anchor_events_with_mbp)
    features_df = filter_features(features_df)
    return features_df
 
 
def main():
    features_df = build_feature_pipeline()
    features_df.to_csv(OUTPUT_PATH)
    return features_df
 
 
if __name__ == "__main__":
    main()

# Sanity checks
"""
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
print(features_df["midprice_change_during"].describe())"""
