"""quickstart_xgboost: A Flower / XGBoost app."""

import os
import numpy as np
import xgboost as xgb
import pandas as pd
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

DATA_PATH = "/home/dschneider/github/fl_spoofing_detection/data/Training and Testing/test_january.csv"
_cached_df = None

def load_dataframe():
    global _cached_df
    if _cached_df is None:
        _cached_df = pd.read_csv(DATA_PATH, delimiter=',')
        _cached_df["ts_event"] = pd.to_datetime(_cached_df["ts_event"])
        _cached_df = _cached_df.sort_values("ts_event").reset_index(drop=True)
        _cached_df["side"] = _cached_df["side"].map({"bid": 0, "ask": 1})
        _cached_df = _cached_df.drop(columns=["order_id", "spoof_prob"], errors="ignore")
    return _cached_df

def load_data(partition_id: int, num_clients: int):
    # Load dataframe with function
    df = load_dataframe()

    # Logic for loading time-series based partitions, could and will be changed to isntrument-/symbol-based partitions
    # For now, each client gets a consecutive time slice (preserves temporal order)
    n = len(df)
    partition_size = n // num_clients
    start = partition_id * partition_size
    end = start + partition_size if partition_id < num_clients - 1 else n  
    partition = df.iloc[start:end].copy()

    # Now split partitions into Train and Test (not a seperate function)
    split_date = df["ts_event"].quantile(0.75)
    train_mask = partition["ts_event"] < split_date
    test_mask = partition["ts_event"] >= split_date

    feature_cols = [c for c in partition.columns if c not in ["weak_label", "ts_event"]]

    X_train = partition.loc[train_mask, feature_cols].values.astype(np.float32)
    y_train = partition.loc[train_mask, "weak_label"].values.astype(np.float32)
    X_test = partition.loc[test_mask, feature_cols].values.astype(np.float32)
    y_test = partition.loc[test_mask, "weak_label"].values.astype(np.float32)
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_test, label=y_test)

    return train_dmatrix, valid_dmatrix, len(y_train), len(y_test)

# scale_pos_weight needs to be calculated for each partition, especially when there are different instruments
def get_scale_pos_weight(partition_id: int, num_clients: int) -> float:
    # Same logic as load_data
    df = load_dataframe()
    n = len(df)
    partition_size = n // num_clients
    start = partition_id * partition_size
    end = start + partition_size if partition_id < num_clients - 1 else n  
    partition = df.iloc[start:end]

    split_date = partition["ts_event"].quantile(0.75)
    y_train = partition[partition["ts_event"] < split_date]["weak_label"]

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
