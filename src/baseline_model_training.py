import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import (train_test_split, TimeSeriesSplit)
import xgboost as xgb
from xgboost import (XGBClassifier, plot_importance)
import matplotlib.pyplot as plt

# Loading data and mapping the bid/ask side to a number
df = pd.read_csv('data/Training and Testing/test_january.csv')
df["ts_event"] = pd.to_datetime(df["ts_event"])
df = df.sort_values("ts_event")
df["side"] = df["side"].map({"bid": 0, "ask": 1})

CLEAN_FEATURES = [
    "initial_size",
    "anchor_price",
    "best_bid",
    "best_ask", 
    "distance_ticks",
    "spread_normalized_distance",
    "pre_add_count",
    "pre_cancel_count",
    "relative_size",
    "depth_ratio",
    "depth_volume"
]

# Splitting data
X = df[CLEAN_FEATURES]
Y = df["weak_label"]
# Split training and test 
split_date = df["ts_event"].quantile(0.75)
train_idx = df["ts_event"] < split_date
test_idx = df["ts_event"] >= split_date

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = Y[train_idx], Y[test_idx]

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos

# Weight for potential class imbalance
scale_pos_weight = n_neg / n_pos
print("scale_pos_weight: ", scale_pos_weight)

# Parameter setting for model
def build_model(scale_pos_weight):
    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=30,
    )

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

# Time Series Cross Validation loop
for fold, (train_i, val_i) in enumerate(tscv.split(X_train)):
    
    X_tr, X_val = X_train.iloc[train_i], X_train.iloc[val_i]
    Y_tr, Y_val = y_train.iloc[train_i], y_train.iloc[val_i]

    model = build_model(scale_pos_weight)

    model.fit(
        X_tr,
        Y_tr,
        eval_set=[(X_val, Y_val)],
        verbose=False
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    fold_precision = precision_score(Y_val, y_pred_binary)
    fold_recall = recall_score(Y_val, y_pred_binary)
    fold_f1 = f1_score(Y_val, y_pred_binary)

    cv_scores.append(fold_f1)

    print(f"Fold {fold} --> Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}, F1: {fold_f1:.4f}")

print("Mean CV F1:", np.mean(cv_scores))

# Final training
final_model = build_model(scale_pos_weight)
final_model.set_params(early_stopping_rounds=None)
final_model.fit(X_train, y_train)
# Final Test
y_test_proba = final_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
# F1-Score needs threshold, but XGBoost uses probabilities so we need to find the best threshold
best_threshold = thresholds[np.argmax(f1_scores)]
print("Best threshold:", best_threshold)

y_test_pred = (y_test_proba >= best_threshold).astype(int)

# Final Precision, Recall and F1-Score 
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
print(f"Final evaluation --> Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
