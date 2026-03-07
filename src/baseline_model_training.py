import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report
)
from sklearn.model_selection import (train_test_split, TimeSeriesSplit)
import xgboost as xgb
from xgboost import (XGBClassifier, plot_importance)
import matplotlib.pyplot as plt

df = pd.read_csv('data/Training and Testing/largedataset.csv')
df["ts_event"] = pd.to_datetime(df["ts_event"])
df = df.sort_values("ts_event")
df["side"] = df["side"].map({"bid": 0, "ask": 1})

# Dropping labeling features here
###########################
df = df.drop(["spoofing_score", "distance_ticks", "order_id", "spread_normalized_distance"], axis=1)
###########################

X = df.drop(columns=["is_spoofing","ts_event"])
Y = df["is_spoofing"]
# Split training and test between 
split_date = df["ts_event"].quantile(0.75)
train_idx = df["ts_event"] < split_date
test_idx = df["ts_event"] >= split_date

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = Y[train_idx], Y[test_idx]

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos

scale_pos_weight = n_neg / n_pos
print("scale_pos_weight: ", scale_pos_weight)

tscv = TimeSeriesSplit(n_splits=5)

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

cv_scores = []

# Initial training is split into several windows for time series cross validation
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

    pr_auc = average_precision_score(Y_val, y_pred_proba)
    roc_auc = roc_auc_score(Y_val, y_pred_proba)

    cv_scores.append(pr_auc)

    print(f"Fold {fold} PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

print("Mean CV PR-AUC:", np.mean(cv_scores))

# Final training
final_model = build_model(scale_pos_weight)
final_model.set_params(early_stopping_rounds=None)
final_model.fit(X_train, y_train)
# Final Test
y_test_proba = final_model.predict_proba(X_test)[:, 1]
pr_auc_test = average_precision_score(y_test, y_test_proba)
roc_auc_test = roc_auc_score(y_test, y_test_proba)
print("TEST PR-AUC:", pr_auc_test)
print("TEST ROC-AUC:", roc_auc_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print("Best threshold:", best_threshold)
y_test_pred = (y_test_proba >= best_threshold).astype(int)
print(classification_report(y_test, y_test_pred))

top_k = int(0.01 * len(y_test))
sorted_idx = np.argsort(-y_test_proba)
top_idx = sorted_idx[:top_k]
precision_top1 = y_test.iloc[top_idx].mean()
print("Precision@1%:", precision_top1)

ax = plot_importance(final_model, max_num_features=10)
ax.figure.savefig('features/plot_importance.pdf')