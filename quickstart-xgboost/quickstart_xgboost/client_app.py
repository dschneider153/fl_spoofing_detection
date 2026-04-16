"""quickstart-xgboost: A Flower / XGBoost app."""

import warnings

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict

from quickstart_xgboost.task import load_data, replace_keys, get_scale_pos_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)


# Flower ClientApp
app = ClientApp()


def _local_boost(bst_input, num_local_round, train_dmatrix):
    # Update trees based on local training data.
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    # Bagging: extract the last N=num_local_round trees for sever aggregation
    bst = bst_input[
        bst_input.num_boosted_rounds()
        - num_local_round : bst_input.num_boosted_rounds()
    ]
    return bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_dmatrix, _, num_train, _ = load_data(partition_id, num_partitions)

    # Read from run config
    num_local_round = context.run_config["local-epochs"]
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    # Override scale_pos_weight with partition-specific value since class imbalance varies across clients
    params["scale_pos_weight"] = get_scale_pos_weight(partition_id, num_partitions)

    global_round = msg.content["config"]["server-round"]
    if global_round == 1:
        # First round local training
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        # Load global model into booster
        bst.load_model(global_model)
        # Local training
        bst = _local_boost(bst, num_local_round, train_dmatrix)

    # Save model
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)

    # Construct reply message
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    model_record = ArrayRecord([model_np])
    metrics = {"num-examples": num_train}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valid_dmatrix, _, num_val = load_data(partition_id, num_partitions)

    if num_val == 0:
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num-examples": 0}
        metric_record = MetricRecord(metrics)
        return Message(content=RecordDict({"metrics": metric_record}), reply_to=msg)

    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    bst = xgb.Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)

    # These two lines were missing
    y_pred_proba = bst.predict(valid_dmatrix)
    y_true = valid_dmatrix.get_label()

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_arr = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-10)
    best_threshold = thresholds[np.argmax(f1_arr)]
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    client_precision = float(precision_score(y_true, y_pred, zero_division=0))
    client_recall = float(recall_score(y_true, y_pred, zero_division=0))
    client_f1 = float(f1_score(y_true, y_pred, zero_division=0))

    metrics = {
        "precision": client_precision,
        "recall": client_recall,
        "f1": client_f1,
        "num-examples": num_val,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)