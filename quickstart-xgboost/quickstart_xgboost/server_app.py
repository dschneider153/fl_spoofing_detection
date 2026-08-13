"""quickstart_xgboost: A Flower / XGBoost app."""

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging

from quickstart_xgboost.task import replace_keys, load_dataframe, CLEAN_FEATURES

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Init global model
    # Init with an empty object; the XGBooster will be created
    # and trained on the client side.
    global_model = b""
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Initialize FedXgbBagging strategy
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start strategy, run FedXgbBagging for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())
    # Load global model into booster
    bst.load_model(global_model)

    # Last evaluation
    df = load_dataframe()
    split_date = df["ts_event"].quantile(0.75)
    test_df = df[df["ts_event"] >= split_date].copy()

    missing = [c for c in CLEAN_FEATURES if c not in test_df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X_test = test_df[CLEAN_FEATURES].values.astype(np.float32)
    y_test = test_df["weak_label"].values

    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred_proba = bst.predict(dtest)

    # Same F1-evaluation block as in baseline
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = (
        2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    )
    best_threshold = thresholds[np.argmax(f1_scores)]
    print("Best threshold:", best_threshold)
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print("\n--- Final Global Model Evaluation ---")
    print(
        f"Final evaluation --> Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}"
    )

    # Save model
    print("\nSaving final model to disk...")
    bst.save_model("federated_final_model.json")
