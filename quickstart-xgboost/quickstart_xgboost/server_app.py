"""quickstart_xgboost: A Flower / XGBoost app."""

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging

from quickstart_xgboost.task import replace_keys, load_dataframe

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

    feature_cols = [c for c in df.columns if c not in ["weak_label", "ts_event"]]
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["weak_label"].values

    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred_proba = bst.predict(dtest)

    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n--- Final Global Model Evaluation ---")
    print(f"TEST PR-AUC:  {pr_auc:.4f}")
    print(f"TEST ROC-AUC: {roc_auc:.4f}")

    # Save model
    print("\nSaving final model to disk...")
    bst.save_model("federated_final_model.json")
