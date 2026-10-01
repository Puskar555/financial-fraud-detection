"""
Task 14 — Decision Threshold Optimization
Financial Fraud Detection Project

Purpose
-------
Analyze the precision/recall trade-off for tuned models using
the held-out validation dataset.

Methodology
-----------
- train.csv is used to fit models.
- validation.csv is used for threshold selection.
- test.csv remains completely untouched.
- Hyperparameters come from Task 13.
- Thresholds are NOT selected using the test set.
- PR-AUC and ROC-AUC are threshold-independent.
- Precision, recall, F1, FP and FN depend on threshold.
"""

import ast
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from ml.src.features.preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    validate_features,
)

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

BASELINE_THRESHOLD = 0.50

# Fine threshold grid.
THRESHOLDS = np.arange(
    0.01,
    0.991,
    0.005,
)

NN_MAX_EPOCHS = 50
NN_BATCH_SIZE = 512
NN_PATIENCE = 7


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds(seed=RANDOM_STATE):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA
# ============================================================

def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' missing."
        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[
        TARGET_COLUMN
    ].copy()

    validate_features(X)

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"{path.name}: target must contain 0 and 1."
        )

    return X, y


# ============================================================
# LOAD TASK 13 CONFIGURATIONS
# ============================================================

def load_best_hyperparameters(path):

    if not path.exists():

        raise FileNotFoundError(
            "\nTask 13 has not finished yet.\n"
            f"Expected file:\n{path}\n\n"
            "Wait for Task 13 to complete before "
            "running Task 14."
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        configs = json.load(
            file
        )

    required = {
        "random_forest",
        "xgboost",
        "neural_network",
    }

    missing = (
        required
        - set(configs.keys())
    )

    if missing:
        raise ValueError(
            f"Missing tuned configurations: {missing}"
        )

    return configs


# ============================================================
# THRESHOLD METRICS
# ============================================================

def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    return {
        "threshold":
            float(threshold),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


# ============================================================
# THRESHOLD TABLE
# ============================================================

def build_threshold_table(
    model_name,
    y_true,
    probabilities,
):

    rows = []

    # Guarantee that 0.50 is included.

    thresholds = np.unique(
        np.append(
            THRESHOLDS,
            BASELINE_THRESHOLD,
        )
    )

    for threshold in thresholds:

        metrics = (
            calculate_threshold_metrics(
                y_true,
                probabilities,
                threshold,
            )
        )

        metrics[
            "model"
        ] = model_name

        rows.append(
            metrics
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SELECT CANDIDATES
# ============================================================

def select_f1_threshold(table):

    index = table[
        "f1"
    ].idxmax()

    row = table.loc[
        index
    ].copy()

    row[
        "selection_rule"
    ] = "maximum_f1"

    return row


def select_recall_constraint(
    table,
    minimum_recall,
):

    candidates = table[
        table["recall"]
        >= minimum_recall
    ].copy()

    if candidates.empty:
        return None

    # Among thresholds satisfying recall,
    # prefer highest precision.
    #
    # If tied, prefer higher threshold,
    # which is more conservative.

    candidates = (
        candidates
        .sort_values(
            [
                "precision",
                "threshold",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    row = candidates.iloc[
        0
    ].copy()

    row[
        "selection_rule"
    ] = (
        f"recall_at_least_"
        f"{int(minimum_recall * 100)}pct"
    )

    return row


def select_precision_constraint(
    table,
    minimum_precision,
):

    candidates = table[
        table["precision"]
        >= minimum_precision
    ].copy()

    if candidates.empty:
        return None

    # Among thresholds satisfying precision,
    # prefer highest recall.

    candidates = (
        candidates
        .sort_values(
            [
                "recall",
                "threshold",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    row = candidates.iloc[
        0
    ].copy()

    row[
        "selection_rule"
    ] = (
        f"precision_at_least_"
        f"{int(minimum_precision * 100)}pct"
    )

    return row


def select_baseline_threshold(
    table,
):

    index = (
        table["threshold"]
        - BASELINE_THRESHOLD
    ).abs().idxmin()

    row = table.loc[
        index
    ].copy()

    row[
        "selection_rule"
    ] = "baseline_0.50"

    return row


def get_candidate_thresholds(
    table,
):

    candidates = []

    candidates.append(
        select_baseline_threshold(
            table
        )
    )

    candidates.append(
        select_f1_threshold(
            table
        )
    )

    for recall_target in [
        0.80,
        0.85,
        0.90,
    ]:

        candidate = (
            select_recall_constraint(
                table,
                recall_target,
            )
        )

        if candidate is not None:

            candidates.append(
                candidate
            )

    candidate = (
        select_precision_constraint(
            table,
            0.90,
        )
    )

    if candidate is not None:

        candidates.append(
            candidate
        )

    return pd.DataFrame(
        candidates
    )


# ============================================================
# NEURAL NETWORK
# ============================================================

def build_neural_network(
    input_dim,
    hidden_layers,
    dropout,
    learning_rate,
):

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Input(
            shape=(input_dim,)
        )
    )

    for units in hidden_layers:

        model.add(
            tf.keras.layers.Dense(
                int(units),
                activation="relu",
            )
        )

        model.add(
            tf.keras.layers.Dropout(
                float(dropout)
            )
        )

    model.add(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(
                learning_rate
            )
        ),

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            )
        ],
    )

    return model


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest(
    X_train,
    y_train,
    X_val,
    config,
):

    print(
        "\nTraining tuned Random Forest..."
    )

    print(
        f"Parameters: {config}"
    )

    model = RandomForestClassifier(
        n_estimators=int(
            config["n_estimators"]
        ),

        max_depth=(
            None
            if config["max_depth"]
            is None
            else int(
                config["max_depth"]
            )
        ),

        min_samples_split=int(
            config[
                "min_samples_split"
            ]
        ),

        max_features=config[
            "max_features"
        ],

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Completed in {elapsed:.2f}s"
    )

    return (
        model,
        probabilities,
        elapsed,
    )


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_xgboost(
    X_train,
    y_train,
    X_val,
    config,
):

    print(
        "\nTraining tuned XGBoost..."
    )

    print(
        f"Parameters: {config}"
    )

    model = XGBClassifier(
        n_estimators=int(
            config[
                "n_estimators"
            ]
        ),

        learning_rate=float(
            config[
                "learning_rate"
            ]
        ),

        max_depth=int(
            config[
                "max_depth"
            ]
        ),

        subsample=float(
            config[
                "subsample"
            ]
        ),

        colsample_bytree=float(
            config[
                "colsample_bytree"
            ]
        ),

        objective=
            "binary:logistic",

        eval_metric=
            "logloss",

        random_state=
            RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist",
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Completed in {elapsed:.2f}s"
    )

    return (
        model,
        probabilities,
        elapsed,
    )


# ============================================================
# TRAIN NEURAL NETWORK
# ============================================================

def train_neural_network(
    X_train,
    y_train,
    X_val,
    config,
):

    print(
        "\nTraining tuned Neural Network..."
    )

    print(
        f"Parameters: {config}"
    )

    preprocessor = (
        build_preprocessor()
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32,
    )

    X_val_processed = np.asarray(
        X_val_processed,
        dtype=np.float32,
    )

    hidden_layers = config[
        "hidden_layers"
    ]

    # Task 13 stores this as a string
    # such as "[64, 32, 16]".

    if isinstance(
        hidden_layers,
        str,
    ):

        hidden_layers = (
            ast.literal_eval(
                hidden_layers
            )
        )

    if not isinstance(
        hidden_layers,
        (list, tuple),
    ):

        raise TypeError(
            "Neural-network hidden_layers "
            "must be a list or tuple."
        )

    tf.keras.backend.clear_session()

    set_random_seeds()

    model = build_neural_network(
        input_dim=
            X_train_processed.shape[1],

        hidden_layers=
            hidden_layers,

        dropout=
            config["dropout"],

        learning_rate=
            config[
                "learning_rate"
            ],
    )

    start = time.perf_counter()

    print(
        "\n*** CLEAN NN TRAINING VERSION ***"
    )

    print(
        "Fixed epochs: 27"
    )

    print(
        "Validation labels are NOT used during training."
    )

    history = model.fit(
        X_train_processed,

        np.asarray(
            y_train,
            dtype=np.float32,
        ),

        epochs=27,

        batch_size=NN_BATCH_SIZE,

        verbose=2,

        shuffle=True,
    )

    probabilities = (
        model.predict(
            X_val_processed,
            batch_size=NN_BATCH_SIZE,
            verbose=0,
        )
        .reshape(-1)
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    epochs_trained = len(
        history.history["loss"]
    )

    print(
        f"Completed in {elapsed:.2f}s"
    )

    print(
        f"Epochs trained: "
        f"{epochs_trained}"
    )

    return (
        model,
        preprocessor,
        probabilities,
        elapsed,
        epochs_trained,
    )


# ============================================================
# MODEL ANALYSIS
# ============================================================

def analyze_model(
    model_name,
    y_val,
    probabilities,
):

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            f"{model_name}: invalid probabilities."
        )

    pr_auc = (
        average_precision_score(
            y_val,
            probabilities,
        )
    )

    roc_auc = (
        roc_auc_score(
            y_val,
            probabilities,
        )
    )

    print(
        f"\n{model_name}"
    )

    print(
        f"Validation PR-AUC : "
        f"{pr_auc:.6f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{roc_auc:.6f}"
    )

    threshold_table = (
        build_threshold_table(
            model_name,
            y_val,
            probabilities,
        )
    )

    candidates = (
        get_candidate_thresholds(
            threshold_table
        )
    )

    print(
        "\nCandidate operating thresholds:"
    )

    columns = [
        "selection_rule",
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    print(
        candidates[
            columns
        ].to_string(
            index=False,
            float_format=
                lambda x: f"{x:.6f}",
        )
    )

    return (
        threshold_table,
        candidates,
        {
            "model":
                model_name,

            "pr_auc":
                float(pr_auc),

            "roc_auc":
                float(roc_auc),
        },
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_random_seeds()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    split_dir = (
        project_root
        / "data"
        / "processed"
        / "splits"
    )

    task13_path = (
        project_root
        / "artifacts"
        / "hyperparameter_tuning"
        / "best_hyperparameters.json"
    )

    artifact_dir = (
        project_root
        / "artifacts"
        / "threshold_optimization"
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "TASK 14 — DECISION THRESHOLD OPTIMIZATION"
    )

    print("=" * 72)

    print(
        "\nLoading Task 13 tuned hyperparameters..."
    )

    configs = (
        load_best_hyperparameters(
            task13_path
        )
    )

    # ========================================================
    # LOAD TRAIN + VALIDATION
    # ========================================================

    print(
        "\nLoading training data..."
    )

    X_train, y_train = (
        load_dataset(
            split_dir
            / "train.csv"
        )
    )

    print(
        "Loading validation data..."
    )

    X_val, y_val = (
        load_dataset(
            split_dir
            / "validation.csv"
        )
    )

    print(
        f"\nTraining   : "
        f"{X_train.shape}"
    )

    print(
        f"Validation : "
        f"{X_val.shape}"
    )

    print(
        f"Training fraud   : "
        f"{int(y_train.sum())}"
    )

    print(
        f"Validation fraud : "
        f"{int(y_val.sum())}"
    )

    print(
        "\nTEST DATASET IS NOT LOADED."
    )

    all_threshold_tables = []
    all_candidates = []
    model_summaries = []

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_model, rf_prob, rf_time = (
        train_random_forest(
            X_train,
            y_train,
            X_val,
            configs[
                "random_forest"
            ],
        )
    )

    (
        rf_table,
        rf_candidates,
        rf_summary,
    ) = analyze_model(
        "random_forest",
        y_val,
        rf_prob,
    )

    all_threshold_tables.append(
        rf_table
    )

    all_candidates.append(
        rf_candidates
    )

    rf_summary[
        "training_seconds"
    ] = rf_time

    model_summaries.append(
        rf_summary
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    (
        xgb_model,
        xgb_prob,
        xgb_time,
    ) = train_xgboost(
        X_train,
        y_train,
        X_val,
        configs[
            "xgboost"
        ],
    )

    (
        xgb_table,
        xgb_candidates,
        xgb_summary,
    ) = analyze_model(
        "xgboost",
        y_val,
        xgb_prob,
    )

    all_threshold_tables.append(
        xgb_table
    )

    all_candidates.append(
        xgb_candidates
    )

    xgb_summary[
        "training_seconds"
    ] = xgb_time

    model_summaries.append(
        xgb_summary
    )

    # ========================================================
    # NEURAL NETWORK
    # ========================================================

    (
        nn_model,
        nn_preprocessor,
        nn_prob,
        nn_time,
        nn_epochs,
    ) = train_neural_network(
        X_train,
        y_train,
        X_val,
        configs[
            "neural_network"
        ],
    )

    (
        nn_table,
        nn_candidates,
        nn_summary,
    ) = analyze_model(
        "neural_network",
        y_val,
        nn_prob,
    )

    all_threshold_tables.append(
        nn_table
    )

    all_candidates.append(
        nn_candidates
    )

    nn_summary[
        "training_seconds"
    ] = nn_time

    nn_summary[
        "epochs_trained"
    ] = nn_epochs

    model_summaries.append(
        nn_summary
    )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    threshold_results = (
        pd.concat(
            all_threshold_tables,
            ignore_index=True,
        )
    )

    candidate_results = (
        pd.concat(
            all_candidates,
            ignore_index=True,
        )
    )

    model_summary = (
        pd.DataFrame(
            model_summaries
        )
    )

    # ========================================================
    # SAVE TABLES
    # ========================================================

    threshold_path = (
        artifact_dir
        / "threshold_results.csv"
    )

    candidates_path = (
        artifact_dir
        / "candidate_thresholds.csv"
    )

    summary_path = (
        artifact_dir
        / "validation_model_summary.csv"
    )

    threshold_results.to_csv(
        threshold_path,
        index=False,
    )

    candidate_results.to_csv(
        candidates_path,
        index=False,
    )

    model_summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    rf_model_path = (
        artifact_dir
        / "tuned_random_forest.joblib"
    )

    xgb_model_path = (
        artifact_dir
        / "tuned_xgboost.joblib"
    )

    nn_model_path = (
        artifact_dir
        / "tuned_neural_network.keras"
    )

    nn_preprocessor_path = (
        artifact_dir
        / "neural_network_preprocessor.joblib"
    )

    joblib.dump(
        rf_model,
        rf_model_path,
    )

    joblib.dump(
        xgb_model,
        xgb_model_path,
    )

    nn_model.save(
        nn_model_path
    )

    joblib.dump(
        nn_preprocessor,
        nn_preprocessor_path,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 100)

    print(
        "VALIDATION MODEL SUMMARY"
    )

    print("=" * 100)

    print(
        model_summary.to_string(
            index=False,
            float_format=
                lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 100)

    print(
        "ALL CANDIDATE OPERATING THRESHOLDS"
    )

    print("=" * 100)

    display_columns = [
        "model",
        "selection_rule",
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    print(
        candidate_results[
            display_columns
        ].to_string(
            index=False,
            float_format=
                lambda x: f"{x:.6f}",
        )
    )

    print(
        "\nSaved artifacts:"
    )

    print(
        f"Threshold table : "
        f"{threshold_path}"
    )

    print(
        f"Candidates      : "
        f"{candidates_path}"
    )

    print(
        f"Model summary   : "
        f"{summary_path}"
    )

    print(
        f"RF model        : "
        f"{rf_model_path}"
    )

    print(
        f"XGB model       : "
        f"{xgb_model_path}"
    )

    print(
        f"NN model        : "
        f"{nn_model_path}"
    )

    print(
        f"NN preprocessor : "
        f"{nn_preprocessor_path}"
    )

    print("\n" + "=" * 72)

    print(
        "TASK 14 — THRESHOLD "
        "OPTIMIZATION COMPLETED"
    )

    print("=" * 72)

    print(
        """
IMPORTANT

- Tuned hyperparameters came from Task 13 CV.
- Threshold analysis used validation.csv.
- test.csv was NOT loaded.
- Threshold 0.50 remains the baseline reference.
- Lower thresholds generally increase recall and false positives.
- Higher thresholds generally increase precision and false negatives.
- Maximum F1 is a statistical candidate, not automatically the
  production/business threshold.
- Final threshold selection must consider fraud-detection costs.
"""
    )


if __name__ == "__main__":
    main()