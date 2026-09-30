"""
Task 11 — TensorFlow/Keras Neural Network Baseline
Financial Fraud Detection Project

Purpose
-------
Train and evaluate a feed-forward neural-network baseline.

Rules
-----
- Train only on training split.
- Evaluate model development only on validation split.
- Test split remains untouched.
- Preprocessing is fitted ONLY on training data.
- No SMOTE/resampling.
- No class weighting in this baseline.
- Threshold 0.50 is only the baseline threshold.
- Early stopping monitors validation PR-AUC.
"""

from pathlib import Path
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
MODEL_NAME = "neural_network_baseline"

EPOCHS = 50
BATCH_SIZE = 512
PATIENCE = 7


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds():
    """
    Set random seeds for reproducibility.
    """

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path: Path):
    """
    Load split and separate predictors from target.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found in {path.name}."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()

    validate_features(X)

    unique_classes = set(y.unique())

    if not unique_classes.issubset({0, 1}):
        raise ValueError(
            f"Invalid target values in {path.name}: "
            f"{sorted(unique_classes)}"
        )

    if len(unique_classes) != 2:
        raise ValueError(
            f"{path.name} must contain both classes."
        )

    return X, y


# ============================================================
# BUILD NEURAL NETWORK
# ============================================================

def build_neural_network(input_dim: int):
    """
    Build a simple feed-forward binary classifier.

    This is intentionally a baseline architecture.
    Hyperparameter tuning happens later.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(input_dim,)
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu",
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu",
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu",
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            ),

            tf.keras.metrics.AUC(
                curve="ROC",
                name="roc_auc",
            ),
        ],
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=BASELINE_THRESHOLD,
):
    """
    Calculate fraud-focused validation metrics.
    """

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

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    metrics = {
        "threshold": float(threshold),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "pr_auc": float(pr_auc),

        "roc_auc": float(roc_auc),

        "true_negatives": int(tn),

        "false_positives": int(fp),

        "false_negatives": int(fn),

        "true_positives": int(tp),
    }

    return metrics, predictions


# ============================================================
# PRINT EVALUATION
# ============================================================

def print_evaluation(
    y_true,
    predictions,
    metrics,
):
    """
    Print validation results.
    """

    print("\n" + "=" * 70)
    print("VALIDATION PERFORMANCE")
    print("=" * 70)

    print(
        f"\nDecision threshold : "
        f"{metrics['threshold']:.2f}"
    )

    print(
        f"Precision          : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall             : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1-score           : "
        f"{metrics['f1_score']:.6f}"
    )

    print(
        f"PR-AUC             : "
        f"{metrics['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC            : "
        f"{metrics['roc_auc']:.6f}"
    )

    print("\nConfusion Matrix")
    print("-" * 40)

    print(
        f"True negatives  : "
        f"{metrics['true_negatives']:,}"
    )

    print(
        f"False positives : "
        f"{metrics['false_positives']:,}"
    )

    print(
        f"False negatives : "
        f"{metrics['false_negatives']:,}"
    )

    print(
        f"True positives  : "
        f"{metrics['true_positives']:,}"
    )

    print("\nClassification Report")
    print("-" * 40)

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
            digits=4,
            zero_division=0,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_random_seeds()

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

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

    artifact_dir = (
        project_root
        / "artifacts"
        / "models"
        / MODEL_NAME
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        split_dir
        / "train.csv"
    )

    validation_path = (
        split_dir
        / "validation.csv"
    )

    print("=" * 70)
    print(
        "TASK 11 — TENSORFLOW/KERAS "
        "NEURAL NETWORK BASELINE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading training data...")

    X_train, y_train = load_dataset(
        train_path
    )

    print(
        "Loading validation data..."
    )

    X_val, y_val = load_dataset(
        validation_path
    )

    print("\nOriginal shapes:")

    print(
        f"Training   : {X_train.shape}"
    )

    print(
        f"Validation : {X_val.shape}"
    )

    print("\nFraud counts:")

    print(
        f"Training   : "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Validation : "
        f"{int(y_val.sum()):,}"
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    print(
        "\nBuilding preprocessing pipeline..."
    )

    preprocessor = build_preprocessor()

    print(
        "Fitting preprocessor on TRAINING data only..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    print(
        "Transforming validation data..."
    )

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    # TensorFlow works cleanly with float32.

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32,
    )

    X_val_processed = np.asarray(
        X_val_processed,
        dtype=np.float32,
    )

    y_train_array = np.asarray(
        y_train,
        dtype=np.float32,
    )

    y_val_array = np.asarray(
        y_val,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------------

    if not np.isfinite(
        X_train_processed
    ).all():

        raise ValueError(
            "Training features contain "
            "non-finite values."
        )

    if not np.isfinite(
        X_val_processed
    ).all():

        raise ValueError(
            "Validation features contain "
            "non-finite values."
        )

    if (
        X_train_processed.shape[1]
        != X_val_processed.shape[1]
    ):

        raise ValueError(
            "Training and validation "
            "feature counts do not match."
        )

    print("\nProcessed shapes:")

    print(
        f"Training   : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Validation : "
        f"{X_val_processed.shape}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    input_dim = (
        X_train_processed.shape[1]
    )

    print(
        f"\nBuilding neural network "
        f"with {input_dim} input features..."
    )

    model = build_neural_network(
        input_dim
    )

    model.summary()

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nTraining neural network..."
    )

    start_time = time.perf_counter()

    history = model.fit(
        X_train_processed,
        y_train_array,

        validation_data=(
            X_val_processed,
            y_val_array,
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stopping
        ],

        verbose=2,

        shuffle=True,
    )

    training_seconds = (
        time.perf_counter()
        - start_time
    )

    epochs_trained = len(
        history.history["loss"]
    )

    print(
        f"\nTraining completed in "
        f"{training_seconds:.2f} seconds."
    )

    print(
        f"Epochs trained: "
        f"{epochs_trained}"
    )

    # --------------------------------------------------------
    # VALIDATION PROBABILITIES
    # --------------------------------------------------------

    print(
        "\nGenerating validation probabilities..."
    )

    validation_probabilities = (
        model.predict(
            X_val_processed,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        .reshape(-1)
    )

    # --------------------------------------------------------
    # PROBABILITY CHECKS
    # --------------------------------------------------------

    if not np.isfinite(
        validation_probabilities
    ).all():

        raise ValueError(
            "Non-finite validation "
            "probabilities detected."
        )

    if (
        (validation_probabilities < 0).any()
        or
        (validation_probabilities > 1).any()
    ):

        raise ValueError(
            "Predicted probabilities must "
            "be between 0 and 1."
        )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    metrics, predictions = (
        calculate_metrics(
            y_val,
            validation_probabilities,
            threshold=BASELINE_THRESHOLD,
        )
    )

    metrics["model"] = MODEL_NAME

    metrics["training_seconds"] = float(
        training_seconds
    )

    metrics["epochs_trained"] = int(
        epochs_trained
    )

    metrics["batch_size"] = int(
        BATCH_SIZE
    )

    metrics["training_rows"] = int(
        len(X_train)
    )

    metrics["validation_rows"] = int(
        len(X_val)
    )

    metrics["training_fraud"] = int(
        y_train.sum()
    )

    metrics["validation_fraud"] = int(
        y_val.sum()
    )

    print_evaluation(
        y_val,
        predictions,
        metrics,
    )

    # --------------------------------------------------------
    # CONSISTENCY CHECKS
    # --------------------------------------------------------

    confusion_total = (
        metrics["true_negatives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_positives"]
    )

    if confusion_total != len(y_val):

        raise ValueError(
            "Confusion matrix total does not "
            "match validation size."
        )

    actual_fraud = (
        metrics["true_positives"]
        + metrics["false_negatives"]
    )

    if actual_fraud != int(
        y_val.sum()
    ):

        raise ValueError(
            "Fraud count does not match "
            "validation target."
        )

    print(
        "Metric consistency checks passed."
    )

    # --------------------------------------------------------
    # SAVE VALIDATION PREDICTIONS
    # --------------------------------------------------------

    validation_results = pd.DataFrame({
        "actual_class":
            y_val.to_numpy(),

        "fraud_probability":
            validation_probabilities,

        "predicted_class":
            predictions,
    })

    prediction_path = (
        artifact_dir
        / "validation_predictions.csv"
    )

    validation_results.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_df) + 1,
        ),
    )

    history_path = (
        artifact_dir
        / "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_path = (
        artifact_dir
        / "validation_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # SAVE PREPROCESSOR
    # --------------------------------------------------------

    preprocessor_path = (
        artifact_dir
        / "preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    # --------------------------------------------------------
    # SAVE KERAS MODEL
    # --------------------------------------------------------

    model_path = (
        artifact_dir
        / "neural_network.keras"
    )

    model.save(
        model_path
    )

    # --------------------------------------------------------
    # REPORT SAVED ARTIFACTS
    # --------------------------------------------------------

    print("\nSaved artifacts:")

    print(
        f"Model        : {model_path}"
    )

    print(
        f"Preprocessor : {preprocessor_path}"
    )

    print(
        f"Metrics      : {metrics_path}"
    )

    print(
        f"Predictions  : {prediction_path}"
    )

    print(
        f"History      : {history_path}"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "TASK 11 — NEURAL NETWORK "
        "BASELINE COMPLETED"
    )

    print("=" * 70)

    print(
        """
IMPORTANT INTERPRETATION RULES

- These are VALIDATION results.
- The test dataset has NOT been evaluated.
- Threshold 0.50 is only the baseline threshold.
- Preprocessing was fitted ONLY on training data.
- Early stopping monitored validation PR-AUC.
- No SMOTE/resampling was used.
- No class weighting was used.
- Architecture/hyperparameters are not tuned.
- Do not select the final model yet.
"""
    )


if __name__ == "__main__":
    main()