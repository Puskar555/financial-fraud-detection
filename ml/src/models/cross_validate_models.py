"""
Task 12 — Stratified K-Fold Cross-Validation
Financial Fraud Detection Project

Purpose
-------
Compare four baseline fraud-detection models using 5-fold
Stratified Cross-Validation on the TRAINING dataset only.

Models
------
1. Logistic Regression
2. Random Forest
3. XGBoost
4. TensorFlow/Keras Neural Network

Important methodology
---------------------
- validation.csv is NOT used.
- test.csv is NOT used.
- CV operates only on train.csv.
- preprocessing is fitted independently inside every fold.
- no SMOTE/resampling.
- no class weighting.
- no threshold optimization.
- threshold = 0.50 for threshold-dependent metrics.
- PR-AUC is the primary model-comparison metric.
"""

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
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

N_SPLITS = 5

THRESHOLD = 0.50

NN_EPOCHS = 50
NN_BATCH_SIZE = 512
NN_PATIENCE = 7

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "neural_network",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seeds(seed=RANDOM_STATE):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(path: Path):
    """
    Load ONLY train.csv.

    Validation and test data are intentionally excluded
    from cross-validation.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()

    validate_features(X)

    classes = set(y.unique())

    if classes != {0, 1}:
        raise ValueError(
            f"Expected target classes {{0, 1}}, "
            f"found {classes}."
        )

    return X, y


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_logistic_regression():
    """
    Same baseline family used in Task 8.
    """

    return LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )


def build_random_forest():
    """
    Same Random Forest baseline configuration
    used in Task 9.
    """

    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_xgboost():
    """
    Same XGBoost baseline configuration used in Task 10.
    """

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.10,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def build_neural_network(input_dim: int):
    """
    Same basic neural-network architecture used in Task 11.
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
    threshold=THRESHOLD,
):
    """
    Calculate identical metrics for every model.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# PRINT FOLD RESULT
# ============================================================

def print_fold_result(
    model_name,
    fold,
    metrics,
    training_seconds,
):
    """
    Print compact metrics after each model/fold.
    """

    print(
        f"\n{model_name} — Fold {fold}"
    )

    print(
        f"PR-AUC    : "
        f"{metrics['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC   : "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"TP={metrics['tp']}, "
        f"FP={metrics['fp']}, "
        f"FN={metrics['fn']}, "
        f"TN={metrics['tn']}"
    )

    print(
        f"Training time: "
        f"{training_seconds:.2f}s"
    )


# ============================================================
# MAIN CROSS-VALIDATION
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

    train_path = (
        project_root
        / "data"
        / "processed"
        / "splits"
        / "train.csv"
    )

    artifact_dir = (
        project_root
        / "artifacts"
        / "cross_validation"
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "TASK 12 — STRATIFIED 5-FOLD CROSS-VALIDATION"
    )
    print("=" * 72)

    print(
        "\nIMPORTANT: Only train.csv is used."
    )

    print(
        "Validation and test datasets remain untouched."
    )

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    print(
        "\nLoading training dataset..."
    )

    X, y = load_training_data(
        train_path
    )

    print(
        f"\nTraining shape : {X.shape}"
    )

    print(
        f"Legitimate     : "
        f"{int((y == 0).sum()):,}"
    )

    print(
        f"Fraud          : "
        f"{int((y == 1).sum()):,}"
    )

    print(
        f"Fraud rate     : "
        f"{y.mean() * 100:.4f}%"
    )

    # --------------------------------------------------------
    # CV CONFIGURATION
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    # --------------------------------------------------------
    # FOLDS
    # --------------------------------------------------------

    for fold, (
        train_indices,
        fold_val_indices,
    ) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        print("\n" + "=" * 72)

        print(
            f"FOLD {fold}/{N_SPLITS}"
        )

        print("=" * 72)

        X_fold_train = X.iloc[
            train_indices
        ].copy()

        y_fold_train = y.iloc[
            train_indices
        ].copy()

        X_fold_val = X.iloc[
            fold_val_indices
        ].copy()

        y_fold_val = y.iloc[
            fold_val_indices
        ].copy()

        print(
            f"\nFold training shape   : "
            f"{X_fold_train.shape}"
        )

        print(
            f"Fold validation shape : "
            f"{X_fold_val.shape}"
        )

        print(
            f"Training fraud         : "
            f"{int(y_fold_train.sum())}"
        )

        print(
            f"Validation fraud       : "
            f"{int(y_fold_val.sum())}"
        )

        # ====================================================
        # PREPROCESSING FOR THIS FOLD
        # ====================================================

        # New preprocessor every fold.
        # Fit ONLY on fold-training observations.

        preprocessor = (
            build_preprocessor()
        )

        X_fold_train_processed = (
            preprocessor.fit_transform(
                X_fold_train
            )
        )

        X_fold_val_processed = (
            preprocessor.transform(
                X_fold_val
            )
        )

        X_fold_train_processed = (
            np.asarray(
                X_fold_train_processed,
                dtype=np.float32,
            )
        )

        X_fold_val_processed = (
            np.asarray(
                X_fold_val_processed,
                dtype=np.float32,
            )
        )

        if not np.isfinite(
            X_fold_train_processed
        ).all():

            raise ValueError(
                f"Non-finite training data "
                f"in fold {fold}."
            )

        if not np.isfinite(
            X_fold_val_processed
        ).all():

            raise ValueError(
                f"Non-finite validation data "
                f"in fold {fold}."
            )

        # ====================================================
        # 1. LOGISTIC REGRESSION
        # ====================================================

        print(
            "\nTraining Logistic Regression..."
        )

        logistic = (
            build_logistic_regression()
        )

        start = time.perf_counter()

        logistic.fit(
            X_fold_train_processed,
            y_fold_train,
        )

        training_seconds = (
            time.perf_counter()
            - start
        )

        probabilities = (
            logistic.predict_proba(
                X_fold_val_processed
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_fold_val,
            probabilities,
        )

        print_fold_result(
            "Logistic Regression",
            fold,
            metrics,
            training_seconds,
        )

        results.append({
            "model":
                "logistic_regression",

            "fold":
                fold,

            **metrics,

            "training_seconds":
                training_seconds,
        })

        # ====================================================
        # 2. RANDOM FOREST
        # ====================================================

        print(
            "\nTraining Random Forest..."
        )

        random_forest = (
            build_random_forest()
        )

        start = time.perf_counter()

        random_forest.fit(
            X_fold_train,
            y_fold_train,
        )

        training_seconds = (
            time.perf_counter()
            - start
        )

        probabilities = (
            random_forest.predict_proba(
                X_fold_val
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_fold_val,
            probabilities,
        )

        print_fold_result(
            "Random Forest",
            fold,
            metrics,
            training_seconds,
        )

        results.append({
            "model":
                "random_forest",

            "fold":
                fold,

            **metrics,

            "training_seconds":
                training_seconds,
        })

        # ====================================================
        # 3. XGBOOST
        # ====================================================

        print(
            "\nTraining XGBoost..."
        )

        xgboost_model = (
            build_xgboost()
        )

        start = time.perf_counter()

        xgboost_model.fit(
            X_fold_train,
            y_fold_train,
        )

        training_seconds = (
            time.perf_counter()
            - start
        )

        probabilities = (
            xgboost_model.predict_proba(
                X_fold_val
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_fold_val,
            probabilities,
        )

        print_fold_result(
            "XGBoost",
            fold,
            metrics,
            training_seconds,
        )

        results.append({
            "model":
                "xgboost",

            "fold":
                fold,

            **metrics,

            "training_seconds":
                training_seconds,
        })

        # ====================================================
        # 4. NEURAL NETWORK
        # ====================================================

        print(
            "\nTraining Neural Network..."
        )

        # Clear previous Keras state before
        # constructing each fold model.

        tf.keras.backend.clear_session()

        set_random_seeds(
            RANDOM_STATE + fold
        )

        neural_network = (
            build_neural_network(
                X_fold_train_processed.shape[1]
            )
        )

        early_stopping = (
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=NN_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
        )

        start = time.perf_counter()

        history = neural_network.fit(
            X_fold_train_processed,

            np.asarray(
                y_fold_train,
                dtype=np.float32,
            ),

            validation_data=(
                X_fold_val_processed,

                np.asarray(
                    y_fold_val,
                    dtype=np.float32,
                ),
            ),

            epochs=NN_EPOCHS,

            batch_size=NN_BATCH_SIZE,

            callbacks=[
                early_stopping
            ],

            verbose=0,

            shuffle=True,
        )

        training_seconds = (
            time.perf_counter()
            - start
        )

        probabilities = (
            neural_network.predict(
                X_fold_val_processed,
                batch_size=NN_BATCH_SIZE,
                verbose=0,
            )
            .reshape(-1)
        )

        metrics = calculate_metrics(
            y_fold_val,
            probabilities,
        )

        epochs_trained = len(
            history.history["loss"]
        )

        print_fold_result(
            "Neural Network",
            fold,
            metrics,
            training_seconds,
        )

        print(
            f"Epochs trained: "
            f"{epochs_trained}"
        )

        results.append({
            "model":
                "neural_network",

            "fold":
                fold,

            **metrics,

            "training_seconds":
                training_seconds,

            "epochs_trained":
                epochs_trained,
        })

        # ----------------------------------------------------
        # MEMORY CLEANUP
        # ----------------------------------------------------

        del logistic
        del random_forest
        del xgboost_model
        del neural_network

        tf.keras.backend.clear_session()

        gc.collect()

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
        .groupby("model")
        .agg(
            pr_auc_mean=(
                "pr_auc",
                "mean",
            ),

            pr_auc_std=(
                "pr_auc",
                "std",
            ),

            roc_auc_mean=(
                "roc_auc",
                "mean",
            ),

            roc_auc_std=(
                "roc_auc",
                "std",
            ),

            precision_mean=(
                "precision",
                "mean",
            ),

            precision_std=(
                "precision",
                "std",
            ),

            recall_mean=(
                "recall",
                "mean",
            ),

            recall_std=(
                "recall",
                "std",
            ),

            f1_mean=(
                "f1",
                "mean",
            ),

            f1_std=(
                "f1",
                "std",
            ),

            training_seconds_mean=(
                "training_seconds",
                "mean",
            ),
        )
        .reset_index()
    )

    # Sort only for convenient inspection.
    # This is NOT final model selection.

    summary = summary.sort_values(
        "pr_auc_mean",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 100)

    print(
        "5-FOLD CROSS-VALIDATION SUMMARY"
    )

    print("=" * 100)

    display_columns = [
        "model",
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "training_seconds_mean",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    fold_results_path = (
        artifact_dir
        / "fold_results.csv"
    )

    summary_path = (
        artifact_dir
        / "cv_summary.csv"
    )

    json_path = (
        artifact_dir
        / "cv_summary.json"
    )

    results_df.to_csv(
        fold_results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary.to_dict(
                orient="records"
            ),
            file,
            indent=4,
        )

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    expected_rows = (
        N_SPLITS
        * len(MODEL_NAMES)
    )

    if len(results_df) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows} "
            f"model/fold results, "
            f"found {len(results_df)}."
        )

    fold_counts = (
        results_df
        .groupby("model")["fold"]
        .nunique()
    )

    if not (
        fold_counts == N_SPLITS
    ).all():

        raise ValueError(
            "Not every model completed "
            "all CV folds."
        )

    print(
        "\nCross-validation consistency "
        "checks passed."
    )

    print("\nSaved artifacts:")

    print(
        f"Fold results : "
        f"{fold_results_path}"
    )

    print(
        f"CV summary   : "
        f"{summary_path}"
    )

    print(
        f"JSON summary : "
        f"{json_path}"
    )

    print("\n" + "=" * 72)

    print(
        "TASK 12 — STRATIFIED "
        "5-FOLD CROSS-VALIDATION COMPLETED"
    )

    print("=" * 72)

    print(
        """
IMPORTANT

- Cross-validation used ONLY train.csv.
- validation.csv remained untouched.
- test.csv remained untouched.
- Preprocessing was fitted separately inside every fold.
- No SMOTE/resampling was used.
- No class weighting was used.
- Threshold 0.50 was used for threshold-dependent metrics.
- PR-AUC was used as the primary ranking statistic.
- These results are for model comparison, not final test performance.
"""
    )


if __name__ == "__main__":
    main()