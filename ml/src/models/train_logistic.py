"""
Task 8 — Logistic Regression Baseline
Financial Fraud Detection Project

This script:

1. Loads training and validation data.
2. Builds a leakage-safe sklearn Pipeline.
3. Fits preprocessing only through the training pipeline.
4. Trains Logistic Regression.
5. Generates validation probabilities.
6. Evaluates the default threshold of 0.50.
7. Reports fraud-focused metrics.
8. Saves the trained pipeline and validation metrics.

IMPORTANT:
- Test data is NOT evaluated here.
- No SMOTE/resampling is performed.
- Threshold 0.50 is only the baseline threshold.
- Threshold optimization will be performed later.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# Import preprocessing function created in Task 7.
#
# Run this script from the PROJECT ROOT:
# python -m ml.src.models.train_logistic
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

MODEL_NAME = "logistic_regression_baseline"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path: Path):
    """Load a dataset and separate predictors and target."""

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

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN].copy()

    validate_features(X)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            f"Invalid target values in {path.name}."
        )

    if len(y.unique()) != 2:
        raise ValueError(
            f"{path.name} must contain both classes."
        )

    return X, y


# ============================================================
# BUILD MODEL PIPELINE
# ============================================================

def build_logistic_pipeline():
    """
    Create preprocessing + Logistic Regression pipeline.

    Preprocessing is inside the Pipeline so that later
    cross-validation can fit preprocessing independently
    within each training fold.
    """

    preprocessor = build_preprocessor()

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=BASELINE_THRESHOLD,
):
    """
    Calculate fraud-focused evaluation metrics.
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
# PRINT METRICS
# ============================================================

def print_evaluation(
    y_true,
    predictions,
    metrics,
):
    """Print baseline model evaluation."""

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
        "TASK 8 — LOGISTIC REGRESSION BASELINE"
    )
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading training data...")

    X_train, y_train = load_dataset(
        train_path
    )

    print("Loading validation data...")

    X_val, y_val = load_dataset(
        validation_path
    )

    print("\nDataset shapes:")

    print(
        f"Training   : "
        f"{X_train.shape}"
    )

    print(
        f"Validation : "
        f"{X_val.shape}"
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

    # ========================================================
    # BUILD PIPELINE
    # ========================================================

    print(
        "\nBuilding preprocessing + "
        "Logistic Regression pipeline..."
    )

    pipeline = build_logistic_pipeline()

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "\nTraining Logistic Regression..."
    )

    start_time = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train,
    )

    training_seconds = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Training completed in "
        f"{training_seconds:.2f} seconds."
    )

    # ========================================================
    # VALIDATION PROBABILITIES
    # ========================================================

    print(
        "\nGenerating validation probabilities..."
    )

    validation_probabilities = (
        pipeline.predict_proba(
            X_val
        )[:, 1]
    )

    if not np.isfinite(
        validation_probabilities
    ).all():
        raise ValueError(
            "Non-finite validation probabilities detected."
        )

    if (
        (validation_probabilities < 0).any()
        or
        (validation_probabilities > 1).any()
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    # ========================================================
    # EVALUATION
    # ========================================================

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

    metrics["training_rows"] = len(X_train)

    metrics["validation_rows"] = len(X_val)

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

    # ========================================================
    # SIMPLE CONSISTENCY CHECK
    # ========================================================

    confusion_total = (
        metrics["true_negatives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_positives"]
    )

    if confusion_total != len(y_val):
        raise ValueError(
            "Confusion matrix count does not "
            "match validation dataset size."
        )

    detected_fraud = (
        metrics["true_positives"]
        + metrics["false_negatives"]
    )

    if detected_fraud != int(
        y_val.sum()
    ):
        raise ValueError(
            "Fraud count does not match "
            "validation target."
        )

    print(
        "Metric consistency checks passed."
    )

    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

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

    # ========================================================
    # SAVE METRICS
    # ========================================================

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

    # ========================================================
    # SAVE COMPLETE PIPELINE
    # ========================================================

    model_path = (
        artifact_dir
        / "logistic_pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print("\nSaved artifacts:")

    print(
        f"Model       : {model_path}"
    )

    print(
        f"Metrics     : {metrics_path}"
    )

    print(
        f"Predictions : {prediction_path}"
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TASK 8 — LOGISTIC REGRESSION "
        "BASELINE COMPLETED"
    )

    print("=" * 70)

    print(
        """
IMPORTANT INTERPRETATION RULES

- These are VALIDATION results, not final test results.
- The test dataset has not been evaluated.
- Threshold 0.50 is only the baseline threshold.
- Accuracy is intentionally not the primary metric.
- No SMOTE or resampling was used.
- Do not claim this is the best model yet.
"""
    )


if __name__ == "__main__":
    main()