"""
Task 9 — Random Forest Baseline
Financial Fraud Detection Project

Purpose
-------
Train and evaluate a Random Forest baseline using exactly the
same train/validation split used for Logistic Regression.

IMPORTANT
---------
- Training data is used for model fitting.
- Validation data is used for baseline evaluation.
- Test data is NOT loaded or evaluated.
- No SMOTE/resampling is performed.
- No class weighting is used in this baseline.
- Threshold 0.50 is a baseline threshold only.
- Hyperparameter tuning happens later.
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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
    validate_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

BASELINE_THRESHOLD = 0.50

MODEL_NAME = "random_forest_baseline"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path: Path):
    """
    Load dataset and separate features from target.
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
# BUILD RANDOM FOREST
# ============================================================

def build_random_forest():
    """
    Construct the untuned Random Forest baseline.

    These are deliberately reasonable baseline settings,
    not optimized hyperparameters.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=BASELINE_THRESHOLD,
):
    """
    Calculate fraud-focused classification metrics.
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
    Print Random Forest validation results.
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
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    feature_names,
):
    """
    Return Random Forest impurity-based feature importance.

    This is descriptive only and must not be interpreted
    as causal importance.
    """

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df


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
        "TASK 9 — RANDOM FOREST BASELINE"
    )

    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading training data..."
    )

    X_train, y_train = load_dataset(
        train_path
    )

    print(
        "Loading validation data..."
    )

    X_val, y_val = load_dataset(
        validation_path
    )

    print("\nDataset shapes:")

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

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print(
        "\nBuilding Random Forest..."
    )

    model = build_random_forest()

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "\nTraining Random Forest..."
    )

    start_time = time.perf_counter()

    model.fit(
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
        model.predict_proba(
            X_val
        )[:, 1]
    )

    # ========================================================
    # PROBABILITY SAFETY CHECK
    # ========================================================

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
    # EVALUATE
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

    # ========================================================
    # CONSISTENCY CHECKS
    # ========================================================

    confusion_total = (
        metrics["true_negatives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_positives"]
    )

    if confusion_total != len(y_val):

        raise ValueError(
            "Confusion matrix does not match "
            "validation dataset size."
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

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = (
        get_feature_importance(
            model,
            X_train.columns,
        )
    )

    print("\n" + "=" * 70)

    print(
        "TOP 15 RANDOM FOREST FEATURE IMPORTANCES"
    )

    print("=" * 70)

    print(
        feature_importance
        .head(15)
        .to_string(index=False)
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
    # SAVE FEATURE IMPORTANCE
    # ========================================================

    importance_path = (
        artifact_dir
        / "feature_importance.csv"
    )

    feature_importance.to_csv(
        importance_path,
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
    # SAVE MODEL
    # ========================================================

    model_path = (
        artifact_dir
        / "random_forest.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    # ========================================================
    # SAVED ARTIFACTS
    # ========================================================

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

    print(
        f"Importance  : {importance_path}"
    )

    # ========================================================
    # FINISH
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TASK 9 — RANDOM FOREST "
        "BASELINE COMPLETED"
    )

    print("=" * 70)

    print(
        """
IMPORTANT INTERPRETATION RULES

- These are validation results.
- The test set has NOT been evaluated.
- Threshold 0.50 is only the baseline threshold.
- Random Forest hyperparameters are NOT tuned.
- No SMOTE/resampling was performed.
- No class weighting was used.
- Feature importance is descriptive, not causal.
- Do not select the final model yet.
"""
    )


if __name__ == "__main__":
    main()