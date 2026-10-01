"""
Task 10 — XGBoost Baseline
Financial Fraud Detection Project

Purpose
-------
Train and evaluate an XGBoost baseline using the same
train/validation split as the previous baseline models.

IMPORTANT
---------
- Training data is used for fitting.
- Validation data is used for evaluation.
- Test data is NOT loaded or evaluated.
- No SMOTE/resampling is performed.
- No scale_pos_weight is used in this initial baseline.
- Threshold 0.50 is a baseline threshold only.
- Hyperparameter tuning happens later.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from ml.src.features.preprocessing import (
    TARGET_COLUMN,
    validate_features,
)

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
BASELINE_THRESHOLD = 0.50
MODEL_NAME = "xgboost_baseline"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path: Path):
    """
    Load a split and separate predictors from target.
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
# BUILD XGBOOST
# ============================================================

def build_xgboost():
    """
    Construct an intentionally untuned XGBoost baseline.

    No scale_pos_weight is used yet so this remains comparable
    to our unweighted Logistic Regression and Random Forest
    baselines.
    """

    model = XGBClassifier(
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
    Print validation performance.
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
    Extract XGBoost feature importances.

    These are descriptive model importances.
    They must not be interpreted as causal effects.
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

    # --------------------------------------------------------
    # PROJECT PATHS
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
    print("TASK 10 — XGBOOST BASELINE")
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

    # --------------------------------------------------------
    # BASELINE CLASS IMBALANCE INFORMATION
    # --------------------------------------------------------

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    imbalance_ratio = (
        negative_count
        / positive_count
    )

    print(
        f"\nTraining imbalance ratio: "
        f"{imbalance_ratio:.2f}:1"
    )

    print(
        "scale_pos_weight       : 1.0 "
        "(baseline — weighting disabled)"
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print(
        "\nBuilding XGBoost..."
    )

    model = build_xgboost()

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    print(
        "\nTraining XGBoost..."
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

    # --------------------------------------------------------
    # VALIDATION PROBABILITIES
    # --------------------------------------------------------

    print(
        "\nGenerating validation probabilities..."
    )

    validation_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
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
            "Probabilities must be "
            "between 0 and 1."
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

    metrics["training_rows"] = len(X_train)

    metrics["validation_rows"] = len(X_val)

    metrics["training_fraud"] = int(
        y_train.sum()
    )

    metrics["validation_fraud"] = int(
        y_val.sum()
    )

    metrics["training_imbalance_ratio"] = float(
        imbalance_ratio
    )

    metrics["scale_pos_weight"] = 1.0

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
            "Confusion matrix total does "
            "not match validation size."
        )

    actual_fraud = (
        metrics["true_posititives"]
        if "true_posititives" in metrics
        else metrics["true_positives"]
    ) + metrics["false_negatives"]

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
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance = (
        get_feature_importance(
            model,
            X_train.columns,
        )
    )

    print("\n" + "=" * 70)

    print(
        "TOP 15 XGBOOST FEATURE IMPORTANCES"
    )

    print("=" * 70)

    print(
        feature_importance
        .head(15)
        .to_string(index=False)
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
    # SAVE FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_path = (
        artifact_dir
        / "feature_importance.csv"
    )

    feature_importance.to_csv(
        importance_path,
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
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        artifact_dir
        / "xgboost_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    # --------------------------------------------------------
    # REPORT ARTIFACTS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "TASK 10 — XGBOOST BASELINE COMPLETED"
    )

    print("=" * 70)

    print(
        """
IMPORTANT INTERPRETATION RULES

- These are VALIDATION results.
- The test dataset has NOT been evaluated.
- Threshold 0.50 is only the baseline threshold.
- Hyperparameters are not tuned.
- scale_pos_weight is intentionally disabled.
- No SMOTE/resampling was used.
- Feature importance is descriptive, not causal.
- Do not select the final model yet.
"""
    )


if __name__ == "__main__":
    main()