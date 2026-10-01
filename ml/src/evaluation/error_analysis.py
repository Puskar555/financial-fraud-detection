"""
Task 15 — Fraud Detection Error Analysis

Purpose
-------
Analyze false positives and false negatives produced by the tuned
fraud-detection models on the validation dataset.

IMPORTANT
---------
- Uses validation.csv only.
- test.csv is NEVER loaded.
- Uses models produced by Task 14.
- Uses Task 14 maximum-F1 thresholds for comparative error analysis.
- No model selection is performed here.
- No threshold is declared a production threshold here.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "validation.csv"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "threshold_optimization"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL PATHS
# ============================================================

RF_MODEL_PATH = (
    ARTIFACT_DIR
    / "tuned_random_forest.joblib"
)

XGB_MODEL_PATH = (
    ARTIFACT_DIR
    / "tuned_xgboost.joblib"
)

NN_MODEL_PATH = (
    ARTIFACT_DIR
    / "tuned_neural_network.keras"
)

NN_PREPROCESSOR_PATH = (
    ARTIFACT_DIR
    / "neural_network_preprocessor.joblib"
)


# ============================================================
# TASK 14 THRESHOLDS
# ============================================================

# Maximum-F1 thresholds found using validation.csv in Task 14.
#
# These are used only as consistent operating points for
# Task 15 error analysis.
#
# They are NOT automatically production thresholds.

THRESHOLDS = {
    "random_forest": 0.375,
    "xgboost": 0.130,
    "neural_network": 0.145,
}


TARGET_COLUMN = "Class"


# ============================================================
# DATA LOADING
# ============================================================

def load_validation_data():
    """
    Load validation data only.

    test.csv is intentionally never referenced.
    """

    print("\nLoading validation data...")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: "
            f"{VALIDATION_PATH}"
        )

    df = pd.read_csv(
        VALIDATION_PATH
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column "
            f"'{TARGET_COLUMN}' not found."
        )

    if df.empty:
        raise ValueError(
            "Validation dataset is empty."
        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[
        TARGET_COLUMN
    ].astype(int)

    print(
        f"Validation shape : {X.shape}"
    )

    print(
        f"Legitimate       : {(y == 0).sum():,}"
    )

    print(
        f"Fraud            : {(y == 1).sum():,}"
    )

    print(
        "\nTEST DATASET IS NOT LOADED."
    )

    return df, X, y


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    """
    Load Task 14 tuned models.
    """

    required_paths = [
        RF_MODEL_PATH,
        XGB_MODEL_PATH,
        NN_MODEL_PATH,
        NN_PREPROCESSOR_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required Task 14 artifact "
                f"not found: {path}"
            )

    print(
        "\nLoading Task 14 models..."
    )

    rf_model = joblib.load(
        RF_MODEL_PATH
    )

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    nn_model = tf.keras.models.load_model(
        NN_MODEL_PATH
    )

    nn_preprocessor = joblib.load(
        NN_PREPROCESSOR_PATH
    )

    print(
        "Models loaded successfully."
    )

    return (
        rf_model,
        xgb_model,
        nn_model,
        nn_preprocessor,
    )


# ============================================================
# PROBABILITY GENERATION
# ============================================================

def generate_probabilities(
    X,
    rf_model,
    xgb_model,
    nn_model,
    nn_preprocessor,
):
    """
    Generate fraud probabilities for all three models.
    """

    print(
        "\nGenerating validation probabilities..."
    )

    rf_prob = rf_model.predict_proba(
        X
    )[:, 1]

    xgb_prob = xgb_model.predict_proba(
        X
    )[:, 1]

    X_nn = nn_preprocessor.transform(
        X
    )

    X_nn = np.asarray(
        X_nn,
        dtype=np.float32,
    )

    nn_prob = (
        nn_model.predict(
            X_nn,
            batch_size=512,
            verbose=0,
        )
        .reshape(-1)
    )

    probabilities = {
        "random_forest": rf_prob,
        "xgboost": xgb_prob,
        "neural_network": nn_prob,
    }

    for model_name, prob in probabilities.items():

        prob = np.asarray(
            prob,
            dtype=float,
        )

        if len(prob) != len(X):

            raise RuntimeError(
                f"{model_name}: "
                f"prediction length mismatch."
            )

        if not np.isfinite(
            prob
        ).all():

            raise RuntimeError(
                f"{model_name}: "
                f"non-finite probabilities."
            )

        if (
            prob.min() < 0
            or prob.max() > 1
        ):

            raise RuntimeError(
                f"{model_name}: "
                f"probabilities outside [0, 1]."
            )

    return probabilities


# ============================================================
# CLASSIFICATION LABEL
# ============================================================

def classify_error(
    actual,
    predicted,
):
    """
    Assign TP, TN, FP or FN.
    """

    if actual == 1 and predicted == 1:
        return "TP"

    if actual == 0 and predicted == 0:
        return "TN"

    if actual == 0 and predicted == 1:
        return "FP"

    if actual == 1 and predicted == 0:
        return "FN"

    raise ValueError(
        "Unexpected classification values."
    )


# ============================================================
# MODEL ERROR TABLE
# ============================================================

def build_model_error_table(
    validation_df,
    y,
    probabilities,
    model_name,
    threshold,
):
    """
    Build row-level error table for one model.
    """

    prob = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        prob >= threshold
    ).astype(int)

    result = (
        validation_df
        .copy()
        .reset_index(drop=True)
    )

    result[
        "validation_row"
    ] = np.arange(
        len(result)
    )

    result[
        "fraud_probability"
    ] = prob

    result[
        "predicted_class"
    ] = predictions

    result[
        "error_type"
    ] = [
        classify_error(
            actual,
            predicted,
        )
        for actual, predicted
        in zip(
            y.to_numpy(),
            predictions,
        )
    ]

    result[
        "model"
    ] = model_name

    result[
        "threshold"
    ] = threshold

    return result


# ============================================================
# SUMMARY METRICS
# ============================================================

def summarize_model(
    model_name,
    y,
    probabilities,
    threshold,
):
    """
    Calculate model metrics at selected threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    return {
        "model": model_name,
        "threshold": threshold,

        "pr_auc":
            average_precision_score(
                y,
                probabilities,
            ),

        "precision":
            precision_score(
                y,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                predictions,
                zero_division=0,
            ),

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ============================================================
# ERROR AMOUNT SUMMARY
# ============================================================

def build_amount_summary(
    error_table,
):
    """
    Summarize Amount by TP/FP/FN/TN.

    Amount is treated as transaction value only.
    """

    if "Amount" not in error_table.columns:
        raise ValueError(
            "Amount column not found."
        )

    summary = (
        error_table
        .groupby(
            "error_type"
        )["Amount"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
                "sum",
            ]
        )
        .reset_index()
    )

    return summary


# ============================================================
# FALSE NEGATIVE SUMMARY
# ============================================================

def summarize_false_negatives(
    error_table,
):
    """
    Return descriptive information about missed fraud.
    """

    fn = error_table[
        error_table[
            "error_type"
        ] == "FN"
    ].copy()

    if fn.empty:

        return {
            "fn_count": 0,
            "fn_amount_sum": 0.0,
            "fn_amount_mean": np.nan,
            "fn_amount_median": np.nan,
            "fn_probability_mean": np.nan,
            "fn_probability_max": np.nan,
        }

    return {
        "fn_count":
            len(fn),

        "fn_amount_sum":
            float(
                fn["Amount"].sum()
            ),

        "fn_amount_mean":
            float(
                fn["Amount"].mean()
            ),

        "fn_amount_median":
            float(
                fn["Amount"].median()
            ),

        "fn_probability_mean":
            float(
                fn[
                    "fraud_probability"
                ].mean()
            ),

        "fn_probability_max":
            float(
                fn[
                    "fraud_probability"
                ].max()
            ),
    }


# ============================================================
# FALSE POSITIVE SUMMARY
# ============================================================

def summarize_false_positives(
    error_table,
):
    """
    Return descriptive information about false alarms.
    """

    fp = error_table[
        error_table[
            "error_type"
        ] == "FP"
    ].copy()

    if fp.empty:

        return {
            "fp_count": 0,
            "fp_amount_sum": 0.0,
            "fp_amount_mean": np.nan,
            "fp_amount_median": np.nan,
            "fp_probability_mean": np.nan,
            "fp_probability_max": np.nan,
        }

    return {
        "fp_count":
            len(fp),

        "fp_amount_sum":
            float(
                fp["Amount"].sum()
            ),

        "fp_amount_mean":
            float(
                fp["Amount"].mean()
            ),

        "fp_amount_median":
            float(
                fp["Amount"].median()
            ),

        "fp_probability_mean":
            float(
                fp[
                    "fraud_probability"
                ].mean()
            ),

        "fp_probability_max":
            float(
                fp[
                    "fraud_probability"
                ].max()
            ),
    }


# ============================================================
# MODEL AGREEMENT ANALYSIS
# ============================================================

def build_agreement_table(
    validation_df,
    probabilities,
):
    """
    Compare predictions made by all three models.
    """

    result = (
        validation_df
        .copy()
        .reset_index(drop=True)
    )

    result[
        "validation_row"
    ] = np.arange(
        len(result)
    )

    prediction_columns = []

    for model_name, prob in probabilities.items():

        threshold = THRESHOLDS[
            model_name
        ]

        probability_column = (
            f"{model_name}_probability"
        )

        prediction_column = (
            f"{model_name}_prediction"
        )

        result[
            probability_column
        ] = prob

        result[
            prediction_column
        ] = (
            prob >= threshold
        ).astype(int)

        prediction_columns.append(
            prediction_column
        )

    result[
        "models_predicting_fraud"
    ] = (
        result[
            prediction_columns
        ]
        .sum(axis=1)
    )

    result[
        "all_models_agree"
    ] = (
        result[
            prediction_columns
        ]
        .nunique(axis=1)
        == 1
    )

    return result


# ============================================================
# MISSED FRAUD AGREEMENT
# ============================================================

def build_missed_fraud_table(
    agreement_table,
):
    """
    Analyze fraud cases missed by one or more models.
    """

    fraud = agreement_table[
        agreement_table[
            TARGET_COLUMN
        ] == 1
    ].copy()

    fraud[
        "models_missing_fraud"
    ] = (
        3
        - fraud[
            "models_predicting_fraud"
        ]
    )

    fraud = fraud.sort_values(
        by=[
            "models_missing_fraud",
            "Amount",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return fraud


# ============================================================
# MODEL DISAGREEMENTS
# ============================================================

def build_disagreement_table(
    agreement_table,
):
    """
    Transactions where the models do not all agree.
    """

    disagreement = agreement_table[
        ~agreement_table[
            "all_models_agree"
        ]
    ].copy()

    disagreement = (
        disagreement.sort_values(
            by=[
                TARGET_COLUMN,
                "Amount",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    return disagreement


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "TASK 15 — FRAUD DETECTION ERROR ANALYSIS"
    )

    print(
        "=" * 72
    )

    (
        validation_df,
        X_val,
        y_val,
    ) = load_validation_data()

    (
        rf_model,
        xgb_model,
        nn_model,
        nn_preprocessor,
    ) = load_models()

    probabilities = (
        generate_probabilities(
            X_val,
            rf_model,
            xgb_model,
            nn_model,
            nn_preprocessor,
        )
    )

    model_summaries = []

    detailed_tables = {}

    fn_summaries = []

    fp_summaries = []

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MODEL ERROR SUMMARY"
    )

    print(
        "=" * 100
    )

    for (
        model_name,
        probability,
    ) in probabilities.items():

        threshold = THRESHOLDS[
            model_name
        ]

        table = (
            build_model_error_table(
                validation_df,
                y_val,
                probability,
                model_name,
                threshold,
            )
        )

        detailed_tables[
            model_name
        ] = table

        summary = summarize_model(
            model_name,
            y_val,
            probability,
            threshold,
        )

        model_summaries.append(
            summary
        )

        fn_summary = (
            summarize_false_negatives(
                table
            )
        )

        fn_summary[
            "model"
        ] = model_name

        fn_summaries.append(
            fn_summary
        )

        fp_summary = summarize_false_positives(
    table
 )

        fp_summary[
            "model"
        ] = model_name

        fp_summaries.append(
            fp_summary
        )

        print(
            f"\n{model_name}"
        )

        print(
            f"Threshold : "
            f"{threshold:.3f}"
        )

        print(
            f"PR-AUC    : "
            f"{summary['pr_auc']:.6f}"
        )

        print(
            f"Precision : "
            f"{summary['precision']:.6f}"
        )

        print(
            f"Recall    : "
            f"{summary['recall']:.6f}"
        )

        print(
            f"F1        : "
            f"{summary['f1']:.6f}"
        )

        print(
            f"TP={summary['tp']}  "
            f"FP={summary['fp']}  "
            f"FN={summary['fn']}  "
            f"TN={summary['tn']}"
        )

    summary_df = pd.DataFrame(
        model_summaries
    )

    fn_summary_df = pd.DataFrame(
        fn_summaries
    )

    fp_summary_df = pd.DataFrame(
        fp_summaries
    )

    # Reorder model column.

    fn_columns = [
        "model"
    ] + [
        column
        for column
        in fn_summary_df.columns
        if column != "model"
    ]

    fp_columns = [
        "model"
    ] + [
        column
        for column
        in fp_summary_df.columns
        if column != "model"
    ]

    fn_summary_df = (
        fn_summary_df[
            fn_columns
        ]
    )

    fp_summary_df = (
        fp_summary_df[
            fp_columns
        ]
    )

    # ========================================================
    # AGREEMENT ANALYSIS
    # ========================================================

    agreement_table = (
        build_agreement_table(
            validation_df,
            probabilities,
        )
    )

    missed_fraud_table = (
        build_missed_fraud_table(
            agreement_table
        )
    )

    disagreement_table = (
        build_disagreement_table(
            agreement_table
        )
    )

    # ========================================================
    # AMOUNT SUMMARIES
    # ========================================================

    amount_summaries = []

    for (
        model_name,
        table,
    ) in detailed_tables.items():

        amount_summary = (
            build_amount_summary(
                table
            )
        )

        amount_summary.insert(
            0,
            "model",
            model_name,
        )

        amount_summaries.append(
            amount_summary
        )

    amount_summary_df = pd.concat(
        amount_summaries,
        ignore_index=True,
    )

    # ========================================================
    # PRINT IMPORTANT ANALYSIS
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FALSE NEGATIVE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        fn_summary_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FALSE POSITIVE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        fp_summary_df.to_string(
            index=False
        )
    )

    fraud_count = int(
        (
            agreement_table[
                TARGET_COLUMN
            ] == 1
        ).sum()
    )

    fraud_detected_by_all = int(
        (
            (
                agreement_table[
                    TARGET_COLUMN
                ] == 1
            )
            &
            (
                agreement_table[
                    "models_predicting_fraud"
                ] == 3
            )
        ).sum()
    )

    fraud_missed_by_all = int(
        (
            (
                agreement_table[
                    TARGET_COLUMN
                ] == 1
            )
            &
            (
                agreement_table[
                    "models_predicting_fraud"
                ] == 0
            )
        ).sum()
    )

    fraud_detected_by_some = int(
        (
            (
                agreement_table[
                    TARGET_COLUMN
                ] == 1
            )
            &
            (
                agreement_table[
                    "models_predicting_fraud"
                ].between(
                    1,
                    2,
                )
            )
        ).sum()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CROSS-MODEL FRAUD AGREEMENT"
    )

    print(
        "=" * 100
    )

    print(
        f"Total fraud cases       : "
        f"{fraud_count}"
    )

    print(
        f"Detected by all 3       : "
        f"{fraud_detected_by_all}"
    )

    print(
        f"Detected by only some   : "
        f"{fraud_detected_by_some}"
    )

    print(
        f"Missed by all 3         : "
        f"{fraud_missed_by_all}"
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    summary_df.to_csv(
        OUTPUT_DIR
        / "model_error_summary.csv",
        index=False,
    )

    fn_summary_df.to_csv(
        OUTPUT_DIR
        / "false_negative_summary.csv",
        index=False,
    )

    fp_summary_df.to_csv(
        OUTPUT_DIR
        / "false_positive_summary.csv",
        index=False,
    )

    amount_summary_df.to_csv(
        OUTPUT_DIR
        / "error_amount_summary.csv",
        index=False,
    )

    agreement_table.to_csv(
        OUTPUT_DIR
        / "model_agreement.csv",
        index=False,
    )

    missed_fraud_table.to_csv(
        OUTPUT_DIR
        / "missed_fraud_cases.csv",
        index=False,
    )

    disagreement_table.to_csv(
        OUTPUT_DIR
        / "model_disagreements.csv",
        index=False,
    )

    for (
        model_name,
        table,
    ) in detailed_tables.items():

        errors_only = table[
            table[
                "error_type"
            ].isin(
                [
                    "FP",
                    "FN",
                ]
            )
        ].copy()

        errors_only.to_csv(
            OUTPUT_DIR
            / f"{model_name}_errors.csv",
            index=False,
        )

    metadata = {
        "dataset":
            "validation.csv",

        "test_dataset_loaded":
            False,

        "threshold_source":
            "Task 14 maximum-F1 "
            "validation thresholds",

        "thresholds":
            THRESHOLDS,

        "purpose":
            "Error analysis only. "
            "Thresholds are not automatically "
            "production operating points.",
    }

    with open(
        OUTPUT_DIR
        / "error_analysis_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SAVED ERROR ANALYSIS ARTIFACTS"
    )

    print(
        "=" * 72
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFiles:"
    )

    print(
        "- model_error_summary.csv"
    )

    print(
        "- false_negative_summary.csv"
    )

    print(
        "- false_positive_summary.csv"
    )

    print(
        "- error_amount_summary.csv"
    )

    print(
        "- model_agreement.csv"
    )

    print(
        "- missed_fraud_cases.csv"
    )

    print(
        "- model_disagreements.csv"
    )

    print(
        "- random_forest_errors.csv"
    )

    print(
        "- xgboost_errors.csv"
    )

    print(
        "- neural_network_errors.csv"
    )

    print(
        "- error_analysis_metadata.json"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TASK 15 — ERROR ANALYSIS COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        "\nIMPORTANT"
    )

    print(
        "- Analysis used validation.csv."
    )

    print(
        "- test.csv was NOT loaded."
    )

    print(
        "- No model was selected here."
    )

    print(
        "- No threshold was declared "
        "a production threshold."
    )

    print(
        "- False negatives represent "
        "fraud cases missed at the "
        "selected analysis threshold."
    )

    print(
        "- False positives represent "
        "legitimate transactions flagged "
        "as fraud."
    )


if __name__ == "__main__":
    main()