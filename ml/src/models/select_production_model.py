"""
Task 16 — Production Model Selection and Packaging

Purpose
-------
Package the selected XGBoost fraud-detection candidate for serving.

IMPORTANT
---------
- This script does NOT load train.csv.
- This script does NOT load validation.csv.
- This script does NOT load test.csv.
- It does NOT retrain a model.
- It does NOT tune a threshold.
- It packages the already-trained Task 14 XGBoost model.
- test.csv remains untouched for final evaluation.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "xgboost"

MODEL_VERSION = "1.0.0"

MODEL_STAGE = "production_candidate"

THRESHOLD = 0.130

THRESHOLD_SOURCE = (
    "Task 14 validation threshold optimization "
    "(maximum-F1 candidate)"
)

EXPECTED_FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
]


# ============================================================
# HASH
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# VALIDATE MODEL
# ============================================================

def validate_xgboost_model(model):

    required_methods = [
        "predict",
        "predict_proba",
    ]

    for method in required_methods:

        if not hasattr(model, method):

            raise TypeError(
                f"Loaded model does not provide "
                f"required method: {method}"
            )

    if (
    hasattr(model, "n_features_in_")
    and model.n_features_in_
    != len(EXPECTED_FEATURES)
    ):
        raise ValueError(
        "Model feature count mismatch. "
        f"Expected {len(EXPECTED_FEATURES)}, "
        f"got {model.n_features_in_}."
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

    source_model = (
        project_root
        / "artifacts"
        / "threshold_optimization"
        / "tuned_xgboost.joblib"
    )

    production_root = (
        project_root
        / "artifacts"
        / "production"
    )

    version_dir = (
        production_root
        / MODEL_NAME
        / MODEL_VERSION
    )

    version_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not source_model.exists():

        raise FileNotFoundError(
            f"Task 14 XGBoost artifact not found: "
            f"{source_model}"
        )

    print("=" * 76)

    print(
        "TASK 16 — PRODUCTION MODEL PACKAGING"
    )

    print("=" * 76)

    print(
        "\nNo dataset will be loaded."
    )

    print(
        "test.csv will NOT be loaded."
    )

    print(
        "\nLoading existing Task 14 "
        "XGBoost artifact..."
    )

    model = joblib.load(
        source_model
    )

    validate_xgboost_model(
        model
    )

    print(
        "Model artifact validation passed."
    )

    # --------------------------------------------------------
    # COPY MODEL INTO VERSIONED PRODUCTION DIRECTORY
    # --------------------------------------------------------

    production_model_path = (
        version_dir
        / "model.joblib"
    )

    shutil.copy2(
        source_model,
        production_model_path,
    )

    model_hash = sha256_file(
        production_model_path
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "model_name":
            MODEL_NAME,

        "model_version":
            MODEL_VERSION,

        "stage":
            MODEL_STAGE,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model_file":
            "model.joblib",

        "model_sha256":
            model_hash,

        "target":
            "Class",

        "positive_class":
            1,

        "negative_class":
            0,

        "expected_feature_count":
            len(EXPECTED_FEATURES),

        "expected_features":
            EXPECTED_FEATURES,

        "decision_threshold":
            THRESHOLD,

        "threshold_source":
            THRESHOLD_SOURCE,

        "probability_rule":
            (
                "predict_proba(X)[:, 1] >= "
                "decision_threshold"
            ),

        "selection_evidence": {
            "tuned_cv_pr_auc":
                0.857653,

            "validation_pr_auc":
                0.845791,

            "validation_roc_auc":
                0.973013,

            "validation_precision_at_threshold":
                0.981818,

            "validation_recall_at_threshold":
                0.760563,

            "validation_f1_at_threshold":
                0.857143,

            "validation_true_positives":
                54,

            "validation_false_positives":
                1,

            "validation_false_negatives":
                17,

            "validation_true_negatives":
                42487,
        },

        "selection_notes": [
            (
                "XGBoost had the highest tuned "
                "cross-validation PR-AUC among "
                "the tuned candidate models."
            ),
            (
                "XGBoost had the highest Task 14 "
                "validation PR-AUC among the "
                "candidate models."
            ),
            (
                "Random Forest remained competitive "
                "and achieved higher recall and F1 "
                "at its own Task 14 maximum-F1 "
                "threshold."
            ),
            (
                "The selected threshold is a "
                "validation-derived operating-point "
                "candidate and is not a test-set "
                "optimized threshold."
            ),
        ],

        "preprocessing": {
            "required":
                False,

            "description":
                (
                    "Task 14 XGBoost consumes the "
                    "30 numerical dataset features "
                    "directly. No StandardScaler "
                    "artifact is required."
                ),
        },

        "data_policy": {
            "training_source":
                "train.csv",

            "threshold_selection_source":
                "validation.csv",

            "test_used_for_training":
                False,

            "test_used_for_model_selection":
                False,

            "test_used_for_threshold_selection":
                False,
        },
    }

    metadata_path = (
        version_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # CURRENT POINTER
    # --------------------------------------------------------

    current = {
        "model_name":
            MODEL_NAME,

        "model_version":
            MODEL_VERSION,

        "stage":
            MODEL_STAGE,

        "artifact_directory":
            str(
                Path(MODEL_NAME)
                / MODEL_VERSION
            ),

        "model_file":
            "model.joblib",

        "metadata_file":
            "metadata.json",

        "decision_threshold":
            THRESHOLD,

        "model_sha256":
            model_hash,
    }

    current_path = (
        production_root
        / "current.json"
    )

    with current_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            current,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # VERIFY COPIED ARTIFACT
    # --------------------------------------------------------

    copied_model = joblib.load(
        production_model_path
    )

    validate_xgboost_model(
        copied_model
    )

    copied_hash = sha256_file(
        production_model_path
    )

    if copied_hash != model_hash:

        raise RuntimeError(
            "Production model checksum "
            "verification failed."
        )

    print(
        "\nProduction candidate:"
    )

    print(
        f"Model      : {MODEL_NAME}"
    )

    print(
        f"Version    : {MODEL_VERSION}"
    )

    print(
        f"Stage      : {MODEL_STAGE}"
    )

    print(
        f"Threshold  : {THRESHOLD:.3f}"
    )

    print(
        f"SHA-256    : {model_hash}"
    )

    print(
        "\nPackaged files:"
    )

    print(
        production_model_path
    )

    print(
        metadata_path
    )

    print(
        current_path
    )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "TASK 16 — PRODUCTION PACKAGING COMPLETED"
    )

    print(
        "=" * 76
    )

    print(
        """
IMPORTANT

- Existing Task 14 model was packaged.
- No model was retrained.
- No dataset was loaded.
- test.csv was NOT loaded.
- Threshold 0.130 came from validation analysis.
- Model artifact is versioned.
- SHA-256 checksum was generated.
- current.json identifies the active production candidate.
- Final untouched-test evaluation remains separate.
"""
    )


if __name__ == "__main__":
    main()