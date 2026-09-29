"""
Leakage-safe preprocessing pipeline for the
Financial Fraud Detection project.

Preprocessing strategy
----------------------
V1-V28:
    Passed through unchanged.

Time:
    StandardScaler.

Amount:
    StandardScaler.

Class:
    Target variable. Never transformed.

IMPORTANT
---------
The preprocessing transformer is fitted ONLY on training data.
Validation and test data are transformed using the fitted
training transformer.

No resampling or SMOTE is performed here.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Class"

SCALED_FEATURES = [
    "Time",
    "Amount",
]

V_FEATURES = [
    f"V{i}"
    for i in range(1, 29)
]

EXPECTED_FEATURES = (
    ["Time"]
    + V_FEATURES
    + ["Amount"]
)


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_features(X: pd.DataFrame) -> None:
    """
    Validate feature dataframe before preprocessing.
    """

    if X.empty:
        raise ValueError(
            "Feature dataframe is empty."
        )

    missing_columns = [
        column
        for column in EXPECTED_FEATURES
        if column not in X.columns
    ]

    unexpected_columns = [
        column
        for column in X.columns
        if column not in EXPECTED_FEATURES
    ]

    if missing_columns:
        raise ValueError(
            f"Missing features: {missing_columns}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected features: {unexpected_columns}"
        )

    if X.shape[1] != len(EXPECTED_FEATURES):
        raise ValueError(
            "Incorrect number of predictor features."
        )

    if X.isna().sum().sum() != 0:
        raise ValueError(
            "Feature dataframe contains missing values."
        )

    if not all(
        pd.api.types.is_numeric_dtype(X[column])
        for column in X.columns
    ):
        raise ValueError(
            "All predictor features must be numeric."
        )

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Features contain infinite or non-finite values."
        )


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Construct the preprocessing transformer.

    Time and Amount are standardized.

    V1-V28 pass through unchanged.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scale_time_amount",
                StandardScaler(),
                SCALED_FEATURES,
            ),
            (
                "v_features",
                "passthrough",
                V_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(path: Path):
    """
    Load one saved dataset split and separate
    predictors from target.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}"
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found "
            f"in {path.name}."
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

    return X, y


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
        / "preprocessing"
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

    test_path = (
        split_dir
        / "test.csv"
    )

    print("=" * 70)
    print(
        "FINANCIAL FRAUD DETECTION — "
        "PREPROCESSING PIPELINE"
    )
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading train split...")

    X_train, y_train = load_split(
        train_path
    )

    print("Loading validation split...")

    X_val, y_val = load_split(
        validation_path
    )

    print("Loading test split...")

    X_test, y_test = load_split(
        test_path
    )

    print("\nOriginal shapes:")

    print(
        f"X_train : {X_train.shape}"
    )

    print(
        f"X_val   : {X_val.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    # ========================================================
    # BUILD PREPROCESSOR
    # ========================================================

    preprocessor = build_preprocessor()

    # ========================================================
    # CRITICAL LEAKAGE-SAFE STEP
    # ========================================================

    print(
        "\nFitting preprocessor on TRAINING DATA ONLY..."
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

    print(
        "Transforming test data..."
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # ========================================================
    # FEATURE NAMES
    # ========================================================

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    print(
        f"\nProcessed feature count: "
        f"{len(feature_names)}"
    )

    print(
        f"Feature names:\n{feature_names}"
    )

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    expected_feature_count = 30

    if X_train_processed.shape[1] != expected_feature_count:
        raise ValueError(
            "Training feature count changed unexpectedly."
        )

    if X_val_processed.shape[1] != expected_feature_count:
        raise ValueError(
            "Validation feature count changed unexpectedly."
        )

    if X_test_processed.shape[1] != expected_feature_count:
        raise ValueError(
            "Test feature count changed unexpectedly."
        )

    if not np.isfinite(
        X_train_processed
    ).all():
        raise ValueError(
            "Training data contains non-finite values "
            "after preprocessing."
        )

    if not np.isfinite(
        X_val_processed
    ).all():
        raise ValueError(
            "Validation data contains non-finite values "
            "after preprocessing."
        )

    if not np.isfinite(
        X_test_processed
    ).all():
        raise ValueError(
            "Test data contains non-finite values "
            "after preprocessing."
        )

    print(
        "\nProcessed shapes:"
    )

    print(
        f"Training   : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Validation : "
        f"{X_val_processed.shape}"
    )

    print(
        f"Test       : "
        f"{X_test_processed.shape}"
    )

    # ========================================================
    # VERIFY STANDARDIZATION
    # ========================================================

    time_index = feature_names.index(
        "Time"
    )

    amount_index = feature_names.index(
        "Amount"
    )

    train_time_mean = (
        X_train_processed[
            :, time_index
        ].mean()
    )

    train_time_std = (
        X_train_processed[
            :, time_index
        ].std()
    )

    train_amount_mean = (
        X_train_processed[
            :, amount_index
        ].mean()
    )

    train_amount_std = (
        X_train_processed[
            :, amount_index
        ].std()
    )

    print("\nTraining scaling verification:")

    print(
        f"Time mean   : "
        f"{train_time_mean:.8f}"
    )

    print(
        f"Time std    : "
        f"{train_time_std:.8f}"
    )

    print(
        f"Amount mean : "
        f"{train_amount_mean:.8f}"
    )

    print(
        f"Amount std  : "
        f"{train_amount_std:.8f}"
    )

    # StandardScaler should produce approximately
    # mean 0 and standard deviation 1 on training data.

    if not np.isclose(
        train_time_mean,
        0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Training Time mean is not approximately zero."
        )

    if not np.isclose(
        train_time_std,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Training Time standard deviation is "
            "not approximately one."
        )

    if not np.isclose(
        train_amount_mean,
        0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Training Amount mean is not approximately zero."
        )

    if not np.isclose(
        train_amount_std,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Training Amount standard deviation is "
            "not approximately one."
        )

    print(
        "\nScaling verification passed."
    )

    # ========================================================
    # VERIFY V FEATURES WERE NOT MODIFIED
    # ========================================================

    print(
        "\nChecking passthrough V1-V28 features..."
    )

    for feature in V_FEATURES:

        processed_index = (
            feature_names.index(
                feature
            )
        )

        original_values = (
            X_train[feature]
            .to_numpy()
        )

        processed_values = (
            X_train_processed[
                :,
                processed_index
            ]
        )

        if not np.allclose(
            original_values,
            processed_values,
        ):
            raise ValueError(
                f"{feature} changed unexpectedly "
                f"during preprocessing."
            )

    print(
        "V1-V28 passthrough verification passed."
    )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    print("\nTarget distributions:")

    print(
        f"Training fraud   : "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Validation fraud : "
        f"{int(y_val.sum()):,}"
    )

    print(
        f"Test fraud       : "
        f"{int(y_test.sum()):,}"
    )

    # ========================================================
    # SAVE PREPROCESSOR
    # ========================================================

    preprocessor_path = (
        artifact_dir
        / "preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    print(
        "\nSaved fitted preprocessor:"
    )

    print(
        preprocessor_path
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TASK 7 — PREPROCESSING "
        "COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print(
        "\nIMPORTANT:"
        "\n- Preprocessor fitted on training data only."
        "\n- Validation/test data were transform-only."
        "\n- V1-V28 were passed through unchanged."
        "\n- Time and Amount were standardized."
        "\n- No SMOTE/resampling was performed."
        "\n- No model was trained."
    )


if __name__ == "__main__":
    main()