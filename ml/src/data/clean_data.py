from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "Class"

EXPECTED_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount",
    "Class",
]


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the credit-card transaction dataset.

    Raises
    ------
    ValueError
        If the dataset does not satisfy the expected schema
        or basic data-quality requirements.
    """

    # -------------------------------------------------
    # 1. Check whether dataset is empty
    # -------------------------------------------------

    if df.empty:
        raise ValueError("Dataset is empty.")

    # -------------------------------------------------
    # 2. Validate expected columns
    # -------------------------------------------------

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns found: {unexpected_columns}"
        )

    if len(df.columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected {len(EXPECTED_COLUMNS)} columns, "
            f"but found {len(df.columns)}."
        )

    # -------------------------------------------------
    # 3. Check numeric data types
    # -------------------------------------------------

    non_numeric_columns = (
        df[EXPECTED_COLUMNS]
        .select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise ValueError(
            f"Non-numeric columns found: {non_numeric_columns}"
        )

    # -------------------------------------------------
    # 4. Check missing values
    # -------------------------------------------------

    missing_values = int(
        df.isna().sum().sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values:,} missing values."
        )

    # -------------------------------------------------
    # 5. Validate target column
    # -------------------------------------------------

    target_values = set(
        df[TARGET_COLUMN].unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Unexpected target values: {target_values}. "
            "Expected only 0 and 1."
        )

    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "Dataset must contain both legitimate "
            "and fraudulent transactions."
        )

    # -------------------------------------------------
    # 6. Check infinite/non-finite feature values
    # -------------------------------------------------

    feature_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column != TARGET_COLUMN
    ]

    feature_values = (
        df[feature_columns]
        .to_numpy()
    )

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Feature data contains infinite "
            "or non-finite values."
        )

    # -------------------------------------------------
    # 7. Basic financial sanity checks
    # -------------------------------------------------

    if (df["Amount"] < 0).any():
        raise ValueError(
            "Negative transaction amounts detected."
        )

    if (df["Time"] < 0).any():
        raise ValueError(
            "Negative Time values detected."
        )

    print("Data validation passed.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the transaction dataset.

    Exact duplicate rows are removed according to the
    documented decision from the exploratory analysis.
    """

    # -------------------------------------------------
    # Validate raw dataset
    # -------------------------------------------------

    print("\nValidating raw dataset...")

    validate_data(df)

    # -------------------------------------------------
    # Record statistics before cleaning
    # -------------------------------------------------

    original_rows = len(df)

    legitimate_before = int(
        (df[TARGET_COLUMN] == 0).sum()
    )

    fraud_before = int(
        (df[TARGET_COLUMN] == 1).sum()
    )

    # -------------------------------------------------
    # Remove exact duplicate rows
    # -------------------------------------------------

    cleaned_df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    removed_duplicates = (
        original_rows - len(cleaned_df)
    )

    # -------------------------------------------------
    # Record statistics after cleaning
    # -------------------------------------------------

    legitimate_after = int(
        (cleaned_df[TARGET_COLUMN] == 0).sum()
    )

    fraud_after = int(
        (cleaned_df[TARGET_COLUMN] == 1).sum()
    )

    # -------------------------------------------------
    # Cleaning report
    # -------------------------------------------------

    print("\nCleaning summary")
    print("-" * 50)

    print(
        f"Original rows      : "
        f"{original_rows:,}"
    )

    print(
        f"Duplicates removed : "
        f"{removed_duplicates:,}"
    )

    print(
        f"Cleaned rows       : "
        f"{len(cleaned_df):,}"
    )

    print()

    print(
        f"Legitimate         : "
        f"{legitimate_before:,} -> "
        f"{legitimate_after:,}"
    )

    print(
        f"Fraud              : "
        f"{fraud_before:,} -> "
        f"{fraud_after:,}"
    )

    # -------------------------------------------------
    # Verify duplicate removal
    # -------------------------------------------------

    remaining_duplicates = int(
        cleaned_df.duplicated().sum()
    )

    if remaining_duplicates > 0:
        raise RuntimeError(
            f"{remaining_duplicates} duplicate rows "
            "remain after cleaning."
        )

    # -------------------------------------------------
    # Validate cleaned dataset again
    # -------------------------------------------------

    print("\nValidating cleaned dataset...")

    validate_data(cleaned_df)

    return cleaned_df


def main() -> None:
    """
    Execute the complete data-cleaning pipeline.
    """

    # -------------------------------------------------
    # Project paths
    # -------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    input_path = (
        project_root
        / "data"
        / "raw"
        / "creditcard.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "creditcard_cleaned.csv"
    )

    # -------------------------------------------------
    # Pipeline information
    # -------------------------------------------------

    print("=" * 60)
    print(
        "FINANCIAL FRAUD DETECTION - DATA CLEANING"
    )
    print("=" * 60)

    print("\nInput dataset:")
    print(input_path)

    # -------------------------------------------------
    # Check input file
    # -------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {input_path}"
        )

    # -------------------------------------------------
    # Load raw data
    # -------------------------------------------------

    df = pd.read_csv(input_path)

    print(
        f"\nLoaded {len(df):,} rows "
        f"and {len(df.columns)} columns."
    )

    # -------------------------------------------------
    # Clean data
    # -------------------------------------------------

    cleaned_df = clean_data(df)

    # -------------------------------------------------
    # Create output directory
    # -------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Save cleaned dataset
    # -------------------------------------------------

    cleaned_df.to_csv(
        output_path,
        index=False,
    )

    # -------------------------------------------------
    # Final report
    # -------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "CLEANING COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    print("\nSaved cleaned dataset to:")
    print(output_path)

    print(
        f"\nFinal dataset shape: "
        f"{cleaned_df.shape}"
    )


if __name__ == "__main__":
    main()