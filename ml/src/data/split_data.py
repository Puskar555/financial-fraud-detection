"""
Create leakage-safe train, validation, and test datasets
for the Financial Fraud Detection project.

Strategy
--------
Training   : 70%
Validation : 15%
Test       : 15%

Stratification preserves the fraud/legitimate class distribution
in every split.

IMPORTANT:
No scaling, SMOTE, feature selection, model fitting, or learned
transformation is performed before splitting.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Class"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# VALIDATION
# ============================================================

def validate_input_data(df: pd.DataFrame) -> None:
    """Validate dataset before splitting."""

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    if df.isna().sum().sum() != 0:
        raise ValueError(
            "Dataset contains missing values."
        )

    if df.duplicated().sum() != 0:
        raise ValueError(
            "Dataset contains duplicate rows."
        )

    target_values = set(df[TARGET_COLUMN].unique())

    if target_values != {0, 1}:
        raise ValueError(
            "Target column must contain both classes 0 and 1."
        )

    print("Input validation passed.")


# ============================================================
# SPLIT FUNCTION
# ============================================================

def split_data(df: pd.DataFrame):
    """
    Split cleaned data into train, validation, and test sets.

    Returns
    -------
    X_train, X_val, X_test,
    y_train, y_val, y_test
    """

    validate_input_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()

    # --------------------------------------------------------
    # First split:
    # 70% training
    # 30% temporary (validation + test)
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(VALIDATION_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --------------------------------------------------------
    # Second split:
    # Divide temporary data equally into validation and test.
    #
    # 30% temporary -> 15% validation + 15% test
    # --------------------------------------------------------

    relative_test_size = (
        TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# SPLIT REPORT
# ============================================================

def print_split_report(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):
    """Print class distribution for every split."""

    print("\n" + "=" * 70)
    print("TRAIN / VALIDATION / TEST SPLIT REPORT")
    print("=" * 70)

    datasets = {
        "Training": (X_train, y_train),
        "Validation": (X_val, y_val),
        "Test": (X_test, y_test),
    }

    total_rows = sum(
        len(y)
        for _, y in datasets.values()
    )

    for name, (X_split, y_split) in datasets.items():

        rows = len(y_split)

        legitimate = int((y_split == 0).sum())
        fraud = int((y_split == 1).sum())

        fraud_rate = (
            fraud / rows * 100
        )

        dataset_percentage = (
            rows / total_rows * 100
        )

        print(f"\n{name} set")
        print("-" * 40)

        print(
            f"Rows             : {rows:,}"
        )

        print(
            f"Dataset share    : "
            f"{dataset_percentage:.2f}%"
        )

        print(
            f"Features         : "
            f"{X_split.shape[1]}"
        )

        print(
            f"Legitimate       : "
            f"{legitimate:,}"
        )

        print(
            f"Fraud            : "
            f"{fraud:,}"
        )

        print(
            f"Fraud rate       : "
            f"{fraud_rate:.4f}%"
        )


# ============================================================
# SAFETY CHECKS
# ============================================================

def validate_splits(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):
    """Verify integrity of generated splits."""

    # --------------------------------------------------------
    # Check row counts
    # --------------------------------------------------------

    total_rows = (
        len(X_train)
        + len(X_val)
        + len(X_test)
    )

    if total_rows == 0:
        raise ValueError(
            "Generated splits are empty."
        )

    # --------------------------------------------------------
    # X/y lengths must match
    # --------------------------------------------------------

    split_pairs = [
        ("Training", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]

    for name, X_split, y_split in split_pairs:

        if len(X_split) != len(y_split):
            raise ValueError(
                f"{name} X/y lengths do not match."
            )

        if len(X_split) == 0:
            raise ValueError(
                f"{name} split is empty."
            )

        if set(y_split.unique()) != {0, 1}:
            raise ValueError(
                f"{name} split does not contain both classes."
            )

    # --------------------------------------------------------
    # Check index overlap
    #
    # Because train_test_split preserves original indices,
    # these sets should be completely disjoint.
    # --------------------------------------------------------

    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    if train_indices & val_indices:
        raise ValueError(
            "Data leakage detected between train and validation."
        )

    if train_indices & test_indices:
        raise ValueError(
            "Data leakage detected between train and test."
        )

    if val_indices & test_indices:
        raise ValueError(
            "Data leakage detected between validation and test."
        )

    print("\nSplit integrity checks passed.")
    print("No row overlap detected.")


# ============================================================
# SAVE SPLITS
# ============================================================

def save_splits(
    output_dir: Path,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):
    """Save train, validation, and test datasets."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Combine features and target before saving.

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    val_df = X_val.copy()
    val_df[TARGET_COLUMN] = y_val

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    # Reset indices only when writing the files.
    # Original indices were retained above for overlap checks.

    train_df.reset_index(
        drop=True
    ).to_csv(
        output_dir / "train.csv",
        index=False,
    )

    val_df.reset_index(
        drop=True
    ).to_csv(
        output_dir / "validation.csv",
        index=False,
    )

    test_df.reset_index(
        drop=True
    ).to_csv(
        output_dir / "test.csv",
        index=False,
    )

    print("\nSaved split datasets:")

    print(
        output_dir / "train.csv"
    )

    print(
        output_dir / "validation.csv"
    )

    print(
        output_dir / "test.csv"
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

    input_path = (
        project_root
        / "data"
        / "processed"
        / "creditcard_cleaned.csv"
    )

    output_dir = (
        project_root
        / "data"
        / "processed"
        / "splits"
    )

    print("=" * 70)
    print("FINANCIAL FRAUD DETECTION — DATA SPLITTING")
    print("=" * 70)

    print(
        f"\nInput dataset:\n{input_path}"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {input_path}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        input_path
    )

    print(
        f"\nLoaded {len(df):,} rows "
        f"and {df.shape[1]} columns."
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_splits(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_split_report(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_splits(
        output_dir,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    print("\n" + "=" * 70)
    print("TASK 6 — DATA SPLITTING COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()