import numpy as np
import pandas as pd
import pytest

from ml.src.data.clean_data import (
    EXPECTED_COLUMNS,
    clean_data,
    validate_data,
)


def create_valid_dataframe() -> pd.DataFrame:
    """
    Create a small valid dataset using the same schema
    as the production fraud dataset.
    """

    rows = []

    for class_value in [0, 1]:
        row = {
            column: 1.0
            for column in EXPECTED_COLUMNS
        }

        row["Class"] = class_value
        rows.append(row)

    return pd.DataFrame(rows)


def test_valid_data_passes_validation():
    df = create_valid_dataframe()

    validate_data(df)


def test_missing_column_is_rejected():
    df = create_valid_dataframe()

    df = df.drop(columns=["V28"])

    with pytest.raises(
        ValueError,
        match="Missing expected columns",
    ):
        validate_data(df)


def test_unexpected_column_is_rejected():
    df = create_valid_dataframe()

    df["UnexpectedFeature"] = 1.0

    with pytest.raises(
        ValueError,
        match="Unexpected columns",
    ):
        validate_data(df)


def test_missing_values_are_rejected():
    df = create_valid_dataframe()

    df.loc[0, "Amount"] = np.nan

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        validate_data(df)


def test_invalid_target_is_rejected():
    df = create_valid_dataframe()

    df.loc[0, "Class"] = 2

    with pytest.raises(
        ValueError,
        match="Unexpected target values",
    ):
        validate_data(df)


def test_infinite_values_are_rejected():
    df = create_valid_dataframe()

    df.loc[0, "V1"] = np.inf

    with pytest.raises(
        ValueError,
        match="infinite or non-finite",
    ):
        validate_data(df)


def test_negative_amount_is_rejected():
    df = create_valid_dataframe()

    df.loc[0, "Amount"] = -100.0

    with pytest.raises(
        ValueError,
        match="Negative transaction amounts",
    ):
        validate_data(df)


def test_duplicate_rows_are_removed():
    df = create_valid_dataframe()

    duplicate_row = df.iloc[[0]].copy()

    df_with_duplicate = pd.concat(
        [df, duplicate_row],
        ignore_index=True,
    )

    assert len(df_with_duplicate) == 3

    cleaned_df = clean_data(df_with_duplicate)

    assert len(cleaned_df) == 2
    assert cleaned_df.duplicated().sum() == 0