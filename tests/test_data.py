"""
Unit tests for data loading, cleaning, and preprocessing.
"""
import os
import pandas as pd
import pytest
from scripts.train import load_and_clean_data, DATA_PATH, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def test_dataset_exists():
    assert os.path.exists(DATA_PATH), f"Dataset file does not exist at {DATA_PATH}"
    assert os.path.getsize(DATA_PATH) > 100_000


def test_data_cleaning():
    df = load_and_clean_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7043
    assert TARGET in df.columns
    assert "customerID" not in df.columns

    # Verify TotalCharges has no nulls and is float
    assert df["TotalCharges"].isnull().sum() == 0
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])

    # Verify target encoding is binary 0 and 1
    unique_targets = set(df[TARGET].unique())
    assert unique_targets.issubset({0, 1})

    # Verify all expected features are present
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        assert col in df.columns, f"Feature {col} missing from cleaned data"
