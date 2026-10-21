"""Tests for data loading, schema validation, and dataset integrity."""

import pandas as pd
import pytest

from config.settings import COLUMNS
from src.data_loader import KDDDataLoader


def test_validate_schema_valid():
    """Verify that a correctly formatted DataFrame passes schema validation."""
    mock_data = {col: [0] for col in COLUMNS}
    df = pd.DataFrame(mock_data)
    # Should not raise exception
    KDDDataLoader.validate_schema(df)


def test_validate_schema_empty():
    """Verify that an empty DataFrame raises ValueError."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Loaded dataset is empty"):
        KDDDataLoader.validate_schema(df)


def test_validate_schema_missing_column():
    """Verify that missing critical columns raises ValueError."""
    mock_data = {col: [0] for col in COLUMNS[:-1]}  # missing last column
    df = pd.DataFrame(mock_data)
    with pytest.raises(ValueError, match="Schema mismatch"):
        KDDDataLoader.validate_schema(df)
