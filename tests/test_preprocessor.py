"""Tests for feature engineering, attack mapping, and data partitioning."""

import numpy as np
import pandas as pd
import pytest

from config.settings import CLASS_MAPPING, COLUMNS, NUMERIC_FEATURES
from src.preprocessor import KDDPreprocessor


def test_map_attack_to_multiclass():
    """Verify attack strings map to the expected tactical multi-class integer IDs."""
    preprocessor = KDDPreprocessor()

    # Normal traffic -> 0
    assert preprocessor.map_attack_to_multiclass("normal") == 0

    # DoS attacks -> 1
    assert preprocessor.map_attack_to_multiclass("neptune") == 1
    assert preprocessor.map_attack_to_multiclass("smurf") == 1
    assert preprocessor.map_attack_to_multiclass("back") == 1

    # Probe attacks -> 2
    assert preprocessor.map_attack_to_multiclass("ipsweep") == 2
    assert preprocessor.map_attack_to_multiclass("nmap") == 2
    assert preprocessor.map_attack_to_multiclass("satan") == 2

    # Privilege Escalation (U2R) -> 3
    assert preprocessor.map_attack_to_multiclass("buffer_overflow") == 3
    assert preprocessor.map_attack_to_multiclass("rootkit") == 3

    # Remote to Local / Unauthorized Access (R2L) -> 4
    assert preprocessor.map_attack_to_multiclass("guess_passwd") == 4
    assert preprocessor.map_attack_to_multiclass("warezclient") == 4

    # Unknown attack -> fallback 0
    assert preprocessor.map_attack_to_multiclass("unknown_variant") == 0


def test_map_attack_to_binary():
    """Verify binary mapping separates normal traffic from attacks."""
    preprocessor = KDDPreprocessor()
    assert preprocessor.map_attack_to_binary("normal") == 0
    assert preprocessor.map_attack_to_binary("neptune") == 1
    assert preprocessor.map_attack_to_binary("nmap") == 1
    assert preprocessor.map_attack_to_binary("rootkit") == 1


def test_fit_and_transform_features_alignment():
    """Verify preprocessor guarantees feature dimension and column order alignment."""
    preprocessor = KDDPreprocessor()

    # Create synthetic raw DataFrame
    data = {
        "protocol_type": ["tcp", "udp", "icmp"],
        "service": ["http", "private", "smtp"],
        "attack": ["normal", "neptune", "ipsweep"],
    }
    for feat in NUMERIC_FEATURES:
        data[feat] = [1.0, 2.0, 3.0]

    df = pd.DataFrame(data)
    transformed_train = preprocessor.fit_transform_features(df)

    assert preprocessor.fitted is True
    assert len(preprocessor.feature_columns) > len(NUMERIC_FEATURES)

    # Test with unseen record (missing smtp service, has http)
    test_data = {
        "protocol_type": ["tcp"],
        "service": ["http"],
    }
    for feat in NUMERIC_FEATURES:
        test_data[feat] = [0.5]

    test_df = pd.DataFrame(test_data)
    transformed_test = preprocessor.transform_features(test_df)

    # Output dimensions and columns must match training exactly
    assert list(transformed_test.columns) == list(transformed_train.columns)
    assert transformed_test.shape[1] == transformed_train.shape[1]
    assert transformed_test["service_smtp"].iloc[0] == 0.0
    assert transformed_test["service_http"].iloc[0] == 1.0


def test_split_data_proportions():
    """Verify data splitting maintains expected partition sizes."""
    preprocessor = KDDPreprocessor()
    n_samples = 100
    X = pd.DataFrame(np.random.randn(n_samples, 5))
    y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

    splits = preprocessor.split_data(X, y, test_size=0.2, val_size=0.3, random_state=42)

    # 100 * 0.2 = 20 in test
    assert len(splits["test_X"]) == 20
    # Remaining 80 split 70/30 -> 56 train, 24 val
    assert len(splits["train_X"]) == 56
    assert len(splits["val_X"]) == 24
