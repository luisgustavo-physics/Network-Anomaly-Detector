"""Tests for model training, prediction, serialization, and feature importance."""

import numpy as np
import pandas as pd
import pytest

from src.model import NetworkAnomalyModel


def test_model_train_predict(tmp_path):
    """Verify model can train, generate predictions, and serialize to disk."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 10), columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(np.random.choice([0, 1, 2, 3, 4], size=50))

    model = NetworkAnomalyModel(model_params={"n_estimators": 5, "random_state": 42})
    assert model.is_trained is False

    model.train(X, y)
    assert model.is_trained is True

    preds = model.predict(X)
    assert len(preds) == 50
    assert set(preds).issubset({0, 1, 2, 3, 4})

    proba = model.predict_proba(X)
    assert proba.shape[0] == 50

    # Test feature importances
    importances = model.get_feature_importances(feature_names=list(X.columns), top_n=5)
    assert len(importances) == 5
    assert importances[0][1] >= importances[1][1]

    # Test serialization and reload
    model_file = tmp_path / "test_model.joblib"
    model.save(model_file)
    assert model_file.exists()

    loaded_model = NetworkAnomalyModel()
    loaded_model.load(model_file)
    assert loaded_model.is_trained is True

    reloaded_preds = loaded_model.predict(X)
    np.testing.assert_array_equal(preds, reloaded_preds)
