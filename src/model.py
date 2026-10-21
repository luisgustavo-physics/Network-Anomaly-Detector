"""Model management, training, inference, and serialization for Network Anomaly Detection."""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config.settings import DEFAULT_MODEL_FILE, MODELS_DIR, RF_PARAMS

logger = logging.getLogger(__name__)


class NetworkAnomalyModel:
    """Wrapper around RandomForestClassifier for multi-class and binary network anomaly detection."""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None) -> None:
        params = RF_PARAMS.copy()
        if model_params:
            params.update(model_params)
        self.params = params
        self.classifier: RandomForestClassifier = RandomForestClassifier(**self.params)
        self.is_trained: bool = False

    def train(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> None:
        """Train the Random Forest model on training partitions.

        Args:
            X_train: Training feature matrix.
            y_train: Target labels.
        """
        logger.info(
            "Starting model training with %d samples and %d features...",
            X_train.shape[0],
            X_train.shape[1],
        )
        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        elapsed = time.time() - start_time
        self.is_trained = True
        logger.info("Training complete in %.2f seconds.", elapsed)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate class predictions for input features."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() before predict().")
        return self.classifier.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Generate class prediction probability distributions."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() before predict_proba().")
        return self.classifier.predict_proba(X)

    def get_feature_importances(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: int = 15,
    ) -> List[Tuple[str, float]]:
        """Extract top feature importances for explainability and red team feature sensitivity analysis.

        Args:
            feature_names: List of column names corresponding to input features.
            top_n: Number of top contributing features to return.

        Returns:
            List of (feature_name, importance_score) sorted descending.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before extracting feature importances.")

        importances = self.classifier.feature_importances_
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Serialize the trained model to disk using joblib.

        Args:
            filepath: Target file path. Defaults to DEFAULT_MODEL_FILE.

        Returns:
            Path to the saved artifact.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")

        target_path = Path(filepath) if filepath else DEFAULT_MODEL_FILE
        target_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.classifier, target_path)
        logger.info("Model artifact successfully saved to: %s", target_path)
        return target_path

    def load(self, filepath: Optional[Union[str, Path]] = None) -> "NetworkAnomalyModel":
        """Deserialize a trained model from disk.

        Args:
            filepath: Path to the serialized .joblib file.

        Returns:
            Self instance with loaded classifier.
        """
        target_path = Path(filepath) if filepath else DEFAULT_MODEL_FILE
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at: {target_path}")

        logger.info("Loading model artifact from: %s", target_path)
        self.classifier = joblib.load(target_path)
        self.is_trained = True
        logger.info("Model loaded successfully. Classes: %s", getattr(self.classifier, "classes_", None))
        return self
