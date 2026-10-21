"""Real-time and batch network flow inference engine."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config.settings import (
    CLASS_LABELS,
    DEFAULT_MODEL_FILE,
    DEFAULT_PREPROCESSOR_FILE,
)
from src.model import NetworkAnomalyModel
from src.preprocessor import KDDPreprocessor

logger = logging.getLogger(__name__)


class NetworkAnomalyPredictor:
    """Production inference interface for scoring live network traffic telemetry."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_FILE
        self.preprocessor_path = (
            Path(preprocessor_path) if preprocessor_path else DEFAULT_PREPROCESSOR_FILE
        )

        self.model = NetworkAnomalyModel()
        self.preprocessor = KDDPreprocessor()
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model and feature preprocessor states."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")
        self.model.load(self.model_path)

        if self.preprocessor_path.exists():
            self.preprocessor.load(self.preprocessor_path)
        else:
            # Fallback: if preprocessor_metadata does not exist, use feature_names_in_ from classifier
            if hasattr(self.model.classifier, "feature_names_in_"):
                self.preprocessor.feature_columns = list(self.model.classifier.feature_names_in_)
                self.preprocessor.fitted = True
            else:
                logger.warning("No preprocessor metadata or model feature_names_in_ available.")

    def predict(
        self,
        traffic_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Predict anomaly category and confidence for one or more network flow records.

        Args:
            traffic_data: Input records as DataFrame, dict, or list of dicts.

        Returns:
            List of result dictionaries containing class_id, label, is_anomaly, and probabilities.
        """
        if isinstance(traffic_data, dict):
            df = pd.DataFrame([traffic_data])
        elif isinstance(traffic_data, list):
            df = pd.DataFrame(traffic_data)
        elif isinstance(traffic_data, pd.DataFrame):
            df = traffic_data.copy()
        else:
            raise TypeError("Expected traffic_data to be a dict, list of dicts, or pd.DataFrame")

        # Transform using aligned preprocessor
        X_aligned = self.preprocessor.transform_features(df)

        predictions = self.model.predict(X_aligned)
        probabilities = self.model.predict_proba(X_aligned)

        results = []
        for i, (pred_class, prob_dist) in enumerate(zip(predictions, probabilities)):
            class_id = int(pred_class)
            label = CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else "Unknown"
            is_anomaly = class_id != 0

            # Map class probabilities
            prob_dict = {
                CLASS_LABELS[c]: round(float(prob_dist[c]), 4)
                for c in range(min(len(CLASS_LABELS), len(prob_dist)))
            }

            results.append({
                "record_index": i,
                "predicted_class_id": class_id,
                "predicted_label": label,
                "is_anomaly": is_anomaly,
                "confidence": round(float(np.max(prob_dist)), 4),
                "class_probabilities": prob_dict,
            })

        return results
