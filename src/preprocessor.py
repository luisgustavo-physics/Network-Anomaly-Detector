"""Feature engineering, categorical encoding, attack mapping, and dataset partitioning."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import (
    ATTACK_CATEGORIES,
    CLASS_MAPPING,
    DEFAULT_PREPROCESSOR_FILE,
    FEATURES_TO_ENCODE,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


class KDDPreprocessor:
    """Preprocesses raw NSL-KDD telemetry data into model-ready feature sets and targets."""

    def __init__(self) -> None:
        self.feature_columns: Optional[List[str]] = None
        self.fitted: bool = False

    @staticmethod
    def map_attack_to_multiclass(attack_label: str) -> int:
        """Map raw attack label string into a tactical multi-class integer ID (0-4).

        0 = Normal
        1 = Denial of Service (DoS)
        2 = Probe / Network Reconnaissance
        3 = Privilege Escalation (User to Root - U2R)
        4 = Unauthorized Access (Remote to Local - R2L)
        """
        attack_str = str(attack_label).strip().lower()

        if attack_str == "normal":
            return CLASS_MAPPING["normal"]

        for category, attack_list in ATTACK_CATEGORIES.items():
            if attack_str in attack_list:
                return CLASS_MAPPING[category]

        # Unknown or unclassified attacks mapped to 0 (or fallback)
        logger.debug("Unmapped attack signature encountered: '%s', mapping to Normal (0)", attack_label)
        return 0

    @staticmethod
    def map_attack_to_binary(attack_label: str) -> int:
        """Map raw attack label string to binary target (0: Normal, 1: Attack)."""
        return 0 if str(attack_label).strip().lower() == "normal" else 1

    def fit_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the categorical feature space on the dataset and produce aligned feature matrix.

        Args:
            df: Raw DataFrame containing 'protocol_type', 'service', and NUMERIC_FEATURES.

        Returns:
            pd.DataFrame of one-hot encoded and aligned features.
        """
        logger.info("Fitting and transforming features across %d records", len(df))

        # Perform one-hot encoding on categorical features
        encoded_cats = pd.get_dummies(df[FEATURES_TO_ENCODE], dtype=float)

        # Extract numeric features and ensure numeric types
        numeric_df = df[NUMERIC_FEATURES].astype(float)

        # Merge encoded categorical and numeric features
        combined_features = encoded_cats.join(numeric_df)

        self.feature_columns = list(combined_features.columns)
        self.fitted = True

        logger.info(
            "Feature space constructed with %d total columns (%d categorical, %d numeric)",
            len(self.feature_columns),
            len(encoded_cats.columns),
            len(NUMERIC_FEATURES),
        )
        return combined_features

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform incoming data using previously established feature column alignment.

        Ensures that real-time or inference samples match the exact shape and column
        order expected by the trained model.

        Args:
            df: Raw or partial DataFrame with telemetry attributes.

        Returns:
            pd.DataFrame aligned strictly to self.feature_columns.
        """
        if not self.fitted or self.feature_columns is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform_features()")

        # One-hot encode incoming categorical variables
        encoded_cats = pd.get_dummies(df[FEATURES_TO_ENCODE], dtype=float)

        # Extract numeric features
        numeric_df = df[[col for col in NUMERIC_FEATURES if col in df.columns]].astype(float)

        combined = encoded_cats.join(numeric_df)

        # Reindex to exact training feature set, filling missing columns with 0.0
        aligned = combined.reindex(columns=self.feature_columns, fill_value=0.0)
        return aligned

    def prepare_dataset(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Extract aligned features, multi-class target, and binary target from raw data.

        Returns:
            Tuple of (X_features, y_multiclass, y_binary)
        """
        # Multi-class target
        multi_y = df["attack"].apply(self.map_attack_to_multiclass)

        # Binary target
        binary_y = df["attack"].apply(self.map_attack_to_binary)

        # Feature matrix
        X = self.fit_transform_features(df)

        return X, multi_y, binary_y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = TEST_SIZE,
        val_size: float = VAL_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """Split dataset into Train, Validation, and Test partitions.

        Replicates the exact Hack The Box methodology:
        1. Split into (Train + Val) and Test (80/20).
        2. Split (Train + Val) into Train and Val (70/30).

        Returns:
            Dict containing:
                'train_X', 'train_y',
                'val_X', 'val_y',
                'test_X', 'test_y'
        """
        logger.info(
            "Splitting dataset: test_size=%.2f, val_size=%.2f, random_state=%d",
            test_size,
            val_size,
            random_state,
        )

        # Primary partition: Train+Val vs Test
        train_val_X, test_X, train_val_y, test_y = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Secondary partition: Train vs Validation
        train_X, val_X, train_y, val_y = train_test_split(
            train_val_X, train_val_y, test_size=val_size, random_state=random_state
        )

        logger.info(
            "Partitions established -> Train: %d samples, Val: %d samples, Test: %d samples",
            len(train_X),
            len(val_X),
            len(test_X),
        )

        return {
            "train_X": train_X,
            "train_y": train_y,
            "val_X": val_X,
            "val_y": val_y,
            "test_X": test_X,
            "test_y": test_y,
        }

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Serialize preprocessor configuration and aligned feature schema."""
        target_path = Path(filepath) if filepath else DEFAULT_PREPROCESSOR_FILE
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "fitted": self.fitted,
            },
            target_path,
        )
        logger.info("Preprocessor metadata successfully saved to: %s", target_path)
        return target_path

    def load(self, filepath: Optional[Union[str, Path]] = None) -> "KDDPreprocessor":
        """Load serialized preprocessor configuration."""
        target_path = Path(filepath) if filepath else DEFAULT_PREPROCESSOR_FILE
        if not target_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at: {target_path}")

        metadata = joblib.load(target_path)
        self.feature_columns = metadata.get("feature_columns")
        self.fitted = metadata.get("fitted", True)
        logger.info(
            "Preprocessor loaded from %s with %d feature columns",
            target_path,
            len(self.feature_columns) if self.feature_columns else 0,
        )
        return self
