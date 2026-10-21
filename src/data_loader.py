"""Data ingestion and validation module for the NSL-KDD dataset."""

import io
import logging
from pathlib import Path
from typing import Optional, Union
import zipfile

import pandas as pd
import requests

from config.settings import COLUMNS, DATA_DIR, DATASET_URL

logger = logging.getLogger(__name__)


class KDDDataLoader:
    """Manages downloading, caching, extraction, and loading of the NSL-KDD dataset."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_and_extract(
        self,
        url: str = DATASET_URL,
        target_filename: str = "KDD+.txt",
        force_download: bool = False,
    ) -> Path:
        """Download the dataset zip file from remote repository and extract locally.

        Args:
            url: Remote URL for the zipped NSL-KDD dataset.
            target_filename: Name of the expected uncompressed data file.
            force_download: Re-download even if target file exists.

        Returns:
            Path to the verified extracted data file.
        """
        target_path = self.data_dir / target_filename
        if target_path.exists() and not force_download:
            logger.info("Dataset already exists at: %s", target_path)
            return target_path

        logger.info("Initiating dataset download from: %s", url)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(self.data_dir)
            logger.info("Dataset successfully extracted to: %s", self.data_dir)
        except requests.RequestException as e:
            logger.error("Failed to download dataset from %s: %s", url, e)
            raise RuntimeError(f"Network error downloading dataset: {e}") from e
        except zipfile.BadZipFile as e:
            logger.error("Downloaded file is not a valid zip archive: %s", e)
            raise ValueError(f"Corrupted zip archive: {e}") from e

        if not target_path.exists():
            # Check if extracted into current directory or subdirectory
            candidates = list(self.data_dir.rglob(target_filename))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(
                f"Target file '{target_filename}' was not found in extracted files at {self.data_dir}"
            )

        return target_path

    def load_data(
        self,
        file_path: Optional[Union[str, Path]] = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Load the raw dataset into a pandas DataFrame and validate schema.

        Args:
            file_path: Optional direct path to KDD+.txt. If None, resolves from data_dir.
            validate: Whether to perform structural and column count checks.

        Returns:
            pd.DataFrame containing the raw network telemetry logs.
        """
        if file_path is None:
            resolved_path = self.data_dir / "KDD+.txt"
        else:
            resolved_path = Path(file_path)

        if not resolved_path.exists():
            # Check local home directory fallback
            root_fallback = Path("/home/luis/KDD+.txt")
            if root_fallback.exists():
                logger.info("Found existing dataset at %s, linking/using fallback.", root_fallback)
                resolved_path = root_fallback
            else:
                logger.info("Dataset file not found at %s. Triggering automatic download.", resolved_path)
                resolved_path = self.download_and_extract()

        if not resolved_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist at {resolved_path}")

        logger.info("Loading NSL-KDD dataset from %s", resolved_path)
        df = pd.read_csv(resolved_path, names=COLUMNS, header=None)

        if validate:
            self.validate_schema(df)

        logger.info("Dataset loaded successfully. Shape: %s", df.shape)
        return df

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> None:
        """Validate that the loaded DataFrame matches expected column structure and integrity.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If column counts or expected critical columns are missing.
        """
        if df.empty:
            raise ValueError("Loaded dataset is empty.")

        if len(df.columns) != len(COLUMNS):
            raise ValueError(
                f"Schema mismatch: expected {len(COLUMNS)} columns, got {len(df.columns)}"
            )

        critical_columns = ["protocol_type", "service", "attack", "duration", "src_bytes"]
        for col in critical_columns:
            if col not in df.columns:
                raise ValueError(f"Missing mandatory telemetry column: '{col}'")
