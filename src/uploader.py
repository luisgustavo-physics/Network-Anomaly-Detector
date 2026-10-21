"""Model artifact deployment and API upload client.

This module handles submitting trained model artifacts to Hack The Box evaluation
endpoints or remote inference microservices, incorporating retry resilience and
security considerations regarding model deserialization.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union

import requests

from config.settings import DEFAULT_API_URL

logger = logging.getLogger(__name__)


class ModelUploader:
    """Uploads serialized model artifacts to remote evaluation or scoring endpoints."""

    def __init__(self, api_url: Optional[str] = None) -> None:
        self.api_url = api_url or DEFAULT_API_URL

    def upload(
        self,
        model_path: Union[str, Path],
        timeout: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> Dict[str, Any]:
        """Upload a serialized model artifact (.joblib) via HTTP POST multipart form.

        Args:
            model_path: Local filesystem path to the model artifact.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum number of connection retry attempts.
            backoff_factor: Exponential backoff factor between retries.

        Returns:
            Dict containing execution status, HTTP status code, and endpoint response.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file to upload not found: {path}")

        logger.info("Preparing to upload '%s' to endpoint: %s", path.name, self.api_url)

        # AI Red Teaming Note:
        # Deserializing joblib/pickle models on remote endpoints presents an Arbitrary
        # Code Execution (ACE) attack surface if endpoints accept untrusted artifacts.
        logger.debug("Verifying model artifact size: %.2f MB", path.stat().st_size / (1024 * 1024))

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                with open(path, "rb") as model_file:
                    files = {"model": (path.name, model_file, "application/octet-stream")}
                    response = requests.post(self.api_url, files=files, timeout=timeout)

                logger.info(
                    "Upload request received HTTP status %d from %s",
                    response.status_code,
                    self.api_url,
                )

                try:
                    response_json = response.json()
                except json.JSONDecodeError:
                    response_json = {"raw_text": response.text}

                return {
                    "success": response.ok,
                    "status_code": response.status_code,
                    "response": response_json,
                    "attempt": attempt,
                }

            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                logger.warning(
                    "Upload attempt %d/%d failed: %s. Retrying in %.2fs...",
                    attempt,
                    max_retries,
                    e,
                    backoff_factor ** attempt,
                )
                if attempt < max_retries:
                    time.sleep(backoff_factor ** attempt)

            except Exception as e:
                logger.error("Unexpected error during model upload: %s", e)
                return {
                    "success": False,
                    "status_code": None,
                    "error": str(e),
                    "attempt": attempt,
                }

        logger.error(
            "All %d upload attempts failed. Endpoint %s is unreachable or inactive.",
            max_retries,
            self.api_url,
        )
        return {
            "success": False,
            "status_code": None,
            "error": f"Failed after {max_retries} attempts: {last_error}",
        }
