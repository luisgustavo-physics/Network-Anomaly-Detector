"""Tests for model artifact upload client and resilience handling."""

from pathlib import Path
import pytest

from src.uploader import ModelUploader


def test_upload_missing_file():
    """Verify that attempting to upload a non-existent file raises FileNotFoundError."""
    uploader = ModelUploader(api_url="http://localhost:8001/api/upload")
    with pytest.raises(FileNotFoundError):
        uploader.upload("non_existent_model.joblib")


def test_upload_unreachable_endpoint(tmp_path):
    """Verify that uploading to an unreachable endpoint fails gracefully with clear status."""
    dummy_model = tmp_path / "model.joblib"
    dummy_model.write_text("mock model content")

    # Use a non-routable port or local closed port
    uploader = ModelUploader(api_url="http://127.0.0.1:59999/api/upload")
    result = uploader.upload(dummy_model, timeout=1, max_retries=1)

    assert result["success"] is False
    assert result["status_code"] is None
    assert "error" in result
