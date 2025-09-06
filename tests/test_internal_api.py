"""
Unit tests for the internal API endpoint: /internal/process-material
"""

import os
from dotenv import load_dotenv
load_dotenv()
import pytest
from fastapi.testclient import TestClient
from api.main import app

# Load the internal API key from environment variable to match the API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = TestClient(app)


def test_process_material_success():
    payload = {
        "source_material": "Sample document text.",
        "prompt": "Summarize the content.",
        "metadata": {"type": "text"}
    }
    response = client.post(
        "/internal/process-material",
        json=payload,
    headers={"X-Internal-API-Key": OPENAI_API_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["received_material_length"] == len(payload["source_material"])
    assert data["prompt"] == payload["prompt"]
    assert data["metadata"] == payload["metadata"]


def test_process_material_unauthorized():
    payload = {
        "source_material": "Sample document text.",
        "prompt": "Summarize the content."
    }
    response = client.post(
        "/internal/process-material",
        json=payload,
        headers={"X-Internal-API-Key": "wrong-key"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing internal API key."


def test_process_material_missing_key():
    payload = {
        "source_material": "Sample document text.",
        "prompt": "Summarize the content."
    }
    response = client.post(
        "/internal/process-material",
        json=payload
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing internal API key."
