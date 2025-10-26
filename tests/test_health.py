import json
import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure api is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.main import app

client = TestClient(app)

def test_health():
	response = client.get("/health")
	assert response.status_code == 200
	assert response.json() == {"status": "ok"}

def test_health_log_structure(caplog):
    import logging
    caplog.set_level(logging.INFO)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    found_log = False
    for record in caplog.records:
        if record.levelname == "INFO":
            try:
                log_entry = json.loads(record.message)
                if log_entry.get("message") == "Health check successful":
                    assert "timestamp" in log_entry
                    assert log_entry["level"] == "INFO"
                    assert log_entry["service"] == "api.main"
                    assert log_entry["endpoint"] == "/health"
                    assert log_entry["status"] == "ok"
                    found_log = True
                    break
            except json.JSONDecodeError:
                continue
    assert found_log, "Did not find a health check log with the expected structured format."