# Test for the /health endpoint
import sys
import os
import pytest
from fastapi.testclient import TestClient
import json
import logging

# Ensure api is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.main import app

client = TestClient(app)

def test_health():
	response = client.get("/health")
	assert response.status_code == 200
	assert response.json() == {"status": "ok"}

def test_health_log_entry(caplog):
    with caplog.at_level(logging.INFO):
        response = client.get("/health")
        assert response.status_code == 200
        assert len(caplog.records) >= 1
        log_record = caplog.records[0]
        parsed_log = json.loads(log_record.message)

        assert "time" in parsed_log
        assert parsed_log["level"] == "INFO"
        assert parsed_log["service"] == "authormaton-core"
        assert parsed_log["message"] == "Health check requested"
