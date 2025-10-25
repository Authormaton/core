import pytest
from unittest.mock import MagicMock
import sys

def pytest_configure(config):
    # Create a mock Settings object
    mock_settings_instance = MagicMock()
    mock_settings_instance.pinecone_api_key.get_secret_value.return_value = "test-api-key"
    mock_settings_instance.pinecone_cloud = "aws"
    mock_settings_instance.pinecone_region = "us-east-1"
    mock_settings_instance.pinecone_index_name = "test-index"
    mock_settings_instance.embedding_dimension = 8
    mock_settings_instance.embed_batch_size = 128
    mock_settings_instance.web_search_engine = "dummy"
    mock_settings_instance.pinecone_max_retries = 3
    mock_settings_instance.pinecone_min_retry_delay = 0.01
    mock_settings_instance.pinecone_max_retry_delay = 0.05
    mock_settings_instance.pinecone_timeout = 1

    # Create a mock module for config.settings
    mock_settings_module = MagicMock()
    mock_settings_module.settings = mock_settings_instance

    # Replace the actual config.settings module in sys.modules
    sys.modules["config.settings"] = mock_settings_module

    # Patch sys.exit to prevent it from terminating the test run
    monkeypatch = MagicMock() # Create a mock monkeypatch for pytest_configure
    monkeypatch.setattr(sys, "exit", lambda x: None)

@pytest.fixture(scope="session", autouse=True)
def mock_settings_module_fixture():
    # This fixture is now mostly for cleanup, as setup is in pytest_configure
    original_settings_module = sys.modules.get("config.settings")
    yield
    if original_settings_module is not None:
        sys.modules["config.settings"] = original_settings_module
    elif "config.settings" in sys.modules:
        del sys.modules["config.settings"]