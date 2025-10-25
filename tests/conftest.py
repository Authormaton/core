import pytest
from unittest.mock import MagicMock
import sys

@pytest.fixture(scope="session", autouse=True)
def mock_settings_module():
    # Store original module if it exists
    original_settings_module = sys.modules.get("config.settings")

    # Create a mock Settings object
    mock_settings_instance = MagicMock()
    mock_settings_instance.pinecone_api_key.get_secret_value.return_value = "test-api-key"
    mock_settings_instance.pinecone_cloud = "aws"
    mock_settings_instance.pinecone_region = "us-east-1"
    mock_settings_instance.pinecone_index_name = "test-index"
    mock_settings_instance.embedding_dimension = 3072
    mock_settings_instance.web_search_engine = "dummy"

    # Create a mock module for config.settings
    mock_settings_module = MagicMock()
    mock_settings_module.settings = mock_settings_instance

    # Replace the actual config.settings module in sys.modules
    sys.modules["config.settings"] = mock_settings_module

    yield

    # Clean up: restore original module or remove if it didn't exist
    if original_settings_module is not None:
        sys.modules["config.settings"] = original_settings_module
    elif "config.settings" in sys.modules:
        del sys.modules["config.settings"]