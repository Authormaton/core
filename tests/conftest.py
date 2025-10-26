import pytest
import os
from unittest.mock import MagicMock, patch
import sys

@pytest.fixture(autouse=True, scope='session')
def set_testing_env_var():
    os.environ["TESTING"] = "True"
    yield
    del os.environ["TESTING"]

@pytest.fixture(autouse=True, scope='session')
def mock_settings_module_for_imports():
    # Create a mock for the settings module
    mock_settings_module = MagicMock()

    # Set attributes on the mocked settings object
    mock_settings_module.settings.pinecone_api_key = "test-api-key"
    mock_settings_module.settings.pinecone_cloud = "aws"
    mock_settings_module.settings.pinecone_region = "us-west-2"
    mock_settings_module.settings.pinecone_index_name = "test-index"
    mock_settings_module.settings.embedding_dimension = 3072
    mock_settings_module.settings.vector_db_max_retries = 3
    mock_settings_module.settings.vector_db_initial_backoff = 0.01
    mock_settings_module.settings.vector_db_timeout = 1

    # Patch sys.modules to replace the actual config.settings with our mock
    with patch.dict(sys.modules, {'config.settings': mock_settings_module}):
        yield