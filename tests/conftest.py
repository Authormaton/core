import pytest
import sys
import os
import types
from pydantic import SecretStr

# Set required env vars and patch config.settings before any other imports
os.environ["INTERNAL_API_KEY"] = "test-key"
os.environ["PINECONE_API_KEY"] = "test-key"
os.environ["PINECONE_CLOUD"] = "aws"
os.environ["PINECONE_REGION"] = "us-east-1"

class DummySettings:
    pinecone_api_key = SecretStr("test-key")
    pinecone_cloud = "aws"
    pinecone_region = "us-east-1"
    pinecone_index_name = "authormaton-core"
    embedding_model = "test-model"
    embedding_dimension = 16
    embed_batch_size = 64
    max_upload_mb = 25

dummy_config = types.ModuleType("config.settings")
dummy_config.settings = DummySettings()
sys.modules["config.settings"] = dummy_config

# Patch Pinecone client for tests to avoid real API calls
from unittest.mock import MagicMock
mock_pc = MagicMock()
mock_pc.list_indexes.return_value = []
mock_pc.create_index.return_value = None
mock_pc.Index.return_value = MagicMock()
mock_pc.describe_index.return_value = {"dimension": 16}

mock_pinecone = MagicMock()
mock_pinecone.Pinecone = MagicMock(return_value=mock_pc)
mock_pinecone.ServerlessSpec = MagicMock()
sys.modules["pinecone"] = mock_pinecone

@pytest.fixture(autouse=True)
def mock_settings():
    # This fixture ensures that the dummy settings are always applied
    # before any test runs, and cleaned up afterwards.
    original_settings = sys.modules.get("config.settings")
    original_pinecone = sys.modules.get("pinecone")

    sys.modules["config.settings"] = dummy_config
    sys.modules["pinecone"] = mock_pinecone

    yield

    if original_settings:
        sys.modules["config.settings"] = original_settings
    else:
        del sys.modules["config.settings"]

    if original_pinecone:
        sys.modules["pinecone"] = original_pinecone
    else:
        del sys.modules["pinecone"]
