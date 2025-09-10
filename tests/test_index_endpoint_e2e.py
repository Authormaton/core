
# Patch config.settings and env vars before any other imports
import sys
import types
import os
from pydantic import SecretStr

# Set required env vars and patch config.settings before any other imports
os.environ["INTERNAL_API_KEY"] = "test-key"

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

# Now import everything else
from dotenv import load_dotenv
load_dotenv()
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch

client = TestClient(app)

def mock_embed_texts_batched(texts, batch_size=32, retries=3):
    embedding_dim = 16
    return [[0.0] * embedding_dim for _ in texts]

PDF_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")

@patch("api.indexing_router.embed_texts_batched", side_effect=mock_embed_texts_batched)
def test_index_pdf(mock_embed):
    sources = [{"text": "Hello world", "source_id": "pdf1", "file_path": "dummy.pdf"}]
    payload = {
        "project_id": "proj1",
        "sources": sources
    }
    headers = {"X-Internal-API-Key": "test-key"}
    response = client.post("/internal/index", json=payload, headers=headers)
    print(response.text)  # Show error detail for debugging
    assert response.status_code == 201
    data = response.json()
    assert data["project_id"] == "proj1"
    assert data["indexed_chunks"] > 0
    assert data["sources_indexed"] == 1