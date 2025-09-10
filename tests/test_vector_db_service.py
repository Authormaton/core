
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
class MockIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.upserted = []
    def upsert(self, items, namespace=None):
        self.upserted.extend(items)
    def query(self, vector, top_k=8, namespace=None, filter=None):
        return {'matches': [{'id': 'test', 'score': 0.99}]}

mock_pc = MagicMock()
mock_pc.list_indexes.return_value = []
mock_pc.create_index.return_value = None
mock_pc.Index.side_effect = lambda name: MockIndex(16)
mock_pc.describe_index.return_value = {"dimension": 16}

mock_pinecone = MagicMock()
mock_pinecone.Pinecone = MagicMock(return_value=mock_pc)
mock_pinecone.ServerlessSpec = MagicMock()
sys.modules["pinecone"] = mock_pinecone

# Now import everything else
import pytest
from services.vector_db_service import VectorDBService

class MockIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.upserted = []
    def upsert(self, items, namespace=None):
        self.upserted.extend(items)
    def query(self, vector, top_k=8, namespace=None, filter=None):
        return {'matches': [{'id': 'test', 'score': 0.99}]}

def test_ensure_index_idempotent(monkeypatch):
    svc = VectorDBService()
    # Simulate: first call creates the index; second call sees it and skips creation.
    mock_pc.create_index.reset_mock()
    import types
    mock_pc.list_indexes.side_effect = [
        [],
        [types.SimpleNamespace(name=dummy_config.settings.pinecone_index_name)],
    ]

    svc.ensure_index(svc.embedding_dimension)
    svc.ensure_index(svc.embedding_dimension)
    # Should only ever create the index once
    assert mock_pc.create_index.call_count == 1
    assert svc.index.dimension == svc.embedding_dimension

def test_upsert_dimension_guard(monkeypatch):
    svc = VectorDBService()
    monkeypatch.setattr(svc, 'index', MockIndex(svc.embedding_dimension))
    ids = ['a', 'b']
    vectors = [[0.0]*svc.embedding_dimension, [0.0]*svc.embedding_dimension]
    metadata = [{}, {}]
    count = svc.upsert(namespace='proj', ids=ids, vectors=vectors, metadata=metadata)
    assert count == 2
    # Wrong dimension
    with pytest.raises(ValueError):
        svc.upsert(namespace='proj', ids=ids, vectors=[[0.0]*10, [0.0]*10], metadata=metadata)

def test_query_dimension_guard(monkeypatch):
    svc = VectorDBService()
    monkeypatch.setattr(svc, 'index', MockIndex(svc.embedding_dimension))
    vector = [0.0]*svc.embedding_dimension
    matches = svc.query(namespace='proj', vector=vector)
    assert matches[0]['id'] == 'test'
    # Wrong dimension
    with pytest.raises(ValueError):
        svc.query(namespace='proj', vector=[0.0]*10)
