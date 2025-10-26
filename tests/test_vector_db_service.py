

import pytest
from unittest.mock import MagicMock, patch, ANY
from services.vector_db_service import VectorDBClient
from pinecone import PineconeException


class DummyIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.upserted = []
    def upsert(self, vectors=None, namespace=None):
        if vectors:
            self.upserted.extend(vectors)
    def query(self, vector, top_k=5):
        return {'matches': [{'id': 'id1', 'score': 0.99}]}

class MockPinecone:
    def __init__(self, api_key):
        self.api_key = api_key

    def Index(self, name):
        return DummyIndex(dimension=8) # Assuming dimension 8 for tests

    def describe_index(self, index_name):
        # Simulate index not found
        return None

    def create_index(self, name, dimension, spec):
        pass

@pytest.fixture
def vdb():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    svc = VectorDBClient(
        dimension=3072,
        index_name="test-index",
        pinecone_client=mock_pinecone_client,
        pinecone_index=mock_pinecone_client.Index("test-index"),
        pinecone_api_key="test-api-key",
        pinecone_cloud="aws",
        pinecone_region="us-west-2",
    )
    return svc

def test_create_index():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    mock_pinecone_client.describe_index = lambda name: None # Simulate index not found
    mock_pinecone_client.create_index = lambda name, dimension, spec: None # Mock create_index
    mock_pinecone_client.Index = lambda name: DummyIndex(dimension=3072) # Mock Index

    svc = VectorDBClient(
        dimension=3072,
        index_name="test-index",
        pinecone_client=mock_pinecone_client,
        pinecone_api_key="test-api-key",
        pinecone_cloud="aws",
        pinecone_region="us-west-2",
    )
    svc.create_index()
    assert svc.index.dimension == 3072

def test_upsert_vectors(vdb):
    ids = ["id1", "id2"]
    vectors = [[0.0]*3072, [1.0]*3072]
    vdb.upsert_vectors(vectors, ids)
    assert len(vdb.index.upserted) == 2
    # Wrong dimension
    with pytest.raises(ValueError):
        vdb.upsert_vectors([[0.0]*5, [1.0]*5], ids)

def test_query(vdb):
    vector = [0.0]*3072
    result = vdb.query(vector)
    assert result['matches'][0]['id'] == 'id1'
    # Wrong dimension
    with pytest.raises(ValueError):
        vdb.query([0.0]*5)

