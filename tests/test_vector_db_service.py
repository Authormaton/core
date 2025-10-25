

import pytest
from unittest.mock import MagicMock
from services.vector_db_service import VectorDBClient
from pinecone.exceptions import PineconeException, PineconeProtocolError

class DummyIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.upserted = []
        self.upsert_calls = 0
        self.query_calls = 0
        self.error_on_upsert_calls = []
        self.error_on_query_calls = []

    def upsert(self, vectors=None, namespace=None, timeout=None):
        self.upsert_calls += 1
        if self.upsert_calls in self.error_on_upsert_calls:
            raise PineconeException("Simulated upsert timeout")
        if vectors:
            self.upserted.extend(vectors)

    def query(self, vector, top_k=5, timeout=None):
        self.query_calls += 1
        if self.query_calls in self.error_on_query_calls:
            raise PineconeException("Simulated query timeout")
        return {'matches': [{'id': 'id1', 'score': 0.99}]}

class MockPinecone:
    def __init__(self, api_key):
        self.api_key = api_key
        self.describe_index_calls = 0
        self.create_index_calls = 0
        self.error_on_describe_index_calls = []
        self.error_on_create_index_calls = []
        self._index_instance = None

    def Index(self, name):
        if not self._index_instance:
            self._index_instance = DummyIndex(dimension=8) # Assuming dimension 8 for tests
        return self._index_instance

    def describe_index(self, index_name):
        self.describe_index_call_count += 1
        if self.fail_describe_index_count > 0:
            self.fail_describe_index_count -= 1
            raise PineconeException("Simulated describe_index failure")
        # Simulate index found after retries
        return MagicMock(dimension=8) # Simulate index found

    def create_index(self, name, dimension, spec):
        self.create_index_call_count += 1
        if self.fail_create_index_count > 0:
            self.fail_create_index_count -= 1
            raise PineconeException("Simulated create_index failure")
        pass

@pytest.fixture
def vdb():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    svc = VectorDBClient(dimension=8, index_name="test-index", pinecone_client=mock_pinecone_client, pinecone_index=mock_pinecone_client.Index("test-index"))
    return svc

def test_create_index():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    mock_pinecone_client.describe_index = lambda name, timeout=None: None # Simulate index not found
    mock_pinecone_client.create_index = lambda name, dimension, spec, timeout=None: None # Mock create_index
    mock_pinecone_client.Index = lambda name: DummyIndex(dimension=8) # Mock Index

    svc = VectorDBClient(dimension=8, index_name="test-index", pinecone_client=mock_pinecone_client)
    svc.create_index()
    assert svc.index.dimension == 8

def test_upsert_vectors(vdb):
    ids = ["id1", "id2"]
    vectors = [[0.0]*8, [1.0]*8]
    vdb.upsert_vectors(vectors, ids)
    assert len(vdb.index.upserted) == 2
    # Wrong dimension
    with pytest.raises(ValueError):
        vdb.upsert_vectors([[0.0]*5, [1.0]*5], ids)

def test_query(vdb):
    vector = [0.0]*8
    result = vdb.query(vector)
    assert result['matches'][0]['id'] == 'id1'
    # Wrong dimension
    with pytest.raises(ValueError):
        vdb.query([0.0]*5)

def test_upsert_vectors_with_retries():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    mock_pinecone_client.Index("test-index").error_on_upsert_calls = [1, 2] # Fail on first two calls
    svc = VectorDBClient(dimension=8, index_name="test-index", pinecone_client=mock_pinecone_client, pinecone_index=mock_pinecone_client.Index("test-index"))
    
    ids = ["id1", "id2"]
    vectors = [[0.0]*8, [1.0]*8]
    svc.upsert_vectors(vectors, ids)
    assert mock_pinecone_client.Index("test-index").upsert_calls == 3 # Should retry twice and succeed on third

def test_query_with_retries():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    mock_pinecone_client.Index("test-index").error_on_query_calls = [1] # Fail on first call
    svc = VectorDBClient(dimension=8, index_name="test-index", pinecone_client=mock_pinecone_client, pinecone_index=mock_pinecone_client.Index("test-index"))
    
    vector = [0.0]*8
    result = svc.query(vector)
    assert mock_pinecone_client.Index("test-index").query_calls == 2 # Should retry once and succeed on second
    assert result['matches'][0]['id'] == 'id1'

def test_create_index_with_retries():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    mock_pinecone_client.error_on_describe_index_calls = [1] # Fail on first describe_index call
    svc = VectorDBClient(dimension=8, index_name="test-index", pinecone_client=mock_pinecone_client)
    
    svc.create_index()
    assert mock_pinecone_client.describe_index_calls == 2 # Should retry once and succeed on second
    assert svc.index.dimension == 8

def test_upsert_vectors_timeout_failure():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    # Fail on all expected calls (max_retries attempts)
    mock_pinecone_client.Index("test-index").error_on_upsert_calls = [1, 2, 3] 
    svc = VectorDBClient(dimension=8, index_name="test-index", pinecone_client=mock_pinecone_client, pinecone_index=mock_pinecone_client.Index("test-index"))
    
    ids = ["id1", "id2"]
    vectors = [[0.0]*8, [1.0]*8]
    with pytest.raises(PineconeException):
        svc.upsert_vectors(vectors, ids)
    assert mock_pinecone_client.Index("test-index").upsert_calls == 3 # Should attempt max_retries times

def test_query_timeout_failure():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    # Fail on all expected calls (max_retries attempts)
    mock_pinecone_client.Index("test-index").error_on_query_calls = [1, 2, 3]
    svc = VectorDBClient(dimension=8, index_name="test-index", pinecone_client=mock_pinecone_client, pinecone_index=mock_pinecone_client.Index("test-index"))
    
    vector = [0.0]*8
    with pytest.raises(PineconeException):
        svc.query(vector)
    assert mock_pinecone_client.Index("test-index").query_calls == 3 # Should attempt max_retries times

def test_create_index_timeout_failure():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    # Fail on all expected calls (max_retries attempts)
    mock_pinecone_client.error_on_describe_index_calls = [1, 2, 3]
    svc = VectorDBClient(dimension=8, index_name="test-index", pinecone_client=mock_pinecone_client)
    
    with pytest.raises(PineconeException):
        svc.create_index()
    assert mock_pinecone_client.describe_index_calls == 3 # Should attempt max_retries times

