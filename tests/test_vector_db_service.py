

import pytest
from unittest.mock import MagicMock, patch, ANY
from services.vector_db_service import VectorDBClient
from pinecone import PineconeException
from config.settings import settings # This will now be the mocked settings from conftest.py

class DummyIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.upserted = []
    def upsert(self, vectors=None, namespace=None, timeout=None):
        if vectors:
            self.upserted.extend(vectors)
    def query(self, vector, top_k=5, timeout=None):
        return {'matches': [{'id': 'id1', 'score': 0.99}]}

class MockPinecone:
    def __init__(self, api_key):
        self.api_key = api_key

    def Index(self, name):
        return DummyIndex(dimension=settings.embedding_dimension) # Use mocked settings

    def describe_index(self, index_name):
        # Simulate index not found
        return None

    def create_index(self, name, dimension, spec, timeout):
        pass

@pytest.fixture
def vdb():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    svc = VectorDBClient(
        dimension=settings.embedding_dimension,
        index_name=settings.pinecone_index_name,
        pinecone_client=mock_pinecone_client,
        pinecone_index=mock_pinecone_client.Index(settings.pinecone_index_name),
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )
    return svc

def test_create_index():
    mock_pinecone_client = MockPinecone(api_key="test-api-key")
    mock_pinecone_client.describe_index = lambda name: None # Simulate index not found
    mock_pinecone_client.create_index = lambda name, dimension, spec, timeout: None # Mock create_index
    mock_pinecone_client.Index = lambda name: DummyIndex(dimension=settings.embedding_dimension) # Mock Index

    svc = VectorDBClient(
        dimension=settings.embedding_dimension,
        index_name=settings.pinecone_index_name,
        pinecone_client=mock_pinecone_client,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )
    svc.create_index()
    assert svc.index.dimension == settings.embedding_dimension

@patch('pinecone.Pinecone')
def test_create_index_with_retries(mock_pinecone_class):
    # Override settings for this specific test if needed, otherwise use global mock
    settings.vector_db_max_retries = 3
    settings.vector_db_initial_backoff = 0.01
    settings.vector_db_timeout = 1

    mock_pinecone_instance = MagicMock()
    mock_pinecone_class.return_value = mock_pinecone_instance

    # Simulate describe_index failing twice, then succeeding
    mock_pinecone_instance.describe_index.side_effect = [
        PineconeException("Transient error"),
        PineconeException("Another transient error"),
        None  # Succeed on the third attempt
    ]
    mock_pinecone_instance.Index.return_value = DummyIndex(dimension=settings.embedding_dimension)

    svc = VectorDBClient(
        dimension=settings.embedding_dimension,
        index_name=settings.pinecone_index_name,
        pinecone_client=mock_pinecone_instance,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )
    svc.create_index()

    # Check that describe_index was called 3 times (2 failures + 1 success)
    assert mock_pinecone_instance.describe_index.call_count == 3
    mock_pinecone_instance.create_index.assert_called_once_with(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        spec=ANY,
        timeout=settings.vector_db_timeout
    )

@patch('pinecone.Pinecone')
def test_create_index_retries_exhausted(mock_pinecone_class):
    settings.vector_db_max_retries = 2 # Override for this specific test

    mock_pinecone_instance = MagicMock()
    mock_pinecone_class.return_value = mock_pinecone_instance

    # Simulate describe_index always failing
    mock_pinecone_instance.describe_index.side_effect = PineconeException("Persistent error")

    svc = VectorDBClient(
        dimension=settings.embedding_dimension,
        index_name=settings.pinecone_index_name,
        pinecone_client=mock_pinecone_instance,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )
    with pytest.raises(PineconeException):
        svc.create_index()

    # Check that describe_index was called max_retries + 1 times
    assert mock_pinecone_instance.describe_index.call_count == settings.vector_db_max_retries + 1


def test_upsert_vectors(vdb):
    ids = ["id1", "id2"]
    vectors = [[0.0]*3072, [1.0]*3072]
    vdb.upsert_vectors(vectors, ids)
    assert len(vdb.index.upserted) == 2
    # Wrong dimension
    with pytest.raises(ValueError):
        vdb.upsert_vectors([[0.0]*5, [1.0]*5], ids)

def test_upsert_vectors_with_retries():
    settings.vector_db_max_retries = 3
    settings.vector_db_initial_backoff = 0.01
    settings.vector_db_timeout = 1

    mock_index = MagicMock()
    mock_index.upsert.side_effect = [
        PineconeException("Upsert error 1"),
        PineconeException("Upsert error 2"),
        None  # Succeed on the third attempt
    ]

    svc = VectorDBClient(
        dimension=settings.embedding_dimension,
        index_name=settings.pinecone_index_name,
        pinecone_client=MagicMock(), # Mock the Pinecone client itself
        pinecone_index=mock_index,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )

    ids = ["id1", "id2"]
    vectors = [[0.0]*settings.embedding_dimension, [1.0]*settings.embedding_dimension]
    svc.upsert_vectors(vectors, ids)

    assert mock_index.upsert.call_count == 3
    mock_index.upsert.assert_called_with(vectors=ANY, timeout=settings.vector_db_timeout)

def test_upsert_vectors_retries_exhausted():
    settings.vector_db_max_retries = 2 # Override for this specific test

    mock_index = MagicMock()
    mock_index.upsert.side_effect = PineconeException("Persistent upsert error")

    svc = VectorDBClient(
        dimension=settings.embedding_dimension,
        index_name=settings.pinecone_index_name,
        pinecone_client=MagicMock(),
        pinecone_index=mock_index,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )

    ids = ["id1", "id2"]
    vectors = [[0.0]*settings.embedding_dimension, [1.0]*settings.embedding_dimension]
    with pytest.raises(PineconeException):
        svc.upsert_vectors(vectors, ids)

    assert mock_index.upsert.call_count == settings.vector_db_max_retries + 1


def test_query(vdb):
    vector = [0.0]*3072
    result = vdb.query(vector)
    assert result['matches'][0]['id'] == 'id1'
    # Wrong dimension
    with pytest.raises(ValueError):
        vdb.query([0.0]*5)

def test_query_with_retries():
    settings.vector_db_max_retries = 3
    settings.vector_db_initial_backoff = 0.01
    settings.vector_db_timeout = 1

    mock_index = MagicMock()
    mock_index.query.side_effect = [
        PineconeException("Query error 1"),
        PineconeException("Query error 2"),
        {'matches': [{'id': 'id1', 'score': 0.99}]}  # Succeed on the third attempt
    ]

    svc = VectorDBClient(
        dimension=settings.embedding_dimension,
        index_name=settings.pinecone_index_name,
        pinecone_client=MagicMock(),
        pinecone_index=mock_index,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )

    vector = [0.0]*settings.embedding_dimension
    result = svc.query(vector)

    assert mock_index.query.call_count == 3
    mock_index.query.assert_called_with(vector=vector, top_k=5, timeout=settings.vector_db_timeout)
    assert result['matches'][0]['id'] == 'id1'

def test_query_retries_exhausted():
    settings.vector_db_max_retries = 2 # Override for this specific test

    mock_index = MagicMock()
    mock_index.query.side_effect = PineconeException("Persistent query error")

    svc = VectorDBClient(
        dimension=settings.embedding_dimension,
        index_name=settings.pinecone_index_name,
        pinecone_client=MagicMock(),
        pinecone_index=mock_index,
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_cloud=settings.pinecone_cloud,
        pinecone_region=settings.pinecone_region,
    )

    vector = [0.0]*settings.embedding_dimension
    with pytest.raises(PineconeException):
        svc.query(vector)

    assert mock_index.query.call_count == settings.vector_db_max_retries + 1

