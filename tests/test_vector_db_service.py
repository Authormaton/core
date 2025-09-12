

import pytest
from services.vector_db_service import VectorDBClient

class DummyIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.upserted = []
    def upsert(self, vectors=None):
        if vectors:
            self.upserted.extend(vectors)
    def query(self, vector, top_k=5):
        return {'matches': [{'id': 'id1', 'score': 0.99}]}

@pytest.fixture
def vdb(monkeypatch):
    svc = VectorDBClient(dimension=8, index_name="test-index")
    monkeypatch.setattr(svc, 'index', DummyIndex(svc.dimension))
    return svc

def test_create_index(monkeypatch):
    svc = VectorDBClient(dimension=8, index_name="test-index")
    monkeypatch.setattr(svc.pc, 'list_indexes', lambda: [])
    monkeypatch.setattr(svc.pc, 'create_index', lambda **kwargs: None)
    monkeypatch.setattr(svc.pc, 'Index', lambda name: DummyIndex(8))
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

