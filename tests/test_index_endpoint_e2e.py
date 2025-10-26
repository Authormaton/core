
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