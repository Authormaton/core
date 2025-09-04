import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import os
import pytest
from dotenv import load_dotenv
from services.chunking_service import chunk_text
from services.embedding_service import embed_texts

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_chunk_text_basic():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, max_length=10, overlap=2)
    assert len(chunks) > 1
    assert all(len(chunk) <= 10 for chunk in chunks)
    # Check overlap
    if len(chunks) > 1:
        assert chunks[0][-2:] == chunks[1][:2]


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_embed_texts_shape(mocker):
    fake_response = type('FakeResponse', (), {
        'data': [
            type('FakeEmbedding', (), {'embedding': [0.1, 0.2, 0.3]})(),
            type('FakeEmbedding', (), {'embedding': [0.4, 0.5, 0.6]})()
        ]
    })()
    mocker.patch("openai.embeddings.create", return_value=fake_response)
    texts = ["hello world", "test embedding"]
    embeddings = embed_texts(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(x, float) for vec in embeddings for x in vec)
    assert all(len(vec) > 0 for vec in embeddings)


def test_chunk_text_invalid_params():
    # max_length <= 0
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        chunk_text("abc", max_length=0, overlap=0)
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        chunk_text("abc", max_length=-1, overlap=0)
    # overlap < 0
    with pytest.raises(ValueError, match="overlap must be >= 0"):
        chunk_text("abc", max_length=5, overlap=-1)
    # overlap >= max_length
    with pytest.raises(ValueError, match="overlap must be less than max_length"):
        chunk_text("abc", max_length=5, overlap=5)
    with pytest.raises(ValueError, match="overlap must be less than max_length"):
        chunk_text("abc", max_length=5, overlap=6)
import pytest

@pytest.mark.integration
def test_embed_texts_openai_live():
    texts = ["OpenAI test", "Embedding API"]
    embeddings = embed_texts(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(x, float) for vec in embeddings for x in vec)
    assert all(len(vec) > 0 for vec in embeddings)

def test_chunk_text_invalid_params():
    # max_length <= 0
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        chunk_text("abc", max_length=0, overlap=0)
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        chunk_text("abc", max_length=-1, overlap=0)
    # overlap < 0
    with pytest.raises(ValueError, match="overlap must be >= 0"):
        chunk_text("abc", max_length=5, overlap=-1)
    # overlap >= max_length
    with pytest.raises(ValueError, match="overlap must be less than max_length"):
        chunk_text("abc", max_length=5, overlap=5)
    with pytest.raises(ValueError, match="overlap must be less than max_length"):
        chunk_text("abc", max_length=5, overlap=6)
    texts = ["hello world", "test embedding"]
    embeddings = embed_texts(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(x, float) for vec in embeddings for x in vec)
    # Embedding size is model-dependent, but should be >0
    assert all(len(vec) > 0 for vec in embeddings)
