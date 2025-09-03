import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from services.chunking_service import chunk_text
from services.embedding_service import embed_texts

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

def test_embed_texts_shape():
    texts = ["hello world", "test embedding"]
    embeddings = embed_texts(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(x, float) for vec in embeddings for x in vec)
    # Embedding size is model-dependent, but should be >0
    assert all(len(vec) > 0 for vec in embeddings)
