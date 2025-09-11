"""
Unit tests for embed_texts_batched (batching, retries, dimension guard).
"""
import sys
import types
import pytest

# Patch config.settings before importing anything that uses it
class DummySettings:
    embedding_dimension = 16
    embed_batch_size = 64
    embedding_model = "test-model"

dummy_config = types.ModuleType("config.settings")
dummy_config.settings = DummySettings()
sys.modules["config.settings"] = dummy_config

from services.embedding_service import embed_texts_batched

class DummyEmbed:
    def __init__(self, dim):
        self.dim = dim
        self.calls = 0
    def __call__(self, texts, model=None):
        self.calls += 1
        return [[0.0]*self.dim for _ in texts]

def test_batching(monkeypatch):
    from config.settings import settings
    dummy = DummyEmbed(settings.embedding_dimension)
    monkeypatch.setattr('services.embedding_service.embed_texts', dummy)
    texts = ["a"]*257
    vectors = embed_texts_batched(texts)
    assert len(vectors) == 257
    assert dummy.calls == (257 // settings.embed_batch_size) + 1

def test_dimension_guard(monkeypatch):
    from config.settings import settings
    def bad_embed(texts, model=None):
        return [[0.0]*10 for _ in texts]
    monkeypatch.setattr('services.embedding_service.embed_texts', bad_embed)
    with pytest.raises(ValueError):
        embed_texts_batched(["a"]*5)
