"""
Service for generating embeddings for text chunks using transformers.
"""


import os
from typing import List
from openai import OpenAI, APIConnectionError, APIError, AuthenticationError, RateLimitError
import time
import random
from config.settings import settings
from services.logging_config import get_logger
from tenacity import retry, wait_exponential, stop_after_attempt, before_log, after_log, retry_if_exception_type

logger = get_logger(__name__)
from services.exceptions import DocumentEmbeddingError

def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment.")
        raise ValueError("OPENAI_API_KEY not set in environment.")
    return api_key

def embed_texts(texts: List[str], model: str = "text-embedding-3-small", timeout: float = 30.0) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI's embedding API (v1 client).
    Returns a list of embedding vectors (as lists of floats).
    """
    if not texts:
        return []
    client = OpenAI(api_key=get_openai_api_key(), timeout=timeout)
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

def embed_texts_batched(texts: List[str]) -> List[List[float]]:
    batch_size = settings.embed_batch_size
    model = settings.embedding_model
    expected_dim = settings.embedding_dimension
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for attempt in range(4):
            try:
                vectors = embed_texts(batch, model=model)
                if any(len(vec) != expected_dim for vec in vectors):
                    logger.error("EMBEDDING_DIMENSION_MISMATCH: One or more vectors have incorrect dimension for batch starting at index %d", i)
                    raise ValueError("EMBEDDING_DIMENSION_MISMATCH: One or more vectors have incorrect dimension")
                    raise DocumentEmbeddingError("EMBEDDING_DIMENSION_MISMATCH: One or more vectors have incorrect dimension")
                all_vectors.extend(vectors)
                break
            except Exception as e:
                logger.warning("Error embedding batch, attempt %d/%d: %s", attempt + 1, 4, e)
                if attempt < 3:
                    time.sleep(2 ** attempt)
                else:
                    logger.exception("Failed to embed batch after multiple retries.")
                    raise
                    raise DocumentEmbeddingError(f"Failed to generate embeddings after multiple retries: {e}") from e
    return all_vectors
