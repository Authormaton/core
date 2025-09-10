"""
Service for generating embeddings for text chunks using transformers.
"""


import os
from typing import List
from openai import OpenAI, APIConnectionError, APIError, AuthenticationError, RateLimitError
import time
import random
from config.settings import settings

def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    return api_key

def embed_texts(texts: List[str], model: str = "text-embedding-3-small", timeout: float = 30.0, max_retries: int = 2) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI's embedding API (v1 client).
    Returns a list of embedding vectors (as lists of floats).
    Retries on rate limit, with exponential backoff.
    """
    if not texts:
        return []
    client = OpenAI(api_key=get_openai_api_key(), timeout=timeout, max_retries=max_retries)
    for attempt in range(max_retries + 1):
        try:
            response = client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except AuthenticationError as e:
            # Credentials are invalid or expired
            raise
        except RateLimitError as e:
            # Client will retry up to max_retries; bubble up if still failing
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt + random.random())
        except (APIConnectionError, APIError) as e:
            # Network or other API errors
            raise

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
                    raise ValueError("EMBEDDING_DIMENSION_MISMATCH: One or more vectors have incorrect dimension")
                all_vectors.extend(vectors)
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                else:
                    raise
    return all_vectors
