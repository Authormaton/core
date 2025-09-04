"""
Service for generating embeddings for text chunks using transformers.
"""


import os
from typing import List
import openai

def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment.")
    return api_key

def embed_texts(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI's embedding API.
    Returns a list of embedding vectors (as lists of floats).
    """
    if not texts:
        return []
    openai.api_key = get_openai_api_key()
    response = openai.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]
    # Expand mask for broadcasting
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
    mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)
    embeddings = sum_hidden / mask_sum
    return embeddings.cpu().tolist()
