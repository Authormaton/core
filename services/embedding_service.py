"""
Service for generating embeddings for text chunks using transformers.
"""

from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

# You can change the model name to a suitable sentence transformer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_tokenizer = None
_model = None

def get_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
    return _tokenizer, _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    Returns a list of embedding vectors (as lists of floats).
    """
    tokenizer, model = get_model_and_tokenizer()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    # Mean pooling
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.cpu().tolist()
