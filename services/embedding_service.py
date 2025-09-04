"""
Service for generating embeddings for text chunks using transformers.
"""


from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
import threading

# You can change the model name to a suitable sentence transformer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


_tokenizer = None
_model = None
_model_lock = threading.Lock()
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    with _model_lock:
        if _tokenizer is not None and _model is not None:
            return _tokenizer, _model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model = model.to(_device)
        model.eval()
        _tokenizer = tokenizer
        _model = model
        return _tokenizer, _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    Returns a list of embedding vectors (as lists of floats).
    """
    if not texts:
        return []
    tokenizer, model = get_model_and_tokenizer()
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model_output = model(**inputs)
    last_hidden = model_output.last_hidden_state
    attention_mask = inputs["attention_mask"].float()
    # Expand mask for broadcasting
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
    mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-9)
    embeddings = sum_hidden / mask_sum
    return embeddings.cpu().tolist()
