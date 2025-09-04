"""
Service for splitting text into chunks for embedding and retrieval.
"""

def chunk_text(text: str, max_length: int = 500, overlap: int = 50):
    """
    Splits text into overlapping chunks of max_length with specified overlap.
    Returns a list of text chunks.
    """
    if not text:
        return []
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= max_length:
        raise ValueError("overlap must be less than max_length.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start += max_length - overlap
    return chunks
