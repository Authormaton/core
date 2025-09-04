
"""
Service for splitting text into chunks for embedding and retrieval.
"""

import logging
from services.exceptions import DocumentChunkError

def chunk_text(text: str, max_length: int = 500, overlap: int = 50):
    """
    Splits text into overlapping chunks of max_length with specified overlap.
    Returns a list of text chunks.
    """
    logger = logging.getLogger(__name__)
    try:
        if not text:
            return []
        if max_length <= 0:
            logger.error("Invalid max_length for chunking: %d", max_length)
            raise DocumentChunkError("max_length must be greater than 0.")
        if overlap < 0:
            logger.error("Invalid overlap for chunking: %d", overlap)
            raise DocumentChunkError("overlap must be >= 0.")
        if overlap >= max_length:
            logger.error("overlap >= max_length for chunking: %d >= %d", overlap, max_length)
            raise DocumentChunkError("overlap must be less than max_length.")
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
    except Exception as e:
        logger.exception("Error during text chunking")
        raise DocumentChunkError(f"Failed to chunk text: {e}") from e
