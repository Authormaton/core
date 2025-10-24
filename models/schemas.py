"""
Pydantic models for request/response schemas.
"""

from pydantic import BaseModel

class UploadResponse(BaseModel):
    """
    Represents the response returned after a file upload operation.

    Attributes:
        filename (str): The name of the file that was uploaded.
        message (str): A human-readable message indicating the result of the upload.
        parsing_status (str): The status of the file parsing (e.g., 'success', 'failed', 'pending').
        text_preview (str | None): An optional preview of the extracted text content from the file.
    """

class DocumentChunk(BaseModel):
    """
    Represents a single chunk of a document with its associated metadata.

    Attributes:
        id (str): A unique identifier for the chunk.
        order (int): The sequential order of the chunk within the document.
        chunk_start (int): The starting character index of the chunk in the original text (inclusive).
        chunk_end (int): The ending character index of the chunk in the original text (exclusive).
        text (str): The textual content of the chunk.
        estimated_tokens (int): An estimation of the number of tokens in the chunk.
    """
    id: str
    order: int
    chunk_start: int
    chunk_end: int
    text: str
    estimated_tokens: int