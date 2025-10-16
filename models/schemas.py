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
