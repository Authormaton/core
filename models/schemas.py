"""
Pydantic models for request/response schemas.
"""

from pydantic import BaseModel

class UploadResponse(BaseModel):
    filename: str
    message: str
