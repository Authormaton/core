"""
Internal API endpoint for secure processing of source materials and a prompt.
"""


import os
import secrets
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
api_key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)

router = APIRouter(prefix="/internal", tags=["internal"])

class SourceMaterialRequest(BaseModel):
    source_material: str  # Could be file content, text, or base64
    prompt: str
    metadata: Optional[dict] = None

# Dependency for internal authentication
def verify_internal_api_key(api_key: str = Depends(api_key_header)):
    # Do not log or expose the secret
    if not api_key or not OPENAI_API_KEY or not secrets.compare_digest(api_key, OPENAI_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key."
        )

@router.post("/process-material", response_model=dict)
def process_material(
    request: SourceMaterialRequest,
    _: str = Depends(verify_internal_api_key)
):
    # TODO: Integrate with parsing, chunking, embedding services
    # Example stub response
    return {
        "status": "success",
        "received_material_length": len(request.source_material),
        "prompt": request.prompt,
        "metadata": request.metadata or {}
    }
