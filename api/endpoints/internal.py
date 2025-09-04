"""
Internal API endpoint for secure processing of source materials and a prompt.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional

# Replace with your actual internal API key or authentication logic
INTERNAL_API_KEY = "your-secure-internal-key"
api_key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)

router = APIRouter(prefix="/internal", tags=["internal"])

class SourceMaterialRequest(BaseModel):
    source_material: str  # Could be file content, text, or base64
    prompt: str
    metadata: Optional[dict] = None

# Dependency for internal authentication
def verify_internal_api_key(api_key: str = Depends(api_key_header)):
    if api_key != INTERNAL_API_KEY:
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
