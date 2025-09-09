"""
Internal API endpoint for secure processing of source materials and a prompt.
"""


import os
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import tempfile
import logging
from services.parsing_service import extract_text_from_pdf, extract_text_from_docx
from services.chunking_service import chunk_text
from services.embedding_service import embed_texts
from services.vector_db_service import VectorDBClient


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # used by embedding service
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY")
api_key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)

if not INTERNAL_API_KEY:
    raise RuntimeError("INTERNAL_API_KEY environment variable is required for internal API authentication.")

router = APIRouter(prefix="/internal", tags=["internal"])



class SourceMaterialRequest(BaseModel):
    source_material: str  # Base64-encoded file content for PDF/DOCX, or plain text
    prompt: str
    metadata: Optional[dict] = None
    file_type: Optional[str] = None  # e.g., 'pdf', 'docx', 'text'

# Dependency for internal authentication
def verify_internal_api_key(api_key: str = Depends(api_key_header)):
    # Do not log or expose the secret
    if not api_key or not INTERNAL_API_KEY or not secrets.compare_digest(api_key, INTERNAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key."
        )

@router.post("/process-material", response_model=dict)

def process_material(
    request: SourceMaterialRequest,
    _: str = Depends(verify_internal_api_key)
):
    logger = logging.getLogger(__name__)
    try:
        # Step 1: Parse document text
        text = None

        import base64
        import binascii
        tmp_file = None
        try:
            if request.file_type == "pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as tmp:
                    try:
                        data = base64.b64decode(request.source_material)
                    except (binascii.Error, ValueError) as decode_err:
                        raise HTTPException(status_code=400, detail="Invalid base64-encoded source_material for PDF upload.")
                    tmp.write(data)
                    tmp.flush()
                    tmp_file = tmp.name
                    text = extract_text_from_pdf(tmp.name)
            elif request.file_type == "docx":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", mode="wb") as tmp:
                    try:
                        data = base64.b64decode(request.source_material)
                    except (binascii.Error, ValueError) as decode_err:
                        raise HTTPException(status_code=400, detail="Invalid base64-encoded source_material for DOCX upload.")
                    tmp.write(data)
                    tmp.flush()
                    tmp_file = tmp.name
                    text = extract_text_from_docx(tmp.name)
            else:
                text = request.source_material
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

        if not text:
            raise HTTPException(status_code=400, detail="No text extracted from source material.")

        # Step 2: Chunk text
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from text.")

        # Step 3: Generate embeddings
        try:
            embeddings = embed_texts(chunks)
        except Exception as e:
            logger.exception("Embedding generation failed")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

        # Step 4: Store embeddings in vector DB (stub)
        # NOTE: Replace with real API key/environment for Pinecone
        # vector_db = VectorDBClient(api_key="YOUR_PINECONE_API_KEY", environment="YOUR_PINECONE_ENV")
        # vector_db.create_index("authormaton-index", dimension=len(embeddings[0]))
        # vector_db.upsert_vectors(embeddings, [str(i) for i in range(len(embeddings))])

        # Step 5: Synthesize draft (stub)
        draft = f"Draft generated for prompt: {request.prompt}\n\n" + "\n---\n".join(chunks[:3])

        return {
            "status": "success",
            "received_material_length": len(request.source_material),
            "prompt": request.prompt,
            "metadata": request.metadata or {},
            "num_chunks": len(chunks),
            "draft_preview": draft,
        }
    except Exception as e:
        logger.exception("Error in process_material pipeline")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
