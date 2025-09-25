"""
Internal API endpoint for secure processing of source materials and a prompt.
"""


import os
import secrets

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import base64
import binascii
import time
import threading
import uuid

try:
    from prometheus_client import Counter, Histogram
except Exception:
    Counter = None
    Histogram = None
import tempfile
import logging
from services.parsing_service import extract_text_from_pdf, extract_text_from_docx
from services.chunking_service import chunk_text
from services.embedding_service import embed_texts
# VectorDBClient intentionally not imported by default here; integrate in production


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # used by embedding service
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY")
api_key_header = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)

if not INTERNAL_API_KEY:
    raise RuntimeError("INTERNAL_API_KEY environment variable is required for internal API authentication.")

router = APIRouter(prefix="/internal", tags=["internal"])

# Simple in-memory job store for background processing (for demo/testing only)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Metrics (optional)
if Counter:
    INTERNAL_REQUESTS = Counter("internal_requests_total", "Internal API requests", ["status"])
    EMBEDDING_DURATION = Histogram("embedding_duration_seconds", "Embedding generation duration")
else:
    INTERNAL_REQUESTS = EMBEDDING_DURATION = None



class SourceMaterialRequest(BaseModel):
    source_material: str = Field(..., description="Base64-encoded file content for PDF/DOCX, or plain text")
    prompt: str = Field(..., max_length=2000)
    metadata: Optional[dict] = None
    file_type: Optional[str] = Field(None, description="pdf|docx|text")

    @validator("file_type")
    def validate_file_type(cls, v):
        if v is None:
            return v
        if v not in {"pdf", "docx", "text"}:
            raise ValueError("file_type must be one of: pdf, docx, text")
        return v

    @validator("source_material")
    def base64_or_text_size(cls, v, values):
        # If file_type is text or None, allow plain text up to a configured size
        file_type = values.get("file_type")
        max_bytes = int(os.getenv("INTERNAL_MAX_BYTES", str(5 * 1024 * 1024)))
        if file_type in ("pdf", "docx"):
            # validate base64 roughly
            try:
                # base64 length roughly 4/3 of the binary size
                approx = (len(v) * 3) // 4
                if approx > max_bytes:
                    raise ValueError("source_material too large")
                base64.b64decode(v, validate=True)
            except (binascii.Error, ValueError) as e:
                raise ValueError("Invalid or too-large base64 source_material") from e
        else:
            # plain text size check
            if len(v.encode("utf-8", errors="ignore")) > max_bytes:
                raise ValueError("source_material too large")
        return v

# Dependency for internal authentication
def verify_internal_api_key(api_key: str = Depends(api_key_header)):
    # Do not log or expose the secret
    if not api_key or not INTERNAL_API_KEY or not secrets.compare_digest(api_key, INTERNAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key."
        )

@router.post("/process-material", response_model=dict)

def _create_job_record(status: str, info: Dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": status, "info": info, "created_at": time.time()}
    return job_id


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _background_process(job_id: str, request: SourceMaterialRequest) -> None:
    logger = logging.getLogger(__name__)
    try:
        _update_job(job_id, status="processing")

        # Decode/parse
        tmp_file = None
        text = None
        try:
            if request.file_type == "pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as tmp:
                    data = base64.b64decode(request.source_material)
                    tmp.write(data)
                    tmp.flush()
                    tmp_file = tmp.name
                    text = extract_text_from_pdf(tmp_file)
            elif request.file_type == "docx":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", mode="wb") as tmp:
                    data = base64.b64decode(request.source_material)
                    tmp.write(data)
                    tmp.flush()
                    tmp_file = tmp.name
                    text = extract_text_from_docx(tmp_file)
            else:
                text = request.source_material
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

        if not text:
            _update_job(job_id, status="failed", error="No text extracted")
            return

        # Chunk
        chunks = chunk_text(text)
        if not chunks:
            _update_job(job_id, status="failed", error="No chunks generated")
            return

        # Embeddings (with optional histogram)
        try:
            if EMBEDDING_DURATION:
                with EMBEDDING_DURATION.time():
                    embeddings = embed_texts(chunks)
            else:
                embeddings = embed_texts(chunks)
            embedding_count = len(embeddings) if embeddings else 0
        except Exception as e:
            logger.exception("Embedding generation failed")
            _update_job(job_id, status="failed", error=str(e))
            return

        # TODO: upsert embeddings into vector DB
        # For now, we store counts and a draft
        draft = f"Draft generated for prompt: {request.prompt}\n\n" + "\n---\n".join(chunks[:3])
        _update_job(job_id, status="completed", result={
            "num_chunks": len(chunks),
            "embedding_count": embedding_count,
            "draft_preview": draft,
        })
    except Exception:
        logger.exception("Unexpected error in background job")
        _update_job(job_id, status="failed", error="Unexpected error")


@router.post("/process-material", response_model=dict)
def process_material(
    request: SourceMaterialRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_internal_api_key),
):
    """Validate input and schedule background processing; return job id."""
    logger = logging.getLogger(__name__)
    try:
        # Create job record and schedule background processing
        job_id = _create_job_record("pending", {"prompt_len": len(request.prompt)})
        background_tasks.add_task(_background_process, job_id, request)

        if INTERNAL_REQUESTS:
            INTERNAL_REQUESTS.labels(status="accepted").inc()

        return {"status": "accepted", "job_id": job_id}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error scheduling background job")
        if INTERNAL_REQUESTS:
            INTERNAL_REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/job/{job_id}")
def job_status(job_id: str, _: str = Depends(verify_internal_api_key)):
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return _jobs[job_id]
