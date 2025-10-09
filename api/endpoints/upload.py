"""
Upload endpoint for document ingestion.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, status
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from services.file_service import save_upload_file
from services.parsing_service import extract_text_from_pdf
from models.schemas import UploadResponse
from services.exceptions import DocumentSaveError, DocumentParseError, DocumentChunkError, DocumentEmbeddingError

from services.logging_config import get_logger, set_log_context, clear_log_context

logger = get_logger(__name__)
router = APIRouter()

# Configuration: allowed types and size (bytes)
ALLOWED_CONTENT_TYPES = {"application/pdf", "text/plain", "text/markdown"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))  # default 25 MB


def _secure_filename(name: str) -> str:
    # Simple sanitization: take only base name and strip suspicious characters
    base = pathlib.Path(name).name
    # remove path separators and control chars
    return "".join(c for c in base if c.isprintable())


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):  # noqa: B008
    """Upload and do a light parse to provide a preview.

    This endpoint is async but offloads blocking file IO to a threadpool.
    """
    # Per-request logging context
    request_id = request.headers.get("X-Request-Id") or None
    if request_id:
        set_log_context(request_id=request_id)
    try:
        # Basic content-type and size checks
        content_type = (file.content_type or "").lower()
        if content_type not in ALLOWED_CONTENT_TYPES:
            logger.warning("Rejected upload due to content-type", extra={"content_type": content_type})
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type")

        # If client provided Content-Length header, check early
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_UPLOAD_BYTES:
                    logger.warning("Rejected upload due to size header too large", extra={"size": content_length})
                    raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")
            except ValueError:
                # ignore invalid header and continue with streaming checks
                pass

        # Sanitize filename
        filename = _secure_filename(file.filename or "upload")

        # Offload blocking save to threadpool
        loop = __import__("asyncio").get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as ex:
            saved_path = await loop.run_in_executor(ex, save_upload_file, file.file, filename, MAX_UPLOAD_BYTES)

        parsing_status = "success"
        text_preview: Optional[str] = None

        if content_type == "application/pdf":
            try:
                text = extract_text_from_pdf(saved_path)
                text_preview = text[:500] if text else None
            except DocumentParseError:
                logger.error("Document parse error", extra={"path": saved_path})
                parsing_status = "failed"
            except Exception:
                logger.exception("Error parsing PDF file for preview", extra={"path": saved_path})
                parsing_status = "failed"
        elif content_type in {"text/plain", "text/markdown"}:
            try:
                with open(saved_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(500)
                text_preview = text if text else None
            except UnicodeDecodeError:
                parsing_status = "failed"
                logger.error("Unicode decode error while reading file for preview", extra={"path": saved_path})
            except OSError:
                parsing_status = "failed"
                logger.error("OS error while reading file for preview", extra={"path": saved_path})

        return UploadResponse(
            filename=filename,
            message="File uploaded and parsed.",
            parsing_status=parsing_status,
            text_preview=text_preview,
        )
    except DocumentSaveError as dse:
        logger.error("Document save error", extra={"error": str(dse)})
        raise HTTPException(status_code=400, detail="Failed to save uploaded document") from dse
    except (DocumentParseError, DocumentChunkError, DocumentEmbeddingError) as de:
        logger.error("Document processing error", extra={"error": str(de)})
        raise HTTPException(status_code=422, detail="Error processing document")
    finally:
        # Clear per-request logging context
        clear_log_context()
