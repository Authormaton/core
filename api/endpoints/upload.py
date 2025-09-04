"""
Upload endpoint for document ingestion.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
from services.file_service import save_upload_file
from services.parsing_service import extract_text_from_pdf
from models.schemas import UploadResponse
from services.exceptions import DocumentSaveError, DocumentParseError, DocumentChunkError, DocumentEmbeddingError


logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):  # noqa: B008
    try:
        saved_path = save_upload_file(file.file, file.filename)
        parsing_status = "success"
        text_preview = None
        if file.content_type == "application/pdf":
            try:
                text = extract_text_from_pdf(saved_path)
                text_preview = text[:500] if text else None
            except DocumentParseError:
                logger.error("Document parse error")
                parsing_status = "failed"
            except Exception:
                logger.exception("Error parsing PDF file for preview: %s", saved_path)
                parsing_status = "failed"
        elif file.content_type in {"text/plain", "text/markdown"}:
            try:
                with open(saved_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read(500)
                text_preview = text if text else None
            except UnicodeDecodeError:
                parsing_status = "failed"
                logger.error("Unicode decode error while reading file for preview: %s", saved_path)
            except OSError:
                parsing_status = "failed"
                logger.error("OS error while reading file for preview: %s", saved_path)
        return UploadResponse(
            filename=file.filename,
            message="File uploaded and parsed.",
            parsing_status=parsing_status,
            text_preview=text_preview
        )
    except DocumentSaveError as dse:
        logger.error("Document save error: %s", dse)
        raise HTTPException(status_code=400, detail=str(dse)) from dse
    except (DocumentParseError, DocumentChunkError, DocumentEmbeddingError) as de:
        logger.error("Document processing error: %s", de)
        raise HTTPException(status_code=422, detail=str(de))
    except Exception:
        logger.exception("Unhandled error in upload_document")
        raise HTTPException(status_code=500, detail="Internal server error")
