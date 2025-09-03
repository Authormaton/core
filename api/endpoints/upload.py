"""
Upload endpoint for document ingestion.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
from services.file_service import save_upload_file
from services.parsing_service import extract_text_from_pdf
from models.schemas import UploadResponse


logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):  # noqa: B008
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    # Validate content type
    allowed_types = {"application/pdf", "text/plain", "text/markdown"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    try:
        saved_path = save_upload_file(file.file, file.filename)
        parsing_status = "success"
        text_preview = None
        if file.content_type == "application/pdf":
            import logging
            try:
                text = extract_text_from_pdf(saved_path)
                text_preview = text[:500] if text else None
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
                logger.exception("Unicode decode error while reading file for preview: %s", saved_path)
            except OSError:
                parsing_status = "failed"
                logger.exception("OS error while reading file for preview: %s", saved_path)
        return UploadResponse(
            filename=file.filename,
            message="File uploaded and parsed.",
            parsing_status=parsing_status,
            text_preview=text_preview
        )
    except HTTPException:
        # Preserve explicitly raised HTTP errors (e.g., 400/415).
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to upload file.") from e
