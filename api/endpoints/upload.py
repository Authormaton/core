"""
Upload endpoint for document ingestion.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from services.file_service import save_upload_file
from models.schemas import UploadResponse

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):  # noqa: B008
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    # Validate content type
    allowed_types = {"application/pdf", "text/plain"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    try:
        save_upload_file(file.file, file.filename)
        return UploadResponse(filename=file.filename, message="File uploaded successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to upload file.") from e
