"""
Upload endpoint for document ingestion.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from services.file_service import save_upload_file
from models.schemas import UploadResponse

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):
    try:
        file_path = save_upload_file(file.file, file.filename)
        return UploadResponse(filename=file.filename, message="File uploaded successfully.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
