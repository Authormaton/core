"""
Indexing router for /internal/index endpoint.
"""
from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel
from config.settings import settings
from services.vector_db_service import VectorDBClient as VectorDBService
from services.embedding_service import embed_texts_batched
from services.chunking_service import chunk_text
from services.parsing_service import extract_text_from_pdf, extract_text_from_docx
import logging
import os

router = APIRouter(prefix="/internal", tags=["internal"])

class IndexRequest(BaseModel):
    project_id: str
    sources: list[dict]

class IndexResponse(BaseModel):
    project_id: str
    indexed_chunks: int
    sources_indexed: int
    skipped_sources: int

@router.post("/index", response_model=IndexResponse, status_code=201)
def index(request: IndexRequest, req: Request):
    vdb = VectorDBService()
    indexed_chunks = 0
    sources_indexed = 0
    skipped_sources = 0
    for src in request.sources:
        text = None
        source_id = src.get("source_id") or src.get("file_id") or "unknown"
        file_path = src.get("file_path", "")
        # Upload size check
        if "text" in src and len(src["text"]) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="UPLOAD_TOO_LARGE")
        if "file_id" in src:
            # TODO: Load file by file_id, check MIME/type, enforce size
            skipped_sources += 1
            continue
        elif "text" in src:
            text = src["text"]
        else:
            skipped_sources += 1
            continue
        # Chunk text
        try:
            chunks = chunk_text(text)
        except Exception:
            raise HTTPException(status_code=415, detail="UNSUPPORTED_FILE_TYPE")
        ids = [f"{source_id}:{i}" for i in range(len(chunks))]
        # Prepare metadata for future implementation that will support metadata
        metadata = [{
            "project_id": request.project_id,
            "source_id": source_id,
            "file_path": file_path,
            "page": 1,
            "chunk_id": ids[i],
            "char_span": [0, len(chunks[i])]
        } for i in range(len(chunks))]
        # TODO: Add metadata support once VectorDBService supports it
        # Embed
        try:
            vectors = embed_texts_batched(chunks)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="EMBEDDING_DIMENSION_MISMATCH")
        # Create index if needed, then upsert vectors
        try:
            # Ensure index is initialized first
            vdb.create_index()
            # Then upsert vectors with the proper API
            vdb.upsert_vectors(vectors=vectors, ids=ids)
            # Note: metadata support will be added in a follow-up
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"VECTOR_DB_OPERATION_FAILED: {str(e)}")
        indexed_chunks += len(chunks)
        sources_indexed += 1
    return IndexResponse(
        project_id=request.project_id,
        indexed_chunks=indexed_chunks,
        sources_indexed=sources_indexed,
        skipped_sources=skipped_sources
    )
