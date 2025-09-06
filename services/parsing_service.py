"""
Handles document parsing and text extraction (PDF, DOCX, etc.).
"""



import logging
from services.exceptions import DocumentParseError

def extract_text_from_pdf(file_path: str) -> str:
    logger = logging.getLogger(__name__)
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except ImportError as e:
        logger.error("PyPDF2 import failed for PDF parsing: %s", e)
        raise DocumentParseError("PyPDF2 is required for PDF parsing. Please install it with 'pip install PyPDF2'.") from e
    except Exception as e:
        logger.exception("Unexpected error during PDF parsing: %s", file_path)
        raise DocumentParseError(f"Failed to parse PDF: {file_path}. Error: {e}") from e
