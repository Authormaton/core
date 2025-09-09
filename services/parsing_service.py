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

def extract_text_from_docx(file_path: str) -> str:
    logger = logging.getLogger(__name__)
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except ImportError as e:
        logger.error("python-docx import failed for DOCX parsing: %s", e)
        raise DocumentParseError("python-docx is required for DOCX parsing. Please install it with 'pip install python-docx'.") from e
    except Exception as e:
        logger.exception("Unexpected error during DOCX parsing: %s", file_path)
        raise DocumentParseError(f"Failed to parse DOCX: {file_path}. Error: {e}") from e
