"""
Handles document parsing and text extraction (PDF, DOCX, etc.).
"""


from typing import List

def extract_text_from_pdf(file_path: str) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError as e:
        raise ImportError("PyPDF2 is required for PDF parsing. Please install it with 'pip install PyPDF2'.") from e
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
