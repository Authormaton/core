"""
Handles document parsing and text extraction (PDF, DOCX, etc.).
"""

from PyPDF2 import PdfReader
from typing import List

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
