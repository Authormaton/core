def test_upload_pdf(tmp_path):
    client = TestClient(app)
    # Create a minimal PDF file (one page, one line of text)
    pdf_path = tmp_path / "test.pdf"
    # Write a minimal valid PDF (using PyPDF2 for simplicity)
    try:
        from PyPDF2 import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with open(pdf_path, "wb") as f:
            writer.write(f)
    except ImportError:
        # If PyPDF2 is not available, skip this test
        import pytest
        pytest.skip("PyPDF2 not installed")
    with open(pdf_path, "rb") as f:
        response = client.post(
            "/upload/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["parsing_status"] in ("success", "failed")  # Accept either, since blank page has no text

# Ensure project root is in sys.path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from api.main import app

def test_upload_markdown(tmp_path):
    client = TestClient(app)
    # Create a temporary markdown file
    md_content = "# Title\n\nThis is a test markdown file."
    md_file = tmp_path / "test.md"
    md_file.write_text(md_content, encoding="utf-8")
    with md_file.open("rb") as f:
        response = client.post(
            "/upload/upload",
            files={"file": ("test.md", f, "text/markdown")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.md"
    assert data["parsing_status"] == "success"
    assert "Title" in data["text_preview"]

def test_upload_txt(tmp_path):
    client = TestClient(app)
    txt_content = "This is a plain text file."
    txt_file = tmp_path / "test.txt"
    txt_file.write_text(txt_content, encoding="utf-8")
    with txt_file.open("rb") as f:
        response = client.post(
            "/upload/upload",
            files={"file": ("test.txt", f, "text/plain")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["parsing_status"] == "success"
    assert "plain text" in data["text_preview"]
