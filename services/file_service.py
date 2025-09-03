"""
Handles file saving, validation, and management for uploads.
"""


import os
import uuid
import tempfile
import contextlib
from typing import IO

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')

os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_upload_file(upload_file: IO, filename: str) -> str:
    # Sanitize filename: remove path, reject empty/unsafe
    base = os.path.basename(filename)
    if not base or base in {'.', '..'} or any(c in base for c in '\/:*?"<>|'):
        raise ValueError("Invalid filename.")

    # Generate a safe unique filename (preserve extension if present)
    ext = os.path.splitext(base)[1]
    safe_name = f"{uuid.uuid4().hex}{ext}"
    final_path = os.path.abspath(os.path.join(UPLOAD_DIR, safe_name))

    # Ensure final path is within UPLOAD_DIR
    upload_dir_abs = os.path.abspath(UPLOAD_DIR)
    if not final_path.startswith(upload_dir_abs + os.sep):
        raise ValueError("Unsafe file path.")

    # Write file atomically in chunks

    try:
        with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, delete=False) as tmp:
            for chunk in iter(lambda: upload_file.read(8192), b""):
                tmp.write(chunk)
            temp_path = tmp.name
        os.replace(temp_path, final_path)
        try:
            os.chmod(final_path, 0o600)
        except OSError:
            pass  # Best effort, ignore chmod errors
    except OSError as e:
        # Clean up temp file if something goes wrong
        with contextlib.suppress(Exception):
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        raise IOError("Failed to save file securely.") from e

    return final_path
