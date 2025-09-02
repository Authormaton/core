"""
Handles file saving, validation, and management for uploads.
"""

import os
from typing import IO

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')

os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_upload_file(upload_file: IO, filename: str) -> str:
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, 'wb') as out_file:
        out_file.write(upload_file.read())
    return file_path
