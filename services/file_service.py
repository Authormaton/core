"""
Handles file saving, validation, and management for uploads.
"""



import os
import uuid
import tempfile
import contextlib
from services.logging_config import get_logger

logger = get_logger(__name__)
from pathlib import Path
import hashlib
import asyncio
import mimetypes
from datetime import datetime, timezone, timedelta
from typing import IO, Optional, List, Dict, Any
from services.exceptions import DocumentSaveError
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MiB
_ENV_UPLOAD_DIR = os.environ.get("UPLOAD_DIR")
UPLOAD_DIR = os.path.abspath(_ENV_UPLOAD_DIR) if _ENV_UPLOAD_DIR else os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Optional comma-separated allowed content types (e.g. 'application/pdf,text/plain')
_ALLOWED_TYPES = os.environ.get("ALLOWED_CONTENT_TYPES")
ALLOWED_CONTENT_TYPES = set([t.strip() for t in _ALLOWED_TYPES.split(',')]) if _ALLOWED_TYPES else None

# Shared thread pool for offloading blocking file operations when used from async code
_THREAD_POOL: Optional[ThreadPoolExecutor] = None

def _get_thread_pool() -> ThreadPoolExecutor:
    global _THREAD_POOL
    if _THREAD_POOL is None:
        _THREAD_POOL = ThreadPoolExecutor(max_workers=4)
    return _THREAD_POOL

def save_upload_file(upload_file: IO, filename: str, max_bytes: Optional[int] = None) -> str:
    # Reject filenames containing path separators to prevent directory traversal
    if '/' in filename or '\\' in filename:
        logger.error("Filename contains path separators: %s", filename)
        raise DocumentSaveError("Invalid filename.")

    # Sanitize filename: remove path, reject empty/unsafe
    base = os.path.basename(filename)
    if not base or base in {'.', '..'} or any(c in base for c in '\\/:*?"<>|'):
        logger.error("Invalid filename for upload: %s", filename)
        raise DocumentSaveError("Invalid filename.")

    # Generate a safe unique filename (preserve extension if present)
    ext = os.path.splitext(base)[1]
    safe_name = f"{uuid.uuid4().hex}{ext}"
    final_path = os.path.abspath(os.path.join(UPLOAD_DIR, safe_name))

    # Ensure final path is within UPLOAD_DIR using resolved paths
    upload_dir_resolved = Path(UPLOAD_DIR).resolve()
    final_path_resolved = Path(final_path).resolve()
    if not final_path_resolved.is_relative_to(upload_dir_resolved):
        logger.error("Unsafe file path detected: %s (resolved to %s)", filename, final_path_resolved)
        raise DocumentSaveError("Unsafe file path.")

    effective_max_bytes = max_bytes if max_bytes is not None else DEFAULT_MAX_UPLOAD_BYTES
    total = 0
    try:
        with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, delete=False) as tmp:
            temp_path = tmp.name
            source = getattr(upload_file, 'file', upload_file)
            for chunk in iter(lambda: source.read(8192), b""):
                if not chunk:
                    break
                tmp.write(chunk)
                total += len(chunk)
                if total > effective_max_bytes:
                    logger.warning("Upload of file %s exceeds max allowed size: %d bytes (max %d)", filename, total, effective_max_bytes)
                    raise DocumentSaveError(f"Uploaded file '{filename}' exceeds maximum allowed size of {effective_max_bytes} bytes.")
        os.replace(temp_path, final_path)
        try:
            os.chmod(final_path, 0o600)
        except OSError:
            pass  # Best effort, ignore chmod errors
    except DocumentSaveError:
        with contextlib.suppress(Exception):
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        raise
    except OSError as e:
        logger.exception("OSError during file save: %s", filename)
        with contextlib.suppress(Exception):
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        raise DocumentSaveError("Failed to save file securely.") from e

    # Optionally detect mime type (by extension); more accurate detection can be added
    try:
        mime, _ = mimetypes.guess_type(final_path)
    except Exception:
        mime = None

    logger.info("Saved upload %s as %s (%d bytes, mime=%s)", base, final_path, total, mime)
    return final_path


def save_upload_file_with_meta(upload_file: IO, filename: str, max_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Save file and return metadata: path, size, sha256, mime_type.

    This keeps the original save behavior but returns useful metadata for callers.
    """
    # Reject filenames containing path separators to prevent directory traversal
    if '/' in filename or '\\' in filename:
        logger.error("Filename contains path separators: %s", filename)
        raise DocumentSaveError("Invalid filename.")

    # If caller passes a custom max_bytes use it, otherwise use default
    if max_bytes is None:
        max_bytes = DEFAULT_MAX_UPLOAD_BYTES

    base = os.path.basename(filename)
    if not base or base in {'.', '..'}:
        raise DocumentSaveError("Invalid filename.")

    ext = os.path.splitext(base)[1]
    safe_name = f"{uuid.uuid4().hex}{ext}"
    final_path = os.path.abspath(os.path.join(UPLOAD_DIR, safe_name))

    upload_dir_abs = os.path.abspath(UPLOAD_DIR)
    upload_dir_resolved = Path(UPLOAD_DIR).resolve()
    final_path_resolved = Path(final_path).resolve()
    if not final_path_resolved.is_relative_to(upload_dir_resolved):
        logger.error("Unsafe file path detected: %s (resolved to %s)", final_path, final_path_resolved)
        raise DocumentSaveError("Unsafe file path.")

    sha256 = hashlib.sha256()
    total = 0

    try:
        with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, delete=False) as tmp:
            temp_path = tmp.name
            source = getattr(upload_file, 'file', upload_file)
            for chunk in iter(lambda: source.read(8192), b""):
                if not chunk:
                    break
                tmp.write(chunk)
                total += len(chunk)
                sha256.update(chunk)
                if total > max_bytes:
                    logger.warning("Upload of file %s exceeds max allowed size: %d bytes (max %d)", filename, total, max_bytes)
                    raise DocumentSaveError(f"Uploaded file '{filename}' exceeds maximum allowed size of {max_bytes} bytes.")
        os.replace(temp_path, final_path)
        try:
            os.chmod(final_path, 0o600)
        except OSError:
            pass
    except DocumentSaveError:
        with contextlib.suppress(Exception):
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        raise
    except OSError as e:
        logger.exception("OSError during file save: %s", filename)
        with contextlib.suppress(Exception):
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        raise DocumentSaveError("Failed to save file securely.") from e

    try:
        mime, _ = mimetypes.guess_type(final_path)
    except Exception:
        mime = None

    meta = {
        'path': final_path,
        'original_name': base,
        'size': total,
        'sha256': sha256.hexdigest(),
        'mime_type': mime,
        'saved_at': datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Saved upload metadata: %s", {k: meta[k] for k in ('original_name','path','size','mime_type')})
    return meta


async def save_upload_file_async(upload_file: IO, filename: str, max_bytes: Optional[int] = None) -> str:
    """Async wrapper that offloads the blocking save call to a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_thread_pool(), save_upload_file, upload_file, filename, max_bytes)


async def save_upload_file_with_meta_async(upload_file: IO, filename: str, max_bytes: Optional[int] = None) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_thread_pool(), save_upload_file_with_meta, upload_file, filename, max_bytes)


def calculate_checksum(path: str) -> str:
    """Compute SHA256 checksum for a file path."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_mime_type(path: str) -> Optional[str]:
    """Detect mime type by extension or fallback to mimetypes. If python-magic is available it will be used."""
    try:
        import magic as _magic  # type: ignore
    except Exception:
        mime, _ = mimetypes.guess_type(path)
        return mime
    try:
        m = _magic.Magic(mime=True)
        return m.from_file(path)
    except Exception:
        return None


def list_uploads() -> List[Dict[str, Any]]:
    """Return list of uploaded files with basic metadata."""
    results: List[Dict[str, Any]] = []
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        results.append({
            'name': name,
            'path': path,
            'size': stat.st_size,
            'mtime': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
    return results


def delete_upload(name_or_path: str) -> bool:
    """Delete an upload by filename or absolute path, ensuring it's within UPLOAD_DIR."""

    try:
        # Resolve UPLOAD_DIR and the target path
        upload_dir_resolved = Path(UPLOAD_DIR).resolve()
        
        # If name_or_path is absolute, ensure it's within UPLOAD_DIR
        if os.path.isabs(name_or_path):
            target_path = Path(name_or_path)
        else:
            # Construct path relative to UPLOAD_DIR
            target_path = Path(UPLOAD_DIR) / name_or_path

        target_resolved = target_path.resolve()

        # 1. Ensure the resolved target starts with the resolved UPLOAD_DIR path
        if not target_resolved.is_relative_to(upload_dir_resolved):
            logger.warning("Attempt to delete file outside UPLOAD_DIR: %s", name_or_path)
            return False

        # 2. Verify the target is a regular file and not a directory or symlink pointing outside
        if not target_resolved.is_file():
            logger.warning("Attempt to delete non-file or invalid file type: %s", name_or_path)
            return False

        # If all checks pass, remove the file
        os.remove(str(target_resolved))
        logger.info("Successfully deleted upload: %s", name_or_path)
        return True
    except FileNotFoundError:
        logger.info("Attempted to delete non-existent file: %s", name_or_path)
        return False
    except Exception:
        logger.exception("Failed to delete upload: %s", name_or_path)
    return False


def cleanup_old_uploads(days: int = 7) -> int:
    """Remove uploads older than `days`. Returns number of files removed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    removed = 0
    for info in list_uploads():
        mtime = datetime.fromisoformat(info['mtime'])
        if mtime < cutoff:
            if delete_upload(info['path']):
                removed += 1
    logger.info("cleanup_old_uploads removed %d files older than %d days", removed, days)
    return removed


def read_file_content(file_path: str) -> Optional[str]:
    """
    Reads the content of a file, attempting to detect encoding if necessary.
    Handles various path edge-cases and raises FileReadError for issues.
    """
    normalized_path = Path(file_path).resolve()

    if not normalized_path.exists():
        raise FileReadError(f"File not found: {file_path}")
    if not normalized_path.is_file():
        raise FileReadError(f"Path is not a file: {file_path}")
    if normalized_path.stat().st_size == 0:
        logger.info("Attempted to read empty file: %s", file_path)
        return "" # Return empty string for empty files

    try:
        # Try reading with UTF-8 first (most common)
        with open(normalized_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s, attempting encoding detection.", file_path)
        
        detected_encoding = None
        if chardet:
            with open(normalized_path, 'rb') as f:
                raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            
            if encoding and confidence > 0.8:
                detected_encoding = encoding
                logger.info("Detected encoding for %s: %s (confidence: %.2f)", file_path, encoding, confidence)
            else:
                logger.warning("Chardet detection for %s was inconclusive (encoding: %s, confidence: %.2f), trying common fallbacks.", file_path, encoding, confidence)
        
        if detected_encoding:
            try:
                with open(normalized_path, 'r', encoding=detected_encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError) as e:
                raise FileReadError(f"Failed to decode file {file_path} with detected encoding {detected_encoding}: {e}") from e
        
        # Fallback to common encodings if chardet fails or is not available or not confident
        for fallback_encoding in ['utf-16', 'latin-1']:
            try:
                with open(normalized_path, 'r', encoding=fallback_encoding) as f:
                    logger.info("Successfully read %s with fallback encoding: %s", file_path, fallback_encoding)
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue
        
        # If all attempts fail, raise an error
        raise FileReadError(f"Failed to decode file {file_path} with any supported encoding.")
    except (IOError, OSError) as e:
        raise FileReadError(f"Error reading file {file_path}: {e}") from e
