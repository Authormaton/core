import pytest
import os
import uuid
from unittest.mock import patch, mock_open, MagicMock, ANY
from io import BytesIO
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib # Added import
import asyncio # Added import
import sys # Added import

# Adjust the import path based on your project structure
from services.file_service import (
    save_upload_file,
    save_upload_file_with_meta,
    save_upload_file_async,
    save_upload_file_with_meta_async,
    calculate_checksum,
    detect_mime_type,
    list_uploads,
    delete_upload,
    cleanup_old_uploads,
    UPLOAD_DIR,
    DEFAULT_MAX_UPLOAD_BYTES,
)
from services.exceptions import DocumentSaveError

# Mock UPLOAD_DIR for testing
@pytest.fixture(autouse=True)
def mock_upload_dir(tmp_path):
    with patch('services.file_service.UPLOAD_DIR', str(tmp_path)):
        yield tmp_path

@pytest.fixture
def mock_file_content():
    return b"This is some test content for the file."

@pytest.fixture
def mock_upload_file(mock_file_content):
    mock_file = MagicMock()
    mock_file.file = BytesIO(mock_file_content)
    return mock_file

class TestFileService:

    # Helper for calculating sha256 from bytes
    @staticmethod # Added staticmethod decorator
    def calculate_checksum_from_bytes(data: bytes) -> str:
        h = hashlib.sha256()
        h.update(data)
        return h.hexdigest()

    # Tests for save_upload_file
    @patch('os.replace')
    @patch('os.chmod')
    @patch('tempfile.NamedTemporaryFile')
    @patch('uuid.uuid4', return_value=MagicMock(hex='test_uuid'))
    def test_save_upload_file_success(self, mock_uuid, mock_tempfile, mock_chmod, mock_replace, mock_upload_dir, mock_upload_file):
        mock_tempfile_instance = MagicMock()
        mock_tempfile_instance.name = str(mock_upload_dir / "temp_file")
        mock_tempfile.return_value.__enter__.return_value = mock_tempfile_instance

        filename = "test_document.txt"
        expected_path = os.path.join(mock_upload_dir, f"test_uuid.txt")

        path = save_upload_file(mock_upload_file, filename)

        mock_tempfile_instance.write.assert_called_once_with(mock_upload_file.file.getvalue())
        mock_replace.assert_called_once_with(mock_tempfile_instance.name, expected_path)
        mock_chmod.assert_called_once_with(expected_path, 0o600)
        assert path == expected_path

    @patch('os.replace')
    @patch('os.chmod')
    @patch('tempfile.NamedTemporaryFile')
    @patch('uuid.uuid4', return_value=MagicMock(hex='test_uuid'))
    def test_save_upload_file_exceeds_max_bytes(self, mock_uuid, mock_tempfile, mock_chmod, mock_replace, mock_upload_dir):
        mock_tempfile_instance = MagicMock()
        mock_tempfile_instance.name = str(mock_upload_dir / "temp_file")
        mock_tempfile.return_value.__enter__.return_value = mock_tempfile_instance

        large_content = b"a" * (DEFAULT_MAX_UPLOAD_BYTES + 1)
        mock_upload_file = MagicMock()
        mock_upload_file.file = BytesIO(large_content)

        filename = "large_file.txt"

        with pytest.raises(DocumentSaveError, match=f"Uploaded file '{filename}' exceeds maximum allowed size of {DEFAULT_MAX_UPLOAD_BYTES} bytes."):
            save_upload_file(mock_upload_file, filename)

        mock_tempfile_instance.write.assert_called() # Should write some chunks before failing
        mock_replace.assert_not_called()
        mock_chmod.assert_not_called()

    @pytest.mark.parametrize("invalid_filename", ["", ".", "..", "/etc/passwd", "../../../evil.txt", "file/with/slash.txt"])
    def test_save_upload_file_invalid_filename(self, invalid_filename, mock_upload_file):
        with pytest.raises(DocumentSaveError, match="Invalid filename."):
            save_upload_file(mock_upload_file, invalid_filename)

    @patch('os.replace')
    @patch('os.chmod')
    @patch('tempfile.NamedTemporaryFile')
    @patch('uuid.uuid4', return_value=MagicMock(hex='test_uuid'))
    def test_save_upload_file_os_error_during_save(self, mock_uuid, mock_tempfile, mock_chmod, mock_replace, mock_upload_dir, mock_upload_file):
        mock_tempfile_instance = MagicMock()
        mock_tempfile_instance.name = str(mock_upload_dir / "temp_file")
        mock_tempfile.return_value.__enter__.return_value = mock_tempfile_instance
        mock_replace.side_effect = OSError("Disk full")

        filename = "test_document.txt"

        with pytest.raises(DocumentSaveError, match="Failed to save file securely."):
            save_upload_file(mock_upload_file, filename)

        mock_tempfile_instance.write.assert_called_once()
        mock_replace.assert_called_once()
        mock_chmod.assert_not_called()

    # Tests for save_upload_file_with_meta
    @patch('os.replace')
    @patch('os.chmod')
    @patch('tempfile.NamedTemporaryFile')
    @patch('uuid.uuid4', return_value=MagicMock(hex='test_uuid'))
    @patch('services.file_service.datetime')
    def test_save_upload_file_with_meta_success(self, mock_datetime, mock_uuid, mock_tempfile, mock_chmod, mock_replace, mock_upload_dir, mock_upload_file, mock_file_content):
        mock_tempfile_instance = MagicMock()
        mock_tempfile_instance.name = str(mock_upload_dir / "temp_file")
        mock_tempfile.return_value.__enter__.return_value = mock_tempfile_instance

        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.timezone = timezone # Ensure timezone is accessible

        filename = "test_document.txt"
        expected_path = os.path.join(mock_upload_dir, f"test_uuid.txt")
        expected_sha256 = self.calculate_checksum_from_bytes(mock_file_content)

        meta = save_upload_file_with_meta(mock_upload_file, filename)

        mock_tempfile_instance.write.assert_called_once_with(mock_upload_file.file.getvalue())
        mock_replace.assert_called_once_with(mock_tempfile_instance.name, expected_path)
        mock_chmod.assert_called_once_with(expected_path, 0o600)
        assert meta['path'] == expected_path
        assert meta['original_name'] == filename
        assert meta['size'] == len(mock_file_content)
        assert meta['sha256'] == expected_sha256
        assert meta['mime_type'] == 'text/plain' # mimetypes.guess_type for .txt
        assert meta['saved_at'] == mock_now.isoformat()

    @patch('os.replace')
    @patch('os.chmod')
    @patch('tempfile.NamedTemporaryFile')
    @patch('uuid.uuid4', return_value=MagicMock(hex='test_uuid'))
    def test_save_upload_file_with_meta_exceeds_max_bytes(self, mock_uuid, mock_tempfile, mock_chmod, mock_replace, mock_upload_dir):
        mock_tempfile_instance = MagicMock()
        mock_tempfile_instance.name = str(mock_upload_dir / "temp_file")
        mock_tempfile.return_value.__enter__.return_value = mock_tempfile_instance

        large_content = b"a" * (DEFAULT_MAX_UPLOAD_BYTES + 1)
        mock_upload_file = MagicMock()
        mock_upload_file.file = BytesIO(large_content)

        filename = "large_file.txt"

        with pytest.raises(DocumentSaveError, match=f"Uploaded file '{filename}' exceeds maximum allowed size of {DEFAULT_MAX_UPLOAD_BYTES} bytes."):
            save_upload_file_with_meta(mock_upload_file, filename)

        mock_tempfile_instance.write.assert_called()
        mock_replace.assert_not_called()
        mock_chmod.assert_not_called()

    @pytest.mark.parametrize("invalid_filename", ["", ".", "..", "/etc/passwd", "../../../evil.txt", "file/with/slash.txt"])
    def test_save_upload_file_with_meta_invalid_filename(self, invalid_filename, mock_upload_file):
        with pytest.raises(DocumentSaveError, match="Invalid filename."):
            save_upload_file_with_meta(mock_upload_file, invalid_filename)

    # Helper for calculating sha256 from bytes
    def calculate_checksum_from_bytes(self, data: bytes) -> str:
        h = hashlib.sha256()
        h.update(data)
        return h.hexdigest()

    # Tests for calculate_checksum
    def test_calculate_checksum_success(self, mock_upload_dir):
        file_path = mock_upload_dir / "checksum_test.txt"
        content = b"This is content for checksum."
        file_path.write_bytes(content)
        expected_checksum = self.calculate_checksum_from_bytes(content)
        assert calculate_checksum(str(file_path)) == expected_checksum

    def test_calculate_checksum_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            calculate_checksum("/non/existent/file.txt")

    # Tests for detect_mime_type
    @patch('mimetypes.guess_type', return_value=('text/plain', None))
    @patch.dict(sys.modules, {'magic': None}) # Simulate magic not being available
    def test_detect_mime_type_success_mimetypes(self, mock_guess_type, mock_upload_dir):
        file_path = mock_upload_dir / "test.txt"
        file_path.touch()
        assert detect_mime_type(str(file_path)) == 'text/plain'
        mock_guess_type.assert_called_once_with(str(file_path))

    @patch('mimetypes.guess_type', return_value=(None, None))
    @patch.dict(sys.modules, {'magic': None}) # Simulate magic not being available
    def test_detect_mime_type_unknown_type(self, mock_guess_type, mock_upload_dir):
        file_path = mock_upload_dir / "test.xyz"
        file_path.touch()
        assert detect_mime_type(str(file_path)) is None
        mock_guess_type.assert_called_once_with(str(file_path))

    @patch('mimetypes.guess_type', return_value=('application/pdf', None))
    @patch.dict(sys.modules, {'magic': None}) # Simulate python-magic not being available
    def test_detect_mime_type_no_magic_fallback_mimetypes(self, mock_guess_type, mock_upload_dir):
        file_path = mock_upload_dir / "document.pdf"
        file_path.touch()
        assert detect_mime_type(str(file_path)) == 'application/pdf'
        mock_guess_type.assert_called_once_with(str(file_path))

    @patch('mimetypes.guess_type', return_value=('application/x-python-code', None))
    @patch.dict(sys.modules, {'magic': MagicMock()}) # Simulate magic being available
    def test_detect_mime_type_with_magic(self, mock_guess_type, mock_upload_dir):
        mock_magic_instance = MagicMock()
        sys.modules['magic'].Magic.return_value = mock_magic_instance
        mock_magic_instance.from_file.return_value = 'application/x-python-code'

        file_path = mock_upload_dir / "script.py"
        file_path.touch()
        assert detect_mime_type(str(file_path)) == 'application/x-python-code'
        mock_magic_instance.from_file.assert_called_once_with(str(file_path))

    # Tests for list_uploads
    def test_list_uploads_empty_dir(self, mock_upload_dir):
        assert list_uploads() == []

    @patch('os.listdir')
    @patch('os.path.isfile')
    def test_list_uploads_multiple_files(self, mock_isfile, mock_listdir, mock_upload_dir):
        file1_name = "file1.txt"
        file2_name = "file2.pdf"
        dir_name = "subdir"

        file1_path = mock_upload_dir / file1_name
        file2_path = mock_upload_dir / file2_name
        dir_path = mock_upload_dir / dir_name

        file1_path.write_text("content1")
        file2_path.write_text("content2")
        dir_path.mkdir()

        mock_listdir.return_value = [file1_name, file2_name, dir_name]
        mock_isfile.side_effect = lambda p: Path(p).name in [file1_name, file2_name]

        uploads = list_uploads()
        assert len(uploads) == 2
        # Sort by name for consistent assertion
        uploads.sort(key=lambda x: x['name'])

        # Since os.stat is not mocked, it will read actual file stats
        stat_file1 = os.stat(file1_path)
        stat_file2 = os.stat(file2_path)

        assert uploads[0]['name'] == file1_name
        assert uploads[0]['path'] == str(file1_path)
        assert uploads[0]['size'] == stat_file1.st_size
        assert uploads[0]['mtime'] == datetime.fromtimestamp(stat_file1.st_mtime, timezone.utc).isoformat()

        assert uploads[1]['name'] == file2_name
        assert uploads[1]['path'] == str(file2_path)
        assert uploads[1]['size'] == stat_file2.st_size
        assert uploads[1]['mtime'] == datetime.fromtimestamp(stat_file2.st_mtime, timezone.utc).isoformat()

    # Tests for delete_upload
    @patch('os.remove')
    def test_delete_upload_by_filename_success(self, mock_remove, mock_upload_dir):
        file_path = mock_upload_dir / "to_delete.txt"
        file_path.touch()
        assert delete_upload("to_delete.txt") is True
        mock_remove.assert_called_once_with(str(file_path))

    @patch('os.remove')
    def test_delete_upload_by_absolute_path_success(self, mock_remove, mock_upload_dir):
        file_path = mock_upload_dir / "to_delete_abs.txt"
        file_path.touch()
        assert delete_upload(str(file_path)) is True
        mock_remove.assert_called_once_with(str(file_path))

    @patch('os.remove')
    def test_delete_upload_file_not_found(self, mock_remove, mock_upload_dir):
        assert delete_upload("non_existent.txt") is False
        mock_remove.assert_not_called()

    @patch('os.remove')
    def test_delete_upload_outside_upload_dir(self, mock_remove, mock_upload_dir):
        # Create a file outside the mocked UPLOAD_DIR
        outside_file = Path("/tmp/evil.txt")
        outside_file.touch()
        assert delete_upload(str(outside_file)) is False
        mock_remove.assert_not_called()
        outside_file.unlink() # Clean up

    @patch('os.remove')
    def test_delete_upload_directory(self, mock_remove, mock_upload_dir):
        dir_path = mock_upload_dir / "a_directory"
        dir_path.mkdir()
        assert delete_upload(str(dir_path)) is False
        mock_remove.assert_not_called()

    @patch('os.remove', side_effect=OSError("Permission denied"))
    def test_delete_upload_os_error(self, mock_remove, mock_upload_dir):
        file_path = mock_upload_dir / "permission_denied.txt"
        file_path.touch()
        assert delete_upload(str(file_path)) is False
        mock_remove.assert_called_once_with(str(file_path))

    # Tests for cleanup_old_uploads
    @patch('services.file_service.list_uploads')
    @patch('services.file_service.delete_upload')
    def test_cleanup_old_uploads_no_files(self, mock_delete_upload, mock_list_uploads):
        mock_list_uploads.return_value = []
        assert cleanup_old_uploads(days=7) == 0
        mock_delete_upload.assert_not_called()

    @patch('services.file_service.list_uploads')
    @patch('services.file_service.delete_upload')
    def test_cleanup_old_uploads_some_files_removed(self, mock_delete_upload, mock_list_uploads):
        now = datetime.now(timezone.utc)
        old_file_mtime = (now - timedelta(days=10)).isoformat()
        recent_file_mtime = (now - timedelta(days=1)).isoformat()

        mock_list_uploads.return_value = [
            {'name': 'old_file.txt', 'path': '/path/to/old_file.txt', 'size': 100, 'mtime': old_file_mtime},
            {'name': 'recent_file.txt', 'path': '/path/to/recent_file.txt', 'size': 50, 'mtime': recent_file_mtime},
        ]
        mock_delete_upload.side_effect = [True, False] # Only old_file.txt is successfully deleted

        assert cleanup_old_uploads(days=7) == 1
        mock_delete_upload.assert_called_once_with('/path/to/old_file.txt')

    @patch('services.file_service.list_uploads')
    @patch('services.file_service.delete_upload')
    def test_cleanup_old_uploads_no_files_older_than_days(self, mock_delete_upload, mock_list_uploads):
        now = datetime.now(timezone.utc)
        recent_file_mtime = (now - timedelta(days=1)).isoformat()

        mock_list_uploads.return_value = [
            {'name': 'recent_file1.txt', 'path': '/path/to/recent_file1.txt', 'size': 100, 'mtime': recent_file_mtime},
            {'name': 'recent_file2.txt', 'path': '/path/to/recent_file2.txt', 'size': 50, 'mtime': recent_file_mtime},
        ]
        assert cleanup_old_uploads(days=7) == 0
        mock_delete_upload.assert_not_called()

    # Async tests
    @pytest.mark.asyncio
    @patch('services.file_service.save_upload_file')
    @patch('asyncio.get_running_loop')
    @patch('services.file_service._get_thread_pool') # Added patch
    async def test_save_upload_file_async(self, mock_get_running_loop, mock_save_upload_file, mock_upload_file, event_loop):
        mock_loop = MagicMock()
        mock_get_running_loop.return_value = mock_loop
        mock_save_upload_file.return_value = "/mock/path/file.txt"

        # Create a real Future and set its result
        future = asyncio.Future()
        future.set_result(mock_save_upload_file.return_value)
        mock_loop.run_in_executor.return_value = future

        result = await save_upload_file_async(mock_upload_file, "test.txt")

        mock_get_running_loop.assert_called_once()
        mock_loop.run_in_executor.assert_called_once_with(ANY, mock_save_upload_file, mock_upload_file, "test.txt", None)
        assert result == "/mock/path/file.txt"

    @pytest.mark.asyncio
    @patch('services.file_service.save_upload_file_with_meta')
    @patch('asyncio.get_running_loop')
    @patch('services.file_service._get_thread_pool') # Added patch
    async def test_save_upload_file_with_meta_async(self, mock_get_running_loop, mock_save_upload_file_with_meta, mock_upload_file, event_loop):
        mock_loop = MagicMock()
        mock_get_running_loop.return_value = mock_loop
        mock_save_upload_file_with_meta.return_value = {"path": "/mock/path/file.txt", "size": 100}

        # Create a real Future and set its result
        future = mock_loop.create_future()
        future.set_result(mock_save_upload_file_with_meta.return_value)
        mock_loop.run_in_executor.return_value = future

        result = await save_upload_file_with_meta_async(mock_upload_file, "test.txt")

        mock_get_running_loop.assert_called_once()
        mock_loop.run_in_executor.assert_called_once_with(ANY, mock_save_upload_file_with_meta, mock_upload_file, "test.txt", None)
        assert result == {"path": "/mock/path/file.txt", "size": 100}
