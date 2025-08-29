"""
Comprehensive unit tests for file operations functionality.

This module tests file reading and writing operations, including read-before-write
integration and various file types handling.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from omnimancer.core.agent.file_system_manager import (
    FileSystemManager,
    FileOperationError,
)
from omnimancer.core.security import SecurityManager


class TestFileReadingFunction:
    """Test file reading functionality for Task 31.1."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_system_manager(self, temp_dir):
        """Create FileSystemManager for testing."""
        # Create mock security manager that allows all operations
        mock_security = Mock()
        mock_security.secure_file_access = AsyncMock(
            return_value={"success": True, "message": "Access granted for testing"}
        )

        return FileSystemManager(
            security_manager=mock_security,
            approval_manager=None,
            require_approval=False,
        )

    @pytest.mark.asyncio
    async def test_read_simple_text_file(self, file_system_manager, temp_dir):
        """Test reading a simple text file."""
        test_file = temp_dir / "simple.txt"
        content = "Hello, World!"
        test_file.write_text(content)

        result = await file_system_manager.read_file(test_file)

        assert result == content
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_read_multiline_text_file(self, file_system_manager, temp_dir):
        """Test reading a multiline text file."""
        test_file = temp_dir / "multiline.txt"
        content = """Line 1
Line 2
Line 3 with special chars: !@#$%^&*()
Line 4"""
        test_file.write_text(content)

        result = await file_system_manager.read_file(test_file)

        assert result == content
        assert result.count("\n") == 3

    @pytest.mark.asyncio
    async def test_read_empty_file(self, file_system_manager, temp_dir):
        """Test reading an empty file."""
        test_file = temp_dir / "empty.txt"
        test_file.write_text("")

        result = await file_system_manager.read_file(test_file)

        assert result == ""
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_read_utf8_file(self, file_system_manager, temp_dir):
        """Test reading UTF-8 encoded file with special characters."""
        test_file = temp_dir / "utf8.txt"
        content = "Hello 世界! 🌍 Café naïve résumé"
        test_file.write_text(content, encoding="utf-8")

        result = await file_system_manager.read_file(test_file, encoding="utf-8")

        assert result == content
        assert "世界" in result
        assert "🌍" in result

    @pytest.mark.asyncio
    async def test_read_different_encodings(self, file_system_manager, temp_dir):
        """Test reading files with different encodings."""
        # Test with latin-1 encoding
        test_file = temp_dir / "latin1.txt"
        content = "Café naïve résumé"
        test_file.write_text(content, encoding="latin-1")

        result = await file_system_manager.read_file(test_file, encoding="latin-1")

        assert result == content

    @pytest.mark.asyncio
    async def test_read_binary_file(self, file_system_manager, temp_dir):
        """Test reading binary files."""
        test_file = temp_dir / "binary.bin"
        binary_data = b"\x00\x01\x02\x03\xff\xfe\xfd"
        test_file.write_bytes(binary_data)

        result = await file_system_manager.read_file(test_file, binary=True)

        assert result == binary_data
        assert isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_read_json_file(self, file_system_manager, temp_dir):
        """Test reading JSON file."""
        test_file = temp_dir / "data.json"
        json_content = '{"name": "test", "value": 123, "active": true}'
        test_file.write_text(json_content)

        result = await file_system_manager.read_file(test_file)

        assert result == json_content
        # Verify it's valid JSON by parsing
        import json

        parsed = json.loads(result)
        assert parsed["name"] == "test"
        assert parsed["value"] == 123

    @pytest.mark.asyncio
    async def test_read_python_file(self, file_system_manager, temp_dir):
        """Test reading Python source code file."""
        test_file = temp_dir / "example.py"
        python_content = '''def hello():
    """Say hello."""
    return "Hello, World!"

if __name__ == "__main__":
    print(hello())
'''
        test_file.write_text(python_content)

        result = await file_system_manager.read_file(test_file)

        assert result == python_content
        assert "def hello():" in result
        assert 'if __name__ == "__main__":' in result

    @pytest.mark.asyncio
    async def test_read_xml_file(self, file_system_manager, temp_dir):
        """Test reading XML file."""
        test_file = temp_dir / "data.xml"
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <item id="1">
        <name>Test Item</name>
        <value>123</value>
    </item>
</root>"""
        test_file.write_text(xml_content)

        result = await file_system_manager.read_file(test_file)

        assert result == xml_content
        assert '<?xml version="1.0"' in result
        assert "<root>" in result

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, file_system_manager, temp_dir):
        """Test reading non-existent file raises appropriate error."""
        test_file = temp_dir / "nonexistent.txt"

        with pytest.raises(FileOperationError) as exc_info:
            await file_system_manager.read_file(test_file)

        assert "File not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_read_directory_as_file(self, file_system_manager, temp_dir):
        """Test attempting to read a directory as a file."""
        test_dir = temp_dir / "subdirectory"
        test_dir.mkdir()

        with pytest.raises(FileOperationError):
            await file_system_manager.read_file(test_dir)

    @pytest.mark.asyncio
    async def test_read_file_permission_denied(self, file_system_manager, temp_dir):
        """Test reading file with permission denied."""
        test_file = temp_dir / "restricted.txt"
        test_file.write_text("Secret content")

        # Mock security manager to deny access
        with patch.object(
            file_system_manager.security, "secure_file_access"
        ) as mock_security:
            mock_security.return_value = {
                "success": False,
                "error": "Permission denied",
            }

            with pytest.raises(FileOperationError) as exc_info:
                await file_system_manager.read_file(test_file)

            assert "Security check failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_read_large_file(self, file_system_manager, temp_dir):
        """Test reading a large file that triggers streaming."""
        test_file = temp_dir / "large.txt"
        # Create content larger than default max_file_size_mb (usually 50MB)
        large_content = "A" * 1000 + "\n"  # 1KB per line
        total_content = large_content * 60000  # ~60MB
        test_file.write_text(total_content)

        result = await file_system_manager.read_file(test_file)

        assert len(result) == len(total_content)
        assert result.startswith("A" * 1000)

    @pytest.mark.asyncio
    async def test_read_file_with_special_characters_in_path(
        self, file_system_manager, temp_dir
    ):
        """Test reading file with special characters in filename."""
        # Create file with special characters in name
        test_file = temp_dir / "file with spaces & symbols!.txt"
        content = "Content in file with special name"
        test_file.write_text(content)

        result = await file_system_manager.read_file(test_file)

        assert result == content

    @pytest.mark.asyncio
    async def test_read_symlink_file(self, file_system_manager, temp_dir):
        """Test reading a symbolic link to a file."""
        # Skip on Windows as symlinks require special permissions
        import os

        if os.name == "nt":
            pytest.skip("Symbolic links require special permissions on Windows")

        original_file = temp_dir / "original.txt"
        symlink_file = temp_dir / "symlink.txt"
        content = "Original file content"

        # Create original file and symlink
        original_file.write_text(content)
        symlink_file.symlink_to(original_file)

        result = await file_system_manager.read_file(symlink_file)

        assert result == content

    @pytest.mark.asyncio
    async def test_read_file_encoding_error(self, file_system_manager, temp_dir):
        """Test reading file with wrong encoding raises error."""
        test_file = temp_dir / "encoded.txt"
        # Write with one encoding, try to read with another
        content = "Café naïve résumé"
        test_file.write_text(content, encoding="latin-1")

        # This should work with correct encoding
        result_correct = await file_system_manager.read_file(
            test_file, encoding="latin-1"
        )
        assert result_correct == content

        # This might cause issues with wrong encoding, but should be handled gracefully
        try:
            result_wrong = await file_system_manager.read_file(
                test_file, encoding="ascii"
            )
            # If it doesn't raise an error, the result should still be readable
            assert isinstance(result_wrong, str)
        except FileOperationError:
            # This is also acceptable - encoding errors should be handled
            pass

    @pytest.mark.asyncio
    async def test_read_file_with_null_bytes(self, file_system_manager, temp_dir):
        """Test reading file containing null bytes."""
        test_file = temp_dir / "null_bytes.txt"
        content = "Text with\x00null\x00bytes"
        test_file.write_text(content, encoding="utf-8")

        # Files with null bytes are detected as binary and return bytes
        result = await file_system_manager.read_file(test_file)

        # The file is detected as binary due to null bytes, so result will be bytes
        assert isinstance(result, bytes)
        assert result == content.encode("utf-8")
        assert b"\x00" in result

    @pytest.mark.asyncio
    async def test_read_file_return_type_consistency(
        self, file_system_manager, temp_dir
    ):
        """Test that read_file returns consistent types."""
        # Text file should return string
        text_file = temp_dir / "text.txt"
        text_file.write_text("text content")

        text_result = await file_system_manager.read_file(text_file)
        assert isinstance(text_result, str)

        # Binary file should return bytes
        binary_file = temp_dir / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02")

        binary_result = await file_system_manager.read_file(binary_file, binary=True)
        assert isinstance(binary_result, bytes)


class TestFileReadingEdgeCases:
    """Test edge cases for file reading functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_system_manager(self, temp_dir):
        """Create FileSystemManager for testing."""
        # Create mock security manager that allows all operations
        mock_security = Mock()
        mock_security.secure_file_access = AsyncMock(
            return_value={"success": True, "message": "Access granted for testing"}
        )

        return FileSystemManager(
            security_manager=mock_security,
            approval_manager=None,
            require_approval=False,
        )

    @pytest.mark.asyncio
    async def test_read_file_concurrent_access(self, file_system_manager, temp_dir):
        """Test reading file concurrently from multiple coroutines."""
        test_file = temp_dir / "concurrent.txt"
        content = "Content for concurrent reading test"
        test_file.write_text(content)

        # Read file concurrently from multiple coroutines
        tasks = [file_system_manager.read_file(test_file) for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # All results should be identical
        for result in results:
            assert result == content

    @pytest.mark.asyncio
    async def test_read_file_path_variations(self, file_system_manager, temp_dir):
        """Test reading files with different path representations."""
        test_file = temp_dir / "pathtest.txt"
        content = "Path test content"
        test_file.write_text(content)

        # Test with string path
        result1 = await file_system_manager.read_file(str(test_file))
        assert result1 == content

        # Test with Path object
        result2 = await file_system_manager.read_file(test_file)
        assert result2 == content

        # Results should be identical
        assert result1 == result2


class TestWriteOperationsIntegration:
    """Test read-before-write integration with write operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_system_manager(self, temp_dir):
        """Create FileSystemManager for testing."""
        # Create mock security manager that allows all operations
        mock_security = Mock()
        mock_security.secure_file_access = AsyncMock(
            return_value={"success": True, "message": "Access granted for testing"}
        )

        return FileSystemManager(
            security_manager=mock_security,
            approval_manager=None,
            require_approval=False,
        )

    @pytest.mark.asyncio
    async def test_write_file_with_read_before_write_new_file(
        self, file_system_manager, temp_dir
    ):
        """Test write_file with read_before_write=True for new file."""
        test_file = temp_dir / "new_file.txt"
        content = "This is new content."

        # Mock user callback that approves
        async def mock_callback(review_data):
            assert review_data["file_exists"] is False
            assert review_data["operation"] == "create"
            assert review_data["new_content"] == content
            return {"approved": True}

        result = await file_system_manager.write_file(
            path=test_file,
            content=content,
            read_before_write=True,
            user_review_callback=mock_callback,
        )

        assert result["success"] is True
        assert result["operation"] == "read_before_write"
        assert result["had_existing_content"] is False
        assert result["user_reviewed"] is True
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_file_with_read_before_write_existing_file(
        self, file_system_manager, temp_dir
    ):
        """Test write_file with read_before_write=True for existing file."""
        test_file = temp_dir / "existing_file.txt"
        original_content = "Original content"
        new_content = "Modified content"

        # Create existing file
        test_file.write_text(original_content)

        # Mock user callback that approves
        async def mock_callback(review_data):
            assert review_data["file_exists"] is True
            assert review_data["operation"] == "modify"
            assert review_data["current_content"] == original_content
            assert review_data["new_content"] == new_content
            return {"approved": True}

        result = await file_system_manager.write_file(
            path=test_file,
            content=new_content,
            read_before_write=True,
            user_review_callback=mock_callback,
        )

        assert result["success"] is True
        assert result["operation"] == "read_before_write"
        assert result["had_existing_content"] is True
        assert result["user_reviewed"] is True
        assert test_file.read_text() == new_content

    @pytest.mark.asyncio
    async def test_write_file_without_read_before_write(
        self, file_system_manager, temp_dir
    ):
        """Test write_file with read_before_write=False (default behavior)."""
        test_file = temp_dir / "regular_file.txt"
        content = "Regular write content"

        result = await file_system_manager.write_file(
            path=test_file, content=content, read_before_write=False
        )

        assert result["success"] is True
        # Should NOT have read_before_write operation info
        assert (
            "operation" not in result or result.get("operation") != "read_before_write"
        )
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_file_with_confirmation_and_read_before_write(
        self, file_system_manager, temp_dir
    ):
        """Test write_file_with_confirmation with read_before_write=True."""
        test_file = temp_dir / "confirmation_test.txt"
        original_content = "Original content"
        new_content = "New content"

        # Create existing file
        test_file.write_text(original_content)

        # Mock confirmation callback that approves
        async def confirmation_callback(file_info):
            return {
                "confirmed": True,
                "action": "overwrite",
                "reason": "User approved overwrite",
            }

        # Mock review callback that approves
        async def review_callback(review_data):
            return {"approved": True}

        result = await file_system_manager.write_file_with_confirmation(
            path=test_file,
            content=new_content,
            confirmation_callback=confirmation_callback,
            read_before_write=True,
            user_review_callback=review_callback,
        )

        assert result["success"] is True
        assert result["operation"] == "read_before_write"
        assert result["file_existed_before"] is True
        assert result["confirmation_requested"] is True
        assert test_file.read_text() == new_content

    @pytest.mark.asyncio
    async def test_write_file_user_rejection_in_read_before_write(
        self, file_system_manager, temp_dir
    ):
        """Test write_file with read_before_write when user rejects."""
        test_file = temp_dir / "rejection_test.txt"
        original_content = "Original content"
        new_content = "Should not be written"

        # Create existing file
        test_file.write_text(original_content)

        # Mock user callback that rejects
        async def mock_callback(review_data):
            return {"approved": False, "reason": "User does not want this change"}

        result = await file_system_manager.write_file(
            path=test_file,
            content=new_content,
            read_before_write=True,
            user_review_callback=mock_callback,
        )

        assert result["success"] is False
        assert "User rejected" in result["error"]
        # File should remain unchanged
        assert test_file.read_text() == original_content


if __name__ == "__main__":
    pytest.main([__file__])
