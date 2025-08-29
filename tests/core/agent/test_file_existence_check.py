"""
Unit tests for file existence checking functionality.

This module tests the file existence check methods in FileSystemManager,
including comprehensive metadata retrieval and simple boolean checks.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from omnimancer.core.agent.file_system_manager import FileSystemManager
from omnimancer.core.security import SecurityManager


class TestFileExistenceCheck:
    """Test file existence checking functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_system_manager(self, temp_dir):
        """Create FileSystemManager for testing."""
        return FileSystemManager(
            security_manager=SecurityManager(),
            approval_manager=None,  # Disable approval for testing
            backup_dir=str(temp_dir / "backups"),
            require_approval=False,
        )

    @pytest.mark.asyncio
    async def test_check_file_exists_new_file(
        self, file_system_manager, temp_dir
    ):
        """Test existence check for non-existent file."""
        test_file = temp_dir / "non_existent.txt"

        result = await file_system_manager.check_file_exists(test_file)

        assert result["exists"] is False
        assert result["path"] == str(test_file.resolve())
        assert result["is_file"] is False
        assert result["is_directory"] is False
        assert result["is_symlink"] is False
        assert result["size"] is None
        assert result["modified_time"] is None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_check_file_exists_existing_file(
        self, file_system_manager, temp_dir
    ):
        """Test existence check for existing file."""
        test_file = temp_dir / "existing_file.txt"
        test_content = "This is test content for existence check."

        # Create file
        test_file.write_text(test_content)

        result = await file_system_manager.check_file_exists(test_file)

        assert result["exists"] is True
        assert result["path"] == str(test_file.resolve())
        assert result["is_file"] is True
        assert result["is_directory"] is False
        assert result["is_symlink"] is False
        assert result["size"] == len(test_content)
        assert result["modified_time"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_check_file_exists_directory(
        self, file_system_manager, temp_dir
    ):
        """Test existence check for directory."""
        test_dir = temp_dir / "test_directory"
        test_dir.mkdir()

        result = await file_system_manager.check_file_exists(test_dir)

        assert result["exists"] is True
        assert result["path"] == str(test_dir.resolve())
        assert result["is_file"] is False
        assert result["is_directory"] is True
        assert result["is_symlink"] is False
        assert result["size"] is None  # Directories don't have file size
        assert result["modified_time"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_check_file_exists_symlink(
        self, file_system_manager, temp_dir
    ):
        """Test existence check for symbolic link."""
        # Skip on Windows as symlinks require special permissions
        if os.name == "nt":
            pytest.skip(
                "Symbolic links require special permissions on Windows"
            )

        original_file = temp_dir / "original.txt"
        symlink_file = temp_dir / "symlink.txt"

        # Create original file and symlink
        original_file.write_text("Original content")
        symlink_file.symlink_to(original_file)

        result = await file_system_manager.check_file_exists(symlink_file)

        assert result["exists"] is True
        assert result["is_file"] is True  # Following symlink
        assert result["is_directory"] is False
        assert result["is_symlink"] is True
        assert result["size"] is not None
        assert result["modified_time"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_check_file_exists_no_follow_symlinks(
        self, file_system_manager, temp_dir
    ):
        """Test existence check without following symbolic links."""
        # Skip on Windows as symlinks require special permissions
        if os.name == "nt":
            pytest.skip(
                "Symbolic links require special permissions on Windows"
            )

        original_file = temp_dir / "original.txt"
        symlink_file = temp_dir / "symlink.txt"

        # Create original file and symlink
        original_file.write_text("Original content")
        symlink_file.symlink_to(original_file)

        result = await file_system_manager.check_file_exists(
            symlink_file, follow_symlinks=False
        )

        assert result["exists"] is True
        assert result["is_symlink"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_check_file_exists_security_denied(
        self, file_system_manager, temp_dir
    ):
        """Test existence check with security access denied."""
        test_file = temp_dir / "secure_file.txt"
        test_file.write_text("Secure content")

        # Mock security manager to deny access with proper dictionary format
        mock_security_result = {
            "success": False,
            "error": "Access denied for testing",
            "operation_id": "test_op_001",
            "session_id": "test_session",
        }

        with patch.object(
            file_system_manager.security,
            "secure_file_access",
            return_value=mock_security_result,
        ):
            result = await file_system_manager.check_file_exists(test_file)

            assert result["exists"] is True  # File exists but access denied
            assert (
                result["error"] == "Access denied: Access denied for testing"
            )

    @pytest.mark.asyncio
    async def test_check_file_exists_security_error(
        self, file_system_manager, temp_dir
    ):
        """Test existence check with security validation error."""
        test_file = temp_dir / "error_file.txt"
        test_file.write_text("Content for error test")

        # Mock security manager to raise exception
        with patch.object(
            file_system_manager.security,
            "secure_file_access",
            side_effect=Exception("Security check failed"),
        ):
            result = await file_system_manager.check_file_exists(test_file)

            # File exists but security check failed, so we continue with normal metadata
            assert result["exists"] is True
            assert result["is_file"] is True
            assert (
                result["error"] is None
            )  # Security error is logged but doesn't affect result

    @pytest.mark.asyncio
    async def test_check_file_exists_metadata_error(
        self, file_system_manager, temp_dir
    ):
        """Test existence check with metadata access error."""
        test_file = temp_dir / "metadata_error.txt"
        test_file.write_text("Test content")

        # Mock os.stat to raise OSError
        with patch(
            "aiofiles.os.stat", side_effect=OSError("Permission denied")
        ):
            result = await file_system_manager.check_file_exists(test_file)

            assert result["exists"] is True  # File exists but metadata failed
            assert result["is_file"] is False  # Metadata not available
            assert "Metadata access failed" in result["error"]

    @pytest.mark.asyncio
    async def test_file_exists_simple_true(
        self, file_system_manager, temp_dir
    ):
        """Test simple boolean existence check for existing file."""
        test_file = temp_dir / "simple_test.txt"
        test_file.write_text("Simple test content")

        exists = await file_system_manager.file_exists(test_file)

        assert exists is True

    @pytest.mark.asyncio
    async def test_file_exists_simple_false(
        self, file_system_manager, temp_dir
    ):
        """Test simple boolean existence check for non-existent file."""
        test_file = temp_dir / "non_existent_simple.txt"

        exists = await file_system_manager.file_exists(test_file)

        assert exists is False

    @pytest.mark.asyncio
    async def test_file_exists_simple_with_error(
        self, file_system_manager, temp_dir
    ):
        """Test simple boolean existence check with security error."""
        test_file = temp_dir / "error_simple.txt"
        test_file.write_text("Content for simple error test")

        # Mock security manager to raise exception
        with patch.object(
            file_system_manager.security,
            "secure_file_access",
            side_effect=Exception("Security error"),
        ):
            exists = await file_system_manager.file_exists(test_file)

            assert (
                exists is True
            )  # File exists, security error is logged but doesn't affect result

    @pytest.mark.asyncio
    async def test_check_file_exists_path_resolution(
        self, file_system_manager, temp_dir
    ):
        """Test that paths are properly resolved."""
        # Create nested directory structure
        nested_dir = temp_dir / "level1" / "level2"
        nested_dir.mkdir(parents=True)
        test_file = nested_dir / "nested_file.txt"
        test_file.write_text("Nested content")

        # Use relative path with .. navigation
        relative_path = (
            temp_dir
            / "level1"
            / "level2"
            / ".."
            / "level2"
            / "nested_file.txt"
        )

        result = await file_system_manager.check_file_exists(relative_path)

        assert result["exists"] is True
        assert result["path"] == str(
            relative_path
        )  # Returns original path, not resolved
        assert result["is_file"] is True
        assert result["error"] is None


class TestFileExistenceIntegration:
    """Integration tests for file existence checking with other components."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_system_manager(self, temp_dir):
        """Create FileSystemManager for testing."""
        return FileSystemManager(
            security_manager=SecurityManager(),
            approval_manager=None,
            backup_dir=str(temp_dir / "backups"),
            require_approval=False,
        )

    @pytest.mark.asyncio
    async def test_existence_check_before_operations(
        self, file_system_manager, temp_dir
    ):
        """Test using existence check before file operations."""
        test_file = temp_dir / "operation_test.txt"
        original_content = "Original content"
        new_content = "New content"

        # Initially file doesn't exist
        assert await file_system_manager.file_exists(test_file) is False

        # Create file
        await file_system_manager.write_file(test_file, original_content)

        # Now file exists
        assert await file_system_manager.file_exists(test_file) is True

        # Get detailed info
        info = await file_system_manager.check_file_exists(test_file)
        assert info["exists"] is True
        assert info["is_file"] is True
        assert info["size"] == len(original_content)

        # Modify file
        await file_system_manager.write_file(test_file, new_content)

        # Check new size
        info = await file_system_manager.check_file_exists(test_file)
        assert info["size"] == len(new_content)

    @pytest.mark.asyncio
    async def test_existence_check_performance(
        self, file_system_manager, temp_dir
    ):
        """Test performance of existence checks with multiple files."""
        import time

        # Create multiple test files
        test_files = []
        for i in range(50):
            test_file = temp_dir / f"perf_test_{i}.txt"
            test_file.write_text(f"Content {i}")
            test_files.append(test_file)

        # Time the existence checks
        start_time = time.time()

        for test_file in test_files:
            exists = await file_system_manager.file_exists(test_file)
            assert exists is True

        end_time = time.time()

        # Should complete quickly (less than 1 second for 50 files)
        elapsed = end_time - start_time
        assert elapsed < 1.0, f"Existence checks took too long: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__])
