"""
Comprehensive integration tests for read-before-write logic (Task 31.5).

This module tests the complete read-before-write functionality end-to-end,
including all components working together: file reading, UI interactions,
error handling, and write operations.
"""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from omnimancer.core.agent.file_system_manager import FileSystemManager
from omnimancer.core.agent.read_before_write_ui import (
    ReadBeforeWriteUI,
    create_review_callback,
)
from omnimancer.core.agent.read_before_write_errors import (
    ReadBeforeWriteErrorHandler,
    ReadBeforeWriteError,
    ReadBeforeWriteErrorType,
    RecoveryStrategy,
    FileReadError,
    FileWriteError,
    CallbackError,
    UserRejectionError,
    ContentValidationError,
)


class TestReadBeforeWriteLogicIntegration:
    """Test complete read-before-write logic integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_security_manager(self):
        """Create mock security manager."""
        mock_security = Mock()
        mock_security.secure_file_access = AsyncMock(
            return_value={"success": True, "message": "Access granted for testing"}
        )
        return mock_security

    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return ReadBeforeWriteErrorHandler(enable_recovery=True, log_errors=True)

    @pytest.fixture
    def file_system_manager(self, temp_dir, mock_security_manager, error_handler):
        """Create FileSystemManager with full configuration for testing."""
        return FileSystemManager(
            security_manager=mock_security_manager,
            approval_manager=None,
            require_approval=False,
            error_handler=error_handler,
        )

    @pytest.mark.asyncio
    async def test_complete_read_before_write_workflow_new_file(
        self, file_system_manager, temp_dir
    ):
        """Test complete workflow for creating a new file with read-before-write."""
        test_file = temp_dir / "new_workflow_test.txt"
        content = "This is new content for workflow test."

        # Mock user callback that approves
        async def approval_callback(review_data):
            # Verify review data structure
            assert "file_path" in review_data
            assert "file_exists" in review_data
            assert "operation" in review_data
            assert "new_content" in review_data
            assert review_data["file_exists"] is False
            assert review_data["operation"] == "create"
            assert review_data["new_content"] == content

            return {"approved": True}

        # Test the complete workflow
        result = await file_system_manager.write_file(
            path=test_file,
            content=content,
            read_before_write=True,
            user_review_callback=approval_callback,
        )

        # Verify successful completion
        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == content

        # Verify file system manager called read_before_write
        assert "had_existing_content" in result
        assert result["had_existing_content"] is False

    @pytest.mark.asyncio
    async def test_complete_read_before_write_workflow_existing_file(
        self, file_system_manager, temp_dir
    ):
        """Test complete workflow for modifying an existing file with read-before-write."""
        test_file = temp_dir / "existing_workflow_test.txt"
        original_content = "Original content for workflow test."
        new_content = "Updated content for workflow test."

        # Create existing file
        test_file.write_text(original_content)

        # Mock user callback that approves
        async def approval_callback(review_data):
            # Verify review data structure for existing file
            assert review_data["file_exists"] is True
            assert review_data["operation"] == "modify"
            assert review_data["current_content"] == original_content
            assert review_data["new_content"] == new_content
            assert "diff" in review_data

            return {"approved": True}

        # Test the complete workflow
        result = await file_system_manager.write_file(
            path=test_file,
            content=new_content,
            read_before_write=True,
            user_review_callback=approval_callback,
        )

        # Verify successful completion
        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == new_content

        # Verify read-before-write was used
        assert result["had_existing_content"] is True

    @pytest.mark.asyncio
    async def test_read_before_write_with_user_content_modification(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write where user modifies content before approval."""
        test_file = temp_dir / "user_modify_test.txt"
        original_content = "Original content."
        proposed_content = "Proposed content."
        user_modified_content = "User modified content."

        # Create existing file
        test_file.write_text(original_content)

        # Mock user callback that modifies content
        async def modification_callback(review_data):
            assert review_data["current_content"] == original_content
            assert review_data["new_content"] == proposed_content

            return {"approved": True, "modified_content": user_modified_content}

        # Test the workflow with user modification
        result = await file_system_manager.write_file(
            path=test_file,
            content=proposed_content,
            read_before_write=True,
            user_review_callback=modification_callback,
        )

        # Verify user's modified content was used
        assert result["success"] is True
        assert test_file.read_text() == user_modified_content

    @pytest.mark.asyncio
    async def test_read_before_write_with_user_rejection(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write where user rejects the changes."""
        test_file = temp_dir / "user_reject_test.txt"
        original_content = "Original content that should remain."
        new_content = "Content that will be rejected."

        # Create existing file
        test_file.write_text(original_content)

        # Mock user callback that rejects
        async def rejection_callback(review_data):
            return {"approved": False, "reason": "User does not want these changes"}

        # Test the workflow with user rejection
        result = await file_system_manager.write_file(
            path=test_file,
            content=new_content,
            read_before_write=True,
            user_review_callback=rejection_callback,
        )

        # Verify rejection was handled correctly
        assert result["success"] is False
        assert "User rejected" in result["error"]
        assert test_file.read_text() == original_content  # Original content preserved

    @pytest.mark.asyncio
    async def test_read_before_write_fallback_on_callback_error(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write falls back to regular write when callback fails."""
        test_file = temp_dir / "callback_error_test.txt"
        content = "Content when callback fails."

        # Mock user callback that raises exception
        async def failing_callback(review_data):
            raise RuntimeError("Callback failed for testing")

        # Test fallback behavior
        result = await file_system_manager.write_file(
            path=test_file,
            content=content,
            read_before_write=True,
            user_review_callback=failing_callback,
        )

        # Should succeed via fallback to regular write
        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_read_before_write_with_binary_files(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write with binary files."""
        test_file = temp_dir / "binary_test.bin"
        binary_content = b"\x00\x01\x02\x03\xff\xfe\xfd"

        # Mock user callback that approves binary content
        async def binary_approval_callback(review_data):
            # Binary content should be handled properly
            assert review_data["file_exists"] is False
            assert review_data["operation"] == "create"
            assert isinstance(review_data["new_content"], bytes)

            return {"approved": True}

        # Test binary file handling
        result = await file_system_manager.write_file(
            path=test_file,
            content=binary_content,
            read_before_write=True,
            user_review_callback=binary_approval_callback,
        )

        # Verify binary file was created correctly
        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_bytes() == binary_content

    @pytest.mark.asyncio
    async def test_read_before_write_with_large_files(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write with large files."""
        test_file = temp_dir / "large_file_test.txt"
        # Create content just under the size limit (50MB default)
        large_content = "A" * (45 * 1024 * 1024)  # 45MB

        # Mock user callback that approves large content
        async def large_file_callback(review_data):
            assert len(review_data["new_content"]) == len(large_content)
            return {"approved": True}

        # Test large file handling
        result = await file_system_manager.write_file(
            path=test_file,
            content=large_content,
            read_before_write=True,
            user_review_callback=large_file_callback,
        )

        # Verify large file was handled correctly
        assert result["success"] is True
        assert test_file.exists()
        assert len(test_file.read_text()) == len(large_content)

    @pytest.mark.asyncio
    async def test_read_before_write_content_validation_failure(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write with content validation failure."""
        test_file = temp_dir / "validation_test.txt"
        # Create content that exceeds size limit (50MB default)
        huge_content = "A" * (51 * 1024 * 1024)  # 51MB

        # Mock user callback that tries to approve oversized content
        async def oversized_callback(review_data):
            return {"approved": True, "modified_content": huge_content}

        # Test content validation
        result = await file_system_manager.write_file(
            path=test_file,
            content="Small initial content",
            read_before_write=True,
            user_review_callback=oversized_callback,
        )

        # Should fail due to content validation
        assert result["success"] is False
        assert "error_details" in result
        assert (
            result["error_details"]["error"]["error_type"] == "content_validation_error"
        )

    @pytest.mark.asyncio
    async def test_read_before_write_with_encoding_handling(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write with different encodings."""
        test_file = temp_dir / "encoding_test.txt"
        unicode_content = "Hello 世界! 🌍 Café naïve résumé"

        # Create file with UTF-8 encoding
        test_file.write_text(unicode_content, encoding="utf-8")

        # Mock user callback that handles unicode content
        async def unicode_callback(review_data):
            assert "世界" in review_data["current_content"]
            assert "🌍" in review_data["current_content"]
            return {"approved": True}

        # Test unicode/encoding handling
        new_content = unicode_content + " - Updated"
        result = await file_system_manager.write_file(
            path=test_file,
            content=new_content,
            encoding="utf-8",
            read_before_write=True,
            user_review_callback=unicode_callback,
        )

        # Verify unicode content was handled correctly
        assert result["success"] is True
        assert test_file.read_text(encoding="utf-8") == new_content

    @pytest.mark.asyncio
    async def test_read_before_write_diff_generation(
        self, file_system_manager, temp_dir
    ):
        """Test that read-before-write generates proper diffs."""
        test_file = temp_dir / "diff_test.txt"
        original_content = """Line 1
Line 2
Line 3
Line 4"""

        new_content = """Line 1
Modified Line 2
Line 3
New Line 4
Line 5"""

        # Create existing file
        test_file.write_text(original_content)

        # Mock callback that examines the diff
        async def diff_examination_callback(review_data):
            diff = review_data.get("diff", "")
            # Verify diff contains expected changes
            assert "- Line 2" in diff or "-Line 2" in diff
            assert "+ Modified Line 2" in diff or "+Modified Line 2" in diff
            assert "+ New Line 4" in diff or "+New Line 4" in diff
            return {"approved": True}

        # Test diff generation
        result = await file_system_manager.write_file(
            path=test_file,
            content=new_content,
            read_before_write=True,
            user_review_callback=diff_examination_callback,
        )

        # Verify successful with proper diff
        assert result["success"] is True
        assert test_file.read_text() == new_content

    @pytest.mark.asyncio
    async def test_read_before_write_with_atomic_operations(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write with atomic operations enabled."""
        test_file = temp_dir / "atomic_test.txt"
        original_content = "Original atomic content"
        new_content = "New atomic content"

        # Create existing file
        test_file.write_text(original_content)

        # Mock callback that approves
        async def atomic_callback(review_data):
            return {"approved": True}

        # Test with atomic operations
        result = await file_system_manager.write_file(
            path=test_file,
            content=new_content,
            atomic=True,
            read_before_write=True,
            user_review_callback=atomic_callback,
        )

        # Verify atomic operation succeeded
        assert result["success"] is True
        assert test_file.read_text() == new_content
        assert result.get("atomic", False)  # Should indicate atomic operation was used

    @pytest.mark.asyncio
    async def test_read_before_write_with_backup_enabled(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write with backup creation."""
        test_file = temp_dir / "backup_test.txt"
        original_content = "Original content for backup"
        new_content = "New content after backup"

        # Create existing file
        test_file.write_text(original_content)

        # Mock callback that approves
        async def backup_callback(review_data):
            return {"approved": True}

        # Test with backup enabled
        result = await file_system_manager.write_file(
            path=test_file,
            content=new_content,
            backup=True,
            read_before_write=True,
            user_review_callback=backup_callback,
        )

        # Verify backup was created and operation succeeded
        assert result["success"] is True
        assert test_file.read_text() == new_content

        # Check for backup file
        assert "backup_path" in result
        backup_path = Path(result["backup_path"])
        assert backup_path.exists()
        assert backup_path.read_text() == original_content

    @pytest.mark.asyncio
    async def test_read_before_write_error_statistics_tracking(
        self, file_system_manager, temp_dir
    ):
        """Test that error statistics are tracked correctly during read-before-write."""
        test_file = temp_dir / "stats_test.txt"

        # Generate several errors with fallback recovery
        async def failing_callback(review_data):
            raise RuntimeError("Callback failed for statistics test")

        # Run multiple operations that will generate errors
        for i in range(3):
            result = await file_system_manager.write_file(
                path=test_file,
                content=f"Content {i}",
                read_before_write=True,
                user_review_callback=failing_callback,
            )
            # Should succeed due to fallback
            assert result["success"] is True

        # Check error statistics
        if (
            hasattr(file_system_manager, "error_handler")
            and file_system_manager.error_handler
        ):
            stats = file_system_manager.error_handler.get_error_statistics()
            assert stats["total_errors"] >= 3
            assert "callback_error" in stats["error_types"]
            assert stats["error_types"]["callback_error"] >= 3


class TestReadBeforeWriteUIIntegration:
    """Test read-before-write UI component integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_create_review_callback_integration(self, temp_dir):
        """Test that create_review_callback produces working callback."""
        # Create mock console to avoid actual terminal output
        mock_console = Mock()
        mock_console.print = Mock()

        # Create callback using utility function
        review_callback = create_review_callback(console=mock_console)

        # Verify callback is callable
        assert callable(review_callback)

        # Test with mock input (simulating user approval)
        review_data = {
            "file_path": str(temp_dir / "test.txt"),
            "file_exists": False,
            "operation": "create",
            "new_content": "Test content",
            "encoding": "utf-8",
        }

        # Mock the user input to return approval
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask", return_value="1"
        ):
            result = await review_callback(review_data)

        # Verify callback returns expected structure
        assert isinstance(result, dict)
        assert "approved" in result

    @pytest.mark.asyncio
    async def test_ui_error_handling_in_callback(self, temp_dir):
        """Test UI error handling during callback execution."""
        # Create callback that will encounter UI errors
        mock_console = Mock()
        mock_console.print = Mock(side_effect=Exception("UI error"))

        review_callback = create_review_callback(console=mock_console)

        review_data = {
            "file_path": str(temp_dir / "test.txt"),
            "file_exists": False,
            "operation": "create",
            "new_content": "Test content",
            "encoding": "utf-8",
        }

        # Test that UI errors are handled gracefully
        result = await review_callback(review_data)

        # Should return rejection due to UI error
        assert result["approved"] is False
        assert "UI error" in result["reason"]


class TestReadBeforeWriteEdgeCases:
    """Test edge cases and boundary conditions for read-before-write."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_system_manager(self, temp_dir):
        """Create FileSystemManager for edge case testing."""
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
    async def test_read_before_write_with_permission_errors(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write behavior with permission errors."""
        test_file = temp_dir / "permission_test.txt"
        test_file.write_text("Original content")

        # Mock security manager to deny permission
        with patch.object(
            file_system_manager.security, "secure_file_access"
        ) as mock_security:
            mock_security.return_value = {
                "success": False,
                "error": "Permission denied",
            }

            async def approval_callback(review_data):
                return {"approved": True}

            # Should fail due to permission error
            result = await file_system_manager.write_file(
                path=test_file,
                content="New content",
                read_before_write=True,
                user_review_callback=approval_callback,
            )

            assert result["success"] is False
            assert "Permission denied" in str(result.get("error", ""))

    @pytest.mark.asyncio
    async def test_read_before_write_with_concurrent_access(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write with concurrent file access."""
        test_file = temp_dir / "concurrent_test.txt"
        test_file.write_text("Original content")

        async def approval_callback(review_data):
            return {"approved": True}

        # Run multiple read-before-write operations concurrently
        tasks = []
        for i in range(3):
            task = file_system_manager.write_file(
                path=test_file,
                content=f"Content {i}",
                read_before_write=True,
                user_review_callback=approval_callback,
            )
            tasks.append(task)

        # Wait for all operations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # At least one should succeed (implementation detail)
        success_count = sum(
            1 for r in results if isinstance(r, dict) and r.get("success")
        )
        assert success_count >= 1

    @pytest.mark.asyncio
    async def test_read_before_write_with_empty_callback_response(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write with malformed callback response."""
        test_file = temp_dir / "empty_response_test.txt"

        # Mock callback that returns empty/malformed response
        async def empty_callback(review_data):
            return {}  # Missing required fields

        # Should handle malformed response gracefully
        result = await file_system_manager.write_file(
            path=test_file,
            content="Test content",
            read_before_write=True,
            user_review_callback=empty_callback,
        )

        # Implementation should handle this gracefully (either succeed via fallback or fail cleanly)
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.asyncio
    async def test_read_before_write_with_file_disappearing(
        self, file_system_manager, temp_dir
    ):
        """Test read-before-write when file disappears between read and write."""
        test_file = temp_dir / "disappearing_test.txt"
        test_file.write_text("Original content")

        async def file_deleting_callback(review_data):
            # Delete the file during the callback
            test_file.unlink()
            return {"approved": True}

        # Test handling of file disappearing
        result = await file_system_manager.write_file(
            path=test_file,
            content="New content",
            read_before_write=True,
            user_review_callback=file_deleting_callback,
        )

        # Should still succeed (creates new file)
        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == "New content"


class TestReadBeforeWriteErrorHandler:
    """Test the ReadBeforeWriteErrorHandler class."""

    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return ReadBeforeWriteErrorHandler(enable_recovery=True, log_errors=True)

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_error_handler_initialization(self, error_handler):
        """Test error handler initializes correctly."""
        assert error_handler.enable_recovery is True
        assert error_handler.log_errors is True
        assert len(error_handler.error_history) == 0
        assert len(error_handler.recovery_strategies) > 0

    def test_handle_file_read_error(self, error_handler, temp_dir):
        """Test handling of file read errors."""
        test_file = temp_dir / "test.txt"
        original_exception = FileNotFoundError("File not found")

        read_error = FileReadError(str(test_file), original_exception)
        result = error_handler.handle_error(read_error)

        assert (
            result["recovery_strategy"]
            == RecoveryStrategy.FALLBACK_TO_REGULAR_WRITE.value
        )
        assert result["recovery_result"]["fallback_action"] == "regular_write"
        assert len(error_handler.error_history) == 1
        assert (
            error_handler.error_history[0].error_type
            == ReadBeforeWriteErrorType.FILE_READ_ERROR
        )

    def test_handle_file_write_error_with_retry(self, error_handler, temp_dir):
        """Test handling of file write errors with retry strategy."""
        test_file = temp_dir / "test.txt"
        original_exception = PermissionError("Permission denied")

        write_error = FileWriteError(str(test_file), original_exception)

        # First attempt - should retry
        result = error_handler.handle_error(write_error, retry_count=0, max_retries=2)
        assert result["recovery_result"]["should_retry"] is True
        assert "attempt 1/2" in result["recovery_result"]["message"]

        # Max retries exceeded - should abort
        result = error_handler.handle_error(write_error, retry_count=2, max_retries=2)
        assert result["recovery_result"]["should_abort"] is True
        assert "Maximum retries" in result["recovery_result"]["message"]

    def test_handle_user_rejection_error(self, error_handler, temp_dir):
        """Test handling of user rejection errors."""
        test_file = temp_dir / "test.txt"
        rejection_reason = "User does not want this change"

        rejection_error = UserRejectionError(str(test_file), rejection_reason)
        result = error_handler.handle_error(rejection_error)

        assert result["recovery_strategy"] == RecoveryStrategy.SKIP_OPERATION.value
        assert result["recovery_result"]["fallback_action"] == "skip"
        assert "Skipping operation" in result["recovery_result"]["message"]

    def test_handle_callback_error(self, error_handler, temp_dir):
        """Test handling of callback errors."""
        test_file = temp_dir / "test.txt"
        original_exception = RuntimeError("Callback failed")

        callback_error = CallbackError(str(test_file), original_exception)
        result = error_handler.handle_error(callback_error)

        assert (
            result["recovery_strategy"]
            == RecoveryStrategy.FALLBACK_TO_REGULAR_WRITE.value
        )
        assert result["recovery_result"]["fallback_action"] == "regular_write"

    def test_handle_content_validation_error(self, error_handler, temp_dir):
        """Test handling of content validation errors."""
        test_file = temp_dir / "test.txt"
        validation_issue = "Content contains suspicious patterns"

        validation_error = ContentValidationError(str(test_file), validation_issue)
        result = error_handler.handle_error(validation_error)

        assert result["recovery_strategy"] == RecoveryStrategy.PROMPT_USER.value
        assert result["recovery_result"]["fallback_action"] == "prompt_user"


if __name__ == "__main__":
    pytest.main([__file__])
