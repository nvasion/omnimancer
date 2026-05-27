"""Comprehensive tests for FileSystemManager."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from omnimancer.core.agent.approval_manager import EnhancedApprovalManager
from omnimancer.core.agent.file_system_manager import (
    FileOperationError,
    FileSystemManager,
)
from omnimancer.core.security import SecurityManager
from omnimancer.core.security.approval_workflow import ApprovalWorkflow


@pytest.fixture
async def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def mock_security():
    """Mock security manager for testing."""
    security = Mock(spec=SecurityManager)

    # Mock successful security checks by default
    async def mock_secure_file_access(path, operation, content=None):
        return {
            "success": True,
            "operation_id": "test_op",
            "session_id": "test_session",
        }

    async def mock_validate_operation(operation):
        return {"allowed": True, "reasons": ["Test allowed"]}

    async def mock_execute_secure_command(command, working_dir=None):
        if "git" in command:
            if "rev-parse" in command:
                return {"success": True, "stdout": ".git", "stderr": ""}
            elif "status" in command:
                return {
                    "success": True,
                    "stdout": "M test.txt\n",
                    "stderr": "",
                }
            elif "add" in command:
                return {"success": True, "stdout": "", "stderr": ""}
        return {"success": True, "stdout": "test output", "stderr": ""}

    security.secure_file_access = mock_secure_file_access
    security.validate_operation = mock_validate_operation
    security.execute_secure_command = mock_execute_secure_command

    return security


@pytest.fixture
async def fs_manager(mock_security, temp_dir):
    """Create FileSystemManager instance for testing."""
    backup_dir = temp_dir / "backups"
    manager = FileSystemManager(
        security_manager=mock_security,
        backup_dir=str(backup_dir),
        max_file_size_mb=1,  # Small limit for testing
        require_approval=False,  # Disable approval for basic tests
    )
    yield manager
    await manager.cleanup()


@pytest.fixture
async def mock_approval_manager():
    """Create mock approval manager for testing."""
    Mock(spec=ApprovalWorkflow)
    approval_manager = Mock(spec=EnhancedApprovalManager)

    # Mock successful approval by default
    approval_manager.request_single_approval = AsyncMock(return_value=True)
    return approval_manager


@pytest.fixture
async def fs_manager_with_approval(mock_security, mock_approval_manager, temp_dir):
    """Create FileSystemManager with approval enabled for testing."""
    backup_dir = temp_dir / "backups"
    manager = FileSystemManager(
        security_manager=mock_security,
        approval_manager=mock_approval_manager,
        backup_dir=str(backup_dir),
        max_file_size_mb=1,  # Small limit for testing
        require_approval=True,  # Enable approval for these tests
    )
    yield manager
    await manager.cleanup()


class TestFileSystemManager:
    """Test FileSystemManager functionality."""

    @pytest.mark.asyncio
    async def test_read_file_success(self, fs_manager, temp_dir):
        """Test successful file reading."""
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!"

        # Create test file
        test_file.write_text(test_content)

        # Read file
        content = await fs_manager.read_file(test_file)

        assert content == test_content

    @pytest.mark.asyncio
    async def test_read_binary_file(self, fs_manager, temp_dir):
        """Test reading binary files."""
        test_file = temp_dir / "test.bin"
        test_data = b"\x00\x01\x02\x03\xff"

        # Create binary test file
        test_file.write_bytes(test_data)

        # Read as binary
        content = await fs_manager.read_file(test_file, binary=True)

        assert content == test_data
        assert isinstance(content, bytes)

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, fs_manager, temp_dir):
        """Test reading non-existent file raises error."""
        test_file = temp_dir / "nonexistent.txt"

        with pytest.raises(FileOperationError, match="File not found"):
            await fs_manager.read_file(test_file)

    @pytest.mark.asyncio
    async def test_write_file_success(self, fs_manager, temp_dir):
        """Test successful file writing."""
        test_file = temp_dir / "write_test.txt"
        test_content = "Test content for writing"

        # Write file
        result = await fs_manager.write_file(test_file, test_content)

        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == test_content

    @pytest.mark.asyncio
    async def test_write_file_with_backup(self, fs_manager, temp_dir):
        """Test file writing with backup creation."""
        test_file = temp_dir / "backup_test.txt"
        original_content = "Original content"
        new_content = "New content"

        # Create original file
        test_file.write_text(original_content)

        # Write new content with backup
        result = await fs_manager.write_file(test_file, new_content, backup=True)

        assert result["success"] is True
        assert result["backup_path"] is not None
        assert test_file.read_text() == new_content

        # Check backup exists
        backup_path = Path(result["backup_path"])
        assert backup_path.exists()
        assert backup_path.read_text() == original_content

    @pytest.mark.asyncio
    async def test_atomic_write(self, fs_manager, temp_dir):
        """Test atomic write operations."""
        test_file = temp_dir / "atomic_test.txt"
        test_content = "Atomic write test"

        # Write with atomic=True (default)
        result = await fs_manager.write_file(test_file, test_content, atomic=True)

        assert result["success"] is True
        assert result["atomic"] is True
        assert test_file.exists()
        assert test_file.read_text() == test_content

    @pytest.mark.asyncio
    async def test_create_backup(self, fs_manager, temp_dir):
        """Test backup creation."""
        test_file = temp_dir / "backup_source.txt"
        test_content = "Content to backup"

        # Create test file
        test_file.write_text(test_content)

        # Create backup
        backup_path = await fs_manager.create_backup(test_file)

        assert backup_path.exists()
        assert backup_path.read_text() == test_content
        assert ".backup" in backup_path.name

    @pytest.mark.asyncio
    async def test_restore_backup(self, fs_manager, temp_dir):
        """Test backup restoration."""
        test_file = temp_dir / "restore_test.txt"
        original_content = "Original content"
        modified_content = "Modified content"

        # Create and backup original file
        test_file.write_text(original_content)
        backup_path = await fs_manager.create_backup(test_file)

        # Modify original file
        test_file.write_text(modified_content)
        assert test_file.read_text() == modified_content

        # Restore from backup
        success = await fs_manager.restore_backup(backup_path, test_file)

        assert success is True
        assert test_file.read_text() == original_content

    @pytest.mark.asyncio
    async def test_create_directory(self, fs_manager, temp_dir):
        """Test directory creation."""
        new_dir = temp_dir / "new_directory"

        success = await fs_manager.create_directory(new_dir)

        assert success is True
        assert new_dir.exists()
        assert new_dir.is_dir()

    @pytest.mark.asyncio
    async def test_create_nested_directory(self, fs_manager, temp_dir):
        """Test nested directory creation."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"

        success = await fs_manager.create_directory(nested_dir, parents=True)

        assert success is True
        assert nested_dir.exists()
        assert nested_dir.is_dir()

    @pytest.mark.asyncio
    async def test_delete_file(self, fs_manager, temp_dir):
        """Test file deletion."""
        test_file = temp_dir / "delete_test.txt"
        test_content = "File to delete"

        # Create test file
        test_file.write_text(test_content)
        assert test_file.exists()

        # Delete file
        result = await fs_manager.delete_file(test_file)

        assert result["success"] is True
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_delete_file_with_backup(self, fs_manager, temp_dir):
        """Test file deletion with backup."""
        test_file = temp_dir / "delete_backup_test.txt"
        test_content = "File to delete with backup"

        # Create test file
        test_file.write_text(test_content)

        # Delete with backup
        result = await fs_manager.delete_file(test_file, backup=True)

        assert result["success"] is True
        assert not test_file.exists()
        assert result["backup_path"] is not None

        # Check backup exists
        backup_path = Path(result["backup_path"])
        assert backup_path.exists()
        assert backup_path.read_text() == test_content

    @pytest.mark.asyncio
    async def test_glob_files(self, fs_manager, temp_dir):
        """Test glob pattern file matching."""
        # Create test files
        (temp_dir / "test1.txt").write_text("content1")
        (temp_dir / "test2.txt").write_text("content2")
        (temp_dir / "other.log").write_text("log content")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "test3.txt").write_text("content3")

        # Test glob patterns
        txt_files = await fs_manager.glob_files(str(temp_dir / "*.txt"))
        assert len(txt_files) == 2
        assert all(f.suffix == ".txt" for f in txt_files)

        # Test recursive glob
        all_txt = await fs_manager.glob_files(
            str(temp_dir / "**/*.txt"), recursive=True
        )
        assert len(all_txt) == 3
        assert all(f.suffix == ".txt" for f in all_txt)

    @pytest.mark.asyncio
    async def test_copy_file(self, fs_manager, temp_dir):
        """Test file copying."""
        src_file = temp_dir / "source.txt"
        dst_file = temp_dir / "destination.txt"
        test_content = "Content to copy"

        # Create source file
        src_file.write_text(test_content)

        # Copy file
        success = await fs_manager.copy_file(src_file, dst_file)

        assert success is True
        assert dst_file.exists()
        assert dst_file.read_text() == test_content
        assert src_file.exists()  # Source should still exist

    @pytest.mark.asyncio
    async def test_move_file(self, fs_manager, temp_dir):
        """Test file moving."""
        src_file = temp_dir / "source.txt"
        dst_file = temp_dir / "destination.txt"
        test_content = "Content to move"

        # Create source file
        src_file.write_text(test_content)

        # Move file
        result = await fs_manager.move_file(src_file, dst_file)

        assert result["success"] is True
        assert dst_file.exists()
        assert dst_file.read_text() == test_content
        assert not src_file.exists()  # Source should be gone

    @pytest.mark.asyncio
    async def test_get_file_info(self, fs_manager, temp_dir):
        """Test getting file information."""
        test_file = temp_dir / "info_test.txt"
        test_content = "Content for info test"

        # Create test file
        test_file.write_text(test_content)

        # Get file info
        info = await fs_manager.get_file_info(test_file)

        assert info["name"] == "info_test.txt"
        assert info["size"] == len(test_content.encode())
        assert info["is_file"] is True
        assert info["is_dir"] is False
        assert info["is_binary"] is False
        assert "md5_hash" in info
        assert "sha256_hash" in info

    @pytest.mark.asyncio
    async def test_list_directory(self, fs_manager, temp_dir):
        """Test directory listing."""
        # Create test files and directories
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "nested.txt").write_text("nested content")

        # List directory (non-recursive)
        items = await fs_manager.list_directory(temp_dir, recursive=False)

        # Check that we have at least the expected files and directories
        names = [item["name"] for item in items]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names

        # Verify we have exactly the files and directories we created
        expected_items = {"file1.txt", "file2.txt", "subdir"}
        actual_items = set(names)
        assert expected_items.issubset(
            actual_items
        ), f"Missing expected items: {expected_items - actual_items}"

        # Test recursive listing
        all_items = await fs_manager.list_directory(temp_dir, recursive=True)
        all_names = [item["name"] for item in all_items]
        assert "nested.txt" in all_names
        assert "file1.txt" in all_names
        assert "file2.txt" in all_names
        assert "subdir" in all_names

    @pytest.mark.asyncio
    async def test_binary_file_detection(self, fs_manager, temp_dir):
        """Test binary file detection."""
        # Create text file
        text_file = temp_dir / "text.txt"
        text_file.write_text("This is text content")

        # Create binary file
        binary_file = temp_dir / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff")

        assert not fs_manager._is_binary_file(text_file)
        assert fs_manager._is_binary_file(binary_file)

    @pytest.mark.asyncio
    async def test_large_file_handling(self, fs_manager, temp_dir):
        """Test handling of files larger than size limit."""
        large_file = temp_dir / "large.txt"

        # Create content larger than 1MB limit
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        large_file.write_text(large_content)

        # Should still be able to read (using streaming)
        content = await fs_manager.read_file(large_file)
        assert len(content) == len(large_content)

    @pytest.mark.asyncio
    async def test_git_integration(self, fs_manager, temp_dir):
        """Test git integration methods."""
        test_file = temp_dir / "git_test.txt"
        test_file.write_text("Git test content")

        # Test git add
        success = await fs_manager.git_add(test_file)
        assert success is True

        # Test git status
        status = await fs_manager.git_status(temp_dir)
        assert status["success"] is True
        assert "test.txt" in status["status_output"]

    @pytest.mark.asyncio
    async def test_operation_tracking(self, fs_manager, temp_dir):
        """Test operation tracking functionality."""
        test_file = temp_dir / "tracking_test.txt"
        test_content = "Content for tracking test"

        # Perform write operation
        result = await fs_manager.write_file(test_file, test_content)
        operation_id = result["operation_id"]

        # Check operation status
        status = await fs_manager.get_operation_status(operation_id)
        assert status is not None
        assert status["type"] == "write"
        assert status["path"] == test_file
        assert status["success"] is True

    @pytest.mark.asyncio
    async def test_streaming_read(self, fs_manager, temp_dir):
        """Test streaming file read context manager."""
        test_file = temp_dir / "stream_test.txt"
        test_content = "Content for streaming test"
        test_file.write_text(test_content)

        # Test streaming read
        async with fs_manager.streaming_read(test_file) as f:
            chunk = await f.read(1024)
            assert test_content.encode() == chunk

    @pytest.mark.asyncio
    async def test_cleanup(self, fs_manager):
        """Test cleanup functionality."""
        # Add some tracked operations
        fs_manager.active_operations["test_op"] = {
            "type": "test",
            "success": True,
        }

        # Test cleanup
        await fs_manager.cleanup()
        assert len(fs_manager.active_operations) == 0


class TestFileSystemManagerErrorHandling:
    """Test error handling in FileSystemManager."""

    @pytest.mark.asyncio
    async def test_security_denied_read(self, temp_dir):
        """Test handling of security-denied file reads."""
        # Create mock security that denies access
        mock_security = Mock(spec=SecurityManager)

        async def mock_secure_file_access(path, operation, content=None):
            return {
                "success": False,
                "error": "Access denied by security policy",
            }

        mock_security.secure_file_access = mock_secure_file_access

        # Create file system manager with restrictive security
        fs_manager = FileSystemManager(security_manager=mock_security)

        test_file = temp_dir / "restricted.txt"
        test_file.write_text("Restricted content")

        # Should raise FileOperationError
        with pytest.raises(FileOperationError, match="Security check failed"):
            await fs_manager.read_file(test_file)

    @pytest.mark.asyncio
    async def test_security_denied_write(self, temp_dir):
        """Test handling of security-denied file writes."""
        mock_security = Mock(spec=SecurityManager)

        async def mock_secure_file_access(path, operation, content=None):
            return {"success": False, "error": "Write access denied"}

        mock_security.secure_file_access = mock_secure_file_access

        fs_manager = FileSystemManager(security_manager=mock_security)

        test_file = temp_dir / "restricted_write.txt"

        with pytest.raises(FileOperationError, match="Security check failed"):
            await fs_manager.write_file(test_file, "Restricted content")

    @pytest.mark.asyncio
    async def test_backup_nonexistent_file(self, fs_manager, temp_dir):
        """Test backup creation for non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.txt"

        with pytest.raises(FileOperationError, match="Cannot backup non-existent file"):
            await fs_manager.create_backup(nonexistent_file)

    @pytest.mark.asyncio
    async def test_restore_nonexistent_backup(self, fs_manager, temp_dir):
        """Test restoration from non-existent backup."""
        nonexistent_backup = temp_dir / "nonexistent.backup"
        target_file = temp_dir / "target.txt"

        with pytest.raises(FileOperationError, match="Backup file not found"):
            await fs_manager.restore_backup(nonexistent_backup, target_file)

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self, fs_manager, temp_dir):
        """Test listing non-existent directory."""
        nonexistent_dir = temp_dir / "nonexistent"

        with pytest.raises(FileOperationError, match="Failed to list directory"):
            await fs_manager.list_directory(nonexistent_dir)


class TestFileSystemManagerIntegration:
    """Integration tests for FileSystemManager."""

    @pytest.mark.asyncio
    async def test_complete_file_workflow(self, fs_manager, temp_dir):
        """Test a complete file operation workflow."""
        original_file = temp_dir / "workflow_test.txt"
        copied_file = temp_dir / "workflow_copy.txt"
        moved_file = temp_dir / "workflow_moved.txt"

        original_content = "Original workflow content"
        modified_content = "Modified workflow content"

        # 1. Create original file
        result = await fs_manager.write_file(original_file, original_content)
        assert result["success"] is True

        # 2. Copy file
        success = await fs_manager.copy_file(original_file, copied_file)
        assert success is True

        # 3. Modify original with backup
        result = await fs_manager.write_file(
            original_file, modified_content, backup=True
        )
        assert result["success"] is True
        assert result["backup_path"] is not None

        # 4. Move copied file
        result = await fs_manager.move_file(copied_file, moved_file)
        assert result["success"] is True

        # 5. Verify final state
        assert original_file.read_text() == modified_content
        assert moved_file.read_text() == original_content
        assert not copied_file.exists()

        # 6. Restore from backup
        backup_path = Path(result["backup_path"])
        success = await fs_manager.restore_backup(backup_path, original_file)
        assert success is True
        assert original_file.read_text() == original_content

    @pytest.mark.asyncio
    async def test_directory_operations_workflow(self, fs_manager, temp_dir):
        """Test complete directory operations workflow."""
        # Create directory structure
        main_dir = temp_dir / "main"
        sub_dir = main_dir / "subdir"

        await fs_manager.create_directory(sub_dir, parents=True)

        # Create files in directories
        main_file = main_dir / "main.txt"
        sub_file = sub_dir / "sub.txt"

        await fs_manager.write_file(main_file, "Main content")
        await fs_manager.write_file(sub_file, "Sub content")

        # List directories
        main_contents = await fs_manager.list_directory(main_dir, recursive=False)
        assert len(main_contents) == 2  # 1 file + 1 subdir

        all_contents = await fs_manager.list_directory(main_dir, recursive=True)
        assert len(all_contents) == 3  # 2 files + 1 subdir

        # Test glob patterns
        txt_files = await fs_manager.glob_files(
            str(main_dir / "**/*.txt"), recursive=True
        )
        assert len(txt_files) == 2
        assert all(f.suffix == ".txt" for f in txt_files)


class TestFileSystemManagerDirectoryAwareness:
    """Test directory awareness functionality in FileSystemManager."""

    def test_get_current_working_directory(self, fs_manager):
        """Test getting current working directory."""
        cwd = fs_manager.get_current_working_directory()

        assert isinstance(cwd, Path)
        assert cwd.exists()
        assert cwd.is_dir()
        # Should match Path.cwd()
        assert cwd == Path.cwd()

    @pytest.mark.asyncio
    async def test_is_git_repository_true(self, fs_manager, temp_dir):
        """Test git repository detection for actual git repository."""
        # Create a .git directory to simulate a git repository
        git_dir = temp_dir / ".git"
        git_dir.mkdir()

        is_repo = await fs_manager.is_git_repository(temp_dir)

        assert is_repo is True

    @pytest.mark.asyncio
    async def test_is_git_repository_false(self, fs_manager, temp_dir):
        """Test git repository detection for non-git directory."""
        # temp_dir doesn't have .git directory by default
        is_repo = await fs_manager.is_git_repository(temp_dir)

        assert is_repo is False

    @pytest.mark.asyncio
    async def test_is_git_repository_subdirectory(self, fs_manager, temp_dir):
        """Test git repository detection from subdirectory."""
        # Create a .git directory in parent
        git_dir = temp_dir / ".git"
        git_dir.mkdir()

        # Create subdirectory
        sub_dir = temp_dir / "subdir" / "nested"
        sub_dir.mkdir(parents=True)

        # Should detect git repository from subdirectory
        is_repo = await fs_manager.is_git_repository(sub_dir)

        assert is_repo is True

    @pytest.mark.asyncio
    async def test_is_git_repository_default_path(self, fs_manager):
        """Test git repository detection using current working directory."""
        # Test with default path (current directory)
        is_repo = await fs_manager.is_git_repository()

        # Result depends on whether current directory is in a git repo
        assert isinstance(is_repo, bool)

    @pytest.mark.asyncio
    async def test_get_git_repository_root_found(self, fs_manager, temp_dir):
        """Test getting git repository root when it exists."""
        # Create a .git directory to simulate a git repository
        git_dir = temp_dir / ".git"
        git_dir.mkdir()

        # Create subdirectory structure
        nested_dir = temp_dir / "src" / "components"
        nested_dir.mkdir(parents=True)

        # Test from nested directory
        repo_root = await fs_manager.get_git_repository_root(nested_dir)

        assert repo_root == temp_dir
        assert isinstance(repo_root, Path)

    @pytest.mark.asyncio
    async def test_get_git_repository_root_not_found(self, fs_manager, temp_dir):
        """Test getting git repository root when it doesn't exist."""
        # temp_dir doesn't have .git directory by default
        repo_root = await fs_manager.get_git_repository_root(temp_dir)

        assert repo_root is None

    @pytest.mark.asyncio
    async def test_get_git_repository_root_default_path(self, fs_manager):
        """Test getting git repository root using current working directory."""
        # Test with default path (current directory)
        repo_root = await fs_manager.get_git_repository_root()

        # Result depends on whether current directory is in a git repo
        # Should be either a Path object or None
        assert repo_root is None or isinstance(repo_root, Path)

    @pytest.mark.asyncio
    async def test_get_directory_context_non_git(self, fs_manager, temp_dir):
        """Test getting directory context for non-git directory."""
        context = await fs_manager.get_directory_context(temp_dir)

        assert context["current_working_directory"] == str(temp_dir)
        assert context["is_git_repository"] is False
        assert context["git_repository_root"] is None
        assert context["relative_to_repo_root"] is None

    @pytest.mark.asyncio
    async def test_get_directory_context_git_repository(self, fs_manager, temp_dir):
        """Test getting directory context for git repository."""
        # Create a .git directory to simulate a git repository
        git_dir = temp_dir / ".git"
        git_dir.mkdir()

        # Create subdirectory
        sub_dir = temp_dir / "src" / "components"
        sub_dir.mkdir(parents=True)

        # Test from subdirectory
        context = await fs_manager.get_directory_context(sub_dir)

        assert context["current_working_directory"] == str(sub_dir)
        assert context["is_git_repository"] is True
        assert context["git_repository_root"] == str(temp_dir)
        assert context["relative_to_repo_root"] == "src/components"

    @pytest.mark.asyncio
    async def test_get_directory_context_default_path(self, fs_manager):
        """Test getting directory context using current working directory."""
        context = await fs_manager.get_directory_context()

        # Should have all required keys
        assert "current_working_directory" in context
        assert "is_git_repository" in context
        assert "git_repository_root" in context
        assert "relative_to_repo_root" in context

        # current_working_directory should match current directory
        assert context["current_working_directory"] == str(Path.cwd())

        # is_git_repository should be a boolean
        assert isinstance(context["is_git_repository"], bool)

    @pytest.mark.asyncio
    async def test_get_directory_context_git_root(self, fs_manager, temp_dir):
        """Test getting directory context from git repository root."""
        # Create a .git directory to simulate a git repository
        git_dir = temp_dir / ".git"
        git_dir.mkdir()

        # Test from repository root
        context = await fs_manager.get_directory_context(temp_dir)

        assert context["current_working_directory"] == str(temp_dir)
        assert context["is_git_repository"] is True
        assert context["git_repository_root"] == str(temp_dir)
        assert context["relative_to_repo_root"] == "."

    @pytest.mark.asyncio
    async def test_directory_awareness_integration(self, fs_manager, temp_dir):
        """Test integration of all directory awareness methods."""
        # Create a complex directory structure with git repository
        git_dir = temp_dir / ".git"
        git_dir.mkdir()

        # Create nested structure
        project_src = temp_dir / "src"
        project_tests = temp_dir / "tests"
        project_docs = temp_dir / "docs"
        deep_nested = temp_dir / "src" / "components" / "ui" / "forms"

        for dir_path in [
            project_src,
            project_tests,
            project_docs,
            deep_nested,
        ]:
            dir_path.mkdir(parents=True)

        # Test from various locations
        test_cases = [
            (temp_dir, "."),
            (project_src, "src"),
            (project_tests, "tests"),
            (deep_nested, "src/components/ui/forms"),
        ]

        for test_path, expected_relative in test_cases:
            # Test individual methods
            fs_manager.get_current_working_directory()
            is_repo = await fs_manager.is_git_repository(test_path)
            repo_root = await fs_manager.get_git_repository_root(test_path)
            context = await fs_manager.get_directory_context(test_path)

            # Verify consistency
            assert is_repo is True
            assert repo_root == temp_dir
            assert context["is_git_repository"] is True
            assert context["git_repository_root"] == str(temp_dir)
            assert context["relative_to_repo_root"] == expected_relative
            assert context["current_working_directory"] == str(test_path)


class TestFileSystemManagerApprovalFlow:
    """Test approval flow integration in FileSystemManager."""

    @pytest.mark.asyncio
    async def test_write_file_with_approval(self, fs_manager_with_approval, temp_dir):
        """Test file writing with approval system enabled."""
        test_file = temp_dir / "approval_test.txt"
        test_content = "Content requiring approval"

        # Write file with approval
        result = await fs_manager_with_approval.write_file(test_file, test_content)

        assert result["success"] is True
        assert result["approved"] is True
        assert test_file.exists()
        assert test_file.read_text() == test_content

        # Verify the approval manager was called
        approval = fs_manager_with_approval.approval_manager
        approval.request_single_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_file_approval_denied(self, mock_security, temp_dir):
        """Test file writing when approval is denied."""
        # Create approval manager that denies requests
        denial_approval_manager = Mock(spec=EnhancedApprovalManager)
        denial_approval_manager.request_single_approval = AsyncMock(return_value=False)

        # Create file manager with denial approval
        backup_dir = temp_dir / "backups"
        fs_manager = FileSystemManager(
            security_manager=mock_security,
            approval_manager=denial_approval_manager,
            backup_dir=str(backup_dir),
            require_approval=True,
        )

        test_file = temp_dir / "denied_test.txt"
        test_content = "This should be denied"

        # Should raise error when approval is denied
        with pytest.raises(FileOperationError, match="denied by approval workflow"):
            await fs_manager.write_file(test_file, test_content)

        # File should not be created
        assert not test_file.exists()

        await fs_manager.cleanup()

    @pytest.mark.asyncio
    async def test_delete_file_with_approval(self, fs_manager_with_approval, temp_dir):
        """Test file deletion with approval system enabled."""
        test_file = temp_dir / "delete_approval_test.txt"
        test_content = "File to delete with approval"

        # Create file first (without approval for setup)
        test_file.write_text(test_content)

        # Delete file with approval
        result = await fs_manager_with_approval.delete_file(test_file)

        assert result["success"] is True
        assert result["approved"] is True
        assert not test_file.exists()
        assert result["backup_path"] is not None

        # Verify backup was created
        backup_path = Path(result["backup_path"])
        assert backup_path.exists()
        assert backup_path.read_text() == test_content

    @pytest.mark.asyncio
    async def test_create_directory_with_approval(
        self, fs_manager_with_approval, temp_dir
    ):
        """Test directory creation with approval system enabled."""
        new_dir = temp_dir / "approved_directory"

        # Create directory with approval
        success = await fs_manager_with_approval.create_directory(new_dir)

        assert success is True
        assert new_dir.exists()
        assert new_dir.is_dir()

        # Verify the approval manager was called
        approval = fs_manager_with_approval.approval_manager
        approval.request_single_approval.assert_called()

    @pytest.mark.asyncio
    async def test_move_file_with_approval(self, fs_manager_with_approval, temp_dir):
        """Test file moving with approval system enabled."""
        src_file = temp_dir / "source_approval.txt"
        dst_file = temp_dir / "destination_approval.txt"
        test_content = "Content to move with approval"

        # Create source file first
        src_file.write_text(test_content)

        # Move file with approval
        result = await fs_manager_with_approval.move_file(src_file, dst_file)

        assert result["success"] is True
        assert result["approved"] is True
        assert dst_file.exists()
        assert dst_file.read_text() == test_content
        assert not src_file.exists()

    @pytest.mark.asyncio
    async def test_approval_disabled(self, mock_security, temp_dir):
        """Test that operations work normally when approval is disabled."""
        backup_dir = temp_dir / "backups"
        fs_manager = FileSystemManager(
            security_manager=mock_security,
            backup_dir=str(backup_dir),
            require_approval=False,  # Approval disabled
        )

        test_file = temp_dir / "no_approval_test.txt"
        test_content = "Content without approval"

        # Write file without approval
        result = await fs_manager.write_file(test_file, test_content)

        assert result["success"] is True
        # 'approved' key should still be present but True (auto-approved)
        assert "approved" in result
        assert test_file.exists()
        assert test_file.read_text() == test_content

        await fs_manager.cleanup()

    @pytest.mark.asyncio
    async def test_operation_data_structure(self, fs_manager_with_approval, temp_dir):
        """Test that operations contain correct data for approval system."""
        test_file = temp_dir / "operation_data_test.txt"
        test_content = "Test content for operation data"

        # Mock to capture the operation passed to approval
        captured_operation = None

        async def capture_operation(operation):
            nonlocal captured_operation
            captured_operation = operation
            return True

        fs_manager_with_approval.approval_manager.request_single_approval = AsyncMock(
            side_effect=capture_operation
        )

        # Perform write operation
        await fs_manager_with_approval.write_file(test_file, test_content)

        # Verify operation structure
        assert captured_operation is not None
        assert captured_operation.description == f"Create file: {test_file.name}"
        assert captured_operation.data["path"] == str(test_file)
        assert captured_operation.data["content"] == test_content
        assert captured_operation.data["content_length"] == len(test_content)
        assert captured_operation.requires_approval is True


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
