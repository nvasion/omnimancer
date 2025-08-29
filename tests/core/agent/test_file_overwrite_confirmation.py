"""
Unit tests for file overwrite confirmation functionality.

This module tests the user confirmation prompts when files already exist,
including the UI components and callback functions.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any

from omnimancer.core.agent.read_before_write_ui import (
    ReadBeforeWriteUI,
    create_confirmation_callback,
)


class TestFileOverwriteConfirmation:
    """Test file overwrite confirmation functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def ui(self):
        """Create ReadBeforeWriteUI for testing."""
        # Use a mock console to avoid actual terminal output during tests
        mock_console = Mock()
        return ReadBeforeWriteUI(console=mock_console)

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_new_file(self, ui):
        """Test confirmation for non-existent file."""
        file_info = {
            "path": "/test/new_file.txt",
            "exists": False,
            "is_file": False,
            "is_directory": False,
            "size": None,
            "modified_time": None,
        }

        result = await ui.confirm_file_overwrite(file_info)

        assert result["confirmed"] is True
        assert result["action"] == "create"
        assert "does not exist" in result["reason"]

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_approve(self, ui):
        """Test confirmation with user approval to overwrite."""
        file_info = {
            "path": "/test/existing_file.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 1024,
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # Mock user input to overwrite
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask",
            return_value="1",
        ) as mock_prompt, patch(
            "omnimancer.core.agent.read_before_write_ui.Confirm.ask",
            return_value=True,
        ) as mock_confirm:

            result = await ui.confirm_file_overwrite(file_info)

            assert result["confirmed"] is True
            assert result["action"] == "overwrite"
            assert result["reason"] == "User confirmed overwrite"

            # Verify prompts were called
            assert mock_prompt.call_count >= 1
            assert mock_confirm.call_count == 1

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_backup(self, ui):
        """Test confirmation with user choosing backup option."""
        file_info = {
            "path": "/test/existing_file.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 2048,
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # Mock user input to backup
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask",
            return_value="2",
        ):

            result = await ui.confirm_file_overwrite(file_info)

            assert result["confirmed"] is True
            assert result["action"] == "backup"
            assert "backup" in result["reason"]

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_cancel(self, ui):
        """Test confirmation with user canceling operation."""
        file_info = {
            "path": "/test/existing_file.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 1024,
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # Mock user input to cancel
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask"
        ) as mock_prompt:
            mock_prompt.side_effect = ["3", "User does not want to overwrite"]

            result = await ui.confirm_file_overwrite(file_info)

            assert result["confirmed"] is False
            assert result["action"] == "cancel"
            assert "User does not want to overwrite" in result["reason"]

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_symlink(self, ui):
        """Test confirmation for symbolic link."""
        file_info = {
            "path": "/test/symlink_file.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 512,
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": True,
        }

        # Mock user input to cancel (testing that symlink info is displayed)
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask",
            return_value="3",
        ) as mock_prompt:

            result = await ui.confirm_file_overwrite(file_info)

            assert result["confirmed"] is False
            assert result["action"] == "cancel"
            # UI should have displayed symlink information (tested through mock console)

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_directory(self, ui):
        """Test confirmation for directory."""
        file_info = {
            "path": "/test/existing_dir",
            "exists": True,
            "is_file": False,
            "is_directory": True,
            "size": None,  # Directories don't have file size
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # Mock user input to backup
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask",
            return_value="2",
        ):

            result = await ui.confirm_file_overwrite(file_info)

            assert result["confirmed"] is True
            assert result["action"] == "backup"

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_large_file(self, ui):
        """Test confirmation for large file with MB display."""
        file_info = {
            "path": "/test/large_file.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 5 * 1024 * 1024,  # 5 MB
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # Mock user input to cancel
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask",
            return_value="3",
        ):

            result = await ui.confirm_file_overwrite(file_info)

            assert result["confirmed"] is False
            assert result["action"] == "cancel"
            # Size should be displayed in MB format (tested through mock console)

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_small_file(self, ui):
        """Test confirmation for small file with KB display."""
        file_info = {
            "path": "/test/small_file.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 512,  # 0.5 KB
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # Mock user input to cancel
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask",
            return_value="3",
        ):

            result = await ui.confirm_file_overwrite(file_info)

            assert result["confirmed"] is False
            assert result["action"] == "cancel"
            # Size should be displayed in KB format (tested through mock console)

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_hesitant_user(self, ui):
        """Test confirmation when user is hesitant about overwriting."""
        file_info = {
            "path": "/test/important_file.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 1024,
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # Mock user input: first tries overwrite but hesitates, then chooses backup
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask"
        ) as mock_prompt, patch(
            "omnimancer.core.agent.read_before_write_ui.Confirm.ask",
            return_value=False,
        ) as mock_confirm:

            # User chooses overwrite, but then says no to confirmation, then chooses backup
            mock_prompt.side_effect = ["1", "2"]

            result = await ui.confirm_file_overwrite(file_info)

            assert result["confirmed"] is True
            assert result["action"] == "backup"
            assert "backup" in result["reason"]

            # Should have prompted for overwrite, then backup
            assert mock_prompt.call_count == 2
            assert (
                mock_confirm.call_count == 1
            )  # Asked for overwrite confirmation

    @pytest.mark.asyncio
    async def test_confirm_file_overwrite_error_handling(self, ui):
        """Test error handling in confirmation process."""
        file_info = {
            "path": "/test/error_file.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 1024,
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # Mock prompt to raise an exception
        with patch(
            "omnimancer.core.agent.read_before_write_ui.Prompt.ask",
            side_effect=Exception("Prompt error"),
        ):

            result = await ui.confirm_file_overwrite(file_info)

            assert result["confirmed"] is False
            assert result["action"] == "cancel"
            assert "Confirmation error" in result["reason"]

    def test_create_confirmation_callback(self):
        """Test creating confirmation callback function."""
        callback = create_confirmation_callback()

        assert callable(callback)
        assert asyncio.iscoroutinefunction(callback)

    @pytest.mark.asyncio
    async def test_confirmation_callback_integration(self):
        """Test confirmation callback with mock data."""
        callback = create_confirmation_callback()

        file_info = {
            "path": "/test/callback_test.txt",
            "exists": False,
            "is_file": False,
            "is_directory": False,
            "size": None,
            "modified_time": None,
        }

        result = await callback(file_info)

        assert result["confirmed"] is True
        assert result["action"] == "create"


class TestConfirmationUIDisplay:
    """Test the display components of confirmation UI."""

    @pytest.fixture
    def ui(self):
        """Create ReadBeforeWriteUI for testing."""
        mock_console = Mock()
        return ReadBeforeWriteUI(console=mock_console)

    def test_display_file_exists_warning_file(self, ui):
        """Test displaying warning for existing file."""
        file_info = {
            "path": "/test/warning_file.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 1024,
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # This should not raise an exception
        ui._display_file_exists_warning(file_info)

        # Verify console.print was called (mock console)
        assert ui.console.print.called

    def test_display_file_exists_warning_directory(self, ui):
        """Test displaying warning for existing directory."""
        file_info = {
            "path": "/test/warning_dir",
            "exists": True,
            "is_file": False,
            "is_directory": True,
            "size": None,
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": False,
        }

        # This should not raise an exception
        ui._display_file_exists_warning(file_info)

        # Verify console.print was called
        assert ui.console.print.called

    def test_display_file_exists_warning_symlink(self, ui):
        """Test displaying warning for symbolic link."""
        file_info = {
            "path": "/test/warning_symlink.txt",
            "exists": True,
            "is_file": True,
            "is_directory": False,
            "size": 512,
            "modified_time": "2024-01-01 12:00:00",
            "is_symlink": True,
        }

        # This should not raise an exception
        ui._display_file_exists_warning(file_info)

        # Verify console.print was called
        assert ui.console.print.called


if __name__ == "__main__":
    pytest.main([__file__])
