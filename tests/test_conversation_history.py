"""
Unit tests for conversation history management in CLI.

This module tests the save, load, and list conversation functionality
including integration with the conversation manager.
"""

from unittest.mock import MagicMock, patch

import pytest

from omnimancer.cli.commands import Command, SlashCommand
from omnimancer.cli.interface import CommandLineInterface
from omnimancer.core.engine import CoreEngine


@pytest.fixture
def mock_engine():
    """Create a mock CoreEngine for testing conversation history."""
    engine = MagicMock(spec=CoreEngine)
    engine.get_conversation_summary = MagicMock(
        return_value={
            "message_count": 5,
            "current_model": "gpt-4",
            "session_id": "test-session",
        }
    )
    engine.save_conversation = MagicMock(
        return_value="conversation_20240101_120000.json"
    )
    engine.load_conversation = MagicMock()
    engine.list_conversations = MagicMock(return_value=[])
    engine.get_conversation_info = MagicMock(
        return_value={
            "filename": "test.json",
            "message_count": 5,
            "current_model": "gpt-4",
            "created_at": "2024-01-01T12:00:00",
        }
    )
    return engine


@pytest.fixture
def cli_interface(mock_engine):
    """Create a CLI interface with mocked engine."""
    return CommandLineInterface(mock_engine)


class TestSaveCommand:
    """Test save conversation functionality."""

    @pytest.mark.asyncio
    async def test_save_command_with_messages(self, cli_interface, mock_engine):
        """Test saving conversation when messages exist."""
        command = Command.create_slash_command(SlashCommand.SAVE, [], "/save")

        with patch.object(cli_interface, "_show_info") as mock_show_info:
            await cli_interface._handle_save_command(command)

            mock_engine.save_conversation.assert_called_once_with(None)
            mock_show_info.assert_called_once_with(
                "Conversation saved as: conversation_20240101_120000.json"
            )

    @pytest.mark.asyncio
    async def test_save_command_with_filename(self, cli_interface, mock_engine):
        """Test saving conversation with custom filename."""
        command = Command.create_slash_command(
            SlashCommand.SAVE, ["my_conversation"], "/save my_conversation"
        )

        with patch.object(cli_interface, "_show_info") as mock_show_info:
            await cli_interface._handle_save_command(command)

            mock_engine.save_conversation.assert_called_once_with("my_conversation")
            mock_show_info.assert_called_once_with(
                "Conversation saved as: conversation_20240101_120000.json"
            )

    @pytest.mark.asyncio
    async def test_save_command_no_messages(self, cli_interface, mock_engine):
        """Test saving conversation when no messages exist."""
        mock_engine.get_conversation_summary.return_value = {
            "message_count": 0,
            "current_model": "gpt-4",
            "session_id": "test-session",
        }

        command = Command.create_slash_command(SlashCommand.SAVE, [], "/save")

        with patch.object(cli_interface, "_show_info") as mock_show_info:
            await cli_interface._handle_save_command(command)

            mock_engine.save_conversation.assert_not_called()
            mock_show_info.assert_called_once_with("No messages to save.")

    @pytest.mark.asyncio
    async def test_save_command_error(self, cli_interface, mock_engine):
        """Test saving conversation when an error occurs."""
        mock_engine.save_conversation.side_effect = Exception("Save error")
        command = Command.create_slash_command(SlashCommand.SAVE, [], "/save")

        with patch.object(cli_interface, "_show_error") as mock_show_error:
            await cli_interface._handle_save_command(command)

            mock_show_error.assert_called_once_with("Save failed: Save error")


class TestLoadCommand:
    """Test load conversation functionality."""

    @pytest.mark.asyncio
    async def test_load_command_success(self, cli_interface, mock_engine):
        """Test loading conversation successfully."""
        command = Command.create_slash_command(
            SlashCommand.LOAD, ["test.json"], "/load test.json"
        )

        with patch.object(cli_interface, "_show_info") as mock_show_info:
            await cli_interface._handle_load_command(command)

            mock_engine.load_conversation.assert_called_once_with("test.json")
            mock_engine.get_conversation_info.assert_called_once_with("test.json")

            # Check that info messages were shown
            assert mock_show_info.call_count == 2
            mock_show_info.assert_any_call("Loaded conversation: test.json")
            mock_show_info.assert_any_call("Messages: 5, Model: gpt-4")

    @pytest.mark.asyncio
    async def test_load_command_no_info(self, cli_interface, mock_engine):
        """Test loading conversation when info is not available."""
        mock_engine.get_conversation_info.return_value = None
        command = Command.create_slash_command(
            SlashCommand.LOAD, ["test.json"], "/load test.json"
        )

        with patch.object(cli_interface, "_show_info") as mock_show_info:
            await cli_interface._handle_load_command(command)

            mock_engine.load_conversation.assert_called_once_with("test.json")
            mock_show_info.assert_called_once_with("Loaded conversation: test.json")

    @pytest.mark.asyncio
    async def test_load_command_no_filename(self, cli_interface, mock_engine):
        """Test load command without filename."""
        command = Command.create_slash_command(SlashCommand.LOAD, [], "/load")

        with (
            patch.object(
                cli_interface, "_show_conversations"
            ) as mock_show_conversations,
            patch.object(cli_interface, "_show_error") as mock_show_error,
        ):
            await cli_interface._handle_load_command(command)

            mock_show_conversations.assert_called_once()
            mock_show_error.assert_called_once_with("Usage: /load <filename>")
            mock_engine.load_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_command_error(self, cli_interface, mock_engine):
        """Test loading conversation when an error occurs."""
        mock_engine.load_conversation.side_effect = Exception("Load error")
        command = Command.create_slash_command(
            SlashCommand.LOAD, ["test.json"], "/load test.json"
        )

        with patch.object(cli_interface, "_show_error") as mock_show_error:
            await cli_interface._handle_load_command(command)

            mock_show_error.assert_called_once_with("Load failed: Load error")


class TestListCommand:
    """Test list conversations functionality."""

    @pytest.mark.asyncio
    async def test_list_command_handler(self, cli_interface):
        """Test /list command handler."""
        command = Command.create_slash_command(SlashCommand.LIST, [], "/list")

        with patch.object(
            cli_interface, "_show_conversations"
        ) as mock_show_conversations:
            await cli_interface._handle_slash_command(command)
            mock_show_conversations.assert_called_once()


class TestShowConversations:
    """Test show conversations functionality."""

    @pytest.mark.asyncio
    async def test_show_conversations_empty(self, cli_interface, mock_engine):
        """Test showing conversations when none exist."""
        mock_engine.list_conversations.return_value = []

        with patch.object(cli_interface, "_show_info") as mock_show_info:
            await cli_interface._show_conversations()
            mock_show_info.assert_called_once_with("No saved conversations found.")

    @pytest.mark.asyncio
    async def test_show_conversations_with_data(self, cli_interface, mock_engine):
        """Test showing conversations with data."""
        conversations = [
            {
                "filename": "conversation1.json",
                "created_at": "2024-01-01T12:00:00",
                "message_count": 5,
                "current_model": "gpt-4",
                "session_id": "session1",
            },
            {
                "filename": "conversation2.json",
                "created_at": "2024-01-02T14:30:00",
                "message_count": 3,
                "current_model": "claude-3",
                "session_id": "session2",
            },
        ]
        mock_engine.list_conversations.return_value = conversations

        with patch.object(cli_interface.console, "print") as mock_print:
            await cli_interface._show_conversations()

            mock_print.assert_called_once()
            # Check that a Panel was printed
            args = mock_print.call_args[0]
            assert len(args) == 1
            assert hasattr(args[0], "title")
            assert args[0].title == "Saved Conversations"

    @pytest.mark.asyncio
    async def test_show_conversations_with_invalid_date(
        self, cli_interface, mock_engine
    ):
        """Test showing conversations with invalid date format."""
        conversations = [
            {
                "filename": "conversation1.json",
                "created_at": "invalid-date",
                "message_count": 5,
                "current_model": "gpt-4",
                "session_id": "session1",
            }
        ]
        mock_engine.list_conversations.return_value = conversations

        with patch.object(cli_interface.console, "print") as mock_print:
            await cli_interface._show_conversations()

            mock_print.assert_called_once()
            # Should still work even with invalid date
            args = mock_print.call_args[0]
            assert len(args) == 1
            assert hasattr(args[0], "title")

    @pytest.mark.asyncio
    async def test_show_conversations_error(self, cli_interface, mock_engine):
        """Test showing conversations when an error occurs."""
        mock_engine.list_conversations.side_effect = Exception("List error")

        with patch.object(cli_interface, "_show_error") as mock_show_error:
            await cli_interface._show_conversations()
            mock_show_error.assert_called_once_with(
                "Failed to list conversations: List error"
            )


class TestConversationIntegration:
    """Test integration between conversation commands."""

    @pytest.mark.asyncio
    async def test_save_load_workflow(self, cli_interface, mock_engine):
        """Test complete save and load workflow."""
        # Test save
        save_command = Command.create_slash_command(
            SlashCommand.SAVE, ["test_conversation"], "/save test_conversation"
        )

        with patch.object(cli_interface, "_show_info"):
            await cli_interface._handle_save_command(save_command)
            mock_engine.save_conversation.assert_called_once_with("test_conversation")

        # Test load
        load_command = Command.create_slash_command(
            SlashCommand.LOAD, ["test_conversation"], "/load test_conversation"
        )

        with patch.object(cli_interface, "_show_info"):
            await cli_interface._handle_load_command(load_command)
            mock_engine.load_conversation.assert_called_once_with("test_conversation")

    @pytest.mark.asyncio
    async def test_list_then_load_workflow(self, cli_interface, mock_engine):
        """Test listing conversations then loading one."""
        # Setup mock conversations
        conversations = [
            {
                "filename": "conversation1.json",
                "created_at": "2024-01-01T12:00:00",
                "message_count": 5,
                "current_model": "gpt-4",
            }
        ]
        mock_engine.list_conversations.return_value = conversations

        # Test list
        with patch.object(cli_interface.console, "print"):
            await cli_interface._show_conversations()
            mock_engine.list_conversations.assert_called_once()

        # Test load
        load_command = Command.create_slash_command(
            SlashCommand.LOAD,
            ["conversation1.json"],
            "/load conversation1.json",
        )

        with patch.object(cli_interface, "_show_info"):
            await cli_interface._handle_load_command(load_command)
            mock_engine.load_conversation.assert_called_once_with("conversation1.json")


class TestConversationValidation:
    """Test conversation command validation."""

    @pytest.mark.asyncio
    async def test_save_command_validation(self, cli_interface):
        """Test save command argument validation."""
        # Valid cases - no args (auto-generate filename)
        command1 = Command.create_slash_command(SlashCommand.SAVE, [], "/save")
        # Valid cases - with filename
        command2 = Command.create_slash_command(
            SlashCommand.SAVE, ["test"], "/save test"
        )

        # Both should work without validation errors
        with patch.object(cli_interface, "_show_info"):
            await cli_interface._handle_save_command(command1)
            await cli_interface._handle_save_command(command2)

    @pytest.mark.asyncio
    async def test_load_command_validation(self, cli_interface):
        """Test load command argument validation."""
        # Invalid case - no filename
        command = Command.create_slash_command(SlashCommand.LOAD, [], "/load")

        with (
            patch.object(cli_interface, "_show_conversations"),
            patch.object(cli_interface, "_show_error") as mock_show_error,
        ):
            await cli_interface._handle_load_command(command)
            mock_show_error.assert_called_once_with("Usage: /load <filename>")


class TestConversationDisplay:
    """Test conversation display formatting."""

    @pytest.mark.asyncio
    async def test_conversation_list_formatting(self, cli_interface, mock_engine):
        """Test that conversation list is properly formatted."""
        conversations = [
            {
                "filename": "test.json",
                "created_at": "2024-01-01T12:00:00",
                "message_count": 10,
                "current_model": "gpt-4",
            }
        ]
        mock_engine.list_conversations.return_value = conversations

        with patch.object(cli_interface.console, "print") as mock_print:
            await cli_interface._show_conversations()

            # Verify the panel was created with proper content
            mock_print.assert_called_once()
            panel = mock_print.call_args[0][0]

            # Check that the content includes expected information
            content = str(panel.renderable)
            assert "test.json" in content
            assert "Messages: 10" in content
            assert "Model: gpt-4" in content

    @pytest.mark.asyncio
    async def test_date_formatting(self, cli_interface, mock_engine):
        """Test that dates are properly formatted in conversation list."""
        conversations = [
            {
                "filename": "test.json",
                "created_at": "2024-01-01T12:30:45",
                "message_count": 5,
                "current_model": "gpt-4",
            }
        ]
        mock_engine.list_conversations.return_value = conversations

        with patch.object(cli_interface.console, "print") as mock_print:
            await cli_interface._show_conversations()

            panel = mock_print.call_args[0][0]
            content = str(panel.renderable)

            # Should format date as YYYY-MM-DD HH:MM
            assert "2024-01-01 12:30" in content
