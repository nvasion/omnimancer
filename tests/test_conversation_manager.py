"""
Tests for the conversation manager module.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from omnimancer.core.conversation_manager import ConversationManager
from omnimancer.core.models import ChatContext, ChatMessage, MessageRole
from omnimancer.utils.errors import ConversationError


@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def conversation_manager(temp_storage_path):
    """Create ConversationManager instance for testing."""
    return ConversationManager(temp_storage_path)


@pytest.fixture
def sample_chat_context():
    """Create a sample ChatContext for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Hello, how are you?",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            model_used=None,
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="I'm doing well, thank you for asking!",
            timestamp=datetime(2024, 1, 1, 12, 0, 5),
            model_used="gpt-4",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="What can you help me with?",
            timestamp=datetime(2024, 1, 1, 12, 0, 10),
            model_used=None,
        ),
    ]

    return ChatContext(
        messages=messages,
        current_model="gpt-4",
        session_id="test-session-123",
        max_context_length=4000,
    )


class TestConversationManagerInitialization:
    """Test ConversationManager initialization."""

    def test_init_creates_directories(self, temp_storage_path):
        """Test that initialization creates required directories."""
        manager = ConversationManager(temp_storage_path)

        assert manager.storage_path == temp_storage_path
        assert manager.conversations_dir == temp_storage_path / "conversations"
        assert manager.conversations_dir.exists()

    def test_init_with_existing_directories(self, temp_storage_path):
        """Test initialization with existing directories."""
        # Pre-create the directory
        conversations_dir = temp_storage_path / "conversations"
        conversations_dir.mkdir(parents=True, exist_ok=True)

        manager = ConversationManager(temp_storage_path)

        assert manager.conversations_dir.exists()


class TestSaveConversation:
    """Test conversation saving functionality."""

    def test_save_conversation_with_auto_filename(
        self, conversation_manager, sample_chat_context
    ):
        """Test saving conversation with auto-generated filename."""
        with patch("omnimancer.core.conversation_manager.datetime") as mock_datetime:
            mock_now = Mock()
            mock_now.strftime.return_value = "20240101_123045"
            mock_now.isoformat.return_value = "2024-01-01T12:30:45"
            mock_datetime.now.return_value = mock_now

            filename = conversation_manager.save_conversation(sample_chat_context)

            assert filename == "conversation_20240101_123045.json"

            # Verify file was created
            file_path = conversation_manager.conversations_dir / filename
            assert file_path.exists()

    def test_save_conversation_with_custom_filename(
        self, conversation_manager, sample_chat_context
    ):
        """Test saving conversation with custom filename."""
        filename = conversation_manager.save_conversation(
            sample_chat_context, "my_conversation"
        )

        assert filename == "my_conversation.json"

        # Verify file was created
        file_path = conversation_manager.conversations_dir / filename
        assert file_path.exists()

    def test_save_conversation_with_json_extension(
        self, conversation_manager, sample_chat_context
    ):
        """Test saving conversation with .json extension already provided."""
        filename = conversation_manager.save_conversation(
            sample_chat_context, "test.json"
        )

        assert filename == "test.json"

        # Verify file was created
        file_path = conversation_manager.conversations_dir / filename
        assert file_path.exists()

    def test_save_conversation_with_metadata(
        self, conversation_manager, sample_chat_context
    ):
        """Test saving conversation with metadata."""
        metadata = {"tags": ["test", "sample"], "priority": "high"}

        filename = conversation_manager.save_conversation(
            sample_chat_context, "test_with_metadata", metadata
        )

        # Verify metadata was saved
        file_path = conversation_manager.conversations_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"] == metadata

    def test_save_conversation_content_structure(
        self, conversation_manager, sample_chat_context
    ):
        """Test that saved conversation has correct structure."""
        filename = conversation_manager.save_conversation(
            sample_chat_context, "structure_test"
        )

        file_path = conversation_manager.conversations_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify structure
        assert data["version"] == "1.0"
        assert "created_at" in data
        assert data["session_id"] == "test-session-123"
        assert data["current_model"] == "gpt-4"
        assert data["max_context_length"] == 4000
        assert len(data["messages"]) == 3

        # Verify message structure
        msg = data["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "Hello, how are you?"
        assert "timestamp" in msg
        assert msg["model_used"] is None

    def test_save_conversation_file_error(
        self, conversation_manager, sample_chat_context
    ):
        """Test handling of file write errors."""
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(ConversationError, match="Failed to save conversation"):
                conversation_manager.save_conversation(
                    sample_chat_context, "error_test"
                )


class TestLoadConversation:
    """Test conversation loading functionality."""

    def test_load_conversation_success(self, conversation_manager, sample_chat_context):
        """Test successful conversation loading."""
        # First save a conversation
        conversation_manager.save_conversation(sample_chat_context, "load_test")

        # Then load it
        loaded_context = conversation_manager.load_conversation("load_test")

        # Verify loaded data
        assert loaded_context.session_id == sample_chat_context.session_id
        assert loaded_context.current_model == sample_chat_context.current_model
        assert (
            loaded_context.max_context_length == sample_chat_context.max_context_length
        )
        assert len(loaded_context.messages) == len(sample_chat_context.messages)

        # Verify first message
        loaded_msg = loaded_context.messages[0]
        original_msg = sample_chat_context.messages[0]
        assert loaded_msg.role == original_msg.role
        assert loaded_msg.content == original_msg.content
        assert loaded_msg.model_used == original_msg.model_used

    def test_load_conversation_with_json_extension(
        self, conversation_manager, sample_chat_context
    ):
        """Test loading conversation with .json extension."""
        # Save conversation
        conversation_manager.save_conversation(sample_chat_context, "extension_test")

        # Load with .json extension
        loaded_context = conversation_manager.load_conversation("extension_test.json")

        assert loaded_context.session_id == sample_chat_context.session_id

    def test_load_conversation_file_not_found(self, conversation_manager):
        """Test loading non-existent conversation."""
        with pytest.raises(ConversationError, match="Conversation file not found"):
            conversation_manager.load_conversation("nonexistent")

    def test_load_conversation_invalid_version(self, conversation_manager):
        """Test loading conversation with unsupported version."""
        # Create file with invalid version
        invalid_data = {"version": "2.0", "messages": []}

        file_path = conversation_manager.conversations_dir / "invalid_version.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(invalid_data, f)

        with pytest.raises(
            ConversationError, match="Unsupported conversation file version"
        ):
            conversation_manager.load_conversation("invalid_version")

    def test_load_conversation_corrupted_file(self, conversation_manager):
        """Test loading corrupted conversation file."""
        # Create corrupted file
        file_path = conversation_manager.conversations_dir / "corrupted.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("invalid json content")

        with pytest.raises(ConversationError, match="Failed to load conversation"):
            conversation_manager.load_conversation("corrupted")


class TestListConversations:
    """Test conversation listing functionality."""

    def test_list_conversations_empty(self, conversation_manager):
        """Test listing conversations when directory is empty."""
        conversations = conversation_manager.list_conversations()
        assert conversations == []

    def test_list_conversations_with_files(
        self, conversation_manager, sample_chat_context
    ):
        """Test listing conversations with existing files."""
        # Save multiple conversations
        conversation_manager.save_conversation(sample_chat_context, "conv1")
        conversation_manager.save_conversation(sample_chat_context, "conv2")

        conversations = conversation_manager.list_conversations()

        assert len(conversations) == 2
        filenames = [conv["filename"] for conv in conversations]
        assert "conv1.json" in filenames
        assert "conv2.json" in filenames

        # Verify conversation info structure
        conv = conversations[0]
        assert "filename" in conv
        assert "created_at" in conv
        assert "session_id" in conv
        assert "current_model" in conv
        assert "message_count" in conv
        assert "file_size" in conv
        assert "modified_at" in conv

    def test_list_conversations_sorted_by_date(
        self, conversation_manager, sample_chat_context
    ):
        """Test that conversations are sorted by creation date."""
        with patch("omnimancer.core.conversation_manager.datetime") as mock_datetime:
            # First conversation
            mock_now_1 = Mock()
            mock_now_1.isoformat.return_value = "2024-01-01T12:00:00"
            mock_datetime.now.return_value = mock_now_1
            conversation_manager.save_conversation(sample_chat_context, "older")

            # Second conversation (newer)
            mock_now_2 = Mock()
            mock_now_2.isoformat.return_value = "2024-01-02T12:00:00"
            mock_datetime.now.return_value = mock_now_2
            conversation_manager.save_conversation(sample_chat_context, "newer")

        conversations = conversation_manager.list_conversations()

        # Should be sorted with newest first
        assert conversations[0]["filename"] == "newer.json"
        assert conversations[1]["filename"] == "older.json"

    def test_list_conversations_skips_invalid_files(
        self, conversation_manager, sample_chat_context
    ):
        """Test that invalid files are skipped during listing."""
        # Save valid conversation
        conversation_manager.save_conversation(sample_chat_context, "valid")

        # Create invalid file
        invalid_path = conversation_manager.conversations_dir / "invalid.json"
        with open(invalid_path, "w", encoding="utf-8") as f:
            f.write("invalid json")

        conversations = conversation_manager.list_conversations()

        # Should only include valid conversation
        assert len(conversations) == 1
        assert conversations[0]["filename"] == "valid.json"


class TestDeleteConversation:
    """Test conversation deletion functionality."""

    def test_delete_conversation_success(
        self, conversation_manager, sample_chat_context
    ):
        """Test successful conversation deletion."""
        # Save conversation
        filename = conversation_manager.save_conversation(
            sample_chat_context, "delete_test"
        )

        # Verify file exists
        file_path = conversation_manager.conversations_dir / filename
        assert file_path.exists()

        # Delete conversation
        result = conversation_manager.delete_conversation("delete_test")

        assert result is True
        assert not file_path.exists()

    def test_delete_conversation_with_json_extension(
        self, conversation_manager, sample_chat_context
    ):
        """Test deleting conversation with .json extension."""
        # Save conversation
        conversation_manager.save_conversation(sample_chat_context, "delete_ext_test")

        # Delete with .json extension
        result = conversation_manager.delete_conversation("delete_ext_test.json")

        assert result is True

    def test_delete_conversation_not_found(self, conversation_manager):
        """Test deleting non-existent conversation."""
        result = conversation_manager.delete_conversation("nonexistent")
        assert result is False

    def test_delete_conversation_permission_error(
        self, conversation_manager, sample_chat_context
    ):
        """Test handling permission errors during deletion."""
        # Save conversation
        conversation_manager.save_conversation(sample_chat_context, "perm_test")

        with patch.object(Path, "unlink", side_effect=PermissionError("Access denied")):
            result = conversation_manager.delete_conversation("perm_test")
            assert result is False


class TestGetConversationInfo:
    """Test getting conversation information."""

    def test_get_conversation_info_success(
        self, conversation_manager, sample_chat_context
    ):
        """Test getting conversation info successfully."""
        # Save conversation with metadata
        metadata = {"tags": ["test"], "priority": "high"}
        filename = conversation_manager.save_conversation(
            sample_chat_context, "info_test", metadata
        )

        info = conversation_manager.get_conversation_info("info_test")

        assert info is not None
        assert info["filename"] == filename
        assert info["session_id"] == "test-session-123"
        assert info["current_model"] == "gpt-4"
        assert info["message_count"] == 3
        assert info["metadata"] == metadata
        assert "created_at" in info
        assert "file_size" in info
        assert "modified_at" in info

    def test_get_conversation_info_with_json_extension(
        self, conversation_manager, sample_chat_context
    ):
        """Test getting info with .json extension."""
        conversation_manager.save_conversation(sample_chat_context, "info_ext_test")

        info = conversation_manager.get_conversation_info("info_ext_test.json")

        assert info is not None
        assert info["filename"] == "info_ext_test.json"

    def test_get_conversation_info_not_found(self, conversation_manager):
        """Test getting info for non-existent conversation."""
        info = conversation_manager.get_conversation_info("nonexistent")
        assert info is None

    def test_get_conversation_info_corrupted_file(self, conversation_manager):
        """Test getting info for corrupted file."""
        # Create corrupted file
        file_path = conversation_manager.conversations_dir / "corrupted_info.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("invalid json")

        info = conversation_manager.get_conversation_info("corrupted_info")
        assert info is None


class TestExportConversation:
    """Test conversation export functionality."""

    def test_export_conversation_json(self, conversation_manager, sample_chat_context):
        """Test exporting conversation to JSON format."""
        # Save conversation
        conversation_manager.save_conversation(sample_chat_context, "export_json_test")

        # Export to JSON (should return original file path)
        export_path = conversation_manager.export_conversation(
            "export_json_test", "json"
        )

        # The method returns the path using the input filename, not the saved filename
        expected_path = str(conversation_manager.conversations_dir / "export_json_test")
        assert export_path == expected_path

    def test_export_conversation_txt(self, conversation_manager, sample_chat_context):
        """Test exporting conversation to TXT format."""
        # Save conversation
        conversation_manager.save_conversation(sample_chat_context, "export_txt_test")

        # Export to TXT
        export_path = conversation_manager.export_conversation("export_txt_test", "txt")

        # Verify export file was created
        export_file = Path(export_path)
        assert export_file.exists()
        assert export_file.suffix == ".txt"

        # Verify content
        content = export_file.read_text(encoding="utf-8")
        assert "Omnimancer Conversation Export" in content
        assert "Session ID: test-session-123" in content
        assert "Model: gpt-4" in content
        assert "Hello, how are you?" in content
        assert "I'm doing well, thank you for asking!" in content

    def test_export_conversation_markdown(
        self, conversation_manager, sample_chat_context
    ):
        """Test exporting conversation to Markdown format."""
        # Save conversation
        conversation_manager.save_conversation(sample_chat_context, "export_md_test")

        # Export to Markdown
        export_path = conversation_manager.export_conversation("export_md_test", "md")

        # Verify export file was created
        export_file = Path(export_path)
        assert export_file.exists()
        assert export_file.suffix == ".md"

        # Verify content
        content = export_file.read_text(encoding="utf-8")
        assert "# Omnimancer Conversation" in content
        assert "**Session ID:** test-session-123" in content
        assert "**Model:** gpt-4" in content
        assert "## 👤 User" in content
        assert "## 🤖 Assistant - gpt-4" in content
        assert "Hello, how are you?" in content

    def test_export_conversation_unsupported_format(
        self, conversation_manager, sample_chat_context
    ):
        """Test exporting to unsupported format."""
        # Save conversation
        conversation_manager.save_conversation(
            sample_chat_context, "export_unsupported_test"
        )

        # Try to export to unsupported format
        with pytest.raises(ConversationError, match="Unsupported export format"):
            conversation_manager.export_conversation("export_unsupported_test", "xml")

    def test_export_conversation_not_found(self, conversation_manager):
        """Test exporting non-existent conversation."""
        with pytest.raises(ConversationError, match="Conversation file not found"):
            conversation_manager.export_conversation("nonexistent", "txt")


class TestPrivateMethods:
    """Test private helper methods."""

    def test_export_to_txt_content(self, conversation_manager, sample_chat_context):
        """Test TXT export content formatting."""
        export_path = conversation_manager._export_to_txt(
            sample_chat_context, "test_txt"
        )

        export_file = Path(export_path)
        content = export_file.read_text(encoding="utf-8")

        # Verify header
        assert "Omnimancer Conversation Export" in content
        assert f"Session ID: {sample_chat_context.session_id}" in content
        assert f"Model: {sample_chat_context.current_model}" in content
        assert f"Messages: {len(sample_chat_context.messages)}" in content

        # Verify message formatting
        assert "USER:" in content
        assert "ASSISTANT (gpt-4):" in content
        assert "[2024-01-01 12:00:00]" in content

    def test_export_to_markdown_content(
        self, conversation_manager, sample_chat_context
    ):
        """Test Markdown export content formatting."""
        export_path = conversation_manager._export_to_markdown(
            sample_chat_context, "test_md"
        )

        export_file = Path(export_path)
        content = export_file.read_text(encoding="utf-8")

        # Verify header
        assert "# Omnimancer Conversation" in content
        assert f"**Session ID:** {sample_chat_context.session_id}" in content
        assert f"**Model:** {sample_chat_context.current_model}" in content

        # Verify message formatting
        assert "## 👤 User" in content
        assert "## 🤖 Assistant - gpt-4" in content
        assert "(2024-01-01 12:00:00)" in content


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_chat_context(self, conversation_manager):
        """Test saving empty chat context."""
        empty_context = ChatContext(
            messages=[],
            current_model="gpt-4",
            session_id="empty-session",
            max_context_length=4000,
        )

        filename = conversation_manager.save_conversation(empty_context, "empty_test")

        # Verify file was created
        file_path = conversation_manager.conversations_dir / filename
        assert file_path.exists()

        # Load and verify
        loaded_context = conversation_manager.load_conversation("empty_test")
        assert len(loaded_context.messages) == 0
        assert loaded_context.session_id == "empty-session"

    def test_special_characters_in_content(self, conversation_manager):
        """Test handling special characters in message content."""
        special_message = ChatMessage(
            role=MessageRole.USER,
            content="Hello! 🌟 This has émojis and spëcial chars: <>&\"'",
            timestamp=datetime.now(),
            model_used=None,
        )

        context = ChatContext(
            messages=[special_message],
            current_model="gpt-4",
            session_id="special-chars",
            max_context_length=4000,
        )

        conversation_manager.save_conversation(context, "special_chars_test")
        loaded_context = conversation_manager.load_conversation("special_chars_test")

        assert loaded_context.messages[0].content == special_message.content

    def test_directory_creation_failure(self, temp_storage_path):
        """Test handling directory creation failure."""
        with patch.object(Path, "mkdir", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                ConversationManager(temp_storage_path)
