"""
Conversation persistence manager for Omnimancer.

This module handles saving, loading, and managing conversation histories
including serialization, file management, and conversation organization.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from ..utils.errors import ConversationError
from .models import ChatContext, ChatMessage, MessageRole


class ConversationManager:
    """
    Manages conversation persistence and file operations.

    This class handles saving conversations to files, loading them back,
    and managing conversation file organization.
    """

    def __init__(self, storage_path):
        """
        Initialize the conversation manager.

        Args:
            storage_path: Base path for storing conversation files (str or Path)
        """
        from pathlib import Path

        # Always ensure Path object and expand user home directory
        if isinstance(storage_path, str):
            self.storage_path = Path(storage_path).expanduser()
        else:
            self.storage_path = Path(storage_path).expanduser()
        self.conversations_dir = self.storage_path / "conversations"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def save_conversation(
        self,
        context: ChatContext,
        filename: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save a conversation to a file.

        Args:
            context: Chat context to save
            filename: Optional filename (auto-generated if not provided)
            metadata: Optional metadata to include

        Returns:
            Filename of the saved conversation

        Raises:
            ConversationError: If saving fails
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.json"

            # Ensure .json extension
            if not filename.endswith(".json"):
                filename += ".json"

            # Prepare conversation data
            conversation_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "session_id": context.session_id,
                "current_model": context.current_model,
                "max_context_length": context.max_context_length,
                "metadata": metadata or {},
                "messages": [
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "model_used": msg.model_used,
                    }
                    for msg in context.messages
                ],
            }

            # Save to file
            file_path = self.conversations_dir / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            return filename

        except Exception as e:
            raise ConversationError(f"Failed to save conversation: {e}")

    def load_conversation(self, filename: str) -> ChatContext:
        """
        Load a conversation from a file.

        Args:
            filename: Name of the conversation file

        Returns:
            Loaded ChatContext

        Raises:
            ConversationError: If loading fails
        """
        try:
            # Ensure .json extension
            if not filename.endswith(".json"):
                filename += ".json"

            file_path = self.conversations_dir / filename

            if not file_path.exists():
                raise ConversationError(f"Conversation file not found: {filename}")

            # Load conversation data
            with open(file_path, "r", encoding="utf-8") as f:
                conversation_data = json.load(f)

            # Validate version
            version = conversation_data.get("version", "1.0")
            if version != "1.0":
                raise ConversationError(
                    f"Unsupported conversation file version: {version}"
                )

            # Reconstruct messages
            messages = []
            for msg_data in conversation_data.get("messages", []):
                message = ChatMessage(
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    model_used=msg_data["model_used"],
                )
                messages.append(message)

            # Create ChatContext
            context = ChatContext(
                messages=messages,
                current_model=conversation_data.get("current_model", ""),
                session_id=conversation_data.get("session_id", ""),
                max_context_length=conversation_data.get("max_context_length", 4000),
            )

            return context

        except ConversationError:
            raise
        except Exception as e:
            raise ConversationError(f"Failed to load conversation: {e}")

    def list_conversations(self) -> List[Dict[str, str]]:
        """
        List all available conversation files.

        Returns:
            List of conversation info dictionaries
        """
        conversations = []

        try:
            for file_path in self.conversations_dir.glob("*.json"):
                try:
                    # Get basic file info
                    stat = file_path.stat()

                    # Try to get conversation metadata
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    conversations.append(
                        {
                            "filename": file_path.name,
                            "created_at": data.get("created_at", ""),
                            "session_id": data.get("session_id", ""),
                            "current_model": data.get("current_model", ""),
                            "message_count": len(data.get("messages", [])),
                            "file_size": stat.st_size,
                            "modified_at": datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                        }
                    )

                except Exception:
                    # Skip files that can't be read
                    continue

            # Sort by creation date (newest first)
            conversations.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        except Exception:
            # Return empty list if directory can't be read
            pass

        return conversations

    def delete_conversation(self, filename: str) -> bool:
        """
        Delete a conversation file.

        Args:
            filename: Name of the conversation file

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Ensure .json extension
            if not filename.endswith(".json"):
                filename += ".json"

            file_path = self.conversations_dir / filename

            if file_path.exists():
                file_path.unlink()
                return True

            return False

        except Exception:
            return False

    def get_conversation_info(self, filename: str) -> Optional[Dict]:
        """
        Get information about a specific conversation.

        Args:
            filename: Name of the conversation file

        Returns:
            Conversation info dictionary or None if not found
        """
        try:
            # Ensure .json extension
            if not filename.endswith(".json"):
                filename += ".json"

            file_path = self.conversations_dir / filename

            if not file_path.exists():
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            stat = file_path.stat()

            return {
                "filename": filename,
                "created_at": data.get("created_at", ""),
                "session_id": data.get("session_id", ""),
                "current_model": data.get("current_model", ""),
                "message_count": len(data.get("messages", [])),
                "file_size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "metadata": data.get("metadata", {}),
            }

        except Exception:
            return None

    def export_conversation(
        self, filename: str, export_format: str = "json"
    ) -> Optional[str]:
        """
        Export a conversation to different formats.

        Args:
            filename: Name of the conversation file
            export_format: Export format ("json", "txt", "md")

        Returns:
            Path to exported file or None if failed
        """
        try:
            context = self.load_conversation(filename)

            base_name = filename.replace(".json", "")

            if export_format == "txt":
                return self._export_to_txt(context, base_name)
            elif export_format == "md":
                return self._export_to_markdown(context, base_name)
            elif export_format == "json":
                # Already in JSON format
                return str(self.conversations_dir / filename)
            else:
                raise ConversationError(f"Unsupported export format: {export_format}")

        except Exception as e:
            raise ConversationError(f"Failed to export conversation: {e}")

    def _export_to_txt(self, context: ChatContext, base_name: str) -> str:
        """Export conversation to plain text format."""
        export_path = self.conversations_dir / f"{base_name}.txt"

        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f"Omnimancer Conversation Export\n")
            f.write(f"Session ID: {context.session_id}\n")
            f.write(f"Model: {context.current_model}\n")
            f.write(f"Messages: {len(context.messages)}\n")
            f.write("=" * 50 + "\n\n")

            for msg in context.messages:
                f.write(f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] ")
                f.write(f"{msg.role.value.upper()}")
                if msg.model_used:
                    f.write(f" ({msg.model_used})")
                f.write(":\n")
                f.write(f"{msg.content}\n\n")

        return str(export_path)

    def _export_to_markdown(self, context: ChatContext, base_name: str) -> str:
        """Export conversation to Markdown format."""
        export_path = self.conversations_dir / f"{base_name}.md"

        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f"# Omnimancer Conversation\n\n")
            f.write(f"**Session ID:** {context.session_id}\n")
            f.write(f"**Model:** {context.current_model}\n")
            f.write(f"**Messages:** {len(context.messages)}\n\n")
            f.write("---\n\n")

            for msg in context.messages:
                timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                role = msg.role.value.title()

                if msg.role == MessageRole.USER:
                    f.write(f"## 👤 {role} ({timestamp})\n\n")
                elif msg.role == MessageRole.ASSISTANT:
                    model_info = f" - {msg.model_used}" if msg.model_used else ""
                    f.write(f"## 🤖 {role}{model_info} ({timestamp})\n\n")
                else:
                    f.write(f"## ⚙️ {role} ({timestamp})\n\n")

                f.write(f"{msg.content}\n\n")

        return str(export_path)
