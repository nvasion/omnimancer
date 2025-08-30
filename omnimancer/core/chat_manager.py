"""
Chat manager for Omnimancer.

This module handles conversation management, context tracking,
and message history for chat sessions.
"""

import uuid
from datetime import datetime
from typing import Optional

from .models import ChatContext, ChatMessage, MessageRole


class ChatManager:
    """
    Manages chat conversations and context.

    This class handles the conversation state, message history,
    and context management for chat sessions.
    """

    def __init__(self):
        """Initialize the chat manager."""
        self.current_context: Optional[ChatContext] = None
        self._initialize_context()

    def _initialize_context(self) -> None:
        """Initialize a new chat context."""
        self.current_context = ChatContext(
            messages=[],
            current_model="",
            session_id=str(uuid.uuid4()),
        )

    def get_current_context(self) -> ChatContext:
        """
        Get the current chat context.

        Returns:
            Current ChatContext instance
        """
        if self.current_context is None:
            self._initialize_context()
        return self.current_context

    def set_current_model(self, model: str) -> None:
        """
        Set the current model being used.

        Args:
            model: Model name
        """
        context = self.get_current_context()
        context.current_model = model

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the current context.

        Args:
            content: Message content
        """
        context = self.get_current_context()
        message = ChatMessage(
            role=MessageRole.USER,
            content=content,
            timestamp=datetime.now(),
            model_used=context.current_model,
        )
        context.add_message(message)

    def add_assistant_message(self, content: str, model_used: str) -> None:
        """
        Add an assistant message to the current context.

        Args:
            content: Message content
            model_used: Model that generated the message
        """
        context = self.get_current_context()
        message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            timestamp=datetime.now(),
            model_used=model_used,
        )
        context.add_message(message)

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the current context.

        Args:
            content: Message content
        """
        context = self.get_current_context()
        message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=content,
            timestamp=datetime.now(),
            model_used=context.current_model,
        )
        context.add_message(message)

    def clear_context(self) -> None:
        """Clear the current conversation context."""
        if self.current_context:
            self.current_context.clear()

    def start_new_session(self) -> str:
        """
        Start a new chat session.

        Returns:
            New session ID
        """
        self._initialize_context()
        return self.current_context.session_id

    def get_message_count(self) -> int:
        """
        Get the number of messages in the current context.

        Returns:
            Message count
        """
        context = self.get_current_context()
        return len(context.messages)

    def get_last_message(self) -> Optional[ChatMessage]:
        """
        Get the last message in the conversation.

        Returns:
            Last ChatMessage or None if no messages
        """
        context = self.get_current_context()
        if context.messages:
            return context.messages[-1]
        return None
