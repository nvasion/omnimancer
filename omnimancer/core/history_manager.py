"""
Command History Manager for Omnimancer CLI.

This module provides command history functionality with persistent storage,
search capabilities, and arrow key navigation similar to claude-code.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HistoryEntry:
    """Represents a single command history entry."""

    def __init__(
        self,
        command: str,
        timestamp: Optional[float] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a history entry.

        Args:
            command: The command text
            timestamp: Unix timestamp (defaults to current time)
            session_id: Optional session identifier
            metadata: Additional metadata about the command
        """
        self.command = command.strip()
        self.timestamp = timestamp or time.time()
        self.session_id = session_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "command": self.command,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        """Create from dictionary."""
        return cls(
            command=data["command"],
            timestamp=data.get("timestamp"),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )

    @property
    def datetime(self) -> datetime:
        """Get datetime representation of timestamp."""
        return datetime.fromtimestamp(self.timestamp)

    def __str__(self) -> str:
        return self.command

    def __repr__(self) -> str:
        return f"HistoryEntry(command='{self.command[:50]}...', timestamp={self.timestamp})"


class HistoryManager:
    """
    Manages command history with persistent storage and search capabilities.

    Features:
    - Persistent storage in JSON format
    - Configurable history size limits
    - Search and filtering
    - Session tracking
    - Automatic cleanup of old entries
    - Duplicate prevention
    """

    def __init__(
        self,
        storage_path: str = "~/.omnimancer",
        max_entries: int = 1000,
        max_age_days: int = 90,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the history manager.

        Args:
            storage_path: Directory for history storage
            max_entries: Maximum number of history entries to keep
            max_age_days: Maximum age in days before entries are cleaned up
            session_id: Optional session identifier
        """
        self.storage_path = Path(storage_path).expanduser()
        self.history_file = self.storage_path / "command_history.json"
        self.max_entries = max_entries
        self.max_age_days = max_age_days
        self.session_id = session_id or self._generate_session_id()

        # In-memory history for current session
        self._history: List[HistoryEntry] = []
        self._current_index = 0  # For arrow key navigation
        self._search_filter = ""
        self._filtered_history: List[HistoryEntry] = []

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing history
        self._load_history()

        # Clean up old entries
        self._cleanup_old_entries()

    def _generate_session_id(self) -> str:
        """Generate a unique session identifier."""
        return f"session_{int(time.time())}_{os.getpid()}"

    def add_command(
        self, command: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a command to history.

        Args:
            command: Command text to add
            metadata: Optional metadata about the command
        """
        command = command.strip()

        # Skip empty commands
        if not command:
            return

        # Skip duplicate consecutive commands
        if self._history and self._history[-1].command == command:
            return

        # Create history entry
        entry = HistoryEntry(
            command=command, session_id=self.session_id, metadata=metadata
        )

        # Add to in-memory history
        self._history.append(entry)

        # Reset navigation index
        self._current_index = len(self._history)

        # Trim history if it exceeds max size
        if len(self._history) > self.max_entries:
            self._history = self._history[-self.max_entries :]

        # Save to disk (async would be better but keeping simple for now)
        self._save_history()

        logger.debug(f"Added command to history: {command[:50]}...")

    def get_previous_command(self) -> Optional[str]:
        """
        Get the previous command in history (up arrow functionality).

        Returns:
            Previous command or None if at beginning
        """
        if not self._get_filtered_history():
            return None

        if self._current_index > 0:
            self._current_index -= 1
            return self._get_filtered_history()[self._current_index].command

        return None

    def get_next_command(self) -> Optional[str]:
        """
        Get the next command in history (down arrow functionality).

        Returns:
            Next command or None if at end
        """
        filtered = self._get_filtered_history()
        if not filtered:
            return None

        if self._current_index < len(filtered) - 1:
            self._current_index += 1
            return filtered[self._current_index].command
        else:
            # Go to end (empty command)
            self._current_index = len(filtered)
            return ""

    def search_history(self, query: str, limit: int = 50) -> List[HistoryEntry]:
        """
        Search command history.

        Args:
            query: Search query (case insensitive)
            limit: Maximum number of results

        Returns:
            List of matching history entries
        """
        query = query.lower()
        results = []

        # Search in reverse order (most recent first)
        for entry in reversed(self._history):
            if query in entry.command.lower():
                results.append(entry)
                if len(results) >= limit:
                    break

        return results

    def set_search_filter(self, filter_text: str) -> None:
        """
        Set a search filter for arrow key navigation.

        Args:
            filter_text: Text to filter commands by
        """
        self._search_filter = filter_text.lower()
        self._filtered_history = []  # Will be rebuilt on next access

        # Reset navigation to end of filtered results
        filtered = self._get_filtered_history()
        self._current_index = len(filtered)

    def clear_search_filter(self) -> None:
        """Clear the current search filter."""
        self._search_filter = ""
        self._filtered_history = []
        self._current_index = len(self._history)

    def _get_filtered_history(self) -> List[HistoryEntry]:
        """Get history filtered by current search filter."""
        if not self._search_filter:
            return self._history

        # Rebuild filtered history if needed
        if not self._filtered_history:
            self._filtered_history = [
                entry
                for entry in self._history
                if self._search_filter in entry.command.lower()
            ]

        return self._filtered_history

    def get_recent_commands(self, limit: int = 20) -> List[HistoryEntry]:
        """
        Get recent commands.

        Args:
            limit: Maximum number of commands to return

        Returns:
            List of recent history entries
        """
        return self._history[-limit:] if self._history else []

    def get_session_commands(
        self, session_id: Optional[str] = None
    ) -> List[HistoryEntry]:
        """
        Get commands from a specific session.

        Args:
            session_id: Session ID (defaults to current session)

        Returns:
            List of commands from the session
        """
        target_session = session_id or self.session_id
        return [entry for entry in self._history if entry.session_id == target_session]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get history statistics.

        Returns:
            Dictionary with history statistics
        """
        if not self._history:
            return {
                "total_commands": 0,
                "current_session_commands": 0,
                "oldest_entry": None,
                "newest_entry": None,
                "unique_commands": 0,
            }

        unique_commands = len(set(entry.command for entry in self._history))
        current_session_count = len(self.get_session_commands())

        return {
            "total_commands": len(self._history),
            "current_session_commands": current_session_count,
            "oldest_entry": self._history[0].datetime.isoformat(),
            "newest_entry": self._history[-1].datetime.isoformat(),
            "unique_commands": unique_commands,
            "sessions": len(
                set(entry.session_id for entry in self._history if entry.session_id)
            ),
        }

    def clear_history(self, confirm: bool = False) -> bool:
        """
        Clear all command history.

        Args:
            confirm: Must be True to actually clear

        Returns:
            True if history was cleared
        """
        if not confirm:
            return False

        self._history.clear()
        self._current_index = 0
        self._filtered_history.clear()

        # Remove history file
        if self.history_file.exists():
            self.history_file.unlink()

        logger.info("Command history cleared")
        return True

    def export_history(self, filepath: str, format: str = "json") -> bool:
        """
        Export history to a file.

        Args:
            filepath: Path to export file
            format: Export format ('json' or 'txt')

        Returns:
            True if export was successful
        """
        try:
            export_path = Path(filepath)

            if format.lower() == "json":
                data = {
                    "exported_at": datetime.now().isoformat(),
                    "session_id": self.session_id,
                    "history": [entry.to_dict() for entry in self._history],
                }

                with open(export_path, "w") as f:
                    json.dump(data, f, indent=2)

            elif format.lower() == "txt":
                with open(export_path, "w") as f:
                    f.write(f"# Omnimancer Command History Export\n")
                    f.write(f"# Exported at: {datetime.now().isoformat()}\n")
                    f.write(f"# Session ID: {self.session_id}\n\n")

                    for entry in self._history:
                        f.write(f"# {entry.datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"{entry.command}\n\n")

            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"History exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False

    def _load_history(self) -> None:
        """Load history from disk."""
        try:
            if not self.history_file.exists():
                return

            with open(self.history_file, "r") as f:
                data = json.load(f)

            # Handle different file formats
            if isinstance(data, list):
                # Old format - list of commands
                for item in data:
                    if isinstance(item, str):
                        self._history.append(HistoryEntry(item))
                    elif isinstance(item, dict):
                        self._history.append(HistoryEntry.from_dict(item))

            elif isinstance(data, dict) and "history" in data:
                # New format - structured data
                for entry_data in data["history"]:
                    self._history.append(HistoryEntry.from_dict(entry_data))

            # Set navigation index to end
            self._current_index = len(self._history)

            logger.debug(f"Loaded {len(self._history)} history entries")

        except Exception as e:
            logger.error(f"Failed to load command history: {e}")
            # Continue with empty history
            self._history = []

    def _save_history(self) -> None:
        """Save history to disk."""
        try:
            data = {
                "format_version": "1.0",
                "saved_at": datetime.now().isoformat(),
                "session_id": self.session_id,
                "history": [entry.to_dict() for entry in self._history],
            }

            # Write atomically
            temp_file = self.history_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)

            temp_file.replace(self.history_file)

        except Exception as e:
            logger.error(f"Failed to save command history: {e}")

    def _cleanup_old_entries(self) -> None:
        """Remove entries older than max_age_days."""
        if not self._history or self.max_age_days <= 0:
            return

        cutoff_time = time.time() - (self.max_age_days * 24 * 60 * 60)
        original_count = len(self._history)

        self._history = [
            entry for entry in self._history if entry.timestamp > cutoff_time
        ]

        removed_count = original_count - len(self._history)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old history entries")
            self._save_history()

    def __len__(self) -> int:
        """Return number of history entries."""
        return len(self._history)

    def __bool__(self) -> bool:
        """Return True if history is not empty."""
        return bool(self._history)
