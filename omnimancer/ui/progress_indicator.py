"""
Progress indicator system for Omnimancer.

This module provides real-time progress indicators showing current operations
and tool usage during AI interactions.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class OperationType(Enum):
    """Types of operations that can be tracked."""

    THINKING = "thinking"
    READ = "read"
    WRITE = "write"
    BASH = "bash"
    RESEARCH = "research"
    GREP = "grep"
    EDIT = "edit"
    VALIDATE = "validate"
    ANALYZE = "analyze"
    GENERATE = "generate"
    NETWORK = "network"
    OTHER = "other"


@dataclass
class OperationInfo:
    """Information about a current operation."""

    operation_type: OperationType
    description: str
    start_time: float = field(default_factory=time.time)
    progress: Optional[float] = None  # 0.0 to 1.0
    details: Optional[str] = None
    status: str = "running"  # "running", "completed", "failed"


class ProgressIndicator:
    """
    Real-time progress indicator for Omnimancer operations.

    Shows current operations, tool usage, and progress information
    in a live-updating display.
    """

    # Operation type icons and colors
    OPERATION_ICONS = {
        OperationType.THINKING: ("🤔", "yellow"),
        OperationType.READ: ("📖", "blue"),
        OperationType.WRITE: ("✍️", "green"),
        OperationType.BASH: ("⚙️", "magenta"),
        OperationType.RESEARCH: ("🔍", "cyan"),
        OperationType.GREP: ("🔎", "blue"),
        OperationType.EDIT: ("📝", "green"),
        OperationType.VALIDATE: ("✅", "green"),
        OperationType.ANALYZE: ("🔬", "purple"),
        OperationType.GENERATE: ("🏗️", "orange"),
        OperationType.NETWORK: ("🌐", "cyan"),
        OperationType.OTHER: ("⚡", "white"),
    }

    def __init__(self, console: Console):
        """
        Initialize the progress indicator.

        Args:
            console: Rich Console instance
        """
        self.console = console
        self.current_operations: Dict[str, OperationInfo] = {}
        self.operation_history: List[OperationInfo] = []
        self.live_display: Optional[Live] = None
        self.display_task: Optional[asyncio.Task] = None
        self.enabled = True
        self.max_history = 5

    def start_operation(
        self,
        operation_id: str,
        operation_type: OperationType,
        description: str,
        details: Optional[str] = None,
    ) -> None:
        """
        Start tracking a new operation.

        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation
            description: Human-readable description
            details: Optional additional details
        """
        if not self.enabled:
            return

        operation = OperationInfo(
            operation_type=operation_type,
            description=description,
            details=details,
        )
        self.current_operations[operation_id] = operation

        # Start live display if this is the first operation
        if len(self.current_operations) == 1 and self.live_display is None:
            self.start_live_display()
        else:
            self._update_display()

    def update_operation(
        self,
        operation_id: str,
        progress: Optional[float] = None,
        details: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        """
        Update an existing operation.

        Args:
            operation_id: Operation identifier
            progress: Progress as float 0.0 to 1.0
            details: Updated details
            status: Updated status
        """
        if not self.enabled or operation_id not in self.current_operations:
            return

        operation = self.current_operations[operation_id]
        if progress is not None:
            operation.progress = progress
        if details is not None:
            operation.details = details
        if status is not None:
            operation.status = status

        self._update_display()

    def complete_operation(self, operation_id: str, status: str = "completed") -> None:
        """
        Complete an operation and move it to history.

        Args:
            operation_id: Operation identifier
            status: Final status ("completed" or "failed")
        """
        if not self.enabled or operation_id not in self.current_operations:
            return

        operation = self.current_operations.pop(operation_id)
        operation.status = status

        # Add to history (keep only recent ones)
        self.operation_history.append(operation)
        if len(self.operation_history) > self.max_history:
            self.operation_history.pop(0)

        # Stop live display if no more operations
        if len(self.current_operations) == 0:
            self.stop_live_display()
        else:
            self._update_display()

    def clear_all_operations(self) -> None:
        """Clear all current operations."""
        if not self.enabled:
            return

        # Move all current operations to history as completed
        for operation_id, operation in self.current_operations.items():
            operation.status = "completed"
            self.operation_history.append(operation)

        self.current_operations.clear()

        # Trim history
        if len(self.operation_history) > self.max_history:
            self.operation_history = self.operation_history[-self.max_history :]

        # Always stop live display when clearing all operations
        self.stop_live_display()

    def start_live_display(self) -> None:
        """Start the live updating display."""
        if not self.enabled or self.live_display is not None:
            return

        layout = self._create_layout()
        self.live_display = Live(
            layout,
            console=self.console,
            auto_refresh=False,  # Disable auto-refresh to prevent runaway updates
            refresh_per_second=2,  # Limit refresh rate
        )
        self.live_display.start()

    def stop_live_display(self) -> None:
        """Stop the live updating display."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

    def _create_layout(self) -> Panel:
        """Create the display layout."""
        # Create table for current operations
        current_table = Table(show_header=False, box=None, padding=(0, 1))
        current_table.add_column("", style="bold")
        current_table.add_column("", style="dim")
        current_table.add_column("", style="")

        for operation_id, operation in self.current_operations.items():
            icon, color = self.OPERATION_ICONS[operation.operation_type]
            elapsed = time.time() - operation.start_time

            # Format time
            if elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            else:
                time_str = f"{elapsed/60:.1f}m"

            # Progress indicator
            if operation.progress is not None:
                progress_str = f" ({operation.progress:.0%})"
            else:
                progress_str = ""

            # Operation details
            details_str = f" - {operation.details}" if operation.details else ""

            current_table.add_row(
                f"[{color}]{icon}[/{color}]",
                f"[{color}]{operation.operation_type.value.title()}[/{color}]",
                f"{operation.description}{progress_str}{details_str} [{time_str}]",
            )

        return Panel(
            current_table,
            title="🤖 Omnimancer Progress",
            border_style="blue",
            padding=(0, 1),
        )

    def _update_display(self) -> None:
        """Update the live display."""
        if self.live_display:
            self.live_display.update(self._create_layout())

    def enable(self) -> None:
        """Enable progress tracking."""
        self.enabled = True

    def disable(self) -> None:
        """Disable progress tracking."""
        self.enabled = False
        self.stop_live_display()

    def show_simple_status(self, message: str) -> None:
        """Show a simple status message without full progress tracking."""
        if not self.enabled:
            return

        self.console.print(f"🤖 {message}", style="blue")


# Global progress indicator instance
_global_progress_indicator: Optional[ProgressIndicator] = None


def get_progress_indicator() -> Optional[ProgressIndicator]:
    """Get the global progress indicator instance."""
    return _global_progress_indicator


def set_progress_indicator(indicator: ProgressIndicator) -> None:
    """Set the global progress indicator instance."""
    global _global_progress_indicator
    _global_progress_indicator = indicator


def start_operation(
    operation_id: str,
    operation_type: OperationType,
    description: str,
    details: Optional[str] = None,
) -> None:
    """Global function to start tracking an operation."""
    indicator = get_progress_indicator()
    if indicator:
        indicator.start_operation(operation_id, operation_type, description, details)


def update_operation(
    operation_id: str,
    progress: Optional[float] = None,
    details: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    """Global function to update an operation."""
    indicator = get_progress_indicator()
    if indicator:
        indicator.update_operation(operation_id, progress, details, status)


def complete_operation(operation_id: str, status: str = "completed") -> None:
    """Global function to complete an operation."""
    indicator = get_progress_indicator()
    if indicator:
        indicator.complete_operation(operation_id, status)
