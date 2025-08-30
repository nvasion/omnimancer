"""
Core Status Data Models and Types for Omnimancer Agents.

This module provides the consolidated core data structures, enums, and base classes
for the agent status tracking system. All other status modules should import
from this module to avoid circular dependencies.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set


class AgentStatus(Enum):
    """Agent status enumeration."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    INITIALIZING = "initializing"
    SHUTTING_DOWN = "shutting_down"


class OperationType(Enum):
    """Types of operations that can be tracked."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    DIRECTORY_CREATE = "directory_create"
    DIRECTORY_DELETE = "directory_delete"
    COMMAND_EXECUTION = "command_execution"
    WEB_REQUEST = "web_request"
    API_CALL = "api_call"
    APPROVAL_REQUEST = "approval_request"
    CONFIGURATION_CHANGE = "configuration_change"
    AGENT_SWITCH = "agent_switch"


class OperationStatus(Enum):
    """Status of individual operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


class EventType(Enum):
    """Types of agent events."""

    AGENT_STATE_CHANGED = "agent_state_changed"
    OPERATION_STARTED = "operation_started"
    OPERATION_PROGRESS = "operation_progress"
    OPERATION_COMPLETED = "operation_completed"
    OPERATION_FAILED = "operation_failed"
    OPERATION_CANCELLED = "operation_cancelled"
    ERROR_OCCURRED = "error_occurred"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"


class StreamPriority(Enum):
    """Priority levels for status stream events."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class StatusDisplayLevel(Enum):
    """Display detail levels for status information."""

    MINIMAL = "minimal"  # Only basic status
    STANDARD = "standard"  # Standard status with operations
    DETAILED = "detailed"  # Full detailed status
    DEBUG = "debug"  # Debug level with all events


class StatusUpdateFrequency(Enum):
    """Update frequency for status display."""

    REAL_TIME = 0.1  # 10 times per second
    FAST = 0.5  # 2 times per second
    NORMAL = 1.0  # 1 time per second
    SLOW = 2.0  # Every 2 seconds
    MINIMAL = 5.0  # Every 5 seconds


@dataclass
class AgentOperation:
    """Data class representing a single agent operation."""

    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: OperationType = OperationType.API_CALL
    description: str = ""
    status: OperationStatus = OperationStatus.PENDING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    progress_percentage: float = 0.0
    error_details: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    parent_operation_id: Optional[str] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """Get operation duration."""
        if self.end_time:
            return self.end_time - self.start_time
        elif self.status in [OperationStatus.RUNNING, OperationStatus.PENDING]:
            return datetime.now() - self.start_time
        return None

    @property
    def is_active(self) -> bool:
        """Check if operation is currently active."""
        return self.status in [
            OperationStatus.PENDING,
            OperationStatus.RUNNING,
            OperationStatus.WAITING_APPROVAL,
        ]

    @property
    def is_completed(self) -> bool:
        """Check if operation is completed (successfully or not)."""
        return self.status in [
            OperationStatus.COMPLETED,
            OperationStatus.FAILED,
            OperationStatus.CANCELLED,
        ]

    def update_progress(
        self, percentage: float, description: Optional[str] = None
    ) -> None:
        """Update operation progress."""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        if description:
            self.description = description
        if percentage >= 100.0 and self.status == OperationStatus.RUNNING:
            self.complete()

    def complete(self, result_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark operation as completed."""
        self.status = OperationStatus.COMPLETED
        self.end_time = datetime.now()
        self.progress_percentage = 100.0
        if result_metadata:
            self.metadata.update(result_metadata)

    def fail(
        self,
        error_details: str,
        error_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark operation as failed."""
        self.status = OperationStatus.FAILED
        self.end_time = datetime.now()
        self.error_details = error_details
        if error_metadata:
            self.metadata.update(error_metadata)

    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel the operation."""
        self.status = OperationStatus.CANCELLED
        self.end_time = datetime.now()
        if reason:
            self.error_details = f"Cancelled: {reason}"


@dataclass
class AgentEvent:
    """Data class representing an agent event."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.AGENT_STATE_CHANGED
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    operation_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    def __post_init__(self):
        """Ensure timestamp is always set."""
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class StatusStreamEvent:
    """Event wrapper for status streaming with priority and metadata."""

    event: AgentEvent
    priority: StreamPriority = StreamPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure timestamp is set."""
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class StreamMetrics:
    """Metrics for status stream performance."""

    events_processed: int = 0
    events_dropped: int = 0
    events_retried: int = 0
    listeners_count: int = 0
    queue_size: int = 0
    avg_processing_time: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)

    def update_processing_time(self, processing_time: float) -> None:
        """Update average processing time with new sample."""
        if self.events_processed == 0:
            self.avg_processing_time = processing_time
        else:
            # Simple moving average
            self.avg_processing_time = (
                self.avg_processing_time * min(self.events_processed, 100)
                + processing_time
            ) / (min(self.events_processed, 100) + 1)


@dataclass
class StatusDisplayConfig:
    """Consolidated configuration for status display system."""

    # Display settings
    enabled: bool = True
    display_level: StatusDisplayLevel = StatusDisplayLevel.STANDARD
    update_frequency: StatusUpdateFrequency = StatusUpdateFrequency.NORMAL
    auto_start_display: bool = False
    max_visible_operations: int = 5
    max_visible_events: int = 10
    show_timestamps: bool = True
    show_progress_bars: bool = True

    # Event filtering
    filtered_event_types: Set[EventType] = field(default_factory=set)
    important_event_types: Set[EventType] = field(
        default_factory=lambda: {
            EventType.AGENT_STATE_CHANGED,
            EventType.ERROR_OCCURRED,
            EventType.APPROVAL_REQUESTED,
        }
    )

    # Terminal settings
    terminal_width: Optional[int] = None  # Auto-detect if None
    terminal_height: Optional[int] = None  # Auto-detect if None
    use_color: bool = True
    use_emoji: bool = True
    compact_mode: bool = False

    # Performance settings
    max_queue_size: int = 1000
    throttle_rate: int = 10  # Events per second
    max_listeners: int = 50
    batch_size: int = 10
    flush_interval: float = 0.1

    # Persistence settings
    save_event_history: bool = False
    event_history_file: Optional[Path] = None
    max_history_size: int = 10000
    max_operation_history: int = 1000
    max_event_history: int = 5000

    # Integration settings
    enable_cli_integration: bool = True
    enable_web_interface: bool = False
    web_interface_port: int = 8080

    # UI theme settings
    theme: str = "default"
    colors: Dict[str, str] = field(
        default_factory=lambda: {
            "primary": "blue",
            "secondary": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "info": "white",
            "dim": "dim",
        }
    )

    # Component visibility
    show_agent_table: bool = True
    show_operation_progress: bool = True
    show_event_log: bool = True
    show_system_stats: bool = True
    show_operation_tree: bool = False

    # Layout settings
    layout_style: str = "comprehensive"  # "simple", "comprehensive", "minimal"
    panel_borders: bool = True
    panel_titles: bool = True

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Ensure update frequency is valid
        if isinstance(self.update_frequency, float):
            # Convert float to enum
            for freq in StatusUpdateFrequency:
                if abs(freq.value - self.update_frequency) < 0.01:
                    self.update_frequency = freq
                    break

        # Ensure reasonable limits
        self.max_visible_operations = max(1, min(50, self.max_visible_operations))
        self.max_visible_events = max(1, min(100, self.max_visible_events))
        self.throttle_rate = max(1, min(100, self.throttle_rate))

        # Set up event history file if saving is enabled
        if self.save_event_history and not self.event_history_file:
            self.event_history_file = (
                Path.home() / ".omnimancer" / "agent_status_history.json"
            )

    def get_color(self, color_name: str, default: str = "white") -> str:
        """Get color by name with fallback."""
        return self.colors.get(color_name, default)


# Base classes for event handling
class EventListener:
    """Base class for event listeners."""

    def __init__(self, event_types: Optional[Set[EventType]] = None):
        """
        Initialize event listener.

        Args:
            event_types: Set of event types to listen for. If None, listens to all events.
        """
        self.event_types = event_types or set(EventType)
        self.active = True

    async def handle_event(self, event: AgentEvent) -> None:
        """
        Handle an agent event.

        Args:
            event: The agent event to handle
        """
        if self.active and event.event_type in self.event_types:
            await self._process_event(event)

    async def _process_event(self, event: AgentEvent) -> None:
        """Process the event. Override in subclasses."""
        pass

    def stop(self) -> None:
        """Stop listening to events."""
        self.active = False


class StatusStreamListener:
    """Base class for status stream listeners."""

    def __init__(self, listener_id: str, event_types: Optional[Set[EventType]] = None):
        """
        Initialize status stream listener.

        Args:
            listener_id: Unique identifier for this listener
            event_types: Set of event types to listen for (None = all events)
        """
        self.listener_id = listener_id
        self.event_types = event_types or set(EventType)
        self.active = True
        self.last_event_time: Optional[datetime] = None
        self.event_count = 0

    async def handle_event(self, stream_event: StatusStreamEvent) -> None:
        """
        Handle a status stream event.

        Args:
            stream_event: The status stream event to handle
        """
        if not self.active or stream_event.event.event_type not in self.event_types:
            return

        try:
            await self._process_event(stream_event)
            self.last_event_time = datetime.now()
            self.event_count += 1
        except Exception as e:
            # Import here to avoid circular imports
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error processing event in listener {self.listener_id}: {e}")

    async def _process_event(self, stream_event: StatusStreamEvent) -> None:
        """Process the event. Override in subclasses."""
        pass

    def stop(self) -> None:
        """Stop listening to events."""
        self.active = False


# Utility functions for common configuration operations
def create_minimal_config() -> StatusDisplayConfig:
    """Create a minimal configuration for low-resource environments."""
    return StatusDisplayConfig(
        display_level=StatusDisplayLevel.MINIMAL,
        update_frequency=StatusUpdateFrequency.SLOW,
        max_visible_operations=3,
        max_visible_events=5,
        compact_mode=True,
        show_progress_bars=False,
        use_emoji=False,
        max_queue_size=100,
        throttle_rate=5,
    )


def create_debug_config() -> StatusDisplayConfig:
    """Create a debug configuration with full visibility."""
    return StatusDisplayConfig(
        display_level=StatusDisplayLevel.DEBUG,
        update_frequency=StatusUpdateFrequency.REAL_TIME,
        max_visible_operations=20,
        max_visible_events=50,
        save_event_history=True,
        max_queue_size=5000,
        throttle_rate=100,
    )
