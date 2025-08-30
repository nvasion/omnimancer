"""
Common types and data structures for Omnimancer Agent operations.

This module contains shared types that are used across different agent components
to avoid circular imports between modules.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class OperationType(Enum):
    """Types of operations that can be performed by the agent."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    DIRECTORY_CREATE = "directory_create"
    DIRECTORY_DELETE = "directory_delete"
    COMMAND_EXECUTE = "command_execute"
    WEB_REQUEST = "web_request"
    MCP_TOOL_CALL = "mcp_tool_call"
    WORKFLOW_STEP = "workflow_step"


@dataclass
class Operation:
    """Represents an operation to be performed by the agent."""

    type: OperationType
    description: str
    data: Dict[str, Any]
    requires_approval: bool = True
    reversible: bool = False
    preview: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class OperationResult:
    """Result of an operation execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    details: Optional[str] = None  # Additional context for errors or operation info
    rollback_data: Optional[Dict[str, Any]] = None
    operation: Optional[Operation] = None
    execution_time: Optional[float] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
