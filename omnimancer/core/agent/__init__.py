"""Omnimancer agent core components."""

from .approval_context import (
    ApprovalContext,
    ApprovalDecision,
    FileChange,
    FileOperationType,
    OperationDetails,
    OperationStatus,
    RiskLevel,
    SecurityFlags,
    create_command_execution_context,
    create_file_operation_context,
    create_web_request_context,
)
from .config import AgentConfig
from .file_system_manager import FileSystemManager
from .status_core import (
    AgentEvent,
    AgentOperation,
    AgentStatus,
    EventListener,
    EventType,
    OperationType,
)
from .status_manager import UnifiedStatusManager as AgentStatusManager
from .status_manager import (
    get_status_manager,
    initialize_status_system,
    shutdown_status_system,
)
from .types import Operation, OperationResult
from .web_client import WebClient

__all__ = [
    "FileSystemManager",
    "WebClient",
    "AgentConfig",
    "AgentStatus",
    "OperationType",
    "OperationStatus",
    "EventType",
    "AgentOperation",
    "AgentEvent",
    "EventListener",
    "AgentStatusManager",
    "get_status_manager",
    "initialize_status_system",
    "shutdown_status_system",
    "ApprovalContext",
    "OperationDetails",
    "ApprovalDecision",
    "RiskLevel",
    "SecurityFlags",
    "FileChange",
    "FileOperationType",
    "OperationStatus",
    "create_file_operation_context",
    "create_command_execution_context",
    "create_web_request_context",
    "Operation",
    "OperationResult",
]
