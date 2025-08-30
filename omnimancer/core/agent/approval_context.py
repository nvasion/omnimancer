"""
Approval Context and Data Structures for Omnimancer Agent Operations.

This module defines the data structures and context objects used
throughout the approval workflow system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .rich_renderer import RiskLevel


class OperationStatus(Enum):
    """Status of an operation."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileOperationType(Enum):
    """Types of file operations."""

    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    MODIFY = "modify"


@dataclass
class FileChange:
    """Represents a file change in an operation."""

    path: Path
    operation: FileOperationType
    content_preview: Optional[str] = None
    size_bytes: Optional[int] = None
    permissions: Optional[str] = None
    backup_path: Optional[Path] = None

    @property
    def relative_path(self) -> str:
        """Get relative path as string."""
        return str(self.path)


@dataclass
class SecurityFlags:
    """Security-related flags for operations."""

    requires_sudo: bool = False
    modifies_system_files: bool = False
    accesses_network: bool = False
    executes_code: bool = False
    modifies_permissions: bool = False
    creates_processes: bool = False
    accesses_sensitive_data: bool = False

    def get_risk_factors(self) -> List[str]:
        """Get list of security risk factors."""
        factors = []

        if self.requires_sudo:
            factors.append("Requires elevated privileges")
        if self.modifies_system_files:
            factors.append("Modifies system files")
        if self.accesses_network:
            factors.append("Makes network requests")
        if self.executes_code:
            factors.append("Executes external code")
        if self.modifies_permissions:
            factors.append("Changes file permissions")
        if self.creates_processes:
            factors.append("Creates new processes")
        if self.accesses_sensitive_data:
            factors.append("Accesses sensitive data")

        return factors


@dataclass
class OperationDetails:
    """Detailed information about an operation requiring approval."""

    # Core operation information
    operation_type: str
    target: Optional[str] = None
    description: Optional[str] = None

    # Risk assessment
    risk_level: Union[RiskLevel, int, str] = RiskLevel.MEDIUM
    risk_factors: List[str] = field(default_factory=list)
    security_flags: SecurityFlags = field(default_factory=SecurityFlags)

    # Command/execution details
    command: Optional[str] = None
    arguments: List[str] = field(default_factory=list)
    working_directory: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)

    # File operations
    files_affected: List[Path] = field(default_factory=list)
    file_changes: List[FileChange] = field(default_factory=list)

    # Timing and resources
    estimated_time: Optional[float] = None
    max_memory_mb: Optional[int] = None
    max_disk_mb: Optional[int] = None

    # Operation metadata
    operation_id: Optional[str] = None
    parent_operation_id: Optional[str] = None
    batch_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing."""
        # Auto-populate risk factors from security flags
        if not self.risk_factors and hasattr(self.security_flags, "get_risk_factors"):
            self.risk_factors = self.security_flags.get_risk_factors()

        # Ensure paths are Path objects
        self.files_affected = [
            Path(f) if not isinstance(f, Path) else f for f in self.files_affected
        ]

    def add_file_change(
        self,
        path: Union[str, Path],
        operation: FileOperationType,
        content_preview: Optional[str] = None,
    ) -> None:
        """Add a file change to this operation."""
        file_change = FileChange(
            path=Path(path) if not isinstance(path, Path) else path,
            operation=operation,
            content_preview=content_preview,
        )
        self.file_changes.append(file_change)

        # Also add to files_affected if not already present
        if file_change.path not in self.files_affected:
            self.files_affected.append(file_change.path)

    def get_total_files_affected(self) -> int:
        """Get total number of files affected."""
        return len(set(self.files_affected + [fc.path for fc in self.file_changes]))

    def get_operation_summary(self) -> str:
        """Get a brief summary of the operation."""
        if self.description:
            return self.description

        summary_parts = [self.operation_type]

        if self.target:
            summary_parts.append(f"on {self.target}")
        elif self.files_affected:
            if len(self.files_affected) == 1:
                summary_parts.append(f"on {self.files_affected[0]}")
            else:
                summary_parts.append(f"on {len(self.files_affected)} files")

        return " ".join(summary_parts)


@dataclass
class ApprovalDecision:
    """Represents a user's approval decision."""

    approved: bool
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    # Decision metadata
    response_time_seconds: Optional[float] = None
    approval_method: Optional[str] = None  # "manual", "auto", "cached", etc.
    confidence_level: Optional[float] = None  # 0.0 - 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "approved": self.approved,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "additional_data": self.additional_data,
            "response_time_seconds": self.response_time_seconds,
            "approval_method": self.approval_method,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalDecision":
        """Create from dictionary."""
        timestamp = (
            datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now()
        )

        return cls(
            approved=data["approved"],
            reason=data.get("reason"),
            timestamp=timestamp,
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            additional_data=data.get("additional_data", {}),
            response_time_seconds=data.get("response_time_seconds"),
            approval_method=data.get("approval_method"),
            confidence_level=data.get("confidence_level"),
        )


@dataclass
class ApprovalContext:
    """Complete context for an approval request."""

    # Core identification
    context_id: str = field(
        default_factory=lambda: f"ctx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    agent_name: str = "Omnimancer Agent"
    timestamp: datetime = field(default_factory=datetime.now)

    # Operation details
    operation_details: OperationDetails = field(
        default_factory=lambda: OperationDetails(operation_type="unknown")
    )

    # Content for display
    diff_content: Optional[str] = None
    log_content: Optional[str] = None
    preview_content: Optional[str] = None

    # Context and history
    previous_operations: List[OperationDetails] = field(default_factory=list)
    related_context: Dict[str, Any] = field(default_factory=dict)

    # Session information
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_history: List[str] = field(default_factory=list)

    # Approval workflow state
    status: OperationStatus = OperationStatus.PENDING
    decision: Optional[ApprovalDecision] = None
    auto_approval_eligible: bool = False
    approval_timeout_seconds: Optional[int] = None

    # Display customization
    display_options: Dict[str, Any] = field(default_factory=dict)

    def add_conversation_entry(self, message: str) -> None:
        """Add an entry to the conversation history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_history.append(f"[{timestamp}] {message}")

    def add_related_context(self, key: str, value: Any) -> None:
        """Add related context information."""
        self.related_context[key] = value

    def set_decision(self, decision: ApprovalDecision) -> None:
        """Set the approval decision and update status."""
        self.decision = decision
        self.status = (
            OperationStatus.APPROVED if decision.approved else OperationStatus.DENIED
        )

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the context for logging."""
        return {
            "context_id": self.context_id,
            "agent_name": self.agent_name,
            "operation_type": self.operation_details.operation_type,
            "risk_level": str(self.operation_details.risk_level),
            "files_affected": len(self.operation_details.files_affected),
            "status": self.status.value,
            "approved": self.decision.approved if self.decision else None,
            "timestamp": self.timestamp.isoformat(),
        }

    def is_completed(self) -> bool:
        """Check if the approval process is completed."""
        return self.status in [
            OperationStatus.APPROVED,
            OperationStatus.DENIED,
            OperationStatus.CANCELLED,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "operation_details": {
                "operation_type": self.operation_details.operation_type,
                "target": self.operation_details.target,
                "description": self.operation_details.description,
                "risk_level": str(self.operation_details.risk_level),
                "risk_factors": self.operation_details.risk_factors,
                "command": self.operation_details.command,
                "arguments": self.operation_details.arguments,
                "files_affected": [
                    str(f) for f in self.operation_details.files_affected
                ],
                "metadata": self.operation_details.metadata,
            },
            "diff_content": self.diff_content,
            "status": self.status.value,
            "decision": self.decision.to_dict() if self.decision else None,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "conversation_history": self.conversation_history,
            "related_context": self.related_context,
        }


# Utility functions for creating common contexts


def create_file_operation_context(
    operation_type: str,
    file_path: Union[str, Path],
    content: Optional[str] = None,
    agent_name: str = "Omnimancer Agent",
) -> ApprovalContext:
    """Create approval context for file operations."""
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

    # Determine file operation type
    file_op_type = FileOperationType.MODIFY
    if "create" in operation_type.lower():
        file_op_type = FileOperationType.CREATE
    elif "delete" in operation_type.lower():
        file_op_type = FileOperationType.DELETE
    elif "read" in operation_type.lower():
        file_op_type = FileOperationType.READ
    elif "write" in operation_type.lower():
        file_op_type = FileOperationType.WRITE

    # Create operation details
    operation_details = OperationDetails(
        operation_type=operation_type,
        target=str(file_path),
        description=f"{operation_type} on {file_path.name}",
        files_affected=[file_path],
    )

    # Add file change
    operation_details.add_file_change(
        file_path,
        file_op_type,
        content[:200] + "..." if content and len(content) > 200 else content,
    )

    # Create context
    context = ApprovalContext(
        agent_name=agent_name,
        operation_details=operation_details,
        preview_content=content,
    )

    return context


def create_command_execution_context(
    command: str,
    arguments: Optional[List[str]] = None,
    working_directory: Optional[str] = None,
    agent_name: str = "Omnimancer Agent",
) -> ApprovalContext:
    """Create approval context for command execution."""
    arguments = arguments or []

    # Assess security flags
    security_flags = SecurityFlags()

    # Basic security assessment based on command
    if command in ["sudo", "su", "doas"]:
        security_flags.requires_sudo = True

    if any(
        sys_path in " ".join([command] + arguments)
        for sys_path in ["/etc", "/usr/bin", "/sys"]
    ):
        security_flags.modifies_system_files = True

    if any(net_cmd in command for net_cmd in ["curl", "wget", "nc", "ssh"]):
        security_flags.accesses_network = True

    if any(
        exec_cmd in command for exec_cmd in ["eval", "exec", "python", "bash", "sh"]
    ):
        security_flags.executes_code = True

    # Create operation details
    operation_details = OperationDetails(
        operation_type="command_execute",
        target=command,
        description=f"Execute: {command} {' '.join(arguments)}",
        command=command,
        arguments=arguments,
        working_directory=working_directory,
        security_flags=security_flags,
    )

    # Create context
    context = ApprovalContext(
        agent_name=agent_name, operation_details=operation_details
    )

    return context


def create_web_request_context(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    agent_name: str = "Omnimancer Agent",
) -> ApprovalContext:
    """Create approval context for web requests."""
    # Create security flags
    security_flags = SecurityFlags(accesses_network=True)

    # Additional assessment based on URL
    if any(sensitive in url.lower() for sensitive in ["admin", "api", "auth", "login"]):
        security_flags.accesses_sensitive_data = True

    # Create operation details
    operation_details = OperationDetails(
        operation_type="web_request",
        target=url,
        description=f"{method} request to {url}",
        security_flags=security_flags,
        metadata={"method": method, "headers": headers or {}, "url": url},
    )

    # Create context
    context = ApprovalContext(
        agent_name=agent_name, operation_details=operation_details
    )

    return context
