"""
Enhanced User Approval and Preview System for Omnimancer Agent.

This module provides comprehensive approval workflows with change preview,
diff visualization, batch approval capabilities, and integration with the
existing security framework.
"""

import difflib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..security.approval_workflow import (
    ApprovalRequest,
    ApprovalStatus,
    ApprovalWorkflow,
)
from .types import Operation, OperationType

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes that can be previewed."""

    FILE_CREATE = "file_create"
    FILE_MODIFY = "file_modify"
    FILE_DELETE = "file_delete"
    DIRECTORY_CREATE = "directory_create"
    DIRECTORY_DELETE = "directory_delete"
    COMMAND_EXECUTE = "command_execute"
    WEB_REQUEST = "web_request"
    MCP_TOOL_CALL = "mcp_tool_call"


class PreviewFormat(Enum):
    """Format for change previews."""

    TEXT = "text"
    DIFF = "diff"
    JSON = "json"
    HTML = "html"


@dataclass
class ChangePreview:
    """Represents a preview of a change."""

    change_type: ChangeType
    description: str
    current_state: Optional[str] = None
    proposed_state: Optional[str] = None
    diff: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Optional[str] = None
    reversible: bool = False

    def generate_diff(self) -> str:
        """Generate unified diff between current and proposed states."""
        if not self.current_state or not self.proposed_state:
            return "No diff available - new file or deletion"

        current_lines = self.current_state.splitlines(keepends=True)
        proposed_lines = self.proposed_state.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                current_lines,
                proposed_lines,
                fromfile=f"{self.metadata.get('path', 'current')}",
                tofile=f"{self.metadata.get('path', 'proposed')}",
                lineterm="",
            )
        )

        return "".join(diff_lines)

    def format_preview(self, format_type: PreviewFormat = PreviewFormat.TEXT) -> str:
        """Format the preview for display."""
        if format_type == PreviewFormat.DIFF:
            if not self.diff:
                self.diff = self.generate_diff()
            return self.diff
        elif format_type == PreviewFormat.JSON:
            return json.dumps(
                {
                    "type": self.change_type.value,
                    "description": self.description,
                    "metadata": self.metadata,
                    "risk_assessment": self.risk_assessment,
                    "reversible": self.reversible,
                },
                indent=2,
            )
        elif format_type == PreviewFormat.HTML:
            return self._format_html_preview()
        else:
            return self._format_text_preview()

    def _format_text_preview(self) -> str:
        """Format as plain text preview."""
        lines = [
            f"Change Type: {self.change_type.value}",
            f"Description: {self.description}",
        ]

        if self.risk_assessment:
            lines.append(f"Risk Assessment: {self.risk_assessment}")

        lines.append(f"Reversible: {'Yes' if self.reversible else 'No'}")

        if self.metadata:
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")

        if self.diff:
            lines.extend(["", "Changes:", self.diff])

        return "\n".join(lines)

    def _format_html_preview(self) -> str:
        """Format as HTML preview."""
        html_parts = [
            "<div class='change-preview'>",
            f"<h3>{self.change_type.value.replace('_', ' ').title()}</h3>",
            f"<p><strong>Description:</strong> {self.description}</p>",
        ]

        if self.risk_assessment:
            html_parts.append(
                f"<p><strong>Risk:</strong> <span class='risk'>{self.risk_assessment}</span></p>"
            )

        html_parts.append(
            f"<p><strong>Reversible:</strong> {'Yes' if self.reversible else 'No'}</p>"
        )

        if self.diff:
            html_parts.extend(
                [
                    "<div class='diff-container'>",
                    "<pre class='diff'>",
                    self.diff.replace("<", "&lt;").replace(">", "&gt;"),
                    "</pre>",
                    "</div>",
                ]
            )

        html_parts.append("</div>")
        return "\n".join(html_parts)


@dataclass
class BatchApprovalRequest:
    """Represents a batch of operations requiring approval."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operations: List[Operation] = field(default_factory=list)
    previews: List[ChangePreview] = field(default_factory=list)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    approver: Optional[str] = None
    approved_at: Optional[datetime] = None
    denial_reason: Optional[str] = None
    approved_operations: Set[int] = field(
        default_factory=set
    )  # Indices of approved operations

    def is_expired(self) -> bool:
        """Check if the batch request has expired."""
        return self.expires_at and datetime.now() > self.expires_at

    def get_approval_summary(self) -> Dict[str, Any]:
        """Get summary of approval status."""
        total = len(self.operations)
        approved = len(self.approved_operations)
        return {
            "total_operations": total,
            "approved_operations": approved,
            "pending_operations": total - approved,
            "approval_rate": approved / total if total > 0 else 0,
            "all_approved": approved == total,
            "partially_approved": 0 < approved < total,
        }


class EnhancedApprovalManager:
    """
    Enhanced approval manager with preview, diff visualization, and batch approval.

    This class bridges the basic ApprovalManager functionality with the advanced
    ApprovalWorkflow security system, adding comprehensive preview capabilities
    and batch approval support.
    """

    def __init__(
        self,
        approval_workflow: Optional[ApprovalWorkflow] = None,
        default_timeout_minutes: int = 30,
        enable_batch_approval: bool = True,
        max_batch_size: int = 10,
    ):
        """
        Initialize enhanced approval manager.

        Args:
            approval_workflow: Optional existing approval workflow instance
            default_timeout_minutes: Default timeout for approval requests
            enable_batch_approval: Whether to enable batch approval functionality
            max_batch_size: Maximum number of operations in a batch
        """
        self.approval_workflow = approval_workflow or ApprovalWorkflow()
        self.default_timeout_minutes = default_timeout_minutes
        self.enable_batch_approval = enable_batch_approval
        self.max_batch_size = max_batch_size

        # Batch approval management
        self.pending_batches: Dict[str, BatchApprovalRequest] = {}
        self.completed_batches: Dict[str, BatchApprovalRequest] = {}

        # Preview generators for different operation types
        self.preview_generators: Dict[OperationType, Callable] = {
            OperationType.FILE_READ: self._generate_file_read_preview,
            OperationType.FILE_WRITE: self._generate_file_write_preview,
            OperationType.FILE_DELETE: self._generate_file_delete_preview,
            OperationType.DIRECTORY_CREATE: self._generate_directory_create_preview,
            OperationType.DIRECTORY_DELETE: self._generate_directory_delete_preview,
            OperationType.COMMAND_EXECUTE: self._generate_command_preview,
            OperationType.WEB_REQUEST: self._generate_web_request_preview,
            OperationType.MCP_TOOL_CALL: self._generate_mcp_tool_preview,
        }

        # User interaction callbacks
        self.approval_callback: Optional[Callable] = None
        self.batch_approval_callback: Optional[Callable] = None

        # Approval history for audit trail
        self.approval_history: List[Dict[str, Any]] = []

    def set_approval_callback(self, callback: Callable):
        """Set callback for single operation approval."""
        self.approval_callback = callback

    def set_batch_approval_callback(self, callback: Callable):
        """Set callback for batch operation approval."""
        self.batch_approval_callback = callback

    async def request_single_approval(self, operation: Operation) -> bool:
        """
        Request approval for a single operation with preview.

        Args:
            operation: Operation requiring approval

        Returns:
            True if approved, False if denied
        """
        try:
            # Generate preview
            preview = await self.generate_operation_preview(operation)
            operation.preview = preview.format_preview()

            # Create approval request through security workflow
            approval_request = await self.approval_workflow.request_approval(
                operation_type=operation.type.value,
                description=operation.description,
                metadata={
                    "operation_data": operation.data,
                    "preview": preview.format_preview(PreviewFormat.JSON),
                    "reversible": operation.reversible,
                },
                timeout_minutes=self.default_timeout_minutes,
            )

            # If auto-approved (low risk), return immediately
            if approval_request.status == ApprovalStatus.APPROVED:
                self._record_approval_history(operation, approval_request, preview)
                return True

            # Request user approval through callback
            if self.approval_callback:
                user_approved = await self.approval_callback(
                    {
                        "operation": operation,
                        "preview": preview,
                        "approval_request": approval_request,
                        "risk_level": approval_request.risk_level,
                    }
                )

                if user_approved:
                    self.approval_workflow.approve_request(
                        approval_request.id, "user", "User approved through UI"
                    )
                    self._record_approval_history(operation, approval_request, preview)
                    return True
                else:
                    self.approval_workflow.deny_request(
                        approval_request.id, "user", "User denied through UI"
                    )
                    self._record_approval_history(
                        operation, approval_request, preview, approved=False
                    )
                    return False

            # No callback available, default to deny
            logger.warning(f"No approval callback set for operation: {operation.type}")
            return False

        except Exception as e:
            logger.error(f"Error in approval request: {e}")
            return False

    async def request_batch_approval(
        self, operations: List[Operation]
    ) -> BatchApprovalRequest:
        """
        Request approval for a batch of operations.

        Args:
            operations: List of operations requiring approval

        Returns:
            BatchApprovalRequest with approval status
        """
        if not self.enable_batch_approval:
            raise ValueError("Batch approval is disabled")

        if len(operations) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(operations)} exceeds maximum {self.max_batch_size}"
            )

        # Generate previews for all operations
        previews = []
        for operation in operations:
            try:
                preview = await self.generate_operation_preview(operation)
                previews.append(preview)
            except Exception as e:
                logger.error(f"Error generating preview for {operation.type}: {e}")
                # Create basic preview on error
                previews.append(
                    ChangePreview(
                        change_type=ChangeType(operation.type.value),
                        description=f"Error generating preview: {str(e)}",
                        metadata={"error": str(e)},
                    )
                )

        # Create batch request
        batch_request = BatchApprovalRequest(
            operations=operations,
            previews=previews,
            expires_at=datetime.now() + timedelta(minutes=self.default_timeout_minutes),
        )

        self.pending_batches[batch_request.id] = batch_request

        # Request user approval through batch callback
        if self.batch_approval_callback:
            try:
                approval_result = await self.batch_approval_callback(batch_request)
                if approval_result:
                    self._process_batch_approval_result(batch_request, approval_result)
            except Exception as e:
                logger.error(f"Error in batch approval callback: {e}")

        return batch_request

    def _process_batch_approval_result(
        self,
        batch_request: BatchApprovalRequest,
        approval_result: Dict[str, Any],
    ):
        """Process the result of batch approval from user."""
        if approval_result.get("approve_all"):
            batch_request.approved_operations = set(
                range(len(batch_request.operations))
            )
            batch_request.status = ApprovalStatus.APPROVED
        elif approval_result.get("deny_all"):
            batch_request.status = ApprovalStatus.DENIED
            batch_request.denial_reason = approval_result.get(
                "reason", "User denied all operations"
            )
        elif "approved_indices" in approval_result:
            batch_request.approved_operations = set(approval_result["approved_indices"])
            if len(batch_request.approved_operations) == len(batch_request.operations):
                batch_request.status = ApprovalStatus.APPROVED
            elif len(batch_request.approved_operations) > 0:
                batch_request.status = (
                    ApprovalStatus.APPROVED
                )  # Partial approval still counts as approved

        batch_request.approver = "user"
        batch_request.approved_at = datetime.now()

        # Move to completed batches
        self.completed_batches[batch_request.id] = batch_request
        if batch_request.id in self.pending_batches:
            del self.pending_batches[batch_request.id]

    async def generate_operation_preview(self, operation: Operation) -> ChangePreview:
        """
        Generate a comprehensive preview for an operation.

        Args:
            operation: Operation to generate preview for

        Returns:
            ChangePreview with diff visualization and metadata
        """
        generator = self.preview_generators.get(operation.type)
        if generator:
            return await generator(operation)
        else:
            # Fallback generic preview
            return ChangePreview(
                change_type=ChangeType(operation.type.value),
                description=operation.description,
                metadata=operation.data,
                reversible=operation.reversible,
            )

    async def _generate_file_read_preview(self, operation: Operation) -> ChangePreview:
        """Generate preview for file read operation."""
        file_path = operation.data.get("path", "unknown")
        return ChangePreview(
            change_type=ChangeType.FILE_MODIFY,
            description=f"Read file: {file_path}",
            metadata={"path": file_path, "operation": "read"},
            risk_assessment="Low - Read-only operation",
            reversible=False,
        )

    async def _generate_file_write_preview(self, operation: Operation) -> ChangePreview:
        """Generate preview for file write operation."""
        file_path = operation.data.get("path", "unknown")
        new_content = operation.data.get("content", "")

        # Try to read current content for diff
        current_content = None
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                current_content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not read current content of {file_path}: {e}")

        preview = ChangePreview(
            change_type=(
                ChangeType.FILE_CREATE
                if not current_content
                else ChangeType.FILE_MODIFY
            ),
            description=f"Write to file: {file_path}",
            current_state=current_content,
            proposed_state=new_content,
            metadata={
                "path": file_path,
                "content_length": len(new_content),
                "operation": "write",
            },
            risk_assessment=self._assess_file_write_risk(file_path, new_content),
            reversible=bool(current_content),  # Reversible if we have backup content
        )

        # Generate diff if we have both states
        if current_content is not None:
            preview.diff = preview.generate_diff()

        return preview

    async def _generate_file_delete_preview(
        self, operation: Operation
    ) -> ChangePreview:
        """Generate preview for file delete operation."""
        file_path = operation.data.get("path", "unknown")

        # Try to read current content for backup
        current_content = None
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                current_content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.debug(
                f"Could not read content of {file_path} for deletion preview: {e}"
            )

        return ChangePreview(
            change_type=ChangeType.FILE_DELETE,
            description=f"Delete file: {file_path}",
            current_state=current_content,
            proposed_state="",  # File will be gone
            metadata={
                "path": file_path,
                "file_size": len(current_content) if current_content else 0,
                "operation": "delete",
            },
            risk_assessment=self._assess_file_delete_risk(file_path),
            reversible=bool(current_content),
        )

    async def _generate_directory_create_preview(
        self, operation: Operation
    ) -> ChangePreview:
        """Generate preview for directory creation."""
        dir_path = operation.data.get("path", "unknown")
        return ChangePreview(
            change_type=ChangeType.DIRECTORY_CREATE,
            description=f"Create directory: {dir_path}",
            metadata={"path": dir_path, "operation": "mkdir"},
            risk_assessment="Low - Directory creation",
            reversible=True,
        )

    async def _generate_directory_delete_preview(
        self, operation: Operation
    ) -> ChangePreview:
        """Generate preview for directory deletion."""
        dir_path = operation.data.get("path", "unknown")

        # Check if directory exists and list contents
        contents = []
        try:
            path = Path(dir_path)
            if path.exists() and path.is_dir():
                contents = [str(item) for item in path.iterdir()]
        except Exception as e:
            logger.debug(f"Could not list contents of {dir_path}: {e}")

        return ChangePreview(
            change_type=ChangeType.DIRECTORY_DELETE,
            description=f"Delete directory: {dir_path}",
            metadata={
                "path": dir_path,
                "contents": contents,
                "item_count": len(contents),
                "operation": "rmdir",
            },
            risk_assessment=self._assess_directory_delete_risk(dir_path, contents),
            reversible=len(contents) == 0,  # Only reversible if empty
        )

    async def _generate_command_preview(self, operation: Operation) -> ChangePreview:
        """Generate preview for command execution."""
        command = operation.data.get("command", "unknown")
        args = operation.data.get("args", [])
        full_command = f"{command} {' '.join(args)}" if args else command

        return ChangePreview(
            change_type=ChangeType.COMMAND_EXECUTE,
            description=f"Execute command: {full_command}",
            metadata={
                "command": command,
                "args": args,
                "full_command": full_command,
                "working_dir": operation.data.get("working_dir"),
                "operation": "execute",
            },
            risk_assessment=self._assess_command_risk(full_command),
            reversible=False,  # Commands are generally not reversible
        )

    async def _generate_web_request_preview(
        self, operation: Operation
    ) -> ChangePreview:
        """Generate preview for web request."""
        url = operation.data.get("url", "unknown")
        method = operation.data.get("method", "GET")

        return ChangePreview(
            change_type=ChangeType.WEB_REQUEST,
            description=f"{method} request to: {url}",
            metadata={
                "url": url,
                "method": method,
                "headers": operation.data.get("headers", {}),
                "operation": "web_request",
            },
            risk_assessment=self._assess_web_request_risk(url, method),
            reversible=method.upper() == "GET",
        )

    async def _generate_mcp_tool_preview(self, operation: Operation) -> ChangePreview:
        """Generate preview for MCP tool call."""
        tool_name = operation.data.get("tool_name", "unknown")
        arguments = operation.data.get("arguments", {})

        return ChangePreview(
            change_type=ChangeType.MCP_TOOL_CALL,
            description=f"Call MCP tool: {tool_name}",
            metadata={
                "tool_name": tool_name,
                "arguments": arguments,
                "operation": "mcp_call",
            },
            risk_assessment=self._assess_mcp_tool_risk(tool_name, arguments),
            reversible=False,  # Tool calls are generally not reversible
        )

    def _assess_file_write_risk(self, file_path: str, content: str) -> str:
        """Assess risk level for file write operations."""
        path_lower = file_path.lower()
        risk_factors = []

        if any(
            pattern in path_lower
            for pattern in [".env", "password", "secret", "key", "credential"]
        ):
            risk_factors.append("sensitive file")

        if any(
            pattern in path_lower for pattern in ["config", "settings", ".ssh", "/etc"]
        ):
            risk_factors.append("configuration file")

        if len(content) > 100000:  # Large files
            risk_factors.append("large file")

        if not risk_factors:
            return "Low - Standard file write"
        elif len(risk_factors) == 1:
            return f"Medium - {risk_factors[0]}"
        else:
            return f"High - Multiple factors: {', '.join(risk_factors)}"

    def _assess_file_delete_risk(self, file_path: str) -> str:
        """Assess risk level for file deletion."""
        if any(
            pattern in file_path.lower()
            for pattern in [".env", "config", "settings", "backup"]
        ):
            return "High - Deleting important file"
        return "Medium - File deletion"

    def _assess_directory_delete_risk(self, dir_path: str, contents: List[str]) -> str:
        """Assess risk level for directory deletion."""
        if len(contents) > 10:
            return "High - Deleting directory with many files"
        elif len(contents) > 0:
            return "Medium - Deleting non-empty directory"
        else:
            return "Low - Deleting empty directory"

    def _assess_command_risk(self, command: str) -> str:
        """Assess risk level for command execution."""
        command_lower = command.lower()

        if any(
            dangerous in command_lower for dangerous in ["rm -rf", "sudo", "chmod 777"]
        ):
            return "Critical - Dangerous command"
        elif any(
            risky in command_lower for risky in ["rm", "mv", "cp", "chmod", "chown"]
        ):
            return "High - File system modification"
        elif any(network in command_lower for network in ["curl", "wget", "nc", "ssh"]):
            return "Medium - Network operation"
        else:
            return "Low - Safe command"

    def _assess_web_request_risk(self, url: str, method: str) -> str:
        """Assess risk level for web requests."""
        if method.upper() in ["POST", "PUT", "DELETE", "PATCH"]:
            return "Medium - Data modification request"
        elif "localhost" in url or "127.0.0.1" in url:
            return "Low - Local request"
        else:
            return "Low - External GET request"

    def _assess_mcp_tool_risk(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Assess risk level for MCP tool calls."""
        if "write" in tool_name.lower() or "delete" in tool_name.lower():
            return "Medium - Potentially modifying tool"
        elif "execute" in tool_name.lower() or "run" in tool_name.lower():
            return "High - Execution tool"
        else:
            return "Low - Query or read tool"

    def _record_approval_history(
        self,
        operation: Operation,
        approval_request: ApprovalRequest,
        preview: ChangePreview,
        approved: bool = True,
    ):
        """Record approval in history for audit trail."""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation.type.value,
            "operation_description": operation.description,
            "approval_id": approval_request.id,
            "risk_level": approval_request.risk_level.value,
            "approved": approved,
            "approver": approval_request.approver,
            "preview_summary": {
                "change_type": preview.change_type.value,
                "risk_assessment": preview.risk_assessment,
                "reversible": preview.reversible,
            },
        }

        self.approval_history.append(history_entry)

        # Keep only last 1000 entries to prevent memory bloat
        if len(self.approval_history) > 1000:
            self.approval_history = self.approval_history[-1000:]

    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get statistics about approval history."""
        if not self.approval_history:
            return {"total_requests": 0}

        total = len(self.approval_history)
        approved = sum(1 for entry in self.approval_history if entry["approved"])

        risk_levels = {}
        operation_types = {}

        for entry in self.approval_history:
            risk = entry["risk_level"]
            op_type = entry["operation_type"]

            risk_levels[risk] = risk_levels.get(risk, 0) + 1
            operation_types[op_type] = operation_types.get(op_type, 0) + 1

        return {
            "total_requests": total,
            "approved_requests": approved,
            "denied_requests": total - approved,
            "approval_rate": approved / total,
            "risk_level_distribution": risk_levels,
            "operation_type_distribution": operation_types,
            "recent_activity": self.approval_history[-10:],  # Last 10 entries
        }

    def cleanup_expired_requests(self):
        """Clean up expired batch requests."""
        expired_ids = []
        for batch_id, batch_request in self.pending_batches.items():
            if batch_request.is_expired():
                batch_request.status = ApprovalStatus.EXPIRED
                self.completed_batches[batch_id] = batch_request
                expired_ids.append(batch_id)

        for batch_id in expired_ids:
            del self.pending_batches[batch_id]

        return len(expired_ids)
