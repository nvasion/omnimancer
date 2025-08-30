"""Approval workflow for high-risk agent operations."""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ApprovalStatus(Enum):
    """Status of approval requests."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class RiskLevel(Enum):
    """Risk levels for operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalRequest:
    """Represents an approval request for a risky operation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requested_by: str = "agent"
    requested_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: Optional[str] = None
    approved_at: Optional[datetime] = None
    denial_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the approval request has expired."""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return False

    def time_remaining(self) -> Optional[timedelta]:
        """Get time remaining before expiration."""
        if self.expires_at:
            remaining = self.expires_at - datetime.now()
            return remaining if remaining.total_seconds() > 0 else timedelta(0)
        return None


class ApprovalWorkflow:
    """Manages approval workflows for high-risk operations."""

    def __init__(
        self,
        default_expiry_minutes: int = 30,
        auto_approve_low_risk: bool = True,
    ):
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.completed_requests: Dict[str, ApprovalRequest] = {}
        self.default_expiry_minutes = default_expiry_minutes
        self.auto_approve_low_risk = auto_approve_low_risk
        self.approval_handlers: Dict[RiskLevel, List[Callable]] = {
            level: [] for level in RiskLevel
        }
        self.risk_assessment_rules = self._get_default_risk_rules()

    def _get_default_risk_rules(self) -> Dict[str, RiskLevel]:
        """Get default risk assessment rules for operation types."""
        return {
            # Low risk operations
            "file_read": RiskLevel.LOW,
            "file_list": RiskLevel.LOW,
            "system_info": RiskLevel.LOW,
            "network_get": RiskLevel.LOW,
            # Medium risk operations
            "file_write": RiskLevel.MEDIUM,
            "file_create": RiskLevel.MEDIUM,
            "network_post": RiskLevel.MEDIUM,
            "command_execute": RiskLevel.MEDIUM,
            # High risk operations
            "file_delete": RiskLevel.HIGH,
            "system_config_change": RiskLevel.HIGH,
            "package_install": RiskLevel.HIGH,
            "service_control": RiskLevel.HIGH,
            # Critical risk operations
            "system_admin": RiskLevel.CRITICAL,
            "credential_access": RiskLevel.CRITICAL,
            "network_server": RiskLevel.CRITICAL,
            "data_export": RiskLevel.CRITICAL,
        }

    def assess_risk_level(
        self, operation_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> RiskLevel:
        """Assess the risk level of an operation."""

        # Base risk from operation type
        base_risk = self.risk_assessment_rules.get(operation_type, RiskLevel.MEDIUM)

        if not metadata:
            return base_risk

        # Risk escalation factors
        risk_factors = []

        # Check for sensitive paths
        if "path" in metadata:
            path = metadata["path"].lower()
            sensitive_patterns = [
                ".ssh",
                ".env",
                "password",
                "secret",
                "credential",
                "/etc",
                "/system",
                "config",
                "settings",
            ]
            if any(pattern in path for pattern in sensitive_patterns):
                risk_factors.append("sensitive_path")

        # Check for dangerous commands
        if "command" in metadata:
            command = metadata["command"].lower()
            dangerous_commands = [
                "rm -rf",
                "sudo",
                "chmod 777",
                "curl",
                "wget",
                "nc",
                "netcat",
                "ssh",
                "scp",
            ]
            if any(cmd in command for cmd in dangerous_commands):
                risk_factors.append("dangerous_command")

        # Check for network operations
        if "network" in metadata and metadata["network"]:
            risk_factors.append("network_operation")

        # Check for external URLs
        if "url" in metadata:
            url = metadata["url"]
            if not any(
                domain in url for domain in ["localhost", "127.0.0.1", "0.0.0.0"]
            ):
                risk_factors.append("external_network")

        # Escalate risk based on factors
        if len(risk_factors) >= 3:
            return RiskLevel.CRITICAL
        elif len(risk_factors) >= 2:
            return RiskLevel.HIGH
        elif len(risk_factors) >= 1:
            # Escalate by one level
            levels = [
                RiskLevel.LOW,
                RiskLevel.MEDIUM,
                RiskLevel.HIGH,
                RiskLevel.CRITICAL,
            ]
            current_index = levels.index(base_risk)
            return levels[min(current_index + 1, len(levels) - 1)]

        return base_risk

    async def request_approval(
        self,
        operation_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_minutes: Optional[int] = None,
    ) -> ApprovalRequest:
        """Request approval for an operation."""

        # Assess risk level
        risk_level = self.assess_risk_level(operation_type, metadata)

        # Create approval request
        request = ApprovalRequest(
            operation_type=operation_type,
            description=description,
            risk_level=risk_level,
            metadata=metadata or {},
        )

        # Set expiration
        expiry_minutes = timeout_minutes or self.default_expiry_minutes
        request.expires_at = datetime.now() + timedelta(minutes=expiry_minutes)

        # Auto-approve low risk operations if configured
        if self.auto_approve_low_risk and risk_level == RiskLevel.LOW:
            request.status = ApprovalStatus.APPROVED
            request.approver = "auto_approval"
            request.approved_at = datetime.now()
            self.completed_requests[request.id] = request
            return request

        # Add to pending requests
        self.pending_requests[request.id] = request

        # Trigger approval handlers
        await self._trigger_approval_handlers(request)

        return request

    async def _trigger_approval_handlers(self, request: ApprovalRequest) -> None:
        """Trigger approval handlers for a request."""

        handlers = self.approval_handlers.get(request.risk_level, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(request)
                else:
                    handler(request)
            except Exception as e:
                print(f"Error in approval handler: {e}")

    def approve_request(
        self, request_id: str, approver: str, reason: Optional[str] = None
    ) -> bool:
        """Approve a pending request."""

        if request_id not in self.pending_requests:
            return False

        request = self.pending_requests[request_id]

        # Check if expired
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            self.completed_requests[request_id] = request
            del self.pending_requests[request_id]
            return False

        # Approve the request
        request.status = ApprovalStatus.APPROVED
        request.approver = approver
        request.approved_at = datetime.now()
        if reason:
            request.metadata["approval_reason"] = reason

        # Move to completed
        self.completed_requests[request_id] = request
        del self.pending_requests[request_id]

        return True

    def deny_request(self, request_id: str, approver: str, reason: str) -> bool:
        """Deny a pending request."""

        if request_id not in self.pending_requests:
            return False

        request = self.pending_requests[request_id]

        # Deny the request
        request.status = ApprovalStatus.DENIED
        request.approver = approver
        request.approved_at = datetime.now()
        request.denial_reason = reason

        # Move to completed
        self.completed_requests[request_id] = request
        del self.pending_requests[request_id]

        return True

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request."""

        if request_id not in self.pending_requests:
            return False

        request = self.pending_requests[request_id]
        request.status = ApprovalStatus.CANCELLED

        # Move to completed
        self.completed_requests[request_id] = request
        del self.pending_requests[request_id]

        return True

    async def wait_for_approval(
        self, request_id: str, check_interval: float = 1.0
    ) -> ApprovalStatus:
        """Wait for an approval request to be resolved."""

        while request_id in self.pending_requests:
            request = self.pending_requests[request_id]

            # Check if expired
            if request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                self.completed_requests[request_id] = request
                del self.pending_requests[request_id]
                return ApprovalStatus.EXPIRED

            # Wait before checking again
            await asyncio.sleep(check_interval)

        # Request was resolved
        if request_id in self.completed_requests:
            return self.completed_requests[request_id].status

        return ApprovalStatus.CANCELLED

    def get_pending_requests(
        self, risk_level: Optional[RiskLevel] = None
    ) -> List[ApprovalRequest]:
        """Get all pending approval requests."""

        requests = list(self.pending_requests.values())

        if risk_level:
            requests = [r for r in requests if r.risk_level == risk_level]

        return sorted(requests, key=lambda r: r.requested_at)

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get an approval request by ID."""

        if request_id in self.pending_requests:
            return self.pending_requests[request_id]
        elif request_id in self.completed_requests:
            return self.completed_requests[request_id]

        return None

    def cleanup_expired_requests(self) -> List[str]:
        """Clean up expired requests and return their IDs."""

        expired_ids = []

        for request_id, request in list(self.pending_requests.items()):
            if request.is_expired():
                request.status = ApprovalStatus.EXPIRED
                self.completed_requests[request_id] = request
                del self.pending_requests[request_id]
                expired_ids.append(request_id)

        return expired_ids

    def add_approval_handler(
        self, risk_level: RiskLevel, handler: Callable[[ApprovalRequest], Any]
    ) -> None:
        """Add an approval handler for a specific risk level."""

        self.approval_handlers[risk_level].append(handler)

    def remove_approval_handler(
        self, risk_level: RiskLevel, handler: Callable[[ApprovalRequest], Any]
    ) -> bool:
        """Remove an approval handler."""

        try:
            self.approval_handlers[risk_level].remove(handler)
            return True
        except ValueError:
            return False

    def get_approval_stats(self) -> Dict[str, Any]:
        """Get approval workflow statistics."""

        pending_count = len(self.pending_requests)
        completed_count = len(self.completed_requests)

        # Count by status
        status_counts = {}
        for request in self.completed_requests.values():
            status = request.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Count by risk level
        risk_counts = {}
        all_requests = list(self.pending_requests.values()) + list(
            self.completed_requests.values()
        )
        for request in all_requests:
            risk = request.risk_level.value
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        return {
            "pending_requests": pending_count,
            "completed_requests": completed_count,
            "total_requests": pending_count + completed_count,
            "status_breakdown": status_counts,
            "risk_level_breakdown": risk_counts,
            "auto_approve_enabled": self.auto_approve_low_risk,
            "default_expiry_minutes": self.default_expiry_minutes,
        }
