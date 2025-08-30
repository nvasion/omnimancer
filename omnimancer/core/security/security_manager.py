"""Main security manager that coordinates all security components."""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .approval_workflow import ApprovalStatus, ApprovalWorkflow, RiskLevel
from .audit_logger import AuditEventType, AuditLevel, AuditLogger
from .permission_controller import (
    PermissionController,
    PermissionOperation,
)
from .sandbox_manager import ResourceLimits, SandboxManager


class SecurityManager:
    """Main security manager that coordinates all security components."""

    def __init__(
        self,
        enable_sandbox: bool = True,
        enable_approval_workflow: bool = True,
        enable_audit_logging: bool = True,
        default_resource_limits: Optional[ResourceLimits] = None,
        audit_log_file: Optional[str] = None,
    ):

        # Initialize components
        self.permissions = PermissionController()

        self.sandbox = (
            SandboxManager(default_resource_limits) if enable_sandbox else None
        )
        self.approval = ApprovalWorkflow() if enable_approval_workflow else None
        self.audit = AuditLogger(audit_log_file) if enable_audit_logging else None

        # Configuration
        self.enable_sandbox = enable_sandbox
        self.enable_approval_workflow = enable_approval_workflow
        self.enable_audit_logging = enable_audit_logging

        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.operation_counter = 0

        # Security policies
        self.security_policies = self._get_default_security_policies()

        if self.audit:
            self.audit.log_event(
                AuditEventType.SYSTEM_EVENT,
                AuditLevel.INFO,
                "SecurityManager initialized",
                session_id=self.session_id,
                metadata={
                    "sandbox_enabled": enable_sandbox,
                    "approval_workflow_enabled": enable_approval_workflow,
                    "audit_logging_enabled": enable_audit_logging,
                },
            )

    def _get_default_security_policies(self) -> Dict[str, Any]:
        """Get default security policies."""
        return {
            "require_approval_for_high_risk": True,
            "sandbox_all_commands": True,
            "log_all_operations": True,
            "block_system_directories": True,
            "block_credential_files": True,
            "max_command_timeout": 300,  # 5 minutes
            "max_file_size_mb": 100,
            "allowed_network_domains": [],
            "blocked_network_domains": ["*.internal", "*.local"],
        }

    async def validate_operation(
        self, operation: PermissionOperation
    ) -> Dict[str, Any]:
        """Validate if an operation is allowed and safe to execute."""

        operation_id = f"op_{self.operation_counter}"
        self.operation_counter += 1

        result = {
            "allowed": False,
            "operation_id": operation_id,
            "session_id": self.session_id,
            "reasons": [],
            "approval_required": False,
            "approval_request_id": None,
            "sandbox_required": False,
        }

        try:
            # Step 1: Basic permission check
            permission_allowed = self.permissions.validate_operation(operation)

            if self.audit:
                self.audit.log_permission_check(
                    operation.operation_type,
                    path=operation.path,
                    allowed=permission_allowed,
                    operation_id=operation_id,
                    session_id=self.session_id,
                    metadata=operation.metadata,
                )

            if not permission_allowed:
                result["reasons"].append("Permission denied by security policy")
                return result

            # Step 2: Risk assessment and approval workflow
            if self.approval and self.security_policies.get(
                "require_approval_for_high_risk", True
            ):
                risk_level = self.approval.assess_risk_level(
                    operation.operation_type, operation.metadata
                )

                # Check if approval is required
                if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    approval_request = await self.approval.request_approval(
                        operation.operation_type,
                        f"Operation: {operation.operation_type}",
                        operation.metadata,
                    )

                    if approval_request.status == ApprovalStatus.PENDING:
                        result["approval_required"] = True
                        result["approval_request_id"] = approval_request.id
                        result["reasons"].append(
                            f"Approval required for {risk_level.value} risk operation"
                        )
                        return result
                    elif approval_request.status != ApprovalStatus.APPROVED:
                        result["reasons"].append(
                            f"Operation not approved: {approval_request.status.value}"
                        )
                        return result

            # Step 3: Determine if sandboxing is required
            if self.sandbox and self.security_policies.get(
                "sandbox_all_commands", True
            ):
                if operation.command or operation.operation_type in [
                    "command_execute",
                    "file_write",
                    "file_create",
                ]:
                    result["sandbox_required"] = True

            # Step 4: All checks passed
            result["allowed"] = True
            result["reasons"].append("All security checks passed")

            return result

        except Exception as e:
            if self.audit:
                self.audit.log_event(
                    AuditEventType.SECURITY_ALERT,
                    AuditLevel.ERROR,
                    f"Error validating operation: {str(e)}",
                    operation_id=operation_id,
                    session_id=self.session_id,
                    metadata={
                        "operation_type": operation.operation_type,
                        "error": str(e),
                    },
                )

            result["reasons"].append(f"Security validation error: {str(e)}")
            return result

    async def execute_secure_command(
        self,
        command: Union[str, List[str]],
        working_dir: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        resource_limits: Optional[ResourceLimits] = None,
        require_approval: bool = False,
    ) -> Dict[str, Any]:
        """Execute a command with full security validation and sandboxing."""

        # Convert command to list if it's a string
        if isinstance(command, str):
            command_list = command.split()
        else:
            command_list = command
            command = " ".join(command_list)

        operation_id = f"cmd_{self.operation_counter}"
        self.operation_counter += 1

        result = {
            "success": False,
            "operation_id": operation_id,
            "session_id": self.session_id,
            "return_code": -1,
            "stdout": "",
            "stderr": "",
            "security_info": {},
        }

        try:
            # Create operation for validation
            operation = PermissionOperation(
                operation_type="command_execute",
                command=command,
                path=working_dir,
                env_vars=env_vars,
                require_approval=require_approval,
            )

            # Validate operation
            validation_result = await self.validate_operation(operation)
            result["security_info"] = validation_result

            if not validation_result["allowed"]:
                if self.audit:
                    self.audit.log_command_execution(
                        command,
                        success=False,
                        operation_id=operation_id,
                        session_id=self.session_id,
                        metadata={"validation_result": validation_result},
                    )

                result["stderr"] = (
                    f"Command blocked: {', '.join(validation_result['reasons'])}"
                )
                return result

            # Handle approval if required
            if validation_result.get("approval_required"):
                result["stderr"] = (
                    f"Approval required. Request ID: {validation_result['approval_request_id']}"
                )
                return result

            # Execute command (with or without sandbox)
            if validation_result.get("sandbox_required") and self.sandbox:
                # Execute in sandbox
                sandbox_result = self.sandbox.execute_sandboxed_command(
                    command_list,
                    working_dir=working_dir,
                    env_vars=env_vars,
                    limits=resource_limits,
                )

                result.update(
                    {
                        "success": sandbox_result["success"],
                        "return_code": sandbox_result["return_code"],
                        "stdout": sandbox_result["stdout"],
                        "stderr": sandbox_result["stderr"],
                    }
                )
                result["security_info"]["sandbox_dir"] = sandbox_result["sandbox_dir"]

            else:
                # Execute directly (not recommended for production)
                import subprocess

                try:
                    process = subprocess.run(
                        command_list,
                        cwd=working_dir,
                        env=env_vars,
                        capture_output=True,
                        text=True,
                        timeout=self.security_policies.get("max_command_timeout", 300),
                    )

                    result.update(
                        {
                            "success": process.returncode == 0,
                            "return_code": process.returncode,
                            "stdout": process.stdout,
                            "stderr": process.stderr,
                        }
                    )

                except subprocess.TimeoutExpired:
                    result["stderr"] = "Command timed out"
                except Exception as e:
                    result["stderr"] = f"Execution error: {str(e)}"

            # Log the execution
            if self.audit:
                self.audit.log_command_execution(
                    command,
                    success=result["success"],
                    exit_code=result["return_code"],
                    operation_id=operation_id,
                    session_id=self.session_id,
                    sandbox_id=result["security_info"].get("sandbox_dir"),
                    metadata={"validation_result": validation_result},
                )

            return result

        except Exception as e:
            if self.audit:
                self.audit.log_event(
                    AuditEventType.SECURITY_ALERT,
                    AuditLevel.ERROR,
                    f"Error executing secure command: {str(e)}",
                    operation_id=operation_id,
                    session_id=self.session_id,
                    metadata={"command": command, "error": str(e)},
                )

            result["stderr"] = f"Security manager error: {str(e)}"
            return result

    async def secure_file_access(
        self, file_path: str, operation: str, content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Securely access a file with full validation."""

        operation_id = f"file_{self.operation_counter}"
        self.operation_counter += 1

        result = {
            "success": False,
            "operation_id": operation_id,
            "session_id": self.session_id,
            "content": None,
            "error": None,
            "security_info": {},
        }

        try:
            # Create operation for validation
            op = PermissionOperation(
                operation_type=f"file_{operation}",
                path=file_path,
                content=content,
            )

            # Validate operation
            validation_result = await self.validate_operation(op)
            result["security_info"] = validation_result

            if not validation_result["allowed"]:
                if self.audit:
                    self.audit.log_file_access(
                        file_path,
                        operation,
                        allowed=False,
                        operation_id=operation_id,
                        session_id=self.session_id,
                        metadata={"validation_result": validation_result},
                    )

                result["error"] = (
                    f"File access blocked: {', '.join(validation_result['reasons'])}"
                )
                return result

            # Handle approval if required
            if validation_result.get("approval_required"):
                result["error"] = (
                    f"Approval required. Request ID: {validation_result['approval_request_id']}"
                )
                return result

            # Perform file operation
            file_path_obj = Path(file_path)

            if operation == "read":
                if file_path_obj.exists() and file_path_obj.is_file():
                    with open(file_path_obj, "r", encoding="utf-8") as f:
                        result["content"] = f.read()
                    result["success"] = True
                else:
                    result["error"] = "File does not exist"

            elif operation == "write":
                if content is None:
                    result["error"] = "No content provided for write operation"
                else:
                    # Check file size limits
                    max_size_mb = self.security_policies.get("max_file_size_mb", 100)
                    if len(content.encode("utf-8")) > max_size_mb * 1024 * 1024:
                        result["error"] = (
                            f"Content exceeds maximum file size ({max_size_mb}MB)"
                        )
                    else:
                        # Ensure directory exists
                        file_path_obj.parent.mkdir(parents=True, exist_ok=True)

                        with open(file_path_obj, "w", encoding="utf-8") as f:
                            f.write(content)
                        result["success"] = True

            elif operation == "delete":
                if file_path_obj.exists():
                    file_path_obj.unlink()
                    result["success"] = True
                else:
                    result["error"] = "File does not exist"

            else:
                result["error"] = f"Unsupported file operation: {operation}"

            # Log the file access
            if self.audit:
                self.audit.log_file_access(
                    file_path,
                    operation,
                    allowed=result["success"],
                    file_size=(len(content.encode("utf-8")) if content else None),
                    operation_id=operation_id,
                    session_id=self.session_id,
                    metadata={"validation_result": validation_result},
                )

            return result

        except Exception as e:
            if self.audit:
                self.audit.log_event(
                    AuditEventType.SECURITY_ALERT,
                    AuditLevel.ERROR,
                    f"Error in secure file access: {str(e)}",
                    operation_id=operation_id,
                    session_id=self.session_id,
                    metadata={
                        "file_path": file_path,
                        "operation": operation,
                        "error": str(e),
                    },
                )

            result["error"] = f"Security manager error: {str(e)}"
            return result

    def get_restricted_paths(self) -> List[str]:
        """Get list of restricted file paths."""
        return self.permissions.get_restricted_paths()

    def get_allowed_commands(self) -> List[str]:
        """Get list of allowed commands."""
        return self.permissions.get_allowed_commands()

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security manager status."""

        status = {
            "session_id": self.session_id,
            "components": {
                "permissions": True,
                "sandbox": self.sandbox is not None,
                "approval_workflow": self.approval is not None,
                "audit_logging": self.audit is not None,
            },
            "policies": self.security_policies,
            "operation_count": self.operation_counter,
        }

        # Add component-specific status
        if self.sandbox:
            status["sandbox_status"] = {
                "active_processes": self.sandbox.get_active_process_count(),
            }

        if self.approval:
            status["approval_status"] = self.approval.get_approval_stats()

        if self.audit:
            status["audit_status"] = self.audit.get_statistics()

        return status

    def update_security_policy(self, policy_name: str, value: Any) -> bool:
        """Update a security policy."""

        if policy_name in self.security_policies:
            old_value = self.security_policies[policy_name]
            self.security_policies[policy_name] = value

            if self.audit:
                self.audit.log_event(
                    AuditEventType.SYSTEM_EVENT,
                    AuditLevel.INFO,
                    f"Security policy updated: {policy_name}",
                    session_id=self.session_id,
                    metadata={
                        "policy_name": policy_name,
                        "old_value": old_value,
                        "new_value": value,
                    },
                )

            return True

        return False

    async def shutdown(self) -> None:
        """Shutdown the security manager gracefully."""

        if self.audit:
            self.audit.log_event(
                AuditEventType.SYSTEM_EVENT,
                AuditLevel.INFO,
                "SecurityManager shutting down",
                session_id=self.session_id,
                metadata={"operation_count": self.operation_counter},
            )

        # Cleanup components
        if self.sandbox:
            self.sandbox.cleanup_all_sandboxes()

        if self.approval:
            # Cancel pending approvals
            pending = self.approval.get_pending_requests()
            for request in pending:
                self.approval.cancel_request(request.id)

        if self.audit:
            self.audit.shutdown()

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        # Note: Cannot reliably run async cleanup in destructor
        # Cleanup should be called explicitly via shutdown()
        pass
