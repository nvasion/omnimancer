"""
Approval Manager CLI Integration for Omnimancer.

This module wires the EnhancedApprovalManager to the CLI workflow for
user-facing approval processes. It integrates approval dialog display
in terminal with clear formatting and captures approval decisions.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from ..core.agent.approval_manager import (
    BatchApprovalRequest,
    EnhancedApprovalManager,
)
from ..core.agent.types import Operation, OperationType
from ..core.security.approval_workflow import (
    ApprovalRequest,
)
from ..core.security.permission_controller import PermissionController
from .approval_formatter import CLIApprovalFormatter
from .approval_prompt import (
    ApprovalDecision,
    CLIApprovalPrompt,
)

logger = logging.getLogger(__name__)


class CLIApprovalIntegration:
    """
    Integration layer that connects EnhancedApprovalManager with CLI approval dialogs.

    This class handles the flow from approval request through user interaction
    to final approval decision, including permission storage for "remember" decisions.
    """

    def __init__(
        self,
        approval_manager: Optional[EnhancedApprovalManager] = None,
        permission_controller: Optional[PermissionController] = None,
        console: Optional[Console] = None,
        enable_auto_approval: bool = True,
        approval_timeout_seconds: int = 300,
    ):
        """
        Initialize CLI approval integration.

        Args:
            approval_manager: Enhanced approval manager instance
            permission_controller: Permission controller for storing decisions
            console: Rich console for display
            enable_auto_approval: Whether to enable auto-approval for remembered decisions
            approval_timeout_seconds: Default timeout for approval prompts
        """
        self.approval_manager = approval_manager or EnhancedApprovalManager()
        self.permission_controller = permission_controller or PermissionController()
        self.console = console or Console()
        self.enable_auto_approval = enable_auto_approval
        self.approval_timeout_seconds = approval_timeout_seconds

        # Initialize UI components
        self.formatter = CLIApprovalFormatter(self.console)
        self.prompt_handler = CLIApprovalPrompt(
            console=self.console,
            formatter=self.formatter,
            default_timeout_seconds=approval_timeout_seconds,
        )

        # Set approval callback on the manager
        self.approval_manager.set_approval_callback(self._handle_single_approval)
        self.approval_manager.set_batch_approval_callback(self._handle_batch_approval)

        # Track approval decisions for session
        self.approval_session_log: List[Dict[str, Any]] = []

    async def request_approval_for_operation(self, operation: Operation) -> bool:
        """
        Request approval for a single operation with CLI integration.

        Args:
            operation: Operation requiring approval

        Returns:
            True if approved, False if denied
        """
        try:
            # Check for auto-approval first if enabled
            if self.enable_auto_approval:
                auto_approval = await self._check_auto_approval(operation)
                if auto_approval is not None:
                    return auto_approval

            # Use the enhanced approval manager for full workflow
            approved = await self.approval_manager.request_single_approval(operation)

            # Log the decision
            self._log_approval_decision(operation, approved)

            return approved

        except Exception as e:
            logger.error(f"Error in approval request: {e}")
            self.console.print(f"[red]❌ Approval system error: {e}[/red]")
            return False

    async def request_batch_approval(
        self, operations: List[Operation]
    ) -> Dict[str, Any]:
        """
        Request approval for multiple operations with CLI integration.

        Args:
            operations: List of operations requiring approval

        Returns:
            Dictionary with approval results
        """
        try:
            # Use the enhanced approval manager for batch workflow
            batch_request = await self.approval_manager.request_batch_approval(
                operations
            )

            # Convert to result format
            summary = batch_request.get_approval_summary()

            result = {
                "approved_operations": list(batch_request.approved_operations),
                "total_operations": len(operations),
                "approval_summary": summary,
                "batch_id": batch_request.id,
            }

            # Log batch decision
            self._log_batch_approval_decision(operations, batch_request)

            return result

        except Exception as e:
            logger.error(f"Error in batch approval request: {e}")
            self.console.print(f"[red]❌ Batch approval system error: {e}[/red]")
            return {
                "approved_operations": [],
                "total_operations": len(operations),
                "approval_summary": {
                    "all_approved": False,
                    "approval_rate": 0.0,
                },
                "error": str(e),
            }

    async def _handle_single_approval(self, approval_context: Dict[str, Any]) -> bool:
        """
        Handle single operation approval through CLI dialog.

        Args:
            approval_context: Context dict with operation, preview, approval_request, etc.

        Returns:
            True if approved, False if denied
        """
        try:
            operation = approval_context["operation"]
            preview = approval_context.get("preview")
            approval_request = approval_context["approval_request"]

            # Present approval dialog and get user decision
            decision = await self.prompt_handler.prompt_for_approval(
                approval_request,
                preview,
                operation.data,
                self.approval_timeout_seconds,
            )

            # Handle "remember" decision
            if decision.should_remember and decision.is_approved:
                await self._store_approval_pattern(operation, approval_request)

            # Record decision in session log
            self._record_session_decision(operation, decision)

            return decision.is_approved

        except Exception as e:
            logger.error(f"Error in single approval handler: {e}")
            return False

    async def _handle_batch_approval(
        self, batch_request: BatchApprovalRequest
    ) -> Dict[str, Any]:
        """
        Handle batch operation approval through CLI dialog.

        Args:
            batch_request: Batch approval request

        Returns:
            Dictionary with approval result information
        """
        try:
            # Present batch approval dialog
            batch_decision = await self.prompt_handler.prompt_for_batch_approval(
                batch_request, self.approval_timeout_seconds
            )

            # Process the batch decision
            if batch_decision.decision_type == "approve_all":
                return {
                    "approve_all": True,
                    "approved_indices": list(range(len(batch_request.operations))),
                }
            elif batch_decision.decision_type == "deny_all":
                return {
                    "deny_all": True,
                    "reason": batch_decision.user_notes or "User denied all operations",
                }
            elif batch_decision.decision_type == "selective":
                return {"approved_indices": batch_decision.approved_indices}
            elif batch_decision.decision_type == "individual":
                # Handle individual decisions with potential "remember" actions
                approved_indices = []
                for i, decision in enumerate(batch_decision.individual_decisions):
                    if decision.is_approved:
                        approved_indices.append(i)

                        # Store remember patterns for individual decisions
                        if decision.should_remember:
                            operation = batch_request.operations[i]
                            # Create minimal approval request for pattern storage
                            temp_request = ApprovalRequest(
                                operation_type=operation.type.value,
                                description=operation.description,
                                metadata=operation.data,
                            )
                            await self._store_approval_pattern(operation, temp_request)

                return {"approved_indices": approved_indices}
            else:
                # Cancelled or timeout
                return {
                    "deny_all": True,
                    "reason": f"Batch approval {batch_decision.decision_type}: {batch_decision.user_notes}",
                }

        except Exception as e:
            logger.error(f"Error in batch approval handler: {e}")
            return {
                "deny_all": True,
                "reason": f"Error in batch approval: {str(e)}",
            }

    async def _check_auto_approval(self, operation: Operation) -> Optional[bool]:
        """
        Check if operation should be auto-approved based on stored patterns.

        Args:
            operation: Operation to check

        Returns:
            True if auto-approved, False if auto-denied, None if no match
        """
        if not self.enable_auto_approval:
            return None

        try:
            # Generate operation signature for matching
            operation_signature = self._generate_operation_signature(operation)

            # Check permission controller for matching approvals
            has_permission = (
                await self.permission_controller.check_operation_permission(
                    operation.type.value, operation_signature, operation.data
                )
            )

            if has_permission:
                # Show auto-approval message
                self.console.print(
                    f"[green]✅ Auto-approved: {operation.description} (remembered decision)[/green]"
                )
                return True

            return None

        except Exception as e:
            logger.error(f"Error checking auto-approval: {e}")
            return None

    async def _store_approval_pattern(
        self, operation: Operation, approval_request: ApprovalRequest
    ):
        """
        Store approval pattern for future auto-approval.

        Args:
            operation: The approved operation
            approval_request: The approval request context
        """
        try:
            # Generate operation signature
            operation_signature = self._generate_operation_signature(operation)

            # Store in permission controller
            await self.permission_controller.grant_operation_permission(
                operation_type=operation.type.value,
                operation_signature=operation_signature,
                metadata={
                    "operation_data": operation.data,
                    "approval_request_id": approval_request.id,
                    "risk_level": approval_request.risk_level.value,
                    "stored_reason": "user_remember_decision",
                },
                expires_hours=24 * 30,  # 30 days default
            )

            self.console.print(
                f"[blue]🧠 Stored approval pattern for similar '{operation.type.value}' operations[/blue]"
            )

        except Exception as e:
            logger.error(f"Error storing approval pattern: {e}")
            self.console.print(
                f"[yellow]⚠️  Failed to store approval pattern: {e}[/yellow]"
            )

    def _generate_operation_signature(self, operation: Operation) -> str:
        """
        Generate a signature for operation pattern matching.

        Args:
            operation: Operation to generate signature for

        Returns:
            String signature for pattern matching
        """
        # Create signature based on operation type and key parameters
        signature_parts = [operation.type.value]

        # Add type-specific signature components
        if operation.type in [
            OperationType.FILE_READ,
            OperationType.FILE_WRITE,
            OperationType.FILE_DELETE,
        ]:
            if "path" in operation.data:
                path = Path(operation.data["path"])
                # Use directory for pattern matching (allows similar files)
                signature_parts.append(f"dir:{path.parent}")
                # Also include file extension for type matching
                if path.suffix:
                    signature_parts.append(f"ext:{path.suffix}")

        elif operation.type == OperationType.COMMAND_EXECUTE:
            if "command" in operation.data:
                command = operation.data["command"]
                # Use base command name for matching
                base_command = command.split()[0] if command.split() else command
                signature_parts.append(f"cmd:{base_command}")

        elif operation.type == OperationType.WEB_REQUEST:
            if "url" in operation.data:
                from urllib.parse import urlparse

                parsed_url = urlparse(operation.data["url"])
                # Use domain for matching
                signature_parts.append(f"domain:{parsed_url.netloc}")
                # Include method
                method = operation.data.get("method", "GET")
                signature_parts.append(f"method:{method}")

        return "|".join(signature_parts)

    def _log_approval_decision(self, operation: Operation, approved: bool):
        """Log approval decision for audit trail."""
        log_entry = {
            "timestamp": (
                operation.created_at.isoformat() if operation.created_at else None
            ),
            "operation_type": operation.type.value,
            "operation_description": operation.description,
            "approved": approved,
            "operation_signature": self._generate_operation_signature(operation),
        }

        self.approval_session_log.append(log_entry)

        # Limit session log size
        if len(self.approval_session_log) > 100:
            self.approval_session_log = self.approval_session_log[-100:]

    def _log_batch_approval_decision(
        self, operations: List[Operation], batch_request: BatchApprovalRequest
    ):
        """Log batch approval decision for audit trail."""
        summary = batch_request.get_approval_summary()

        for i, operation in enumerate(operations):
            approved = i in batch_request.approved_operations
            self._log_approval_decision(operation, approved)

        # Add batch summary log
        batch_log_entry = {
            "timestamp": batch_request.created_at.isoformat(),
            "operation_type": "batch_approval",
            "operation_description": f"Batch of {len(operations)} operations",
            "approved": summary["all_approved"],
            "batch_summary": summary,
            "batch_id": batch_request.id,
        }

        self.approval_session_log.append(batch_log_entry)

    def _record_session_decision(
        self, operation: Operation, decision: ApprovalDecision
    ):
        """Record detailed session decision information."""
        decision_record = {
            "operation_type": operation.type.value,
            "operation_description": operation.description,
            "decision_type": decision.decision.value,
            "approved": decision.is_approved,
            "remember": decision.should_remember,
            "response_time_seconds": decision.response_time_seconds,
            "timeout_occurred": decision.timeout_occurred,
            "user_notes": decision.user_notes,
            "timestamp": decision.created_at.isoformat(),
        }

        # Store in session log
        session_entry = {
            **decision_record,
            "operation_signature": self._generate_operation_signature(operation),
        }

        self.approval_session_log.append(session_entry)

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about approval decisions in this session."""
        if not self.approval_session_log:
            return {"total_decisions": 0}

        total = len(
            [entry for entry in self.approval_session_log if "decision_type" in entry]
        )
        approved = len(
            [
                entry
                for entry in self.approval_session_log
                if entry.get("approved", False)
            ]
        )
        remembered = len(
            [
                entry
                for entry in self.approval_session_log
                if entry.get("remember", False)
            ]
        )

        # Calculate average response time
        response_times = [
            entry.get("response_time_seconds", 0)
            for entry in self.approval_session_log
            if "response_time_seconds" in entry
        ]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        return {
            "total_decisions": total,
            "approved_decisions": approved,
            "denied_decisions": total - approved,
            "remembered_decisions": remembered,
            "approval_rate": approved / total if total > 0 else 0,
            "remember_rate": remembered / total if total > 0 else 0,
            "average_response_time_seconds": avg_response_time,
            "recent_decisions": (
                self.approval_session_log[-10:] if self.approval_session_log else []
            ),
        }

    def add_no_approval_flag_support(self, no_approval_enabled: bool = False):
        """
        Configure support for --no-approval flag.

        Args:
            no_approval_enabled: Whether --no-approval flag is active
        """
        if no_approval_enabled:
            self.console.print(
                "[yellow]⚠️  WARNING: Approval system bypassed with --no-approval flag[/yellow]"
            )
            self.console.print(
                "[yellow]   All operations will be executed without user confirmation[/yellow]"
            )

            # Set approval manager to auto-approve everything
            self.approval_manager.approval_callback = lambda ctx: True
            self.enable_auto_approval = True
        else:
            # Restore normal approval behavior
            self.approval_manager.set_approval_callback(self._handle_single_approval)

    async def cleanup(self):
        """Clean up resources and save session information."""
        try:
            # Save session statistics if significant activity
            if len(self.approval_session_log) > 0:
                stats = self.get_session_statistics()
                logger.info(
                    f"Approval session completed with {stats['total_decisions']} decisions"
                )

            # Clean up approval manager resources
            if self.approval_manager:
                await self._cleanup_approval_manager()

            # Clean up expired approvals in permission controller
            if self.permission_controller:
                expired_count = self.permission_controller.cleanup_expired_approvals()
                if expired_count > 0:
                    logger.info(
                        f"Cleaned up {expired_count} expired approvals during shutdown"
                    )

            # Clean up prompt handler resources
            if self.prompt_handler:
                self.prompt_handler.cleanup_resources()

            # Clear session log
            self.approval_session_log.clear()

        except Exception as e:
            logger.error(f"Error during approval integration cleanup: {e}")

    async def _cleanup_approval_manager(self):
        """Clean up approval manager resources and pending requests."""
        try:
            # Cancel any pending approval requests
            await self._cancel_pending_approvals()

            # Clean up expired batch requests
            if hasattr(self.approval_manager, "cleanup_expired_requests"):
                expired_count = self.approval_manager.cleanup_expired_requests()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired batch requests")

            # Log pending operations that will be cancelled
            pending_operations = getattr(
                self.approval_manager, "pending_operations", {}
            )
            if pending_operations:
                logger.warning(
                    f"Cancelling {len(pending_operations)} pending approval operations"
                )

        except Exception as e:
            logger.error(f"Error cleaning up approval manager: {e}")

    async def _cancel_pending_approvals(self):
        """Cancel any pending approval requests gracefully."""
        try:
            # Check if approval manager has pending requests
            if hasattr(self.approval_manager, "pending_batches"):
                pending_batches = self.approval_manager.pending_batches
                if pending_batches:
                    logger.info(
                        f"Cancelling {len(pending_batches)} pending batch approvals"
                    )

                    # Cancel each pending batch
                    cancelled_batches = []
                    for batch_id, batch_request in pending_batches.items():
                        try:
                            # Mark as cancelled
                            batch_request.status = getattr(
                                batch_request, "status", None
                            )
                            if hasattr(batch_request, "cancel"):
                                batch_request.cancel("System shutdown")
                            cancelled_batches.append(batch_id)

                            # Log the cancellation
                            self._log_cancelled_batch(batch_request)

                        except Exception as batch_error:
                            logger.error(
                                f"Error cancelling batch {batch_id}: {batch_error}"
                            )

                    # Move cancelled batches to completed
                    if hasattr(self.approval_manager, "completed_batches"):
                        for batch_id in cancelled_batches:
                            batch_request = pending_batches[batch_id]
                            self.approval_manager.completed_batches[batch_id] = (
                                batch_request
                            )
                            del pending_batches[batch_id]

        except Exception as e:
            logger.error(f"Error cancelling pending approvals: {e}")

    def _log_cancelled_batch(self, batch_request):
        """Log information about a cancelled batch request."""
        try:
            # Add cancellation entry to session log
            cancellation_log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation_type": "batch_cancellation",
                "operation_description": f'Cancelled batch of {len(getattr(batch_request, "operations", []))} operations',
                "approved": False,
                "batch_id": getattr(batch_request, "id", "unknown"),
                "cancellation_reason": "System shutdown",
            }

            self.approval_session_log.append(cancellation_log_entry)

        except Exception as e:
            logger.error(f"Error logging cancelled batch: {e}")

    async def handle_operation_cancellation(
        self, operation_id: str, reason: str = "User cancelled"
    ):
        """
        Handle cancellation of a specific operation with cleanup.

        Args:
            operation_id: ID of operation to cancel
            reason: Reason for cancellation
        """
        try:
            # Log the cancellation
            cancellation_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation_type": "operation_cancellation",
                "operation_description": f"Cancelled operation {operation_id}",
                "approved": False,
                "cancellation_reason": reason,
                "operation_id": operation_id,
            }

            self.approval_session_log.append(cancellation_entry)

            # Clean up any related approval state
            await self._cleanup_operation_state(operation_id)

            logger.info(f"Operation {operation_id} cancelled: {reason}")

        except Exception as e:
            logger.error(
                f"Error handling operation cancellation for {operation_id}: {e}"
            )

    async def _cleanup_operation_state(self, operation_id: str):
        """Clean up state related to a cancelled operation."""
        try:
            # Remove from any pending operation tracking
            if hasattr(self.approval_manager, "pending_operations"):
                pending_ops = self.approval_manager.pending_operations
                if operation_id in pending_ops:
                    del pending_ops[operation_id]

            # Clean up any temporary resources related to the operation
            # This could include temporary files, locks, etc.

        except Exception as e:
            logger.error(f"Error cleaning up operation state for {operation_id}: {e}")


# Utility functions for easy integration


async def create_cli_approval_integration(
    approval_manager: Optional[EnhancedApprovalManager] = None,
    permission_controller: Optional[PermissionController] = None,
    console: Optional[Console] = None,
    config: Optional[Dict[str, Any]] = None,
) -> CLIApprovalIntegration:
    """
    Create and configure CLI approval integration.

    Args:
        approval_manager: Enhanced approval manager instance
        permission_controller: Permission controller instance
        console: Rich console instance
        config: Configuration dictionary

    Returns:
        Configured CLIApprovalIntegration instance
    """
    config = config or {}

    integration = CLIApprovalIntegration(
        approval_manager=approval_manager,
        permission_controller=permission_controller,
        console=console,
        enable_auto_approval=config.get("enable_auto_approval", True),
        approval_timeout_seconds=config.get("approval_timeout_seconds", 300),
    )

    return integration


def inject_approval_integration_into_agent_engine(
    agent_engine, cli_approval_integration: CLIApprovalIntegration
):
    """
    Inject CLI approval integration into an agent engine.

    Args:
        agent_engine: Agent engine to modify
        cli_approval_integration: CLI approval integration instance
    """
    # Replace the approval manager's approval callback
    if hasattr(agent_engine, "approval") and agent_engine.approval:
        agent_engine.approval.set_approval_callback(
            cli_approval_integration._handle_single_approval
        )
        agent_engine.approval.set_batch_approval_callback(
            cli_approval_integration._handle_batch_approval
        )

        # Store reference for cleanup
        agent_engine._cli_approval_integration = cli_approval_integration

        logger.info("CLI approval integration injected into agent engine")
    else:
        logger.warning("Agent engine has no approval manager to integrate with")
