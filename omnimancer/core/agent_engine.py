"""
Agent Engine for Omnimancer CLI.

This module provides the AgentEngine class that extends CoreEngine with
autonomous operation capabilities including file system management,
program execution, web client operations, and approval workflows.
"""

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..events import emitter as fleet_events
from ..utils.errors import AgentError, PermissionError, SecurityError
from .agent.approval_interface import ApprovalInterface
from .agent.approval_manager import EnhancedApprovalManager
from .agent.file_system_manager import FileSystemManager as EnhancedFileSystemManager
from .agent.status_core import EventType
from .agent.types import Operation, OperationResult, OperationType
from .agent.workflow_orchestrator import WorkflowContext, WorkflowOrchestrator
from .config_manager import ConfigManager
from .engine import CoreEngine
from .fallback_manager import ProviderRank
from .mcp_integration_layer import (
    ExecutionPriority,
    ToolCapability,
    ToolExecutionContext,
)
from .models import PermissionRule, PermissionsConfig
from .security.approval_workflow import ApprovalWorkflow
from .security.permission_rules import PermissionDecision, PermissionRuleEngine

logger = logging.getLogger(__name__)


# Manager facades live in agent_managers; re-exported here so existing
# ``from omnimancer.core.agent_engine import ProgramExecutor`` imports keep working.
from .agent_managers import (  # noqa: E402
    ApprovalManager,
    BaseManager,
    MCPIntegrator,
    ProgramExecutor,
    ProviderFallback,
    WebClient,
)


class AgentEngine(CoreEngine):
    """
    Agent-enabled engine that extends CoreEngine with autonomous operation capabilities.

    This class adds file system management, program execution, web client operations,
    MCP tool integration, approval workflows, and provider fallback to the base engine.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        base_path: Optional[Path] = None,
    ):
        """
        Initialize the agent engine.

        Args:
            config_manager: Configuration manager instance
            base_path: Base path for file system operations
                (defaults to current directory)
        """
        super().__init__(config_manager)

        # Initialize enhanced approval system first
        self.approval_workflow = ApprovalWorkflow()
        self.enhanced_approval = EnhancedApprovalManager(self.approval_workflow)
        self.approval_interface = ApprovalInterface(self.enhanced_approval)
        self.approval = ApprovalManager()  # Keep legacy for backward compatibility

        # Initialize agent-specific managers with approval integration
        self.file_system = EnhancedFileSystemManager(
            approval_manager=self.enhanced_approval,
            require_approval=True,  # Enable approval by default for agent mode
        )

        # Setup autonomous file modification workflow
        self._setup_autonomous_file_workflow()
        self.executor = ProgramExecutor()
        self.web_client = WebClient()
        self.mcp_integrator = MCPIntegrator(self.mcp_manager)

        # Initialize workflow orchestrator for continuous multi-step execution
        self.workflow_orchestrator = WorkflowOrchestrator(
            file_system=self.file_system,
            approval_manager=self.enhanced_approval,
            executor=self.executor,
            engine=self,  # Pass engine reference for AI calls
        )

        self.fallback = ProviderFallback(self)

        # Agent state
        self.agent_mode_enabled: bool = False
        self.pending_operations: List[Operation] = []
        self.operation_history: List[Dict[str, Any]] = []
        self.current_workflow: Optional[str] = None
        self._session_permissions = PermissionsConfig()

    def set_read_only(self, enabled: bool) -> None:
        """Deny mutating and command operations for this process only.

        Args:
            enabled: Whether session-level read-only rules should be active.
        """
        denied_tools = (
            OperationType.FILE_WRITE,
            OperationType.FILE_DELETE,
            OperationType.DIRECTORY_CREATE,
            OperationType.DIRECTORY_DELETE,
            OperationType.COMMAND_EXECUTE,
        )
        rules = (
            [PermissionRule(tool=operation.value) for operation in denied_tools]
            if enabled
            else []
        )
        self._session_permissions = PermissionsConfig(always_deny=rules)

    def _permission_decision(self, tool: str, target: str = "") -> PermissionDecision:
        """Apply session rules before the persisted configuration rules."""
        session_config = getattr(self, "_session_permissions", None)
        session_decision = PermissionRuleEngine(session_config).evaluate(tool, target)
        if session_decision == PermissionDecision.DENY:
            return session_decision
        return super()._permission_decision(tool, target)

    def configure_approval_settings(
        self,
        require_approval: bool = True,
        enable_batch_approval: bool = True,
        max_batch_size: int = 10,
    ) -> None:
        """Configure approval system settings."""
        self.file_system.require_approval = require_approval
        self.enhanced_approval.enable_batch_approval = enable_batch_approval
        self.enhanced_approval.max_batch_size = max_batch_size

    def set_approval_callbacks(
        self,
        approval_callback: Optional[Callable[..., Any]] = None,
        batch_approval_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Set custom approval callbacks for user interaction."""
        if approval_callback:
            self.enhanced_approval.set_approval_callback(approval_callback)
        if batch_approval_callback:
            self.enhanced_approval.set_batch_approval_callback(batch_approval_callback)

    @staticmethod
    def _hook_context_for_operation(
        operation: Operation,
    ) -> Tuple[Dict[str, Any], str]:
        """Build a JSON-safe hook payload and matcher target for an operation.

        The match target is the most specific actionable string available (the
        command, path, or URL) so matchers like ``^rm\\b`` work intuitively;
        it falls back to the operation type otherwise.
        """
        op_type = (
            operation.type.value
            if hasattr(operation.type, "value")
            else str(operation.type)
        )
        data = operation.data or {}
        target = ""
        for key in ("command", "path", "file_path", "url"):
            value = data.get(key)
            if isinstance(value, str) and value:
                target = value
                break
        context: Dict[str, Any] = {
            "tool": op_type,
            "description": operation.description,
            "requires_approval": operation.requires_approval,
        }
        if target:
            context["target"] = target
        return context, target or op_type

    async def execute_with_approval(self, operation: Operation) -> OperationResult:
        """
        Execute operation with approval workflow.

        Args:
            operation: Operation to execute

        Returns:
            Result of the operation
        """
        fleet_op_id: Optional[str] = None
        hook_ctx: Optional[Dict[str, Any]] = None
        match_target = ""
        try:
            # Generate preview
            preview = await self._generate_preview(operation)
            operation.preview = preview

            hook_ctx, match_target = self._hook_context_for_operation(operation)
            op_tool = hook_ctx["tool"]

            # Fleet event feed: tool_start + operation tracking. Emission is
            # drop-on-full and no-ops when events are disabled — it can never
            # stall or reorder the gate stages below.
            event_data: Dict[str, Any] = {
                "tool": operation.data.get("_tool_name", op_tool),
                "op_type": op_tool,
                "description": operation.description,
                "requires_approval": operation.requires_approval,
                "invocation": operation.data.get("_invocation", "marker"),
            }
            if hook_ctx.get("target"):
                event_data["target"] = hook_ctx["target"]
            fleet_op_id = await fleet_events.start_tool_operation(
                operation.type, operation.description, event_data
            )
            if fleet_op_id is not None:
                # Lets the command manager attach tool_progress events from
                # live process output to this operation.
                operation.data["_fleet_op_id"] = fleet_op_id

            # Apply config-driven permission rules (deny > ask > allow).
            decision = self._permission_decision(op_tool, match_target)
            if decision == PermissionDecision.DENY:
                await fleet_events.emit_event(
                    EventType.APPROVAL_DENIED,
                    {
                        "tool": event_data["tool"],
                        "target": event_data.get("target"),
                        "source": "permission_rule",
                    },
                    operation_id=fleet_op_id,
                )
                await fleet_events.end_tool_operation(
                    fleet_op_id,
                    success=False,
                    error=f"denied by permission rule ({op_tool})",
                )
                return OperationResult(
                    success=False,
                    error=f"Operation denied by permission rule ({op_tool}).",
                )
            elif decision == PermissionDecision.ALLOW:
                # Auto-approve and authorize sensitive project-local writes.
                operation.requires_approval = False
            elif decision == PermissionDecision.ASK:
                # Force a prompt even if it would otherwise be auto/remembered.
                operation.requires_approval = True
                operation.data["_force_prompt"] = True

            # Fire tool_use_request hooks; a blocking hook can veto the tool.
            outcome = await self._fire_hook(
                "tool_use_request", hook_ctx, match_target=match_target
            )
            if not outcome.allowed:
                await fleet_events.emit_event(
                    EventType.APPROVAL_DENIED,
                    {
                        "tool": event_data["tool"],
                        "target": event_data.get("target"),
                        "source": "hook",
                    },
                    operation_id=fleet_op_id,
                )
                await fleet_events.end_tool_operation(
                    fleet_op_id,
                    success=False,
                    error=f"blocked by {outcome.reason}",
                )
                return OperationResult(
                    success=False,
                    error=f"Operation blocked by {outcome.reason}.",
                )

            # Request approval if needed
            if operation.requires_approval:
                await fleet_events.emit_event(
                    EventType.APPROVAL_REQUESTED,
                    {
                        "tool": event_data["tool"],
                        "target": event_data.get("target"),
                    },
                    operation_id=fleet_op_id,
                )
                approval_result = await self.approval.request_approval(operation)

                # Handle both old bool and new tuple return
                # for backward compatibility
                if isinstance(approval_result, tuple):
                    approved, was_cancelled = approval_result
                else:
                    # Legacy bool return
                    approved = approval_result
                    was_cancelled = operation.data.get("was_cancelled", False)

                if not approved:
                    await fleet_events.emit_event(
                        EventType.APPROVAL_DENIED,
                        {
                            "tool": event_data["tool"],
                            "target": event_data.get("target"),
                            "source": "user",
                            "cancelled": was_cancelled,
                        },
                        operation_id=fleet_op_id,
                    )
                    await fleet_events.end_tool_operation(
                        fleet_op_id,
                        success=False,
                        error=(
                            "cancelled by user"
                            if was_cancelled
                            else "not approved by user"
                        ),
                        was_cancelled=was_cancelled,
                    )
                    return OperationResult(
                        success=False,
                        error=(
                            "User cancelled operation"
                            if was_cancelled
                            else "Operation not approved by user"
                        ),
                        was_cancelled=was_cancelled,
                    )
                await fleet_events.emit_event(
                    EventType.APPROVAL_GRANTED,
                    {
                        "tool": event_data["tool"],
                        "target": event_data.get("target"),
                    },
                    operation_id=fleet_op_id,
                )
                # Set approval flag to enable security check in execute_operation
                operation.data["_approval_granted"] = True

            # Execute operation using appropriate manager
            result = await self._execute_operation(operation)

            # Observe-only post-execution hooks.
            post_ctx = dict(hook_ctx)
            post_ctx["success"] = result.success
            if result.error:
                post_ctx["error"] = result.error
            await self._fire_hook("post_tool", post_ctx, match_target=match_target)

            # Record in history
            self.operation_history.append(
                {
                    "operation": operation,
                    "result": result,
                    "timestamp": time.time(),
                }
            )

            await fleet_events.end_tool_operation(
                fleet_op_id,
                success=result.success,
                error=result.error,
                was_cancelled=bool(getattr(result, "was_cancelled", False)),
            )

            return result

        except asyncio.CancelledError:
            # A cancelled turn (Ctrl+C) must still close the event lifecycle
            # or tool_start records stay unmatched. Shielded because the
            # emission must survive the surrounding cancellation; hooks are
            # deliberately not fired here (no subprocess spawns during
            # teardown). The cancellation always propagates.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.shield(
                    fleet_events.end_tool_operation(
                        fleet_op_id,
                        success=False,
                        error="turn cancelled",
                        was_cancelled=True,
                    )
                )
            raise

        except Exception as e:
            # Enhanced error context for agent operations
            error_context = {
                "operation_type": (
                    operation.type.value
                    if hasattr(operation.type, "value")
                    else str(operation.type)
                ),
                "operation_data": operation.data,
                "requires_approval": operation.requires_approval,
                "error_type": type(e).__name__,
                "managers_enabled": {
                    "file_system": getattr(self.file_system, "enabled", True),
                    "program_executor": getattr(self.executor, "enabled", True),
                    "web_client": getattr(self.web_client, "enabled", True),
                    "mcp_integrator": getattr(self.mcp_integrator, "enabled", True),
                },
            }
            logger.error(
                f"Operation execution failed: {e}",
                extra={"context": error_context},
            )

            # Create user-friendly error message with context
            if isinstance(e, AgentError):
                user_error = f"Agent operation failed: {e}"
            elif isinstance(e, SecurityError):
                user_error = f"Security violation: {e}"
            elif isinstance(e, PermissionError):
                user_error = f"Permission denied: {e}"
            elif "timeout" in str(e).lower():
                user_error = (
                    "Operation timeout: The operation took too long to complete"
                )
            else:
                user_error = f"Operation execution failed: {e}"

            # Terminal observability on crash: previously an exception skipped
            # post_tool entirely, so hooks and the event feed never saw the
            # operation end. _fire_hook never raises.
            if hook_ctx is not None:
                post_ctx = dict(hook_ctx)
                post_ctx["success"] = False
                post_ctx["error"] = str(e)
                await self._fire_hook("post_tool", post_ctx, match_target=match_target)
            await fleet_events.end_tool_operation(
                fleet_op_id, success=False, error=user_error
            )

            return OperationResult(
                success=False,
                error=user_error,
                details=(
                    f"Operation: {operation.type}, "
                    f"Error type: {type(e).__name__}, "
                    f"Context: {error_context}"
                ),
            )

    async def execute_with_enhanced_approval(
        self, operation: Operation
    ) -> OperationResult:
        """Execute operation with enhanced approval workflow.

        Args:
            operation: Operation to execute

        Returns:
            Result of the operation
        """
        try:
            # Request approval through enhanced system if needed
            if operation.requires_approval:
                approved = await self.enhanced_approval.request_single_approval(
                    operation
                )
                if not approved:
                    return OperationResult(
                        success=False, error="Operation not approved by user"
                    )

            # Execute operation using appropriate manager
            result = await self._execute_operation(operation)

            # Record in history
            self.operation_history.append(
                {
                    "operation": operation,
                    "result": result,
                    "timestamp": time.time(),
                }
            )

            return result

        except Exception as e:
            # Enhanced error context for enhanced approval operations
            error_context = {
                "operation_type": (
                    operation.type.value
                    if hasattr(operation.type, "value")
                    else str(operation.type)
                ),
                "operation_data": operation.data,
                "requires_approval": operation.requires_approval,
                "error_type": type(e).__name__,
                "enhanced_approval_enabled": hasattr(self, "enhanced_approval"),
            }
            logger.error(
                f"Enhanced approval operation execution failed: {e}",
                extra={"context": error_context},
            )

            # Create user-friendly error message with context
            if isinstance(e, AgentError):
                user_error = f"Enhanced approval operation failed: {e}"
            elif isinstance(e, SecurityError):
                user_error = f"Security violation in enhanced approval: {e}"
            elif isinstance(e, PermissionError):
                user_error = f"Permission denied in enhanced approval: {e}"
            else:
                user_error = f"Enhanced approval operation failed: {e}"

            return OperationResult(
                success=False,
                error=user_error,
                details=(
                    f"Operation: {operation.type}, "
                    f"Error type: {type(e).__name__}, "
                    f"Context: {error_context}"
                ),
            )

    async def execute_batch_with_approval(
        self, operations: List[Operation]
    ) -> List[OperationResult]:
        """
        Execute a batch of operations with enhanced batch approval workflow.

        Args:
            operations: List of operations to execute

        Returns:
            List of operation results
        """
        try:
            # Filter operations that require approval
            approval_required_ops = [op for op in operations if op.requires_approval]
            no_approval_ops = [op for op in operations if not op.requires_approval]

            results = []

            # Execute operations that don't require approval
            for operation in no_approval_ops:
                result = await self._execute_operation(operation)
                results.append(result)
                self.operation_history.append(
                    {
                        "operation": operation,
                        "result": result,
                        "timestamp": time.time(),
                    }
                )

            # Handle batch approval for operations that require it
            if approval_required_ops:
                batch_request = await self.enhanced_approval.request_batch_approval(
                    approval_required_ops
                )

                # Execute approved operations
                for i, operation in enumerate(approval_required_ops):
                    if i in batch_request.approved_operations:
                        result = await self._execute_operation(operation)
                        results.append(result)
                        self.operation_history.append(
                            {
                                "operation": operation,
                                "result": result,
                                "timestamp": time.time(),
                            }
                        )
                    else:
                        # Operation was not approved
                        results.append(
                            OperationResult(
                                success=False,
                                error="Operation not approved in batch",
                            )
                        )

            return results

        except Exception as e:
            # Enhanced error context for batch operations
            error_context = {
                "operation_count": len(operations),
                "operations_with_approval": len(
                    [op for op in operations if op.requires_approval]
                ),
                "operation_types": [
                    (op.type.value if hasattr(op.type, "value") else str(op.type))
                    for op in operations
                ],
                "error_type": type(e).__name__,
            }
            logger.error(
                f"Batch operation execution failed: {e}",
                extra={"context": error_context},
            )

            # Create user-friendly error message
            if isinstance(e, AgentError):
                user_error = f"Batch agent operations failed: {e}"
            elif isinstance(e, SecurityError):
                user_error = f"Security violation in batch operations: {e}"
            else:
                user_error = f"Batch operation execution failed: {e}"

            # Return failure result for each operation with context
            return [
                OperationResult(
                    success=False,
                    error=user_error,
                    details=(
                        f"Batch error: {type(e).__name__}, "
                        f"Total operations: {len(operations)}"
                    ),
                )
                for _ in operations
            ]

    async def _generate_preview(self, operation: Operation) -> str:
        """Generate preview of operation."""
        manager = self._get_manager_for_operation(operation)
        if manager:
            return await manager.preview_operation(operation)
        return f"Unknown operation: {operation.type.value}"

    async def _execute_operation(self, operation: Operation) -> OperationResult:
        """Execute operation using appropriate manager."""
        manager = self._get_manager_for_operation(operation)
        if not manager:
            return OperationResult(
                success=False,
                error=f"No manager available for operation: {operation.type}",
            )

        return await manager.execute_operation(operation)

    def _get_manager_for_operation(
        self, operation: Operation
    ) -> Optional[Union[BaseManager, EnhancedFileSystemManager]]:
        """Get appropriate manager for operation type."""
        if operation.type in [
            OperationType.FILE_READ,
            OperationType.FILE_WRITE,
            OperationType.FILE_DELETE,
            OperationType.DIRECTORY_CREATE,
            OperationType.DIRECTORY_DELETE,
        ]:
            return self.file_system
        elif operation.type == OperationType.COMMAND_EXECUTE:
            return self.executor
        elif operation.type == OperationType.WEB_REQUEST:
            return self.web_client
        elif operation.type == OperationType.MCP_TOOL_CALL:
            return self.mcp_integrator
        return None

    def enable_agent_mode(self) -> None:
        """Enable autonomous agent mode."""
        self.agent_mode_enabled = True
        logger.info("Agent mode enabled")

    def disable_agent_mode(self) -> None:
        """Disable autonomous agent mode."""
        self.agent_mode_enabled = False
        logger.info("Agent mode disabled")

    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of executed operations."""
        return self.operation_history.copy()

    def clear_operation_history(self) -> None:
        """Clear operation history."""
        self.operation_history.clear()

    async def rollback_operation(self, operation_index: int) -> bool:
        """
        Attempt to rollback a previous operation.

        Args:
            operation_index: Index of operation in history to rollback

        Returns:
            True if rollback was successful, False otherwise
        """
        if operation_index >= len(self.operation_history):
            return False

        history_entry = self.operation_history[operation_index]
        operation = history_entry["operation"]
        result = history_entry["result"]

        if not result.rollback_data:
            logger.warning(
                f"No rollback data available for operation {operation_index}"
            )
            return False

        try:
            # Attempt rollback based on operation type
            if operation.type == OperationType.FILE_WRITE:
                # Restore backup content
                path = operation.data["path"]
                backup_content = result.rollback_data["backup_content"]
                rollback_op = Operation(
                    type=OperationType.FILE_WRITE,
                    description=f"Rollback write to {path}",
                    data={
                        "path": path,
                        "content": backup_content,
                        "create_backup": False,
                    },
                    requires_approval=False,
                )
                rollback_result = await self.execute_with_approval(rollback_op)
                return rollback_result.success

            elif operation.type == OperationType.FILE_DELETE:
                # Restore deleted file
                path = result.rollback_data["path"]
                backup_content = result.rollback_data["backup_content"]
                rollback_op = Operation(
                    type=OperationType.FILE_WRITE,
                    description=f"Restore deleted file {path}",
                    data={
                        "path": path,
                        "content": backup_content,
                        "create_backup": False,
                    },
                    requires_approval=False,
                )
                rollback_result = await self.execute_with_approval(rollback_op)
                return rollback_result.success

            # Add more rollback logic for other operation types as needed

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

        return False

    def configure_fallback_providers(
        self,
        providers: List[str],
        rankings: Optional[Dict[str, ProviderRank]] = None,
    ) -> None:
        """
        Configure fallback providers with optional rankings.

        Args:
            providers: List of provider names in priority order
            rankings: Optional dict mapping provider names to rankings
        """
        self.fallback.enhanced_fallback.set_fallback_providers(providers, rankings)
        logger.info(f"Configured fallback providers: {providers}")

    def get_provider_fallback_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for provider fallback performance."""
        return self.fallback.enhanced_fallback.get_provider_stats()

    def get_fallback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent fallback history."""
        return self.fallback.enhanced_fallback.get_fallback_history(limit)

    def reset_provider_stats(self, provider_name: Optional[str] = None) -> None:
        """Reset fallback statistics for specific provider or all providers."""
        self.fallback.enhanced_fallback.reset_provider_stats(provider_name)
        logger.info(f"Reset fallback stats for {provider_name or 'all providers'}")

    def configure_circuit_breaker(
        self, threshold: int = 5, recovery_time: int = 600
    ) -> None:
        """
        Configure circuit breaker for provider fallback.

        Args:
            threshold: Number of consecutive failures before circuit break
            recovery_time: Recovery time in seconds before re-enabling provider
        """
        self.fallback.enhanced_fallback.configure_circuit_breaker(
            threshold, recovery_time
        )
        logger.info(
            "Configured circuit breaker: "
            f"threshold={threshold}, "
            f"recovery_time={recovery_time}s"
        )

    async def health_check_providers(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all configured providers."""
        return await self.fallback.enhanced_fallback.health_check_all_providers()

    async def initialize_mcp_integrator(self) -> bool:
        """Initialize the enhanced MCP integrator."""
        return await self.mcp_integrator.initialize()

    async def discover_mcp_tools(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Discover available MCP tools with descriptions."""
        return await self.mcp_integrator.discover_tools(force_refresh)

    def find_mcp_tools_by_capability(self, capability: ToolCapability) -> List[str]:
        """Find MCP tools that have a specific capability."""
        return self.mcp_integrator.find_tools_by_capability(capability)

    def find_best_mcp_tool_for_task(self, task_description: str) -> Optional[str]:
        """Find the best MCP tool for a given task description."""
        return self.mcp_integrator.find_best_tool_for_task(task_description)

    def get_mcp_tool_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for MCP tools."""
        return self.mcp_integrator.get_tool_metrics()

    def get_mcp_capability_summary(self) -> Dict[str, int]:
        """Get summary of MCP tools by capability."""
        return self.mcp_integrator.get_capability_summary()

    async def execute_mcp_tool_with_context(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        task_context: Optional[str] = None,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Execute an MCP tool with enhanced context and monitoring.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            task_context: Optional context description
            priority: Execution priority
            timeout: Timeout in seconds

        Returns:
            Dictionary with execution results and metadata
        """
        # Ensure MCP integrator is initialized
        await self.initialize_mcp_integrator()

        # Create execution context
        context = ToolExecutionContext(
            task_context=task_context,
            execution_priority=priority,
            timeout_seconds=timeout,
            metadata={"agent_engine_call": True},
        )

        # Execute tool
        result = (
            await self.mcp_integrator.enhanced_integrator.execute_tool_with_context(
                tool_name, arguments, context
            )
        )

        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "execution_time": result.execution_time,
            "server_name": result.server_name,
            "attempt_count": result.attempt_count,
            "context_used": result.context_used,
            "metadata": result.metadata,
        }

    async def health_check_mcp_integration(self) -> Dict[str, Any]:
        """Perform comprehensive health check on MCP integration."""
        return await self.mcp_integrator.health_check()

    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get statistics about approval history and usage."""
        return self.enhanced_approval.get_approval_statistics()

    def configure_approval_interface(
        self,
        show_colors: bool = True,
        auto_show_diff: bool = True,
        max_diff_lines: int = 50,
    ) -> None:
        """Configure the approval interface display options."""
        self.approval_interface.set_colors_enabled(show_colors)
        self.approval_interface.set_auto_show_diff(auto_show_diff)
        self.approval_interface.set_max_diff_lines(max_diff_lines)

    def enable_batch_approval(
        self, enabled: bool = True, max_batch_size: int = 10
    ) -> None:
        """Enable or disable batch approval functionality."""
        self.enhanced_approval.enable_batch_approval = enabled
        self.enhanced_approval.max_batch_size = max_batch_size

    def cleanup_expired_approval_requests(self) -> int:
        """Clean up expired approval requests and return count of cleaned items."""
        result = self.enhanced_approval.cleanup_expired_requests()
        return result  # type: ignore[no-any-return]

    def get_pending_approval_requests(self) -> Dict[str, Any]:
        """Get information about pending approval requests."""
        return {
            "pending_batches": len(self.enhanced_approval.pending_batches),
            "completed_batches": len(self.enhanced_approval.completed_batches),
            "approval_workflow_pending": len(self.approval_workflow.pending_requests),
            "approval_workflow_completed": len(
                self.approval_workflow.completed_requests
            ),
        }

    async def generate_operation_preview(self, operation: Operation) -> str:
        """Generate a detailed preview for an operation."""
        preview = await self.enhanced_approval.generate_operation_preview(operation)
        return preview.format_preview()

    def set_approval_auto_approve_low_risk(self, enabled: bool = True) -> None:
        """Enable or disable automatic approval of low-risk operations."""
        self.approval_workflow.auto_approve_low_risk = enabled

    # Directory awareness methods

    def get_current_working_directory(self) -> Path:
        """Get the current working directory."""
        return self.file_system.get_current_working_directory()

    async def is_git_repository(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Check if the given path (or current directory) is a Git repository."""
        return await self.file_system.is_git_repository(path)

    async def get_git_repository_root(
        self, path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """Get the root directory of the Git repository, if any."""
        return await self.file_system.get_git_repository_root(path)

    async def get_directory_context(
        self, path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive directory context."""
        return await self.file_system.get_directory_context(path)

    # Continuous workflow execution methods

    async def execute_continuous_workflow(
        self, workflow_name: str, parameters: Optional[Dict[str, Any]] = None
    ) -> WorkflowContext:
        """
        Execute a continuous multi-step workflow.

        This enables the AI to automatically flow through multiple operations
        without stopping, similar to how Claude Code works.

        Args:
            workflow_name: Name of the workflow to execute
            parameters: Optional parameters for the workflow

        Returns:
            WorkflowContext with execution results
        """
        logger.info(f"Starting continuous workflow: {workflow_name}")
        self.current_workflow = workflow_name

        # Create workflow context
        context = WorkflowContext(working_directory=Path.cwd())

        # Execute the workflow
        result = await self.workflow_orchestrator.execute_workflow(
            workflow_name, context=context, parameters=parameters
        )

        self.current_workflow = None
        return result

    async def analyze_workspace(self) -> WorkflowContext:
        """
        Analyze the current workspace automatically.

        This executes multiple steps in sequence:
        1. List directory contents
        2. Detect technology stack
        3. Check configuration files
        4. Analyze project structure
        5. Generate summary

        Returns:
            WorkflowContext with analysis results
        """
        return await self.execute_continuous_workflow("project_analysis")

    async def modify_file_with_workflow(
        self, file_path: str, changes: Dict[str, Any]
    ) -> WorkflowContext:
        """
        Modify a file using the continuous workflow.

        This executes multiple steps:
        1. Read original file
        2. Prepare changes
        3. Show diff for review
        4. Apply approved changes
        5. Validate changes

        Args:
            file_path: Path to the file to modify
            changes: Dictionary describing the changes

        Returns:
            WorkflowContext with modification results
        """
        parameters = {"file_path": file_path, "changes": changes}
        return await self.execute_continuous_workflow("file_modification", parameters)

    def register_custom_workflow(self, name: str, steps: List[Any]) -> None:
        """
        Register a custom workflow for continuous execution.

        Args:
            name: Name of the workflow
            steps: List of workflow steps
        """
        self.workflow_orchestrator.register_workflow(name, steps)

    # Read-before-write functionality

    async def write_file_with_review(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
        user_review_callback: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        """
        Write file with read-before-write review process.

        This method reads existing file content, presents it to the user for review
        alongside the new content, and allows the user to approve, modify, or reject
        the changes before writing.

        Args:
            path: Path to the file to be written
            content: New content to write
            encoding: File encoding for text files
            user_review_callback: Optional callback for user review interface

        Returns:
            Dict with operation result and review metadata
        """
        return await self.file_system.read_before_write(
            path=path,
            new_content=content,
            encoding=encoding,
            user_review_callback=user_review_callback,
        )

    async def preview_file_modification(
        self,
        path: Union[str, Path],
        new_content: Union[str, bytes],
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """
        Preview file modification without making changes.

        Args:
            path: Path to the file
            new_content: Proposed new content
            encoding: File encoding for text files

        Returns:
            Dict with preview information including diff
        """
        return await self.file_system.preview_file_modification(
            path=path, new_content=new_content, encoding=encoding
        )

    def set_read_before_write_callback(self, callback: Callable[..., Any]) -> None:
        """
        Set a default callback for read-before-write operations.

        This callback will be used when write_file_with_review is called
        without specifying a user_review_callback.

        Args:
            callback: Async function that takes review_data and returns user decision
        """
        self._default_review_callback = callback

    async def write_file_with_default_review(
        self,
        path: Union[str, Path],
        content: Union[str, bytes],
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """
        Write file using the default review callback if set.

        Args:
            path: Path to the file to be written
            content: New content to write
            encoding: File encoding for text files

        Returns:
            Dict with operation result and review metadata
        """
        callback = getattr(self, "_default_review_callback", None)
        if not callback:
            # Fall back to regular write if no callback is set
            logger.warning(
                "No default review callback set, falling back to regular write"
            )
            return await self.file_system.write_file(
                path=path, content=content, encoding=encoding
            )

        return await self.write_file_with_review(
            path=path,
            content=content,
            encoding=encoding,
            user_review_callback=callback,
        )

    def _setup_autonomous_file_workflow(self) -> None:
        """Setup autonomous file modification workflow with simple approval."""

        async def autonomous_file_review_callback(
            review_data: Dict[str, Any],
        ) -> Dict[str, Any]:
            file_path = review_data.get("file_path", "unknown")
            operation_type = review_data.get("operation", "modify")
            logger.debug(f"Approval requested for {file_path} ({operation_type})")
            return {"approved": True, "reason": "Auto-approved in agent mode"}

        self.set_read_before_write_callback(autonomous_file_review_callback)

        self.file_system._original_write_file = (  # type: ignore[attr-defined]
            self.file_system.write_file
        )

        async def autonomous_write_file(
            path: Union[str, Path],
            content: Union[str, bytes],
            encoding: str = "utf-8",
            **kwargs: Any,
        ) -> Any:
            autonomous_mode = kwargs.pop("autonomous_mode", True)
            if autonomous_mode:
                kwargs["read_before_write"] = True
                kwargs["user_review_callback"] = autonomous_file_review_callback
            _write = self.file_system._original_write_file  # type: ignore[attr-defined]
            return await _write(
                path=path,
                content=content,
                encoding=encoding,
                **kwargs,
            )

        self.file_system.write_file = autonomous_write_file  # type: ignore[assignment]
        logger.info("Autonomous file modification workflow initialized")
