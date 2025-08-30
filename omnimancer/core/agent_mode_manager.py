"""
Agent Mode Manager for Omnimancer.

This module provides the core agent mode functionality including state management,
operation queuing, and integration with the approval system.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agent.approval_interface import ApprovalInterface
from .agent.approval_manager import EnhancedApprovalManager
from .agent_engine import Operation, OperationResult, OperationType

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Agent mode states."""

    OFF = "off"
    ON = "on"
    PAUSED = "paused"


class AgentOperationStatus(Enum):
    """Status of agent operations."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class AgentOperation:
    """Represents an operation in the agent queue."""

    id: str
    operation: Operation
    status: AgentOperationStatus = AgentOperationStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[OperationResult] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 0  # Higher number = higher priority


@dataclass
class AgentModeSettings:
    """Settings for agent mode."""

    auto_approve_low_risk: bool = False
    auto_approve_read_only: bool = True
    max_concurrent_operations: int = 3
    operation_timeout: int = 300  # seconds
    retry_delay: int = 5  # seconds
    enable_batch_approval: bool = True
    persist_state: bool = True


class AgentModeManager:
    """
    Manages agent mode state and operations.

    Provides functionality for:
    - Agent mode state management (on/off/paused)
    - Operation queuing and execution
    - Integration with approval system
    - Progress tracking and reporting
    - Persistent state management
    """

    def __init__(self, config_manager, storage_path: str = "~/.omnimancer"):
        """
        Initialize the agent mode manager.

        Args:
            config_manager: Configuration manager instance
            storage_path: Path for persistent storage
        """
        self.config_manager = config_manager
        self.storage_path = Path(storage_path).expanduser()
        self.state_file = self.storage_path / "agent_state.json"

        # Initialize state
        self.mode = AgentMode.ON
        self.settings = AgentModeSettings()
        self.operation_queue: List[AgentOperation] = []
        self.active_operations: Dict[str, AgentOperation] = {}
        self.completed_operations: List[AgentOperation] = []

        # Initialize approval system
        self.approval_manager = EnhancedApprovalManager()
        self.approval_interface = ApprovalInterface(self.approval_manager)

        # Event handlers
        self.operation_callbacks: List[Callable[[AgentOperation], None]] = []
        self.mode_change_callbacks: List[Callable[[AgentMode, AgentMode], None]] = []

        # Execution control
        self._execution_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Load persistent state
        self._load_state()

        # Start execution loop if agent mode is enabled
        if self.mode == AgentMode.ON:
            # The execution loop will be started when the CLI initializes
            logger.info("Agent mode will be started automatically")

    async def enable_agent_mode(self, auto_approve: bool = False) -> bool:
        """
        Enable agent mode.

        Args:
            auto_approve: Whether to enable auto-approval for low-risk operations

        Returns:
            True if enabled successfully, False otherwise
        """
        try:
            old_mode = self.mode
            self.mode = AgentMode.ON

            if auto_approve:
                self.settings.auto_approve_low_risk = True
                self.settings.auto_approve_read_only = True

            # Start execution loop
            if not self._execution_task or self._execution_task.done():
                self._shutdown_event.clear()
                self._execution_task = asyncio.create_task(self._execution_loop())

            # Save state
            self._save_state()

            # Notify callbacks
            self._notify_mode_change(old_mode, self.mode)

            logger.info("Agent mode enabled")
            return True

        except Exception as e:
            logger.error(f"Failed to enable agent mode: {e}")
            self.mode = AgentMode.OFF
            return False

    async def disable_agent_mode(self, wait_for_completion: bool = True) -> bool:
        """
        Disable agent mode.

        Args:
            wait_for_completion: Whether to wait for active operations to complete

        Returns:
            True if disabled successfully, False otherwise
        """
        try:
            old_mode = self.mode
            self.mode = AgentMode.OFF

            # Signal shutdown
            self._shutdown_event.set()

            if wait_for_completion and self.active_operations:
                logger.info(
                    f"Waiting for {len(self.active_operations)} active operations to complete..."
                )
                # Wait with timeout
                try:
                    await asyncio.wait_for(self._wait_for_operations(), timeout=30)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for operations to complete, forcing shutdown"
                    )
                    self._cancel_active_operations()
            else:
                self._cancel_active_operations()

            # Stop execution task
            if self._execution_task and not self._execution_task.done():
                self._execution_task.cancel()
                try:
                    await self._execution_task
                except asyncio.CancelledError:
                    pass

            # Save state
            self._save_state()

            # Notify callbacks
            self._notify_mode_change(old_mode, self.mode)

            logger.info("Agent mode disabled")
            return True

        except Exception as e:
            logger.error(f"Failed to disable agent mode: {e}")
            return False

    async def _start_execution_loop_when_ready(self):
        """Start execution loop when event loop is ready."""
        try:
            # Wait a bit for initialization to complete
            await asyncio.sleep(0.1)

            if self.mode == AgentMode.ON and (
                not self._execution_task or self._execution_task.done()
            ):
                self._shutdown_event.clear()
                self._execution_task = asyncio.create_task(self._execution_loop())
                logger.info("Agent execution loop started automatically")
        except Exception as e:
            logger.error(f"Failed to start agent execution loop: {e}")

    def pause_agent_mode(self) -> bool:
        """
        Pause agent mode (stop processing new operations).

        Returns:
            True if paused successfully, False otherwise
        """
        if self.mode != AgentMode.ON:
            return False

        old_mode = self.mode
        self.mode = AgentMode.PAUSED
        self._save_state()
        self._notify_mode_change(old_mode, self.mode)

        logger.info("Agent mode paused")
        return True

    def resume_agent_mode(self) -> bool:
        """
        Resume agent mode from paused state.

        Returns:
            True if resumed successfully, False otherwise
        """
        if self.mode != AgentMode.PAUSED:
            return False

        old_mode = self.mode
        self.mode = AgentMode.ON
        self._save_state()
        self._notify_mode_change(old_mode, self.mode)

        logger.info("Agent mode resumed")
        return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent mode status.

        Returns:
            Dictionary containing status information
        """
        return {
            "mode": self.mode.value,
            "settings": {
                "auto_approve_low_risk": self.settings.auto_approve_low_risk,
                "auto_approve_read_only": self.settings.auto_approve_read_only,
                "max_concurrent_operations": self.settings.max_concurrent_operations,
                "operation_timeout": self.settings.operation_timeout,
                "enable_batch_approval": self.settings.enable_batch_approval,
            },
            "operations": {
                "queued": len(
                    [
                        op
                        for op in self.operation_queue
                        if op.status == AgentOperationStatus.QUEUED
                    ]
                ),
                "in_progress": len(self.active_operations),
                "completed": len(
                    [
                        op
                        for op in self.completed_operations
                        if op.status == AgentOperationStatus.COMPLETED
                    ]
                ),
                "failed": len(
                    [
                        op
                        for op in self.completed_operations
                        if op.status == AgentOperationStatus.FAILED
                    ]
                ),
                "requires_approval": len(
                    [
                        op
                        for op in self.operation_queue
                        if op.status == AgentOperationStatus.REQUIRES_APPROVAL
                    ]
                ),
            },
            "execution_task_active": self._execution_task is not None
            and not self._execution_task.done(),
        }

    def queue_operation(self, operation: Operation, priority: int = 0) -> str:
        """
        Queue an operation for execution.

        Args:
            operation: Operation to queue
            priority: Priority level (higher = more urgent)

        Returns:
            Operation ID
        """
        op_id = f"agent_op_{int(time.time() * 1000)}_{len(self.operation_queue)}"
        agent_operation = AgentOperation(
            id=op_id, operation=operation, priority=priority
        )

        # Insert based on priority
        inserted = False
        for i, existing_op in enumerate(self.operation_queue):
            if priority > existing_op.priority:
                self.operation_queue.insert(i, agent_operation)
                inserted = True
                break

        if not inserted:
            self.operation_queue.append(agent_operation)

        logger.info(f"Queued operation {op_id}: {operation.description}")
        self._save_state()

        return op_id

    def cancel_operation(self, operation_id: str) -> bool:
        """
        Cancel a queued or active operation.

        Args:
            operation_id: ID of operation to cancel

        Returns:
            True if cancelled successfully, False otherwise
        """
        # Check queued operations
        for i, op in enumerate(self.operation_queue):
            if op.id == operation_id:
                op.status = AgentOperationStatus.CANCELLED
                self.operation_queue.pop(i)
                self.completed_operations.append(op)
                logger.info(f"Cancelled queued operation {operation_id}")
                self._save_state()
                return True

        # Check active operations
        if operation_id in self.active_operations:
            op = self.active_operations[operation_id]
            op.status = AgentOperationStatus.CANCELLED
            # Note: Actual cancellation of running operation depends on implementation
            logger.info(f"Marked active operation {operation_id} for cancellation")
            return True

        return False

    def get_operation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get operation history.

        Args:
            limit: Maximum number of operations to return

        Returns:
            List of operation dictionaries
        """
        all_operations = (
            self.operation_queue
            + list(self.active_operations.values())
            + self.completed_operations
        )

        # Sort by creation time, most recent first
        all_operations.sort(key=lambda x: x.created_at, reverse=True)

        history = []
        for op in all_operations[:limit]:
            duration = None
            if op.started_at and op.completed_at:
                duration = (op.completed_at - op.started_at).total_seconds()

            history.append(
                {
                    "id": op.id,
                    "type": (
                        op.operation.type.value if op.operation.type else "unknown"
                    ),
                    "description": op.operation.description,
                    "status": op.status.value,
                    "created_at": op.created_at.isoformat(),
                    "duration": duration,
                    "retry_count": op.retry_count,
                    "error": op.error,
                }
            )

        return history

    def add_operation_callback(self, callback: Callable[[AgentOperation], None]):
        """Add callback for operation state changes."""
        self.operation_callbacks.append(callback)

    def add_mode_change_callback(
        self, callback: Callable[[AgentMode, AgentMode], None]
    ):
        """Add callback for mode changes."""
        self.mode_change_callbacks.append(callback)

    def update_settings(self, **kwargs):
        """Update agent mode settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self._save_state()

    async def _execution_loop(self):
        """Main execution loop for processing operations."""
        logger.info("Agent execution loop started")

        try:
            while not self._shutdown_event.is_set():
                if self.mode == AgentMode.ON:
                    await self._process_operations()

                # Wait briefly before next iteration
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=1.0)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue loop

        except asyncio.CancelledError:
            logger.info("Agent execution loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in agent execution loop: {e}")
        finally:
            logger.info("Agent execution loop stopped")

    async def _process_operations(self):
        """Process queued operations."""
        # Remove completed active operations
        completed_ids = []
        for op_id, op in self.active_operations.items():
            if op.status in [
                AgentOperationStatus.COMPLETED,
                AgentOperationStatus.FAILED,
                AgentOperationStatus.CANCELLED,
            ]:
                completed_ids.append(op_id)
                self.completed_operations.append(op)

        for op_id in completed_ids:
            del self.active_operations[op_id]

        # Start new operations if capacity allows
        available_slots = self.settings.max_concurrent_operations - len(
            self.active_operations
        )

        if available_slots > 0:
            # Get operations ready for execution
            ready_operations = [
                op
                for op in self.operation_queue
                if op.status == AgentOperationStatus.QUEUED
            ]

            for op in ready_operations[:available_slots]:
                try:
                    await self._start_operation(op)
                except Exception as e:
                    logger.error(f"Failed to start operation {op.id}: {e}")
                    op.status = AgentOperationStatus.FAILED
                    op.error = str(e)

    async def _start_operation(self, agent_operation: AgentOperation):
        """Start executing an operation."""
        operation = agent_operation.operation

        # Check if approval is required
        if operation.requires_approval:
            # Handle approval
            approval_granted = await self._handle_approval(agent_operation)
            if not approval_granted:
                agent_operation.status = AgentOperationStatus.CANCELLED
                agent_operation.error = "Approval denied"
                return

        # Move to active operations
        agent_operation.status = AgentOperationStatus.IN_PROGRESS
        agent_operation.started_at = datetime.now()
        self.operation_queue.remove(agent_operation)
        self.active_operations[agent_operation.id] = agent_operation

        # Notify callbacks
        self._notify_operation_change(agent_operation)

        logger.info(f"Starting operation {agent_operation.id}: {operation.description}")

        # Execute operation asynchronously
        asyncio.create_task(self._execute_operation(agent_operation))

    async def _execute_operation(self, agent_operation: AgentOperation):
        """Execute an operation."""
        try:
            # This would be replaced with actual operation execution
            # For now, simulate execution
            await asyncio.sleep(1)

            # Simulate success
            agent_operation.status = AgentOperationStatus.COMPLETED
            agent_operation.completed_at = datetime.now()
            agent_operation.result = OperationResult(
                success=True,
                data={"message": "Operation completed successfully"},
                metadata={},
            )

            logger.info(f"Completed operation {agent_operation.id}")

        except Exception as e:
            agent_operation.status = AgentOperationStatus.FAILED
            agent_operation.completed_at = datetime.now()
            agent_operation.error = str(e)

            logger.error(f"Operation {agent_operation.id} failed: {e}")

            # Handle retry logic
            if agent_operation.retry_count < agent_operation.max_retries:
                agent_operation.retry_count += 1
                agent_operation.status = AgentOperationStatus.QUEUED
                agent_operation.started_at = None
                agent_operation.completed_at = None

                # Re-queue for retry
                self.operation_queue.append(agent_operation)
                del self.active_operations[agent_operation.id]

                logger.info(
                    f"Queued operation {agent_operation.id} for retry ({agent_operation.retry_count}/{agent_operation.max_retries})"
                )

        finally:
            # Notify callbacks
            self._notify_operation_change(agent_operation)
            self._save_state()

    async def _handle_approval(self, agent_operation: AgentOperation) -> bool:
        """Handle approval for an operation."""
        operation = agent_operation.operation

        # Check auto-approval rules
        if self._check_auto_approval(operation):
            logger.info(f"Auto-approved operation {agent_operation.id}")
            return True

        # Mark as requiring approval
        agent_operation.status = AgentOperationStatus.REQUIRES_APPROVAL

        # Use approval interface
        approval_data = {
            "operation": operation,
            "preview": None,  # Would generate preview based on operation
            "approval_request": None,  # Would create formal approval request
            "risk_level": operation.data.get("risk_level", "medium"),
        }

        try:
            approved = await self.approval_interface.handle_single_approval(
                approval_data
            )
            return approved
        except Exception as e:
            logger.error(
                f"Approval handling failed for operation {agent_operation.id}: {e}"
            )
            return False

    def _check_auto_approval(self, operation: Operation) -> bool:
        """Check if operation qualifies for auto-approval."""
        if not operation.requires_approval:
            return True

        # Auto-approve read-only operations
        if (
            self.settings.auto_approve_read_only
            and operation.type == OperationType.FILE_READ
        ):
            return True

        # Auto-approve low-risk operations
        if self.settings.auto_approve_low_risk:
            risk_level = operation.data.get("risk_level", "medium")
            if risk_level.lower() in ["low", "minimal"]:
                return True

        return False

    async def _wait_for_operations(self):
        """Wait for all active operations to complete."""
        while self.active_operations:
            await asyncio.sleep(0.5)

    def _cancel_active_operations(self):
        """Cancel all active operations."""
        for op in self.active_operations.values():
            op.status = AgentOperationStatus.CANCELLED
            op.completed_at = datetime.now()
            self.completed_operations.append(op)

        self.active_operations.clear()

    def _notify_operation_change(self, operation: AgentOperation):
        """Notify callbacks of operation state change."""
        for callback in self.operation_callbacks:
            try:
                callback(operation)
            except Exception as e:
                logger.error(f"Error in operation callback: {e}")

    def _notify_mode_change(self, old_mode: AgentMode, new_mode: AgentMode):
        """Notify callbacks of mode change."""
        for callback in self.mode_change_callbacks:
            try:
                callback(old_mode, new_mode)
            except Exception as e:
                logger.error(f"Error in mode change callback: {e}")

    def _save_state(self):
        """Save persistent state to disk."""
        if not self.settings.persist_state:
            return

        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)

            state_data = {
                "mode": self.mode.value,
                "settings": {
                    "auto_approve_low_risk": self.settings.auto_approve_low_risk,
                    "auto_approve_read_only": self.settings.auto_approve_read_only,
                    "max_concurrent_operations": self.settings.max_concurrent_operations,
                    "operation_timeout": self.settings.operation_timeout,
                    "retry_delay": self.settings.retry_delay,
                    "enable_batch_approval": self.settings.enable_batch_approval,
                    "persist_state": self.settings.persist_state,
                },
                "operation_count": len(self.operation_queue)
                + len(self.active_operations)
                + len(self.completed_operations),
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")

    def _load_state(self):
        """Load persistent state from disk."""
        try:
            if not self.state_file.exists():
                return

            with open(self.state_file, "r") as f:
                state_data = json.load(f)

            # Restore mode - always enable agent mode by default
            self.mode = AgentMode.ON
            logger.info("Agent mode enabled by default")

            # Restore settings
            if "settings" in state_data:
                settings_data = state_data["settings"]
                for key, value in settings_data.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)

            logger.info("Loaded agent state from disk")

        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
