"""
Status-Integrated Agent Engine for Omnimancer.

This module provides an enhanced AgentEngine that integrates with the status
tracking and display system, providing real-time updates on agent operations
and state changes.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..agent_engine import AgentEngine
from ..config_manager import ConfigManager
from .status_core import (
    AgentOperation,
    AgentStatus,
)
from .status_core import OperationType as StatusOperationType
from .status_manager import UnifiedStatusManager as AgentStatusManager
from .status_manager import (
    get_status_manager,
)
from .types import Operation, OperationResult, OperationType

logger = logging.getLogger(__name__)


class StatusIntegratedAgentEngine(AgentEngine):
    """
    Agent engine enhanced with comprehensive status tracking and display.

    This class extends the base AgentEngine to provide real-time status updates,
    operation tracking, and integration with the status display system.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        base_path: Optional[Path] = None,
        status_manager: Optional[AgentStatusManager] = None,
        enable_status_display: bool = True,
    ):
        """
        Initialize the status-integrated agent engine.

        Args:
            config_manager: Configuration manager instance
            base_path: Base path for file system operations
            status_manager: Optional status manager (uses global if not provided)
            enable_status_display: Whether to enable status display
        """
        super().__init__(config_manager, base_path)

        # Status system integration
        self.status_manager = status_manager or get_status_manager()
        self.enable_status_display = enable_status_display

        # Operation mapping for status tracking
        self.operation_mapping = {
            OperationType.FILE_READ: StatusOperationType.FILE_READ,
            OperationType.FILE_WRITE: StatusOperationType.FILE_WRITE,
            OperationType.FILE_DELETE: StatusOperationType.FILE_DELETE,
            OperationType.DIRECTORY_CREATE: StatusOperationType.DIRECTORY_CREATE,
            OperationType.DIRECTORY_DELETE: StatusOperationType.DIRECTORY_DELETE,
            OperationType.COMMAND_EXECUTION: StatusOperationType.COMMAND_EXECUTION,
            OperationType.WEB_REQUEST: StatusOperationType.WEB_REQUEST,
            OperationType.API_CALL: StatusOperationType.API_CALL,
            OperationType.APPROVAL_REQUEST: StatusOperationType.APPROVAL_REQUEST,
        }

        # Active operation tracking
        self.active_status_operations: Dict[str, str] = (
            {}
        )  # agent_operation_id -> status_operation_id

        # Agent identification
        self.agent_id = "main_agent"  # Could be made configurable

        logger.info(
            "StatusIntegratedAgentEngine initialized with status tracking enabled"
        )

    async def initialize_status_system(self) -> None:
        """Initialize the integrated status system."""
        try:
            # Initialize status manager
            await self.status_manager.initialize()

            # Set initial agent status
            await self.status_manager.set_agent_status(
                self.agent_id,
                AgentStatus.ENABLED,
                {
                    "initialization_time": datetime.now(),
                    "engine_type": "StatusIntegratedAgentEngine",
                    "base_path": (
                        str(self.file_system.base_path)
                        if hasattr(self, "file_system")
                        else None
                    ),
                },
            )

            logger.info(f"Status system initialized for agent {self.agent_id}")

        except Exception as e:
            logger.error(f"Failed to initialize status system: {e}")
            # Don't fail the entire engine if status system fails
            self.enable_status_display = False

    async def shutdown_status_system(self) -> None:
        """Shutdown the integrated status system."""
        try:
            # Set agent status to shutting down
            await self.status_manager.set_agent_status(
                self.agent_id,
                AgentStatus.SHUTTING_DOWN,
                {"shutdown_time": datetime.now()},
            )

            # Cancel any active operations
            for agent_op_id, status_op_id in list(
                self.active_status_operations.items()
            ):
                await self.status_manager.cancel_operation(
                    status_op_id, "Engine shutdown"
                )

            # Shutdown status manager
            await self.status_manager.shutdown()

            logger.info("Status system shutdown complete")

        except Exception as e:
            logger.error(f"Error during status system shutdown: {e}")

    async def execute_with_approval(self, operation: Operation) -> OperationResult:
        """
        Execute operation with approval workflow and status tracking.

        Args:
            operation: Operation to execute

        Returns:
            Result of the operation
        """
        status_operation = None

        try:
            # Create status operation for tracking
            if self.enable_status_display:
                status_operation = await self._create_status_operation(operation)

            # Set agent to running state
            if self.enable_status_display:
                await self.status_manager.set_agent_status(
                    self.agent_id,
                    AgentStatus.RUNNING,
                    {
                        "current_operation": (
                            operation.type.value
                            if hasattr(operation.type, "value")
                            else str(operation.type)
                        )
                    },
                )

            # Execute the operation using parent class method
            result = await super().execute_with_approval(operation)

            # Update status based on result
            if self.enable_status_display and status_operation:
                if result.success:
                    await self.status_manager.complete_operation(
                        status_operation.operation_id,
                        {
                            "result_data": result.data,
                            "execution_details": result.details,
                        },
                    )
                else:
                    await self.status_manager.fail_operation(
                        status_operation.operation_id,
                        result.error or "Operation failed without specific error",
                        {
                            "error_details": result.details,
                            "failure_context": result.error,
                        },
                    )

            # Set agent back to enabled state
            if self.enable_status_display:
                await self.status_manager.set_agent_status(
                    self.agent_id,
                    AgentStatus.ENABLED,
                    {
                        "last_operation_result": (
                            "success" if result.success else "failed"
                        )
                    },
                )

            return result

        except Exception as e:
            # Handle operation failure in status system
            if self.enable_status_display and status_operation:
                await self.status_manager.fail_operation(
                    status_operation.operation_id,
                    f"Exception during execution: {e}",
                    {
                        "exception_type": type(e).__name__,
                        "exception_details": str(e),
                    },
                )

            # Set agent to error state
            if self.enable_status_display:
                await self.status_manager.set_agent_status(
                    self.agent_id,
                    AgentStatus.ERROR,
                    {"error_message": str(e), "error_time": datetime.now()},
                )

            # Re-raise the exception
            raise
        finally:
            # Clean up operation tracking
            if status_operation and operation.id in self.active_status_operations:
                del self.active_status_operations[operation.id]

    async def _create_status_operation(self, operation: Operation) -> AgentOperation:
        """
        Create a status operation for tracking.

        Args:
            operation: The agent operation to track

        Returns:
            Created status operation
        """
        # Map operation type
        status_op_type = self.operation_mapping.get(
            operation.type, StatusOperationType.API_CALL
        )

        # Create operation description
        description = self._generate_operation_description(operation)

        # Create status operation
        status_operation = AgentOperation(
            operation_id=str(uuid.uuid4()),
            operation_type=status_op_type,
            description=description,
            agent_id=self.agent_id,
            metadata={
                "original_operation_id": operation.id,
                "original_operation_type": (
                    operation.type.value
                    if hasattr(operation.type, "value")
                    else str(operation.type)
                ),
                "operation_data": operation.data,
                "requires_approval": operation.requires_approval,
            },
        )

        # Start tracking
        await self.status_manager.start_operation(status_operation)

        # Keep reference for updates
        self.active_status_operations[operation.id] = status_operation.operation_id

        return status_operation

    def _generate_operation_description(self, operation: Operation) -> str:
        """
        Generate a user-friendly description for an operation.

        Args:
            operation: The operation to describe

        Returns:
            Human-readable description
        """
        op_type = operation.type
        data = operation.data

        try:
            if hasattr(op_type, "value"):
                type_str = op_type.value
            else:
                type_str = str(op_type)

            if type_str == "FILE_READ" and "path" in data:
                return f"Reading file: {Path(data['path']).name}"
            elif type_str == "FILE_WRITE" and "path" in data:
                return f"Writing to file: {Path(data['path']).name}"
            elif type_str == "FILE_DELETE" and "path" in data:
                return f"Deleting file: {Path(data['path']).name}"
            elif type_str == "DIRECTORY_CREATE" and "path" in data:
                return f"Creating directory: {Path(data['path']).name}"
            elif type_str == "DIRECTORY_DELETE" and "path" in data:
                return f"Deleting directory: {Path(data['path']).name}"
            elif type_str == "COMMAND_EXECUTION" and "command" in data:
                cmd = data["command"]
                return f"Executing: {cmd[:30]}{'...' if len(cmd) > 30 else ''}"
            elif type_str == "WEB_REQUEST" and "url" in data:
                return f"Web request to: {data['url']}"
            elif type_str == "API_CALL":
                return "Making API call"
            elif type_str == "APPROVAL_REQUEST":
                return "Requesting user approval"
            else:
                return f"Executing {type_str.lower().replace('_', ' ')}"

        except Exception as e:
            logger.warning(f"Failed to generate operation description: {e}")
            return f"Executing {type_str.lower().replace('_', ' ') if type_str else 'operation'}"

    async def update_operation_progress(
        self,
        operation_id: str,
        progress: float,
        description: Optional[str] = None,
    ) -> None:
        """
        Update progress of an operation.

        Args:
            operation_id: ID of the agent operation
            progress: Progress percentage (0-100)
            description: Optional progress description
        """
        if (
            not self.enable_status_display
            or operation_id not in self.active_status_operations
        ):
            return

        status_operation_id = self.active_status_operations[operation_id]

        try:
            await self.status_manager.update_operation_progress(
                status_operation_id, progress, description
            )
        except Exception as e:
            logger.error(f"Failed to update operation progress: {e}")

    async def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current agent status.

        Returns:
            Dictionary with status information
        """
        if not self.enable_status_display:
            return {"status_display": "disabled"}

        try:
            agent_status = self.status_manager.get_agent_status(self.agent_id)
            agent_metadata = self.status_manager.get_agent_metadata(self.agent_id)
            active_operations = self.status_manager.get_active_operations(self.agent_id)
            system_stats = self.status_manager.get_stats()

            return {
                "agent_id": self.agent_id,
                "status": agent_status.value if agent_status else "unknown",
                "metadata": agent_metadata,
                "active_operations_count": len(active_operations),
                "active_operations": [
                    {
                        "id": op.operation_id,
                        "type": op.operation_type.value,
                        "description": op.description,
                        "progress": op.progress_percentage,
                        "duration": (op.duration.total_seconds() if op.duration else 0),
                    }
                    for op in active_operations
                ],
                "system_stats": system_stats,
                "status_stream_running": self.status_manager.running,
            }

        except Exception as e:
            logger.error(f"Failed to get status summary: {e}")
            return {"error": f"Failed to get status: {e}"}

    def is_status_display_enabled(self) -> bool:
        """Check if status display is enabled."""
        return self.enable_status_display

    def enable_status_display_mode(self) -> None:
        """Enable status display mode."""
        self.enable_status_display = True

    def disable_status_display_mode(self) -> None:
        """Disable status display mode."""
        self.enable_status_display = False

    async def start_status_monitoring(self) -> bool:
        """
        Start status monitoring and display.

        Returns:
            True if monitoring started successfully
        """
        try:
            if not self.enable_status_display:
                return False

            # Initialize if not already done
            if not self.status_manager._event_processor_task:
                await self.status_manager.initialize()

            # Status manager handles stream initialization

            logger.info("Status monitoring started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start status monitoring: {e}")
            return False

    async def stop_status_monitoring(self) -> bool:
        """
        Stop status monitoring and display.

        Returns:
            True if monitoring stopped successfully
        """
        try:
            # The unified status manager handles stream shutdown

            # Update agent status
            await self.status_manager.set_agent_status(
                self.agent_id,
                AgentStatus.DISABLED,
                {"monitoring_stopped": datetime.now()},
            )

            logger.info("Status monitoring stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop status monitoring: {e}")
            return False


# Factory function to create a fully integrated engine
async def create_status_integrated_engine(
    config_manager: ConfigManager,
    base_path: Optional[Path] = None,
    auto_initialize: bool = True,
) -> StatusIntegratedAgentEngine:
    """
    Create a status-integrated agent engine with automatic setup.

    Args:
        config_manager: Configuration manager instance
        base_path: Base path for file system operations
        auto_initialize: Whether to automatically initialize the status system

    Returns:
        Fully configured status-integrated engine
    """
    # Create engine
    engine = StatusIntegratedAgentEngine(config_manager, base_path)

    # Auto-initialize if requested
    if auto_initialize:
        try:
            await engine.initialize_status_system()
            await engine.start_status_monitoring()
            logger.info("Status-integrated engine created and initialized")
        except Exception as e:
            logger.error(f"Failed to auto-initialize status system: {e}")
            # Continue with engine creation but status display disabled
            engine.disable_status_display_mode()

    return engine


# Context manager for temporary status tracking
class StatusTrackingContext:
    """Context manager for temporary status tracking within operations."""

    def __init__(self, engine: StatusIntegratedAgentEngine, operation_name: str):
        self.engine = engine
        self.operation_name = operation_name
        self.operation_id = None

    async def __aenter__(self):
        if self.engine.enable_status_display:
            # Create a temporary operation
            from .types import Operation as TempOperation
            from .types import OperationType as TempOperationType

            temp_operation = TempOperation(
                id=str(uuid.uuid4()),
                type=TempOperationType.API_CALL,  # Generic type
                data={"description": self.operation_name},
                requires_approval=False,
            )

            await self.engine._create_status_operation(temp_operation)
            self.operation_id = temp_operation.id

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.engine.enable_status_display and self.operation_id:
            status_operation_id = self.engine.active_status_operations.get(
                self.operation_id
            )
            if status_operation_id:
                if exc_type:
                    await self.engine.status_manager.fail_operation(
                        status_operation_id,
                        f"Context operation failed: {exc_val}",
                        {"exception_type": exc_type.__name__},
                    )
                else:
                    await self.engine.status_manager.complete_operation(
                        status_operation_id, {"context_result": "success"}
                    )

            # Clean up
            if self.operation_id in self.engine.active_status_operations:
                del self.engine.active_status_operations[self.operation_id]

    async def update_progress(self, progress: float, description: Optional[str] = None):
        """Update progress of the tracked operation."""
        if self.operation_id:
            await self.engine.update_operation_progress(
                self.operation_id, progress, description
            )
