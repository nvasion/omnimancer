"""
Complete File Modification Workflow with User Approval.

This module provides the end-to-end workflow for file modifications,
integrating all UI components and approval mechanisms into a cohesive system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress

from ...cli.approval_formatter import CLIApprovalFormatter
from ...cli.approval_prompt import (
    ApprovalDecision,
    ApprovalDecisionType,
    CLIApprovalPrompt,
)
from ..security.approval_workflow import (
    RiskLevel,
)
from .approval_manager import EnhancedApprovalManager
from .file_content_display import (
    FileDisplayConfig,
    UnifiedFileContentDisplay,
)
from .file_system_manager import FileSystemManager
from .proposed_changes_integration import (
    ChangeDisplayMode,
    ChangeSet,
    ProposedChangesIntegration,
)

logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """States of the file modification workflow."""

    INITIALIZED = "initialized"
    DISPLAYING_CHANGES = "displaying_changes"
    AWAITING_APPROVAL = "awaiting_approval"
    PROCESSING_APPROVAL = "processing_approval"
    APPLYING_CHANGES = "applying_changes"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class WorkflowResult(Enum):
    """Possible results of the workflow."""

    APPROVED_AND_APPLIED = "approved_and_applied"
    APPROVED_NOT_APPLIED = "approved_not_applied"
    DENIED = "denied"
    CANCELLED = "cancelled"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class WorkflowConfig:
    """Configuration for the file modification workflow."""

    # Display settings
    default_display_mode: ChangeDisplayMode = ChangeDisplayMode.UNIFIED
    show_file_stats: bool = True
    syntax_highlighting: bool = True

    # Approval settings
    require_confirmation: bool = True
    auto_apply_approved: bool = True
    remember_decisions: bool = True
    approval_timeout_seconds: int = 300  # 5 minutes

    # Risk thresholds
    auto_approve_risk_threshold: RiskLevel = RiskLevel.LOW
    require_explicit_approval: RiskLevel = RiskLevel.HIGH

    # Batch settings
    batch_approval_threshold: int = 5  # Show batch interface for 5+ changes
    allow_selective_approval: bool = True

    # Performance settings
    max_concurrent_operations: int = 5  # Maximum concurrent file operations

    # Persistence
    save_workflow_history: bool = True
    history_retention_days: int = 30


@dataclass
class WorkflowContext:
    """Context information for the workflow."""

    operation_id: str
    operation_type: str
    initiated_by: str = "user"
    initiated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # State tracking
    current_state: WorkflowState = WorkflowState.INITIALIZED
    state_history: List[Dict[str, Any]] = field(default_factory=list)

    # Results
    final_result: Optional[WorkflowResult] = None
    user_decisions: List[ApprovalDecision] = field(default_factory=list)
    applied_changes: List[str] = field(default_factory=list)
    failed_changes: List[tuple] = field(default_factory=list)


class FileModificationWorkflow:
    """
    Complete file modification workflow with integrated user approval.

    Orchestrates the entire process from displaying changes to applying
    approved modifications, with comprehensive error handling and state management.
    """

    def __init__(
        self,
        file_system_manager: Optional[FileSystemManager] = None,
        approval_manager: Optional[EnhancedApprovalManager] = None,
        console: Optional[Console] = None,
        config: Optional[WorkflowConfig] = None,
    ):
        """
        Initialize the file modification workflow.

        Args:
            file_system_manager: File system manager instance
            approval_manager: Approval manager instance
            console: Rich console for output
            config: Workflow configuration
        """
        self.console = console or Console()
        self.config = config or WorkflowConfig()
        self.file_system_manager = file_system_manager or FileSystemManager()
        self.approval_manager = approval_manager or EnhancedApprovalManager()

        # Initialize UI components
        self.unified_display = UnifiedFileContentDisplay(
            console=self.console,
            config=FileDisplayConfig(
                syntax_highlighting=self.config.syntax_highlighting,
                show_file_stats=self.config.show_file_stats,
            ),
        )

        self.changes_integration = ProposedChangesIntegration(
            file_system_manager=self.file_system_manager,
            approval_manager=self.approval_manager,
            console=self.console,
        )

        self.approval_prompt = CLIApprovalPrompt(
            console=self.console,
            default_timeout_seconds=self.config.approval_timeout_seconds,
        )

        self.approval_formatter = CLIApprovalFormatter(console=self.console)

        # Workflow tracking
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.workflow_history: List[WorkflowContext] = []

        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_approval_decision: Optional[Callable] = None
        self.on_workflow_complete: Optional[Callable] = None

    async def execute_file_modification_workflow(
        self,
        operation_id: str,
        file_paths: Optional[List[str]] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowContext:
        """
        Execute the complete file modification workflow.

        Args:
            operation_id: ID of the operation to process
            file_paths: Optional list of specific files to process
            context_metadata: Additional context information

        Returns:
            WorkflowContext containing the complete workflow state and results
        """
        # Initialize workflow context
        workflow_context = WorkflowContext(
            operation_id=operation_id,
            operation_type="file_modification",
            metadata=context_metadata or {},
        )

        self.active_workflows[operation_id] = workflow_context

        try:
            # Step 1: Fetch proposed changes
            await self._transition_state(
                workflow_context, WorkflowState.DISPLAYING_CHANGES
            )
            change_set = await self.changes_integration.fetch_proposed_changes(
                operation_id, file_paths
            )

            if not change_set.changes:
                workflow_context.final_result = WorkflowResult.CANCELLED
                await self._transition_state(workflow_context, WorkflowState.COMPLETED)
                self.console.print("[yellow]No changes found to process.[/yellow]")
                return workflow_context

            # Step 2: Display changes and get approval
            await self._transition_state(
                workflow_context, WorkflowState.AWAITING_APPROVAL
            )
            approval_result = await self._handle_change_approval(
                workflow_context, change_set
            )

            if not approval_result.get("approved", False):
                workflow_context.final_result = WorkflowResult.DENIED
                await self._transition_state(workflow_context, WorkflowState.COMPLETED)
                return workflow_context

            # Step 3: Apply approved changes
            if self.config.auto_apply_approved:
                await self._transition_state(
                    workflow_context, WorkflowState.APPLYING_CHANGES
                )
                apply_result = await self._apply_approved_changes(
                    workflow_context, change_set, approval_result
                )

                if apply_result["success"]:
                    workflow_context.final_result = WorkflowResult.APPROVED_AND_APPLIED
                else:
                    workflow_context.final_result = WorkflowResult.APPROVED_NOT_APPLIED
            else:
                workflow_context.final_result = WorkflowResult.APPROVED_NOT_APPLIED

            await self._transition_state(workflow_context, WorkflowState.COMPLETED)

        except asyncio.TimeoutError:
            workflow_context.final_result = WorkflowResult.TIMEOUT
            await self._transition_state(workflow_context, WorkflowState.ERROR)
            self.console.print("[red]Workflow timed out waiting for user input.[/red]")

        except Exception as e:
            logger.error(f"Error in workflow {operation_id}: {e}")
            workflow_context.final_result = WorkflowResult.ERROR
            await self._transition_state(workflow_context, WorkflowState.ERROR)
            self.console.print(f"[red]Workflow error: {str(e)}[/red]")

        finally:
            # Clean up and store history
            await self._finalize_workflow(workflow_context)

        return workflow_context

    async def execute_workflow(self, changeset: ChangeSet) -> WorkflowContext:
        """
        Execute workflow for a given ChangeSet.

        Args:
            changeset: The ChangeSet to process

        Returns:
            WorkflowContext containing the complete workflow state and results
        """
        operation_id = changeset.id

        # Initialize workflow context
        workflow_context = WorkflowContext(
            operation_id=operation_id,
            operation_type="file_modification",
            metadata={},
        )

        self.active_workflows[operation_id] = workflow_context

        try:
            # Step 1: Display changes (already have them in changeset)
            await self._transition_state(
                workflow_context, WorkflowState.DISPLAYING_CHANGES
            )

            # Step 2: Handle approval
            await self._transition_state(
                workflow_context, WorkflowState.AWAITING_APPROVAL
            )
            approval_result = await self._handle_change_approval(
                workflow_context, changeset
            )

            if not approval_result.get("approved", False):
                workflow_context.current_state = WorkflowState.CANCELLED
                return workflow_context

            # Step 3: Apply changes
            await self._transition_state(
                workflow_context, WorkflowState.APPLYING_CHANGES
            )
            apply_result = await self._apply_approved_changes(
                workflow_context, changeset, approval_result
            )

            # Step 4: Complete
            if apply_result.get("success", False):
                await self._transition_state(workflow_context, WorkflowState.COMPLETED)
            else:
                await self._transition_state(workflow_context, WorkflowState.ERROR)
                workflow_context.failed_changes = apply_result.get("failed_changes", [])

        except Exception as e:
            workflow_context.current_state = WorkflowState.ERROR
            workflow_context.failed_changes.append((str(e), "Exception"))
        finally:
            # Clean up and store history
            await self._finalize_workflow(workflow_context)

        return workflow_context

    async def execute_single_file_workflow(
        self,
        file_path: str,
        operation_type: str,
        current_content: Optional[str] = None,
        new_content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowContext:
        """
        Execute workflow for a single file modification.

        Args:
            file_path: Path to the file
            operation_type: Type of operation (create, modify, delete)
            current_content: Current file content (for modify operations)
            new_content: New content to write (for create/modify operations)
            metadata: Additional metadata

        Returns:
            WorkflowContext with results
        """
        operation_id = f"single-file-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        workflow_context = WorkflowContext(
            operation_id=operation_id,
            operation_type=f"single_file_{operation_type}",
            metadata=metadata or {},
        )

        try:
            await self._transition_state(
                workflow_context, WorkflowState.DISPLAYING_CHANGES
            )

            # Display based on operation type
            current_model = workflow_context.metadata.get(
                "current_model", "Omnimancer AI"
            )
            if operation_type == "create":
                display_result = await self.unified_display.display_file_creation(
                    file_path,
                    new_content or "",
                    {"interactive": True, "current_model": current_model},
                )
            elif operation_type == "modify":
                display_result = await self.unified_display.display_file_modification(
                    file_path,
                    current_content or "",
                    new_content or "",
                    {"interactive": True, "current_model": current_model},
                )
            elif operation_type == "delete":
                display_result = await self.unified_display.display_file_deletion(
                    file_path,
                    current_content or "",
                    {"interactive": True, "current_model": current_model},
                )
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")

            # Process approval decision
            if display_result.get("approved", False):
                workflow_context.final_result = WorkflowResult.APPROVED_AND_APPLIED

                # Apply the change (bypass autonomous approval since we already got approval)
                if operation_type == "create" and new_content is not None:
                    # Call original write_file directly to bypass autonomous approval wrapper
                    write_method = getattr(
                        self.file_system_manager,
                        "_original_write_file",
                        self.file_system_manager.write_file,
                    )
                    await write_method(
                        path=file_path,
                        content=new_content,
                        read_before_write=False,  # Skip read_before_write since we already got user approval
                    )
                elif operation_type == "modify" and new_content is not None:
                    # Call original write_file directly to bypass autonomous approval wrapper
                    write_method = getattr(
                        self.file_system_manager,
                        "_original_write_file",
                        self.file_system_manager.write_file,
                    )
                    await write_method(
                        path=file_path,
                        content=new_content,
                        read_before_write=False,  # Skip read_before_write since we already got user approval
                    )
                elif operation_type == "delete":
                    await self.file_system_manager.delete_file(file_path)

                workflow_context.applied_changes.append(file_path)
            else:
                workflow_context.final_result = WorkflowResult.DENIED

            await self._transition_state(workflow_context, WorkflowState.COMPLETED)

        except Exception as e:
            logger.error(f"Error in single file workflow: {e}")
            workflow_context.final_result = WorkflowResult.ERROR
            await self._transition_state(workflow_context, WorkflowState.ERROR)

        return workflow_context

    async def get_workflow_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a workflow.

        Args:
            operation_id: ID of the workflow to check

        Returns:
            Status dictionary or None if workflow not found
        """
        workflow = self.active_workflows.get(operation_id)
        if not workflow:
            return None

        return {
            "operation_id": workflow.operation_id,
            "current_state": workflow.current_state.value,
            "initiated_at": workflow.initiated_at.isoformat(),
            "elapsed_seconds": (datetime.now() - workflow.initiated_at).total_seconds(),
            "decisions_count": len(workflow.user_decisions),
            "applied_changes": workflow.applied_changes,
            "failed_changes": workflow.failed_changes,
        }

    async def cancel_workflow(
        self, operation_id: str, reason: str = "User cancelled"
    ) -> bool:
        """
        Cancel an active workflow.

        Args:
            operation_id: ID of the workflow to cancel
            reason: Reason for cancellation

        Returns:
            True if successfully cancelled, False if workflow not found
        """
        workflow = self.active_workflows.get(operation_id)
        if not workflow:
            return False

        workflow.final_result = WorkflowResult.CANCELLED
        workflow.metadata["cancellation_reason"] = reason

        await self._transition_state(workflow, WorkflowState.CANCELLED)
        await self._finalize_workflow(workflow)

        return True

    async def _handle_change_approval(
        self, workflow_context: WorkflowContext, change_set: ChangeSet
    ) -> Dict[str, Any]:
        """Handle the approval process for changes."""
        # Determine if we should use batch interface
        use_batch_interface = (
            len(change_set.changes) >= self.config.batch_approval_threshold
        )

        if use_batch_interface:
            return await self.changes_integration.display_proposed_changes(
                change_set,
                display_mode=self.config.default_display_mode,
                interactive=True,
            )
        else:
            # Handle individual approval for each change
            approved_changes = []

            for i, change in enumerate(change_set.changes):
                self.console.print(
                    f"\n[bold]Change {i+1} of {len(change_set.changes)}[/bold]"
                )

                # Create single-change set for display
                single_change_set = ChangeSet(
                    id=f"{change_set.id}-{i}",
                    description=f"Change for {change.file_path}",
                    changes=[change],
                )

                result = await self.changes_integration.display_proposed_changes(
                    single_change_set,
                    display_mode=self.config.default_display_mode,
                    interactive=True,
                )

                if result.get("approved", False):
                    approved_changes.append(i)

                # Record decision
                decision = ApprovalDecision(
                    decision=(
                        ApprovalDecisionType.APPROVED
                        if result.get("approved")
                        else ApprovalDecisionType.DENIED
                    ),
                    user_notes=result.get("reason", ""),
                )
                workflow_context.user_decisions.append(decision)

            return {
                "approved": len(approved_changes) > 0,
                "selected_changes": approved_changes,
            }

    async def _apply_approved_changes(
        self,
        workflow_context: WorkflowContext,
        change_set: ChangeSet,
        approval_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply the approved changes."""
        selected_changes = approval_result.get("selected_changes")

        # Show progress for multiple changes
        if len(change_set.changes) > 1:
            with Progress(console=self.console) as progress:
                task = progress.add_task(
                    "Applying changes...", total=len(change_set.changes)
                )

                apply_result = await self.changes_integration.apply_proposed_changes(
                    change_set.id, selected_changes
                )

                progress.update(task, completed=len(change_set.changes))
        else:
            apply_result = await self.changes_integration.apply_proposed_changes(
                change_set.id, selected_changes
            )

        # Update workflow context
        workflow_context.applied_changes.extend(apply_result.get("applied", []))
        workflow_context.failed_changes.extend(apply_result.get("failed", []))

        # Display results
        if apply_result["success"]:
            self.console.print(
                f"[green]✓ Successfully applied {len(apply_result.get('applied', []))} changes[/green]"
            )
        else:
            failed_count = len(apply_result.get("failed", []))
            self.console.print(f"[red]✗ {failed_count} changes failed to apply[/red]")

            for file_path, error in apply_result.get("failed", []):
                self.console.print(f"  [red]• {file_path}: {error}[/red]")

        return apply_result

    async def _transition_state(
        self, workflow_context: WorkflowContext, new_state: WorkflowState
    ):
        """Transition workflow to a new state."""
        old_state = workflow_context.current_state
        workflow_context.current_state = new_state

        # Record state change
        workflow_context.state_history.append(
            {
                "from_state": old_state.value,
                "to_state": new_state.value,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Call callback if registered
        if self.on_state_change:
            await self.on_state_change(workflow_context, old_state, new_state)

    async def _finalize_workflow(self, workflow_context: WorkflowContext):
        """Finalize workflow and clean up."""
        # Remove from active workflows
        if workflow_context.operation_id in self.active_workflows:
            del self.active_workflows[workflow_context.operation_id]

        # Add to history if configured
        if self.config.save_workflow_history:
            self.workflow_history.append(workflow_context)

            # Clean old history
            cutoff_date = datetime.now() - timedelta(
                days=self.config.history_retention_days
            )
            self.workflow_history = [
                w for w in self.workflow_history if w.initiated_at > cutoff_date
            ]

        # Call completion callback
        if self.on_workflow_complete:
            await self.on_workflow_complete(workflow_context)

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get statistics about workflow usage."""
        total_workflows = len(self.workflow_history)
        if total_workflows == 0:
            return {"total_workflows": 0}

        # Count results
        result_counts = {}
        for workflow in self.workflow_history:
            result = workflow.final_result.value if workflow.final_result else "unknown"
            result_counts[result] = result_counts.get(result, 0) + 1

        # Calculate success rate
        successful = result_counts.get("approved_and_applied", 0)
        success_rate = (
            (successful / total_workflows) * 100 if total_workflows > 0 else 0
        )

        return {
            "total_workflows": total_workflows,
            "active_workflows": len(self.active_workflows),
            "result_distribution": result_counts,
            "success_rate": success_rate,
            "average_changes_per_workflow": (
                sum(
                    len(w.applied_changes) + len(w.failed_changes)
                    for w in self.workflow_history
                )
                / total_workflows
                if total_workflows > 0
                else 0
            ),
        }
