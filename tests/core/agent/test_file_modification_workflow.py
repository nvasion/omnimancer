"""
Tests for the File Modification Workflow module.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from rich.console import Console

from omnimancer.core.agent.file_modification_workflow import (
    FileModificationWorkflow,
    WorkflowConfig,
    WorkflowContext,
    WorkflowState,
    WorkflowResult,
)
from omnimancer.core.agent.proposed_changes_integration import (
    ChangeSet,
    ProposedChange,
)
from omnimancer.core.agent.approval_manager import ChangeType
from omnimancer.core.security.approval_workflow import RiskLevel


class TestFileModificationWorkflow:
    """Test suite for FileModificationWorkflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console = Mock()
        # Add get_time method that Rich Progress expects
        self.console.get_time.return_value = 0.0
        self.file_system_manager = Mock()
        self.approval_manager = Mock()

        self.config = WorkflowConfig(
            approval_timeout_seconds=30,
            auto_apply_approved=True,
            batch_approval_threshold=3,
        )

        self.workflow = FileModificationWorkflow(
            file_system_manager=self.file_system_manager,
            approval_manager=self.approval_manager,
            console=self.console,
            config=self.config,
        )

    def create_test_change_set(self, num_changes=2):
        """Create a test ChangeSet."""
        changes = []
        for i in range(num_changes):
            change = ProposedChange(
                file_path=f"/test/file{i}.py",
                operation_type=ChangeType.FILE_MODIFY,
                original_content=f"original content {i}",
                modified_content=f"modified content {i}",
                risk_level=RiskLevel.LOW,
            )
            changes.append(change)

        return ChangeSet(
            id="test-changeset", description="Test change set", changes=changes
        )

    @pytest.mark.asyncio
    async def test_execute_file_modification_workflow_success(self):
        """Test successful execution of file modification workflow."""
        # Mock the changes integration
        change_set = self.create_test_change_set(2)

        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch:
            mock_fetch.return_value = change_set

            with patch.object(
                self.workflow, "_handle_change_approval"
            ) as mock_approval:
                mock_approval.return_value = {
                    "approved": True,
                    "all_changes": True,
                }

                # Create a mock that actually updates the workflow context
                async def mock_apply_changes(
                    workflow_context, change_set, approval_result
                ):
                    # Simulate updating the workflow context like the real method does
                    applied_files = ["/test/file0.py", "/test/file1.py"]
                    workflow_context.applied_changes.extend(applied_files)
                    return {
                        "success": True,
                        "applied": applied_files,
                        "failed": [],
                    }

                with patch.object(
                    self.workflow,
                    "_apply_approved_changes",
                    side_effect=mock_apply_changes,
                ):
                    result = (
                        await self.workflow.execute_file_modification_workflow(
                            "test-op"
                        )
                    )

                    assert result.operation_id == "test-op"
                    assert (
                        result.final_result
                        == WorkflowResult.APPROVED_AND_APPLIED
                    )
                    assert result.current_state == WorkflowState.COMPLETED
                    assert len(result.applied_changes) == 2

    @pytest.mark.asyncio
    async def test_execute_file_modification_workflow_denied(self):
        """Test workflow when changes are denied."""
        change_set = self.create_test_change_set(1)

        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch:
            mock_fetch.return_value = change_set

            with patch.object(
                self.workflow, "_handle_change_approval"
            ) as mock_approval:
                mock_approval.return_value = {"approved": False}

                result = (
                    await self.workflow.execute_file_modification_workflow(
                        "test-op-denied"
                    )
                )

                assert result.final_result == WorkflowResult.DENIED
                assert result.current_state == WorkflowState.COMPLETED
                assert len(result.applied_changes) == 0

    @pytest.mark.asyncio
    async def test_execute_file_modification_workflow_no_changes(self):
        """Test workflow when no changes are found."""
        empty_change_set = ChangeSet(
            id="empty", description="Empty change set", changes=[]
        )

        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch:
            mock_fetch.return_value = empty_change_set

            result = await self.workflow.execute_file_modification_workflow(
                "test-op-empty"
            )

            assert result.final_result == WorkflowResult.CANCELLED
            assert result.current_state == WorkflowState.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_single_file_workflow_create(self):
        """Test single file creation workflow."""
        self.file_system_manager.create_file = AsyncMock(
            return_value={"success": True}
        )
        self.file_system_manager._original_write_file = AsyncMock(
            return_value={"success": True}
        )

        with patch.object(
            self.workflow.unified_display,
            "display_file_creation",
            new=AsyncMock(return_value={"approved": True}),
        ) as mock_display:

            result = await self.workflow.execute_single_file_workflow(
                "/test/new_file.py", "create", new_content="def hello(): pass"
            )

            assert result.final_result == WorkflowResult.APPROVED_AND_APPLIED
            assert "/test/new_file.py" in result.applied_changes
            self.file_system_manager._original_write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_single_file_workflow_modify(self):
        """Test single file modification workflow."""
        self.file_system_manager.modify_file = AsyncMock(
            return_value={"success": True}
        )
        self.file_system_manager._original_write_file = AsyncMock(
            return_value={"success": True}
        )

        with patch.object(
            self.workflow.unified_display,
            "display_file_modification",
            new=AsyncMock(return_value={"approved": True}),
        ) as mock_display:

            result = await self.workflow.execute_single_file_workflow(
                "/test/existing_file.py",
                "modify",
                current_content="old content",
                new_content="new content",
            )

            assert result.final_result == WorkflowResult.APPROVED_AND_APPLIED
            assert "/test/existing_file.py" in result.applied_changes
            self.file_system_manager._original_write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_single_file_workflow_delete(self):
        """Test single file deletion workflow."""
        self.file_system_manager.delete_file = AsyncMock(
            return_value={"success": True}
        )

        with patch.object(
            self.workflow.unified_display, "display_file_deletion"
        ) as mock_display:
            mock_display.return_value = {"approved": True}

            result = await self.workflow.execute_single_file_workflow(
                "/test/file_to_delete.py",
                "delete",
                current_content="content to delete",
            )

            assert result.final_result == WorkflowResult.APPROVED_AND_APPLIED
            assert "/test/file_to_delete.py" in result.applied_changes
            self.file_system_manager.delete_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_single_file_workflow_denied(self):
        """Test single file workflow when user denies."""
        with patch.object(
            self.workflow.unified_display, "display_file_creation"
        ) as mock_display:
            mock_display.return_value = {"approved": False}

            result = await self.workflow.execute_single_file_workflow(
                "/test/denied_file.py", "create", new_content="content"
            )

            assert result.final_result == WorkflowResult.DENIED
            assert len(result.applied_changes) == 0

    @pytest.mark.asyncio
    async def test_handle_change_approval_batch_interface(self):
        """Test change approval with batch interface."""
        change_set = self.create_test_change_set(5)  # Exceeds batch threshold
        workflow_context = WorkflowContext(
            operation_id="test", operation_type="test"
        )

        with patch.object(
            self.workflow.changes_integration, "display_proposed_changes"
        ) as mock_display:
            mock_display.return_value = {"approved": True, "all_changes": True}

            result = await self.workflow._handle_change_approval(
                workflow_context, change_set
            )

            assert result["approved"] is True
            mock_display.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_change_approval_individual(self):
        """Test individual change approval."""
        change_set = self.create_test_change_set(2)  # Below batch threshold
        workflow_context = WorkflowContext(
            operation_id="test", operation_type="test"
        )

        with patch.object(
            self.workflow.changes_integration, "display_proposed_changes"
        ) as mock_display:
            # Mock approval for first change, denial for second
            mock_display.side_effect = [
                {"approved": True},
                {"approved": False},
            ]

            result = await self.workflow._handle_change_approval(
                workflow_context, change_set
            )

            assert result["approved"] is True
            assert result["selected_changes"] == [
                0
            ]  # Only first change approved
            assert len(workflow_context.user_decisions) == 2

    @pytest.mark.asyncio
    async def test_apply_approved_changes_success(self):
        """Test successful application of approved changes."""
        change_set = self.create_test_change_set(2)
        workflow_context = WorkflowContext(
            operation_id="test", operation_type="test"
        )
        approval_result = {"approved": True, "selected_changes": [0, 1]}

        with patch.object(
            self.workflow.changes_integration, "apply_proposed_changes"
        ) as mock_apply:
            mock_apply.return_value = {
                "success": True,
                "applied": ["/test/file0.py", "/test/file1.py"],
                "failed": [],
            }

            # Mock Progress to avoid Rich console issues
            with patch(
                "omnimancer.core.agent.file_modification_workflow.Progress"
            ):
                result = await self.workflow._apply_approved_changes(
                    workflow_context, change_set, approval_result
                )

            assert result["success"] is True
            assert len(workflow_context.applied_changes) == 2
            assert len(workflow_context.failed_changes) == 0

    @pytest.mark.asyncio
    async def test_apply_approved_changes_partial_failure(self):
        """Test partial failure when applying changes."""
        change_set = self.create_test_change_set(2)
        workflow_context = WorkflowContext(
            operation_id="test", operation_type="test"
        )
        approval_result = {"approved": True, "selected_changes": [0, 1]}

        with patch.object(
            self.workflow.changes_integration, "apply_proposed_changes"
        ) as mock_apply:
            mock_apply.return_value = {
                "success": False,
                "applied": ["/test/file0.py"],
                "failed": [("/test/file1.py", "Permission denied")],
            }

            # Mock Progress to avoid Rich console issues
            with patch(
                "omnimancer.core.agent.file_modification_workflow.Progress"
            ):
                result = await self.workflow._apply_approved_changes(
                    workflow_context, change_set, approval_result
                )

            assert result["success"] is False
            assert len(workflow_context.applied_changes) == 1
            assert len(workflow_context.failed_changes) == 1

    @pytest.mark.asyncio
    async def test_get_workflow_status(self):
        """Test getting workflow status."""
        workflow_context = WorkflowContext(
            operation_id="status-test", operation_type="test"
        )
        workflow_context.current_state = WorkflowState.AWAITING_APPROVAL
        workflow_context.applied_changes = ["/test/file.py"]

        self.workflow.active_workflows["status-test"] = workflow_context

        status = await self.workflow.get_workflow_status("status-test")

        assert status is not None
        assert status["operation_id"] == "status-test"
        assert status["current_state"] == WorkflowState.AWAITING_APPROVAL.value
        assert status["applied_changes"] == ["/test/file.py"]
        assert "elapsed_seconds" in status

    @pytest.mark.asyncio
    async def test_get_workflow_status_not_found(self):
        """Test getting status for non-existent workflow."""
        status = await self.workflow.get_workflow_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_cancel_workflow(self):
        """Test cancelling an active workflow."""
        workflow_context = WorkflowContext(
            operation_id="cancel-test", operation_type="test"
        )
        self.workflow.active_workflows["cancel-test"] = workflow_context

        with patch.object(
            self.workflow, "_finalize_workflow"
        ) as mock_finalize:
            success = await self.workflow.cancel_workflow(
                "cancel-test", "User requested"
            )

            assert success is True
            assert workflow_context.final_result == WorkflowResult.CANCELLED
            assert (
                workflow_context.metadata["cancellation_reason"]
                == "User requested"
            )
            mock_finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_workflow_not_found(self):
        """Test cancelling non-existent workflow."""
        success = await self.workflow.cancel_workflow("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test workflow state transitions."""
        workflow_context = WorkflowContext(
            operation_id="state-test", operation_type="test"
        )

        initial_state = workflow_context.current_state
        assert initial_state == WorkflowState.INITIALIZED

        await self.workflow._transition_state(
            workflow_context, WorkflowState.DISPLAYING_CHANGES
        )

        assert (
            workflow_context.current_state == WorkflowState.DISPLAYING_CHANGES
        )
        assert len(workflow_context.state_history) == 1
        assert (
            workflow_context.state_history[0]["from_state"]
            == WorkflowState.INITIALIZED.value
        )
        assert (
            workflow_context.state_history[0]["to_state"]
            == WorkflowState.DISPLAYING_CHANGES.value
        )

    @pytest.mark.asyncio
    async def test_state_change_callback(self):
        """Test state change callback functionality."""
        callback_calls = []

        async def state_change_callback(context, old_state, new_state):
            callback_calls.append((context.operation_id, old_state, new_state))

        self.workflow.on_state_change = state_change_callback

        workflow_context = WorkflowContext(
            operation_id="callback-test", operation_type="test"
        )

        await self.workflow._transition_state(
            workflow_context, WorkflowState.AWAITING_APPROVAL
        )

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "callback-test"
        assert callback_calls[0][1] == WorkflowState.INITIALIZED
        assert callback_calls[0][2] == WorkflowState.AWAITING_APPROVAL

    def test_get_workflow_statistics(self):
        """Test workflow statistics calculation."""
        # Add some completed workflows to history
        self.workflow.workflow_history = [
            WorkflowContext(
                operation_id="op1",
                operation_type="test",
                final_result=WorkflowResult.APPROVED_AND_APPLIED,
                applied_changes=["/file1.py"],
                failed_changes=[],
            ),
            WorkflowContext(
                operation_id="op2",
                operation_type="test",
                final_result=WorkflowResult.DENIED,
                applied_changes=[],
                failed_changes=[],
            ),
            WorkflowContext(
                operation_id="op3",
                operation_type="test",
                final_result=WorkflowResult.APPROVED_AND_APPLIED,
                applied_changes=["/file2.py", "/file3.py"],
                failed_changes=[("/file4.py", "Error")],
            ),
        ]

        stats = self.workflow.get_workflow_statistics()

        assert stats["total_workflows"] == 3
        assert stats["result_distribution"]["approved_and_applied"] == 2
        assert stats["result_distribution"]["denied"] == 1
        assert stats["success_rate"] == 2 / 3 * 100  # 66.67%
        assert stats["average_changes_per_workflow"] == (1 + 0 + 3) / 3  # 1.33

    def test_get_workflow_statistics_empty(self):
        """Test workflow statistics when no workflows exist."""
        stats = self.workflow.get_workflow_statistics()
        assert stats["total_workflows"] == 0

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self):
        """Test timeout handling in workflow."""
        change_set = self.create_test_change_set(1)

        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch:
            mock_fetch.return_value = change_set

            with patch.object(
                self.workflow, "_handle_change_approval"
            ) as mock_approval:
                mock_approval.side_effect = asyncio.TimeoutError(
                    "Timeout occurred"
                )

                result = (
                    await self.workflow.execute_file_modification_workflow(
                        "timeout-test"
                    )
                )

                assert result.final_result == WorkflowResult.TIMEOUT
                assert result.current_state == WorkflowState.ERROR

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test error handling in workflow."""
        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch:
            mock_fetch.side_effect = Exception("Fetch error")

            result = await self.workflow.execute_file_modification_workflow(
                "error-test"
            )

            assert result.final_result == WorkflowResult.ERROR
            assert result.current_state == WorkflowState.ERROR

    def test_workflow_config_defaults(self):
        """Test workflow configuration defaults."""
        default_config = WorkflowConfig()

        assert default_config.require_confirmation is True
        assert default_config.auto_apply_approved is True
        assert default_config.approval_timeout_seconds == 300
        assert default_config.batch_approval_threshold == 5
        assert default_config.save_workflow_history is True
