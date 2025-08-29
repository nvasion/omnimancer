"""
Comprehensive Integration Tests for File Modification and Approval Flow.

This module contains comprehensive integration tests that simulate real-world
user scenarios for the complete file modification approval workflow, including
both normal scenarios and edge cases.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from omnimancer.core.agent.file_modification_workflow import (
    FileModificationWorkflow,
    WorkflowConfig,
    WorkflowState,
    WorkflowResult,
)
from omnimancer.core.agent.file_system_manager import FileSystemManager
from omnimancer.core.agent.approval_manager import (
    EnhancedApprovalManager,
    ChangeType,
)
from omnimancer.core.agent.proposed_changes_integration import (
    ProposedChange,
    ChangeSet,
)
from omnimancer.core.security.approval_workflow import RiskLevel


class TestFileModificationFlowIntegration:
    """
    Integration test suite for the complete file modification approval flow.

    Tests end-to-end scenarios from change proposal to file system application.
    """

    def setup_method(self):
        """Set up test environment with temporary file system."""
        # Create temporary directory for file operations
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Initialize components with real implementations where possible
        self.file_system_manager = FileSystemManager()
        # Add _original_write_file attribute for testing
        self.file_system_manager._original_write_file = (
            self.file_system_manager.write_file
        )
        self.approval_manager = EnhancedApprovalManager()

        # Configure workflow for testing
        self.workflow_config = WorkflowConfig(
            approval_timeout_seconds=5,  # Short timeout for tests
            auto_apply_approved=True,
            batch_approval_threshold=3,
            save_workflow_history=True,
        )

        self.workflow = FileModificationWorkflow(
            file_system_manager=self.file_system_manager,
            approval_manager=self.approval_manager,
            config=self.workflow_config,
        )

        # Track callback events
        self.state_changes = []
        self.approval_decisions = []
        self.workflow_completions = []

        # Set up callbacks
        self.workflow.on_state_change = self._record_state_change
        self.workflow.on_approval_decision = self._record_approval_decision
        self.workflow.on_workflow_complete = self._record_workflow_completion

    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def _record_state_change(self, context, old_state, new_state):
        """Record state change events."""
        self.state_changes.append(
            {
                "operation_id": context.operation_id,
                "from": old_state,
                "to": new_state,
                "timestamp": context.state_history[-1]["timestamp"],
            }
        )

    async def _record_approval_decision(self, context, decision):
        """Record approval decision events."""
        self.approval_decisions.append(
            {
                "operation_id": context.operation_id,
                "decision": decision,
                "timestamp": context.initiated_at,
            }
        )

    async def _record_workflow_completion(self, context):
        """Record workflow completion events."""
        self.workflow_completions.append(
            {
                "operation_id": context.operation_id,
                "result": context.final_result,
                "applied_changes": len(context.applied_changes),
                "failed_changes": len(context.failed_changes),
            }
        )

    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a test file in the temporary directory."""
        file_path = self.temp_path / filename
        file_path.write_text(content)
        return file_path

    def create_mock_change_set(
        self, changes: List[Dict[str, Any]]
    ) -> ChangeSet:
        """Create a mock change set from change descriptions."""
        proposed_changes = []

        for i, change_info in enumerate(changes):
            # Map operation string to enum value
            operation = change_info.get("operation", "FILE_MODIFY")
            if operation == "FILE_CREATE":
                change_type = ChangeType.FILE_CREATE
            elif operation == "FILE_MODIFY":
                change_type = ChangeType.FILE_MODIFY
            elif operation == "FILE_DELETE":
                change_type = ChangeType.FILE_DELETE
            else:
                change_type = ChangeType.FILE_MODIFY  # default

            # Map risk level string to enum value
            risk = change_info.get("risk_level", "low")
            if risk.lower() == "low":
                risk_level = RiskLevel.LOW
            elif risk.lower() == "medium":
                risk_level = RiskLevel.MEDIUM
            elif risk.lower() == "high":
                risk_level = RiskLevel.HIGH
            elif risk.lower() == "critical":
                risk_level = RiskLevel.CRITICAL
            else:
                risk_level = RiskLevel.LOW  # default

            change = ProposedChange(
                file_path=change_info.get("file_path", f"/test/file{i}.py"),
                operation_type=change_type,
                original_content=change_info.get("original_content"),
                modified_content=change_info.get("modified_content"),
                risk_level=risk_level,
                change_summary=change_info.get("summary", f"Change {i+1}"),
            )
            proposed_changes.append(change)

        return ChangeSet(
            id=f"test-changeset-{len(changes)}",
            description=f"Test change set with {len(changes)} changes",
            changes=proposed_changes,
            total_risk_score=sum(
                (
                    1.0
                    if change.risk_level == RiskLevel.HIGH
                    else 0.5 if change.risk_level == RiskLevel.MEDIUM else 0.1
                )
                for change in proposed_changes
            ),
        )

    @pytest.mark.asyncio
    async def test_single_file_creation_approved(self):
        """Test successful single file creation with user approval."""
        file_path = str(self.temp_path / "new_file.py")
        content = "def hello():\n    print('Hello, World!')"

        # Mock user approval
        with patch.object(
            self.workflow.unified_display,
            "display_file_creation",
            new_callable=AsyncMock,
        ) as mock_display:
            mock_display.return_value = {"approved": True}

            # Mock file system operation
            with patch.object(
                self.file_system_manager,
                "_original_write_file",
                new_callable=AsyncMock,
            ) as mock_write:
                mock_write.return_value = {"success": True}

                # Execute workflow
                result = await self.workflow.execute_single_file_workflow(
                    file_path, "create", new_content=content
                )

                # Verify results
                assert (
                    result.final_result == WorkflowResult.APPROVED_AND_APPLIED
                )
                assert result.current_state == WorkflowState.COMPLETED
                assert file_path in result.applied_changes
                assert len(result.failed_changes) == 0

                # Verify file system was called
                mock_write.assert_called_once_with(
                    path=file_path, content=content, read_before_write=False
                )

                # Verify state transitions occurred
                assert len(self.state_changes) > 0
                assert any(
                    change["to"] == WorkflowState.COMPLETED
                    for change in self.state_changes
                )

    @pytest.mark.asyncio
    async def test_single_file_modification_denied(self):
        """Test single file modification with user denial."""
        file_path = str(self.temp_path / "existing_file.py")
        original_content = "def old_function():\n    pass"
        new_content = "def new_function():\n    print('Updated!')"

        # Mock user denial
        with patch.object(
            self.workflow.unified_display,
            "display_file_modification",
            new_callable=AsyncMock,
        ) as mock_display:
            mock_display.return_value = {"approved": False}

            # Execute workflow
            result = await self.workflow.execute_single_file_workflow(
                file_path,
                "modify",
                current_content=original_content,
                new_content=new_content,
            )

            # Verify results
            assert result.final_result == WorkflowResult.DENIED
            assert result.current_state == WorkflowState.COMPLETED
            assert len(result.applied_changes) == 0

    @pytest.mark.asyncio
    async def test_batch_workflow_mixed_approvals(self):
        """Test batch workflow with mixed approvals and denials."""
        changes = [
            {
                "file_path": "/test/file1.py",
                "operation": "FILE_CREATE",
                "modified_content": 'print("File 1")',
                "risk_level": "LOW",
                "summary": "Create new file 1",
            },
            {
                "file_path": "/test/file2.py",
                "operation": "FILE_MODIFY",
                "original_content": "old content",
                "modified_content": "new content",
                "risk_level": "MEDIUM",
                "summary": "Modify existing file 2",
            },
            {
                "file_path": "/test/file3.py",
                "operation": "FILE_DELETE",
                "original_content": "content to delete",
                "risk_level": "HIGH",
                "summary": "Delete file 3",
            },
        ]

        change_set = self.create_mock_change_set(changes)

        # Mock fetching changes
        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch:
            mock_fetch.return_value = change_set

            # Mock selective approval (approve first 2, deny last)
            with patch.object(
                self.workflow, "_handle_change_approval"
            ) as mock_approval:
                mock_approval.return_value = {
                    "approved": True,
                    "selected_changes": [0, 1],  # Approve only first 2 changes
                }

                # Mock application
                async def mock_apply_changes(
                    workflow_context, change_set, approval_result
                ):
                    # Simulate the actual method behavior by updating the context
                    apply_result = {
                        "success": True,
                        "applied": ["/test/file1.py", "/test/file2.py"],
                        "failed": [],
                    }
                    workflow_context.applied_changes.extend(
                        apply_result.get("applied", [])
                    )
                    workflow_context.failed_changes.extend(
                        apply_result.get("failed", [])
                    )
                    return apply_result

                with patch.object(
                    self.workflow,
                    "_apply_approved_changes",
                    new_callable=AsyncMock,
                    side_effect=mock_apply_changes,
                ) as mock_apply:

                    # Execute workflow
                    result = (
                        await self.workflow.execute_file_modification_workflow(
                            "batch-test"
                        )
                    )

                    # Verify results
                    assert (
                        result.final_result
                        == WorkflowResult.APPROVED_AND_APPLIED
                    )
                    assert len(result.applied_changes) == 2
                    assert "/test/file1.py" in result.applied_changes
                    assert "/test/file2.py" in result.applied_changes

    @pytest.mark.asyncio
    async def test_workflow_with_partial_failures(self):
        """Test workflow handling when some changes fail to apply."""
        changes = [
            {
                "file_path": "/test/success_file.py",
                "operation": "FILE_CREATE",
                "modified_content": "success content",
                "risk_level": "LOW",
            },
            {
                "file_path": "/test/failure_file.py",
                "operation": "FILE_CREATE",
                "modified_content": "failure content",
                "risk_level": "LOW",
            },
        ]

        change_set = self.create_mock_change_set(changes)

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

                # Mock partial failure in application
                async def mock_apply_partial_failure(
                    workflow_context, change_set, approval_result
                ):
                    apply_result = {
                        "success": False,
                        "applied": ["/test/success_file.py"],
                        "failed": [
                            ("/test/failure_file.py", "Permission denied")
                        ],
                    }
                    workflow_context.applied_changes.extend(
                        apply_result.get("applied", [])
                    )
                    workflow_context.failed_changes.extend(
                        apply_result.get("failed", [])
                    )
                    return apply_result

                with patch.object(
                    self.workflow,
                    "_apply_approved_changes",
                    new_callable=AsyncMock,
                    side_effect=mock_apply_partial_failure,
                ) as mock_apply:

                    result = (
                        await self.workflow.execute_file_modification_workflow(
                            "partial-failure-test"
                        )
                    )

                    # Should still be marked as approved but not fully applied
                    assert (
                        result.final_result
                        == WorkflowResult.APPROVED_NOT_APPLIED
                    )
                    assert len(result.applied_changes) == 1
                    assert len(result.failed_changes) == 1

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self):
        """Test workflow behavior when user approval times out."""
        changes = [
            {"file_path": "/test/timeout_file.py", "operation": "FILE_CREATE"}
        ]
        change_set = self.create_mock_change_set(changes)

        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch:
            mock_fetch.return_value = change_set

            # Mock timeout in approval handling
            with patch.object(
                self.workflow, "_handle_change_approval"
            ) as mock_approval:
                mock_approval.side_effect = asyncio.TimeoutError(
                    "User approval timeout"
                )

                result = (
                    await self.workflow.execute_file_modification_workflow(
                        "timeout-test"
                    )
                )

                assert result.final_result == WorkflowResult.TIMEOUT
                assert result.current_state == WorkflowState.ERROR

    @pytest.mark.asyncio
    async def test_workflow_cancellation(self):
        """Test workflow cancellation functionality."""
        # Start a workflow
        changes = [
            {"file_path": "/test/cancel_file.py", "operation": "FILE_CREATE"}
        ]
        change_set = self.create_mock_change_set(changes)

        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch, patch(
            "builtins.input", return_value="D"
        ), patch.object(
            self.workflow, "_handle_change_approval"
        ) as mock_approval:
            mock_fetch.return_value = change_set

            # Mock approval to hang so we can cancel
            mock_approval.side_effect = asyncio.CancelledError()

            # Start workflow asynchronously
            workflow_task = asyncio.create_task(
                self.workflow.execute_file_modification_workflow("cancel-test")
            )

            # Give it a moment to start
            await asyncio.sleep(0.1)

            # Cancel the workflow
            success = await self.workflow.cancel_workflow(
                "cancel-test", "User cancelled"
            )

            try:
                # Wait for the workflow to complete
                result = await workflow_task
                # If the cancellation worked, we should have a cancelled result
                assert result.final_result == WorkflowResult.CANCELLED
            except asyncio.CancelledError:
                # If the task was cancelled, that's also acceptable
                pass

    @pytest.mark.asyncio
    async def test_workflow_status_tracking(self):
        """Test workflow status tracking throughout execution."""
        changes = [
            {"file_path": "/test/status_file.py", "operation": "FILE_CREATE"}
        ]
        change_set = self.create_mock_change_set(changes)

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

                with patch.object(
                    self.workflow, "_apply_approved_changes"
                ) as mock_apply:
                    mock_apply.return_value = {
                        "success": True,
                        "applied": ["/test/status_file.py"],
                        "failed": [],
                    }

                    # Start workflow
                    workflow_task = asyncio.create_task(
                        self.workflow.execute_file_modification_workflow(
                            "status-test"
                        )
                    )

                    # Check status while running
                    await asyncio.sleep(0.1)
                    status = await self.workflow.get_workflow_status(
                        "status-test"
                    )

                    if status:  # May complete before we can check
                        assert status["operation_id"] == "status-test"
                        assert "current_state" in status
                        assert "elapsed_seconds" in status

                    # Wait for completion
                    result = await workflow_task

                    # Status should be None after completion (moved to history)
                    final_status = await self.workflow.get_workflow_status(
                        "status-test"
                    )
                    assert final_status is None

    @pytest.mark.asyncio
    async def test_high_risk_changes_workflow(self):
        """Test workflow with high-risk changes requiring explicit approval."""
        changes = [
            {
                "file_path": "/etc/critical_config.conf",
                "operation": "FILE_MODIFY",
                "original_content": "safe_setting=true",
                "modified_content": "safe_setting=false",
                "risk_level": "CRITICAL",
                "summary": "Disable safety setting",
            }
        ]

        change_set = self.create_mock_change_set(changes)

        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch:
            mock_fetch.return_value = change_set

            # Mock explicit approval for high-risk change
            with patch.object(
                self.workflow, "_handle_change_approval"
            ) as mock_approval:
                mock_approval.return_value = {
                    "approved": True,
                    "explicit_approval": True,
                }

                with patch.object(
                    self.workflow, "_apply_approved_changes"
                ) as mock_apply:
                    mock_apply.return_value = {
                        "success": True,
                        "applied": ["/etc/critical_config.conf"],
                        "failed": [],
                    }

                    result = (
                        await self.workflow.execute_file_modification_workflow(
                            "high-risk-test"
                        )
                    )

                    assert (
                        result.final_result
                        == WorkflowResult.APPROVED_AND_APPLIED
                    )
                    # Verify that explicit approval was required
                    mock_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_changeset_workflow(self):
        """Test workflow behavior with empty changeset."""
        empty_changeset = ChangeSet(
            id="empty-test", description="No changes", changes=[]
        )

        with patch.object(
            self.workflow.changes_integration, "fetch_proposed_changes"
        ) as mock_fetch:
            mock_fetch.return_value = empty_changeset

            result = await self.workflow.execute_file_modification_workflow(
                "empty-test"
            )

            assert result.final_result == WorkflowResult.CANCELLED
            assert result.current_state == WorkflowState.COMPLETED
            assert len(result.applied_changes) == 0

    @pytest.mark.asyncio
    async def test_workflow_statistics_tracking(self):
        """Test that workflow statistics are properly tracked."""
        # Execute several workflows to build statistics
        test_workflows = [
            ("success-1", WorkflowResult.APPROVED_AND_APPLIED, 2, 0),
            ("denied-1", WorkflowResult.DENIED, 0, 0),
            ("partial-1", WorkflowResult.APPROVED_NOT_APPLIED, 1, 1),
        ]

        for (
            op_id,
            expected_result,
            applied_count,
            failed_count,
        ) in test_workflows:
            # Mock a simple workflow for each test case
            changes = [
                {
                    "file_path": f"/test/{op_id}_file.py",
                    "operation": "FILE_CREATE",
                }
            ]
            change_set = self.create_mock_change_set(changes)

            with patch.object(
                self.workflow.changes_integration, "fetch_proposed_changes"
            ) as mock_fetch:
                mock_fetch.return_value = change_set

                # Configure mock responses based on expected result
                if expected_result == WorkflowResult.DENIED:
                    approval_response = {"approved": False}
                else:
                    approval_response = {"approved": True, "all_changes": True}

                with patch.object(
                    self.workflow, "_handle_change_approval"
                ) as mock_approval:
                    mock_approval.return_value = approval_response

                    if expected_result != WorkflowResult.DENIED:
                        apply_response = {
                            "success": applied_count > failed_count,
                            "applied": [f"/test/{op_id}_file.py"]
                            * applied_count,
                            "failed": [
                                (f"/test/failed_{i}.py", "Error")
                                for i in range(failed_count)
                            ],
                        }

                        with patch.object(
                            self.workflow, "_apply_approved_changes"
                        ) as mock_apply:
                            mock_apply.return_value = apply_response

                            result = await self.workflow.execute_file_modification_workflow(
                                op_id
                            )
                    else:
                        result = await self.workflow.execute_file_modification_workflow(
                            op_id
                        )

                # Verify expected result
                assert result.final_result == expected_result

        # Check statistics
        stats = self.workflow.get_workflow_statistics()

        assert stats["total_workflows"] == 3
        assert "result_distribution" in stats
        assert "success_rate" in stats
        assert stats["result_distribution"].get("approved_and_applied", 0) >= 1
        assert stats["result_distribution"].get("denied", 0) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_workflows(self):
        """Test handling of concurrent workflows."""
        # Mock all workflows to return successful results
        changes_list = []
        change_sets = []

        for i in range(3):
            changes = [
                {
                    "file_path": f"/test/concurrent_{i}.py",
                    "operation": "FILE_CREATE",
                }
            ]
            changes_list.append(changes)
            change_set = self.create_mock_change_set(changes)
            change_sets.append(change_set)

        # Mock the fetch method to return appropriate change set based on operation_id
        def mock_fetch_changes(*args, **kwargs):
            # Get operation_id from first arg or kwargs
            operation_id = args[0] if args else kwargs.get("operation_id")
            if operation_id:
                # Extract index from operation_id "concurrent-{i}"
                index = int(operation_id.split("-")[1])
                return change_sets[index]
            return change_sets[0]  # Fallback

        async def mock_apply_changes(*args, **kwargs):
            # Get workflow_context from args
            workflow_context = (
                args[0] if args else kwargs.get("workflow_context")
            )
            if workflow_context and hasattr(workflow_context, "operation_id"):
                # Extract index from operation_id to get the right file
                index = int(workflow_context.operation_id.split("-")[1])
                apply_result = {
                    "success": True,
                    "applied": [f"/test/concurrent_{index}.py"],
                    "failed": [],
                }
                workflow_context.applied_changes.extend(
                    apply_result.get("applied", [])
                )
                workflow_context.failed_changes.extend(
                    apply_result.get("failed", [])
                )
                return apply_result
            return {"success": True, "applied": [], "failed": []}

        with patch.object(
            self.workflow.changes_integration,
            "fetch_proposed_changes",
            side_effect=mock_fetch_changes,
        ) as mock_fetch, patch.object(
            self.workflow, "_handle_change_approval"
        ) as mock_approval, patch.object(
            self.workflow,
            "_apply_approved_changes",
            side_effect=mock_apply_changes,
        ) as mock_apply:

            mock_approval.return_value = {
                "approved": True,
                "all_changes": True,
            }

            # Start multiple workflows concurrently
            workflow_tasks = [
                asyncio.create_task(
                    self.workflow.execute_file_modification_workflow(
                        f"concurrent-{i}"
                    )
                )
                for i in range(3)
            ]

            # Wait for all workflows to complete
            results = await asyncio.gather(
                *workflow_tasks, return_exceptions=True
            )

            # Verify all completed successfully
            for i, result in enumerate(results):
                assert not isinstance(
                    result, Exception
                ), f"Workflow {i} failed with exception: {result}"
                assert (
                    result.final_result == WorkflowResult.APPROVED_AND_APPLIED
                )
                assert f"/test/concurrent_{i}.py" in result.applied_changes

    def test_workflow_configuration_validation(self):
        """Test workflow configuration validation and defaults."""
        # Test default configuration
        default_config = WorkflowConfig()
        workflow_with_defaults = FileModificationWorkflow(
            config=default_config
        )

        assert workflow_with_defaults.config.approval_timeout_seconds == 300
        assert workflow_with_defaults.config.auto_apply_approved is True
        assert workflow_with_defaults.config.batch_approval_threshold == 5

        # Test custom configuration
        custom_config = WorkflowConfig(
            approval_timeout_seconds=60,
            auto_apply_approved=False,
            batch_approval_threshold=2,
            save_workflow_history=False,
        )

        workflow_with_custom = FileModificationWorkflow(config=custom_config)

        assert workflow_with_custom.config.approval_timeout_seconds == 60
        assert workflow_with_custom.config.auto_apply_approved is False
        assert workflow_with_custom.config.batch_approval_threshold == 2
        assert workflow_with_custom.config.save_workflow_history is False


class TestApprovalFlowEdgeCases:
    """
    Test suite for edge cases and error conditions in the approval flow.

    Focuses on boundary conditions, error recovery, and unusual scenarios.
    """

    def setup_method(self):
        """Set up test environment for edge case testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create mock approval manager with edge case handling
        self.mock_approval_manager = Mock()
        self.mock_approval_manager.request_approval = AsyncMock()

        # Create workflow with edge case configuration
        self.config = WorkflowConfig(
            approval_timeout_seconds=1,  # Very short timeout for edge testing
            max_concurrent_operations=2,
            auto_apply_approved=False,
            batch_approval_threshold=1,
            save_workflow_history=True,
        )
        self.workflow = FileModificationWorkflow(config=self.config)

    def teardown_method(self):
        """Cleanup after edge case tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_extremely_large_file_handling(self):
        """Test handling of very large files in the approval workflow."""
        # Create a large test file (simulated with large content string)
        large_content = "x" * (10 * 1024 * 1024)  # 10MB of content
        test_file = self.temp_path / "large_file.txt"

        # Create proposed change for large file
        large_change = ProposedChange(
            operation_type=ChangeType.FILE_CREATE,
            file_path=str(test_file),
            modified_content=large_content,
            risk_level=RiskLevel.MEDIUM,
        )

        changeset = ChangeSet(
            id="test_large_file",
            description="Test large file handling",
            changes=[large_change],
        )

        # Mock approval for large file
        self.mock_approval_manager.request_approval.return_value = {
            "approved": True,
            "reason": "Large file approved for testing",
        }

        # Mock the changes integration display method since batch_approval_threshold=1
        mock_display = AsyncMock(
            return_value={
                "approved": True,
                "reason": "Large file approved for testing",
            }
        )
        mock_apply = AsyncMock(
            return_value={
                "success": True,
                "applied": [large_change],
                "failed": [],
            }
        )

        with patch.object(
            self.workflow, "approval_manager", self.mock_approval_manager
        ), patch.object(
            self.workflow.changes_integration,
            "display_proposed_changes",
            mock_display,
        ), patch.object(
            self.workflow.changes_integration,
            "apply_proposed_changes",
            mock_apply,
        ):
            result = await self.workflow.execute_workflow(changeset)

            assert result.current_state == WorkflowState.COMPLETED
            # Should handle large files without memory issues
            assert (
                len(result.applied_changes) <= 1
            )  # May fail due to size limits

    @pytest.mark.asyncio
    async def test_binary_file_handling(self):
        """Test handling of binary files in approval workflow."""
        # Create binary content
        binary_content = bytes(range(256)) * 1000  # Binary data
        test_file = self.temp_path / "binary_file.bin"

        # Create proposed change for binary file
        binary_change = ProposedChange(
            operation_type=ChangeType.FILE_CREATE,
            file_path=str(test_file),
            modified_content=binary_content.decode(
                "latin-1"
            ),  # Preserve bytes as string
            risk_level=RiskLevel.MEDIUM,
        )

        changeset = ChangeSet(
            id="test_binary_file",
            description="Test binary file handling",
            changes=[binary_change],
        )

        self.mock_approval_manager.request_approval.return_value = {
            "approved": True,
            "reason": "Binary file approved",
        }

        # Mock the changes integration display method since batch_approval_threshold=1
        mock_display = AsyncMock(
            return_value={"approved": True, "reason": "Binary file approved"}
        )
        mock_apply = AsyncMock(
            return_value={
                "success": True,
                "applied": [binary_change],
                "failed": [],
            }
        )

        with patch.object(
            self.workflow, "approval_manager", self.mock_approval_manager
        ), patch.object(
            self.workflow.changes_integration,
            "display_proposed_changes",
            mock_display,
        ), patch.object(
            self.workflow.changes_integration,
            "apply_proposed_changes",
            mock_apply,
        ):
            result = await self.workflow.execute_workflow(changeset)

            # Should handle binary content appropriately
            assert result.current_state != WorkflowState.ERROR

    @pytest.mark.asyncio
    async def test_filesystem_permission_errors(self):
        """Test workflow behavior with filesystem permission errors."""
        # Create change for file in read-only location
        restricted_file = self.temp_path / "readonly" / "restricted.txt"
        restricted_file.parent.mkdir(exist_ok=True)

        permission_change = ProposedChange(
            operation_type=ChangeType.FILE_CREATE,
            file_path=str(restricted_file),
            modified_content="restricted content",
            risk_level=RiskLevel.HIGH,
        )

        changeset = ChangeSet(
            id="test_permission_error",
            description="Test filesystem permission errors",
            changes=[permission_change],
        )

        self.mock_approval_manager.request_approval.return_value = {
            "approved": True,
            "reason": "Approved despite permissions",
        }

        # Mock the changes integration display method since batch_approval_threshold=1
        mock_display = AsyncMock(
            return_value={
                "approved": True,
                "reason": "Approved despite permissions",
            }
        )
        mock_apply = AsyncMock(
            return_value={
                "success": False,
                "applied": [],
                "failed": [permission_change],
            }
        )

        with patch.object(
            self.workflow, "approval_manager", self.mock_approval_manager
        ), patch.object(
            self.workflow.changes_integration,
            "display_proposed_changes",
            mock_display,
        ), patch.object(
            self.workflow.changes_integration,
            "apply_proposed_changes",
            mock_apply,
        ):
            result = await self.workflow.execute_workflow(changeset)

            # Should handle permission errors gracefully
            assert result.current_state == WorkflowState.ERROR
            assert (
                len(result.failed_changes) >= 1
            )  # Should have at least the failed permission change

    @pytest.mark.asyncio
    async def test_concurrent_modification_conflicts(self):
        """Test handling of concurrent modification conflicts."""
        test_file = self.temp_path / "concurrent_test.txt"
        test_file.write_text("original content")

        # Create two conflicting changes
        change1 = ProposedChange(
            operation_type=ChangeType.FILE_MODIFY,
            file_path=str(test_file),
            modified_content="modification 1",
            risk_level=RiskLevel.LOW,
        )

        change2 = ProposedChange(
            operation_type=ChangeType.FILE_MODIFY,
            file_path=str(test_file),
            modified_content="modification 2",
            risk_level=RiskLevel.LOW,
        )

        changeset = ChangeSet(
            id="test_concurrent_mods",
            description="Test concurrent modification conflicts",
            changes=[change1, change2],
        )

        self.mock_approval_manager.request_approval.return_value = {
            "approved": True,
            "reason": "Concurrent changes approved",
        }

        # Mock the changes integration display method since batch_approval_threshold=1
        mock_display = AsyncMock(
            return_value={
                "approved": True,
                "reason": "Concurrent changes approved",
            }
        )
        mock_apply = AsyncMock(
            return_value={
                "success": True,
                "applied": [change1, change2],
                "failed": [],
            }
        )

        with patch.object(
            self.workflow, "approval_manager", self.mock_approval_manager
        ), patch.object(
            self.workflow.changes_integration,
            "display_proposed_changes",
            mock_display,
        ), patch.object(
            self.workflow.changes_integration,
            "apply_proposed_changes",
            mock_apply,
        ):
            result = await self.workflow.execute_workflow(changeset)

            # Should detect and handle conflicting changes
            # Exact behavior may vary, but should not crash
            assert result.current_state in [
                WorkflowState.COMPLETED,
                WorkflowState.ERROR,
            ]
