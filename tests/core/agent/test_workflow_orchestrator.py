"""
Tests for the workflow orchestrator and continuous execution functionality.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from omnimancer.core.agent.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowContext,
    WorkflowStep,
    WorkflowStepType,
    WorkflowStatus,
)


class TestWorkflowOrchestrator:
    """Test cases for the WorkflowOrchestrator class."""

    @pytest.fixture
    def mock_file_system(self):
        """Mock file system manager."""
        mock = Mock()
        mock.read_file = AsyncMock(return_value="test content")
        mock.write_file = AsyncMock(return_value=True)
        return mock

    @pytest.fixture
    def mock_approval_manager(self):
        """Mock approval manager."""
        mock = Mock()
        mock.request_approval = AsyncMock(return_value=True)
        return mock

    @pytest.fixture
    def mock_executor(self):
        """Mock program executor."""
        mock = Mock()
        mock.execute_command = AsyncMock(
            return_value={"success": True, "output": "test output"}
        )
        return mock

    @pytest.fixture
    def orchestrator(
        self, mock_file_system, mock_approval_manager, mock_executor
    ):
        """Create a workflow orchestrator with mocked dependencies."""
        return WorkflowOrchestrator(
            file_system=mock_file_system,
            approval_manager=mock_approval_manager,
            executor=mock_executor,
        )

    @pytest.fixture
    def sample_context(self):
        """Create a sample workflow context."""
        return WorkflowContext(working_directory=Path.cwd())

    def test_workflow_orchestrator_initialization(self, orchestrator):
        """Test that the workflow orchestrator initializes correctly."""
        assert orchestrator.file_system is not None
        assert orchestrator.approval_manager is not None
        assert orchestrator.executor is not None
        assert "project_analysis" in orchestrator.workflows
        assert "file_modification" in orchestrator.workflows

    def test_register_workflow(self, orchestrator):
        """Test registering a custom workflow."""
        steps = [
            WorkflowStep(
                name="test_step",
                type=WorkflowStepType.CUSTOM,
                description="Test step",
            )
        ]

        orchestrator.register_workflow("test_workflow", steps)
        assert "test_workflow" in orchestrator.workflows
        assert len(orchestrator.workflows["test_workflow"]) == 1

    @pytest.mark.asyncio
    async def test_project_analysis_workflow(
        self, orchestrator, sample_context
    ):
        """Test the built-in project analysis workflow."""
        with patch("pathlib.Path.iterdir") as mock_iterdir:
            # Mock directory contents
            mock_files = [
                Mock(is_dir=lambda: False, name="package.json"),
                Mock(is_dir=lambda: True, name=".git"),
                Mock(is_dir=lambda: False, name="requirements.txt"),
            ]
            mock_iterdir.return_value = mock_files

            result = await orchestrator.execute_workflow(
                "project_analysis", sample_context
            )

            assert result is not None
            assert len(result.history) > 0

            # Check that steps were executed
            completed_steps = [
                s
                for s in result.history
                if s.status == WorkflowStatus.COMPLETED
            ]
            assert len(completed_steps) > 0

            # Check context data
            assert "project_files" in result.data
            assert "tech_stack" in result.data

    @pytest.mark.asyncio
    async def test_file_modification_workflow(
        self, orchestrator, sample_context
    ):
        """Test the built-in file modification workflow."""
        parameters = {
            "file_path": "test.txt",
            "changes": {"content": "new content"},
        }

        result = await orchestrator.execute_workflow(
            "file_modification", sample_context, parameters
        )

        assert result is not None
        assert len(result.history) > 0

        # Check that file operations were called
        orchestrator.file_system.read_file.assert_called_once()

        # Verify context data
        assert "original_content" in result.data
        assert "modified_content" in result.data

    @pytest.mark.asyncio
    async def test_workflow_step_dependencies(self, orchestrator):
        """Test that workflow step dependencies are respected."""
        steps = [
            WorkflowStep(
                name="step1",
                type=WorkflowStepType.CUSTOM,
                description="First step",
                action=AsyncMock(return_value="step1_result"),
            ),
            WorkflowStep(
                name="step2",
                type=WorkflowStepType.CUSTOM,
                description="Second step",
                action=AsyncMock(return_value="step2_result"),
                dependencies=["step1"],
            ),
            WorkflowStep(
                name="step3",
                type=WorkflowStepType.CUSTOM,
                description="Third step",
                action=AsyncMock(return_value="step3_result"),
                dependencies=["step1", "step2"],
            ),
        ]

        orchestrator.register_workflow("dependency_test", steps)

        context = WorkflowContext(working_directory=Path.cwd())
        result = await orchestrator.execute_workflow(
            "dependency_test", context
        )

        # Check that all steps completed
        completed_steps = [
            s for s in result.history if s.status == WorkflowStatus.COMPLETED
        ]
        assert len(completed_steps) == 3

        # Check execution order
        step_names = [s.name for s in result.history]
        assert step_names.index("step1") < step_names.index("step2")
        assert step_names.index("step2") < step_names.index("step3")

    @pytest.mark.asyncio
    async def test_workflow_approval_flow(self, orchestrator, sample_context):
        """Test workflow steps that require approval."""
        steps = [
            WorkflowStep(
                name="approval_step",
                type=WorkflowStepType.CUSTOM,
                description="Step requiring approval",
                action=AsyncMock(return_value="approved_result"),
                requires_approval=True,
            )
        ]

        orchestrator.register_workflow("approval_test", steps)

        result = await orchestrator.execute_workflow(
            "approval_test", sample_context
        )

        # Check that approval was requested
        orchestrator.approval_manager.request_approval.assert_called()

        # Check that step completed
        completed_steps = [
            s for s in result.history if s.status == WorkflowStatus.COMPLETED
        ]
        assert len(completed_steps) == 1

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, orchestrator, sample_context):
        """Test workflow error handling and continue_on_error."""

        def failing_action(*args, **kwargs):
            raise ValueError("Test error")

        steps = [
            WorkflowStep(
                name="failing_step",
                type=WorkflowStepType.CUSTOM,
                description="Step that fails",
                action=failing_action,
                continue_on_error=True,
            ),
            WorkflowStep(
                name="success_step",
                type=WorkflowStepType.CUSTOM,
                description="Step that succeeds",
                action=AsyncMock(return_value="success"),
            ),
        ]

        orchestrator.register_workflow("error_test", steps)

        result = await orchestrator.execute_workflow(
            "error_test", sample_context
        )

        # Check that first step failed but workflow continued
        failed_steps = [
            s for s in result.history if s.status == WorkflowStatus.FAILED
        ]
        completed_steps = [
            s for s in result.history if s.status == WorkflowStatus.COMPLETED
        ]

        assert len(failed_steps) == 1
        assert len(completed_steps) == 1
        assert failed_steps[0].error == "Test error"

    @pytest.mark.asyncio
    async def test_workflow_context_sharing(self, orchestrator):
        """Test that context data is shared between workflow steps."""

        async def step1_action(context, params):
            context.set("shared_value", "test_data")
            return "step1_complete"

        async def step2_action(context, params):
            shared_value = context.get("shared_value")
            return f"received: {shared_value}"

        steps = [
            WorkflowStep(
                name="step1",
                type=WorkflowStepType.CUSTOM,
                description="Set context data",
                action=step1_action,
            ),
            WorkflowStep(
                name="step2",
                type=WorkflowStepType.CUSTOM,
                description="Use context data",
                action=step2_action,
                dependencies=["step1"],
            ),
        ]

        orchestrator.register_workflow("context_test", steps)

        context = WorkflowContext(working_directory=Path.cwd())
        result = await orchestrator.execute_workflow("context_test", context)

        # Check that context was shared
        assert result.get("shared_value") == "test_data"

        # Check step results
        step2 = next(s for s in result.history if s.name == "step2")
        assert step2.result == "received: test_data"


class TestWorkflowContext:
    """Test cases for the WorkflowContext class."""

    def test_context_initialization(self):
        """Test WorkflowContext initialization."""
        context = WorkflowContext(working_directory=Path.cwd())

        assert context.working_directory == Path.cwd()
        assert isinstance(context.data, dict)
        assert len(context.history) == 0

    def test_context_data_operations(self):
        """Test context data operations."""
        context = WorkflowContext(working_directory=Path.cwd())

        # Test set and get
        context.set("key1", "value1")
        assert context.get("key1") == "value1"

        # Test get with default
        assert context.get("nonexistent", "default") == "default"

        # Test update
        context.update(key2="value2", key3="value3")
        assert context.get("key2") == "value2"
        assert context.get("key3") == "value3"


class TestWorkflowIntegration:
    """Integration tests for workflow functionality."""

    @pytest.mark.asyncio
    async def test_continuous_execution_flow(self):
        """Test that workflows execute continuously without stopping."""
        orchestrator = WorkflowOrchestrator()

        # Track execution order
        execution_order = []

        async def tracked_action(name):
            async def action(context, params):
                execution_order.append(name)
                await asyncio.sleep(0.01)  # Simulate work
                return f"{name}_result"

            return action

        steps = [
            WorkflowStep(
                name="analyze",
                type=WorkflowStepType.ANALYZE,
                description="Analyze environment",
                action=await tracked_action("analyze"),
            ),
            WorkflowStep(
                name="prepare",
                type=WorkflowStepType.CUSTOM,
                description="Prepare changes",
                action=await tracked_action("prepare"),
                dependencies=["analyze"],
            ),
            WorkflowStep(
                name="execute",
                type=WorkflowStepType.CUSTOM,
                description="Execute changes",
                action=await tracked_action("execute"),
                dependencies=["prepare"],
            ),
            WorkflowStep(
                name="validate",
                type=WorkflowStepType.VALIDATE,
                description="Validate results",
                action=await tracked_action("validate"),
                dependencies=["execute"],
            ),
        ]

        orchestrator.register_workflow("continuous_test", steps)

        context = WorkflowContext(working_directory=Path.cwd())
        result = await orchestrator.execute_workflow(
            "continuous_test", context
        )

        # Verify continuous execution
        assert execution_order == ["analyze", "prepare", "execute", "validate"]

        # Verify all steps completed
        completed_steps = [
            s for s in result.history if s.status == WorkflowStatus.COMPLETED
        ]
        assert len(completed_steps) == 4

        # Verify timing shows continuous execution
        for i in range(1, len(result.history)):
            prev_step = result.history[i - 1]
            curr_step = result.history[i]

            # Each step should start after the previous one completes
            assert prev_step.completed_at <= curr_step.started_at
