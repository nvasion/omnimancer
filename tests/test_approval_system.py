"""
Comprehensive tests for the enhanced approval system.
"""

import tempfile
from pathlib import Path

import pytest

from omnimancer.core.agent.approval_interface import (
    ApprovalInterface,
)
from omnimancer.core.agent.approval_manager import (
    BatchApprovalRequest,
    ChangePreview,
    ChangeType,
    EnhancedApprovalManager,
    PreviewFormat,
)
from omnimancer.core.agent_engine import (
    Operation,
    OperationType,
)
from omnimancer.core.security.approval_workflow import (
    ApprovalWorkflow,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def approval_workflow():
    """Create approval workflow instance."""
    return ApprovalWorkflow(default_expiry_minutes=30, auto_approve_low_risk=True)


@pytest.fixture
def approval_manager(approval_workflow):
    """Create enhanced approval manager."""
    return EnhancedApprovalManager(
        approval_workflow=approval_workflow,
        enable_batch_approval=True,
        max_batch_size=10,
    )


@pytest.fixture
def approval_interface(approval_manager):
    """Create approval interface."""
    return ApprovalInterface(approval_manager)


class TestChangePreview:
    """Test change preview functionality."""

    def test_create_change_preview(self):
        """Test creating a change preview."""
        preview = ChangePreview(
            change_type=ChangeType.FILE_MODIFY,
            description="Test file modification",
            current_state="old content",
            proposed_state="new content",
            metadata={"path": "/test/file.txt"},
            risk_assessment="Low - Safe operation",
            reversible=True,
        )

        assert preview.change_type == ChangeType.FILE_MODIFY
        assert preview.description == "Test file modification"
        assert preview.reversible is True
        assert preview.metadata["path"] == "/test/file.txt"

    def test_generate_diff(self):
        """Test diff generation."""
        preview = ChangePreview(
            change_type=ChangeType.FILE_MODIFY,
            description="Test diff",
            current_state="line 1\nline 2\nline 3",
            proposed_state="line 1\nmodified line 2\nline 3",
            metadata={"path": "test.txt"},
        )

        diff = preview.generate_diff()
        assert diff is not None
        assert "modified line 2" in diff
        assert "-line 2" in diff or "+modified line 2" in diff

    def test_format_preview_text(self):
        """Test text format preview."""
        preview = ChangePreview(
            change_type=ChangeType.FILE_CREATE,
            description="Create new file",
            metadata={"path": "/new/file.txt", "size": 100},
            risk_assessment="Low - File creation",
            reversible=False,
        )

        text_preview = preview.format_preview(PreviewFormat.TEXT)
        assert "Change Type: file_create" in text_preview
        assert "Create new file" in text_preview
        assert "Reversible: No" in text_preview
        assert "path: /new/file.txt" in text_preview

    def test_format_preview_json(self):
        """Test JSON format preview."""
        preview = ChangePreview(
            change_type=ChangeType.FILE_DELETE,
            description="Delete file",
            metadata={"path": "/delete/file.txt"},
            risk_assessment="Medium - File deletion",
        )

        json_preview = preview.format_preview(PreviewFormat.JSON)
        assert '"type": "file_delete"' in json_preview
        assert '"description": "Delete file"' in json_preview
        assert '"risk_assessment": "Medium - File deletion"' in json_preview

    def test_format_preview_html(self):
        """Test HTML format preview."""
        preview = ChangePreview(
            change_type=ChangeType.COMMAND_EXECUTE,
            description="Execute command",
            diff="+ new line\n- old line",
            risk_assessment="High - Command execution",
        )

        html_preview = preview.format_preview(PreviewFormat.HTML)
        assert "<div class='change-preview'>" in html_preview
        assert "<h3>Command Execute</h3>" in html_preview
        assert "<pre class='diff'>" in html_preview
        assert "Execute command" in html_preview


class TestEnhancedApprovalManager:
    """Test enhanced approval manager functionality."""

    @pytest.mark.asyncio
    async def test_initialize_approval_manager(self, approval_manager):
        """Test approval manager initialization."""
        assert approval_manager.approval_workflow is not None
        assert approval_manager.enable_batch_approval is True
        assert approval_manager.max_batch_size == 10
        assert len(approval_manager.preview_generators) == 8

    @pytest.mark.asyncio
    async def test_generate_file_write_preview(self, approval_manager, temp_dir):
        """Test file write preview generation."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("original content")

        operation = Operation(
            type=OperationType.FILE_WRITE,
            description="Write to test file",
            data={"path": str(test_file), "content": "new content"},
        )

        preview = await approval_manager.generate_operation_preview(operation)

        assert preview.change_type == ChangeType.FILE_MODIFY
        assert preview.current_state == "original content"
        assert preview.proposed_state == "new content"
        assert preview.reversible is True
        assert preview.diff is not None
        assert "original content" in preview.diff
        assert "new content" in preview.diff

    @pytest.mark.asyncio
    async def test_generate_file_create_preview(self, approval_manager, temp_dir):
        """Test file creation preview generation."""
        new_file = temp_dir / "new_file.txt"

        operation = Operation(
            type=OperationType.FILE_WRITE,
            description="Create new file",
            data={"path": str(new_file), "content": "file content"},
        )

        preview = await approval_manager.generate_operation_preview(operation)

        assert preview.change_type == ChangeType.FILE_CREATE
        assert preview.current_state is None
        assert preview.proposed_state == "file content"
        assert preview.reversible is False

    @pytest.mark.asyncio
    async def test_generate_file_delete_preview(self, approval_manager, temp_dir):
        """Test file deletion preview generation."""
        # Create test file
        test_file = temp_dir / "delete_me.txt"
        test_file.write_text("content to be deleted")

        operation = Operation(
            type=OperationType.FILE_DELETE,
            description="Delete test file",
            data={"path": str(test_file)},
        )

        preview = await approval_manager.generate_operation_preview(operation)

        assert preview.change_type == ChangeType.FILE_DELETE
        assert preview.current_state == "content to be deleted"
        assert preview.proposed_state == ""
        assert preview.reversible is True
        assert "Medium" in preview.risk_assessment or "High" in preview.risk_assessment

    @pytest.mark.asyncio
    async def test_generate_command_preview(self, approval_manager):
        """Test command execution preview generation."""
        operation = Operation(
            type=OperationType.COMMAND_EXECUTE,
            description="Run ls command",
            data={
                "command": "ls",
                "args": ["-la", "/tmp"],
                "working_dir": "/home/user",
            },
        )

        preview = await approval_manager.generate_operation_preview(operation)

        assert preview.change_type == ChangeType.COMMAND_EXECUTE
        assert preview.description == "Execute command: ls -la /tmp"
        assert preview.metadata["command"] == "ls"
        assert preview.metadata["args"] == ["-la", "/tmp"]
        assert preview.metadata["working_dir"] == "/home/user"
        assert preview.reversible is False

    @pytest.mark.asyncio
    async def test_generate_web_request_preview(self, approval_manager):
        """Test web request preview generation."""
        operation = Operation(
            type=OperationType.WEB_REQUEST,
            description="GET request to API",
            data={
                "url": "https://api.example.com/data",
                "method": "GET",
                "headers": {"Authorization": "Bearer token"},
            },
        )

        preview = await approval_manager.generate_operation_preview(operation)

        assert preview.change_type == ChangeType.WEB_REQUEST
        assert preview.description == "GET request to: https://api.example.com/data"
        assert preview.metadata["url"] == "https://api.example.com/data"
        assert preview.metadata["method"] == "GET"
        assert preview.reversible is True  # GET requests are reversible

    @pytest.mark.asyncio
    async def test_generate_mcp_tool_preview(self, approval_manager):
        """Test MCP tool call preview generation."""
        operation = Operation(
            type=OperationType.MCP_TOOL_CALL,
            description="Call file reading tool",
            data={
                "tool_name": "file_read",
                "arguments": {"path": "/test/file.txt"},
            },
        )

        preview = await approval_manager.generate_operation_preview(operation)

        assert preview.change_type == ChangeType.MCP_TOOL_CALL
        assert preview.description == "Call MCP tool: file_read"
        assert preview.metadata["tool_name"] == "file_read"
        assert preview.metadata["arguments"] == {"path": "/test/file.txt"}
        assert preview.reversible is False

    @pytest.mark.asyncio
    async def test_risk_assessment_file_operations(self, approval_manager):
        """Test risk assessment for file operations."""
        # Test sensitive file
        sensitive_preview = await approval_manager._generate_file_write_preview(
            Operation(
                type=OperationType.FILE_WRITE,
                description="Write to .env file",
                data={"path": "/app/.env", "content": "SECRET=value"},
            )
        )
        assert (
            "sensitive file" in sensitive_preview.risk_assessment.lower()
            or "medium" in sensitive_preview.risk_assessment.lower()
            or "high" in sensitive_preview.risk_assessment.lower()
        )

        # Test normal file
        normal_preview = await approval_manager._generate_file_write_preview(
            Operation(
                type=OperationType.FILE_WRITE,
                description="Write to normal file",
                data={"path": "/tmp/normal.txt", "content": "content"},
            )
        )
        assert "low" in normal_preview.risk_assessment.lower()

    @pytest.mark.asyncio
    async def test_command_risk_assessment(self, approval_manager):
        """Test risk assessment for commands."""
        # Dangerous command
        dangerous_preview = await approval_manager._generate_command_preview(
            Operation(
                type=OperationType.COMMAND_EXECUTE,
                description="Remove files",
                data={"command": "rm", "args": ["-rf", "/tmp/*"]},
            )
        )
        assert (
            "high" in dangerous_preview.risk_assessment.lower()
            or "critical" in dangerous_preview.risk_assessment.lower()
        )

        # Safe command
        safe_preview = await approval_manager._generate_command_preview(
            Operation(
                type=OperationType.COMMAND_EXECUTE,
                description="List files",
                data={"command": "ls", "args": ["-la"]},
            )
        )
        assert "low" in safe_preview.risk_assessment.lower()

    def test_approval_statistics_empty(self, approval_manager):
        """Test approval statistics with no history."""
        stats = approval_manager.get_approval_statistics()
        assert stats["total_requests"] == 0

    def test_approval_statistics_with_history(self, approval_manager):
        """Test approval statistics with history."""
        # Add some fake history entries
        approval_manager.approval_history = [
            {
                "timestamp": "2023-01-01T12:00:00",
                "operation_type": "file_write",
                "approved": True,
                "risk_level": "low",
            },
            {
                "timestamp": "2023-01-01T12:01:00",
                "operation_type": "file_delete",
                "approved": False,
                "risk_level": "high",
            },
            {
                "timestamp": "2023-01-01T12:02:00",
                "operation_type": "file_write",
                "approved": True,
                "risk_level": "medium",
            },
        ]

        stats = approval_manager.get_approval_statistics()
        assert stats["total_requests"] == 3
        assert stats["approved_requests"] == 2
        assert stats["denied_requests"] == 1
        assert stats["approval_rate"] == 2 / 3
        assert "low" in stats["risk_level_distribution"]
        assert "file_write" in stats["operation_type_distribution"]
        assert stats["operation_type_distribution"]["file_write"] == 2

    def test_cleanup_expired_requests(self, approval_manager):
        """Test cleanup of expired requests."""
        from datetime import datetime, timedelta

        # Create expired batch request
        expired_batch = BatchApprovalRequest(
            operations=[], expires_at=datetime.now() - timedelta(minutes=30)
        )
        approval_manager.pending_batches[expired_batch.id] = expired_batch

        # Create non-expired batch request
        active_batch = BatchApprovalRequest(
            operations=[], expires_at=datetime.now() + timedelta(minutes=30)
        )
        approval_manager.pending_batches[active_batch.id] = active_batch

        cleaned_count = approval_manager.cleanup_expired_requests()

        assert cleaned_count == 1
        assert expired_batch.id not in approval_manager.pending_batches
        assert active_batch.id in approval_manager.pending_batches
        assert expired_batch.id in approval_manager.completed_batches


class TestBatchApprovalRequest:
    """Test batch approval request functionality."""

    def test_create_batch_request(self):
        """Test creating batch approval request."""
        operations = [
            Operation(
                type=OperationType.FILE_READ,
                description="Read file 1",
                data={},
            ),
            Operation(
                type=OperationType.FILE_READ,
                description="Read file 2",
                data={},
            ),
        ]

        batch = BatchApprovalRequest(operations=operations)

        assert len(batch.operations) == 2
        assert batch.id is not None
        assert batch.created_at is not None
        assert len(batch.approved_operations) == 0

    def test_batch_approval_summary(self):
        """Test batch approval summary."""
        operations = [
            Operation(type=OperationType.FILE_READ, description="Op 1", data={}),
            Operation(type=OperationType.FILE_WRITE, description="Op 2", data={}),
            Operation(type=OperationType.FILE_DELETE, description="Op 3", data={}),
        ]

        batch = BatchApprovalRequest(operations=operations)
        batch.approved_operations = {0, 2}  # Approve operations 0 and 2

        summary = batch.get_approval_summary()

        assert summary["total_operations"] == 3
        assert summary["approved_operations"] == 2
        assert summary["pending_operations"] == 1
        assert summary["approval_rate"] == 2 / 3
        assert summary["partially_approved"] is True
        assert summary["all_approved"] is False

    def test_batch_all_approved(self):
        """Test batch with all operations approved."""
        operations = [
            Operation(type=OperationType.FILE_READ, description="Op 1", data={}),
            Operation(type=OperationType.FILE_READ, description="Op 2", data={}),
        ]

        batch = BatchApprovalRequest(operations=operations)
        batch.approved_operations = {0, 1}  # Approve all operations

        summary = batch.get_approval_summary()

        assert summary["all_approved"] is True
        assert summary["partially_approved"] is False
        assert summary["approval_rate"] == 1.0

    def test_batch_expiry(self):
        """Test batch request expiry."""
        from datetime import datetime, timedelta

        # Create expired batch
        expired_batch = BatchApprovalRequest(
            operations=[], expires_at=datetime.now() - timedelta(minutes=1)
        )
        assert expired_batch.is_expired() is True

        # Create active batch
        active_batch = BatchApprovalRequest(
            operations=[], expires_at=datetime.now() + timedelta(minutes=30)
        )
        assert active_batch.is_expired() is False


class TestApprovalIntegration:
    """Test integration between approval components."""

    @pytest.mark.asyncio
    async def test_single_approval_workflow(self, approval_manager):
        """Test complete single approval workflow."""
        operation = Operation(
            type=OperationType.FILE_WRITE,
            description="Write test file",
            data={"path": "/tmp/test.txt", "content": "test content"},
            requires_approval=True,
        )

        # Mock the approval callback to approve
        async def mock_approval_callback(approval_data):
            assert "operation" in approval_data
            assert "preview" in approval_data
            assert "approval_request" in approval_data
            return True

        approval_manager.set_approval_callback(mock_approval_callback)

        # Test approval
        approved = await approval_manager.request_single_approval(operation)
        assert approved is True

        # Check that history was recorded
        assert len(approval_manager.approval_history) == 1
        history_entry = approval_manager.approval_history[0]
        assert history_entry["operation_type"] == "file_write"
        assert history_entry["approved"] is True

    @pytest.mark.asyncio
    async def test_single_approval_denial(self, approval_manager):
        """Test single approval denial."""
        operation = Operation(
            type=OperationType.FILE_DELETE,
            description="Delete important file",
            data={"path": "/important/file.txt"},
            requires_approval=True,
        )

        # Mock the approval callback to deny
        async def mock_denial_callback(approval_data):
            return False

        approval_manager.set_approval_callback(mock_denial_callback)

        # Test denial
        approved = await approval_manager.request_single_approval(operation)
        assert approved is False

        # Check that denial was recorded
        assert len(approval_manager.approval_history) == 1
        history_entry = approval_manager.approval_history[0]
        assert history_entry["approved"] is False

    @pytest.mark.asyncio
    async def test_batch_approval_workflow(self, approval_manager):
        """Test complete batch approval workflow."""
        operations = [
            Operation(
                type=OperationType.FILE_READ,
                description="Read file 1",
                data={},
            ),
            Operation(
                type=OperationType.FILE_WRITE,
                description="Write file 2",
                data={},
            ),
            Operation(
                type=OperationType.FILE_DELETE,
                description="Delete file 3",
                data={},
            ),
        ]

        # Mock batch approval callback
        async def mock_batch_callback(batch_request):
            # Approve first two operations
            return {"approved_indices": [0, 1]}

        approval_manager.set_batch_approval_callback(mock_batch_callback)

        # Test batch approval
        batch_request = await approval_manager.request_batch_approval(operations)

        assert batch_request.id is not None
        assert len(batch_request.operations) == 3
        assert len(batch_request.previews) == 3
        assert batch_request.approved_operations == {0, 1}

        summary = batch_request.get_approval_summary()
        assert summary["approved_operations"] == 2
        assert summary["pending_operations"] == 1

    @pytest.mark.asyncio
    async def test_low_risk_auto_approval(self, approval_manager):
        """Test automatic approval of low-risk operations."""
        # Create low-risk operation (file read)
        operation = Operation(
            type=OperationType.FILE_READ,
            description="Read configuration file",
            data={"path": "/config/app.conf"},
            requires_approval=True,
        )

        # Should be auto-approved due to low risk
        approved = await approval_manager.request_single_approval(operation)
        assert approved is True

        # Check that it was auto-approved
        assert len(approval_manager.approval_history) == 1
        history_entry = approval_manager.approval_history[0]
        assert history_entry["approved"] is True


class TestApprovalInterface:
    """Test approval interface functionality."""

    def test_interface_initialization(self, approval_interface):
        """Test approval interface initialization."""
        assert approval_interface.approval_manager is not None
        assert approval_interface.show_colors is True
        assert approval_interface.auto_show_diff is True
        assert approval_interface.max_diff_lines == 50

    def test_configure_interface(self, approval_interface):
        """Test interface configuration."""
        approval_interface.set_colors_enabled(False)
        approval_interface.set_auto_show_diff(False)
        approval_interface.set_max_diff_lines(20)

        assert approval_interface.show_colors is False
        assert approval_interface.auto_show_diff is False
        assert approval_interface.max_diff_lines == 20

    def test_colorize_text(self, approval_interface):
        """Test text colorization."""
        # With colors enabled
        approval_interface.set_colors_enabled(True)
        colored = approval_interface._colorize("test", "red")
        assert "\033[91m" in colored  # Red color code
        assert "test" in colored
        assert "\033[0m" in colored  # Reset code

        # With colors disabled
        approval_interface.set_colors_enabled(False)
        plain = approval_interface._colorize("test", "red")
        assert plain == "test"
        assert "\033[" not in plain

    def test_risk_color_mapping(self, approval_interface):
        """Test risk level color mapping."""
        from omnimancer.core.security.approval_workflow import RiskLevel

        assert approval_interface._get_risk_color(RiskLevel.LOW) == "green"
        assert approval_interface._get_risk_color(RiskLevel.MEDIUM) == "yellow"
        assert approval_interface._get_risk_color(RiskLevel.HIGH) == "red"
        assert approval_interface._get_risk_color(RiskLevel.CRITICAL) == "magenta"


@pytest.mark.asyncio
async def test_end_to_end_approval_flow(approval_manager, temp_dir):
    """Test complete end-to-end approval flow."""
    # Create a test file to modify
    test_file = temp_dir / "test_file.txt"
    test_file.write_text("original content\nline 2\nline 3")

    # Create operation
    operation = Operation(
        type=OperationType.FILE_WRITE,
        description="Modify test file",
        data={
            "path": str(test_file),
            "content": "modified content\nline 2\nline 3\nnew line 4",
        },
        requires_approval=True,
        reversible=True,
    )

    # Track approval callback calls
    callback_calls = []

    async def test_callback(approval_data):
        callback_calls.append(approval_data)

        # Verify all expected data is present
        assert "operation" in approval_data
        assert "preview" in approval_data
        assert "approval_request" in approval_data
        assert "risk_level" in approval_data

        # Verify preview has diff
        preview = approval_data["preview"]
        assert preview.change_type == ChangeType.FILE_MODIFY
        assert preview.current_state is not None
        assert preview.proposed_state is not None
        assert preview.diff is not None
        assert "original content" in preview.diff
        assert "modified content" in preview.diff
        assert "new line 4" in preview.diff

        return True  # Approve the operation

    approval_manager.set_approval_callback(test_callback)

    # Execute approval request
    approved = await approval_manager.request_single_approval(operation)

    # Verify results
    assert approved is True
    assert len(callback_calls) == 1
    assert len(approval_manager.approval_history) == 1

    # Verify history entry
    history_entry = approval_manager.approval_history[0]
    assert history_entry["operation_type"] == "file_write"
    assert history_entry["approved"] is True
    assert history_entry["preview_summary"]["change_type"] == "file_modify"
    assert history_entry["preview_summary"]["reversible"] is True

    # Verify statistics
    stats = approval_manager.get_approval_statistics()
    assert stats["total_requests"] == 1
    assert stats["approved_requests"] == 1
    assert stats["approval_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
