"""
Tests for the Approval Dialog System.

This module tests the structured approval dialog layout,
approval context handling, and dialog interaction logic.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from omnimancer.core.agent.approval_dialog import (
    ApprovalDialog, DialogState, DialogSection, DialogOptions,
    create_approval_dialog, show_quick_approval
)
from omnimancer.core.agent.approval_context import (
    ApprovalContext, OperationDetails, ApprovalDecision, 
    FileChange, FileOperationType, SecurityFlags, OperationStatus,
    create_file_operation_context, create_command_execution_context,
    create_web_request_context
)
from omnimancer.core.agent.rich_renderer import RiskLevel, create_renderer
from omnimancer.core.agent.diff_renderer import DiffType, create_diff_renderer


class TestDialogOptions:
    """Test DialogOptions configuration."""
    
    def test_default_options(self):
        """Test default dialog options."""
        options = DialogOptions()
        
        assert options.show_diff is True
        assert options.show_risk_assessment is True
        assert options.show_file_tree is True
        assert options.show_operation_details is True
        assert options.max_height == 40
        assert options.max_width == 120
        assert options.compact_mode is False
        assert options.auto_scroll is True
        assert options.enable_shortcuts is True
        assert options.timeout_seconds is None
        assert options.diff_type == DiffType.UNIFIED
        assert options.syntax_highlighting is True
        assert options.show_line_numbers is True
    
    def test_custom_options(self):
        """Test custom dialog options."""
        options = DialogOptions(
            show_diff=False,
            compact_mode=True,
            max_height=20,
            timeout_seconds=30,
            diff_type=DiffType.SIDE_BY_SIDE
        )
        
        assert options.show_diff is False
        assert options.compact_mode is True
        assert options.max_height == 20
        assert options.timeout_seconds == 30
        assert options.diff_type == DiffType.SIDE_BY_SIDE


class TestFileChange:
    """Test FileChange dataclass."""
    
    def test_file_change_creation(self):
        """Test creating file changes."""
        change = FileChange(
            path=Path("/test/file.py"),
            operation=FileOperationType.MODIFY,
            content_preview="def test():\n    pass",
            size_bytes=1024
        )
        
        assert change.path == Path("/test/file.py")
        assert change.operation == FileOperationType.MODIFY
        assert change.content_preview == "def test():\n    pass"
        assert change.size_bytes == 1024
        assert change.relative_path == "/test/file.py"


class TestSecurityFlags:
    """Test SecurityFlags functionality."""
    
    def test_default_security_flags(self):
        """Test default security flags."""
        flags = SecurityFlags()
        
        assert flags.requires_sudo is False
        assert flags.modifies_system_files is False
        assert flags.accesses_network is False
        assert flags.executes_code is False
        assert flags.modifies_permissions is False
        assert flags.creates_processes is False
        assert flags.accesses_sensitive_data is False
    
    def test_security_flags_with_values(self):
        """Test security flags with specific values."""
        flags = SecurityFlags(
            requires_sudo=True,
            accesses_network=True,
            executes_code=True
        )
        
        assert flags.requires_sudo is True
        assert flags.accesses_network is True
        assert flags.executes_code is True
        assert flags.modifies_system_files is False
    
    def test_get_risk_factors(self):
        """Test extracting risk factors from security flags."""
        flags = SecurityFlags(
            requires_sudo=True,
            modifies_system_files=True,
            accesses_network=True
        )
        
        factors = flags.get_risk_factors()
        
        assert "Requires elevated privileges" in factors
        assert "Modifies system files" in factors
        assert "Makes network requests" in factors
        assert len(factors) == 3
    
    def test_empty_risk_factors(self):
        """Test risk factors with no flags set."""
        flags = SecurityFlags()
        factors = flags.get_risk_factors()
        
        assert factors == []


class TestOperationDetails:
    """Test OperationDetails functionality."""
    
    def test_basic_operation_details(self):
        """Test creating basic operation details."""
        details = OperationDetails(
            operation_type="file_write",
            target="/test/file.txt",
            description="Write test file"
        )
        
        assert details.operation_type == "file_write"
        assert details.target == "/test/file.txt"
        assert details.description == "Write test file"
        assert details.risk_level == RiskLevel.MEDIUM
        assert details.files_affected == []
        assert details.file_changes == []
    
    def test_operation_with_files(self):
        """Test operation details with file operations."""
        details = OperationDetails(
            operation_type="file_modify",
            files_affected=["/test/file1.py", "/test/file2.py"]
        )
        
        # Files should be converted to Path objects
        assert all(isinstance(f, Path) for f in details.files_affected)
        assert len(details.files_affected) == 2
    
    def test_add_file_change(self):
        """Test adding file changes to operation."""
        details = OperationDetails(operation_type="file_edit")
        
        details.add_file_change(
            "/test/new_file.py",
            FileOperationType.CREATE,
            "print('hello')"
        )
        
        assert len(details.file_changes) == 1
        assert len(details.files_affected) == 1
        
        change = details.file_changes[0]
        assert change.path == Path("/test/new_file.py")
        assert change.operation == FileOperationType.CREATE
        assert change.content_preview == "print('hello')"
    
    def test_get_total_files_affected(self):
        """Test counting total affected files."""
        details = OperationDetails(
            operation_type="batch_edit",
            files_affected=["/test/file1.py", "/test/file2.py"]
        )
        
        details.add_file_change("/test/file3.py", FileOperationType.CREATE)
        # file1.py is in both lists (should count once)
        details.add_file_change("/test/file1.py", FileOperationType.MODIFY)
        
        total = details.get_total_files_affected()
        assert total == 3  # file1.py, file2.py, file3.py
    
    def test_get_operation_summary(self):
        """Test generating operation summaries."""
        # With description
        details1 = OperationDetails(
            operation_type="file_write",
            description="Custom description"
        )
        assert details1.get_operation_summary() == "Custom description"
        
        # With target
        details2 = OperationDetails(
            operation_type="file_delete",
            target="/test/file.txt"
        )
        assert details2.get_operation_summary() == "file_delete on /test/file.txt"
        
        # With multiple files
        details3 = OperationDetails(
            operation_type="batch_process",
            files_affected=[Path("/test/file1.py"), Path("/test/file2.py")]
        )
        assert details3.get_operation_summary() == "batch_process on 2 files"
        
        # With single file
        details4 = OperationDetails(
            operation_type="file_read",
            files_affected=[Path("/test/single.py")]
        )
        assert details4.get_operation_summary() == "file_read on /test/single.py"


class TestApprovalDecision:
    """Test ApprovalDecision functionality."""
    
    def test_basic_approval_decision(self):
        """Test creating approval decisions."""
        decision = ApprovalDecision(
            approved=True,
            reason="Operation is safe"
        )
        
        assert decision.approved is True
        assert decision.reason == "Operation is safe"
        assert isinstance(decision.timestamp, datetime)
        assert decision.user_id is None
        assert decision.additional_data == {}
    
    def test_decision_serialization(self):
        """Test converting decisions to/from dictionaries."""
        timestamp = datetime.now()
        decision = ApprovalDecision(
            approved=False,
            reason="High risk operation",
            timestamp=timestamp,
            user_id="test_user",
            additional_data={"risk_score": 8.5}
        )
        
        # Test to_dict
        data = decision.to_dict()
        
        assert data['approved'] is False
        assert data['reason'] == "High risk operation"
        assert data['timestamp'] == timestamp.isoformat()
        assert data['user_id'] == "test_user"
        assert data['additional_data'] == {"risk_score": 8.5}
        
        # Test from_dict
        decision2 = ApprovalDecision.from_dict(data)
        
        assert decision2.approved == decision.approved
        assert decision2.reason == decision.reason
        assert decision2.timestamp == decision.timestamp
        assert decision2.user_id == decision.user_id
        assert decision2.additional_data == decision.additional_data


class TestApprovalContext:
    """Test ApprovalContext functionality."""
    
    def test_basic_approval_context(self):
        """Test creating approval contexts."""
        operation_details = OperationDetails(
            operation_type="test_operation",
            description="Test operation"
        )
        
        context = ApprovalContext(
            agent_name="Test Agent",
            operation_details=operation_details
        )
        
        assert context.agent_name == "Test Agent"
        assert context.operation_details == operation_details
        assert context.status == OperationStatus.PENDING
        assert context.decision is None
        assert context.context_id.startswith("ctx_")
    
    def test_conversation_history(self):
        """Test managing conversation history."""
        context = ApprovalContext()
        
        context.add_conversation_entry("User requested file operation")
        context.add_conversation_entry("Agent analyzing risk")
        
        assert len(context.conversation_history) == 2
        assert "User requested file operation" in context.conversation_history[0]
        assert "Agent analyzing risk" in context.conversation_history[1]
        # Should have timestamps
        assert "[" in context.conversation_history[0]
        assert "]" in context.conversation_history[0]
    
    def test_set_decision(self):
        """Test setting approval decisions."""
        context = ApprovalContext()
        decision = ApprovalDecision(approved=True, reason="Approved by user")
        
        context.set_decision(decision)
        
        assert context.decision == decision
        assert context.status == OperationStatus.APPROVED
        
        # Test denial
        context2 = ApprovalContext()
        denial = ApprovalDecision(approved=False, reason="Too risky")
        
        context2.set_decision(denial)
        
        assert context2.decision == denial
        assert context2.status == OperationStatus.DENIED
    
    def test_is_completed(self):
        """Test completion status checking."""
        context = ApprovalContext()
        
        assert context.is_completed() is False
        
        # Test approved
        context.set_decision(ApprovalDecision(approved=True))
        assert context.is_completed() is True
        
        # Test denied
        context2 = ApprovalContext()
        context2.set_decision(ApprovalDecision(approved=False))
        assert context2.is_completed() is True
        
        # Test cancelled
        context3 = ApprovalContext()
        context3.status = OperationStatus.CANCELLED
        assert context3.is_completed() is True
    
    def test_context_summary(self):
        """Test generating context summaries."""
        operation_details = OperationDetails(
            operation_type="file_write",
            risk_level=RiskLevel.HIGH,
            files_affected=[Path("/test/file.py")]
        )
        
        context = ApprovalContext(
            agent_name="Test Agent",
            operation_details=operation_details
        )
        
        summary = context.get_context_summary()
        
        assert summary['agent_name'] == "Test Agent"
        assert summary['operation_type'] == "file_write"
        assert summary['risk_level'] == str(RiskLevel.HIGH)
        assert summary['files_affected'] == 1
        assert summary['status'] == OperationStatus.PENDING.value
        assert summary['approved'] is None


class TestApprovalDialog:
    """Test ApprovalDialog functionality."""
    
    def test_dialog_initialization(self):
        """Test dialog initialization."""
        dialog = ApprovalDialog()
        
        assert dialog.renderer is not None
        assert dialog.diff_renderer is not None
        assert dialog.console is not None
        assert dialog.options is not None
        assert dialog.state == DialogState.INITIALIZING
        assert dialog.current_context is None
        assert dialog.decision is None
        assert isinstance(dialog.layout, Layout)
    
    def test_dialog_with_custom_components(self):
        """Test dialog with custom components."""
        renderer = create_renderer()
        console = Console()
        options = DialogOptions(compact_mode=True)
        
        dialog = ApprovalDialog(
            renderer=renderer,
            console=console,
            options=options
        )
        
        assert dialog.renderer == renderer
        assert dialog.console == console
        assert dialog.options.compact_mode is True
    
    @pytest.mark.asyncio
    @patch('omnimancer.core.agent.approval_dialog.InteractiveInputHandler.handle_input_loop')
    @patch('omnimancer.core.agent.approval_dialog.Live')
    async def test_show_approval_dialog_mock(self, mock_live, mock_input_loop):
        """Test showing approval dialog with mocked input."""
        # Mock the input handler to return approval
        from omnimancer.core.agent.input_handler import KeyAction
        mock_input_loop.return_value = KeyAction.APPROVE
        
        # Mock the Live context manager
        mock_live_instance = Mock()
        mock_live.return_value.__enter__ = Mock(return_value=mock_live_instance)
        mock_live.return_value.__exit__ = Mock(return_value=None)
        
        # Create test context
        operation_details = OperationDetails(
            operation_type="file_write",
            target="/test/file.py",
            description="Write test file"
        )
        
        context = ApprovalContext(
            agent_name="Test Agent",
            operation_details=operation_details,
            diff_content="@@ -1,1 +1,2 @@\n print('hello')\n+print('world')"
        )
        
        # Create dialog - no need to mock console anymore
        dialog = ApprovalDialog()
        
        # Show dialog
        decision = await dialog.show_approval_dialog(context)
        
        # Verify results
        assert isinstance(decision, ApprovalDecision)
        assert decision.approved is True  # Mock returns approval
        assert dialog.state == DialogState.COMPLETED
        assert dialog.current_context == context
        assert dialog.decision == decision
        assert decision.additional_data["input_action"] == "approve"
    
    def test_dialog_state_management(self):
        """Test dialog state management."""
        dialog = ApprovalDialog()
        
        assert dialog.get_current_state() == DialogState.INITIALIZING
        
        dialog.state = DialogState.DISPLAYING
        assert dialog.get_current_state() == DialogState.DISPLAYING
        
        decision = ApprovalDecision(approved=True)
        dialog.decision = decision
        assert dialog.get_decision() == decision
    
    def test_render_header(self):
        """Test header rendering."""
        dialog = ApprovalDialog()
        context = ApprovalContext(
            agent_name="Test Agent",
            timestamp=datetime.now()
        )
        
        header = dialog._render_header(context)
        
        assert isinstance(header, Panel)
        assert "Operation Approval Required" in str(header.title)
    
    def test_render_operation_summary(self):
        """Test operation summary rendering."""
        dialog = ApprovalDialog()
        operation = OperationDetails(
            operation_type="file_write",
            target="/test/file.py",
            risk_level=RiskLevel.MEDIUM,
            description="Write test file"
        )
        
        summary = dialog._render_operation_summary(operation)
        
        assert isinstance(summary, Panel)
        assert "Operation Summary" in str(summary.title)
    
    def test_render_risk_assessment(self):
        """Test risk assessment rendering."""
        dialog = ApprovalDialog()
        operation = OperationDetails(
            operation_type="system_command",
            risk_level=RiskLevel.HIGH,
            risk_factors=["Executes system command", "Modifies critical files"]
        )
        
        risk_panel = dialog._render_risk_assessment(operation)
        
        assert isinstance(risk_panel, Panel)
        assert "Risk Assessment" in str(risk_panel.title)
    
    def test_render_diff_panel(self):
        """Test diff panel rendering."""
        dialog = ApprovalDialog()
        context = ApprovalContext(
            diff_content="@@ -1,2 +1,3 @@\n print('hello')\n+print('world')\n return True"
        )
        
        diff_panel = dialog._render_diff_panel(context)
        
        assert isinstance(diff_panel, Panel)
        assert "File Changes" in str(diff_panel.title)
    
    def test_render_diff_panel_no_content(self):
        """Test diff panel with no content."""
        dialog = ApprovalDialog()
        context = ApprovalContext()  # No diff_content
        
        diff_panel = dialog._render_diff_panel(context)
        
        assert isinstance(diff_panel, Panel)
        assert "No file changes" in str(diff_panel.renderable)
    
    def test_render_controls(self):
        """Test controls rendering."""
        dialog = ApprovalDialog()
        
        controls = dialog._render_controls()
        
        assert isinstance(controls, Panel)
        assert "Controls" in str(controls.title)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_approval_dialog(self):
        """Test dialog creation utility."""
        dialog = create_approval_dialog()
        
        assert isinstance(dialog, ApprovalDialog)
        assert dialog.renderer is not None
        assert dialog.options is not None
    
    def test_create_approval_dialog_with_options(self):
        """Test dialog creation with custom options."""
        options = DialogOptions(compact_mode=True)
        dialog = create_approval_dialog(options=options)
        
        assert dialog.options.compact_mode is True
    
    @pytest.mark.asyncio
    async def test_show_quick_approval(self):
        """Test quick approval utility."""
        with patch('omnimancer.core.agent.approval_dialog.ApprovalDialog.show_approval_dialog') as mock_show:
            # Mock the dialog to return approval
            mock_show.return_value = ApprovalDecision(approved=True, reason="Quick approval")
            
            result = await show_quick_approval(
                operation_type="file_read",
                target="/test/file.py",
                risk_level=RiskLevel.LOW
            )
            
            assert result is True
            mock_show.assert_called_once()


class TestContextCreationUtilities:
    """Test context creation utility functions."""
    
    def test_create_file_operation_context(self):
        """Test file operation context creation."""
        context = create_file_operation_context(
            operation_type="file_write",
            file_path="/test/new_file.py",
            content="print('hello world')",
            agent_name="Test Agent"
        )
        
        assert context.agent_name == "Test Agent"
        assert context.operation_details.operation_type == "file_write"
        assert context.operation_details.target == "/test/new_file.py"
        assert len(context.operation_details.files_affected) == 1
        assert len(context.operation_details.file_changes) == 1
        assert context.preview_content == "print('hello world')"
        
        # Check file change details
        file_change = context.operation_details.file_changes[0]
        assert file_change.operation == FileOperationType.WRITE
        assert file_change.path == Path("/test/new_file.py")
    
    def test_create_command_execution_context(self):
        """Test command execution context creation."""
        context = create_command_execution_context(
            command="python",
            arguments=["script.py", "--verbose"],
            working_directory="/test/dir",
            agent_name="Command Agent"
        )
        
        assert context.agent_name == "Command Agent"
        assert context.operation_details.operation_type == "command_execute"
        assert context.operation_details.command == "python"
        assert context.operation_details.arguments == ["script.py", "--verbose"]
        assert context.operation_details.working_directory == "/test/dir"
        assert context.operation_details.target == "python"
    
    def test_create_web_request_context(self):
        """Test web request context creation."""
        headers = {"User-Agent": "Omnimancer/1.0"}
        context = create_web_request_context(
            url="https://api.example.com/data",
            method="POST",
            headers=headers,
            agent_name="Web Agent"
        )
        
        assert context.agent_name == "Web Agent"
        assert context.operation_details.operation_type == "web_request"
        assert context.operation_details.target == "https://api.example.com/data"
        assert context.operation_details.metadata['method'] == "POST"
        assert context.operation_details.metadata['headers'] == headers
        assert context.operation_details.security_flags.accesses_network is True
    
    def test_command_security_assessment(self):
        """Test security assessment for commands."""
        # Test sudo command
        context1 = create_command_execution_context("sudo", ["rm", "/etc/config"])
        assert context1.operation_details.security_flags.requires_sudo is True
        
        # Test network command
        context2 = create_command_execution_context("curl", ["https://example.com"])
        assert context2.operation_details.security_flags.accesses_network is True
        
        # Test code execution
        context3 = create_command_execution_context("python", ["-c", "print('hello')"])
        assert context3.operation_details.security_flags.executes_code is True
        
        # Test system file modification
        context4 = create_command_execution_context("rm", ["/etc/hosts"])
        assert context4.operation_details.security_flags.modifies_system_files is True
    
    def test_web_request_security_assessment(self):
        """Test security assessment for web requests."""
        # Test sensitive URL
        context1 = create_web_request_context("https://admin.example.com/api")
        assert context1.operation_details.security_flags.accesses_sensitive_data is True
        
        # Test regular URL
        context2 = create_web_request_context("https://httpbin.org/get")
        assert context2.operation_details.security_flags.accesses_sensitive_data is False
        assert context2.operation_details.security_flags.accesses_network is True


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_dialog_with_keyboard_interrupt(self):
        """Test dialog handling keyboard interrupt."""
        dialog = ApprovalDialog()
        context = ApprovalContext()
        
        with patch.object(dialog, '_wait_for_decision', side_effect=KeyboardInterrupt):
            decision = await dialog.show_approval_dialog(context)
            
            assert decision.approved is False
            assert "cancelled by user" in decision.reason
            assert dialog.state == DialogState.CANCELLED
    
    @pytest.mark.asyncio
    async def test_dialog_with_exception(self):
        """Test dialog handling exceptions."""
        dialog = ApprovalDialog()
        context = ApprovalContext()
        
        with patch.object(dialog, '_render_header', side_effect=Exception("Test error")):
            decision = await dialog.show_approval_dialog(context)
            
            assert decision.approved is False
            assert "Dialog error" in decision.reason
            assert dialog.state == DialogState.CANCELLED
    
    def test_empty_operation_details(self):
        """Test operation details with minimal data."""
        details = OperationDetails(operation_type="unknown")
        
        summary = details.get_operation_summary()
        assert summary == "unknown"
        
        assert details.get_total_files_affected() == 0
    
    def test_context_with_no_diff_content(self):
        """Test context without diff content."""
        context = ApprovalContext()
        dialog = ApprovalDialog()
        
        diff_panel = dialog._render_diff_panel(context)
        
        assert isinstance(diff_panel, Panel)
        assert "No file changes" in str(diff_panel.renderable)
    
    def test_malformed_context_data(self):
        """Test handling malformed context data."""
        # Test with None values
        details = OperationDetails(
            operation_type="test",
            target=None,
            description=None,
            files_affected=[]
        )
        
        summary = details.get_operation_summary()
        assert summary == "test"  # Should handle None gracefully
    
    def test_large_file_list(self):
        """Test handling large numbers of files."""
        files = [Path(f"/test/file_{i}.py") for i in range(100)]
        details = OperationDetails(
            operation_type="batch_process",
            files_affected=files
        )
        
        # Should handle large lists without issues
        total = details.get_total_files_affected()
        assert total == 100
        
        summary = details.get_operation_summary()
        assert "100 files" in summary