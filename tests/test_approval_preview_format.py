"""
Test that approval handler properly creates ChangePreview objects from Operations.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.approval_integration import CLIApprovalIntegration
from omnimancer.core.agent.approval_manager import ChangePreview, ChangeType
from omnimancer.core.agent.types import Operation, OperationType


@pytest.mark.asyncio
async def test_handle_single_approval_creates_change_preview():
    """Test that _handle_single_approval creates ChangePreview from Operation."""
    # Create mock prompt handler
    mock_prompt_handler = MagicMock()
    mock_prompt_handler.prompt_for_approval = AsyncMock(
        return_value=MagicMock(is_approved=True, should_remember=False)
    )

    # Create CLI approval integration and mock its prompt_handler
    cli_integration = CLIApprovalIntegration(
        approval_timeout_seconds=30,
    )
    cli_integration.prompt_handler = mock_prompt_handler

    # Create test operation
    operation = Operation(
        type=OperationType.COMMAND_EXECUTE,
        description="Install pdf2docx package",
        data={"command": "pip install pdf2docx"},
        requires_approval=True,
        reversible=False,
        preview="pip install pdf2docx",
    )

    # Call handler
    result = await cli_integration._handle_single_approval(operation)

    # Verify prompt_for_approval was called
    mock_prompt_handler.prompt_for_approval.assert_called_once()

    # Get the arguments passed to prompt_for_approval
    call_args = mock_prompt_handler.prompt_for_approval.call_args
    call_args[0][0]
    change_preview = call_args[0][1]

    # Verify change_preview is a ChangePreview object, not a string
    assert isinstance(change_preview, ChangePreview)
    assert change_preview.change_type == ChangeType.COMMAND_EXECUTE
    assert change_preview.description == operation.description
    assert change_preview.proposed_state == operation.preview
    assert change_preview.reversible == operation.reversible
    assert change_preview.metadata == operation.data

    # Verify result - _handle_single_approval returns (approved, was_cancelled)
    assert result == (True, False)


@pytest.mark.asyncio
async def test_handle_single_approval_maps_operation_types():
    """Test that all OperationType values map correctly to ChangeType."""
    # Create mock prompt handler
    mock_prompt_handler = MagicMock()
    mock_prompt_handler.prompt_for_approval = AsyncMock(
        return_value=MagicMock(is_approved=True, should_remember=False)
    )

    # Create CLI approval integration and mock its prompt_handler
    cli_integration = CLIApprovalIntegration(
        approval_timeout_seconds=30,
    )
    cli_integration.prompt_handler = mock_prompt_handler

    # Test cases mapping OperationType to expected ChangeType
    test_cases = [
        # No FILE_READ in ChangeType, maps to FILE_MODIFY
        (OperationType.FILE_READ, ChangeType.FILE_MODIFY),
        # Rich preview: a write to a nonexistent path is a create
        # (writes to existing files map to FILE_MODIFY — covered separately)
        (OperationType.FILE_WRITE, ChangeType.FILE_CREATE),
        (OperationType.FILE_DELETE, ChangeType.FILE_DELETE),
        (OperationType.DIRECTORY_CREATE, ChangeType.DIRECTORY_CREATE),
        (OperationType.DIRECTORY_DELETE, ChangeType.DIRECTORY_DELETE),
        (OperationType.COMMAND_EXECUTE, ChangeType.COMMAND_EXECUTE),
        (OperationType.WEB_REQUEST, ChangeType.WEB_REQUEST),
        (OperationType.MCP_TOOL_CALL, ChangeType.MCP_TOOL_CALL),
    ]

    for operation_type, expected_change_type in test_cases:
        # Create test operation
        operation = Operation(
            type=operation_type,
            description=f"Test {operation_type.value}",
            data={"test": "data"},
            requires_approval=True,
            preview=f"Preview for {operation_type.value}",
        )

        # Call handler
        await cli_integration._handle_single_approval(operation)

        # Get the change_preview that was passed
        call_args = mock_prompt_handler.prompt_for_approval.call_args
        change_preview = call_args[0][1]

        # Verify change_type matches expected
        msg = (
            f"Failed for {operation_type.value}: "
            f"expected {expected_change_type.value}, "
            f"got {change_preview.change_type.value}"
        )
        assert change_preview.change_type == expected_change_type, msg


def _integration_with_mock_prompt():
    mock_prompt_handler = MagicMock()
    mock_prompt_handler.prompt_for_approval = AsyncMock(
        return_value=MagicMock(is_approved=True, should_remember=False)
    )
    cli_integration = CLIApprovalIntegration(approval_timeout_seconds=30)
    cli_integration.prompt_handler = mock_prompt_handler
    return cli_integration, mock_prompt_handler


@pytest.mark.asyncio
async def test_file_write_to_existing_file_shows_real_diff(tmp_path):
    """Approving an edit must show an actual diff of the change.

    The preview used to set proposed_state to the short *string* preview
    ("Content preview: <100 chars>...") and never generated a diff, so the
    user approved writes blind.
    """
    target = tmp_path / "settings.tsx"
    target.write_text("line one\nline two\nline three\n")
    new_content = "line one\nline CHANGED\nline three\n"

    cli_integration, mock_prompt_handler = _integration_with_mock_prompt()
    operation = Operation(
        type=OperationType.FILE_WRITE,
        description=f"Edit file: {target}",
        data={"path": str(target), "content": new_content},
        requires_approval=True,
        preview="Update file: ...\nContent preview: line one...",
    )

    result = await cli_integration._handle_single_approval(operation)

    assert result == (True, False)
    change_preview = mock_prompt_handler.prompt_for_approval.call_args[0][1]
    assert change_preview.change_type == ChangeType.FILE_MODIFY
    # Real states, not the operation's string preview
    assert change_preview.proposed_state == new_content
    assert change_preview.current_state == "line one\nline two\nline three\n"
    # A unified diff of the actual change
    assert change_preview.diff
    assert "-line two" in change_preview.diff
    assert "+line CHANGED" in change_preview.diff
    # Existing file content means the change is recoverable
    assert change_preview.reversible is True


@pytest.mark.asyncio
async def test_file_write_to_new_file_is_create(tmp_path):
    """Writing a brand-new file previews as a create with full content."""
    target = tmp_path / "brand_new.py"
    new_content = "print('hi')\n"

    cli_integration, mock_prompt_handler = _integration_with_mock_prompt()
    operation = Operation(
        type=OperationType.FILE_WRITE,
        description=f"Write file: {target}",
        data={"path": str(target), "content": new_content},
        requires_approval=True,
        preview="Create file: ...",
    )

    await cli_integration._handle_single_approval(operation)

    change_preview = mock_prompt_handler.prompt_for_approval.call_args[0][1]
    assert change_preview.change_type == ChangeType.FILE_CREATE
    assert change_preview.proposed_state == new_content


@pytest.mark.asyncio
async def test_approval_formatter_receives_valid_preview():
    """Test approval formatter receives a ChangePreview with change_type."""
    from omnimancer.cli.approval_formatter import CLIApprovalFormatter
    from omnimancer.core.security.approval_workflow import ApprovalRequest, RiskLevel

    # Create approval formatter
    formatter = CLIApprovalFormatter()

    # Create test data
    approval_request = ApprovalRequest(
        operation_type="command_execute",
        description="Install package",
        risk_level=RiskLevel.MEDIUM,
    )

    change_preview = ChangePreview(
        change_type=ChangeType.COMMAND_EXECUTE,
        description="Install pdf2docx",
        proposed_state="pip install pdf2docx",
        reversible=False,
    )

    # Format header (this is where the error was occurring)
    header = formatter._format_header(approval_request, change_preview)

    # Verify it didn't crash and has content
    assert header is not None
    assert "Approval Required" in header.title


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
