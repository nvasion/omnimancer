"""
Test that approval handler properly creates ChangePreview objects from Operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

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
    approval_request = call_args[0][0]
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
        (OperationType.FILE_READ, ChangeType.FILE_MODIFY),  # No FILE_READ in ChangeType
        (OperationType.FILE_WRITE, ChangeType.FILE_MODIFY),  # Maps to FILE_MODIFY
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
        assert (
            change_preview.change_type == expected_change_type
        ), f"Failed for {operation_type.value}: expected {expected_change_type.value}, got {change_preview.change_type.value}"


@pytest.mark.asyncio
async def test_approval_formatter_receives_valid_preview():
    """Test that approval formatter receives a ChangePreview object with change_type attribute."""
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
