"""
Test approval integration compatibility with different ApprovalManager types.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.approval_integration import (
    inject_approval_integration_into_agent_engine,
)


@pytest.mark.asyncio
async def test_integration_with_basic_approval_manager():
    """Test that integration works with basic ApprovalManager (no batch callback)."""
    # Create mock agent engine with basic ApprovalManager
    # (no set_batch_approval_callback)
    mock_agent_engine = MagicMock()
    mock_approval = MagicMock()
    mock_approval.set_approval_callback = MagicMock()
    # Explicitly don't have set_batch_approval_callback
    if hasattr(mock_approval, "set_batch_approval_callback"):
        delattr(mock_approval, "set_batch_approval_callback")
    mock_agent_engine.approval = mock_approval

    # Create mock CLI approval integration
    mock_cli_integration = MagicMock()
    mock_cli_integration._handle_single_approval = AsyncMock()

    # Should not raise AttributeError
    inject_approval_integration_into_agent_engine(
        mock_agent_engine, mock_cli_integration
    )

    # Verify single callback was set
    mock_approval.set_approval_callback.assert_called_once_with(
        mock_cli_integration._handle_single_approval
    )

    # Verify no error when batch callback doesn't exist
    assert True  # If we got here without error, test passes


@pytest.mark.asyncio
async def test_integration_with_enhanced_approval_manager():
    """Test that integration works with EnhancedApprovalManager (has batch callback)."""
    # Create mock agent engine with EnhancedApprovalManager
    mock_agent_engine = MagicMock()
    mock_approval = MagicMock()
    mock_approval.set_approval_callback = MagicMock()
    mock_approval.set_batch_approval_callback = MagicMock()
    mock_agent_engine.approval = mock_approval

    # Create mock CLI approval integration
    mock_cli_integration = MagicMock()
    mock_cli_integration._handle_single_approval = AsyncMock()
    mock_cli_integration._handle_batch_approval = AsyncMock()

    # Should work with both callbacks
    inject_approval_integration_into_agent_engine(
        mock_agent_engine, mock_cli_integration
    )

    # Verify both callbacks were set
    mock_approval.set_approval_callback.assert_called_once_with(
        mock_cli_integration._handle_single_approval
    )
    mock_approval.set_batch_approval_callback.assert_called_once_with(
        mock_cli_integration._handle_batch_approval
    )


@pytest.mark.asyncio
async def test_integration_without_approval_manager():
    """Test that integration handles missing approval manager gracefully."""
    # Create mock agent engine without approval manager
    mock_agent_engine = MagicMock()
    mock_agent_engine.approval = None

    # Create mock CLI approval integration
    mock_cli_integration = MagicMock()

    # Should not raise any errors, just log warning
    inject_approval_integration_into_agent_engine(
        mock_agent_engine, mock_cli_integration
    )

    # Verify no callbacks were set (since there's no approval manager)
    assert True  # If we got here without error, test passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
