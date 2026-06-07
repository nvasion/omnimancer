"""Regression test: 'remember' approvals must auto-approve later operations.

The agent engine drives approvals through the manager callback
(_handle_single_approval); the auto-approval check must therefore run inside
that callback, not only in request_approval_for_operation.
"""

from unittest.mock import AsyncMock, MagicMock

from omnimancer.cli.approval_integration import CLIApprovalIntegration
from omnimancer.core.agent.types import Operation, OperationType
from omnimancer.core.security.permission_controller import PermissionController


def _integration():
    integration = CLIApprovalIntegration(
        permission_controller=PermissionController(),
        console=MagicMock(),
        enable_auto_approval=True,
    )
    # Fail loudly if the interactive prompt is ever reached.
    integration.prompt_handler.prompt_for_approval = AsyncMock(
        side_effect=AssertionError("should not prompt when remembered")
    )
    return integration


def _write_op(path="/tmp/proj/hello.txt"):
    return Operation(
        type=OperationType.FILE_WRITE,
        description=f"Write file: {path}",
        data={"path": path},
        requires_approval=True,
    )


class TestRememberAutoApproval:
    async def test_remembered_write_is_auto_approved(self):
        integration = _integration()
        op = _write_op()

        # Simulate the user choosing "remember".
        from omnimancer.cli.approval_integration import ApprovalRequest

        await integration._store_approval_pattern(
            op,
            ApprovalRequest(operation_type=op.type.value, description=op.description),
        )

        # A later, similar write must not prompt.
        approved, cancelled = await integration._handle_single_approval(_write_op())
        assert approved is True
        assert cancelled is False
        integration.prompt_handler.prompt_for_approval.assert_not_called()

    async def test_unremembered_write_still_prompts(self):
        integration = _integration()
        # Make the prompt return a deny decision instead of asserting.
        decision = MagicMock()
        decision.should_remember = False
        decision.is_approved = False
        from omnimancer.cli.approval_prompt import ApprovalDecisionType

        decision.decision = ApprovalDecisionType.DENIED
        integration.prompt_handler.prompt_for_approval = AsyncMock(
            return_value=decision
        )

        approved, _ = await integration._handle_single_approval(_write_op())
        assert approved is False
        integration.prompt_handler.prompt_for_approval.assert_called_once()
