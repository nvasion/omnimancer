"""Session approval modes: /accept normal → accept-edits → accept-all.

Seated in ApprovalIntegration._check_auto_approval AFTER the
_force_prompt check, so permission-rule ASK (and DENY, which never
reaches the approval workflow) always beat the session mode.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.approval_integration import ApprovalMode, CLIApprovalIntegration
from omnimancer.core.agent.types import Operation, OperationType


@pytest.fixture
def integration():
    permission_controller = MagicMock()
    permission_controller.check_operation_permission = AsyncMock(return_value=False)
    return CLIApprovalIntegration(
        approval_manager=MagicMock(),
        permission_controller=permission_controller,
        console=MagicMock(),
    )


def _operation(op_type, data=None):
    return Operation(
        type=op_type,
        description=f"test {op_type.value}",
        data=data or {},
    )


class TestSessionModes:
    @pytest.mark.asyncio
    async def test_normal_mode_defers(self, integration):
        assert integration.session_approval_mode is ApprovalMode.NORMAL
        result = await integration._check_auto_approval(
            _operation(OperationType.FILE_WRITE)
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_accept_edits_approves_file_write(self, integration):
        integration.session_approval_mode = ApprovalMode.ACCEPT_EDITS
        result = await integration._check_auto_approval(
            _operation(OperationType.FILE_WRITE)
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_accept_edits_still_prompts_for_commands(self, integration):
        integration.session_approval_mode = ApprovalMode.ACCEPT_EDITS
        result = await integration._check_auto_approval(
            _operation(OperationType.COMMAND_EXECUTE)
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_accept_edits_still_prompts_for_deletes(self, integration):
        integration.session_approval_mode = ApprovalMode.ACCEPT_EDITS
        result = await integration._check_auto_approval(
            _operation(OperationType.FILE_DELETE)
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_accept_all_approves_commands(self, integration):
        integration.session_approval_mode = ApprovalMode.ACCEPT_ALL
        result = await integration._check_auto_approval(
            _operation(OperationType.COMMAND_EXECUTE)
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_force_prompt_beats_accept_all(self, integration):
        integration.session_approval_mode = ApprovalMode.ACCEPT_ALL
        result = await integration._check_auto_approval(
            _operation(OperationType.FILE_WRITE, data={"_force_prompt": True})
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_auto_approval_beats_modes(self, integration):
        integration.enable_auto_approval = False
        integration.session_approval_mode = ApprovalMode.ACCEPT_ALL
        result = await integration._check_auto_approval(
            _operation(OperationType.FILE_WRITE)
        )
        assert result is None


class TestModeCycle:
    def test_cycle_order(self, integration):
        assert integration.cycle_approval_mode() is ApprovalMode.ACCEPT_EDITS
        assert integration.cycle_approval_mode() is ApprovalMode.ACCEPT_ALL
        assert integration.cycle_approval_mode() is ApprovalMode.NORMAL


class TestAcceptCommandHandler:
    @pytest.fixture
    def harness(self, integration):
        from omnimancer.cli.command_dispatch import CommandDispatchMixin

        class _Harness(CommandDispatchMixin):
            def __init__(self):
                self.console = MagicMock()
                self.engine = MagicMock()
                self.approval_integration = integration
                self.messages = []

            def _show_error(self, message):
                self.messages.append(("error", message))

            def _show_info(self, message):
                self.messages.append(("info", message))

            def _show_success(self, message):
                self.messages.append(("success", message))

        return _Harness()

    @pytest.mark.asyncio
    async def test_explicit_mode(self, harness, integration):
        await harness._handle_accept_command(["edits"])
        assert integration.session_approval_mode is ApprovalMode.ACCEPT_EDITS
        await harness._handle_accept_command(["all"])
        assert integration.session_approval_mode is ApprovalMode.ACCEPT_ALL
        await harness._handle_accept_command(["off"])
        assert integration.session_approval_mode is ApprovalMode.NORMAL

    @pytest.mark.asyncio
    async def test_bare_command_cycles(self, harness, integration):
        await harness._handle_accept_command([])
        assert integration.session_approval_mode is ApprovalMode.ACCEPT_EDITS

    @pytest.mark.asyncio
    async def test_invalid_arg_errors(self, harness, integration):
        await harness._handle_accept_command(["bogus"])
        assert integration.session_approval_mode is ApprovalMode.NORMAL
        assert any(kind == "error" for kind, _ in harness.messages)
