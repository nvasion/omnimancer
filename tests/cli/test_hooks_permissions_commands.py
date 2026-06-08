"""Tests for the /hooks and /permissions CLI commands."""

from unittest.mock import MagicMock

import pytest

from omnimancer.cli.command_dispatch import CommandDispatchMixin
from omnimancer.cli.commands import Command, SlashCommand
from omnimancer.core.models import Config, HookCommand, PermissionRule, ProviderConfig


class _Dispatcher(CommandDispatchMixin):
    """Minimal harness exercising just the command handlers."""

    def __init__(self, config: Config):
        self.engine = MagicMock()
        self.engine.config_manager.get_config.return_value = config
        self.console = MagicMock()
        self.errors: list = []
        self.successes: list = []
        self.infos: list = []

    def _show_error(self, msg):
        self.errors.append(msg)

    def _show_success(self, msg):
        self.successes.append(msg)

    def _show_info(self, msg):
        self.infos.append(msg)


def _config() -> Config:
    return Config(
        default_provider="openai",
        providers={"openai": ProviderConfig(api_key="k", model="gpt-4")},
        storage_path="/tmp/omnimancer_cmd_test",
    )


def _cmd(slash: SlashCommand, *args: str) -> Command:
    return Command.create_slash_command(slash, list(args), raw_input="x")


@pytest.fixture
def disp():
    return _Dispatcher(_config())


class TestHooksCommand:
    @pytest.mark.asyncio
    async def test_add_hook(self, disp):
        await disp._handle_hooks_command(
            _cmd(
                SlashCommand.HOOKS,
                "add",
                "pre_send_message",
                "logit",
                "--blocking",
                "echo",
                "hi",
            )
        )
        hooks = disp.engine.config_manager.get_config().hooks.pre_send_message
        assert len(hooks) == 1
        assert hooks[0].name == "logit"
        assert hooks[0].command == "echo hi"
        assert hooks[0].blocking is True
        disp.engine.config_manager.save_config.assert_called()

    @pytest.mark.asyncio
    async def test_add_hook_with_matcher_and_timeout(self, disp):
        await disp._handle_hooks_command(
            _cmd(
                SlashCommand.HOOKS,
                "add",
                "tool_use_request",
                "guard",
                "--matcher",
                "^rm",
                "--timeout",
                "5",
                "false",
            )
        )
        hook = disp.engine.config_manager.get_config().hooks.tool_use_request[0]
        assert hook.matcher == "^rm"
        assert hook.timeout == 5
        assert hook.command == "false"

    @pytest.mark.asyncio
    async def test_add_hook_rejects_unknown_event(self, disp):
        await disp._handle_hooks_command(
            _cmd(SlashCommand.HOOKS, "add", "bogus_event", "n", "echo")
        )
        assert disp.errors
        assert not disp.engine.config_manager.get_config().hooks.pre_send_message

    @pytest.mark.asyncio
    async def test_remove_hook(self, disp):
        config = disp.engine.config_manager.get_config()
        config.hooks.post_tool.append(HookCommand(name="x", command="echo"))
        await disp._handle_hooks_command(
            _cmd(SlashCommand.HOOKS, "remove", "post_tool", "x")
        )
        assert config.hooks.post_tool == []
        assert disp.successes

    @pytest.mark.asyncio
    async def test_toggle_off(self, disp):
        await disp._handle_hooks_command(_cmd(SlashCommand.HOOKS, "off"))
        assert disp.engine.config_manager.get_config().hooks.enabled is False

    @pytest.mark.asyncio
    async def test_list_no_hooks(self, disp):
        await disp._handle_hooks_command(_cmd(SlashCommand.HOOKS))
        assert disp.infos  # "No hooks configured"


class TestPermissionsCommand:
    @pytest.mark.asyncio
    async def test_add_deny_rule(self, disp):
        await disp._handle_permissions_command(
            _cmd(SlashCommand.PERMISSIONS, "deny", "command_execute", "^rm")
        )
        rules = disp.engine.config_manager.get_config().permissions.always_deny
        assert len(rules) == 1
        assert rules[0].tool == "command_execute"
        assert rules[0].matcher == "^rm"
        disp.engine.config_manager.save_config.assert_called()

    @pytest.mark.asyncio
    async def test_add_allow_rule_without_matcher(self, disp):
        await disp._handle_permissions_command(
            _cmd(SlashCommand.PERMISSIONS, "allow", "file_write")
        )
        rules = disp.engine.config_manager.get_config().permissions.always_allow
        assert rules[0].tool == "file_write"
        assert rules[0].matcher is None

    @pytest.mark.asyncio
    async def test_remove_rule_by_index(self, disp):
        config = disp.engine.config_manager.get_config()
        config.permissions.always_ask.append(PermissionRule(tool="file_delete"))
        await disp._handle_permissions_command(
            _cmd(SlashCommand.PERMISSIONS, "remove", "ask", "1")
        )
        assert config.permissions.always_ask == []
        assert disp.successes

    @pytest.mark.asyncio
    async def test_remove_bad_index(self, disp):
        await disp._handle_permissions_command(
            _cmd(SlashCommand.PERMISSIONS, "remove", "deny", "5")
        )
        assert disp.errors

    @pytest.mark.asyncio
    async def test_toggle_on(self, disp):
        config = disp.engine.config_manager.get_config()
        config.permissions.enabled = False
        await disp._handle_permissions_command(_cmd(SlashCommand.PERMISSIONS, "on"))
        assert config.permissions.enabled is True

    @pytest.mark.asyncio
    async def test_list_empty(self, disp):
        await disp._handle_permissions_command(_cmd(SlashCommand.PERMISSIONS))
        assert disp.infos
