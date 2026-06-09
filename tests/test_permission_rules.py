"""Tests for config-driven permission rules and the sensitive-path override."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.core.agent.types import Operation, OperationResult, OperationType
from omnimancer.core.agent_engine import AgentEngine
from omnimancer.core.models import (
    Config,
    PermissionRule,
    PermissionsConfig,
    ProviderConfig,
)
from omnimancer.core.security.permission_controller import PermissionController
from omnimancer.core.security.permission_rules import (
    PermissionDecision,
    PermissionRuleEngine,
)


def _rules(**kw) -> PermissionsConfig:
    return PermissionsConfig(**kw)


class TestPermissionRuleEngine:
    def test_no_rules_is_default(self):
        eng = PermissionRuleEngine(PermissionsConfig())
        assert eng.evaluate("file_write", "/tmp/x") == PermissionDecision.DEFAULT

    def test_disabled_is_default(self):
        eng = PermissionRuleEngine(
            _rules(
                enabled=False,
                always_deny=[PermissionRule(tool="*")],
            )
        )
        assert eng.evaluate("command_execute", "rm -rf /") == PermissionDecision.DEFAULT

    def test_allow_rule(self):
        eng = PermissionRuleEngine(
            _rules(always_allow=[PermissionRule(tool="file_write", matcher=r"\.env$")])
        )
        assert eng.evaluate("file_write", "/proj/.env") == PermissionDecision.ALLOW
        assert eng.evaluate("file_write", "/proj/main.py") == PermissionDecision.DEFAULT

    def test_deny_rule(self):
        eng = PermissionRuleEngine(
            _rules(
                always_deny=[PermissionRule(tool="command_execute", matcher=r"^rm\b")]
            )
        )
        assert eng.evaluate("command_execute", "rm -rf /") == PermissionDecision.DENY
        assert eng.evaluate("command_execute", "ls -la") == PermissionDecision.DEFAULT

    def test_ask_rule(self):
        eng = PermissionRuleEngine(_rules(always_ask=[PermissionRule(tool="*")]))
        assert eng.evaluate("file_delete", "/proj/x") == PermissionDecision.ASK

    def test_precedence_deny_over_ask_over_allow(self):
        eng = PermissionRuleEngine(
            _rules(
                always_allow=[PermissionRule(tool="*")],
                always_ask=[PermissionRule(tool="*")],
                always_deny=[
                    PermissionRule(tool="command_execute", matcher="dangerous")
                ],
            )
        )
        # command_execute matches all three; deny wins.
        assert eng.evaluate("command_execute", "dangerous") == PermissionDecision.DENY
        # file_write matches ask + allow; ask wins.
        assert eng.evaluate("file_write", "x") == PermissionDecision.ASK

    def test_wildcard_tool_matches_any(self):
        eng = PermissionRuleEngine(_rules(always_allow=[PermissionRule(tool="*")]))
        assert eng.evaluate("anything", "y") == PermissionDecision.ALLOW

    def test_tool_specific_rule_does_not_match_other_tools(self):
        eng = PermissionRuleEngine(
            _rules(always_deny=[PermissionRule(tool="file_delete")])
        )
        assert eng.evaluate("file_write", "x") == PermissionDecision.DEFAULT
        assert eng.evaluate("file_delete", "x") == PermissionDecision.DENY

    def test_invalid_matcher_is_skipped(self):
        eng = PermissionRuleEngine(
            _rules(always_deny=[PermissionRule(tool="*", matcher="(")])
        )
        # Invalid regex → rule ignored, not crashing.
        assert eng.evaluate("file_write", "x") == PermissionDecision.DEFAULT

    def test_none_config_is_default(self):
        assert (
            PermissionRuleEngine(None).evaluate("x", "y") == PermissionDecision.DEFAULT
        )


class TestSensitivePathOverride:
    """The .env regression: approval/allow must override sensitive-path block."""

    def setup_method(self):
        self.controller = PermissionController()

    def test_project_env_blocked_by_default(self):
        path = os.path.join(os.getcwd(), ".env")
        assert self.controller.validate_path_access(path, "write") is False

    def test_project_env_allowed_when_approved(self):
        path = os.path.join(os.getcwd(), ".env")
        assert (
            self.controller.validate_path_access(path, "write", allow_sensitive=True)
            is True
        )

    @pytest.mark.parametrize("hard", ["/etc/passwd", "/usr/bin/x", "/root/.bashrc"])
    def test_hard_paths_stay_blocked_even_when_approved(self, hard):
        assert (
            self.controller.validate_path_access(hard, "write", allow_sensitive=True)
            is False
        )

    def test_ssh_key_stays_blocked_when_approved(self):
        path = os.path.expanduser("~/.ssh/id_rsa")
        assert (
            self.controller.validate_path_access(path, "write", allow_sensitive=True)
            is False
        )

    def test_secret_named_file_in_project_overridable(self):
        path = os.path.join(os.getcwd(), "my_secret.txt")
        assert self.controller.validate_path_access(path, "write") is False
        assert (
            self.controller.validate_path_access(path, "write", allow_sensitive=True)
            is True
        )


def _config_with(perms: PermissionsConfig) -> Config:
    return Config(
        default_provider="openai",
        providers={"openai": ProviderConfig(api_key="k", model="gpt-4")},
        storage_path="/tmp/omnimancer_perm_test",
        permissions=perms,
    )


class TestAgentEnginePermissionWiring:
    """execute_with_approval honours permission rules (deny/allow/ask)."""

    def _engine(self, perms: PermissionsConfig):
        engine = AgentEngine.__new__(AgentEngine)
        cm = MagicMock()
        cm.get_config.return_value = _config_with(perms)
        engine.config_manager = cm
        engine._generate_preview = AsyncMock(return_value="preview")
        engine._execute_operation = AsyncMock(
            return_value=OperationResult(success=True, data="done")
        )
        engine.approval = MagicMock()
        engine.approval.request_approval = AsyncMock(return_value=True)
        engine.operation_history = []
        return engine

    def _op(self, command="rm -rf /", requires_approval=True) -> Operation:
        return Operation(
            type=OperationType.COMMAND_EXECUTE,
            description="run",
            data={"command": command},
            requires_approval=requires_approval,
        )

    @pytest.mark.asyncio
    async def test_deny_rule_blocks_without_executing(self):
        perms = PermissionsConfig(
            always_deny=[PermissionRule(tool="command_execute", matcher=r"^rm\b")]
        )
        engine = self._engine(perms)
        result = await engine.execute_with_approval(self._op("rm -rf /"))
        assert not result.success
        assert "denied by permission rule" in result.error.lower()
        engine._execute_operation.assert_not_called()
        engine.approval.request_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_rule_skips_approval_and_executes(self):
        perms = PermissionsConfig(always_allow=[PermissionRule(tool="command_execute")])
        engine = self._engine(perms)
        result = await engine.execute_with_approval(self._op("ls -la"))
        assert result.success
        # Approval prompt skipped because the rule auto-allowed it.
        engine.approval.request_approval.assert_not_called()
        engine._execute_operation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ask_rule_forces_prompt_even_if_not_required(self):
        perms = PermissionsConfig(always_ask=[PermissionRule(tool="*")])
        engine = self._engine(perms)
        # Operation would normally NOT require approval, but ask forces it.
        op = self._op("ls -la", requires_approval=False)
        result = await engine.execute_with_approval(op)
        assert result.success
        engine.approval.request_approval.assert_awaited_once()
        assert op.data.get("_force_prompt") is True

    @pytest.mark.asyncio
    async def test_no_rules_uses_default_flow(self):
        engine = self._engine(PermissionsConfig())
        op = self._op("ls -la", requires_approval=False)
        result = await engine.execute_with_approval(op)
        assert result.success
        # No rule → default flow; requires_approval stayed False → no prompt.
        engine.approval.request_approval.assert_not_called()
        engine._execute_operation.assert_awaited_once()
