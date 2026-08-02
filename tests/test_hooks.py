"""Tests for the lifecycle hooks system (HooksManager + config models)."""

import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.core.agent.types import Operation, OperationResult, OperationType
from omnimancer.core.agent_engine import AgentEngine
from omnimancer.core.engine import CoreEngine
from omnimancer.core.hooks import HooksManager
from omnimancer.core.models import Config, HookCommand, HooksConfig, ProviderConfig


def _hook(command: str, **kw) -> HookCommand:
    kw.setdefault("name", kw.get("name", "h"))
    return HookCommand(command=command, **kw)


class TestHooksConfigModel:
    def test_defaults_are_empty_and_enabled(self):
        cfg = HooksConfig()
        assert cfg.enabled is True
        assert cfg.pre_send_message == []
        assert cfg.turn_complete == []
        assert cfg.hooks_for("tool_use_request") == []

    def test_hooks_for_unknown_event_is_empty(self):
        cfg = HooksConfig(pre_send_message=[_hook("true")])
        assert cfg.hooks_for("does_not_exist") == []
        assert len(cfg.hooks_for("pre_send_message")) == 1

    def test_empty_name_or_command_rejected(self):
        with pytest.raises(ValueError):
            HookCommand(name="", command="true")
        with pytest.raises(ValueError):
            HookCommand(name="x", command="   ")

    def test_roundtrips_through_json(self):
        cfg = HooksConfig(
            post_tool=[_hook("echo hi", name="log", blocking=True)],
            turn_complete=[_hook("true", name="turn")],
        )
        restored = HooksConfig.model_validate(cfg.model_dump(mode="json"))
        assert restored.post_tool[0].name == "log"
        assert restored.post_tool[0].blocking is True
        assert restored.turn_complete[0].name == "turn"


class TestHooksManagerFiring:
    @pytest.mark.asyncio
    async def test_no_hooks_is_allowed_noop(self):
        out = await HooksManager(HooksConfig()).fire("pre_send_message", {})
        assert out.allowed is True
        assert out.results == []

    @pytest.mark.asyncio
    async def test_master_switch_disables_all(self):
        cfg = HooksConfig(
            enabled=False,
            pre_send_message=[_hook("false", blocking=True)],
        )
        out = await HooksManager(cfg).fire("pre_send_message", {})
        assert out.allowed is True
        assert out.results == []

    @pytest.mark.asyncio
    async def test_successful_hook_runs_and_allows(self):
        cfg = HooksConfig(pre_send_message=[_hook("exit 0", name="ok")])
        out = await HooksManager(cfg).fire("pre_send_message", {})
        assert out.allowed is True
        assert out.results[0].succeeded is True
        assert out.results[0].returncode == 0

    @pytest.mark.asyncio
    async def test_blocking_hook_nonzero_vetoes(self):
        cfg = HooksConfig(
            tool_use_request=[_hook("exit 3", name="deny", blocking=True)]
        )
        out = await HooksManager(cfg).fire("tool_use_request", {})
        assert out.allowed is False
        assert out.results[0].blocked is True
        assert "deny" in out.reason

    @pytest.mark.asyncio
    async def test_nonblocking_failure_does_not_veto(self):
        cfg = HooksConfig(post_tool=[_hook("exit 1", name="noisy")])
        out = await HooksManager(cfg).fire("post_tool", {})
        assert out.allowed is True
        assert out.results[0].succeeded is False
        assert out.results[0].blocked is False

    @pytest.mark.asyncio
    async def test_disabled_hook_is_skipped(self):
        cfg = HooksConfig(
            pre_send_message=[_hook("exit 1", name="off", enabled=False, blocking=True)]
        )
        out = await HooksManager(cfg).fire("pre_send_message", {})
        assert out.allowed is True
        assert out.results == []

    @pytest.mark.asyncio
    async def test_blocking_timeout_vetoes(self):
        cfg = HooksConfig(
            pre_send_message=[
                _hook(
                    f"{sys.executable} -c 'import time; time.sleep(5)'",
                    name="slow",
                    blocking=True,
                    timeout=1,
                )
            ]
        )
        out = await HooksManager(cfg).fire("pre_send_message", {})
        assert out.allowed is False
        assert out.results[0].timed_out is True


class TestHooksManagerMatcher:
    @pytest.mark.asyncio
    async def test_matcher_runs_only_on_match(self):
        cfg = HooksConfig(
            tool_use_request=[
                _hook("exit 1", name="rm-guard", matcher=r"^rm\b", blocking=True)
            ]
        )
        mgr = HooksManager(cfg)
        # Non-matching target → hook skipped → allowed.
        out = await mgr.fire("tool_use_request", {}, match_target="ls -la")
        assert out.allowed is True and out.results == []
        # Matching target → blocking hook runs → vetoed.
        out = await mgr.fire("tool_use_request", {}, match_target="rm -rf /")
        assert out.allowed is False

    @pytest.mark.asyncio
    async def test_invalid_matcher_skips_hook(self):
        cfg = HooksConfig(
            pre_send_message=[_hook("exit 1", name="bad", matcher="(", blocking=True)]
        )
        out = await HooksManager(cfg).fire("pre_send_message", {}, match_target="x")
        assert out.allowed is True  # invalid regex → hook skipped, not crashed


class TestHooksManagerContext:
    @pytest.mark.asyncio
    async def test_payload_delivered_on_stdin(self, tmp_path):
        out_file = tmp_path / "stdin.txt"
        cfg = HooksConfig(pre_send_message=[_hook(f"cat > {out_file}", name="capture")])
        await HooksManager(cfg).fire("pre_send_message", {"message": "hello world"})
        captured = out_file.read_text()
        assert "hello world" in captured
        assert "pre_send_message" in captured  # event injected into payload

    @pytest.mark.asyncio
    async def test_scalar_context_exported_as_env(self, tmp_path):
        out_file = tmp_path / "env.txt"
        cfg = HooksConfig(
            post_tool=[
                _hook(
                    f'echo "$OMNIMANCER_HOOK_EVENT:$OMNIMANCER_HOOK_TOOL" > {out_file}',
                    name="env",
                )
            ]
        )
        await HooksManager(cfg).fire("post_tool", {"tool": "Bash"})
        assert out_file.read_text().strip() == "post_tool:Bash"

    @pytest.mark.asyncio
    async def test_nonserialisable_context_does_not_crash(self):
        cfg = HooksConfig(pre_send_message=[_hook("true", name="ok")])
        out = await HooksManager(cfg).fire("pre_send_message", {"obj": object()})
        assert out.allowed is True
        assert out.results[0].succeeded is True

    @pytest.mark.asyncio
    async def test_turn_complete_payload_is_not_modified(self, tmp_path):
        out_file = tmp_path / "turn.json"
        recorder = tmp_path / "record_turn.py"
        recorder.write_text(
            "import pathlib, sys\n"
            "pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())\n"
        )
        payload = {
            "type": "agent-turn-complete",
            "turn-id": "turn-1",
            "last-assistant-message": "done",
            "session_id": "session-1",
            "usage": {
                "input_tokens": 2,
                "output_tokens": 1,
                "total_cost_usd": 0.01,
            },
            "cwd": str(tmp_path),
        }
        cfg = HooksConfig(
            turn_complete=[
                _hook(
                    f"{sys.executable} {recorder} {out_file}",
                    name="capture-turn",
                    blocking=True,
                )
            ]
        )

        outcome = await HooksManager(cfg).fire("turn_complete", payload)

        assert outcome.allowed is True
        assert json.loads(out_file.read_text()) == payload


class TestHooksManagerRobustness:
    def test_none_config_is_safe(self):
        mgr = HooksManager(None)
        assert mgr._selected("pre_send_message", "") == []


def _config_with(hooks: HooksConfig) -> Config:
    return Config(
        default_provider="openai",
        providers={"openai": ProviderConfig(api_key="k", model="gpt-4")},
        storage_path="/tmp/omnimancer_hooks_test",
        hooks=hooks,
    )


class TestEngineHookWiring:
    """pre/post_send_message hooks fire from CoreEngine.send_message."""

    def _engine(self, hooks: HooksConfig):
        engine = CoreEngine.__new__(CoreEngine)
        cm = MagicMock()
        cm.get_config.return_value = _config_with(hooks)
        engine.config_manager = cm
        provider = MagicMock()
        provider.get_provider_name.return_value = "openai"
        provider.model = "gpt-4"
        provider.send_message = AsyncMock()
        engine.current_provider = provider
        return engine, provider

    @pytest.mark.asyncio
    async def test_blocking_pre_send_hook_stops_send(self):
        hooks = HooksConfig(
            pre_send_message=[HookCommand(name="veto", command="exit 1", blocking=True)]
        )
        engine, provider = self._engine(hooks)
        resp = await engine.send_message("hello")
        assert not resp.is_success
        assert "blocked" in resp.error.lower()
        provider.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonblocking_pre_send_hook_allows_send(self):
        hooks = HooksConfig(
            pre_send_message=[HookCommand(name="log", command="exit 1")]
        )
        engine, provider = self._engine(hooks)
        provider.send_message.return_value = MagicMock(is_success=True, content="hi")
        engine.chat_manager = MagicMock()
        engine.chat_manager.get_current_context.return_value = MagicMock()
        await engine.send_message("hello")
        provider.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_matcher_only_blocks_matching_message(self):
        hooks = HooksConfig(
            pre_send_message=[
                HookCommand(
                    name="secret-guard",
                    command="exit 1",
                    matcher="SECRET",
                    blocking=True,
                )
            ]
        )
        engine, provider = self._engine(hooks)
        engine.chat_manager = MagicMock()
        engine.chat_manager.get_current_context.return_value = MagicMock()
        provider.send_message.return_value = MagicMock(is_success=True, content="ok")
        # Non-matching message goes through.
        await engine.send_message("hello")
        provider.send_message.assert_awaited_once()
        # Matching message is blocked.
        resp = await engine.send_message("this has a SECRET in it")
        assert not resp.is_success


class TestAgentEngineToolHookWiring:
    """tool_use_request hooks fire from AgentEngine.execute_with_approval."""

    def _agent_engine(self, hooks: HooksConfig):
        engine = AgentEngine.__new__(AgentEngine)
        cm = MagicMock()
        cm.get_config.return_value = _config_with(hooks)
        engine.config_manager = cm
        # _generate_preview / _execute_operation are exercised only past the
        # hook gate; stub them so the test isolates hook behaviour.
        engine._generate_preview = AsyncMock(return_value="preview")
        engine._execute_operation = AsyncMock(
            return_value=OperationResult(success=True, data="done")
        )
        engine.operation_history = []
        return engine

    def _operation(self, command: str) -> Operation:
        return Operation(
            type=OperationType.COMMAND_EXECUTE,
            description="run a command",
            data={"command": command},
            requires_approval=False,
        )

    @pytest.mark.asyncio
    async def test_blocking_tool_hook_vetoes_execution(self):
        hooks = HooksConfig(
            tool_use_request=[
                HookCommand(
                    name="rm-guard", command="exit 1", matcher=r"^rm\b", blocking=True
                )
            ]
        )
        engine = self._agent_engine(hooks)
        result = await engine.execute_with_approval(self._operation("rm -rf /"))
        assert not result.success
        assert "blocked" in result.error.lower()
        engine._execute_operation.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonmatching_tool_hook_allows_execution(self):
        hooks = HooksConfig(
            tool_use_request=[
                HookCommand(
                    name="rm-guard", command="exit 1", matcher=r"^rm\b", blocking=True
                )
            ]
        )
        engine = self._agent_engine(hooks)
        result = await engine.execute_with_approval(self._operation("ls -la"))
        assert result.success
        engine._execute_operation.assert_awaited_once()
