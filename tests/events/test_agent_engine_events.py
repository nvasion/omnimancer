"""Operation-gate event instrumentation tests (WU-A4).

Drives a skeleton AgentEngine through every gate outcome and asserts the
JSONL event feed sees it: allow, permission-deny, hook-veto, user
deny/cancel/grant, executor exception (which now also fires post_tool),
and native-vs-marker invocation tagging. Gate ordering itself is pinned
by tests/test_permission_rules.py and tests/test_hooks.py.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.core.agent.types import Operation, OperationResult, OperationType
from omnimancer.core.agent_engine import AgentEngine
from omnimancer.core.hooks import HookOutcome
from omnimancer.core.models import (
    Config,
    EventsConfig,
    PermissionRule,
    PermissionsConfig,
    ProviderConfig,
)
from omnimancer.events import emitter


def _config_with(perms: PermissionsConfig) -> Config:
    return Config(
        default_provider="p",
        providers={"p": ProviderConfig(model="m")},
        storage_path="/tmp/omni-events-test",
        permissions=perms,
    )


def _engine(perms: PermissionsConfig = None) -> AgentEngine:
    engine = AgentEngine.__new__(AgentEngine)
    cm = MagicMock()
    cm.get_config.return_value = _config_with(perms or PermissionsConfig())
    engine.config_manager = cm
    engine._generate_preview = AsyncMock(return_value="preview")
    engine._execute_operation = AsyncMock(
        return_value=OperationResult(success=True, data="done")
    )
    engine._fire_hook = AsyncMock(return_value=HookOutcome(event="tool_use_request"))
    engine.approval = MagicMock()
    engine.approval.request_approval = AsyncMock(return_value=(True, False))
    engine.operation_history = []
    # The exception handler's error_context reads these managers.
    engine.file_system = MagicMock(enabled=True)
    engine.executor = MagicMock(enabled=True)
    engine.web_client = MagicMock(enabled=True)
    engine.mcp_integrator = MagicMock(enabled=True)
    return engine


def _op(command: str = "ls -la", requires_approval: bool = False, **data) -> Operation:
    return Operation(
        type=OperationType.COMMAND_EXECUTE,
        description=f"Execute: {command}",
        data={"command": command, **data},
        requires_approval=requires_approval,
    )


@pytest.fixture
async def event_file(tmp_path):
    ok = await emitter.init_events(
        "sess-gate", "headless", EventsConfig(directory=str(tmp_path))
    )
    assert ok
    yield tmp_path / "omn-sess-gate.jsonl"
    await emitter.shutdown_events()


async def _events(path: Path, expected: int, timeout: float = 3.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            lines = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            if len(lines) >= expected:
                return lines
        await asyncio.sleep(0.02)
    raise AssertionError(f"expected {expected} events in {path} within {timeout}s")


class TestGateEvents:
    async def test_allowed_operation_emits_start_and_end(self, event_file):
        engine = _engine()
        result = await engine.execute_with_approval(_op())
        assert result.success
        events = await _events(event_file, 2)
        assert [e["event"] for e in events] == ["tool_start", "tool_end"]
        start = events[0]
        assert start["data"]["op_type"] == "command_execute"
        assert start["data"]["target"] == "ls -la"
        assert start["data"]["invocation"] == "marker"  # untagged default
        assert start["data"]["tool"] == "command_execute"
        end = events[1]
        assert end["data"]["success"] is True
        assert end["data"]["op_id"] == start["data"]["op_id"]

    async def test_native_stamp_carries_tool_name(self, event_file):
        engine = _engine()
        op = _op(_tool_name="Bash", _invocation="native")
        await engine.execute_with_approval(op)
        events = await _events(event_file, 2)
        assert events[0]["data"]["tool"] == "Bash"
        assert events[0]["data"]["invocation"] == "native"

    async def test_permission_deny_emits_denied_and_end(self, event_file):
        perms = PermissionsConfig(
            always_deny=[PermissionRule(tool="command_execute", matcher=r"^rm\b")]
        )
        engine = _engine(perms)
        result = await engine.execute_with_approval(_op("rm -rf /"))
        assert not result.success
        events = await _events(event_file, 3)
        names = [e["event"] for e in events]
        assert names == ["tool_start", "approval_denied", "tool_end"]
        assert events[1]["data"]["source"] == "permission_rule"
        assert events[2]["data"]["success"] is False
        assert "approval_requested" not in names

    async def test_hook_veto_emits_denied_and_end(self, event_file):
        engine = _engine()
        veto = HookOutcome(event="tool_use_request", allowed=False)
        engine._fire_hook = AsyncMock(return_value=veto)
        result = await engine.execute_with_approval(_op())
        assert not result.success
        events = await _events(event_file, 3)
        assert [e["event"] for e in events] == [
            "tool_start",
            "approval_denied",
            "tool_end",
        ]
        assert events[1]["data"]["source"] == "hook"

    async def test_user_denial_emits_requested_then_denied(self, event_file):
        engine = _engine()
        engine.approval.request_approval = AsyncMock(return_value=(False, False))
        result = await engine.execute_with_approval(_op(requires_approval=True))
        assert not result.success
        events = await _events(event_file, 4)
        assert [e["event"] for e in events] == [
            "tool_start",
            "approval_requested",
            "approval_denied",
            "tool_end",
        ]
        assert events[2]["data"]["source"] == "user"
        assert events[2]["data"]["cancelled"] is False

    async def test_user_cancel_marks_tool_end_cancelled(self, event_file):
        engine = _engine()
        engine.approval.request_approval = AsyncMock(return_value=(False, True))
        result = await engine.execute_with_approval(_op(requires_approval=True))
        assert result.was_cancelled
        events = await _events(event_file, 4)
        assert events[2]["data"]["cancelled"] is True
        assert events[3]["event"] == "tool_end"
        assert events[3]["data"]["was_cancelled"] is True

    async def test_approval_grant_emits_granted_then_success(self, event_file):
        engine = _engine()
        result = await engine.execute_with_approval(_op(requires_approval=True))
        assert result.success
        events = await _events(event_file, 4)
        assert [e["event"] for e in events] == [
            "tool_start",
            "approval_requested",
            "approval_granted",
            "tool_end",
        ]

    async def test_exception_fires_post_tool_and_tool_end(self, event_file):
        engine = _engine()
        engine._execute_operation = AsyncMock(side_effect=RuntimeError("boom"))
        result = await engine.execute_with_approval(_op())
        assert not result.success
        events = await _events(event_file, 2)
        assert events[-1]["event"] == "tool_end"
        assert events[-1]["data"]["success"] is False
        # post_tool now fires on the exception path (previously skipped).
        fired = [call.args[0] for call in engine._fire_hook.await_args_list]
        assert fired == ["tool_use_request", "post_tool"]
        post_ctx = engine._fire_hook.await_args_list[1].args[1]
        assert post_ctx["success"] is False
        assert "boom" in post_ctx["error"]

    async def test_history_timestamp_is_wall_clock(self, event_file):
        engine = _engine()
        await engine.execute_with_approval(_op())
        ts = engine.operation_history[0]["timestamp"]
        assert abs(time.time() - ts) < 60

    async def test_events_disabled_gate_still_works(self):
        # No init_events in this test: every emit must silently no-op.
        engine = _engine()
        result = await engine.execute_with_approval(_op())
        assert result.success
        engine._execute_operation.assert_awaited_once()
