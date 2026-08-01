"""Emitter facade + status-bus wiring tests (WU-A3).

Pins the two load-bearing guarantees: mapped bus events land as
omn.event.v1 JSONL lines, and emission never blocks even with full
queues or a wedged listener.
"""

import asyncio
import json
import time
from pathlib import Path

import pytest

from omnimancer.core.agent.status_core import (
    AgentOperation,
    EventType,
    StatusDisplayConfig,
    StreamPriority,
)
from omnimancer.core.agent.status_manager import UnifiedStatusManager
from omnimancer.core.agent.types import OperationType as AgentOperationType
from omnimancer.core.models import EventsConfig
from omnimancer.events import emitter
from omnimancer.events.schema import translate_operation_type


@pytest.fixture
def events_dir(tmp_path):
    return tmp_path / "events"


@pytest.fixture
async def live_pipeline(events_dir):
    """Initialized pipeline writing to a tmp dir; always torn down."""
    ok = await emitter.init_events(
        "sess-test", "interactive", EventsConfig(directory=str(events_dir))
    )
    assert ok
    yield events_dir / "omn-sess-test.jsonl"
    await emitter.shutdown_events()


async def _read_lines(path: Path, expected: int, timeout: float = 3.0) -> list:
    """Poll until the JSONL file holds `expected` lines, then parse them."""
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
    raise AssertionError(f"expected {expected} lines in {path} within {timeout}s")


class TestInitGating:
    async def test_disabled_config_is_noop(self, events_dir):
        ok = await emitter.init_events(
            "s", "headless", EventsConfig(enabled=False, directory=str(events_dir))
        )
        assert ok is False
        assert not emitter.events_enabled()
        assert not events_dir.exists()

    async def test_env_kill_switch(self, events_dir, monkeypatch):
        monkeypatch.setenv(emitter.ENV_KILL_SWITCH, "0")
        ok = await emitter.init_events(
            "s", "headless", EventsConfig(directory=str(events_dir))
        )
        assert ok is False
        assert not events_dir.exists()

    async def test_emit_before_init_is_noop(self):
        # Must not raise, must not create anything.
        await emitter.emit_event(EventType.TURN_START, {"turn": 1})


class TestPipeline:
    async def test_lifecycle_and_custom_events_land(self, live_pipeline):
        manager = emitter._state.manager
        operation = AgentOperation(
            operation_type=translate_operation_type(AgentOperationType.FILE_WRITE),
            description="write foo.py",
            agent_id="main",
            metadata={"tool": "Write", "invocation": "native"},
        )
        await manager.start_operation(operation)
        await manager.complete_operation(operation.operation_id)
        await emitter.emit_event(EventType.TURN_END, {"turn": 1})

        lines = await _read_lines(live_pipeline, 3)
        by_event = {line["event"]: line for line in lines}
        assert by_event["tool_start"]["data"]["tool"] == "Write"
        assert by_event["tool_start"]["data"]["op_id"] == operation.operation_id
        assert by_event["tool_end"]["data"]["success"] is True
        # Terminal events carry the operation's identity too — renderers
        # must never fall back to a bare "tool_end" label.
        assert by_event["tool_end"]["data"]["tool"] == "Write"
        assert by_event["turn_end"]["data"]["turn"] == 1
        # Envelope invariants
        for line in lines:
            assert line["v"] == 1
            assert line["session_id"] == "sess-test"
            assert line["mode"] == "interactive"
            assert line["ts"].endswith("+00:00")
        # seq strictly increases in file order
        seqs = [line["seq"] for line in lines]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    async def test_failed_and_cancelled_map_to_tool_end(self, live_pipeline):
        manager = emitter._state.manager
        failed = AgentOperation(description="boom", agent_id="main")
        await manager.start_operation(failed)
        await manager.fail_operation(failed.operation_id, "exploded")
        lines = await _read_lines(live_pipeline, 2)
        end = [line for line in lines if line["event"] == "tool_end"][-1]
        assert end["data"]["success"] is False
        assert end["data"]["error"] == "exploded"

    async def test_agent_context_scopes_identity(self, live_pipeline):
        with emitter.agent_context("subagent-tester-abcd1234", "main"):
            await emitter.emit_event(EventType.APPROVAL_REQUESTED, {"tool": "Bash"})
        await emitter.emit_event(EventType.APPROVAL_GRANTED, {"tool": "Bash"})
        lines = await _read_lines(live_pipeline, 2)
        requested = [line for line in lines if line["event"] == "approval_requested"][0]
        granted = [line for line in lines if line["event"] == "approval_granted"][0]
        assert requested["agent_id"] == "subagent-tester-abcd1234"
        assert requested["parent_id"] == "main"
        assert granted["agent_id"] == "main"
        assert granted["parent_id"] is None

    async def test_unmapped_event_type_is_skipped(self, live_pipeline):
        await emitter.emit_event(EventType.AGENT_STATE_CHANGED, {"x": 1})
        await emitter.emit_event(EventType.SESSION_START, {"model": "m"})
        lines = await _read_lines(live_pipeline, 1)
        assert [line["event"] for line in lines] == ["session_start"]


class TestNeverBlocks:
    async def test_emit_stream_event_full_queues_return_fast(self):
        """Bounded stream queues must drop, not block, when processors wedge."""
        manager = UnifiedStatusManager(StatusDisplayConfig(max_queue_size=2))
        manager.running = True  # simulate wedged processors: running, not draining

        async def flood() -> int:
            dropped = 0
            for _ in range(50):
                ok = await manager.emit_stream_event(
                    _make_event(), StreamPriority.NORMAL
                )
                if not ok:
                    dropped += 1
            return dropped

        start = time.monotonic()
        dropped = await asyncio.wait_for(flood(), timeout=0.5)
        elapsed = time.monotonic() - start
        assert dropped == 48  # queue holds 2, the rest drop
        assert manager.metrics.events_dropped == 48
        assert elapsed < 0.25


def _make_event():
    from omnimancer.core.agent.status_core import AgentEvent

    return AgentEvent(event_type=EventType.OPERATION_PROGRESS, data={})
