"""Regression tests for the Phase A review findings (gpt-5.6-sol, cycle 1).

Each test pins one fix: cancellation closes the event lifecycle, bus
shutdown emits terminal events, retention cleanup is scoped to session
files, event files are owner-only, threadsafe emission keeps subagent
parentage, and free-text payload fields are clipped.
"""

import asyncio
import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omnimancer.core.agent.status_core import EventType
from omnimancer.core.agent.types import OperationType as AgentOperationType
from omnimancer.core.models import EventsConfig
from omnimancer.events import emitter
from omnimancer.events.jsonl_writer import cleanup_old_files
from tests.events.test_agent_engine_events import _engine, _op


@pytest.fixture
async def event_file(tmp_path):
    ok = await emitter.init_events(
        "sess-fixes", "headless", EventsConfig(directory=str(tmp_path))
    )
    assert ok
    yield tmp_path / "omn-sess-fixes.jsonl"
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
    raise AssertionError(f"expected {expected} events in {path}")


class TestCancellationLifecycle:
    async def test_cancelled_turn_emits_terminal_tool_end(self, event_file):
        engine = _engine()
        engine._execute_operation = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await engine.execute_with_approval(_op())
        events = await _events(event_file, 2)
        end = events[-1]
        assert end["event"] == "tool_end"
        assert end["data"]["was_cancelled"] is True

    async def test_bus_shutdown_closes_active_operations(self, tmp_path):
        ok = await emitter.init_events(
            "sess-shutdown", "headless", EventsConfig(directory=str(tmp_path))
        )
        assert ok
        op_id = await emitter.start_tool_operation(
            AgentOperationType.COMMAND_EXECUTE, "long", {"tool": "Bash"}
        )
        assert op_id is not None
        # Shut down with the operation still open: a terminal cancelled
        # tool_end must land before the pipeline stops.
        await emitter.shutdown_events()
        path = tmp_path / "omn-sess-shutdown.jsonl"
        lines = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        ends = [line for line in lines if line["event"] == "tool_end"]
        assert len(ends) == 1
        assert ends[0]["data"]["was_cancelled"] is True
        assert ends[0]["data"]["op_id"] == op_id


class TestRetentionScope:
    def test_cleanup_only_touches_omn_namespaced_names(self, tmp_path):
        old = time.time() - 10 * 24 * 3600
        session = tmp_path / "omn-aaaabbbb-cccc-dddd-eeee-ffff00001111.jsonl"
        # Foreign data AND un-prefixed uuid files (another app could name
        # files by bare uuid too) must both survive.
        foreign = tmp_path / "user-data.jsonl"
        bare_uuid = tmp_path / "aaaabbbb-cccc-dddd-eeee-ffff00001111.jsonl"
        for path in (session, foreign, bare_uuid):
            path.write_text("{}\n")
            os.utime(path, (old, old))
        deleted = cleanup_old_files(
            tmp_path, retention_days=7, name_re=emitter.SESSION_FILE_RE
        )
        assert deleted == 1
        assert not session.exists()
        assert foreign.exists()
        assert bare_uuid.exists()


class TestFilePermissions:
    async def test_event_file_and_dir_are_owner_only(self, tmp_path, event_file):
        await emitter.emit_event(EventType.TURN_START, {"turn": 1})
        await _events(event_file, 1)
        assert stat.S_IMODE(event_file.stat().st_mode) == 0o600


class TestThreadsafeParentage:
    async def test_threadsafe_emit_carries_parent_id(self, event_file):
        with emitter.agent_context("subagent-x-11112222", "main"):
            emitter.emit_event_threadsafe(
                EventType.OPERATION_PROGRESS,
                {"stream": "stdout", "bytes": 1},
                operation_id="op-1",
            )
        events = await _events(event_file, 1)
        assert events[0]["agent_id"] == "subagent-x-11112222"
        assert events[0]["parent_id"] == "main"


class TestPayloadClipping:
    async def test_target_and_error_are_clipped(self, event_file):
        engine = _engine()
        long_command = "echo " + "a" * 1000
        result = await engine.execute_with_approval(_op(long_command))
        assert result.success
        events = await _events(event_file, 2)
        start = events[0]
        assert len(start["data"]["target"]) <= 200
        assert len(start["data"]["description"]) <= 200
