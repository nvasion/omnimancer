"""Long-command tool_progress plumbing tests (WU-A6).

The progress callback must: not exist when the pipeline is down (keeping
the non-streaming execution path), throttle to its ceiling, and deliver
real subprocess output as tool_progress events end to end.
"""

import asyncio
import json
import time
from pathlib import Path

import pytest

from omnimancer.core.agent.types import OperationType as AgentOperationType
from omnimancer.core.agent_managers import ProgramExecutor
from omnimancer.core.models import EventsConfig
from omnimancer.events import emitter


@pytest.fixture
async def event_file(tmp_path):
    ok = await emitter.init_events(
        "sess-progress", "headless", EventsConfig(directory=str(tmp_path))
    )
    assert ok
    yield tmp_path / "sess-progress.jsonl"
    await emitter.shutdown_events()


async def _events_named(path: Path, name: str, minimum: int, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            events = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            matched = [event for event in events if event["event"] == name]
            if len(matched) >= minimum:
                return matched
        await asyncio.sleep(0.02)
    raise AssertionError(f"expected >= {minimum} {name} events in {path}")


class TestBuildProgressCallback:
    def test_none_without_operation_id(self, event_file):
        assert emitter.build_progress_callback(None) is None

    async def test_none_when_pipeline_down(self):
        assert emitter.build_progress_callback("op-1") is None

    async def test_throttles_to_ceiling(self, event_file):
        callback = emitter.build_progress_callback("op-throttle", max_per_sec=4.0)
        assert callback is not None
        # 50 rapid chunks inside one throttle window: only the first emits.
        for _ in range(50):
            callback("stdout", "x" * 10)
        events = await _events_named(event_file, "tool_progress", 1)
        assert len(events) == 1
        assert events[0]["data"]["stream"] == "stdout"
        assert events[0]["data"]["bytes"] == 10  # first chunk only
        assert events[0]["data"]["op_id"] == "op-throttle"


class TestCommandProgressEndToEnd:
    async def test_echo_produces_progress_and_result(self, event_file):
        executor = ProgramExecutor()
        op_id = await emitter.start_tool_operation(
            AgentOperationType.COMMAND_EXECUTE, "Execute: echo", {"tool": "Bash"}
        )
        assert op_id is not None
        result = await executor._execute_command(
            "echo fleet-progress-test", progress_op_id=op_id
        )
        assert result.success
        assert "fleet-progress-test" in result.data["stdout"]
        events = await _events_named(event_file, "tool_progress", 1)
        assert any(
            "fleet-progress-test" in event["data"].get("preview", "")
            for event in events
        )

    async def test_no_pipeline_keeps_nonstreaming_path(self):
        executor = ProgramExecutor()
        result = await executor._execute_command("echo plain-path")
        assert result.success
        assert "plain-path" in result.data["stdout"]
