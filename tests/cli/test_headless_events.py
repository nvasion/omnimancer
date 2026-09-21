"""Headless event-feed wiring + stream-json schema lock (WU-A5).

Two contracts pinned here:

1. The stream-json NDJSON lines fleet workers and codex-orchestrator parse
   keep their exact key sets — the event feed must never leak into stdout,
   and any key change must consciously update this test.
2. A HeadlessRunner run produces the session/turn lifecycle in the JSONL
   event file, even though stdout stays byte-compatible.
"""

import json
from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.headless import HeadlessOutputEmitter, HeadlessRunner, OutputFormat
from omnimancer.core.models import ChatResponse, Config, EventsConfig, ProviderConfig


class TestStreamJsonSchemaLock:
    """Exact key sets of every stream-json line type."""

    def _emitter(self) -> tuple:
        emitter = HeadlessOutputEmitter(
            OutputFormat.STREAM_JSON, "sess-lock", verbose=False
        )
        buf = StringIO()
        emitter._stdout = buf
        return emitter, buf

    def _keys(self, buf: StringIO) -> set:
        return set(json.loads(buf.getvalue().strip()))

    def test_init_keys(self):
        emitter, buf = self._emitter()
        emitter.emit_init("m")
        assert self._keys(buf) == {"type", "subtype", "model", "session_id"}

    def test_assistant_keys(self):
        emitter, buf = self._emitter()
        emitter.emit_assistant("hi", "m", "end_turn")
        line = json.loads(buf.getvalue().strip())
        assert set(line) == {"type", "message", "session_id"}
        assert set(line["message"]) == {"model", "content", "stop_reason"}

    def test_tool_use_keys(self):
        emitter, buf = self._emitter()
        emitter.emit_tool_use("Bash", {"command": "ls"})
        line = json.loads(buf.getvalue().strip())
        assert set(line) == {"type", "tool", "session_id"}
        assert set(line["tool"]) == {"name", "arguments"}

    def test_tool_result_keys(self):
        emitter, buf = self._emitter()
        emitter.emit_tool_result("Bash", "ok", None)
        line = json.loads(buf.getvalue().strip())
        assert set(line) == {"type", "tool", "session_id"}
        assert set(line["tool"]) == {"name", "content", "error"}

    def test_error_keys(self):
        emitter, buf = self._emitter()
        emitter.emit_error("boom")
        assert self._keys(buf) == {
            "type",
            "is_error",
            "message",
            "session_id",
            "model",
            "provider",
            "stop_cause",
            "resume_session_id",
        }

    def test_result_keys(self):
        emitter, buf = self._emitter()
        emitter.emit_result("hi", "m", {}, 0.0, "end_turn")
        assert self._keys(buf) == {
            "type",
            "subtype",
            "is_error",
            "result",
            "model",
            "provider",
            "num_turns",
            "usage",
            "total_cost_usd",
            "stop_reason",
            "stop_cause",
            "session_id",
        }


def _headless_config(events_dir) -> Config:
    return Config(
        default_provider="p",
        providers={"p": ProviderConfig(model="test-model")},
        storage_path="/tmp/omni-headless-events-test",
        events=EventsConfig(directory=str(events_dir)),
    )


def _mock_engine(events_dir) -> MagicMock:
    engine = MagicMock()
    engine.runtime_identity.return_value = ("p", "test-model")
    engine.config_manager.get_config.return_value = _headless_config(events_dir)
    engine.provider_supports_tools.return_value = True
    engine.send_message_with_tools = AsyncMock(
        return_value=ChatResponse(
            content="DONE",
            model_used="test-model",
            tokens_used=3,
            input_tokens=2,
            output_tokens=1,
            stop_reason="end_turn",
        )
    )
    engine._fire_hook = AsyncMock()
    return engine


class TestHeadlessRunnerEventFeed:
    @pytest.mark.asyncio
    async def test_lifecycle_events_written_stdout_clean(self, tmp_path, capsys):
        engine = _mock_engine(tmp_path)
        runner = HeadlessRunner(engine, OutputFormat.STREAM_JSON)
        status = await runner.run("say DONE")
        assert status == 0

        # Event feed: full session/turn lifecycle in the JSONL file.
        event_path = tmp_path / f"omn-{runner._turn_notifier.session_id}.jsonl"
        assert event_path.exists()
        events = [
            json.loads(line)
            for line in event_path.read_text().splitlines()
            if line.strip()
        ]
        names = [event["event"] for event in events]
        assert names[0] == "session_start"
        assert "turn_start" in names
        assert "turn_end" in names
        assert names[-1] == "session_end"
        by_name = {event["event"]: event for event in events}
        assert by_name["session_start"]["data"]["model"] == "test-model"
        assert by_name["session_start"]["data"]["read_only"] is False
        assert by_name["turn_start"]["data"]["prompt_preview"] == "say DONE"
        assert by_name["session_end"]["data"]["status"] == 0
        assert all(event["mode"] == "headless" for event in events)

        # stdout contract: only the existing stream-json types, no leakage.
        stdout_types = {
            json.loads(line)["type"]
            for line in capsys.readouterr().out.splitlines()
            if line.strip()
        }
        assert stdout_types <= {
            "system",
            "assistant",
            "tool_use",
            "tool_result",
            "result",
        }

    @pytest.mark.asyncio
    async def test_events_disabled_produces_no_file(self, tmp_path):
        engine = _mock_engine(tmp_path)
        config = _headless_config(tmp_path)
        config.events.enabled = False
        engine.config_manager.get_config.return_value = config
        runner = HeadlessRunner(engine, OutputFormat.STREAM_JSON)
        status = await runner.run("say DONE")
        assert status == 0
        assert not list(tmp_path.glob("*.jsonl"))

    @pytest.mark.asyncio
    async def test_session_start_reports_runtime_identity(self, tmp_path):
        engine = MagicMock()
        engine.runtime_identity.return_value = ("gateway", "qwen3.5-9b")
        engine.config_manager.get_config.return_value = _headless_config(tmp_path)
        engine.provider_supports_tools.return_value = True
        engine.send_message_with_tools = AsyncMock(
            return_value=ChatResponse(
                content="DONE",
                model_used="test-model",
                tokens_used=3,
                input_tokens=2,
                output_tokens=1,
                stop_reason="end_turn",
            )
        )
        engine._fire_hook = AsyncMock()
        runner = HeadlessRunner(engine, OutputFormat.STREAM_JSON)
        status = await runner.run("say DONE")
        assert status == 0

        event_path = tmp_path / f"omn-{runner._turn_notifier.session_id}.jsonl"
        events = [
            json.loads(line)
            for line in event_path.read_text().splitlines()
            if line.strip()
        ]
        by_name = {event["event"]: event for event in events}
        assert by_name["session_start"]["data"]["model"] == "qwen3.5-9b"
        assert by_name["session_start"]["data"]["provider"] == "gateway"
