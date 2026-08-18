"""Tests for headless checkpointing and resume (omn -p ... / omn --resume).

A rate-limited or interrupted headless run used to lose all progress — the
orchestrator had to restart the whole task, re-burning every input token. The
runner now checkpoints the conversation each iteration; a failed run exits
with a resumable status and `--resume <session-id>` continues where it left
off.
"""

import json
from datetime import datetime
from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.core.chat_manager import ChatManager
from omnimancer.core.models import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    ToolCall,
    ToolResultRecord,
)


@pytest.fixture(autouse=True)
def _checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIMANCER_CHECKPOINT_DIR", str(tmp_path))
    return tmp_path


class TestMessageSerialization:
    def test_round_trip_preserves_tool_fields(self):
        from omnimancer.cli.headless_checkpoint import (
            message_from_dict,
            message_to_dict,
        )

        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="doing it\n[Called tools: file_read({...})]",
            timestamp=datetime(2026, 8, 17, 12, 0, 0),
            model_used="claude-sonnet-4-6",
            tool_calls=[
                ToolCall(
                    name="file_read",
                    arguments={"path": "/a.py"},
                    id="call_1",
                    server_name="fs",
                )
            ],
            raw_content="doing it",
        )
        restored = message_from_dict(message_to_dict(msg))
        assert restored.role == MessageRole.ASSISTANT
        assert restored.content == msg.content
        assert restored.model_used == "claude-sonnet-4-6"
        assert restored.raw_content == "doing it"
        assert restored.tool_calls[0].name == "file_read"
        assert restored.tool_calls[0].arguments == {"path": "/a.py"}
        assert restored.tool_calls[0].id == "call_1"
        assert restored.tool_calls[0].server_name == "fs"

    def test_round_trip_preserves_tool_results(self):
        from omnimancer.cli.headless_checkpoint import (
            message_from_dict,
            message_to_dict,
        )

        msg = ChatMessage(
            role=MessageRole.USER,
            content="Tool results:\n\n[file_read] Result: ...",
            timestamp=datetime(2026, 8, 17, 12, 0, 1),
            model_used="claude-sonnet-4-6",
            tool_results=[ToolResultRecord(tool_call_id="call_1", content="data")],
        )
        restored = message_from_dict(message_to_dict(msg))
        assert restored.tool_results[0].tool_call_id == "call_1"
        assert restored.tool_results[0].content == "data"

    def test_plain_message_round_trip(self):
        from omnimancer.cli.headless_checkpoint import (
            message_from_dict,
            message_to_dict,
        )

        msg = ChatMessage(
            role=MessageRole.USER,
            content="hello",
            timestamp=datetime(2026, 8, 17, 12, 0, 2),
            model_used="m",
        )
        restored = message_from_dict(message_to_dict(msg))
        assert restored.tool_calls is None
        assert restored.tool_results is None
        assert restored.content == "hello"


class TestCheckpointStore:
    def _checkpoint(self, session_id="sess-abc"):
        from omnimancer.cli.headless_checkpoint import HeadlessCheckpoint

        return HeadlessCheckpoint(
            session_id=session_id,
            prompt="fix the bug",
            iteration=3,
            current_message="Tool results:\n\n[x] Result: y",
            messages=[
                {
                    "role": "user",
                    "content": "hi",
                    "timestamp": "2026-08-17T12:00:00",
                    "model_used": "m",
                }
            ],
            tool_log=[{"name": "file_read", "arguments": {}, "error": None}],
            no_tool_nudges=1,
            usage={"input_tokens": 100, "output_tokens": 50, "total_cost_usd": 0.01},
            provider="claude",
            model="claude-sonnet-4-6",
        )

    def test_save_and_load_round_trip(self, _checkpoint_dir):
        from omnimancer.cli.headless_checkpoint import load_checkpoint, save_checkpoint

        path = save_checkpoint(self._checkpoint())
        assert path.exists()
        assert path.parent == _checkpoint_dir

        loaded = load_checkpoint("sess-abc")
        assert loaded is not None
        assert loaded.prompt == "fix the bug"
        assert loaded.iteration == 3
        assert loaded.current_message.startswith("Tool results:")
        assert loaded.no_tool_nudges == 1
        assert loaded.usage["input_tokens"] == 100
        assert loaded.tool_log[0]["name"] == "file_read"

    def test_load_missing_returns_none(self):
        from omnimancer.cli.headless_checkpoint import load_checkpoint

        assert load_checkpoint("nope") is None

    def test_delete_checkpoint(self):
        from omnimancer.cli.headless_checkpoint import (
            delete_checkpoint,
            load_checkpoint,
            save_checkpoint,
        )

        save_checkpoint(self._checkpoint())
        delete_checkpoint("sess-abc")
        assert load_checkpoint("sess-abc") is None
        # Deleting again is a no-op, not an error.
        delete_checkpoint("sess-abc")

    def test_rejects_path_traversal_session_ids(self):
        from omnimancer.cli.headless_checkpoint import load_checkpoint

        assert load_checkpoint("../../etc/passwd") is None
        assert load_checkpoint("a/b") is None

    def test_corrupt_file_returns_none(self, _checkpoint_dir):
        from omnimancer.cli.headless_checkpoint import load_checkpoint

        (_checkpoint_dir / "bad.json").write_text("{not json")
        assert load_checkpoint("bad") is None


def _mock_engine(responses):
    engine = MagicMock()
    engine.runtime_identity.return_value = ("claude", "test-model")
    engine.provider_supports_tools = MagicMock(return_value=True)
    engine.provider_supports_native_tool_history = MagicMock(return_value=False)
    engine.chat_manager = ChatManager()
    if isinstance(responses, list):
        engine.send_message_with_tools = AsyncMock(side_effect=responses)
    else:
        engine.send_message_with_tools = AsyncMock(return_value=responses)
    engine.agent_engine = MagicMock()
    return engine


_RATE_LIMITED = ChatResponse(
    content="",
    model_used="",
    tokens_used=0,
    error="Claude API rate limit exceeded",
)

_DONE = ChatResponse(
    content="All finished.\nDONE",
    model_used="test-model",
    tokens_used=10,
    input_tokens=6,
    output_tokens=4,
    stop_reason="end_turn",
)


class TestRunnerCheckpointing:
    @pytest.mark.asyncio
    async def test_rate_limit_exits_4_and_keeps_checkpoint(self, _checkpoint_dir):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat
        from omnimancer.cli.headless_checkpoint import load_checkpoint

        engine = _mock_engine(_RATE_LIMITED)
        runner = HeadlessRunner(engine=engine, output_format=OutputFormat.JSON)
        stdout, stderr = StringIO(), StringIO()
        runner._emitter._stdout = stdout
        runner._emitter._stderr = stderr

        exit_code = await runner.run("do the thing")

        assert exit_code == 4
        blob = json.loads(stdout.getvalue().strip())
        assert blob["is_error"] is True
        assert blob["stop_cause"] == "rate_limited"
        session_id = blob["resume_session_id"]
        assert session_id
        assert load_checkpoint(session_id) is not None
        assert "--resume" in stderr.getvalue()

    @pytest.mark.asyncio
    async def test_generic_error_still_exits_1_with_checkpoint(self):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        engine = _mock_engine(
            ChatResponse(
                content="", model_used="", tokens_used=0, error="server exploded"
            )
        )
        runner = HeadlessRunner(engine=engine, output_format=OutputFormat.JSON)
        stdout = StringIO()
        runner._emitter._stdout = stdout
        runner._emitter._stderr = StringIO()

        exit_code = await runner.run("do the thing")

        assert exit_code == 1
        blob = json.loads(stdout.getvalue().strip())
        assert blob["stop_cause"] == "error"

    @pytest.mark.asyncio
    async def test_clean_completion_deletes_checkpoint(self, _checkpoint_dir):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        engine = _mock_engine(_DONE)
        runner = HeadlessRunner(engine=engine, output_format=OutputFormat.TEXT)
        runner._emitter._stdout = StringIO()
        runner._emitter._stderr = StringIO()

        exit_code = await runner.run("do the thing")

        assert exit_code == 0
        assert list(_checkpoint_dir.glob("*.json")) == []

    @pytest.mark.asyncio
    async def test_max_iterations_keeps_checkpoint(self, _checkpoint_dir):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        chatty = ChatResponse(
            content="working...",
            model_used="test-model",
            tokens_used=5,
            stop_reason="tool_use",
            tool_calls=[ToolCall(name="file_read", arguments={"path": "/a"})],
        )
        engine = _mock_engine([chatty] * 2)
        engine.agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True, data="x", error=None, was_cancelled=False
            )
        )
        runner = HeadlessRunner(
            engine=engine, output_format=OutputFormat.TEXT, max_iterations=2
        )
        runner._emitter._stdout = StringIO()
        runner._emitter._stderr = StringIO()

        exit_code = await runner.run("do the thing")

        assert exit_code == 3
        assert len(list(_checkpoint_dir.glob("*.json"))) == 1

    @pytest.mark.asyncio
    async def test_resume_restores_context_and_continues(self, _checkpoint_dir):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat
        from omnimancer.cli.headless_checkpoint import (
            HeadlessCheckpoint,
            load_checkpoint,
            save_checkpoint,
        )

        save_checkpoint(
            HeadlessCheckpoint(
                session_id="resume-me",
                prompt="original task",
                iteration=2,
                current_message="Tool results:\n\n[file_read] Result: content",
                messages=[
                    {
                        "role": "user",
                        "content": "AGENTPROMPT\n\nUser: original task",
                        "timestamp": "2026-08-17T12:00:00",
                        "model_used": "test-model",
                    },
                    {
                        "role": "assistant",
                        "content": "reading\n[Called tools: file_read({})]",
                        "timestamp": "2026-08-17T12:00:01",
                        "model_used": "test-model",
                        "tool_calls": [
                            {"name": "file_read", "arguments": {}, "id": "c1"}
                        ],
                        "raw_content": "reading",
                    },
                ],
                tool_log=[{"name": "file_read", "arguments": {}, "error": None}],
                no_tool_nudges=0,
                usage={
                    "input_tokens": 40,
                    "output_tokens": 20,
                    "total_cost_usd": 0.004,
                },
            )
        )

        engine = _mock_engine(_DONE)
        runner = HeadlessRunner(
            engine=engine,
            output_format=OutputFormat.JSON,
            resume_session_id="resume-me",
        )
        stdout = StringIO()
        runner._emitter._stdout = stdout
        runner._emitter._stderr = StringIO()

        exit_code = await runner.run("")

        assert exit_code == 0
        # The pending tool-results message was re-sent, not the agent prompt.
        sent = engine.send_message_with_tools.call_args_list[0][0][0]
        assert sent.startswith("Tool results:")
        # Restored history was placed in the chat context before sending.
        roles = [m.role for m in engine.chat_manager.current_context.messages]
        assert roles[0] == MessageRole.USER
        assert roles[1] == MessageRole.ASSISTANT
        # Token totals carried over from the checkpoint.
        blob = json.loads(stdout.getvalue().strip())
        assert blob["usage"]["input_tokens"] == 40 + (_DONE.input_tokens or 0)
        # Clean completion removes the checkpoint.
        assert load_checkpoint("resume-me") is None

    @pytest.mark.asyncio
    async def test_resume_with_missing_checkpoint_errors(self):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        engine = _mock_engine(_DONE)
        runner = HeadlessRunner(
            engine=engine,
            output_format=OutputFormat.JSON,
            resume_session_id="ghost",
        )
        stdout = StringIO()
        runner._emitter._stdout = stdout
        runner._emitter._stderr = StringIO()

        exit_code = await runner.run("")

        assert exit_code == 1
        assert engine.send_message_with_tools.await_count == 0
        blob = json.loads(stdout.getvalue().strip())
        assert "checkpoint" in blob["error"].lower()

    @pytest.mark.asyncio
    async def test_checkpointing_disabled_via_env(self, _checkpoint_dir, monkeypatch):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        monkeypatch.setenv("OMNIMANCER_CHECKPOINT", "0")
        engine = _mock_engine(_RATE_LIMITED)
        runner = HeadlessRunner(engine=engine, output_format=OutputFormat.JSON)
        runner._emitter._stdout = StringIO()
        runner._emitter._stderr = StringIO()

        exit_code = await runner.run("do the thing")

        assert exit_code == 4
        assert list(_checkpoint_dir.glob("*.json")) == []
