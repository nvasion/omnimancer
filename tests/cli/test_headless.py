"""Tests for headless pipe mode (omn -p)."""

import json
from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.core.models import ChatResponse, ToolCall


class TestChatResponseNewFields:
    """Test ChatResponse token split and stop_reason fields."""

    def test_new_fields_default_to_none(self):
        resp = ChatResponse(content="hi", model_used="test", tokens_used=10)
        assert resp.input_tokens is None
        assert resp.output_tokens is None
        assert resp.stop_reason is None

    def test_new_fields_set_correctly(self):
        resp = ChatResponse(
            content="hi",
            model_used="test",
            tokens_used=150,
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
        )
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50
        assert resp.stop_reason == "end_turn"

    def test_backward_compat_is_success(self):
        resp = ChatResponse(
            content="hi",
            model_used="test",
            tokens_used=10,
            input_tokens=5,
            output_tokens=5,
            stop_reason="end_turn",
        )
        assert resp.is_success is True

    def test_backward_compat_error(self):
        resp = ChatResponse(
            content="",
            model_used="",
            tokens_used=0,
            error="fail",
            stop_reason="error",
        )
        assert resp.is_success is False


class TestOutputFormat:
    """Test OutputFormat enum."""

    def test_enum_values(self):
        from omnimancer.cli.headless import OutputFormat

        assert OutputFormat.TEXT.value == "text"
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.STREAM_JSON.value == "stream-json"


class TestTokenAccumulator:
    """Test cumulative token tracking."""

    def test_accumulates_tokens(self):
        from omnimancer.cli.headless import TokenAccumulator

        acc = TokenAccumulator()
        acc.add(
            ChatResponse(
                content="a",
                model_used="m",
                tokens_used=10,
                input_tokens=6,
                output_tokens=4,
                cost_estimate=0.001,
            )
        )
        acc.add(
            ChatResponse(
                content="b",
                model_used="m",
                tokens_used=20,
                input_tokens=12,
                output_tokens=8,
                cost_estimate=0.002,
            )
        )

        total = acc.total
        assert total["input_tokens"] == 18
        assert total["output_tokens"] == 12
        assert abs(total["total_cost_usd"] - 0.003) < 1e-9

    def test_handles_none_token_fields(self):
        from omnimancer.cli.headless import TokenAccumulator

        acc = TokenAccumulator()
        acc.add(ChatResponse(content="a", model_used="m", tokens_used=10))
        acc.add(ChatResponse(content="b", model_used="m", tokens_used=20))

        total = acc.total
        assert total["input_tokens"] == 0
        assert total["output_tokens"] == 0

    def test_empty_accumulator(self):
        from omnimancer.cli.headless import TokenAccumulator

        acc = TokenAccumulator()
        total = acc.total
        assert total["input_tokens"] == 0
        assert total["output_tokens"] == 0
        assert total["total_cost_usd"] == 0.0


class TestHeadlessOutputEmitterText:
    """Test text output format."""

    def test_emit_assistant_prints_to_stdout(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(OutputFormat.TEXT, "sess-1", verbose=False)
        emitter._stdout = buf

        emitter.emit_assistant("Hello world", "claude", "end_turn")
        assert "Hello world" in buf.getvalue()

    def test_emit_result_prints_text(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(OutputFormat.TEXT, "sess-1", verbose=False)
        emitter._stdout = buf

        emitter.emit_result(
            "Final answer",
            "claude",
            {"input_tokens": 10, "output_tokens": 5},
            0.001,
            "end_turn",
        )
        assert "Final answer" in buf.getvalue()

    def test_tool_events_hidden_without_verbose(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(OutputFormat.TEXT, "sess-1", verbose=False)
        emitter._stdout = buf

        emitter.emit_tool_use("file_read", {"path": "/x"})
        emitter.emit_tool_result("file_read", "contents", None)
        assert buf.getvalue() == ""

    def test_tool_events_shown_with_verbose(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(OutputFormat.TEXT, "sess-1", verbose=True)
        emitter._stdout = buf

        emitter.emit_tool_use("file_read", {"path": "/x"})
        assert "file_read" in buf.getvalue()


class TestHeadlessOutputEmitterJSON:
    """Test JSON output format."""

    def test_emit_result_produces_valid_json(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(OutputFormat.JSON, "sess-1", verbose=False)
        emitter._stdout = buf

        emitter.emit_assistant("intermediate", "claude", None)
        emitter.emit_result(
            "Final",
            "claude",
            {"input_tokens": 10, "output_tokens": 5},
            0.001,
            "end_turn",
        )

        output = json.loads(buf.getvalue())
        assert output["result"] == "Final"
        assert output["session_id"] == "sess-1"
        assert output["model"] == "claude"
        assert output["provider"] == ""
        assert output["usage"]["input_tokens"] == 10
        assert output["total_cost_usd"] == 0.001
        assert output["stop_reason"] == "end_turn"

    def test_json_mode_suppresses_intermediate_output(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(OutputFormat.JSON, "sess-1", verbose=False)
        emitter._stdout = buf

        emitter.emit_assistant("thinking...", "claude", None)
        emitter.emit_tool_use("file_read", {"path": "/x"})
        emitter.emit_tool_result("file_read", "data", None)

        assert buf.getvalue() == ""


class TestHeadlessOutputEmitterStreamJSON:
    """Test stream-json (NDJSON) output format."""

    def test_emit_init_writes_ndjson(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(
            OutputFormat.STREAM_JSON, "sess-1", verbose=False
        )
        emitter._stdout = buf

        emitter.emit_init("claude-sonnet")

        line = json.loads(buf.getvalue().strip())
        assert line["type"] == "system"
        assert line["subtype"] == "init"
        assert line["session_id"] == "sess-1"
        assert line["model"] == "claude-sonnet"

    def test_emit_assistant_writes_ndjson(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(
            OutputFormat.STREAM_JSON, "sess-1", verbose=False
        )
        emitter._stdout = buf

        emitter.emit_assistant("Hello", "claude", "end_turn")

        line = json.loads(buf.getvalue().strip())
        assert line["type"] == "assistant"
        assert line["message"]["content"] == "Hello"
        assert line["message"]["model"] == "claude"

    def test_emit_tool_use_writes_ndjson(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(
            OutputFormat.STREAM_JSON, "sess-1", verbose=False
        )
        emitter._stdout = buf

        emitter.emit_tool_use("file_read", {"path": "/src/main.py"})

        line = json.loads(buf.getvalue().strip())
        assert line["type"] == "tool_use"
        assert line["tool"]["name"] == "file_read"
        assert line["tool"]["arguments"]["path"] == "/src/main.py"

    def test_emit_tool_result_writes_ndjson(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(
            OutputFormat.STREAM_JSON, "sess-1", verbose=False
        )
        emitter._stdout = buf

        emitter.emit_tool_result("file_read", "print('hi')", None)

        line = json.loads(buf.getvalue().strip())
        assert line["type"] == "tool_result"
        assert line["tool"]["name"] == "file_read"
        assert line["tool"]["content"] == "print('hi')"
        assert line["tool"]["error"] is None

    def test_emit_tool_result_with_error(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(
            OutputFormat.STREAM_JSON, "sess-1", verbose=False
        )
        emitter._stdout = buf

        emitter.emit_tool_result("file_read", "", "File not found")

        line = json.loads(buf.getvalue().strip())
        assert line["tool"]["error"] == "File not found"

    def test_emit_result_writes_final_ndjson(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(
            OutputFormat.STREAM_JSON, "sess-1", verbose=False
        )
        emitter._stdout = buf

        emitter.emit_result(
            "Done",
            "claude",
            {"input_tokens": 50, "output_tokens": 25},
            0.005,
            "end_turn",
        )

        line = json.loads(buf.getvalue().strip())
        assert line["type"] == "result"
        assert line["result"] == "Done"
        assert line["usage"]["input_tokens"] == 50

    def test_session_id_in_all_events(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        buf = StringIO()
        emitter = HeadlessOutputEmitter(
            OutputFormat.STREAM_JSON, "sess-42", verbose=False
        )
        emitter._stdout = buf

        emitter.emit_init("m")
        emitter.emit_assistant("hi", "m", "end_turn")
        emitter.emit_tool_use("t", {})
        emitter.emit_tool_result("t", "ok", None)
        emitter.emit_result("done", "m", {}, 0, "end_turn")

        lines = [json.loads(line) for line in buf.getvalue().strip().split("\n")]
        assert len(lines) == 5
        for line in lines:
            assert line["session_id"] == "sess-42"


class TestHeadlessOutputEmitterError:
    """Test error emission."""

    def test_emit_error_to_stderr(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        stderr_buf = StringIO()
        emitter = HeadlessOutputEmitter(OutputFormat.TEXT, "sess-1", verbose=False)
        emitter._stderr = stderr_buf

        emitter.emit_error("Something broke")
        assert "Something broke" in stderr_buf.getvalue()

    def test_stream_json_emit_error_to_both(self):
        from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

        stdout_buf = StringIO()
        stderr_buf = StringIO()
        emitter = HeadlessOutputEmitter(
            OutputFormat.STREAM_JSON, "sess-1", verbose=False
        )
        emitter._stdout = stdout_buf
        emitter._stderr = stderr_buf

        emitter.emit_error("API failed")

        line = json.loads(stdout_buf.getvalue().strip())
        assert line["type"] == "error"
        assert "API failed" in stderr_buf.getvalue()


class TestHeadlessRunner:
    """Test the headless execution runner."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(
            return_value=ChatResponse(
                content="Hello from the AI",
                model_used="test-model",
                tokens_used=15,
                input_tokens=10,
                output_tokens=5,
                stop_reason="end_turn",
                tool_calls=None,
            )
        )

        mock_agent_engine = MagicMock()
        mock_engine.agent_engine = mock_agent_engine

        stdout_buf = StringIO()
        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.TEXT,
            no_approval=True,
            verbose=False,
        )
        runner._emitter._stdout = stdout_buf

        exit_code = await runner.run("say hello")
        assert exit_code == 0
        assert "Hello from the AI" in stdout_buf.getvalue()

    @pytest.mark.asyncio
    async def test_tool_call_round_trip(self):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        first_response = ChatResponse(
            content="I'll read the file.",
            model_used="test-model",
            tokens_used=20,
            input_tokens=15,
            output_tokens=5,
            stop_reason="tool_use",
            tool_calls=[ToolCall(name="file_read", arguments={"path": "/main.py"})],
        )
        second_response = ChatResponse(
            content="The file contains a hello world program.",
            model_used="test-model",
            tokens_used=30,
            input_tokens=20,
            output_tokens=10,
            stop_reason="end_turn",
            tool_calls=None,
        )
        done_response = ChatResponse(
            content="DONE",
            model_used="test-model",
            tokens_used=2,
            stop_reason="end_turn",
            tool_calls=None,
        )

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(
            side_effect=[first_response, second_response, done_response]
        )

        mock_agent_engine = MagicMock()
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True,
                data="print('hello')",
                error=None,
                was_cancelled=False,
            )
        )
        mock_engine.agent_engine = mock_agent_engine

        stdout_buf = StringIO()
        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.STREAM_JSON,
            no_approval=True,
            verbose=False,
        )
        runner._emitter._stdout = stdout_buf

        exit_code = await runner.run("read main.py")
        assert exit_code == 0

        lines = [json.loads(line) for line in stdout_buf.getvalue().strip().split("\n")]
        types = [line["type"] for line in lines]
        assert "system" in types
        assert "tool_use" in types
        assert "tool_result" in types
        assert "result" in types

    @pytest.mark.asyncio
    async def test_native_tool_history_branch(self):
        """Native providers get structured results in headless mode too."""
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        first = ChatResponse(
            content="",
            model_used="test-model",
            tokens_used=20,
            stop_reason="tool_use",
            tool_calls=[
                ToolCall(name="file_read", arguments={"path": "/a"}, id="call_7")
            ],
        )
        final = ChatResponse(
            content="Done.",
            model_used="test-model",
            tokens_used=10,
            stop_reason="end_turn",
            tool_calls=None,
        )

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.provider_supports_native_tool_history = MagicMock(return_value=True)
        mock_engine.record_tool_results = MagicMock()
        mock_engine.send_message_with_tools = AsyncMock(side_effect=[first, final])

        mock_agent_engine = MagicMock()
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True, data="contents", error=None, was_cancelled=False
            )
        )
        mock_engine.agent_engine = mock_agent_engine

        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.TEXT,
            no_approval=True,
            verbose=False,
        )
        runner._emitter._stdout = StringIO()

        exit_code = await runner.run("read a")
        assert exit_code == 0

        mock_engine.record_tool_results.assert_called_once()
        _, records = mock_engine.record_tool_results.call_args[0]
        assert records[0].tool_call_id == "call_7"
        assert mock_engine.send_message_with_tools.call_args_list[1][0][0] == ""

    @pytest.mark.asyncio
    async def test_text_mimicked_tool_call_is_recovered(self):
        """A '[Called tools: ...]' emitted as text still executes (headless)."""
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        mimicked = ChatResponse(
            content='[Called tools: file_read({"path": "/main.py"})]',
            model_used="test-model",
            tokens_used=20,
            stop_reason="end_turn",
            tool_calls=None,
        )
        final = ChatResponse(
            content="Done.",
            model_used="test-model",
            tokens_used=10,
            stop_reason="end_turn",
            tool_calls=None,
        )

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(side_effect=[mimicked, final])

        mock_agent_engine = MagicMock()
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True,
                data="print('hello')",
                error=None,
                was_cancelled=False,
            )
        )
        mock_engine.agent_engine = mock_agent_engine

        stdout_buf = StringIO()
        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.STREAM_JSON,
            no_approval=True,
            verbose=False,
        )
        runner._emitter._stdout = stdout_buf

        exit_code = await runner.run("read main.py")
        assert exit_code == 0

        mock_agent_engine.execute_with_approval.assert_called_once()
        assert mock_engine.send_message_with_tools.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_results_labeled_with_arguments(self):
        """Results fed back to the model identify the call they belong to."""
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        first_response = ChatResponse(
            content="",
            model_used="test-model",
            tokens_used=20,
            stop_reason="tool_use",
            tool_calls=[ToolCall(name="file_read", arguments={"path": "/main.py"})],
        )
        second_response = ChatResponse(
            content="Done.",
            model_used="test-model",
            tokens_used=10,
            stop_reason="end_turn",
            tool_calls=None,
        )

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(
            side_effect=[first_response, second_response]
        )
        mock_agent_engine = MagicMock()
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True, data="contents", error=None, was_cancelled=False
            )
        )
        mock_engine.agent_engine = mock_agent_engine

        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.TEXT,
            no_approval=True,
            verbose=False,
        )
        runner._emitter._stdout = StringIO()

        await runner.run("read main.py")

        results_message = mock_engine.send_message_with_tools.call_args_list[1][0][0]
        assert "/main.py" in results_message

    @pytest.mark.asyncio
    async def test_json_output_format(self):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(
            return_value=ChatResponse(
                content="Answer",
                model_used="claude",
                tokens_used=15,
                input_tokens=10,
                output_tokens=5,
                cost_estimate=0.001,
                stop_reason="end_turn",
                tool_calls=None,
            )
        )
        mock_engine.agent_engine = MagicMock()

        stdout_buf = StringIO()
        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.JSON,
            no_approval=True,
            verbose=False,
        )
        runner._emitter._stdout = stdout_buf

        exit_code = await runner.run("question")
        assert exit_code == 0

        output = json.loads(stdout_buf.getvalue())
        assert output["result"] == "Answer"
        assert output["model"] == "claude"
        assert output["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_engine_error_returns_exit_code_1(self):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(
            return_value=ChatResponse(
                content="",
                model_used="",
                tokens_used=0,
                error="Rate limit exceeded",
            )
        )
        mock_engine.agent_engine = MagicMock()

        stderr_buf = StringIO()
        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.TEXT,
            no_approval=True,
            verbose=False,
        )
        runner._emitter._stderr = stderr_buf

        exit_code = await runner.run("test")
        assert exit_code == 1
        assert "Rate limit" in stderr_buf.getvalue()

    @pytest.mark.asyncio
    async def test_json_error_is_structured_json(self):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(
            return_value=ChatResponse(
                content="", model_used="", tokens_used=0, error="context overflow"
            )
        )
        mock_engine.agent_engine = MagicMock()

        stdout_buf = StringIO()
        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.JSON,
            no_approval=True,
        )
        runner._emitter._stdout = stdout_buf

        exit_code = await runner.run("question")
        assert exit_code == 1
        # stdout must be valid JSON in json mode, even on error.
        payload = json.loads(stdout_buf.getvalue())
        assert payload["is_error"] is True
        assert payload["error"] == "context overflow"
        assert payload["type"] == "result"

    @pytest.mark.asyncio
    async def test_json_result_includes_tool_calls(self):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        responses = [
            ChatResponse(
                content="Looking...",
                model_used="claude",
                tokens_used=5,
                stop_reason="tool_use",
                tool_calls=[ToolCall(name="find_files", arguments={"pattern": "*.py"})],
            ),
            ChatResponse(
                content="Here is the explanation.",
                model_used="claude",
                tokens_used=8,
                stop_reason="end_turn",
                tool_calls=None,
            ),
            ChatResponse(
                content="DONE",
                model_used="claude",
                tokens_used=2,
                stop_reason="end_turn",
                tool_calls=None,
            ),
        ]

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(side_effect=responses)
        mock_agent_engine = MagicMock()
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True, data="a.py\nb.py", error=None, was_cancelled=False
            )
        )
        mock_engine.agent_engine = mock_agent_engine

        stdout_buf = StringIO()
        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.JSON,
            no_approval=True,
        )
        runner._emitter._stdout = stdout_buf

        exit_code = await runner.run("explain this repo")
        assert exit_code == 0
        payload = json.loads(stdout_buf.getvalue())
        # The bare DONE acknowledgment must not replace the real summary.
        assert payload["result"] == "Here is the explanation."
        assert payload["num_turns"] == 3
        assert len(payload["tool_calls"]) == 1
        assert payload["tool_calls"][0]["name"] == "find_files"

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        import json

        from omnimancer.cli.headless import HeadlessRunner, OutputFormat
        from omnimancer.cli.tool_handler import MAX_TOOL_ITERATIONS

        # Distinct args each call so loop-detection does not trip; this tests
        # the hard iteration cap.
        def make_response(*_args, **_kwargs):
            make_response.n += 1
            return ChatResponse(
                content="Working...",
                model_used="test",
                tokens_used=10,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        name="Read", arguments={"file_path": f"/x{make_response.n}"}
                    )
                ],
            )

        make_response.n = 0

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(side_effect=make_response)

        mock_agent_engine = MagicMock()
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True,
                data="ok",
                error=None,
                was_cancelled=False,
            )
        )
        mock_engine.agent_engine = mock_agent_engine

        stdout_buf = StringIO()
        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.JSON,
            no_approval=True,
            verbose=False,
        )
        runner._emitter._stdout = stdout_buf

        exit_code = await runner.run("loop")
        assert exit_code == 3
        assert mock_engine.send_message_with_tools.call_count == MAX_TOOL_ITERATIONS

        # Verify the emitted JSON result line contains stop_cause
        stdout_content = runner._emitter._stdout.getvalue()
        result_line = stdout_content.strip()
        result_data = json.loads(result_line)
        assert result_data["stop_cause"] == "max_iterations"

    @pytest.mark.asyncio
    async def test_max_iterations_override(self):
        """--max-iterations raises (or lowers) the headless cap."""
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        def make_response(*_args, **_kwargs):
            make_response.n += 1
            return ChatResponse(
                content="Working...",
                model_used="test",
                tokens_used=10,
                stop_reason="tool_use",
                tool_calls=[
                    ToolCall(
                        name="Read", arguments={"file_path": f"/y{make_response.n}"}
                    )
                ],
            )

        make_response.n = 0

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(side_effect=make_response)

        mock_agent_engine = MagicMock()
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True, data="ok", error=None, was_cancelled=False
            )
        )
        mock_engine.agent_engine = mock_agent_engine

        runner = HeadlessRunner(
            engine=mock_engine,
            output_format=OutputFormat.TEXT,
            no_approval=True,
            verbose=False,
            max_iterations=2,
        )
        runner._emitter._stdout = StringIO()

        await runner.run("loop")
        assert mock_engine.send_message_with_tools.call_count == 2

    @pytest.mark.asyncio
    async def test_no_approval_flag_auto_approves_operations(self):
        """no_approval must install an auto-approve callback on the engine.

        Regression: HeadlessRunner stored no_approval but never used it. No
        approval callback exists in headless mode, and ApprovalManager denies
        by default without one, so every write/exec failed with "Operation
        not approved by user" — even when spawned with
        --dangerously-skip-permissions.
        """
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat
        from omnimancer.core.agent.types import Operation, OperationType
        from omnimancer.core.agent_managers import ApprovalManager

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(
            return_value=ChatResponse(
                content="done",
                model_used="m",
                tokens_used=1,
                stop_reason="end_turn",
                tool_calls=None,
            )
        )
        agent_engine = MagicMock()
        agent_engine.approval = ApprovalManager()
        mock_engine.agent_engine = agent_engine

        runner = HeadlessRunner(
            engine=mock_engine, output_format=OutputFormat.TEXT, no_approval=True
        )
        runner._emitter._stdout = StringIO()
        assert await runner.run("hi") == 0

        op = Operation(
            type=OperationType.FILE_WRITE,
            description="write a file",
            data={"path": "/x"},
            requires_approval=True,
        )
        assert await agent_engine.approval.request_approval(op) is True
        # The enhanced approval path must be covered too.
        agent_engine.enhanced_approval.set_approval_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_without_no_approval_operations_still_denied(self):
        """Without the flag, headless keeps its deny-by-default behavior."""
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat
        from omnimancer.core.agent.types import Operation, OperationType
        from omnimancer.core.agent_managers import ApprovalManager

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(
            return_value=ChatResponse(
                content="done",
                model_used="m",
                tokens_used=1,
                stop_reason="end_turn",
                tool_calls=None,
            )
        )
        agent_engine = MagicMock()
        agent_engine.approval = ApprovalManager()
        mock_engine.agent_engine = agent_engine

        runner = HeadlessRunner(
            engine=mock_engine, output_format=OutputFormat.TEXT, no_approval=False
        )
        runner._emitter._stdout = StringIO()
        assert await runner.run("hi") == 0

        op = Operation(
            type=OperationType.FILE_WRITE,
            description="write a file",
            data={"path": "/x"},
            requires_approval=True,
        )
        assert await agent_engine.approval.request_approval(op) is False

    @pytest.mark.asyncio
    async def test_repeated_tool_call_stops_early(self):
        import json

        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        # Same tool call every time → executed twice, nudged twice, then
        # the run stops on the 5th occurrence rather than running to the cap.
        repeated = ChatResponse(
            content="",
            model_used="test",
            tokens_used=1,
            stop_reason="tool_use",
            tool_calls=[ToolCall(name="Write", arguments={"file_path": "/x"})],
        )

        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(return_value=repeated)
        mock_agent_engine = MagicMock()
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True, data="ok", error=None, was_cancelled=False
            )
        )
        mock_engine.agent_engine = mock_agent_engine

        runner = HeadlessRunner(
            engine=mock_engine, output_format=OutputFormat.JSON, no_approval=True
        )
        runner._emitter._stdout = StringIO()

        exit_code = await runner.run("loop")
        assert exit_code == 3
        assert mock_engine.send_message_with_tools.call_count == 5
        # Only the first two occurrences actually executed.
        assert mock_agent_engine.execute_with_approval.call_count == 2

        # Verify the emitted JSON result line contains stop_cause
        stdout_content = runner._emitter._stdout.getvalue()
        result_line = stdout_content.strip()
        result_data = json.loads(result_line)
        assert result_data["stop_cause"] == "repeat_abort"


class TestNoToolCallNudge:
    """A tool-less narration turn must nudge the model, not end the run."""

    @staticmethod
    def _mock_engine(responses):
        mock_engine = MagicMock()
        mock_engine.runtime_identity.return_value = ("p", "test-model")
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.send_message_with_tools = AsyncMock(side_effect=responses)
        mock_agent_engine = MagicMock()
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=MagicMock(
                success=True, data="ok", error=None, was_cancelled=False
            )
        )
        mock_engine.agent_engine = mock_agent_engine
        return mock_engine, mock_agent_engine

    @pytest.mark.asyncio
    async def test_narration_turn_is_nudged_then_work_continues(self):
        from omnimancer.cli.headless import (
            NO_TOOL_CALL_NUDGE,
            HeadlessRunner,
            OutputFormat,
        )

        responses = [
            ChatResponse(
                content="Let me look into the existing routes first.",
                model_used="m",
                tokens_used=5,
                stop_reason="end_turn",
                tool_calls=None,
            ),
            ChatResponse(
                content="Reading now.",
                model_used="m",
                tokens_used=5,
                stop_reason="tool_use",
                tool_calls=[ToolCall(name="file_read", arguments={"path": "/a.py"})],
            ),
            ChatResponse(
                content="All changes are in place.\nDONE",
                model_used="m",
                tokens_used=5,
                stop_reason="end_turn",
                tool_calls=None,
            ),
        ]
        mock_engine, mock_agent_engine = self._mock_engine(responses)

        runner = HeadlessRunner(
            engine=mock_engine, output_format=OutputFormat.TEXT, no_approval=True
        )
        runner._emitter._stdout = StringIO()

        exit_code = await runner.run("add a route")
        assert exit_code == 0
        # Narration did not end the run: the tool call after the nudge ran.
        assert mock_agent_engine.execute_with_approval.call_count == 1
        # The second request carried the nudge.
        nudge_message = mock_engine.send_message_with_tools.call_args_list[1][0][0]
        assert nudge_message == NO_TOOL_CALL_NUDGE

    @pytest.mark.asyncio
    async def test_persistent_narration_ends_after_max_nudges(self):
        import json

        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        narration = ChatResponse(
            content="I will now plan my approach.",
            model_used="m",
            tokens_used=5,
            stop_reason="end_turn",
            tool_calls=None,
        )
        mock_engine, mock_agent_engine = self._mock_engine(
            [narration, narration, narration, narration]
        )

        runner = HeadlessRunner(
            engine=mock_engine, output_format=OutputFormat.JSON, no_approval=True
        )
        runner._emitter._stdout = StringIO()

        exit_code = await runner.run("do the thing")
        assert exit_code == 0
        # Initial turn + 2 nudged retries, then the run ends.
        assert mock_engine.send_message_with_tools.call_count == 3
        assert mock_agent_engine.execute_with_approval.call_count == 0

        # Verify the emitted JSON result line contains stop_cause
        stdout_content = runner._emitter._stdout.getvalue()
        result_line = stdout_content.strip()
        result_data = json.loads(result_line)
        assert result_data["stop_cause"] == "nudge_exhausted"

    @pytest.mark.asyncio
    async def test_done_reply_ends_run_without_nudge(self):
        import json

        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        responses = [
            ChatResponse(
                content="Nothing to do here.\nDONE",
                model_used="m",
                tokens_used=5,
                stop_reason="end_turn",
                tool_calls=None,
            ),
        ]
        mock_engine, _ = self._mock_engine(responses)

        runner = HeadlessRunner(
            engine=mock_engine, output_format=OutputFormat.JSON, no_approval=True
        )
        runner._emitter._stdout = StringIO()

        exit_code = await runner.run("check something")
        assert exit_code == 0
        assert mock_engine.send_message_with_tools.call_count == 1

        # Verify the emitted JSON result line contains stop_cause
        stdout_content = runner._emitter._stdout.getvalue()
        result_line = stdout_content.strip()
        result_data = json.loads(result_line)
        assert result_data["stop_cause"] == "done"


class TestMaxIterationsEnv:
    """OMNIMANCER_MAX_ITERATIONS resolves the iteration cap."""

    def test_env_var_respected(self, monkeypatch):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        monkeypatch.setenv("OMNIMANCER_MAX_ITERATIONS", "150")
        runner = HeadlessRunner(
            engine=MagicMock(), output_format=OutputFormat.TEXT, no_approval=True
        )
        assert runner._max_iterations == 150

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat

        monkeypatch.setenv("OMNIMANCER_MAX_ITERATIONS", "150")
        runner = HeadlessRunner(
            engine=MagicMock(),
            output_format=OutputFormat.TEXT,
            no_approval=True,
            max_iterations=7,
        )
        assert runner._max_iterations == 7

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        from omnimancer.cli.headless import HeadlessRunner, OutputFormat
        from omnimancer.cli.tool_handler import MAX_TOOL_ITERATIONS

        for bad in ("abc", "0", "-5"):
            monkeypatch.setenv("OMNIMANCER_MAX_ITERATIONS", bad)
            runner = HeadlessRunner(
                engine=MagicMock(), output_format=OutputFormat.TEXT, no_approval=True
            )
            assert runner._max_iterations == MAX_TOOL_ITERATIONS
