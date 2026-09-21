"""The isolated tool loop shared by planner, worker and judge (PRD §3–§5)."""

from unittest.mock import AsyncMock, MagicMock

from omnimancer.core.models import (
    ChatContext,
    ChatResponse,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from omnimancer.h2l.loop import IsolatedLoop


def _resp(content="", tool_calls=None, error=None, **usage):
    return ChatResponse(
        content=content,
        model_used="m",
        tokens_used=0,
        tool_calls=tool_calls,
        error=error,
        **usage,
    )


def _provider(responses, native=False):
    provider = MagicMock()
    provider.model = "m"
    provider.send_message_with_tools = AsyncMock(side_effect=responses)
    provider.supports_native_tool_history.return_value = native
    return provider


TOOLS = [ToolDefinition(name="Read", description="", parameters={})]


def _context():
    return ChatContext(messages=[], current_model="m", session_id="s")


class TestIsolatedLoop:
    async def test_history_accumulates_across_turns(self):
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="Read", arguments={"a": 1})]),
                _resp(tool_calls=[ToolCall(name="Read", arguments={"a": 2})]),
                _resp(content="final"),
            ]
        )
        execute = AsyncMock(return_value=ToolResult(content="data"))
        context = _context()
        loop = IsolatedLoop(provider, context, TOOLS, execute, max_iterations=10)
        result = await loop.run("start")
        assert result.stop == "done"
        assert result.content == "final"
        assert result.iterations == 3
        roles = [m.role for m in context.messages]
        # user, assistant, user(results), assistant, user(results), assistant
        assert roles == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert "[Called tools:" in context.messages[1].content
        assert context.messages[2].content.startswith("Tool results:")
        # The third call carries the accumulated context, not a fresh one.
        assert provider.send_message_with_tools.call_args_list[2][0][1] is context
        assert len(result.tool_log) == 2

    async def test_native_history_records_results_and_sends_empty_message(self):
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="Read", arguments={}, id="c1")]),
                _resp(content="ok"),
            ],
            native=True,
        )
        context = _context()
        loop = IsolatedLoop(
            provider,
            context,
            TOOLS,
            AsyncMock(return_value=ToolResult(content="x")),
            max_iterations=5,
        )
        await loop.run("start")
        second_message = provider.send_message_with_tools.call_args_list[1][0][0]
        assert second_message == ""
        results_msg = context.messages[2]
        assert results_msg.tool_results[0].tool_call_id == "c1"
        assert results_msg.tool_results[0].content == "x"

    async def test_cancelled_result_aborts(self):
        provider = _provider(
            [_resp(tool_calls=[ToolCall(name="Read", arguments={})]), _resp("x")]
        )
        loop = IsolatedLoop(
            provider,
            _context(),
            TOOLS,
            AsyncMock(return_value=ToolResult(content="", cancelled=True)),
            max_iterations=5,
        )
        result = await loop.run("start")
        assert result.stop == "cancelled"
        assert provider.send_message_with_tools.call_count == 1

    async def test_provider_error_surfaces(self):
        loop = IsolatedLoop(
            _provider([_resp(error="boom")]), _context(), TOOLS, AsyncMock(), 5
        )
        result = await loop.run("start")
        assert result.stop == "error"
        assert result.error == "boom"

    async def test_provider_exception_surfaces(self):
        provider = _provider([RuntimeError("connection lost")])
        loop = IsolatedLoop(provider, _context(), TOOLS, AsyncMock(), 5)
        result = await loop.run("start")
        assert result.stop == "error"
        assert "connection lost" in (result.error or "")

    async def test_iteration_cap(self):
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="Read", arguments={"i": i})])
                for i in range(9)
            ]
        )
        loop = IsolatedLoop(
            provider,
            _context(),
            TOOLS,
            AsyncMock(return_value=ToolResult(content="x")),
            max_iterations=3,
        )
        result = await loop.run("start")
        assert result.stop == "max_iterations"
        assert result.iterations == 3

    async def test_repeated_identical_call_aborts_after_threshold(self):
        same = ToolCall(name="Read", arguments={"a": 1})
        provider = _provider([_resp(tool_calls=[same]) for _ in range(10)])
        loop = IsolatedLoop(
            provider,
            _context(),
            TOOLS,
            AsyncMock(return_value=ToolResult(content="x")),
            max_iterations=10,
        )
        result = await loop.run("start")
        assert result.stop == "repeat_abort"

    async def test_stop_tool_ends_loop_after_execution(self):
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="submit", arguments={"k": 1})]),
                _resp(content="should not be reached"),
            ]
        )
        execute = AsyncMock(return_value=ToolResult(content="received"))
        loop = IsolatedLoop(
            provider,
            _context(),
            TOOLS,
            execute,
            max_iterations=5,
            stop_tools={"submit"},
        )
        result = await loop.run("start")
        assert result.stop == "stopped"
        assert provider.send_message_with_tools.call_count == 1
        execute.assert_awaited_once()

    async def test_stop_tool_with_error_keeps_looping(self):
        # A rejected submission goes back to the model so it can fix it.
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="submit", arguments={"k": 1})]),
                _resp(tool_calls=[ToolCall(name="submit", arguments={"k": 2})]),
            ]
        )
        execute = AsyncMock(
            side_effect=[
                ToolResult(content="", error="rejected"),
                ToolResult(content="received"),
            ]
        )
        loop = IsolatedLoop(
            provider, _context(), TOOLS, execute, 5, stop_tools={"submit"}
        )
        result = await loop.run("start")
        assert result.stop == "stopped"
        assert provider.send_message_with_tools.call_count == 2

    async def test_described_tool_calls_recovered_from_text(self):
        provider = _provider(
            [
                _resp(content='[Called tools: Read({"a": 1})]'),
                _resp(content="done"),
            ]
        )
        execute = AsyncMock(return_value=ToolResult(content="x"))
        loop = IsolatedLoop(provider, _context(), TOOLS, execute, 5)
        result = await loop.run("start")
        assert result.stop == "done"
        execute.assert_awaited_once()

    async def test_usage_accumulates(self):
        provider = _provider(
            [
                _resp(
                    tool_calls=[ToolCall(name="Read", arguments={})],
                    input_tokens=10,
                    output_tokens=2,
                ),
                _resp(content="x", input_tokens=20, output_tokens=3),
            ]
        )
        loop = IsolatedLoop(
            provider,
            _context(),
            TOOLS,
            AsyncMock(return_value=ToolResult(content="r")),
            5,
        )
        result = await loop.run("start")
        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 5
        assert result.usage.calls == 2

    async def test_unbounded_when_max_iterations_is_none(self):
        # Well past any former cap: the loop ends when the model is done.
        responses = [
            _resp(tool_calls=[ToolCall(name="Read", arguments={"i": i})])
            for i in range(40)
        ] + [_resp(content="finally")]
        loop = IsolatedLoop(
            _provider(responses),
            _context(),
            TOOLS,
            AsyncMock(return_value=ToolResult(content="x")),
            max_iterations=None,
        )
        result = await loop.run("start")
        assert result.stop == "done"
        assert result.iterations == 41

    async def test_truncated_tool_call_is_flagged_to_the_model(self):
        cut = _resp(tool_calls=[ToolCall(name="Read", arguments={})])
        cut.stop_reason = "length"
        provider = _provider([cut, _resp(content="done")])
        loop = IsolatedLoop(
            provider,
            _context(),
            TOOLS,
            AsyncMock(return_value=ToolResult(content="x")),
            5,
        )
        result = await loop.run("start")
        assert result.truncations == 1
        follow_up = provider.send_message_with_tools.call_args_list[1][0][0]
        assert "cut off at the output token limit" in follow_up

    async def test_truncated_prose_continues_instead_of_ending(self):
        cut = _resp(content="I was in the middle of")
        cut.stop_reason = "length"
        provider = _provider([cut, _resp(content="finished")])
        loop = IsolatedLoop(provider, _context(), TOOLS, AsyncMock(), 5)
        result = await loop.run("start")
        assert result.stop == "done"
        assert result.content == "finished"
        assert provider.send_message_with_tools.call_count == 2

    async def test_truncation_continues_are_bounded(self):
        cut = _resp(content="again")
        cut.stop_reason = "length"
        provider = _provider([cut] * 10)
        loop = IsolatedLoop(provider, _context(), TOOLS, AsyncMock(), None)
        result = await loop.run("start")
        assert result.stop == "done"
        assert provider.send_message_with_tools.call_count == 3

    async def test_served_model_recorded(self):
        first = _resp(tool_calls=[ToolCall(name="Read", arguments={})])
        first.model_used = "qwen3.8-max"
        second = _resp(content="done")
        second.model_used = ""
        loop = IsolatedLoop(
            _provider([first, second]),
            _context(),
            TOOLS,
            AsyncMock(return_value=ToolResult(content="x")),
            5,
        )
        result = await loop.run("start")
        assert result.model_used == "qwen3.8-max"

    async def test_heartbeat_fires_while_provider_is_slow(self):
        import asyncio

        async def slow(*args, **kwargs):
            await asyncio.sleep(0.08)
            return _resp(content="late")

        provider = MagicMock()
        provider.model = "m"
        provider.send_message_with_tools = slow
        provider.supports_native_tool_history.return_value = False
        beats = []

        async def on_wait(seconds):
            beats.append(seconds)

        loop = IsolatedLoop(
            provider,
            _context(),
            TOOLS,
            AsyncMock(),
            5,
            on_wait=on_wait,
            heartbeat=0.02,
        )
        result = await loop.run("start")
        assert result.stop == "done" and result.content == "late"
        assert len(beats) >= 2
        assert beats == sorted(beats)

    async def test_heartbeat_does_not_swallow_provider_errors(self):
        provider = _provider([RuntimeError("boom")])
        loop = IsolatedLoop(
            provider, _context(), TOOLS, AsyncMock(), 5, on_wait=AsyncMock()
        )
        result = await loop.run("start")
        assert result.stop == "error" and "boom" in result.error

    async def test_tool_call_callback_observes_each_call(self):
        provider = _provider(
            [_resp(tool_calls=[ToolCall(name="Read", arguments={})]), _resp("x")]
        )
        seen = []

        async def on_call(tc, result):
            seen.append((tc.name, result.content))

        loop = IsolatedLoop(
            provider,
            _context(),
            TOOLS,
            AsyncMock(return_value=ToolResult(content="r")),
            5,
            on_tool_call=on_call,
        )
        await loop.run("start")
        assert seen == [("Read", "r")]
