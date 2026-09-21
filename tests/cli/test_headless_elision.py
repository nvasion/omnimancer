"""Tests for headless tool-result history elision.

The headless loop retransmits the whole conversation to the provider on every
iteration; _elide_stale_tool_results bounds that by replacing the oldest tool
results with a stub once the retained budget is exceeded.
"""

from datetime import datetime

from omnimancer.cli.headless import (
    ELIDED_RESULT_STUB,
    TOOL_RESULT_HISTORY_BUDGET,
    _elide_stale_tool_results,
    _message_tool_result_size,
    _resolve_tool_result_budget,
)
from omnimancer.core.models import (
    ChatContext,
    ChatMessage,
    MessageRole,
    ToolResultRecord,
)


def _ctx(messages):
    return ChatContext(messages=messages, current_model="m", session_id="s")


def _msg(role, content, tool_results=None):
    return ChatMessage(
        role=role,
        content=content,
        timestamp=datetime.now(),
        model_used="m",
        tool_results=tool_results,
    )


def _native_results(payload):
    """User message carrying native tool results of the given payload."""
    return _msg(
        MessageRole.USER,
        f"Tool results:\n\n{payload}",
        tool_results=[ToolResultRecord(tool_call_id="c1", content=payload)],
    )


class TestMessageToolResultSize:
    def test_native_records_counted(self):
        msg = _native_results("A" * 100)
        assert _message_tool_result_size(msg) == 100

    def test_text_protocol_results_counted(self):
        msg = _msg(MessageRole.USER, "Tool results:\n\n" + "A" * 100)
        assert _message_tool_result_size(msg) == len(msg.content)

    def test_plain_user_message_not_counted(self):
        msg = _msg(MessageRole.USER, "please fix the bug")
        assert _message_tool_result_size(msg) == 0

    def test_assistant_message_not_counted(self):
        msg = _msg(MessageRole.ASSISTANT, "Tool results: just narration")
        assert _message_tool_result_size(msg) == 0


class TestElideStaleToolResults:
    def test_oldest_results_elided_beyond_budget(self):
        old = _native_results("A" * 3000)
        mid = _native_results("B" * 3000)
        new = _native_results("C" * 3000)
        ctx = _ctx([_msg(MessageRole.USER, "task"), old, mid, new])

        _elide_stale_tool_results(ctx, budget=6500)

        assert old.content == ELIDED_RESULT_STUB
        assert old.tool_results[0].content == ELIDED_RESULT_STUB
        assert "B" * 3000 in mid.content
        assert "C" * 3000 in new.content

    def test_newest_batch_never_elided_even_over_budget(self):
        newest = _native_results("A" * 10_000)
        ctx = _ctx([newest])

        _elide_stale_tool_results(ctx, budget=100)

        assert "A" * 10_000 in newest.content

    def test_text_protocol_messages_elided(self):
        old = _msg(MessageRole.USER, "Tool results:\n\n" + "A" * 3000)
        new = _msg(MessageRole.USER, "Tool results:\n\n" + "B" * 3000)
        ctx = _ctx([old, new])

        _elide_stale_tool_results(ctx, budget=3500)

        assert old.content == ELIDED_RESULT_STUB
        assert "B" * 3000 in new.content

    def test_non_result_messages_untouched(self):
        task = _msg(MessageRole.USER, "implement the feature")
        answer = _msg(MessageRole.ASSISTANT, "working on it")
        old = _native_results("A" * 3000)
        new = _native_results("B" * 3000)
        ctx = _ctx([task, answer, old, new])

        _elide_stale_tool_results(ctx, budget=3000)

        assert task.content == "implement the feature"
        assert answer.content == "working on it"

    def test_idempotent_across_iterations(self):
        old = _native_results("A" * 3000)
        new = _native_results("B" * 3000)
        ctx = _ctx([old, new])

        _elide_stale_tool_results(ctx, budget=3500)
        first_pass = old.content
        _elide_stale_tool_results(ctx, budget=3500)

        assert old.content == first_pass == ELIDED_RESULT_STUB
        assert "B" * 3000 in new.content

    def test_everything_kept_within_budget(self):
        msgs = [_native_results("A" * 1000) for _ in range(3)]
        ctx = _ctx(list(msgs))

        _elide_stale_tool_results(ctx, budget=TOOL_RESULT_HISTORY_BUDGET)

        for m in msgs:
            assert ELIDED_RESULT_STUB not in m.content


class TestResolveToolResultBudget:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("OMNIMANCER_TOOL_RESULT_BUDGET", raising=False)
        assert _resolve_tool_result_budget() == TOOL_RESULT_HISTORY_BUDGET

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_TOOL_RESULT_BUDGET", "12345")
        assert _resolve_tool_result_budget() == 12345

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_TOOL_RESULT_BUDGET", "lots")
        assert _resolve_tool_result_budget() == TOOL_RESULT_HISTORY_BUDGET

    def test_non_positive_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_TOOL_RESULT_BUDGET", "-5")
        assert _resolve_tool_result_budget() == TOOL_RESULT_HISTORY_BUDGET
