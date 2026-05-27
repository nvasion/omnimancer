"""Tests for StreamingDisplay component."""

import json

import pytest
from rich.console import Console

from omnimancer.core.models import StreamEvent, StreamEventType
from omnimancer.ui.streaming_display import StreamingDisplay


@pytest.fixture
def console():
    return Console(file=None, force_terminal=True)


@pytest.fixture
def display(console):
    return StreamingDisplay(console, model="test-model")


class TestStreamingDisplayEventHandling:
    def test_message_start_sets_model(self, display):
        display.handle_event(
            StreamEvent(type=StreamEventType.MESSAGE_START, model="claude-sonnet-4-6")
        )
        assert display.model == "claude-sonnet-4-6"

    def test_text_delta_accumulates(self, display):
        display.handle_event(StreamEvent(
            type=StreamEventType.TEXT_DELTA, text="Hello"
        ))
        display.handle_event(
            StreamEvent(type=StreamEventType.TEXT_DELTA, text=" world")
        )
        assert display.accumulated_text == "Hello world"

    def test_tool_use_collected(self, display):
        display.handle_event(
            StreamEvent(
                type=StreamEventType.TOOL_USE_START,
                tool_name="file_read",
                tool_id="toolu_1",
            )
        )
        display.handle_event(
            StreamEvent(
                type=StreamEventType.TOOL_USE_DELTA,
                partial_json='{"path": "/main.py"}',
            )
        )
        display.handle_event(StreamEvent(type=StreamEventType.TOOL_USE_END))

        assert len(display.tool_calls) == 1
        assert display.tool_calls[0].name == "file_read"
        assert display.tool_calls[0].arguments == {"path": "/main.py"}

    def test_multiple_tool_calls(self, display):
        tool_pairs = [
            ("file_read", {"path": "a.py"}),
            ("command_exec", {"command": "ls"}),
        ]
        for name, args in tool_pairs:
            display.handle_event(
                StreamEvent(
                    type=StreamEventType.TOOL_USE_START,
                    tool_name=name,
                )
            )
            display.handle_event(
                StreamEvent(
                    type=StreamEventType.TOOL_USE_DELTA,
                    partial_json=json.dumps(args),
                )
            )
            display.handle_event(StreamEvent(type=StreamEventType.TOOL_USE_END))

        assert len(display.tool_calls) == 2
        assert display.tool_calls[0].name == "file_read"
        assert display.tool_calls[1].name == "command_exec"

    def test_invalid_tool_json_uses_empty_dict(self, display):
        display.handle_event(
            StreamEvent(type=StreamEventType.TOOL_USE_START, tool_name="broken")
        )
        display.handle_event(
            StreamEvent(
                type=StreamEventType.TOOL_USE_DELTA,
                partial_json="not valid json{",
            )
        )
        display.handle_event(StreamEvent(type=StreamEventType.TOOL_USE_END))

        assert display.tool_calls[0].arguments == {}

    def test_empty_tool_json_uses_empty_dict(self, display):
        display.handle_event(
            StreamEvent(type=StreamEventType.TOOL_USE_START, tool_name="empty")
        )
        display.handle_event(StreamEvent(type=StreamEventType.TOOL_USE_END))

        assert display.tool_calls[0].arguments == {}

    def test_initial_state(self, display):
        assert display.accumulated_text == ""
        assert display.tool_calls == []
        assert display.model == "test-model"


class TestStreamingDisplayRender:
    def test_render_with_no_text(self, display):
        panel = display._render()
        assert "..." in str(panel.renderable)

    def test_render_with_text(self, display):
        display.accumulated_text = "Hello world"
        panel = display._render()
        assert "Hello world" in str(panel.renderable)

    def test_render_shows_model_in_title(self, display):
        display.model = "claude-sonnet-4-6"
        panel = display._render()
        assert "claude-sonnet-4-6" in panel.title
