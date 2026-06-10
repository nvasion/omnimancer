"""Tests for streaming integration in the CLI interface."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.cli.interface import CommandLineInterface
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.engine import CoreEngine
from omnimancer.core.models import (
    ChatResponse,
    ProviderConfig,
    StreamEvent,
    StreamEventType,
    ToolCall,
)


@pytest.fixture
def mock_engine():
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "config.json"
    config_manager = ConfigManager(str(config_path))
    config = config_manager.get_config()
    config.providers = {
        "test": ProviderConfig(
            model="test-model",
            api_key="test-key",
            provider_type="test",
            supports_tools=True,
            supports_multimodal=False,
        )
    }
    config.default_provider = "test"

    engine = MagicMock(spec=CoreEngine)
    engine.config_manager = config_manager
    engine.current_provider = None
    engine.chat_manager = MagicMock()
    return engine


@pytest.fixture
def interface(mock_engine):
    return CommandLineInterface(mock_engine, no_approval=True)


def make_text_stream_events(text: str, model: str = "test-model") -> list:
    response = ChatResponse(
        content=text,
        model_used=model,
        tokens_used=10,
        input_tokens=5,
        output_tokens=10,
        cost_estimate=0.001,
        timestamp=datetime.now(),
    )
    return [
        StreamEvent(type=StreamEventType.MESSAGE_START, model=model),
        StreamEvent(type=StreamEventType.TEXT_DELTA, text=text),
        StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, response=response),
    ]


def make_tool_stream_events(
    text: str, tool_name: str, tool_args: dict, model: str = "test-model"
) -> list:
    import json

    response = ChatResponse(
        content=text,
        model_used=model,
        tokens_used=20,
        input_tokens=10,
        output_tokens=20,
        cost_estimate=0.002,
        timestamp=datetime.now(),
        tool_calls=[ToolCall(name=tool_name, arguments=tool_args)],
    )
    return [
        StreamEvent(type=StreamEventType.MESSAGE_START, model=model),
        StreamEvent(type=StreamEventType.TEXT_DELTA, text=text),
        StreamEvent(type=StreamEventType.TOOL_USE_START, tool_name=tool_name),
        StreamEvent(
            type=StreamEventType.TOOL_USE_DELTA,
            partial_json=json.dumps(tool_args),
        ),
        StreamEvent(type=StreamEventType.TOOL_USE_END),
        StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, response=response),
    ]


class TestStreamToolResponse:
    @pytest.mark.asyncio
    async def test_returns_response_from_stream(self, interface, mock_engine):
        events = make_text_stream_events("Hello from stream")

        async def fake_stream(msg, tools):
            for e in events:
                yield e

        mock_engine.send_message_with_tools_stream = fake_stream

        with patch("omnimancer.ui.streaming_display.Live"):
            response = await interface._stream_tool_response("Hi", [])

        assert response.content == "Hello from stream"
        assert response.model_used == "test-model"

    @pytest.mark.asyncio
    async def test_collects_tool_calls(self, interface, mock_engine):
        events = make_tool_stream_events(
            "Reading file.", "file_read", {"path": "/a.py"}
        )

        async def fake_stream(msg, tools):
            for e in events:
                yield e

        mock_engine.send_message_with_tools_stream = fake_stream

        with patch("omnimancer.ui.streaming_display.Live"):
            response = await interface._stream_tool_response("Read a.py", [])

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "file_read"
        assert response.tool_calls[0].arguments == {"path": "/a.py"}

    @pytest.mark.asyncio
    async def test_updates_chat_history(self, interface, mock_engine):
        events = make_text_stream_events("Response text")

        async def fake_stream(msg, tools):
            for e in events:
                yield e

        mock_engine.send_message_with_tools_stream = fake_stream

        with patch("omnimancer.ui.streaming_display.Live"):
            await interface._stream_tool_response("User msg", [])

        mock_engine.chat_manager.add_user_message.assert_called_once_with("User msg")
        mock_engine.chat_manager.add_assistant_message.assert_called_once_with(
            "Response text",
            "test-model",
            tool_calls=None,
            raw_content="Response text",
        )

    @pytest.mark.asyncio
    async def test_records_tool_calls_in_history(self, interface, mock_engine):
        """The assistant turn recorded in history must mention its tool calls.

        Otherwise the next iteration's context shows an empty assistant
        message followed by an unlabeled result — the model can't tell it
        already made the call and repeats it.
        """
        events = make_tool_stream_events(
            "Reading file.", "file_read", {"path": "/a.py"}
        )

        async def fake_stream(msg, tools):
            for e in events:
                yield e

        mock_engine.send_message_with_tools_stream = fake_stream

        with patch("omnimancer.ui.streaming_display.Live"):
            await interface._stream_tool_response("Read a.py", [])

        recorded = mock_engine.chat_manager.add_assistant_message.call_args[0][0]
        assert "file_read" in recorded
        assert "/a.py" in recorded

    @pytest.mark.asyncio
    async def test_returns_error_on_empty_stream(self, interface, mock_engine):
        async def fake_stream(msg, tools):
            return
            yield

        mock_engine.send_message_with_tools_stream = fake_stream

        with patch("omnimancer.ui.streaming_display.Live"):
            response = await interface._stream_tool_response("Hi", [])

        assert response.error == "Stream failed"
        assert not response.is_success


class TestStreamChatResponse:
    @pytest.mark.asyncio
    async def test_returns_response_from_stream(self, interface, mock_engine):
        events = make_text_stream_events("Streamed reply")

        async def fake_stream(msg):
            for e in events:
                yield e

        mock_engine.send_message_stream = fake_stream

        with patch("omnimancer.ui.streaming_display.Live"):
            response = await interface._stream_chat_response("Hello")

        assert response.content == "Streamed reply"
        assert response.is_success

    @pytest.mark.asyncio
    async def test_updates_chat_history(self, interface, mock_engine):
        events = make_text_stream_events("Reply")

        async def fake_stream(msg):
            for e in events:
                yield e

        mock_engine.send_message_stream = fake_stream

        with patch("omnimancer.ui.streaming_display.Live"):
            await interface._stream_chat_response("Question")

        mock_engine.chat_manager.add_user_message.assert_called_once_with("Question")
        mock_engine.chat_manager.add_assistant_message.assert_called_once_with(
            "Reply", "test-model"
        )

    @pytest.mark.asyncio
    async def test_returns_error_on_empty_stream(self, interface, mock_engine):
        async def fake_stream(msg):
            return
            yield

        mock_engine.send_message_stream = fake_stream

        with patch("omnimancer.ui.streaming_display.Live"):
            response = await interface._stream_chat_response("Hello")

        assert response.error == "Stream failed"


class TestStreamingRouting:
    @pytest.mark.asyncio
    async def test_tool_flow_uses_streaming_when_supported(
        self, interface, mock_engine
    ):
        """Uses _stream_tool_response when provider supports it."""
        provider = MagicMock()
        provider.supports_streaming.return_value = True
        mock_engine.current_provider = provider
        mock_engine.agent_engine = MagicMock()

        response = ChatResponse(
            content="Done",
            model_used="test-model",
            tokens_used=10,
            input_tokens=5,
            output_tokens=10,
            cost_estimate=0.001,
            timestamp=datetime.now(),
            tool_calls=None,
        )

        with patch.object(
            interface,
            "_stream_tool_response",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_stream:
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("Do stuff")

        mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_flow_skips_streaming_when_not_supported(
        self, interface, mock_engine
    ):
        """Uses send_message_with_tools when not supported."""
        provider = MagicMock()
        provider.supports_streaming.return_value = False
        mock_engine.current_provider = provider
        mock_engine.agent_engine = MagicMock()

        response = ChatResponse(
            content="Done",
            model_used="test-model",
            tokens_used=10,
            input_tokens=5,
            output_tokens=10,
            cost_estimate=0.001,
            timestamp=datetime.now(),
            tool_calls=None,
        )
        mock_engine.send_message_with_tools = AsyncMock(return_value=response)

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("Do stuff")

        mock_engine.send_message_with_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_uses_streaming_when_supported(self, interface, mock_engine):
        """Non-agent chat uses _stream_chat_response."""
        from omnimancer.cli.commands import Command, CommandType

        provider = MagicMock()
        provider.supports_streaming.return_value = True
        mock_engine.current_provider = provider
        interface.agent_manager = None

        response = ChatResponse(
            content="Hi",
            model_used="test-model",
            tokens_used=5,
            input_tokens=3,
            output_tokens=5,
            cost_estimate=0.0005,
            timestamp=datetime.now(),
        )

        command = Command(
            type=CommandType.CHAT_MESSAGE,
            content="Hello",
            parameters={},
            raw_input="Hello",
        )

        with patch.object(
            interface,
            "_stream_chat_response",
            new_callable=AsyncMock,
            return_value=response,
        ) as mock_stream:
            with patch.object(interface, "_show_user_message"):
                with patch.object(interface, "_show_token_status"):
                    cancel_handler = interface.cancellation_handler
                    with patch.object(
                        cancel_handler,
                        "start_cancellable_operation",
                    ) as mock_cancel:

                        async def run_op(**kwargs):
                            await kwargs["operation"]()

                        mock_cancel.side_effect = run_op
                        await interface._handle_chat_message(command)

        mock_stream.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_chat_skips_streaming_when_not_supported(
        self, interface, mock_engine
    ):
        """Non-agent chat path uses send_message when streaming not supported."""
        from omnimancer.cli.commands import Command, CommandType

        mock_engine.current_provider = None
        interface.agent_manager = None

        response = MagicMock()
        response.is_success = True
        response.content = "Hi"
        response.model_used = "test-model"
        mock_engine.send_message = AsyncMock(return_value=response)

        command = Command(
            type=CommandType.CHAT_MESSAGE,
            content="Hello",
            parameters={},
            raw_input="Hello",
        )

        with patch.object(interface, "_show_user_message"):
            with patch.object(interface, "_show_assistant_message") as mock_show:
                with patch.object(interface, "_show_token_status"):
                    cancel_handler = interface.cancellation_handler
                    with patch.object(
                        cancel_handler,
                        "start_cancellable_operation",
                    ) as mock_cancel:

                        async def run_op(**kwargs):
                            await kwargs["operation"]()

                        mock_cancel.side_effect = run_op
                        await interface._handle_chat_message(command)

        mock_engine.send_message.assert_called_once()
        mock_show.assert_called_once_with("Hi", "test-model")
