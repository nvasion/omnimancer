"""SSE streaming for the OpenAI provider family (vLLM-compatible).

Feeds canned OpenAI chat.completion.chunk SSE sequences through the
provider's stream parser and asserts the StreamEvent contract that
ui/streaming_display.py and interface._stream_tool_response consume.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core.models import (
    ChatContext,
    ChatMessage,
    MessageRole,
    StreamEventType,
    ToolDefinition,
)
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.utils.errors import AuthenticationError


@pytest.fixture
def provider():
    return OpenAIProvider(api_key="sk-test", model="qwen3-coder-30b")


@pytest.fixture
def context():
    return ChatContext(
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content="Hi",
                timestamp=datetime.now(),
                model_used="qwen3-coder-30b",
            )
        ],
        current_model="qwen3-coder-30b",
        session_id="s",
    )


@pytest.fixture
def tools():
    return [
        ToolDefinition(
            name="Read",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        )
    ]


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""

    def json(self):
        return {"error": {"message": "unauthorized"}}


def _patch_stream(lines, status_code=200):
    """Patch httpx.AsyncClient so client.stream() yields the given SSE lines."""
    response = _FakeStreamResponse(lines, status_code)
    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    patcher = patch("httpx.AsyncClient")
    mock_client_cls = patcher.start()
    client = mock_client_cls.return_value.__aenter__.return_value
    client.stream = MagicMock(return_value=stream_cm)
    return patcher, client


TEXT_LINES = [
    'data: {"id":"c1","model":"qwen3-coder-30b","choices":'
    '[{"index":0,"delta":{"role":"assistant","content":""},'
    '"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"content":"Hello"},'
    '"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"content":" world"},'
    '"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
    "data: [DONE]",
]


TOOL_LINES = [
    'data: {"id":"c2","model":"qwen3-coder-30b","choices":'
    '[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
    '"id":"call_abc","type":"function","function":{"name":"Read",'
    '"arguments":""}}]},"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"{\\"file_"}}]},"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"path\\": \\"x\\"}"}}]},"finish_reason":null}]}',
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
    "data: [DONE]",
]


class TestSupportsStreaming:
    def test_openai_supports_streaming(self, provider):
        assert provider.supports_streaming() is True


class TestTextStreaming:
    @pytest.mark.asyncio
    async def test_text_deltas_and_completion(self, provider, context):
        patcher, client = _patch_stream(TEXT_LINES)
        try:
            events = [e async for e in provider.send_message_stream("Hi", context)]
        finally:
            patcher.stop()

        assert events[0].type == StreamEventType.MESSAGE_START
        assert events[0].model == "qwen3-coder-30b"
        deltas = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert [d.text for d in deltas] == ["Hello", " world"]
        final = events[-1]
        assert final.type == StreamEventType.MESSAGE_COMPLETE
        assert final.response.content == "Hello world"
        assert final.response.stop_reason == "stop"
        assert final.response.input_tokens == 10
        assert final.response.output_tokens == 5

    @pytest.mark.asyncio
    async def test_request_body_enables_stream_and_usage(self, provider, context):
        patcher, client = _patch_stream(TEXT_LINES)
        try:
            async for _ in provider.send_message_stream("Hi", context):
                pass
        finally:
            patcher.stop()

        body = client.stream.call_args.kwargs["json"]
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["model"] == "qwen3-coder-30b"

    @pytest.mark.asyncio
    async def test_lines_after_done_ignored(self, provider, context):
        lines = TEXT_LINES[:-1] + [
            "data: [DONE]",
            'data: {"choices":[{"index":0,"delta":{"content":"IGNORED"},'
            '"finish_reason":null}]}',
        ]
        patcher, _ = _patch_stream(lines)
        try:
            events = [e async for e in provider.send_message_stream("Hi", context)]
        finally:
            patcher.stop()

        final = events[-1]
        assert "IGNORED" not in final.response.content


class TestToolCallStreaming:
    @pytest.mark.asyncio
    async def test_tool_call_fragments_accumulate(self, provider, context, tools):
        patcher, client = _patch_stream(TOOL_LINES)
        try:
            events = [
                e
                async for e in provider.send_message_with_tools_stream(
                    "Hi", context, tools
                )
            ]
        finally:
            patcher.stop()

        starts = [e for e in events if e.type == StreamEventType.TOOL_USE_START]
        assert len(starts) == 1
        assert starts[0].tool_name == "Read"
        assert starts[0].tool_id == "call_abc"
        assert any(e.type == StreamEventType.TOOL_USE_DELTA for e in events)
        assert any(e.type == StreamEventType.TOOL_USE_END for e in events)

        final = events[-1]
        assert final.type == StreamEventType.MESSAGE_COMPLETE
        calls = final.response.tool_calls
        assert len(calls) == 1
        assert calls[0].name == "Read"
        assert calls[0].arguments == {"file_path": "x"}
        assert calls[0].id == "call_abc"
        assert final.response.stop_reason == "tool_calls"

        body = client.stream.call_args.kwargs["json"]
        assert body["tools"][0]["function"]["name"] == "Read"

    @pytest.mark.asyncio
    async def test_two_tool_calls_second_id_synthesized(self, provider, context, tools):
        lines = [
            'data: {"id":"c3","model":"m","choices":[{"index":0,'
            '"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
            '"id":"call_a","function":{"name":"Read","arguments":'
            '"{\\"file_path\\": \\"a\\"}"}}]},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,'
            '"function":{"name":"Read","arguments":'
            '"{\\"file_path\\": \\"b\\"}"}}]},"finish_reason":null}]}',
            'data: {"choices":[{"index":0,"delta":{},'
            '"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        patcher, _ = _patch_stream(lines)
        try:
            events = [
                e
                async for e in provider.send_message_with_tools_stream(
                    "Hi", context, tools
                )
            ]
        finally:
            patcher.stop()

        final = events[-1]
        calls = final.response.tool_calls
        assert len(calls) == 2
        assert calls[0].arguments == {"file_path": "a"}
        assert calls[1].arguments == {"file_path": "b"}
        assert calls[0].id == "call_a"
        assert calls[1].id  # synthesized, non-empty


class TestStreamErrors:
    @pytest.mark.asyncio
    async def test_non_200_raises_before_streaming(self, provider, context):
        patcher, _ = _patch_stream([], status_code=401)
        try:
            with pytest.raises(AuthenticationError):
                async for _ in provider.send_message_stream("Hi", context):
                    pass
        finally:
            patcher.stop()
