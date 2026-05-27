"""Tests for Claude provider, focusing on native tool calling."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core.models import (
    ChatContext,
    ChatMessage,
    ChatResponse,
    MessageRole,
    ToolCall,
    ToolDefinition,
)
from omnimancer.providers.claude import ClaudeProvider
from omnimancer.utils.errors import NetworkError, ProviderError


@pytest.fixture
def claude_provider():
    return ClaudeProvider(api_key="sk-test-key-123", model="claude-sonnet-4-20250514")


@pytest.fixture
def sample_chat_context():
    return ChatContext(
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content="Hello",
                timestamp=datetime.now(),
                model_used="claude-sonnet-4-20250514",
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Hi there!",
                timestamp=datetime.now(),
                model_used="claude-sonnet-4-20250514",
            ),
        ],
        current_model="claude-sonnet-4-20250514",
        session_id="test-session",
    )


@pytest.fixture
def sample_tools():
    return [
        ToolDefinition(
            name="file_read",
            description="Read file contents",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read",
                    }
                },
                "required": ["path"],
            },
        ),
        ToolDefinition(
            name="command_exec",
            description="Execute a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute",
                    }
                },
                "required": ["command"],
            },
        ),
    ]


@pytest.fixture
def mock_tool_use_response():
    """Mock Anthropic API response with a tool_use content block."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll read that file for you."},
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "file_read",
                "input": {"path": "/src/main.py"},
            },
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


@pytest.fixture
def mock_text_only_response():
    """Mock Anthropic API response with text only (no tool calls)."""
    return {
        "id": "msg_456",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Here's what I found."},
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 80, "output_tokens": 30},
    }


@pytest.fixture
def mock_multi_tool_response():
    """Mock response with multiple tool calls."""
    return {
        "id": "msg_789",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me read the file and run the tests."},
            {
                "type": "tool_use",
                "id": "toolu_a",
                "name": "file_read",
                "input": {"path": "/src/main.py"},
            },
            {
                "type": "tool_use",
                "id": "toolu_b",
                "name": "command_exec",
                "input": {"command": "pytest tests/"},
            },
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 120, "output_tokens": 80},
    }


class TestClaudeProviderToolCalling:
    """Test native tool calling for Claude provider."""

    @pytest.mark.asyncio
    async def test_send_message_with_tools_returns_tool_calls(
        self, claude_provider, sample_chat_context, sample_tools, mock_tool_use_response
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tool_use_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await claude_provider.send_message_with_tools(
                "Read /src/main.py", sample_chat_context, sample_tools
            )

        assert response.content == "I'll read that file for you."
        assert response.model_used == "claude-sonnet-4-20250514"
        assert response.tokens_used == 50
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "file_read"
        assert response.tool_calls[0].arguments == {"path": "/src/main.py"}

    @pytest.mark.asyncio
    async def test_send_message_with_tools_text_only(
        self,
        claude_provider,
        sample_chat_context,
        sample_tools,
        mock_text_only_response,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_text_only_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await claude_provider.send_message_with_tools(
                "What is Python?", sample_chat_context, sample_tools
            )

        assert response.content == "Here's what I found."
        assert response.tool_calls is None

    @pytest.mark.asyncio
    async def test_send_message_with_tools_multiple_tool_calls(
        self,
        claude_provider,
        sample_chat_context,
        sample_tools,
        mock_multi_tool_response,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_multi_tool_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await claude_provider.send_message_with_tools(
                "Read main.py and run tests", sample_chat_context, sample_tools
            )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "file_read"
        assert response.tool_calls[1].name == "command_exec"
        assert response.tool_calls[1].arguments == {"command": "pytest tests/"}

    @pytest.mark.asyncio
    async def test_send_message_with_tools_empty_tools_list(
        self, claude_provider, sample_chat_context, mock_text_only_response
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_text_only_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await claude_provider.send_message_with_tools(
                "Hello", sample_chat_context, []
            )

        assert response.content == "Here's what I found."
        assert response.tool_calls is None

    @pytest.mark.asyncio
    async def test_send_message_with_tools_sends_correct_api_format(
        self, claude_provider, sample_chat_context, sample_tools, mock_text_only_response
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_text_only_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await claude_provider.send_message_with_tools(
                "Test message", sample_chat_context, sample_tools
            )

            call_kwargs = mock_post.call_args
            request_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

            assert "tools" in request_body
            assert len(request_body["tools"]) == 2

            tool = request_body["tools"][0]
            assert tool["name"] == "file_read"
            assert tool["description"] == "Read file contents"
            assert "input_schema" in tool

    @pytest.mark.asyncio
    async def test_send_message_with_tools_timeout(
        self, claude_provider, sample_chat_context, sample_tools
    ):
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("timeout")
            )

            with pytest.raises(NetworkError, match="timed out"):
                await claude_provider.send_message_with_tools(
                    "Read file", sample_chat_context, sample_tools
                )

    @pytest.mark.asyncio
    async def test_send_message_with_tools_network_error(
        self, claude_provider, sample_chat_context, sample_tools
    ):
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("connection failed")
            )

            with pytest.raises(NetworkError):
                await claude_provider.send_message_with_tools(
                    "Read file", sample_chat_context, sample_tools
                )

    def test_convert_tools_to_claude_format(self, claude_provider, sample_tools):
        claude_tools = claude_provider._convert_tools_to_claude_format(sample_tools)

        assert len(claude_tools) == 2

        tool = claude_tools[0]
        assert tool["name"] == "file_read"
        assert tool["description"] == "Read file contents"
        assert tool["input_schema"] == {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to read",
                }
            },
            "required": ["path"],
        }

    def test_convert_tools_empty_list(self, claude_provider):
        assert claude_provider._convert_tools_to_claude_format([]) == []

    def test_parse_tool_calls_from_response(self, claude_provider, mock_tool_use_response):
        content_blocks = mock_tool_use_response["content"]
        text, tool_calls = claude_provider._parse_response_content(content_blocks)

        assert text == "I'll read that file for you."
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "file_read"
        assert tool_calls[0].arguments == {"path": "/src/main.py"}

    def test_parse_text_only_response(self, claude_provider, mock_text_only_response):
        content_blocks = mock_text_only_response["content"]
        text, tool_calls = claude_provider._parse_response_content(content_blocks)

        assert text == "Here's what I found."
        assert len(tool_calls) == 0

    def test_parse_multiple_tool_calls(self, claude_provider, mock_multi_tool_response):
        content_blocks = mock_multi_tool_response["content"]
        text, tool_calls = claude_provider._parse_response_content(content_blocks)

        assert text == "Let me read the file and run the tests."
        assert len(tool_calls) == 2


class TestClaudeProviderBasic:
    """Test basic Claude provider functionality."""

    def test_supports_tools(self, claude_provider):
        assert claude_provider.supports_tools() is True

    def test_supports_multimodal(self, claude_provider):
        assert claude_provider.supports_multimodal() is True

    def test_default_model(self):
        provider = ClaudeProvider(api_key="sk-test")
        assert provider.model == "claude-sonnet-4-6"

    def test_custom_model(self):
        provider = ClaudeProvider(api_key="sk-test", model="claude-opus-4-20250514")
        assert provider.model == "claude-opus-4-20250514"

    def test_get_model_info(self, claude_provider):
        info = claude_provider.get_model_info()
        assert info.name == "claude-sonnet-4-20250514"
        assert info.provider == "claude"
        assert info.supports_tools is True


class TestClaudeProviderStreaming:
    """Test Claude provider SSE streaming."""

    @pytest.fixture
    def streaming_provider(self):
        return ClaudeProvider(api_key="sk-test-key-123", model="claude-sonnet-4-20250514")

    @pytest.fixture
    def sample_context(self):
        return ChatContext(messages=[], current_model="test", session_id="test")

    def test_supports_streaming(self, streaming_provider):
        assert streaming_provider.supports_streaming() is True

    @pytest.mark.asyncio
    async def test_stream_text_only(self, streaming_provider, sample_context):
        from omnimancer.core.models import StreamEventType

        sse_lines = [
            'event: message_start',
            'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"usage":{"input_tokens":25,"output_tokens":0}}}',
            '',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
            '',
            'event: content_block_stop',
            'data: {"type":"content_block_stop","index":0}',
            '',
            'event: message_delta',
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}',
            '',
            'event: message_stop',
            'data: {"type":"message_stop"}',
            '',
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = fake_aiter_lines

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)

            mock_client_cls.return_value = mock_client

            events = []
            async for event in streaming_provider.send_message_stream("Hello", sample_context):
                events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.MESSAGE_START in types
        assert StreamEventType.TEXT_DELTA in types
        assert StreamEventType.MESSAGE_COMPLETE in types

        text_chunks = [e.text for e in events if e.type == StreamEventType.TEXT_DELTA]
        assert "".join(text_chunks) == "Hello world"

        complete = [e for e in events if e.type == StreamEventType.MESSAGE_COMPLETE][0]
        assert complete.response.content == "Hello world"
        assert complete.response.input_tokens == 25
        assert complete.response.output_tokens == 10
        assert complete.response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_stream_with_tool_use(self, streaming_provider, sample_context):
        from omnimancer.core.models import StreamEventType

        sse_lines = [
            'event: message_start',
            'data: {"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"usage":{"input_tokens":50,"output_tokens":0}}}',
            '',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Reading file."}}',
            '',
            'event: content_block_stop',
            'data: {"type":"content_block_stop","index":0}',
            '',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_abc","name":"file_read"}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\""}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":": \\"/main.py\\"}"}}',
            '',
            'event: content_block_stop',
            'data: {"type":"content_block_stop","index":1}',
            '',
            'event: message_delta',
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}',
            '',
            'event: message_stop',
            'data: {"type":"message_stop"}',
            '',
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response.aiter_lines = fake_aiter_lines

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            mock_stream_ctx = AsyncMock()
            mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(return_value=mock_stream_ctx)

            mock_client_cls.return_value = mock_client

            events = []
            async for event in streaming_provider.send_message_with_tools_stream(
                "Read main.py", sample_context, []
            ):
                events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.TOOL_USE_START in types
        assert StreamEventType.TOOL_USE_DELTA in types
        assert StreamEventType.TOOL_USE_END in types

        tool_start = [e for e in events if e.type == StreamEventType.TOOL_USE_START][0]
        assert tool_start.tool_name == "file_read"
        assert tool_start.tool_id == "toolu_abc"

        complete = [e for e in events if e.type == StreamEventType.MESSAGE_COMPLETE][0]
        assert complete.response.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_stream_http_error(self, streaming_provider, sample_context):
        import httpx

        from omnimancer.core.models import StreamEventType
        from omnimancer.utils.errors import NetworkError

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream = MagicMock(
                side_effect=httpx.TimeoutException("timeout")
            )
            mock_client_cls.return_value = mock_client

            with pytest.raises(NetworkError, match="timed out"):
                async for _ in streaming_provider.send_message_stream("hi", sample_context):
                    pass
