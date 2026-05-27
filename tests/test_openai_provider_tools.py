"""Tests for OpenAI provider native tool calling."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core.models import ChatContext, ChatMessage, MessageRole, ToolDefinition
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.utils.errors import NetworkError


@pytest.fixture
def openai_provider():
    return OpenAIProvider(api_key="sk-test-key", model="gpt-4")


@pytest.fixture
def sample_chat_context():
    return ChatContext(
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content="Hello",
                timestamp=datetime.now(),
                model_used="gpt-4",
            ),
        ],
        current_model="gpt-4",
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
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        ),
    ]


@pytest.fixture
def mock_tool_call_response():
    return {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "file_read",
                                "arguments": '{"path": "/src/main.py"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }


@pytest.fixture
def mock_text_response():
    return {
        "id": "chatcmpl-456",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here's the answer.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
    }


class TestOpenAIToolCalling:

    @pytest.mark.asyncio
    async def test_send_message_with_tools_returns_tool_calls(
        self,
        openai_provider,
        sample_chat_context,
        sample_tools,
        mock_tool_call_response,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tool_call_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await openai_provider.send_message_with_tools(
                "Read main.py", sample_chat_context, sample_tools
            )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "file_read"
        assert response.tool_calls[0].arguments == {"path": "/src/main.py"}

    @pytest.mark.asyncio
    async def test_send_message_with_tools_text_only(
        self, openai_provider, sample_chat_context, sample_tools, mock_text_response
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_text_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await openai_provider.send_message_with_tools(
                "What is Python?", sample_chat_context, sample_tools
            )

        assert response.content == "Here's the answer."
        assert response.tool_calls is None

    @pytest.mark.asyncio
    async def test_sends_correct_openai_tool_format(
        self, openai_provider, sample_chat_context, sample_tools, mock_text_response
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_text_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await openai_provider.send_message_with_tools(
                "Test", sample_chat_context, sample_tools
            )

            call_kwargs = mock_post.call_args
            request_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

            assert "tools" in request_body
            tool = request_body["tools"][0]
            assert tool["type"] == "function"
            assert tool["function"]["name"] == "file_read"
            assert "parameters" in tool["function"]

    def test_convert_tools_to_openai_format(self, openai_provider, sample_tools):
        openai_tools = openai_provider._convert_tools_to_openai_format(sample_tools)

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "file_read"

    def test_convert_empty_tools(self, openai_provider):
        assert openai_provider._convert_tools_to_openai_format([]) == []

    @pytest.mark.asyncio
    async def test_timeout_raises_network_error(
        self, openai_provider, sample_chat_context, sample_tools
    ):
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("timeout")
            )

            with pytest.raises(NetworkError, match="timed out"):
                await openai_provider.send_message_with_tools(
                    "Read file", sample_chat_context, sample_tools
                )
