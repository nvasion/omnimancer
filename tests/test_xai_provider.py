"""
Unit tests for xAI (Grok) provider implementation.

This module tests the XAIProvider class functionality including
message sending, tool calling, multimodal support, and error handling.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from omnimancer.core.models import (
    ChatContext,
    ChatMessage,
    EnhancedModelInfo,
    MessageRole,
    ToolDefinition,
)
from omnimancer.providers.xai import XAIProvider
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
)


@pytest.fixture
def xai_provider():
    """Create an XAIProvider instance for testing."""
    return XAIProvider(
        api_key="test-xai-key",
        model="grok-beta",
        max_tokens=4096,
        temperature=0.7,
        grok_mode="balanced",
        enable_web_search=True,
    )


@pytest.fixture
def xai_provider_creative():
    """Create an XAIProvider instance with creative mode."""
    return XAIProvider(
        api_key="test-xai-key",
        model="grok-beta",
        grok_mode="creative",
        system_prompt="You are a creative writing assistant.",
    )


@pytest.fixture
def xai_provider_precise():
    """Create an XAIProvider instance with precise mode."""
    return XAIProvider(
        api_key="test-xai-key",
        model="grok-beta",
        grok_mode="precise",
        enable_web_search=False,
    )


@pytest.fixture
def sample_chat_context():
    """Create a sample chat context for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="What are the latest developments in AI?",
            timestamp=datetime.now(),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Recent AI developments include advances in reasoning and multimodal capabilities.",
            timestamp=datetime.now(),
            model_used="grok-beta",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="grok-beta",
        session_id="test-session",
        max_context_length=4000,
    )


@pytest.fixture
def sample_tools():
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="web_search",
            description="Search the web for current information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        ),
        ToolDefinition(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        ),
    ]


@pytest.fixture
def mock_successful_response():
    """Create a mock successful API response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Based on the latest information, AI has made significant strides in reasoning capabilities.",
                }
            }
        ],
        "usage": {
            "total_tokens": 50,
            "prompt_tokens": 25,
            "completion_tokens": 25,
        },
    }


@pytest.fixture
def mock_tool_response():
    """Create a mock API response with tool calls."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll search for the latest AI developments.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query": "latest AI developments 2024"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {
            "total_tokens": 40,
            "prompt_tokens": 20,
            "completion_tokens": 20,
        },
    }


class TestXAIProviderInitialization:
    """Test XAIProvider initialization and configuration."""

    def test_initialization_with_defaults(self):
        """Test provider initialization with default values."""
        provider = XAIProvider(api_key="test-key", model="grok-beta")

        assert provider.api_key == "test-key"
        assert provider.model == "grok-beta"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.7
        assert provider.grok_mode == "balanced"
        assert provider.system_prompt is None
        assert provider.enable_web_search is True
        assert provider.enable_real_time is True

    def test_initialization_with_custom_values(self, xai_provider):
        """Test provider initialization with custom values."""
        assert xai_provider.api_key == "test-xai-key"
        assert xai_provider.model == "grok-beta"
        assert xai_provider.max_tokens == 4096
        assert xai_provider.temperature == 0.7
        assert xai_provider.grok_mode == "balanced"
        assert xai_provider.enable_web_search is True

    def test_initialization_with_system_prompt(self, xai_provider_creative):
        """Test initialization with system prompt."""
        assert (
            xai_provider_creative.system_prompt
            == "You are a creative writing assistant."
        )
        assert xai_provider_creative.grok_mode == "creative"

    def test_initialization_precise_mode(self, xai_provider_precise):
        """Test initialization with precise mode."""
        assert xai_provider_precise.grok_mode == "precise"
        assert xai_provider_precise.enable_web_search is False


class TestXAIProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, xai_provider, sample_chat_context, mock_successful_response
    ):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await xai_provider.send_message(
                "Tell me about AI", sample_chat_context
            )

            assert (
                response.content
                == "Based on the latest information, AI has made significant strides in reasoning capabilities."
            )
            assert response.model_used == "grok-beta"
            assert response.tokens_used == 50
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_with_web_search(
        self, xai_provider, sample_chat_context
    ):
        """Test message sending with web search enabled."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await xai_provider.send_message("Test message", sample_chat_context)

            # Check that web search tool is included
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert "tools" in payload
            assert payload["tools"] == [{"type": "web_search"}]

    @pytest.mark.asyncio
    async def test_send_message_no_web_search(
        self, xai_provider_precise, sample_chat_context
    ):
        """Test message sending without web search."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await xai_provider_precise.send_message("Test message", sample_chat_context)

            # Check that web search tool is not included
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert "tools" not in payload or payload.get("tools") == []

    @pytest.mark.asyncio
    async def test_send_message_with_system_prompt(
        self, xai_provider_creative, sample_chat_context
    ):
        """Test message sending with system prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Creative response",
                    }
                }
            ],
            "usage": {"total_tokens": 15},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await xai_provider_creative.send_message(
                "Write a story", sample_chat_context
            )

            # Check that system prompt is included in messages
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            messages = payload["messages"]

            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a creative writing assistant."

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, xai_provider, sample_chat_context):
        """Test message sending with timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(NetworkError, match="Request to xAI API timed out"):
                await xai_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_error(self, xai_provider, sample_chat_context):
        """Test message sending with network error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(NetworkError, match="Network error"):
                await xai_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, xai_provider, sample_chat_context
    ):
        """Test message sending with authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(AuthenticationError, match="Invalid xAI API key"):
                await xai_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, xai_provider, sample_chat_context
    ):
        """Test message sending with rate limit error."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RateLimitError, match="xAI API rate limit exceeded"):
                await xai_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_model_not_found(
        self, xai_provider, sample_chat_context
    ):
        """Test message sending with model not found error."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ModelNotFoundError, match="xAI model .* not found"):
                await xai_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_empty_response(self, xai_provider, sample_chat_context):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="Empty response from xAI API"):
                await xai_provider.send_message("Test message", sample_chat_context)


class TestXAIProviderToolCalling:
    """Test tool calling functionality."""

    @pytest.mark.asyncio
    async def test_send_message_with_tools_success(
        self,
        xai_provider,
        sample_chat_context,
        sample_tools,
        mock_tool_response,
    ):
        """Test successful message sending with tools."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tool_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await xai_provider.send_message_with_tools(
                "Search for AI news", sample_chat_context, sample_tools
            )

            assert response.content == "I'll search for the latest AI developments."
            assert response.model_used == "grok-beta"
            assert response.tokens_used == 40
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "web_search"

    @pytest.mark.asyncio
    async def test_send_message_with_tools_payload_format(
        self, xai_provider, sample_chat_context, sample_tools
    ):
        """Test that tools are formatted correctly in the request payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await xai_provider.send_message_with_tools(
                "Test message", sample_chat_context, sample_tools
            )

            # Check that tools are formatted correctly
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert "tools" in payload
            assert "tool_choice" in payload
            assert payload["tool_choice"] == "auto"

            tools = payload["tools"]
            assert len(tools) == 2

            # Check first tool
            assert tools[0]["type"] == "function"
            assert tools[0]["function"]["name"] == "web_search"
            assert (
                tools[0]["function"]["description"]
                == "Search the web for current information"
            )

            # Check second tool
            assert tools[1]["type"] == "function"
            assert tools[1]["function"]["name"] == "calculate"

    @pytest.mark.asyncio
    async def test_convert_tools_to_xai_format(self, xai_provider, sample_tools):
        """Test tool conversion to xAI format."""
        xai_tools = xai_provider._convert_tools_to_xai_format(sample_tools)

        assert len(xai_tools) == 2

        # Check first tool
        tool1 = xai_tools[0]
        assert tool1["type"] == "function"
        assert tool1["function"]["name"] == "web_search"
        assert (
            tool1["function"]["description"] == "Search the web for current information"
        )
        assert "parameters" in tool1["function"]

        # Check second tool
        tool2 = xai_tools[1]
        assert tool2["type"] == "function"
        assert tool2["function"]["name"] == "calculate"
        assert tool2["function"]["description"] == "Perform mathematical calculations"


class TestXAIProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, xai_provider):
        """Test successful credential validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await xai_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, xai_provider):
        """Test credential validation failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await xai_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_exception(self, xai_provider):
        """Test credential validation with exception."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Connection error")
            )

            result = await xai_provider.validate_credentials()
            assert result is False


class TestXAIProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_grok_beta(self, xai_provider):
        """Test getting model info for Grok Beta."""
        model_info = xai_provider.get_model_info()

        assert isinstance(model_info, EnhancedModelInfo)
        assert model_info.name == "grok-beta"
        assert model_info.provider == "xai"
        assert (
            model_info.description
            == "Grok Beta - Advanced reasoning with real-time information"
        )
        assert model_info.max_tokens == 131072
        assert model_info.cost_per_million_input == 5.0
        assert model_info.cost_per_million_output == 15.0
        assert model_info.swe_score == 70.1
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is True
        assert model_info.latest_version is True
        assert model_info.is_free is False

    def test_get_model_info_grok_vision(self):
        """Test getting model info for Grok Vision."""
        provider = XAIProvider(api_key="test-key", model="grok-vision-beta")
        model_info = provider.get_model_info()

        assert model_info.name == "grok-vision-beta"
        assert (
            model_info.description
            == "Grok Vision Beta - Multimodal model with image understanding"
        )
        assert model_info.max_tokens == 8192
        assert model_info.swe_score == 65.3
        assert model_info.supports_multimodal is True

    def test_get_model_info_unknown_model(self):
        """Test getting model info for unknown model."""
        provider = XAIProvider(api_key="test-key", model="unknown-model")
        model_info = provider.get_model_info()

        assert model_info.name == "unknown-model"
        assert model_info.provider == "xai"
        assert model_info.description == "xAI model unknown-model"
        assert model_info.swe_score == 65.0  # Default value

    def test_get_available_models(self, xai_provider):
        """Test getting list of available models."""
        models = xai_provider.get_available_models()

        assert len(models) == 2

        # Check that all models are EnhancedModelInfo instances
        for model in models:
            assert isinstance(model, EnhancedModelInfo)
            assert model.provider == "xai"
            assert model.supports_tools is True
            assert model.supports_multimodal is True

        # Check specific models
        grok_beta = next(m for m in models if m.name == "grok-beta")
        grok_vision = next(m for m in models if m.name == "grok-vision-beta")

        assert grok_beta.latest_version is True
        assert grok_vision.latest_version is False
        assert grok_beta.context_window == 131072
        assert grok_vision.context_window == 8192


class TestXAIProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools(self, xai_provider):
        """Test tool support."""
        assert xai_provider.supports_tools() is True

    def test_supports_multimodal(self, xai_provider):
        """Test multimodal support."""
        assert xai_provider.supports_multimodal() is True

    def test_supports_streaming(self, xai_provider):
        """Test streaming support."""
        assert xai_provider.supports_streaming() is True


class TestXAIProviderTemperatureSettings:
    """Test temperature settings based on Grok mode."""

    def test_get_temperature_for_mode_balanced(self, xai_provider):
        """Test temperature for balanced mode."""
        temp = xai_provider._get_temperature_for_mode()
        assert temp == 0.7

    def test_get_temperature_for_mode_precise(self, xai_provider_precise):
        """Test temperature for precise mode."""
        temp = xai_provider_precise._get_temperature_for_mode()
        assert temp == 0.1

    def test_get_temperature_for_mode_creative(self, xai_provider_creative):
        """Test temperature for creative mode."""
        temp = xai_provider_creative._get_temperature_for_mode()
        assert temp == 1.0

    def test_get_temperature_for_mode_unknown(self):
        """Test temperature for unknown mode falls back to default."""
        provider = XAIProvider(
            api_key="test",
            model="grok-beta",
            grok_mode="unknown",
            temperature=0.5,
        )
        temp = provider._get_temperature_for_mode()
        assert temp == 0.5  # Falls back to default temperature


class TestXAIProviderMessagePreparation:
    """Test message preparation for API requests."""

    def test_prepare_messages_with_context(self, xai_provider, sample_chat_context):
        """Test preparing messages with conversation context."""
        messages = xai_provider._prepare_messages("New message", sample_chat_context)

        assert len(messages) == 3  # 2 from context + 1 new

        # Check context messages
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What are the latest developments in AI?"
        assert messages[1]["role"] == "assistant"
        assert (
            messages[1]["content"]
            == "Recent AI developments include advances in reasoning and multimodal capabilities."
        )

        # Check new message
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "New message"

    def test_prepare_messages_with_system_prompt(
        self, xai_provider_creative, sample_chat_context
    ):
        """Test preparing messages with system prompt."""
        messages = xai_provider_creative._prepare_messages(
            "Write a story", sample_chat_context
        )

        assert len(messages) == 4  # system + 2 from context + 1 new

        # Check system message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a creative writing assistant."

        # Check that context and new message follow
        assert messages[1]["role"] == "user"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Write a story"

    def test_prepare_messages_empty_context(self, xai_provider):
        """Test preparing messages with empty context."""
        empty_context = ChatContext(
            messages=[], current_model="grok-beta", session_id="test-session"
        )

        messages = xai_provider._prepare_messages("Hello", empty_context)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"


class TestXAIProviderResponseHandling:
    """Test response handling functionality."""

    def test_handle_response_success(self, xai_provider, mock_successful_response):
        """Test handling successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        response = xai_provider._handle_response(mock_response)

        assert (
            response.content
            == "Based on the latest information, AI has made significant strides in reasoning capabilities."
        )
        assert response.model_used == "grok-beta"
        assert response.tokens_used == 50
        assert response.timestamp is not None

    def test_handle_response_with_tools_success(self, xai_provider, mock_tool_response):
        """Test handling response with tool calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tool_response

        response = xai_provider._handle_response_with_tools(mock_response)

        assert response.content == "I'll search for the latest AI developments."
        assert response.model_used == "grok-beta"
        assert response.tokens_used == 40
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "web_search"
        assert '"query": "latest AI developments 2024"' in str(
            response.tool_calls[0].arguments
        )

    def test_handle_response_with_error_json(self, xai_provider):
        """Test handling response with JSON error message."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid request format"}
        }

        with pytest.raises(
            ProviderError, match="xAI API error: Invalid request format"
        ):
            xai_provider._handle_response(mock_response)

    def test_handle_response_with_error_no_json(self, xai_provider):
        """Test handling response with error but no JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ProviderError, match="xAI API error: HTTP 500"):
            xai_provider._handle_response(mock_response)
