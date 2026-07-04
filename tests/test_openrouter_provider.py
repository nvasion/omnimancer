"""
Unit tests for OpenRouter provider implementation.

This module tests the OpenRouterProvider class functionality including
construction, message sending, tool calling, SSL fallback behaviour,
credential validation, model info, and capability methods.
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
from omnimancer.providers.openrouter import OpenRouterProvider
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openrouter_provider():
    """Create an OpenRouterProvider instance for testing."""
    return OpenRouterProvider(
        api_key="test-openrouter-key",
        model="anthropic/claude-3.5-sonnet",
        max_tokens=4096,
        temperature=0.7,
    )


@pytest.fixture
def openrouter_provider_gpt4():
    """Create an OpenRouterProvider with a GPT-4o model."""
    return OpenRouterProvider(
        api_key="test-openrouter-key",
        model="openai/gpt-4o",
        enable_fallback=False,
    )


@pytest.fixture
def openrouter_provider_llama():
    """Create an OpenRouterProvider with a Llama model (no multimodal)."""
    return OpenRouterProvider(
        api_key="test-openrouter-key",
        model="meta-llama/llama-3.1-405b-instruct",
    )


@pytest.fixture
def openrouter_provider_no_fallback():
    """Create an OpenRouterProvider with fallback disabled."""
    return OpenRouterProvider(
        api_key="test-openrouter-key",
        model="anthropic/claude-3.5-sonnet",
        enable_fallback=False,
    )


@pytest.fixture
def sample_chat_context():
    """Create a sample chat context for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="What is OpenRouter?",
            timestamp=datetime.now(),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="OpenRouter is an API aggregator for AI models.",
            timestamp=datetime.now(),
            model_used="anthropic/claude-3.5-sonnet",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="anthropic/claude-3.5-sonnet",
        session_id="test-session",
        max_context_length=4000,
    )


@pytest.fixture
def sample_tools():
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="fetch_url",
            description="Fetch content from a URL",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"],
            },
        ),
        ToolDefinition(
            name="summarize_text",
            description="Summarize a block of text",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"},
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum summary length",
                    },
                },
                "required": ["text"],
            },
        ),
    ]


@pytest.fixture
def mock_successful_response():
    """Create a mock successful OpenRouter API response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "OpenRouter provides unified access to many AI models."
                    ),
                }
            }
        ],
        "model": "anthropic/claude-3.5-sonnet",
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 15,
            "total_tokens": 35,
        },
    }


@pytest.fixture
def mock_tool_response():
    """Create a mock OpenRouter API response with tool calls."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I will fetch that URL for you.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "fetch_url",
                                "arguments": '{"url": "https://example.com"}',
                            },
                        }
                    ],
                }
            }
        ],
        "model": "anthropic/claude-3.5-sonnet",
        "usage": {"total_tokens": 30},
    }


# ---------------------------------------------------------------------------
# TestOpenRouterProviderInitialization
# ---------------------------------------------------------------------------


class TestOpenRouterProviderInitialization:
    """Test OpenRouterProvider initialization and configuration."""

    def test_initialization_with_defaults(self):
        """Test provider initialization with default values."""
        provider = OpenRouterProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model == "anthropic/claude-3.5-sonnet"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.7
        assert provider.top_p == 1.0
        assert provider.frequency_penalty == 0.0
        assert provider.presence_penalty == 0.0
        assert provider.enable_fallback is True
        assert provider.max_cost_per_token is None
        assert provider.prefer_cheaper_models is False
        assert provider.show_fallback_warnings is True

    def test_initialization_with_custom_values(self, openrouter_provider):
        """Test provider initialization with custom values."""
        assert openrouter_provider.api_key == "test-openrouter-key"
        assert openrouter_provider.model == "anthropic/claude-3.5-sonnet"
        assert openrouter_provider.max_tokens == 4096
        assert openrouter_provider.temperature == 0.7

    def test_initialization_default_base_url(self, openrouter_provider):
        """Test that the default base URL is correct."""
        assert openrouter_provider.base_url == "https://openrouter.ai/api/v1"

    def test_initialization_custom_base_url(self):
        """Test provider initialization with a custom base URL."""
        provider = OpenRouterProvider(
            api_key="test-key",
            base_url="https://my-proxy.example.com/api/v1",
        )
        assert provider.base_url == "https://my-proxy.example.com/api/v1"

    def test_initialization_base_url_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from base URL."""
        provider = OpenRouterProvider(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
        )
        assert not provider.base_url.endswith("/")

    def test_initialization_default_referrer_header(self, openrouter_provider):
        """Test that default referrer is set."""
        assert "github.com/omnimancer-cli" in openrouter_provider.openrouter_referrer

    def test_initialization_custom_referrer(self):
        """Test provider initialization with custom referrer."""
        provider = OpenRouterProvider(
            api_key="test-key",
            openrouter_referrer="https://my-app.example.com",
        )
        assert provider.openrouter_referrer == "https://my-app.example.com"

    def test_initialization_fallback_disabled(self, openrouter_provider_no_fallback):
        """Test provider initialization with fallback disabled."""
        assert openrouter_provider_no_fallback.enable_fallback is False


# ---------------------------------------------------------------------------
# TestOpenRouterProviderHeaders
# ---------------------------------------------------------------------------


class TestOpenRouterProviderHeaders:
    """Test header generation."""

    def test_get_headers_includes_authorization(self, openrouter_provider):
        """Test that headers include the Authorization bearer token."""
        headers = openrouter_provider._get_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-openrouter-key"

    def test_get_headers_includes_referer(self, openrouter_provider):
        """Test that headers include HTTP-Referer."""
        headers = openrouter_provider._get_headers()

        assert "HTTP-Referer" in headers

    def test_get_headers_includes_title(self, openrouter_provider):
        """Test that headers include X-Title."""
        headers = openrouter_provider._get_headers()

        assert "X-Title" in headers
        assert headers["X-Title"] == "Omnimancer CLI"

    def test_get_headers_includes_content_type(self, openrouter_provider):
        """Test that headers include Content-Type."""
        headers = openrouter_provider._get_headers()

        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# TestOpenRouterProviderMessagePreparation
# ---------------------------------------------------------------------------


class TestOpenRouterProviderMessagePreparation:
    """Test message preparation for OpenRouter API format."""

    def test_prepare_messages_includes_context(
        self, openrouter_provider, sample_chat_context
    ):
        """Test that context messages are included in prepared messages."""
        messages = openrouter_provider._prepare_messages(
            "New question", sample_chat_context
        )

        # 2 from context + 1 new
        assert len(messages) == 3

    def test_prepare_messages_includes_new_message(
        self, openrouter_provider, sample_chat_context
    ):
        """Test that the new message is appended at the end."""
        messages = openrouter_provider._prepare_messages(
            "Tell me more", sample_chat_context
        )

        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Tell me more"

    def test_prepare_messages_empty_context(self, openrouter_provider):
        """Test message preparation with empty context."""
        empty_context = ChatContext(
            messages=[],
            current_model="anthropic/claude-3.5-sonnet",
            session_id="test-session",
        )
        messages = openrouter_provider._prepare_messages("Hello", empty_context)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_convert_tools_to_openrouter_format(
        self, openrouter_provider, sample_tools
    ):
        """Test tool conversion to OpenRouter format."""
        converted = openrouter_provider._convert_tools_to_openrouter_format(
            sample_tools
        )

        assert len(converted) == 2

        tool1 = converted[0]
        assert tool1["type"] == "function"
        assert tool1["function"]["name"] == "fetch_url"
        assert tool1["function"]["description"] == "Fetch content from a URL"
        assert "parameters" in tool1["function"]

        tool2 = converted[1]
        assert tool2["type"] == "function"
        assert tool2["function"]["name"] == "summarize_text"


# ---------------------------------------------------------------------------
# TestOpenRouterProviderMessageSending
# ---------------------------------------------------------------------------


class TestOpenRouterProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, openrouter_provider, sample_chat_context, mock_successful_response
    ):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await openrouter_provider.send_message(
                "Tell me more", sample_chat_context
            )

            expected = "OpenRouter provides unified access to many AI models."
            assert response.content == expected
            assert response.model_used == "anthropic/claude-3.5-sonnet"
            assert response.tokens_used == 35
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_with_fallback_route_header(
        self, openrouter_provider, sample_chat_context, mock_successful_response
    ):
        """Test that fallback=True adds 'route': 'fallback' to payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await openrouter_provider.send_message("Hello", sample_chat_context)

            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]
            assert payload.get("route") == "fallback"

    @pytest.mark.asyncio
    async def test_send_message_no_fallback_route_when_disabled(
        self,
        openrouter_provider_no_fallback,
        sample_chat_context,
        mock_successful_response,
    ):
        """Test that fallback=False does NOT add 'route': 'fallback'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await openrouter_provider_no_fallback.send_message(
                "Hello", sample_chat_context
            )

            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]
            assert "route" not in payload

    @pytest.mark.asyncio
    async def test_send_message_model_fallback_warning_added(
        self, openrouter_provider, sample_chat_context
    ):
        """Test that a fallback warning is prepended when a different model is used."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Response text"}}],
            "model": "openai/gpt-4o",  # Different from requested model
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await openrouter_provider.send_message(
                "Hello", sample_chat_context
            )

            assert "Model Fallback Notice" in response.content
            assert response.model_used == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, openrouter_provider, sample_chat_context
    ):
        """Test send_message with 401 raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(AuthenticationError, match="Invalid OpenRouter API key"):
                await openrouter_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, openrouter_provider, sample_chat_context
    ):
        """Test send_message with 429 raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                RateLimitError, match="OpenRouter API rate limit exceeded"
            ):
                await openrouter_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_model_not_found(
        self, openrouter_provider, sample_chat_context
    ):
        """Test send_message with 404 raises ModelNotFoundError."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ModelNotFoundError, match="not found"):
                await openrouter_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, openrouter_provider, sample_chat_context):
        """Test send_message with timeout raises NetworkError."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timed out")
            )

            with pytest.raises(NetworkError, match="timed out"):
                await openrouter_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_non_ssl_connection_error(
        self, openrouter_provider, sample_chat_context
    ):
        """Test send_message with a non-SSL connection error raises NetworkError."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with pytest.raises(NetworkError, match="Connection error"):
                await openrouter_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_empty_choices(
        self, openrouter_provider, sample_chat_context
    ):
        """Test that empty choices list raises ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [],
            "model": "anthropic/claude-3.5-sonnet",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="Empty response from OpenRouter"):
                await openrouter_provider.send_message("Hello", sample_chat_context)


# ---------------------------------------------------------------------------
# TestOpenRouterProviderToolCalling
# ---------------------------------------------------------------------------


class TestOpenRouterProviderToolCalling:
    """Test tool calling functionality."""

    @pytest.mark.asyncio
    async def test_send_message_with_tools_success(
        self,
        openrouter_provider,
        sample_chat_context,
        sample_tools,
        mock_tool_response,
    ):
        """Test successful message sending with tool calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tool_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await openrouter_provider.send_message_with_tools(
                "Fetch example.com", sample_chat_context, sample_tools
            )

            assert response.content == "I will fetch that URL for you."
            assert response.model_used == "anthropic/claude-3.5-sonnet"
            assert response.tokens_used == 30
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "fetch_url"

    @pytest.mark.asyncio
    async def test_send_message_with_tools_payload_format(
        self, openrouter_provider, sample_chat_context, sample_tools
    ):
        """Test that tools are formatted correctly in the request payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "model": "anthropic/claude-3.5-sonnet",
            "usage": {"total_tokens": 5},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await openrouter_provider.send_message_with_tools(
                "Fetch a page", sample_chat_context, sample_tools
            )

            call_kwargs = mock_post.call_args[1]
            payload = call_kwargs["json"]

            assert "tools" in payload
            assert "tool_choice" in payload
            assert payload["tool_choice"] == "auto"

            tools = payload["tools"]
            assert len(tools) == 2
            assert tools[0]["type"] == "function"
            assert tools[0]["function"]["name"] == "fetch_url"


# ---------------------------------------------------------------------------
# TestOpenRouterProviderCredentialValidation
# ---------------------------------------------------------------------------


class TestOpenRouterProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, openrouter_provider):
        """Test successful credential validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await openrouter_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, openrouter_provider):
        """Test credential validation failure (non-200 response)."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await openrouter_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_exception(self, openrouter_provider):
        """Test credential validation when exception occurs."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Connection error")
            )

            result = await openrouter_provider.validate_credentials()
            assert result is False


# ---------------------------------------------------------------------------
# TestOpenRouterProviderModelInfo
# ---------------------------------------------------------------------------


class TestOpenRouterProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_claude_sonnet(self, openrouter_provider):
        """Test getting model info for Claude 3.5 Sonnet."""
        model_info = openrouter_provider.get_model_info()

        assert isinstance(model_info, EnhancedModelInfo)
        assert model_info.name == "anthropic/claude-3.5-sonnet"
        assert model_info.provider == "openrouter"
        assert "Claude 3.5 Sonnet" in model_info.description
        assert model_info.max_tokens == 200000
        assert model_info.cost_per_million_input == 3.0
        assert model_info.cost_per_million_output == 15.0
        assert model_info.swe_score == 88.7
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is True
        assert model_info.latest_version is True
        assert model_info.is_free is False

    def test_get_model_info_gpt4o(self, openrouter_provider_gpt4):
        """Test getting model info for GPT-4o via OpenRouter."""
        model_info = openrouter_provider_gpt4.get_model_info()

        assert model_info.name == "openai/gpt-4o"
        assert model_info.max_tokens == 128000
        assert model_info.swe_score == 71.2
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is True

    def test_get_model_info_llama(self, openrouter_provider_llama):
        """Test getting model info for Llama via OpenRouter."""
        model_info = openrouter_provider_llama.get_model_info()

        assert model_info.name == "meta-llama/llama-3.1-405b-instruct"
        assert model_info.supports_multimodal is False
        assert model_info.supports_tools is True

    def test_get_model_info_unknown_model(self):
        """Test getting model info for an unknown model returns defaults."""
        provider = OpenRouterProvider(
            api_key="test-key",
            model="vendor/unknown-model",
        )
        model_info = provider.get_model_info()

        assert model_info.name == "vendor/unknown-model"
        assert model_info.provider == "openrouter"
        assert model_info.swe_score == 50.0

    def test_get_available_models(self, openrouter_provider):
        """Test getting the list of available models."""
        models = openrouter_provider.get_available_models()

        assert len(models) == 6
        for model in models:
            assert isinstance(model, EnhancedModelInfo)
            assert model.provider == "openrouter"

        model_names = [m.name for m in models]
        assert "anthropic/claude-3.5-sonnet" in model_names
        assert "openai/gpt-4o" in model_names
        assert "meta-llama/llama-3.1-405b-instruct" in model_names

        latest = next(m for m in models if m.name == "anthropic/claude-3.5-sonnet")
        assert latest.latest_version is True


# ---------------------------------------------------------------------------
# TestOpenRouterProviderCapabilities
# ---------------------------------------------------------------------------


class TestOpenRouterProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools(self, openrouter_provider):
        """Test that OpenRouter supports tool calling."""
        assert openrouter_provider.supports_tools() is True

    def test_supports_multimodal_claude(self, openrouter_provider):
        """Test that Claude models support multimodal via OpenRouter."""
        assert openrouter_provider.supports_multimodal() is True

    def test_supports_multimodal_gpt4o(self, openrouter_provider_gpt4):
        """Test that GPT-4o supports multimodal via OpenRouter."""
        assert openrouter_provider_gpt4.supports_multimodal() is True

    def test_supports_multimodal_llama(self, openrouter_provider_llama):
        """Test that Llama models do NOT support multimodal via OpenRouter."""
        assert openrouter_provider_llama.supports_multimodal() is False

    def test_supports_streaming(self, openrouter_provider):
        """Test that OpenRouter supports streaming."""
        assert openrouter_provider.supports_streaming() is True
