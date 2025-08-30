"""
Unit tests for Mistral AI provider implementation.

This module tests the MistralProvider class functionality including
message sending, tool calling, safety settings, and error handling.
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
from omnimancer.providers.mistral import MistralProvider
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
)


@pytest.fixture
def mistral_provider():
    """Create a MistralProvider instance for testing."""
    return MistralProvider(
        api_key="test-mistral-key",
        model="mistral-large-latest",
        max_tokens=4096,
        temperature=0.7,
        top_p=1.0,
        safe_prompt=False,
    )


@pytest.fixture
def mistral_provider_safe():
    """Create a MistralProvider instance with safe prompt enabled."""
    return MistralProvider(
        api_key="test-mistral-key",
        model="mistral-large-latest",
        safe_prompt=True,
        random_seed=42,
    )


@pytest.fixture
def mistral_provider_json():
    """Create a MistralProvider instance with JSON response format."""
    return MistralProvider(
        api_key="test-mistral-key",
        model="mistral-large-latest",
        response_format={"type": "json_object"},
    )


@pytest.fixture
def sample_chat_context():
    """Create a sample chat context for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Explain machine learning concepts",
            timestamp=datetime.now(),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Machine learning is a subset of AI that enables systems to learn from data.",
            timestamp=datetime.now(),
            model_used="mistral-large-latest",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="mistral-large-latest",
        session_id="test-session",
        max_context_length=4000,
    )


@pytest.fixture
def sample_tools():
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="code_interpreter",
            description="Execute Python code and return results",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        ),
        ToolDefinition(
            name="file_search",
            description="Search for files in the codebase",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern or filename",
                    }
                },
                "required": ["pattern"],
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
                    "content": "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning.",
                }
            }
        ],
        "usage": {
            "total_tokens": 45,
            "prompt_tokens": 20,
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
                    "content": "I'll execute the code for you.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "code_interpreter",
                                "arguments": '{"code": "print(\\"Hello, World!\\")\\nresult = 2 + 2\\nprint(f\\"2 + 2 = {result}\\")"}',
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {
            "total_tokens": 35,
            "prompt_tokens": 15,
            "completion_tokens": 20,
        },
    }


class TestMistralProviderInitialization:
    """Test MistralProvider initialization and configuration."""

    def test_initialization_with_defaults(self):
        """Test provider initialization with default values."""
        provider = MistralProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model == "mistral-large-latest"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.7
        assert provider.top_p == 1.0
        assert provider.safe_prompt is False
        assert provider.random_seed is None
        assert provider.response_format is None

    def test_initialization_with_custom_values(self, mistral_provider):
        """Test provider initialization with custom values."""
        assert mistral_provider.api_key == "test-mistral-key"
        assert mistral_provider.model == "mistral-large-latest"
        assert mistral_provider.max_tokens == 4096
        assert mistral_provider.temperature == 0.7
        assert mistral_provider.top_p == 1.0
        assert mistral_provider.safe_prompt is False

    def test_initialization_with_safety_settings(self, mistral_provider_safe):
        """Test initialization with safety settings."""
        assert mistral_provider_safe.safe_prompt is True
        assert mistral_provider_safe.random_seed == 42

    def test_initialization_with_json_format(self, mistral_provider_json):
        """Test initialization with JSON response format."""
        assert mistral_provider_json.response_format == {"type": "json_object"}


class TestMistralProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, mistral_provider, sample_chat_context, mock_successful_response
    ):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await mistral_provider.send_message(
                "Explain ML types", sample_chat_context
            )

            assert (
                response.content
                == "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning."
            )
            assert response.model_used == "mistral-large-latest"
            assert response.tokens_used == 45
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_with_safety_settings(
        self, mistral_provider_safe, sample_chat_context
    ):
        """Test message sending with safety settings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Safe response"}}],
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await mistral_provider_safe.send_message(
                "Test message", sample_chat_context
            )

            # Check that safety settings are included
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert payload["safe_prompt"] is True
            assert payload["random_seed"] == 42

    @pytest.mark.asyncio
    async def test_send_message_with_json_format(
        self, mistral_provider_json, sample_chat_context
    ):
        """Test message sending with JSON response format."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"result": "JSON response"}',
                    }
                }
            ],
            "usage": {"total_tokens": 15},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await mistral_provider_json.send_message("Return JSON", sample_chat_context)

            # Check that response format is included
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert payload["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, mistral_provider, sample_chat_context):
        """Test message sending with timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(NetworkError, match="Request to Mistral API timed out"):
                await mistral_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_error(
        self, mistral_provider, sample_chat_context
    ):
        """Test message sending with network error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(NetworkError, match="Network error"):
                await mistral_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, mistral_provider, sample_chat_context
    ):
        """Test message sending with authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(AuthenticationError, match="Invalid Mistral API key"):
                await mistral_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, mistral_provider, sample_chat_context
    ):
        """Test message sending with rate limit error."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RateLimitError, match="Mistral API rate limit exceeded"):
                await mistral_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_model_not_found(
        self, mistral_provider, sample_chat_context
    ):
        """Test message sending with model not found error."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ModelNotFoundError, match="Mistral model .* not found"):
                await mistral_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_empty_response(
        self, mistral_provider, sample_chat_context
    ):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="Empty response from Mistral API"):
                await mistral_provider.send_message("Test message", sample_chat_context)


class TestMistralProviderToolCalling:
    """Test tool calling functionality."""

    @pytest.mark.asyncio
    async def test_send_message_with_tools_success(
        self,
        mistral_provider,
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

            response = await mistral_provider.send_message_with_tools(
                "Run some Python code", sample_chat_context, sample_tools
            )

            assert response.content == "I'll execute the code for you."
            assert response.model_used == "mistral-large-latest"
            assert response.tokens_used == 35
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "code_interpreter"

    @pytest.mark.asyncio
    async def test_send_message_with_tools_payload_format(
        self, mistral_provider, sample_chat_context, sample_tools
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

            await mistral_provider.send_message_with_tools(
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
            assert tools[0]["function"]["name"] == "code_interpreter"
            assert (
                tools[0]["function"]["description"]
                == "Execute Python code and return results"
            )

            # Check second tool
            assert tools[1]["type"] == "function"
            assert tools[1]["function"]["name"] == "file_search"

    @pytest.mark.asyncio
    async def test_send_message_with_tools_and_safety(
        self, mistral_provider_safe, sample_chat_context, sample_tools
    ):
        """Test tool calling with safety settings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Safe tool response",
                    }
                }
            ],
            "usage": {"total_tokens": 15},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await mistral_provider_safe.send_message_with_tools(
                "Test message", sample_chat_context, sample_tools
            )

            # Check that safety settings are preserved with tools
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert payload["safe_prompt"] is True
            assert payload["random_seed"] == 42
            assert "tools" in payload

    @pytest.mark.asyncio
    async def test_convert_tools_to_mistral_format(
        self, mistral_provider, sample_tools
    ):
        """Test tool conversion to Mistral format."""
        mistral_tools = mistral_provider._convert_tools_to_mistral_format(sample_tools)

        assert len(mistral_tools) == 2

        # Check first tool
        tool1 = mistral_tools[0]
        assert tool1["type"] == "function"
        assert tool1["function"]["name"] == "code_interpreter"
        assert (
            tool1["function"]["description"] == "Execute Python code and return results"
        )
        assert "parameters" in tool1["function"]

        # Check second tool
        tool2 = mistral_tools[1]
        assert tool2["type"] == "function"
        assert tool2["function"]["name"] == "file_search"
        assert tool2["function"]["description"] == "Search for files in the codebase"


class TestMistralProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, mistral_provider):
        """Test successful credential validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await mistral_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, mistral_provider):
        """Test credential validation failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await mistral_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_exception(self, mistral_provider):
        """Test credential validation with exception."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Connection error")
            )

            result = await mistral_provider.validate_credentials()
            assert result is False


class TestMistralProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_large_model(self, mistral_provider):
        """Test getting model info for Mistral Large."""
        model_info = mistral_provider.get_model_info()

        assert isinstance(model_info, EnhancedModelInfo)
        assert model_info.name == "mistral-large-latest"
        assert model_info.provider == "mistral"
        assert (
            model_info.description
            == "Mistral Large - Most capable model for complex tasks"
        )
        assert model_info.max_tokens == 128000
        assert model_info.cost_per_million_input == 2.0
        assert model_info.cost_per_million_output == 6.0
        assert model_info.swe_score == 64.6
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is False
        assert model_info.latest_version is True
        assert model_info.is_free is False

    def test_get_model_info_codestral(self):
        """Test getting model info for Codestral."""
        provider = MistralProvider(api_key="test-key", model="codestral-latest")
        model_info = provider.get_model_info()

        assert model_info.name == "codestral-latest"
        assert (
            model_info.description
            == "Codestral - Specialized for code generation and analysis"
        )
        assert model_info.swe_score == 78.2  # Higher SWE score for code model
        assert model_info.supports_tools is True

    def test_get_model_info_tiny_model(self):
        """Test getting model info for Mistral Tiny."""
        provider = MistralProvider(api_key="test-key", model="mistral-tiny")
        model_info = provider.get_model_info()

        assert model_info.name == "mistral-tiny"
        assert model_info.description == "Mistral Tiny - Ultra-fast for simple tasks"
        assert model_info.swe_score == 35.1
        assert model_info.supports_tools is False  # Tiny doesn't support tools

    def test_get_model_info_unknown_model(self):
        """Test getting model info for unknown model."""
        provider = MistralProvider(api_key="test-key", model="unknown-model")
        model_info = provider.get_model_info()

        assert model_info.name == "unknown-model"
        assert model_info.provider == "mistral"
        assert model_info.description == "Mistral model unknown-model"
        assert model_info.swe_score == 50.0  # Default value

    def test_get_available_models(self, mistral_provider):
        """Test getting list of available models."""
        models = mistral_provider.get_available_models()

        assert len(models) == 5

        # Check that all models are EnhancedModelInfo instances
        for model in models:
            assert isinstance(model, EnhancedModelInfo)
            assert model.provider == "mistral"

        # Check specific models
        large_model = next(m for m in models if m.name == "mistral-large-latest")
        tiny_model = next(m for m in models if m.name == "mistral-tiny")
        codestral_model = next(m for m in models if m.name == "codestral-latest")

        assert large_model.latest_version is True
        assert large_model.supports_tools is True
        assert tiny_model.supports_tools is False
        assert codestral_model.swe_score == 78.2  # Highest SWE score


class TestMistralProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools_large_model(self, mistral_provider):
        """Test tool support for large models."""
        assert mistral_provider.supports_tools() is True

    def test_supports_tools_tiny_model(self):
        """Test tool support for tiny models."""
        provider = MistralProvider(api_key="test-key", model="mistral-tiny")
        assert provider.supports_tools() is False

    def test_supports_multimodal(self, mistral_provider):
        """Test multimodal support."""
        assert mistral_provider.supports_multimodal() is False

    def test_supports_streaming(self, mistral_provider):
        """Test streaming support."""
        assert mistral_provider.supports_streaming() is True


class TestMistralProviderMessagePreparation:
    """Test message preparation for API requests."""

    def test_prepare_messages_with_context(self, mistral_provider, sample_chat_context):
        """Test preparing messages with conversation context."""
        messages = mistral_provider._prepare_messages(
            "New message", sample_chat_context
        )

        assert len(messages) == 3  # 2 from context + 1 new

        # Check context messages
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Explain machine learning concepts"
        assert messages[1]["role"] == "assistant"
        assert (
            messages[1]["content"]
            == "Machine learning is a subset of AI that enables systems to learn from data."
        )

        # Check new message
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "New message"

    def test_prepare_messages_empty_context(self, mistral_provider):
        """Test preparing messages with empty context."""
        empty_context = ChatContext(
            messages=[],
            current_model="mistral-large-latest",
            session_id="test-session",
        )

        messages = mistral_provider._prepare_messages("Hello", empty_context)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"


class TestMistralProviderResponseHandling:
    """Test response handling functionality."""

    def test_handle_response_success(self, mistral_provider, mock_successful_response):
        """Test handling successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        response = mistral_provider._handle_response(mock_response)

        assert (
            response.content
            == "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning."
        )
        assert response.model_used == "mistral-large-latest"
        assert response.tokens_used == 45
        assert response.timestamp is not None

    def test_handle_response_with_tools_success(
        self, mistral_provider, mock_tool_response
    ):
        """Test handling response with tool calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tool_response

        response = mistral_provider._handle_response_with_tools(mock_response)

        assert response.content == "I'll execute the code for you."
        assert response.model_used == "mistral-large-latest"
        assert response.tokens_used == 35
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "code_interpreter"

    def test_handle_response_with_error_json(self, mistral_provider):
        """Test handling response with JSON error message."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid request format"}
        }

        with pytest.raises(
            ProviderError, match="Mistral API error: Invalid request format"
        ):
            mistral_provider._handle_response(mock_response)

    def test_handle_response_with_error_no_json(self, mistral_provider):
        """Test handling response with error but no JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ProviderError, match="Mistral API error: HTTP 500"):
            mistral_provider._handle_response(mock_response)
