"""
Unit tests for Google Gemini provider implementation.

This module tests the GeminiProvider class functionality including
message sending, credential validation, model information, and tool calling.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from omnimancer.core.models import (
    ChatContext,
    ChatMessage,
    MessageRole,
    ToolDefinition,
)
from omnimancer.providers.gemini import GeminiProvider
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
)


@pytest.fixture
def gemini_provider():
    """Create a GeminiProvider instance for testing."""
    return GeminiProvider(
        api_key="AIzaSyTest123456789",
        model="gemini-1.5-pro",
        max_tokens=4096,
        temperature=0.7,
    )


@pytest.fixture
def sample_chat_context():
    """Create a sample chat context for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Hello",
            timestamp=datetime.now(),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hi there! How can I help you?",
            timestamp=datetime.now(),
            model_used="gemini-1.5-pro",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="gemini-1.5-pro",
        session_id="test-session",
        max_context_length=4000,
    )


@pytest.fixture
def mock_successful_response():
    """Create a mock successful API response."""
    return {
        "candidates": [
            {"content": {"parts": [{"text": "Hello! How can I help you today?"}]}}
        ],
        "usageMetadata": {"totalTokenCount": 25},
    }


@pytest.fixture
def mock_tool_response():
    """Create a mock API response with tool calls."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "I'll help you with that calculation."},
                        {
                            "functionCall": {
                                "name": "calculate",
                                "args": {"expression": "2 + 2"},
                            }
                        },
                    ]
                }
            }
        ],
        "usageMetadata": {"totalTokenCount": 30},
    }


class TestGeminiProviderInitialization:
    """Test GeminiProvider initialization and configuration."""

    def test_initialization_with_defaults(self):
        """Test provider initialization with default values."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789")

        assert provider.api_key == "AIzaSyTest123456789"
        assert provider.model == "gemini-1.5-pro"
        assert provider.max_tokens == 8192
        assert provider.temperature == 0.7

    def test_initialization_with_custom_values(self):
        """Test provider initialization with custom values."""
        provider = GeminiProvider(
            api_key="AIzaSyCustomKey123",
            model="gemini-1.5-flash",
            max_tokens=2048,
            temperature=0.5,
        )

        assert provider.api_key == "AIzaSyCustomKey123"
        assert provider.model == "gemini-1.5-flash"
        assert provider.max_tokens == 2048
        assert provider.temperature == 0.5

    def test_base_url_configuration(self):
        """Test that the base URL is correctly configured."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789")
        assert provider.BASE_URL == "https://generativelanguage.googleapis.com/v1beta"


class TestGeminiProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_missing_api_key(self):
        """Test validation with missing API key."""
        provider = GeminiProvider(api_key="")

        with pytest.raises(AuthenticationError, match="Gemini API key is required"):
            await provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_validate_credentials_invalid_format(self):
        """Test validation with invalid API key format."""
        provider = GeminiProvider(api_key="invalid-key-format")

        with pytest.raises(AuthenticationError, match="Invalid Gemini API key format"):
            await provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, gemini_provider):
        """Test successful credential validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await gemini_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_invalid_key(self, gemini_provider):
        """Test validation with invalid API key."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "API_KEY_INVALID"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(AuthenticationError, match="Invalid Gemini API key"):
                await gemini_provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_validate_credentials_forbidden(self, gemini_provider):
        """Test validation with forbidden access."""
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                AuthenticationError, match="Gemini API access forbidden"
            ):
                await gemini_provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_validate_credentials_model_not_found(self, gemini_provider):
        """Test validation with model not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ModelNotFoundError,
                match="Gemini model 'gemini-1.5-pro' not found",
            ):
                await gemini_provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_validate_credentials_network_timeout(self, gemini_provider):
        """Test validation with network timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(
                NetworkError, match="Gemini API validation request timed out"
            ):
                await gemini_provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_validate_credentials_network_error(self, gemini_provider):
        """Test validation with network error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(
                NetworkError,
                match="Network error during Gemini API validation",
            ):
                await gemini_provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_validate_credentials_malformed_json_response(self, gemini_provider):
        """Test validation with malformed JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ProviderError,
                match="Invalid response from Gemini API during validation",
            ):
                await gemini_provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_validate_credentials_unexpected_error(self, gemini_provider):
        """Test validation with unexpected error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Unexpected error")
            )

            with pytest.raises(
                ProviderError,
                match="Unexpected error during Gemini API validation",
            ):
                await gemini_provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_validate_credentials_generic_400_error(self, gemini_provider):
        """Test validation with generic 400 error without API key message."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Bad request - generic error"}
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ProviderError,
                match="Gemini API validation error: Bad request - generic error",
            ):
                await gemini_provider.validate_credentials()


class TestGeminiProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_gemini_1_5_pro(self):
        """Test getting model info for Gemini 1.5 Pro."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")
        model_info = provider.get_model_info()

        assert model_info.name == "gemini-1.5-pro"
        assert model_info.provider == "gemini"
        assert (
            model_info.description == "Gemini 1.5 Pro - Most capable multimodal model"
        )
        assert model_info.max_tokens == 2097152
        assert model_info.cost_per_token == 0.0000035
        assert model_info.available is True
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is True
        assert model_info.latest_version is True

    def test_get_model_info_gemini_1_5_flash(self):
        """Test getting model info for Gemini 1.5 Flash."""
        provider = GeminiProvider(
            api_key="AIzaSyTest123456789", model="gemini-1.5-flash"
        )
        model_info = provider.get_model_info()

        assert model_info.name == "gemini-1.5-flash"
        assert model_info.provider == "gemini"
        assert (
            model_info.description
            == "Gemini 1.5 Flash - Fast and efficient multimodal model"
        )
        assert model_info.max_tokens == 1048576
        assert model_info.cost_per_token == 0.00000035
        assert model_info.available is True
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is True
        assert model_info.latest_version is False

    def test_get_model_info_gemini_1_0_pro(self):
        """Test getting model info for Gemini 1.0 Pro."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.0-pro")
        model_info = provider.get_model_info()

        assert model_info.name == "gemini-1.0-pro"
        assert model_info.provider == "gemini"
        assert (
            model_info.description == "Gemini 1.0 Pro - Reliable text generation model"
        )
        assert model_info.max_tokens == 32768
        assert model_info.cost_per_token == 0.0000005
        assert model_info.available is True
        assert model_info.supports_tools is False
        assert model_info.supports_multimodal is False
        assert model_info.latest_version is False

    def test_get_model_info_unknown_model(self):
        """Test getting model info for unknown model."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="unknown-model")
        model_info = provider.get_model_info()

        assert model_info.name == "unknown-model"
        assert model_info.provider == "gemini"
        assert model_info.description == "Gemini model unknown-model"
        assert model_info.max_tokens == 32768
        assert model_info.cost_per_token == 0.0000035

    def test_get_available_models(self, gemini_provider):
        """Test getting list of available models."""
        models = gemini_provider.get_available_models()

        assert len(models) == 3

        # Check Gemini 1.5 Pro
        pro_model = next(m for m in models if m.name == "gemini-1.5-pro")
        assert pro_model.provider == "gemini"
        assert pro_model.supports_tools is True
        assert pro_model.supports_multimodal is True
        assert pro_model.latest_version is True

        # Check Gemini 1.5 Flash
        flash_model = next(m for m in models if m.name == "gemini-1.5-flash")
        assert flash_model.provider == "gemini"
        assert flash_model.supports_tools is True
        assert flash_model.supports_multimodal is True
        assert flash_model.latest_version is False

        # Check Gemini 1.0 Pro
        old_model = next(m for m in models if m.name == "gemini-1.0-pro")
        assert old_model.provider == "gemini"
        assert old_model.supports_tools is False
        assert old_model.supports_multimodal is False
        assert old_model.latest_version is False

    def test_get_available_models_consistency(self, gemini_provider):
        """Test that available models list is consistent across calls."""
        models1 = gemini_provider.get_available_models()
        models2 = gemini_provider.get_available_models()

        assert len(models1) == len(models2)

        # Check that model names are the same
        names1 = {m.name for m in models1}
        names2 = {m.name for m in models2}
        assert names1 == names2

    def test_model_info_matches_available_models(self):
        """Test that get_model_info() returns consistent info with get_available_models()."""
        provider = GeminiProvider(
            api_key="AIzaSyTest123456789", model="gemini-1.5-flash"
        )

        # Get model info for current model
        current_model_info = provider.get_model_info()

        # Get available models and find the matching one
        available_models = provider.get_available_models()
        matching_model = next(
            m for m in available_models if m.name == "gemini-1.5-flash"
        )

        # Compare key attributes
        assert current_model_info.name == matching_model.name
        assert current_model_info.provider == matching_model.provider
        assert current_model_info.description == matching_model.description
        assert current_model_info.max_tokens == matching_model.max_tokens
        assert current_model_info.cost_per_token == matching_model.cost_per_token
        assert current_model_info.supports_tools == matching_model.supports_tools
        assert (
            current_model_info.supports_multimodal == matching_model.supports_multimodal
        )


class TestGeminiProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools_gemini_1_5_pro(self):
        """Test tool support for Gemini 1.5 Pro."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")
        assert provider.supports_tools() is True

    def test_supports_tools_gemini_1_5_flash(self):
        """Test tool support for Gemini 1.5 Flash."""
        provider = GeminiProvider(
            api_key="AIzaSyTest123456789", model="gemini-1.5-flash"
        )
        assert provider.supports_tools() is True

    def test_supports_tools_gemini_1_0_pro(self):
        """Test tool support for Gemini 1.0 Pro."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.0-pro")
        assert provider.supports_tools() is False

    def test_supports_multimodal_gemini_1_5_pro(self):
        """Test multimodal support for Gemini 1.5 Pro."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")
        assert provider.supports_multimodal() is True

    def test_supports_multimodal_gemini_1_5_flash(self):
        """Test multimodal support for Gemini 1.5 Flash."""
        provider = GeminiProvider(
            api_key="AIzaSyTest123456789", model="gemini-1.5-flash"
        )
        assert provider.supports_multimodal() is True

    def test_supports_multimodal_gemini_1_0_pro(self):
        """Test multimodal support for Gemini 1.0 Pro."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.0-pro")
        assert provider.supports_multimodal() is False


class TestGeminiProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, gemini_provider, sample_chat_context, mock_successful_response
    ):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await gemini_provider.send_message("Hello", sample_chat_context)

            assert response.content == "Hello! How can I help you today?"
            assert response.model_used == "gemini-1.5-pro"
            assert response.tokens_used == 25
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_empty_candidates(
        self, gemini_provider, sample_chat_context
    ):
        """Test handling of empty candidates in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ProviderError, match="Empty candidates in Gemini API response"
            ):
                await gemini_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_empty_parts(self, gemini_provider, sample_chat_context):
        """Test handling of empty parts in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": [{"content": {"parts": []}}]}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ProviderError, match="Empty parts in Gemini API response"
            ):
                await gemini_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, gemini_provider, sample_chat_context
    ):
        """Test handling of authentication errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "API_KEY_INVALID"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(AuthenticationError, match="Invalid Gemini API key"):
                await gemini_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, gemini_provider, sample_chat_context
    ):
        """Test handling of rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RateLimitError, match="Gemini API rate limit exceeded"):
                await gemini_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_model_not_found(
        self, gemini_provider, sample_chat_context
    ):
        """Test handling of model not found errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ModelNotFoundError,
                match="Gemini model 'gemini-1.5-pro' not found",
            ):
                await gemini_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_timeout(
        self, gemini_provider, sample_chat_context
    ):
        """Test handling of network timeouts."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(NetworkError, match="Request to Gemini API timed out"):
                await gemini_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_error(
        self, gemini_provider, sample_chat_context
    ):
        """Test handling of network errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(NetworkError, match="Network error"):
                await gemini_provider.send_message("Hello", sample_chat_context)


class TestGeminiProviderToolCalling:
    """Test tool calling functionality."""

    @pytest.fixture
    def sample_tools(self):
        """Create sample tool definitions."""
        return [
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
                server_name="math_server",
            ),
            ToolDefinition(
                name="search",
                description="Search for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["query"],
                },
                server_name="search_server",
            ),
        ]

    @pytest.mark.asyncio
    async def test_send_message_with_tools_success(
        self,
        gemini_provider,
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

            response = await gemini_provider.send_message_with_tools(
                "Calculate 2 + 2", sample_chat_context, sample_tools
            )

            assert response.content == "I'll help you with that calculation."
            assert response.model_used == "gemini-1.5-pro"
            assert response.tokens_used == 30
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "calculate"
            assert response.tool_calls[0].arguments == {"expression": "2 + 2"}

    @pytest.mark.asyncio
    async def test_send_message_with_tools_no_tool_support(
        self, sample_chat_context, sample_tools
    ):
        """Test tool calling with model that doesn't support tools."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.0-pro")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I can't use tools, but I can help with math."}
                        ]
                    }
                }
            ],
            "usageMetadata": {"totalTokenCount": 15},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await provider.send_message_with_tools(
                "Calculate 2 + 2", sample_chat_context, sample_tools
            )

            assert response.content == "I can't use tools, but I can help with math."
            assert response.tool_calls is None

    @pytest.mark.asyncio
    async def test_send_message_with_tools_empty_tools(
        self, gemini_provider, sample_chat_context
    ):
        """Test tool calling with empty tools list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "Hello! How can I help you?"}]}}
            ],
            "usageMetadata": {"totalTokenCount": 20},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await gemini_provider.send_message_with_tools(
                "Hello", sample_chat_context, []
            )

            assert response.content == "Hello! How can I help you?"
            assert response.tool_calls is None

    def test_convert_tools_to_gemini_format(self, gemini_provider, sample_tools):
        """Test conversion of tools to Gemini format."""
        gemini_tools = gemini_provider._convert_tools_to_gemini_format(sample_tools)

        assert len(gemini_tools) == 1
        assert "function_declarations" in gemini_tools[0]

        function_declarations = gemini_tools[0]["function_declarations"]
        assert len(function_declarations) == 2

        # Check calculate function
        calc_func = next(f for f in function_declarations if f["name"] == "calculate")
        assert calc_func["description"] == "Perform mathematical calculations"
        assert "expression" in calc_func["parameters"]["properties"]

        # Check search function
        search_func = next(f for f in function_declarations if f["name"] == "search")
        assert search_func["description"] == "Search for information"
        assert "query" in search_func["parameters"]["properties"]

    def test_convert_tools_to_gemini_format_empty(self, gemini_provider):
        """Test conversion of empty tools list."""
        gemini_tools = gemini_provider._convert_tools_to_gemini_format([])
        assert gemini_tools == []


class TestGeminiProviderContentPreparation:
    """Test content preparation for API requests."""

    def test_prepare_contents_with_context(self, gemini_provider, sample_chat_context):
        """Test preparing contents with conversation context."""
        contents = gemini_provider._prepare_contents("New message", sample_chat_context)

        assert len(contents) == 3  # 2 from context + 1 new message

        # Check first message (user)
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Hello"

        # Check second message (assistant -> model)
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["text"] == "Hi there! How can I help you?"

        # Check new message
        assert contents[2]["role"] == "user"
        assert contents[2]["parts"][0]["text"] == "New message"

    def test_prepare_contents_empty_context(self, gemini_provider):
        """Test preparing contents with empty context."""
        empty_context = ChatContext(
            messages=[],
            current_model="gemini-1.5-pro",
            session_id="test",
            max_context_length=4000,
        )

        contents = gemini_provider._prepare_contents("Hello", empty_context)

        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Hello"


class TestGeminiProviderResponseHandling:
    """Test response handling functionality."""

    def test_handle_response_success(self, gemini_provider, mock_successful_response):
        """Test handling successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        chat_response = gemini_provider._handle_response(mock_response)

        assert chat_response.content == "Hello! How can I help you today?"
        assert chat_response.model_used == "gemini-1.5-pro"
        assert chat_response.tokens_used == 25
        assert chat_response.timestamp is not None

    def test_handle_response_with_tools_success(
        self, gemini_provider, mock_tool_response
    ):
        """Test handling response with tool calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tool_response

        chat_response = gemini_provider._handle_response_with_tools(mock_response)

        assert chat_response.content == "I'll help you with that calculation."
        assert chat_response.model_used == "gemini-1.5-pro"
        assert chat_response.tokens_used == 30
        assert chat_response.tool_calls is not None
        assert len(chat_response.tool_calls) == 1
        assert chat_response.tool_calls[0].name == "calculate"
        assert chat_response.tool_calls[0].arguments == {"expression": "2 + 2"}

    def test_handle_response_with_tools_text_only(self, gemini_provider):
        """Test handling response with only text (no tool calls)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Here's your answer: 4"}]}}],
            "usageMetadata": {"totalTokenCount": 15},
        }

        chat_response = gemini_provider._handle_response_with_tools(mock_response)

        assert chat_response.content == "Here's your answer: 4"
        assert chat_response.tool_calls is None

    def test_handle_response_error_status(self, gemini_provider):
        """Test handling error response status."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": {"message": "Internal server error"}
        }

        with pytest.raises(ProviderError, match=r".*Gemini API server error.*"):
            gemini_provider._handle_response(mock_response)


class TestGeminiProviderIntegration:
    """Integration tests for GeminiProvider."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, gemini_provider):
        """Test a full conversation flow with multiple messages."""
        # Mock responses for a conversation
        responses = [
            {
                "candidates": [
                    {"content": {"parts": [{"text": "Hello! How can I help?"}]}}
                ],
                "usageMetadata": {"totalTokenCount": 10},
            },
            {
                "candidates": [
                    {"content": {"parts": [{"text": "Sure, I can help with math."}]}}
                ],
                "usageMetadata": {"totalTokenCount": 15},
            },
        ]

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = responses[call_count]
            call_count += 1
            return mock_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=mock_post
            )

            # First message
            context = ChatContext([], "gemini-1.5-pro", "session1", 4000)
            response1 = await gemini_provider.send_message("Hello", context)

            assert response1.content == "Hello! How can I help?"
            assert response1.tokens_used == 10

            # Add response to context
            context.messages.extend(
                [
                    ChatMessage(MessageRole.USER, "Hello", datetime.now(), ""),
                    ChatMessage(
                        MessageRole.ASSISTANT,
                        response1.content,
                        datetime.now(),
                        "gemini-1.5-pro",
                    ),
                ]
            )

            # Second message
            response2 = await gemini_provider.send_message(
                "Can you help with math?", context
            )

            assert response2.content == "Sure, I can help with math."
            assert response2.tokens_used == 15

    @pytest.mark.asyncio
    async def test_error_recovery(self, gemini_provider, sample_chat_context):
        """Test error recovery and retry behavior."""
        # First call fails, second succeeds
        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("First call timeout")
            else:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "candidates": [
                        {"content": {"parts": [{"text": "Success after retry"}]}}
                    ],
                    "usageMetadata": {"totalTokenCount": 20},
                }
                return mock_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=mock_post
            )

            # First call should fail
            with pytest.raises(NetworkError):
                await gemini_provider.send_message("Hello", sample_chat_context)

            # Second call should succeed
            response = await gemini_provider.send_message("Hello", sample_chat_context)
            assert response.content == "Success after retry"
            assert call_count == 2


class TestGeminiProviderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_model_info_data_consistency(self):
        """Test that model info data is consistent across different access methods."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")

        # Get model info directly
        model_info = provider.get_model_info()

        # Get from available models list
        available_models = provider.get_available_models()
        matching_model = next(m for m in available_models if m.name == "gemini-1.5-pro")

        # Verify all key attributes match
        assert model_info.name == matching_model.name
        assert model_info.provider == matching_model.provider
        assert model_info.description == matching_model.description
        assert model_info.max_tokens == matching_model.max_tokens
        assert model_info.cost_per_token == matching_model.cost_per_token
        assert model_info.supports_tools == matching_model.supports_tools
        assert model_info.supports_multimodal == matching_model.supports_multimodal
        assert model_info.latest_version == matching_model.latest_version

    def test_all_models_have_required_attributes(self):
        """Test that all available models have required attributes."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789")
        models = provider.get_available_models()

        for model in models:
            # Check required attributes exist and have valid values
            assert isinstance(model.name, str) and model.name
            assert isinstance(model.provider, str) and model.provider == "gemini"
            assert isinstance(model.description, str) and model.description
            assert isinstance(model.max_tokens, int) and model.max_tokens > 0
            assert isinstance(model.cost_per_token, float) and model.cost_per_token >= 0
            assert isinstance(model.available, bool)
            assert isinstance(model.supports_tools, bool)
            assert isinstance(model.supports_multimodal, bool)
            assert isinstance(model.latest_version, bool)

    def test_model_capabilities_consistency(self):
        """Test that model capabilities are consistent with model names."""
        test_cases = [
            ("gemini-1.5-pro", True, True, True),  # tools, multimodal, latest
            (
                "gemini-1.5-flash",
                True,
                True,
                False,
            ),  # tools, multimodal, not latest
            (
                "gemini-1.0-pro",
                False,
                False,
                False,
            ),  # no tools, no multimodal, not latest
        ]

        for (
            model_name,
            expected_tools,
            expected_multimodal,
            expected_latest,
        ) in test_cases:
            provider = GeminiProvider(api_key="AIzaSyTest123456789", model=model_name)

            # Test provider methods
            assert provider.supports_tools() == expected_tools
            assert provider.supports_multimodal() == expected_multimodal

            # Test model info
            model_info = provider.get_model_info()
            assert model_info.supports_tools == expected_tools
            assert model_info.supports_multimodal == expected_multimodal
            assert model_info.latest_version == expected_latest

    @pytest.mark.asyncio
    async def test_credential_validation_edge_cases(self):
        """Test credential validation with various edge cases."""
        # Test with None API key
        provider = GeminiProvider(api_key=None, model="gemini-1.5-pro")
        with pytest.raises(AuthenticationError, match="Gemini API key is required"):
            await provider.validate_credentials()

        # Test with empty string API key
        provider = GeminiProvider(api_key="", model="gemini-1.5-pro")
        with pytest.raises(AuthenticationError, match="Gemini API key is required"):
            await provider.validate_credentials()

        # Test with whitespace-only API key
        provider = GeminiProvider(api_key="   ", model="gemini-1.5-pro")
        with pytest.raises(AuthenticationError, match="Invalid Gemini API key format"):
            await provider.validate_credentials()

        # Test with API key that doesn't start with AIza
        provider = GeminiProvider(api_key="sk-1234567890", model="gemini-1.5-pro")
        with pytest.raises(AuthenticationError, match="Invalid Gemini API key format"):
            await provider.validate_credentials()

    @pytest.mark.asyncio
    async def test_send_message_with_various_response_formats(
        self, sample_chat_context
    ):
        """Test handling of various API response formats."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")

        # Test response with multiple text parts (Gemini only uses first text part)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "First part"},
                            {"text": " Second part"},
                        ]
                    }
                }
            ],
            "usageMetadata": {"totalTokenCount": 20},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await provider.send_message("Test", sample_chat_context)
            # Gemini provider only takes the first text part
            assert response.content == "First part"

    def test_provider_string_representations(self):
        """Test string representations of the provider."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")

        # Test __str__ method (inherited from BaseProvider)
        str_repr = str(provider)
        assert "gemini" in str_repr.lower()
        assert "gemini-1.5-pro" in str_repr

        # Test __repr__ method (inherited from BaseProvider)
        repr_str = repr(provider)
        assert "GeminiProvider" in repr_str
        assert "gemini-1.5-pro" in repr_str

    def test_provider_name_method(self):
        """Test the get_provider_name method."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")
        assert provider.get_provider_name() == "gemini"

    def test_max_tokens_method(self):
        """Test the get_max_tokens method."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")
        max_tokens = provider.get_max_tokens()
        assert max_tokens == 2097152  # Should match model info

        # Test with different model
        provider_flash = GeminiProvider(
            api_key="AIzaSyTest123456789", model="gemini-1.5-flash"
        )
        max_tokens_flash = provider_flash.get_max_tokens()
        assert max_tokens_flash == 1048576

    def test_cost_estimation(self):
        """Test cost estimation functionality."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")

        # Test cost estimation
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
        expected_cost = 1500 * 0.0000035  # (1000 + 500) * cost_per_token
        assert (
            abs(cost - expected_cost) < 0.0000001
        )  # Allow for floating point precision

        # Test with different model
        provider_old = GeminiProvider(
            api_key="AIzaSyTest123456789", model="gemini-1.0-pro"
        )
        cost_old = provider_old.estimate_cost(input_tokens=1000, output_tokens=500)
        expected_cost_old = 1500 * 0.0000005
        assert abs(cost_old - expected_cost_old) < 0.0000001


class TestGeminiProviderComprehensiveValidation:
    """Comprehensive validation tests for all provider functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_comprehensive_error_scenarios(self):
        """Test comprehensive error scenarios for credential validation."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")

        # Test various HTTP status codes
        error_scenarios = [
            (403, AuthenticationError, "Forbidden"),
            (404, ModelNotFoundError, "Not found"),
            (500, ProviderError, "Internal server error"),
            (502, ProviderError, "Bad gateway"),
            (503, ProviderError, "Service unavailable"),
        ]

        for status_code, expected_exception, description in error_scenarios:
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_response.json.return_value = {"error": {"message": description}}

            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    return_value=mock_response
                )

                with pytest.raises(expected_exception):
                    await provider.validate_credentials()

    def test_model_info_completeness_for_all_models(self):
        """Test that model info is complete for all supported models."""
        supported_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]

        for model_name in supported_models:
            provider = GeminiProvider(api_key="AIzaSyTest123456789", model=model_name)
            model_info = provider.get_model_info()

            # Verify all required fields are present and valid
            assert model_info.name == model_name
            assert model_info.provider == "gemini"
            assert (
                isinstance(model_info.description, str)
                and len(model_info.description) > 0
            )
            assert isinstance(model_info.max_tokens, int) and model_info.max_tokens > 0
            assert (
                isinstance(model_info.cost_per_token, float)
                and model_info.cost_per_token >= 0
            )
            assert isinstance(model_info.available, bool)
            assert isinstance(model_info.supports_tools, bool)
            assert isinstance(model_info.supports_multimodal, bool)
            assert isinstance(model_info.latest_version, bool)

    def test_available_models_list_integrity(self):
        """Test the integrity of the available models list."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789")
        models = provider.get_available_models()

        # Should have exactly 3 models
        assert len(models) == 3

        # Check that all expected models are present
        model_names = {model.name for model in models}
        expected_names = {
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        }
        assert model_names == expected_names

        # Check that exactly one model is marked as latest
        latest_models = [model for model in models if model.latest_version]
        assert len(latest_models) == 1
        assert latest_models[0].name == "gemini-1.5-pro"

        # Check that all models are marked as available
        assert all(model.available for model in models)

        # Check provider consistency
        assert all(model.provider == "gemini" for model in models)

    @pytest.mark.asyncio
    async def test_tool_calling_comprehensive_scenarios(self, sample_chat_context):
        """Test comprehensive tool calling scenarios."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")

        # Test with complex tool definitions
        complex_tools = [
            ToolDefinition(
                name="complex_calculation",
                description="Perform complex mathematical calculations with multiple parameters",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression",
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Decimal precision",
                            "default": 2,
                        },
                        "format": {
                            "type": "string",
                            "enum": ["decimal", "scientific"],
                            "default": "decimal",
                        },
                    },
                    "required": ["expression"],
                },
                server_name="math_server",
            )
        ]

        # Mock response with complex tool call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I'll calculate that for you."},
                            {
                                "functionCall": {
                                    "name": "complex_calculation",
                                    "args": {
                                        "expression": "sqrt(2) * pi",
                                        "precision": 4,
                                        "format": "decimal",
                                    },
                                }
                            },
                        ]
                    }
                }
            ],
            "usageMetadata": {"totalTokenCount": 35},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await provider.send_message_with_tools(
                "Calculate sqrt(2) * pi with 4 decimal places",
                sample_chat_context,
                complex_tools,
            )

            assert response.content == "I'll calculate that for you."
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1

            tool_call = response.tool_calls[0]
            assert tool_call.name == "complex_calculation"
            assert tool_call.arguments["expression"] == "sqrt(2) * pi"
            assert tool_call.arguments["precision"] == 4
            assert tool_call.arguments["format"] == "decimal"

    @pytest.mark.asyncio
    async def test_error_handling_comprehensive_coverage(self, sample_chat_context):
        """Test comprehensive error handling coverage."""
        provider = GeminiProvider(api_key="AIzaSyTest123456789", model="gemini-1.5-pro")

        # Test various network-related errors
        network_errors = [
            (httpx.TimeoutException("Timeout"), NetworkError, "timed out"),
            (
                httpx.ConnectError("Connection failed"),
                NetworkError,
                "Network error",
            ),
            (httpx.ReadError("Read failed"), NetworkError, "Network error"),
            (httpx.WriteError("Write failed"), NetworkError, "Network error"),
        ]

        for error, expected_exception, expected_message in network_errors:
            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    side_effect=error
                )

                with pytest.raises(expected_exception, match=expected_message):
                    await provider.send_message("Test", sample_chat_context)
