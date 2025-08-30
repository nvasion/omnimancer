"""
Unit tests for Cohere provider implementation.

This module tests the CohereProvider class functionality including
message sending, credential validation, and model information.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from omnimancer.core.models import (
    ChatContext,
    ChatMessage,
    MessageRole,
)
from omnimancer.providers.cohere import CohereProvider
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)


@pytest.fixture
def cohere_provider():
    """Create a CohereProvider instance for testing."""
    return CohereProvider(
        api_key="co_test123456789",
        model="command-r",
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
            model_used="command-r",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="command-r",
        session_id="test-session",
        max_context_length=4000,
    )


@pytest.fixture
def mock_successful_response():
    """Create a mock successful API response."""
    return {
        "text": "Hello! How can I help you today?",
        "meta": {"billed_units": {"output_tokens": 25}},
    }


class TestCohereProviderInitialization:
    """Test CohereProvider initialization and configuration."""

    def test_initialization_with_defaults(self):
        """Test provider initialization with default values."""
        provider = CohereProvider(api_key="co_test123456789")

        assert provider.api_key == "co_test123456789"
        assert provider.model == "command-r"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.7

    def test_initialization_with_custom_values(self):
        """Test provider initialization with custom values."""
        provider = CohereProvider(
            api_key="co_custom123",
            model="command-r-plus",
            max_tokens=2048,
            temperature=0.5,
        )

        assert provider.api_key == "co_custom123"
        assert provider.model == "command-r-plus"
        assert provider.max_tokens == 2048
        assert provider.temperature == 0.5

    def test_base_url_configuration(self):
        """Test that the base URL is correctly configured."""
        provider = CohereProvider(api_key="co_test123456789")
        assert provider.BASE_URL == "https://api.cohere.ai/v1"


class TestCohereProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, cohere_provider):
        """Test successful credential validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await cohere_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_invalid_key(self, cohere_provider):
        """Test validation with invalid API key."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await cohere_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_network_error(self, cohere_provider):
        """Test validation with network error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            result = await cohere_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_timeout(self, cohere_provider):
        """Test validation with timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            result = await cohere_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_unexpected_error(self, cohere_provider):
        """Test validation with unexpected error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Unexpected error")
            )

            result = await cohere_provider.validate_credentials()
            assert result is False


class TestCohereProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_command_r(self):
        """Test getting model info for Command R."""
        provider = CohereProvider(api_key="co_test123456789", model="command-r")
        model_info = provider.get_model_info()

        assert model_info.name == "command-r"
        assert model_info.provider == "cohere"
        assert (
            model_info.description
            == "Command R - Cohere's flagship conversational model"
        )
        assert model_info.max_tokens == 128000
        assert model_info.cost_per_token == 0.0000005
        assert model_info.available is True
        assert model_info.supports_tools is False
        assert model_info.supports_multimodal is False
        assert model_info.latest_version is False

    def test_get_model_info_command_r_plus(self):
        """Test getting model info for Command R+."""
        provider = CohereProvider(api_key="co_test123456789", model="command-r-plus")
        model_info = provider.get_model_info()

        assert model_info.name == "command-r-plus"
        assert model_info.provider == "cohere"
        assert (
            model_info.description
            == "Command R+ - Enhanced version with improved capabilities"
        )
        assert model_info.max_tokens == 128000
        assert model_info.cost_per_token == 0.000003
        assert model_info.available is True
        assert model_info.supports_tools is False
        assert model_info.supports_multimodal is False
        assert model_info.latest_version is True

    def test_get_model_info_command_light(self):
        """Test getting model info for Command Light."""
        provider = CohereProvider(api_key="co_test123456789", model="command-light")
        model_info = provider.get_model_info()

        assert model_info.name == "command-light"
        assert model_info.provider == "cohere"
        assert (
            model_info.description
            == "Command Light - Fast and efficient model for simple tasks"
        )
        assert model_info.max_tokens == 4096
        assert model_info.cost_per_token == 0.0000003
        assert model_info.available is True
        assert model_info.supports_tools is False
        assert model_info.supports_multimodal is False
        assert model_info.latest_version is False

    def test_get_model_info_command(self):
        """Test getting model info for Command."""
        provider = CohereProvider(api_key="co_test123456789", model="command")
        model_info = provider.get_model_info()

        assert model_info.name == "command"
        assert model_info.provider == "cohere"
        assert (
            model_info.description
            == "Command - Previous generation conversational model"
        )
        assert model_info.max_tokens == 4096
        assert model_info.cost_per_token == 0.000001
        assert model_info.available is True
        assert model_info.supports_tools is False
        assert model_info.supports_multimodal is False
        assert model_info.latest_version is False

    def test_get_model_info_unknown_model(self):
        """Test getting model info for unknown model."""
        provider = CohereProvider(api_key="co_test123456789", model="unknown-model")
        model_info = provider.get_model_info()

        assert model_info.name == "unknown-model"
        assert model_info.provider == "cohere"
        assert model_info.description == "Cohere model unknown-model"
        assert model_info.max_tokens == 4096
        assert model_info.cost_per_token == 0.000001

    def test_get_available_models(self, cohere_provider):
        """Test getting list of available models."""
        models = cohere_provider.get_available_models()

        assert len(models) == 4

        # Check Command R
        command_r = next(m for m in models if m.name == "command-r")
        assert command_r.provider == "cohere"
        assert command_r.supports_tools is False
        assert command_r.supports_multimodal is False
        assert command_r.latest_version is False

        # Check Command R+
        command_r_plus = next(m for m in models if m.name == "command-r-plus")
        assert command_r_plus.provider == "cohere"
        assert command_r_plus.supports_tools is False
        assert command_r_plus.supports_multimodal is False
        assert command_r_plus.latest_version is True

        # Check Command Light
        command_light = next(m for m in models if m.name == "command-light")
        assert command_light.provider == "cohere"
        assert command_light.supports_tools is False
        assert command_light.supports_multimodal is False
        assert command_light.latest_version is False

        # Check Command
        command = next(m for m in models if m.name == "command")
        assert command.provider == "cohere"
        assert command.supports_tools is False
        assert command.supports_multimodal is False
        assert command.latest_version is False

    def test_get_available_models_consistency(self, cohere_provider):
        """Test that available models list is consistent across calls."""
        models1 = cohere_provider.get_available_models()
        models2 = cohere_provider.get_available_models()

        assert len(models1) == len(models2)

        # Check that model names are the same
        names1 = {m.name for m in models1}
        names2 = {m.name for m in models2}
        assert names1 == names2

    def test_model_info_matches_available_models(self):
        """Test that get_model_info() returns consistent info with get_available_models()."""
        provider = CohereProvider(api_key="co_test123456789", model="command-r-plus")

        # Get model info for current model
        current_model_info = provider.get_model_info()

        # Get available models and find the matching one
        available_models = provider.get_available_models()
        matching_model = next(m for m in available_models if m.name == "command-r-plus")

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


class TestCohereProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools(self, cohere_provider):
        """Test tool support for Cohere models."""
        assert cohere_provider.supports_tools() is False

    def test_supports_multimodal(self, cohere_provider):
        """Test multimodal support for Cohere models."""
        assert cohere_provider.supports_multimodal() is False


class TestCohereProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, cohere_provider, sample_chat_context, mock_successful_response
    ):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await cohere_provider.send_message("Hello", sample_chat_context)

            assert response.content == "Hello! How can I help you today?"
            assert response.model_used == "command-r"
            assert response.tokens_used == 25
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_empty_text(self, cohere_provider, sample_chat_context):
        """Test handling of empty text in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": ""}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="Empty response from Cohere API"):
                await cohere_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, cohere_provider, sample_chat_context
    ):
        """Test handling of authentication errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(AuthenticationError, match="Invalid Cohere API key"):
                await cohere_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, cohere_provider, sample_chat_context
    ):
        """Test handling of rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RateLimitError, match="Cohere API rate limit exceeded"):
                await cohere_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_model_not_found(
        self, cohere_provider, sample_chat_context
    ):
        """Test handling of model not found errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "model 'invalid-model' not found"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ModelNotFoundError,
                match="Cohere model 'command-r' not found or not accessible",
            ):
                await cohere_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_bad_request_generic(
        self, cohere_provider, sample_chat_context
    ):
        """Test handling of generic bad request errors."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Invalid request format"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ProviderError, match="Cohere API error: Invalid request format"
            ):
                await cohere_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_timeout(
        self, cohere_provider, sample_chat_context
    ):
        """Test handling of network timeouts."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(NetworkError, match="Request to Cohere API timed out"):
                await cohere_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_error(
        self, cohere_provider, sample_chat_context
    ):
        """Test handling of network errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(NetworkError, match="Network error"):
                await cohere_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_unexpected_error(
        self, cohere_provider, sample_chat_context
    ):
        """Test handling of unexpected errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Unexpected error")
            )

            with pytest.raises(ProviderError, match="Unexpected error"):
                await cohere_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_malformed_json_response(
        self, cohere_provider, sample_chat_context
    ):
        """Test handling of malformed JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ProviderUnavailableError, match="Cohere API server error"
            ):
                await cohere_provider.send_message("Hello", sample_chat_context)


class TestCohereChatHistoryPreparation:
    """Test chat history preparation for API requests."""

    def test_prepare_chat_history_with_context(
        self, cohere_provider, sample_chat_context
    ):
        """Test preparing chat history with conversation context."""
        chat_history = cohere_provider._prepare_chat_history(sample_chat_context)

        assert len(chat_history) == 2

        # Check first message (user)
        assert chat_history[0]["role"] == "USER"
        assert chat_history[0]["message"] == "Hello"

        # Check second message (assistant -> chatbot)
        assert chat_history[1]["role"] == "CHATBOT"
        assert chat_history[1]["message"] == "Hi there! How can I help you?"

    def test_prepare_chat_history_empty_context(self, cohere_provider):
        """Test preparing chat history with empty context."""
        empty_context = ChatContext(
            messages=[], current_model="command-r", session_id="test-session"
        )

        chat_history = cohere_provider._prepare_chat_history(empty_context)
        assert chat_history == []

    def test_prepare_chat_history_with_system_messages(self, cohere_provider):
        """Test that system messages are excluded from chat history."""
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant",
                timestamp=datetime.now(),
                model_used="",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="Hello",
                timestamp=datetime.now(),
                model_used="",
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Hi there!",
                timestamp=datetime.now(),
                model_used="command-r",
            ),
        ]
        context = ChatContext(
            messages=messages,
            current_model="command-r",
            session_id="test-session",
        )

        chat_history = cohere_provider._prepare_chat_history(context)

        # Should only have 2 messages (system message excluded)
        assert len(chat_history) == 2
        assert chat_history[0]["role"] == "USER"
        assert chat_history[0]["message"] == "Hello"
        assert chat_history[1]["role"] == "CHATBOT"
        assert chat_history[1]["message"] == "Hi there!"

    def test_prepare_chat_history_role_conversion(self, cohere_provider):
        """Test that roles are correctly converted to Cohere format."""
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="User message",
                timestamp=datetime.now(),
                model_used="",
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Assistant message",
                timestamp=datetime.now(),
                model_used="command-r",
            ),
        ]
        context = ChatContext(
            messages=messages,
            current_model="command-r",
            session_id="test-session",
        )

        chat_history = cohere_provider._prepare_chat_history(context)

        assert chat_history[0]["role"] == "USER"  # user -> USER
        assert chat_history[1]["role"] == "CHATBOT"  # assistant -> CHATBOT


class TestCohereProviderResponseHandling:
    """Test response handling functionality."""

    def test_handle_response_success_with_tokens(self, cohere_provider):
        """Test handling successful response with token information."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Test response",
            "meta": {"billed_units": {"output_tokens": 50}},
        }

        response = cohere_provider._handle_response(mock_response)

        assert response.content == "Test response"
        assert response.model_used == "command-r"
        assert response.tokens_used == 50
        assert response.timestamp is not None

    def test_handle_response_success_without_tokens(self, cohere_provider):
        """Test handling successful response without token information."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Test response"}

        response = cohere_provider._handle_response(mock_response)

        assert response.content == "Test response"
        assert response.model_used == "command-r"
        assert response.tokens_used == 0
        assert response.timestamp is not None

    def test_handle_response_error_with_message(self, cohere_provider):
        """Test handling error response with message."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"message": "Internal server error"}

        with pytest.raises(ProviderUnavailableError, match="Cohere API server error"):
            cohere_provider._handle_response(mock_response)

    def test_handle_response_error_without_message(self, cohere_provider):
        """Test handling error response without message."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ProviderUnavailableError, match="Cohere API server error"):
            cohere_provider._handle_response(mock_response)


class TestCohereProviderUtilityMethods:
    """Test utility methods."""

    def test_get_provider_name(self, cohere_provider):
        """Test getting provider name."""
        assert cohere_provider.get_provider_name() == "cohere"

    def test_string_representation(self, cohere_provider):
        """Test string representations."""
        assert str(cohere_provider) == "cohere:command-r"
        assert repr(cohere_provider) == "CohereProvider(model='command-r')"

    def test_get_max_tokens(self, cohere_provider):
        """Test getting max tokens from model info."""
        max_tokens = cohere_provider.get_max_tokens()
        assert max_tokens == 128000  # command-r max tokens

    def test_estimate_cost(self, cohere_provider):
        """Test cost estimation."""
        cost = cohere_provider.estimate_cost(input_tokens=100, output_tokens=50)
        expected_cost = 150 * 0.0000005  # (100 + 50) * command-r cost per token
        assert cost == expected_cost
