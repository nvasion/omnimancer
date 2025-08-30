"""
Unit tests for Azure OpenAI provider implementation.

This module tests the AzureProvider class functionality including
message sending, Azure-specific configuration, and error handling.
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
)
from omnimancer.providers.azure import AzureProvider
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
)


@pytest.fixture
def azure_provider():
    """Create an AzureProvider instance for testing."""
    return AzureProvider(
        api_key="test-azure-key",
        model="gpt-4",
        azure_endpoint="https://test-resource.openai.azure.com",
        azure_deployment="gpt-4-deployment",
        api_version="2024-02-15-preview",
        max_tokens=4096,
        temperature=0.7,
    )


@pytest.fixture
def azure_provider_minimal():
    """Create an AzureProvider instance with minimal configuration."""
    return AzureProvider(
        api_key="test-azure-key",
        model="gpt-35-turbo",
        azure_endpoint="https://minimal-resource.openai.azure.com",
    )


@pytest.fixture
def sample_chat_context():
    """Create a sample chat context for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="Hello, how can you help me?",
            timestamp=datetime.now(),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="I'm here to help you with various tasks. What would you like to know?",
            timestamp=datetime.now(),
            model_used="gpt-4",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="gpt-4",
        session_id="test-session",
        max_context_length=4000,
    )


@pytest.fixture
def mock_successful_response():
    """Create a mock successful API response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I can help you with a wide range of topics including programming, writing, analysis, and more.",
                }
            }
        ],
        "usage": {
            "total_tokens": 35,
            "prompt_tokens": 15,
            "completion_tokens": 20,
        },
    }


class TestAzureProviderInitialization:
    """Test AzureProvider initialization and configuration."""

    def test_initialization_with_full_config(self, azure_provider):
        """Test provider initialization with full Azure configuration."""
        assert azure_provider.api_key == "test-azure-key"
        assert azure_provider.model == "gpt-4"
        assert azure_provider.azure_endpoint == "https://test-resource.openai.azure.com"
        assert azure_provider.azure_deployment == "gpt-4-deployment"
        assert azure_provider.api_version == "2024-02-15-preview"
        assert azure_provider.max_tokens == 4096
        assert azure_provider.temperature == 0.7
        assert azure_provider.top_p == 1.0

    def test_initialization_with_minimal_config(self, azure_provider_minimal):
        """Test provider initialization with minimal configuration."""
        assert azure_provider_minimal.api_key == "test-azure-key"
        assert azure_provider_minimal.model == "gpt-35-turbo"
        assert (
            azure_provider_minimal.azure_endpoint
            == "https://minimal-resource.openai.azure.com"
        )
        assert (
            azure_provider_minimal.azure_deployment == "gpt-35-turbo"
        )  # Defaults to model name
        assert (
            azure_provider_minimal.api_version == "2024-02-15-preview"
        )  # Default version
        assert azure_provider_minimal.max_tokens == 4096  # Default value
        assert azure_provider_minimal.temperature == 0.7  # Default value

    def test_initialization_missing_endpoint(self):
        """Test that missing Azure endpoint raises ValueError."""
        with pytest.raises(ValueError, match="azure_endpoint is required"):
            AzureProvider(api_key="test-key", model="gpt-4")


class TestAzureProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, azure_provider, sample_chat_context, mock_successful_response
    ):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await azure_provider.send_message(
                "What can you help me with?", sample_chat_context
            )

            assert (
                response.content
                == "I can help you with a wide range of topics including programming, writing, analysis, and more."
            )
            assert response.model_used == "gpt-4"
            assert response.tokens_used == 35
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_url_construction(
        self, azure_provider, sample_chat_context
    ):
        """Test that Azure URL is constructed correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await azure_provider.send_message("Test message", sample_chat_context)

            # Check that the correct Azure URL was called
            call_args = mock_post.call_args
            expected_url = (
                "https://test-resource.openai.azure.com/openai/deployments/"
                "gpt-4-deployment/chat/completions?api-version=2024-02-15-preview"
            )
            assert call_args[0][0] == expected_url

    @pytest.mark.asyncio
    async def test_send_message_headers(self, azure_provider, sample_chat_context):
        """Test that Azure-specific headers are set correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await azure_provider.send_message("Test message", sample_chat_context)

            # Check that Azure-specific headers are set
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]

            assert headers["Content-Type"] == "application/json"
            assert headers["api-key"] == "test-azure-key"

    @pytest.mark.asyncio
    async def test_send_message_missing_endpoint(self):
        """Test message sending with missing Azure endpoint."""
        # Since azure_endpoint is required during initialization, this test
        # should verify that the ValueError is raised during __init__
        with pytest.raises(ValueError, match="azure_endpoint is required"):
            AzureProvider(api_key="test-key", model="gpt-4")

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, azure_provider, sample_chat_context):
        """Test message sending with timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(
                NetworkError, match="Request to Azure OpenAI API timed out"
            ):
                await azure_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_error(
        self, azure_provider, sample_chat_context
    ):
        """Test message sending with network error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(NetworkError, match="Network error"):
                await azure_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, azure_provider, sample_chat_context
    ):
        """Test message sending with authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                AuthenticationError, match="Invalid Azure OpenAI API key"
            ):
                await azure_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, azure_provider, sample_chat_context
    ):
        """Test message sending with rate limit error."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                RateLimitError, match="Azure OpenAI API rate limit exceeded"
            ):
                await azure_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_deployment_not_found(
        self, azure_provider, sample_chat_context
    ):
        """Test message sending with deployment not found error."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ModelNotFoundError, match="Azure deployment .* not found"
            ):
                await azure_provider.send_message("Test message", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_empty_response(
        self, azure_provider, sample_chat_context
    ):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ProviderError, match="Empty response from Azure OpenAI API"
            ):
                await azure_provider.send_message("Test message", sample_chat_context)


class TestAzureProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, azure_provider):
        """Test successful credential validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await azure_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, azure_provider):
        """Test credential validation failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await azure_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_missing_endpoint(self):
        """Test credential validation with missing endpoint."""
        # Since azure_endpoint is required during initialization, this test
        # should verify that the ValueError is raised during __init__
        with pytest.raises(ValueError, match="azure_endpoint is required"):
            AzureProvider(api_key="test-key", model="gpt-4")

    @pytest.mark.asyncio
    async def test_validate_credentials_exception(self, azure_provider):
        """Test credential validation with exception."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Connection error")
            )

            result = await azure_provider.validate_credentials()
            assert result is False


class TestAzureProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_gpt4(self, azure_provider):
        """Test getting model info for GPT-4."""
        model_info = azure_provider.get_model_info()

        assert isinstance(model_info, EnhancedModelInfo)
        assert model_info.name == "gpt-4"
        assert model_info.provider == "azure"
        assert model_info.description == "GPT-4 via Azure OpenAI Service"
        assert model_info.max_tokens == 8192
        assert model_info.cost_per_million_input == 30.0
        assert model_info.cost_per_million_output == 60.0
        assert model_info.swe_score == 67.0
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is False
        assert model_info.is_free is False

    def test_get_model_info_gpt35_turbo(self, azure_provider_minimal):
        """Test getting model info for GPT-3.5 Turbo."""
        model_info = azure_provider_minimal.get_model_info()

        assert model_info.name == "gpt-35-turbo"
        assert model_info.description == "GPT-3.5 Turbo via Azure OpenAI Service"
        assert model_info.max_tokens == 4096
        assert model_info.cost_per_million_input == 0.5
        assert model_info.cost_per_million_output == 1.5
        assert model_info.swe_score == 48.1

    def test_get_model_info_unknown_model(self):
        """Test getting model info for unknown model."""
        provider = AzureProvider(
            api_key="test-key",
            model="unknown-model",
            azure_endpoint="https://test.openai.azure.com",
        )
        model_info = provider.get_model_info()

        assert model_info.name == "unknown-model"
        assert model_info.provider == "azure"
        assert model_info.description == "Azure OpenAI model unknown-model"
        assert model_info.swe_score == 50.0  # Default value

    def test_get_available_models(self, azure_provider):
        """Test getting list of available models."""
        models = azure_provider.get_available_models()

        assert len(models) >= 4  # Should have at least GPT-4, GPT-3.5, etc.

        # Check that all models are EnhancedModelInfo instances
        for model in models:
            assert isinstance(model, EnhancedModelInfo)
            assert model.provider == "azure"

        # Check specific models
        gpt4_models = [m for m in models if "gpt-4" in m.name]
        gpt35_models = [m for m in models if "gpt-35-turbo" in m.name]

        assert len(gpt4_models) >= 1
        assert len(gpt35_models) >= 1

        # Check that GPT-4 models support tools
        for model in gpt4_models:
            assert model.supports_tools is True


class TestAzureProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools_gpt4(self, azure_provider):
        """Test tool support for GPT-4 models."""
        assert azure_provider.supports_tools() is True

    def test_supports_tools_gpt35(self, azure_provider_minimal):
        """Test tool support for GPT-3.5 models."""
        assert azure_provider_minimal.supports_tools() is True

    def test_supports_multimodal_standard_models(self, azure_provider):
        """Test multimodal support for standard models."""
        assert azure_provider.supports_multimodal() is False

    def test_supports_multimodal_vision_model(self):
        """Test multimodal support for vision models."""
        provider = AzureProvider(
            api_key="test-key",
            model="gpt-4-vision-preview",
            azure_endpoint="https://test.openai.azure.com",
        )
        assert provider.supports_multimodal() is True

    def test_supports_streaming(self, azure_provider):
        """Test streaming support."""
        assert azure_provider.supports_streaming() is True


class TestAzureProviderMessagePreparation:
    """Test message preparation for API requests."""

    def test_prepare_messages_with_context(self, azure_provider, sample_chat_context):
        """Test preparing messages with conversation context."""
        messages = azure_provider._prepare_messages("New message", sample_chat_context)

        assert len(messages) == 3  # 2 from context + 1 new

        # Check context messages
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, how can you help me?"
        assert messages[1]["role"] == "assistant"
        assert (
            messages[1]["content"]
            == "I'm here to help you with various tasks. What would you like to know?"
        )

        # Check new message
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "New message"

    def test_prepare_messages_empty_context(self, azure_provider):
        """Test preparing messages with empty context."""
        empty_context = ChatContext(
            messages=[], current_model="gpt-4", session_id="test-session"
        )

        messages = azure_provider._prepare_messages("Hello", empty_context)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"


class TestAzureProviderResponseHandling:
    """Test response handling functionality."""

    def test_handle_response_success(self, azure_provider, mock_successful_response):
        """Test handling successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        response = azure_provider._handle_response(mock_response)

        assert (
            response.content
            == "I can help you with a wide range of topics including programming, writing, analysis, and more."
        )
        assert response.model_used == "gpt-4"
        assert response.tokens_used == 35
        assert response.timestamp is not None

    def test_handle_response_with_error_json(self, azure_provider):
        """Test handling response with JSON error message."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid request format"}
        }

        with pytest.raises(
            ProviderError,
            match="Azure OpenAI API error: Invalid request format",
        ):
            azure_provider._handle_response(mock_response)

    def test_handle_response_with_error_no_json(self, azure_provider):
        """Test handling response with error but no JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ProviderError, match="Azure OpenAI API error: HTTP 500"):
            azure_provider._handle_response(mock_response)


class TestAzureProviderURLConstruction:
    """Test Azure-specific URL construction."""

    def test_build_url_with_all_params(self, azure_provider):
        """Test URL construction with all parameters."""
        url = azure_provider._build_azure_url("chat/completions")
        expected = (
            "https://test-resource.openai.azure.com/openai/deployments/"
            "gpt-4-deployment/chat/completions?api-version=2024-02-15-preview"
        )
        assert url == expected

    def test_build_url_with_minimal_params(self, azure_provider_minimal):
        """Test URL construction with minimal parameters."""
        url = azure_provider_minimal._build_azure_url("chat/completions")
        expected = (
            "https://minimal-resource.openai.azure.com/openai/deployments/"
            "gpt-35-turbo/chat/completions?api-version=2024-02-15-preview"
        )
        assert url == expected

    def test_build_url_missing_endpoint(self):
        """Test URL construction with missing endpoint."""
        # This test is no longer valid since azure_endpoint is required during initialization
        # The provider will raise ValueError during __init__ if endpoint is missing
        with pytest.raises(ValueError, match="azure_endpoint is required"):
            AzureProvider(api_key="test-key", model="gpt-4")
