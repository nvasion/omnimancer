"""
Unit tests for Ollama provider implementation.

This module tests the OllamaProvider class functionality including
message sending, server connectivity, model discovery, and error handling.
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
from omnimancer.providers.ollama import OllamaProvider
from omnimancer.utils.errors import (
    ModelNotFoundError,
    NetworkError,
    ProviderError,
)


@pytest.fixture
def ollama_provider():
    """Create an OllamaProvider instance for testing."""
    return OllamaProvider(
        api_key="",  # Ollama doesn't use API keys
        model="llama2",
        base_url="http://localhost:11434",
        max_tokens=4096,
        temperature=0.7,
    )


@pytest.fixture
def custom_ollama_provider():
    """Create an OllamaProvider instance with custom server URL."""
    return OllamaProvider(
        api_key="",
        model="codellama",
        base_url="http://custom-server:8080",
        max_tokens=2048,
        temperature=0.5,
        timeout=30.0,
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
            model_used="llama2",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="llama2",
        session_id="test-session",
        max_context_length=4000,
    )


@pytest.fixture
def mock_successful_response():
    """Create a mock successful API response."""
    return {
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        },
        "done": True,
    }


@pytest.fixture
def mock_server_models():
    """Create mock server models response."""
    return {
        "models": [
            {
                "name": "llama2:latest",
                "size": 3826793677,
                "digest": "sha256:1234567890abcdef",
                "modified_at": "2024-01-15T10:30:00Z",
            },
            {
                "name": "codellama:7b",
                "size": 3826793677,
                "digest": "sha256:abcdef1234567890",
                "modified_at": "2024-01-15T11:00:00Z",
            },
            {
                "name": "llava:latest",
                "size": 4661224676,
                "digest": "sha256:fedcba0987654321",
                "modified_at": "2024-01-15T12:00:00Z",
            },
        ]
    }


class TestOllamaProviderInitialization:
    """Test OllamaProvider initialization and configuration."""

    def test_initialization_with_defaults(self):
        """Test provider initialization with default values."""
        provider = OllamaProvider(model="llama2")

        assert provider.api_key == "local"
        assert provider.model == "llama2"
        assert provider.base_url == "http://localhost:11434"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.7
        assert provider.timeout == 60.0

    def test_initialization_with_custom_values(self, custom_ollama_provider):
        """Test provider initialization with custom values."""
        assert custom_ollama_provider.api_key == "local"
        assert custom_ollama_provider.model == "codellama"
        assert custom_ollama_provider.base_url == "http://custom-server:8080"
        assert custom_ollama_provider.max_tokens == 2048
        assert custom_ollama_provider.temperature == 0.5
        assert custom_ollama_provider.timeout == 30.0

    def test_initialization_with_empty_api_key(self):
        """Test that empty API key defaults to 'local'."""
        provider = OllamaProvider(api_key="", model="llama2")
        assert provider.api_key == "local"

    def test_initialization_with_api_key(self):
        """Test that provided API key is maintained."""
        provider = OllamaProvider(api_key="test-key", model="llama2")
        assert provider.api_key == "test-key"


class TestOllamaProviderServerConnectivity:
    """Test server connectivity and availability checks."""

    @pytest.mark.asyncio
    async def test_check_server_availability_success(self, ollama_provider):
        """Test successful server availability check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            # Should not raise an exception
            await ollama_provider._check_server_availability()

    @pytest.mark.asyncio
    async def test_check_server_availability_connection_error(self, ollama_provider):
        """Test server availability check with connection error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with pytest.raises(NetworkError, match="Cannot connect to Ollama server"):
                await ollama_provider._check_server_availability()

    @pytest.mark.asyncio
    async def test_check_server_availability_timeout(self, ollama_provider):
        """Test server availability check with timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(NetworkError, match="Ollama server is not responding"):
                await ollama_provider._check_server_availability()

    @pytest.mark.asyncio
    async def test_check_server_availability_bad_status(self, ollama_provider):
        """Test server availability check with bad status code."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(NetworkError, match="Ollama server returned status 500"):
                await ollama_provider._check_server_availability()


class TestOllamaProviderModelDiscovery:
    """Test model discovery functionality."""

    @pytest.mark.asyncio
    async def test_get_server_models_success(self, ollama_provider, mock_server_models):
        """Test successful model discovery."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_server_models

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            models = await ollama_provider._get_server_models()

            assert len(models) == 3
            assert models[0]["name"] == "llama2:latest"
            assert models[1]["name"] == "codellama:7b"
            assert models[2]["name"] == "llava:latest"

    @pytest.mark.asyncio
    async def test_get_server_models_empty_response(self, ollama_provider):
        """Test model discovery with empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            models = await ollama_provider._get_server_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_get_server_models_network_error(self, ollama_provider):
        """Test model discovery with network error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(NetworkError, match="Error fetching models from Ollama"):
                await ollama_provider._get_server_models()

    @pytest.mark.asyncio
    async def test_get_server_models_bad_status(self, ollama_provider):
        """Test model discovery with bad status code."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                NetworkError, match="Failed to fetch models from Ollama server"
            ):
                await ollama_provider._get_server_models()

    @pytest.mark.asyncio
    async def test_get_available_models_async_success(
        self, ollama_provider, mock_server_models
    ):
        """Test async model listing with successful server response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_server_models

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            models = await ollama_provider.get_available_models_async()

            assert len(models) == 3

            # Check first model
            assert models[0].name == "llama2:latest"
            assert models[0].provider == "ollama"
            assert models[0].cost_per_token == 0.0
            assert models[0].available is True
            assert models[0].supports_tools is False
            assert "3.6GB" in models[0].description

            # Check multimodal model
            llava_model = next(m for m in models if "llava" in m.name)
            assert llava_model.supports_multimodal is True

    @pytest.mark.asyncio
    async def test_get_available_models_async_fallback(self, ollama_provider):
        """Test async model listing fallback to current model."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            models = await ollama_provider.get_available_models_async()

            # Should fallback to current model
            assert len(models) == 1
            assert models[0].name == "llama2"
            assert models[0].provider == "ollama"


class TestOllamaProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(
        self, ollama_provider, mock_server_models
    ):
        """Test successful credential validation."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        # Mock model availability check
        mock_models_response = MagicMock()
        mock_models_response.status_code = 200
        mock_models_response.json.return_value = {
            "models": [{"name": "llama2"}, {"name": "codellama"}]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=[mock_tags_response, mock_models_response]
            )

            result = await ollama_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_model_not_available(self, ollama_provider):
        """Test validation when model is not available."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        # Mock model availability check - model not in list
        mock_models_response = MagicMock()
        mock_models_response.status_code = 200
        mock_models_response.json.return_value = {
            "models": [{"name": "codellama"}, {"name": "mistral"}]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=[mock_tags_response, mock_models_response]
            )

            result = await ollama_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_server_unavailable(self, ollama_provider):
        """Test validation when server is unavailable."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            result = await ollama_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_unexpected_error(self, ollama_provider):
        """Test validation with unexpected error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Unexpected error")
            )

            result = await ollama_provider.validate_credentials()
            assert result is False


class TestOllamaProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, ollama_provider, sample_chat_context, mock_successful_response
    ):
        """Test successful message sending."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        # Mock chat response
        mock_chat_response = MagicMock()
        mock_chat_response.status_code = 200
        mock_chat_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_tags_response
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_chat_response
            )

            response = await ollama_provider.send_message("Hello", sample_chat_context)

            assert response.content == "Hello! How can I help you today?"
            assert response.model_used == "llama2"
            assert response.tokens_used > 0  # Should have estimated tokens
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_server_unavailable(
        self, ollama_provider, sample_chat_context
    ):
        """Test message sending when server is unavailable."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with pytest.raises(NetworkError, match="Cannot connect to Ollama server"):
                await ollama_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, ollama_provider, sample_chat_context):
        """Test message sending with timeout."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_tags_response
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(
                NetworkError, match="Request to Ollama server timed out"
            ):
                await ollama_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_model_not_found(
        self, ollama_provider, sample_chat_context
    ):
        """Test message sending with model not found."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        # Mock model not found response
        mock_chat_response = MagicMock()
        mock_chat_response.status_code = 404
        mock_chat_response.json.return_value = {"error": "model 'llama2' not found"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_tags_response
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_chat_response
            )

            with pytest.raises(
                ModelNotFoundError,
                match="Use 'ollama pull llama2' to download it",
            ):
                await ollama_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_empty_response(
        self, ollama_provider, sample_chat_context
    ):
        """Test handling of empty response content."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        # Mock empty response
        mock_chat_response = MagicMock()
        mock_chat_response.status_code = 200
        mock_chat_response.json.return_value = {
            "message": {"role": "assistant", "content": ""}
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_tags_response
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_chat_response
            )

            with pytest.raises(ProviderError, match="Empty response from Ollama"):
                await ollama_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_malformed_response(
        self, ollama_provider, sample_chat_context
    ):
        """Test handling of malformed response."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        # Mock malformed response
        mock_chat_response = MagicMock()
        mock_chat_response.status_code = 200
        mock_chat_response.json.side_effect = ValueError("Invalid JSON")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_tags_response
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_chat_response
            )

            with pytest.raises(
                ProviderError, match="Invalid response format from Ollama"
            ):
                await ollama_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_error(
        self, ollama_provider, sample_chat_context
    ):
        """Test handling of network errors during message sending."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_tags_response
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Network error")
            )

            with pytest.raises(
                NetworkError, match="Network error connecting to Ollama"
            ):
                await ollama_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_unexpected_error(
        self, ollama_provider, sample_chat_context
    ):
        """Test handling of unexpected errors."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_tags_response
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Unexpected error")
            )

            with pytest.raises(ProviderError, match="Unexpected error with Ollama"):
                await ollama_provider.send_message("Hello", sample_chat_context)


class TestOllamaProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_basic_model(self, ollama_provider):
        """Test getting model info for basic model."""
        model_info = ollama_provider.get_model_info()

        assert model_info.name == "llama2"
        assert model_info.provider == "ollama"
        assert model_info.description == "Ollama model llama2 (local inference)"
        assert model_info.max_tokens == 4096
        assert model_info.cost_per_token == 0.0
        assert model_info.available is True
        assert model_info.supports_tools is False
        assert model_info.supports_multimodal is False
        assert model_info.latest_version is False

    def test_get_model_info_multimodal_model(self):
        """Test getting model info for multimodal model."""
        provider = OllamaProvider(model="llava:latest")
        model_info = provider.get_model_info()

        assert model_info.name == "llava:latest"
        assert model_info.provider == "ollama"
        assert model_info.supports_multimodal is True

    def test_get_available_models_fallback(self, ollama_provider):
        """Test getting available models when server is not accessible."""
        models = ollama_provider.get_available_models()

        # Should return at least the current model
        assert len(models) >= 1
        assert models[0].name == "llama2"
        assert models[0].provider == "ollama"
        assert models[0].cost_per_token == 0.0


class TestOllamaProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools(self, ollama_provider):
        """Test tool support for Ollama models."""
        assert ollama_provider.supports_tools() is False

    def test_supports_multimodal_basic_model(self, ollama_provider):
        """Test multimodal support for basic models."""
        assert ollama_provider.supports_multimodal() is False

    def test_supports_multimodal_llava_model(self):
        """Test multimodal support for LLaVA models."""
        provider = OllamaProvider(model="llava:latest")
        assert provider.supports_multimodal() is True

    def test_supports_multimodal_bakllava_model(self):
        """Test multimodal support for BakLLaVA models."""
        provider = OllamaProvider(model="bakllava:7b")
        assert provider.supports_multimodal() is True

    def test_supports_multimodal_moondream_model(self):
        """Test multimodal support for Moondream models."""
        provider = OllamaProvider(model="moondream:latest")
        assert provider.supports_multimodal() is True

    def test_supports_streaming(self, ollama_provider):
        """Test streaming support for Ollama models."""
        assert ollama_provider.supports_streaming() is True

    def test_model_supports_multimodal_with_model_name(self, ollama_provider):
        """Test multimodal check with specific model name."""
        assert ollama_provider._model_supports_multimodal("llava:7b") is True
        assert ollama_provider._model_supports_multimodal("llama2:latest") is False
        assert ollama_provider._model_supports_multimodal("bakllava:13b") is True


class TestOllamaProviderMessagePreparation:
    """Test message preparation for API requests."""

    def test_prepare_messages_with_context(self, ollama_provider, sample_chat_context):
        """Test preparing messages with conversation context."""
        messages = ollama_provider._prepare_messages("New message", sample_chat_context)

        assert len(messages) == 3  # 2 from context + 1 new

        # Check context messages
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there! How can I help you?"

        # Check new message
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "New message"

    def test_prepare_messages_empty_context(self, ollama_provider):
        """Test preparing messages with empty context."""
        empty_context = ChatContext(
            messages=[], current_model="llama2", session_id="test-session"
        )

        messages = ollama_provider._prepare_messages("Hello", empty_context)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_prepare_messages_with_system_messages(self, ollama_provider):
        """Test that system messages are included in Ollama format."""
        messages_list = [
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
        ]
        context = ChatContext(
            messages=messages_list,
            current_model="llama2",
            session_id="test-session",
        )

        messages = ollama_provider._prepare_messages("New message", context)

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "New message"


class TestOllamaProviderResponseHandling:
    """Test response handling functionality."""

    def test_handle_response_success(self, ollama_provider):
        """Test handling successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Test response from Ollama",
            },
            "done": True,
        }

        response = ollama_provider._handle_response(mock_response)

        assert response.content == "Test response from Ollama"
        assert response.model_used == "llama2"
        assert response.tokens_used > 0  # Should have estimated tokens
        assert response.timestamp is not None

    def test_handle_response_model_not_found(self, ollama_provider):
        """Test handling model not found response."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "model 'nonexistent' not found"}

        with pytest.raises(
            ModelNotFoundError, match="Use 'ollama pull llama2' to download it"
        ):
            ollama_provider._handle_response(mock_response)

    def test_handle_response_model_not_found_no_json(self, ollama_provider):
        """Test handling model not found without JSON error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ModelNotFoundError, match="Model 'llama2' not found"):
            ollama_provider._handle_response(mock_response)

    def test_handle_response_server_error(self, ollama_provider):
        """Test handling server error response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}

        with pytest.raises(ProviderError, match=r".*Internal server error.*"):
            ollama_provider._handle_response(mock_response)

    def test_handle_response_server_error_no_json(self, ollama_provider):
        """Test handling server error without JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ProviderError, match=r".*HTTP 500.*"):
            ollama_provider._handle_response(mock_response)

    def test_handle_response_empty_content(self, ollama_provider):
        """Test handling response with empty content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": ""}
        }

        with pytest.raises(ProviderError, match="Empty response from Ollama"):
            ollama_provider._handle_response(mock_response)

    def test_handle_response_malformed_json(self, ollama_provider):
        """Test handling response with malformed JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ProviderError, match="Invalid response format from Ollama"):
            ollama_provider._handle_response(mock_response)


class TestOllamaProviderUtilityMethods:
    """Test utility methods."""

    def test_get_provider_name(self, ollama_provider):
        """Test getting provider name."""
        assert ollama_provider.get_provider_name() == "ollama"

    def test_string_representation(self, ollama_provider):
        """Test string representations."""
        assert str(ollama_provider) == "ollama:llama2"
        assert repr(ollama_provider) == "OllamaProvider(model='llama2')"

    def test_get_max_tokens(self, ollama_provider):
        """Test getting max tokens from model info."""
        max_tokens = ollama_provider.get_max_tokens()
        assert max_tokens == 4096

    def test_estimate_cost(self, ollama_provider):
        """Test cost estimation for local models."""
        cost = ollama_provider.estimate_cost(input_tokens=100, output_tokens=50)
        assert cost == 0.0  # Local models are free


class TestOllamaProviderCustomConfiguration:
    """Test custom configuration options."""

    def test_custom_server_url(self, custom_ollama_provider):
        """Test custom server URL configuration."""
        assert custom_ollama_provider.base_url == "http://custom-server:8080"

    def test_custom_timeout(self, custom_ollama_provider):
        """Test custom timeout configuration."""
        assert custom_ollama_provider.timeout == 30.0

    @pytest.mark.asyncio
    async def test_custom_server_url_in_requests(
        self, custom_ollama_provider, sample_chat_context
    ):
        """Test that custom server URL is used in requests."""
        # Mock server availability check
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200

        # Mock chat response
        mock_chat_response = MagicMock()
        mock_chat_response.status_code = 200
        mock_chat_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Response from custom server",
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_tags_response
            )
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_chat_response
            )

            await custom_ollama_provider.send_message("Hello", sample_chat_context)

            # Verify the custom URL was used
            get_call = mock_client.return_value.__aenter__.return_value.get.call_args
            assert "http://custom-server:8080/api/tags" in str(get_call)

            post_call = mock_client.return_value.__aenter__.return_value.post.call_args
            assert "http://custom-server:8080/api/chat" in str(post_call)
