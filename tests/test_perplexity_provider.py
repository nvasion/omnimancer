"""
Unit tests for Perplexity provider implementation.

This module tests the PerplexityProvider class functionality including
message sending, search capabilities, model information, and error handling.
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
from omnimancer.providers.perplexity import PerplexityProvider
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
)


@pytest.fixture
def perplexity_provider():
    """Create a PerplexityProvider instance for testing."""
    return PerplexityProvider(
        api_key="test-perplexity-key",
        model="llama-3.1-sonar-small-128k-online",
        max_tokens=4096,
        temperature=0.2,
        search_enabled=True,
        search_recency_filter="month",
    )


@pytest.fixture
def offline_perplexity_provider():
    """Create a PerplexityProvider instance for offline model."""
    return PerplexityProvider(
        api_key="test-perplexity-key",
        model="llama-3.1-sonar-small-128k-chat",
        search_enabled=False,
    )


@pytest.fixture
def sample_chat_context():
    """Create a sample chat context for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="What's the weather like?",
            timestamp=datetime.now(),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="I can help you check the weather. What location are you interested in?",
            timestamp=datetime.now(),
            model_used="llama-3.1-sonar-small-128k-online",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="llama-3.1-sonar-small-128k-online",
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
                    "content": "Based on current weather data, it's sunny with 75°F temperature.",
                }
            }
        ],
        "usage": {
            "total_tokens": 45,
            "prompt_tokens": 20,
            "completion_tokens": 25,
        },
        "citations": [
            {
                "title": "Weather.com",
                "url": "https://weather.com/current-conditions",
            }
        ],
        "related_questions": [
            "What's the forecast for tomorrow?",
            "Will it rain this week?",
        ],
    }


@pytest.fixture
def mock_successful_response_with_metadata():
    """Create a mock successful API response with full metadata."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The latest AI developments include significant advances in reasoning capabilities.",
                }
            }
        ],
        "usage": {
            "total_tokens": 60,
            "prompt_tokens": 30,
            "completion_tokens": 30,
        },
        "citations": [
            {
                "title": "AI Research Paper",
                "url": "https://arxiv.org/abs/2024.01234",
            },
            {
                "title": "Tech News Article",
                "url": "https://technews.com/ai-advances",
            },
        ],
        "images": [
            {
                "url": "https://example.com/ai-diagram.png",
                "description": "AI architecture diagram",
            }
        ],
        "related_questions": [
            "What are the implications of these advances?",
            "How will this affect the industry?",
        ],
    }


class TestPerplexityProviderInitialization:
    """Test PerplexityProvider initialization and configuration."""

    def test_initialization_with_defaults(self):
        """Test provider initialization with default values."""
        provider = PerplexityProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model == "llama-3.1-sonar-small-128k-online"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.2
        assert provider.search_enabled is True
        assert provider.search_recency_filter == "month"
        assert provider.return_citations is False
        assert provider.return_images is False
        assert provider.return_related_questions is False

    def test_initialization_with_custom_values(self, perplexity_provider):
        """Test provider initialization with custom values."""
        assert perplexity_provider.api_key == "test-perplexity-key"
        assert perplexity_provider.model == "llama-3.1-sonar-small-128k-online"
        assert perplexity_provider.max_tokens == 4096
        assert perplexity_provider.temperature == 0.2
        assert perplexity_provider.search_enabled is True
        assert perplexity_provider.search_recency_filter == "month"

    def test_initialization_offline_model(self, offline_perplexity_provider):
        """Test initialization with offline model."""
        assert offline_perplexity_provider.model == "llama-3.1-sonar-small-128k-chat"
        assert offline_perplexity_provider.search_enabled is False


class TestPerplexityProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self,
        perplexity_provider,
        sample_chat_context,
        mock_successful_response,
    ):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await perplexity_provider.send_message(
                "What's the weather today?", sample_chat_context
            )

            assert (
                response.content
                == "Based on current weather data, it's sunny with 75°F temperature."
            )
            assert response.model_used == "llama-3.1-sonar-small-128k-online"
            assert response.tokens_used == 45
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_with_citations(
        self,
        perplexity_provider,
        sample_chat_context,
        mock_successful_response_with_metadata,
    ):
        """Test message sending with citations and metadata."""
        perplexity_provider.return_citations = True
        perplexity_provider.return_related_questions = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response_with_metadata

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await perplexity_provider.send_message(
                "Tell me about AI advances", sample_chat_context
            )

            # Check that citations are included in content
            assert "**Sources:**" in response.content
            assert "AI Research Paper" in response.content
            assert "Tech News Article" in response.content

            # Check that related questions are included
            assert "**Related Questions:**" in response.content
            assert "What are the implications" in response.content

    @pytest.mark.asyncio
    async def test_send_message_online_model_parameters(
        self, perplexity_provider, sample_chat_context
    ):
        """Test that online models include search parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await perplexity_provider.send_message("Test message", sample_chat_context)

            # Check that the request payload includes search parameters
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert payload["model"] == "llama-3.1-sonar-small-128k-online"
            assert payload["search_recency_filter"] == "month"
            assert payload["return_citations"] is False

    @pytest.mark.asyncio
    async def test_send_message_offline_model_no_search_params(
        self, offline_perplexity_provider, sample_chat_context
    ):
        """Test that offline models don't include search parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await offline_perplexity_provider.send_message(
                "Test message", sample_chat_context
            )

            # Check that the request payload doesn't include search parameters
            call_args = mock_post.call_args
            payload = call_args[1]["json"]

            assert "search_recency_filter" not in payload

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, perplexity_provider, sample_chat_context):
        """Test message sending with timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(
                NetworkError, match="Request to Perplexity API timed out"
            ):
                await perplexity_provider.send_message(
                    "Test message", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_network_error(
        self, perplexity_provider, sample_chat_context
    ):
        """Test message sending with network error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            with pytest.raises(NetworkError, match="Network error"):
                await perplexity_provider.send_message(
                    "Test message", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, perplexity_provider, sample_chat_context
    ):
        """Test message sending with authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(AuthenticationError, match="Invalid Perplexity API key"):
                await perplexity_provider.send_message(
                    "Test message", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, perplexity_provider, sample_chat_context
    ):
        """Test message sending with rate limit error."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                RateLimitError, match="Perplexity API rate limit exceeded"
            ):
                await perplexity_provider.send_message(
                    "Test message", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_model_not_found(
        self, perplexity_provider, sample_chat_context
    ):
        """Test message sending with model not found error."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                ModelNotFoundError, match="Perplexity model .* not found"
            ):
                await perplexity_provider.send_message(
                    "Test message", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_empty_response(
        self, perplexity_provider, sample_chat_context
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
                ProviderError, match="Empty response from Perplexity API"
            ):
                await perplexity_provider.send_message(
                    "Test message", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_unexpected_error(
        self, perplexity_provider, sample_chat_context
    ):
        """Test handling of unexpected errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Unexpected error")
            )

            with pytest.raises(ProviderError, match="Unexpected error"):
                await perplexity_provider.send_message(
                    "Test message", sample_chat_context
                )


class TestPerplexityProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, perplexity_provider):
        """Test successful credential validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await perplexity_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, perplexity_provider):
        """Test credential validation failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await perplexity_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_exception(self, perplexity_provider):
        """Test credential validation with exception."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Connection error")
            )

            result = await perplexity_provider.validate_credentials()
            assert result is False


class TestPerplexityProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_online_model(self, perplexity_provider):
        """Test getting model info for online model."""
        model_info = perplexity_provider.get_model_info()

        assert isinstance(model_info, EnhancedModelInfo)
        assert model_info.name == "llama-3.1-sonar-small-128k-online"
        assert model_info.provider == "perplexity"
        assert (
            model_info.description == "Llama 3.1 Sonar Small with real-time web search"
        )
        assert model_info.max_tokens == 127072
        assert model_info.cost_per_million_input == 0.2
        assert model_info.cost_per_million_output == 0.2
        assert model_info.swe_score == 45.2
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is False
        assert model_info.is_free is False

    def test_get_model_info_offline_model(self, offline_perplexity_provider):
        """Test getting model info for offline model."""
        model_info = offline_perplexity_provider.get_model_info()

        assert model_info.name == "llama-3.1-sonar-small-128k-chat"
        assert model_info.supports_tools is False

    def test_get_model_info_unknown_model(self):
        """Test getting model info for unknown model."""
        provider = PerplexityProvider(api_key="test-key", model="unknown-model")
        model_info = provider.get_model_info()

        assert model_info.name == "unknown-model"
        assert model_info.provider == "perplexity"
        assert model_info.description == "Perplexity model unknown-model"
        assert model_info.swe_score == 50.0  # Default value

    def test_get_available_models(self, perplexity_provider):
        """Test getting list of available models."""
        models = perplexity_provider.get_available_models()

        assert len(models) == 5

        # Check that all models are EnhancedModelInfo instances
        for model in models:
            assert isinstance(model, EnhancedModelInfo)
            assert model.provider == "perplexity"

        # Check specific models
        online_models = [m for m in models if "online" in m.name]
        chat_models = [m for m in models if "chat" in m.name]

        assert len(online_models) == 3
        assert len(chat_models) == 2

        # Check that online models support tools
        for model in online_models:
            assert model.supports_tools is True

        # Check that chat models don't support tools
        for model in chat_models:
            assert model.supports_tools is False


class TestPerplexityProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools_online_model(self, perplexity_provider):
        """Test tool support for online models."""
        assert perplexity_provider.supports_tools() is True

    def test_supports_tools_offline_model(self, offline_perplexity_provider):
        """Test tool support for offline models."""
        assert offline_perplexity_provider.supports_tools() is False

    def test_supports_multimodal(self, perplexity_provider):
        """Test multimodal support."""
        assert perplexity_provider.supports_multimodal() is False

    def test_supports_streaming(self, perplexity_provider):
        """Test streaming support."""
        assert perplexity_provider.supports_streaming() is True


class TestPerplexityProviderMessagePreparation:
    """Test message preparation for API requests."""

    def test_prepare_messages_with_context(
        self, perplexity_provider, sample_chat_context
    ):
        """Test preparing messages with conversation context."""
        messages = perplexity_provider._prepare_messages(
            "New message", sample_chat_context
        )

        assert len(messages) == 3  # 2 from context + 1 new

        # Check context messages
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What's the weather like?"
        assert messages[1]["role"] == "assistant"
        assert (
            messages[1]["content"]
            == "I can help you check the weather. What location are you interested in?"
        )

        # Check new message
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "New message"

    def test_prepare_messages_empty_context(self, perplexity_provider):
        """Test preparing messages with empty context."""
        empty_context = ChatContext(
            messages=[],
            current_model="llama-3.1-sonar-small-128k-online",
            session_id="test-session",
        )

        messages = perplexity_provider._prepare_messages("Hello", empty_context)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"


class TestPerplexityProviderResponseHandling:
    """Test response handling functionality."""

    def test_handle_response_success(
        self, perplexity_provider, mock_successful_response
    ):
        """Test handling successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_successful_response

        response = perplexity_provider._handle_response(mock_response)

        assert (
            response.content
            == "Based on current weather data, it's sunny with 75°F temperature."
        )
        assert response.model_used == "llama-3.1-sonar-small-128k-online"
        assert response.tokens_used == 45
        assert response.timestamp is not None

    def test_handle_response_with_error_json(self, perplexity_provider):
        """Test handling response with JSON error message."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Invalid request format"}
        }

        with pytest.raises(
            ProviderError, match="Perplexity API error: Invalid request format"
        ):
            perplexity_provider._handle_response(mock_response)

    def test_handle_response_with_error_no_json(self, perplexity_provider):
        """Test handling response with error but no JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ProviderError, match="Perplexity API error: HTTP 500"):
            perplexity_provider._handle_response(mock_response)


class TestPerplexityProviderContentFormatting:
    """Test content formatting with metadata."""

    def test_format_content_with_citations_only(self, perplexity_provider):
        """Test formatting content with citations only."""
        perplexity_provider.return_citations = True
        perplexity_provider.return_related_questions = False

        content = "Test response content"
        citations = [
            {"title": "Source 1", "url": "https://example1.com"},
            {"title": "Source 2", "url": "https://example2.com"},
        ]

        formatted = perplexity_provider._format_content_with_metadata(
            content, citations, [], []
        )

        assert "Test response content" in formatted
        assert "**Sources:**" in formatted
        assert "1. [Source 1](https://example1.com)" in formatted
        assert "2. [Source 2](https://example2.com)" in formatted
        assert "**Related Questions:**" not in formatted

    def test_format_content_with_related_questions_only(self, perplexity_provider):
        """Test formatting content with related questions only."""
        perplexity_provider.return_citations = False
        perplexity_provider.return_related_questions = True

        content = "Test response content"
        related_questions = ["What about this?", "How does that work?"]

        formatted = perplexity_provider._format_content_with_metadata(
            content, [], [], related_questions
        )

        assert "Test response content" in formatted
        assert "**Sources:**" not in formatted
        assert "**Related Questions:**" in formatted
        assert "• What about this?" in formatted
        assert "• How does that work?" in formatted

    def test_format_content_with_all_metadata(self, perplexity_provider):
        """Test formatting content with all metadata types."""
        perplexity_provider.return_citations = True
        perplexity_provider.return_related_questions = True

        content = "Test response content"
        citations = [{"title": "Source", "url": "https://example.com"}]
        images = [{"url": "https://example.com/image.png"}]
        related_questions = ["Related question?"]

        formatted = perplexity_provider._format_content_with_metadata(
            content, citations, images, related_questions
        )

        assert "Test response content" in formatted
        assert "**Sources:**" in formatted
        assert "**Related Questions:**" in formatted

    def test_format_content_no_metadata(self, perplexity_provider):
        """Test formatting content without metadata."""
        perplexity_provider.return_citations = False
        perplexity_provider.return_related_questions = False

        content = "Test response content"

        formatted = perplexity_provider._format_content_with_metadata(
            content, [], [], []
        )

        assert formatted == "Test response content"


class TestPerplexityProviderModelDetection:
    """Test online/offline model detection."""

    def test_is_online_model_true(self, perplexity_provider):
        """Test detection of online models."""
        assert perplexity_provider._is_online_model() is True

    def test_is_online_model_false(self, offline_perplexity_provider):
        """Test detection of offline models."""
        assert offline_perplexity_provider._is_online_model() is False

    def test_is_online_model_various_models(self):
        """Test online model detection for various model names."""
        online_models = [
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
        ]

        offline_models = [
            "llama-3.1-sonar-small-128k-chat",
            "llama-3.1-sonar-large-128k-chat",
            "unknown-model",
        ]

        for model in online_models:
            provider = PerplexityProvider(api_key="test", model=model)
            assert provider._is_online_model() is True

        for model in offline_models:
            provider = PerplexityProvider(api_key="test", model=model)
            assert provider._is_online_model() is False
