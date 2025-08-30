"""
Shared base class and utilities for AI provider testing.

This module provides common test patterns, fixtures, and assertion helpers
that can be shared across all provider test implementations.

Usage Example:
    To use this base class in provider tests, inherit from ProviderTestBase
    and implement the abstract methods:

    class TestMyProvider(ProviderTestBase):
        def get_provider_instance(self, **kwargs):
            return MyProvider(api_key="test-key", model="test-model", **kwargs)

        def get_expected_api_endpoint(self):
            return "https://api.myprovider.com/v1/chat"

        def get_expected_headers(self, provider):
            return {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json"
            }

    This will automatically provide standard tests for initialization,
    message sending, error handling, and credential validation.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from omnimancer.core.models import (
    ChatContext,
    ChatMessage,
    MessageRole,
    ModelInfo,
)
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    RateLimitError,
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
            model_used="test-model",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="test-model",
        session_id="test-session",
        max_context_length=4000,
    )


@pytest.fixture
def mock_successful_response():
    """Create a mock successful HTTP response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the AI model.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75,
        },
    }
    return mock_response


@pytest.fixture
def mock_auth_error_response():
    """Create a mock authentication error response."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {
        "error": {"message": "Invalid API key", "type": "authentication_error"}
    }
    return mock_response


@pytest.fixture
def mock_rate_limit_response():
    """Create a mock rate limit error response."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.json.return_value = {
        "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
    }
    return mock_response


@pytest.fixture
def mock_model_not_found_response():
    """Create a mock model not found error response."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.json.return_value = {
        "error": {
            "message": "Model not found",
            "type": "model_not_found_error",
        }
    }
    return mock_response


class ProviderTestBase(ABC):
    """
    Abstract base class for provider tests.

    Provides common test methods that all providers should implement.
    Subclasses should override abstract methods to provide provider-specific
    fixtures and configurations.
    """

    @abstractmethod
    def get_provider_instance(self, **kwargs):
        """Return a configured provider instance for testing."""
        pass

    @abstractmethod
    def get_expected_api_endpoint(self):
        """Return the expected API endpoint URL for this provider."""
        pass

    @abstractmethod
    def get_expected_headers(self, provider):
        """Return expected HTTP headers for requests."""
        pass

    def test_initialization_basic(self):
        """Test provider initializes with basic required parameters."""
        provider = self.get_provider_instance()
        assert provider is not None
        assert hasattr(provider, "api_key")
        assert hasattr(provider, "model")

    async def test_send_message_success(
        self, sample_chat_context, mock_successful_response
    ):
        """Test successful message sending."""
        provider = self.get_provider_instance()

        with patch(
            "httpx.AsyncClient.post", return_value=mock_successful_response
        ) as mock_post:
            result = await provider.send_message("Test message", sample_chat_context)

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            mock_post.assert_called_once()

    async def test_send_message_authentication_error(
        self, sample_chat_context, mock_auth_error_response
    ):
        """Test handling of authentication errors."""
        provider = self.get_provider_instance()

        with patch("httpx.AsyncClient.post", return_value=mock_auth_error_response):
            with pytest.raises(AuthenticationError):
                await provider.send_message("Test message", sample_chat_context)

    async def test_send_message_rate_limit_error(
        self, sample_chat_context, mock_rate_limit_response
    ):
        """Test handling of rate limit errors."""
        provider = self.get_provider_instance()

        with patch("httpx.AsyncClient.post", return_value=mock_rate_limit_response):
            with pytest.raises(RateLimitError):
                await provider.send_message("Test message", sample_chat_context)

    async def test_send_message_model_not_found(
        self, sample_chat_context, mock_model_not_found_response
    ):
        """Test handling of model not found errors."""
        provider = self.get_provider_instance()

        with patch(
            "httpx.AsyncClient.post",
            return_value=mock_model_not_found_response,
        ):
            with pytest.raises(ModelNotFoundError):
                await provider.send_message("Test message", sample_chat_context)

    async def test_send_message_network_error(self, sample_chat_context):
        """Test handling of network errors."""
        provider = self.get_provider_instance()

        with patch(
            "httpx.AsyncClient.post",
            side_effect=httpx.ConnectError("Connection failed"),
        ):
            with pytest.raises(NetworkError):
                await provider.send_message("Test message", sample_chat_context)

    async def test_send_message_timeout(self, sample_chat_context):
        """Test handling of request timeouts."""
        provider = self.get_provider_instance()

        with patch(
            "httpx.AsyncClient.post",
            side_effect=httpx.TimeoutException("Request timeout"),
        ):
            with pytest.raises(NetworkError):
                await provider.send_message("Test message", sample_chat_context)

    async def test_validate_credentials_success(self, mock_successful_response):
        """Test successful credential validation."""
        provider = self.get_provider_instance()

        with patch("httpx.AsyncClient.post", return_value=mock_successful_response):
            result = await provider.validate_credentials()
            assert result is True

    async def test_validate_credentials_failure(self, mock_auth_error_response):
        """Test credential validation failure."""
        provider = self.get_provider_instance()

        with patch("httpx.AsyncClient.post", return_value=mock_auth_error_response):
            result = await provider.validate_credentials()
            assert result is False

    def test_get_model_info(self):
        """Test model info retrieval."""
        provider = self.get_provider_instance()
        model_info = provider.get_model_info()

        assert isinstance(model_info, ModelInfo)
        assert model_info.name == provider.model
        assert model_info.max_tokens > 0

    def test_request_headers_format(self):
        """Test that request headers are properly formatted."""
        provider = self.get_provider_instance()
        headers = self.get_expected_headers(provider)

        assert isinstance(headers, dict)
        assert "Authorization" in headers or "api-key" in headers
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"


class ProviderTestUtils:
    """Utility methods for provider testing."""

    @staticmethod
    def assert_valid_api_response(response_data):
        """Assert that an API response has the expected structure."""
        assert "choices" in response_data
        assert len(response_data["choices"]) > 0
        assert "message" in response_data["choices"][0]
        assert "content" in response_data["choices"][0]["message"]

    @staticmethod
    def assert_valid_error_response(response_data):
        """Assert that an error response has the expected structure."""
        assert "error" in response_data
        assert "message" in response_data["error"]
        assert isinstance(response_data["error"]["message"], str)

    @staticmethod
    def create_mock_stream_response(chunks):
        """Create a mock streaming response."""

        class MockStreamResponse:
            def __init__(self, chunks):
                self.chunks = chunks
                self.index = 0

            async def __aiter__(self):
                for chunk in self.chunks:
                    yield chunk

        return MockStreamResponse(chunks)


# Common test data
COMMON_TEST_MESSAGES = [
    "Hello, how are you?",
    "What is the weather like today?",
    "Can you help me with a coding problem?",
    "Explain quantum physics in simple terms.",
    "Write a short story about a robot.",
]

COMMON_ERROR_SCENARIOS = [
    {"status_code": 400, "error_type": "bad_request"},
    {"status_code": 401, "error_type": "authentication_error"},
    {"status_code": 403, "error_type": "forbidden"},
    {"status_code": 429, "error_type": "rate_limit_error"},
    {"status_code": 500, "error_type": "server_error"},
    {"status_code": 503, "error_type": "service_unavailable"},
]
