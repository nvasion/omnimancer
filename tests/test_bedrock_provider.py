"""
Unit tests for AWS Bedrock provider implementation.

This module tests the BedrockProvider class functionality including
construction, request preparation (including ARN-based model handling),
message sending, tool calling, credential validation, and capability methods.

ARN handling is tested through observable public behaviour (request body content
and URL routing) rather than through the private helper methods themselves, so
the test suite remains resilient to internal refactoring.
"""

import json
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
from omnimancer.providers.bedrock import BedrockProvider
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

_SONNET_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
_SONNET_35_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
_SONNET_ARN = (
    "arn:aws:bedrock:us-east-1::foundation-model/"
    "anthropic.claude-3-sonnet-20240229-v1:0"
)


@pytest.fixture
def bedrock_provider():
    """Create a BedrockProvider instance with a standard model ID."""
    return BedrockProvider(
        api_key="test-bedrock-key",
        model=_SONNET_MODEL,
        aws_region="us-east-1",
    )


@pytest.fixture
def bedrock_provider_sonnet35():
    """Create a BedrockProvider instance with Claude 3.5 Sonnet."""
    return BedrockProvider(
        api_key="test-bedrock-key",
        model=_SONNET_35_MODEL,
        aws_region="us-east-1",
    )


@pytest.fixture
def bedrock_provider_arn():
    """Create a BedrockProvider instance using an ARN model identifier."""
    return BedrockProvider(
        api_key="test-bedrock-key",
        model=_SONNET_ARN,
        aws_region="us-east-1",
    )


@pytest.fixture
def sample_chat_context():
    """Create a sample chat context for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="What is AWS Bedrock?",
            timestamp=datetime.now(),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="AWS Bedrock is a fully managed service for foundation models.",
            timestamp=datetime.now(),
            model_used=_SONNET_MODEL,
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model=_SONNET_MODEL,
        session_id="test-session",
        max_context_length=4000,
    )


@pytest.fixture
def sample_tools():
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="query_database",
            description="Run a query against the database",
            parameters={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query to execute"}
                },
                "required": ["sql"],
            },
        ),
        ToolDefinition(
            name="send_email",
            description="Send an email to a recipient",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
        ),
    ]


@pytest.fixture
def mock_converse_response():
    """Create a mock successful Bedrock Converse API response."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": (
                            "AWS Bedrock enables you to use "
                            "foundation models from leading AI companies."
                        )
                    }
                ],
            }
        },
        "usage": {
            "inputTokens": 20,
            "outputTokens": 15,
            "totalTokens": 35,
        },
    }


@pytest.fixture
def mock_tool_response():
    """Create a mock Bedrock Converse API response with tool use."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will query the database."},
                    {
                        "type": "toolUse",
                        "name": "query_database",
                        "input": {"sql": "SELECT * FROM users LIMIT 10"},
                    },
                ],
            }
        },
        "usage": {"outputTokens": 20},
    }


# ---------------------------------------------------------------------------
# TestBedrockProviderInitialization
# ---------------------------------------------------------------------------


class TestBedrockProviderInitialization:
    """Test BedrockProvider initialization and configuration."""

    def test_initialization_with_valid_params(self, bedrock_provider):
        """Test provider initialization with valid parameters."""
        assert bedrock_provider.api_key == "test-bedrock-key"
        assert bedrock_provider.model == _SONNET_MODEL
        assert bedrock_provider.aws_region == "us-east-1"

    def test_initialization_default_region(self):
        """Test that default region is us-east-1."""
        provider = BedrockProvider(api_key="key", model=_SONNET_MODEL)
        assert provider.aws_region == "us-east-1"

    def test_initialization_base_url_contains_region(self, bedrock_provider):
        """Test that base URL contains the configured region."""
        assert "us-east-1" in bedrock_provider.base_url

    def test_initialization_custom_region(self):
        """Test provider initialization with a custom region."""
        provider = BedrockProvider(
            api_key="key",
            model=_SONNET_MODEL,
            aws_region="eu-west-1",
        )
        assert provider.aws_region == "eu-west-1"
        assert "eu-west-1" in provider.base_url

    def test_initialization_default_generation_params(self, bedrock_provider):
        """Test default generation parameter values."""
        assert bedrock_provider.max_tokens == 4096
        assert bedrock_provider.temperature == 0.7
        assert bedrock_provider.top_p == 1.0
        assert bedrock_provider.top_k == 250

    def test_initialization_missing_api_key_raises_error(self):
        """Test that missing api_key raises ValueError."""
        with pytest.raises(ValueError, match="API key is required"):
            BedrockProvider(api_key="", model=_SONNET_MODEL)

    def test_initialization_missing_model_raises_error(self):
        """Test that missing model raises ValueError."""
        with pytest.raises(ValueError, match="Model ID is required"):
            BedrockProvider(api_key="test-key", model="")

    def test_initialization_with_arn_model(self, bedrock_provider_arn):
        """Test provider initialization with an ARN model identifier."""
        assert bedrock_provider_arn.model == _SONNET_ARN


# ---------------------------------------------------------------------------
# TestBedrockProviderRequestPreparation
# ---------------------------------------------------------------------------


class TestBedrockProviderRequestPreparation:
    """Test request body preparation for Bedrock Converse API.

    ARN-related behaviour is verified through the *output* of request
    preparation (what goes into the JSON body), not by calling private
    helpers such as _is_arn directly.
    """

    def test_prepare_request_includes_user_message(
        self, bedrock_provider, sample_chat_context
    ):
        """Test that the request body includes the current user message."""
        body_str = bedrock_provider._prepare_bedrock_request(
            "New question", sample_chat_context
        )
        body = json.loads(body_str)

        last_message = body["messages"][-1]
        assert last_message["role"] == "user"
        assert last_message["content"][0]["text"] == "New question"

    def test_prepare_request_includes_context_messages(
        self, bedrock_provider, sample_chat_context
    ):
        """Test that context messages are included in the request."""
        body_str = bedrock_provider._prepare_bedrock_request(
            "Follow-up", sample_chat_context
        )
        body = json.loads(body_str)

        # 2 messages from context + 1 new message = 3 total
        assert len(body["messages"]) == 3

    def test_prepare_request_includes_inference_config(
        self, bedrock_provider, sample_chat_context
    ):
        """Test that the request body includes inference configuration."""
        body_str = bedrock_provider._prepare_bedrock_request(
            "Hello", sample_chat_context
        )
        body = json.loads(body_str)

        assert "inferenceConfig" in body
        assert body["inferenceConfig"]["maxTokens"] == bedrock_provider.max_tokens
        assert body["inferenceConfig"]["temperature"] == bedrock_provider.temperature

    def test_prepare_request_standard_model_has_no_model_id_field(
        self, bedrock_provider, sample_chat_context
    ):
        """Standard (non-ARN) model does NOT add a modelId to the body."""
        body_str = bedrock_provider._prepare_bedrock_request(
            "Hello", sample_chat_context
        )
        body = json.loads(body_str)

        assert "modelId" not in body

    def test_prepare_request_arn_model_includes_model_id_in_body(
        self, bedrock_provider_arn, sample_chat_context
    ):
        """When the model is an ARN, modelId must appear in the request body."""
        body_str = bedrock_provider_arn._prepare_bedrock_request(
            "Hello", sample_chat_context
        )
        body = json.loads(body_str)

        assert "modelId" in body
        assert body["modelId"] == _SONNET_ARN

    def test_prepare_request_with_tools_includes_tool_config(
        self, bedrock_provider, sample_chat_context, sample_tools
    ):
        """Test that tool configuration is included in the request."""
        body_str = bedrock_provider._prepare_bedrock_request_with_tools(
            "Query the DB", sample_chat_context, sample_tools
        )
        body = json.loads(body_str)

        assert "toolConfig" in body
        assert "tools" in body["toolConfig"]
        assert len(body["toolConfig"]["tools"]) == 2

    def test_convert_tool_to_bedrock_format(self, bedrock_provider, sample_tools):
        """Test tool conversion to Bedrock format."""
        tool = sample_tools[0]
        bedrock_tool = bedrock_provider._convert_tool_to_bedrock_format(tool)

        assert bedrock_tool["name"] == "query_database"
        assert bedrock_tool["description"] == "Run a query against the database"
        assert "input_schema" in bedrock_tool


# ---------------------------------------------------------------------------
# TestBedrockProviderARNRoutingBehaviour
# ---------------------------------------------------------------------------


class TestBedrockProviderARNRoutingBehaviour:
    """Test that ARN-format models result in correct URL construction.

    These tests verify observable *behaviour* when an ARN model is used
    (the URL sent to the HTTP client) rather than poking at private helpers.
    """

    @pytest.mark.asyncio
    async def test_send_message_with_tools_uses_extracted_id_in_url_for_arn(
        self, bedrock_provider_arn, sample_chat_context, sample_tools
    ):
        """When using an ARN model, the URL should contain the bare model ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "OK"}],
                }
            },
            "usage": {"outputTokens": 5},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await bedrock_provider_arn.send_message_with_tools(
                "Query the DB", sample_chat_context, sample_tools
            )

            call_url = mock_post.call_args[0][0]
            # The URL must NOT include the full ARN; it should end with the
            # bare model ID extracted from the ARN.
            bare_model_id = _SONNET_ARN.split("/")[-1]
            assert bare_model_id in call_url
            assert "arn:aws:bedrock:" not in call_url

    @pytest.mark.asyncio
    async def test_send_message_standard_model_uses_model_id_in_url(
        self, bedrock_provider, sample_chat_context
    ):
        """For standard (non-ARN) models the model ID is used directly in the URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}},
            "usage": {"outputTokens": 5},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await bedrock_provider.send_message("Hello", sample_chat_context)

            call_url = mock_post.call_args[0][0]
            assert _SONNET_MODEL in call_url


# ---------------------------------------------------------------------------
# TestBedrockProviderMessageSending
# ---------------------------------------------------------------------------


class TestBedrockProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, bedrock_provider, sample_chat_context, mock_converse_response
    ):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_converse_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await bedrock_provider.send_message(
                "Tell me more", sample_chat_context
            )

            expected = (
                "AWS Bedrock enables you to use "
                "foundation models from leading AI companies."
            )
            assert response.content == expected
            assert response.model_used == _SONNET_MODEL
            assert response.tokens_used == 15
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, bedrock_provider, sample_chat_context
    ):
        """Test send_message with 401 is surfaced as ProviderError.

        Note: BedrockProvider.send_message wraps non-httpx errors in ProviderError.
        Use _handle_response directly to assert AuthenticationError specifically.
        """
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="Invalid API key for Bedrock"):
                await bedrock_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, bedrock_provider, sample_chat_context
    ):
        """Test send_message with 429 is surfaced as ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="rate limit exceeded"):
                await bedrock_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_model_not_found(
        self, bedrock_provider, sample_chat_context
    ):
        """Test send_message with 404 is surfaced as ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="not found"):
                await bedrock_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, bedrock_provider, sample_chat_context):
        """Test send_message with timeout raises NetworkError."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timed out")
            )

            with pytest.raises(NetworkError, match="timed out"):
                await bedrock_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_error(
        self, bedrock_provider, sample_chat_context
    ):
        """Test send_message with request error raises NetworkError."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection refused")
            )

            with pytest.raises(NetworkError, match="Network error"):
                await bedrock_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_invalid_response_format(
        self, bedrock_provider, sample_chat_context
    ):
        """Test that an invalid response format raises ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="Invalid response format"):
                await bedrock_provider.send_message("Hello", sample_chat_context)


# ---------------------------------------------------------------------------
# TestBedrockProviderToolCalling
# ---------------------------------------------------------------------------


class TestBedrockProviderToolCalling:
    """Test tool calling functionality."""

    @pytest.mark.asyncio
    async def test_send_message_with_tools_success(
        self,
        bedrock_provider,
        sample_chat_context,
        sample_tools,
        mock_tool_response,
    ):
        """Test successful message sending with tool use."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tool_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await bedrock_provider.send_message_with_tools(
                "Query the database", sample_chat_context, sample_tools
            )

            assert response.content == "I will query the database."
            assert response.model_used == _SONNET_MODEL
            assert response.tokens_used == 20
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "query_database"

    @pytest.mark.asyncio
    async def test_send_message_with_tools_authentication_error(
        self, bedrock_provider, sample_chat_context, sample_tools
    ):
        """Test tool calling with 401 is surfaced as ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="Invalid API key for Bedrock"):
                await bedrock_provider.send_message_with_tools(
                    "Query", sample_chat_context, sample_tools
                )


# ---------------------------------------------------------------------------
# TestBedrockProviderCredentialValidation
# ---------------------------------------------------------------------------


class TestBedrockProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, bedrock_provider):
        """Test successful credential validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await bedrock_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, bedrock_provider):
        """Test credential validation failure (non-200 response)."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await bedrock_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_exception(self, bedrock_provider):
        """Test credential validation when exception is raised."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection error")
            )

            result = await bedrock_provider.validate_credentials()
            assert result is False


# ---------------------------------------------------------------------------
# TestBedrockProviderResponseHandling
# ---------------------------------------------------------------------------


class TestBedrockProviderResponseHandling:
    """Test _handle_response for proper error type classification.

    These tests call _handle_response directly so they can assert specific
    error types without send_message's catch-all interfering.
    """

    def test_handle_response_success(self, bedrock_provider, mock_converse_response):
        """Test handling a successful 200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_converse_response

        response = bedrock_provider._handle_response(mock_response)

        expected = (
            "AWS Bedrock enables you to use "
            "foundation models from leading AI companies."
        )
        assert response.content == expected
        assert response.model_used == _SONNET_MODEL
        assert response.tokens_used == 15

    def test_handle_response_401_raises_authentication_error(self, bedrock_provider):
        """Test that 401 raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with pytest.raises(AuthenticationError, match="Invalid API key for Bedrock"):
            bedrock_provider._handle_response(mock_response)

    def test_handle_response_429_raises_rate_limit_error(self, bedrock_provider):
        """Test that 429 raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with pytest.raises(RateLimitError, match="AWS Bedrock API rate limit exceeded"):
            bedrock_provider._handle_response(mock_response)

    def test_handle_response_404_raises_model_not_found_error(self, bedrock_provider):
        """Test that 404 raises ModelNotFoundError."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with pytest.raises(ModelNotFoundError, match="not found"):
            bedrock_provider._handle_response(mock_response)

    def test_handle_response_500_raises_provider_error(self, bedrock_provider):
        """Test that a 500 error raises ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"message": "Internal server error"}
        mock_response.text = "Internal server error"

        with pytest.raises(ProviderError, match="AWS Bedrock API error"):
            bedrock_provider._handle_response(mock_response)

    def test_handle_response_empty_content_raises_provider_error(
        self, bedrock_provider
    ):
        """Test that empty content blocks raise ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {"message": {"role": "assistant", "content": []}},
            "usage": {},
        }

        with pytest.raises(ProviderError, match="Empty content"):
            bedrock_provider._handle_response(mock_response)


# ---------------------------------------------------------------------------
# TestBedrockProviderModelInfo
# ---------------------------------------------------------------------------


class TestBedrockProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_sonnet(self, bedrock_provider):
        """Test getting model info for Claude 3 Sonnet."""
        model_info = bedrock_provider.get_model_info()

        assert isinstance(model_info, EnhancedModelInfo)
        assert model_info.name == _SONNET_MODEL
        assert model_info.provider == "bedrock"
        assert "Sonnet" in model_info.description
        assert model_info.max_tokens == 200000
        assert model_info.cost_per_million_input == 3.0
        assert model_info.cost_per_million_output == 15.0
        assert model_info.swe_score == 73.0
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is True
        assert model_info.is_free is False

    def test_get_model_info_sonnet35_is_latest_version(self, bedrock_provider_sonnet35):
        """Test that Claude 3.5 Sonnet is marked as latest version."""
        model_info = bedrock_provider_sonnet35.get_model_info()

        assert model_info.latest_version is True
        assert model_info.swe_score == 88.7

    def test_get_model_info_unknown_model(self):
        """Test getting model info for an unknown model returns defaults."""
        provider = BedrockProvider(
            api_key="key",
            model="anthropic.claude-custom-v1:0",
        )
        model_info = provider.get_model_info()

        assert model_info.name == "anthropic.claude-custom-v1:0"
        assert model_info.provider == "bedrock"
        assert model_info.swe_score == 70.0

    def test_get_available_models(self, bedrock_provider):
        """Test getting list of available models."""
        models = bedrock_provider.get_available_models()

        assert len(models) == 4
        for model in models:
            assert isinstance(model, EnhancedModelInfo)
            assert model.provider == "bedrock"
            assert model.supports_tools is True

        model_names = [m.name for m in models]
        assert _SONNET_MODEL in model_names
        assert _SONNET_35_MODEL in model_names

        latest = next(m for m in models if m.name == _SONNET_35_MODEL)
        assert latest.latest_version is True


# ---------------------------------------------------------------------------
# TestBedrockProviderCapabilities
# ---------------------------------------------------------------------------


class TestBedrockProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools(self, bedrock_provider):
        """Test that Bedrock supports tool calling."""
        assert bedrock_provider.supports_tools() is True

    def test_supports_multimodal(self, bedrock_provider):
        """Test that Bedrock supports multimodal inputs."""
        assert bedrock_provider.supports_multimodal() is True

    def test_supports_streaming(self, bedrock_provider):
        """Test that Bedrock supports streaming."""
        assert bedrock_provider.supports_streaming() is True
