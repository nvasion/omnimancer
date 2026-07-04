"""
Unit tests for Google Vertex AI provider implementation.

This module tests the VertexAIProvider class functionality including
construction, authentication, message sending, tool calling, and capabilities.
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
from omnimancer.providers.vertex import VertexAIProvider
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
def vertex_provider():
    """Create a VertexAIProvider instance for testing (gemini-1.5-pro)."""
    return VertexAIProvider(
        api_key="test-vertex-key",
        model="gemini-1.5-pro",
        vertex_project="test-project",
        vertex_location="us-central1",
    )


@pytest.fixture
def vertex_provider_flash():
    """Create a VertexAIProvider instance with gemini-1.5-flash."""
    return VertexAIProvider(
        api_key="test-vertex-key",
        model="gemini-1.5-flash",
        vertex_project="test-project",
        vertex_location="us-east1",
    )


@pytest.fixture
def vertex_provider_vision():
    """Create a VertexAIProvider with gemini-1.0-pro-vision."""
    return VertexAIProvider(
        api_key="test-vertex-key",
        model="gemini-1.0-pro-vision",
        vertex_project="test-project",
    )


@pytest.fixture
def vertex_provider_no_tools():
    """Create a VertexAIProvider with a vision-only model."""
    return VertexAIProvider(
        api_key="test-vertex-key",
        model="gemini-1.0-pro-vision",
        vertex_project="test-project",
    )


@pytest.fixture
def sample_chat_context():
    """Create a sample chat context for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="What can Vertex AI do?",
            timestamp=datetime.now(),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Vertex AI provides ML infrastructure and tools.",
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
def sample_tools():
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="search_web",
            description="Search the internet for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        ),
        ToolDefinition(
            name="run_code",
            description="Execute code in a sandbox",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to run"},
                    "language": {
                        "type": "string",
                        "description": "Programming language",
                    },
                },
                "required": ["code"],
            },
        ),
    ]


@pytest.fixture
def mock_vertex_response():
    """Create a mock successful Vertex AI API response."""
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Vertex AI supports many ML capabilities."}],
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "totalTokenCount": 30,
        },
    }


@pytest.fixture
def mock_vertex_tool_response():
    """Create a mock Vertex AI API response with function calls."""
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "I will search for information."},
                        {
                            "functionCall": {
                                "name": "search_web",
                                "args": {"query": "Vertex AI capabilities"},
                            }
                        },
                    ],
                }
            }
        ],
        "usageMetadata": {"totalTokenCount": 25},
    }


# ---------------------------------------------------------------------------
# TestVertexAIProviderInitialization
# ---------------------------------------------------------------------------


class TestVertexAIProviderInitialization:
    """Test VertexAIProvider initialization and configuration."""

    def test_initialization_with_required_params(self, vertex_provider):
        """Test provider initialization with required parameters."""
        assert vertex_provider.api_key == "test-vertex-key"
        assert vertex_provider.model == "gemini-1.5-pro"
        assert vertex_provider.vertex_project == "test-project"
        assert vertex_provider.vertex_location == "us-central1"

    def test_initialization_default_model(self):
        """Test provider initialization uses default model."""
        provider = VertexAIProvider(
            api_key="test-key",
            vertex_project="test-project",
        )
        assert provider.model == "gemini-1.5-pro"

    def test_initialization_default_location(self):
        """Test provider initialization uses default location."""
        provider = VertexAIProvider(
            api_key="test-key",
            vertex_project="test-project",
        )
        assert provider.vertex_location == "us-central1"

    def test_initialization_custom_location(self, vertex_provider_flash):
        """Test provider initialization with custom location."""
        assert vertex_provider_flash.vertex_location == "us-east1"

    def test_initialization_default_generation_params(self, vertex_provider):
        """Test default generation parameter values."""
        assert vertex_provider.max_tokens == 8192
        assert vertex_provider.temperature == 0.7
        assert vertex_provider.top_p == 0.95
        assert vertex_provider.top_k == 40

    def test_initialization_base_url_contains_project(self, vertex_provider):
        """Test that base URL contains the project ID."""
        assert "test-project" in vertex_provider.base_url

    def test_initialization_base_url_contains_location(self, vertex_provider):
        """Test that base URL contains the location."""
        assert "us-central1" in vertex_provider.base_url

    def test_initialization_missing_project_raises_error(self):
        """Test that missing vertex_project raises ValueError."""
        with pytest.raises(ValueError, match="vertex_project is required"):
            VertexAIProvider(api_key="test-key", model="gemini-1.5-pro")

    def test_initialization_sets_auth_token_from_api_key(self, vertex_provider):
        """Test that api_key is stored as auth_token."""
        assert vertex_provider.auth_token == "test-vertex-key"

    def test_initialization_with_safety_settings(self):
        """Test provider initialization with custom safety settings."""
        safety = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
        }
        provider = VertexAIProvider(
            api_key="test-key",
            vertex_project="proj",
            safety_settings=safety,
        )
        assert provider.safety_settings == safety


# ---------------------------------------------------------------------------
# TestVertexAIProviderAuthHeaders
# ---------------------------------------------------------------------------


class TestVertexAIProviderAuthHeaders:
    """Test authentication header generation."""

    def test_get_auth_headers_includes_bearer_token(self, vertex_provider):
        """Test that auth headers include the Bearer token."""
        headers = vertex_provider._get_auth_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-vertex-key"
        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# TestVertexAIProviderRequestPreparation
# ---------------------------------------------------------------------------


class TestVertexAIProviderRequestPreparation:
    """Test request preparation for Vertex AI API."""

    def test_prepare_vertex_request_includes_user_message(
        self, vertex_provider, sample_chat_context
    ):
        """Test that the request includes the current user message."""
        request = vertex_provider._prepare_vertex_request(
            "Tell me more", sample_chat_context
        )

        contents = request["contents"]
        last_message = contents[-1]
        assert last_message["role"] == "user"
        assert last_message["parts"][0]["text"] == "Tell me more"

    def test_prepare_vertex_request_includes_context(
        self, vertex_provider, sample_chat_context
    ):
        """Test that request includes conversation context."""
        request = vertex_provider._prepare_vertex_request(
            "New question", sample_chat_context
        )

        contents = request["contents"]
        # 2 from context + 1 new message
        assert len(contents) == 3

    def test_prepare_vertex_request_includes_generation_config(
        self, vertex_provider, sample_chat_context
    ):
        """Test that request includes generation configuration."""
        request = vertex_provider._prepare_vertex_request("Hello", sample_chat_context)

        assert "generationConfig" in request
        gen_config = request["generationConfig"]
        assert gen_config["maxOutputTokens"] == vertex_provider.max_tokens
        assert gen_config["temperature"] == vertex_provider.temperature

    def test_prepare_vertex_request_with_tools(
        self, vertex_provider, sample_chat_context, sample_tools
    ):
        """Test that tools are included in the request."""
        request = vertex_provider._prepare_vertex_request_with_tools(
            "Search for info", sample_chat_context, sample_tools
        )

        assert "tools" in request
        assert len(request["tools"]) == 2

    def test_convert_tool_to_vertex_format(self, vertex_provider, sample_tools):
        """Test tool conversion to Vertex AI format."""
        tool = sample_tools[0]
        vertex_tool = vertex_provider._convert_tool_to_vertex_format(tool)

        assert "functionDeclarations" in vertex_tool
        func_decls = vertex_tool["functionDeclarations"]
        assert len(func_decls) == 1
        assert func_decls[0]["name"] == "search_web"
        assert func_decls[0]["description"] == "Search the internet for information"


# ---------------------------------------------------------------------------
# TestVertexAIProviderMessageSending
# ---------------------------------------------------------------------------


class TestVertexAIProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, vertex_provider, sample_chat_context, mock_vertex_response
    ):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vertex_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await vertex_provider.send_message(
                "Tell me more", sample_chat_context
            )

            assert response.content == "Vertex AI supports many ML capabilities."
            assert response.model_used == "gemini-1.5-pro"
            assert response.tokens_used == 30
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, vertex_provider, sample_chat_context
    ):
        """Test send_message with 401 is surfaced as ProviderError.

        Note: VertexAIProvider.send_message wraps non-httpx errors in ProviderError.
        Use _handle_response directly to assert AuthenticationError specifically.
        """
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="Invalid Vertex AI credentials"):
                await vertex_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, vertex_provider, sample_chat_context
    ):
        """Test send_message with 429 is surfaced as ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="rate limit exceeded"):
                await vertex_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_model_not_found(
        self, vertex_provider, sample_chat_context
    ):
        """Test send_message with 404 is surfaced as ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="not found"):
                await vertex_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_timeout(self, vertex_provider, sample_chat_context):
        """Test send_message with timeout raises NetworkError."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timed out")
            )

            with pytest.raises(NetworkError, match="timed out"):
                await vertex_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_network_error(
        self, vertex_provider, sample_chat_context
    ):
        """Test send_message with request error raises NetworkError."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection refused")
            )

            with pytest.raises(NetworkError, match="Network error"):
                await vertex_provider.send_message("Hello", sample_chat_context)

    @pytest.mark.asyncio
    async def test_send_message_empty_candidates(
        self, vertex_provider, sample_chat_context
    ):
        """Test send_message with empty candidates raises ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candidates": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(ProviderError, match="Empty candidates"):
                await vertex_provider.send_message("Hello", sample_chat_context)


# ---------------------------------------------------------------------------
# TestVertexAIProviderToolCalling
# ---------------------------------------------------------------------------


class TestVertexAIProviderToolCalling:
    """Test tool calling functionality."""

    @pytest.mark.asyncio
    async def test_send_message_with_tools_success(
        self,
        vertex_provider,
        sample_chat_context,
        sample_tools,
        mock_vertex_tool_response,
    ):
        """Test successful tool calling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vertex_tool_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            response = await vertex_provider.send_message_with_tools(
                "Search for info", sample_chat_context, sample_tools
            )

            assert response.content == "I will search for information."
            assert response.model_used == "gemini-1.5-pro"
            assert response.tokens_used == 25
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].name == "search_web"

    @pytest.mark.asyncio
    async def test_send_message_with_tools_timeout(
        self, vertex_provider, sample_chat_context, sample_tools
    ):
        """Test tool calling with timeout raises NetworkError."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timed out")
            )

            with pytest.raises(NetworkError, match="timed out"):
                await vertex_provider.send_message_with_tools(
                    "Search", sample_chat_context, sample_tools
                )


# ---------------------------------------------------------------------------
# TestVertexAIProviderCredentialValidation
# ---------------------------------------------------------------------------


class TestVertexAIProviderCredentialValidation:
    """Test credential validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, vertex_provider):
        """Test successful credential validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await vertex_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, vertex_provider):
        """Test credential validation failure (non-200 response)."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await vertex_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_exception(self, vertex_provider):
        """Test credential validation when exception is raised."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Connection error")
            )

            result = await vertex_provider.validate_credentials()
            assert result is False


# ---------------------------------------------------------------------------
# TestVertexAIProviderModelInfo
# ---------------------------------------------------------------------------


class TestVertexAIProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_gemini_15_pro(self, vertex_provider):
        """Test getting model info for gemini-1.5-pro."""
        model_info = vertex_provider.get_model_info()

        assert isinstance(model_info, EnhancedModelInfo)
        assert model_info.name == "gemini-1.5-pro"
        assert model_info.provider == "vertex"
        assert "1.5 Pro" in model_info.description
        assert model_info.max_tokens == 2097152
        assert model_info.cost_per_million_input == 1.25
        assert model_info.cost_per_million_output == 5.0
        assert model_info.swe_score == 71.9
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is True
        assert model_info.latest_version is True
        assert model_info.is_free is False

    def test_get_model_info_gemini_15_flash(self, vertex_provider_flash):
        """Test getting model info for gemini-1.5-flash."""
        model_info = vertex_provider_flash.get_model_info()

        assert model_info.name == "gemini-1.5-flash"
        assert model_info.swe_score == 61.5
        assert model_info.max_tokens == 1048576
        assert model_info.supports_tools is True
        assert model_info.supports_multimodal is True

    def test_get_model_info_unknown_model(self):
        """Test getting model info for unknown model returns defaults."""
        provider = VertexAIProvider(
            api_key="test-key",
            model="gemini-custom-model",
            vertex_project="test-project",
        )
        model_info = provider.get_model_info()

        assert model_info.name == "gemini-custom-model"
        assert model_info.provider == "vertex"
        assert "gemini-custom-model" in model_info.description
        assert model_info.swe_score == 55.0

    def test_get_available_models(self, vertex_provider):
        """Test getting list of available models."""
        models = vertex_provider.get_available_models()

        assert len(models) == 4
        for model in models:
            assert isinstance(model, EnhancedModelInfo)
            assert model.provider == "vertex"

        model_names = [m.name for m in models]
        assert "gemini-1.5-pro" in model_names
        assert "gemini-1.5-flash" in model_names
        assert "gemini-1.0-pro" in model_names
        assert "gemini-1.0-pro-vision" in model_names

        pro_model = next(m for m in models if m.name == "gemini-1.5-pro")
        assert pro_model.latest_version is True


# ---------------------------------------------------------------------------
# TestVertexAIProviderCapabilities
# ---------------------------------------------------------------------------


class TestVertexAIProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools_pro_model(self, vertex_provider):
        """Test that gemini-1.5-pro supports tools."""
        assert vertex_provider.supports_tools() is True

    def test_supports_tools_vision_model(self, vertex_provider_vision):
        """Test that gemini-1.0-pro-vision does NOT support tools."""
        assert vertex_provider_vision.supports_tools() is False

    def test_supports_multimodal_pro_15(self, vertex_provider):
        """Test that gemini-1.5-pro supports multimodal."""
        assert vertex_provider.supports_multimodal() is True

    def test_supports_multimodal_flash(self, vertex_provider_flash):
        """Test that gemini-1.5-flash supports multimodal."""
        assert vertex_provider_flash.supports_multimodal() is True

    def test_supports_multimodal_vision(self, vertex_provider_vision):
        """Test that gemini-1.0-pro-vision supports multimodal."""
        assert vertex_provider_vision.supports_multimodal() is True

    def test_supports_multimodal_pro_10(self):
        """Test that gemini-1.0-pro does NOT support multimodal."""
        provider = VertexAIProvider(
            api_key="test-key",
            model="gemini-1.0-pro",
            vertex_project="test-project",
        )
        assert provider.supports_multimodal() is False

    def test_supports_streaming(self, vertex_provider):
        """Test that Vertex AI supports streaming."""
        assert vertex_provider.supports_streaming() is True


# ---------------------------------------------------------------------------
# TestVertexAIProviderResponseHandling
# ---------------------------------------------------------------------------


class TestVertexAIProviderResponseHandling:
    """Test _handle_response for proper error type classification.

    These tests call _handle_response directly so they can assert
    specific error types without send_message's catch-all interfering.
    """

    def test_handle_response_success(self, vertex_provider, mock_vertex_response):
        """Test handling a successful 200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vertex_response

        response = vertex_provider._handle_response(mock_response)

        assert response.content == "Vertex AI supports many ML capabilities."
        assert response.model_used == "gemini-1.5-pro"
        assert response.tokens_used == 30

    def test_handle_response_401_raises_authentication_error(self, vertex_provider):
        """Test that 401 raises AuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with pytest.raises(AuthenticationError, match="Invalid Vertex AI credentials"):
            vertex_provider._handle_response(mock_response)

    def test_handle_response_429_raises_rate_limit_error(self, vertex_provider):
        """Test that 429 raises RateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with pytest.raises(RateLimitError, match="Vertex AI API rate limit exceeded"):
            vertex_provider._handle_response(mock_response)

    def test_handle_response_404_raises_model_not_found_error(self, vertex_provider):
        """Test that 404 raises ModelNotFoundError."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with pytest.raises(ModelNotFoundError, match="not found"):
            vertex_provider._handle_response(mock_response)

    def test_handle_response_500_raises_provider_error(self, vertex_provider):
        """Test that a 500 error raises ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ProviderError, match="Vertex AI API error"):
            vertex_provider._handle_response(mock_response)

    def test_handle_response_empty_parts_raises_provider_error(self, vertex_provider):
        """Test that empty parts in the response raises ProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"role": "model", "parts": []}}],
            "usageMetadata": {"totalTokenCount": 5},
        }

        with pytest.raises(ProviderError, match="Empty parts"):
            vertex_provider._handle_response(mock_response)


# ---------------------------------------------------------------------------
# TestVertexAIProviderSafetySettings
# ---------------------------------------------------------------------------


class TestVertexAIProviderSafetySettings:
    """Test safety settings formatting."""

    def test_format_safety_settings_returns_list(self):
        """Test that safety settings are formatted as a list."""
        provider = VertexAIProvider(
            api_key="test-key",
            vertex_project="proj",
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            },
        )
        safety_list = provider._format_safety_settings()

        assert isinstance(safety_list, list)
        assert len(safety_list) == 4  # All 4 default categories

    def test_format_safety_settings_includes_configured_category(self):
        """Test that configured safety threshold is applied."""
        provider = VertexAIProvider(
            api_key="test-key",
            vertex_project="proj",
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            },
        )
        safety_list = provider._format_safety_settings()

        harassment_entry = next(
            e for e in safety_list if e["category"] == "HARM_CATEGORY_HARASSMENT"
        )
        assert harassment_entry["threshold"] == "BLOCK_ONLY_HIGH"
