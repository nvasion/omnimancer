"""
Unit tests for Claude-code provider implementation.

This module tests the ClaudeCodeProvider class functionality including
construction, installation validation, message sending, and capability methods.
"""

import subprocess
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core.models import (
    ChatContext,
    ChatMessage,
    EnhancedModelInfo,
    MessageRole,
)
from omnimancer.providers.claude_code import ClaudeCodeProvider
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_valid_subprocess():
    """Patch subprocess.run to simulate a working Claude installation."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Claude CLI version 1.0.0"
    mock_result.stderr = ""
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        yield mock_run


@pytest.fixture
def claude_code_provider(mock_valid_subprocess):
    """Create a ClaudeCodeProvider instance for testing (sonnet mode)."""
    return ClaudeCodeProvider(model="claude-code-sonnet")


@pytest.fixture
def claude_code_provider_opus(mock_valid_subprocess):
    """Create a ClaudeCodeProvider instance with opus mode."""
    return ClaudeCodeProvider(
        api_key="local",
        model="claude-code-opus",
        max_tokens=4096,
    )


@pytest.fixture
def sample_chat_context():
    """Create a sample chat context for testing."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content="What is Python?",
            timestamp=datetime.now(),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Python is a high-level programming language.",
            timestamp=datetime.now(),
            model_used="claude-code-sonnet",
        ),
    ]
    return ChatContext(
        messages=messages,
        current_model="claude-code-sonnet",
        session_id="test-session",
        max_context_length=4000,
    )


# ---------------------------------------------------------------------------
# TestClaudeCodeProviderInitialization
# ---------------------------------------------------------------------------


class TestClaudeCodeProviderInitialization:
    """Test ClaudeCodeProvider initialization and configuration."""

    def test_initialization_with_defaults(self, mock_valid_subprocess):
        """Test provider initialization with default values."""
        provider = ClaudeCodeProvider()

        assert provider.model == "claude-code-sonnet"
        assert provider.api_key == "local"
        assert provider.max_tokens == 4096
        assert provider.temperature == 0.7
        assert provider.claude_code_mode == "sonnet"
        assert provider.claude_code_path == "claude"

    def test_initialization_with_opus_model(self, mock_valid_subprocess):
        """Test provider initialization with opus model."""
        provider = ClaudeCodeProvider(model="claude-code-opus")

        assert provider.model == "claude-code-opus"
        assert provider.claude_code_mode == "opus"

    def test_initialization_with_custom_path(self, mock_valid_subprocess):
        """Test provider initialization with custom claude_code_path."""
        provider = ClaudeCodeProvider(
            model="claude-code-sonnet",
            claude_code_path="/usr/local/bin/claude",
        )

        assert provider.claude_code_path == "/usr/local/bin/claude"

    def test_initialization_custom_working_directory(self, mock_valid_subprocess):
        """Test provider initialization with custom working directory."""
        provider = ClaudeCodeProvider(
            model="claude-code-sonnet",
            working_directory="/tmp/test",
        )

        assert provider.working_directory == "/tmp/test"

    def test_initialization_with_no_api_key_uses_local(self, mock_valid_subprocess):
        """Test that missing api_key defaults to 'local'."""
        provider = ClaudeCodeProvider(api_key="", model="claude-code-sonnet")

        assert provider.api_key == "local"

    def test_initialization_triggers_installation_validation(self):
        """Test that __init__ validates the Claude installation."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Claude version 1.0"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            ClaudeCodeProvider(model="claude-code-sonnet")
            assert mock_run.called

    def test_initialization_raises_on_missing_executable(self):
        """Test FileNotFoundError during install validation raises ProviderError."""
        with patch("subprocess.run", side_effect=FileNotFoundError("No such file")):
            with pytest.raises(ProviderError, match="Claude executable not found"):
                ClaudeCodeProvider(model="claude-code-sonnet")

    def test_initialization_raises_on_nonzero_returncode(self):
        """Test that a failing subprocess raises ProviderError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(
                ProviderError, match="Claude command not found or not working"
            ):
                ClaudeCodeProvider(model="claude-code-sonnet")

    def test_initialization_raises_on_timeout(self):
        """Test that subprocess timeout raises ProviderError."""
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 10)
        ):
            with pytest.raises(ProviderError, match="Claude command timed out"):
                ClaudeCodeProvider(model="claude-code-sonnet")


# ---------------------------------------------------------------------------
# TestClaudeCodeProviderModeExtraction
# ---------------------------------------------------------------------------


class TestClaudeCodeProviderModeExtraction:
    """Test claude_code_mode extraction from model name."""

    def test_mode_from_opus_model(self, mock_valid_subprocess):
        """Test mode extraction for opus model."""
        provider = ClaudeCodeProvider(model="claude-code-opus")
        assert provider.claude_code_mode == "opus"

    def test_mode_from_sonnet_model(self, mock_valid_subprocess):
        """Test mode extraction for sonnet model."""
        provider = ClaudeCodeProvider(model="claude-code-sonnet")
        assert provider.claude_code_mode == "sonnet"

    def test_mode_defaults_to_sonnet_for_unknown_model(self, mock_valid_subprocess):
        """Test that unknown model names default to sonnet mode."""
        provider = ClaudeCodeProvider(model="claude-code-unknown")
        assert provider.claude_code_mode == "sonnet"


# ---------------------------------------------------------------------------
# TestClaudeCodeProviderValidation
# ---------------------------------------------------------------------------


class TestClaudeCodeProviderValidation:
    """Test validate_credentials functionality."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, claude_code_provider):
        """Test successful credential validation."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Hi there!"

        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(return_value=mock_result),
        ):
            result = await claude_code_provider.validate_credentials()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_empty_stdout(self, claude_code_provider):
        """Test credential validation with empty stdout returns False."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(return_value=mock_result),
        ):
            result = await claude_code_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_nonzero_returncode(self, claude_code_provider):
        """Test credential validation with non-zero return code returns False."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(return_value=mock_result),
        ):
            result = await claude_code_provider.validate_credentials()
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_credentials_exception(self, claude_code_provider):
        """Test credential validation when exception is raised returns False."""
        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(side_effect=Exception("Claude not found")),
        ):
            result = await claude_code_provider.validate_credentials()
            assert result is False


# ---------------------------------------------------------------------------
# TestClaudeCodeProviderMessageSending
# ---------------------------------------------------------------------------


class TestClaudeCodeProviderMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, claude_code_provider, sample_chat_context
    ):
        """Test successful message sending."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Python is a versatile language."

        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(return_value=mock_result),
        ):
            response = await claude_code_provider.send_message(
                "Tell me more", sample_chat_context
            )

            assert response.content == "Python is a versatile language."
            assert response.model_used == "claude-code-sonnet"
            assert response.tokens_used == 0
            assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_send_message_empty_response(
        self, claude_code_provider, sample_chat_context
    ):
        """Test that empty stdout raises ProviderError."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(return_value=mock_result),
        ):
            with pytest.raises(ProviderError, match="Empty response from Claude-code"):
                await claude_code_provider.send_message(
                    "Tell me more", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_authentication_error(
        self, claude_code_provider, sample_chat_context
    ):
        """Test that auth error in stderr is surfaced as ProviderError.

        Note: ClaudeCodeProvider.send_message wraps all exceptions in ProviderError.
        Use _handle_claude_code_response directly to assert AuthenticationError.
        """
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Authentication error: invalid api key"
        mock_result.stdout = ""

        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(return_value=mock_result),
        ):
            with pytest.raises(ProviderError, match="authentication"):
                await claude_code_provider.send_message(
                    "Tell me more", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_rate_limit_error(
        self, claude_code_provider, sample_chat_context
    ):
        """Test that rate limit in stderr is surfaced as ProviderError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "rate limit exceeded"
        mock_result.stdout = ""

        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(return_value=mock_result),
        ):
            with pytest.raises(ProviderError, match="rate limit"):
                await claude_code_provider.send_message(
                    "Tell me more", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_model_not_found_error(
        self, claude_code_provider, sample_chat_context
    ):
        """Test that not found in stderr is surfaced as ProviderError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "model not found"
        mock_result.stdout = ""

        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(return_value=mock_result),
        ):
            with pytest.raises(ProviderError, match="not found"):
                await claude_code_provider.send_message(
                    "Tell me more", sample_chat_context
                )

    @pytest.mark.asyncio
    async def test_send_message_generic_error(
        self, claude_code_provider, sample_chat_context
    ):
        """Test that generic stderr raises ProviderError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "unexpected failure"
        mock_result.stdout = ""

        with patch.object(
            claude_code_provider,
            "_execute_claude_code",
            new=AsyncMock(return_value=mock_result),
        ):
            with pytest.raises(ProviderError, match="Claude-code error"):
                await claude_code_provider.send_message(
                    "Tell me more", sample_chat_context
                )


# ---------------------------------------------------------------------------
# TestClaudeCodeProviderConversationPreparation
# ---------------------------------------------------------------------------


class TestClaudeCodeProviderConversationPreparation:
    """Test conversation preparation logic."""

    def test_prepare_conversation_with_context(
        self, claude_code_provider, sample_chat_context
    ):
        """Test conversation is prepared with context messages."""
        conversation = claude_code_provider._prepare_conversation(
            "New question", sample_chat_context
        )

        assert "Human: What is Python?" in conversation
        assert "Assistant: Python is a high-level programming language." in conversation
        assert "Human: New question" in conversation

    def test_prepare_conversation_empty_context(self, claude_code_provider):
        """Test conversation preparation with empty context."""
        empty_context = ChatContext(
            messages=[],
            current_model="claude-code-sonnet",
            session_id="test-session",
        )
        conversation = claude_code_provider._prepare_conversation(
            "Hello", empty_context
        )

        assert conversation == "Human: Hello"

    def test_prepare_conversation_system_message(self, mock_valid_subprocess):
        """Test that system messages are included in the conversation."""
        provider = ClaudeCodeProvider(model="claude-code-sonnet")
        context = ChatContext(
            messages=[
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful assistant.",
                    timestamp=datetime.now(),
                    model_used="",
                )
            ],
            current_model="claude-code-sonnet",
            session_id="test-session",
        )
        conversation = provider._prepare_conversation("Hi", context)

        assert "System: You are a helpful assistant." in conversation
        assert "Human: Hi" in conversation


# ---------------------------------------------------------------------------
# TestClaudeCodeProviderModelInfo
# ---------------------------------------------------------------------------


class TestClaudeCodeProviderModelInfo:
    """Test model information functionality."""

    def test_get_model_info_sonnet(self, claude_code_provider):
        """Test getting model info for claude-code-sonnet."""
        model_info = claude_code_provider.get_model_info()

        assert isinstance(model_info, EnhancedModelInfo)
        assert model_info.name == "claude-code-sonnet"
        assert model_info.provider == "claude-code"
        assert "Sonnet" in model_info.description
        assert model_info.max_tokens == 200000
        assert model_info.swe_score == 73.0
        assert model_info.supports_tools is False
        assert model_info.supports_multimodal is True
        assert model_info.is_free is True
        assert model_info.cost_per_million_input == 0.0
        assert model_info.cost_per_million_output == 0.0

    def test_get_model_info_opus(self, claude_code_provider_opus):
        """Test getting model info for claude-code-opus."""
        model_info = claude_code_provider_opus.get_model_info()

        assert isinstance(model_info, EnhancedModelInfo)
        assert model_info.name == "claude-code-opus"
        assert model_info.provider == "claude-code"
        assert "Opus" in model_info.description
        assert model_info.max_tokens == 200000
        assert model_info.swe_score == 84.9
        assert model_info.supports_tools is False
        assert model_info.supports_multimodal is True
        assert model_info.is_free is True
        assert model_info.latest_version is True

    def test_get_model_info_unknown_model(self, mock_valid_subprocess):
        """Test getting model info for an unknown model."""
        provider = ClaudeCodeProvider(model="claude-code-custom")
        model_info = provider.get_model_info()

        assert model_info.name == "claude-code-custom"
        assert model_info.provider == "claude-code"
        assert "claude-code-custom" in model_info.description
        assert model_info.max_tokens == 200000
        assert model_info.swe_score == 75.0
        assert model_info.is_free is True

    def test_get_available_models(self, claude_code_provider):
        """Test getting list of available models."""
        models = claude_code_provider.get_available_models()

        assert len(models) == 2
        for model in models:
            assert isinstance(model, EnhancedModelInfo)
            assert model.provider == "claude-code"
            assert model.is_free is True
            assert model.supports_tools is False
            assert model.supports_multimodal is True

        model_names = [m.name for m in models]
        assert "claude-code-opus" in model_names
        assert "claude-code-sonnet" in model_names

        opus_model = next(m for m in models if m.name == "claude-code-opus")
        assert opus_model.latest_version is True


# ---------------------------------------------------------------------------
# TestClaudeCodeProviderResponseHandling
# ---------------------------------------------------------------------------


class TestClaudeCodeProviderResponseHandling:
    """Test _handle_claude_code_response for proper error classification.

    These tests call _handle_claude_code_response directly so they can assert
    specific error types without the send_message catch-all interfering.
    """

    def test_handle_response_success(self, claude_code_provider):
        """Test that a returncode=0 response returns a ChatResponse."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Here is the answer."
        mock_result.stderr = ""

        response = claude_code_provider._handle_claude_code_response(mock_result)

        assert response.content == "Here is the answer."
        assert response.model_used == "claude-code-sonnet"
        assert response.tokens_used == 0

    def test_handle_response_empty_output_raises_provider_error(
        self, claude_code_provider
    ):
        """Test that empty stdout raises ProviderError."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "   "  # only whitespace

        with pytest.raises(ProviderError, match="Empty response from Claude-code"):
            claude_code_provider._handle_claude_code_response(mock_result)

    def test_handle_response_auth_error_raises_authentication_error(
        self, claude_code_provider
    ):
        """Test that 'authentication'/'api key' in stderr raises AuthenticationError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Authentication failure: invalid api key"
        mock_result.stdout = ""

        with pytest.raises(AuthenticationError):
            claude_code_provider._handle_claude_code_response(mock_result)

    def test_handle_response_rate_limit_raises_rate_limit_error(
        self, claude_code_provider
    ):
        """Test that 'rate limit' in stderr raises RateLimitError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "rate limit exceeded"
        mock_result.stdout = ""

        with pytest.raises(RateLimitError):
            claude_code_provider._handle_claude_code_response(mock_result)

    def test_handle_response_not_found_raises_model_not_found_error(
        self, claude_code_provider
    ):
        """Test that 'not found' in stderr raises ModelNotFoundError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "model not found"
        mock_result.stdout = ""

        with pytest.raises(ModelNotFoundError):
            claude_code_provider._handle_claude_code_response(mock_result)

    def test_handle_response_generic_error_raises_provider_error(
        self, claude_code_provider
    ):
        """Test that an unknown error in stderr raises generic ProviderError."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "unknown internal error"
        mock_result.stdout = ""

        with pytest.raises(ProviderError, match="Claude-code error"):
            claude_code_provider._handle_claude_code_response(mock_result)


# ---------------------------------------------------------------------------
# TestClaudeCodeProviderCapabilities
# ---------------------------------------------------------------------------


class TestClaudeCodeProviderCapabilities:
    """Test provider capability methods."""

    def test_supports_tools_is_false(self, claude_code_provider):
        """Test that Claude-code does not support tool calling."""
        assert claude_code_provider.supports_tools() is False

    def test_supports_multimodal_is_true(self, claude_code_provider):
        """Test that Claude-code supports multimodal inputs."""
        assert claude_code_provider.supports_multimodal() is True

    def test_supports_streaming_is_false(self, claude_code_provider):
        """Test that Claude-code does not support streaming."""
        assert claude_code_provider.supports_streaming() is False
