"""
Tests for the core engine module.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from omnimancer.core.chat_manager import ChatManager
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.engine import CoreEngine
from omnimancer.core.models import (
    ChatResponse,
    EnhancedModelInfo,
    ModelInfo,
    ProviderConfig,
)
from omnimancer.core.provider_initializer import ProviderInitializer
from omnimancer.providers.base import BaseProvider
from omnimancer.utils.errors import ConfigurationError


@pytest.fixture
def mock_config_manager():
    """Create mock config manager."""
    mock = Mock(spec=ConfigManager)
    mock_config = Mock()
    mock_config.default_provider = "openai"
    mock_config.providers = {
        "openai": ProviderConfig(api_key="sk-test123", model="gpt-4"),
        "claude": ProviderConfig(api_key="sk-ant-test123", model="claude-3-sonnet"),
    }
    mock_config.storage_path = Path("~/.omnimancer")
    mock.get_config.return_value = mock_config
    mock.get_custom_models.return_value = []  # Return empty list instead of Mock
    return mock


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    mock = Mock(spec=BaseProvider)
    mock.get_provider_name.return_value = "openai"
    mock.model = "gpt-4"
    mock.get_available_models.return_value = [
        ModelInfo(
            name="gpt-4",
            provider="openai",
            description="GPT-4",
            max_tokens=8192,
            cost_per_token=0.03,
            available=True,
        ),
        ModelInfo(
            name="gpt-3.5-turbo",
            provider="openai",
            description="GPT-3.5 Turbo",
            max_tokens=4096,
            cost_per_token=0.002,
            available=True,
        ),
    ]
    mock.get_model_info.return_value = ModelInfo(
        name="gpt-4",
        provider="openai",
        description="GPT-4",
        max_tokens=8192,
        cost_per_token=0.03,
        available=True,
    )
    mock.send_message = AsyncMock(
        return_value=ChatResponse(
            content="Test response", model_used="gpt-4", tokens_used=100
        )
    )
    mock.validate_credentials = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_providers(mock_provider):
    """Create mock providers dictionary."""
    claude_provider = Mock(spec=BaseProvider)
    claude_provider.get_provider_name.return_value = "claude"
    claude_provider.model = "claude-3-sonnet"
    claude_provider.get_available_models.return_value = [
        ModelInfo(
            name="claude-3-sonnet",
            provider="claude",
            description="Claude 3 Sonnet",
            max_tokens=200000,
            cost_per_token=0.015,
            available=True,
        ),
        ModelInfo(
            name="claude-3-haiku",
            provider="claude",
            description="Claude 3 Haiku",
            max_tokens=200000,
            cost_per_token=0.0025,
            available=True,
        ),
    ]
    claude_provider.get_model_info.return_value = ModelInfo(
        name="claude-3-sonnet",
        provider="claude",
        description="Claude 3 Sonnet",
        max_tokens=200000,
        cost_per_token=0.015,
        available=True,
    )
    claude_provider.send_message = AsyncMock(
        return_value=ChatResponse(
            content="Claude response",
            model_used="claude-3-sonnet",
            tokens_used=150,
        )
    )
    claude_provider.validate_credentials = AsyncMock(return_value=True)

    return {"openai": mock_provider, "claude": claude_provider}


@pytest.fixture
def core_engine(mock_config_manager):
    """Create CoreEngine instance for testing."""
    with (
        patch("omnimancer.core.engine.HealthMonitor") as mock_health_monitor_class,
        patch("omnimancer.core.engine.ConversationManager") as mock_conv_manager_class,
    ):

        # Mock the HealthMonitor class to return a mock instance
        mock_health_monitor = Mock()
        mock_health_monitor_class.return_value = mock_health_monitor

        # Mock the ConversationManager class to return a mock instance
        mock_conv_manager = Mock()
        mock_conv_manager_class.return_value = mock_conv_manager

        engine = CoreEngine(mock_config_manager)

        # Set the mocked instances as attributes for easy access in tests
        engine._mock_health_monitor = mock_health_monitor
        engine._mock_conversation_manager = mock_conv_manager

        return engine


class TestCoreEngineInitialization:
    """Test CoreEngine initialization."""

    def test_init(self, mock_config_manager):
        """Test CoreEngine initialization."""
        with (
            patch("omnimancer.core.engine.HealthMonitor"),
            patch("omnimancer.core.engine.ConversationManager"),
        ):

            engine = CoreEngine(mock_config_manager)

            assert engine.config_manager == mock_config_manager
            assert isinstance(engine.chat_manager, ChatManager)
            assert engine.conversation_manager is not None  # Mocked
            assert engine.health_monitor is not None  # Mocked
            assert isinstance(engine.provider_initializer, ProviderInitializer)
            assert engine.providers == {}
            assert engine.current_provider is None
            assert engine._initialized is False

    def test_init_with_custom_storage_path(self, mock_config_manager, tmp_path):
        """Test initialization with custom storage path."""
        mock_config = mock_config_manager.get_config.return_value
        # Use tmp_path instead of a root directory that requires permissions
        mock_config.storage_path = tmp_path / "custom_storage"

        with (
            patch("omnimancer.core.engine.HealthMonitor"),
            patch(
                "omnimancer.core.engine.ConversationManager"
            ) as mock_conv_manager_class,
        ):

            engine = CoreEngine(mock_config_manager)

            # Verify conversation manager was initialized with custom path
            assert engine.conversation_manager is not None
            # Verify ConversationManager was called with the custom path
            mock_conv_manager_class.assert_called_once_with(tmp_path / "custom_storage")


class TestProviderInitialization:
    """Test provider initialization functionality."""

    @pytest.mark.asyncio
    async def test_initialize_providers_success(self, core_engine, mock_providers):
        """Test successful provider initialization."""
        with patch.object(
            core_engine.provider_initializer,
            "initialize_providers",
            return_value=mock_providers,
        ) as mock_init:

            await core_engine.initialize_providers()

            assert core_engine.providers == mock_providers
            assert (
                core_engine.current_provider == mock_providers["openai"]
            )  # default provider
            assert core_engine._initialized is True
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_providers_no_default_uses_first(
        self, core_engine, mock_providers
    ):
        """Test initialization when no default provider is set."""
        # Remove default provider
        mock_config = core_engine.config_manager.get_config.return_value
        mock_config.default_provider = None

        with patch.object(
            core_engine.provider_initializer,
            "initialize_providers",
            return_value=mock_providers,
        ):

            await core_engine.initialize_providers()

            # Should use first available provider
            assert core_engine.current_provider is not None
            assert core_engine.current_provider.get_provider_name() in [
                "openai",
                "claude",
            ]

    @pytest.mark.asyncio
    async def test_initialize_providers_failure(self, core_engine):
        """Test provider initialization failure."""
        with patch.object(
            core_engine.provider_initializer,
            "initialize_providers",
            side_effect=Exception("Init failed"),
        ):

            with pytest.raises(
                ConfigurationError, match="Provider initialization failed"
            ):
                await core_engine.initialize_providers()

            assert core_engine._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_providers_empty_providers(self, core_engine):
        """Test initialization with no providers."""
        with patch.object(
            core_engine.provider_initializer,
            "initialize_providers",
            return_value={},
        ):

            await core_engine.initialize_providers()

            assert core_engine.providers == {}
            assert core_engine.current_provider is None
            assert core_engine._initialized is True


class TestModelSwitching:
    """Test model switching functionality."""

    @pytest.mark.asyncio
    async def test_switch_model_success(self, core_engine, mock_providers):
        """Test successful model switching."""
        core_engine.providers = mock_providers
        core_engine.current_provider = mock_providers["openai"]

        result = await core_engine.switch_model("claude", "claude-3-haiku")

        assert result is True
        assert core_engine.current_provider == mock_providers["claude"]
        assert core_engine.current_provider.model == "claude-3-haiku"

    @pytest.mark.asyncio
    async def test_switch_model_provider_only(self, core_engine, mock_providers):
        """Test switching provider without specifying model."""
        core_engine.providers = mock_providers
        core_engine.current_provider = mock_providers["openai"]

        result = await core_engine.switch_model("claude")

        assert result is True
        assert core_engine.current_provider == mock_providers["claude"]
        # Model should remain the same as provider's current model
        assert core_engine.current_provider.model == "claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_switch_model_provider_not_available(
        self, core_engine, mock_providers
    ):
        """Test switching to unavailable provider."""
        core_engine.providers = mock_providers

        result = await core_engine.switch_model("nonexistent")

        assert result is False
        assert core_engine.current_provider is None

    @pytest.mark.asyncio
    async def test_switch_model_invalid_model(self, core_engine, mock_providers):
        """Test switching to invalid model."""
        core_engine.providers = mock_providers

        result = await core_engine.switch_model("openai", "nonexistent-model")

        assert result is False

    @pytest.mark.asyncio
    async def test_switch_model_exception(self, core_engine, mock_providers):
        """Test model switching with exception."""
        core_engine.providers = mock_providers
        mock_providers["openai"].get_available_models.side_effect = Exception(
            "Model fetch failed"
        )

        result = await core_engine.switch_model("openai", "gpt-4")

        assert result is False


class TestMessageSending:
    """Test message sending functionality."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, core_engine, mock_provider):
        """Test successful message sending."""
        core_engine.current_provider = mock_provider

        with (
            patch.object(
                core_engine.chat_manager, "get_current_context"
            ) as mock_context,
            patch.object(core_engine.chat_manager, "add_user_message") as mock_add_user,
            patch.object(
                core_engine.chat_manager, "add_assistant_message"
            ) as mock_add_assistant,
        ):

            mock_context.return_value = Mock()

            response = await core_engine.send_message("Hello")

            assert response.is_success is True
            assert response.content == "Test response"
            mock_provider.send_message.assert_called_once()
            mock_add_user.assert_called_once_with("Hello")
            mock_add_assistant.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_no_provider(self, core_engine):
        """Test sending message with no provider."""
        response = await core_engine.send_message("Hello")

        assert response.is_success is False
        assert "No provider available" in response.error

    @pytest.mark.asyncio
    async def test_send_message_provider_failure(self, core_engine, mock_provider):
        """Test message sending with provider failure."""
        core_engine.current_provider = mock_provider
        mock_provider.send_message.side_effect = Exception("Provider error")

        with patch.object(
            core_engine.chat_manager,
            "get_current_context",
            return_value=Mock(),
        ):
            response = await core_engine.send_message("Hello")

            assert response.is_success is False
            assert "Failed to send message" in response.error

    @pytest.mark.asyncio
    async def test_send_message_unsuccessful_response(self, core_engine, mock_provider):
        """Test handling unsuccessful provider response."""
        core_engine.current_provider = mock_provider
        mock_provider.send_message.return_value = ChatResponse(
            content="", model_used="gpt-4", tokens_used=0, error="API error"
        )

        with (
            patch.object(
                core_engine.chat_manager,
                "get_current_context",
                return_value=Mock(),
            ),
            patch.object(core_engine.chat_manager, "add_user_message") as mock_add_user,
            patch.object(
                core_engine.chat_manager, "add_assistant_message"
            ) as mock_add_assistant,
        ):

            response = await core_engine.send_message("Hello")

            assert response.is_success is False
            assert response.error == "API error"
            # Should add user message but not assistant message on failure
            mock_add_user.assert_called_once_with("Hello")
            mock_add_assistant.assert_not_called()


class TestModelInformation:
    """Test model information retrieval."""

    def test_get_available_models(self, core_engine, mock_providers):
        """Test getting available models from all providers."""
        core_engine.providers = mock_providers

        models = core_engine.get_available_models()

        assert len(models) == 4  # 2 from each provider
        model_names = [m.name for m in models]
        assert "gpt-4" in model_names
        assert "claude-3-sonnet" in model_names

    def test_get_available_models_with_enhanced_info(self, core_engine, mock_providers):
        """Test getting models with enhanced model info."""
        # Mock one provider to return EnhancedModelInfo
        enhanced_model = EnhancedModelInfo(
            name="gpt-4",
            provider="openai",
            description="GPT-4",
            max_tokens=8192,
            cost_per_million_input=10.0,
            cost_per_million_output=30.0,
            swe_score=85.5,
        )
        mock_providers["openai"].get_available_models.return_value = [enhanced_model]
        core_engine.providers = mock_providers

        models = core_engine.get_available_models()

        # Should convert EnhancedModelInfo to ModelInfo
        assert len(models) >= 1
        assert any(m.name == "gpt-4" for m in models)

    def test_get_available_models_provider_error(self, core_engine, mock_providers):
        """Test handling provider errors when getting models."""
        mock_providers["openai"].get_available_models.side_effect = Exception(
            "Provider error"
        )
        core_engine.providers = mock_providers

        models = core_engine.get_available_models()

        # Should still return models from working providers
        assert len(models) == 2  # Only from claude provider
        model_names = [m.name for m in models]
        assert "claude-3-sonnet" in model_names

    def test_get_all_models(self, core_engine, mock_providers):
        """Test getting all models organized by provider."""
        core_engine.providers = mock_providers

        result = core_engine.get_all_models()

        assert "openai" in result
        assert "claude" in result
        assert len(result["openai"]) == 2
        assert len(result["claude"]) == 2

        # Check model structure
        openai_model = result["openai"][0]
        assert "name" in openai_model
        assert "provider" in openai_model
        assert "supports_tools" in openai_model

    def test_get_all_models_with_enhanced_info(self, core_engine, mock_providers):
        """Test getting all models with enhanced information."""
        enhanced_model = EnhancedModelInfo(
            name="gpt-4",
            provider="openai",
            description="GPT-4",
            max_tokens=8192,
            cost_per_million_input=10.0,
            cost_per_million_output=30.0,
            swe_score=85.5,
        )
        mock_providers["openai"].get_available_models.return_value = [enhanced_model]
        core_engine.providers = mock_providers

        result = core_engine.get_all_models()

        openai_model = result["openai"][0]
        assert "swe_score" in openai_model
        assert "cost_display" in openai_model
        assert openai_model["swe_score"] == 85.5

    def test_get_current_model_info(self, core_engine, mock_provider):
        """Test getting current model information."""
        core_engine.current_provider = mock_provider

        info = core_engine.get_current_model_info()

        assert info is not None
        assert info["name"] == "gpt-4"
        assert info["provider"] == "openai"
        assert "supports_tools" in info

    def test_get_current_model_info_no_provider(self, core_engine):
        """Test getting model info with no current provider."""
        info = core_engine.get_current_model_info()
        assert info is None

    def test_get_current_model_info_error(self, core_engine, mock_provider):
        """Test handling error when getting model info."""
        core_engine.current_provider = mock_provider
        mock_provider.get_model_info.side_effect = Exception("Model info error")

        info = core_engine.get_current_model_info()
        assert info is None


class TestConfigurationInfo:
    """Test configuration information retrieval."""

    def test_get_current_config(self, core_engine, mock_provider):
        """Test getting current configuration."""
        core_engine.current_provider = mock_provider

        config = core_engine.get_current_config()

        assert "default_provider" in config
        assert "providers" in config
        assert "current_provider" in config
        assert "current_model" in config
        assert config["default_provider"] == "openai"
        assert config["current_provider"] == "openai"
        assert config["current_model"] == "gpt-4"

    def test_get_current_config_masks_api_keys(self, core_engine):
        """Test that API keys are masked in configuration."""
        config = core_engine.get_current_config()

        for provider_config in config["providers"].values():
            api_key = provider_config["api_key"]
            assert "***" in api_key or api_key == "Not set"

    def test_get_current_config_error(self, core_engine):
        """Test handling error when getting configuration."""
        core_engine.config_manager.get_config.side_effect = Exception("Config error")

        config = core_engine.get_current_config()

        assert "error" in config

    def test_get_conversation_summary(self, core_engine, mock_provider):
        """Test getting conversation summary."""
        core_engine.current_provider = mock_provider

        with patch.object(
            core_engine.chat_manager, "get_current_context"
        ) as mock_context:
            mock_ctx = Mock()
            mock_ctx.messages = [Mock(), Mock(), Mock()]  # 3 messages
            mock_ctx.session_id = "test-session"
            mock_context.return_value = mock_ctx

            summary = core_engine.get_conversation_summary()

            assert summary["message_count"] == 3
            assert summary["current_model"] == "gpt-4"
            assert summary["session_id"] == "test-session"

    def test_get_conversation_summary_error(self, core_engine):
        """Test handling error when getting conversation summary."""
        with patch.object(
            core_engine.chat_manager,
            "get_current_context",
            side_effect=Exception("Context error"),
        ):

            summary = core_engine.get_conversation_summary()

            assert "error" in summary


class TestProviderValidation:
    """Test provider validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_current_provider_success(self, core_engine, mock_provider):
        """Test successful provider validation."""
        core_engine.current_provider = mock_provider

        result = await core_engine.validate_current_provider()

        assert result is True
        mock_provider.validate_credentials.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_current_provider_no_provider(self, core_engine):
        """Test validation with no current provider."""
        result = await core_engine.validate_current_provider()
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_current_provider_failure(self, core_engine, mock_provider):
        """Test provider validation failure."""
        core_engine.current_provider = mock_provider
        mock_provider.validate_credentials.return_value = False

        result = await core_engine.validate_current_provider()

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_current_provider_exception(
        self, core_engine, mock_provider
    ):
        """Test provider validation with exception."""
        core_engine.current_provider = mock_provider
        mock_provider.validate_credentials.side_effect = Exception("Validation error")

        result = await core_engine.validate_current_provider()

        assert result is False


class TestHealthChecking:
    """Test health checking functionality."""

    @pytest.mark.asyncio
    async def test_check_provider_health_specific(self, core_engine):
        """Test checking health of specific provider."""
        health_status = {
            "status": "healthy",
            "available": True,
            "credentials_valid": True,
        }

        with patch.object(
            core_engine.health_monitor,
            "check_provider_health",
            new_callable=AsyncMock,
            return_value=health_status,
        ) as mock_check:

            result = await core_engine.check_provider_health("openai")

            assert "openai" in result
            assert result["openai"] == health_status
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_provider_health_all(self, core_engine):
        """Test checking health of all providers."""
        all_health = {
            "openai": {"status": "healthy", "available": True},
            "claude": {"status": "healthy", "available": True},
        }

        with patch.object(
            core_engine.health_monitor,
            "check_all_providers_health",
            new_callable=AsyncMock,
            return_value=all_health,
        ) as mock_check:

            result = await core_engine.check_provider_health()

            assert result == all_health
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_provider_health_not_configured(self, core_engine):
        """Test checking health of non-configured provider."""
        result = await core_engine.check_provider_health("nonexistent")

        assert "nonexistent" in result
        assert result["nonexistent"]["status"] == "error"
        assert "not configured" in result["nonexistent"]["message"]

    @pytest.mark.asyncio
    async def test_check_provider_health_exception(self, core_engine):
        """Test health check with exception."""
        with patch.object(
            core_engine.health_monitor,
            "check_provider_health",
            side_effect=Exception("Health check error"),
        ):

            result = await core_engine.check_provider_health("openai")

            assert "openai" in result
            assert result["openai"]["status"] == "error"


class TestConversationManagement:
    """Test conversation management functionality."""

    def test_save_conversation(self, core_engine):
        """Test saving conversation."""
        with (
            patch.object(
                core_engine.chat_manager, "get_current_context"
            ) as mock_context,
            patch.object(
                core_engine.conversation_manager,
                "save_conversation",
                return_value="conversation_123.json",
            ) as mock_save,
        ):

            mock_context.return_value = Mock()

            result = core_engine.save_conversation("test_conversation")

            assert result == "conversation_123.json"
            mock_save.assert_called_once()

    def test_save_conversation_error(self, core_engine):
        """Test handling error when saving conversation."""
        with patch.object(
            core_engine.chat_manager,
            "get_current_context",
            side_effect=Exception("Save error"),
        ):

            with pytest.raises(Exception):
                core_engine.save_conversation("test_conversation")

    def test_list_conversations(self, core_engine):
        """Test listing conversations."""
        conversations = [
            {"filename": "conv1.json", "created_at": "2024-01-01"},
            {"filename": "conv2.json", "created_at": "2024-01-02"},
        ]

        with patch.object(
            core_engine.conversation_manager,
            "list_conversations",
            return_value=conversations,
        ) as mock_list:

            result = core_engine.list_conversations()

            assert result == conversations
            mock_list.assert_called_once()

    def test_list_conversations_error(self, core_engine):
        """Test handling error when listing conversations."""
        with patch.object(
            core_engine.conversation_manager,
            "list_conversations",
            side_effect=Exception("List error"),
        ):

            result = core_engine.list_conversations()

            assert result == []

    def test_load_conversation_success(self, core_engine):
        """Test successful conversation loading."""
        mock_context = Mock()

        with patch.object(
            core_engine.conversation_manager,
            "load_conversation",
            return_value=mock_context,
        ) as mock_load:

            result = core_engine.load_conversation("test.json")

            assert result is True
            mock_load.assert_called_once_with("test.json")
            # Check that the context was set
            assert core_engine.chat_manager.current_context == mock_context

    def test_load_conversation_error(self, core_engine):
        """Test handling error when loading conversation."""
        with patch.object(
            core_engine.conversation_manager,
            "load_conversation",
            side_effect=Exception("Load error"),
        ):

            result = core_engine.load_conversation("test.json")

            assert result is False


class TestMCPManagement:
    """Test MCP (Model Context Protocol) management."""

    @pytest.mark.asyncio
    async def test_initialize_mcp(self, core_engine):
        """Test MCP initialization (placeholder)."""
        # Mock mcp_manager if it exists
        if hasattr(core_engine, "mcp_manager"):
            # Set to None to test the no-MCP case
            core_engine.mcp_manager = None

        # This should not raise exception when MCP is not configured
        await core_engine.initialize_mcp()

    @pytest.mark.asyncio
    async def test_shutdown_mcp(self, core_engine):
        """Test MCP shutdown (placeholder)."""
        # Mock mcp_manager if it exists
        if hasattr(core_engine, "mcp_manager"):
            # Set to None to test the no-MCP case
            core_engine.mcp_manager = None

        # This should not raise exception when MCP is not configured
        await core_engine.shutdown_mcp()


class TestUtilityMethods:
    """Test utility and helper methods."""

    def test_get_models_list_with_models(self, core_engine, mock_providers):
        """Test getting models list with available models."""
        core_engine.providers = mock_providers

        models_list = core_engine._get_models_list()

        assert "gpt-4 (openai)" in models_list
        assert "claude-3-sonnet (claude)" in models_list

    def test_get_models_list_no_models(self, core_engine):
        """Test getting models list with no models."""
        models_list = core_engine._get_models_list()

        assert models_list == "No models available."


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_multiple_operations_without_initialization(self, core_engine):
        """Test operations before provider initialization."""
        # Test various operations without initializing providers

        # Should handle gracefully
        result = await core_engine.switch_model("openai")
        assert result is False

        response = await core_engine.send_message("Hello")
        assert response.is_success is False

        models = core_engine.get_available_models()
        assert models == []

        validation = await core_engine.validate_current_provider()
        assert validation is False

    def test_config_manager_error_handling(self, core_engine):
        """Test handling of config manager errors."""
        core_engine.config_manager.get_config.side_effect = Exception("Config error")

        # Should handle config errors gracefully
        config = core_engine.get_current_config()
        assert "error" in config

        # Conversation summary should still work since it doesn't use config manager
        summary = core_engine.get_conversation_summary()
        assert isinstance(summary, dict)
        assert "message_count" in summary

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, core_engine, mock_providers):
        """Test concurrent operations on the engine."""
        core_engine.providers = mock_providers
        core_engine.current_provider = mock_providers["openai"]

        # Run multiple operations concurrently
        tasks = [
            core_engine.send_message("Message 1"),
            core_engine.send_message("Message 2"),
            core_engine.validate_current_provider(),
            core_engine.check_provider_health("openai"),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should handle concurrent operations without issues
        assert len(results) == 4
        # First two should be ChatResponse objects
        assert isinstance(results[0], ChatResponse)
        assert isinstance(results[1], ChatResponse)
        # Third should be boolean
        assert isinstance(results[2], bool)
        # Fourth should be dict
        assert isinstance(results[3], dict)
