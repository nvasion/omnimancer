"""
Tests for the testing infrastructure and fixtures.

This module verifies that all test fixtures and mock factories
are working correctly.
"""

from datetime import datetime
from pathlib import Path

import pytest

from omnimancer.cli.interface import CommandLineInterface
from omnimancer.core.models import ChatResponse, MessageRole, ModelInfo
from tests.conftest import (
    create_chat_message,
    create_chat_response,
    create_model_info,
    generate_conversation_data,
    generate_model_list,
)


class TestFixtures:
    """Test that all fixtures are working correctly."""

    def test_temp_dir_fixture(self, temp_dir):
        """Test temporary directory fixture."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()

    def test_test_config_fixture(self, test_config):
        """Test configuration fixture."""
        assert test_config.default_provider == "openai"
        assert "openai" in test_config.providers
        assert "claude" in test_config.providers
        assert test_config.providers["openai"].api_key == "test-openai-key"

    def test_mock_providers(self, mock_openai_provider, mock_claude_provider):
        """Test mock provider fixtures."""
        assert mock_openai_provider.get_provider_name() == "openai"
        assert mock_claude_provider.get_provider_name() == "claude"
        assert mock_openai_provider.model == "gpt-4"
        assert mock_claude_provider.model == "claude-3-sonnet"

    def test_sample_chat_messages(self, sample_chat_messages):
        """Test sample chat messages fixture."""
        assert len(sample_chat_messages) == 4
        assert sample_chat_messages[0].role == MessageRole.USER
        assert sample_chat_messages[1].role == MessageRole.ASSISTANT
        assert "Hello" in sample_chat_messages[0].content

    def test_sample_chat_context(self, sample_chat_context):
        """Test sample chat context fixture."""
        assert len(sample_chat_context.messages) == 4
        assert sample_chat_context.current_model == "gpt-4"
        assert sample_chat_context.session_id == "test-session-123"

    def test_mock_engine(self, mock_engine):
        """Test mock engine fixture."""
        assert "openai" in mock_engine.providers
        assert "claude" in mock_engine.providers
        assert mock_engine.current_provider is not None

        summary = mock_engine.get_conversation_summary()
        assert summary["message_count"] == 5
        assert summary["current_model"] == "gpt-4"

    def test_mock_cli_interface(self, mock_cli_interface):
        """Test mock CLI interface fixture."""
        assert isinstance(mock_cli_interface, CommandLineInterface)
        assert mock_cli_interface.engine is not None


class TestFactoryFunctions:
    """Test factory functions for creating test objects."""

    def test_create_chat_response(self):
        """Test chat response factory."""
        response = create_chat_response(
            content="Test content", model_used="test-model", tokens_used=20
        )

        assert isinstance(response, ChatResponse)
        assert response.content == "Test content"
        assert response.model_used == "test-model"
        assert response.tokens_used == 20
        assert response.is_success

    def test_create_chat_response_with_error(self):
        """Test chat response factory with error."""
        response = create_chat_response(content="", error="Test error")

        assert not response.is_success
        assert response.error == "Test error"

    def test_create_model_info(self):
        """Test model info factory."""
        model = create_model_info(
            name="test-model",
            provider="test-provider",
            description="Test description",
        )

        assert isinstance(model, ModelInfo)
        assert model.name == "test-model"
        assert model.provider == "test-provider"
        assert model.description == "Test description"
        assert model.available

    def test_create_chat_message(self):
        """Test chat message factory."""
        message = create_chat_message(role=MessageRole.USER, content="Test message")

        assert message.role == MessageRole.USER
        assert message.content == "Test message"
        assert isinstance(message.timestamp, datetime)


class TestMockProviderFactory:
    """Test the mock provider factory."""

    def test_create_working_provider(self, mock_provider_factory):
        """Test creating a working provider."""
        provider = mock_provider_factory.create_working_provider(
            name="test-provider",
            model="test-model",
            response_content="Test response",
        )

        assert provider.get_provider_name() == "test-provider"
        assert provider.model == "test-model"

        # Test async methods
        import asyncio

        async def test_async():
            response = await provider.send_message("test")
            assert response.content == "Test response"
            assert response.model_used == "test-model"

            is_valid = await provider.validate_credentials()
            assert is_valid

        asyncio.run(test_async())

    def test_create_failing_provider(self, mock_provider_factory):
        """Test creating a failing provider."""
        provider = mock_provider_factory.create_failing_provider(
            name="failing-provider", error_message="Test error"
        )

        assert provider.get_provider_name() == "failing-provider"

        # Test that methods fail appropriately
        with pytest.raises(Exception, match="Test error"):
            provider.get_available_models()

    def test_create_rate_limited_provider(self, mock_provider_factory):
        """Test creating a rate limited provider."""
        provider = mock_provider_factory.create_rate_limited_provider(
            name="rate-limited"
        )

        assert provider.get_provider_name() == "rate-limited"

        import asyncio

        async def test_async():
            response = await provider.send_message("test")
            assert not response.is_success
            assert "Rate limit" in response.error

        asyncio.run(test_async())


class TestErrorResponseFixtures:
    """Test error response fixtures."""

    def test_api_error_response(self, api_error_response):
        """Test API error response fixture."""
        assert not api_error_response.is_success
        assert "API request failed" in api_error_response.error
        assert api_error_response.tokens_used == 0

    def test_network_error_response(self, network_error_response):
        """Test network error response fixture."""
        assert not network_error_response.is_success
        assert "Network error" in network_error_response.error

    def test_auth_error_response(self, auth_error_response):
        """Test authentication error response fixture."""
        assert not auth_error_response.is_success
        assert "Authentication failed" in auth_error_response.error


class TestDataGenerators:
    """Test data generator functions."""

    def test_generate_conversation_data(self):
        """Test conversation data generator."""
        conversations = generate_conversation_data(3)

        assert len(conversations) == 3
        assert all("filename" in conv for conv in conversations)
        assert all("created_at" in conv for conv in conversations)
        assert all("message_count" in conv for conv in conversations)

        # Check that message counts increase
        assert conversations[0]["message_count"] == 2
        assert conversations[1]["message_count"] == 4
        assert conversations[2]["message_count"] == 6

    def test_generate_model_list(self):
        """Test model list generator."""
        models = generate_model_list(2)

        assert len(models) >= 2
        assert all(isinstance(model, ModelInfo) for model in models)

        # Check that we have models from different providers
        providers = {model.provider for model in models}
        assert len(providers) >= 2


class TestAsyncFixtures:
    """Test async-related fixtures and utilities."""

    @pytest.mark.asyncio
    async def test_async_mock_methods(self, mock_engine):
        """Test that async mock methods work correctly."""
        # Test send_message
        response = await mock_engine.send_message("test message")
        assert isinstance(response, ChatResponse)
        assert response.content == "Test response"

        # Test switch_model
        result = await mock_engine.switch_model("claude", "claude-3-sonnet")
        assert result is True

        # Test initialize_providers
        await mock_engine.initialize_providers()
        mock_engine.initialize_providers.assert_called_once()

    @pytest.mark.asyncio
    async def test_provider_async_methods(self, mock_openai_provider):
        """Test async methods on mock providers."""
        response = await mock_openai_provider.send_message("test", [])
        assert response.content == "Hello! How can I help you?"

        is_valid = await mock_openai_provider.validate_credentials()
        assert is_valid


class TestMockManagerFixtures:
    """Test mock manager fixtures."""

    def test_mock_config_manager(self, mock_config_manager, test_config):
        """Test mock configuration manager."""
        config = mock_config_manager.get_config()
        assert config == test_config

        # Test that methods can be called
        mock_config_manager.set_default_provider("claude")
        mock_config_manager.set_default_provider.assert_called_once_with("claude")

    def test_mock_chat_manager(self, mock_chat_manager, sample_chat_context):
        """Test mock chat manager."""
        context = mock_chat_manager.get_current_context()
        assert context == sample_chat_context

        # Test method calls
        mock_chat_manager.add_user_message("test")
        mock_chat_manager.add_user_message.assert_called_once_with("test")

    def test_mock_conversation_manager(self, mock_conversation_manager):
        """Test mock conversation manager."""
        # Test save
        filename = mock_conversation_manager.save_conversation(None)
        assert filename == "conversation_20240101_120000.json"

        # Test list
        conversations = mock_conversation_manager.list_conversations()
        assert len(conversations) == 1
        assert conversations[0]["filename"] == "conversation1.json"


class TestIntegrationMarkers:
    """Test that pytest markers are working."""

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test integration marker."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test slow marker."""
        assert True

    @pytest.mark.network
    def test_network_marker(self):
        """Test network marker."""
        assert True
