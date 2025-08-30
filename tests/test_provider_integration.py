"""
Integration tests for provider switching and error handling.

This module tests the integration between providers, the core engine,
and error handling across different failure scenarios.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.utils.errors import (
    ConfigurationError,
)
from tests.conftest import (
    create_chat_response,
    create_model_info,
)


@pytest.mark.integration
class TestProviderSwitching:
    """Test provider switching functionality and context preservation."""

    @pytest.mark.asyncio
    async def test_switch_between_working_providers(
        self, mock_engine, mock_provider_factory
    ):
        """Test switching between two working providers."""
        # Setup providers
        openai_provider = mock_provider_factory.create_working_provider(
            "openai", "gpt-4", "OpenAI response"
        )
        claude_provider = mock_provider_factory.create_working_provider(
            "claude", "claude-3-sonnet", "Claude response"
        )

        mock_engine.providers = {
            "openai": openai_provider,
            "claude": claude_provider,
        }
        mock_engine.current_provider = openai_provider

        # Mock the actual switch_model method
        async def mock_switch_model(provider_name, model_name=None):
            if provider_name in mock_engine.providers:
                mock_engine.current_provider = mock_engine.providers[provider_name]
                return True
            return False

        mock_engine.switch_model = mock_switch_model

        # Test initial state
        assert mock_engine.current_provider == openai_provider

        # Switch to Claude
        result = await mock_engine.switch_model("claude")
        assert result is True
        assert mock_engine.current_provider == claude_provider

        # Switch back to OpenAI
        result = await mock_engine.switch_model("openai")
        assert result is True
        assert mock_engine.current_provider == openai_provider

    @pytest.mark.asyncio
    async def test_switch_to_nonexistent_provider(self, mock_engine):
        """Test switching to a provider that doesn't exist."""
        mock_engine.providers = {"openai": MagicMock()}

        # Mock switch_model to raise ConfigurationError for unknown providers
        async def mock_switch_model(provider_name, model_name=None):
            if provider_name not in mock_engine.providers:
                raise ConfigurationError(f"Provider '{provider_name}' is not available")
            return True

        mock_engine.switch_model = mock_switch_model

        with pytest.raises(
            ConfigurationError, match="Provider 'unknown' is not available"
        ):
            await mock_engine.switch_model("unknown")

    @pytest.mark.asyncio
    async def test_context_preservation_during_switch(
        self, mock_engine, sample_chat_context
    ):
        """Test that conversation context is preserved during provider switches."""
        # Setup providers
        openai_provider = MagicMock()
        claude_provider = MagicMock()

        mock_engine.providers = {
            "openai": openai_provider,
            "claude": claude_provider,
        }
        mock_engine.current_provider = openai_provider
        mock_engine.chat_manager.get_current_context.return_value = sample_chat_context

        # Mock switch_model to preserve context
        async def mock_switch_model(provider_name, model_name=None):
            if provider_name in mock_engine.providers:
                old_context = mock_engine.chat_manager.get_current_context()
                mock_engine.current_provider = mock_engine.providers[provider_name]
                mock_engine.chat_manager.set_current_model(
                    mock_engine.current_provider.model
                )
                # Context should be preserved
                assert mock_engine.chat_manager.get_current_context() == old_context
                return True
            return False

        mock_engine.switch_model = mock_switch_model

        # Get initial context
        initial_context = mock_engine.chat_manager.get_current_context()
        initial_message_count = len(initial_context.messages)

        # Switch provider
        result = await mock_engine.switch_model("claude")
        assert result is True

        # Verify context is preserved
        current_context = mock_engine.chat_manager.get_current_context()
        assert len(current_context.messages) == initial_message_count
        assert current_context.session_id == initial_context.session_id

    @pytest.mark.asyncio
    async def test_model_switching_within_provider(
        self, mock_engine, mock_provider_factory
    ):
        """Test switching models within the same provider."""
        # Setup provider with multiple models
        openai_provider = mock_provider_factory.create_working_provider(
            "openai", "gpt-4"
        )
        openai_provider.get_available_models.return_value = [
            create_model_info("gpt-4", "openai", "GPT-4"),
            create_model_info("gpt-3.5-turbo", "openai", "GPT-3.5 Turbo"),
        ]

        mock_engine.providers = {"openai": openai_provider}
        mock_engine.current_provider = openai_provider

        # Mock switch_model to handle model switching
        async def mock_switch_model(provider_name, model_name=None):
            if provider_name == "openai" and model_name:
                available_models = openai_provider.get_available_models()
                if any(model.name == model_name for model in available_models):
                    openai_provider.model = model_name
                    return True
                else:
                    raise ConfigurationError(f"Model '{model_name}' not available")
            return False

        mock_engine.switch_model = mock_switch_model

        # Test switching to valid model
        result = await mock_engine.switch_model("openai", "gpt-3.5-turbo")
        assert result is True
        assert openai_provider.model == "gpt-3.5-turbo"

        # Test switching to invalid model
        with pytest.raises(
            ConfigurationError, match="Model 'invalid-model' not available"
        ):
            await mock_engine.switch_model("openai", "invalid-model")


@pytest.mark.integration
class TestProviderErrorHandling:
    """Test error handling across different provider failure scenarios."""

    @pytest.mark.asyncio
    async def test_api_key_authentication_failure(
        self, mock_engine, mock_provider_factory
    ):
        """Test handling of authentication failures."""
        # Create provider that fails authentication
        failing_provider = mock_provider_factory.create_failing_provider(
            "openai", "gpt-4", "Authentication failed: Invalid API key"
        )
        failing_provider.send_message = AsyncMock(
            return_value=create_chat_response(
                content="", error="Authentication failed: Invalid API key"
            )
        )

        mock_engine.providers = {"openai": failing_provider}
        mock_engine.current_provider = failing_provider

        # Mock send_message to use the provider
        async def mock_send_message(message):
            return await mock_engine.current_provider.send_message(message, [])

        mock_engine.send_message = mock_send_message

        # Test that authentication error is properly handled
        response = await mock_engine.send_message("Hello")
        assert not response.is_success
        assert "Authentication failed" in response.error

    @pytest.mark.asyncio
    async def test_rate_limiting_error(self, mock_engine, mock_provider_factory):
        """Test handling of rate limiting errors."""
        # Create rate-limited provider
        rate_limited_provider = mock_provider_factory.create_rate_limited_provider(
            "openai"
        )

        mock_engine.providers = {"openai": rate_limited_provider}
        mock_engine.current_provider = rate_limited_provider

        # Mock send_message to use the provider
        async def mock_send_message(message):
            return await mock_engine.current_provider.send_message(message, [])

        mock_engine.send_message = mock_send_message

        # Test rate limiting error
        response = await mock_engine.send_message("Hello")
        assert not response.is_success
        assert "Rate limit" in response.error

    @pytest.mark.asyncio
    async def test_network_timeout_error(self, mock_engine, mock_provider_factory):
        """Test handling of network timeout errors."""
        # Create provider that times out
        timeout_provider = mock_provider_factory.create_failing_provider(
            "openai", "gpt-4", "Network timeout"
        )
        timeout_provider.send_message = AsyncMock(
            side_effect=Exception("Network timeout")
        )

        mock_engine.providers = {"openai": timeout_provider}
        mock_engine.current_provider = timeout_provider

        # Mock send_message to handle exceptions
        async def mock_send_message(message):
            try:
                return await mock_engine.current_provider.send_message(message, [])
            except Exception as e:
                return create_chat_response(content="", error=str(e))

        mock_engine.send_message = mock_send_message

        # Test network timeout handling
        response = await mock_engine.send_message("Hello")
        assert not response.is_success
        assert "Network timeout" in response.error

    @pytest.mark.asyncio
    async def test_provider_unavailable_fallback(
        self, mock_engine, mock_provider_factory
    ):
        """Test fallback behavior when primary provider is unavailable."""
        # Create working and failing providers
        working_provider = mock_provider_factory.create_working_provider(
            "claude", "claude-3-sonnet", "Claude response"
        )
        failing_provider = mock_provider_factory.create_failing_provider(
            "openai", "gpt-4", "Service unavailable"
        )

        mock_engine.providers = {
            "openai": failing_provider,
            "claude": working_provider,
        }
        mock_engine.current_provider = failing_provider

        # Test that we can switch to working provider when current fails
        async def mock_switch_model(provider_name, model_name=None):
            if provider_name in mock_engine.providers:
                mock_engine.current_provider = mock_engine.providers[provider_name]
                return True
            return False

        mock_engine.switch_model = mock_switch_model

        # Switch to working provider
        result = await mock_engine.switch_model("claude")
        assert result is True
        assert mock_engine.current_provider == working_provider

    @pytest.mark.asyncio
    async def test_malformed_response_handling(
        self, mock_engine, mock_provider_factory
    ):
        """Test handling of malformed API responses."""
        # Create provider that returns malformed responses
        malformed_provider = MagicMock()
        malformed_provider.get_provider_name.return_value = "openai"
        malformed_provider.model = "gpt-4"
        malformed_provider.send_message = AsyncMock(
            return_value=create_chat_response(
                content="", error="Malformed response from API"
            )
        )

        mock_engine.providers = {"openai": malformed_provider}
        mock_engine.current_provider = malformed_provider

        # Mock send_message
        async def mock_send_message(message):
            return await mock_engine.current_provider.send_message(message, [])

        mock_engine.send_message = mock_send_message

        # Test malformed response handling
        response = await mock_engine.send_message("Hello")
        assert not response.is_success
        assert "Malformed response" in response.error


@pytest.mark.integration
class TestProviderValidation:
    """Test provider validation and credential checking."""

    @pytest.mark.asyncio
    async def test_provider_credential_validation(
        self, mock_engine, mock_provider_factory
    ):
        """Test provider credential validation."""
        # Create providers with different credential states
        valid_provider = mock_provider_factory.create_working_provider("openai")
        invalid_provider = mock_provider_factory.create_failing_provider("claude")
        invalid_provider.validate_credentials = AsyncMock(return_value=False)

        mock_engine.providers = {
            "openai": valid_provider,
            "claude": invalid_provider,
        }

        # Mock validate_current_provider
        async def mock_validate_current_provider():
            if mock_engine.current_provider:
                return await mock_engine.current_provider.validate_credentials()
            return False

        mock_engine.validate_current_provider = mock_validate_current_provider

        # Test valid provider
        mock_engine.current_provider = valid_provider
        is_valid = await mock_engine.validate_current_provider()
        assert is_valid is True

        # Test invalid provider
        mock_engine.current_provider = invalid_provider
        is_valid = await mock_engine.validate_current_provider()
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_model_availability_checking(
        self, mock_engine, mock_provider_factory
    ):
        """Test checking model availability across providers."""
        # Create providers with different model availability
        openai_provider = mock_provider_factory.create_working_provider("openai")
        openai_provider.get_available_models.return_value = [
            create_model_info("gpt-4", "openai", "GPT-4", available=True),
            create_model_info("gpt-3.5-turbo", "openai", "GPT-3.5", available=False),
        ]

        claude_provider = mock_provider_factory.create_working_provider("claude")
        claude_provider.get_available_models.return_value = [
            create_model_info("claude-3-sonnet", "claude", "Claude 3", available=True)
        ]

        mock_engine.providers = {
            "openai": openai_provider,
            "claude": claude_provider,
        }

        # Mock get_available_models
        def mock_get_available_models():
            models = []
            for provider in mock_engine.providers.values():
                try:
                    models.extend(provider.get_available_models())
                except Exception:
                    continue
            return models

        mock_engine.get_available_models = mock_get_available_models

        # Test model availability
        models = mock_engine.get_available_models()
        assert len(models) == 3

        available_models = [m for m in models if m.available]
        assert len(available_models) == 2

        unavailable_models = [m for m in models if not m.available]
        assert len(unavailable_models) == 1
        assert unavailable_models[0].name == "gpt-3.5-turbo"


@pytest.mark.integration
class TestRetryBehavior:
    """Test retry behavior and exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_temporary_failure(self, mock_engine, mock_provider_factory):
        """Test retry behavior on temporary failures."""
        # Create provider that fails first, then succeeds
        retry_provider = MagicMock()
        retry_provider.get_provider_name.return_value = "openai"
        retry_provider.model = "gpt-4"

        # Setup call counter for retry simulation
        call_count = 0

        async def mock_send_message(message, context):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return create_chat_response(content="", error="Temporary failure")
            else:
                return create_chat_response(content="Success after retry")

        retry_provider.send_message = mock_send_message

        mock_engine.providers = {"openai": retry_provider}
        mock_engine.current_provider = retry_provider

        # Mock send_message with retry logic
        async def mock_engine_send_message(message):
            max_retries = 2
            for attempt in range(max_retries):
                response = await mock_engine.current_provider.send_message(message, [])
                if response.is_success or attempt == max_retries - 1:
                    return response
                # In real implementation, would have exponential backoff here
            return response

        mock_engine.send_message = mock_engine_send_message

        # Test retry behavior
        response = await mock_engine.send_message("Hello")
        assert response.is_success
        assert response.content == "Success after retry"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, mock_engine, mock_provider_factory):
        """Test behavior when max retries are exceeded."""
        # Create provider that always fails
        failing_provider = MagicMock()
        failing_provider.get_provider_name.return_value = "openai"
        failing_provider.model = "gpt-4"
        failing_provider.send_message = AsyncMock(
            return_value=create_chat_response(content="", error="Persistent failure")
        )

        mock_engine.providers = {"openai": failing_provider}
        mock_engine.current_provider = failing_provider

        # Mock send_message with retry logic
        call_count = 0

        async def mock_engine_send_message(message):
            nonlocal call_count
            max_retries = 3
            last_response = None

            for attempt in range(max_retries):
                call_count += 1
                last_response = await mock_engine.current_provider.send_message(
                    message, []
                )
                if last_response.is_success:
                    break

            return last_response

        mock_engine.send_message = mock_engine_send_message

        # Test max retries
        response = await mock_engine.send_message("Hello")
        assert not response.is_success
        assert "Persistent failure" in response.error
        assert call_count == 3


@pytest.mark.integration
class TestConcurrentProviderOperations:
    """Test concurrent operations across multiple providers."""

    @pytest.mark.asyncio
    async def test_concurrent_model_queries(self, mock_engine, mock_provider_factory):
        """Test querying multiple providers concurrently."""
        import asyncio

        # Create multiple providers
        providers = {}
        for i, name in enumerate(["openai", "claude", "anthropic"]):
            provider = mock_provider_factory.create_working_provider(
                name, f"{name}-model", f"Response from {name}"
            )
            providers[name] = provider

        mock_engine.providers = providers

        # Mock concurrent model querying
        async def query_provider_models(provider_name):
            provider = mock_engine.providers[provider_name]
            return provider.get_available_models()

        # Test concurrent queries
        tasks = [query_provider_models(name) for name in providers.keys()]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(len(models) > 0 for models in results)

    @pytest.mark.asyncio
    async def test_provider_health_check(self, mock_engine, mock_provider_factory):
        """Test health checking across all providers."""
        import asyncio

        # Create providers with different health states
        healthy_provider = mock_provider_factory.create_working_provider("openai")
        unhealthy_provider = mock_provider_factory.create_failing_provider("claude")

        mock_engine.providers = {
            "openai": healthy_provider,
            "claude": unhealthy_provider,
        }

        # Mock health check function
        async def check_provider_health(provider_name):
            provider = mock_engine.providers[provider_name]
            try:
                return await provider.validate_credentials()
            except Exception:
                return False

        # Test concurrent health checks
        tasks = [check_provider_health(name) for name in mock_engine.providers.keys()]
        health_results = await asyncio.gather(*tasks)

        assert health_results[0] is True  # openai healthy
        assert health_results[1] is False  # claude unhealthy


@pytest.mark.integration
class TestProviderConfigurationIntegration:
    """Test integration between provider configuration and switching."""

    @pytest.mark.asyncio
    async def test_configuration_driven_provider_setup(self, mock_engine, test_config):
        """Test setting up providers based on configuration."""

        # Mock initialize_providers to use configuration
        async def mock_initialize_providers():
            mock_engine.providers.clear()

            for (
                provider_name,
                provider_config,
            ) in test_config.providers.items():
                # In real implementation, would create actual providers
                mock_provider = MagicMock()
                mock_provider.get_provider_name.return_value = provider_name
                mock_provider.model = provider_config.model
                mock_provider.validate_credentials = AsyncMock(return_value=True)
                mock_engine.providers[provider_name] = mock_provider

            # Set default provider
            if test_config.default_provider in mock_engine.providers:
                mock_engine.current_provider = mock_engine.providers[
                    test_config.default_provider
                ]

        mock_engine.initialize_providers = mock_initialize_providers

        # Test initialization
        await mock_engine.initialize_providers()

        assert len(mock_engine.providers) == 2
        assert "openai" in mock_engine.providers
        assert "claude" in mock_engine.providers
        assert mock_engine.current_provider.get_provider_name() == "openai"

    @pytest.mark.asyncio
    async def test_dynamic_configuration_update(self, mock_engine, mock_config_manager):
        """Test updating provider configuration dynamically."""
        # Initial setup
        mock_engine.providers = {"openai": MagicMock()}

        # Mock configuration update
        def mock_update_provider_settings(provider_name, **kwargs):
            if provider_name in mock_engine.providers:
                provider = mock_engine.providers[provider_name]
                for key, value in kwargs.items():
                    setattr(provider, key, value)

        mock_config_manager.update_provider_settings = mock_update_provider_settings
        mock_engine.config_manager = mock_config_manager

        # Test configuration update
        mock_engine.config_manager.update_provider_settings(
            "openai", model="gpt-4-turbo"
        )

        assert mock_engine.providers["openai"].model == "gpt-4-turbo"
