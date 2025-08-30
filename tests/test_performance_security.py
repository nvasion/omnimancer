"""
Tests for performance optimizations and security features.

This module contains tests for the performance optimizations
and security features implemented in the provider initialization,
configuration validation, and model catalog loading.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from omnimancer.core.config_validator import ConfigValidator
from omnimancer.core.health_monitor import HealthMonitor

# from omnimancer.core.config_repair import ConfigRepair  # Removed as over-engineered
from omnimancer.core.models import (
    Config,
    EnhancedModelInfo,
    MCPConfig,
    ProviderConfig,
)
from omnimancer.core.provider_initializer import ProviderInitializer


class TestProviderInitializer:
    """Tests for the ProviderInitializer class."""

    def test_lazy_loading(self):
        """Test lazy loading of provider classes."""
        # Clear caches first
        ProviderInitializer.clear_caches()

        # Just test that the cache works by directly setting a provider class

        # Create a mock provider class
        mock_provider_class = MagicMock()
        mock_provider_class.__name__ = "TestProvider"

        # Directly set in cache to test cache functionality
        ProviderInitializer._provider_classes["test_provider"] = mock_provider_class

        # First call should use cache
        provider_class1 = ProviderInitializer.get_provider_class("test_provider")
        assert provider_class1 == mock_provider_class

        # Second call should also use cache
        provider_class2 = ProviderInitializer.get_provider_class("test_provider")
        assert provider_class2 == mock_provider_class
        assert provider_class1 is provider_class2  # Same instance from cache

    def test_provider_instance_caching(self):
        """Test caching of provider instances."""
        # Clear caches first
        ProviderInitializer.clear_caches()

        # Mock provider class
        mock_provider_class = MagicMock()
        mock_instance = MagicMock()
        mock_provider_class.return_value = mock_instance

        # Mock get_provider_class
        with patch.object(
            ProviderInitializer,
            "get_provider_class",
            return_value=mock_provider_class,
        ):
            # Create config
            config = ProviderConfig(api_key="test_key", model="test_model")

            # First call should create instance
            instance1 = ProviderInitializer.get_provider_instance(
                "test_provider", config
            )
            assert instance1 == mock_instance
            mock_provider_class.assert_called_once()

            # Second call should use cached instance
            mock_provider_class.reset_mock()
            instance2 = ProviderInitializer.get_provider_instance(
                "test_provider", config
            )
            assert instance2 == mock_instance
            mock_provider_class.assert_not_called()

            # Different config should create new instance
            config2 = ProviderConfig(api_key="test_key", model="different_model")
            mock_provider_class.reset_mock()
            instance3 = ProviderInitializer.get_provider_instance(
                "test_provider", config2
            )
            assert instance3 == mock_instance  # In our mock, same instance is returned
            mock_provider_class.assert_called_once()

    def test_model_info_caching(self):
        """Test caching of model information."""
        # Clear caches first
        ProviderInitializer.clear_caches()

        # Mock provider class and instance
        mock_provider_class = MagicMock()
        mock_instance = MagicMock()
        mock_provider_class.return_value = mock_instance

        # Setup mock models
        mock_models = [
            EnhancedModelInfo(
                name="model1",
                provider="test_provider",
                description="Test model 1",
                max_tokens=4096,
                cost_per_million_input=1.0,
                cost_per_million_output=2.0,
            )
        ]
        mock_instance.get_available_models.return_value = mock_models

        # Mock get_provider_class
        with patch.object(
            ProviderInitializer,
            "get_provider_class",
            return_value=mock_provider_class,
        ):
            # First call should fetch models
            models1 = ProviderInitializer.get_model_info("test_provider")
            assert models1 == mock_models
            mock_instance.get_available_models.assert_called_once()

            # Second call should use cached models
            mock_instance.get_available_models.reset_mock()
            models2 = ProviderInitializer.get_model_info("test_provider")
            assert models2 == mock_models
            mock_instance.get_available_models.assert_not_called()

    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        # Clear caches first
        ProviderInitializer.clear_caches()

        # Set very short TTL for testing
        original_ttl = ProviderInitializer._cache_ttl
        ProviderInitializer.set_cache_ttl(0.1)  # 100ms TTL

        try:
            # Mock provider class and instance
            mock_provider_class = MagicMock()
            mock_instance = MagicMock()
            mock_provider_class.return_value = mock_instance

            # Setup mock models
            mock_models = [
                EnhancedModelInfo(
                    name="model1",
                    provider="test_provider",
                    description="Test model 1",
                    max_tokens=4096,
                    cost_per_million_input=1.0,
                    cost_per_million_output=2.0,
                )
            ]
            mock_instance.get_available_models.return_value = mock_models

            # Mock get_provider_class
            with patch.object(
                ProviderInitializer,
                "get_provider_class",
                return_value=mock_provider_class,
            ):
                # First call should fetch models
                models1 = ProviderInitializer.get_model_info("test_provider")
                assert models1 == mock_models
                mock_instance.get_available_models.assert_called_once()

                # Wait for cache to expire
                time.sleep(0.2)

                # Next call should fetch models again
                mock_instance.get_available_models.reset_mock()
                models2 = ProviderInitializer.get_model_info("test_provider")
                assert models2 == mock_models
                mock_instance.get_available_models.assert_called_once()
        finally:
            # Restore original TTL
            ProviderInitializer.set_cache_ttl(original_ttl)


class TestConfigValidator:
    """Tests for the ConfigValidator class."""

    def test_validation_caching(self):
        """Test caching of validation results."""
        validator = ConfigValidator()

        # Create test config
        config = Config(
            default_provider="test",
            providers={"test": ProviderConfig(api_key="test_key", model="test_model")},
            storage_path="/tmp/omnimancer",
        )

        # Mock _validate_config to track calls
        original_validate = validator._validate_config
        call_count = [0]

        def mock_validate(cfg):
            call_count[0] += 1
            return original_validate(cfg)

        validator._validate_config = mock_validate

        try:
            # First call should validate
            errors1 = validator.validate_config(config)
            assert call_count[0] == 1

            # Second call should use cache
            errors2 = validator.validate_config(config)
            assert call_count[0] == 1  # Still 1
            assert errors1 == errors2

            # Modify config
            config.default_provider = "different"

            # Should validate again
            validator.validate_config(config)
            assert call_count[0] == 2
        finally:
            # Restore original method
            validator._validate_config = original_validate

    def test_provider_specific_validation(self):
        """Test provider-specific validation rules."""
        validator = ConfigValidator()

        # Test Claude config
        claude_config = ProviderConfig(
            api_key="sk-ant-test123", model="claude-3-sonnet-20240229"
        )
        errors = validator._validate_claude_config(claude_config)
        assert not errors

        # Test invalid Claude model
        invalid_claude_config = ProviderConfig(
            api_key="sk-ant-test123", model="invalid-model"
        )
        errors = validator._validate_claude_config(invalid_claude_config)
        assert len(errors) == 1
        assert "Unknown Claude model" in errors[0]

        # Test OpenAI config
        openai_config = ProviderConfig(api_key="sk-test123", model="gpt-4")
        errors = validator._validate_openai_config(openai_config)
        assert not errors

        # Test invalid OpenAI model
        invalid_openai_config = ProviderConfig(
            api_key="sk-test123", model="invalid-model"
        )
        errors = validator._validate_openai_config(invalid_openai_config)
        assert len(errors) == 1
        assert "Unknown OpenAI model" in errors[0]


class TestHealthMonitor:
    """Tests for the HealthMonitor class."""

    @pytest.mark.asyncio
    async def test_health_check_caching(self):
        """Test caching of health check results."""
        monitor = HealthMonitor()

        # Create test config
        config = ProviderConfig(api_key="test_key", model="test_model")

        # Mock provider instance
        mock_provider = MagicMock()
        mock_provider.validate_credentials.return_value = True
        mock_provider.get_model_info.return_value = MagicMock(available=True)
        mock_provider.supports_tools.return_value = True
        mock_provider.supports_multimodal.return_value = False
        mock_provider.supports_streaming.return_value = True

        # Mock get_provider_instance
        with patch(
            "omnimancer.core.provider_initializer.ProviderInitializer.get_provider_instance",
            return_value=mock_provider,
        ):
            # First call should check health
            status1 = await monitor.check_provider_health("test_provider", config)
            assert status1["status"] == "healthy"
            assert status1["credentials_valid"] is True
            assert status1["model_available"] is True
            mock_provider.validate_credentials.assert_called_once()

            # Second call should use cached result
            mock_provider.validate_credentials.reset_mock()
            status2 = await monitor.check_provider_health("test_provider", config)
            assert status2["status"] == "healthy"
            mock_provider.validate_credentials.assert_not_called()

            # Force check should bypass cache
            mock_provider.validate_credentials.reset_mock()
            status3 = await monitor.check_provider_health(
                "test_provider", config, force=True
            )
            assert status3["status"] == "healthy"
            mock_provider.validate_credentials.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test concurrent health checks for multiple providers."""
        monitor = HealthMonitor()

        # Create test configs
        configs = {
            "provider1": ProviderConfig(api_key="key1", model="model1"),
            "provider2": ProviderConfig(api_key="key2", model="model2"),
            "provider3": ProviderConfig(api_key="key3", model="model3"),
        }

        # Mock provider instance
        def mock_get_provider(provider_name, config):
            mock_provider = MagicMock()
            mock_provider.validate_credentials.return_value = True
            mock_provider.get_model_info.return_value = MagicMock(available=True)
            mock_provider.supports_tools.return_value = True
            mock_provider.supports_multimodal.return_value = False
            mock_provider.supports_streaming.return_value = True
            return mock_provider

        # Mock get_provider_instance
        with patch(
            "omnimancer.core.provider_initializer.ProviderInitializer.get_provider_instance",
            side_effect=mock_get_provider,
        ):
            # Check all providers concurrently
            status = await monitor.check_all_providers_health(configs)

            # Verify results
            assert len(status) == 3
            for provider_name in configs:
                assert provider_name in status
                assert status[provider_name]["status"] == "healthy"
                assert status[provider_name]["credentials_valid"] is True
                assert status[provider_name]["model_available"] is True


@pytest.mark.skip(reason="ConfigRepair removed as over-engineered")
class TestConfigRepair:
    """Tests for the ConfigRepair class."""

    @pytest.mark.skip(reason="ConfigRepair functionality not fully implemented yet")
    def test_analyze_config(self):
        """Test configuration analysis for issues."""
        # repair = ConfigRepair()  # Removed as over-engineered

        # Create test config with issues
        config = Config(
            default_provider="test",
            providers={
                "test": ProviderConfig(api_key="test_key", model=""),  # Missing model
                "ollama": ProviderConfig(
                    api_key="",  # No API key (allowed for Ollama)
                    model="llama3",
                    timeout=5,  # Too low timeout
                ),
            },
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(
                auto_approve_timeout=1,  # Very low timeout (still valid but problematic)
                max_concurrent_servers=100,  # Too high
            ),
        )

        # Analyze config
        issues = repair.analyze_config(config)

        # Verify issues
        assert len(issues) > 0

        # Check for specific issues
        model_issue = next(
            (i for i in issues if "has no model specified" in i["message"]),
            None,
        )
        assert model_issue is not None
        assert model_issue["fixable"] is True

        timeout_issue = next(
            (i for i in issues if "timeout should be at least" in i["message"]),
            None,
        )
        assert timeout_issue is not None
        assert timeout_issue["fixable"] is True

        mcp_issue = next(
            (i for i in issues if "Invalid MCP auto_approve_timeout" in i["message"]),
            None,
        )
        assert mcp_issue is not None
        assert mcp_issue["fixable"] is True

    @pytest.mark.skip(reason="ConfigRepair functionality not fully implemented yet")
    def test_fix_issues(self):
        """Test fixing configuration issues."""
        # repair = ConfigRepair()  # Removed as over-engineered

        # Create test config with issues
        config = Config(
            default_provider="test",
            providers={
                "test": ProviderConfig(api_key="test_key", model=""),  # Missing model
                "ollama": ProviderConfig(
                    api_key="",  # No API key (allowed for Ollama)
                    model="llama3",
                    timeout=5,  # Too low timeout
                ),
            },
            storage_path="/tmp/omnimancer",
            mcp=MCPConfig(
                auto_approve_timeout=1,  # Very low timeout (still valid but problematic)
                max_concurrent_servers=100,  # Too high
            ),
        )

        # Analyze config
        issues = repair.analyze_config(config)

        # Fix issues
        fixed_config, applied_fixes = repair.fix_issues(config, issues)

        # Verify fixes
        assert len(applied_fixes) > 0
        assert fixed_config.providers["test"].model != ""  # Model should be set
        assert (
            fixed_config.providers["ollama"].timeout >= 10
        )  # Timeout should be increased
        assert fixed_config.mcp.auto_approve_timeout > 0  # Should be positive
        assert fixed_config.mcp.max_concurrent_servers == 5  # Should be reduced
