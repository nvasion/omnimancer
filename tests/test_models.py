"""
Unit tests for the core models.
"""

import pytest
from pydantic import ValidationError

from omnimancer.core.models import (
    ChatSettings,
    Config,
    ConfigProfile,
    EnhancedModelInfo,
    MCPConfig,
    MCPServerConfig,
    ModelInfo,
    ProviderConfig,
)


class TestProviderConfig:
    """Test cases for ProviderConfig model."""

    def test_provider_config_basic(self):
        """Test basic ProviderConfig creation."""
        config = ProviderConfig(model="gpt-4", api_key="sk-test123")

        assert config.model == "gpt-4"
        assert config.api_key == "sk-test123"
        assert config.enabled is True
        assert config.supports_tools is False

    def test_provider_config_validation_temperature(self):
        """Test temperature validation."""
        # Valid temperature
        config = ProviderConfig(model="gpt-4", temperature=0.7)
        assert config.temperature == 0.7

        # Invalid temperature - too high
        with pytest.raises(ValidationError):
            ProviderConfig(model="gpt-4", temperature=3.0)

        # Invalid temperature - negative
        with pytest.raises(ValidationError):
            ProviderConfig(model="gpt-4", temperature=-0.5)

    def test_provider_config_validation_top_p(self):
        """Test top_p validation."""
        # Valid top_p
        config = ProviderConfig(model="gpt-4", top_p=0.9)
        assert config.top_p == 0.9

        # Invalid top_p - too high
        with pytest.raises(ValidationError):
            ProviderConfig(model="gpt-4", top_p=1.5)

        # Invalid top_p - negative
        with pytest.raises(ValidationError):
            ProviderConfig(model="gpt-4", top_p=-0.1)

    def test_provider_config_azure_specific(self):
        """Test Azure-specific configuration."""
        config = ProviderConfig(
            model="gpt-4",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="test-deployment",
            api_version="2024-02-15-preview",
        )

        assert config.azure_endpoint == "https://test.openai.azure.com"
        assert config.azure_deployment == "test-deployment"
        assert config.api_version == "2024-02-15-preview"

    def test_provider_config_ollama_specific(self):
        """Test Ollama-specific configuration."""
        config = ProviderConfig(
            model="llama2",
            base_url="http://localhost:11434",
            num_predict=100,
            num_ctx=2048,
            repeat_penalty=1.1,
        )

        assert config.base_url == "http://localhost:11434"
        assert config.num_predict == 100
        assert config.num_ctx == 2048
        assert config.repeat_penalty == 1.1

    def test_provider_config_template_openai(self):
        """Test OpenAI provider template."""
        template = ProviderConfig.get_provider_config_template("openai")

        assert "model" in template
        assert "api_key" in template
        assert "temperature" in template
        assert "max_tokens" in template

    def test_provider_config_template_claude(self):
        """Test Claude provider template."""
        template = ProviderConfig.get_provider_config_template("claude")

        assert "model" in template
        assert "api_key" in template
        assert "temperature" in template

    def test_provider_config_all_templates(self):
        """Test getting all provider templates."""
        templates = ProviderConfig.get_all_provider_templates()

        assert isinstance(templates, dict)
        assert len(templates) > 0

        # Check some expected providers
        expected_providers = ["claude", "openai", "gemini", "ollama"]
        for provider in expected_providers:
            assert provider in templates
            assert "model" in templates[provider]

    def test_provider_config_string_representation(self):
        """Test string representation masks sensitive data."""
        config = ProviderConfig(
            model="gpt-4", api_key="sk-1234567890abcdef1234567890abcdef"
        )

        str_repr = str(config)
        assert "***masked***" in str_repr
        assert "sk-1234567890abcdef1234567890abcdef" not in str_repr
        assert "gpt-4" in str_repr


class TestMCPServerConfig:
    """Test cases for MCPServerConfig model."""

    def test_mcp_server_config_basic(self):
        """Test basic MCPServerConfig creation."""
        config = MCPServerConfig(
            name="filesystem", command="fs-server", args=["--root", "/tmp"]
        )

        assert config.name == "filesystem"
        assert config.command == "fs-server"
        assert config.args == ["--root", "/tmp"]
        assert config.enabled is True
        assert config.timeout == 30

    def test_mcp_server_config_validation_empty_name(self):
        """Test validation of empty name."""
        with pytest.raises(ValidationError):
            MCPServerConfig(name="", command="test")

    def test_mcp_server_config_validation_empty_command(self):
        """Test validation of empty command."""
        with pytest.raises(ValidationError):
            MCPServerConfig(name="test", command="")

    def test_mcp_server_config_with_env(self):
        """Test MCPServerConfig with environment variables."""
        config = MCPServerConfig(
            name="git", command="git-server", env={"GIT_DIR": "/repo/.git"}
        )

        assert config.env["GIT_DIR"] == "/repo/.git"


class TestMCPConfig:
    """Test cases for MCPConfig model."""

    def test_mcp_config_basic(self):
        """Test basic MCPConfig creation."""
        config = MCPConfig()

        assert config.enabled is True
        assert config.auto_approve_timeout == 30
        assert config.max_concurrent_servers == 10
        assert isinstance(config.servers, dict)

    def test_mcp_config_with_servers(self):
        """Test MCPConfig with servers."""
        server_config = MCPServerConfig(name="fs", command="fs-server")
        config = MCPConfig(servers={"filesystem": server_config})

        assert "filesystem" in config.servers
        assert config.servers["filesystem"].name == "fs"

    def test_mcp_config_validation_timeouts(self):
        """Test validation of timeout values."""
        # Valid timeout
        config = MCPConfig(auto_approve_timeout=60)
        assert config.auto_approve_timeout == 60

        # Invalid timeout - zero
        with pytest.raises(ValidationError):
            MCPConfig(auto_approve_timeout=0)

        # Invalid timeout - negative
        with pytest.raises(ValidationError):
            MCPConfig(auto_approve_timeout=300)

    def test_mcp_config_get_enabled_servers(self):
        """Test getting only enabled servers."""
        server1 = MCPServerConfig(name="fs", command="fs-server", enabled=True)
        server2 = MCPServerConfig(name="git", command="git-server", enabled=False)

        config = MCPConfig(servers={"filesystem": server1, "git": server2})

        enabled = config.get_enabled_servers()
        assert len(enabled) == 1
        assert "filesystem" in enabled
        assert "git" not in enabled

    def test_mcp_config_add_remove_server(self):
        """Test adding and removing servers."""
        config = MCPConfig()
        server = MCPServerConfig(name="test", command="test-server")

        # Add server
        config.add_server("test", server)
        assert "test" in config.servers

        # Remove server
        result = config.remove_server("test")
        assert result is True
        assert "test" not in config.servers

        # Remove non-existent server
        result = config.remove_server("nonexistent")
        assert result is False


class TestChatSettings:
    """Test cases for ChatSettings model."""

    def test_chat_settings_basic(self):
        """Test basic ChatSettings creation."""
        settings = ChatSettings()

        assert settings.context_length == 4000
        assert settings.save_history is True
        assert settings.max_tokens is None
        assert settings.temperature is None

    def test_chat_settings_custom(self):
        """Test ChatSettings with custom values."""
        settings = ChatSettings(
            max_tokens=2048,
            temperature=0.7,
            context_length=8000,
            save_history=False,
        )

        assert settings.max_tokens == 2048
        assert settings.temperature == 0.7
        assert settings.context_length == 8000
        assert settings.save_history is False


class TestConfig:
    """Test cases for Config model."""

    def test_config_basic(self):
        """Test basic Config creation."""
        provider_config = ProviderConfig(model="gpt-4", api_key="test")
        config = Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp/omnimancer",
        )

        assert config.default_provider == "openai"
        assert "openai" in config.providers
        assert config.storage_path == "/tmp/omnimancer"
        assert isinstance(config.chat_settings, ChatSettings)
        assert isinstance(config.mcp, MCPConfig)

    def test_config_validation_empty_default_provider(self):
        """Test validation of empty default provider."""
        provider_config = ProviderConfig(model="gpt-4", api_key="test")

        with pytest.raises(ValidationError):
            Config(
                default_provider="",
                providers={"openai": provider_config},
                storage_path="/tmp",
            )

    def test_config_validation_empty_storage_path(self):
        """Test validation of empty storage path."""
        provider_config = ProviderConfig(model="gpt-4", api_key="test")

        with pytest.raises(ValidationError):
            Config(
                default_provider="openai",
                providers={"openai": provider_config},
                storage_path="",
            )

    def test_config_create_profile(self):
        """Test creating a configuration profile."""
        provider_config = ProviderConfig(model="gpt-4", api_key="test")
        config = Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp",
        )

        profile = config.create_profile("development", "Dev environment")

        assert profile.name == "development"
        assert profile.description == "Dev environment"
        assert profile.default_provider == "openai"
        assert "development" in config.profiles

    def test_config_switch_profile(self):
        """Test switching configuration profiles."""
        provider_config = ProviderConfig(model="gpt-4", api_key="test")
        config = Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp",
        )

        # Create profile
        config.create_profile("test", "Test profile")

        # Switch to profile
        config.switch_profile("test")
        assert config.active_profile == "test"

        # Try to switch to non-existent profile
        with pytest.raises(ValueError):
            config.switch_profile("nonexistent")

    def test_config_delete_profile(self):
        """Test deleting a configuration profile."""
        provider_config = ProviderConfig(model="gpt-4", api_key="test")
        config = Config(
            default_provider="openai",
            providers={"openai": provider_config},
            storage_path="/tmp",
        )

        # Create and switch to profile
        config.create_profile("test", "Test profile")
        config.switch_profile("test")

        # Delete profile
        result = config.delete_profile("test")
        assert result is True
        assert "test" not in config.profiles
        assert config.active_profile is None

        # Try to delete non-existent profile
        result = config.delete_profile("nonexistent")
        assert result is False


class TestEnhancedModelInfo:
    """Test cases for EnhancedModelInfo model."""

    def test_enhanced_model_info_basic(self):
        """Test basic EnhancedModelInfo creation."""
        model = EnhancedModelInfo(
            name="gpt-4",
            provider="openai",
            description="GPT-4 model",
            max_tokens=8192,
            cost_per_million_input=30.0,
            cost_per_million_output=60.0,
        )

        assert model.name == "gpt-4"
        assert model.provider == "openai"
        assert model.cost_per_million_input == 30.0
        assert model.cost_per_million_output == 60.0
        assert model.context_window == 4096  # default

    def test_enhanced_model_info_swe_score(self):
        """Test SWE score functionality."""
        model = EnhancedModelInfo(
            name="gpt-4",
            provider="openai",
            description="GPT-4 model",
            max_tokens=8192,
            cost_per_million_input=30.0,
            cost_per_million_output=60.0,
            swe_score=75.5,
        )

        assert model.swe_score == 75.5
        assert model.get_swe_rating() == "★★★"
        assert "75.5%" in model.get_swe_display()

    def test_enhanced_model_info_cost_display(self):
        """Test cost display functionality."""
        # Paid model
        model = EnhancedModelInfo(
            name="gpt-4",
            provider="openai",
            description="GPT-4 model",
            max_tokens=8192,
            cost_per_million_input=30.0,
            cost_per_million_output=60.0,
        )

        cost_display = model.get_cost_display()
        assert "$30.00 in, $60.00 out" == cost_display

        # Free model
        free_model = EnhancedModelInfo(
            name="llama2",
            provider="ollama",
            description="Llama 2 model",
            max_tokens=4096,
            cost_per_million_input=0.0,
            cost_per_million_output=0.0,
            is_free=True,
        )

        assert free_model.get_cost_display() == "Free"

    def test_enhanced_model_info_validation(self):
        """Test model info validation."""
        model = EnhancedModelInfo(
            name="test",
            provider="test",
            description="Test model",
            max_tokens=4096,
            cost_per_million_input=10.0,
            cost_per_million_output=20.0,
            swe_score=85.0,
        )

        assert model.validate_pricing() is True
        assert model.validate_swe_score() is True

        # Test invalid SWE score
        model.swe_score = 150.0
        assert model.validate_swe_score() is False

    def test_enhanced_model_info_to_model_info(self):
        """Test conversion to legacy ModelInfo."""
        enhanced = EnhancedModelInfo(
            name="gpt-4",
            provider="openai",
            description="GPT-4 model",
            max_tokens=8192,
            cost_per_million_input=30.0,
            cost_per_million_output=60.0,
        )

        legacy = enhanced.to_model_info()

        assert isinstance(legacy, ModelInfo)
        assert legacy.name == "gpt-4"
        assert legacy.provider == "openai"
        assert legacy.max_tokens == 8192
        # Cost should be average per token
        expected_cost = (30.0 + 60.0) / 2 / 1_000_000
        assert abs(legacy.cost_per_token - expected_cost) < 1e-10


class TestConfigProfile:
    """Test cases for ConfigProfile model."""

    def test_config_profile_basic(self):
        """Test basic ConfigProfile creation."""
        provider_config = ProviderConfig(model="gpt-4", api_key="test")
        profile = ConfigProfile(
            name="development",
            description="Development environment",
            default_provider="openai",
            providers={"openai": provider_config},
        )

        assert profile.name == "development"
        assert profile.description == "Development environment"
        assert profile.default_provider == "openai"
        assert "openai" in profile.providers

    def test_config_profile_validation_empty_name(self):
        """Test validation of empty profile name."""
        provider_config = ProviderConfig(model="gpt-4", api_key="test")

        with pytest.raises(ValidationError):
            ConfigProfile(
                name="",
                default_provider="openai",
                providers={"openai": provider_config},
            )

    def test_config_profile_validation_empty_default_provider(self):
        """Test validation of empty default provider."""
        provider_config = ProviderConfig(model="gpt-4", api_key="test")

        with pytest.raises(ValidationError):
            ConfigProfile(
                name="test",
                default_provider="",
                providers={"openai": provider_config},
            )
