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


class TestStreamEventType:
    """Test StreamEventType enum."""

    def test_all_event_types_exist(self):
        from omnimancer.core.models import StreamEventType

        assert StreamEventType.MESSAGE_START.value == "message_start"
        assert StreamEventType.TEXT_DELTA.value == "text_delta"
        assert StreamEventType.TOOL_USE_START.value == "tool_use_start"
        assert StreamEventType.TOOL_USE_DELTA.value == "tool_use_delta"
        assert StreamEventType.TOOL_USE_END.value == "tool_use_end"
        assert StreamEventType.MESSAGE_COMPLETE.value == "message_complete"
        assert StreamEventType.ERROR.value == "error"

    def test_event_type_count(self):
        from omnimancer.core.models import StreamEventType

        assert len(StreamEventType) == 7


class TestStreamEvent:
    """Test StreamEvent dataclass."""

    def test_defaults(self):
        from omnimancer.core.models import StreamEvent, StreamEventType

        event = StreamEvent(type=StreamEventType.TEXT_DELTA)
        assert event.text == ""
        assert event.tool_name == ""
        assert event.tool_id == ""
        assert event.partial_json == ""
        assert event.response is None
        assert event.model == ""
        assert event.error == ""

    def test_text_delta(self):
        from omnimancer.core.models import StreamEvent, StreamEventType

        event = StreamEvent(type=StreamEventType.TEXT_DELTA, text="Hello")
        assert event.type == StreamEventType.TEXT_DELTA
        assert event.text == "Hello"

    def test_message_start(self):
        from omnimancer.core.models import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.MESSAGE_START,
            model="claude-sonnet-4-6",
        )
        assert event.model == "claude-sonnet-4-6"

    def test_tool_use_start(self):
        from omnimancer.core.models import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.TOOL_USE_START,
            tool_name="file_read",
            tool_id="toolu_123",
        )
        assert event.tool_name == "file_read"
        assert event.tool_id == "toolu_123"

    def test_tool_use_delta(self):
        from omnimancer.core.models import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.TOOL_USE_DELTA,
            partial_json='{"path": "/src',
        )
        assert event.partial_json == '{"path": "/src'

    def test_message_complete_with_response(self):
        from omnimancer.core.models import ChatResponse, StreamEvent, StreamEventType

        response = ChatResponse(
            content="Hello!",
            model_used="claude-sonnet-4-6",
            tokens_used=50,
            input_tokens=100,
            output_tokens=50,
        )
        event = StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, response=response)
        assert event.response is not None
        assert event.response.content == "Hello!"
        assert event.response.input_tokens == 100

    def test_error_event(self):
        from omnimancer.core.models import StreamEvent, StreamEventType

        event = StreamEvent(type=StreamEventType.ERROR, error="Connection lost")
        assert event.error == "Connection lost"


class TestParseDescribedToolCalls:
    """parse_described_tool_calls recovers calls a model emitted as text.

    Weak models mimic the "[Called tools: ...]" history notation instead of
    issuing native tool calls; the parser turns that text back into ToolCall
    objects so the agent loop can keep going.
    """

    def test_round_trips_describe_output(self):
        from omnimancer.core.models import (
            ToolCall,
            describe_tool_calls,
            parse_described_tool_calls,
        )

        calls = [
            ToolCall(name="Grep", arguments={"pattern": "auth", "glob": "*.ts"}),
            ToolCall(name="Read", arguments={"file_path": "/src/main.py"}),
        ]
        parsed = parse_described_tool_calls(describe_tool_calls(calls))

        assert [(c.name, c.arguments) for c in parsed] == [
            (c.name, c.arguments) for c in calls
        ]

    def test_plain_text_returns_empty(self):
        from omnimancer.core.models import parse_described_tool_calls

        assert parse_described_tool_calls("The auth flow looks correct.") == []
        assert parse_described_tool_calls("") == []
        assert parse_described_tool_calls(None) == []

    def test_nested_json_with_braces_and_commas(self):
        from omnimancer.core.models import parse_described_tool_calls

        text = (
            '[Called tools: Grep({"glob": "*.{ts,tsx}", '
            '"output_mode": "files_with_matches", '
            '"pattern": "AUTH_SESSION_KEY|automarketer_auth_session"})]'
        )
        parsed = parse_described_tool_calls(text)

        assert len(parsed) == 1
        assert parsed[0].name == "Grep"
        assert parsed[0].arguments["glob"] == "*.{ts,tsx}"

    def test_marker_embedded_in_other_text(self):
        from omnimancer.core.models import parse_described_tool_calls

        text = (
            "Let me search for that.\n"
            '[Called tools: Bash({"command": "git log --oneline -5"})]'
        )
        parsed = parse_described_tool_calls(text)

        assert len(parsed) == 1
        assert parsed[0].name == "Bash"
        assert parsed[0].arguments == {"command": "git log --oneline -5"}

    def test_malformed_json_is_skipped(self):
        from omnimancer.core.models import parse_described_tool_calls

        assert parse_described_tool_calls("[Called tools: Grep({'bad': json})]") == []


class TestToolCallArgumentNormalization:
    """ToolCall normalizes arguments that arrive as JSON-encoded strings.

    OpenAI-protocol servers send function.arguments as a JSON string (and
    some models double-encode it). Providers that passed the raw value
    through crashed the tool handler with "'str' object has no attribute
    'get'".
    """

    def test_json_string_arguments_become_dict(self):
        from omnimancer.core.models import ToolCall

        tc = ToolCall(name="Glob", arguments='{"pattern": "**/*.py"}')
        assert tc.arguments == {"pattern": "**/*.py"}

    def test_double_encoded_arguments_become_dict(self):
        import json

        from omnimancer.core.models import ToolCall

        double = json.dumps(json.dumps({"pattern": "TODO"}))
        tc = ToolCall(name="Grep", arguments=double)
        assert tc.arguments == {"pattern": "TODO"}

    def test_none_arguments_become_empty_dict(self):
        from omnimancer.core.models import ToolCall

        tc = ToolCall(name="Read", arguments=None)
        assert tc.arguments == {}

    def test_dict_arguments_unchanged(self):
        from omnimancer.core.models import ToolCall

        tc = ToolCall(name="Read", arguments={"file_path": "/src/main.py"})
        assert tc.arguments == {"file_path": "/src/main.py"}

    def test_unparseable_string_left_for_handler_to_report(self):
        from omnimancer.core.models import ToolCall

        tc = ToolCall(name="Glob", arguments="not json at all")
        assert tc.arguments == "not json at all"

    def test_non_object_json_left_for_handler_to_report(self):
        from omnimancer.core.models import ToolCall

        tc = ToolCall(name="Glob", arguments="[1, 2, 3]")
        assert tc.arguments == "[1, 2, 3]"
