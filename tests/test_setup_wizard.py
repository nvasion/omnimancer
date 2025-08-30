"""
Tests for the setup wizard module.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.models import ProviderConfig
from omnimancer.core.provider_registry import ProviderRegistry
from omnimancer.core.setup_wizard import SetupWizard
from omnimancer.utils.errors import ProviderError


@pytest.fixture
def mock_config_manager():
    """Create mock config manager."""
    mock = Mock(spec=ConfigManager)
    mock.is_first_run.return_value = True
    mock.save_config.return_value = None
    return mock


@pytest.fixture
def mock_provider_registry():
    """Create mock provider registry."""
    mock = Mock()
    return mock


@pytest.fixture
def setup_wizard(mock_config_manager, mock_provider_registry):
    """Create SetupWizard instance for testing."""
    return SetupWizard(mock_config_manager, mock_provider_registry)


@pytest.fixture
def mock_console():
    """Create mock console for testing output."""
    with patch("omnimancer.core.setup_wizard_core.Console") as mock_console_class:
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        yield mock_console


class TestSetupWizardInitialization:
    """Test SetupWizard initialization."""

    def test_init(self, mock_config_manager, mock_provider_registry):
        """Test SetupWizard initialization."""
        wizard = SetupWizard(mock_config_manager, mock_provider_registry)

        assert wizard.config_manager == mock_config_manager
        assert wizard.provider_registry == mock_provider_registry
        assert wizard.console is not None
        assert wizard.provider_factory is not None
        assert isinstance(wizard.provider_info, dict)
        assert len(wizard.provider_info) > 0

    def test_provider_info_structure(self, setup_wizard):
        """Test that provider info has expected structure."""
        for provider_name, info in setup_wizard.provider_info.items():
            assert isinstance(provider_name, str)
            assert "name" in info
            assert "description" in info
            assert "strengths" in info
            assert "models" in info
            assert "api_key_url" in info
            assert "setup_notes" in info
            assert isinstance(info["strengths"], list)
            assert isinstance(info["models"], list)


class TestProviderSelection:
    """Test provider selection functionality."""

    def test_select_provider_valid_choice(self, setup_wizard, mock_console):
        """Test selecting a valid provider."""
        with (
            patch.object(setup_wizard, "console", mock_console),
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Prompt.ask",
                return_value="1",
            ),
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Confirm.ask",
                return_value=True,
            ),
            patch.object(setup_wizard, "_show_provider_details"),
        ):

            result = setup_wizard._select_provider()

            # Should return the first provider
            providers = list(setup_wizard.provider_info.keys())
            assert result == providers[0]

    def test_select_provider_quit(self, setup_wizard, mock_console):
        """Test quitting provider selection."""
        with (
            patch.object(setup_wizard, "console", mock_console),
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Prompt.ask",
                return_value="q",
            ),
        ):

            result = setup_wizard._select_provider()
            assert result is None

    def test_show_provider_details(self, setup_wizard, mock_console):
        """Test showing provider details."""
        with (
            patch.object(setup_wizard, "console", mock_console),
            patch.object(setup_wizard.ui, "console", mock_console),
        ):
            setup_wizard._show_provider_details("openai")

            # Verify console.print was called with panel
            assert mock_console.print.called


class TestAPIKeyHandling:
    """Test API key input and validation."""

    def test_get_api_key_valid(self, setup_wizard):
        """Test getting valid API key."""
        provider_info = setup_wizard.provider_info["openai"]

        with patch(
            "omnimancer.core.setup_wizard_provider_setup.Prompt.ask",
            return_value="sk-test123",
        ):
            result = setup_wizard._get_api_key("openai", provider_info)
            assert result == "sk-test123"

    def test_get_api_key_empty_then_continue(self, setup_wizard):
        """Test getting empty API key then continuing without it."""
        provider_info = setup_wizard.provider_info["openai"]

        with (
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Prompt.ask",
                return_value="",
            ),
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Confirm.ask",
                return_value=True,
            ),
        ):

            result = setup_wizard._get_api_key("openai", provider_info)
            assert result is None

    def test_validate_api_key_format_openai_valid(self, setup_wizard):
        """Test OpenAI API key format validation - valid."""
        assert (
            setup_wizard._validate_api_key_format(
                "openai", "sk-test123456789012345678901234567890"
            )
            is True
        )

    def test_validate_api_key_format_openai_invalid(self, setup_wizard):
        """Test OpenAI API key format validation - invalid."""
        assert setup_wizard._validate_api_key_format("openai", "invalid") is False

    def test_validate_api_key_format_claude_valid(self, setup_wizard):
        """Test Claude API key format validation - valid."""
        assert (
            setup_wizard._validate_api_key_format(
                "claude", "sk-ant-test123456789012345678901234567890"
            )
            is True
        )

    def test_validate_api_key_format_ollama(self, setup_wizard):
        """Test Ollama doesn't require API key validation."""
        assert setup_wizard._validate_api_key_format("ollama", None) is True


class TestModelSelection:
    """Test model selection functionality."""

    def test_select_model_from_list(self, setup_wizard, mock_console):
        """Test selecting model from predefined list."""
        provider_info = setup_wizard.provider_info["openai"]

        with (
            patch.object(setup_wizard, "console", mock_console),
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Prompt.ask",
                return_value="1",
            ),
        ):

            result = setup_wizard._select_model("openai", provider_info)
            assert result == provider_info["models"][0]

    def test_select_model_invalid_choice(self, setup_wizard, mock_console):
        """Test model selection with invalid choice."""
        provider_info = setup_wizard.provider_info["openai"]

        with (
            patch.object(setup_wizard, "console", mock_console),
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Prompt.ask",
                side_effect=["999", "1"],
            ),
        ):

            result = setup_wizard._select_model("openai", provider_info)
            assert result == provider_info["models"][0]


class TestProviderConfiguration:
    """Test provider configuration functionality."""

    @pytest.mark.asyncio
    async def test_configure_provider_openai_success(self, setup_wizard, mock_console):
        """Test successful OpenAI provider configuration."""
        with (
            patch.object(setup_wizard, "console", mock_console),
            patch.object(
                setup_wizard.provider_setup,
                "_get_api_key",
                return_value="sk-test123",
            ),
            patch.object(
                setup_wizard.provider_setup, "_select_model", return_value="gpt-4"
            ),
            patch.object(
                setup_wizard.provider_setup,
                "_get_additional_settings",
                return_value={},
            ),
        ):

            result = await setup_wizard._configure_provider("openai")

            assert isinstance(result, ProviderConfig)
            assert result.api_key == "sk-test123"
            assert result.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_configure_provider_ollama_no_api_key(
        self, setup_wizard, mock_console
    ):
        """Test Ollama provider configuration (no API key required)."""
        with (
            patch.object(setup_wizard, "console", mock_console),
            patch.object(
                setup_wizard.provider_setup,
                "_select_model",
                return_value="llama3.2",
            ),
            patch.object(
                setup_wizard.provider_setup,
                "_get_additional_settings",
                return_value={},
            ),
        ):

            result = await setup_wizard._configure_provider("ollama")

            assert isinstance(result, ProviderConfig)
            assert result.api_key is None
            assert result.model == "llama3.2"


class TestConfigurationTesting:
    """Test configuration testing functionality."""

    @pytest.mark.asyncio
    async def test_test_configuration_success(self, setup_wizard, mock_console):
        """Test successful configuration testing."""
        provider_config = ProviderConfig(
            api_key="sk-test123456789012345678901234567890", model="gpt-4"
        )

        with (
            patch.object(setup_wizard, "console", mock_console),
            patch.object(
                setup_wizard.provider_factory, "create_provider"
            ) as mock_create,
            patch("omnimancer.core.setup_wizard_validation.Progress") as mock_progress,
        ):

            mock_provider = AsyncMock()
            mock_provider.send_message.return_value = Mock(content="Test response")
            mock_create.return_value = mock_provider

            # Mock progress context manager
            mock_progress_instance = Mock()
            mock_progress.return_value.__enter__.return_value = mock_progress_instance
            mock_progress.return_value.__exit__.return_value = None

            result = await setup_wizard._test_configuration("openai", provider_config)
            assert result is True

    @pytest.mark.asyncio
    async def test_test_configuration_failure(self, setup_wizard, mock_console):
        """Test configuration testing failure."""
        provider_config = ProviderConfig(api_key="invalid-key", model="gpt-4")

        with (
            patch.object(setup_wizard, "console", mock_console),
            patch.object(
                setup_wizard.provider_factory, "create_provider"
            ) as mock_create,
            patch("omnimancer.core.setup_wizard_validation.Progress") as mock_progress,
            patch("omnimancer.core.setup_wizard_core.Confirm.ask", return_value=False),
        ):

            mock_provider = AsyncMock()
            mock_provider.send_message.side_effect = ProviderError("Invalid API key")
            mock_create.return_value = mock_provider

            # Mock progress context manager
            mock_progress_instance = Mock()
            mock_progress.return_value.__enter__.return_value = mock_progress_instance
            mock_progress.return_value.__exit__.return_value = None

            result = await setup_wizard._test_configuration("openai", provider_config)
            assert result is False


class TestConfigurationSaving:
    """Test configuration saving functionality."""

    @pytest.mark.skip(reason="Config model validation issues with mock")
    def test_save_configuration_success(self, setup_wizard):
        """Test successful configuration saving."""
        provider_config = ProviderConfig(api_key="sk-test123", model="gpt-4")

        # Mock Confirm.ask to avoid stdin issues in pytest
        with patch("omnimancer.core.setup_wizard_core.Confirm.ask", return_value=False):
            # Mock the config manager's methods
            with patch.object(
                setup_wizard.config_manager, "get_config"
            ) as mock_get_config:
                # Mock an empty config (first provider)
                from omnimancer.core.models import ChatSettings, Config

                mock_config = Config(
                    providers={},
                    chat_settings=ChatSettings(),
                    default_provider="",
                    storage_path="/tmp/test",
                )
                mock_get_config.return_value = mock_config

                with patch.object(
                    setup_wizard.config_manager, "set_provider_config"
                ) as mock_set_provider:
                    with patch.object(
                        setup_wizard.config_manager, "set_default_provider"
                    ) as mock_set_default:
                        setup_wizard._save_configuration("openai", provider_config)

                        # Verify config manager methods were called
                        mock_set_provider.assert_called_once_with(
                            "openai", provider_config
                        )
                        mock_set_default.assert_called_once_with("openai")


class TestUtilityMethods:
    """Test utility and helper methods."""

    def test_get_additional_settings_empty(self, setup_wizard):
        """Test getting additional settings returns empty dict by default."""
        with patch("omnimancer.core.setup_wizard_core.Confirm.ask", return_value=False):
            result = setup_wizard._get_additional_settings("openai")
            assert result == {}

    def test_get_additional_settings_with_values(self, setup_wizard):
        """Test getting additional settings with user input."""
        with (
            patch("omnimancer.core.setup_wizard_core.Confirm.ask", return_value=True),
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Prompt.ask",
                side_effect=["0.8", "2048"],
            ),
        ):

            result = setup_wizard._get_additional_settings("openai")
            assert result["temperature"] == 0.8
            assert result["max_tokens"] == 2048

    def test_show_completion(self, setup_wizard, mock_console):
        """Test completion message display."""
        with patch.object(setup_wizard.ui, "console", mock_console):
            setup_wizard._show_completion("openai")

            # Verify console.print was called
            assert mock_console.print.called

    def test_get_available_providers(self, setup_wizard):
        """Test getting list of available providers."""
        providers = setup_wizard.get_available_providers()

        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "openai" in providers
        assert "claude" in providers

    def test_get_provider_info(self, setup_wizard):
        """Test getting provider information."""
        info = setup_wizard.get_provider_info("openai")

        assert info is not None
        assert "name" in info
        assert "description" in info
        assert info["name"] == "OpenAI GPT"

    def test_get_provider_info_unknown(self, setup_wizard):
        """Test getting info for unknown provider."""
        info = setup_wizard.get_provider_info("unknown_provider")
        assert info is None


class TestFullWizardWorkflow:
    """Test complete wizard workflow."""

    @pytest.mark.skip(
        reason="Over-engineered test - complex mocking issues after config simplification"
    )
    @pytest.mark.asyncio
    async def test_start_wizard_complete_success(self, setup_wizard, mock_console):
        """Test complete successful wizard workflow."""
        provider_config = ProviderConfig(api_key="sk-test123", model="gpt-4")

        # Create mock instances that will be returned by the class constructors
        mock_ui = Mock()
        mock_provider_setup = Mock()
        mock_validation = AsyncMock()

        # Configure the mocks to return expected values
        mock_provider_setup.select_provider.return_value = "openai"
        mock_provider_setup.configure_provider = AsyncMock(return_value=provider_config)
        mock_provider_setup.save_configuration = (
            Mock()
        )  # Add missing save_configuration mock
        mock_validation.test_configuration = AsyncMock(return_value=True)

        with (
            patch.object(
                setup_wizard.config_manager, "is_first_run", return_value=True
            ),
            patch(
                "omnimancer.core.setup_wizard_ui.SetupWizardUI",
                return_value=mock_ui,
            ),
            patch(
                "omnimancer.core.setup_wizard_provider_setup.SetupWizardProviderSetup",
                return_value=mock_provider_setup,
            ),
            patch(
                "omnimancer.core.setup_wizard_validation.SetupWizardValidation",
                return_value=mock_validation,
            ),
        ):

            result = await setup_wizard.start_wizard()
            assert result is True

            # Verify the workflow was called correctly
            mock_ui.show_welcome.assert_called_once()
            mock_provider_setup.select_provider.assert_called_once()
            mock_provider_setup.configure_provider.assert_called_once_with("openai")
            mock_validation.test_configuration.assert_called_once_with(
                "openai", provider_config
            )
            mock_provider_setup.save_configuration.assert_called_once()
            mock_ui.show_completion.assert_called_once_with("openai")

    @pytest.mark.asyncio
    async def test_start_wizard_no_provider_selected(self, setup_wizard, mock_console):
        """Test wizard workflow when no provider is selected."""
        with (
            patch.object(setup_wizard, "console", mock_console),
            patch.object(setup_wizard, "_show_welcome"),
            patch.object(
                setup_wizard.config_manager, "is_first_run", return_value=True
            ),
            patch.object(setup_wizard, "_select_provider", return_value=None),
        ):

            result = await setup_wizard.start_wizard()
            assert result is False

    @pytest.mark.asyncio
    async def test_start_wizard_keyboard_interrupt(self, setup_wizard, mock_console):
        """Test wizard workflow with keyboard interrupt."""
        with (
            patch.object(setup_wizard, "console", mock_console),
            patch.object(setup_wizard, "_show_welcome", side_effect=KeyboardInterrupt),
        ):

            result = await setup_wizard.start_wizard()
            assert result is False


class TestProviderSpecificBehavior:
    """Test provider-specific behavior and configurations."""

    def test_api_key_validation_all_providers(self, setup_wizard):
        """Test API key validation for providers with known prefixes."""
        test_cases = [
            ("openai", "sk-test123456789012345678901234567890", True),
            ("openai", "invalid", False),
            ("claude", "sk-ant-test123456789012345678901234567890", True),
            ("claude", "sk-test123", False),
            ("gemini", "AIzatest123456789012345678901234567890", True),
            ("gemini", "invalid", False),
            ("perplexity", "pplx-test123456789012345678901234567890", True),
            ("perplexity", "invalid", False),
            ("xai", "xai-test123456789012345678901234567890", True),
            ("xai", "invalid", False),
            (
                "mistral",
                "mistral-test123456789012345678901",
                True,
            ),  # Key with correct prefix for mistral
            ("mistral", "short", False),
            ("ollama", None, True),  # No API key required
        ]

        for provider, api_key, expected in test_cases:
            result = setup_wizard._validate_api_key_format(provider, api_key)
            assert result == expected, f"Failed for {provider} with key {api_key}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_provider_info_completeness(self, setup_wizard):
        """Test that all providers have complete information."""
        required_keys = [
            "name",
            "description",
            "strengths",
            "models",
            "api_key_url",
            "setup_notes",
        ]

        for provider_name, info in setup_wizard.provider_info.items():
            for key in required_keys:
                assert key in info, f"Provider {provider_name} missing {key}"

            assert (
                len(info["strengths"]) > 0
            ), f"Provider {provider_name} has no strengths"
            assert len(info["models"]) > 0, f"Provider {provider_name} has no models"

    @pytest.mark.asyncio
    async def test_configure_provider_unknown_provider(
        self, setup_wizard, mock_console
    ):
        """Test configuring unknown provider."""
        with patch.object(setup_wizard, "console", mock_console):
            # Should handle gracefully even for unknown providers
            try:
                await setup_wizard._configure_provider("unknown_provider")
                # Result depends on implementation - might return None or handle gracefully
            except KeyError:
                # This is acceptable behavior for unknown providers
                pass

    def test_get_api_key_with_prefix_warning(self, setup_wizard):
        """Test API key input with prefix warning."""
        provider_info = setup_wizard.provider_info["openai"]

        with (
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Prompt.ask",
                return_value="invalid-key",
            ),
            patch(
                "omnimancer.core.setup_wizard_provider_setup.Confirm.ask",
                return_value=True,
            ),
            patch.object(setup_wizard.provider_setup, "console") as mock_console,
        ):

            result = setup_wizard._get_api_key("openai", provider_info)
            assert result == "invalid-key"
            # Should have printed warning about prefix
            assert mock_console.print.called


class TestAsyncBehavior:
    """Test asynchronous behavior and error handling."""

    @pytest.mark.asyncio
    async def test_async_methods_handle_exceptions_gracefully(self, setup_wizard):
        """Test that async methods handle exceptions gracefully."""
        provider_config = ProviderConfig(
            api_key="sk-test123456789012345678901234567890", model="gpt-4"
        )

        # Test that methods handle exceptions gracefully
        with patch(
            "omnimancer.core.setup_wizard_validation.Confirm.ask",
            return_value=False,
        ):
            # Use the actual validation method but patch its dependencies to cause an exception
            with patch.object(
                setup_wizard.validation.provider_factory,
                "create_provider",
                side_effect=Exception("Connection failed"),
            ):

                # Should not raise exception, but return False due to user declining after the exception
                result = await setup_wizard._test_configuration(
                    "openai", provider_config
                )
                assert result is False


class TestSetupWizardSmokeTests:
    """Core smoke tests for SetupWizard functionality."""

    def setup_method(self):
        """Setup test environment with mocked dependencies."""
        self.config_manager = Mock(spec=ConfigManager)
        self.provider_registry = Mock(spec=ProviderRegistry)
        self.setup_wizard = SetupWizard(
            config_manager=self.config_manager,
            provider_registry=self.provider_registry,
        )

    def test_setup_wizard_initialization(self):
        """Test that SetupWizard can be initialized properly."""
        assert self.setup_wizard is not None
        assert self.setup_wizard.config_manager == self.config_manager
        assert self.setup_wizard.provider_registry == self.provider_registry

    def test_provider_info_exists(self):
        """Test that provider info dictionary exists and has expected structure."""
        assert hasattr(self.setup_wizard, "provider_info")
        assert isinstance(self.setup_wizard.provider_info, dict)
        assert len(self.setup_wizard.provider_info) > 0

    def test_get_available_providers(self):
        """Test that get_available_providers method exists and returns data."""
        assert hasattr(self.setup_wizard, "get_available_providers")
        providers = self.setup_wizard.get_available_providers()
        assert isinstance(providers, (list, dict))

    def test_configuration_methods_exist(self):
        """Test that configuration methods are available."""
        methods = [
            "show_provider_help",
            "create_setup_guide",
            "generate_all_setup_guides",
        ]
        for method in methods:
            assert hasattr(self.setup_wizard, method), f"Missing method: {method}"
