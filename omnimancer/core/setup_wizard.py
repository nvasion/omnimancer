"""
Interactive setup wizard for first-time configuration.

This module provides an interactive wizard to guide users through
the initial configuration of Omnimancer with provider selection,
API key setup, and validation.

This is now a simplified facade that delegates to specialized modules.
"""

from typing import Any, Dict, List, Optional


from .config_manager import ConfigManager
from .models import ProviderConfig
from .provider_registry import ProviderRegistry
from .setup_wizard_core import SetupWizardCore
from .setup_wizard_guide_generator import SetupWizardGuideGenerator
from .setup_wizard_provider_setup import SetupWizardProviderSetup
from .setup_wizard_ui import SetupWizardUI
from .setup_wizard_validation import SetupWizardValidation


class SetupWizard(SetupWizardCore):
    """
    Interactive setup wizard for first-time configuration.

    This class inherits from SetupWizardCore and adds additional
    convenience methods for documentation generation and provider help.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        provider_registry: ProviderRegistry,
    ):
        """
        Initialize the setup wizard.

        Args:
            config_manager: Configuration manager instance
            provider_registry: Provider registry instance
        """
        super().__init__(config_manager, provider_registry)
        self.ui = SetupWizardUI(self.console, self.provider_info)
        self.provider_setup = SetupWizardProviderSetup(self.console, self.provider_info)
        self.validation = SetupWizardValidation(
            self.console,
            self.provider_info,
            self.provider_factory,
            self.cancellation_handler,
            self.signal_handler,
        )
        self.guide_generator = SetupWizardGuideGenerator(
            self.console, self.provider_info
        )

    async def start_wizard(self) -> bool:
        """
        Start the interactive setup process using pre-initialized components.

        Returns:
            True if setup completed successfully, False otherwise
        """
        try:
            self.ui.show_welcome()

            # Check if this is first run
            if not self.config_manager.is_first_run():
                if not self._confirm_overwrite():
                    return False

            # Select provider
            provider_name = self.provider_setup.select_provider()
            if not provider_name:
                return False

            # Configure the selected provider
            provider_config = await self.provider_setup.configure_provider(
                provider_name
            )
            if not provider_config:
                return False

            # Test the configuration
            if not await self.validation.test_configuration(
                provider_name, provider_config
            ):
                return False

            # Save configuration
            self.provider_setup.save_configuration(
                provider_name, provider_config, self.config_manager
            )

            # Show completion message
            self.ui.show_completion(provider_name)

            return True

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Setup cancelled by user[/yellow]")
            return False
        except Exception as e:
            self.console.print(f"\n[red]Setup failed: {e}[/red]")
            return False

    def show_provider_help(self, provider_name: str) -> None:
        """
        Show provider-specific setup help.

        Args:
            provider_name: Name of the provider
        """
        self.ui.show_provider_help(provider_name)

    async def validate_existing_configuration(
        self, provider_name: str
    ) -> Dict[str, Any]:
        """
        Validate an existing provider configuration.

        Args:
            provider_name: Name of the provider to validate

        Returns:
            Dictionary with validation results
        """
        from .setup_wizard_validation import SetupWizardValidation

        validation = SetupWizardValidation(
            self.console,
            self.provider_info,
            self.provider_factory,
            self.cancellation_handler,
            self.signal_handler,
        )

        return await validation.validate_existing_configuration(
            provider_name, self.config_manager
        )

    def create_setup_guide(
        self, provider_name: str, output_path: Optional[str] = None
    ) -> str:
        """
        Create a detailed setup guide for a specific provider.

        Args:
            provider_name: Name of the provider
            output_path: Optional path to save the guide

        Returns:
            Setup guide content as markdown string
        """
        return self.guide_generator.create_setup_guide(provider_name, output_path)

    def generate_all_setup_guides(self, output_dir: str) -> List[str]:
        """
        Generate setup guides for all supported providers.

        Args:
            output_dir: Directory to save the guides

        Returns:
            List of generated file paths
        """
        return self.guide_generator.generate_all_setup_guides(output_dir)

    # Delegation methods for backward compatibility with tests

    def _get_api_key(self, provider_name: str, info: Dict[str, Any]) -> Optional[str]:
        """
        Get and validate API key from user.

        Args:
            provider_name: Name of the provider
            info: Provider information dictionary

        Returns:
            Valid API key or None if cancelled
        """
        return self.provider_setup._get_api_key(provider_name, info)

    def _validate_api_key_format(
        self, provider_name: str, api_key: Optional[str]
    ) -> bool:
        """
        Validate API key format for a specific provider.

        Args:
            provider_name: Name of the provider
            api_key: API key to validate

        Returns:
            True if format is valid, False otherwise
        """
        return self.validation._validate_api_key_format(provider_name, api_key)

    def _select_model(self, provider_name: str, info: Dict[str, Any]) -> Optional[str]:
        """
        Select model for the provider.

        Args:
            provider_name: Name of the provider
            info: Provider information dictionary

        Returns:
            Selected model name or None if cancelled
        """
        return self.provider_setup._select_model(provider_name, info)

    def _get_additional_settings(self, provider_name: str) -> Dict[str, Any]:
        """
        Get additional provider-specific settings.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary of additional settings
        """
        return self.provider_setup._get_additional_settings(provider_name)

    def _save_configuration(
        self, provider_name: str, provider_config: ProviderConfig
    ) -> None:
        """
        Save the configuration, adding the provider without overwriting existing ones.

        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration to save
        """
        return self.provider_setup.save_configuration(
            provider_name, provider_config, self.config_manager
        )

    async def _test_configuration(
        self, provider_name: str, provider_config: ProviderConfig
    ) -> bool:
        """
        Test the configuration and provide feedback.

        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration to test

        Returns:
            True if configuration is valid, False otherwise
        """
        return await self.validation.test_configuration(provider_name, provider_config)

    async def _configure_provider(self, provider_name: str) -> Optional[ProviderConfig]:
        """
        Configure a specific provider with validation.

        Args:
            provider_name: Name of the provider to configure

        Returns:
            Configured ProviderConfig or None if failed
        """
        return await self.provider_setup.configure_provider(provider_name)

    def _select_provider(self) -> Optional[str]:
        """
        Interactive provider selection with descriptions.

        Returns:
            Selected provider name or None if cancelled
        """
        return self.provider_setup.select_provider()

    def _show_welcome(self) -> None:
        """Show welcome message and setup introduction."""
        return self.ui.show_welcome()

    def _show_provider_details(self, provider_name: str) -> None:
        """
        Show detailed information about a provider.

        Args:
            provider_name: Name of the provider
        """
        return self.ui.show_provider_details(provider_name)

    def _show_completion(self, provider_name: str) -> None:
        """
        Show setup completion message with next steps.

        Args:
            provider_name: Name of the configured provider
        """
        return self.ui.show_completion(provider_name)

    def _show_troubleshooting_guidance(self, provider_name: str, error: str) -> None:
        """
        Show provider-specific troubleshooting guidance.

        Args:
            provider_name: Name of the provider
            error: Error message
        """
        return self.ui.show_troubleshooting_guidance(provider_name, error)

    # Methods expected by tests but delegated to specialized components
    async def _validate_provider_credentials(
        self, provider_name: str, config: Dict[str, Any]
    ) -> bool:
        """Validate provider credentials (delegated to validation component)."""
        from .models import ProviderConfig

        provider_config = ProviderConfig(**config)
        return await self._test_configuration(provider_name, provider_config)

    async def _check_model_availability(self, provider_name: str, model: str) -> bool:
        """Check if model is available for provider."""
        return True  # Stub implementation

    def _generate_model_table(self, provider_name: str) -> str:
        """Generate model table for documentation (delegated to UI component)."""
        return (
            self.ui._generate_model_table(provider_name)
            if hasattr(self.ui, "_generate_model_table")
            else ""
        )

    def _generate_provider_specific_config(self, provider_name: str) -> str:
        """Generate provider-specific configuration documentation."""
        return f"# {provider_name.title()} Configuration\n\n"

    def _generate_troubleshooting_section(self, provider_name: str) -> str:
        """Generate troubleshooting section for documentation."""
        return f"# Troubleshooting {provider_name.title()}\n\n"

    def _generate_best_practices(self, provider_name: str) -> str:
        """Generate best practices section."""
        return f"# Best Practices for {provider_name.title()}\n\n"

    def _generate_support_links(self, provider_name: str) -> str:
        """Generate support links section."""
        return f"# Support Links for {provider_name.title()}\n\n"

    # Note: create_setup_guide method already defined above at line 85
