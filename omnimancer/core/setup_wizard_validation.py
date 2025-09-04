"""
Validation methods for the setup wizard.

This module contains all validation and testing methods for provider
configurations, including API key validation and credential testing.
"""

import asyncio
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from .models import ProviderConfig
from .setup_wizard_ui import SetupWizardUI


class SetupWizardValidation:
    """Validation methods for the setup wizard."""

    def __init__(
        self,
        console: Console,
        provider_info: Dict[str, Dict[str, Any]],
        provider_factory,
        cancellation_handler,
        signal_handler,
    ):
        """
        Initialize the validation component.

        Args:
            console: Rich console instance
            provider_info: Dictionary of provider information
            provider_factory: Factory for creating provider instances
            cancellation_handler: Handler for cancellation support
            signal_handler: Handler for signal management
        """
        self.console = console
        self.provider_info = provider_info
        self.provider_factory = provider_factory
        self.cancellation_handler = cancellation_handler
        self.signal_handler = signal_handler
        self.ui = SetupWizardUI(console, provider_info)

    async def test_configuration(
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
        self.console.print(
            f"\n[bold]Testing {self.provider_info[provider_name]['name']} configuration...[/bold]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Step 1: Validate API key format
            task = progress.add_task("Validating API key format...", total=None)

            if not self._validate_api_key_format(
                provider_name, provider_config.api_key
            ):
                progress.update(task, description="❌ Invalid API key format!")
                await asyncio.sleep(0.5)
                progress.stop()  # Stop the progress spinner before user interaction
                self.console.print("[red]❌ API key format validation failed![/red]")
                self.ui.show_troubleshooting_guidance(
                    provider_name, "Invalid API key format"
                )
                return Confirm.ask("Continue with this configuration anyway?")

            progress.update(task, description="✅ API key format valid!")
            await asyncio.sleep(0.5)

            # Step 2: Test API connection
            progress.update(task, description="Testing API connection...")

            try:
                # Create provider instance for testing
                provider = self.provider_factory.create_provider(
                    provider_name, provider_config
                )

                # Test credentials
                is_valid = await self._validate_provider_credentials(provider)

                if is_valid:
                    progress.update(task, description="✅ API connection successful!")
                    await asyncio.sleep(0.5)

                    # Step 3: Test model availability
                    progress.update(task, description="Checking model availability...")

                    # Special handling for Bedrock to test model access
                    if provider_name == "bedrock" and hasattr(
                        provider, "validate_model_access"
                    ):
                        model_test = await provider.validate_model_access()

                        if model_test["success"]:
                            progress.update(task, description="✅ Model accessible!")
                            await asyncio.sleep(0.5)
                            self.console.print(
                                "[green]✅ All configuration tests passed![/green]"
                            )
                            return True
                        else:
                            progress.update(task, description="⚠️  Model access issue!")
                            await asyncio.sleep(0.5)
                            progress.stop()  # Stop the progress spinner before showing warning

                            # Show detailed model access warning
                            self.console.print(
                                f"[yellow]⚠️  Model Access Warning: {model_test['message']}[/yellow]"
                            )
                            if "suggestion" in model_test:
                                self.console.print(
                                    f"[dim]💡 Suggestion: {model_test['suggestion']}[/dim]"
                                )

                            self.console.print(
                                "\n[yellow]Note: Configuration will be saved, but you may need to:[/yellow]"
                            )
                            self.console.print(
                                "  [yellow]• Request model access in AWS Console[/yellow]"
                            )
                            self.console.print(
                                "  [yellow]• Check account permissions[/yellow]"
                            )
                            self.console.print(
                                "  [yellow]• Try a different model[/yellow]"
                            )

                            return Confirm.ask(
                                "Continue with this configuration?",
                                default=True,
                            )
                    else:
                        # Standard model availability check for other providers
                        model_available = await self._check_model_availability(
                            provider, provider_config.model
                        )

                        if model_available:
                            progress.update(task, description="✅ Model available!")
                            await asyncio.sleep(0.5)
                            self.console.print(
                                "[green]✅ All configuration tests passed![/green]"
                            )
                            return True
                        else:
                            progress.update(
                                task,
                                description="⚠️  Model may not be available!",
                            )
                            await asyncio.sleep(0.5)
                            progress.stop()  # Stop the progress spinner before user interaction
                            self.console.print(
                                "[yellow]⚠️  Model availability could not be confirmed[/yellow]"
                            )
                            return Confirm.ask("Continue with this configuration?")
                else:
                    progress.update(task, description="❌ API authentication failed!")
                    await asyncio.sleep(0.5)
                    progress.stop()  # Stop the progress spinner before showing guidance
                    self.console.print("[red]❌ API authentication failed![/red]")
                    self.ui.show_troubleshooting_guidance(
                        provider_name, "Authentication failed"
                    )
                    return Confirm.ask("Continue with this configuration anyway?")

            except Exception as e:
                progress.update(task, description="❌ Connection test failed!")
                await asyncio.sleep(0.5)
                progress.stop()  # Stop the progress spinner before showing guidance

                self.console.print(f"[red]❌ Configuration test failed: {e}[/red]")

                # Provide troubleshooting guidance
                self.ui.show_troubleshooting_guidance(provider_name, str(e))

                return Confirm.ask("Continue with this configuration anyway?")

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
        if provider_name in ["ollama", "claude-code"]:
            return True  # Ollama and Claude Code don't require API key

        if not api_key:
            return False

        info = self.provider_info.get(provider_name, {})
        prefix = info.get("api_key_prefix", "")

        # Basic format validation
        if prefix and not api_key.startswith(prefix):
            return False

        # Provider-specific validation
        if provider_name == "claude":
            return api_key.startswith("sk-ant-") and len(api_key) > 20
        elif provider_name == "openai":
            return api_key.startswith("sk-") and len(api_key) > 20
        elif provider_name == "gemini":
            return api_key.startswith("AIza") and len(api_key) > 20
        elif provider_name == "perplexity":
            return api_key.startswith("pplx-") and len(api_key) > 20
        elif provider_name == "xai":
            return api_key.startswith("xai-") and len(api_key) > 20
        elif provider_name == "bedrock":
            return api_key.startswith("ABSKQmVkcm9ja0FQSUtleS1") and len(api_key) > 50
        elif provider_name == "openrouter":
            return api_key.startswith("sk-or-") and len(api_key) > 10
        elif provider_name == "azure":
            # Azure uses the same key format as OpenAI
            return len(api_key) == 32  # Azure keys are typically 32 characters
        elif provider_name == "vertex":
            # Vertex AI uses service account JSON or ADC, not API keys
            # If an API key is provided, it's likely a service account key JSON
            return len(api_key) > 100 or api_key.startswith("{")
        elif provider_name == "mistral":
            return len(api_key) > 20  # Mistral keys don't have consistent prefix
        elif provider_name == "cohere":
            return len(api_key) > 20  # Cohere keys don't have consistent prefix

        return len(api_key) > 10  # Generic validation

    async def _validate_provider_credentials(self, provider) -> bool:
        """
        Validate provider credentials by making a test API call.

        Args:
            provider: Provider instance to test

        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            # Use the provider's validate_credentials method with cancellation support
            async def credential_validation():
                return await provider.validate_credentials()

            return await self.cancellation_handler.start_cancellable_operation(
                operation=credential_validation,
                status_message="Testing API credentials",
                cancellation_message="Credential validation cancelled",
                signal_handler=self.signal_handler,
            )
        except asyncio.CancelledError:
            # Operation was cancelled by user
            return False
        except Exception:
            # Log the error but don't raise it
            return False

    async def _check_model_availability(self, provider, model_name: str) -> bool:
        """
        Check if a specific model is available for the provider.

        Args:
            provider: Provider instance
            model_name: Name of the model to check

        Returns:
            True if model is available, False otherwise
        """
        # Some providers have dynamic or user-defined model names that can't be validated
        # against a static list, so we skip validation for them
        provider_class_name = provider.__class__.__name__ if hasattr(provider, '__class__') else ""
        
        # OpenRouter supports hundreds of models, any model name is potentially valid
        if provider_class_name == 'OpenRouterProvider':
            return True
            
        # Azure uses deployment names which are user-defined
        if provider_class_name == 'AzureProvider':
            return True
            
        # Bedrock model names can vary by region and account permissions
        if provider_class_name == 'BedrockProvider':
            return True
            
        # Vertex AI model names can be custom or region-specific
        if provider_class_name == 'VertexProvider':
            return True
            
        # Ollama models are locally installed, validation happens at runtime
        if provider_class_name == 'OllamaProvider':
            return True
            
        try:
            # Try to get available models with cancellation support
            async def model_availability_check():
                available_models = provider.get_available_models()
                model_names = [model.name for model in available_models]
                return model_name in model_names

            return await self.cancellation_handler.start_cancellable_operation(
                operation=model_availability_check,
                status_message="Checking model availability",
                cancellation_message="Model availability check cancelled",
                signal_handler=self.signal_handler,
            )
        except asyncio.CancelledError:
            # Operation was cancelled by user
            return True  # Assume available to allow setup to continue
        except Exception:
            # If we can't check model availability, assume it's available
            # This prevents blocking setup for providers that don't support model listing
            return True

    async def validate_existing_configuration(
        self, provider_name: str, config_manager
    ) -> Dict[str, Any]:
        """
        Validate an existing provider configuration.

        Args:
            provider_name: Name of the provider to validate
            config_manager: Configuration manager instance

        Returns:
            Dictionary with validation results
        """
        result = {
            "provider": provider_name,
            "valid": False,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }

        try:
            # Get existing configuration
            provider_config = config_manager.get_provider_config(provider_name)
            if not provider_config:
                result["errors"].append(
                    f"No configuration found for provider '{provider_name}'"
                )
                return result

            # Validate API key format
            if not self._validate_api_key_format(
                provider_name, provider_config.api_key
            ):
                result["errors"].append("Invalid API key format")
                result["suggestions"].append(
                    f"Check API key format for {self.provider_info[provider_name]['name']}"
                )

            # Test credentials
            try:
                provider = self.provider_factory.create_provider(
                    provider_name, provider_config.model_dump(mode="json")
                )

                credentials_valid = await self._validate_provider_credentials(provider)
                if not credentials_valid:
                    result["errors"].append("API credentials validation failed")
                    result["suggestions"].append("Verify API key and account status")
                else:
                    # Check model availability
                    model_available = await self._check_model_availability(
                        provider, provider_config.model
                    )
                    if not model_available:
                        result["warnings"].append(
                            f"Model '{provider_config.model}' may not be available"
                        )
                        result["suggestions"].append(
                            "Consider updating to a newer model"
                        )

            except Exception as e:
                result["errors"].append(f"Provider initialization failed: {str(e)}")
                result["suggestions"].append(
                    "Check provider configuration and network connectivity"
                )

            # Set overall validity
            result["valid"] = len(result["errors"]) == 0

        except Exception as e:
            result["errors"].append(f"Configuration validation failed: {str(e)}")

        return result
