"""
Provider setup methods for the setup wizard.

This module contains all provider configuration and setup methods,
including provider selection, API key management, and model selection.
"""

from typing import Any, Dict, Optional

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .models import ProviderConfig
from .setup_wizard_ui import SetupWizardUI


class SetupWizardProviderSetup:
    """Provider setup methods for the setup wizard."""

    def __init__(self, console: Console, provider_info: Dict[str, Dict[str, Any]]):
        """
        Initialize the provider setup component.

        Args:
            console: Rich console instance
            provider_info: Dictionary of provider information
        """
        self.console = console
        self.provider_info = provider_info
        self.ui = SetupWizardUI(console, provider_info)

    def select_provider(self) -> Optional[str]:
        """
        Interactive provider selection with descriptions.

        Returns:
            Selected provider name or None if cancelled
        """
        self.console.print("[bold]Available AI Providers:[/bold]\n")

        # Create provider selection table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Provider", style="bold")
        table.add_column("Description", style="dim")
        table.add_column("Best For", style="green")

        providers = list(self.provider_info.keys())
        for i, provider_key in enumerate(providers, 1):
            info = self.provider_info[provider_key]
            strengths = ", ".join(info["strengths"][:2])  # Show first 2 strengths
            table.add_row(str(i), info["name"], info["description"], strengths)

        self.console.print(table)
        self.console.print()

        # Get user selection
        while True:
            try:
                choice = Prompt.ask(
                    "Select a provider",
                    choices=[str(i) for i in range(1, len(providers) + 1)] + ["q"],
                    default="1",
                )

                if choice.lower() == "q":
                    return None

                provider_index = int(choice) - 1
                selected_provider = providers[provider_index]

                # Show detailed provider information
                self.ui.show_provider_details(selected_provider)

                if Confirm.ask(
                    f"Configure {self.provider_info[selected_provider]['name']}?"
                ):
                    return selected_provider

            except (ValueError, IndexError):
                self.console.print("[red]Invalid selection. Please try again.[/red]")

    async def configure_provider(self, provider_name: str) -> Optional[ProviderConfig]:
        """
        Configure a specific provider with validation.

        Args:
            provider_name: Name of the provider to configure

        Returns:
            Configured ProviderConfig or None if failed
        """
        info = self.provider_info[provider_name]

        self.console.print(f"\n[bold]Configuring {info['name']}[/bold]")
        self.console.print(f"[dim]{info['setup_notes']}[/dim]")
        self.console.print(f"[dim]Get your API key from: {info['api_key_url']}[/dim]\n")

        # Get API key (if required)
        api_key = None
        if provider_name not in [
            "ollama",
            "claude-code",
        ]:  # Ollama and Claude Code don't need API key
            api_key = self._get_api_key(provider_name, info)
            if not api_key:
                return None
        elif provider_name == "claude-code":
            # Claude Code uses local authentication, set a placeholder
            api_key = "local"

        # Select model
        model = self._select_model(provider_name, info)
        if not model:
            return None

        # Get additional settings
        additional_settings = self._get_additional_settings(provider_name)

        # Create provider config
        config_data = {
            "api_key": api_key,
            "model": model,
            **additional_settings,
        }

        return ProviderConfig(**config_data)

    def _get_api_key(self, provider_name: str, info: Dict[str, Any]) -> Optional[str]:
        """
        Get and validate API key from user.

        Args:
            provider_name: Name of the provider
            info: Provider information dictionary

        Returns:
            Valid API key or None if cancelled
        """
        while True:
            api_key = Prompt.ask(f"Enter your {info['name']} API key", password=True)

            if not api_key:
                if not Confirm.ask("API key is required. Continue without it?"):
                    continue
                return None

            # Basic format validation
            if info["api_key_prefix"] and not api_key.startswith(
                info["api_key_prefix"]
            ):
                self.console.print(
                    f"[yellow]⚠️  API key should start with '{info['api_key_prefix']}'[/yellow]"
                )
                if not Confirm.ask("Continue with this API key anyway?"):
                    continue

            return api_key

    def _select_model(self, provider_name: str, info: Dict[str, Any]) -> Optional[str]:
        """
        Select model for the provider.

        Args:
            provider_name: Name of the provider
            info: Provider information dictionary

        Returns:
            Selected model name or None if cancelled
        """
        models = info["models"]

        # Allow custom model IDs for all providers
        self.console.print(f"\n[bold]Model options for {info['name']}:[/bold]")
        self.console.print("You can either:")
        self.console.print("  1. Select from common models")
        self.console.print("  2. Enter a custom model ID")
        self.console.print()

        choice = Prompt.ask("Choose option", choices=["1", "2"], default="1")

        if choice == "2":
            self.console.print(
                f"\n[bold cyan]Custom Model ID Entry for {info['name']}:[/bold cyan]"
            )

            # Provider-specific examples
            if provider_name == "bedrock":
                self.console.print("Examples:")
                self.console.print(
                    "  • Model ID: [dim]anthropic.claude-opus-4-20250514-v1:0[/dim]"
                )
                self.console.print(
                    "  • Inference Profile: [dim]us.anthropic.claude-opus-4-20250514-v1:0[/dim]"
                )
                self.console.print(
                    "  • Foundation ARN: [dim]arn:aws:bedrock:us-west-2::foundation-model/meta.llama4-maverick-17b-instruct-v1:0[/dim]"
                )
                self.console.print(
                    "  • Inference Profile ARN: [dim]arn:aws:bedrock:us-west-2::inference-profile/us.meta.llama4-maverick-17b-instruct-v1:0[/dim]"
                )
            elif provider_name == "openai":
                self.console.print("Examples:")
                self.console.print("  • [dim]gpt-4o[/dim]")
                self.console.print("  • [dim]gpt-4o-mini[/dim]")
                self.console.print("  • [dim]o1-preview[/dim]")
                self.console.print("  • [dim]gpt-4-turbo[/dim]")
            elif provider_name == "claude":
                self.console.print("Examples:")
                self.console.print("  • [dim]claude-sonnet-4-20250514[/dim]")
                self.console.print("  • [dim]claude-opus-4-20250514[/dim]")
                self.console.print("  • [dim]claude-3-5-sonnet-20241022[/dim]")
            elif provider_name == "gemini":
                self.console.print("Examples:")
                self.console.print("  • [dim]gemini-1.5-pro[/dim]")
                self.console.print("  • [dim]gemini-1.5-flash[/dim]")
                self.console.print("  • [dim]gemini-2.0-flash-exp[/dim]")
            elif provider_name == "openrouter":
                self.console.print("Examples:")
                self.console.print("  • [dim]anthropic/claude-3.5-sonnet[/dim]")
                self.console.print("  • [dim]openai/gpt-4o[/dim]")
                self.console.print("  • [dim]meta-llama/llama-3.1-70b-instruct[/dim]")
            elif provider_name == "ollama":
                self.console.print("Examples:")
                self.console.print("  • [dim]llama3:8b[/dim]")
                self.console.print("  • [dim]mistral:7b[/dim]")
                self.console.print("  • [dim]codellama:13b[/dim]")
            elif provider_name == "perplexity":
                self.console.print("Examples:")
                self.console.print("  • [dim]llama-3.1-sonar-large-128k-online[/dim]")
                self.console.print("  • [dim]llama-3.1-sonar-small-128k-online[/dim]")
            elif provider_name == "xai":
                self.console.print("Examples:")
                self.console.print("  • [dim]grok-beta[/dim]")
                self.console.print("  • [dim]grok-vision-beta[/dim]")
            elif provider_name == "mistral":
                self.console.print("Examples:")
                self.console.print("  • [dim]mistral-large-latest[/dim]")
                self.console.print("  • [dim]mistral-small-latest[/dim]")
                self.console.print("  • [dim]open-mistral-7b[/dim]")
            elif provider_name == "cohere":
                self.console.print("Examples:")
                self.console.print("  • [dim]command-r[/dim]")
                self.console.print("  • [dim]command-r-plus[/dim]")
                self.console.print("  • [dim]command[/dim]")
            elif provider_name == "azure":
                self.console.print("Examples:")
                self.console.print("  • [dim]gpt-4[/dim] (your deployment name)")
                self.console.print("  • [dim]gpt-35-turbo[/dim] (your deployment name)")
                self.console.print(
                    "  • [dim]my-custom-model[/dim] (your deployment name)"
                )
            elif provider_name == "vertex":
                self.console.print("Examples:")
                self.console.print("  • [dim]gemini-1.5-pro[/dim]")
                self.console.print("  • [dim]gemini-1.5-flash[/dim]")
                self.console.print("  • [dim]gemini-pro[/dim]")
            elif provider_name == "claude-code":
                self.console.print("Examples:")
                self.console.print("  • [dim]claude-code-sonnet[/dim]")
                self.console.print("  • [dim]claude-code-opus[/dim]")
                self.console.print("  • [dim]claude-code-haiku[/dim]")

            self.console.print()

            custom_model = Prompt.ask("Enter custom model ID")
            if custom_model:
                return custom_model
            else:
                self.console.print(
                    "[yellow]No model entered, falling back to common models...[/yellow]"
                )

        self.console.print(f"\n[bold]Common models for {info['name']}:[/bold]")

        # Create model selection table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Model", style="bold")
        table.add_column("Description", style="dim")

        # Get model descriptions (simplified for now)
        model_descriptions = {
            "claude-3-5-sonnet-20241022": "Latest Claude model with enhanced capabilities",
            "claude-3-haiku-20240307": "Fast and efficient Claude model",
            "claude-3-opus-20240229": "Most capable Claude model",
            "gpt-4o": "Latest GPT-4 with vision and tool use",
            "gpt-4o-mini": "Efficient GPT-4 variant",
            "gpt-4-turbo": "Enhanced GPT-4 with larger context",
            "gemini-1.5-pro": "Advanced Gemini with large context",
            "gemini-1.5-flash": "Fast Gemini variant",
            "sonar-pro": "Advanced search-enabled model",
            "grok-3": "Latest Grok model",
            "mistral-small-3.1": "Efficient Mistral model",
            "command-r": "Cohere's reasoning model",
            "llama3.2": "Meta's latest Llama model",
            # Bedrock specific models
            "anthropic.claude-3-sonnet-20240229-v1:0": "Claude 3 Sonnet on Bedrock",
            "anthropic.claude-3-haiku-20240307-v1:0": "Claude 3 Haiku on Bedrock",
            "meta.llama3-1-8b-instruct-v1:0": "Llama 3.1 8B on Bedrock",
            "arn:aws:bedrock:us-west-2::foundation-model/meta.llama4-maverick-17b-instruct-v1:0": "Meta Llama 4 Maverick (ARN)",
        }

        for i, model in enumerate(models, 1):
            description = model_descriptions.get(model, "High-quality AI model")
            table.add_row(str(i), model, description)

        # Add custom option for non-Bedrock providers
        if provider_name != "bedrock":
            table.add_row(str(len(models) + 1), "[Custom]", "Enter custom model name")

        self.console.print(table)

        # Get user selection
        while True:
            try:
                max_choice = len(models) + (1 if provider_name != "bedrock" else 0)
                choice = Prompt.ask(
                    "Select a model",
                    choices=[str(i) for i in range(1, max_choice + 1)],
                    default="1",
                )

                choice_num = int(choice)

                # Handle custom option for non-Bedrock providers
                if provider_name != "bedrock" and choice_num == len(models) + 1:
                    custom_model = Prompt.ask("Enter custom model name")
                    if custom_model:
                        return custom_model
                    else:
                        continue

                model_index = choice_num - 1
                return models[model_index]

            except (ValueError, IndexError):
                self.console.print("[red]Invalid selection. Please try again.[/red]")

    def _get_additional_settings(self, provider_name: str) -> Dict[str, Any]:
        """
        Get additional provider-specific settings.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary of additional settings
        """
        settings = {}

        # Common settings
        if Confirm.ask("Configure advanced settings?", default=False):
            # Temperature
            temp_str = Prompt.ask("Temperature (0.0-1.0)", default="0.7")
            try:
                settings["temperature"] = float(temp_str)
            except ValueError:
                settings["temperature"] = 0.7

            # Max tokens
            tokens_str = Prompt.ask("Max tokens", default="4096")
            try:
                settings["max_tokens"] = int(tokens_str)
            except ValueError:
                settings["max_tokens"] = 4096

        # Provider-specific settings
        if provider_name == "ollama":
            base_url = Prompt.ask("Ollama server URL", default="http://localhost:11434")
            settings["base_url"] = base_url
        elif provider_name == "bedrock":
            # AWS region selection is required for Bedrock
            regions = [
                "us-east-1",
                "us-west-2",
                "eu-west-1",
                "eu-central-1",
                "ap-southeast-1",
                "ap-northeast-1",
                "ap-south-1",
            ]

            self.console.print("\n[bold]Available AWS Regions:[/bold]")
            for i, region in enumerate(regions, 1):
                self.console.print(f"  {i}. {region}")

            while True:
                region_choice = Prompt.ask(
                    f"Select AWS region [1-{len(regions)}]", default="1"
                )
                try:
                    region_idx = int(region_choice) - 1
                    if 0 <= region_idx < len(regions):
                        settings["aws_region"] = regions[region_idx]
                        break
                    else:
                        self.console.print(
                            "[red]Invalid selection. Please try again.[/red]"
                        )
                except ValueError:
                    self.console.print("[red]Please enter a number.[/red]")

        return settings

    def save_configuration(
        self,
        provider_name: str,
        provider_config: ProviderConfig,
        config_manager,
    ) -> None:
        """
        Save the configuration, adding the provider without overwriting existing ones.

        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration to save
            config_manager: Configuration manager instance
        """
        try:
            # Check if this is the first provider being configured
            config = config_manager.get_config()
            is_first_provider = not config.providers

            # Add or update the provider configuration
            config_manager.set_provider_config(provider_name, provider_config)

            # Set as default provider if it's the first one or if user confirms
            if is_first_provider:
                config_manager.set_default_provider(provider_name)
                self.console.print(
                    f"[green]✅ {self.provider_info[provider_name]['name']} set as default provider![/green]"
                )
            else:
                # Ask if user wants to set this as the default provider
                if Confirm.ask(
                    f"Set {self.provider_info[provider_name]['name']} as your default provider?",
                    default=False,
                ):
                    config_manager.set_default_provider(provider_name)
                    self.console.print(
                        f"[green]✅ {self.provider_info[provider_name]['name']} set as default provider![/green]"
                    )
                else:
                    self.console.print(
                        f"[green]✅ {self.provider_info[provider_name]['name']} added to your providers![/green]"
                    )

            self.console.print("[green]✅ Configuration saved successfully![/green]")

        except Exception as e:
            self.console.print(f"[red]❌ Failed to save configuration: {e}[/red]")
            raise
