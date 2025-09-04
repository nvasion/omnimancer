"""
Core orchestration methods for the setup wizard.

This module contains the main orchestration logic for coordinating
the setup wizard workflow.
"""

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.prompt import Confirm

from ..core.signal_handler import SignalHandler
from ..providers.factory import ProviderFactory
from ..ui.cancellation_handler import CancellationHandler
from .config_manager import ConfigManager
from .provider_registry import ProviderRegistry


class SetupWizardCore:
    """Core orchestration methods for the setup wizard."""

    def __init__(
        self,
        config_manager: ConfigManager,
        provider_registry: ProviderRegistry,
    ):
        """
        Initialize the setup wizard core.

        Args:
            config_manager: Configuration manager instance
            provider_registry: Provider registry instance
        """
        self.config_manager = config_manager
        self.provider_registry = provider_registry
        self.console = Console()
        self.provider_factory = ProviderFactory()

        # Initialize cancellation support
        self.cancellation_handler = CancellationHandler(self.console)
        self.signal_handler = SignalHandler()

        # Provider descriptions and strengths
        self.provider_info = {
            "claude": {
                "name": "Anthropic Claude",
                "description": "Advanced reasoning and analysis with strong safety features",
                "strengths": [
                    "Code analysis",
                    "Complex reasoning",
                    "Safety-focused",
                    "Long conversations",
                ],
                "models": [
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                    "claude-3-5-sonnet-20241022",
                ],
                "api_key_url": "https://console.anthropic.com/",
                "api_key_prefix": "sk-ant-",
                "setup_notes": "Get your API key from the Anthropic Console",
            },
            "openai": {
                "name": "OpenAI GPT",
                "description": "Versatile models with strong general capabilities",
                "strengths": [
                    "General purpose",
                    "Code generation",
                    "Creative writing",
                    "Tool usage",
                ],
                "models": [
                    "gpt-4o",
                    "o1",
                    "o3",
                    "o3-mini",
                    "o4-mini",
                    "o1-mini",
                    "o1-pro",
                    "gpt-4o-mini",
                ],
                "api_key_url": "https://platform.openai.com/api-keys",
                "api_key_prefix": "sk-",
                "setup_notes": "Get your API key from the OpenAI Platform",
            },
            "gemini": {
                "name": "Google Gemini",
                "description": "Multimodal AI with strong integration capabilities",
                "strengths": [
                    "Multimodal",
                    "Large context",
                    "Fast responses",
                    "Google integration",
                ],
                "models": [
                    "gemini-2.5-pro-preview-05-06",
                    "gemini-2.5-flash-preview-04-17",
                    "gemini-2.0-flash",
                    "gemini-1.5-pro",
                ],
                "api_key_url": "https://makersuite.google.com/app/apikey",
                "api_key_prefix": "AIza",
                "setup_notes": "Get your API key from Google AI Studio",
            },
            "perplexity": {
                "name": "Perplexity AI",
                "description": "Search-enhanced AI with real-time information access",
                "strengths": [
                    "Web search",
                    "Real-time info",
                    "Research",
                    "Current events",
                ],
                "models": [
                    "sonar-pro",
                    "sonar",
                    "deep-research",
                    "sonar-reasoning-pro",
                    "sonar-reasoning",
                ],
                "api_key_url": "https://www.perplexity.ai/settings/api",
                "api_key_prefix": "pplx-",
                "setup_notes": "Get your API key from Perplexity Settings",
            },
            "xai": {
                "name": "xAI Grok",
                "description": "Advanced reasoning with real-time information",
                "strengths": [
                    "Real-time data",
                    "Reasoning",
                    "Multimodal",
                    "Tool usage",
                ],
                "models": ["grok-3", "grok-3-fast", "grok-4"],
                "api_key_url": "https://console.x.ai/",
                "api_key_prefix": "xai-",
                "setup_notes": "Get your API key from the xAI Console",
            },
            "mistral": {
                "name": "Mistral AI",
                "description": "Efficient European AI with strong performance",
                "strengths": [
                    "Efficiency",
                    "Multilingual",
                    "Code generation",
                    "Privacy-focused",
                ],
                "models": [
                    "mistral-small-3.1",
                    "mistral-medium",
                    "mistral-large",
                ],
                "api_key_url": "https://console.mistral.ai/",
                "api_key_prefix": "",
                "setup_notes": "Get your API key from the Mistral Console",
            },
            "cohere": {
                "name": "Cohere",
                "description": "Enterprise-focused AI with strong text generation",
                "strengths": [
                    "Enterprise features",
                    "Text generation",
                    "Embeddings",
                    "Classification",
                ],
                "models": ["command-r", "command-r-plus", "command-light"],
                "api_key_url": "https://dashboard.cohere.ai/api-keys",
                "api_key_prefix": "",
                "setup_notes": "Get your API key from the Cohere Dashboard",
            },
            "ollama": {
                "name": "Ollama (Local)",
                "description": "Run models locally on your machine",
                "strengths": [
                    "Privacy",
                    "No API costs",
                    "Offline usage",
                    "Custom models",
                ],
                "models": [
                    "llama3.3:latest",
                    "qwen3:latest",
                    "mistral-small3.1:latest",
                    "phi4:latest",
                    "devstral:latest",
                ],
                "api_key_url": "https://ollama.ai/",
                "api_key_prefix": "",
                "setup_notes": "Install Ollama locally - no API key required",
            },
            "azure": {
                "name": "Azure OpenAI",
                "description": "OpenAI models through Microsoft Azure cloud platform",
                "strengths": [
                    "Enterprise security",
                    "Azure integration",
                    "Compliance",
                    "Scalability",
                ],
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-1"],
                "api_key_url": "https://portal.azure.com/",
                "api_key_prefix": "",
                "setup_notes": "Requires Azure subscription and OpenAI service deployment",
            },
            "vertex": {
                "name": "Google Vertex AI",
                "description": "Google's enterprise AI platform with Gemini models",
                "strengths": [
                    "Enterprise features",
                    "Google Cloud integration",
                    "MLOps",
                    "Custom models",
                ],
                "models": ["gemini-1.5-pro", "gemini-1.5-flash", "text-bison"],
                "api_key_url": "https://console.cloud.google.com/",
                "api_key_prefix": "",
                "setup_notes": "Requires Google Cloud project and service account credentials",
            },
            "bedrock": {
                "name": "AWS Bedrock",
                "description": "Amazon's managed AI service with multiple model providers",
                "strengths": [
                    "AWS integration",
                    "Multiple models",
                    "Enterprise security",
                    "Serverless",
                ],
                "models": [
                    "arn:aws:bedrock:us-west-2::foundation-model/meta.llama4-maverick-17b-instruct-v1:0",
                    "arn:aws:bedrock:us-west-2::inference-profile/us.meta.llama4-maverick-17b-instruct-v1:0",
                    "anthropic.claude-3-sonnet-20240229-v1:0",
                    "anthropic.claude-3-haiku-20240307-v1:0",
                    "meta.llama3-1-8b-instruct-v1:0",
                ],
                "api_key_url": "https://console.aws.amazon.com/bedrock/home",
                "api_key_prefix": "ABSKQmVkcm9ja0FQSUtleS1",
                "setup_notes": "Get a 30-day API key from the AWS Bedrock Console. You can enter custom model IDs or ARNs during setup.",
            },
            "openrouter": {
                "name": "OpenRouter",
                "description": "Access to multiple AI models through a single API",
                "strengths": [
                    "Model variety",
                    "Cost optimization",
                    "Unified API",
                    "Real-time switching",
                ],
                "models": [
                    "anthropic/claude-3.5-sonnet",
                    "openai/gpt-4o",
                    "google/gemini-pro-1.5",
                    "meta-llama/llama-3.1-405b-instruct",
                    "qwen/qwen-2.5-72b-instruct",
                ],
                "api_key_url": "https://openrouter.ai/keys",
                "api_key_prefix": "sk-or-",
                "setup_notes": "Get your API key from OpenRouter dashboard",
            },
            "claude-code": {
                "name": "Claude Code (Local)",
                "description": "Free local Claude integration for development",
                "strengths": [
                    "Free usage",
                    "Local processing",
                    "Development focus",
                    "No API limits",
                ],
                "models": ["claude-code-opus", "claude-code-sonnet"],
                "api_key_url": "",
                "api_key_prefix": "",
                "setup_notes": "No API key required - uses your existing Claude Code authentication",
            },
        }

    async def start_wizard(self) -> bool:
        """
        Start the interactive setup process.

        Returns:
            True if setup completed successfully, False otherwise
        """
        try:
            from .setup_wizard_provider_setup import SetupWizardProviderSetup
            from .setup_wizard_ui import SetupWizardUI
            from .setup_wizard_validation import SetupWizardValidation

            # Create helper instances
            ui = SetupWizardUI(self.console, self.provider_info)
            provider_setup = SetupWizardProviderSetup(self.console, self.provider_info)
            validation = SetupWizardValidation(
                self.console,
                self.provider_info,
                self.provider_factory,
                self.cancellation_handler,
                self.signal_handler,
            )

            ui.show_welcome()

            # Check if this is first run
            if not self.config_manager.is_first_run():
                if not self._confirm_overwrite():
                    return False

            # Select provider
            provider_name = provider_setup.select_provider()
            if not provider_name:
                return False

            # Configure the selected provider
            provider_config = await provider_setup.configure_provider(provider_name)
            if not provider_config:
                return False

            # Test the configuration
            if not await validation.test_configuration(provider_name, provider_config):
                return False

            # Save configuration
            provider_setup.save_configuration(
                provider_name, provider_config, self.config_manager
            )

            # Show completion message
            ui.show_completion(provider_name)

            return True

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Setup cancelled by user[/yellow]")
            return False
        except Exception as e:
            self.console.print(f"\n[red]Setup failed: {e}[/red]")
            return False

    def _confirm_overwrite(self) -> bool:
        """
        Confirm if user wants to add another provider to existing configuration.

        Returns:
            True if user confirms, False otherwise
        """
        self.console.print("[yellow]⚠️  Existing configuration detected[/yellow]")
        return Confirm.ask("Do you want to continue and add another provider?")

    def get_available_providers(self) -> List[str]:
        """
        Get list of available providers for setup.

        Returns:
            List of provider names
        """
        return list(self.provider_info.keys())

    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider information dictionary or None if not found
        """
        return self.provider_info.get(provider_name)

    def get_common_setup_issues(self, provider_name: str) -> List[Dict[str, str]]:
        """
        Get common setup issues and solutions for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            List of issue dictionaries with 'issue' and 'solution' keys
        """
        common_issues = {
            "claude": [
                {
                    "issue": "API key authentication fails",
                    "solution": "Ensure your API key starts with 'sk-ant-' and is from console.anthropic.com",
                },
                {
                    "issue": "Rate limit errors",
                    "solution": "Check your usage limits in the Anthropic Console and consider upgrading your plan",
                },
                {
                    "issue": "Model not found",
                    "solution": "Verify the model name matches exactly (e.g., 'claude-3-5-sonnet-20241022')",
                },
            ],
            "openai": [
                {
                    "issue": "API key authentication fails",
                    "solution": "Ensure your API key starts with 'sk-' and is from platform.openai.com",
                },
                {
                    "issue": "Insufficient quota",
                    "solution": "Add billing information to your OpenAI account or check usage limits",
                },
                {
                    "issue": "Model access denied",
                    "solution": "Some models require special access - check your account permissions",
                },
            ],
            "gemini": [
                {
                    "issue": "API key authentication fails",
                    "solution": "Ensure your API key starts with 'AIza' and is from Google AI Studio",
                },
                {
                    "issue": "Safety filter blocks",
                    "solution": "Adjust safety settings or rephrase your input to avoid triggering filters",
                },
                {
                    "issue": "Quota exceeded",
                    "solution": "Check your usage in Google AI Studio and consider upgrading limits",
                },
            ],
            "ollama": [
                {
                    "issue": "Connection refused",
                    "solution": "Start Ollama server with 'ollama serve' and ensure it's running on the correct port",
                },
                {
                    "issue": "Model not found",
                    "solution": "Install the model with 'ollama pull <model-name>' before using it",
                },
                {
                    "issue": "Slow responses",
                    "solution": "Ensure sufficient RAM and consider using smaller models for better performance",
                },
            ],
        }

        return common_issues.get(
            provider_name,
            [
                {
                    "issue": "API authentication fails",
                    "solution": "Verify your API key is correct and your account has sufficient credits",
                },
                {
                    "issue": "Network connection issues",
                    "solution": "Check your internet connection and firewall settings",
                },
            ],
        )
