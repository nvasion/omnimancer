"""
UI components for the setup wizard.

This module contains all UI and display methods for the setup wizard,
including welcome screens, provider details, and completion messages.
"""

from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class SetupWizardUI:
    """UI components for the setup wizard."""

    def __init__(self, console: Console, provider_info: Dict[str, Dict[str, Any]]):
        """
        Initialize the UI component.

        Args:
            console: Rich console instance
            provider_info: Dictionary of provider information
        """
        self.console = console
        self.provider_info = provider_info

    def show_welcome(self) -> None:
        """Show welcome message and setup introduction."""
        welcome_text = Text()
        welcome_text.append("Welcome to Omnimancer Setup!\n\n", style="bold blue")
        welcome_text.append(
            "This wizard will help you add AI providers to your configuration.\n"
        )
        welcome_text.append(
            "You can run this anytime to add more providers without affecting existing ones.\n"
        )

        panel = Panel(
            welcome_text,
            title="🚀 Omnimancer Setup Wizard",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def show_provider_details(self, provider_name: str) -> None:
        """
        Show detailed information about a provider.

        Args:
            provider_name: Name of the provider
        """
        info = self.provider_info[provider_name]

        details = Text()
        details.append(f"{info['name']}\n", style="bold blue")
        details.append(f"{info['description']}\n\n", style="dim")
        details.append("Strengths:\n", style="bold")
        for strength in info["strengths"]:
            details.append(f"  • {strength}\n", style="green")
        details.append(
            f"\nAvailable Models: {', '.join(info['models'][:3])}\n",
            style="dim",
        )

        panel = Panel(
            details,
            title=f"📋 {info['name']} Details",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)

    def show_completion(self, provider_name: str) -> None:
        """
        Show setup completion message with next steps.

        Args:
            provider_name: Name of the configured provider
        """
        info = self.provider_info[provider_name]

        completion_text = Text()
        completion_text.append("🎉 Setup Complete!\n\n", style="bold green")
        completion_text.append(f"You've successfully configured {info['name']}.\n\n")
        completion_text.append("Next Steps:\n", style="bold")
        completion_text.append("  • Start chatting by typing your message\n")
        completion_text.append("  • Use /help to see available commands\n")
        completion_text.append("  • Use /providers to add more AI providers\n")
        completion_text.append("  • Use /models to see available models\n\n")
        completion_text.append("Happy chatting! 🚀", style="bold blue")

        panel = Panel(
            completion_text,
            title="✨ Welcome to Omnimancer!",
            border_style="green",
            padding=(1, 2),
        )

        self.console.print(panel)

    def show_troubleshooting_guidance(self, provider_name: str, error: str) -> None:
        """
        Show provider-specific troubleshooting guidance.

        Args:
            provider_name: Name of the provider
            error: Error message
        """
        info = self.provider_info[provider_name]

        guidance = Text()
        guidance.append("Troubleshooting Guide:\n\n", style="bold yellow")

        # Show error-specific guidance
        if (
            "api key" in error.lower()
            or "authentication" in error.lower()
            or "format" in error.lower()
        ):
            guidance.append("🔑 API Key Issues:\n", style="bold red")
            guidance.append(f"  • Get your API key from: {info['api_key_url']}\n")
            if info["api_key_prefix"]:
                guidance.append(
                    f"  • Ensure the key starts with '{info['api_key_prefix']}'\n"
                )
            guidance.append(
                "  • Check that your account has sufficient credits/quota\n"
            )
            guidance.append("  • Verify the API key hasn't expired\n\n")

        if (
            "connection" in error.lower()
            or "network" in error.lower()
            or "timeout" in error.lower()
        ):
            guidance.append("🌐 Connection Issues:\n", style="bold blue")
            guidance.append("  • Check your internet connection\n")
            guidance.append("  • Verify firewall/proxy settings\n")
            guidance.append("  • Try again in a few moments (temporary outage)\n")
            guidance.append("  • Check provider status page for known issues\n\n")

        if "model" in error.lower():
            guidance.append("🤖 Model Issues:\n", style="bold green")
            guidance.append("  • Verify the model name is spelled correctly\n")
            guidance.append("  • Check if you have access to the requested model\n")
            guidance.append("  • Try a different model from the same provider\n\n")

        # Provider-specific guidance
        self._add_provider_specific_guidance(guidance, provider_name, error)

        # Show common issues for this provider
        from .setup_wizard_core import SetupWizardCore

        # Create a temporary instance just to access the common issues method
        temp_core = SetupWizardCore(None, None)
        common_issues = temp_core.get_common_setup_issues(provider_name)

        if common_issues:
            guidance.append("💡 Common Issues:\n", style="bold magenta")
            for issue in common_issues[:3]:  # Show top 3 common issues
                guidance.append(f"  • {issue['issue']}\n", style="dim")
                guidance.append(f"    → {issue['solution']}\n")
            guidance.append("\n")

        # Add helpful links
        guidance.append("📚 Helpful Links:\n", style="bold cyan")
        guidance.append(f"  • API Documentation: {info['api_key_url']}\n")
        if provider_name == "claude":
            guidance.append("  • Status Page: https://status.anthropic.com/\n")
            guidance.append("  • Support: https://support.anthropic.com/\n")
        elif provider_name == "openai":
            guidance.append("  • Status Page: https://status.openai.com/\n")
            guidance.append("  • Support: https://help.openai.com/\n")
        elif provider_name == "gemini":
            guidance.append("  • Status Page: https://status.cloud.google.com/\n")
            guidance.append("  • Support: https://support.google.com/\n")
        elif provider_name == "openrouter":
            guidance.append("  • Models: https://openrouter.ai/models\n")
            guidance.append("  • Documentation: https://openrouter.ai/docs\n")
        elif provider_name == "ollama":
            guidance.append("  • Documentation: https://ollama.ai/docs\n")
            guidance.append("  • GitHub: https://github.com/ollama/ollama\n")

        panel = Panel(
            guidance,
            title=f"🔧 {info['name']} Troubleshooting",
            border_style="yellow",
            padding=(1, 2),
        )

        self.console.print(panel)

    def _add_provider_specific_guidance(
        self, guidance: Text, provider_name: str, error: str
    ) -> None:
        """
        Add provider-specific troubleshooting guidance.

        Args:
            guidance: Text object to append guidance to
            provider_name: Name of the provider
            error: Error message
        """
        if provider_name == "claude":
            guidance.append("🎭 Claude-Specific:\n", style="bold")
            guidance.append(
                "  • Ensure you're using the correct API version (2023-06-01)\n"
            )
            guidance.append(
                "  • Check message format (Claude uses specific message structure)\n"
            )
            guidance.append(
                "  • Verify your account tier supports the selected model\n"
            )
            guidance.append("  • Try reducing max_tokens if getting context errors\n\n")

        elif provider_name == "openai":
            guidance.append("🤖 OpenAI-Specific:\n", style="bold")
            guidance.append(
                "  • Check if you have access to GPT-4 models (requires separate approval)\n"
            )
            guidance.append("  • Verify your organization ID if using team accounts\n")
            guidance.append("  • Ensure billing is set up for API usage\n")
            guidance.append("  • Try gpt-3.5-turbo if GPT-4 access is limited\n\n")

        elif provider_name == "gemini":
            guidance.append("🔮 Gemini-Specific:\n", style="bold")
            guidance.append(
                "  • Enable the Generative AI API in Google Cloud Console\n"
            )
            guidance.append("  • Check safety settings if requests are being blocked\n")
            guidance.append(
                "  • Verify your Google Cloud project has billing enabled\n"
            )
            guidance.append("  • Try gemini-1.5-flash for faster/cheaper responses\n\n")

        elif provider_name == "perplexity":
            guidance.append("🔍 Perplexity-Specific:\n", style="bold")
            guidance.append(
                "  • Ensure you have a Perplexity Pro subscription for API access\n"
            )
            guidance.append("  • Check your monthly API usage limits\n")
            guidance.append(
                "  • Verify search functionality is enabled for your account\n"
            )
            guidance.append(
                "  • Try 'sonar' model if 'sonar-pro' is not accessible\n\n"
            )

        elif provider_name == "xai":
            guidance.append("🚀 xAI-Specific:\n", style="bold")
            guidance.append(
                "  • Ensure you have access to the xAI API (currently in beta)\n"
            )
            guidance.append("  • Check your account status and API limits\n")
            guidance.append("  • Verify you're using the correct endpoint URL\n")
            guidance.append("  • Try 'grok-3-fast' for quicker responses\n\n")

        elif provider_name == "mistral":
            guidance.append("🌪️ Mistral-Specific:\n", style="bold")
            guidance.append(
                "  • Ensure you have a Mistral AI account with API access\n"
            )
            guidance.append("  • Check your usage quotas and billing status\n")
            guidance.append("  • Verify the model name matches exactly\n")
            guidance.append("  • Try 'mistral-small-3.1' for cost-effective usage\n\n")

        elif provider_name == "cohere":
            guidance.append("🧠 Cohere-Specific:\n", style="bold")
            guidance.append("  • Ensure you have a Cohere account with API access\n")
            guidance.append("  • Check your trial credits or subscription status\n")
            guidance.append("  • Verify the model supports your use case\n")
            guidance.append("  • Try 'command-light' for faster responses\n\n")

        elif provider_name == "azure":
            guidance.append("☁️ Azure OpenAI-Specific:\n", style="bold")
            guidance.append("  • Ensure you're using the correct endpoint URL\n")
            guidance.append("  • Verify deployment name matches your Azure setup\n")
            guidance.append("  • Check API version is compatible (2024-02-01 or later)\n")
            guidance.append("  • Confirm resource group has the deployment\n")
            guidance.append("  • Test with Azure OpenAI Studio first\n\n")

        elif provider_name == "vertex":
            guidance.append("🔺 Vertex AI-Specific:\n", style="bold")
            guidance.append("  • Enable Vertex AI API in Google Cloud Console\n")
            guidance.append("  • Set up Application Default Credentials (ADC)\n")
            guidance.append("  • Verify project ID and location are correct\n")
            guidance.append("  • Check IAM permissions for Vertex AI\n")
            guidance.append("  • Ensure billing is enabled for the project\n\n")

        elif provider_name == "bedrock":
            guidance.append("🪨 AWS Bedrock-Specific:\n", style="bold")
            guidance.append("  • Request model access in AWS Console first\n")
            guidance.append("  • Verify AWS region supports your model\n")
            guidance.append("  • Check IAM permissions include bedrock:InvokeModel\n")
            guidance.append("  • Ensure AWS credentials are properly configured\n")
            guidance.append("  • Try 'anthropic.claude-3-sonnet' if available\n\n")

        elif provider_name == "claude-code":
            guidance.append("💻 Claude Code-Specific:\n", style="bold")
            guidance.append("  • Ensure Claude Code is running locally\n")
            guidance.append("  • Check the local server is accessible\n")
            guidance.append("  • Verify no firewall blocking local connections\n")
            guidance.append("  • Try restarting Claude Code if connection fails\n")
            guidance.append("  • This is for local development only\n\n")

        elif provider_name == "openrouter":
            guidance.append("🔀 OpenRouter-Specific:\n", style="bold")
            guidance.append("  • Ensure your API key starts with 'sk-or-'\n")
            guidance.append("  • Check account has sufficient credits\n")
            guidance.append("  • Verify model is available (models vary by region)\n")
            guidance.append("  • Try 'anthropic/claude-3.5-sonnet' for best results\n")
            guidance.append("  • Enable fallback if models are unavailable\n")
            guidance.append("  • Check https://openrouter.ai/models for model list\n\n")
        
        elif provider_name == "ollama":
            guidance.append("🏠 Ollama-Specific:\n", style="bold")
            guidance.append("  • Start Ollama server: 'ollama serve'\n")
            guidance.append("  • Install required model: 'ollama pull <model-name>'\n")
            guidance.append("  • Check server URL (default: http://localhost:11434)\n")
            guidance.append("  • Ensure sufficient RAM for the model size\n")
            guidance.append("  • List available models: 'ollama list'\n")
            guidance.append("  • Check Ollama logs for detailed error messages\n\n")

    def show_provider_help(self, provider_name: str) -> None:
        """
        Show provider-specific setup help.

        Args:
            provider_name: Name of the provider
        """
        if provider_name not in self.provider_info:
            self.console.print(f"[red]Unknown provider: {provider_name}[/red]")
            return

        info = self.provider_info[provider_name]

        help_text = Text()
        help_text.append(f"{info['name']} Setup Guide\n\n", style="bold blue")
        help_text.append(f"{info['description']}\n\n")
        help_text.append("Strengths:\n", style="bold")
        for strength in info["strengths"]:
            help_text.append(f"  • {strength}\n", style="green")
        help_text.append(f"\nAPI Key: {info['api_key_url']}\n", style="bold")
        help_text.append(f"Setup Notes: {info['setup_notes']}\n")
        help_text.append(f"Available Models: {', '.join(info['models'])}\n")

        panel = Panel(
            help_text,
            title=f"📚 {info['name']} Help",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)
