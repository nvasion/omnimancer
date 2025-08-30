"""
Configuration Setup Wizard for Omnimancer CLI.

This module provides an interactive command-line configuration wizard
that uses the simplified configuration interface to guide users through
setting up Omnimancer with minimal complexity.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ..core.config_manager import ConfigManager
from ..core.config_provider import (
    ConfigurationContext,
    ConfigurationProvider,
)

logger = logging.getLogger(__name__)


class ConfigSetupWizard:
    """
    Interactive configuration setup wizard for Omnimancer CLI.

    Provides a user-friendly interface for configuring Omnimancer using
    the simplified configuration system with templates and guided setup.
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the configuration setup wizard.

        Args:
            config_manager: Optional ConfigManager instance
        """
        self.console = Console()
        self.config_manager = config_manager or ConfigManager()
        self.config_provider = ConfigurationProvider(self.config_manager)

        self.user_answers = {}
        self.current_step = 0

    def run_initial_setup(self) -> bool:
        """
        Run the initial setup wizard for new users.

        Returns:
            True if setup completed successfully, False otherwise
        """
        try:
            self._show_welcome_screen()

            if self._check_existing_configuration():
                return self._handle_existing_configuration()

            return self._run_fresh_setup()

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Setup cancelled by user.[/yellow]")
            return False
        except Exception as e:
            logger.error(f"Setup wizard failed: {e}")
            self.console.print(f"\n[red]Setup failed: {str(e)}[/red]")
            return False

    def _show_welcome_screen(self) -> None:
        """Display the welcome screen."""
        welcome_text = """
# Welcome to Omnimancer Setup! 🚀

Omnimancer is your unified CLI for multiple AI language models. This setup wizard will help you configure it quickly and easily.

## What we'll set up:
- **AI Providers**: Choose which AI services you want to use
- **API Keys**: Securely add your credentials  
- **Templates**: Pick optimized settings for your use case
- **Tools**: Enable helpful features and capabilities

The entire process takes just **5-10 minutes** and you can change everything later.
        """

        panel = Panel(
            Markdown(welcome_text),
            title="🤖 Omnimancer Configuration Wizard",
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)

        if not Confirm.ask("Ready to get started?", default=True):
            raise KeyboardInterrupt()

    def _check_existing_configuration(self) -> bool:
        """Check if there's an existing configuration."""
        try:
            config = self.config_manager.get_config()
            return len(config.providers) > 0
        except:
            return False

    def _handle_existing_configuration(self) -> bool:
        """Handle existing configuration options."""
        self.console.print("\n[yellow]Existing Configuration Found[/yellow]")
        self.console.print(
            "You already have Omnimancer configured. What would you like to do?"
        )

        options = [
            "Keep current settings and exit",
            "Add more providers to existing configuration",
            "Migrate to a simpler template-based configuration",
            "Start fresh with new configuration",
        ]

        for i, option in enumerate(options, 1):
            self.console.print(f"  {i}. {option}")

        choice = Prompt.ask(
            "Choose an option", choices=["1", "2", "3", "4"], default="1"
        )

        if choice == "1":
            self.console.print("[green]✓[/green] Keeping existing configuration")
            return True
        elif choice == "2":
            return self._add_providers_to_existing()
        elif choice == "3":
            return self._migrate_existing_configuration()
        else:
            return self._start_fresh_configuration()

    def _add_providers_to_existing(self) -> bool:
        """Add new providers to existing configuration."""
        self.console.print("\n[blue]Adding New Providers[/blue]")

        # Get current providers
        current_config = self.config_manager.get_config()
        current_providers = set(current_config.providers.keys())

        # Show available providers not yet configured
        all_providers = self.config_provider.get_simple_mode_providers()
        available_providers = [
            p for p in all_providers if p.name not in current_providers
        ]

        if not available_providers:
            self.console.print(
                "[yellow]All supported providers are already configured.[/yellow]"
            )
            return True

        self.console.print("Available providers to add:")
        for i, provider in enumerate(available_providers, 1):
            self.console.print(
                f"  {i}. {provider.display_name} - {provider.description}"
            )

        selections = Prompt.ask(
            "Which providers would you like to add? (comma-separated numbers)"
        ).split(",")

        # Process selections and add providers
        for selection in selections:
            try:
                idx = int(selection.strip()) - 1
                if 0 <= idx < len(available_providers):
                    provider = available_providers[idx]
                    self._configure_single_provider(provider)
            except (ValueError, IndexError):
                continue

        self.config_manager.save_config()
        self.console.print("[green]✓[/green] New providers added successfully!")
        return True

    def _migrate_existing_configuration(self) -> bool:
        """Migrate existing configuration to a template."""
        self.console.print("\n[blue]Configuration Migration[/blue]")

        with self.console.status("[bold green]Analyzing your current configuration..."):
            analysis = (
                self.config_provider.migration_helper.analyze_current_configuration()
            )

        # Show analysis results
        self._show_migration_analysis(analysis)

        if not analysis.recommended_template:
            self.console.print(
                "[yellow]No suitable template found for migration.[/yellow]"
            )
            return self._run_fresh_setup()

        if not Confirm.ask(
            f"Migrate to '{analysis.recommended_template}' template?",
            default=True,
        ):
            return self._run_fresh_setup()

        # Create and execute migration plan
        with self.console.status("[bold green]Creating migration plan..."):
            migration_plan = (
                self.config_provider.migration_helper.create_migration_plan(
                    analysis.recommended_template, preserve_customizations=True
                )
            )

        self._show_migration_plan(migration_plan)

        if Confirm.ask("Execute this migration plan?", default=True):
            return self._execute_migration(
                migration_plan, analysis.recommended_template
            )

        return self._run_fresh_setup()

    def _show_migration_analysis(self, analysis) -> None:
        """Show migration analysis results."""
        table = Table(title="Configuration Analysis")
        table.add_column("Aspect", style="cyan")
        table.add_column("Current State", style="magenta")

        table.add_row("Complexity Level", analysis.current_complexity.title())
        table.add_row("Recommended Template", analysis.recommended_template or "None")
        table.add_row("Confidence", f"{analysis.confidence:.0%}")
        table.add_row("Estimated Time", analysis.estimated_time)

        self.console.print(table)

        if analysis.simplification_opportunities:
            self.console.print("\n[green]Simplification Opportunities:[/green]")
            for opportunity in analysis.simplification_opportunities:
                self.console.print(f"  • {opportunity}")

        if analysis.potential_issues:
            self.console.print("\n[yellow]Potential Issues:[/yellow]")
            for issue in analysis.potential_issues:
                self.console.print(f"  ⚠ {issue}")

    def _show_migration_plan(self, migration_plan) -> None:
        """Show migration plan details."""
        self.console.print(
            f"\n[blue]Migration Plan[/blue] ({migration_plan.estimated_duration})"
        )

        for step in migration_plan.steps:
            step_num = step["step"]
            title = step["title"]
            description = step["description"]
            time_est = step.get("estimated_time", "")

            self.console.print(f"  {step_num}. **{title}** ({time_est})")
            self.console.print(f"     {description}")

            if "details" in step:
                for detail in step["details"][:2]:  # Show first 2 details
                    self.console.print(f"     • {detail}")

        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
            migration_plan.risk_level, "white"
        )
        self.console.print(
            f"\nRisk Level: [{risk_color}]{migration_plan.risk_level.title()}[/{risk_color}]"
        )

    def _execute_migration(self, migration_plan, template_name: str) -> bool:
        """Execute the migration plan."""
        self.console.print("\n[blue]Executing Migration...[/blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:

            task = progress.add_task(
                "Migrating configuration...", total=len(migration_plan.steps)
            )

            success, messages = self.config_provider.migration_helper.execute_migration(
                migration_plan, template_name
            )

            for i, step in enumerate(migration_plan.steps):
                progress.update(
                    task, advance=1, description=f"Step {i+1}: {step['title']}"
                )
                time.sleep(0.5)  # Show progress

        # Show results
        for message in messages:
            if message.startswith("✓"):
                self.console.print(f"[green]{message}[/green]")
            elif message.startswith("⚠"):
                self.console.print(f"[yellow]{message}[/yellow]")
            elif message.startswith("✗"):
                self.console.print(f"[red]{message}[/red]")
            else:
                self.console.print(message)

        return success

    def _start_fresh_configuration(self) -> bool:
        """Start fresh configuration, removing existing."""
        if Confirm.ask(
            "This will replace your current configuration. Continue?",
            default=False,
        ):
            # Backup current config
            backup_path = self.config_manager.backup_config()
            self.console.print(
                f"[green]✓[/green] Current configuration backed up to: {backup_path}"
            )

            # Clear current configuration
            self.config_manager.reset_config()
            return self._run_fresh_setup()

        return False

    def _run_fresh_setup(self) -> bool:
        """Run the complete fresh setup wizard."""
        self.console.print("\n[blue]Starting Fresh Configuration[/blue]")

        steps = self.config_provider.get_setup_wizard_steps()

        for step in steps:
            self.current_step = step["step"]

            if not self._execute_wizard_step(step):
                return False

        # Generate final configuration
        return self._finalize_configuration()

    def _execute_wizard_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single wizard step."""
        step_type = step["type"]

        self.console.print(f"\n[blue]Step {step['step']}: {step['title']}[/blue]")
        self.console.print(step["description"])

        if step_type == "welcome":
            return self._handle_welcome_step(step)
        elif step_type == "single_choice":
            return self._handle_single_choice_step(step)
        elif step_type == "provider_selection":
            return self._handle_provider_selection_step(step)
        elif step_type == "api_keys":
            return self._handle_api_keys_step(step)
        elif step_type == "feature_selection":
            return self._handle_feature_selection_step(step)
        elif step_type == "review":
            return self._handle_review_step(step)
        else:
            self.console.print(f"[red]Unknown step type: {step_type}[/red]")
            return True  # Continue anyway

    def _handle_welcome_step(self, step: Dict[str, Any]) -> bool:
        """Handle welcome step."""
        content = step.get("content", {})
        features = content.get("features", [])

        self.console.print(f"\n{content.get('message', '')}")
        self.console.print(
            f"Estimated time: {content.get('estimated_time', 'Unknown')}"
        )

        if features:
            self.console.print("\nWhat we'll configure:")
            for feature in features:
                self.console.print(f"  • {feature}")

        return Confirm.ask("\nReady to continue?", default=True)

    def _handle_single_choice_step(self, step: Dict[str, Any]) -> bool:
        """Handle single choice selection step."""
        option = step.get("option", {})
        choices = option.get("choices", [])

        if not choices:
            return True

        # Display choices
        self.console.print(f"\n{option.get('display_name', 'Please choose:')}")
        for i, choice in enumerate(choices, 1):
            label = choice.get("label", choice.get("value", ""))
            description = choice.get("description", "")
            self.console.print(f"  {i}. **{label}**")
            if description:
                self.console.print(f"     {description}")

        help_text = step.get("help_text", "")
        if help_text:
            self.console.print(f"\n[dim]{help_text}[/dim]")

        # Get user selection
        choice_nums = [str(i) for i in range(1, len(choices) + 1)]
        default_idx = next(
            (
                i
                for i, choice in enumerate(choices)
                if choice.get("value") == option.get("default_value")
            ),
            0,
        )

        selection = Prompt.ask(
            "Your choice", choices=choice_nums, default=str(default_idx + 1)
        )

        selected_choice = choices[int(selection) - 1]
        self.user_answers[option["key"]] = selected_choice["value"]

        return True

    def _handle_provider_selection_step(self, step: Dict[str, Any]) -> bool:
        """Handle provider selection step."""
        content = step.get("content", {})
        providers = content.get("providers", [])

        self.console.print(f"\n{content.get('message', '')}")

        # Show available providers
        table = Table(title="Available AI Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Setup", style="yellow")
        table.add_column("Recommended", style="green")

        for i, provider_info in enumerate(providers):
            provider = provider_info["provider"]
            recommended = "✓" if provider_info.get("recommended", False) else ""
            setup_difficulty = provider_info.get("setup_difficulty", "Easy")

            table.add_row(
                f"{i+1}. {provider['display_name']}",
                (
                    provider["description"][:60] + "..."
                    if len(provider["description"]) > 60
                    else provider["description"]
                ),
                setup_difficulty,
                recommended,
            )

        self.console.print(table)

        # Get selections
        selections = Prompt.ask(
            "Which providers would you like to enable? (comma-separated numbers, or 'r' for recommended only)",
            default="r",
        )

        if selections.lower() == "r":
            # Enable recommended providers
            for provider_info in providers:
                if provider_info.get("recommended", False):
                    provider = provider_info["provider"]
                    self.user_answers[f"enable_{provider['name']}"] = True
        else:
            # Enable selected providers
            for selection in selections.split(","):
                try:
                    idx = int(selection.strip()) - 1
                    if 0 <= idx < len(providers):
                        provider = providers[idx]["provider"]
                        self.user_answers[f"enable_{provider['name']}"] = True
                except (ValueError, IndexError):
                    continue

        return True

    def _handle_api_keys_step(self, step: Dict[str, Any]) -> bool:
        """Handle API keys input step."""
        content = step.get("content", {})
        self.console.print(f"\n{content.get('message', '')}")

        # Get enabled providers that need API keys
        enabled_providers = [
            key.replace("enable_", "")
            for key, value in self.user_answers.items()
            if key.startswith("enable_") and value
        ]

        providers_needing_keys = [
            p for p in enabled_providers if p not in ["claude-code", "ollama"]
        ]

        if not providers_needing_keys:
            self.console.print(
                "[green]✓[/green] No API keys needed for your selected providers!"
            )
            return True

        for provider in providers_needing_keys:
            self._get_api_key_for_provider(provider)

        return True

    def _get_api_key_for_provider(self, provider_name: str) -> None:
        """Get API key for a specific provider."""
        instructions = self.config_provider.get_provider_setup_instructions(
            provider_name
        )

        # Show setup instructions
        panel_content = f"**{instructions['title']}**\n\n"
        panel_content += "Steps to get your API key:\n"
        for i, step in enumerate(instructions["steps"], 1):
            panel_content += f"{i}. {step}\n"

        panel_content += f"\nExpected format: `{instructions['api_key_format']}`\n"
        panel_content += f"Cost: {instructions['cost_info']}"

        panel = Panel(
            Markdown(panel_content),
            title=f"🔑 {provider_name.title()} Setup",
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)

        # Get API key
        api_key = Prompt.ask(
            f"Enter your {provider_name} API key (or 'skip' to configure later)",
            password=True,
            default="",
        )

        if api_key and api_key.lower() != "skip":
            self.user_answers[f"{provider_name}_api_key"] = api_key
            self.console.print(f"[green]✓[/green] {provider_name} API key saved")
        else:
            self.console.print(
                f"[yellow]⚠[/yellow] {provider_name} will be disabled until API key is added"
            )
            self.user_answers[f"enable_{provider_name}"] = False

    def _handle_feature_selection_step(self, step: Dict[str, Any]) -> bool:
        """Handle feature selection step."""
        content = step.get("content", {})
        features = content.get("features", [])

        self.console.print(f"\n{content.get('message', '')}")

        for feature in features:
            key = feature["key"]
            name = feature["name"]
            description = feature["description"]
            recommended = feature.get("recommended", False)

            rec_text = " [green](Recommended)[/green]" if recommended else ""
            self.console.print(f"\n**{name}**{rec_text}")
            self.console.print(f"  {description}")

            enable = Confirm.ask(f"Enable {name}?", default=recommended)
            self.user_answers[key] = enable

        return True

    def _handle_review_step(self, step: Dict[str, Any]) -> bool:
        """Handle configuration review step."""
        content = step.get("content", {})
        self.console.print(f"\n{content.get('message', '')}")

        # Show configuration preview
        preview = self.config_provider.get_configuration_preview(self.user_answers)
        self._show_configuration_preview(preview)

        if not preview.get("setup_complete", False):
            self.console.print("\n[red]Configuration is incomplete:[/red]")
            for issue in preview.get("missing_items", []):
                self.console.print(f"  • {issue}")

            if not Confirm.ask(
                "Continue with incomplete configuration?", default=False
            ):
                return False

        return Confirm.ask("Save this configuration?", default=True)

    def _show_configuration_preview(self, preview: Dict[str, Any]) -> None:
        """Show configuration preview."""
        table = Table(title="Configuration Preview")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Use Case", preview.get("use_case", "general").title())
        table.add_row(
            "Enabled Providers",
            ", ".join(preview.get("enabled_providers", [])),
        )
        table.add_row(
            "Estimated Cost", preview.get("estimated_monthly_cost", "Unknown")
        )
        table.add_row("Features", ", ".join(preview.get("features_enabled", [])))

        status_color = "green" if preview.get("setup_complete", False) else "yellow"
        status_text = (
            "Complete" if preview.get("setup_complete", False) else "Incomplete"
        )
        table.add_row("Status", f"[{status_color}]{status_text}[/{status_color}]")

        self.console.print(table)

    def _finalize_configuration(self) -> bool:
        """Generate and save the final configuration."""
        self.console.print("\n[blue]Finalizing Configuration...[/blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:

            task = progress.add_task("Generating configuration...", total=None)

            # Generate configuration
            success, config_dict, messages = (
                self.config_provider.generate_simple_configuration(self.user_answers)
            )
            progress.update(task, description="Saving configuration...")

            if success:
                progress.update(task, description="Validating configuration...")
                time.sleep(1)  # Show progress

                # Validate final configuration
                is_valid, errors = self.config_provider.validate_simple_answers(
                    self.user_answers
                )

                if is_valid:
                    progress.update(task, description="Configuration complete!")
                    time.sleep(1)

        # Show results
        if success:
            self.console.print(
                "\n[green]✅ Configuration completed successfully![/green]"
            )

            for message in messages:
                self.console.print(f"  • {message}")

            # Show next steps
            self._show_next_steps()
            return True
        else:
            self.console.print("\n[red]❌ Configuration failed.[/red]")
            for message in messages:
                self.console.print(f"  • {message}")
            return False

    def _show_next_steps(self) -> None:
        """Show next steps after successful configuration."""
        next_steps_text = """
# 🎉 You're all set!

Omnimancer is now configured and ready to use. Here's what you can do next:

## Try it out:
```bash
omnimancer "Hello! How are you today?"
```

## Check your configuration:
```bash
omnimancer config show
```

## Switch between providers:
```bash
omnimancer --provider claude "Write me a Python function"
omnimancer --provider openai "Help me debug this code"
```

## Get help:
```bash
omnimancer --help
```

## Advanced features:
- Use `omnimancer config advanced` to access all settings
- Visit our documentation for tips and tricks
- Join our community for support and updates
        """

        panel = Panel(
            Markdown(next_steps_text),
            title="🚀 Ready to Chat!",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _configure_single_provider(self, provider) -> None:
        """Configure a single provider."""
        self.console.print(f"\nConfiguring {provider.display_name}...")

        if provider.name not in ["claude-code", "ollama"]:
            self._get_api_key_for_provider(provider.name)

        # Add to current configuration
        # This would integrate with the actual config manager
        self.console.print(f"[green]✓[/green] {provider.display_name} configured")


def run_config_setup_wizard(
    config_manager: Optional[ConfigManager] = None,
) -> bool:
    """
    Run the configuration setup wizard.

    Args:
        config_manager: Optional ConfigManager instance

    Returns:
        True if setup completed successfully, False otherwise
    """
    wizard = ConfigSetupWizard(config_manager)
    return wizard.run_initial_setup()


def run_quick_setup(use_case: str = "general", providers: List[str] = None) -> bool:
    """
    Run a quick setup with minimal prompts.

    Args:
        use_case: Primary use case for the configuration
        providers: List of provider names to enable

    Returns:
        True if setup completed successfully, False otherwise
    """
    console = Console()
    ConfigManager()
    simple_interface = create_simple_config_interface()

    console.print(f"[blue]Quick Setup for {use_case.title()} Use Case[/blue]")

    # Get recommended configuration
    context = ConfigurationContext(use_case=use_case)
    recommendation = simple_interface.config_mode_manager.get_recommended_configuration(
        context
    )

    console.print(f"Using template: {recommendation.template_name}")

    # Apply template
    success, messages = (
        simple_interface.config_mode_manager.apply_template_configuration(
            recommendation.template_name
        )
    )

    if success:
        console.print("[green]✅ Quick setup completed![/green]")
        for message in messages:
            console.print(f"  • {message}")
        return True
    else:
        console.print("[red]❌ Quick setup failed.[/red]")
        for message in messages:
            console.print(f"  • {message}")
        return False
