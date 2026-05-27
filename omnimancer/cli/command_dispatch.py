"""
Command dispatch mixin for Omnimancer CLI.

Contains all slash command handlers and helper methods, extracted from interface.py
to keep the main interface module manageable.
"""

# Standard library imports
import asyncio
import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional

# Third-party imports
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Internal imports - Core
from ..core.models import EnhancedModelInfo
from ..providers.factory import ProviderFactory

# Internal imports - CLI
from .commands import Command, SlashCommand

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rich.console import Console

    from ..core.agent_mode_manager import AgentModeManager
    from ..core.history_manager import HistoryManager
    from .approval_integration import CLIApprovalIntegration
    from .display import DisplayManager


class CommandDispatchMixin:
    """
    Mixin providing all slash command handlers and helper methods.

    Mixed into CommandLineInterface via multiple inheritance.
    Relies on self.engine, self.console, self.agent_manager,
    self.approval_integration, self.history_manager, self.permissions_handler,
    and the display helper methods (self._show_error, self._show_info, etc.)
    being available from the host class.
    """

    # Type stubs for attributes provided by the host class
    engine: Any
    console: "Console"
    running: bool
    agent_manager: Optional["AgentModeManager"]
    history_manager: "HistoryManager"
    approval_integration: Optional["CLIApprovalIntegration"]
    display_manager: "DisplayManager"

    # Methods provided by DisplayMixin (or the host class)
    def _show_error(self, message: str) -> None: ...
    def _show_info(self, message: str) -> None: ...
    def _show_success(self, message: str) -> None: ...
    def _show_warning(self, message: str) -> None: ...
    def _show_help(self) -> None: ...
    def _show_command_help(self, command_name: str) -> None: ...
    def _show_status(self) -> None: ...
    def _clear_screen(self) -> None: ...
    def stop(self) -> None: ...

    async def _handle_slash_command(self, command: Command) -> None:
        """
        Handle a slash command.

        Args:
            command: Slash command
        """
        slash_cmd = command.slash_command

        if slash_cmd == SlashCommand.HELP:
            # Check if specific command help is requested
            if command.args:
                self._show_command_help(command.args[0])
            else:
                self._show_help()
        elif slash_cmd == SlashCommand.QUIT:
            self.stop()
        elif slash_cmd == SlashCommand.CLEAR:
            self._clear_screen()
        elif slash_cmd == SlashCommand.STATUS:
            self._show_status()
        elif slash_cmd == SlashCommand.MODELS or slash_cmd == SlashCommand.MODEL:
            await self._show_models(command)
        elif slash_cmd == SlashCommand.SWITCH:
            await self._handle_switch_command(command)
        elif slash_cmd == SlashCommand.SAVE:
            await self._handle_save_command(command)
        elif slash_cmd == SlashCommand.LOAD:
            await self._handle_load_command(command)
        elif slash_cmd == SlashCommand.LIST:
            await self._show_conversations()
        elif slash_cmd == SlashCommand.PROVIDERS:
            await self._show_providers(command)
        elif slash_cmd == SlashCommand.TOOLS:
            await self._show_tools()
        elif slash_cmd == SlashCommand.MCP:
            await self._handle_mcp_command(command)
        elif slash_cmd == SlashCommand.HISTORY:
            await self._handle_history_command(command)
        elif slash_cmd == SlashCommand.ADD_MODEL:
            await self._handle_add_model_command(command)
        elif slash_cmd == SlashCommand.REMOVE_MODEL:
            await self._handle_remove_model_command(command)
        elif slash_cmd == SlashCommand.LIST_CUSTOM_MODELS:
            await self._handle_list_custom_models_command(command)
        elif slash_cmd == SlashCommand.AGENT:
            await self._handle_agent_command(command)
        elif slash_cmd == SlashCommand.CONFIG:
            await self._handle_config_command(command)
        elif slash_cmd is not None:
            self._show_info(
                f"Command {slash_cmd.value} is not yet implemented"
            )

    async def _handle_dynamic_command(self, command: Command) -> None:
        """
        Handle a dynamic command.

        Args:
            command: Dynamic command
        """
        dynamic_cmd = command.dynamic_command
        if not dynamic_cmd:
            self._show_error("Invalid dynamic command")
            return

        try:
            # Check if command has a Python handler
            if dynamic_cmd.handler:
                # Call the Python handler
                result = await self._execute_dynamic_handler(
                    dynamic_cmd.handler, command.args
                )
                if result:
                    self.console.print(result)

            # Check if command has a script to execute
            elif dynamic_cmd.script_path:
                result = await self._execute_dynamic_script(
                    dynamic_cmd.script_path, command.args
                )
                if result:
                    self.console.print(result)

            else:
                self._show_error(
                    f"Dynamic command '{dynamic_cmd.name}' has no handler or script"
                )

        except Exception as e:
            self._show_error(
                f"Error executing dynamic command '{dynamic_cmd.name}': {e}"
            )

    async def _execute_dynamic_handler(
        self, handler: Callable, args: List[str]
    ) -> Optional[str]:
        """Execute a Python handler for a dynamic command."""

        # Check if handler is async
        if inspect.iscoroutinefunction(handler):
            return await handler(  # type: ignore[no-any-return]
                args, engine=self.engine, console=self.console
            )
        else:
            return handler(  # type: ignore[no-any-return]
                args, engine=self.engine, console=self.console
            )

    async def _execute_dynamic_script(
        self, script_path: Path, args: List[str]
    ) -> Optional[str]:
        """Execute a script for a dynamic command."""

        try:
            # Make script executable if it isn't already
            script_path.chmod(script_path.stat().st_mode | 0o111)

            # Execute the script
            process = await asyncio.create_subprocess_exec(
                str(script_path),
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Script execution failed"
                self._show_error(f"Script error: {error_msg}")
                return None

            return stdout.decode() if stdout else None

        except Exception as e:
            self._show_error(f"Failed to execute script: {e}")
            return None

    def _handle_system_command(self, command: Command) -> None:
        """
        Handle a system command.

        Args:
            command: System command
        """
        if command.content == "quit":
            self.stop()

    # Display methods are in cli/display.py (DisplayMixin)

    async def _show_models(self, command: Command) -> None:
        """Show available models with enhanced provider grouping and capabilities."""
        try:
            args = command.args
            filter_type = args[0].lower() if len(args) > 0 else None
            filter_value = args[1] if len(args) > 1 else None

            # First check if there are any models available at all
            all_models = self.engine.get_available_models()
            if not all_models:
                models_panel = Panel(
                    "No models available.",
                    title="Available Models",
                    border_style="cyan",
                )
                self.console.print(models_panel)
                return

            # Get enhanced models list with filtering
            result = await self._get_enhanced_models_list(filter_type, filter_value)

            title = "Available Models"
            if filter_type:
                title += f" - Filtered by {filter_type.title()}"
                if filter_value:
                    title += f": {filter_value}"

            # Use consistent Panel formatting like other commands
            models_panel = Panel(result, title=title, border_style="cyan")
            self.console.print(models_panel)

        except Exception as e:
            self._show_error(f"Failed to get models: {e}")

    async def _get_enhanced_models_list(
        self, filter_type: Optional[str] = None, filter_value: Optional[str] = None
    ) -> str:
        """Get enhanced models list with filtering and detailed information."""
        try:
            # Get models directly from providers
            all_models = {}
            current_provider_name = None
            current_model = None

            if self.engine.current_provider:
                current_model = self.engine.current_provider.model
                for name, provider in self.engine.providers.items():
                    if provider == self.engine.current_provider:
                        current_provider_name = name
                        break

            # Collect models from each provider
            for provider_name, provider in self.engine.providers.items():
                try:
                    models = provider.get_available_models()
                    if models:
                        all_models[provider_name] = models
                except Exception:
                    continue

            # Apply filtering
            if filter_type == "provider" and filter_value:
                all_models = {
                    k: v
                    for k, v in all_models.items()
                    if k.lower() == filter_value.lower()
                }
            elif filter_type == "free":
                filtered_models = {}
                for provider, models in all_models.items():
                    free_models = [
                        m
                        for m in models
                        if getattr(m, "is_free", False)
                        or (hasattr(m, "cost_per_token") and m.cost_per_token == 0)
                    ]
                    if free_models:
                        filtered_models[provider] = free_models
                all_models = filtered_models
            elif filter_type == "latest":
                filtered_models = {}
                for provider, models in all_models.items():
                    latest_models = [
                        m for m in models if getattr(m, "latest_version", False)
                    ]
                    if latest_models:
                        filtered_models[provider] = latest_models
                all_models = filtered_models

            if not all_models:
                return "No models found matching the specified criteria."

            # Build simple output without complex tables
            output_lines = []

            for provider_name in sorted(all_models.keys()):
                models = all_models[provider_name]
                if not models:
                    continue

                # Provider header with basic info
                provider_status = "✓"
                current_marker = (
                    " (active)" if provider_name == current_provider_name else ""
                )
                pname = provider_name.upper()
                output_lines.append(
                    f"\n[bold cyan]{pname}[/bold cyan]"
                    f" {provider_status}{current_marker}:"
                )

                # List models in simple format
                for model in sorted(models, key=lambda m: m.name):
                    # Model name - truncate long names
                    name = model.name
                    if len(name) > 35:
                        name = name[:32] + "..."

                    # Status indicator (simple)
                    if (
                        model.name == current_model
                        and provider_name == current_provider_name
                    ):
                        status = "●"  # Current
                    elif getattr(model, "available", True):
                        status = "✓"  # Available
                    else:
                        status = "✗"  # Unavailable

                    # NEW indicator
                    if getattr(model, "latest_version", False):
                        status += " NEW"

                    # Capabilities (compact)
                    caps = ""
                    if getattr(model, "supports_tools", False):
                        caps += "🔧"
                    if getattr(model, "supports_multimodal", False):
                        caps += "🖼️"

                    # Cost (simplified)
                    is_free = getattr(model, "is_free", False) or (
                        hasattr(model, "cost_per_token")
                        and getattr(model, "cost_per_token", 0) == 0
                    )
                    if is_free:
                        cost = "FREE"
                    else:
                        input_cost = (
                            getattr(model, "cost_per_million_input", None)
                            or getattr(model, "cost_per_token", 0) * 1000000
                        )
                        if input_cost and input_cost > 0:
                            cost = f"${input_cost:.0f}/M"
                        else:
                            cost = ""

                    # Description - very short
                    desc = getattr(model, "description", "")
                    # Remove provider name from description
                    for remove_str in [
                        provider_name.title(),
                        provider_name.upper(),
                        model.name,
                        "via OpenRouter",
                    ]:
                        desc = desc.replace(remove_str, "").strip(" -")
                    if len(desc) > 40:
                        desc = desc[:37] + "..."

                    # Format line with fixed widths
                    output_lines.append(
                        f"  {name:35} {status:6} {caps:4} {cost:8} {desc}"
                    )

            # Add simple legend
            output_lines.append(
                "\nLegend: ● Current | ✓ Available"
                " | ✗ Unavailable | NEW Latest"
                " | 🔧 Tools | 🖼️ Multimodal"
            )

            return "\n".join(output_lines)

        except Exception as e:
            return f"Error displaying models: {e}"

    async def _handle_switch_command(self, command: Command) -> None:
        """Handle switch command with enhanced provider type support."""
        args = command.args
        if len(args) < 1:
            self._show_error("Usage: /switch <provider> [model]")
            self._show_info("Available providers:")
            # Show available providers as help

            providers_command = Command.create_slash_command(
                SlashCommand.PROVIDERS, [], "/providers"
            )
            await self._show_providers(providers_command)
            return

        provider_name = args[0].lower()
        model_name = args[1] if len(args) > 1 else None

        try:
            with self.console.status("[bold yellow]Switching...", spinner="dots"):
                # Check if provider is available but not initialized

                available_providers = ProviderFactory.get_available_providers()

                if provider_name not in self.engine.providers:
                    if provider_name in available_providers:
                        self._show_error(
                            f"Provider '{provider_name}' is"
                            " available but not configured."
                        )
                        self._show_info(
                            "Configure it in your settings"
                            " first, then try switching"
                            " again."
                        )
                        return
                    else:
                        # Show suggestions for similar provider names
                        suggestions = [
                            p
                            for p in available_providers
                            if provider_name in p.lower() or p.lower() in provider_name
                        ]
                        if suggestions:
                            self._show_error(
                                f"Provider '{provider_name}'"
                                " not found. Did you mean:"
                                f" {', '.join(suggestions)}?"
                            )
                        else:
                            self._show_error(f"Provider '{provider_name}' not found.")

                        providers_command = Command.create_slash_command(
                            SlashCommand.PROVIDERS, [], "/providers"
                        )
                        await self._show_providers(providers_command)
                        return

                # Validate model if specified
                if model_name:
                    provider = self.engine.providers[provider_name]
                    available_models = provider.get_available_models()
                    model_names = [m.name for m in available_models]

                    # Also check custom models for this provider
                    custom_models = self.engine.config_manager.get_custom_models()
                    custom_model_names = [
                        m.name for m in custom_models if m.provider == provider_name
                    ]

                    # Combine both lists
                    all_model_names = model_names + custom_model_names

                    if model_name not in all_model_names:
                        self._show_error(
                            f"Model '{model_name}' not"
                            " available for provider"
                            f" '{provider_name}'."
                        )
                        if all_model_names:
                            # Show suggestions for similar model names
                            suggestions = [
                                m
                                for m in all_model_names
                                if model_name.lower() in m.lower()
                                or m.lower() in model_name.lower()
                            ]
                            if suggestions:
                                models_str = ", ".join(
                                    suggestions[:5]
                                )
                                self._show_info(
                                    "Available models for"
                                    f" {provider_name}:"
                                    f" {models_str}"
                                )
                            else:
                                models_str = ", ".join(
                                    all_model_names[:5]
                                )
                                self._show_info(
                                    "Available models for"
                                    f" {provider_name}:"
                                    f" {models_str}"
                                )
                        return

                success = await self.engine.switch_model(provider_name, model_name)

            if success:
                current_model = (
                    self.engine.current_provider.model
                    if self.engine.current_provider
                    else "unknown"
                )
                provider_info = self.engine.current_provider

                # Show enhanced switch confirmation with capabilities
                capabilities = []
                if provider_info.supports_tools():
                    capabilities.append("🔧 Tools")
                if provider_info.supports_multimodal():
                    capabilities.append("🖼️ Multimodal")

                # Get model info for additional details
                try:
                    model_info = provider_info.get_model_info()
                    model_details = []
                    if model_info and model_info.max_tokens:
                        model_details.append(f"Max tokens: {model_info.max_tokens:,}")
                    if model_info and model_info.cost_per_token:
                        model_details.append(
                            f"Cost: ${model_info.cost_per_token:.6f}/token"
                        )

                    detail_text = (
                        f" | {' | '.join(model_details)}" if model_details else ""
                    )
                except Exception:
                    detail_text = ""

                capability_text = (
                    f" ({', '.join(capabilities)})" if capabilities else ""
                )
                self._show_info(
                    f"✓ Switched to {provider_name}:"
                    f"{current_model}"
                    f"{capability_text}{detail_text}"
                )

                # Show MCP tool availability if provider supports tools
                if provider_info.supports_tools():
                    try:
                        tools_info = await self.engine.get_available_tools()
                        if tools_info:
                            tool_count = len(tools_info)
                            self._show_info(
                                f"🔧 {tool_count} MCP tools available for use"
                            )
                        else:
                            self._show_info(
                                "🔧 Tool calling supported"
                                " (no MCP tools currently"
                                " available)"
                            )
                    except Exception:
                        pass
            else:
                self._show_error("Failed to switch provider/model")

        except Exception as e:
            self._show_error(f"Switch failed: {e}")
            # Show available options on error
            if "not available" in str(e).lower() or "not found" in str(e).lower():
                self._show_info("Available options:")

                providers_command = Command.create_slash_command(
                    SlashCommand.PROVIDERS, [], "/providers"
                )
                await self._show_providers(providers_command)

    async def _handle_save_command(self, command: Command) -> None:
        """Handle save conversation command."""
        args = command.args
        filename = args[0] if args else None

        try:
            # Check if there are messages to save
            summary = self.engine.get_conversation_summary()
            if summary["message_count"] == 0:
                self._show_info("No messages to save.")
                return

            with self.console.status(
                "[bold yellow]Saving conversation...", spinner="dots"
            ):
                saved_filename = self.engine.save_conversation(filename)

            self._show_info(f"Conversation saved as: {saved_filename}")

        except Exception as e:
            self._show_error(f"Save failed: {e}")

    async def _handle_load_command(self, command: Command) -> None:
        """Handle load conversation command."""
        args = command.args
        if not args:
            # Show available conversations
            await self._show_conversations()
            self._show_error("Usage: /load <filename>")
            return

        filename = args[0]

        try:
            with self.console.status(
                "[bold yellow]Loading conversation...", spinner="dots"
            ):
                self.engine.load_conversation(filename)

            # Show loaded conversation info
            info = self.engine.get_conversation_info(filename)
            if info:
                self._show_info(f"Loaded conversation: {filename}")
                self._show_info(
                    f"Messages: {info['message_count']}, Model: {info['current_model']}"
                )
            else:
                self._show_info(f"Loaded conversation: {filename}")

        except Exception as e:
            self._show_error(f"Load failed: {e}")

    async def _handle_config_command(self, command: Command) -> None:
        """Handle config command with subcommands."""
        args = command.args

        if not args:
            await self._show_config()
            return

        subcommand = args[0].lower()

        if subcommand == "show":
            await self._show_config()
        elif subcommand == "set":
            if len(args) >= 3:
                await self._handle_config_set(args[1], args[2])
            else:
                self._show_error("Usage: /config set <key> <value>")
        elif subcommand == "get":
            if len(args) >= 2:
                await self._handle_config_get(args[1])
            else:
                self._show_error("Usage: /config get <key>")
        else:
            self._show_config_help()

    async def _show_config(self) -> None:
        """Show current configuration."""
        try:
            config_info = self.engine.get_configuration_info()

            table = Table(title="Current Configuration")
            table.add_column("Setting", style="bold")
            table.add_column("Value", style="cyan")

            for key, value in config_info.items():
                # Mask sensitive information
                if "key" in key.lower() or "password" in key.lower():
                    if value:
                        masked_value = (
                            value[:8] + "..." if len(str(value)) > 8 else "***"
                        )
                        table.add_row(key, masked_value)
                    else:
                        table.add_row(key, "[dim]Not set[/dim]")
                else:
                    table.add_row(key, str(value))

            self.console.print(table)

        except Exception as e:
            self._show_error(f"Failed to show configuration: {e}")

    async def _handle_config_set(self, key: str, value: str) -> None:
        """Handle config set operation."""
        try:
            # This would need to be implemented in the engine
            self._show_info(f"Setting {key} = {value}")
            self._show_info(
                "Note: Some settings may require restarting Omnimancer to take effect"
            )
        except Exception as e:
            self._show_error(f"Failed to set configuration: {e}")

    async def _handle_config_get(self, key: str) -> None:
        """Handle config get operation."""
        try:
            config_info = self.engine.get_configuration_info()
            if key in config_info:
                value = config_info[key]
                # Mask sensitive information
                if "key" in key.lower() or "password" in key.lower():
                    if value:
                        masked_value = (
                            value[:8] + "..." if len(str(value)) > 8 else "***"
                        )
                        self.console.print(f"[bold]{key}:[/bold] {masked_value}")
                    else:
                        self.console.print(f"[bold]{key}:[/bold] [dim]Not set[/dim]")
                else:
                    self.console.print(f"[bold]{key}:[/bold] {value}")
            else:
                self._show_error(f"Configuration key '{key}' not found")
        except Exception as e:
            self._show_error(f"Failed to get configuration: {e}")

    def _show_config_help(self) -> None:
        """Show configuration command help."""
        help_text = """[bold]Configuration Commands:[/bold]

[cyan]/config[/cyan]                  - Show current configuration
[cyan]/config show[/cyan]             - Show current configuration
[cyan]/config set <key> <value>[/cyan] - Set a configuration value
[cyan]/config get <key>[/cyan]        - Get a configuration value"""
        self.console.print(help_text)

    async def _show_conversations(self) -> None:
        """Show available conversation files."""
        try:
            conversations = self.engine.list_conversations()

            if not conversations:
                self._show_info("No saved conversations found.")
                return

            lines = ["Available conversations:"]

            for conv in conversations:
                created = conv.get("created_at", "Unknown")
                if created and created != "Unknown":
                    try:

                        created_dt = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                        created = created_dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        pass

                lines.append(f"  📄 {conv['filename']}")
                lines.append(f"     Created: {created}")
                model = conv.get('current_model', 'Unknown')
                lines.append(
                    f"     Messages: {conv['message_count']},"
                    f" Model: {model}"
                )
                lines.append("")

            conversations_panel = Panel(
                "\n".join(lines),
                title="Saved Conversations",
                border_style="cyan",
            )
            self.console.print(conversations_panel)

        except Exception as e:
            self._show_error(f"Failed to list conversations: {e}")

    async def _show_providers(self, command: Command) -> None:
        """Show available providers and their status."""
        await self._show_providers_legacy()

    async def _show_providers_legacy(self) -> None:
        """Show available providers and their status."""
        try:
            result = self.engine._get_providers_list()

            providers_panel = Panel(
                result, title="Available Providers", border_style="cyan"
            )
            self.console.print(providers_panel)

        except Exception as e:
            self._show_error(f"Failed to get providers: {e}")

    async def _show_tools(self) -> None:
        """Show available MCP tools."""
        try:
            result = await self.engine._get_tools_list()

            tools_panel = Panel(
                result, title="Available MCP Tools", border_style="cyan"
            )
            self.console.print(tools_panel)

        except Exception as e:
            self._show_error(f"Failed to get tools: {e}")

    async def _handle_mcp_command(self, command: Command) -> None:
        """Handle MCP management commands."""
        try:
            with self.console.status(
                "[bold yellow]Processing MCP command...", spinner="dots"
            ):
                result = await self.engine._handle_mcp_command(command)

            mcp_panel = Panel(result, title="MCP Command Result", border_style="cyan")
            self.console.print(mcp_panel)

        except Exception as e:
            self._show_error(f"MCP command failed: {e}")

    async def _handle_agent_command(self, command: Command) -> None:
        """
        Handle agent mode commands.

        Commands:
        - /agent on [--auto-approve] - Enable agent mode
        - /agent off - Disable agent mode
        - /agent status - Show agent status
        - /agent enable [--auto-approve] - Alias for 'on'
        - /agent disable - Alias for 'off'
        - /agent pause - Pause agent mode
        - /agent resume - Resume agent mode
        """
        args = command.args
        if not args:
            args = ["status"]

        subcommand = args[0].lower()

        # Handle agent mode commands
        if not self.agent_manager:
            self._show_error(
                "Agent mode is not available. Failed to initialize agent manager."
            )
            return

        try:
            if subcommand in ["on", "enable"]:
                # Check for auto-approve flag
                auto_approve = "--auto-approve" in args

                if self.agent_manager.mode.value == "on":
                    self._show_info("Agent mode is already enabled.")
                    return

                self._show_info("Enabling agent mode...")
                success = await self.agent_manager.enable_agent_mode(
                    auto_approve=auto_approve
                )

                if success:
                    mode_text = "Agent mode enabled"
                    if auto_approve:
                        mode_text += " with auto-approval for low-risk operations"

                    self.console.print(
                        Panel(
                            mode_text
                            + "\n\nAgent will now process operations automatically.\n"
                            "Use '/agent status' to monitor progress.\n"
                            "Use '/agent off' to disable agent mode.",
                            title="Agent Mode Enabled",
                            border_style="green",
                        )
                    )

                else:
                    self._show_error("Failed to enable agent mode.")

            elif subcommand in ["off", "disable"]:
                if self.agent_manager.mode.value == "off":
                    self._show_info("Agent mode is already disabled.")
                    return

                # Check if there are active operations
                status = self.agent_manager.get_status()
                active_count = status["operations"]["in_progress"]

                if active_count > 0:

                    if not Confirm.ask(
                        f"There are {active_count} active operations. Disable anyway?"
                    ):
                        self._show_info("Agent mode remains enabled.")
                        return

                self._show_info("Disabling agent mode...")
                success = await self.agent_manager.disable_agent_mode(
                    wait_for_completion=True
                )

                if success:
                    self.console.print(
                        Panel(
                            "Agent mode disabled successfully.\n\n"
                            "All operations have been stopped or completed.",
                            title="Agent Mode Disabled",
                            border_style="red",
                        )
                    )

                else:
                    self._show_error("Failed to disable agent mode.")

            elif subcommand == "status":
                status = self.agent_manager.get_status()
                ops = status.get("operations", {})
                in_prog = ops.get("in_progress", 0)
                mode = status.get("mode", "unknown")
                self.console.print(Panel(
                    f"Mode: {mode}\n"
                    f"Operations in progress: {in_prog}",
                    title="Agent Status",
                ))

            elif subcommand == "pause":
                success = self.agent_manager.pause_agent_mode()
                if success:
                    self._show_success(
                        "Agent mode paused. Use '/agent resume' to continue."
                    )
                else:
                    self._show_error(
                        "Failed to pause agent mode. Agent may not be running."
                    )

            elif subcommand == "resume":
                success = self.agent_manager.resume_agent_mode()
                if success:
                    self._show_success("Agent mode resumed.")
                else:
                    self._show_error(
                        "Failed to resume agent mode. Use '/agent on' to enable."
                    )

            else:
                self._show_error(f"Unknown agent subcommand: {subcommand}")
                self._show_info(
                    "Available commands: on, off, status, pause, resume"
                )

        except Exception as e:
            self._show_error(f"Agent command failed: {e}")

    async def _handle_history_command(self, command: Command) -> None:
        """
        Handle history commands.

        Commands:
        - /history - Show recent commands
        - /history recent [count] - Show recent commands (default: 20)
        - /history search <query> - Search command history
        - /history stats - Show history statistics
        - /history clear --confirm - Clear all history
        - /history export <file> [format] - Export history to file
        """
        args = command.args

        if not args:
            # Default to showing recent commands
            args = ["recent"]

        action = args[0].lower()

        try:
            if action == "recent":
                # Show recent commands
                count = 20
                if len(args) > 1:
                    try:
                        count = int(args[1])
                        count = max(1, min(count, 100))  # Limit between 1-100
                    except ValueError:
                        self._show_error(
                            "Invalid count for recent"
                            " commands. Using default"
                            " (20)."
                        )

                recent_commands = self.history_manager.get_recent_commands(count)

                if not recent_commands:
                    self._show_info("No command history available.")
                    return

                count = len(recent_commands)
                table = Table(
                    title=f"Recent Commands (Last {count})"
                )
                table.add_column("Index", style="dim", width=6)
                table.add_column("Time", style="cyan", width=16)
                table.add_column("Command", style="white")

                for i, entry in enumerate(reversed(recent_commands), 1):
                    time_str = entry.datetime.strftime("%m-%d %H:%M:%S")
                    # Truncate long commands
                    cmd_display = (
                        entry.command[:80] + "..."
                        if len(entry.command) > 80
                        else entry.command
                    )
                    table.add_row(str(i), time_str, cmd_display)

                self.console.print(table)

            elif action == "search":
                if len(args) < 2:
                    self._show_error(
                        "Search requires a query. Usage: /history search <query>"
                    )
                    return

                query = " ".join(args[1:])
                results = self.history_manager.search_history(query, limit=30)

                if not results:
                    self._show_info(f"No commands found matching '{query}'.")
                    return

                table = Table(
                    title=f"Search Results for '{query}' ({len(results)} found)"
                )
                table.add_column("Index", style="dim", width=6)
                table.add_column("Time", style="cyan", width=16)
                table.add_column("Command", style="white")

                for i, entry in enumerate(results, 1):
                    time_str = entry.datetime.strftime("%m-%d %H:%M:%S")
                    # Highlight matching text
                    cmd_display = entry.command
                    if len(cmd_display) > 80:
                        cmd_display = cmd_display[:80] + "..."

                    table.add_row(str(i), time_str, cmd_display)

                self.console.print(table)

            elif action == "stats":
                # Show history statistics
                stats = self.history_manager.get_statistics()

                stats_text = f"""Total Commands: {stats['total_commands']}
Current Session: {stats['current_session_commands']}
Unique Commands: {stats['unique_commands']}
Sessions Tracked: {stats.get('sessions', 1)}

Oldest Entry: {stats['oldest_entry'] or 'None'}
Newest Entry: {stats['newest_entry'] or 'None'}"""

                self.console.print(
                    Panel(
                        stats_text,
                        title="Command History Statistics",
                        border_style="blue",
                    )
                )

            elif action == "clear":
                # Clear history with confirmation
                if "--confirm" not in args:
                    self._show_info(
                        "To clear command history, use: /history clear --confirm"
                    )
                    self._show_info("This action cannot be undone!")
                    return

                success = self.history_manager.clear_history(confirm=True)
                if success:
                    self._show_success("Command history cleared successfully.")
                else:
                    self._show_error("Failed to clear command history.")

            elif action == "export":
                if len(args) < 2:
                    self._show_error(
                        "Export requires a filename."
                        " Usage: /history export"
                        " <file> [format]"
                    )
                    return

                filename = args[1]
                format_type = args[2] if len(args) > 2 else "json"

                if format_type not in ["json", "txt"]:
                    self._show_error(
                        "Format must be 'json' or 'txt'. Defaulting to 'json'."
                    )
                    format_type = "json"

                success = self.history_manager.export_history(filename, format_type)
                if success:
                    self._show_success(
                        f"History exported to {filename} in {format_type} format."
                    )
                else:
                    self._show_error(f"Failed to export history to {filename}.")

            else:
                self._show_error(f"Unknown history action: {action}")
                self._show_info(
                    "Available actions: recent, search, stats, clear, export"
                )

        except Exception as e:
            self._show_error(f"History command error: {e}")

    # Completion methods are in cli/completion.py (CompletionMixin)

    async def _handle_add_model_command(self, command: Command) -> None:
        """Handle the add-model command."""
        try:

            args = command.args
            if len(args) < 2:
                self._show_error(
                    "Add-model command requires at least model ID and provider"
                )
                return

            model_name = args[0]
            provider = args[1]

            # Find where key=value parameters start
            param_start = 2
            description_parts = []

            for i, arg in enumerate(args[2:], 2):
                if "=" in arg:
                    param_start = i
                    break
                description_parts.append(arg)

            if description_parts:
                description = " ".join(description_parts).strip(
                    "\"'"
                )  # Remove quotes if present
            else:
                description = f"Custom {model_name} model"

            # Parse optional parameters
            max_tokens = 4096
            cost_input = 1.0
            cost_output = 3.0
            swe_score = 50.0
            supports_tools = False
            supports_multimodal = False
            is_free = False

            # Parse additional arguments as key=value pairs
            for arg in args[param_start:]:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = key.lower()

                    try:
                        if key == "max_tokens":
                            max_tokens = int(value)
                        elif key == "cost_input":
                            cost_input = float(value)
                        elif key == "cost_output":
                            cost_output = float(value)
                        elif key == "swe_score":
                            swe_score = float(value)
                        elif key == "supports_tools":
                            supports_tools = value.lower() in [
                                "true",
                                "yes",
                                "1",
                            ]
                        elif key == "supports_multimodal":
                            supports_multimodal = value.lower() in [
                                "true",
                                "yes",
                                "1",
                            ]
                        elif key == "is_free":
                            is_free = value.lower() in ["true", "yes", "1"]
                    except ValueError:
                        self._show_warning(f"Invalid value for {key}: {value}")

            # Create enhanced model info
            model_info = EnhancedModelInfo(
                name=model_name,
                provider=provider,
                description=description,
                max_tokens=max_tokens,
                cost_per_million_input=cost_input,
                cost_per_million_output=cost_output,
                swe_score=swe_score,
                available=True,
                supports_tools=supports_tools,
                supports_multimodal=supports_multimodal,
                latest_version=False,
                deprecated=False,
                release_date=datetime.now(),
                context_window=max_tokens,
                is_free=is_free,
            )

            # Update SWE rating
            model_info.update_swe_rating()

            # Add to configuration
            self.engine.config_manager.add_custom_model(model_info)

            self._show_success(
                f"Added custom model '{model_name}' for provider '{provider}'"
            )

        except Exception as e:
            self._show_error(f"Failed to add custom model: {e}")

    async def _handle_remove_model_command(self, command: Command) -> None:
        """Handle the remove-model command."""
        try:
            args = command.args
            if len(args) != 2:
                self._show_error("Remove-model command requires model ID and provider")
                return

            model_name = args[0]
            provider = args[1]

            # Remove from configuration
            success = self.engine.config_manager.remove_custom_model(
                model_name, provider
            )

            if success:
                self._show_success(
                    f"Removed custom model '{model_name}' from provider '{provider}'"
                )
            else:
                self._show_error(
                    f"Custom model '{model_name}' for provider '{provider}' not found"
                )

        except Exception as e:
            self._show_error(f"Failed to remove custom model: {e}")

    async def _handle_list_custom_models_command(self, command: Command) -> None:
        """Handle the list-custom-models command."""
        try:

            custom_models = self.engine.config_manager.get_custom_models()

            if not custom_models:
                self._show_info("No custom models configured")
                return

            # Create table for custom models
            table = Table(
                title="Custom Models",
                show_header=True,
                header_style="bold blue",
            )
            table.add_column("Model", style="cyan", width=25)
            table.add_column("Provider", style="green", width=15)
            table.add_column("Description", style="white", width=35)
            table.add_column("Tokens", justify="right", style="yellow", width=8)
            table.add_column("Cost/1M", justify="right", style="red", width=12)
            table.add_column("SWE", justify="center", style="magenta", width=6)
            table.add_column("Tools", justify="center", style="blue", width=6)
            table.add_column("MM", justify="center", style="purple", width=4)

            for model in custom_models:
                cost_in = model.cost_per_million_input
                cost_out = model.cost_per_million_output
                cost_display = (
                    f"${cost_in:.1f}/${cost_out:.1f}"
                )
                swe_display = f"{model.swe_score:.1f}" if model.swe_score else "N/A"
                tools_display = "✓" if model.supports_tools else "✗"
                mm_display = "✓" if model.supports_multimodal else "✗"

                table.add_row(
                    model.name,
                    model.provider,
                    (
                        model.description[:35] + "..."
                        if len(model.description) > 35
                        else model.description
                    ),
                    f"{model.max_tokens:,}",
                    cost_display,
                    swe_display,
                    tools_display,
                    mm_display,
                )

            # Display in a panel
            models_panel = Panel(
                table,
                title="🎯 Custom Models Configuration",
                subtitle=f"Total: {len(custom_models)} custom models",
                border_style="blue",
            )

            self.console.print(models_panel)

        except Exception as e:
            self._show_error(f"Failed to list custom models: {e}")
