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
from ..core.models import EnhancedModelInfo, FallbackConfig
from ..core.prompt_enhancer import PROFILES as ENHANCE_PROFILES
from ..core.prompt_enhancer import enhance as enhance_prompt
from ..providers.factory import ProviderFactory

# Internal imports - CLI
from .commands import Command, SlashCommand

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rich.console import Console

    from ..core.agent_mode_manager import AgentModeManager
    from ..core.history_manager import HistoryManager
    from .approval_integration import CLIApprovalIntegration
    from .completion import CompletionManager
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
    completion_manager: "CompletionManager"

    # Methods provided by DisplayMixin (or the host class)
    def _show_error(self, message: str) -> None: ...
    def _show_info(self, message: str) -> None: ...
    def _show_success(self, message: str) -> None: ...
    def _show_warning(self, message: str) -> None: ...
    def _show_help(self) -> None: ...
    def _show_command_help(self, command_name: str) -> None: ...
    def _show_status(self) -> None: ...
    def _clear_screen(self) -> None: ...
    async def _handle_chat_message(self, command: "Command") -> None: ...
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
        elif slash_cmd == SlashCommand.MODEL:
            await self._handle_model_command(command.args)
        elif slash_cmd == SlashCommand.MODELS:
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
        elif slash_cmd == SlashCommand.HOOKS:
            await self._handle_hooks_command(command)
        elif slash_cmd == SlashCommand.PERMISSIONS:
            await self._handle_permissions_command(command)
        elif slash_cmd == SlashCommand.ACCEPT:
            await self._handle_accept_command(command.args)
        elif slash_cmd == SlashCommand.ENHANCE:
            await self._handle_enhance_command(command.args)
        elif slash_cmd == SlashCommand.PROMPTS:
            await self._handle_prompts_command(command)
        elif slash_cmd == SlashCommand.SUBAGENTS:
            await self._handle_subagents_command(command)
        elif slash_cmd == SlashCommand.FALLBACK:
            await self._handle_fallback_command(command)
        elif slash_cmd is not None:
            self._show_info(f"Command {slash_cmd.value} is not yet implemented")

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

    async def _refresh_model_catalogs(self, provider_name: Optional[str]) -> None:
        """'/models refresh [provider]' — pull live catalogs from endpoints.

        Assigns each provider's fetch_enhanced_models() result to its
        _catalog_models (preferred by get_available_models), so served
        context sizes (e.g. vLLM max_model_len) replace static guesses.
        """
        providers = self.engine.providers or {}
        if provider_name is not None:
            key = self._resolve_provider_key(provider_name)
            if key not in providers:
                self._show_error(
                    f"Provider '{provider_name}' is not configured. "
                    f"Configured: {', '.join(sorted(providers)) or '(none)'}"
                )
                return
            targets = {key: providers[key]}
        else:
            targets = dict(providers)

        for name, provider in targets.items():
            try:
                enhanced = await provider.fetch_enhanced_models()
            except Exception as e:
                self._show_error(f"{name}: refresh failed ({e})")
                continue
            if enhanced:
                provider._catalog_models = enhanced
                self._show_success(f"{name}: {len(enhanced)} models")
            else:
                # An empty fetch (endpoint down, no /models) must not
                # clobber a previously good catalog.
                self._show_info(f"{name}: no models returned; catalog kept")

    async def _show_models(self, command: Command) -> None:
        """Show available models with enhanced provider grouping and capabilities."""
        try:
            args = command.args
            filter_type = args[0].lower() if len(args) > 0 else None
            filter_value = args[1] if len(args) > 1 else None

            if filter_type == "refresh":
                await self._refresh_model_catalogs(filter_value)
                return

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

            # Merge in custom models (added via /add-model or /switch) —
            # they live in config, not in any provider's static catalog.
            try:
                for custom in self.engine.config_manager.get_custom_models():
                    bucket = all_models.setdefault(custom.provider, [])
                    if not any(m.name == custom.name for m in bucket):
                        bucket.append(custom)
            except Exception:
                pass

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

    def _register_model_on_the_fly(self, provider_name: str, model_name: str) -> None:
        """Register an uncataloged model as a custom model so /switch works.

        The entry persists in config, shows up in /models, and can be removed
        with /remove-model if the endpoint turns out to reject it.
        """
        model_info = EnhancedModelInfo(
            name=model_name,
            provider=provider_name,
            description="Added on the fly via /switch",
            max_tokens=4096,
            cost_per_million_input=0.0,
            cost_per_million_output=0.0,
            swe_score=50.0,
            available=True,
            supports_tools=True,
            supports_multimodal=False,
            latest_version=False,
            deprecated=False,
            release_date=datetime.now(),
            context_window=4096,
            is_free=False,
        )
        self.engine.config_manager.add_custom_model(model_info)
        self._show_warning(
            f"Model '{model_name}' isn't in the {provider_name} catalog — "
            "registered it as a custom model and switching anyway. "
            f"(/remove-model {model_name} {provider_name} to undo)"
        )

    async def _prompt_model_selection(self) -> str:
        """One-line selection prompt; empty string cancels."""
        message = "Select model (number or name, Enter to cancel): "
        prompt_input = getattr(self, "prompt_input", None)
        if prompt_input is not None:
            try:
                return str(await prompt_input.prompt_async(message)).strip()
            except (EOFError, KeyboardInterrupt):
                return ""
        try:
            return (await asyncio.to_thread(input, message)).strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def _current_provider_key(self) -> Optional[str]:
        """Alias/config name of the active provider, by IDENTITY against
        engine.providers. (get_conversation_summary carries no provider
        key — reading one there silently yielded None.)"""
        current = getattr(self.engine, "current_provider", None)
        if current is None:
            return None
        providers = getattr(self.engine, "providers", None) or {}
        for name, provider in providers.items():
            if provider is current:
                return str(name)
        return None

    async def _handle_model_command(self, args: List[str]) -> None:
        """'/model' — picker; '/model <name>' — set it on the current
        provider. (Filters live on /models.)"""
        provider_name = self._current_provider_key()
        if not provider_name:
            self._show_error("No active provider — use /switch first.")
            return

        choices = self.completion_manager.model_names(provider_name, "")

        if args:
            model = args[0]
            if model not in choices:
                self._show_error(
                    f"'{model}' is not in '{provider_name}''s catalog. "
                    f"Use '/switch {provider_name} {model}' to register "
                    "it on the fly, or '/models refresh "
                    f"{provider_name}' to pull the live list."
                )
                return
            await self.engine.switch_model(provider_name, model)
            self._show_success(f"Model set to {model} ({provider_name}).")
            return

        if not choices:
            self._show_info(
                f"No models known for '{provider_name}'. "
                f"Try '/models refresh {provider_name}'."
            )
            return

        table = Table(title=f"Models — {provider_name}")
        table.add_column("#", style="bold", justify="right")
        table.add_column("Model", style="cyan")
        for i, name in enumerate(choices, 1):
            table.add_row(str(i), name)
        self.console.print(table)

        selection = await self._prompt_model_selection()
        if not selection:
            return

        if selection.isdigit() and 1 <= int(selection) <= len(choices):
            model = choices[int(selection) - 1]
        elif selection in choices:
            model = selection
        else:
            self._show_error(f"'{selection}' is not in the list.")
            return

        await self.engine.switch_model(provider_name, model)
        self._show_success(f"Model set to {model} ({provider_name}).")

    def _default_enhance_profile(self) -> str:
        """Configured default enhancement profile, falling back to 'code'."""
        try:
            settings = getattr(
                self.engine.config_manager.get_config(), "enhancement", None
            )
            profile = getattr(settings, "default_profile", "code")
            return profile if profile in ENHANCE_PROFILES else "code"
        except Exception:
            return "code"

    async def _prompt_enhance_confirm(self) -> str:
        """y/n confirmation for /enhance (the e: prefix skips this)."""
        message = "Send enhanced prompt? [y/n]: "
        prompt_input = getattr(self, "prompt_input", None)
        if prompt_input is not None:
            try:
                return str(await prompt_input.prompt_async(message)).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "n"
        try:
            return (await asyncio.to_thread(input, message)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "n"

    async def _handle_enhance_command(self, args: List[str]) -> None:
        """'/enhance [chat|code|image|research] <draft>' — rewrite the draft
        with the configured enhancement model, show it, confirm, send."""
        if not args:
            self._show_error("Usage: /enhance [chat|code|image|research] <draft>")
            return

        if args[0].lower() in ENHANCE_PROFILES:
            profile = args[0].lower()
            draft = " ".join(args[1:]).strip()
        else:
            profile = self._default_enhance_profile()
            draft = " ".join(args).strip()
        if not draft:
            self._show_error("Provide a draft to enhance.")
            return

        enhanced, ok = await enhance_prompt(draft, profile, self.engine.config_manager)
        if not ok:
            self._show_error(
                "Enhancement failed (is the enhancement provider reachable?). "
                "Draft not sent."
            )
            return

        self.console.print(
            Panel(
                enhanced,
                title=f"Enhanced prompt ({profile})",
                border_style="magenta",
            )
        )
        confirm = await self._prompt_enhance_confirm()
        if confirm not in ("y", "yes"):
            self._show_info("Not sent.")
            return

        await self._handle_chat_message(Command.create_chat_message(enhanced))

    async def _handle_accept_command(self, args: List[str]) -> None:
        """'/accept [edits|all|off]' — session approval mode.

        Bare '/accept' cycles normal → accept-edits → accept-all.
        """
        from .approval_integration import ApprovalMode

        integration = getattr(self, "approval_integration", None)
        if integration is None:
            self._show_error(
                "Approval integration is not initialized; "
                "/accept has nothing to configure."
            )
            return

        if not args:
            mode = integration.cycle_approval_mode()
        else:
            arg_to_mode = {
                "edits": ApprovalMode.ACCEPT_EDITS,
                "all": ApprovalMode.ACCEPT_ALL,
                "off": ApprovalMode.NORMAL,
                "normal": ApprovalMode.NORMAL,
            }
            mode = arg_to_mode.get(args[0].lower())
            if mode is None:
                self._show_error(
                    f"Unknown mode '{args[0]}'. Use /accept [edits|all|off]."
                )
                return
            integration.session_approval_mode = mode

        descriptions = {
            ApprovalMode.NORMAL: "normal — every operation prompts as usual",
            ApprovalMode.ACCEPT_EDITS: (
                "accept-edits — file writes auto-approve; deletes and "
                "commands still prompt"
            ),
            ApprovalMode.ACCEPT_ALL: (
                "accept-all — everything auto-approves (deny/ask permission "
                "rules and hard security limits still apply)"
            ),
        }
        self._show_success(f"Approval mode: {descriptions[mode]}")

    def _resolve_provider_key(self, name: str) -> str:
        """Resolve a user-typed provider name to its configured key.

        Config keys are matched case-insensitively so an entry like
        'MyGateway' stays reachable; unknown names fall back to lowercase
        (the historical behavior for registered provider names).
        """
        providers: dict = getattr(self.engine, "providers", None) or {}
        for key in providers:
            if str(key).lower() == name.lower():
                return str(key)
        return name.lower()

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

        provider_name = self._resolve_provider_key(args[0])
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
                        # Static catalogs are forever stale and the endpoint
                        # accepts any model string — register it on the fly
                        # instead of refusing.
                        self._register_model_on_the_fly(provider_name, model_name)

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
            info = self.engine.get_conversation_summary()
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
                # Join the remainder so values may contain spaces.
                await self._handle_config_set(args[1], " ".join(args[2:]))
            else:
                self._show_error("Usage: /config set <key> <value>")
        elif subcommand == "get":
            if len(args) >= 2:
                await self._handle_config_get(args[1])
            else:
                self._show_error("Usage: /config get <key>")
        elif subcommand in ("set-provider", "add-provider"):
            await self._handle_config_set_provider(args[1:])
        elif subcommand in ("remove-provider", "delete-provider"):
            await self._handle_config_remove_provider(args[1:])
        else:
            self._show_config_help()

    async def _show_config(self) -> None:
        """Show current configuration."""
        try:
            config_info = self.engine.get_current_config()

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
        """Handle config set operation.

        Supported keys:
          - ``default_provider``
          - ``providers.<name>.<field>`` (e.g. providers.openai.api_key,
            providers.digitalocean.base_url, providers.claude.model)
        """
        try:
            config_manager = self.engine.config_manager

            if key == "default_provider":
                config_manager.set_default_provider(value)
                self._show_success(f"default_provider set to '{value}'")
            elif key.startswith("providers."):
                parts = key.split(".")
                if len(parts) != 3:
                    self._show_error(
                        "Usage: /config set providers.<name>.<field> <value>"
                    )
                    return
                _, provider_name, field = parts
                self._set_provider_field(provider_name, field, value)
            else:
                self._show_error(
                    f"Unsupported config key: '{key}'. Use 'default_provider' or "
                    "'providers.<name>.<field>'."
                )
                return

            self._show_info(
                "Restart or switch providers (/provider) for changes to take effect."
            )
        except Exception as e:
            self._show_error(f"Failed to set configuration: {e}")

    def _coerce_provider_value(self, field: str, value: str) -> Any:
        """Coerce a string value to the type declared on ProviderConfig."""
        import typing

        from ..core.models import ProviderConfig

        field_info = ProviderConfig.model_fields.get(field)
        if field_info is None:
            valid = ", ".join(sorted(ProviderConfig.model_fields))
            raise ValueError(
                f"Unknown provider setting '{field}'. Valid settings: {valid}"
            )

        annotation = field_info.annotation
        types = set(typing.get_args(annotation)) or {annotation}
        if bool in types:
            return value.strip().lower() in ("1", "true", "yes", "on")
        if int in types:
            return int(value)
        if float in types:
            return float(value)
        return value

    def _set_provider_field(self, provider_name: str, field: str, value: str) -> None:
        """Set a single field on a provider's configuration (creating it if needed)."""
        from ..core.models import ProviderConfig

        config_manager = self.engine.config_manager

        if field == "api_key":
            # set_api_key encrypts and creates a minimal entry if missing.
            config_manager.set_api_key(provider_name, value)
            self._show_success(f"api_key set for provider '{provider_name}'")
            return

        coerced = self._coerce_provider_value(field, value)
        existing = config_manager.get_provider_config(provider_name)
        data = existing.model_dump() if existing else {"model": ""}
        data[field] = coerced
        new_config = ProviderConfig(**data)
        config_manager.set_provider_config(provider_name, new_config)
        self._show_success(f"{field} set to '{value}' for provider '{provider_name}'")

    async def _handle_config_set_provider(self, args: List[str]) -> None:
        """Handle '/config set-provider <name> [--type T] [--api-key K]
        [--base-url U] [--model M]'."""
        if not args:
            self._show_error(
                "Usage: /config set-provider <name> [--type TYPE] "
                "[--api-key KEY] [--base-url URL] [--model MODEL]"
            )
            return

        provider_name = args[0]
        flags = {
            "--api-key": "api_key",
            "--base-url": "base_url",
            "--model": "model",
            "--type": "provider_type",
        }
        values: dict = {}
        i = 1
        while i < len(args):
            token = args[i]
            if token in flags and i + 1 < len(args):
                values[flags[token]] = args[i + 1]
                i += 2
            else:
                self._show_error(f"Unrecognized or incomplete option: {token}")
                return

        if not values:
            self._show_error(
                "Provide at least one of --type, --api-key, --base-url, " "or --model."
            )
            return

        try:
            from ..core.models import ProviderConfig

            config_manager = self.engine.config_manager
            registered = sorted(ProviderFactory.get_available_providers())
            existing = config_manager.get_provider_config(provider_name)

            # Validate everything BEFORE any write so a bad invocation
            # persists nothing (set_api_key would create a partial entry).
            provider_type = values.get("provider_type") or (
                existing.provider_type if existing else None
            )
            if provider_type and provider_type not in registered:
                self._show_error(
                    f"Unknown provider type '{provider_type}'. "
                    f"Registered types: {', '.join(registered)}"
                )
                return
            if (
                existing is None
                and provider_name not in registered
                and not provider_type
            ):
                self._show_error(
                    f"'{provider_name}' is not a registered provider. To "
                    "configure a custom endpoint under this name, add "
                    "--type <provider_type> (e.g. --type openai-compatible). "
                    f"Registered types: {', '.join(registered)}"
                )
                return
            if existing is None and not values.get("model"):
                self._show_error(
                    "New provider entries need --model <name> — an empty "
                    "model is never valid."
                )
                return

            # API key first (encrypts; creates a minimal entry if needed).
            if "api_key" in values:
                config_manager.set_api_key(provider_name, values["api_key"])

            existing = config_manager.get_provider_config(provider_name)
            data = existing.model_dump() if existing else {"model": ""}
            for field in ("base_url", "model", "provider_type"):
                if field in values:
                    data[field] = values[field]
            config_manager.set_provider_config(provider_name, ProviderConfig(**data))

            fields = ", ".join(sorted(values))
            self._show_success(f"Provider '{provider_name}' configured ({fields}).")
            self._show_info(
                "Use '/config set default_provider "
                f"{provider_name}' to make it the default."
            )
        except Exception as e:
            self._show_error(f"Failed to configure provider: {e}")

    # ------------------------------------------------------------------ hooks

    _HOOK_EVENTS = (
        "pre_send_message",
        "post_send_message",
        "tool_use_request",
        "post_tool",
    )

    async def _handle_hooks_command(self, command: Command) -> None:
        """Handle '/hooks [list|add|remove|on|off]'."""
        args = command.args
        config = self.engine.config_manager.get_config()
        hooks = config.hooks

        if not args or args[0].lower() == "list":
            self._show_hooks(hooks)
            return

        sub = args[0].lower()
        if sub in ("on", "off"):
            hooks.enabled = sub == "on"
            self.engine.config_manager.save_config()
            self._show_success(f"Hooks {'enabled' if hooks.enabled else 'disabled'}.")
        elif sub == "add":
            self._hooks_add(args[1:])
        elif sub == "remove":
            self._hooks_remove(args[1:])
        else:
            self._show_error(
                "Usage: /hooks [list | on | off | "
                "add <event> <name> [--matcher RE] [--blocking] "
                "[--timeout N] <command> | remove <event> <name>]"
            )

    def _hooks_add(self, args: List[str]) -> None:
        if len(args) < 3:
            self._show_error(
                "Usage: /hooks add <event> <name> [--matcher RE] "
                "[--blocking] [--timeout N] <command>"
            )
            return
        event, name, rest = args[0], args[1], args[2:]
        if event not in self._HOOK_EVENTS:
            self._show_error(
                f"Unknown event '{event}'. One of: {', '.join(self._HOOK_EVENTS)}"
            )
            return

        matcher: Optional[str] = None
        blocking = False
        timeout = 30
        cmd_parts: List[str] = []
        i = 0
        while i < len(rest):
            token = rest[i]
            if token == "--matcher" and i + 1 < len(rest):
                matcher = rest[i + 1]
                i += 2
            elif token == "--blocking":
                blocking = True
                i += 1
            elif token == "--timeout" and i + 1 < len(rest):
                try:
                    timeout = int(rest[i + 1])
                except ValueError:
                    self._show_error("--timeout must be an integer.")
                    return
                i += 2
            else:
                cmd_parts.append(token)
                i += 1

        command_str = " ".join(cmd_parts)
        if not command_str:
            self._show_error("A command is required.")
            return

        try:
            from ..core.models import HookCommand

            config = self.engine.config_manager.get_config()
            hook = HookCommand(
                name=name,
                command=command_str,
                matcher=matcher,
                blocking=blocking,
                timeout=timeout,
            )
            getattr(config.hooks, event).append(hook)
            self.engine.config_manager.save_config()
            self._show_success(f"Added {event} hook '{name}'.")
        except Exception as e:
            self._show_error(f"Failed to add hook: {e}")

    def _hooks_remove(self, args: List[str]) -> None:
        if len(args) < 2:
            self._show_error("Usage: /hooks remove <event> <name>")
            return
        event, name = args[0], args[1]
        if event not in self._HOOK_EVENTS:
            self._show_error(f"Unknown event '{event}'.")
            return
        config = self.engine.config_manager.get_config()
        hook_list = getattr(config.hooks, event)
        for idx, hook in enumerate(hook_list):
            if hook.name == name:
                hook_list.pop(idx)
                self.engine.config_manager.save_config()
                self._show_success(f"Removed {event} hook '{name}'.")
                return
        self._show_error(f"No {event} hook named '{name}'.")

    def _show_hooks(self, hooks: Any) -> None:
        state = "enabled" if hooks.enabled else "disabled"
        table = Table(title=f"Hooks ({state})")
        table.add_column("Event", style="bold")
        table.add_column("Name", style="cyan")
        table.add_column("Command")
        table.add_column("Matcher", style="magenta")
        table.add_column("Blocking", justify="center")
        any_rows = False
        for event in self._HOOK_EVENTS:
            for hook in getattr(hooks, event):
                any_rows = True
                table.add_row(
                    event,
                    hook.name,
                    hook.command,
                    hook.matcher or "—",
                    "yes" if hook.blocking else "no",
                )
        if any_rows:
            self.console.print(table)
        else:
            self._show_info("No hooks configured. Add one with '/hooks add'.")

    # ------------------------------------------------------------ permissions

    _PERM_LISTS = {
        "allow": "always_allow",
        "deny": "always_deny",
        "ask": "always_ask",
    }

    async def _handle_permissions_command(self, command: Command) -> None:
        """Handle '/permissions [list|allow|deny|ask|remove|on|off]'."""
        args = command.args
        config = self.engine.config_manager.get_config()
        perms = config.permissions

        if not args or args[0].lower() == "list":
            self._show_permissions(perms)
            return

        sub = args[0].lower()
        if sub in ("on", "off"):
            perms.enabled = sub == "on"
            self.engine.config_manager.save_config()
            self._show_success(
                f"Permission rules {'enabled' if perms.enabled else 'disabled'}."
            )
        elif sub in self._PERM_LISTS:
            self._permissions_add(sub, args[1:])
        elif sub == "remove":
            self._permissions_remove(args[1:])
        else:
            self._show_error(
                "Usage: /permissions [list | on | off | "
                "<allow|deny|ask> <tool> [matcher] | "
                "remove <allow|deny|ask> <index>]"
            )

    def _permissions_add(self, kind: str, args: List[str]) -> None:
        if not args:
            self._show_error(
                f"Usage: /permissions {kind} <tool> [matcher]  "
                "(tool is an operation type like file_write, "
                "command_execute, or '*')"
            )
            return
        tool = args[0]
        matcher = args[1] if len(args) > 1 else None
        try:
            from ..core.models import PermissionRule

            config = self.engine.config_manager.get_config()
            rule = PermissionRule(tool=tool, matcher=matcher)
            getattr(config.permissions, self._PERM_LISTS[kind]).append(rule)
            self.engine.config_manager.save_config()
            target = f" matching /{matcher}/" if matcher else ""
            self._show_success(f"Added {kind} rule for '{tool}'{target}.")
        except Exception as e:
            self._show_error(f"Failed to add rule: {e}")

    def _permissions_remove(self, args: List[str]) -> None:
        if len(args) < 2 or args[0].lower() not in self._PERM_LISTS:
            self._show_error("Usage: /permissions remove <allow|deny|ask> <index>")
            return
        kind = args[0].lower()
        try:
            idx = int(args[1]) - 1
        except ValueError:
            self._show_error("Index must be a number (see '/permissions list').")
            return
        config = self.engine.config_manager.get_config()
        rule_list = getattr(config.permissions, self._PERM_LISTS[kind])
        if not (0 <= idx < len(rule_list)):
            self._show_error(f"No {kind} rule at index {idx + 1}.")
            return
        removed = rule_list.pop(idx)
        self.engine.config_manager.save_config()
        self._show_success(f"Removed {kind} rule for '{removed.tool}'.")

    def _show_permissions(self, perms: Any) -> None:
        state = "enabled" if perms.enabled else "disabled"
        table = Table(title=f"Permission rules ({state})")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Decision", style="bold")
        table.add_column("Tool", style="cyan")
        table.add_column("Matcher", style="magenta")
        any_rows = False
        for kind, attr in self._PERM_LISTS.items():
            for idx, rule in enumerate(getattr(perms, attr), start=1):
                any_rows = True
                table.add_row(str(idx), kind, rule.tool, rule.matcher or "—")
        if any_rows:
            self.console.print(table)
        else:
            self._show_info(
                "No permission rules. Add one with "
                "'/permissions deny|ask|allow <tool> [matcher]'."
            )

    # ------------------------------------------------------------ MCP prompts

    async def _handle_prompts_command(self, command: Command) -> None:
        """Handle '/prompts [list | <name> [key=value ...]]'.

        Lists prompts exposed by connected MCP servers, or renders one and shows
        the result so it can be used as a prompt.
        """
        manager = getattr(self.engine, "mcp_manager", None)
        if not manager:
            self._show_info("No MCP servers configured.")
            return

        args = command.args
        if not args or args[0].lower() == "list":
            try:
                prompts = await manager.get_available_prompts()
            except Exception as e:
                self._show_error(f"Failed to list MCP prompts: {e}")
                return
            self._show_mcp_prompts(prompts)
            return

        name = args[0]
        arguments: dict = {}
        for token in args[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                arguments[key] = value
            else:
                self._show_error(f"Prompt arguments must be key=value (got '{token}').")
                return
        try:
            rendered = await manager.get_prompt(name, arguments)
        except Exception as e:
            self._show_error(f"Failed to render prompt '{name}': {e}")
            return
        self.console.print(Panel(rendered, title=f"MCP prompt: {name}"))

    def _show_mcp_prompts(self, prompts: List[dict]) -> None:
        if not prompts:
            self._show_info("No MCP prompts available from connected servers.")
            return
        table = Table(title="MCP prompts")
        table.add_column("Name", style="cyan")
        table.add_column("Server", style="dim")
        table.add_column("Arguments", style="magenta")
        table.add_column("Description")
        for p in prompts:
            arg_names = ", ".join(a.get("name", "") for a in (p.get("arguments") or []))
            table.add_row(
                p.get("name", ""),
                p.get("server", ""),
                arg_names or "—",
                p.get("description") or "",
            )
        self.console.print(table)

    # -------------------------------------------------------------- subagents

    async def _handle_subagents_command(self, command: Command) -> None:
        """Handle '/subagents [list | run <name> <task...>]'."""
        config = self.engine.config_manager.get_config()
        subagents = getattr(config, "subagents", {}) or {}
        args = command.args

        if not args or args[0].lower() == "list":
            self._show_subagents(subagents)
            return

        sub = args[0].lower()
        if sub == "run":
            if len(args) < 3:
                self._show_error("Usage: /subagents run <name> <task>")
                return
            name = args[1]
            task = " ".join(args[2:])
            definition = subagents.get(name)
            if definition is None:
                self._show_error(f"No subagent named '{name}'. See '/subagents list'.")
                return
            from .subagent import SubAgentRunner

            self._show_info(f"Running subagent '{name}'…")
            result = await SubAgentRunner(self.engine).run(definition, task)
            if not result.success:
                self._show_error(f"Subagent '{name}' failed: {result.error}")
                return
            tools_used = ", ".join(result.tool_calls) or "none"
            self.console.print(
                Panel(
                    result.output or "(no output)",
                    title=(
                        f"Subagent: {name} "
                        f"({result.iterations} turn(s), tools: {tools_used})"
                    ),
                )
            )
        else:
            self._show_error("Usage: /subagents [list | run <name> <task>]")

    def _show_subagents(self, subagents: dict) -> None:
        if not subagents:
            self._show_info(
                "No subagents configured. Define them under 'subagents' in config."
            )
            return
        table = Table(title="Subagents")
        table.add_column("Name", style="cyan")
        table.add_column("Model", style="dim")
        table.add_column("Tools", style="magenta")
        table.add_column("Description")
        for name, defn in subagents.items():
            tools = "all" if defn.tools is None else (", ".join(defn.tools) or "none")
            table.add_row(name, defn.model or "(inherit)", tools, defn.description)
        self.console.print(table)

    async def _handle_config_remove_provider(self, args: List[str]) -> None:
        """Handle '/config remove-provider <name>'."""
        if not args:
            self._show_error("Usage: /config remove-provider <name>")
            return

        provider_name = args[0]
        try:
            config_manager = self.engine.config_manager
            config = config_manager.get_config()

            if provider_name not in config.providers:
                self._show_error(f"Provider '{provider_name}' is not configured.")
                return

            del config.providers[provider_name]

            # Repoint the default provider if it referenced the removed one.
            if config.default_provider == provider_name:
                if config.providers:
                    config.default_provider = next(iter(config.providers))
                    self._show_warning(
                        "Default provider was removed; now set to "
                        f"'{config.default_provider}'."
                    )
                else:
                    self._show_warning(
                        "Removed the last provider; configure another with "
                        "'/config set-provider'."
                    )

            config_manager.save_config(config)
            self._show_success(f"Provider '{provider_name}' removed.")
        except Exception as e:
            self._show_error(f"Failed to remove provider: {e}")

    async def _handle_config_get(self, key: str) -> None:
        """Handle config get operation."""
        try:
            config_info = self.engine.get_current_config()
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
        from ..providers.factory import ProviderFactory

        endpoints = ", ".join(sorted(ProviderFactory.get_available_providers()))
        help_text = f"""[bold]Configuration Commands:[/bold]

[cyan]/config[/cyan]                              - Show current configuration
[cyan]/config show[/cyan]                         - Show current configuration
[cyan]/config get <key>[/cyan]                    - Get a configuration value
[cyan]/config set default_provider <name>[/cyan] - Set the default provider
[cyan]/config set providers.<name>.<field> <value>[/cyan]
                                    - Set a provider field
                                      (api_key, base_url, model, ...)
[cyan]/config set-provider <name> [--api-key K] [--base-url U] [--model M][/cyan]
                                    - Create/update a provider in one step
[cyan]/config remove-provider <name>[/cyan]       - Remove a provider

[bold]Endpoints[/bold] (each is a provider you can configure):
  {endpoints}

[bold]Examples:[/bold]
  /config set-provider digitalocean --api-key $DO_KEY \\
      --base-url https://inference.do-ai.run/v1 --model llama3.3-70b-instruct
  /config set providers.openai.base_url https://my-proxy/v1
  /config set default_provider claude

[bold]Environment overrides[/bold] (ephemeral, win over the saved config):
  ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY / DIGITALOCEAN_INFERENCE_KEY
  OMNIMANCER_<PROVIDER>_API_KEY / _BASE_URL / _MODEL
  OMNIMANCER_DEFAULT_PROVIDER"""
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
                model = conv.get("current_model", "Unknown")
                lines.append(
                    f"     Messages: {conv['message_count']}," f" Model: {model}"
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
                self.console.print(
                    Panel(
                        f"Mode: {mode}\n" f"Operations in progress: {in_prog}",
                        title="Agent Status",
                    )
                )

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
                self._show_info("Available commands: on, off, status, pause, resume")

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
                table = Table(title=f"Recent Commands (Last {count})")
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
                cost_display = f"${cost_in:.1f}/${cost_out:.1f}"
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

    # ------------------------------------------------------------------
    # /fallback
    # ------------------------------------------------------------------

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _validate_toggle_arg(arg: str) -> Optional[bool]:
        """Return True/False for 'on'/'off', or None if the value is invalid.

        Centralises validation of boolean toggle arguments so individual
        subhandlers don't duplicate the same ``if toggle not in ("on", "off")``
        check.
        """
        if arg == "on":
            return True
        if arg == "off":
            return False
        return None

    def _persist_fallback_config(self, fb: FallbackConfig) -> None:
        """Persist *fb* to disk and immediately sync the engine.

        This single method replaces the repeated trio of:
          ``config.fallback = fb``
          ``self.engine.config_manager.save_config()``
          ``self.engine.configure_fallback()``
        across every subcommand handler.
        """
        config = self.engine.config_manager.get_config()
        config.fallback = fb
        self.engine.config_manager.save_config()
        # Sync the engine's runtime handler immediately so changes take effect
        # without requiring a restart.
        self.engine.configure_fallback()

    # --- dispatcher -------------------------------------------------------

    async def _handle_fallback_command(self, command: Command) -> None:
        """Handle the /fallback command family.

        Sub-commands
        ------------
        /fallback                         show current fallback configuration
        /fallback status                  same as above
        /fallback auto [on|off]           toggle or query auto-fallback
        /fallback order <p1> <p2> …       set the ordered fallback provider list
        /fallback order clear             clear the fallback order (use any)
        /fallback on-rate-limit [on|off]  toggle fallback on 429 errors
        /fallback on-quota [on|off]       toggle fallback on quota errors
        /fallback help                    show this help text
        """
        try:
            config = self.engine.config_manager.get_config()
            # Use a fresh FallbackConfig() default if the config object does
            # not yet have a fallback section (e.g., existing installs).
            fb: FallbackConfig = getattr(config, "fallback", FallbackConfig())

            args = list(command.args) if command.args else []
            sub = args[0].lower() if args else "status"

            dispatch: dict[str, Any] = {
                "status": lambda: self._show_fallback_status(fb),
                "": lambda: self._show_fallback_status(fb),
                "help": self._show_fallback_help,
                "auto": lambda: self._handle_auto_subcommand(fb, args),
                "order": lambda: self._handle_order_subcommand(fb, args),
                "on-rate-limit": lambda: self._handle_on_rate_limit_subcommand(
                    fb, args
                ),
                "on-quota": lambda: self._handle_on_quota_subcommand(fb, args),
            }

            handler = dispatch.get(sub)
            if handler is None:
                self._show_error(
                    f"Unknown sub-command '{sub}'. "
                    "Run [cyan]/fallback help[/cyan] for usage."
                )
            else:
                result = handler()
                # Subhandlers may be coroutines in the future; await if needed.
                if asyncio.iscoroutine(result):
                    await result

        except ValueError as exc:
            # Raised by subhandlers for invalid user input.
            self._show_error(str(exc))
        except Exception as exc:
            logger.debug("Fallback command error", exc_info=True)
            self._show_error(f"Failed to update fallback config: {exc}")

    # --- subhandlers ------------------------------------------------------

    def _handle_auto_subcommand(self, fb: FallbackConfig, args: list[str]) -> None:
        """Handle ``/fallback auto [on|off]``."""
        if len(args) < 2:
            state = "on" if fb.auto_fallback else "off"
            self._show_info(
                f"Auto-fallback is currently [bold]{state}[/bold]. "
                f"Use [cyan]/fallback auto on[/cyan] or "
                f"[cyan]/fallback auto off[/cyan] to change."
            )
            return

        enabled = self._validate_toggle_arg(args[1].lower())
        if enabled is None:
            raise ValueError("Usage: /fallback auto [on|off]")

        fb.auto_fallback = enabled
        self._persist_fallback_config(fb)
        label = "enabled" if enabled else "disabled"
        self._show_success(f"Auto-fallback {label}.")

    def _handle_order_subcommand(self, fb: FallbackConfig, args: list[str]) -> None:
        """Handle ``/fallback order [<p1> <p2> … | clear]``."""
        if len(args) < 2:
            order_str = " → ".join(fb.fallback_order) if fb.fallback_order else "(any)"
            self._show_info(f"Current fallback order: {order_str}")
            return

        if args[1].lower() == "clear":
            fb.fallback_order = []
            self._persist_fallback_config(fb)
            self._show_success(
                "Fallback order cleared (will use any available provider)."
            )
            return

        # Validate that each name is a known provider.
        known = set(self.engine.providers.keys())
        new_order: list[str] = []
        unknown: list[str] = []
        for name in args[1:]:
            if name in known:
                new_order.append(name)
            else:
                unknown.append(name)

        if unknown:
            self._show_warning(
                f"Unknown provider(s) skipped: {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(known)) or 'none'}"
            )
        if not new_order:
            raise ValueError("No valid providers specified. Nothing changed.")

        fb.fallback_order = new_order
        self._persist_fallback_config(fb)
        self._show_success(f"Fallback order set: {' → '.join(new_order)}")

    def _handle_on_rate_limit_subcommand(
        self, fb: FallbackConfig, args: list[str]
    ) -> None:
        """Handle ``/fallback on-rate-limit [on|off]``."""
        if len(args) < 2:
            state = "on" if fb.fallback_on_rate_limit else "off"
            self._show_info(f"Fallback on rate-limit (429) is [bold]{state}[/bold].")
            return

        enabled = self._validate_toggle_arg(args[1].lower())
        if enabled is None:
            raise ValueError("Usage: /fallback on-rate-limit [on|off]")

        fb.fallback_on_rate_limit = enabled
        self._persist_fallback_config(fb)
        label = "enabled" if enabled else "disabled"
        self._show_success(f"Fallback on rate-limit {label}.")

    def _handle_on_quota_subcommand(self, fb: FallbackConfig, args: list[str]) -> None:
        """Handle ``/fallback on-quota [on|off]``."""
        if len(args) < 2:
            state = "on" if fb.fallback_on_quota else "off"
            self._show_info(f"Fallback on quota errors is [bold]{state}[/bold].")
            return

        enabled = self._validate_toggle_arg(args[1].lower())
        if enabled is None:
            raise ValueError("Usage: /fallback on-quota [on|off]")

        fb.fallback_on_quota = enabled
        self._persist_fallback_config(fb)
        label = "enabled" if enabled else "disabled"
        self._show_success(f"Fallback on quota errors {label}.")

    # --- display helpers --------------------------------------------------

    def _show_fallback_status(self, fb: FallbackConfig) -> None:
        """Print a Rich table with the current fallback configuration."""
        table = Table(
            title="Fallback Configuration", show_header=True, header_style="bold cyan"
        )
        table.add_column("Setting", style="bold")
        table.add_column("Value")
        table.add_column("Description", style="dim")

        auto_label = "[green]on[/green]" if fb.auto_fallback else "[yellow]off[/yellow]"
        rl_label = (
            "[green]on[/green]" if fb.fallback_on_rate_limit else "[yellow]off[/yellow]"
        )
        q_label = (
            "[green]on[/green]" if fb.fallback_on_quota else "[yellow]off[/yellow]"
        )
        order_str = (
            " [dim]→[/dim] ".join(fb.fallback_order)
            if fb.fallback_order
            else "[dim](any available)[/dim]"
        )

        table.add_row("auto", auto_label, "Switch without asking")
        table.add_row("on-rate-limit", rl_label, "Fallback on 429 / rate-limit")
        table.add_row("on-quota", q_label, "Fallback on quota exceeded")
        table.add_row("order", order_str, "Ordered list of providers to try")

        self.console.print(table)
        self.console.print(
            "\n[dim]Tip: [cyan]/fallback auto on[/cyan] to auto-switch silently, "
            "[cyan]/fallback order claude openai[/cyan] to set priority.[/dim]"
        )

    def _show_fallback_help(self) -> None:
        """Print /fallback usage."""
        help_text = (
            "[bold cyan]/fallback[/bold cyan] — manage rate-limit fallback "
            "behaviour\n\n"
            "[bold]Sub-commands[/bold]\n"
            "  [cyan]/fallback[/cyan]                         "
            "show current configuration\n"
            "  [cyan]/fallback auto on|off[/cyan]             "
            "auto-switch on rate limit (no prompt)\n"
            "  [cyan]/fallback order <p1> <p2> …[/cyan]       "
            "set ordered fallback providers\n"
            "  [cyan]/fallback order clear[/cyan]              "
            "clear order (use any available)\n"
            "  [cyan]/fallback on-rate-limit on|off[/cyan]    "
            "enable/disable fallback on 429\n"
            "  [cyan]/fallback on-quota on|off[/cyan]         "
            "enable/disable fallback on quota errors\n\n"
            "[bold]Examples[/bold]\n"
            "  [dim]/fallback auto on[/dim]                   "
            "silently switch when rate-limited\n"
            "  [dim]/fallback order claude openai gemini[/dim]  "
            "try in this order\n"
            "  [dim]/fallback order clear[/dim]               "
            "reset to default (any provider)\n\n"
            "When [bold]auto[/bold] is off (default), omnimancer will pause and ask:\n"
            "  [yellow]⚠ Rate limit hit on claude. Fall back to openai? [Y/n][/yellow]"
        )
        self.console.print(Panel(help_text, title="Fallback Help", border_style="cyan"))
