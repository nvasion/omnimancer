"""
Agent CLI Handler for Omnimancer.

This module provides the AgentCLIHandler class that processes /agents commands
and coordinates with the agent engine to expose agent functionality to users.
"""

import logging
import time
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.status import Status
from rich.table import Table

from ...core.agent.approval_manager import EnhancedApprovalManager
from ...core.agent.persona import (
    PersonaManager,
    PersonaStatus,
    get_persona_manager,
)
from ...core.agent_engine import AgentEngine
from ...utils.errors import AgentError

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Agent-specific error."""

    pass


class AgentNotFoundError(AgentError):
    """Agent not found error."""

    pass


class PermissionDeniedError(AgentError):
    """Permission denied error."""

    pass


class AgentBusyError(AgentError):
    """Agent is busy error."""

    pass


class ConfigurationError(AgentError):
    """Configuration error."""

    pass


class AgentCLIHandler:
    """
    Handler class that processes /agents commands and coordinates with the agent engine.

    This class serves as the bridge between CLI commands and the agent engine,
    providing user-friendly interfaces for agent management.
    """

    def __init__(
        self,
        agent_engine: Optional[AgentEngine] = None,
        approval_manager: Optional[EnhancedApprovalManager] = None,
        console: Optional[Console] = None,
        persona_manager: Optional[PersonaManager] = None,
        verbose: bool = False,
    ):
        """
        Initialize the agent CLI handler.

        Args:
            agent_engine: Agent engine instance for coordination
            approval_manager: Enhanced approval manager for permissions
            console: Rich console for output formatting
            persona_manager: Persona manager for agent personality management
            verbose: Enable verbose logging and debugging output
        """
        try:
            self.agent_engine = agent_engine
            self.approval_manager = approval_manager
            self.console = console or Console()
            self.verbose = verbose

            # Initialize persona manager with error handling
            try:
                self.persona_manager = persona_manager or get_persona_manager()
            except Exception as e:
                logger.error(f"Failed to initialize PersonaManager: {e}")
                if self.verbose:
                    self.console.print(
                        f"[yellow]Warning: PersonaManager initialization failed: {e}[/yellow]"
                    )
                raise ConfigurationError(f"Failed to initialize persona system: {e}")

            # Current agent tracking
            self.current_agent = None

            if self.verbose:
                self.console.print(
                    "[dim]Agent CLI Handler initialized successfully[/dim]"
                )

        except Exception as e:
            logger.error(f"Failed to initialize AgentCLIHandler: {e}")
            if console:
                console.print(
                    f"[red]Error: Failed to initialize agent system: {e}[/red]"
                )
            raise ConfigurationError(f"Agent handler initialization failed: {e}")

    async def handle_command(self, args: List[str]) -> None:
        """
        Handle /agents command with subcommands.

        Args:
            args: Command arguments from CLI parser
        """
        if not args:
            # Default to list if no subcommand provided
            args = ["list"]

        subcommand = args[0].lower()

        try:
            if self.verbose:
                self.console.print(
                    f"[dim]Executing agents subcommand: {subcommand}[/dim]"
                )

            if subcommand == "list":
                await self.handle_list(args[1:] if len(args) > 1 else [])
            elif subcommand == "enable":
                await self.handle_enable(args[1:] if len(args) > 1 else [])
            elif subcommand == "disable":
                await self.handle_disable(args[1:] if len(args) > 1 else [])
            elif subcommand == "status":
                await self.handle_status(args[1:] if len(args) > 1 else [])
            elif subcommand == "config":
                await self.handle_config(args[1:] if len(args) > 1 else [])
            elif subcommand == "switch":
                await self.handle_switch(args[1:] if len(args) > 1 else [])
            elif subcommand == "current":
                await self.handle_current()
            elif subcommand == "info":
                await self.handle_info(args[1:] if len(args) > 1 else [])
            else:
                self._show_error_with_suggestions(
                    f"Unknown agents subcommand: '{subcommand}'",
                    f"Did you mean one of: list, enable, disable, status, config, switch, current, info?",
                )
                self._show_help()

        except AgentNotFoundError as e:
            logger.error(f"Agent not found: {e}")
            self._show_error_with_suggestions(
                str(e), "Use '/agents list' to see available agents."
            )
        except PermissionDeniedError as e:
            logger.error(f"Permission denied: {e}")
            self._show_error_with_suggestions(
                f"Permission denied: {e}",
                "Check your user permissions or contact an administrator.",
            )
        except AgentBusyError as e:
            logger.error(f"Agent busy: {e}")
            self._show_error_with_suggestions(
                f"Agent is busy: {e}",
                "Wait for the current operation to complete or use '/agents status' to check progress.",
            )
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            self._show_error_with_suggestions(
                f"Configuration error: {e}",
                "Check your agent configuration and try again.",
            )
        except Exception as e:
            logger.error(f"Agent command failed unexpectedly: {e}", exc_info=True)
            self._show_error_with_suggestions(
                f"Unexpected error occurred: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}",
                "Try the command again or use '/agents status' to check system health.",
            )

            if self.verbose:
                self.console.print(f"[dim]Full error details: {e}[/dim]")

    async def handle_list(self, args: List[str]) -> None:
        """
        Handle 'list' subcommand to show available agents.

        Args:
            args: Additional arguments
        """
        try:
            table = Table(title="Available Agent Personas")
            table.add_column("Agent ID", style="cyan", width=12)
            table.add_column("Name", style="bold", width=18)
            table.add_column("Description", style="white", width=40)
            table.add_column("Provider", style="green", width=12)
            table.add_column("Model", style="blue", width=25)
            table.add_column("Status", style="yellow", width=10)

            # Get personas from PersonaManager
            personas = self.persona_manager.get_all_personas()

            for agent_id, persona in personas.items():
                # Map persona status to display
                if persona.status == PersonaStatus.ACTIVE:
                    status = "🟢 Active"
                elif persona.status == PersonaStatus.AVAILABLE:
                    status = "⚪ Available"
                elif persona.status == PersonaStatus.DISABLED:
                    status = "🔴 Disabled"
                elif persona.status == PersonaStatus.ERROR:
                    status = "❌ Error"
                elif persona.status == PersonaStatus.LOADING:
                    status = "🔄 Loading"
                else:
                    status = "❓ Unknown"

                # Get provider info from configuration
                config = persona.configuration
                provider_name = (
                    config.provider_type.value if config.provider_type else "Unknown"
                )

                table.add_row(
                    agent_id,
                    persona.name,
                    (
                        persona.description[:37] + "..."
                        if len(persona.description) > 40
                        else persona.description
                    ),
                    provider_name,
                    config.model_id or "Unknown",
                    status,
                )

            self.console.print(table)

            # Show current agent if any
            active_persona = self.persona_manager.active_persona
            if active_persona:
                self.console.print(
                    f"\n[bold green]Current active agent:[/bold green] {active_persona.name} ({active_persona.id})"
                )
            else:
                self.console.print(
                    f"\n[dim]No agent currently active. Use '/agents enable <agent_id>' to activate.[/dim]"
                )

        except Exception as e:
            self._show_error(f"Failed to list agents: {e}")

    async def handle_enable(self, args: List[str]) -> None:
        """
        Handle 'enable' subcommand to activate an agent.

        Args:
            args: Agent ID to enable
        """
        if not args:
            self._show_error_with_suggestions(
                "Agent enable requires an agent ID.",
                "Use '/agents list' to see available agents, then try '/agents enable <agent_id>'",
            )
            return

        agent_id = args[0].lower()

        try:
            if self.verbose:
                self.console.print(
                    f"[dim]Checking if agent '{agent_id}' exists...[/dim]"
                )

            # Check if persona exists
            persona = self.persona_manager.get_persona(agent_id)
            if not persona:
                # Suggest similar agents
                available_agents = list(self.persona_manager.get_all_personas().keys())
                suggestions = [
                    agent
                    for agent in available_agents
                    if agent_id in agent or agent in agent_id
                ]

                if suggestions:
                    suggestion_text = f"Did you mean: {', '.join(suggestions)[:50]}?"
                else:
                    suggestion_text = (
                        f"Available agents: {', '.join(available_agents)[:50]}..."
                    )

                raise AgentNotFoundError(
                    f"Agent '{agent_id}' not found. {suggestion_text}"
                )

            # Check if agent is already active
            if persona.is_active:
                self._show_info_with_action(
                    f"Agent '{agent_id}' is already active.",
                    f"Use '/agents status {agent_id}' to see details or '/agents disable {agent_id}' to deactivate.",
                )
                return

            # Check permissions if approval manager is available
            if self.approval_manager:
                try:
                    # This would check if user has permission to enable agents
                    if not await self._check_permission_for_action(
                        "agent_enable", agent_id
                    ):
                        raise PermissionDeniedError(
                            f"You don't have permission to enable agent '{agent_id}'"
                        )
                except Exception as perm_e:
                    if self.verbose:
                        self.console.print(
                            f"[dim]Permission check failed: {perm_e}[/dim]"
                        )
                    # Continue if permission system is not available

            # Show progress indicator for activation
            with Status(
                f"[bold blue]Activating {persona.name}...",
                console=self.console,
            ):
                if self.verbose:
                    self.console.print(
                        f"[dim]Calling PersonaManager.activate_persona('{agent_id}')...[/dim]"
                    )

                # Small delay to show the spinner (activation is usually instant)
                time.sleep(0.5)

                # Activate the persona using PersonaManager
                success = self.persona_manager.activate_persona(agent_id)

                if not success:
                    raise ConfigurationError(
                        f"Failed to activate persona '{agent_id}'. Check system configuration."
                    )

            # Update current agent tracking
            self.current_agent = agent_id

            # Show success message
            config = persona.configuration
            capabilities_list = [
                cap.value.replace("_", " ").title() for cap in persona.capabilities
            ]

            self.console.print(
                Panel(
                    f"[bold green]Agent Enabled Successfully[/bold green]\n\n"
                    f"[bold]Agent:[/bold] {persona.name}\n"
                    f"[bold]Provider:[/bold] {config.provider_type.value if config.provider_type else 'Unknown'}\n"
                    f"[bold]Model:[/bold] {config.model_id or 'Unknown'}\n"
                    f"[bold]Capabilities:[/bold] {', '.join(capabilities_list)}\n\n"
                    f"The {persona.name} is now active and ready for use.\n"
                    f"Use '/agents status' to monitor or '/agents disable {agent_id}' to deactivate.",
                    title=f"✅ {persona.name} Active",
                    border_style="green",
                )
            )

        except (AgentNotFoundError, PermissionDeniedError, ConfigurationError):
            # These are handled by the main command handler
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error enabling agent '{agent_id}': {e}",
                exc_info=True,
            )
            raise ConfigurationError(f"Failed to enable agent '{agent_id}': {e}")

    async def handle_disable(self, args: List[str]) -> None:
        """
        Handle 'disable' subcommand to deactivate an agent.

        Args:
            args: Agent ID to disable (optional if only one is active)
        """
        try:
            agent_id = None
            persona = None

            if args:
                agent_id = args[0].lower()
                if self.verbose:
                    self.console.print(f"[dim]Looking up agent '{agent_id}'...[/dim]")

                persona = self.persona_manager.get_persona(agent_id)
                if not persona:
                    available_agents = list(
                        self.persona_manager.get_all_personas().keys()
                    )
                    suggestions = [
                        agent
                        for agent in available_agents
                        if agent_id in agent or agent in agent_id
                    ]

                    if suggestions:
                        suggestion_text = (
                            f"Did you mean: {', '.join(suggestions)[:50]}?"
                        )
                    else:
                        suggestion_text = f"Use '/agents list' to see available agents."

                    raise AgentNotFoundError(
                        f"Agent '{agent_id}' not found. {suggestion_text}"
                    )
            else:
                # Use currently active persona
                if self.verbose:
                    self.console.print(
                        "[dim]Looking for currently active persona...[/dim]"
                    )

                active_persona = self.persona_manager.active_persona
                if active_persona:
                    agent_id = active_persona.id
                    persona = active_persona
                else:
                    self._show_info_with_action(
                        "No agent is currently active.",
                        "Use '/agents list' to see available agents or '/agents enable <agent_id>' to activate one.",
                    )
                    return

            if not persona.is_active:
                self._show_info_with_action(
                    f"Agent '{agent_id}' is already disabled.",
                    f"Use '/agents enable {agent_id}' to activate it or '/agents list' to see active agents.",
                )
                return

            # Check permissions if approval manager is available
            if self.approval_manager:
                try:
                    if not await self._check_permission_for_action(
                        "agent_disable", agent_id
                    ):
                        raise PermissionDeniedError(
                            f"You don't have permission to disable agent '{agent_id}'"
                        )
                except Exception as perm_e:
                    if self.verbose:
                        self.console.print(
                            f"[dim]Permission check failed: {perm_e}[/dim]"
                        )
                    # Continue if permission system is not available

            # Check for active operations (placeholder - would integrate with agent engine)
            active_operations = 0  # This would come from agent engine status

            # Show confirmation prompt for destructive action
            if active_operations > 0:
                warning_msg = f"⚠️  Agent '{agent_id}' has {active_operations} active operations.\nDisabling will terminate these operations."
                if not Confirm.ask(f"{warning_msg}\n\nDisable anyway?", default=False):
                    self._show_info("Agent remains enabled.")
                    return
            elif not args:  # Only confirm if no specific agent was provided
                if not Confirm.ask(
                    f"Disable the active agent '{persona.name}'?", default=True
                ):
                    self._show_info("Agent remains enabled.")
                    return

            # Show progress indicator for deactivation
            with Status(
                f"[bold red]Deactivating {persona.name}...",
                console=self.console,
            ):
                if self.verbose:
                    self.console.print(
                        f"[dim]Calling PersonaManager.deactivate_persona()...[/dim]"
                    )

                # Small delay to show the spinner
                time.sleep(0.3)

                # Deactivate the persona using PersonaManager
                success = self.persona_manager.deactivate_persona()

                if not success:
                    raise ConfigurationError(
                        f"Failed to deactivate persona '{agent_id}'. Check system state."
                    )

            # Update current agent tracking
            if self.current_agent == agent_id:
                self.current_agent = None

            self.console.print(
                Panel(
                    f"[bold red]Agent Disabled Successfully[/bold red]\n\n"
                    f"[bold]Agent:[/bold] {persona.name}\n"
                    f"All operations have been stopped.\n\n"
                    f"Use '/agents enable {agent_id}' to reactivate or '/agents list' to see other options.",
                    title=f"🔴 {persona.name} Disabled",
                    border_style="red",
                )
            )

        except (AgentNotFoundError, PermissionDeniedError, ConfigurationError):
            # These are handled by the main command handler
            raise
        except Exception as e:
            logger.error(f"Unexpected error disabling agent: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to disable agent: {e}")

    async def handle_status(self, args: List[str]) -> None:
        """
        Handle 'status' subcommand to show agent status.

        Args:
            args: Optional agent ID for specific status
        """
        try:
            if args:
                # Show specific agent status
                agent_id = args[0].lower()
                persona = self.persona_manager.get_persona(agent_id)
                if not persona:
                    raise AgentNotFoundError(f"Agent '{agent_id}' not found")

                # Map persona status to colors and text
                status_mapping = {
                    PersonaStatus.ACTIVE: ("green", "Active"),
                    PersonaStatus.AVAILABLE: ("yellow", "Available"),
                    PersonaStatus.DISABLED: ("red", "Disabled"),
                    PersonaStatus.ERROR: ("red", "Error"),
                    PersonaStatus.LOADING: ("blue", "Loading"),
                }

                status_color, status_text = status_mapping.get(
                    persona.status, ("gray", "Unknown")
                )
                config = persona.configuration
                capabilities_list = [
                    cap.value.replace("_", " ").title() for cap in persona.capabilities
                ]

                self.console.print(
                    Panel(
                        f"[bold]Name:[/bold] {persona.name}\n"
                        f"[bold]Status:[/bold] [{status_color}]{status_text}[/{status_color}]\n"
                        f"[bold]Provider:[/bold] {config.provider_type.value if config.provider_type else 'Unknown'}\n"
                        f"[bold]Model:[/bold] {config.model_id or 'Unknown'}\n"
                        f"[bold]Description:[/bold] {persona.description}\n"
                        f"[bold]Capabilities:[/bold] {', '.join(capabilities_list)}\n",
                        title=f"Agent Status: {persona.name}",
                        border_style=status_color,
                    )
                )
            else:
                # Show general agent status
                personas = self.persona_manager.get_all_personas()
                active_count = sum(1 for p in personas.values() if p.is_active)
                total_count = len(personas)

                status_text = f"[bold]Agent System Status[/bold]\n\n"
                status_text += f"Available agents: {total_count}\n"
                status_text += f"Active agents: {active_count}\n"

                active_persona = self.persona_manager.active_persona
                if active_persona:
                    status_text += f"Current active agent: {active_persona.name} ({active_persona.id})\n\n"
                    config = active_persona.configuration
                    status_text += f"[bold green]Active Agent Details:[/bold green]\n"
                    status_text += f"• Name: {active_persona.name}\n"
                    status_text += f"• Provider: {config.provider_type.value if config.provider_type else 'Unknown'}\n"
                    status_text += f"• Model: {config.model_id or 'Unknown'}\n"
                else:
                    status_text += f"Current active agent: None\n\n"
                    status_text += f"[dim]Use '/agents enable <agent_id>' to activate an agent.[/dim]"

                self.console.print(
                    Panel(
                        status_text,
                        title="🤖 Agent System Status",
                        border_style="blue",
                    )
                )

        except AgentNotFoundError as e:
            self._show_error(str(e))
        except Exception as e:
            self._show_error(f"Failed to get agent status: {e}")

    async def handle_config(self, args: List[str]) -> None:
        """
        Handle 'config' subcommand to view/modify agent settings.

        Args:
            args: Agent ID and optional config parameters
        """
        if not args:
            self._show_error(
                "Config command requires an agent ID. Use '/agents list' to see available agents."
            )
            return

        agent_id = args[0].lower()

        try:
            persona = self.persona_manager.get_persona(agent_id)
            if not persona:
                raise AgentNotFoundError(f"Agent '{agent_id}' not found")

            config = persona.configuration
            capabilities_list = [
                cap.value.replace("_", " ").title() for cap in persona.capabilities
            ]
            status_text = (
                "Active"
                if persona.is_active
                else (
                    "Available"
                    if persona.status == PersonaStatus.AVAILABLE
                    else persona.status.value.title()
                )
            )

            # For now, just show current configuration
            # In the future, this would allow modifying settings
            self.console.print(
                Panel(
                    f"[bold]Configuration for {persona.name}[/bold]\n\n"
                    f"[bold]Provider:[/bold] {config.provider_type.value if config.provider_type else 'Unknown'}\n"
                    f"[bold]Model:[/bold] {config.model_id or 'Unknown'}\n"
                    f"[bold]Temperature:[/bold] {config.temperature if config.temperature is not None else 'Default'}\n"
                    f"[bold]Max Tokens:[/bold] {config.max_tokens or 'Default'}\n"
                    f"[bold]Tools Enabled:[/bold] {'Yes' if config.tools_enabled else 'No'}\n"
                    f"[bold]Web Search:[/bold] {'Yes' if config.web_search_enabled else 'No'}\n"
                    f"[bold]File Operations:[/bold] {'Yes' if config.file_operations_enabled else 'No'}\n"
                    f"[bold]Approval Required:[/bold] {'Yes' if config.approval_required else 'No'}\n"
                    f"[bold]Capabilities:[/bold] {', '.join(capabilities_list)}\n"
                    f"[bold]Status:[/bold] {status_text}\n\n"
                    f"[dim]Configuration editing will be available in future versions.[/dim]",
                    title=f"⚙️ {persona.name} Configuration",
                    border_style="cyan",
                )
            )

        except AgentNotFoundError as e:
            self._show_error(str(e))
        except Exception as e:
            self._show_error(f"Failed to show agent config: {e}")

    async def handle_switch(self, args: List[str]) -> None:
        """
        Handle 'switch' subcommand to switch between agents.

        Args:
            args: Agent ID to switch to
        """
        if not args:
            self._show_error(
                "Switch command requires an agent ID. Use '/agents list' to see available agents."
            )
            return

        # Switch is essentially disable current + enable new
        await self.handle_enable(args)

    async def handle_current(self) -> None:
        """Handle 'current' subcommand to show currently active agent."""
        active_persona = self.persona_manager.active_persona
        if active_persona:
            config = active_persona.configuration
            self.console.print(
                f"[bold green]Current active agent:[/bold green] {active_persona.name} ({active_persona.id})"
            )
            self.console.print(
                f"[bold]Provider:[/bold] {config.provider_type.value if config.provider_type else 'Unknown'}"
            )
            self.console.print(f"[bold]Model:[/bold] {config.model_id or 'Unknown'}")
        else:
            self.console.print("[dim]No agent is currently active.[/dim]")

    async def handle_info(self, args: List[str]) -> None:
        """
        Handle 'info' subcommand to show detailed agent information.

        Args:
            args: Agent ID to get info for
        """
        if not args:
            self._show_error(
                "Info command requires an agent ID. Use '/agents list' to see available agents."
            )
            return

        agent_id = args[0].lower()

        try:
            persona = self.persona_manager.get_persona(agent_id)
            if not persona:
                raise AgentNotFoundError(f"Agent '{agent_id}' not found")

            config = persona.configuration
            capabilities_list = [
                cap.value.replace("_", " ").title() for cap in persona.capabilities
            ]

            # Get status with appropriate emoji
            if persona.is_active:
                status_display = "🟢 Active"
            elif persona.status == PersonaStatus.AVAILABLE:
                status_display = "⚪ Available"
            elif persona.status == PersonaStatus.DISABLED:
                status_display = "🔴 Disabled"
            elif persona.status == PersonaStatus.ERROR:
                status_display = "❌ Error"
            elif persona.status == PersonaStatus.LOADING:
                status_display = "🔄 Loading"
            else:
                status_display = f"❓ {persona.status.value.title()}"

            info_text = f"[bold]{persona.name}[/bold]\n\n"
            info_text += f"[bold]Description:[/bold] {persona.description}\n\n"
            info_text += f"[bold]Technical Details:[/bold]\n"
            info_text += f"• Provider: {config.provider_type.value if config.provider_type else 'Unknown'}\n"
            info_text += f"• Model: {config.model_id or 'Unknown'}\n"
            info_text += f"• Temperature: {config.temperature if config.temperature is not None else 'Default'}\n"
            info_text += f"• Max Tokens: {config.max_tokens or 'Default'}\n"
            info_text += f"• Status: {status_display}\n\n"
            info_text += f"[bold]Capabilities:[/bold]\n"
            for capability in capabilities_list:
                info_text += f"• {capability}\n"

            info_text += f"\n[bold]Settings:[/bold]\n"
            info_text += f"• Tools Enabled: {'Yes' if config.tools_enabled else 'No'}\n"
            info_text += (
                f"• Web Search: {'Yes' if config.web_search_enabled else 'No'}\n"
            )
            info_text += f"• File Operations: {'Yes' if config.file_operations_enabled else 'No'}\n"
            info_text += (
                f"• Approval Required: {'Yes' if config.approval_required else 'No'}\n"
            )

            info_text += f"\n[bold]Usage:[/bold]\n"
            info_text += f"• Enable: [cyan]/agents enable {agent_id}[/cyan]\n"
            info_text += f"• Disable: [cyan]/agents disable {agent_id}[/cyan]\n"
            info_text += f"• Status: [cyan]/agents status {agent_id}[/cyan]\n"

            self.console.print(
                Panel(
                    info_text,
                    title=f"ℹ️ {persona.name} Information",
                    border_style="blue",
                )
            )

        except AgentNotFoundError as e:
            self._show_error(str(e))
        except Exception as e:
            self._show_error(f"Failed to show agent info: {e}")

    def _show_help(self) -> None:
        """Show help for agents command."""
        help_text = """[bold]Agent Commands:[/bold]

[cyan]/agents list[/cyan]                    - List all available agent personas
[cyan]/agents enable <agent_id>[/cyan]       - Enable/activate an agent persona  
[cyan]/agents disable [agent_id][/cyan]      - Disable active agent
[cyan]/agents switch <agent_id>[/cyan]       - Switch to different agent persona
[cyan]/agents status [agent_id][/cyan]       - Show agent status (general or specific)
[cyan]/agents current[/cyan]                 - Show currently active agent
[cyan]/agents config <agent_id>[/cyan]       - View agent configuration
[cyan]/agents info <agent_id>[/cyan]         - Show detailed agent information

[bold]Available Agent Types:[/bold]
• [green]coding[/green]      - Development-focused with Claude Sonnet + dev tools
• [blue]research[/blue]     - Research-oriented with Perplexity + web search  
• [magenta]creative[/magenta]    - Creative writing with high-temperature models
• [yellow]performance[/yellow] - Fast and cost-efficient with optimized models
• [cyan]general[/cyan]      - Balanced general-purpose configuration

[bold]Examples:[/bold]
[cyan]/agents enable coding[/cyan]           - Activate the coding agent
[cyan]/agents status[/cyan]                  - Show overall agent system status
[cyan]/agents info research[/cyan]           - Get details about the research agent"""

        self.console.print(
            Panel(
                help_text,
                title="🤖 Agent Management Help",
                border_style="blue",
            )
        )

    def _show_error(self, message: str) -> None:
        """Show error message."""
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def _show_error_with_suggestions(self, error_message: str, suggestion: str) -> None:
        """Show error message with actionable suggestions."""
        self.console.print(
            Panel(
                f"[bold red]Error:[/bold red] {error_message}\n\n"
                f"[bold]Suggestion:[/bold] {suggestion}",
                title="❌ Error",
                border_style="red",
            )
        )

    def _show_info(self, message: str) -> None:
        """Show info message."""
        self.console.print(f"[blue]Info:[/blue] {message}")

    def _show_info_with_action(self, info_message: str, action: str) -> None:
        """Show info message with suggested action."""
        self.console.print(
            Panel(
                f"[blue]Info:[/blue] {info_message}\n\n"
                f"[bold]Next Steps:[/bold] {action}",
                title="ℹ️ Information",
                border_style="blue",
            )
        )

    def _show_success(self, message: str) -> None:
        """Show success message."""
        self.console.print(f"[bold green]Success:[/bold green] {message}")

    async def _check_permission_for_action(self, action: str, target: str) -> bool:
        """
        Check if user has permission for a specific action.

        Args:
            action: The action being attempted (e.g., 'agent_enable', 'agent_disable')
            target: The target of the action (e.g., agent_id)

        Returns:
            True if permission is granted, False otherwise
        """
        try:
            if not self.approval_manager:
                return True  # No approval manager means no restrictions

            # This would integrate with the EnhancedApprovalManager
            # For now, we'll assume all actions are permitted
            if self.verbose:
                self.console.print(
                    f"[dim]Checking permission for action '{action}' on target '{target}'[/dim]"
                )

            return True  # Placeholder - would implement real permission checking

        except Exception as e:
            logger.error(f"Permission check failed for action '{action}': {e}")
            return True  # Fail open for now
