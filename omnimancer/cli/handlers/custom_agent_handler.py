"""
Custom Agent CLI Handler for Omnimancer.

This module provides the CustomAgentHandler class that processes custom agent
management commands including creation, listing, loading, and configuration.
"""

import logging
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from ..agent_manager import create_agent_manager
from ..commands import Command

logger = logging.getLogger(__name__)


class CustomAgentHandler:
    """Handles custom agent CLI commands."""

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the handler.

        Args:
            console: Rich console for output
        """
        self.console = console or Console()
        self.agent_manager = create_agent_manager(self.console)

    async def handle_custom_agent_command(self, command: Command) -> None:
        """
        Handle /agent commands for custom agent management.

        Args:
            command: Parsed command object
        """
        try:
            args = command.parameters.get("args", [])

            if not args:
                await self._show_help()
                return

            subcommand = args[0].lower()
            remaining_args = args[1:]

            # Route to appropriate handler
            if subcommand == "list":
                await self._handle_list_agents(remaining_args)
            elif subcommand == "show":
                await self._handle_show_agent(remaining_args)
            elif subcommand == "create":
                await self._handle_create_agent(remaining_args)
            elif subcommand == "load":
                await self._handle_load_agent(remaining_args)
            elif subcommand == "delete":
                await self._handle_delete_agent(remaining_args)
            elif subcommand == "clone":
                await self._handle_clone_agent(remaining_args)
            elif subcommand == "search":
                await self._handle_search_agents(remaining_args)
            elif subcommand == "export":
                await self._handle_export_agent(remaining_args)
            elif subcommand == "import":
                await self._handle_import_agent(remaining_args)
            elif subcommand == "stats":
                await self._handle_show_statistics(remaining_args)
            elif subcommand == "active":
                await self._handle_show_active(remaining_args)
            else:
                self.console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
                await self._show_help()

        except Exception as e:
            self.console.print(f"[red]Error handling agent command: {e}[/red]")
            logger.error(f"Custom agent command error: {e}", exc_info=True)

    async def _show_help(self):
        """Show help for custom agent commands."""
        help_text = """
[bold cyan]Custom Agent Management Commands[/bold cyan]

[yellow]Basic Operations:[/yellow]
  /agent list [--category <cat>] [--status <status>] [--details]
    List all custom agents with optional filtering
    
  /agent show <name|id>
    Show detailed information about a specific agent
    
  /agent create
    Launch interactive wizard to create a new agent
    
  /agent load <name|id>
    Load and activate an agent for the current session
    
[yellow]Management:[/yellow]
  /agent delete <name|id>
    Delete an agent configuration
    
  /agent clone <source> <new_name>
    Clone an existing agent with a new name
    
  /agent search <query>
    Search agents by name, description, or capabilities
    
[yellow]Import/Export:[/yellow]
  /agent export <name|id> [--path <file>]
    Export agent configuration to file
    
  /agent import <file> [--name <new_name>]
    Import agent configuration from file
    
[yellow]Information:[/yellow]
  /agent stats
    Show repository and session statistics
    
  /agent active
    Show currently active agent
    
[yellow]Examples:[/yellow]
  /agent list --category development --details
  /agent show my-coding-assistant
  /agent load my-coding-assistant
  /agent clone my-coding-assistant my-new-assistant
  /agent search "web development"
"""

        self.console.print(
            Panel(help_text, title="🤖 Custom Agent Commands", expand=False)
        )

    async def _handle_list_agents(self, args: List[str]):
        """Handle agent listing command."""
        # Parse options
        category = None
        status = None
        show_details = False
        sort_by = "name"

        i = 0
        while i < len(args):
            if args[i] == "--category" and i + 1 < len(args):
                category = args[i + 1]
                i += 2
            elif args[i] == "--status" and i + 1 < len(args):
                status = args[i + 1]
                i += 2
            elif args[i] == "--details":
                show_details = True
                i += 1
            elif args[i] == "--sort" and i + 1 < len(args):
                sort_by = args[i + 1]
                i += 2
            else:
                i += 1

        await self.agent_manager.list_agents(
            category=category,
            status=status,
            show_details=show_details,
            sort_by=sort_by,
        )

    async def _handle_show_agent(self, args: List[str]):
        """Handle show agent command."""
        if not args:
            self.console.print("[red]Please specify an agent name or ID.[/red]")
            self.console.print("Usage: /agent show <name|id>")
            return

        identifier = " ".join(args)  # Handle names with spaces
        await self.agent_manager.show_agent(identifier)

    async def _handle_create_agent(self, args: List[str]):
        """Handle create agent command."""
        # Check for quick creation options
        interactive = True

        if "--quick" in args:
            interactive = False
            self.console.print(
                "[yellow]Quick creation not yet implemented. Using interactive mode.[/yellow]"
            )

        agent = await self.agent_manager.create_agent(interactive=interactive)

        if agent:
            self.console.print(
                f"\\n[green]Agent '{agent.name}' created successfully![/green]"
            )

            # Ask if user wants to load it immediately
            if Confirm.ask("Load this agent now?", default=True):
                await self.agent_manager.load_agent(agent.id, activate=True)

    async def _handle_load_agent(self, args: List[str]):
        """Handle load agent command."""
        if not args:
            self.console.print("[red]Please specify an agent name or ID.[/red]")
            self.console.print("Usage: /agent load <name|id>")
            return

        identifier = " ".join(args)
        await self.agent_manager.load_agent(identifier, activate=True)

    async def _handle_delete_agent(self, args: List[str]):
        """Handle delete agent command."""
        if not args:
            self.console.print("[red]Please specify an agent name or ID.[/red]")
            self.console.print("Usage: /agent delete <name|id>")
            return

        identifier = " ".join(args)

        # Check for --force flag
        force = "--force" in args
        if force:
            args = [arg for arg in args if arg != "--force"]
            identifier = " ".join(args)

        await self.agent_manager.delete_agent(identifier, confirm=not force)

    async def _handle_clone_agent(self, args: List[str]):
        """Handle clone agent command."""
        if len(args) < 2:
            self.console.print("[red]Please specify source agent and new name.[/red]")
            self.console.print("Usage: /agent clone <source> <new_name>")
            return

        source_identifier = args[0]
        new_name = " ".join(args[1:])  # Handle names with spaces

        await self.agent_manager.clone_agent(source_identifier, new_name)

    async def _handle_search_agents(self, args: List[str]):
        """Handle search agents command."""
        if not args:
            self.console.print("[red]Please specify a search query.[/red]")
            self.console.print("Usage: /agent search <query>")
            return

        query = " ".join(args)

        # Check for field restrictions
        fields = None
        if "--fields" in args:
            field_index = args.index("--fields")
            if field_index + 1 < len(args):
                fields = args[field_index + 1].split(",")
                # Remove --fields and field list from query
                args = args[:field_index] + args[field_index + 2 :]
                query = " ".join(args)

        await self.agent_manager.search_agents(query, fields)

    async def _handle_export_agent(self, args: List[str]):
        """Handle export agent command."""
        if not args:
            self.console.print("[red]Please specify an agent name or ID.[/red]")
            self.console.print("Usage: /agent export <name|id> [--path <file>]")
            return

        # Parse arguments
        export_path = None
        identifier_args = []

        i = 0
        while i < len(args):
            if args[i] == "--path" and i + 1 < len(args):
                export_path = Path(args[i + 1])
                i += 2
            else:
                identifier_args.append(args[i])
                i += 1

        if not identifier_args:
            self.console.print("[red]Please specify an agent name or ID.[/red]")
            return

        identifier = " ".join(identifier_args)
        await self.agent_manager.export_agent(identifier, export_path)

    async def _handle_import_agent(self, args: List[str]):
        """Handle import agent command."""
        if not args:
            self.console.print("[red]Please specify import file path.[/red]")
            self.console.print("Usage: /agent import <file> [--name <new_name>]")
            return

        # Parse arguments
        new_name = None
        import_file = None

        i = 0
        while i < len(args):
            if args[i] == "--name" and i + 1 < len(args):
                new_name = args[i + 1]
                i += 2
            else:
                if not import_file:
                    import_file = args[i]
                i += 1

        if not import_file:
            self.console.print("[red]Please specify import file path.[/red]")
            return

        import_path = Path(import_file)
        agent = await self.agent_manager.import_agent(import_path, new_name)

        if agent:
            # Ask if user wants to load it immediately
            if Confirm.ask("Load this agent now?", default=False):
                await self.agent_manager.load_agent(agent.id, activate=True)

    async def _handle_show_statistics(self, args: List[str]):
        """Handle show statistics command."""
        self.agent_manager.show_statistics()

    async def _handle_show_active(self, args: List[str]):
        """Handle show active agent command."""
        active_agent = self.agent_manager.get_active_agent()

        if not active_agent:
            self.console.print("[yellow]No agent is currently active.[/yellow]")
            return

        self.console.print(f"[green]Active Agent:[/green]")
        await self.agent_manager.show_agent(active_agent.id)


# Utility functions for integration


def create_custom_agent_handler(
    console: Optional[Console] = None,
) -> CustomAgentHandler:
    """Create and initialize a custom agent handler."""
    return CustomAgentHandler(console=console)


async def handle_agent_command(
    command: Command, console: Optional[Console] = None
) -> None:
    """Handle agent command using the custom agent handler."""
    handler = create_custom_agent_handler(console)
    await handler.handle_custom_agent_command(command)
