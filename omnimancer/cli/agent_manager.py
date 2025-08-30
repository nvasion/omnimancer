"""
Agent Discovery and Management Interface for Omnimancer.

This module provides CLI commands for managing custom agents including
listing, searching, loading, activation, and detailed inspection.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.box import ROUNDED, SIMPLE
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from ..core.agent.agent_config import (
    AgentRepository,
    CustomAgentConfig,
    CustomAgentStatus,
)
from ..core.agent.persona import PersonaCategory
from .agent_wizard import AgentCreationWizard

logger = logging.getLogger(__name__)


class AgentManager:
    """Manager for custom agent discovery and operations."""

    def __init__(
        self,
        console: Optional[Console] = None,
        repository: Optional[AgentRepository] = None,
    ):
        """
        Initialize agent manager.

        Args:
            console: Rich console for output
            repository: Agent repository for storage operations
        """
        self.console = console or Console()
        self.repository = repository or AgentRepository()

        # Cache for active agent
        self._active_agent: Optional[CustomAgentConfig] = None
        self._session_stats: Dict[str, Any] = {
            "agents_loaded": 0,
            "agents_activated": 0,
            "session_start": datetime.now(),
        }

    async def list_agents(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None,
        show_details: bool = False,
        sort_by: str = "name",
    ) -> List[CustomAgentConfig]:
        """
        List all custom agents with optional filtering.

        Args:
            category: Filter by category (optional)
            status: Filter by status (optional)
            show_details: Show detailed information
            sort_by: Sort field (name, created_at, last_used, usage_count)

        Returns:
            List of matching agent configurations
        """
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task("Loading agents...", total=None)

                # Get all agents
                all_agents = self.repository.list_all()

                # Apply filters
                filtered_agents = all_agents

                if category:
                    try:
                        cat_filter = PersonaCategory(category.lower())
                        filtered_agents = [
                            a for a in filtered_agents if a.category == cat_filter
                        ]
                    except ValueError:
                        self.console.print(f"[red]Invalid category: {category}[/red]")
                        return []

                if status:
                    try:
                        status_filter = CustomAgentStatus(status.lower())
                        filtered_agents = [
                            a for a in filtered_agents if a.status == status_filter
                        ]
                    except ValueError:
                        self.console.print(f"[red]Invalid status: {status}[/red]")
                        return []

                # Sort agents
                filtered_agents = self._sort_agents(filtered_agents, sort_by)

                progress.update(task, completed=100)

            # Display results
            if not filtered_agents:
                self.console.print(
                    "[yellow]No agents found matching the criteria.[/yellow]"
                )
                return []

            if show_details:
                self._show_detailed_agent_list(filtered_agents)
            else:
                self._show_summary_agent_list(filtered_agents)

            return filtered_agents

        except Exception as e:
            self.console.print(f"[red]Error listing agents: {e}[/red]")
            logger.error(f"Agent listing error: {e}", exc_info=True)
            return []

    async def show_agent(self, identifier: str) -> Optional[CustomAgentConfig]:
        """
        Show detailed information about a specific agent.

        Args:
            identifier: Agent ID or name

        Returns:
            Agent configuration if found
        """
        try:
            # Try to find by ID first, then by name
            agent = self.repository.get(identifier)
            if not agent:
                agent = self.repository.get_by_name(identifier)

            if not agent:
                self.console.print(f"[red]Agent '{identifier}' not found.[/red]")
                return None

            self._show_agent_details(agent)
            return agent

        except Exception as e:
            self.console.print(f"[red]Error showing agent: {e}[/red]")
            logger.error(f"Agent show error: {e}", exc_info=True)
            return None

    async def search_agents(
        self, query: str, fields: Optional[List[str]] = None
    ) -> List[CustomAgentConfig]:
        """
        Search agents by query string.

        Args:
            query: Search query
            fields: Fields to search in (name, description, capabilities)

        Returns:
            List of matching agents
        """
        try:
            if not query.strip():
                self.console.print("[red]Search query cannot be empty.[/red]")
                return []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"Searching for '{query}'...", total=None)

                results = self.repository.search(query, fields)

                progress.update(task, completed=100)

            if not results:
                self.console.print(
                    f"[yellow]No agents found matching '{query}'.[/yellow]"
                )
                return []

            self.console.print(
                f"\\n[green]Found {len(results)} agent(s) matching '{query}':[/green]"
            )
            self._show_summary_agent_list(results)

            return results

        except Exception as e:
            self.console.print(f"[red]Error searching agents: {e}[/red]")
            logger.error(f"Agent search error: {e}", exc_info=True)
            return []

    async def load_agent(
        self, identifier: str, activate: bool = True
    ) -> Optional[CustomAgentConfig]:
        """
        Load an agent configuration.

        Args:
            identifier: Agent ID or name
            activate: Whether to activate the agent for current session

        Returns:
            Loaded agent configuration
        """
        try:
            # Find the agent
            agent = self.repository.get(identifier)
            if not agent:
                agent = self.repository.get_by_name(identifier)

            if not agent:
                self.console.print(f"[red]Agent '{identifier}' not found.[/red]")
                return None

            # Check if agent is in usable state
            if agent.status not in [
                CustomAgentStatus.ACTIVE,
                CustomAgentStatus.DRAFT,
            ]:
                self.console.print(
                    f"[yellow]Warning: Agent status is '{agent.status.value}'.[/yellow]"
                )
                if not Confirm.ask("Load anyway?", default=False):
                    return None

            # Load the agent
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"Loading agent '{agent.name}'...", total=None)

                # Update usage statistics
                agent.increment_usage()
                self.repository.update(agent)

                if activate:
                    self._active_agent = agent
                    self._session_stats["agents_activated"] += 1

                self._session_stats["agents_loaded"] += 1

                progress.update(task, completed=100)

            self.console.print(
                f"[green]✅ Agent '{agent.name}' loaded successfully![/green]"
            )

            if activate:
                self.console.print(
                    f"[cyan]🚀 Agent '{agent.name}' is now active for this session.[/cyan]"
                )
                self._show_agent_summary(agent)

            return agent

        except Exception as e:
            self.console.print(f"[red]Error loading agent: {e}[/red]")
            logger.error(f"Agent loading error: {e}", exc_info=True)
            return None

    async def delete_agent(self, identifier: str, confirm: bool = True) -> bool:
        """
        Delete an agent configuration.

        Args:
            identifier: Agent ID or name
            confirm: Whether to ask for confirmation

        Returns:
            True if deleted successfully
        """
        try:
            # Find the agent
            agent = self.repository.get(identifier)
            if not agent:
                agent = self.repository.get_by_name(identifier)

            if not agent:
                self.console.print(f"[red]Agent '{identifier}' not found.[/red]")
                return False

            # Show agent info before deletion
            self.console.print(f"\\n[yellow]Agent to delete:[/yellow]")
            self._show_agent_summary(agent)

            # Confirm deletion
            if confirm:
                if not Confirm.ask(
                    f"\\n[red]Are you sure you want to delete '{agent.name}'?[/red]",
                    default=False,
                ):
                    self.console.print("[yellow]Deletion cancelled.[/yellow]")
                    return False

            # Check if agent is currently active
            if self._active_agent and self._active_agent.id == agent.id:
                self.console.print(
                    "[yellow]Warning: Deleting currently active agent.[/yellow]"
                )
                self._active_agent = None

            # Delete the agent
            success = self.repository.delete(agent.id)

            if success:
                self.console.print(
                    f"[green]✅ Agent '{agent.name}' deleted successfully.[/green]"
                )
            else:
                self.console.print(
                    f"[red]❌ Failed to delete agent '{agent.name}'.[/red]"
                )

            return success

        except Exception as e:
            self.console.print(f"[red]Error deleting agent: {e}[/red]")
            logger.error(f"Agent deletion error: {e}", exc_info=True)
            return False

    async def create_agent(
        self, interactive: bool = True
    ) -> Optional[CustomAgentConfig]:
        """
        Create a new agent using the wizard.

        Args:
            interactive: Whether to use interactive wizard

        Returns:
            Created agent configuration
        """
        try:
            if interactive:
                wizard = AgentCreationWizard(
                    console=self.console, repository=self.repository
                )
                return await wizard.run_wizard()
            else:
                # Quick creation - would need additional parameters
                self.console.print(
                    "[yellow]Non-interactive creation not yet implemented.[/yellow]"
                )
                return None

        except Exception as e:
            self.console.print(f"[red]Error creating agent: {e}[/red]")
            logger.error(f"Agent creation error: {e}", exc_info=True)
            return None

    async def clone_agent(
        self, identifier: str, new_name: str
    ) -> Optional[CustomAgentConfig]:
        """
        Clone an existing agent with a new name.

        Args:
            identifier: Source agent ID or name
            new_name: Name for the cloned agent

        Returns:
            Cloned agent configuration
        """
        try:
            # Find source agent
            source_agent = self.repository.get(identifier)
            if not source_agent:
                source_agent = self.repository.get_by_name(identifier)

            if not source_agent:
                self.console.print(f"[red]Source agent '{identifier}' not found.[/red]")
                return None

            # Check name availability
            if self.repository.get_by_name(new_name):
                self.console.print(
                    f"[red]Agent named '{new_name}' already exists.[/red]"
                )
                return None

            # Create clone
            import uuid

            cloned_config = CustomAgentConfig(
                id=str(uuid.uuid4()),
                name=new_name,
                description=f"Clone of {source_agent.name}\\n{source_agent.description}",
                category=source_agent.category,
                base_template_id=source_agent.base_template_id,
                model_settings=source_agent.model_settings,
                context_parameters=source_agent.context_parameters,
                behavior_rules=source_agent.behavior_rules,
                capabilities=source_agent.capabilities.copy(),
                enabled_tools=source_agent.enabled_tools.copy(),
                security_overrides=source_agent.security_overrides,
                agent_overrides=source_agent.agent_overrides,
                status=CustomAgentStatus.DRAFT,  # Start as draft
            )

            # Save cloned agent
            saved_clone = self.repository.create(cloned_config)

            self.console.print(
                f"[green]✅ Agent '{new_name}' cloned from '{source_agent.name}'![/green]"
            )
            self._show_agent_summary(saved_clone)

            return saved_clone

        except Exception as e:
            self.console.print(f"[red]Error cloning agent: {e}[/red]")
            logger.error(f"Agent cloning error: {e}", exc_info=True)
            return None

    async def export_agent(
        self, identifier: str, export_path: Optional[Path] = None
    ) -> bool:
        """
        Export an agent configuration to file.

        Args:
            identifier: Agent ID or name
            export_path: Export file path (optional)

        Returns:
            True if exported successfully
        """
        try:
            # Find the agent
            agent = self.repository.get(identifier)
            if not agent:
                agent = self.repository.get_by_name(identifier)

            if not agent:
                self.console.print(f"[red]Agent '{identifier}' not found.[/red]")
                return False

            # Determine export path
            if not export_path:
                safe_name = "".join(
                    c if c.isalnum() or c in "-_" else "_" for c in agent.name
                )
                export_path = Path.cwd() / f"{safe_name}_agent.json"

            # Export the agent
            success = self.repository.export_config(agent.id, export_path)

            if success:
                self.console.print(
                    f"[green]✅ Agent '{agent.name}' exported to {export_path}[/green]"
                )
            else:
                self.console.print(
                    f"[red]❌ Failed to export agent '{agent.name}'.[/red]"
                )

            return success

        except Exception as e:
            self.console.print(f"[red]Error exporting agent: {e}[/red]")
            logger.error(f"Agent export error: {e}", exc_info=True)
            return False

    async def import_agent(
        self, import_path: Path, new_name: Optional[str] = None
    ) -> Optional[CustomAgentConfig]:
        """
        Import an agent configuration from file.

        Args:
            import_path: Import file path
            new_name: Optional new name to avoid conflicts

        Returns:
            Imported agent configuration
        """
        try:
            if not import_path.exists():
                self.console.print(f"[red]Import file not found: {import_path}[/red]")
                return None

            # Import the agent
            imported_agent = self.repository.import_config(import_path, new_id=True)

            if not imported_agent:
                self.console.print(
                    f"[red]Failed to import agent from {import_path}[/red]"
                )
                return None

            # Rename if requested
            if new_name:
                imported_agent.name = new_name
                imported_agent = self.repository.update(imported_agent)

            self.console.print(
                f"[green]✅ Agent '{imported_agent.name}' imported successfully![/green]"
            )
            self._show_agent_summary(imported_agent)

            return imported_agent

        except Exception as e:
            self.console.print(f"[red]Error importing agent: {e}[/red]")
            logger.error(f"Agent import error: {e}", exc_info=True)
            return None

    def get_active_agent(self) -> Optional[CustomAgentConfig]:
        """Get the currently active agent."""
        return self._active_agent

    def show_statistics(self):
        """Show agent repository and session statistics."""
        try:
            # Repository statistics
            repo_stats = self.repository.get_statistics()

            # Create statistics display
            stats_table = Table(show_header=False, box=SIMPLE)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")

            # Repository stats
            stats_table.add_row("Total Agents", str(repo_stats["total_agents"]))

            if repo_stats["status_distribution"]:
                for status, count in repo_stats["status_distribution"].items():
                    stats_table.add_row(f"  {status.title()}", str(count))

            stats_table.add_row(
                "Storage Location", str(repo_stats["storage_directory"])
            )

            # Session stats
            stats_table.add_row("", "")  # Separator
            stats_table.add_row(
                "Session Started",
                self._session_stats["session_start"].strftime("%Y-%m-%d %H:%M"),
            )
            stats_table.add_row(
                "Agents Loaded", str(self._session_stats["agents_loaded"])
            )
            stats_table.add_row(
                "Agents Activated",
                str(self._session_stats["agents_activated"]),
            )

            if self._active_agent:
                stats_table.add_row("Active Agent", self._active_agent.name)

            self.console.print(
                Panel(stats_table, title="📊 Agent Statistics", box=ROUNDED)
            )

        except Exception as e:
            self.console.print(f"[red]Error showing statistics: {e}[/red]")
            logger.error(f"Statistics display error: {e}", exc_info=True)

    def _show_summary_agent_list(self, agents: List[CustomAgentConfig]):
        """Display agents in summary table format."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="yellow")
        table.add_column("Model", style="green")
        table.add_column("Status", style="white")
        table.add_column("Tools", style="blue")
        table.add_column("Last Used", style="dim")

        for agent in agents:
            # Format model info
            model_info = f"{agent.model_settings.provider_type.value}:{agent.model_settings.model_name}"
            if len(model_info) > 20:
                model_info = model_info[:17] + "..."

            # Format status with emoji
            status_emoji = {
                CustomAgentStatus.ACTIVE: "✅",
                CustomAgentStatus.DRAFT: "📝",
                CustomAgentStatus.DISABLED: "❌",
                CustomAgentStatus.ARCHIVED: "📦",
                CustomAgentStatus.ERROR: "⚠️",
            }
            status_display = (
                f"{status_emoji.get(agent.status, '')} {agent.status.value}"
            )

            # Format tools count
            tools_count = f"{len(agent.enabled_tools)} tools"

            # Format last used
            if agent.metadata.last_used:
                last_used = agent.metadata.last_used.strftime("%m-%d %H:%M")
            else:
                last_used = "Never"

            table.add_row(
                agent.name[:25] + ("..." if len(agent.name) > 25 else ""),
                agent.category.value.title(),
                model_info,
                status_display,
                tools_count,
                last_used,
            )

        self.console.print(table)
        self.console.print(f"\\n[dim]Showing {len(agents)} agent(s)[/dim]")

    def _show_detailed_agent_list(self, agents: List[CustomAgentConfig]):
        """Display agents with detailed information."""
        for i, agent in enumerate(agents):
            if i > 0:
                self.console.print()  # Spacing between agents

            self._show_agent_summary(agent)

    def _show_agent_details(self, agent: CustomAgentConfig):
        """Show comprehensive details for a single agent."""
        # Header
        header = f"🤖 {agent.name}"
        if agent.description:
            header += f" - {agent.description}"

        self.console.print(Panel(header, style="bold cyan", box=ROUNDED))

        # Basic information table
        info_table = Table(show_header=False, box=None)
        info_table.add_column("Field", style="cyan", width=20)
        info_table.add_column("Value", style="white")

        info_table.add_row("ID", agent.id)
        info_table.add_row("Category", agent.category.value.title())
        info_table.add_row("Status", f"{agent.status.value.title()}")
        info_table.add_row("Created", agent.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        info_table.add_row(
            "Last Modified", agent.updated_at.strftime("%Y-%m-%d %H:%M:%S")
        )
        info_table.add_row("Usage Count", str(agent.metadata.usage_count))

        if agent.metadata.last_used:
            info_table.add_row(
                "Last Used",
                agent.metadata.last_used.strftime("%Y-%m-%d %H:%M:%S"),
            )

        if agent.base_template_id:
            info_table.add_row("Base Template", agent.base_template_id)

        self.console.print(Panel(info_table, title="📋 Basic Information", box=ROUNDED))

        # Model configuration
        model_table = Table(show_header=False, box=None)
        model_table.add_column("Field", style="cyan", width=20)
        model_table.add_column("Value", style="white")

        model_table.add_row("Provider", agent.model_settings.provider_type.value)
        model_table.add_row("Model", agent.model_settings.model_name)
        model_table.add_row("Temperature", str(agent.model_settings.temperature))

        if agent.model_settings.max_tokens:
            model_table.add_row("Max Tokens", str(agent.model_settings.max_tokens))

        self.console.print(
            Panel(model_table, title="🧠 Model Configuration", box=ROUNDED)
        )

        # Tools and capabilities
        tools_text = "\\n".join(
            [
                f"• {tool.value.replace('_', ' ').title()}"
                for tool in agent.enabled_tools
            ]
        )
        caps_text = "\\n".join(
            [f"• {cap.value.replace('_', ' ').title()}" for cap in agent.capabilities]
        )

        tools_panel = Panel(
            tools_text or "[dim]None[/dim]",
            title="🔧 Enabled Tools",
            box=ROUNDED,
        )
        caps_panel = Panel(
            caps_text or "[dim]None[/dim]",
            title="⭐ Capabilities",
            box=ROUNDED,
        )

        self.console.print(Columns([tools_panel, caps_panel], equal=True))

        # Context and behavior
        context_table = Table(show_header=False, box=None)
        context_table.add_column("Field", style="cyan", width=20)
        context_table.add_column("Value", style="white")

        context_table.add_row(
            "Context Window",
            f"{agent.context_parameters.context_window_size:,} tokens",
        )
        context_table.add_row(
            "Memory Enabled",
            "✅" if agent.context_parameters.conversation_memory else "❌",
        )

        if agent.context_parameters.conversation_memory:
            context_table.add_row(
                "Memory Limit",
                f"{agent.context_parameters.memory_limit} messages",
            )

        context_table.add_row(
            "Response Format", agent.context_parameters.response_format
        )
        context_table.add_row("Safety Level", agent.behavior_rules.safety_level.title())
        context_table.add_row(
            "Reasoning Style",
            agent.behavior_rules.reasoning_style.replace("_", " ").title(),
        )
        context_table.add_row(
            "Creativity Level", agent.behavior_rules.creativity_level.title()
        )

        self.console.print(
            Panel(context_table, title="💭 Context & Behavior", box=ROUNDED)
        )

        # System prompt (if present)
        if agent.context_parameters.system_prompt:
            prompt_preview = agent.context_parameters.system_prompt[:300]
            if len(agent.context_parameters.system_prompt) > 300:
                prompt_preview += "\\n\\n[dim]... (truncated)[/dim]"

            self.console.print(
                Panel(
                    prompt_preview,
                    title="📝 System Prompt (Preview)",
                    box=ROUNDED,
                )
            )

    def _show_agent_summary(self, agent: CustomAgentConfig):
        """Show a concise summary of an agent."""
        summary_text = Text.assemble(
            f"🤖 ",
            ("", "bold cyan"),
            f"{agent.name}",
            ("", "bold cyan"),
            f" ({agent.category.value}) - ",
            f"{agent.model_settings.provider_type.value}:{agent.model_settings.model_name}",
            f"\\n   {len(agent.enabled_tools)} tools, {len(agent.capabilities)} capabilities",
            f" • Status: {agent.status.value}",
        )

        self.console.print(Panel(summary_text, box=SIMPLE))

    def _sort_agents(
        self, agents: List[CustomAgentConfig], sort_by: str
    ) -> List[CustomAgentConfig]:
        """Sort agents by specified field."""
        sort_functions = {
            "name": lambda a: a.name.lower(),
            "created_at": lambda a: a.created_at,
            "updated_at": lambda a: a.updated_at,
            "last_used": lambda a: a.metadata.last_used or datetime.min,
            "usage_count": lambda a: a.metadata.usage_count,
            "category": lambda a: a.category.value,
        }

        if sort_by not in sort_functions:
            logger.warning(f"Unknown sort field: {sort_by}, using name")
            sort_by = "name"

        return sorted(agents, key=sort_functions[sort_by])


# Utility functions for easy integration


def create_agent_manager(console: Optional[Console] = None) -> AgentManager:
    """Create and initialize an agent manager."""
    return AgentManager(console=console)


async def list_available_agents(
    console: Optional[Console] = None,
) -> List[CustomAgentConfig]:
    """Quick function to list all available agents."""
    manager = create_agent_manager(console)
    return await manager.list_agents()


async def load_agent_by_name(
    name: str, console: Optional[Console] = None
) -> Optional[CustomAgentConfig]:
    """Quick function to load an agent by name."""
    manager = create_agent_manager(console)
    return await manager.load_agent(name)


def get_agent_statistics(console: Optional[Console] = None) -> Dict[str, Any]:
    """Get agent repository statistics."""
    repository = AgentRepository()
    return repository.get_statistics()
