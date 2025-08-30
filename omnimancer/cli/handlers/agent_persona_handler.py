"""
Handler for agent persona CLI commands.

This module provides the command handling logic for managing agent personas
through the CLI interface.
"""

from typing import Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.agent.agent_switcher import (
    SwitchState,
    get_agent_switcher,
)
from ...core.agent.persona import (
    PersonaStatus,
    get_persona_manager,
)
from ..commands import Command


class AgentPersonaHandler:
    """Handles agent persona-related CLI commands."""

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the handler.

        Args:
            console: Rich console for output
        """
        self.console = console or Console()
        self.persona_manager = get_persona_manager()
        self.agent_switcher = get_agent_switcher(self.persona_manager)

    async def handle_agent_command(self, command: Command) -> None:
        """
        Main entry point for handling agent commands.

        Args:
            command: The command to handle
        """
        args = command.parameters.get("args", [])

        if not args:
            # Default to 'list' if no subcommand
            await self.handle_list_personas()
            return

        subcommand = args[0].lower()

        # Route to appropriate handler
        if subcommand == "list":
            await self.handle_list_personas()
        elif subcommand == "use":
            if len(args) < 2:
                self._show_error("Usage: /agent use <persona_name>")
                return
            await self.handle_use_persona(args[1])
        elif subcommand == "current":
            await self.handle_current_persona()
        elif subcommand == "info":
            if len(args) < 2:
                self._show_error("Usage: /agent info <persona_name>")
                return
            await self.handle_persona_info(args[1])
        elif subcommand == "status":
            # Show detailed status including switch state
            await self.handle_persona_status()
        elif subcommand == "history":
            # Show switch history
            await self.handle_switch_history()
        elif subcommand == "recommend":
            user_query = " ".join(args[1:]) if len(args) > 1 else None
            await self.handle_recommend_persona(user_query)
        elif subcommand == "compare":
            await self.handle_compare_personas()
        elif subcommand == "preview":
            if len(args) < 2:
                self._show_error("Usage: /agent preview <persona_name>")
                return
            await self.handle_preview_persona(args[1])
        elif subcommand == "discover":
            await self.handle_discover_personas()
        elif subcommand == "help":
            self._show_agent_help()
        else:
            self._show_error(f"Unknown agent subcommand: {subcommand}")
            self._show_agent_help()

    async def handle_list_personas(self) -> None:
        """List all available agent personas."""
        personas = self.persona_manager.get_all_personas()

        if not personas:
            self._show_warning("No agent personas available.")
            return

        # Create a table for personas
        table = Table(
            title="🤖 Available Agent Personas",
            box=ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Icon", style="cyan", justify="center")
        table.add_column("ID", style="green")
        table.add_column("Name", style="bold white")
        table.add_column("Category", style="yellow")
        table.add_column("Status", justify="center")
        table.add_column("Description", style="dim")

        # Sort personas by category and name
        sorted_personas = sorted(
            personas.values(), key=lambda p: (p.category.value, p.name)
        )

        active_persona = self.persona_manager.active_persona

        for persona in sorted_personas:
            # Format status with color
            status = self._format_status(persona.status)

            # Highlight active persona
            if active_persona and persona.id == active_persona.id:
                status = "[bold green]● ACTIVE[/bold green]"
                name_style = "[bold cyan]" + persona.name + "[/bold cyan]"
            else:
                name_style = persona.name

            table.add_row(
                persona.icon,
                persona.id,
                name_style,
                persona.category.value.title(),
                status,
                (
                    persona.description[:50] + "..."
                    if len(persona.description) > 50
                    else persona.description
                ),
            )

        self.console.print(table)

        # Show usage hint
        self.console.print(
            "\n[dim]Use '/agent use <persona_id>' to switch to a persona[/dim]"
        )
        self.console.print(
            "[dim]Use '/agent info <persona_id>' for detailed information[/dim]"
        )

    async def handle_use_persona(self, persona_id: str) -> None:
        """
        Switch to a specified persona.

        Args:
            persona_id: ID of the persona to switch to
        """
        # Check if persona exists
        persona = self.persona_manager.get_persona(persona_id.lower())
        if not persona:
            self._show_error(f"Persona '{persona_id}' not found.")
            self._suggest_personas(persona_id)
            return

        # Check if already active
        if self.persona_manager.active_persona == persona:
            self._show_info(f"Already using {persona.name} ({persona.icon})")
            return

        # Show switching message
        self.console.print(
            f"\n[yellow]Switching to {persona.name} ({persona.icon})...[/yellow]"
        )

        # Perform the switch
        success, message = self.agent_switcher.switch_persona(
            persona_id.lower(), reason="User requested switch via CLI"
        )

        if success:
            self._show_success(f"✓ {message}")

            # Show persona capabilities
            if persona.capabilities:
                caps_text = ", ".join(
                    [
                        cap.value.replace("_", " ").title()
                        for cap in persona.capabilities
                    ]
                )
                self.console.print(f"[dim]Capabilities: {caps_text}[/dim]")
        else:
            self._show_error(f"Failed to switch persona: {message}")

    async def handle_current_persona(self) -> None:
        """Show the currently active persona."""
        active_persona = self.persona_manager.active_persona

        if not active_persona:
            self._show_info("No agent persona is currently active.")
            self.console.print(
                "[dim]Use '/agent use <persona_id>' to activate a persona[/dim]"
            )
            return

        # Create a panel with current persona info
        info_lines = [
            f"[bold cyan]{active_persona.icon} {active_persona.name}[/bold cyan]",
            f"[yellow]ID:[/yellow] {active_persona.id}",
            f"[yellow]Category:[/yellow] {active_persona.category.value.title()}",
            f"[yellow]Status:[/yellow] {self._format_status(active_persona.status)}",
            "",
            f"[dim]{active_persona.description}[/dim]",
        ]

        if active_persona.capabilities:
            caps_text = ", ".join(
                [
                    cap.value.replace("_", " ").title()
                    for cap in active_persona.capabilities
                ]
            )
            info_lines.append("")
            info_lines.append(f"[yellow]Capabilities:[/yellow] {caps_text}")

        # Add usage stats if available
        if active_persona.metadata:
            info_lines.append("")
            info_lines.append(
                f"[dim]Usage Count: {active_persona.metadata.usage_count}[/dim]"
            )
            if active_persona.metadata.last_used:
                info_lines.append(
                    f"[dim]Last Used: {active_persona.metadata.last_used.strftime('%Y-%m-%d %H:%M')}[/dim]"
                )

        panel = Panel(
            "\n".join(info_lines),
            title="Current Agent Persona",
            border_style="green",
        )

        self.console.print(panel)

    async def handle_persona_info(self, persona_id: str) -> None:
        """
        Show detailed information about a persona.

        Args:
            persona_id: ID of the persona
        """
        persona = self.persona_manager.get_persona(persona_id.lower())

        if not persona:
            self._show_error(f"Persona '{persona_id}' not found.")
            self._suggest_personas(persona_id)
            return

        # Create detailed info panel
        info_sections = []

        # Basic info section
        basic_info = [
            f"[bold cyan]{persona.icon} {persona.name}[/bold cyan]",
            f"[yellow]ID:[/yellow] {persona.id}",
            f"[yellow]Category:[/yellow] {persona.category.value.title()}",
            f"[yellow]Status:[/yellow] {self._format_status(persona.status)}",
            "",
            f"{persona.description}",
        ]
        info_sections.append(Panel("\n".join(basic_info), title="Basic Information"))

        # Capabilities section
        if persona.capabilities:
            cap_list = []
            for cap in sorted(persona.capabilities, key=lambda c: c.value):
                cap_name = cap.value.replace("_", " ").title()
                cap_list.append(f"• {cap_name}")

            info_sections.append(Panel("\n".join(cap_list), title="Capabilities"))

        # Configuration section
        if persona.configuration:
            config_info = [
                f"[yellow]Template:[/yellow] {persona.configuration.template_id}",
                f"[yellow]Primary Provider:[/yellow] {persona.configuration.primary_provider}",
                f"[yellow]Tools Enabled:[/yellow] {'Yes' if persona.configuration.tools_enabled else 'No'}",
                f"[yellow]Web Search:[/yellow] {'Yes' if persona.configuration.web_search_enabled else 'No'}",
                f"[yellow]File Operations:[/yellow] {'Yes' if persona.configuration.file_operations_enabled else 'No'}",
                f"[yellow]Approval Required:[/yellow] {'Yes' if persona.configuration.approval_required else 'No'}",
            ]

            if (
                hasattr(persona.configuration, "fallback_providers")
                and persona.configuration.fallback_providers
            ):
                fallback_list = persona.configuration.fallback_providers
                if isinstance(fallback_list, (list, tuple)):
                    config_info.append(
                        f"[yellow]Fallback Providers:[/yellow] {', '.join(fallback_list)}"
                    )
                else:
                    config_info.append(
                        f"[yellow]Fallback Providers:[/yellow] {str(fallback_list)}"
                    )

            info_sections.append(Panel("\n".join(config_info), title="Configuration"))

        # Metadata section
        if persona.metadata:
            meta_info = [
                f"[yellow]Version:[/yellow] {persona.metadata.version}",
                f"[yellow]Author:[/yellow] {persona.metadata.author}",
                f"[yellow]Usage Count:[/yellow] {persona.metadata.usage_count}",
                f"[yellow]Built-in:[/yellow] {'Yes' if persona.metadata.is_builtin else 'No'}",
            ]

            if persona.metadata.created_at:
                meta_info.append(
                    f"[yellow]Created:[/yellow] {persona.metadata.created_at.strftime('%Y-%m-%d %H:%M')}"
                )

            if persona.metadata.last_used:
                meta_info.append(
                    f"[yellow]Last Used:[/yellow] {persona.metadata.last_used.strftime('%Y-%m-%d %H:%M')}"
                )

            if persona.metadata.tags:
                meta_info.append(
                    f"[yellow]Tags:[/yellow] {', '.join(persona.metadata.tags)}"
                )

            info_sections.append(Panel("\n".join(meta_info), title="Metadata"))

        # Display all sections
        for section in info_sections:
            self.console.print(section)

        # Show activation hint if not active
        if self.persona_manager.active_persona != persona:
            self.console.print(
                f"\n[dim]Use '/agent use {persona.id}' to activate this persona[/dim]"
            )

    async def handle_persona_status(self) -> None:
        """Show detailed agent persona status."""
        # Current persona
        active_persona = self.persona_manager.active_persona

        # Switch state
        switch_state = self.agent_switcher.get_current_state()

        # Session state info
        session_state = self.agent_switcher.current_session_state

        # Create status sections
        status_lines = []

        # Active persona section
        status_lines.append("[bold]Active Persona:[/bold]")
        if active_persona:
            status_lines.append(
                f"  {active_persona.icon} {active_persona.name} ({active_persona.id})"
            )
            status_lines.append(
                f"  Status: {self._format_status(active_persona.status)}"
            )
        else:
            status_lines.append("  [dim]No persona active[/dim]")

        status_lines.append("")

        # Switch state section
        status_lines.append("[bold]Switch State:[/bold]")
        status_lines.append(f"  {self._format_switch_state(switch_state)}")

        status_lines.append("")

        # Session state section
        status_lines.append("[bold]Session State:[/bold]")
        if session_state:
            status_lines.append(
                f"  Session ID: {session_state.session_id or '[dim]None[/dim]'}"
            )
            status_lines.append(
                f"  History Items: {len(session_state.conversation_history)}"
            )
            status_lines.append(f"  Active Tools: {len(session_state.active_tools)}")
            status_lines.append(
                f"  Active Operations: {len(session_state.active_operations)}"
            )
            if session_state.last_modified:
                status_lines.append(
                    f"  Last Modified: {session_state.last_modified.strftime('%H:%M:%S')}"
                )
        else:
            status_lines.append("  [dim]No session state[/dim]")

        status_lines.append("")

        # Manager stats
        stats = self.persona_manager.get_stats()
        status_lines.append("[bold]Manager Statistics:[/bold]")
        status_lines.append(f"  Total Personas: {stats['total_personas']}")
        status_lines.append(f"  Built-in: {stats['builtin_personas']}")
        status_lines.append(f"  Custom: {stats['custom_personas']}")
        status_lines.append(f"  Available: {stats['available_personas']}")

        panel = Panel(
            "\n".join(status_lines),
            title="🤖 Agent Persona Status",
            border_style="blue",
        )

        self.console.print(panel)

    async def handle_switch_history(self) -> None:
        """Show persona switch history."""
        history = self.agent_switcher.get_switch_history()

        if not history:
            self._show_info("No persona switch history available.")
            return

        # Create history table
        table = Table(
            title="Persona Switch History",
            box=ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("#", style="dim", width=3)
        table.add_column("Time", style="cyan")
        table.add_column("From", style="yellow")
        table.add_column("To", style="green")
        table.add_column("Reason", style="dim")
        table.add_column("Status", justify="center")

        for idx, context in enumerate(history[-10:], 1):  # Show last 10 switches
            from_name = (
                context.from_persona.name if context.from_persona else "[dim]None[/dim]"
            )
            to_name = context.to_persona.name
            time_str = context.timestamp.strftime("%H:%M:%S")

            # Determine status
            if context.error_message:
                status = "[red]✗ Failed[/red]"
            else:
                status = "[green]✓ Success[/green]"

            reason = (
                context.switch_reason[:30] + "..."
                if len(context.switch_reason) > 30
                else context.switch_reason
            )

            table.add_row(
                str(idx),
                time_str,
                from_name,
                to_name,
                reason or "[dim]Manual[/dim]",
                status,
            )

        self.console.print(table)

    def _show_agent_help(self) -> None:
        """Show help for agent commands."""
        help_text = """
[bold cyan]Agent Persona Commands:[/bold cyan]

[yellow]Basic Commands:[/yellow]
  /agent list              - List all available agent personas
  /agent use <persona>     - Switch to a specific persona
  /agent current           - Show the currently active persona
  /agent info <persona>    - Show detailed information about a persona

[yellow]Discovery & Selection:[/yellow]
  /agent discover          - Interactive persona discovery guide
  /agent recommend [query] - Get persona recommendations for a task
  /agent compare           - Compare capabilities of all personas
  /agent preview <persona> - Preview persona without switching

[yellow]Advanced Commands:[/yellow]
  /agent status            - Show detailed agent system status
  /agent history           - Show persona switch history

[yellow]Available Personas:[/yellow]
  • coding      - Optimized for software development
  • research    - Configured for research with web search
  • creative    - Set up for creative writing
  • performance - Optimized for speed and efficiency
  • general     - Balanced general-purpose configuration

[yellow]Examples:[/yellow]
  /agent discover                    - Start interactive discovery
  /agent recommend "debug python"    - Get recommendations for debugging
  /agent compare                     - See capability comparison table
  /agent preview coding              - Preview coding persona behavior
  /agent use coding                  - Switch to coding persona
"""
        panel = Panel(
            help_text.strip(), title="Agent Persona Help", border_style="cyan"
        )
        self.console.print(panel)

    def _format_status(self, status: PersonaStatus) -> str:
        """Format persona status with color."""
        status_colors = {
            PersonaStatus.AVAILABLE: "[green]Available[/green]",
            PersonaStatus.ACTIVE: "[bold green]Active[/bold green]",
            PersonaStatus.DISABLED: "[dim]Disabled[/dim]",
            PersonaStatus.ERROR: "[red]Error[/red]",
            PersonaStatus.LOADING: "[yellow]Loading...[/yellow]",
        }
        return status_colors.get(status, str(status.value))

    def _format_switch_state(self, state: SwitchState) -> str:
        """Format switch state with color."""
        state_colors = {
            SwitchState.IDLE: "[green]Idle[/green]",
            SwitchState.PREPARING: "[yellow]Preparing[/yellow]",
            SwitchState.SWITCHING: "[yellow]Switching[/yellow]",
            SwitchState.VALIDATING: "[yellow]Validating[/yellow]",
            SwitchState.COMPLETE: "[green]Complete[/green]",
            SwitchState.ERROR: "[red]Error[/red]",
            SwitchState.ROLLBACK: "[red]Rollback[/red]",
        }
        return state_colors.get(state, str(state.value))

    def _suggest_personas(self, typed: str) -> None:
        """Suggest similar persona names."""
        personas = self.persona_manager.get_all_personas()
        suggestions = []

        typed_lower = typed.lower()
        for pid, persona in personas.items():
            if typed_lower in pid or typed_lower in persona.name.lower():
                suggestions.append(f"{pid} ({persona.name})")

        if suggestions:
            self.console.print("[dim]Did you mean one of these?[/dim]")
            for suggestion in suggestions[:3]:
                self.console.print(f"  • {suggestion}")

    def _show_info(self, message: str) -> None:
        """Show info message."""
        self.console.print(f"[cyan]ℹ {message}[/cyan]")

    def _show_success(self, message: str) -> None:
        """Show success message."""
        self.console.print(f"[green]{message}[/green]")

    def _show_warning(self, message: str) -> None:
        """Show warning message."""
        self.console.print(f"[yellow]⚠ {message}[/yellow]")

    def _show_error(self, message: str) -> None:
        """Show error message."""
        self.console.print(f"[red]✗ {message}[/red]")

    async def handle_recommend_persona(self, user_query: Optional[str] = None) -> None:
        """
        Recommend a persona based on user query or task.

        Args:
            user_query: Optional user query to base recommendations on
        """
        self.console.print("\n[bold cyan]🎯 Persona Recommendations[/bold cyan]")

        personas = self.persona_manager.get_all_personas()
        if not personas:
            self._show_warning("No personas available for recommendations.")
            return

        # If no query provided, show general recommendations
        if not user_query:
            self.console.print(
                "\n[dim]Here are some general persona recommendations based on common tasks:[/dim]\n"
            )

            # Show recommended personas for common tasks
            recommendations = [
                ("💻 For software development and coding tasks", "coding"),
                ("🔍 For research and information gathering", "research"),
                ("✍️ For creative writing and content creation", "creative"),
                ("⚡ For quick responses and simple tasks", "performance"),
                ("🎯 For balanced general-purpose work", "general"),
            ]

            for task_desc, persona_id in recommendations:
                persona = personas.get(persona_id)
                if persona:
                    self.console.print(f"{task_desc}")
                    self.console.print(
                        f"  → Use: [cyan]/agent use {persona_id}[/cyan] ({persona.name})"
                    )
                    self.console.print("")

        else:
            # Analyze query and recommend appropriate persona
            self.console.print(f'[dim]Based on your query: "{user_query}"[/dim]\n')

            query_lower = user_query.lower()
            recommendations = []

            # Simple keyword-based matching
            if any(
                word in query_lower
                for word in [
                    "code",
                    "program",
                    "debug",
                    "develop",
                    "build",
                    "fix",
                ]
            ):
                if "coding" in personas:
                    recommendations.append(
                        (
                            personas["coding"],
                            "Optimized for programming and development tasks",
                        )
                    )

            if any(
                word in query_lower
                for word in [
                    "research",
                    "find",
                    "search",
                    "learn",
                    "study",
                    "investigate",
                ]
            ):
                if "research" in personas:
                    recommendations.append(
                        (
                            personas["research"],
                            "Enhanced with web search capabilities",
                        )
                    )

            if any(
                word in query_lower
                for word in [
                    "write",
                    "creative",
                    "story",
                    "content",
                    "blog",
                    "article",
                ]
            ):
                if "creative" in personas:
                    recommendations.append(
                        (
                            personas["creative"],
                            "Specialized for creative writing tasks",
                        )
                    )

            if any(
                word in query_lower
                for word in ["fast", "quick", "speed", "urgent", "simple"]
            ):
                if "performance" in personas:
                    recommendations.append(
                        (
                            personas["performance"],
                            "Optimized for speed and efficiency",
                        )
                    )

            # If no specific matches, recommend general
            if not recommendations and "general" in personas:
                recommendations.append(
                    (personas["general"], "Balanced for general-purpose tasks")
                )

            if recommendations:
                for persona, reason in recommendations[:3]:  # Show top 3
                    self.console.print(
                        f"[green]✓[/green] [bold]{persona.icon} {persona.name}[/bold]"
                    )
                    self.console.print(f"  {reason}")
                    self.console.print(f"  Use: [cyan]/agent use {persona.id}[/cyan]")
                    self.console.print("")
            else:
                self.console.print(
                    "[dim]No specific recommendations found. Try /agent discover for interactive selection.[/dim]"
                )

        # Show additional help
        self.console.print(
            "[dim]💡 Tip: Use '/agent compare' to see capability differences[/dim]"
        )
        self.console.print(
            "[dim]💡 Tip: Use '/agent preview <persona>' to test without switching[/dim]"
        )

    async def handle_compare_personas(self) -> None:
        """Show capability comparison matrix for all personas."""
        personas = self.persona_manager.get_all_personas()
        if not personas:
            self._show_warning("No personas available for comparison.")
            return

        # Create comparison table
        table = Table(
            title="🔀 Persona Capability Comparison",
            box=ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Persona", style="bold white", width=12)
        table.add_column("Category", style="yellow", width=12)
        table.add_column("Tools", justify="center", width=8)
        table.add_column("Web Search", justify="center", width=10)
        table.add_column("File Ops", justify="center", width=8)
        table.add_column("Speed", justify="center", width=8)
        table.add_column("Best For", style="dim", width=25)

        # Sort personas by category and name
        sorted_personas = sorted(
            personas.values(), key=lambda p: (p.category.value, p.name)
        )

        for persona in sorted_personas:
            # Get capabilities from configuration
            config = persona.configuration
            tools_enabled = "✅" if (config and config.tools_enabled) else "❌"
            web_search = (
                "✅"
                if (config and getattr(config, "web_search_enabled", False))
                else "❌"
            )
            file_ops = (
                "✅"
                if (config and getattr(config, "file_operations_enabled", False))
                else "❌"
            )

            # Determine speed based on persona type
            speed_rating = (
                "⚡"
                if persona.id == "performance"
                else "🐢" if persona.id == "research" else "⚖️"
            )

            # Best use case
            use_cases = {
                "coding": "Software development",
                "research": "Research & analysis",
                "creative": "Writing & content",
                "performance": "Quick tasks",
                "general": "Balanced workflows",
            }
            best_for = use_cases.get(persona.id, persona.description[:25])

            table.add_row(
                f"{persona.icon} {persona.name}",
                persona.category.value.title(),
                tools_enabled,
                web_search,
                file_ops,
                speed_rating,
                best_for,
            )

        self.console.print(table)

        # Legend
        legend_lines = [
            "[bold]Legend:[/bold]",
            "✅ = Enabled/Available  ❌ = Disabled/Unavailable",
            "⚡ = Fast  ⚖️ = Balanced  🐢 = Thorough",
            "",
            "[dim]💡 Use '/agent info <persona>' for detailed information[/dim]",
            "[dim]💡 Use '/agent use <persona>' to switch to a persona[/dim]",
        ]

        legend_panel = Panel(
            "\n".join(legend_lines), title="Usage Guide", border_style="dim"
        )

        self.console.print("\n")
        self.console.print(legend_panel)

    async def handle_preview_persona(self, persona_id: str) -> None:
        """
        Preview a persona without switching to it.

        Args:
            persona_id: ID of the persona to preview
        """
        persona = self.persona_manager.get_persona(persona_id.lower())
        if not persona:
            self._show_error(f"Persona '{persona_id}' not found.")
            self._suggest_personas(persona_id)
            return

        # Show preview header
        self.console.print(
            f"\n[bold cyan]👁️ Preview: {persona.icon} {persona.name}[/bold cyan]"
        )
        self.console.print(
            f"[dim]This is how '{persona.name}' would behave if activated[/dim]\n"
        )

        # Show configuration preview
        config_lines = [
            f"[bold]Configuration Preview[/bold]",
            f"• Template: {persona.configuration.template_id if persona.configuration else 'None'}",
            f"• Provider: {persona.configuration.primary_provider if persona.configuration else 'None'}",
        ]

        if persona.configuration:
            config_lines.extend(
                [
                    f"• Tools: {'✅ Enabled' if persona.configuration.tools_enabled else '❌ Disabled'}",
                    f"• Web Search: {'✅ Enabled' if getattr(persona.configuration, 'web_search_enabled', False) else '❌ Disabled'}",
                    f"• File Operations: {'✅ Enabled' if getattr(persona.configuration, 'file_operations_enabled', False) else '❌ Disabled'}",
                    f"• Approval Required: {'✅ Yes' if getattr(persona.configuration, 'approval_required', True) else '❌ No'}",
                ]
            )

        config_panel = Panel(
            "\n".join(config_lines), title="Configuration", border_style="blue"
        )

        # Show capabilities preview
        if persona.capabilities:
            caps_text = ", ".join(
                [cap.value.replace("_", " ").title() for cap in persona.capabilities]
            )
            caps_panel = Panel(caps_text, title="Capabilities", border_style="green")
        else:
            caps_panel = Panel(
                "[dim]No specific capabilities defined[/dim]",
                title="Capabilities",
                border_style="dim",
            )

        # Show behavioral description
        behavior_lines = [
            persona.description,
            "",
            "[bold]Expected Behavior:[/bold]",
        ]

        # Add behavior predictions based on persona type
        behavior_predictions = {
            "coding": [
                "• Will focus on code quality and best practices",
                "• Provides detailed technical explanations",
                "• Suggests testing and documentation",
                "• Uses development tools and file operations",
            ],
            "research": [
                "• Will search for current information online",
                "• Provides comprehensive analysis",
                "• Cites sources and references",
                "• Takes more time for thorough responses",
            ],
            "creative": [
                "• Focuses on engaging and original content",
                "• Uses varied writing styles and techniques",
                "• Emphasizes creativity over strict accuracy",
                "• May suggest multiple creative approaches",
            ],
            "performance": [
                "• Provides quick, concise responses",
                "• Prioritizes speed over depth",
                "• Uses minimal tools to avoid delays",
                "• Best for simple, straightforward tasks",
            ],
            "general": [
                "• Provides balanced responses",
                "• Adapts approach to task requirements",
                "• Uses moderate depth and detail",
                "• Good all-around performance",
            ],
        }

        if persona.id in behavior_predictions:
            behavior_lines.extend(behavior_predictions[persona.id])
        else:
            behavior_lines.append(
                "• Behavior will depend on configured template and settings"
            )

        behavior_panel = Panel(
            "\n".join(behavior_lines),
            title="Behavioral Preview",
            border_style="yellow",
        )

        # Display all panels
        self.console.print(config_panel)
        self.console.print(caps_panel)
        self.console.print(behavior_panel)

        # Show action options
        action_lines = [
            f"[bold]Ready to switch?[/bold]",
            f"• [cyan]/agent use {persona.id}[/cyan] - Switch to this persona",
            f"• [cyan]/agent compare[/cyan] - Compare with other personas",
            f"• [cyan]/agent info {persona.id}[/cyan] - View detailed information",
        ]

        action_panel = Panel(
            "\n".join(action_lines), title="Next Steps", border_style="cyan"
        )

        self.console.print(action_panel)

    async def handle_discover_personas(self) -> None:
        """Provide interactive persona discovery interface."""
        self.console.print("\n[bold cyan]🧭 Persona Discovery[/bold cyan]")

        personas = self.persona_manager.get_all_personas()
        if not personas:
            self._show_warning("No personas available for discovery.")
            return

        # Show discovery intro
        intro_lines = [
            "[bold]Welcome to Persona Discovery![/bold]",
            "",
            "This guide will help you find the perfect agent persona for your needs.",
            "Each persona is optimized for different types of tasks and workflows.",
            "",
            "[dim]💡 Tip: You can always switch personas later with '/agent use <name>'[/dim]",
        ]

        intro_panel = Panel(
            "\n".join(intro_lines),
            title="Getting Started",
            border_style="cyan",
        )

        self.console.print(intro_panel)

        # Group personas by category
        categories = {}
        for persona in personas.values():
            cat = persona.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(persona)

        # Show personas by category
        for category, persona_list in categories.items():
            category_title = f"📂 {category.title()} Personas"
            self.console.print(f"\n[bold yellow]{category_title}[/bold yellow]")

            for persona in sorted(persona_list, key=lambda p: p.name):
                # Create persona card
                card_lines = [
                    f"[bold]{persona.icon} {persona.name}[/bold]",
                    persona.description,
                    "",
                ]

                # Add key features
                features = []
                if persona.configuration:
                    if persona.configuration.tools_enabled:
                        features.append("🛠️ Tools enabled")
                    if getattr(persona.configuration, "web_search_enabled", False):
                        features.append("🌐 Web search")
                    if getattr(persona.configuration, "file_operations_enabled", False):
                        features.append("📁 File operations")

                if features:
                    card_lines.extend(
                        [
                            "[dim]Key features:[/dim]",
                            "[dim]" + " • ".join(features) + "[/dim]",
                            "",
                        ]
                    )

                # Add usage stats
                if persona.metadata and persona.metadata.usage_count > 0:
                    card_lines.append(
                        f"[dim]Used {persona.metadata.usage_count} times[/dim]"
                    )

                # Add quick actions
                card_lines.extend(
                    [
                        "",
                        f"[cyan]Preview:[/cyan] /agent preview {persona.id}",
                        f"[green]Use:[/green] /agent use {persona.id}",
                    ]
                )

                # Show card
                persona_panel = Panel(
                    "\n".join(card_lines),
                    title=f"{persona.id}",
                    border_style=(
                        "blue"
                        if persona == self.persona_manager.active_persona
                        else "dim"
                    ),
                    width=50,
                )

                self.console.print(persona_panel)

        # Show discovery footer
        footer_lines = [
            "[bold]Need help choosing?[/bold]",
            "",
            '• [cyan]/agent recommend "<your task>"[/cyan] - Get recommendations for a specific task',
            "• [cyan]/agent compare[/cyan] - See side-by-side capability comparison",
            "• [cyan]/agent info <persona>[/cyan] - Get detailed information about a persona",
            "• [cyan]/agent preview <persona>[/cyan] - Preview how a persona would behave",
            "",
            "[bold]Popular starting choices:[/bold]",
            "• [green]coding[/green] - For developers and programmers",
            "• [green]research[/green] - For information gathering and analysis",
            "• [green]general[/green] - For balanced, general-purpose work",
        ]

        footer_panel = Panel(
            "\n".join(footer_lines), title="Next Steps", border_style="green"
        )

        self.console.print("\n")
        self.console.print(footer_panel)
