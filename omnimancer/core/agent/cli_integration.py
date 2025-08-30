"""
CLI Integration for Status-Enhanced Agent System.

This module provides integration between the CLI interface and the
status-enhanced agent engine, enabling status display in CLI commands.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from ..config_manager import ConfigManager
from .status_integrated_engine import (
    StatusIntegratedAgentEngine,
    create_status_integrated_engine,
)
from .status_ui_components import (
    ComprehensiveStatusDisplay,
)
from .terminal_status_listener import TerminalStatusListener

logger = logging.getLogger(__name__)


class CLIStatusIntegration:
    """
    Integration layer between CLI and status-enhanced agent system.

    Provides methods for CLI commands to interact with and display
    agent status information.
    """

    def __init__(
        self,
        engine: StatusIntegratedAgentEngine,
        console: Optional[Console] = None,
    ):
        """
        Initialize CLI status integration.

        Args:
            engine: Status-integrated agent engine
            console: Rich console for output
        """
        self.engine = engine
        self.console = console or Console()
        self.status_listener: Optional[TerminalStatusListener] = None
        self.live_display: Optional[Live] = None
        self.ui_display = ComprehensiveStatusDisplay()

    async def start_live_status_display(self) -> bool:
        """
        Start live status display in the terminal.

        Returns:
            True if display started successfully
        """
        try:
            if not self.engine.is_status_display_enabled():
                self.console.print("[yellow]Status display is disabled[/yellow]")
                return False

            # Ensure status monitoring is started
            await self.engine.start_status_monitoring()

            # Create terminal listener if not exists
            if not self.status_listener:
                self.status_listener = TerminalStatusListener(
                    "cli_status_display", console=self.console
                )

                # Add to status manager (which handles streaming)
                await self.engine.status_manager.add_listener(self.status_listener)

            # Start live display
            await self.status_listener.start_live_display()

            self.console.print("[green]✅ Live status display started[/green]")
            return True

        except Exception as e:
            logger.error(f"Failed to start live status display: {e}")
            self.console.print(f"[red]❌ Failed to start status display: {e}[/red]")
            return False

    async def stop_live_status_display(self) -> bool:
        """
        Stop live status display.

        Returns:
            True if display stopped successfully
        """
        try:
            if self.status_listener:
                await self.status_listener.stop_live_display()

                # Remove from status manager
                await self.engine.status_manager.remove_listener(
                    self.status_listener.listener_id
                )

                self.status_listener = None

            self.console.print("[yellow]⏹️ Live status display stopped[/yellow]")
            return True

        except Exception as e:
            logger.error(f"Failed to stop live status display: {e}")
            self.console.print(f"[red]❌ Failed to stop status display: {e}[/red]")
            return False

    async def show_status_snapshot(self, detailed: bool = False) -> None:
        """
        Show a snapshot of current status.

        Args:
            detailed: Whether to show detailed information
        """
        try:
            if not self.engine.is_status_display_enabled():
                self.console.print("[yellow]Status display is disabled[/yellow]")
                return

            # Get status data
            status_summary = await self.engine.get_status_summary()

            if "error" in status_summary:
                self.console.print(
                    f"[red]❌ Error getting status: {status_summary['error']}[/red]"
                )
                return

            # Display basic info
            status_summary.get("status", "unknown")
            status_summary.get("active_operations_count", 0)

            # Create status display
            if detailed:
                self._show_detailed_status(status_summary)
            else:
                self._show_basic_status(status_summary)

        except Exception as e:
            logger.error(f"Failed to show status snapshot: {e}")
            self.console.print(f"[red]❌ Failed to get status: {e}[/red]")

    def _show_basic_status(self, status_summary: Dict[str, Any]) -> None:
        """Show basic status information."""
        agent_id = status_summary.get("agent_id", "unknown")
        status = status_summary.get("status", "unknown")
        active_ops = status_summary.get("active_operations_count", 0)

        # Status color coding
        if status == "enabled":
            status_color = "green"
            status_emoji = "🟢"
        elif status == "running":
            status_color = "blue"
            status_emoji = "🔵"
        elif status == "error":
            status_color = "red"
            status_emoji = "🔴"
        else:
            status_color = "yellow"
            status_emoji = "🟡"

        content = f"[bold]Agent Status[/bold]\n\n"
        content += f"Agent ID: [cyan]{agent_id}[/cyan]\n"
        content += f"Status: [{status_color}]{status_emoji} {status.title()}[/{status_color}]\n"
        content += f"Active Operations: [blue]{active_ops}[/blue]\n"

        if status_summary.get("status_stream_running", False):
            content += f"Status Stream: [green]🟢 Running[/green]\n"
        else:
            content += f"Status Stream: [red]🔴 Stopped[/red]\n"

        self.console.print(Panel(content, title="📊 Agent Status", border_style="blue"))

    def _show_detailed_status(self, status_summary: Dict[str, Any]) -> None:
        """Show detailed status information."""
        # Basic info
        self._show_basic_status(status_summary)

        # Active operations
        active_operations = status_summary.get("active_operations", [])
        if active_operations:
            self.console.print("\n[bold]Active Operations:[/bold]")

            for op in active_operations:
                op_type = op.get("type", "unknown")
                description = op.get("description", "No description")
                progress = op.get("progress", 0)
                duration = op.get("duration", 0)

                # Progress bar
                bar_width = 20
                filled = int((progress / 100) * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)

                op_info = f"  [{progress:5.1f}%] {bar} {op_type}: {description}"
                if duration > 0:
                    op_info += f" ({duration:.1f}s)"

                self.console.print(op_info)

        # System stats
        system_stats = status_summary.get("system_stats", {})
        if system_stats:
            self.console.print(f"\n[bold]System Statistics:[/bold]")
            self.console.print(
                f"  Events Emitted: {system_stats.get('events_emitted', 0)}"
            )
            self.console.print(
                f"  Operations Started: {system_stats.get('operations_started', 0)}"
            )
            self.console.print(
                f"  Operations Completed: {system_stats.get('operations_completed', 0)}"
            )
            self.console.print(
                f"  Operations Failed: {system_stats.get('operations_failed', 0)}"
            )

    async def execute_with_status_display(
        self, operation_func: Callable, *args, **kwargs
    ) -> Any:
        """
        Execute a function while displaying status updates.

        Args:
            operation_func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the function
        """
        try:
            # Start live display if not already running
            display_started = False
            if not self.status_listener:
                display_started = await self.start_live_status_display()

            # Execute the operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)

            # Stop display if we started it
            if display_started:
                await self.stop_live_status_display()

            return result

        except Exception:
            # Stop display if we started it
            if display_started and self.status_listener:
                await self.stop_live_status_display()
            raise

    def is_live_display_running(self) -> bool:
        """Check if live display is currently running."""
        return (
            self.status_listener is not None
            and hasattr(self.status_listener, "live_display")
            and self.status_listener.live_display is not None
        )

    async def get_quick_status(self) -> str:
        """Get a quick one-line status summary."""
        try:
            if not self.engine.is_status_display_enabled():
                return "Status display disabled"

            status_summary = await self.engine.get_status_summary()

            if "error" in status_summary:
                return f"Status error: {status_summary['error']}"

            agent_id = status_summary.get("agent_id", "agent")
            status = status_summary.get("status", "unknown")
            active_ops = status_summary.get("active_operations_count", 0)

            return f"{agent_id}: {status} ({active_ops} active operations)"

        except Exception as e:
            return f"Status unavailable: {e}"


class CLIStatusCommand:
    """
    CLI command handler for status-related operations.

    Provides subcommands for managing agent status display.
    """

    def __init__(self, integration: CLIStatusIntegration):
        """
        Initialize status command handler.

        Args:
            integration: CLI status integration instance
        """
        self.integration = integration
        self.console = integration.console

    async def handle_command(self, args: List[str]) -> None:
        """
        Handle status command with subcommands.

        Args:
            args: Command arguments
        """
        if not args:
            args = ["show"]  # Default to show

        subcommand = args[0].lower()

        try:
            if subcommand == "show":
                detailed = "--detailed" in args or "-d" in args
                await self.integration.show_status_snapshot(detailed)

            elif subcommand == "start":
                success = await self.integration.start_live_status_display()
                if success:
                    self.console.print(
                        "[green]Live status display started. Press Ctrl+C to stop.[/green]"
                    )
                    try:
                        # Keep running until interrupted
                        while self.integration.is_live_display_running():
                            await asyncio.sleep(1)
                    except KeyboardInterrupt:
                        await self.integration.stop_live_status_display()
                        self.console.print(
                            "\n[yellow]Live status display stopped.[/yellow]"
                        )

            elif subcommand == "stop":
                await self.integration.stop_live_status_display()

            elif subcommand == "quick":
                status = await self.integration.get_quick_status()
                self.console.print(status)

            elif subcommand in ["help", "--help", "-h"]:
                self._show_help()

            else:
                self.console.print(
                    f"[red]Unknown status subcommand: {subcommand}[/red]"
                )
                self._show_help()

        except Exception as e:
            logger.error(f"Status command failed: {e}")
            self.console.print(f"[red]❌ Status command failed: {e}[/red]")

    def _show_help(self) -> None:
        """Show help for status commands."""
        help_text = """[bold]Status Commands:[/bold]

[cyan]/status show[/cyan]              - Show current status snapshot
[cyan]/status show --detailed[/cyan]   - Show detailed status information
[cyan]/status start[/cyan]             - Start live status display
[cyan]/status stop[/cyan]              - Stop live status display  
[cyan]/status quick[/cyan]             - Get quick one-line status
[cyan]/status help[/cyan]              - Show this help message

[bold]Examples:[/bold]
[cyan]/status show -d[/cyan]           - Show detailed status
[cyan]/status start[/cyan]             - Start live monitoring
"""

        self.console.print(
            Panel(help_text, title="📊 Status Command Help", border_style="blue")
        )


# Factory function for creating CLI integration
async def create_cli_status_integration(
    config_manager: ConfigManager, console: Optional[Console] = None
) -> CLIStatusIntegration:
    """
    Create a CLI status integration with a status-enhanced engine.

    Args:
        config_manager: Configuration manager
        console: Rich console for output

    Returns:
        Configured CLI status integration
    """
    try:
        # Create status-integrated engine
        engine = await create_status_integrated_engine(config_manager)

        # Create CLI integration
        integration = CLIStatusIntegration(engine, console)

        logger.info("CLI status integration created successfully")
        return integration

    except Exception as e:
        logger.error(f"Failed to create CLI status integration: {e}")
        raise


# Utility function to add status integration to existing engines
def add_status_integration_to_engine(
    engine, console: Optional[Console] = None
) -> CLIStatusIntegration:
    """
    Add status integration to an existing engine (if it supports it).

    Args:
        engine: Existing engine instance
        console: Rich console for output

    Returns:
        CLI status integration instance
    """
    if isinstance(engine, StatusIntegratedAgentEngine):
        return CLIStatusIntegration(engine, console)
    else:
        logger.warning(
            f"Engine {type(engine).__name__} does not support status integration"
        )
        # Return a dummy integration that does nothing
        return DummyCLIStatusIntegration(console)


class DummyCLIStatusIntegration:
    """Dummy implementation for engines that don't support status integration."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    async def start_live_status_display(self) -> bool:
        self.console.print(
            "[yellow]Status display not available for this engine[/yellow]"
        )
        return False

    async def stop_live_status_display(self) -> bool:
        return True

    async def show_status_snapshot(self, detailed: bool = False) -> None:
        self.console.print(
            "[yellow]Status information not available for this engine[/yellow]"
        )

    async def execute_with_status_display(
        self, operation_func: Callable, *args, **kwargs
    ) -> Any:
        # Just execute without status display
        if asyncio.iscoroutinefunction(operation_func):
            return await operation_func(*args, **kwargs)
        else:
            return operation_func(*args, **kwargs)

    def is_live_display_running(self) -> bool:
        return False

    async def get_quick_status(self) -> str:
        return "Status not available"
