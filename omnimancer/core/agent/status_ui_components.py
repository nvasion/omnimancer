"""
Rich-based Terminal UI Components for Agent Status Display.

This module provides reusable Rich UI components for displaying agent status
information in a comprehensive and user-friendly terminal interface.
"""

from datetime import datetime
from typing import Any, Dict, List, Union

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .status_core import (
    AgentEvent,
    AgentOperation,
    AgentStatus,
    EventType,
    OperationStatus,
)


class AgentStatusTable:
    """Rich table component for displaying agent status information."""

    def __init__(self, title: str = "Agent Status"):
        self.title = title

    def create_table(
        self,
        agent_statuses: Dict[str, AgentStatus],
        agent_metadata: Dict[str, Dict[str, Any]] = None,
    ) -> Table:
        """
        Create a Rich table displaying agent statuses.

        Args:
            agent_statuses: Dictionary of agent IDs to their current status
            agent_metadata: Optional metadata for each agent

        Returns:
            Rich Table object
        """
        table = Table(title=self.title, show_header=True, header_style="bold magenta")

        table.add_column("Agent ID", style="cyan", width=12)
        table.add_column("Status", style="bold", width=12)
        table.add_column("State", width=15)
        table.add_column("Last Update", style="dim", width=12)
        table.add_column("Operations", justify="right", width=10)

        agent_metadata = agent_metadata or {}

        for agent_id, status in agent_statuses.items():
            metadata = agent_metadata.get(agent_id, {})

            # Status with emoji and color
            status_display = self._format_agent_status(status)

            # State information
            state = metadata.get("state", "Unknown")

            # Last update time
            last_update = metadata.get("last_update", datetime.now())
            if isinstance(last_update, datetime):
                time_str = last_update.strftime("%H:%M:%S")
            else:
                time_str = str(last_update)

            # Active operations count
            operations_count = metadata.get("active_operations", 0)

            table.add_row(
                agent_id,
                status_display,
                state,
                time_str,
                str(operations_count),
            )

        return table

    def _format_agent_status(self, status: AgentStatus) -> Text:
        """Format agent status with appropriate color and emoji."""
        status_mapping = {
            AgentStatus.ENABLED: ("🟢", "green", "Enabled"),
            AgentStatus.DISABLED: ("🔴", "red", "Disabled"),
            AgentStatus.RUNNING: ("🔵", "blue", "Running"),
            AgentStatus.PAUSED: ("🟡", "yellow", "Paused"),
            AgentStatus.ERROR: ("❌", "red", "Error"),
            AgentStatus.INITIALIZING: ("🔄", "blue", "Starting"),
            AgentStatus.SHUTTING_DOWN: ("⏹️", "orange", "Stopping"),
        }

        emoji, color, text = status_mapping.get(status, ("❓", "white", status.value))

        return Text(f"{emoji} {text}", style=color)


class OperationProgress:
    """Rich progress component for displaying operation progress."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeElapsedColumn(),
            console=None,
            auto_refresh=True,
            refresh_per_second=2,
        )

    def create_progress_display(
        self, operations: List[AgentOperation]
    ) -> Union[Progress, Text]:
        """
        Create a progress display for active operations.

        Args:
            operations: List of operations to display

        Returns:
            Progress widget or Text if no operations
        """
        if not operations:
            return Text("No active operations", style="dim")

        # Clear existing tasks
        for task_id in list(self.progress.task_ids):
            self.progress.remove_task(task_id)

        # Add current operations
        for operation in operations:
            if operation.is_active:
                description = (
                    f"[{operation.agent_id or 'system'}] {operation.description[:40]}"
                )
                if len(operation.description) > 40:
                    description += "..."

                self.progress.add_task(
                    description=description,
                    total=100,
                    completed=operation.progress_percentage,
                )

        return (
            self.progress
            if self.progress.task_ids
            else Text("No active operations", style="dim")
        )


class EventLogPanel:
    """Rich panel component for displaying recent events."""

    def __init__(self, max_events: int = 10):
        self.max_events = max_events

    def create_event_log(
        self, events: List[AgentEvent], title: str = "Recent Events"
    ) -> Panel:
        """
        Create a panel displaying recent events.

        Args:
            events: List of recent events
            title: Panel title

        Returns:
            Rich Panel object
        """
        if not events:
            content = Text("No recent events", style="dim")
        else:
            # Take only the most recent events
            recent_events = (
                events[-self.max_events :] if len(events) > self.max_events else events
            )

            lines = []
            for event in recent_events:
                # Format timestamp
                timestamp = event.timestamp.strftime("%H:%M:%S.%f")[
                    :-3
                ]  # Include milliseconds

                # Format event type with color
                event_display = self._format_event_type(event.event_type)

                # Agent ID
                agent_id = event.agent_id or "system"

                # Event description based on type and data
                description = self._get_event_description(event)

                # Combine into a line
                line = (
                    f"[dim]{timestamp}[/dim] [{agent_id}] {event_display} {description}"
                )
                lines.append(line)

            content = "\n".join(lines)

        return Panel(content, title=title, border_style="yellow", padding=(0, 1))

    def _format_event_type(self, event_type: EventType) -> Text:
        """Format event type with appropriate color."""
        type_colors = {
            EventType.AGENT_STATE_CHANGED: ("blue", "📝"),
            EventType.OPERATION_STARTED: ("green", "▶️"),
            EventType.OPERATION_PROGRESS: ("cyan", "📊"),
            EventType.OPERATION_COMPLETED: ("green", "✅"),
            EventType.OPERATION_FAILED: ("red", "❌"),
            EventType.OPERATION_CANCELLED: ("yellow", "⏹️"),
            EventType.ERROR_OCCURRED: ("red", "🚨"),
            EventType.APPROVAL_REQUESTED: ("magenta", "🔐"),
            EventType.APPROVAL_GRANTED: ("green", "🔓"),
            EventType.APPROVAL_DENIED: ("red", "🔒"),
        }

        color, emoji = type_colors.get(event_type, ("white", "📌"))
        return Text(f"{emoji}", style=color)

    def _get_event_description(self, event: AgentEvent) -> str:
        """Generate event description based on event type and data."""
        if event.event_type == EventType.AGENT_STATE_CHANGED:
            old_status = event.data.get("old_status", "unknown")
            new_status = event.data.get("new_status", "unknown")
            return f"Status: {old_status} → {new_status}"

        elif event.event_type == EventType.OPERATION_STARTED:
            op_type = event.data.get("operation_type", "unknown")
            description = event.data.get("description", "No description")
            return (
                f"{op_type}: {description[:30]}{'...' if len(description) > 30 else ''}"
            )

        elif event.event_type == EventType.OPERATION_PROGRESS:
            progress = event.data.get("progress", 0)
            return f"Progress: {progress:.1f}%"

        elif event.event_type == EventType.OPERATION_COMPLETED:
            duration = event.data.get("duration", 0)
            return f"Completed in {duration:.2f}s"

        elif event.event_type == EventType.OPERATION_FAILED:
            error = event.data.get("error", "Unknown error")
            return f"Failed: {error[:30]}{'...' if len(error) > 30 else ''}"

        elif event.event_type == EventType.ERROR_OCCURRED:
            error = event.data.get("error", "Unknown error")
            return f"Error: {error[:30]}{'...' if len(error) > 30 else ''}"

        else:
            return event.event_type.value.replace("_", " ").title()


class SystemStatsPanel:
    """Rich panel component for displaying system statistics."""

    def create_stats_panel(
        self, stats: Dict[str, Any], title: str = "System Statistics"
    ) -> Panel:
        """
        Create a panel displaying system statistics.

        Args:
            stats: Dictionary of statistics
            title: Panel title

        Returns:
            Rich Panel object
        """
        # Create a table for stats
        table = Table(show_header=False, show_edge=False, padding=(0, 2))
        table.add_column("Metric", style="bold", width=20)
        table.add_column("Value", justify="right", width=10)

        # Add key statistics
        key_stats = [
            ("Active Agents", stats.get("enabled_agents", 0)),
            ("Total Agents", stats.get("total_agents", 0)),
            ("Operations", f"{stats.get('operations_started', 0)}"),
            ("Completed", f"{stats.get('operations_completed', 0)}"),
            ("Failed", f"{stats.get('operations_failed', 0)}"),
            ("Events", f"{stats.get('events_emitted', 0)}"),
            ("Listeners", f"{stats.get('event_listeners', 0)}"),
        ]

        for metric, value in key_stats:
            # Color coding for certain metrics
            if metric == "Failed" and int(str(value)) > 0:
                value_style = "red"
            elif metric in ["Completed", "Active Agents"] and int(str(value)) > 0:
                value_style = "green"
            else:
                value_style = "white"

            table.add_row(metric, Text(str(value), style=value_style))

        return Panel(table, title=title, border_style="blue", padding=(0, 1))


class OperationDetailsTree:
    """Rich tree component for displaying operation hierarchy."""

    def create_operation_tree(
        self, operations: List[AgentOperation], title: str = "Operations"
    ) -> Panel:
        """
        Create a tree view of operations grouped by agent.

        Args:
            operations: List of operations to display
            title: Tree title

        Returns:
            Rich Panel containing the tree
        """
        if not operations:
            return Panel(
                Text("No operations to display", style="dim"),
                title=title,
                border_style="green",
            )

        # Group operations by agent
        by_agent: Dict[str, List[AgentOperation]] = {}
        for operation in operations:
            agent_id = operation.agent_id or "system"
            if agent_id not in by_agent:
                by_agent[agent_id] = []
            by_agent[agent_id].append(operation)

        # Create tree
        tree = Tree(f"[bold]{title}[/bold]")

        for agent_id, agent_operations in by_agent.items():
            # Add agent branch
            agent_branch = tree.add(
                f"[cyan]{agent_id}[/cyan] ({len(agent_operations)} operations)"
            )

            for operation in agent_operations[:5]:  # Limit to 5 operations per agent
                # Format operation
                status_emoji = {
                    OperationStatus.PENDING: "⏳",
                    OperationStatus.RUNNING: "▶️",
                    OperationStatus.COMPLETED: "✅",
                    OperationStatus.FAILED: "❌",
                    OperationStatus.CANCELLED: "⏹️",
                    OperationStatus.WAITING_APPROVAL: "🔐",
                }.get(operation.status, "❓")

                duration_str = ""
                if operation.duration:
                    duration_str = f" ({operation.duration.total_seconds():.1f}s)"

                operation_text = (
                    f"{status_emoji} {operation.operation_type.value} "
                    f"[{operation.progress_percentage:.0f}%]{duration_str}"
                )

                if operation.description:
                    operation_text += f"\n    {operation.description[:50]}"
                    if len(operation.description) > 50:
                        operation_text += "..."

                agent_branch.add(operation_text)

            if len(agent_operations) > 5:
                agent_branch.add(
                    f"[dim]... and {len(agent_operations) - 5} more operations[/dim]"
                )

        return Panel(tree, title=title, border_style="green")


class ComprehensiveStatusDisplay:
    """
    Comprehensive status display combining all UI components.

    Creates a rich, multi-panel display showing all aspects of agent status.
    """

    def __init__(self):
        self.agent_table = AgentStatusTable()
        self.operation_progress = OperationProgress()
        self.event_log = EventLogPanel()
        self.stats_panel = SystemStatsPanel()
        self.operation_tree = OperationDetailsTree()

    def create_comprehensive_display(
        self,
        agent_statuses: Dict[str, AgentStatus],
        agent_metadata: Dict[str, Dict[str, Any]],
        active_operations: List[AgentOperation],
        recent_events: List[AgentEvent],
        system_stats: Dict[str, Any],
    ) -> Layout:
        """
        Create a comprehensive status display with all components.

        Args:
            agent_statuses: Current agent statuses
            agent_metadata: Agent metadata
            active_operations: Active operations
            recent_events: Recent events
            system_stats: System statistics

        Returns:
            Rich Layout object
        """
        # Create layout
        layout = Layout()

        # Split into main sections
        layout.split_row(Layout(name="left"), Layout(name="right"))

        # Left side: Agent info and operations
        layout["left"].split_column(
            Layout(
                self.agent_table.create_table(agent_statuses, agent_metadata),
                name="agents",
            ),
            Layout(
                self.operation_progress.create_progress_display(active_operations),
                name="progress",
            ),
        )

        # Right side: Events, stats, and operation tree
        layout["right"].split_column(
            Layout(self.event_log.create_event_log(recent_events), name="events"),
            Layout(name="bottom_right"),
        )

        # Split bottom right into stats and tree
        layout["bottom_right"].split_row(
            Layout(self.stats_panel.create_stats_panel(system_stats), name="stats"),
            Layout(
                self.operation_tree.create_operation_tree(active_operations),
                name="tree",
            ),
        )

        return layout

    def create_simple_display(
        self,
        agent_statuses: Dict[str, AgentStatus],
        recent_events: List[AgentEvent],
        system_stats: Dict[str, Any],
    ) -> Group:
        """
        Create a simpler status display for constrained terminals.

        Args:
            agent_statuses: Current agent statuses
            recent_events: Recent events
            system_stats: System statistics

        Returns:
            Rich Group object
        """
        components = [
            self.stats_panel.create_stats_panel(system_stats, "📊 System Status"),
            Rule(),
            self.agent_table.create_table(agent_statuses),
            Rule(),
            self.event_log.create_event_log(recent_events, "📝 Recent Activity"),
        ]

        return Group(*components)


# Utility functions for creating common display components
def create_agent_status_summary(
    agent_statuses: Dict[str, AgentStatus],
) -> Text:
    """Create a brief text summary of agent statuses."""
    if not agent_statuses:
        return Text("No agents available", style="dim")

    enabled = sum(
        1 for status in agent_statuses.values() if status == AgentStatus.ENABLED
    )
    total = len(agent_statuses)

    summary = f"Agents: {enabled}/{total} enabled"

    if enabled > 0:
        style = "green"
    elif total > 0:
        style = "yellow"
    else:
        style = "red"

    return Text(summary, style=style)


def create_operation_summary(operations: List[AgentOperation]) -> Text:
    """Create a brief text summary of operations."""
    if not operations:
        return Text("No active operations", style="dim")

    active = sum(1 for op in operations if op.is_active)
    completed = sum(1 for op in operations if op.status == OperationStatus.COMPLETED)
    failed = sum(1 for op in operations if op.status == OperationStatus.FAILED)

    summary = f"Ops: {active} active, {completed} done, {failed} failed"

    if failed > 0:
        style = "red"
    elif active > 0:
        style = "blue"
    else:
        style = "green"

    return Text(summary, style=style)
