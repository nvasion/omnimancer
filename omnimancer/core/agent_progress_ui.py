"""
Agent Progress UI Components for Omnimancer.

This module provides Rich-based UI components for displaying agent mode
progress, operation status, and real-time updates.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .agent_mode_manager import (
    AgentMode,
    AgentModeManager,
    AgentOperation,
    AgentOperationStatus,
)


@dataclass
class OperationProgress:
    """Tracks progress of an individual operation."""

    operation_id: str
    task_id: Optional[TaskID] = None
    progress: float = 0.0
    status_text: str = "Queued"
    started_at: Optional[datetime] = None
    estimated_duration: Optional[float] = None


class AgentProgressUI:
    """
    Rich-based UI for displaying agent mode progress and status.

    Provides real-time updates of:
    - Agent mode status
    - Operation queue status
    - Individual operation progress
    - System metrics
    """

    def __init__(
        self,
        agent_manager: AgentModeManager,
        console: Optional[Console] = None,
    ):
        """
        Initialize the progress UI.

        Args:
            agent_manager: Agent mode manager instance
            console: Rich console instance (creates new if None)
        """
        self.agent_manager = agent_manager
        self.console = console or Console()

        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )

        # Operation tracking
        self.operation_progress: Dict[str, OperationProgress] = {}
        self.operation_task_ids: Dict[str, TaskID] = {}

        # UI state
        self.live_display: Optional[Live] = None
        self.update_interval = 0.5  # seconds
        self._update_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Register callbacks
        self.agent_manager.add_operation_callback(self._on_operation_change)
        self.agent_manager.add_mode_change_callback(self._on_mode_change)

    def start_monitoring(self):
        """Start real-time monitoring display."""
        if self._update_task and not self._update_task.done():
            return  # Already running

        self._stop_event.clear()
        self._update_task = asyncio.create_task(self._monitoring_loop())

    def stop_monitoring(self):
        """Stop real-time monitoring display."""
        self._stop_event.set()
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()

    @asynccontextmanager
    async def live_status_display(self):
        """Context manager for live status display."""
        layout = self._create_status_layout()

        with Live(layout, console=self.console, refresh_per_second=2) as live:
            self.live_display = live
            try:
                yield live
            finally:
                self.live_display = None

    def show_status_panel(self) -> Panel:
        """
        Create and return a status panel.

        Returns:
            Rich Panel with current status
        """
        status_info = self.agent_manager.get_status()

        # Mode status
        mode_color = {"off": "red", "on": "green", "paused": "yellow"}.get(
            status_info["mode"], "white"
        )

        mode_text = Text(f"Agent Mode: ", style="bold")
        mode_text.append(status_info["mode"].upper(), style=f"bold {mode_color}")

        # Operation counts
        ops = status_info["operations"]
        ops_text = f"""
Queued: {ops['queued']}
In Progress: {ops['in_progress']}
Completed: {ops['completed']}
Failed: {ops['failed']}
Requires Approval: {ops['requires_approval']}"""

        # Settings info
        settings = status_info["settings"]
        settings_text = f"""
Auto-approve Low Risk: {'Yes' if settings['auto_approve_low_risk'] else 'No'}
Auto-approve Read-only: {'Yes' if settings['auto_approve_read_only'] else 'No'}
Max Concurrent: {settings['max_concurrent_operations']}
Batch Approval: {'Enabled' if settings['enable_batch_approval'] else 'Disabled'}"""

        status_content = (
            f"{mode_text}\n\nOperations:{ops_text}\n\nSettings:{settings_text}"
        )

        return Panel(
            status_content,
            title="Agent Status",
            border_style="blue",
            padding=(1, 2),
        )

    def show_operations_table(self, limit: int = 10) -> Table:
        """
        Create and return an operations table.

        Args:
            limit: Maximum number of operations to show

        Returns:
            Rich Table with operation history
        """
        table = Table(
            title=f"Recent Operations (Last {limit})",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
        )

        table.add_column("ID", style="dim", width=12)
        table.add_column("Type", style="cyan", width=15)
        table.add_column("Description", width=30)
        table.add_column("Status", width=12)
        table.add_column("Duration", width=10)
        table.add_column("Retries", width=8)

        history = self.agent_manager.get_operation_history(limit)

        for op in history:
            # Format status with color
            status = op["status"]
            status_color = {
                "queued": "yellow",
                "in_progress": "blue",
                "completed": "green",
                "failed": "red",
                "cancelled": "red",
                "requires_approval": "magenta",
            }.get(status, "white")

            status_text = Text(status.replace("_", " ").title(), style=status_color)

            # Format duration
            duration = op.get("duration")
            if duration:
                if duration < 60:
                    duration_str = f"{duration:.1f}s"
                else:
                    duration_str = f"{duration/60:.1f}m"
            else:
                duration_str = "-"

            # Format retry count
            retry_count = op.get("retry_count", 0)
            retry_str = str(retry_count) if retry_count > 0 else "-"

            table.add_row(
                op["id"][-8:],  # Show last 8 chars of ID
                op["type"],
                (
                    op["description"][:28] + "..."
                    if len(op["description"]) > 28
                    else op["description"]
                ),
                status_text,
                duration_str,
                retry_str,
            )

        if not history:
            table.add_row("", "", "No operations yet", "", "", "")

        return table

    def show_progress_panel(self) -> Panel:
        """
        Create and return a progress panel for active operations.

        Returns:
            Rich Panel with progress information
        """
        active_ops = list(self.agent_manager.active_operations.values())

        if not active_ops:
            content = Text("No operations currently running", style="dim italic")
            return Panel(
                Align.center(content),
                title="Active Operations",
                border_style="blue",
                height=5,
            )

        # Create progress display
        progress_text = []
        for op in active_ops:
            # Calculate elapsed time
            elapsed = datetime.now() - op.started_at if op.started_at else timedelta(0)
            elapsed_str = f"{elapsed.total_seconds():.1f}s"

            # Get progress if available
            op_progress = self.operation_progress.get(op.id)
            if op_progress and op_progress.progress > 0:
                progress_str = f"[{int(op_progress.progress)}%]"
                status = op_progress.status_text
            else:
                progress_str = "[---]"
                status = op.status.value.replace("_", " ").title()

            line = f"• {op.operation.type.value}: {op.operation.description[:30]}"
            if len(op.operation.description) > 30:
                line += "..."
            line += f" {progress_str} ({elapsed_str}) - {status}"

            progress_text.append(line)

        content = "\n".join(progress_text)

        return Panel(
            content,
            title="Active Operations",
            border_style="green",
            padding=(1, 2),
        )

    def show_approval_queue_panel(self) -> Panel:
        """
        Create and return a panel showing operations requiring approval.

        Returns:
            Rich Panel with approval queue
        """
        approval_ops = [
            op
            for op in self.agent_manager.operation_queue
            if op.status == AgentOperationStatus.REQUIRES_APPROVAL
        ]

        if not approval_ops:
            content = Text("No operations waiting for approval", style="dim italic")
            return Panel(
                Align.center(content),
                title="Approval Queue",
                border_style="yellow",
                height=5,
            )

        approval_text = []
        for i, op in enumerate(approval_ops[:5], 1):  # Show up to 5
            risk = op.operation.data.get("risk_level", "medium")
            risk_color = {
                "low": "green",
                "medium": "yellow",
                "high": "red",
                "critical": "magenta",
            }.get(risk.lower(), "white")

            line = f"{i}. {op.operation.description[:40]}"
            if len(op.operation.description) > 40:
                line += "..."
            line += f" [Risk: {risk}]"

            # Create colored text
            text = Text(line)
            text.stylize(risk_color, len(line) - len(f"[Risk: {risk}]"), len(line))
            approval_text.append(text)

        if len(approval_ops) > 5:
            approval_text.append(
                Text(f"... and {len(approval_ops) - 5} more", style="dim")
            )

        content = "\n".join(str(text) for text in approval_text)

        return Panel(
            content,
            title=f"Approval Queue ({len(approval_ops)})",
            border_style="yellow",
            padding=(1, 2),
        )

    def show_agent_dashboard(self) -> Layout:
        """
        Create a comprehensive agent dashboard layout.

        Returns:
            Rich Layout with full dashboard
        """
        layout = Layout()

        # Split into sections
        layout.split(
            Layout(name="header", size=8),
            Layout(name="body"),
            Layout(name="footer", size=4),
        )

        # Header: Status and settings
        layout["header"].split_row(
            Layout(self.show_status_panel(), name="status"),
            Layout(self.show_approval_queue_panel(), name="approvals"),
        )

        # Body: Operations and progress
        layout["body"].split_row(
            Layout(self.show_operations_table(8), name="operations"),
            Layout(self.show_progress_panel(), name="progress"),
        )

        # Footer: Help text
        help_text = Text(
            "Commands: /agent on|off|status  •  Ctrl+C to exit dashboard  •  Use /agent off to disable agent mode",
            style="dim",
        )
        layout["footer"] = Panel(Align.center(help_text), style="dim")

        return layout

    def update_operation_progress(
        self, operation_id: str, progress: float, status_text: str = ""
    ):
        """
        Update progress for a specific operation.

        Args:
            operation_id: ID of the operation
            progress: Progress percentage (0-100)
            status_text: Current status description
        """
        if operation_id not in self.operation_progress:
            self.operation_progress[operation_id] = OperationProgress(
                operation_id=operation_id, started_at=datetime.now()
            )

        op_progress = self.operation_progress[operation_id]
        op_progress.progress = progress
        if status_text:
            op_progress.status_text = status_text

        # Update Rich progress bar if exists
        if operation_id in self.operation_task_ids:
            task_id = self.operation_task_ids[operation_id]
            self.progress.update(task_id, completed=progress, description=status_text)

    def _create_status_layout(self) -> Layout:
        """Create the main status layout for live display."""
        return self.show_agent_dashboard()

    async def _monitoring_loop(self):
        """Main monitoring loop for real-time updates."""
        try:
            while not self._stop_event.is_set():
                # Update live display if active
                if self.live_display:
                    self.live_display.update(self._create_status_layout())

                # Wait for next update
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.update_interval
                    )
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue monitoring

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.console.print(f"[red]Error in monitoring loop: {e}[/red]")

    def _on_operation_change(self, operation: AgentOperation):
        """Handle operation state changes."""
        # Update progress tracking
        if operation.status == AgentOperationStatus.IN_PROGRESS:
            # Start tracking progress
            if operation.id not in self.operation_progress:
                self.operation_progress[operation.id] = OperationProgress(
                    operation_id=operation.id,
                    started_at=operation.started_at,
                    status_text="Starting...",
                )

        elif operation.status in [
            AgentOperationStatus.COMPLETED,
            AgentOperationStatus.FAILED,
            AgentOperationStatus.CANCELLED,
        ]:
            # Update final progress
            if operation.id in self.operation_progress:
                op_progress = self.operation_progress[operation.id]
                op_progress.progress = (
                    100.0 if operation.status == AgentOperationStatus.COMPLETED else 0.0
                )
                op_progress.status_text = operation.status.value.replace(
                    "_", " "
                ).title()

            # Clean up progress tracking after delay
            asyncio.create_task(self._cleanup_operation_progress(operation.id))

    def _on_mode_change(self, old_mode: AgentMode, new_mode: AgentMode):
        """Handle agent mode changes."""
        if new_mode == AgentMode.ON:
            # Start monitoring if not already running
            if not self._update_task or self._update_task.done():
                self.start_monitoring()
        elif new_mode == AgentMode.OFF:
            # Stop monitoring
            self.stop_monitoring()

    async def _cleanup_operation_progress(self, operation_id: str, delay: float = 5.0):
        """Clean up operation progress tracking after delay."""
        await asyncio.sleep(delay)

        # Remove from tracking
        self.operation_progress.pop(operation_id, None)

        # Remove from Rich progress
        if operation_id in self.operation_task_ids:
            task_id = self.operation_task_ids.pop(operation_id)
            try:
                self.progress.remove_task(task_id)
            except KeyError:
                pass  # Task already removed


class AgentStatusIndicator:
    """Simple status indicator for embedding in other UI components."""

    def __init__(self, agent_manager: AgentModeManager):
        """
        Initialize status indicator.

        Args:
            agent_manager: Agent mode manager instance
        """
        self.agent_manager = agent_manager

    def get_status_badge(self) -> Text:
        """
        Get a small status badge for display.

        Returns:
            Rich Text object with status badge
        """
        status = self.agent_manager.get_status()
        mode = status["mode"]

        if mode == "on":
            active_count = status["operations"]["in_progress"]
            if active_count > 0:
                return Text(f"🤖 AGENT[{active_count}]", style="bold green")
            else:
                return Text("🤖 AGENT", style="bold green")
        elif mode == "paused":
            return Text("🤖 PAUSED", style="bold yellow")
        else:
            return Text("🤖 OFF", style="dim red")

    def get_mini_status(self) -> str:
        """
        Get minimal status string.

        Returns:
            Short status string
        """
        status = self.agent_manager.get_status()
        mode = status["mode"]

        if mode == "on":
            active = status["operations"]["in_progress"]
            queued = status["operations"]["queued"]
            return f"Agent: ON ({active} active, {queued} queued)"
        elif mode == "paused":
            return "Agent: PAUSED"
        else:
            return "Agent: OFF"
