"""
Batch Approval Display Components for Omnimancer CLI.

This module provides Rich-based UI components for displaying batch approval
information including action lists, summaries, and risk assessments.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from ..core.agent.approval_manager import BatchApprovalRequest, ChangePreview
from ..core.agent.types import Operation, OperationType
from ..core.security.approval_workflow import ApprovalStatus

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display modes for batch approval interface."""

    COMPACT = "compact"
    DETAILED = "detailed"
    SUMMARY_ONLY = "summary"


@dataclass
class BatchDisplayConfig:
    """Configuration for batch approval display."""

    mode: DisplayMode = DisplayMode.DETAILED
    show_risk_colors: bool = True
    show_line_numbers: bool = True
    max_preview_length: int = 100
    items_per_page: int = 10
    show_timestamps: bool = True
    compact_operation_names: bool = True


class BatchApprovalPanel:
    """
    Rich-based UI components for displaying batch approval information.

    Provides structured display of batch approval requests including
    action lists, summaries, risk assessments, and interactive elements.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        config: Optional[BatchDisplayConfig] = None,
    ):
        """
        Initialize batch approval panel.

        Args:
            console: Rich console for output
            config: Display configuration settings
        """
        self.console = console or Console()
        self.config = config or BatchDisplayConfig()

    def render_batch_overview(self, batch_request: BatchApprovalRequest) -> Panel:
        """
        Render overview panel for batch approval request.

        Args:
            batch_request: Batch request to display

        Returns:
            Rich Panel containing batch overview
        """
        # Calculate batch statistics
        total_operations = len(batch_request.operations)
        approved_count = len(batch_request.approved_operations)
        pending_count = total_operations - approved_count

        # Risk level distribution
        risk_counts = self._calculate_risk_distribution(batch_request.previews)

        # Create overview table
        overview_table = Table.grid(padding=1)
        overview_table.add_column("Label", style="bold cyan")
        overview_table.add_column("Value", style="bright_white")

        # Basic info
        overview_table.add_row("Batch ID:", batch_request.id[:8] + "...")
        overview_table.add_row(
            "Created:", batch_request.created_at.strftime("%Y-%m-%d %H:%M:%S")
        )
        overview_table.add_row("Status:", self._format_status(batch_request.status))

        if batch_request.expires_at:
            time_remaining = batch_request.expires_at - datetime.now()
            if time_remaining.total_seconds() > 0:
                overview_table.add_row(
                    "Expires:", f"{time_remaining.seconds // 60} minutes"
                )
            else:
                overview_table.add_row("Expires:", "[red]EXPIRED[/red]")

        # Operation counts
        overview_table.add_row("Total Operations:", str(total_operations))
        if approved_count > 0:
            overview_table.add_row("Approved:", f"[green]{approved_count}[/green]")
        if pending_count > 0:
            overview_table.add_row("Pending:", f"[yellow]{pending_count}[/yellow]")

        # Risk distribution
        if risk_counts:
            overview_table.add_row("", "")  # Separator
            overview_table.add_row("Risk Distribution:", "")
            for risk_level, count in risk_counts.items():
                color = self._get_risk_color(risk_level)
                overview_table.add_row(
                    f"  {risk_level.title()}:", f"[{color}]{count}[/{color}]"
                )

        return Panel(
            overview_table,
            title="[bold]Batch Approval Overview[/bold]",
            title_align="left",
            border_style="blue",
        )

    def render_operations_table(
        self,
        batch_request: BatchApprovalRequest,
        page: int = 0,
        show_approved: bool = True,
    ) -> Panel:
        """
        Render table of operations in the batch.

        Args:
            batch_request: Batch request to display
            page: Page number for pagination
            show_approved: Whether to show already approved operations

        Returns:
            Rich Panel containing operations table
        """
        operations = batch_request.operations
        previews = batch_request.previews

        # Filter operations if needed
        if not show_approved:
            filtered_ops = []
            filtered_previews = []
            for i, (op, preview) in enumerate(zip(operations, previews)):
                if i not in batch_request.approved_operations:
                    filtered_ops.append(op)
                    filtered_previews.append(preview)
            operations = filtered_ops
            previews = filtered_previews

        # Pagination
        start_idx = page * self.config.items_per_page
        end_idx = min(start_idx + self.config.items_per_page, len(operations))

        if start_idx >= len(operations):
            return Panel(
                "[yellow]No operations to display on this page.[/yellow]",
                title="Operations",
                border_style="yellow",
            )

        # Create operations table
        table = Table(show_header=True, header_style="bold magenta")

        if self.config.show_line_numbers:
            table.add_column("#", style="dim", width=4)

        table.add_column("Type", style="cyan", width=15)
        table.add_column("Target", style="bright_white", width=30)
        table.add_column("Risk", width=8, justify="center")
        table.add_column("Status", width=10, justify="center")

        if self.config.mode == DisplayMode.DETAILED:
            table.add_column("Description", style="dim", width=40)

        # Add rows
        for i in range(start_idx, end_idx):
            operation = operations[i]
            preview = previews[i] if i < len(previews) else None

            # Determine status
            original_idx = (
                i if show_approved else self._get_original_index(batch_request, i)
            )
            is_approved = original_idx in batch_request.approved_operations
            status = (
                "[green]✓ Approved[/green]"
                if is_approved
                else "[yellow]○ Pending[/yellow]"
            )

            # Format operation type
            op_type = self._format_operation_type(operation.type)

            # Format target
            target = self._format_target(operation)

            # Risk assessment
            risk_display = (
                self._format_risk_assessment(preview) if preview else "Unknown"
            )

            # Build row
            row = []
            if self.config.show_line_numbers:
                row.append(str(i + 1))

            row.extend([op_type, target, risk_display, status])

            if self.config.mode == DisplayMode.DETAILED:
                description = operation.description or "No description"
                if len(description) > self.config.max_preview_length:
                    description = description[: self.config.max_preview_length] + "..."
                row.append(description)

            table.add_row(*row)

        # Panel title with pagination info
        total_ops = len(operations)
        page_info = (
            f"Page {page + 1} of {(total_ops - 1) // self.config.items_per_page + 1}"
            if total_ops > self.config.items_per_page
            else ""
        )
        title = f"[bold]Operations ({start_idx + 1}-{end_idx} of {total_ops})[/bold]"
        if page_info:
            title += f" - {page_info}"

        return Panel(
            table,
            title=title,
            title_align="left",
            border_style="green" if show_approved else "yellow",
        )

    def render_action_details(
        self, operation: Operation, preview: Optional[ChangePreview] = None
    ) -> Panel:
        """
        Render detailed view of a single operation.

        Args:
            operation: Operation to display
            preview: Optional change preview

        Returns:
            Rich Panel with operation details
        """
        details_group = []

        # Operation information
        info_table = Table.grid(padding=1)
        info_table.add_column("Label", style="bold cyan")
        info_table.add_column("Value", style="bright_white")

        info_table.add_row("Type:", self._format_operation_type(operation.type))
        info_table.add_row("Description:", operation.description or "No description")
        info_table.add_row(
            "Requires Approval:",
            "Yes" if operation.requires_approval else "No",
        )
        info_table.add_row("Reversible:", "Yes" if operation.reversible else "No")

        if operation.data:
            info_table.add_row("", "")  # Separator
            info_table.add_row("Operation Data:", "")
            for key, value in operation.data.items():
                display_value = str(value)
                if len(display_value) > 100:
                    display_value = display_value[:100] + "..."
                info_table.add_row(f"  {key}:", display_value)

        details_group.append(
            Panel(info_table, title="Operation Details", border_style="blue")
        )

        # Preview information
        if preview:
            preview_table = Table.grid(padding=1)
            preview_table.add_column("Label", style="bold cyan")
            preview_table.add_column("Value", style="bright_white")

            preview_table.add_row(
                "Change Type:",
                preview.change_type.value.replace("_", " ").title(),
            )
            preview_table.add_row(
                "Risk Assessment:", self._format_risk_assessment(preview)
            )
            preview_table.add_row("Reversible:", "Yes" if preview.reversible else "No")

            if preview.metadata:
                preview_table.add_row("", "")  # Separator
                preview_table.add_row("Metadata:", "")
                for key, value in preview.metadata.items():
                    preview_table.add_row(f"  {key}:", str(value))

            details_group.append(
                Panel(
                    preview_table,
                    title="Change Preview",
                    border_style="yellow",
                )
            )

            # Show diff if available
            if preview.diff:
                diff_text = preview.diff
                if len(diff_text) > 500:  # Limit diff display
                    diff_text = diff_text[:500] + "\n\n... (diff truncated)"

                details_group.append(
                    Panel(
                        Text(diff_text, style="dim"),
                        title="Changes",
                        border_style="magenta",
                    )
                )

        return Panel(
            Group(*details_group),
            title=f"[bold]Action Details - {operation.type.value}[/bold]",
            title_align="left",
            border_style="white",
        )

    def render_batch_summary(self, batch_request: BatchApprovalRequest) -> Panel:
        """
        Render summary of batch operations and risk assessment.

        Args:
            batch_request: Batch request to summarize

        Returns:
            Rich Panel with batch summary
        """
        summary_parts = []

        # Operation type breakdown
        type_counts = {}
        for operation in batch_request.operations:
            type_name = operation.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        type_table = Table(show_header=True, header_style="bold")
        type_table.add_column("Operation Type", style="cyan")
        type_table.add_column("Count", style="green", justify="right")

        for op_type, count in sorted(type_counts.items()):
            formatted_type = op_type.replace("_", " ").title()
            type_table.add_row(formatted_type, str(count))

        summary_parts.append(
            Panel(type_table, title="Operation Types", border_style="cyan")
        )

        # Risk assessment summary
        risk_counts = self._calculate_risk_distribution(batch_request.previews)
        overall_risk = self._assess_overall_risk(risk_counts)

        risk_table = Table(show_header=True, header_style="bold")
        risk_table.add_column("Risk Level", style="yellow")
        risk_table.add_column("Count", style="green", justify="right")
        risk_table.add_column("Percentage", style="blue", justify="right")

        total_with_risk = sum(risk_counts.values())
        for risk_level, count in risk_counts.items():
            if count > 0:
                percentage = (
                    (count / total_with_risk) * 100 if total_with_risk > 0 else 0
                )
                color = self._get_risk_color(risk_level)
                risk_table.add_row(
                    f"[{color}]{risk_level.title()}[/{color}]",
                    str(count),
                    f"{percentage:.1f}%",
                )

        risk_panel = Panel(
            risk_table,
            title=f"Risk Assessment - Overall: {overall_risk}",
            border_style="red",
        )
        summary_parts.append(risk_panel)

        return Panel(
            Group(*summary_parts),
            title="[bold]Batch Summary[/bold]",
            title_align="left",
            border_style="blue",
        )

    def render_progress_display(self, batch_request: BatchApprovalRequest) -> Panel:
        """
        Render progress display for batch processing.

        Args:
            batch_request: Batch request to show progress for

        Returns:
            Rich Panel with progress information
        """
        total_operations = len(batch_request.operations)
        approved_count = len(batch_request.approved_operations)

        # Create progress bar
        progress = Progress(
            TextColumn("[bold blue]Processing batch...", justify="right"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total} operations"),
            TimeElapsedColumn(),
        )

        progress.add_task(
            "batch_progress", total=total_operations, completed=approved_count
        )

        return Panel(
            progress,
            title=f"[bold]Batch Progress[/bold]",
            title_align="left",
            border_style="green",
        )

    # Helper methods

    def _calculate_risk_distribution(
        self, previews: List[ChangePreview]
    ) -> Dict[str, int]:
        """Calculate distribution of risk levels in previews."""
        risk_counts = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
            "unknown": 0,
        }

        for preview in previews:
            if preview.risk_assessment:
                risk_level = self._extract_risk_level(preview.risk_assessment)
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            else:
                risk_counts["unknown"] += 1

        return risk_counts

    def _extract_risk_level(self, risk_assessment: str) -> str:
        """Extract risk level from risk assessment string."""
        risk_lower = risk_assessment.lower()
        if "critical" in risk_lower:
            return "critical"
        elif "high" in risk_lower:
            return "high"
        elif "medium" in risk_lower:
            return "medium"
        elif "low" in risk_lower:
            return "low"
        else:
            return "unknown"

    def _assess_overall_risk(self, risk_counts: Dict[str, int]) -> str:
        """Assess overall risk level based on distribution."""
        if risk_counts.get("critical", 0) > 0:
            return "[red]CRITICAL[/red]"
        elif risk_counts.get("high", 0) > 0:
            return "[red]HIGH[/red]"
        elif risk_counts.get("medium", 0) > 0:
            return "[yellow]MEDIUM[/yellow]"
        elif risk_counts.get("low", 0) > 0:
            return "[green]LOW[/green]"
        else:
            return "[dim]UNKNOWN[/dim]"

    def _get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level."""
        color_map = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "bright_red",
            "unknown": "dim",
        }
        return color_map.get(risk_level.lower(), "white")

    def _format_status(self, status: ApprovalStatus) -> str:
        """Format approval status with colors."""
        status_colors = {
            ApprovalStatus.PENDING: "yellow",
            ApprovalStatus.APPROVED: "green",
            ApprovalStatus.DENIED: "red",
            ApprovalStatus.EXPIRED: "dim",
        }
        color = status_colors.get(status, "white")
        return f"[{color}]{status.value.title()}[/{color}]"

    def _format_operation_type(self, op_type: OperationType) -> str:
        """Format operation type for display."""
        if self.config.compact_operation_names:
            # Use shorter names
            type_map = {
                OperationType.FILE_READ: "Read",
                OperationType.FILE_WRITE: "Write",
                OperationType.FILE_DELETE: "Delete",
                OperationType.DIRECTORY_CREATE: "MkDir",
                OperationType.DIRECTORY_DELETE: "RmDir",
                OperationType.COMMAND_EXECUTE: "Exec",
                OperationType.WEB_REQUEST: "Web",
                OperationType.MCP_TOOL_CALL: "MCP",
            }
            return type_map.get(op_type, op_type.value)
        else:
            return op_type.value.replace("_", " ").title()

    def _format_target(self, operation: Operation) -> str:
        """Format operation target for display."""
        if operation.data:
            # Extract meaningful target from operation data
            if "path" in operation.data:
                path = operation.data["path"]
                if len(path) > 25:
                    return f"...{path[-25:]}"
                return path
            elif "url" in operation.data:
                url = operation.data["url"]
                if len(url) > 25:
                    return f"...{url[-25:]}"
                return url
            elif "command" in operation.data:
                cmd = operation.data["command"]
                if len(cmd) > 25:
                    return f"...{cmd[-25:]}"
                return cmd

        return (
            operation.description[:25] + "..."
            if len(operation.description or "") > 25
            else (operation.description or "No target")
        )

    def _format_risk_assessment(self, preview: Optional[ChangePreview]) -> str:
        """Format risk assessment for display."""
        if not preview or not preview.risk_assessment:
            return "[dim]Unknown[/dim]"

        risk_level = self._extract_risk_level(preview.risk_assessment)
        color = self._get_risk_color(risk_level)
        return f"[{color}]{risk_level.title()}[/{color}]"

    def _get_original_index(
        self, batch_request: BatchApprovalRequest, filtered_index: int
    ) -> int:
        """Get original index when operations are filtered."""
        # This would need more sophisticated logic if we implement complex filtering
        # For now, assume simple filtering where we just skip approved items
        current_filtered = 0
        for i, operation in enumerate(batch_request.operations):
            if i not in batch_request.approved_operations:
                if current_filtered == filtered_index:
                    return i
                current_filtered += 1
        return filtered_index


def create_batch_approval_panel(
    console: Optional[Console] = None,
    config: Optional[BatchDisplayConfig] = None,
) -> BatchApprovalPanel:
    """
    Create a batch approval panel with default configuration.

    Args:
        console: Rich console for output
        config: Display configuration settings

    Returns:
        Configured BatchApprovalPanel instance
    """
    return BatchApprovalPanel(console=console, config=config)


def create_compact_batch_panel(
    console: Optional[Console] = None,
) -> BatchApprovalPanel:
    """
    Create a batch approval panel optimized for compact display.

    Args:
        console: Rich console for output

    Returns:
        BatchApprovalPanel configured for compact mode
    """
    config = BatchDisplayConfig(
        mode=DisplayMode.COMPACT,
        show_line_numbers=False,
        max_preview_length=50,
        items_per_page=20,
        compact_operation_names=True,
    )
    return BatchApprovalPanel(console=console, config=config)
