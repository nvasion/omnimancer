"""
CLI Approval Dialog Formatter for Omnimancer.

This module provides terminal UI formatting for approval requests from
EnhancedApprovalManager, converting them into rich, readable terminal output
with proper formatting for action context, impact assessment, and risk scores.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..core.agent.approval_manager import (
    BatchApprovalRequest,
    ChangePreview,
    ChangeType,
)
from ..core.security.approval_workflow import (
    ApprovalRequest,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class CLIApprovalFormatter:
    """
    Terminal UI formatter for approval requests.

    Converts ApprovalRequest and ChangePreview objects into rich,
    readable terminal output with color coding, visual indicators,
    and structured data display.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the approval formatter.

        Args:
            console: Rich Console instance (creates new if None)
        """
        self.console = console or Console()

        # Risk level color mapping
        self.risk_colors = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "red",
            RiskLevel.CRITICAL: "bold red",
        }

        # Change type icons
        self.change_type_icons = {
            ChangeType.FILE_CREATE: "📄",
            ChangeType.FILE_MODIFY: "✏️ ",
            ChangeType.FILE_DELETE: "🗑️ ",
            ChangeType.DIRECTORY_CREATE: "📁",
            ChangeType.DIRECTORY_DELETE: "📂",
            ChangeType.COMMAND_EXECUTE: "⚡",
            ChangeType.WEB_REQUEST: "🌐",
            ChangeType.MCP_TOOL_CALL: "🔧",
        }

        # Risk level icons
        self.risk_icons = {
            RiskLevel.LOW: "✅",
            RiskLevel.MEDIUM: "⚠️ ",
            RiskLevel.HIGH: "⛔",
            RiskLevel.CRITICAL: "🚨",
        }

    def format_approval_dialog(
        self,
        approval_request: ApprovalRequest,
        preview: Optional[ChangePreview] = None,
        operation_data: Optional[Dict[str, Any]] = None,
    ) -> Group:
        """
        Format a complete approval dialog for terminal display.

        Args:
            approval_request: The approval request from workflow
            preview: Optional change preview with diff information
            operation_data: Additional operation data for context

        Returns:
            Rich Group with formatted approval dialog components
        """
        components = []

        # Header with operation title and risk level
        components.append(self._format_header(approval_request, preview))

        # Separator
        components.append(Rule(style="dim"))

        # Operation details
        components.append(
            self._format_operation_details(approval_request, operation_data)
        )

        # Change preview and diff if available
        if preview:
            components.append(self._format_change_preview(preview))

            # Show diff if available
            if preview.diff:
                components.append(self._format_diff_display(preview))

        # Risk assessment
        components.append(self._format_risk_assessment(approval_request))

        # Impact assessment
        if preview and preview.metadata:
            components.append(self._format_impact_assessment(preview))

        # Metadata information
        if approval_request.metadata:
            components.append(self._format_metadata(approval_request.metadata))

        # Time information
        components.append(self._format_time_info(approval_request))

        # Separator before options
        components.append(Rule(style="dim"))

        # Available options
        components.append(self._format_approval_options())

        return Group(*components)

    def format_batch_approval_dialog(
        self, batch_request: BatchApprovalRequest
    ) -> Group:
        """
        Format a batch approval dialog for multiple operations.

        Args:
            batch_request: BatchApprovalRequest containing multiple operations

        Returns:
            Rich Group with formatted batch approval dialog
        """
        components = []

        # Batch header
        components.append(self._format_batch_header(batch_request))
        components.append(Rule(style="dim"))

        # Summary table
        components.append(self._format_batch_summary(batch_request))

        # Individual operations
        components.append(Text("Operations:", style="bold"))
        for i, (operation, preview) in enumerate(
            zip(batch_request.operations, batch_request.previews)
        ):
            # Create a mini approval request for consistent formatting
            mini_request = ApprovalRequest(
                operation_type=operation.type.value,
                description=operation.description,
                metadata=operation.data,
            )

            operation_panel = Panel(
                self._format_single_batch_operation(mini_request, preview, i + 1),
                title=f"Operation {i + 1}",
                border_style="dim",
            )
            components.append(operation_panel)

        # Batch approval options
        components.append(Rule(style="dim"))
        components.append(
            self._format_batch_approval_options(len(batch_request.operations))
        )

        return Group(*components)

    def _format_header(
        self,
        approval_request: ApprovalRequest,
        preview: Optional[ChangePreview],
    ) -> Panel:
        """Format the approval dialog header."""
        # Get icon for change type
        icon = ""
        if preview:
            icon = self.change_type_icons.get(preview.change_type, "❓")

        # Risk level with color and icon
        risk_icon = self.risk_icons.get(approval_request.risk_level, "❓")
        risk_color = self.risk_colors.get(approval_request.risk_level, "white")

        title = f"{icon} {approval_request.operation_type.replace('_', ' ').title()}"
        risk_text = Text(
            f"{risk_icon} Risk Level: {approval_request.risk_level.value.upper()}",
            style=risk_color,
        )

        header_content = Group(
            Text(title, style="bold white"),
            Text(approval_request.description, style="dim white"),
            Text(),
            risk_text,
        )

        return Panel(
            header_content,
            title="🔐 Approval Required",
            title_align="left",
            border_style=risk_color,
        )

    def _format_operation_details(
        self,
        approval_request: ApprovalRequest,
        operation_data: Optional[Dict[str, Any]],
    ) -> Panel:
        """Format operation details table."""
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column("Field", style="cyan", min_width=15)
        table.add_column("Value", style="white")

        # Basic operation info
        table.add_row("Operation Type", approval_request.operation_type)
        table.add_row("Requested By", approval_request.requested_by)
        table.add_row(
            "Request Time", approval_request.requested_at.strftime("%H:%M:%S")
        )

        # Operation-specific details from metadata or operation data
        data_source = operation_data or approval_request.metadata
        if data_source:
            # Common operation details
            if "path" in data_source:
                table.add_row("Target Path", str(data_source["path"]))

            if "command" in data_source:
                table.add_row("Command", str(data_source["command"]))

            if "url" in data_source:
                table.add_row("URL", str(data_source["url"]))

            if "content_length" in data_source:
                size_kb = int(data_source["content_length"]) / 1024
                table.add_row("Content Size", f"{size_kb:.1f} KB")

        return Panel(
            table,
            title="Operation Details",
            title_align="left",
            border_style="blue",
        )

    def _format_change_preview(self, preview: ChangePreview) -> Panel:
        """Format change preview information."""
        preview_content = []

        # Change description
        preview_content.append(Text(preview.description, style="white"))
        preview_content.append(Text())

        # Key attributes
        attrs_table = Table(show_header=False, box=None, pad_edge=False)
        attrs_table.add_column("Attribute", style="cyan", min_width=12)
        attrs_table.add_column("Value", style="white")

        attrs_table.add_row("Reversible", "✅ Yes" if preview.reversible else "❌ No")

        if preview.risk_assessment:
            attrs_table.add_row("Risk", preview.risk_assessment)

        # File-specific information
        if preview.change_type in [
            ChangeType.FILE_CREATE,
            ChangeType.FILE_MODIFY,
        ]:
            if preview.proposed_state:
                lines = len(preview.proposed_state.splitlines())
                attrs_table.add_row("New Lines", str(lines))

        if preview.change_type == ChangeType.FILE_DELETE:
            if preview.current_state:
                lines = len(preview.current_state.splitlines())
                attrs_table.add_row("Lines Lost", str(lines))

        # Directory-specific information
        if preview.change_type == ChangeType.DIRECTORY_DELETE:
            item_count = preview.metadata.get("item_count", 0)
            attrs_table.add_row("Items Inside", str(item_count))

        preview_content.append(attrs_table)

        return Panel(
            Group(*preview_content),
            title=f"{self.change_type_icons.get(preview.change_type, '❓')} Change Preview",
            title_align="left",
            border_style="yellow",
        )

    def _format_diff_display(self, preview: ChangePreview) -> Panel:
        """Format diff visualization with syntax highlighting."""
        if not preview.diff:
            return Panel(
                Text("No diff available", style="dim"),
                title="📋 Changes",
                border_style="dim",
            )

        # Try to detect file type for syntax highlighting
        if "path" in preview.metadata:
            path = Path(preview.metadata["path"])
            path.suffix.lstrip(".")

        # Limit diff length to prevent terminal overflow
        diff_text = preview.diff
        if len(diff_text) > 2000:
            lines = diff_text.splitlines()
            if len(lines) > 50:
                truncated_lines = (
                    lines[:25]
                    + ["...", "# Diff truncated - too long to display", "..."]
                    + lines[-25:]
                )
                diff_text = "\n".join(truncated_lines)

        try:
            # Use syntax highlighting for diff
            syntax = Syntax(
                diff_text,
                "diff",
                theme="monokai",
                line_numbers=False,
                word_wrap=False,
            )
        except Exception:
            # Fallback to plain text if syntax highlighting fails
            syntax = Text(diff_text, style="white")

        return Panel(
            syntax,
            title="📋 Changes (Diff)",
            title_align="left",
            border_style="magenta",
        )

    def _format_risk_assessment(self, approval_request: ApprovalRequest) -> Panel:
        """Format risk assessment with visual indicators."""
        risk_level = approval_request.risk_level
        risk_color = self.risk_colors[risk_level]
        risk_icon = self.risk_icons[risk_level]

        # Risk level bar
        risk_value = {
            RiskLevel.LOW: 25,
            RiskLevel.MEDIUM: 50,
            RiskLevel.HIGH: 75,
            RiskLevel.CRITICAL: 100,
        }[risk_level]

        # Create visual risk meter
        progress = Progress(
            TextColumn("[bold blue]Risk Level"),
            BarColumn(bar_width=20),
            TextColumn(f"{risk_icon} {risk_level.value.upper()}"),
            expand=False,
        )

        progress.add_task("risk", total=100, completed=risk_value)

        # Risk description
        risk_descriptions = {
            RiskLevel.LOW: "Low risk operation. Minimal security impact expected.",
            RiskLevel.MEDIUM: "Medium risk operation. Review recommended before approval.",
            RiskLevel.HIGH: "High risk operation. Careful consideration required.",
            RiskLevel.CRITICAL: "Critical risk operation. High security impact possible.",
        }

        description = Text(risk_descriptions[risk_level], style=risk_color)

        risk_content = Group(progress, Text(), description)

        return Panel(
            risk_content,
            title=f"{risk_icon} Risk Assessment",
            title_align="left",
            border_style=risk_color,
        )

    def _format_impact_assessment(self, preview: ChangePreview) -> Panel:
        """Format impact assessment from preview metadata."""
        impact_items = []

        # File system impact
        if preview.change_type in [
            ChangeType.FILE_CREATE,
            ChangeType.FILE_MODIFY,
            ChangeType.FILE_DELETE,
        ]:
            if "path" in preview.metadata:
                path = preview.metadata["path"]
                impact_items.append(f"• File system: {path}")

        if preview.change_type in [
            ChangeType.DIRECTORY_CREATE,
            ChangeType.DIRECTORY_DELETE,
        ]:
            if "path" in preview.metadata:
                path = preview.metadata["path"]
                item_count = preview.metadata.get("item_count", 0)
                impact_items.append(f"• Directory: {path} ({item_count} items)")

        # Command execution impact
        if preview.change_type == ChangeType.COMMAND_EXECUTE:
            if "command" in preview.metadata:
                command = preview.metadata["command"]
                impact_items.append(f"• System command: {command}")

        # Network impact
        if preview.change_type == ChangeType.WEB_REQUEST:
            if "url" in preview.metadata:
                url = preview.metadata["url"]
                method = preview.metadata.get("method", "GET")
                impact_items.append(f"• Network {method}: {url}")

        # Reversibility impact
        reversible_text = "✅ Reversible" if preview.reversible else "❌ Not reversible"
        impact_items.append(f"• Recovery: {reversible_text}")

        if not impact_items:
            impact_items.append("• No specific impacts identified")

        impact_text = Text("\n".join(impact_items), style="white")

        return Panel(
            impact_text,
            title="🎯 Impact Assessment",
            title_align="left",
            border_style="cyan",
        )

    def _format_metadata(self, metadata: Dict[str, Any]) -> Panel:
        """Format metadata information in a readable way."""
        # Filter out large or sensitive metadata
        filtered_metadata = {}
        for key, value in metadata.items():
            if key in ["preview", "operation_data"]:
                continue  # Skip large nested objects

            # Limit string length
            if isinstance(value, str) and len(value) > 100:
                filtered_metadata[key] = value[:100] + "..."
            else:
                filtered_metadata[key] = value

        if not filtered_metadata:
            return Panel(
                Text("No additional metadata", style="dim"),
                title="📊 Additional Information",
                border_style="dim",
            )

        try:
            # Pretty print metadata as JSON
            metadata_json = json.dumps(filtered_metadata, indent=2, default=str)
            syntax = Syntax(metadata_json, "json", theme="monokai", line_numbers=False)
        except Exception:
            # Fallback to simple text display
            lines = [f"{k}: {v}" for k, v in filtered_metadata.items()]
            syntax = Text("\n".join(lines), style="white")

        return Panel(
            syntax,
            title="📊 Additional Information",
            title_align="left",
            border_style="dim",
        )

    def _format_time_info(self, approval_request: ApprovalRequest) -> Panel:
        """Format timing information."""
        time_content = []

        # Request time
        request_time = approval_request.requested_at.strftime("%Y-%m-%d %H:%M:%S")
        time_content.append(Text(f"Requested: {request_time}", style="white"))

        # Expiration time
        if approval_request.expires_at:
            expires_time = approval_request.expires_at.strftime("%Y-%m-%d %H:%M:%S")
            time_remaining = approval_request.time_remaining()

            if time_remaining and time_remaining.total_seconds() > 0:
                minutes_left = int(time_remaining.total_seconds() / 60)
                time_content.append(
                    Text(
                        f"Expires: {expires_time} ({minutes_left}m remaining)",
                        style="yellow",
                    )
                )
            else:
                time_content.append(
                    Text(f"Expires: {expires_time} (EXPIRED)", style="red")
                )

        return Panel(
            Group(*time_content),
            title="⏰ Timing",
            title_align="left",
            border_style="blue",
        )

    def _format_approval_options(self) -> Panel:
        """Format available approval options."""
        options_text = [
            Text("Available Options:", style="bold white"),
            Text(),
            Text(
                "• [bold green]y[/bold green] or [bold green]yes[/bold green] - Approve this operation",
                style="green",
            ),
            Text(
                "• [bold blue]r[/bold blue] or [bold blue]remember[/bold blue] - Approve and remember for similar operations",
                style="blue",
            ),
            Text(
                "• [bold red]n[/bold red] or [bold red]no[/bold red] - Deny this operation",
                style="red",
            ),
            Text(
                "• [bold yellow]q[/bold yellow] or [bold yellow]quit[/bold yellow] - Cancel and exit",
                style="yellow",
            ),
            Text(),
            Text("Press Ctrl+C to cancel at any time.", style="dim"),
        ]

        return Panel(
            Group(*options_text),
            title="⌨️  Options",
            title_align="left",
            border_style="white",
        )

    def _format_batch_header(self, batch_request: BatchApprovalRequest) -> Panel:
        """Format batch approval header."""
        summary = batch_request.get_approval_summary()

        header_content = Group(
            Text("Batch Operation Approval", style="bold white"),
            Text(
                f"Multiple operations require approval ({summary['total_operations']} operations)",
                style="dim white",
            ),
            Text(),
            Text(f"🔢 Operations: {summary['total_operations']}", style="blue"),
            Text(
                f"⏰ Created: {batch_request.created_at.strftime('%H:%M:%S')}",
                style="white",
            ),
        )

        return Panel(
            header_content,
            title="🔐 Batch Approval Required",
            title_align="left",
            border_style="blue",
        )

    def _format_batch_summary(self, batch_request: BatchApprovalRequest) -> Panel:
        """Format batch operations summary table."""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Op", width=3)
        table.add_column("Type", width=15)
        table.add_column("Description", width=40)
        table.add_column("Risk", width=8)

        for i, (operation, preview) in enumerate(
            zip(batch_request.operations, batch_request.previews)
        ):
            # Determine risk level from preview or use medium as default
            risk_level = RiskLevel.MEDIUM
            if preview.risk_assessment:
                if "high" in preview.risk_assessment.lower():
                    risk_level = RiskLevel.HIGH
                elif "low" in preview.risk_assessment.lower():
                    risk_level = RiskLevel.LOW
                elif "critical" in preview.risk_assessment.lower():
                    risk_level = RiskLevel.CRITICAL

            risk_color = self.risk_colors[risk_level]
            risk_icon = self.risk_icons[risk_level]

            # Truncate description if too long
            desc = operation.description
            if len(desc) > 37:
                desc = desc[:37] + "..."

            table.add_row(
                str(i + 1),
                operation.type.value.replace("_", " ").title(),
                desc,
                Text(f"{risk_icon} {risk_level.value}", style=risk_color),
            )

        return Panel(
            table,
            title="📋 Operations Summary",
            title_align="left",
            border_style="cyan",
        )

    def _format_single_batch_operation(
        self,
        approval_request: ApprovalRequest,
        preview: ChangePreview,
        operation_number: int,
    ) -> Group:
        """Format a single operation within a batch."""
        components = []

        # Operation header
        icon = self.change_type_icons.get(preview.change_type, "❓")
        components.append(
            Text(f"{icon} {approval_request.description}", style="bold white")
        )

        # Key details in compact format
        details_text = []
        if "path" in approval_request.metadata:
            details_text.append(f"Path: {approval_request.metadata['path']}")

        if preview.risk_assessment:
            details_text.append(f"Risk: {preview.risk_assessment}")

        if details_text:
            components.append(Text(" | ".join(details_text), style="dim white"))

        return Group(*components)

    def _format_batch_approval_options(self, operation_count: int) -> Panel:
        """Format batch approval options."""
        options_text = [
            Text("Batch Approval Options:", style="bold white"),
            Text(),
            Text(
                "• [bold green]all[/bold green] - Approve all operations",
                style="green",
            ),
            Text(
                "• [bold red]none[/bold red] - Deny all operations",
                style="red",
            ),
            Text(
                f"• [bold blue]select[/bold blue] - Choose specific operations (1-{operation_count})",
                style="blue",
            ),
            Text(
                "• [bold yellow]individual[/bold yellow] - Review each operation separately",
                style="yellow",
            ),
            Text(
                "• [bold yellow]q[/bold yellow] or [bold yellow]quit[/bold yellow] - Cancel batch",
                style="yellow",
            ),
            Text(),
            Text("Press Ctrl+C to cancel at any time.", style="dim"),
        ]

        return Panel(
            Group(*options_text),
            title="⌨️  Batch Options",
            title_align="left",
            border_style="white",
        )

    def format_approval_denied(
        self, approval_request: ApprovalRequest, reason: str = ""
    ) -> Panel:
        """Format approval denied message."""
        content = [
            Text("❌ Operation Denied", style="bold red"),
            Text(),
            Text(f"Operation: {approval_request.description}", style="white"),
            Text(f"Type: {approval_request.operation_type}", style="dim"),
        ]

        if reason:
            content.extend([Text(), Text(f"Reason: {reason}", style="yellow")])

        return Panel(
            Group(*content),
            title="🚫 Approval Denied",
            title_align="left",
            border_style="red",
        )

    def format_approval_approved(
        self, approval_request: ApprovalRequest, remember: bool = False
    ) -> Panel:
        """Format approval approved message."""
        content = [
            Text("✅ Operation Approved", style="bold green"),
            Text(),
            Text(f"Operation: {approval_request.description}", style="white"),
            Text(f"Type: {approval_request.operation_type}", style="dim"),
        ]

        if remember:
            content.extend(
                [
                    Text(),
                    Text(
                        "🧠 Similar operations will be auto-approved in the future",
                        style="blue",
                    ),
                ]
            )

        return Panel(
            Group(*content),
            title="✅ Approval Granted",
            title_align="left",
            border_style="green",
        )
