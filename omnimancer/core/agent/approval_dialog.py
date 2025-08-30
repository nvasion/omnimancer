"""
Structured Approval Dialog Layout for Omnimancer Agent Operations.

This module provides the main approval dialog interface that combines
rich text rendering, diff display, and interactive controls for
operation approval workflows.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Union

from rich.align import Align
from rich.box import MINIMAL, ROUNDED, SIMPLE
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .approval_context import (
    ApprovalContext,
    ApprovalDecision,
    OperationDetails,
)
from .diff_renderer import DiffType, EnhancedDiffRenderer, create_diff_renderer
from .input_handler import (
    InputState,
    InteractiveInputHandler,
    KeyAction,
    create_input_handler,
)
from .rich_renderer import (
    RichTextRenderer,
    RiskLevel,
    create_renderer,
)

logger = logging.getLogger(__name__)


class DialogState(Enum):
    """States of the approval dialog."""

    INITIALIZING = "initializing"
    DISPLAYING = "displaying"
    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DialogSection(Enum):
    """Sections of the approval dialog."""

    HEADER = "header"
    OPERATION = "operation"
    DIFF = "diff"
    RISK = "risk"
    CONTROLS = "controls"
    STATUS = "status"


@dataclass
class DialogOptions:
    """Configuration options for the approval dialog."""

    # Display options
    show_diff: bool = True
    show_risk_assessment: bool = True
    show_file_tree: bool = True
    show_operation_details: bool = True

    # Layout options
    max_height: int = 40
    max_width: int = 120
    compact_mode: bool = False

    # Interactive options
    auto_scroll: bool = True
    enable_shortcuts: bool = True
    timeout_seconds: Optional[int] = None

    # Rendering options
    diff_type: DiffType = DiffType.UNIFIED
    syntax_highlighting: bool = True
    show_line_numbers: bool = True


class ApprovalDialog:
    """
    Structured approval dialog for agent operations.

    Provides a comprehensive interface for reviewing and approving
    agent operations with rich formatting, diff display, and
    interactive controls.
    """

    def __init__(
        self,
        renderer: Optional[RichTextRenderer] = None,
        diff_renderer: Optional[EnhancedDiffRenderer] = None,
        console: Optional[Console] = None,
        options: Optional[DialogOptions] = None,
        input_handler: Optional[InteractiveInputHandler] = None,
    ):
        """
        Initialize the approval dialog.

        Args:
            renderer: Rich text renderer
            diff_renderer: Enhanced diff renderer
            console: Rich console for output
            options: Dialog configuration options
            input_handler: Interactive input handler for keyboard controls
        """
        self.renderer = renderer or create_renderer()
        self.diff_renderer = diff_renderer or create_diff_renderer(self.renderer)
        self.console = console or self.renderer.console
        self.options = options or DialogOptions()

        # Interactive input handling
        self.input_handler = input_handler or create_input_handler(
            timeout_seconds=self.options.timeout_seconds
        )
        self.input_handler.set_callbacks(
            approval_callback=self._handle_user_approval,
            denial_callback=self._handle_user_denial,
            quit_callback=self._handle_user_quit,
            display_update_callback=self._handle_display_update,
        )

        # Dialog state
        self.state = DialogState.INITIALIZING
        self.current_context: Optional[ApprovalContext] = None
        self.decision: Optional[ApprovalDecision] = None

        # Layout components
        self.layout = Layout()
        self.sections: Dict[str, Layout] = {}

        # Legacy shortcuts for compatibility
        self.shortcuts: Dict[str, str] = {
            "y": "Approve",
            "n": "Deny",
            "d": "Show Details",
            "f": "Show Full Diff",
            "s": "Show Summary",
            "q": "Quit",
            "h": "Help",
        }

        # Initialize layout
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Set up the dialog layout structure."""
        # Calculate responsive dimensions
        self.renderer.get_terminal_width()
        term_height = self.renderer.get_terminal_height()

        # Initialize sections dictionary
        self.sections = {}

        # Adjust for compact mode
        if self.options.compact_mode or term_height < 25:
            self.layout.split(
                Layout(name="header", size=3),
                Layout(name="main", ratio=1),
                Layout(name="controls", size=4),
            )
            # Store basic sections
            self.sections.update(
                {
                    "header": self.layout["header"],
                    "main": self.layout["main"],
                    "controls": self.layout["controls"],
                }
            )
        else:
            self.layout.split(
                Layout(name="header", size=3),
                Layout(name="operation", size=6),
                Layout(name="main", ratio=1),
                Layout(name="risk", size=5),
                Layout(name="controls", size=4),
            )
            # Store all sections including operation and risk
            self.sections.update(
                {
                    "header": self.layout["header"],
                    "operation": self.layout["operation"],
                    "main": self.layout["main"],
                    "risk": self.layout["risk"],
                    "controls": self.layout["controls"],
                }
            )

        # Set up main content area
        if not self.options.compact_mode:
            self.layout["main"].split_row(
                Layout(name="diff", ratio=2), Layout(name="details", ratio=1)
            )
        else:
            # Single column for compact mode
            self.layout["main"].split(
                Layout(name="diff", ratio=2), Layout(name="details", ratio=1)
            )

        # Add main content sections
        self.sections.update(
            {
                "diff": self.layout["main"]["diff"],
                "details": self.layout["main"]["details"],
            }
        )

    def _render_header(self, context: ApprovalContext) -> Panel:
        """Render the dialog header."""
        title = Text()
        title.append("🔒 ", style="warning")
        title.append("Operation Approval Required", style="header")

        subtitle = Text()
        subtitle.append(f"Agent: ", style="label")
        subtitle.append(context.agent_name, style="value")
        subtitle.append(" | ", style="dim")
        subtitle.append(f"Time: ", style="label")
        subtitle.append(context.timestamp.strftime("%H:%M:%S"), style="value")

        return Panel(
            Align.center(subtitle),
            title=title,
            border_style="cyan",
            box=ROUNDED,
        )

    def _render_operation_summary(self, operation: OperationDetails) -> Panel:
        """Render operation summary information."""
        # Create summary table
        table = Table(show_header=False, box=None, padding=0)
        table.add_column("Field", style="label", width=15)
        table.add_column("Value", style="value")

        # Add operation details
        table.add_row("Operation", operation.operation_type)
        table.add_row("Target", operation.target or "N/A")
        table.add_row("Risk Level", self._format_risk_level(operation.risk_level))

        if operation.estimated_time:
            table.add_row("Est. Time", f"{operation.estimated_time}s")

        if operation.files_affected:
            table.add_row("Files", str(len(operation.files_affected)))

        # Add description if available
        if operation.description:
            description = Text(operation.description)
            if len(operation.description) > 60:
                description = Text(operation.description[:57] + "...")
            table.add_row("Description", description)

        return Panel(
            table,
            title="📋 Operation Summary",
            border_style="blue",
            box=SIMPLE,
        )

    def _render_risk_assessment(self, operation: OperationDetails) -> Panel:
        """Render risk assessment panel."""
        # Risk indicator
        risk_indicator = self.renderer.render_risk_indicator(
            operation.risk_level, show_label=True, show_bar=True
        )

        # Risk factors
        factors = []
        if operation.risk_factors:
            for factor in operation.risk_factors[:3]:  # Show top 3
                factors.append(f"• {factor}")

        if len(operation.risk_factors) > 3:
            factors.append(f"• ... and {len(operation.risk_factors) - 3} more")

        content = []
        content.append(risk_indicator)

        if factors:
            content.append(Rule(style="dim"))
            content.append(Text("\n".join(factors), style="warning"))

        return Panel(
            Align.left("\n".join([str(c) for c in content])),
            title="⚠️ Risk Assessment",
            border_style=self._get_risk_border_color(operation.risk_level),
            box=ROUNDED,
        )

    def _render_diff_panel(self, context: ApprovalContext) -> Panel:
        """Render file differences panel."""
        if not context.diff_content:
            return Panel(
                Align.center("No file changes to display"),
                title="📄 Changes",
                border_style="dim",
                box=SIMPLE,
            )

        # Render diff using enhanced diff renderer
        renderables = self.diff_renderer.render_diff_set(
            context.diff_content,
            diff_type=self.options.diff_type,
            show_summary=not self.options.compact_mode,
            show_file_tree=self.options.show_file_tree,
        )

        # Combine renderables
        if len(renderables) == 1:
            content = renderables[0]
        else:
            # Join with rules between sections
            combined = []
            for i, renderable in enumerate(renderables):
                if i > 0:
                    combined.append(Rule(style="dim"))
                combined.append(renderable)
            content = "\n".join([str(r) for r in combined])

        return Panel(
            content,
            title="📄 File Changes",
            border_style="yellow",
            box=SIMPLE,
            expand=True,
        )

    def _render_operation_details(self, operation: OperationDetails) -> Panel:
        """Render detailed operation information."""
        sections = []

        # Command/Script details
        if operation.command:
            sections.append(self._render_command_details(operation))

        # File operations
        if operation.files_affected:
            sections.append(self._render_file_details(operation))

        # Environment/Context
        if operation.environment_vars or operation.working_directory:
            sections.append(self._render_environment_details(operation))

        # Metadata
        if operation.metadata:
            sections.append(self._render_metadata(operation))

        if not sections:
            content = Align.center("No additional details available")
        else:
            content = "\n".join([str(s) for s in sections])

        return Panel(
            content,
            title="🔍 Details",
            border_style="cyan",
            box=SIMPLE,
            expand=True,
        )

    def _render_command_details(self, operation: OperationDetails) -> Table:
        """Render command execution details."""
        table = Table(show_header=True, box=MINIMAL)
        table.add_column("Property", style="label")
        table.add_column("Value", style="value")

        table.add_row("Command", operation.command)
        if operation.arguments:
            table.add_row("Arguments", " ".join(operation.arguments))
        if operation.working_directory:
            table.add_row("Working Dir", operation.working_directory)

        return table

    def _render_file_details(self, operation: OperationDetails) -> Table:
        """Render file operation details."""
        table = Table(show_header=True, box=MINIMAL)
        table.add_column("File", style="value", width=30)
        table.add_column("Operation", style="label")
        table.add_column("Size", style="dim", justify="right")

        for file_path in operation.files_affected[:5]:  # Show first 5
            # Determine operation type from file path context
            op_type = "modify"  # Default
            size = "N/A"

            table.add_row(str(file_path), op_type, size)

        if len(operation.files_affected) > 5:
            table.add_row(
                f"... and {len(operation.files_affected) - 5} more files",
                "",
                "",
            )

        return table

    def _render_environment_details(self, operation: OperationDetails) -> Table:
        """Render environment and context details."""
        table = Table(show_header=True, box=MINIMAL)
        table.add_column("Variable", style="label")
        table.add_column("Value", style="value")

        if operation.working_directory:
            table.add_row("PWD", operation.working_directory)

        if operation.environment_vars:
            for key, value in list(operation.environment_vars.items())[:3]:
                # Mask sensitive values
                if any(
                    sensitive in key.lower()
                    for sensitive in ["key", "token", "password", "secret"]
                ):
                    value = "***"
                elif len(str(value)) > 40:
                    value = str(value)[:37] + "..."

                table.add_row(key, str(value))

        return table

    def _render_metadata(self, operation: OperationDetails) -> Table:
        """Render operation metadata."""
        table = Table(show_header=True, box=MINIMAL)
        table.add_column("Key", style="label")
        table.add_column("Value", style="value")

        for key, value in list(operation.metadata.items())[:3]:
            if len(str(value)) > 40:
                value = str(value)[:37] + "..."
            table.add_row(key, str(value))

        return table

    def _render_controls(self) -> Panel:
        """Render control buttons and shortcuts."""
        if self.options.enable_shortcuts:
            shortcuts_help = self.renderer.render_shortcuts_help(
                self.shortcuts,
                columns=2 if self.renderer.get_terminal_width() > 80 else 1,
            )
            content = shortcuts_help
        else:
            # Simple button layout
            content = Text("Use 'y' to approve, 'n' to deny, 'q' to quit")

        return Panel(content, title="⌨️ Controls", border_style="green", box=SIMPLE)

    def _format_risk_level(self, risk_level: Union[RiskLevel, int, str]) -> Text:
        """Format risk level with appropriate styling."""
        if isinstance(risk_level, int):
            risk_text = f"Level {risk_level}"
            if risk_level <= 3:
                style = "green"
            elif risk_level <= 6:
                style = "yellow"
            else:
                style = "red"
        else:
            risk_text = str(risk_level).upper()
            style = self.renderer.get_risk_color(risk_level)

        return Text(risk_text, style=style)

    def _get_risk_border_color(self, risk_level: Union[RiskLevel, int, str]) -> str:
        """Get appropriate border color for risk level."""
        if isinstance(risk_level, int):
            if risk_level <= 3:
                return "green"
            elif risk_level <= 6:
                return "yellow"
            else:
                return "red"

        risk_colors = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "red",
            RiskLevel.CRITICAL: "bold red",
        }

        return risk_colors.get(risk_level, "yellow")

    async def show_approval_dialog(self, context: ApprovalContext) -> ApprovalDecision:
        """
        Display the approval dialog and wait for user decision.

        Args:
            context: Approval context with operation details

        Returns:
            User's approval decision
        """
        try:
            self.state = DialogState.DISPLAYING
            self.current_context = context

            # Render all dialog sections
            self.sections["header"].update(self._render_header(context))

            if not self.options.compact_mode and "operation" in self.sections:
                self.sections["operation"].update(
                    self._render_operation_summary(context.operation_details)
                )

            if not self.options.compact_mode and "risk" in self.sections:
                self.sections["risk"].update(
                    self._render_risk_assessment(context.operation_details)
                )

            self.sections["diff"].update(self._render_diff_panel(context))
            self.sections["details"].update(
                self._render_operation_details(context.operation_details)
            )
            self.sections["controls"].update(self._render_controls())

            # Show the dialog with Live display
            with Live(
                self.layout,
                console=self.console,
                refresh_per_second=4,
                auto_refresh=True,
            ) as live:
                self.state = DialogState.WAITING_INPUT
                self.live_display = live  # Store for display updates

                # Wait for user input
                decision = await self._wait_for_decision()

                self.state = DialogState.COMPLETED
                self.decision = decision

                return decision

        except KeyboardInterrupt:
            self.state = DialogState.CANCELLED
            return ApprovalDecision(
                approved=False,
                reason="Operation cancelled by user",
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Dialog error: {e}", exc_info=True)
            self.state = DialogState.CANCELLED
            return ApprovalDecision(
                approved=False,
                reason=f"Dialog error: {e}",
                timestamp=datetime.now(),
            )

    async def _wait_for_decision(self) -> ApprovalDecision:
        """Wait for user input and return decision."""
        try:
            # Use the interactive input handler to get user decision
            final_action = await self.input_handler.handle_input_loop()

            # Create decision based on the action
            if final_action == KeyAction.APPROVE:
                return ApprovalDecision(
                    approved=True,
                    reason="Approved by user",
                    timestamp=datetime.now(),
                    additional_data={
                        "input_action": "approve",
                        "interactive": True,
                    },
                )
            elif final_action == KeyAction.DENY:
                return ApprovalDecision(
                    approved=False,
                    reason="Denied by user",
                    timestamp=datetime.now(),
                    additional_data={
                        "input_action": "deny",
                        "interactive": True,
                    },
                )
            else:  # QUIT or timeout
                return ApprovalDecision(
                    approved=False,
                    reason="Operation cancelled or timed out",
                    timestamp=datetime.now(),
                    additional_data={
                        "input_action": "quit",
                        "interactive": True,
                    },
                )

        except Exception as e:
            logger.error(f"Error in input handling: {e}")
            return ApprovalDecision(
                approved=False,
                reason=f"Input handling error: {e}",
                timestamp=datetime.now(),
                additional_data={"input_error": str(e)},
            )

    def get_current_state(self) -> DialogState:
        """Get current dialog state."""
        return self.state

    def get_decision(self) -> Optional[ApprovalDecision]:
        """Get the approval decision if available."""
        return self.decision

    # Callback methods for input handler integration

    async def _handle_user_approval(self) -> bool:
        """Handle user approval action."""
        logger.info("User approved operation")
        return True

    async def _handle_user_denial(self) -> bool:
        """Handle user denial action."""
        logger.info("User denied operation")
        return True

    async def _handle_user_quit(self) -> bool:
        """Handle user quit action."""
        logger.info("User quit approval dialog")
        return True

    async def _handle_display_update(self, input_state: InputState) -> None:
        """Handle display updates based on input state changes."""
        try:
            # Update dialog based on input state
            if hasattr(self, "live_display") and self.current_context:

                # Update sections based on input state
                if input_state.help_visible:
                    # Show help overlay
                    help_panel = Panel(
                        self.input_handler.get_help_text(),
                        title="🔍 Keyboard Shortcuts",
                        border_style="cyan",
                        box=ROUNDED,
                    )

                    # Replace main content with help
                    if "main" in self.sections:
                        self.sections["main"].update(help_panel)

                else:
                    # Show normal content - update diff section if focused
                    if (
                        input_state.current_section == "diff"
                        and "diff" in self.sections
                    ):
                        # Potentially update diff rendering based on expanded state
                        diff_panel = self._render_diff_panel(
                            self.current_context,
                            expanded=input_state.diff_expanded,
                        )
                        self.sections["diff"].update(diff_panel)

                    # Update details section if focused
                    elif (
                        input_state.current_section == "details"
                        and "details" in self.sections
                    ):
                        details_panel = self._render_operation_details(
                            self.current_context.operation_details,
                            expanded=input_state.details_expanded,
                        )
                        self.sections["details"].update(details_panel)

                # Update status/controls with current input state
                if "controls" in self.sections:
                    controls_panel = self._render_controls_with_state(input_state)
                    self.sections["controls"].update(controls_panel)

                # Refresh the display
                self.live_display.refresh()

        except Exception as e:
            logger.error(f"Error updating display: {e}")

    def _render_diff_panel(
        self, context: ApprovalContext, expanded: bool = False
    ) -> Panel:
        """Render file differences panel with optional expansion."""
        if not context.diff_content:
            return Panel(
                Align.center("No file changes to display"),
                title="📄 Changes",
                border_style="dim",
                box=SIMPLE,
            )

        # Render diff using enhanced diff renderer
        renderables = self.diff_renderer.render_diff_set(
            context.diff_content,
            diff_type=(DiffType.SIDE_BY_SIDE if expanded else self.options.diff_type),
            show_summary=not self.options.compact_mode or expanded,
            show_file_tree=self.options.show_file_tree,
        )

        # Combine renderables
        if len(renderables) == 1:
            content = renderables[0]
        else:
            # Join with rules between sections
            combined = []
            for i, renderable in enumerate(renderables):
                if i > 0:
                    combined.append(Rule(style="dim"))
                combined.append(renderable)
            content = "\n".join([str(r) for r in combined])

        title = "📄 File Changes (Expanded)" if expanded else "📄 File Changes"

        return Panel(
            content,
            title=title,
            border_style="yellow" if expanded else "blue",
            box=SIMPLE,
            expand=True,
        )

    def _render_operation_details(
        self, operation: OperationDetails, expanded: bool = False
    ) -> Panel:
        """Render detailed operation information with optional expansion."""
        sections = []

        # Command/Script details
        if operation.command:
            sections.append(self._render_command_details(operation))

        # File operations
        if operation.files_affected:
            sections.append(self._render_file_details(operation, show_all=expanded))

        # Environment/Context
        if operation.environment_vars or operation.working_directory:
            sections.append(
                self._render_environment_details(operation, show_all=expanded)
            )

        # Metadata
        if operation.metadata:
            sections.append(self._render_metadata(operation, show_all=expanded))

        if not sections:
            content = Align.center("No additional details available")
        else:
            content = "\n".join([str(s) for s in sections])

        title = "🔍 Details (Expanded)" if expanded else "🔍 Details"

        return Panel(
            content,
            title=title,
            border_style="cyan" if expanded else "green",
            box=SIMPLE,
            expand=True,
        )

    def _render_file_details(
        self, operation: OperationDetails, show_all: bool = False
    ) -> Table:
        """Render file operation details with optional full listing."""
        table = Table(show_header=True, box=MINIMAL)
        table.add_column("File", style="value", width=30)
        table.add_column("Operation", style="label")
        table.add_column("Size", style="dim", justify="right")

        file_limit = len(operation.files_affected) if show_all else 5

        for file_path in operation.files_affected[:file_limit]:
            # Determine operation type from file path context
            op_type = "modify"  # Default
            size = "N/A"

            table.add_row(str(file_path), op_type, size)

        if not show_all and len(operation.files_affected) > 5:
            table.add_row(
                f"... and {len(operation.files_affected) - 5} more files",
                "",
                "",
            )

        return table

    def _render_environment_details(
        self, operation: OperationDetails, show_all: bool = False
    ) -> Table:
        """Render environment and context details with optional full listing."""
        table = Table(show_header=True, box=MINIMAL)
        table.add_column("Variable", style="label")
        table.add_column("Value", style="value")

        if operation.working_directory:
            table.add_row("PWD", operation.working_directory)

        if operation.environment_vars:
            var_limit = len(operation.environment_vars) if show_all else 3
            for key, value in list(operation.environment_vars.items())[:var_limit]:
                # Mask sensitive values
                if any(
                    sensitive in key.lower()
                    for sensitive in ["key", "token", "password", "secret"]
                ):
                    value = "***"
                elif len(str(value)) > 40:
                    value = str(value)[:37] + "..."

                table.add_row(key, str(value))

        return table

    def _render_metadata(
        self, operation: OperationDetails, show_all: bool = False
    ) -> Table:
        """Render operation metadata with optional full listing."""
        table = Table(show_header=True, box=MINIMAL)
        table.add_column("Key", style="label")
        table.add_column("Value", style="value")

        item_limit = len(operation.metadata) if show_all else 3
        for key, value in list(operation.metadata.items())[:item_limit]:
            if len(str(value)) > 40:
                value = str(value)[:37] + "..."
            table.add_row(key, str(value))

        return table

    def _render_controls_with_state(self, input_state: InputState) -> Panel:
        """Render control buttons with current state information."""
        if self.options.enable_shortcuts:
            # Show shortcuts with current state
            shortcuts_text = self.renderer.render_shortcuts_help(
                self.shortcuts,
                columns=2 if self.renderer.get_terminal_width() > 80 else 1,
            )

            # Add state information
            state_info = Text()
            state_info.append(f"Mode: {input_state.mode.value} | ", style="dim")
            state_info.append(f"Section: {input_state.current_section} | ", style="dim")
            state_info.append(f"Scroll: {input_state.scroll_position}", style="dim")

            content = f"{shortcuts_text}\n{state_info}"
        else:
            # Simple button layout with state
            content = Text(
                f"Use 'y' to approve, 'n' to deny, 'q' to quit | Mode: {input_state.mode.value}"
            )

        return Panel(content, title="⌨️ Controls", border_style="green", box=SIMPLE)


# Utility functions


def create_approval_dialog(
    renderer: Optional[RichTextRenderer] = None,
    options: Optional[DialogOptions] = None,
) -> ApprovalDialog:
    """Create an approval dialog with default configuration."""
    return ApprovalDialog(renderer=renderer, options=options)


async def show_quick_approval(
    operation_type: str,
    target: str,
    risk_level: Union[RiskLevel, int] = RiskLevel.MEDIUM,
    console: Optional[Console] = None,
) -> bool:
    """Quick approval dialog for simple operations."""
    context = ApprovalContext(
        agent_name="Omnimancer",
        operation_details=OperationDetails(
            operation_type=operation_type, target=target, risk_level=risk_level
        ),
        timestamp=datetime.now(),
    )

    # Use compact dialog for quick approval
    options = DialogOptions(compact_mode=True, show_file_tree=False, timeout_seconds=30)
    dialog = create_approval_dialog(options=options)

    if console:
        dialog.console = console

    decision = await dialog.show_approval_dialog(context)
    return decision.approved
