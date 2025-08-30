"""
Rich Text Rendering System for Omnimancer Terminal Interface.

This module provides comprehensive rich text rendering capabilities for the
approval interface and other terminal outputs, with color support, syntax
highlighting, and responsive layouts.
"""

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.box import ROUNDED, Box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


class RiskLevel(Enum):
    """Risk levels for operations with associated colors."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OperationType(Enum):
    """Types of operations for color coding."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    COMMAND_EXECUTE = "command_execute"
    WEB_REQUEST = "web_request"
    SYSTEM_CHANGE = "system_change"
    DATA_ACCESS = "data_access"


@dataclass
class ColorScheme:
    """Color scheme configuration for different UI elements."""

    # Risk level colors
    risk_none: str = "dim white"
    risk_low: str = "green"
    risk_medium: str = "yellow"
    risk_high: str = "bright_red"
    risk_critical: str = "bold red on white"

    # Operation type colors
    op_read: str = "cyan"
    op_write: str = "yellow"
    op_delete: str = "red"
    op_execute: str = "magenta"
    op_web: str = "blue"
    op_system: str = "bright_red"

    # Diff colors
    diff_added: str = "green"
    diff_removed: str = "red"
    diff_modified: str = "yellow"
    diff_context: str = "dim white"
    diff_header: str = "bold cyan"
    diff_line_number: str = "dim blue"

    # UI element colors
    header: str = "bold cyan"
    subheader: str = "cyan"
    label: str = "bright_blue"
    value: str = "white"
    success: str = "bold green"
    error: str = "bold red"
    warning: str = "yellow"
    info: str = "blue"
    dim: str = "dim white"

    # Interactive element colors
    shortcut_key: str = "bold yellow"
    shortcut_desc: str = "white"
    selected: str = "reverse"
    focused: str = "bold cyan"


@dataclass
class TerminalCapabilities:
    """Terminal capabilities detection and configuration."""

    width: int = 80
    height: int = 24
    supports_color: bool = True
    supports_unicode: bool = True
    supports_emoji: bool = True
    color_depth: int = 256  # 8, 16, 256, or 16777216 (true color)

    @classmethod
    def detect(cls) -> "TerminalCapabilities":
        """Detect current terminal capabilities."""
        # Get terminal dimensions
        size = shutil.get_terminal_size((80, 24))

        # Detect color support
        supports_color = (
            os.environ.get("COLORTERM") in ("truecolor", "24bit")
            or os.environ.get("TERM", "").endswith("256color")
            or "color" in os.environ.get("TERM", "").lower()
            or os.environ.get("TERM", "").startswith("xterm")
        )

        # Detect Unicode support
        supports_unicode = True
        try:
            "✓".encode(
                os.environ.get("LANG", "ascii").split(".")[1]
                if "." in os.environ.get("LANG", "")
                else "utf-8"
            )
        except (UnicodeEncodeError, LookupError):
            supports_unicode = False

        # Detect emoji support (simplified)
        supports_emoji = (
            supports_unicode and os.name != "nt"
        )  # Windows terminal emoji support is limited

        # Detect color depth
        color_depth = 8
        if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
            color_depth = 16777216
        elif "256color" in os.environ.get("TERM", ""):
            color_depth = 256
        elif supports_color:
            color_depth = 16

        return cls(
            width=size.columns,
            height=size.lines,
            supports_color=supports_color,
            supports_unicode=supports_unicode,
            supports_emoji=supports_emoji,
            color_depth=color_depth,
        )


class RichTextRenderer:
    """
    Rich text rendering system for terminal output.

    Provides comprehensive formatting, color coding, and layout capabilities
    for the approval interface and other terminal displays.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        color_scheme: Optional[ColorScheme] = None,
        auto_detect: bool = True,
    ):
        """
        Initialize the rich text renderer.

        Args:
            console: Rich console instance (creates new if not provided)
            color_scheme: Color scheme to use (uses default if not provided)
            auto_detect: Whether to auto-detect terminal capabilities
        """
        self.console = console or Console()
        self.color_scheme = color_scheme or ColorScheme()

        # Detect terminal capabilities
        if auto_detect:
            self.capabilities = TerminalCapabilities.detect()
        else:
            self.capabilities = TerminalCapabilities()

        # Create theme from color scheme
        self.theme = self._create_theme()

        # Apply theme to console
        self.console = Console(theme=self.theme)

        # Layout cache for responsive designs
        self._layout_cache: Dict[str, Layout] = {}

    def _create_theme(self) -> Theme:
        """Create Rich theme from color scheme."""
        theme_dict = {
            # Risk levels
            "risk.none": self.color_scheme.risk_none,
            "risk.low": self.color_scheme.risk_low,
            "risk.medium": self.color_scheme.risk_medium,
            "risk.high": self.color_scheme.risk_high,
            "risk.critical": self.color_scheme.risk_critical,
            # Operations
            "op.read": self.color_scheme.op_read,
            "op.write": self.color_scheme.op_write,
            "op.delete": self.color_scheme.op_delete,
            "op.execute": self.color_scheme.op_execute,
            "op.web": self.color_scheme.op_web,
            "op.system": self.color_scheme.op_system,
            # Diff
            "diff.added": self.color_scheme.diff_added,
            "diff.removed": self.color_scheme.diff_removed,
            "diff.modified": self.color_scheme.diff_modified,
            "diff.context": self.color_scheme.diff_context,
            "diff.header": self.color_scheme.diff_header,
            "diff.line_number": self.color_scheme.diff_line_number,
            # UI elements
            "header": self.color_scheme.header,
            "subheader": self.color_scheme.subheader,
            "label": self.color_scheme.label,
            "value": self.color_scheme.value,
            "success": self.color_scheme.success,
            "error": self.color_scheme.error,
            "warning": self.color_scheme.warning,
            "info": self.color_scheme.info,
            "dim": self.color_scheme.dim,
            # Interactive
            "shortcut.key": self.color_scheme.shortcut_key,
            "shortcut.desc": self.color_scheme.shortcut_desc,
            "selected": self.color_scheme.selected,
            "focused": self.color_scheme.focused,
        }

        return Theme(theme_dict)

    def get_risk_color(self, risk_level: Union[RiskLevel, str, int, None]) -> str:
        """
        Get color style for risk level.

        Args:
            risk_level: Risk level (enum, string, or numeric 0-10)

        Returns:
            Style string for the risk level
        """
        if risk_level is None:
            return "risk.none"

        if isinstance(risk_level, int):
            # Convert numeric risk to level
            if risk_level <= 2:
                risk_level = RiskLevel.LOW
            elif risk_level <= 4:
                risk_level = RiskLevel.MEDIUM
            elif risk_level <= 7:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
        elif isinstance(risk_level, str):
            try:
                risk_level = RiskLevel(risk_level.lower())
            except ValueError:
                risk_level = RiskLevel.NONE

        risk_styles = {
            RiskLevel.NONE: "risk.none",
            RiskLevel.LOW: "risk.low",
            RiskLevel.MEDIUM: "risk.medium",
            RiskLevel.HIGH: "risk.high",
            RiskLevel.CRITICAL: "risk.critical",
        }

        return risk_styles.get(risk_level, "risk.none")

    def get_operation_color(self, operation_type: Union[OperationType, str]) -> str:
        """
        Get color style for operation type.

        Args:
            operation_type: Type of operation

        Returns:
            Style string for the operation type
        """
        if isinstance(operation_type, str):
            # Map string to operation type
            op_mapping = {
                "read": OperationType.FILE_READ,
                "write": OperationType.FILE_WRITE,
                "delete": OperationType.FILE_DELETE,
                "execute": OperationType.COMMAND_EXECUTE,
                "web": OperationType.WEB_REQUEST,
                "system": OperationType.SYSTEM_CHANGE,
            }

            for key, value in op_mapping.items():
                if key in operation_type.lower():
                    operation_type = value
                    break
            else:
                operation_type = OperationType.FILE_READ  # Default

        op_styles = {
            OperationType.FILE_READ: "op.read",
            OperationType.FILE_WRITE: "op.write",
            OperationType.FILE_DELETE: "op.delete",
            OperationType.COMMAND_EXECUTE: "op.execute",
            OperationType.WEB_REQUEST: "op.web",
            OperationType.SYSTEM_CHANGE: "op.system",
            OperationType.DATA_ACCESS: "op.read",
        }

        return op_styles.get(operation_type, "op.read")

    def render_code_block(
        self,
        code: str,
        language: Optional[str] = None,
        theme: str = "monokai",
        line_numbers: bool = True,
        highlight_lines: Optional[List[int]] = None,
    ) -> Syntax:
        """
        Render a code block with syntax highlighting.

        Args:
            code: Code content to render
            language: Programming language for syntax highlighting
            theme: Pygments theme to use
            line_numbers: Whether to show line numbers
            highlight_lines: Lines to highlight

        Returns:
            Syntax object for rendering
        """
        if not language:
            # Try to auto-detect language
            if "import " in code or "def " in code or "class " in code:
                language = "python"
            elif "function" in code or "const " in code or "var " in code:
                language = "javascript"
            elif "{" in code and "}" in code and ":" in code:
                language = "json"
            else:
                language = "text"

        syntax = Syntax(
            code,
            language,
            theme=theme,
            line_numbers=line_numbers,
            highlight_lines=set(highlight_lines) if highlight_lines else None,
            word_wrap=True,
        )

        return syntax

    def render_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        title: Optional[str] = None,
        box_style: Box = ROUNDED,
        header_style: str = "header",
        show_footer: bool = False,
    ) -> Table:
        """
        Render a formatted table.

        Args:
            headers: Column headers
            rows: Table data rows
            title: Optional table title
            box_style: Box drawing style
            header_style: Style for headers
            show_footer: Whether to show footer

        Returns:
            Table object for rendering
        """
        table = Table(
            title=title,
            box=box_style,
            header_style=header_style,
            show_footer=show_footer,
            width=(
                None if self.capabilities.width > 120 else self.capabilities.width - 2
            ),
        )

        # Add columns
        for header in headers:
            table.add_column(header, justify="left")

        # Add rows
        for row in rows:
            # Convert all values to strings and apply appropriate styles
            styled_row = []
            for i, value in enumerate(row):
                if isinstance(value, bool):
                    styled_value = Text(
                        "✓" if value else "✗",
                        style="success" if value else "error",
                    )
                elif (
                    isinstance(value, (int, float)) and i > 0
                ):  # Numeric columns (not first)
                    styled_value = Text(str(value), style="value")
                else:
                    styled_value = str(value)
                styled_row.append(styled_value)

            table.add_row(*styled_row)

        return table

    def render_panel(
        self,
        content: Union[str, Text, Any],
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        box_style: Box = ROUNDED,
        border_style: Optional[str] = None,
        padding: Tuple[int, int] = (0, 1),
        expand: bool = True,
    ) -> Panel:
        """
        Render a panel with content.

        Args:
            content: Content to display in panel
            title: Panel title
            subtitle: Panel subtitle
            box_style: Box drawing style
            border_style: Border color/style
            padding: Padding (vertical, horizontal)
            expand: Whether to expand to full width

        Returns:
            Panel object for rendering
        """
        # Apply responsive width
        width = None
        if not expand and self.capabilities.width < 100:
            width = self.capabilities.width - 4

        panel = Panel(
            content,
            title=title,
            subtitle=subtitle,
            box=box_style,
            border_style=border_style or "dim",
            padding=padding,
            expand=expand,
            width=width,
        )

        return panel

    def render_risk_indicator(
        self,
        risk_level: Union[RiskLevel, int],
        show_label: bool = True,
        show_bar: bool = True,
    ) -> Union[Text, Columns]:
        """
        Render a risk level indicator.

        Args:
            risk_level: Risk level to display
            show_label: Whether to show text label
            show_bar: Whether to show visual bar

        Returns:
            Formatted risk indicator
        """
        if isinstance(risk_level, int):
            numeric_risk = risk_level
            # Convert to RiskLevel
            if risk_level <= 2:
                risk_enum = RiskLevel.LOW
            elif risk_level <= 4:
                risk_enum = RiskLevel.MEDIUM
            elif risk_level <= 7:
                risk_enum = RiskLevel.HIGH
            else:
                risk_enum = RiskLevel.CRITICAL
        else:
            risk_enum = risk_level
            # Estimate numeric value
            risk_values = {
                RiskLevel.NONE: 0,
                RiskLevel.LOW: 2,
                RiskLevel.MEDIUM: 5,
                RiskLevel.HIGH: 7,
                RiskLevel.CRITICAL: 9,
            }
            numeric_risk = risk_values.get(risk_enum, 0)

        elements = []

        if show_label:
            risk_text = Text()
            risk_text.append("Risk: ", style="label")
            risk_text.append(
                risk_enum.value.upper(), style=self.get_risk_color(risk_enum)
            )
            risk_text.append(f" ({numeric_risk}/10)", style="dim")
            elements.append(risk_text)

        if show_bar:
            # Create visual risk bar
            bar_width = min(20, self.capabilities.width // 4)
            filled = int((numeric_risk / 10) * bar_width)
            empty = bar_width - filled

            bar = Text()
            bar.append("[", style="dim")

            # Gradient coloring for filled portion
            for i in range(filled):
                progress = (i + 1) / bar_width
                if progress < 0.3:
                    style = "green"
                elif progress < 0.6:
                    style = "yellow"
                else:
                    style = "red"
                bar.append("█", style=style)

            bar.append("░" * empty, style="dim")
            bar.append("]", style="dim")
            elements.append(bar)

        if len(elements) == 1:
            return elements[0]
        else:
            return Columns(elements, padding=1)

    def render_progress_bar(
        self,
        task_description: str,
        total: Optional[int] = None,
        completed: Optional[int] = None,
    ) -> Progress:
        """
        Create a progress bar for operations.

        Args:
            task_description: Description of the task
            total: Total number of steps
            completed: Number of completed steps

        Returns:
            Progress bar object
        """
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )

        task = progress.add_task(task_description, total=total)

        if completed is not None and total is not None:
            progress.update(task, completed=completed)

        return progress

    def create_responsive_layout(
        self, sections: Dict[str, Any], vertical_threshold: int = 100
    ) -> Union[Columns, Layout]:
        """
        Create a responsive layout that adapts to terminal width.

        Args:
            sections: Dictionary of section name to content
            vertical_threshold: Width below which to use vertical layout

        Returns:
            Layout object appropriate for terminal width
        """
        if self.capabilities.width < vertical_threshold:
            # Vertical layout for narrow terminals
            layout = Layout()
            for i, (name, content) in enumerate(sections.items()):
                if i > 0:
                    layout.add_split(Layout(Rule(style="dim")))
                layout.add_split(Layout(content, name=name))
            return layout
        else:
            # Horizontal layout for wide terminals
            columns = []
            for name, content in sections.items():
                columns.append(content)
            return Columns(columns, equal=True, padding=1)

    def format_shortcut(self, key: str, description: str, enabled: bool = True) -> Text:
        """
        Format a keyboard shortcut with description.

        Args:
            key: Keyboard shortcut key(s)
            description: Description of the action
            enabled: Whether the shortcut is currently enabled

        Returns:
            Formatted shortcut text
        """
        text = Text()

        if enabled:
            text.append(f"[{key}]", style="shortcut.key")
            text.append(f" {description}", style="shortcut.desc")
        else:
            text.append(f"[{key}]", style="dim")
            text.append(f" {description}", style="dim")

        return text

    def render_shortcuts_help(
        self, shortcuts: Dict[str, str], columns: int = 2
    ) -> Union[Table, Columns]:
        """
        Render a help panel for keyboard shortcuts.

        Args:
            shortcuts: Dictionary of key to description
            columns: Number of columns for display

        Returns:
            Formatted shortcuts help
        """
        if self.capabilities.width < 60:
            # Single column for narrow terminals
            table = Table(show_header=False, box=None, padding=0)
            table.add_column("Shortcut", style="shortcut.key")
            table.add_column("Action", style="shortcut.desc")

            for key, desc in shortcuts.items():
                table.add_row(f"[{key}]", desc)

            return table
        else:
            # Multi-column layout
            shortcut_texts = []
            for key, desc in shortcuts.items():
                shortcut_texts.append(self.format_shortcut(key, desc))

            # Split into columns
            items_per_col = len(shortcut_texts) // columns + (
                1 if len(shortcut_texts) % columns else 0
            )
            columns_list = []

            for i in range(columns):
                start = i * items_per_col
                end = min(start + items_per_col, len(shortcut_texts))
                if start < len(shortcut_texts):
                    col_items = Text("\n").join(shortcut_texts[start:end])
                    columns_list.append(col_items)

            return Columns(columns_list, equal=True, padding=2)

    def apply_diff_highlighting(
        self, diff_text: str, file_type: Optional[str] = None
    ) -> Text:
        """
        Apply syntax highlighting to diff text.

        Args:
            diff_text: Diff content to highlight
            file_type: File type for syntax highlighting

        Returns:
            Highlighted diff text
        """
        result = Text()

        for line in diff_text.split("\n"):
            if line.startswith("+++") or line.startswith("---"):
                # File headers
                result.append(line + "\n", style="diff.header")
            elif line.startswith("@@"):
                # Hunk headers
                result.append(line + "\n", style="diff.line_number")
            elif line.startswith("+"):
                # Added lines
                result.append(line + "\n", style="diff.added")
            elif line.startswith("-"):
                # Removed lines
                result.append(line + "\n", style="diff.removed")
            else:
                # Context lines
                result.append(line + "\n", style="diff.context")

        return result

    def render_with_fallback(self, rich_content: Any, plain_text: str) -> None:
        """
        Render content with fallback for limited terminals.

        Args:
            rich_content: Rich formatted content
            plain_text: Plain text fallback
        """
        if self.capabilities.supports_color and self.capabilities.supports_unicode:
            self.console.print(rich_content)
        else:
            # Fallback to plain text
            print(plain_text)

    def get_terminal_width(self) -> int:
        """Get current terminal width."""
        return self.capabilities.width

    def get_terminal_height(self) -> int:
        """Get current terminal height."""
        return self.capabilities.height

    def supports_emoji(self) -> bool:
        """Check if terminal supports emoji."""
        return self.capabilities.supports_emoji

    def supports_color(self) -> bool:
        """Check if terminal supports color."""
        return self.capabilities.supports_color


# Utility functions for easy use


def create_renderer(auto_detect: bool = True) -> RichTextRenderer:
    """Create a rich text renderer with auto-detection."""
    return RichTextRenderer(auto_detect=auto_detect)


def render_risk_badge(
    risk_level: Union[int, str], renderer: Optional[RichTextRenderer] = None
) -> Text:
    """Quick function to render a risk badge."""
    renderer = renderer or create_renderer()
    return renderer.render_risk_indicator(risk_level, show_bar=False)


def render_operation_badge(
    operation_type: str, renderer: Optional[RichTextRenderer] = None
) -> Text:
    """Quick function to render an operation type badge."""
    renderer = renderer or create_renderer()

    text = Text()
    text.append("⚡ ", style=renderer.get_operation_color(operation_type))
    text.append(
        operation_type.replace("_", " ").title(),
        style=renderer.get_operation_color(operation_type),
    )

    return text
