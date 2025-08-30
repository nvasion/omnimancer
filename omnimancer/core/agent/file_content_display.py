"""
Unified File Content Display Module for Omnimancer.

This module integrates existing UI components to provide a cohesive interface
for displaying file content and modifications according to the designed interaction flow.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from omnimancer.cli.approval_formatter import CLIApprovalFormatter
from omnimancer.cli.approval_prompt import CLIApprovalPrompt

from .approval_dialog import ApprovalDialog, DialogOptions
from .diff_renderer import (
    DiffType,
    EnhancedDiffRenderer,
    FileChange,
    FileChangeType,
)
from .rich_renderer import create_renderer

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display modes for file content."""

    FULL_CONTENT = "full"
    PREVIEW = "preview"
    DIFF = "diff"
    SUMMARY = "summary"


@dataclass
class FileDisplayConfig:
    """Configuration for file content display."""

    display_mode: DisplayMode = DisplayMode.FULL_CONTENT
    syntax_highlighting: bool = True
    show_line_numbers: bool = True
    max_preview_lines: int = 100
    max_content_size: int = 1024 * 1024  # 1MB
    truncate_large_files: bool = True
    diff_type: DiffType = DiffType.UNIFIED
    show_risk_assessment: bool = True
    show_file_stats: bool = True


class UnifiedFileContentDisplay:
    """
    Unified interface for displaying file content and modifications.

    Integrates existing UI components to provide consistent file content
    display across all operations (create, modify, delete, batch).
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        config: Optional[FileDisplayConfig] = None,
    ):
        """
        Initialize the unified file content display.

        Args:
            console: Rich console for output
            config: Display configuration
        """
        self.console = console or Console()
        self.config = config or FileDisplayConfig()

        # Initialize component instances
        self.diff_renderer = EnhancedDiffRenderer(console=self.console)
        self.rich_renderer = create_renderer()
        self.unified_approval_ui = CLIApprovalPrompt(console=self.console)
        self.formatter = CLIApprovalFormatter(console=self.console)

        # Dialog options for approval flow
        self.dialog_options = DialogOptions(
            show_diff=True,
            show_risk_assessment=self.config.show_risk_assessment,
            show_file_tree=True,
            show_operation_details=True,
            diff_type=self.config.diff_type,
            syntax_highlighting=self.config.syntax_highlighting,
            show_line_numbers=self.config.show_line_numbers,
        )

        self.approval_dialog = ApprovalDialog(
            renderer=self.rich_renderer,
            diff_renderer=self.diff_renderer,
            console=self.console,
            options=self.dialog_options,
        )

    async def display_file_creation(
        self,
        file_path: str,
        content: str,
        operation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Display interface for new file creation.

        Args:
            file_path: Path where file will be created
            content: Content to be written to the file
            operation_context: Additional context about the operation

        Returns:
            Display result with user decision if interactive
        """
        try:
            # Prepare review data for existing UI
            review_data = {
                "file_path": file_path,
                "file_exists": False,
                "current_content": None,
                "new_content": content,
                "operation": "create",
                "encoding": "utf-8",
            }

            # Add operation context if provided
            if operation_context:
                review_data.update(operation_context)

            # Determine if content should be truncated
            if (
                self.config.truncate_large_files
                and len(content) > self.config.max_content_size
            ):
                review_data["truncated"] = True
                review_data["original_size"] = len(content)
                review_data["new_content"] = content[: self.config.max_content_size]

            # Use unified approval UI for interactive review
            if operation_context and operation_context.get("interactive", True):
                current_model = operation_context.get("current_model", "Omnimancer AI")
                return await self.unified_approval_ui.prompt_for_file_modification_approval(
                    review_data, current_model
                )

            # Non-interactive display
            self._display_file_creation_summary(file_path, content)
            return {"displayed": True, "interactive": False}

        except Exception as e:
            logger.error(f"Error displaying file creation: {e}")
            return {"error": str(e), "displayed": False}

    async def display_file_modification(
        self,
        file_path: str,
        current_content: str,
        new_content: str,
        operation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Display interface for file modification with diff.

        Args:
            file_path: Path to the file being modified
            current_content: Current file content
            new_content: Proposed new content
            operation_context: Additional context about the operation

        Returns:
            Display result with user decision if interactive
        """
        try:
            # Create file change object for diff renderer
            file_change = FileChange(
                file_path=file_path,
                change_type=FileChangeType.MODIFIED,
                old_content=current_content,
                new_content=new_content,
                language=self._detect_language(file_path),
            )

            # Generate diff
            diff_output = self.diff_renderer.render_unified_diff(file_change)

            # Prepare review data
            review_data = {
                "file_path": file_path,
                "file_exists": True,
                "current_content": current_content,
                "new_content": new_content,
                "diff": diff_output,
                "operation": "modify",
                "encoding": "utf-8",
            }

            if operation_context:
                review_data.update(operation_context)

            # Use unified approval UI for interactive review
            if operation_context and operation_context.get("interactive", True):
                current_model = operation_context.get("current_model", "Omnimancer AI")
                return await self.unified_approval_ui.prompt_for_file_modification_approval(
                    review_data, current_model
                )

            # Non-interactive display
            self._display_file_modification_summary(file_path, file_change)
            return {"displayed": True, "interactive": False}

        except Exception as e:
            logger.error(f"Error displaying file modification: {e}")
            return {"error": str(e), "displayed": False}

    async def display_file_deletion(
        self,
        file_path: str,
        content: str,
        operation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Display interface for file deletion.

        Args:
            file_path: Path to the file being deleted
            content: Current content of the file
            operation_context: Additional context about the operation

        Returns:
            Display result with user decision if interactive
        """
        try:
            # Create warning panel for deletion
            warning_text = Text()
            warning_text.append("⚠️  FILE DELETION WARNING\n", style="bold red")
            warning_text.append(f"File: {file_path}\n", style="yellow")
            warning_text.append(f"Size: {len(content)} bytes\n", style="dim")
            warning_text.append("This operation cannot be undone!", style="bold red")

            warning_panel = Panel(
                warning_text,
                title="[bold red]Confirm Deletion[/bold red]",
                border_style="red",
            )

            self.console.print(warning_panel)

            # Show file content preview
            if self.config.display_mode == DisplayMode.PREVIEW:
                preview_lines = min(
                    self.config.max_preview_lines, len(content.splitlines())
                )
                preview_content = "\n".join(content.splitlines()[:preview_lines])
                if preview_lines < len(content.splitlines()):
                    preview_content += f"\n... ({len(content.splitlines()) - preview_lines} more lines)"
            else:
                preview_content = content

            # Display content with syntax highlighting
            language = self._detect_language(file_path)
            syntax = Syntax(
                preview_content,
                language,
                theme="monokai",
                line_numbers=self.config.show_line_numbers,
            )

            content_panel = Panel(
                syntax,
                title=f"[red]Content to be deleted[/red]",
                border_style="red",
            )

            self.console.print(content_panel)

            # Interactive confirmation if needed
            if operation_context and operation_context.get("interactive", True):
                from rich.prompt import Confirm

                confirmed = Confirm.ask(
                    "Are you sure you want to delete this file?", default=False
                )
                return {"approved": confirmed, "interactive": True}

            return {"displayed": True, "interactive": False}

        except Exception as e:
            logger.error(f"Error displaying file deletion: {e}")
            return {"error": str(e), "displayed": False}

    def display_batch_operations(
        self,
        operations: List[Dict[str, Any]],
        operation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Display interface for batch file operations.

        Args:
            operations: List of file operations to display
            operation_context: Additional context about the operations

        Returns:
            Display result with batch summary
        """
        try:
            # Create summary table
            summary_table = Table(title="Batch File Operations Summary")
            summary_table.add_column("Type", style="cyan")
            summary_table.add_column("Count", style="magenta")
            summary_table.add_column("Risk Level", style="yellow")

            # Count operations by type
            op_counts = {}
            risk_levels = {}

            for op in operations:
                op_type = op.get("type", "unknown")
                op_counts[op_type] = op_counts.get(op_type, 0) + 1

                risk = op.get("risk_level", "low")
                if risk not in risk_levels:
                    risk_levels[risk] = []
                risk_levels[risk].append(op)

            # Add rows to summary table
            for op_type, count in op_counts.items():
                risk_summary = self._get_risk_summary(operations, op_type)
                summary_table.add_row(op_type.title(), str(count), risk_summary)

            self.console.print(summary_table)

            # Show detailed list if not too many operations
            if len(operations) <= 10:
                self._display_operation_list(operations)
            else:
                self.console.print(
                    f"\n[dim]Showing first 10 of {len(operations)} operations...[/dim]"
                )
                self._display_operation_list(operations[:10])

            return {
                "displayed": True,
                "operation_count": len(operations),
                "types": list(op_counts.keys()),
            }

        except Exception as e:
            logger.error(f"Error displaying batch operations: {e}")
            return {"error": str(e), "displayed": False}

    def _display_file_creation_summary(self, file_path: str, content: str):
        """Display non-interactive summary of file creation."""
        info_text = Text()
        info_text.append("📄 Creating new file\n", style="bold green")
        info_text.append(f"Path: {file_path}\n", style="cyan")
        info_text.append(f"Size: {len(content)} bytes\n", style="dim")
        info_text.append(f"Lines: {len(content.splitlines())}\n", style="dim")

        panel = Panel(info_text, title="File Creation", border_style="green")
        self.console.print(panel)

    def _display_file_modification_summary(
        self, file_path: str, file_change: FileChange
    ):
        """Display non-interactive summary of file modification."""
        info_text = Text()
        info_text.append("✏️  Modifying file\n", style="bold yellow")
        info_text.append(f"Path: {file_path}\n", style="cyan")

        if file_change.lines_added or file_change.lines_removed:
            info_text.append(
                f"Lines added: +{file_change.lines_added}\n", style="green"
            )
            info_text.append(
                f"Lines removed: -{file_change.lines_removed}\n", style="red"
            )

        panel = Panel(info_text, title="File Modification", border_style="yellow")
        self.console.print(panel)

    def _display_operation_list(self, operations: List[Dict[str, Any]]):
        """Display a list of operations."""
        for i, op in enumerate(operations, 1):
            op_type = op.get("type", "unknown")
            file_path = op.get("file_path", "N/A")
            risk = op.get("risk_level", "low")

            # Choose icon based on operation type
            icon = self._get_operation_icon(op_type)
            risk_color = self._get_risk_color(risk)

            self.console.print(
                f"{i}. {icon} [{risk_color}]{op_type.title()}[/{risk_color}]: {file_path}"
            )

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".m": "matlab",
            ".jl": "julia",
            ".sh": "bash",
            ".ps1": "powershell",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".less": "less",
            ".sql": "sql",
            ".md": "markdown",
            ".rst": "rst",
            ".tex": "latex",
        }

        path = Path(file_path)
        extension = path.suffix.lower()
        return extension_map.get(extension, "text")

    def _get_operation_icon(self, op_type: str) -> str:
        """Get icon for operation type."""
        icons = {
            "create": "📄",
            "modify": "✏️",
            "delete": "🗑️",
            "move": "📦",
            "copy": "📋",
            "unknown": "❓",
        }
        return icons.get(op_type.lower(), "❓")

    def _get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level."""
        colors = {
            "low": "green",
            "medium": "yellow",
            "high": "orange1",
            "critical": "red",
        }
        return colors.get(risk_level.lower(), "white")

    def _get_risk_summary(self, operations: List[Dict[str, Any]], op_type: str) -> str:
        """Get risk summary for operations of a specific type."""
        type_ops = [op for op in operations if op.get("type") == op_type]
        if not type_ops:
            return "N/A"

        risk_counts = {}
        for op in type_ops:
            risk = op.get("risk_level", "low")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        # Format risk summary
        summary_parts = []
        for risk in ["critical", "high", "medium", "low"]:
            if risk in risk_counts:
                color = self._get_risk_color(risk)
                summary_parts.append(f"[{color}]{risk}: {risk_counts[risk]}[/{color}]")

        return ", ".join(summary_parts) if summary_parts else "Unknown"


# Factory function for easy instantiation
def create_file_content_display(
    console: Optional[Console] = None,
    config: Optional[FileDisplayConfig] = None,
) -> UnifiedFileContentDisplay:
    """
    Create a unified file content display instance.

    Args:
        console: Optional Rich console
        config: Optional display configuration

    Returns:
        UnifiedFileContentDisplay instance
    """
    return UnifiedFileContentDisplay(console=console, config=config)
