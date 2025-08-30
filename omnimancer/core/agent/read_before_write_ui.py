"""
User interface for read-before-write functionality.

This module provides UI components for presenting file modification previews
to users and collecting their approval/modification decisions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ...ui.cancellation_handler import get_active_cancellation_handler

logger = logging.getLogger(__name__)


class ReadBeforeWriteUI:
    """
    User interface for read-before-write operations.

    Provides rich terminal UI for displaying file modification previews,
    diffs, and collecting user decisions.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the read-before-write UI.

        Args:
            console: Optional Rich console instance
        """
        self.console = console or Console()

    async def review_file_modification(
        self, review_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Present file modification review to user and collect decision.

        Args:
            review_data: Dictionary containing:
                - file_path: Path to the file
                - file_exists: Whether file currently exists
                - current_content: Current file content (if exists)
                - new_content: Proposed new content
                - diff: Unified diff between current and new content
                - operation: 'create' or 'modify'
                - encoding: File encoding

        Returns:
            Dictionary with user decision:
                - approved: Boolean indicating if change is approved
                - modified_content: Optional modified content from user
                - reason: Optional reason for decision
        """
        try:
            # Ensure we start on a new line, clearing any spinner artifacts
            self.console.print()

            # Display file modification preview
            self._display_modification_header(review_data)

            # Show content preview based on operation type
            if review_data["operation"] == "create":
                self._display_new_file_preview(review_data)
            else:
                self._display_modification_preview(review_data)

            # Get user decision
            decision = await self._get_user_decision(review_data)

            return decision

        except Exception as e:
            logger.error(f"Error in file modification review: {e}")
            return {"approved": False, "reason": f"UI error: {str(e)}"}

    def _display_modification_header(self, review_data: Dict[str, Any]):
        """Display header information about the file modification."""
        file_path = review_data["file_path"]
        operation = review_data["operation"]

        # Create header panel
        header_text = Text()
        header_text.append("File Modification Review\n", style="bold cyan")
        header_text.append(f"Operation: ", style="bold")
        header_text.append(f"{operation.title()}\n", style="yellow")
        header_text.append(f"File: ", style="bold")
        header_text.append(f"{file_path}", style="green")

        header_panel = Panel(
            header_text,
            title="[bold blue]Read-Before-Write Review[/bold blue]",
            border_style="blue",
        )

        self.console.print(header_panel)

    def _display_new_file_preview(self, review_data: Dict[str, Any]):
        """Display preview for new file creation."""
        new_content = review_data["new_content"]
        file_path = review_data["file_path"]

        # Determine file type for syntax highlighting
        file_extension = Path(file_path).suffix.lower()
        lexer = self._get_lexer_for_extension(file_extension)

        # Show new content with syntax highlighting
        if len(new_content) > 2000:  # Truncate very long content
            preview_content = new_content[:2000] + "\n... (content truncated)"
        else:
            preview_content = new_content

        syntax = Syntax(
            preview_content,
            lexer,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )

        content_panel = Panel(
            syntax,
            title="[bold green]New File Content[/bold green]",
            border_style="green",
        )

        self.console.print(content_panel)

        # Show file stats
        self._display_content_stats(new_content, is_new=True)

    def _display_modification_preview(self, review_data: Dict[str, Any]):
        """Display preview for file modification with diff."""
        current_content = review_data.get("current_content", "")
        new_content = review_data["new_content"]
        diff = review_data.get("diff", "")

        # Display current content summary
        if current_content:
            self._display_current_content_summary(current_content)

        # Display diff if available
        if diff:
            self._display_diff(diff)
        else:
            # Fallback to side-by-side preview for very different content
            self._display_side_by_side_preview(current_content, new_content)

        # Show content statistics
        self._display_content_comparison_stats(current_content, new_content)

    def _display_current_content_summary(self, current_content: str):
        """Display summary of current file content."""
        lines = current_content.split("\n")
        total_lines = len(lines)

        # Show first few and last few lines
        preview_lines = []
        if total_lines <= 20:
            preview_lines = lines
        else:
            preview_lines = (
                lines[:10]
                + [f"... ({total_lines - 20} lines omitted) ..."]
                + lines[-10:]
            )

        preview_text = "\n".join(preview_lines)

        current_panel = Panel(
            preview_text,
            title="[bold yellow]Current File Content (Preview)[/bold yellow]",
            border_style="yellow",
        )

        self.console.print(current_panel)

    def _display_diff(self, diff: str):
        """Display unified diff with syntax highlighting."""
        # Color code the diff
        diff_text = Text()

        for line in diff.split("\n"):
            if line.startswith("+++") or line.startswith("---"):
                diff_text.append(line + "\n", style="bold white")
            elif line.startswith("@@"):
                diff_text.append(line + "\n", style="bold cyan")
            elif line.startswith("+"):
                diff_text.append(line + "\n", style="bold green")
            elif line.startswith("-"):
                diff_text.append(line + "\n", style="bold red")
            else:
                diff_text.append(line + "\n", style="white")

        diff_panel = Panel(
            diff_text,
            title="[bold magenta]Changes (Unified Diff)[/bold magenta]",
            border_style="magenta",
        )

        self.console.print(diff_panel)

    def _display_side_by_side_preview(self, current_content: str, new_content: str):
        """Display side-by-side preview when diff is not available."""
        # Create table for side-by-side view
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Current Content", style="yellow", width=50)
        table.add_column("New Content", style="green", width=50)

        # Truncate content for display
        current_preview = (
            current_content[:1000] + "..."
            if len(current_content) > 1000
            else current_content
        )
        new_preview = (
            new_content[:1000] + "..." if len(new_content) > 1000 else new_content
        )

        table.add_row(current_preview, new_preview)

        preview_panel = Panel(
            table,
            title="[bold magenta]Content Comparison[/bold magenta]",
            border_style="magenta",
        )

        self.console.print(preview_panel)

    def _display_content_stats(self, content: str, is_new: bool = False):
        """Display content statistics."""
        lines = content.split("\n")
        chars = len(content)
        words = len(content.split())

        stats_text = Text()
        stats_text.append("Content Statistics:\n", style="bold")
        stats_text.append(f"Lines: {len(lines)}\n")
        stats_text.append(f"Characters: {chars}\n")
        stats_text.append(f"Words: {words}")

        title = (
            "[bold blue]New File Statistics[/bold blue]"
            if is_new
            else "[bold blue]Content Statistics[/bold blue]"
        )

        stats_panel = Panel(stats_text, title=title, border_style="blue", width=40)

        self.console.print(stats_panel)

    def _display_content_comparison_stats(self, current_content: str, new_content: str):
        """Display comparison statistics between current and new content."""
        current_lines = len(current_content.split("\n")) if current_content else 0
        new_lines = len(new_content.split("\n"))
        current_chars = len(current_content) if current_content else 0
        new_chars = len(new_content)

        # Create comparison table
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="yellow")
        table.add_column("New", style="green")
        table.add_column("Change", style="magenta")

        line_change = new_lines - current_lines
        char_change = new_chars - current_chars

        table.add_row(
            "Lines",
            str(current_lines),
            str(new_lines),
            f"{'+' if line_change >= 0 else ''}{line_change}",
        )
        table.add_row(
            "Characters",
            str(current_chars),
            str(new_chars),
            f"{'+' if char_change >= 0 else ''}{char_change}",
        )

        stats_panel = Panel(
            table,
            title="[bold blue]Content Comparison[/bold blue]",
            border_style="blue",
        )

        self.console.print(stats_panel)

    async def _get_user_decision(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get user decision about the file modification."""
        self.console.print()  # Add some spacing

        # Present options to user
        options_text = Text()
        options_text.append("Options:\n", style="bold cyan")
        options_text.append("1. ", style="bold green")
        options_text.append("Approve changes\n", style="green")
        options_text.append("2. ", style="bold yellow")
        options_text.append("Edit content before writing\n", style="yellow")
        options_text.append("3. ", style="bold red")
        options_text.append("Reject changes\n", style="red")

        options_panel = Panel(
            options_text,
            title="[bold cyan]Choose Action[/bold cyan]",
            border_style="cyan",
        )

        self.console.print(options_panel)

        # Pause the spinner during user interaction
        cancellation_handler = get_active_cancellation_handler()
        if cancellation_handler:
            cancellation_handler.pause_status_display()

        try:
            # Get user choice with Rich prompt (now that spinner is paused)
            choice = Prompt.ask(
                "[bold cyan]Your choice[/bold cyan]",
                choices=["1", "2", "3", "approve", "edit", "reject"],
                default="1",
            )
        except (EOFError, KeyboardInterrupt):
            return {"approved": False, "reason": "Operation cancelled by user"}
        finally:
            # Resume the spinner after user interaction
            if cancellation_handler:
                cancellation_handler.resume_status_display()

        # Process the user's choice
        if choice in ["1", "approve"]:
            return {"approved": True, "reason": "User approved changes"}
        elif choice in ["2", "edit"]:
            return await self._handle_edit_content(review_data)
        elif choice in ["3", "reject"]:
            reason = Prompt.ask(
                "[yellow]Reason for rejection (optional)[/yellow]",
                default="User rejected changes",
            )
            return {"approved": False, "reason": reason}

    async def _handle_edit_content(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user editing of content before writing."""
        self.console.print("\n[yellow]Content editing functionality:[/yellow]")
        self.console.print("You can:")
        self.console.print("1. Provide replacement content inline")
        self.console.print("2. Open external editor (if configured)")
        self.console.print("3. Go back to original content")

        edit_choice = Prompt.ask(
            "Edit method",
            choices=["1", "2", "3", "inline", "editor", "back"],
            default="1",
        )

        if edit_choice in ["1", "inline"]:
            return await self._inline_content_edit(review_data)
        elif edit_choice in ["2", "editor"]:
            return await self._external_editor_edit(review_data)
        else:
            # Go back to original decision
            return await self._get_user_decision(review_data)

    async def _inline_content_edit(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle inline content editing."""
        self.console.print("\n[yellow]Inline content editing:[/yellow]")
        self.console.print(
            "Enter new content (type END_CONTENT on a new line to finish):"
        )

        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "END_CONTENT":
                    break
                lines.append(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                self.console.print("\n[red]Edit cancelled[/red]")
                return {"approved": False, "reason": "Edit cancelled by user"}

        modified_content = "\n".join(lines)

        # Confirm the edit
        self.console.print(
            f"\n[cyan]Modified content ({len(modified_content)} characters):[/cyan]"
        )
        preview = (
            modified_content[:500] + "..."
            if len(modified_content) > 500
            else modified_content
        )
        self.console.print(Panel(preview, title="Modified Content Preview"))

        confirm = Confirm.ask("Use this modified content?", default=True)

        if confirm:
            return {
                "approved": True,
                "modified_content": modified_content,
                "reason": "User provided modified content",
            }
        else:
            return await self._get_user_decision(review_data)

    async def _external_editor_edit(
        self, review_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle external editor content editing."""
        self.console.print(
            "[yellow]External editor functionality not yet implemented[/yellow]"
        )
        self.console.print("[yellow]Falling back to inline editing...[/yellow]")
        return await self._inline_content_edit(review_data)

    async def confirm_file_overwrite(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prompt user for confirmation when a file already exists.

        Args:
            file_info: Dictionary containing file information from check_file_exists():
                - path: File path
                - exists: Whether file exists
                - is_file: Whether it's a file
                - is_directory: Whether it's a directory
                - size: File size in bytes (if applicable)
                - modified_time: Last modified time

        Returns:
            Dictionary with user decision:
                - confirmed: Boolean indicating if user confirmed overwrite
                - action: 'overwrite', 'cancel', or 'backup'
                - reason: Optional reason for decision
        """
        try:
            if not file_info.get("exists", False):
                # File doesn't exist, no confirmation needed
                return {
                    "confirmed": True,
                    "action": "create",
                    "reason": "File does not exist, proceeding with creation",
                }

            # Ensure we start on a new line, clearing any spinner artifacts
            self.console.print()

            # Display file existence warning
            self._display_file_exists_warning(file_info)

            # Get user confirmation
            return await self._get_overwrite_decision(file_info)

        except Exception as e:
            logger.error(f"Error in file overwrite confirmation: {e}")
            return {
                "confirmed": False,
                "action": "cancel",
                "reason": f"Confirmation error: {str(e)}",
            }

    def _display_file_exists_warning(self, file_info: Dict[str, Any]):
        """Display warning about existing file."""
        file_path = file_info["path"]
        file_type = "directory" if file_info.get("is_directory") else "file"

        # Create warning header
        warning_text = Text()
        warning_text.append("⚠️  File Exists Warning\n", style="bold yellow")
        warning_text.append(f"The {file_type} already exists:\n", style="yellow")
        warning_text.append(f"{file_path}\n\n", style="cyan")

        # Add file details if available
        if file_info.get("size") is not None:
            size_mb = file_info["size"] / (1024 * 1024)
            if size_mb >= 1:
                warning_text.append(f"Size: {size_mb:.2f} MB\n", style="white")
            else:
                size_kb = file_info["size"] / 1024
                warning_text.append(f"Size: {size_kb:.2f} KB\n", style="white")

        if file_info.get("modified_time"):
            warning_text.append(
                f"Last modified: {file_info['modified_time']}\n", style="white"
            )

        if file_info.get("is_symlink"):
            warning_text.append("Type: Symbolic link\n", style="magenta")

        warning_panel = Panel(
            warning_text,
            title="[bold red]File Already Exists[/bold red]",
            border_style="red",
        )

        self.console.print(warning_panel)

    async def _get_overwrite_decision(
        self, file_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get user decision about overwriting existing file."""
        self.console.print()  # Add some spacing

        # Present options to user
        options_text = Text()
        options_text.append("What would you like to do?\n\n", style="bold cyan")
        options_text.append("1. ", style="bold green")
        options_text.append("Overwrite the existing file\n", style="green")
        options_text.append("2. ", style="bold yellow")
        options_text.append("Create backup and overwrite\n", style="yellow")
        options_text.append("3. ", style="bold red")
        options_text.append("Cancel operation\n", style="red")

        options_panel = Panel(
            options_text,
            title="[bold cyan]Choose Action[/bold cyan]",
            border_style="cyan",
        )

        self.console.print(options_panel)

        # Pause the spinner during user interaction
        cancellation_handler = get_active_cancellation_handler()
        if cancellation_handler:
            cancellation_handler.pause_status_display()

        try:
            # Get user choice with Rich prompt (now that spinner is paused)
            choice = Prompt.ask(
                "[bold cyan]Your choice[/bold cyan]",
                choices=["1", "2", "3", "overwrite", "backup", "cancel"],
                default="3",  # Default to cancel for safety
            )
        except (EOFError, KeyboardInterrupt):
            return {
                "confirmed": False,
                "action": "cancel",
                "reason": "Operation cancelled by user",
            }
        finally:
            # Resume the spinner after user interaction
            if cancellation_handler:
                cancellation_handler.resume_status_display()

        # Process the user's choice
        if choice in ["1", "overwrite"]:
            confirm = Confirm.ask(
                "[yellow]Are you sure you want to overwrite the existing file?[/yellow]",
                default=False,
            )
            if confirm:
                return {
                    "confirmed": True,
                    "action": "overwrite",
                    "reason": "User confirmed overwrite",
                }
            else:
                return await self._get_overwrite_decision(file_info)  # Ask again

        elif choice in ["2", "backup"]:
            return {
                "confirmed": True,
                "action": "backup",
                "reason": "User chose to create backup before overwriting",
            }

        elif choice in ["3", "cancel"]:
            reason = Prompt.ask(
                "[yellow]Reason for cancellation (optional)[/yellow]",
                default="User chose to cancel operation",
            )
            return {"confirmed": False, "action": "cancel", "reason": reason}

    def _get_lexer_for_extension(self, extension: str) -> str:
        """Get appropriate lexer for file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".md": "markdown",
            ".sh": "bash",
            ".sql": "sql",
            ".cpp": "cpp",
            ".c": "c",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
        }

        return extension_map.get(extension, "text")


# Convenience functions for creating UI callbacks
def create_review_callback(console: Optional[Console] = None):
    """
    Create a review callback function for use with FileSystemManager.

    Args:
        console: Optional Rich console instance

    Returns:
        Async callback function for file modification review
    """
    ui = ReadBeforeWriteUI(console)

    async def review_callback(review_data: Dict[str, Any]) -> Dict[str, Any]:
        return await ui.review_file_modification(review_data)

    return review_callback


def create_confirmation_callback(console: Optional[Console] = None):
    """
    Create a confirmation callback function for file existence checks.

    Args:
        console: Optional Rich console instance

    Returns:
        Async callback function for file overwrite confirmation
    """
    ui = ReadBeforeWriteUI(console)

    async def confirmation_callback(
        file_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await ui.confirm_file_overwrite(file_info)

    return confirmation_callback
