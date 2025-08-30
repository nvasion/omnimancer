"""
Enhanced Diff Display and Code Highlighting for Omnimancer Approval Interface.

This module provides sophisticated diff visualization with syntax highlighting,
side-by-side comparisons, and enhanced formatting for code changes in the
approval workflow.
"""

import difflib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.box import ROUNDED, SIMPLE
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .rich_renderer import RichTextRenderer, create_renderer


class DiffType(Enum):
    """Types of diff display formats."""

    UNIFIED = "unified"
    SIDE_BY_SIDE = "side_by_side"
    INLINE = "inline"
    CONTEXT = "context"


class FileChangeType(Enum):
    """Types of file changes."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"


@dataclass
class FileChange:
    """Represents a single file change in a diff."""

    file_path: str
    change_type: FileChangeType
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    diff_text: Optional[str] = None
    language: Optional[str] = None
    lines_added: int = 0
    lines_removed: int = 0


@dataclass
class DiffChunk:
    """Represents a chunk of changes within a file."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    context_before: List[str]
    changes: List[Tuple[str, str]]  # (change_type, line_content)
    context_after: List[str]


class EnhancedDiffRenderer:
    """
    Enhanced diff renderer with syntax highlighting and multiple display formats.

    Provides sophisticated diff visualization capabilities for the approval interface,
    including side-by-side comparisons, syntax highlighting, and file tree views.
    """

    def __init__(
        self,
        renderer: Optional[RichTextRenderer] = None,
        console: Optional[Console] = None,
    ):
        """
        Initialize the enhanced diff renderer.

        Args:
            renderer: Rich text renderer instance
            console: Rich console instance
        """
        self.renderer = renderer or create_renderer()
        self.console = console or self.renderer.console

        # Diff rendering configuration
        self.max_line_length = 120
        self.context_lines = 3
        self.show_line_numbers = True
        self.highlight_syntax = True
        self.word_level_diff = True

        # Language detection patterns
        self.language_patterns = {
            r"\.py$": "python",
            r"\.js$": "javascript",
            r"\.ts$": "typescript",
            r"\.jsx$": "jsx",
            r"\.tsx$": "tsx",
            r"\.json$": "json",
            r"\.yaml$|\.yml$": "yaml",
            r"\.md$": "markdown",
            r"\.sh$": "bash",
            r"\.sql$": "sql",
            r"\.css$": "css",
            r"\.html$": "html",
            r"\.xml$": "xml",
            r"\.go$": "go",
            r"\.rs$": "rust",
            r"\.java$": "java",
            r"\.cpp$|\.cc$|\.cxx$": "cpp",
            r"\.c$": "c",
            r"\.h$": "c",
            r"\.php$": "php",
            r"\.rb$": "ruby",
        }

    def detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect programming language from file path.

        Args:
            file_path: Path to the file

        Returns:
            Detected language or None
        """
        for pattern, language in self.language_patterns.items():
            if re.search(pattern, file_path, re.IGNORECASE):
                return language
        return None

    def parse_unified_diff(self, diff_text: str) -> List[FileChange]:
        """
        Parse unified diff format into structured file changes.

        Args:
            diff_text: Raw unified diff text

        Returns:
            List of parsed file changes
        """
        file_changes = []
        current_change = None

        lines = diff_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # File header detection
            if line.startswith("diff --git"):
                if current_change:
                    file_changes.append(current_change)

                # Parse git diff header
                parts = line.split()
                if len(parts) >= 4:
                    old_file = parts[2][2:]  # Remove 'a/' prefix
                    new_file = parts[3][2:]  # Remove 'b/' prefix

                    current_change = FileChange(
                        file_path=new_file,
                        change_type=FileChangeType.MODIFIED,
                        old_path=old_file,
                        new_path=new_file,
                        language=self.detect_language(new_file),
                    )

            elif line.startswith("new file mode"):
                if current_change:
                    current_change.change_type = FileChangeType.ADDED

            elif line.startswith("deleted file mode"):
                if current_change:
                    current_change.change_type = FileChangeType.DELETED

            elif line.startswith("rename from"):
                if current_change:
                    current_change.change_type = FileChangeType.RENAMED
                    current_change.old_path = line[12:]  # Remove 'rename from '

            elif line.startswith("rename to"):
                if current_change:
                    current_change.new_path = line[10:]  # Remove 'rename to '
                    current_change.file_path = current_change.new_path

            elif line.startswith("@@"):
                # Hunk header - extract line numbers and changes
                if current_change:
                    # Collect diff content for this hunk
                    diff_lines = [line]
                    i += 1

                    while (
                        i < len(lines)
                        and not lines[i].startswith("@@")
                        and not lines[i].startswith("diff")
                    ):
                        diff_line = lines[i]
                        diff_lines.append(diff_line)

                        # Count additions and deletions
                        if diff_line.startswith("+") and not diff_line.startswith(
                            "+++"
                        ):
                            current_change.lines_added += 1
                        elif diff_line.startswith("-") and not diff_line.startswith(
                            "---"
                        ):
                            current_change.lines_removed += 1

                        i += 1

                    # Store diff text for this file
                    if current_change.diff_text is None:
                        current_change.diff_text = "\n".join(diff_lines)
                    else:
                        current_change.diff_text += "\n" + "\n".join(diff_lines)

                    i -= 1  # Back up one line since we'll increment at end of loop

            i += 1

        # Add the last file change
        if current_change:
            file_changes.append(current_change)

        return file_changes

    def render_file_tree(self, file_changes: List[FileChange]) -> Tree:
        """
        Render a tree view of all changed files.

        Args:
            file_changes: List of file changes to display

        Returns:
            Rich Tree object with file changes
        """
        tree = Tree("📁 Changed Files", style="bold blue")

        # Group files by directory
        file_groups: Dict[str, List[FileChange]] = {}

        for change in file_changes:
            directory = str(Path(change.file_path).parent)
            if directory == ".":
                directory = "root"

            if directory not in file_groups:
                file_groups[directory] = []
            file_groups[directory].append(change)

        # Add each directory and its files
        for directory, changes in sorted(file_groups.items()):
            if len(file_groups) > 1:
                dir_node = tree.add(f"📂 {directory}", style="cyan")
            else:
                dir_node = tree

            for change in sorted(changes, key=lambda x: x.file_path):
                # Choose icon based on change type
                icon_map = {
                    FileChangeType.ADDED: "🆕",
                    FileChangeType.MODIFIED: "📝",
                    FileChangeType.DELETED: "🗑️",
                    FileChangeType.RENAMED: "🔄",
                    FileChangeType.COPIED: "📋",
                }

                icon = icon_map.get(change.change_type, "📄")

                # Color based on change type
                color_map = {
                    FileChangeType.ADDED: "green",
                    FileChangeType.MODIFIED: "yellow",
                    FileChangeType.DELETED: "red",
                    FileChangeType.RENAMED: "blue",
                    FileChangeType.COPIED: "cyan",
                }

                color = color_map.get(change.change_type, "white")

                # File name with stats
                file_name = Path(change.file_path).name
                stats = (
                    f"(+{change.lines_added}/-{change.lines_removed})"
                    if change.lines_added or change.lines_removed
                    else ""
                )

                dir_node.add(f"{icon} {file_name} {stats}", style=color)

        return tree

    def render_unified_diff(self, file_change: FileChange) -> Panel:
        """
        Render a unified diff view for a single file.

        Args:
            file_change: File change to render

        Returns:
            Rich Panel containing the diff
        """
        if not file_change.diff_text:
            return Panel("No changes to display", title=file_change.file_path)

        # Apply syntax highlighting and diff coloring
        diff_content = self.renderer.apply_diff_highlighting(
            file_change.diff_text, file_type=file_change.language
        )

        # Create title with change type and stats
        change_type_colors = {
            FileChangeType.ADDED: "green",
            FileChangeType.MODIFIED: "yellow",
            FileChangeType.DELETED: "red",
            FileChangeType.RENAMED: "blue",
            FileChangeType.COPIED: "cyan",
        }

        title_color = change_type_colors.get(file_change.change_type, "white")
        stats = f"(+{file_change.lines_added}/-{file_change.lines_removed})"

        title = Text()
        title.append(file_change.file_path, style=f"bold {title_color}")
        title.append(f" {stats}", style="dim")

        return Panel(diff_content, title=title, border_style=title_color, expand=False)

    def render_side_by_side_diff(self, file_change: FileChange) -> Panel:
        """
        Render a side-by-side diff view for a single file.

        Args:
            file_change: File change to render

        Returns:
            Rich Panel containing the side-by-side diff
        """
        if not file_change.old_content or not file_change.new_content:
            return self.render_unified_diff(file_change)

        # Split content into lines
        old_lines = file_change.old_content.split("\n")
        new_lines = file_change.new_content.split("\n")

        # Create side-by-side comparison
        table = Table(
            show_header=True,
            header_style="bold",
            box=SIMPLE,
            expand=True,
            min_width=self.renderer.get_terminal_width() - 4,
        )

        table.add_column("Old", style="red", width=50)
        table.add_column("New", style="green", width=50)

        # Use difflib for intelligent line matching
        differ = difflib.SequenceMatcher(None, old_lines, new_lines)

        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == "equal":
                # Lines are identical
                for i in range(i2 - i1):
                    old_line = old_lines[i1 + i] if i1 + i < len(old_lines) else ""
                    new_line = new_lines[j1 + i] if j1 + i < len(new_lines) else ""

                    table.add_row(
                        Text(old_line, style="dim"),
                        Text(new_line, style="dim"),
                    )

            elif tag == "delete":
                # Lines deleted from old
                for i in range(i1, i2):
                    table.add_row(
                        Text(old_lines[i], style="red"), Text("", style="dim")
                    )

            elif tag == "insert":
                # Lines added to new
                for j in range(j1, j2):
                    table.add_row(
                        Text("", style="dim"),
                        Text(new_lines[j], style="green"),
                    )

            elif tag == "replace":
                # Lines changed
                old_chunk = old_lines[i1:i2]
                new_chunk = new_lines[j1:j2]

                # Pad shorter chunk with empty lines
                max_len = max(len(old_chunk), len(new_chunk))

                for i in range(max_len):
                    old_line = old_chunk[i] if i < len(old_chunk) else ""
                    new_line = new_chunk[i] if i < len(new_chunk) else ""

                    table.add_row(
                        Text(old_line, style="red" if old_line else "dim"),
                        Text(new_line, style="green" if new_line else "dim"),
                    )

        # Create title
        title = Text()
        title.append(file_change.file_path, style="bold yellow")
        title.append(
            f" (+{file_change.lines_added}/-{file_change.lines_removed})",
            style="dim",
        )

        return Panel(table, title=title, border_style="yellow", expand=False)

    def render_word_diff(self, old_line: str, new_line: str) -> Tuple[Text, Text]:
        """
        Generate word-level diff highlighting for two lines.

        Args:
            old_line: Original line content
            new_line: New line content

        Returns:
            Tuple of (old_formatted, new_formatted) Text objects
        """
        if not self.word_level_diff:
            return Text(old_line), Text(new_line)

        # Split into words for better diffing
        old_words = re.findall(r"\S+|\s+", old_line)
        new_words = re.findall(r"\S+|\s+", new_line)

        differ = difflib.SequenceMatcher(None, old_words, new_words)

        old_formatted = Text()
        new_formatted = Text()

        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == "equal":
                # Words are identical
                for i in range(i2 - i1):
                    word = old_words[i1 + i]
                    old_formatted.append(word)
                    new_formatted.append(word)

            elif tag == "delete":
                # Words deleted from old
                deleted_text = "".join(old_words[i1:i2])
                old_formatted.append(deleted_text, style="bold red on white")

            elif tag == "insert":
                # Words added to new
                added_text = "".join(new_words[j1:j2])
                new_formatted.append(added_text, style="bold green on white")

            elif tag == "replace":
                # Words changed
                old_text = "".join(old_words[i1:i2])
                new_text = "".join(new_words[j1:j2])

                old_formatted.append(old_text, style="bold red on white")
                new_formatted.append(new_text, style="bold green on white")

        return old_formatted, new_formatted

    def render_diff_summary(self, file_changes: List[FileChange]) -> Table:
        """
        Render a summary table of all file changes.

        Args:
            file_changes: List of file changes

        Returns:
            Rich Table with summary information
        """
        table = Table(
            title="📊 Diff Summary",
            show_header=True,
            header_style="bold cyan",
            box=ROUNDED,
        )

        table.add_column("File", style="white", width=40)
        table.add_column("Change", style="yellow", justify="center")
        table.add_column("Added", style="green", justify="right")
        table.add_column("Removed", style="red", justify="right")
        table.add_column("Language", style="blue", justify="center")

        total_added = 0
        total_removed = 0

        for change in file_changes:
            # Change type with icon
            change_icons = {
                FileChangeType.ADDED: "🆕 Added",
                FileChangeType.MODIFIED: "📝 Modified",
                FileChangeType.DELETED: "🗑️ Deleted",
                FileChangeType.RENAMED: "🔄 Renamed",
                FileChangeType.COPIED: "📋 Copied",
            }

            change_text = change_icons.get(change.change_type, "❓ Unknown")

            table.add_row(
                change.file_path,
                change_text,
                str(change.lines_added),
                str(change.lines_removed),
                change.language or "—",
            )

            total_added += change.lines_added
            total_removed += change.lines_removed

        # Add totals row
        table.add_row("—" * 20, "—" * 10, "—" * 5, "—" * 5, "—" * 8, style="dim")

        table.add_row(
            Text("TOTALS", style="bold"),
            f"{len(file_changes)} files",
            Text(str(total_added), style="bold green"),
            Text(str(total_removed), style="bold red"),
            "—",
        )

        return table

    def render_diff_set(
        self,
        diff_text: str,
        diff_type: DiffType = DiffType.UNIFIED,
        show_summary: bool = True,
        show_file_tree: bool = True,
    ) -> List[RenderableType]:
        """
        Render a complete diff set with multiple files.

        Args:
            diff_text: Raw diff text (unified diff format)
            diff_type: Type of diff display to use
            show_summary: Whether to show summary table
            show_file_tree: Whether to show file tree

        Returns:
            List of Rich renderables
        """
        # Parse the diff
        file_changes = self.parse_unified_diff(diff_text)

        if not file_changes:
            return [Panel("No changes detected", style="yellow")]

        renderables = []

        # Add summary table
        if show_summary:
            renderables.append(self.render_diff_summary(file_changes))
            renderables.append(Rule(style="dim"))

        # Add file tree
        if show_file_tree:
            renderables.append(self.render_file_tree(file_changes))
            renderables.append(Rule(style="dim"))

        # Render each file's diff
        for change in file_changes:
            if diff_type == DiffType.SIDE_BY_SIDE:
                renderables.append(self.render_side_by_side_diff(change))
            else:  # Default to unified
                renderables.append(self.render_unified_diff(change))

            renderables.append(Rule(style="dim"))

        return renderables

    def render_file_content_comparison(
        self,
        old_content: str,
        new_content: str,
        file_path: str,
        language: Optional[str] = None,
    ) -> Panel:
        """
        Render a comparison between two complete file contents.

        Args:
            old_content: Original file content
            new_content: New file content
            file_path: Path to the file
            language: Programming language for syntax highlighting

        Returns:
            Rich Panel with file comparison
        """
        # Auto-detect language if not provided
        if not language:
            language = self.detect_language(file_path)

        # Create unified diff
        diff_lines = list(
            difflib.unified_diff(
                old_content.split("\n"),
                new_content.split("\n"),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm="",
            )
        )

        diff_text = "\n".join(diff_lines)

        # Apply syntax highlighting
        highlighted_diff = self.renderer.apply_diff_highlighting(
            diff_text, file_type=language
        )

        return Panel(
            highlighted_diff,
            title=f"📄 {file_path}",
            border_style="blue",
            expand=False,
        )


# Utility functions for easy integration


def create_diff_renderer(
    renderer: Optional[RichTextRenderer] = None,
) -> EnhancedDiffRenderer:
    """Create an enhanced diff renderer with default configuration."""
    return EnhancedDiffRenderer(renderer=renderer)


def render_git_diff(
    diff_output: str,
    diff_type: DiffType = DiffType.UNIFIED,
    console: Optional[Console] = None,
) -> None:
    """
    Quick function to render git diff output.

    Args:
        diff_output: Raw git diff output
        diff_type: Type of diff display
        console: Rich console for output
    """
    renderer = create_diff_renderer()
    console = console or renderer.console

    renderables = renderer.render_diff_set(diff_output, diff_type)

    for renderable in renderables:
        console.print(renderable)


def compare_files(
    old_file: str,
    new_file: str,
    file_path: str,
    console: Optional[Console] = None,
) -> None:
    """
    Compare two file contents and display the diff.

    Args:
        old_file: Path to old file or content string
        new_file: Path to new file or content string
        file_path: Display path for the file
        console: Rich console for output
    """
    renderer = create_diff_renderer()
    console = console or renderer.console

    # Read file contents if they're paths
    old_content = old_file
    new_content = new_file

    if Path(old_file).exists():
        old_content = Path(old_file).read_text()

    if Path(new_file).exists():
        new_content = Path(new_file).read_text()

    comparison = renderer.render_file_content_comparison(
        old_content, new_content, file_path
    )

    console.print(comparison)
