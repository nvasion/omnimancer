"""
Proposed Changes Display Integration Module.

This module integrates the display of proposed changes with the file system manager
and approval workflow, providing a seamless interface for showing file modifications.
"""

import difflib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..security.approval_workflow import RiskLevel
from .approval_manager import (
    ChangeType,
    EnhancedApprovalManager,
)
from .diff_renderer import (
    DiffType,
    EnhancedDiffRenderer,
    FileChange,
    FileChangeType,
)
from .file_content_display import (
    DisplayMode,
    FileDisplayConfig,
    UnifiedFileContentDisplay,
)
from .file_system_manager import FileSystemManager

logger = logging.getLogger(__name__)


class ChangeDisplayMode(Enum):
    """Display modes for proposed changes."""

    INLINE = "inline"  # Changes shown inline with markers
    SIDE_BY_SIDE = "side_by_side"  # Original and modified side by side
    UNIFIED = "unified"  # Unified diff format
    OVERLAY = "overlay"  # Changes overlaid with transparency
    SUMMARY = "summary"  # Summary of changes only


@dataclass
class ProposedChange:
    """Represents a proposed change to a file."""

    file_path: str
    operation_type: ChangeType
    original_content: Optional[str] = None
    modified_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    change_summary: Optional[str] = None
    line_changes: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ChangeSet:
    """Collection of related proposed changes."""

    id: str
    description: str
    changes: List[ProposedChange]
    total_risk_score: float = 0.0
    approved: bool = False
    applied: bool = False
    created_at: datetime = field(default_factory=datetime.now)


class ProposedChangesIntegration:
    """
    Integrates proposed changes display with file system operations.

    Provides a cohesive interface for fetching, displaying, and managing
    proposed changes with clear visual distinction from original content.
    """

    def __init__(
        self,
        file_system_manager: Optional[FileSystemManager] = None,
        approval_manager: Optional[EnhancedApprovalManager] = None,
        console: Optional[Console] = None,
    ):
        """
        Initialize the proposed changes integration.

        Args:
            file_system_manager: File system manager instance
            approval_manager: Approval manager instance
            console: Rich console for output
        """
        self.console = console or Console()
        self.file_system_manager = file_system_manager or FileSystemManager()
        self.approval_manager = approval_manager or EnhancedApprovalManager()

        # Initialize display components
        self.display_config = FileDisplayConfig(
            display_mode=DisplayMode.DIFF,
            diff_type=DiffType.UNIFIED,
            syntax_highlighting=True,
            show_line_numbers=True,
        )

        self.unified_display = UnifiedFileContentDisplay(
            console=self.console, config=self.display_config
        )

        self.diff_renderer = EnhancedDiffRenderer(console=self.console)

        # Change tracking
        self.pending_changes: Dict[str, ChangeSet] = {}
        self.change_history: List[ChangeSet] = []

        # Display preferences
        self.color_scheme = {
            "added": "green",
            "removed": "red",
            "modified": "yellow",
            "unchanged": "dim white",
            "context": "bright_black",
        }

    async def fetch_proposed_changes(
        self, operation_id: str, file_paths: Optional[List[str]] = None
    ) -> ChangeSet:
        """
        Fetch proposed changes for display.

        Args:
            operation_id: ID of the operation with proposed changes
            file_paths: Optional list of specific file paths to fetch

        Returns:
            ChangeSet containing all proposed changes
        """
        try:
            changes = []

            # Get operations from approval manager
            if hasattr(self.approval_manager, "get_pending_operations"):
                operations = await self.approval_manager.get_pending_operations(
                    operation_id
                )
            else:
                operations = []

            for op in operations:
                if file_paths and op.get("file_path") not in file_paths:
                    continue

                # Create ProposedChange from operation
                change = await self._create_proposed_change(op)
                changes.append(change)

            # Create change set
            change_set = ChangeSet(
                id=operation_id,
                description=f"Changes for operation {operation_id}",
                changes=changes,
            )

            # Calculate total risk
            change_set.total_risk_score = self._calculate_total_risk(changes)

            # Store in pending changes
            self.pending_changes[operation_id] = change_set

            return change_set

        except Exception as e:
            logger.error(f"Error fetching proposed changes: {e}")
            return ChangeSet(id=operation_id, description="Error", changes=[])

    async def display_proposed_changes(
        self,
        change_set: ChangeSet,
        display_mode: ChangeDisplayMode = ChangeDisplayMode.UNIFIED,
        interactive: bool = True,
    ) -> Dict[str, Any]:
        """
        Display proposed changes with clear visual distinction.

        Args:
            change_set: Set of changes to display
            display_mode: How to display the changes
            interactive: Whether to allow user interaction

        Returns:
            Display result with user decisions if interactive
        """
        try:
            # Display header
            self._display_change_set_header(change_set)

            # Display each change based on mode
            if display_mode == ChangeDisplayMode.SUMMARY:
                self._display_change_summary(change_set)
            else:
                for change in change_set.changes:
                    await self._display_single_change(change, display_mode)

            # Get user decision if interactive
            if interactive:
                decision = await self._get_change_approval(change_set)
                return decision

            return {"displayed": True, "interactive": False}

        except Exception as e:
            logger.error(f"Error displaying proposed changes: {e}")
            return {"error": str(e), "displayed": False}

    async def display_inline_changes(
        self, file_path: str, original_content: str, modified_content: str
    ) -> None:
        """
        Display changes inline with color coding and annotations.

        Args:
            file_path: Path to the file
            original_content: Original file content
            modified_content: Modified file content
        """
        # Generate line-by-line diff
        original_lines = original_content.splitlines(keepends=True)
        modified_lines = modified_content.splitlines(keepends=True)

        differ = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )

        # Create annotated display
        annotated_text = Text()
        line_num = 0

        for line in differ:
            line_num += 1
            if line.startswith("+++") or line.startswith("---"):
                continue
            elif line.startswith("@@"):
                annotated_text.append(f"\n{line}\n", style="bold cyan")
            elif line.startswith("+"):
                annotated_text.append(f"{line_num:4d} + {line[1:]}", style="green")
            elif line.startswith("-"):
                annotated_text.append(f"{line_num:4d} - {line[1:]}", style="red")
            else:
                annotated_text.append(f"{line_num:4d}   {line}", style="dim white")

        # Display in panel
        panel = Panel(
            annotated_text,
            title=f"[bold]Inline Changes: {file_path}[/bold]",
            border_style="yellow",
        )

        self.console.print(panel)

    async def display_side_by_side_changes(
        self, file_path: str, original_content: str, modified_content: str
    ) -> None:
        """
        Display changes in side-by-side format.

        Args:
            file_path: Path to the file
            original_content: Original file content
            modified_content: Modified file content
        """
        # Create file change for diff renderer
        file_change = FileChange(
            file_path=file_path,
            change_type=FileChangeType.MODIFIED,
            old_content=original_content,
            new_content=modified_content,
            language=self._detect_language(file_path),
        )

        # Use diff renderer for side-by-side display
        self.diff_renderer.render_side_by_side_diff(file_change)

    async def apply_proposed_changes(
        self, change_set_id: str, selected_changes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Apply approved proposed changes to files.

        Args:
            change_set_id: ID of the change set to apply
            selected_changes: Optional list of change indices to apply (None = all)

        Returns:
            Result of applying changes
        """
        try:
            change_set = self.pending_changes.get(change_set_id)
            if not change_set:
                return {"success": False, "error": "Change set not found"}

            if not change_set.approved:
                return {"success": False, "error": "Changes not approved"}

            applied = []
            failed = []

            # Determine which changes to apply
            changes_to_apply = change_set.changes
            if selected_changes is not None:
                changes_to_apply = [
                    change_set.changes[i]
                    for i in selected_changes
                    if i < len(change_set.changes)
                ]

            # Apply each change
            for change in changes_to_apply:
                try:
                    result = await self._apply_single_change(change)
                    if result["success"]:
                        applied.append(change.file_path)
                    else:
                        failed.append((change.file_path, result.get("error")))
                except Exception as e:
                    failed.append((change.file_path, str(e)))

            # Update change set status
            if not failed:
                change_set.applied = True
                self.change_history.append(change_set)
                del self.pending_changes[change_set_id]

            return {
                "success": len(failed) == 0,
                "applied": applied,
                "failed": failed,
            }

        except Exception as e:
            logger.error(f"Error applying proposed changes: {e}")
            return {"success": False, "error": str(e)}

    def get_change_statistics(self, change_set: ChangeSet) -> Dict[str, Any]:
        """
        Get statistics about a change set.

        Args:
            change_set: Change set to analyze

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_changes": len(change_set.changes),
            "files_affected": len(set(c.file_path for c in change_set.changes)),
            "operations": {},
            "risk_distribution": {},
            "total_lines_added": 0,
            "total_lines_removed": 0,
        }

        for change in change_set.changes:
            # Count operations
            op_type = (
                change.operation_type.value if change.operation_type else "unknown"
            )
            stats["operations"][op_type] = stats["operations"].get(op_type, 0) + 1

            # Count risk levels
            risk = change.risk_level.value if change.risk_level else "unknown"
            stats["risk_distribution"][risk] = (
                stats["risk_distribution"].get(risk, 0) + 1
            )

            # Count line changes
            if change.line_changes:
                stats["total_lines_added"] += change.line_changes.get("added", 0)
                stats["total_lines_removed"] += change.line_changes.get("removed", 0)

        return stats

    async def _create_proposed_change(
        self, operation: Dict[str, Any]
    ) -> ProposedChange:
        """Create a ProposedChange from an operation."""
        file_path = operation.get("file_path", "")
        op_type = operation.get("type", "unknown")

        # Map operation type to ChangeType
        change_type_map = {
            "create": ChangeType.FILE_CREATE,
            "modify": ChangeType.FILE_MODIFY,
            "delete": ChangeType.FILE_DELETE,
            "move": ChangeType.FILE_MODIFY,
            "copy": ChangeType.FILE_CREATE,
        }

        change_type = change_type_map.get(op_type, ChangeType.FILE_MODIFY)

        # Get content
        original_content = operation.get("original_content")
        modified_content = operation.get("modified_content")

        # Calculate line changes if both contents available
        line_changes = None
        if original_content and modified_content:
            line_changes = self._calculate_line_changes(
                original_content, modified_content
            )

        return ProposedChange(
            file_path=file_path,
            operation_type=change_type,
            original_content=original_content,
            modified_content=modified_content,
            metadata=operation.get("metadata", {}),
            risk_level=RiskLevel(operation.get("risk_level", "low")),
            change_summary=operation.get("summary"),
            line_changes=line_changes,
        )

    def _calculate_line_changes(self, original: str, modified: str) -> Dict[str, Any]:
        """Calculate line changes between two contents."""
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()

        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)

        added = 0
        removed = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "delete":
                removed += i2 - i1
            elif tag == "insert":
                added += j2 - j1
            elif tag == "replace":
                removed += i2 - i1
                added += j2 - j1

        return {
            "added": added,
            "removed": removed,
            "modified": len(original_lines) - removed,
        }

    def _calculate_total_risk(self, changes: List[ProposedChange]) -> float:
        """Calculate total risk score for a set of changes."""
        risk_scores = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 3.0,
            RiskLevel.HIGH: 7.0,
            RiskLevel.CRITICAL: 10.0,
        }

        total = sum(risk_scores.get(change.risk_level, 1.0) for change in changes)
        return total / len(changes) if changes else 0.0

    def _display_change_set_header(self, change_set: ChangeSet):
        """Display header for a change set."""
        stats = self.get_change_statistics(change_set)

        # Create header table
        header_table = Table.grid(padding=1)
        header_table.add_column("Label", style="bold cyan")
        header_table.add_column("Value", style="bright_white")

        header_table.add_row("Change Set ID:", change_set.id[:8] + "...")
        header_table.add_row("Description:", change_set.description)
        header_table.add_row("Total Changes:", str(stats["total_changes"]))
        header_table.add_row("Files Affected:", str(stats["files_affected"]))
        header_table.add_row("Risk Score:", f"{change_set.total_risk_score:.1f}")

        # Create risk distribution bar
        risk_bar = self._create_risk_distribution_bar(stats["risk_distribution"])
        header_table.add_row("Risk Distribution:", risk_bar)

        panel = Panel(
            header_table,
            title="[bold]Proposed Changes Overview[/bold]",
            border_style="blue",
        )

        self.console.print(panel)

    def _display_change_summary(self, change_set: ChangeSet):
        """Display summary of changes."""
        summary_table = Table(title="Changes Summary")
        summary_table.add_column("#", style="dim", width=3)
        summary_table.add_column("File", style="cyan")
        summary_table.add_column("Operation", style="yellow")
        summary_table.add_column("Risk", style="magenta")
        summary_table.add_column("Summary", style="white")

        for i, change in enumerate(change_set.changes, 1):
            risk_color = self._get_risk_color(change.risk_level)
            summary = change.change_summary or "No summary available"

            summary_table.add_row(
                str(i),
                change.file_path,
                (change.operation_type.value if change.operation_type else "unknown"),
                f"[{risk_color}]{change.risk_level.value if change.risk_level else 'unknown'}[/{risk_color}]",
                summary[:50] + "..." if len(summary) > 50 else summary,
            )

        self.console.print(summary_table)

    async def _display_single_change(
        self, change: ProposedChange, display_mode: ChangeDisplayMode
    ):
        """Display a single change based on display mode."""
        if display_mode == ChangeDisplayMode.INLINE:
            if change.original_content and change.modified_content:
                await self.display_inline_changes(
                    change.file_path,
                    change.original_content,
                    change.modified_content,
                )
        elif display_mode == ChangeDisplayMode.SIDE_BY_SIDE:
            if change.original_content and change.modified_content:
                await self.display_side_by_side_changes(
                    change.file_path,
                    change.original_content,
                    change.modified_content,
                )
        else:  # UNIFIED
            if change.modified_content:
                await self.unified_display.display_file_modification(
                    change.file_path,
                    change.original_content or "",
                    change.modified_content,
                    {"interactive": False},
                )

    async def _get_change_approval(self, change_set: ChangeSet) -> Dict[str, Any]:
        """Get user approval for changes."""
        from rich.prompt import Prompt

        # Show options
        self.console.print("\n[bold]Options:[/bold]")
        self.console.print("  [green]A[/green] - Approve all changes")
        self.console.print("  [yellow]S[/yellow] - Select specific changes")
        self.console.print("  [red]D[/red] - Deny all changes")
        self.console.print("  [cyan]V[/cyan] - View detailed changes")

        choice = Prompt.ask("Your choice", choices=["A", "S", "D", "V"], default="V")

        if choice == "A":
            change_set.approved = True
            return {"approved": True, "all_changes": True}
        elif choice == "S":
            selected = await self._select_specific_changes(change_set)
            change_set.approved = len(selected) > 0
            return {
                "approved": len(selected) > 0,
                "selected_changes": selected,
            }
        elif choice == "D":
            return {"approved": False}
        else:  # V
            # Show detailed view and ask again
            for change in change_set.changes:
                await self._display_single_change(change, ChangeDisplayMode.UNIFIED)
            return await self._get_change_approval(change_set)

    async def _select_specific_changes(self, change_set: ChangeSet) -> List[int]:
        """Let user select specific changes to apply."""
        from rich.prompt import Prompt

        self._display_change_summary(change_set)

        selection = Prompt.ask(
            "Enter change numbers to approve (comma-separated)", default=""
        )

        if not selection:
            return []

        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            return [i for i in indices if 0 <= i < len(change_set.changes)]
        except ValueError:
            self.console.print("[red]Invalid selection[/red]")
            return []

    async def _apply_single_change(self, change: ProposedChange) -> Dict[str, Any]:
        """Apply a single change to the file system."""
        try:
            if change.operation_type == ChangeType.FILE_CREATE:
                return await self.file_system_manager.create_file(
                    change.file_path, change.modified_content or ""
                )
            elif change.operation_type == ChangeType.FILE_MODIFY:
                return await self.file_system_manager.modify_file(
                    change.file_path, change.modified_content or ""
                )
            elif change.operation_type == ChangeType.FILE_DELETE:
                return await self.file_system_manager.delete_file(change.file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {change.operation_type}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_risk_distribution_bar(self, risk_distribution: Dict[str, int]) -> str:
        """Create a visual risk distribution bar."""
        total = sum(risk_distribution.values())
        if total == 0:
            return "No risks"

        bar_parts = []
        for risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            count = risk_distribution.get(risk, 0)
            if count > 0:
                percentage = (count / total) * 100
                color = self._get_risk_color(RiskLevel[risk])
                bar_parts.append(
                    f"[{color}]{risk}: {count} ({percentage:.0f}%)[/{color}]"
                )

        return " | ".join(bar_parts)

    def _get_risk_color(self, risk_level: RiskLevel) -> str:
        """Get color for risk level."""
        colors = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "orange1",
            RiskLevel.CRITICAL: "red",
        }
        return colors.get(risk_level, "white")

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        extension = Path(file_path).suffix.lower()
        language_map = {
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
        }
        return language_map.get(extension, "text")
