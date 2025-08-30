"""
User Interface for Approval System.

This module provides interactive user interfaces for handling approval requests,
including CLI prompts, formatted displays, and batch approval workflows.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from ..security.approval_workflow import ApprovalRequest, RiskLevel
from .approval_manager import (
    BatchApprovalRequest,
    ChangePreview,
    EnhancedApprovalManager,
)
from .types import Operation, OperationType


class ApprovalChoice(Enum):
    """User approval choices."""

    APPROVE = "approve"
    DENY = "deny"
    SKIP = "skip"
    APPROVE_ALL = "approve_all"
    DENY_ALL = "deny_all"
    VIEW_DETAILS = "details"
    QUIT = "quit"


class ApprovalInterface:
    """
    Interactive user interface for handling approval requests.

    Provides CLI-based approval interface with formatted previews,
    diff visualization, and batch approval capabilities.
    """

    def __init__(self, approval_manager: EnhancedApprovalManager):
        """
        Initialize approval interface.

        Args:
            approval_manager: Enhanced approval manager instance
        """
        self.approval_manager = approval_manager
        self.approval_manager.set_approval_callback(self.handle_single_approval)
        self.approval_manager.set_batch_approval_callback(self.handle_batch_approval)

        # Interface configuration
        self.show_colors = True
        self.auto_show_diff = True
        self.max_diff_lines = 50
        self.page_size = 10

    async def handle_single_approval(self, approval_data: Dict[str, Any]) -> bool:
        """
        Handle approval request for a single operation.

        Args:
            approval_data: Dictionary containing operation, preview, and approval request

        Returns:
            True if approved, False if denied
        """
        operation = approval_data["operation"]
        preview = approval_data["preview"]
        approval_request = approval_data["approval_request"]
        risk_level = approval_data["risk_level"]

        # Display operation summary
        self._print_header(
            f"Approval Request - {operation.type.value.replace('_', ' ').title()}"
        )

        # Show operation details
        self._print_operation_info(operation, risk_level)

        # Show preview
        self._print_preview(preview)

        # Get user decision
        while True:
            choice = await self._get_user_choice(
                [
                    ApprovalChoice.APPROVE,
                    ApprovalChoice.DENY,
                    ApprovalChoice.VIEW_DETAILS,
                    ApprovalChoice.QUIT,
                ]
            )

            if choice == ApprovalChoice.APPROVE:
                self._print_success("✓ Operation approved")
                return True
            elif choice == ApprovalChoice.DENY:
                self._print_warning("✗ Operation denied")
                return False
            elif choice == ApprovalChoice.VIEW_DETAILS:
                await self._show_detailed_view(operation, preview, approval_request)
            elif choice == ApprovalChoice.QUIT:
                self._print_info("Approval cancelled by user")
                return False

    async def handle_batch_approval(
        self, batch_request: BatchApprovalRequest
    ) -> Dict[str, Any]:
        """
        Handle approval request for a batch of operations.

        Args:
            batch_request: Batch approval request

        Returns:
            Dictionary with approval decisions
        """
        self._print_header(
            f"Batch Approval Request - {len(batch_request.operations)} Operations"
        )

        # Show batch summary
        self._print_batch_summary(batch_request)

        # Handle different batch approval modes
        approval_mode = await self._get_batch_approval_mode()

        if approval_mode == "all_at_once":
            return await self._handle_bulk_approval(batch_request)
        elif approval_mode == "individual":
            return await self._handle_individual_approval(batch_request)
        elif approval_mode == "filtered":
            return await self._handle_filtered_approval(batch_request)
        else:
            return {
                "deny_all": True,
                "reason": "User cancelled batch approval",
            }

    async def _handle_bulk_approval(
        self, batch_request: BatchApprovalRequest
    ) -> Dict[str, Any]:
        """Handle bulk approval/denial of all operations."""
        self._print_info("Choose action for all operations:")

        choice = await self._get_user_choice(
            [
                ApprovalChoice.APPROVE_ALL,
                ApprovalChoice.DENY_ALL,
                ApprovalChoice.QUIT,
            ]
        )

        if choice == ApprovalChoice.APPROVE_ALL:
            self._print_success(
                f"✓ All {len(batch_request.operations)} operations approved"
            )
            return {"approve_all": True}
        elif choice == ApprovalChoice.DENY_ALL:
            reason = input("Reason for denial (optional): ").strip()
            self._print_warning(
                f"✗ All {len(batch_request.operations)} operations denied"
            )
            return {
                "deny_all": True,
                "reason": reason or "User denied all operations",
            }
        else:
            return {"deny_all": True, "reason": "User cancelled"}

    async def _handle_individual_approval(
        self, batch_request: BatchApprovalRequest
    ) -> Dict[str, Any]:
        """Handle individual approval of each operation in batch."""
        approved_indices = []

        for i, (operation, preview) in enumerate(
            zip(batch_request.operations, batch_request.previews)
        ):
            self._print_separator()
            self._print_info(f"Operation {i + 1} of {len(batch_request.operations)}")

            # Show operation details
            self._print_operation_info(operation)
            self._print_preview(preview, compact=True)

            # Get decision for this operation
            while True:
                choice = await self._get_user_choice(
                    [
                        ApprovalChoice.APPROVE,
                        ApprovalChoice.DENY,
                        ApprovalChoice.SKIP,
                        ApprovalChoice.VIEW_DETAILS,
                        ApprovalChoice.APPROVE_ALL,
                        ApprovalChoice.DENY_ALL,
                        ApprovalChoice.QUIT,
                    ]
                )

                if choice == ApprovalChoice.APPROVE:
                    approved_indices.append(i)
                    self._print_success("✓ Approved")
                    break
                elif choice == ApprovalChoice.DENY:
                    self._print_warning("✗ Denied")
                    break
                elif choice == ApprovalChoice.SKIP:
                    self._print_info("→ Skipped")
                    break
                elif choice == ApprovalChoice.VIEW_DETAILS:
                    await self._show_detailed_view(operation, preview)
                elif choice == ApprovalChoice.APPROVE_ALL:
                    # Approve remaining operations
                    approved_indices.extend(range(i, len(batch_request.operations)))
                    self._print_success(f"✓ Approved all remaining operations")
                    return {"approved_indices": approved_indices}
                elif choice == ApprovalChoice.DENY_ALL:
                    self._print_warning("✗ Denied all remaining operations")
                    return {"approved_indices": approved_indices}
                elif choice == ApprovalChoice.QUIT:
                    return {"approved_indices": approved_indices}

        return {"approved_indices": approved_indices}

    async def _handle_filtered_approval(
        self, batch_request: BatchApprovalRequest
    ) -> Dict[str, Any]:
        """Handle filtered approval based on operation types or risk levels."""
        self._print_info("Filter options:")
        print("1. Approve all low-risk operations")
        print("2. Approve all read-only operations")
        print("3. Approve by operation type")
        print("4. Custom selection")
        print("5. Cancel")

        choice = input("Choose filter option (1-5): ").strip()
        approved_indices = []

        if choice == "1":
            # Approve low-risk operations
            for i, preview in enumerate(batch_request.previews):
                if preview.risk_assessment and "low" in preview.risk_assessment.lower():
                    approved_indices.append(i)
            self._print_success(
                f"✓ Approved {len(approved_indices)} low-risk operations"
            )

        elif choice == "2":
            # Approve read-only operations
            for i, operation in enumerate(batch_request.operations):
                if operation.type in [OperationType.FILE_READ]:
                    approved_indices.append(i)
            self._print_success(
                f"✓ Approved {len(approved_indices)} read-only operations"
            )

        elif choice == "3":
            # Approve by operation type
            operation_types = list(set(op.type for op in batch_request.operations))
            print("Available operation types:")
            for idx, op_type in enumerate(operation_types, 1):
                print(f"{idx}. {op_type.value}")

            type_choice = input(
                "Select operation types to approve (comma-separated numbers): "
            ).strip()
            try:
                selected_indices = [int(x.strip()) - 1 for x in type_choice.split(",")]
                selected_types = [
                    operation_types[i]
                    for i in selected_indices
                    if 0 <= i < len(operation_types)
                ]

                for i, operation in enumerate(batch_request.operations):
                    if operation.type in selected_types:
                        approved_indices.append(i)

                self._print_success(
                    f"✓ Approved {len(approved_indices)} operations of selected types"
                )
            except (ValueError, IndexError):
                self._print_error("Invalid selection")
                return {"deny_all": True, "reason": "Invalid filter selection"}

        elif choice == "4":
            # Custom selection
            print("Enter operation indices to approve (comma-separated, 1-based):")
            indices_input = input(
                f"Operations 1-{len(batch_request.operations)}: "
            ).strip()
            try:
                approved_indices = [
                    int(x.strip()) - 1 for x in indices_input.split(",")
                ]
                approved_indices = [
                    i
                    for i in approved_indices
                    if 0 <= i < len(batch_request.operations)
                ]
                self._print_success(
                    f"✓ Approved {len(approved_indices)} selected operations"
                )
            except ValueError:
                self._print_error("Invalid indices")
                return {"deny_all": True, "reason": "Invalid custom selection"}

        else:
            return {"deny_all": True, "reason": "User cancelled"}

        return {"approved_indices": approved_indices}

    async def _get_batch_approval_mode(self) -> str:
        """Get user preference for batch approval mode."""
        self._print_info("Batch approval modes:")
        print("1. All at once (approve/deny all)")
        print("2. Individual review")
        print("3. Filtered approval")
        print("0. Cancel")

        while True:
            choice = input("Select mode (0-3): ").strip()
            if choice == "1":
                return "all_at_once"
            elif choice == "2":
                return "individual"
            elif choice == "3":
                return "filtered"
            elif choice == "0":
                return "cancel"
            else:
                self._print_error("Invalid choice. Please select 0-3.")

    async def _get_user_choice(
        self, available_choices: List[ApprovalChoice]
    ) -> ApprovalChoice:
        """Get user choice from available options."""
        choice_map = {
            "y": ApprovalChoice.APPROVE,
            "yes": ApprovalChoice.APPROVE,
            "a": ApprovalChoice.APPROVE,
            "approve": ApprovalChoice.APPROVE,
            "n": ApprovalChoice.DENY,
            "no": ApprovalChoice.DENY,
            "d": ApprovalChoice.DENY,
            "deny": ApprovalChoice.DENY,
            "s": ApprovalChoice.SKIP,
            "skip": ApprovalChoice.SKIP,
            "aa": ApprovalChoice.APPROVE_ALL,
            "all": ApprovalChoice.APPROVE_ALL,
            "approve_all": ApprovalChoice.APPROVE_ALL,
            "dd": ApprovalChoice.DENY_ALL,
            "deny_all": ApprovalChoice.DENY_ALL,
            "v": ApprovalChoice.VIEW_DETAILS,
            "details": ApprovalChoice.VIEW_DETAILS,
            "view": ApprovalChoice.VIEW_DETAILS,
            "q": ApprovalChoice.QUIT,
            "quit": ApprovalChoice.QUIT,
            "exit": ApprovalChoice.QUIT,
        }

        # Build prompt with available choices
        prompt_parts = []
        for choice in available_choices:
            if choice == ApprovalChoice.APPROVE:
                prompt_parts.append("[Y]es/[A]pprove")
            elif choice == ApprovalChoice.DENY:
                prompt_parts.append("[N]o/[D]eny")
            elif choice == ApprovalChoice.SKIP:
                prompt_parts.append("[S]kip")
            elif choice == ApprovalChoice.APPROVE_ALL:
                prompt_parts.append("Approve [A]ll")
            elif choice == ApprovalChoice.DENY_ALL:
                prompt_parts.append("[D]eny All")
            elif choice == ApprovalChoice.VIEW_DETAILS:
                prompt_parts.append("[V]iew Details")
            elif choice == ApprovalChoice.QUIT:
                prompt_parts.append("[Q]uit")

        prompt = f"Choose: {'/'.join(prompt_parts)}: "

        while True:
            user_input = input(prompt).strip().lower()

            if user_input in choice_map:
                choice = choice_map[user_input]
                if choice in available_choices:
                    return choice
                else:
                    self._print_error(
                        f"'{user_input}' is not available in this context"
                    )
            else:
                self._print_error(f"Invalid choice: '{user_input}'")

    async def _show_detailed_view(
        self,
        operation: Operation,
        preview: ChangePreview,
        approval_request: Optional[ApprovalRequest] = None,
    ):
        """Show detailed view of operation and preview."""
        self._print_header("Detailed View")

        # Operation details
        print("OPERATION DETAILS:")
        print(f"  Type: {operation.type.value}")
        print(f"  Description: {operation.description}")
        print(f"  Requires Approval: {operation.requires_approval}")
        print(f"  Reversible: {operation.reversible}")

        if operation.data:
            print("  Data:")
            for key, value in operation.data.items():
                print(f"    {key}: {value}")

        print()

        # Preview details
        print("PREVIEW DETAILS:")
        print(f"  Change Type: {preview.change_type.value}")
        print(f"  Risk Assessment: {preview.risk_assessment}")
        print(f"  Reversible: {preview.reversible}")

        if preview.metadata:
            print("  Metadata:")
            for key, value in preview.metadata.items():
                print(f"    {key}: {value}")

        print()

        # Show diff if available
        if preview.diff and self.auto_show_diff:
            print("CHANGES:")
            diff_lines = preview.diff.split("\n")
            if len(diff_lines) > self.max_diff_lines:
                print(f"Showing first {self.max_diff_lines} lines of diff:")
                for line in diff_lines[: self.max_diff_lines]:
                    self._print_diff_line(line)
                print(f"... ({len(diff_lines) - self.max_diff_lines} more lines)")
            else:
                for line in diff_lines:
                    self._print_diff_line(line)

        # Approval request details
        if approval_request:
            print()
            print("APPROVAL REQUEST:")
            print(f"  Request ID: {approval_request.id}")
            print(f"  Risk Level: {approval_request.risk_level.value}")
            print(f"  Requested At: {approval_request.requested_at}")
            if approval_request.expires_at:
                remaining = approval_request.time_remaining()
                print(f"  Time Remaining: {remaining}")

        input("\\nPress Enter to continue...")

    def _print_operation_info(
        self, operation: Operation, risk_level: Optional[RiskLevel] = None
    ):
        """Print formatted operation information."""
        print(f"Operation: {operation.type.value.replace('_', ' ').title()}")
        print(f"Description: {operation.description}")

        if risk_level:
            risk_color = self._get_risk_color(risk_level)
            print(f"Risk Level: {self._colorize(risk_level.value.upper(), risk_color)}")

        print(f"Reversible: {'Yes' if operation.reversible else 'No'}")
        print()

    def _print_preview(self, preview: ChangePreview, compact: bool = False):
        """Print formatted preview information."""
        if compact:
            print(f"Preview: {preview.description}")
            if preview.risk_assessment:
                print(f"Risk: {preview.risk_assessment}")
        else:
            print("PREVIEW:")
            print(f"  {preview.description}")
            if preview.risk_assessment:
                print(f"  Risk: {preview.risk_assessment}")

            if preview.diff and len(preview.diff.split("\n")) <= 10:
                print("  Changes:")
                for line in preview.diff.split("\n")[:5]:  # Show first 5 lines
                    print(f"    {line}")
                if len(preview.diff.split("\n")) > 5:
                    print("    ... (use 'details' to see full diff)")
        print()

    def _print_batch_summary(self, batch_request: BatchApprovalRequest):
        """Print summary of batch request."""
        print(f"Total Operations: {len(batch_request.operations)}")
        print(f"Created: {batch_request.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if batch_request.expires_at:
            print(f"Expires: {batch_request.expires_at.strftime('%Y-%m-%d %H:%M:%S')}")

        # Operation type breakdown
        type_counts = {}
        for operation in batch_request.operations:
            type_counts[operation.type] = type_counts.get(operation.type, 0) + 1

        print("\\nOperation Types:")
        for op_type, count in type_counts.items():
            print(f"  {op_type.value}: {count}")

        # Risk level breakdown
        risk_counts = {}
        for preview in batch_request.previews:
            if preview.risk_assessment:
                # Extract risk level from assessment
                if "low" in preview.risk_assessment.lower():
                    risk = "low"
                elif "medium" in preview.risk_assessment.lower():
                    risk = "medium"
                elif "high" in preview.risk_assessment.lower():
                    risk = "high"
                elif "critical" in preview.risk_assessment.lower():
                    risk = "critical"
                else:
                    risk = "unknown"
                risk_counts[risk] = risk_counts.get(risk, 0) + 1

        if risk_counts:
            print("\\nRisk Distribution:")
            for risk, count in risk_counts.items():
                color = self._get_risk_color_by_name(risk)
                print(f"  {self._colorize(risk.title(), color)}: {count}")

        print()

    def _print_diff_line(self, line: str):
        """Print a diff line with appropriate coloring."""
        if not self.show_colors:
            print(line)
            return

        if line.startswith("+"):
            print(self._colorize(line, "green"))
        elif line.startswith("-"):
            print(self._colorize(line, "red"))
        elif line.startswith("@@"):
            print(self._colorize(line, "cyan"))
        else:
            print(line)

    def _get_risk_color(self, risk_level: RiskLevel) -> str:
        """Get color for risk level."""
        colors = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "red",
            RiskLevel.CRITICAL: "magenta",
        }
        return colors.get(risk_level, "white")

    def _get_risk_color_by_name(self, risk_name: str) -> str:
        """Get color for risk level by name."""
        colors = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "magenta",
        }
        return colors.get(risk_name.lower(), "white")

    def _colorize(self, text: str, color: str) -> str:
        """Add color to text if colors are enabled."""
        if not self.show_colors:
            return text

        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "reset": "\033[0m",
        }

        color_code = colors.get(color.lower(), colors["white"])
        reset_code = colors["reset"]
        return f"{color_code}{text}{reset_code}"

    def _print_header(self, title: str):
        """Print formatted header."""
        print()
        print("=" * 60)
        print(f" {title}")
        print("=" * 60)
        print()

    def _print_separator(self):
        """Print separator line."""
        print("-" * 40)

    def _print_success(self, message: str):
        """Print success message."""
        print(self._colorize(message, "green"))

    def _print_warning(self, message: str):
        """Print warning message."""
        print(self._colorize(message, "yellow"))

    def _print_error(self, message: str):
        """Print error message."""
        print(self._colorize(message, "red"))

    def _print_info(self, message: str):
        """Print info message."""
        print(self._colorize(message, "cyan"))

    def set_colors_enabled(self, enabled: bool):
        """Enable or disable colored output."""
        self.show_colors = enabled

    def set_auto_show_diff(self, enabled: bool):
        """Enable or disable automatic diff display."""
        self.auto_show_diff = enabled

    def set_max_diff_lines(self, max_lines: int):
        """Set maximum number of diff lines to display."""
        self.max_diff_lines = max_lines
