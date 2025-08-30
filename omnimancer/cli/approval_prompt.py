"""
Interactive Approval Prompt Handler for Omnimancer.

This module provides an interactive prompt system that presents approval dialogs
to users and captures their responses including 'Yes', 'Yes and remember', and 'No'
options with proper keyboard input handling.
"""

import asyncio
import logging
import signal
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from omnimancer.ui.cancellation_handler import get_active_cancellation_handler

from ..core.agent.approval_manager import BatchApprovalRequest, ChangePreview
from ..core.security.approval_workflow import ApprovalRequest
from .approval_formatter import CLIApprovalFormatter

logger = logging.getLogger(__name__)


class ApprovalDecisionType(Enum):
    """Types of approval decisions."""

    APPROVED = "approved"
    APPROVED_AND_REMEMBER = "approved_and_remember"
    REVIEW_AND_EDIT = "review_and_edit"
    DENIED = "denied"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ApprovalDecision:
    """Represents a user's approval decision."""

    decision: ApprovalDecisionType
    remember: bool = False
    user_notes: str = ""
    response_time_seconds: float = 0.0
    timeout_occurred: bool = False
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def is_approved(self) -> bool:
        """Check if this decision approves the operation."""
        return self.decision in [
            ApprovalDecisionType.APPROVED,
            ApprovalDecisionType.APPROVED_AND_REMEMBER,
            ApprovalDecisionType.REVIEW_AND_EDIT,
        ]

    @property
    def should_remember(self) -> bool:
        """Check if this decision should be remembered."""
        return (
            self.decision == ApprovalDecisionType.APPROVED_AND_REMEMBER or self.remember
        )


@dataclass
class BatchApprovalDecision:
    """Represents a batch approval decision."""

    decision_type: str  # "approve_all", "deny_all", "selective", "individual"
    approved_indices: List[int] = None
    user_notes: str = ""
    individual_decisions: List[ApprovalDecision] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.approved_indices is None:
            self.approved_indices = []
        if self.individual_decisions is None:
            self.individual_decisions = []


class CLIApprovalPrompt:
    """
    Interactive prompt system for approval dialogs.

    Integrates with CLIApprovalFormatter to display approval dialogs and capture
    user input with keyboard shortcuts, timeout handling, and graceful cancellation.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        formatter: Optional[CLIApprovalFormatter] = None,
        default_timeout_seconds: int = 300,
    ):  # 5 minutes default
        """
        Initialize the approval prompt handler.

        Args:
            console: Rich Console instance
            formatter: CLIApprovalFormatter for display
            default_timeout_seconds: Default timeout for approval prompts
        """
        self.console = console or Console()
        self.formatter = formatter or CLIApprovalFormatter(self.console)
        self.default_timeout_seconds = default_timeout_seconds

        # Signal handling for graceful interruption
        self._original_sigint_handler = None
        self._interrupted = False

        # Valid response mappings
        self.approval_responses = {
            "y": ApprovalDecisionType.APPROVED,
            "yes": ApprovalDecisionType.APPROVED,
            "approve": ApprovalDecisionType.APPROVED,
            "1": ApprovalDecisionType.APPROVED,  # Support legacy numeric responses
            "r": ApprovalDecisionType.APPROVED_AND_REMEMBER,  # Remember decision
            "remember": ApprovalDecisionType.APPROVED_AND_REMEMBER,
            "e": ApprovalDecisionType.REVIEW_AND_EDIT,  # Edit content
            "edit": ApprovalDecisionType.REVIEW_AND_EDIT,
            "2": ApprovalDecisionType.REVIEW_AND_EDIT,  # Support legacy numeric responses
            "n": ApprovalDecisionType.DENIED,
            "no": ApprovalDecisionType.DENIED,
            "reject": ApprovalDecisionType.DENIED,
            "3": ApprovalDecisionType.DENIED,  # Support legacy numeric responses
            "q": ApprovalDecisionType.CANCELLED,
            "quit": ApprovalDecisionType.CANCELLED,
            "cancel": ApprovalDecisionType.CANCELLED,
        }

    async def prompt_for_file_modification_approval(
        self,
        review_data: Dict[str, Any],
        current_model: str = "Unknown",
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Present file modification approval with unified UI.

        Args:
            review_data: File modification data
            current_model: Current AI model name for UI headers
            timeout_seconds: Timeout in seconds

        Returns:
            Dictionary with approval decision and any modified content
        """
        start_time = datetime.now()
        timeout = timeout_seconds or self.default_timeout_seconds

        try:
            # Set up interrupt handling
            self._setup_interrupt_handling()

            # Display file modification preview with current model
            self._display_file_modification_preview(review_data, current_model)

            # Get user decision with timeout
            decision = await self._get_file_modification_decision_with_timeout(
                review_data, timeout
            )

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            decision["response_time_seconds"] = response_time

            return decision

        except KeyboardInterrupt:
            self.console.print(
                "\n[red]❌ File modification cancelled by user (Ctrl+C)[/red]"
            )
            return {
                "approved": False,
                "reason": "Cancelled by user (Ctrl+C)",
                "response_time_seconds": (datetime.now() - start_time).total_seconds(),
            }

        except asyncio.TimeoutError:
            self.console.print(
                f"\n[red]⏰ File modification timed out after {timeout} seconds - Operation denied[/red]"
            )
            return {
                "approved": False,
                "reason": f"Timed out after {timeout} seconds",
                "response_time_seconds": timeout,
                "timeout_occurred": True,
            }

        except Exception as e:
            logger.error(f"Error in file modification approval: {e}")
            return {
                "approved": False,
                "reason": f"Error occurred: {str(e)}",
                "response_time_seconds": (datetime.now() - start_time).total_seconds(),
            }

        finally:
            # Restore original interrupt handling
            self._restore_interrupt_handling()

    async def prompt_for_approval(
        self,
        approval_request: ApprovalRequest,
        preview: Optional[ChangePreview] = None,
        operation_data: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> ApprovalDecision:
        """
        Present approval dialog and capture user decision.

        Args:
            approval_request: The approval request to present
            preview: Optional change preview with diff information
            operation_data: Additional operation data for context
            timeout_seconds: Timeout in seconds (uses default if None)

        Returns:
            ApprovalDecision with user's choice
        """
        start_time = datetime.now()
        timeout = timeout_seconds or self.default_timeout_seconds

        try:
            # Set up interrupt handling
            self._setup_interrupt_handling()

            # Display the approval dialog
            dialog = self.formatter.format_approval_dialog(
                approval_request, preview, operation_data
            )

            self.console.print(dialog)
            self.console.print()  # Extra line for spacing

            # Get user input with timeout
            decision = await self._get_user_decision_with_timeout(timeout)

            # Handle special case for "remember" - confirm user understands
            if decision.decision == ApprovalDecisionType.APPROVED_AND_REMEMBER:
                if await self._confirm_remember_decision(approval_request):
                    decision.remember = True
                else:
                    # User declined remember, convert to regular approval
                    decision.decision = ApprovalDecisionType.APPROVED
                    decision.remember = False

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            decision.response_time_seconds = response_time

            # Display confirmation
            if decision.is_approved:
                confirmation = self.formatter.format_approval_approved(
                    approval_request, decision.should_remember
                )
            else:
                confirmation = self.formatter.format_approval_denied(
                    approval_request, self._get_decision_reason(decision)
                )

            self.console.print(confirmation)

            return decision

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully with cleanup
            response_time = (datetime.now() - start_time).total_seconds()
            decision = ApprovalDecision(
                decision=ApprovalDecisionType.CANCELLED,
                user_notes="Cancelled by user (Ctrl+C)",
                response_time_seconds=response_time,
            )

            self.console.print("\n[red]❌ Operation cancelled by user (Ctrl+C)[/red]")

            # Log cancellation details for debugging
            logger.info(
                f"Approval cancelled after {response_time:.2f} seconds: {approval_request.operation_type}"
            )

            # Perform any necessary cleanup
            await self._handle_cancellation_cleanup(approval_request, "user_interrupt")

            return decision

        except asyncio.TimeoutError:
            # Handle timeout with cleanup
            decision = ApprovalDecision(
                decision=ApprovalDecisionType.TIMEOUT,
                timeout_occurred=True,
                user_notes=f"Timed out after {timeout} seconds",
                response_time_seconds=timeout,
            )

            self.console.print(
                f"\n[red]⏰ Approval timed out after {timeout} seconds - Operation denied[/red]"
            )

            # Log timeout details
            logger.warning(
                f"Approval timeout after {timeout}s: {approval_request.operation_type}"
            )

            # Perform cleanup for timeout
            await self._handle_cancellation_cleanup(approval_request, "timeout")

            return decision

        except Exception as e:
            logger.error(f"Error in approval prompt: {e}")
            decision = ApprovalDecision(
                decision=ApprovalDecisionType.DENIED,
                user_notes=f"Error occurred: {str(e)}",
                response_time_seconds=(datetime.now() - start_time).total_seconds(),
            )

            self.console.print(f"[red]❌ Error in approval prompt: {e}[/red]")
            return decision

        finally:
            # Restore original interrupt handling
            self._restore_interrupt_handling()

    async def prompt_for_batch_approval(
        self,
        batch_request: BatchApprovalRequest,
        timeout_seconds: Optional[int] = None,
    ) -> BatchApprovalDecision:
        """
        Present batch approval dialog and capture user decisions.

        Args:
            batch_request: BatchApprovalRequest with multiple operations
            timeout_seconds: Timeout in seconds (uses default if None)

        Returns:
            BatchApprovalDecision with user's choices
        """
        timeout = timeout_seconds or self.default_timeout_seconds

        try:
            self._setup_interrupt_handling()

            # Display batch approval dialog
            dialog = self.formatter.format_batch_approval_dialog(batch_request)
            self.console.print(dialog)
            self.console.print()

            # Get batch decision
            batch_decision = await self._get_batch_decision_with_timeout(
                batch_request, timeout
            )

            # Display confirmation
            self._display_batch_confirmation(batch_decision, batch_request)

            return batch_decision

        except KeyboardInterrupt:
            self.console.print("\n[red]❌ Batch approval cancelled by user[/red]")
            return BatchApprovalDecision(
                decision_type="cancelled",
                user_notes="Cancelled by user (Ctrl+C)",
            )

        except asyncio.TimeoutError:
            self.console.print(
                f"\n[red]⏰ Batch approval timed out after {timeout} seconds[/red]"
            )
            return BatchApprovalDecision(
                decision_type="timeout",
                user_notes=f"Timed out after {timeout} seconds",
            )

        finally:
            self._restore_interrupt_handling()

    async def _get_user_decision_with_timeout(
        self, timeout_seconds: int
    ) -> ApprovalDecision:
        """Get user decision with timeout handling."""
        try:
            # Use asyncio.wait_for for timeout
            response = await asyncio.wait_for(
                self._get_user_input_async(
                    "Enter your decision (y=approve/r=remember/e=edit/n=reject/q=quit): "
                ),
                timeout=timeout_seconds,
            )

            response_lower = response.strip().lower()

            # Map response to decision type
            decision_type = self.approval_responses.get(response_lower)

            if decision_type is None:
                # Invalid response, ask again with clearer options
                self.console.print("[yellow]Invalid response. Please enter:[/yellow]")
                self.console.print("  [green]y[/green] = Yes (approve)")
                self.console.print(
                    "  [blue]r[/blue] = Remember (approve and auto-approve similar)"
                )
                self.console.print(
                    "  [cyan]e[/cyan] = Edit (review and modify content)"
                )
                self.console.print("  [red]n[/red] = No (reject)")
                self.console.print(
                    "  [yellow]q[/yellow] = Quit (cancel and exit to omnimancer prompt)"
                )

                # Recursive call with reduced timeout
                remaining_timeout = max(timeout_seconds - 10, 30)
                return await self._get_user_decision_with_timeout(remaining_timeout)

            return ApprovalDecision(decision=decision_type)

        except asyncio.TimeoutError:
            raise  # Re-raise timeout for upper level handling

    async def _get_batch_decision_with_timeout(
        self, batch_request: BatchApprovalRequest, timeout_seconds: int
    ) -> BatchApprovalDecision:
        """Get batch decision with timeout handling."""
        try:
            response = await asyncio.wait_for(
                self._get_user_input_async(
                    "Enter batch decision (all/none/select/individual/q): "
                ),
                timeout=timeout_seconds,
            )

            response_lower = response.strip().lower()

            if response_lower in ["all", "approve-all", "yes-all"]:
                return BatchApprovalDecision(
                    decision_type="approve_all",
                    approved_indices=list(range(len(batch_request.operations))),
                )

            elif response_lower in ["none", "deny-all", "no-all"]:
                return BatchApprovalDecision(decision_type="deny_all")

            elif response_lower in ["select", "selective", "s"]:
                return await self._handle_selective_approval(
                    batch_request, timeout_seconds
                )

            elif response_lower in ["individual", "one-by-one", "i"]:
                return await self._handle_individual_approval(
                    batch_request, timeout_seconds
                )

            elif response_lower in ["q", "quit", "cancel"]:
                return BatchApprovalDecision(
                    decision_type="cancelled", user_notes="Cancelled by user"
                )

            else:
                # Invalid response
                self.console.print("[yellow]Invalid response. Please enter:[/yellow]")
                self.console.print("  [green]all[/green] = Approve all operations")
                self.console.print("  [red]none[/red] = Deny all operations")
                self.console.print("  [blue]select[/blue] = Choose specific operations")
                self.console.print(
                    "  [yellow]individual[/yellow] = Review each operation separately"
                )
                self.console.print("  [yellow]q[/yellow] = Cancel batch")

                remaining_timeout = max(timeout_seconds - 15, 30)
                return await self._get_batch_decision_with_timeout(
                    batch_request, remaining_timeout
                )

        except asyncio.TimeoutError:
            raise

    async def _handle_selective_approval(
        self, batch_request: BatchApprovalRequest, timeout_seconds: int
    ) -> BatchApprovalDecision:
        """Handle selective operation approval."""
        operation_count = len(batch_request.operations)

        self.console.print(
            f"\n[blue]Select operations to approve (1-{operation_count}):[/blue]"
        )
        self.console.print(
            "Enter numbers separated by spaces or commas (e.g., '1 3 5' or '1,3,5')"
        )
        self.console.print("Or enter ranges (e.g., '1-3 5 7-9')")

        try:
            response = await asyncio.wait_for(
                self._get_user_input_async(
                    f"Select operations (1-{operation_count}): "
                ),
                timeout=timeout_seconds,
            )

            selected_indices = self._parse_operation_selection(
                response, operation_count
            )

            if selected_indices is None:
                # Invalid selection, try again
                self.console.print("[red]Invalid selection format[/red]")
                return await self._handle_selective_approval(
                    batch_request, timeout_seconds - 10
                )

            return BatchApprovalDecision(
                decision_type="selective", approved_indices=selected_indices
            )

        except asyncio.TimeoutError:
            raise

    async def _handle_individual_approval(
        self, batch_request: BatchApprovalRequest, timeout_seconds: int
    ) -> BatchApprovalDecision:
        """Handle individual operation approval."""
        individual_decisions = []
        approved_indices = []

        self.console.print(
            "\n[blue]Individual approval mode - reviewing each operation:[/blue]"
        )

        for i, (operation, preview) in enumerate(
            zip(batch_request.operations, batch_request.previews)
        ):
            self.console.print(
                f"\n[bold]--- Operation {i + 1} of {len(batch_request.operations)} ---[/bold]"
            )

            # Create individual approval request
            individual_request = ApprovalRequest(
                operation_type=operation.type.value,
                description=operation.description,
                metadata=operation.data,
            )

            # Get approval for this individual operation
            decision = await self.prompt_for_approval(
                individual_request,
                preview,
                operation.data,
                timeout_seconds=min(timeout_seconds, 120),  # 2 minute max per operation
            )

            individual_decisions.append(decision)

            if decision.is_approved:
                approved_indices.append(i)

            # Check if user wants to cancel the rest
            if decision.decision == ApprovalDecisionType.CANCELLED:
                self.console.print("[yellow]Cancelled remaining operations[/yellow]")
                break

        return BatchApprovalDecision(
            decision_type="individual",
            approved_indices=approved_indices,
            individual_decisions=individual_decisions,
        )

    def _parse_operation_selection(
        self, response: str, max_operations: int
    ) -> Optional[List[int]]:
        """Parse operation selection string into list of indices."""
        try:
            indices = set()

            # Replace commas with spaces and split
            parts = response.replace(",", " ").split()

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if "-" in part:
                    # Handle ranges like "1-3"
                    range_parts = part.split("-")
                    if len(range_parts) != 2:
                        return None

                    start = int(range_parts[0])
                    end = int(range_parts[1])

                    if start < 1 or end > max_operations or start > end:
                        return None

                    indices.update(range(start - 1, end))  # Convert to 0-based indexing
                else:
                    # Handle individual numbers
                    num = int(part)
                    if num < 1 or num > max_operations:
                        return None
                    indices.add(num - 1)  # Convert to 0-based indexing

            return sorted(list(indices))

        except ValueError:
            return None

    async def _confirm_remember_decision(
        self, approval_request: ApprovalRequest
    ) -> bool:
        """Confirm user wants to remember this approval decision."""
        self.console.print()
        self.console.print("[blue]🧠 Remember Decision Confirmation[/blue]")
        self.console.print(
            f"You chose to remember this approval for similar '{approval_request.operation_type}' operations."
        )
        self.console.print(
            "This means similar operations will be automatically approved without asking."
        )
        self.console.print()

        try:
            response = await asyncio.wait_for(
                self._get_user_input_async(
                    "Are you sure you want to remember this? (y/n): "
                ),
                timeout=30,  # 30 second timeout for confirmation
            )

            return response.strip().lower() in ["y", "yes", "true", "1"]

        except asyncio.TimeoutError:
            self.console.print(
                "[yellow]Confirmation timed out - not remembering decision[/yellow]"
            )
            return False

    async def _get_user_input_async(self, prompt_text: str) -> str:
        """Get user input asynchronously."""

        # Use asyncio.to_thread for truly async input
        def get_input():
            return input(prompt_text)

        try:
            return await asyncio.to_thread(get_input)
        except KeyboardInterrupt:
            # Handle Ctrl+C during input
            raise

    def _setup_interrupt_handling(self):
        """Set up signal handling for graceful interruption."""

        def signal_handler(signum, frame):
            self._interrupted = True
            # Don't call sys.exit here, let the calling code handle it
            raise KeyboardInterrupt("User interrupted with Ctrl+C")

        self._original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    def _restore_interrupt_handling(self):
        """Restore original signal handling."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._original_sigint_handler = None

    def _get_decision_reason(self, decision: ApprovalDecision) -> str:
        """Get human-readable reason for decision."""
        if decision.decision == ApprovalDecisionType.DENIED:
            return decision.user_notes or "User denied operation"
        elif decision.decision == ApprovalDecisionType.CANCELLED:
            return decision.user_notes or "User cancelled operation"
        elif decision.decision == ApprovalDecisionType.TIMEOUT:
            return decision.user_notes or "Operation timed out"
        else:
            return decision.user_notes or ""

    def _display_batch_confirmation(
        self,
        batch_decision: BatchApprovalDecision,
        batch_request: BatchApprovalRequest,
    ):
        """Display confirmation of batch decision."""
        if batch_decision.decision_type == "approve_all":
            self.console.print(
                f"[green]✅ Approved all {len(batch_request.operations)} operations[/green]"
            )

        elif batch_decision.decision_type == "deny_all":
            self.console.print(
                f"[red]❌ Denied all {len(batch_request.operations)} operations[/red]"
            )

        elif batch_decision.decision_type == "selective":
            approved_count = len(batch_decision.approved_indices)
            total_count = len(batch_request.operations)
            self.console.print(
                f"[blue]✅ Approved {approved_count} of {total_count} operations[/blue]"
            )

            # Show which operations were approved
            if approved_count > 0:
                approved_nums = [str(i + 1) for i in batch_decision.approved_indices]
                self.console.print(
                    f"   Approved operations: {', '.join(approved_nums)}"
                )

        elif batch_decision.decision_type == "individual":
            approved_count = len(batch_decision.approved_indices)
            total_count = len(batch_request.operations)
            self.console.print(
                f"[blue]✅ Individual review complete: {approved_count} of {total_count} approved[/blue]"
            )

        elif batch_decision.decision_type in ["cancelled", "timeout"]:
            self.console.print("[yellow]❌ Batch approval cancelled[/yellow]")

    async def _handle_cancellation_cleanup(
        self, approval_request: ApprovalRequest, reason: str
    ):
        """
        Handle cleanup tasks when an approval is cancelled or times out.

        Args:
            approval_request: The approval request that was cancelled
            reason: Reason for cancellation ("user_interrupt", "timeout", etc.)
        """
        try:
            # Log the cancellation for debugging
            logger.debug(
                f"Performing cancellation cleanup for {approval_request.operation_type} - Reason: {reason}"
            )

            # Clean up any temporary resources that might have been allocated
            # during the approval process

            # If there are any file handles, locks, or other resources
            # related to the approval request, clean them up here

            # Record cancellation metrics for later analysis
            cancellation_data = {
                "operation_type": approval_request.operation_type,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "risk_level": (
                    approval_request.risk_level.value
                    if approval_request.risk_level
                    else "unknown"
                ),
            }

            # In a full implementation, this could be sent to a metrics system
            logger.info(f"Approval cancellation recorded: {cancellation_data}")

        except Exception as e:
            logger.error(f"Error during cancellation cleanup: {e}")
            # Don't re-raise - cleanup errors shouldn't fail the cancellation

    def cleanup_resources(self):
        """
        Clean up any resources held by the approval prompt handler.

        This method should be called when the approval prompt handler
        is no longer needed to ensure proper resource cleanup.
        """
        try:
            # Restore interrupt handling if still set
            self._restore_interrupt_handling()

            # Clear any cached data
            self._interrupted = False

            # Any other cleanup tasks
            logger.debug("Approval prompt handler resources cleaned up")

        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")

    def _display_file_modification_preview(
        self, review_data: Dict[str, Any], current_model: str
    ):
        """Display file modification preview using current model name."""
        # Ensure we start on a new line, clearing any spinner artifacts
        self.console.print()

        # Display header with current model name
        self._display_modification_header(review_data, current_model)

        # Show content preview based on operation type
        if review_data["operation"] == "create":
            self._display_new_file_preview(review_data)
        else:
            self._display_modification_preview(review_data)

    def _display_modification_header(
        self, review_data: Dict[str, Any], current_model: str
    ):
        """Display header information about the file modification."""
        file_path = review_data["file_path"]
        operation = review_data["operation"]

        # Create header panel with current model
        header_text = Text()
        header_text.append(
            f"{current_model} - File Modification Review\n", style="bold cyan"
        )
        header_text.append(f"Operation: ", style="bold")
        header_text.append(f"{operation.title()}\n", style="yellow")
        header_text.append(f"File: ", style="bold")
        header_text.append(f"{file_path}", style="green")

        header_panel = Panel(
            header_text,
            title="[bold cyan]File Modification[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

        self.console.print(header_panel)

    def _display_new_file_preview(self, review_data: Dict[str, Any]):
        """Display preview of new file creation."""
        new_content = review_data.get("new_content", "")
        file_path = Path(review_data["file_path"])

        # Show content preview with syntax highlighting if possible
        try:
            syntax = Syntax(
                new_content[:2000],
                file_path.suffix.lstrip(".") or "text",
                theme="monokai",
                line_numbers=True,
            )
            preview_panel = Panel(
                syntax,
                title="[bold green]New File Content (Preview)[/bold green]",
                border_style="green",
            )
        except Exception:
            # Fallback to plain text
            preview_content = (
                new_content[:2000] + "..." if len(new_content) > 2000 else new_content
            )
            preview_panel = Panel(
                preview_content,
                title="[bold green]New File Content (Preview)[/bold green]",
                border_style="green",
            )

        self.console.print(preview_panel)

        # Show content statistics
        self._display_content_stats(new_content, is_new=True)

    def _display_modification_preview(self, review_data: Dict[str, Any]):
        """Display preview of file modification."""
        current_content = review_data.get("current_content", "")
        new_content = review_data.get("new_content", "")
        diff = review_data.get("diff", "")

        # Show current content summary if it exists
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
        new_lines = len(new_content.split("\n")) if new_content else 0
        current_chars = len(current_content) if current_content else 0
        new_chars = len(new_content) if new_content else 0

        line_change = new_lines - current_lines
        char_change = new_chars - current_chars

        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Metric", style="bold")
        table.add_column("Current", style="yellow")
        table.add_column("New", style="green")
        table.add_column("Change", style="cyan")

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

    async def _get_file_modification_decision_with_timeout(
        self, review_data: Dict[str, Any], timeout_seconds: int
    ) -> Dict[str, Any]:
        """Get user decision about file modification with timeout."""
        self.console.print()  # Add some spacing

        # Present options to user
        options_text = Text()
        options_text.append("Options:\n", style="bold cyan")
        options_text.append("y. ", style="bold green")
        options_text.append("Approve changes\n", style="green")
        options_text.append("r. ", style="bold blue")
        options_text.append(
            "Remember (approve and auto-approve similar)\n", style="blue"
        )
        options_text.append("e. ", style="bold cyan")
        options_text.append("Edit content before writing\n", style="cyan")
        options_text.append("n. ", style="bold red")
        options_text.append("Reject changes\n", style="red")
        options_text.append("q. ", style="bold yellow")
        options_text.append(
            "Quit (cancel and exit to omnimancer prompt)\n", style="yellow"
        )

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
            # Get user choice with timeout
            response = await asyncio.wait_for(
                self._get_user_input_async("Enter your decision (y/r/e/n/q): "),
                timeout=timeout_seconds,
            )

            response_lower = response.strip().lower()

            # Process the user's choice
            if response_lower in ["y", "yes", "approve", "1"]:
                return {"approved": True, "reason": "User approved changes"}
            elif response_lower in ["r", "remember"]:
                return {
                    "approved": True,
                    "remember": True,
                    "reason": "User approved and remembered decision",
                }
            elif response_lower in ["e", "edit", "2"]:
                return await self._handle_edit_content(review_data)
            elif response_lower in ["n", "no", "reject", "3"]:
                # Ask for optional reason
                try:
                    reason_response = await asyncio.wait_for(
                        self._get_user_input_async("Reason for rejection (optional): "),
                        timeout=30,
                    )
                    reason = reason_response.strip() or "User rejected changes"
                except asyncio.TimeoutError:
                    reason = "User rejected changes"

                return {"approved": False, "reason": reason}
            elif response_lower in ["q", "quit", "cancel"]:
                return {
                    "approved": False,
                    "reason": "User cancelled operation",
                    "cancelled": True,
                }
            else:
                # Invalid response, show help and try again
                self.console.print("[yellow]Invalid response. Please enter:[/yellow]")
                self.console.print("  [green]y[/green] = Yes (approve)")
                self.console.print(
                    "  [blue]r[/blue] = Remember (approve and auto-approve similar)"
                )
                self.console.print(
                    "  [cyan]e[/cyan] = Edit (review and modify content)"
                )
                self.console.print("  [red]n[/red] = No (reject)")
                self.console.print(
                    "  [yellow]q[/yellow] = Quit (cancel and exit to omnimancer prompt)"
                )

                # Retry with reduced timeout
                remaining_timeout = max(timeout_seconds - 10, 30)
                return await self._get_file_modification_decision_with_timeout(
                    review_data, remaining_timeout
                )

        except asyncio.TimeoutError:
            raise  # Re-raise timeout for upper level handling

        finally:
            # Resume the spinner after user interaction
            if cancellation_handler:
                cancellation_handler.resume_status_display()

    async def _handle_edit_content(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user editing of content before writing."""
        self.console.print("\n[yellow]Content editing functionality:[/yellow]")
        self.console.print("You can:")
        self.console.print("1. Provide replacement content inline")
        self.console.print("2. Open external editor (if configured)")
        self.console.print("3. Go back to original content")

        try:
            edit_choice = await asyncio.wait_for(
                self._get_user_input_async("Edit method (1=inline/2=editor/3=back): "),
                timeout=60,
            )

            edit_choice_lower = edit_choice.strip().lower()

            if edit_choice_lower in ["1", "inline"]:
                return await self._inline_content_edit(review_data)
            elif edit_choice_lower in ["2", "editor"]:
                return await self._external_editor_edit(review_data)
            else:
                # Go back to original decision
                return await self._get_file_modification_decision_with_timeout(
                    review_data, 300
                )

        except asyncio.TimeoutError:
            self.console.print(
                "[yellow]Edit choice timed out - going back to approval options[/yellow]"
            )
            return await self._get_file_modification_decision_with_timeout(
                review_data, 300
            )

    async def _inline_content_edit(self, review_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle inline content editing."""
        self.console.print("\n[yellow]Inline content editing:[/yellow]")
        self.console.print(
            "Enter new content (type END_CONTENT on a new line to finish):"
        )

        lines = []
        while True:
            try:
                line = await asyncio.to_thread(input)
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

        try:
            confirm_response = await asyncio.wait_for(
                self._get_user_input_async("Use this modified content? (y/n): "),
                timeout=30,
            )

            if confirm_response.strip().lower() in ["y", "yes"]:
                return {
                    "approved": True,
                    "modified_content": modified_content,
                    "reason": "User provided modified content",
                }
            else:
                return await self._get_file_modification_decision_with_timeout(
                    review_data, 300
                )

        except asyncio.TimeoutError:
            self.console.print(
                "[yellow]Confirmation timed out - using modified content[/yellow]"
            )
            return {
                "approved": True,
                "modified_content": modified_content,
                "reason": "User provided modified content (auto-confirmed due to timeout)",
            }

    async def _external_editor_edit(
        self, review_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle external editor content editing."""
        self.console.print(
            "[yellow]External editor functionality not yet implemented[/yellow]"
        )
        self.console.print("[yellow]Falling back to inline editing...[/yellow]")
        return await self._inline_content_edit(review_data)


# Utility functions for easy integration


async def prompt_single_approval(
    approval_request: ApprovalRequest,
    preview: Optional[ChangePreview] = None,
    operation_data: Optional[Dict[str, Any]] = None,
    console: Optional[Console] = None,
    timeout_seconds: int = 300,
) -> ApprovalDecision:
    """
    Convenience function for single approval prompts.

    Args:
        approval_request: The approval request
        preview: Optional change preview
        operation_data: Optional operation data
        console: Rich console (creates new if None)
        timeout_seconds: Timeout in seconds

    Returns:
        ApprovalDecision with user's choice
    """
    prompt_handler = CLIApprovalPrompt(
        console=console, default_timeout_seconds=timeout_seconds
    )
    return await prompt_handler.prompt_for_approval(
        approval_request, preview, operation_data, timeout_seconds
    )


async def prompt_batch_approval(
    batch_request: BatchApprovalRequest,
    console: Optional[Console] = None,
    timeout_seconds: int = 600,
) -> BatchApprovalDecision:
    """
    Convenience function for batch approval prompts.

    Args:
        batch_request: The batch approval request
        console: Rich console (creates new if None)
        timeout_seconds: Timeout in seconds

    Returns:
        BatchApprovalDecision with user's choices
    """
    prompt_handler = CLIApprovalPrompt(
        console=console, default_timeout_seconds=timeout_seconds
    )
    return await prompt_handler.prompt_for_batch_approval(
        batch_request, timeout_seconds
    )
