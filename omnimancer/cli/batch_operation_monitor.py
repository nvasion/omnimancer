"""
Batch Operation Monitoring and Error Recovery for Omnimancer CLI.

This module provides comprehensive feedback system for batch operations
including progress tracking, result reporting, and error recovery mechanisms.
"""

import asyncio
import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ..core.agent.approval_manager import BatchApprovalRequest
from ..core.agent.types import Operation

logger = logging.getLogger(__name__)


class OperationStatus(Enum):
    """Status of individual operations in a batch."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class BatchOperationResult(Enum):
    """Overall result of batch operation."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class OperationResult:
    """Result of a single operation execution."""

    operation_id: int
    operation: Operation
    status: OperationStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    execution_details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[timedelta]:
        """Get operation duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class BatchExecutionSummary:
    """Summary of batch execution results."""

    batch_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    skipped_operations: int = 0
    retried_operations: int = 0
    result: BatchOperationResult = BatchOperationResult.SUCCESS
    error_summary: List[str] = field(default_factory=list)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> Optional[timedelta]:
        """Get total execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_operations == 0:
            return 100.0
        return (self.successful_operations / self.total_operations) * 100


class BatchOperationMonitor:
    """
    Monitors and manages batch operation execution with progress tracking,
    error recovery, and comprehensive result reporting.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        enable_live_display: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        save_results: bool = True,
    ):
        """
        Initialize batch operation monitor.

        Args:
            console: Rich console for display
            enable_live_display: Whether to show live progress updates
            max_retries: Maximum retry attempts for failed operations
            retry_delay: Delay between retry attempts in seconds
            save_results: Whether to save execution results to files
        """
        self.console = console or Console()
        self.enable_live_display = enable_live_display
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.save_results = save_results

        # Tracking data
        self.operation_results: Dict[int, OperationResult] = {}
        self.current_summary: Optional[BatchExecutionSummary] = None
        self.progress_tracker: Optional[Progress] = None
        self.live_display: Optional[Live] = None

        # History storage
        self.results_directory = Path.home() / ".omnimancer" / "batch_results"
        if self.save_results:
            self.results_directory.mkdir(parents=True, exist_ok=True)

    async def execute_batch(
        self, batch_request: BatchApprovalRequest, operation_executor: callable
    ) -> BatchExecutionSummary:
        """
        Execute a batch of operations with monitoring and error recovery.

        Args:
            batch_request: Batch approval request to execute
            operation_executor: Async function to execute individual operations

        Returns:
            BatchExecutionSummary with complete execution results
        """
        # Initialize tracking
        self.current_summary = BatchExecutionSummary(
            batch_id=batch_request.id,
            start_time=datetime.now(),
            total_operations=len(batch_request.operations),
        )

        # Setup progress tracking
        if self.enable_live_display:
            await self._setup_progress_display(batch_request)

        try:
            # Execute operations
            await self._execute_operations(batch_request, operation_executor)

            # Finalize results
            self.current_summary.end_time = datetime.now()
            self._calculate_final_results()

            # Save results if enabled
            if self.save_results:
                await self._save_execution_results()

            return self.current_summary

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            self.current_summary.result = BatchOperationResult.FAILURE
            self.current_summary.error_summary.append(
                f"Batch execution error: {str(e)}"
            )
            return self.current_summary
        finally:
            if self.live_display:
                self.live_display.stop()

    async def _setup_progress_display(self, batch_request: BatchApprovalRequest):
        """Setup live progress display."""
        self.progress_tracker = Progress(
            TextColumn("[bold blue]Executing batch operations...", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TextColumn("{task.completed}/{task.total} operations"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

        # Add progress task
        self.progress_task = self.progress_tracker.add_task(
            "batch_execution", total=len(batch_request.operations)
        )

        # Create live display layout
        layout = Layout()
        layout.split_column(
            Layout(self.progress_tracker, name="progress", size=3),
            Layout(name="details"),
        )

        self.live_display = Live(layout, console=self.console, refresh_per_second=4)
        self.live_display.start()

    async def _execute_operations(
        self, batch_request: BatchApprovalRequest, operation_executor: callable
    ):
        """Execute all operations in the batch with error handling."""
        approved_indices = batch_request.approved_operations

        for i, operation in enumerate(batch_request.operations):
            # Skip if not approved
            if i not in approved_indices:
                self._record_operation_result(
                    i,
                    operation,
                    OperationStatus.SKIPPED,
                    "Operation not approved",
                )
                continue

            # Execute operation with retry logic
            await self._execute_single_operation(i, operation, operation_executor)

            # Update progress display
            if self.progress_tracker:
                self.progress_tracker.update(self.progress_task, advance=1)

                if self.live_display:
                    # Update details section with current status
                    details_content = self._create_status_display()
                    self.live_display.layout["details"].update(details_content)

    async def _execute_single_operation(
        self,
        operation_id: int,
        operation: Operation,
        operation_executor: callable,
    ):
        """Execute a single operation with retry logic."""
        result = OperationResult(
            operation_id=operation_id,
            operation=operation,
            status=OperationStatus.PENDING,
            start_time=datetime.now(),
        )

        self.operation_results[operation_id] = result

        for attempt in range(self.max_retries + 1):
            try:
                result.status = OperationStatus.IN_PROGRESS
                result.retry_count = attempt

                if attempt > 0:
                    result.status = OperationStatus.RETRYING
                    await asyncio.sleep(
                        self.retry_delay * attempt
                    )  # Exponential backoff

                # Execute the operation
                execution_result = await operation_executor(operation)

                # Success
                result.status = OperationStatus.COMPLETED
                result.end_time = datetime.now()
                result.execution_details = execution_result or {}

                self._log_operation_event(
                    operation_id,
                    "COMPLETED",
                    f"Successfully executed {operation.type.value}",
                )
                break

            except Exception as e:
                error_msg = str(e)
                result.error_message = error_msg

                if attempt < self.max_retries:
                    # Will retry
                    self._log_operation_event(
                        operation_id,
                        "RETRY",
                        f"Attempt {attempt + 1} failed: {error_msg}. Retrying...",
                    )
                    continue
                else:
                    # Final failure
                    result.status = OperationStatus.FAILED
                    result.end_time = datetime.now()

                    self._log_operation_event(
                        operation_id,
                        "FAILED",
                        f"Operation failed after {attempt + 1} attempts: {error_msg}",
                    )
                    break

    def _record_operation_result(
        self,
        operation_id: int,
        operation: Operation,
        status: OperationStatus,
        message: Optional[str] = None,
    ):
        """Record the result of an operation."""
        result = OperationResult(
            operation_id=operation_id,
            operation=operation,
            status=status,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error_message=(message if status == OperationStatus.FAILED else None),
        )

        self.operation_results[operation_id] = result
        self._log_operation_event(operation_id, status.value.upper(), message or "")

    def _log_operation_event(self, operation_id: int, event_type: str, message: str):
        """Log an operation event."""
        if self.current_summary:
            self.current_summary.execution_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operation_id": operation_id,
                    "event_type": event_type,
                    "message": message,
                }
            )

    def _create_status_display(self) -> Panel:
        """Create status display for live updates."""
        # Count status
        status_counts = {status: 0 for status in OperationStatus}
        recent_events = []

        for result in self.operation_results.values():
            status_counts[result.status] += 1

            # Collect recent events
            if result.error_message and result.status == OperationStatus.FAILED:
                recent_events.append(
                    f"[red]Failed[/red]: {result.operation.type.value} - {result.error_message}"
                )
            elif result.status == OperationStatus.COMPLETED:
                recent_events.append(
                    f"[green]Completed[/green]: {result.operation.type.value}"
                )

        # Create status table
        status_table = Table.grid(padding=1)
        status_table.add_column("Status", style="bold")
        status_table.add_column("Count", justify="right")

        for status, count in status_counts.items():
            if count > 0:
                color = {
                    OperationStatus.COMPLETED: "green",
                    OperationStatus.FAILED: "red",
                    OperationStatus.IN_PROGRESS: "yellow",
                    OperationStatus.RETRYING: "orange",
                    OperationStatus.SKIPPED: "dim",
                }.get(status, "white")

                status_table.add_row(
                    f"[{color}]{status.value.title()}[/{color}]",
                    f"[{color}]{count}[/{color}]",
                )

        # Recent events
        events_text = (
            "\n".join(recent_events[-5:])
            if recent_events
            else "[dim]No recent events[/dim]"
        )

        content = f"{status_table}\n\n[bold]Recent Events:[/bold]\n{events_text}"

        return Panel(
            content, title="[bold]Execution Status[/bold]", border_style="blue"
        )

    def _calculate_final_results(self):
        """Calculate final batch execution results."""
        if not self.current_summary:
            return

        # Count results
        for result in self.operation_results.values():
            if result.status == OperationStatus.COMPLETED:
                self.current_summary.successful_operations += 1
            elif result.status == OperationStatus.FAILED:
                self.current_summary.failed_operations += 1
                if result.error_message:
                    self.current_summary.error_summary.append(
                        f"Operation {result.operation_id}: {result.error_message}"
                    )
            elif result.status == OperationStatus.SKIPPED:
                self.current_summary.skipped_operations += 1

            if result.retry_count > 0:
                self.current_summary.retried_operations += 1

        # Determine overall result
        if self.current_summary.failed_operations == 0:
            self.current_summary.result = BatchOperationResult.SUCCESS
        elif self.current_summary.successful_operations > 0:
            self.current_summary.result = BatchOperationResult.PARTIAL_SUCCESS
        else:
            self.current_summary.result = BatchOperationResult.FAILURE

    async def _save_execution_results(self):
        """Save execution results to files."""
        if not self.current_summary or not self.save_results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id_short = self.current_summary.batch_id[:8]

        # Save JSON summary
        json_path = self.results_directory / f"batch_{batch_id_short}_{timestamp}.json"
        json_data = {
            "summary": {
                "batch_id": self.current_summary.batch_id,
                "start_time": self.current_summary.start_time.isoformat(),
                "end_time": (
                    self.current_summary.end_time.isoformat()
                    if self.current_summary.end_time
                    else None
                ),
                "duration_seconds": (
                    self.current_summary.duration.total_seconds()
                    if self.current_summary.duration
                    else None
                ),
                "total_operations": self.current_summary.total_operations,
                "successful_operations": self.current_summary.successful_operations,
                "failed_operations": self.current_summary.failed_operations,
                "skipped_operations": self.current_summary.skipped_operations,
                "retried_operations": self.current_summary.retried_operations,
                "result": self.current_summary.result.value,
                "success_rate": self.current_summary.success_rate,
                "error_summary": self.current_summary.error_summary,
            },
            "operations": [
                {
                    "id": result.operation_id,
                    "type": result.operation.type.value,
                    "description": result.operation.description,
                    "status": result.status.value,
                    "start_time": (
                        result.start_time.isoformat() if result.start_time else None
                    ),
                    "end_time": (
                        result.end_time.isoformat() if result.end_time else None
                    ),
                    "duration_seconds": (
                        result.duration.total_seconds() if result.duration else None
                    ),
                    "retry_count": result.retry_count,
                    "error_message": result.error_message,
                    "execution_details": result.execution_details,
                }
                for result in self.operation_results.values()
            ],
            "execution_log": self.current_summary.execution_log,
        }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Batch execution results saved to {json_path}")

    def display_execution_summary(
        self, summary: Optional[BatchExecutionSummary] = None
    ):
        """Display comprehensive execution summary."""
        if not summary:
            summary = self.current_summary

        if not summary:
            self.console.print("[red]No execution summary available[/red]")
            return

        # Main summary panel
        summary_table = Table.grid(padding=1)
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="bright_white")

        summary_table.add_row("Batch ID:", summary.batch_id[:12] + "...")
        summary_table.add_row(
            "Execution Time:",
            str(summary.duration) if summary.duration else "Unknown",
        )
        summary_table.add_row("Total Operations:", str(summary.total_operations))
        summary_table.add_row(
            "Successful:", f"[green]{summary.successful_operations}[/green]"
        )
        summary_table.add_row("Failed:", f"[red]{summary.failed_operations}[/red]")
        summary_table.add_row("Skipped:", f"[dim]{summary.skipped_operations}[/dim]")
        summary_table.add_row(
            "Retried:", f"[yellow]{summary.retried_operations}[/yellow]"
        )
        summary_table.add_row(
            "Success Rate:", f"[green]{summary.success_rate:.1f}%[/green]"
        )
        summary_table.add_row("Result:", self._format_batch_result(summary.result))

        summary_panel = Panel(
            summary_table,
            title="[bold]Batch Execution Summary[/bold]",
            border_style=(
                "green" if summary.result == BatchOperationResult.SUCCESS else "red"
            ),
        )

        self.console.print(summary_panel)

        # Error details if any
        if summary.error_summary:
            error_text = "\n".join([f"• {error}" for error in summary.error_summary])
            error_panel = Panel(
                error_text,
                title="[bold red]Errors[/bold red]",
                border_style="red",
            )
            self.console.print(error_panel)

        # Operation details table
        if self.operation_results:
            self._display_operation_details()

    def _display_operation_details(self):
        """Display detailed operation results table."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", width=4)
        table.add_column("Type", width=12)
        table.add_column("Status", width=10, justify="center")
        table.add_column("Duration", width=8, justify="right")
        table.add_column("Retries", width=7, justify="center")
        table.add_column("Error", width=40)

        for result in sorted(
            self.operation_results.values(), key=lambda x: x.operation_id
        ):
            status_display = self._format_operation_status(result.status)
            duration_display = (
                f"{result.duration.total_seconds():.1f}s" if result.duration else "N/A"
            )
            retry_display = str(result.retry_count) if result.retry_count > 0 else "-"
            error_display = (
                result.error_message[:40] + "..."
                if result.error_message and len(result.error_message) > 40
                else (result.error_message or "")
            )

            table.add_row(
                str(result.operation_id + 1),
                result.operation.type.value.replace("_", " ").title(),
                status_display,
                duration_display,
                retry_display,
                error_display,
            )

        self.console.print("\n[bold]Operation Details:[/bold]")
        self.console.print(table)

    def _format_batch_result(self, result: BatchOperationResult) -> str:
        """Format batch result with colors."""
        color_map = {
            BatchOperationResult.SUCCESS: "green",
            BatchOperationResult.PARTIAL_SUCCESS: "yellow",
            BatchOperationResult.FAILURE: "red",
            BatchOperationResult.CANCELLED: "dim",
        }
        color = color_map.get(result, "white")
        return f"[{color}]{result.value.replace('_', ' ').title()}[/{color}]"

    def _format_operation_status(self, status: OperationStatus) -> str:
        """Format operation status with colors."""
        color_map = {
            OperationStatus.COMPLETED: "green",
            OperationStatus.FAILED: "red",
            OperationStatus.SKIPPED: "dim",
            OperationStatus.IN_PROGRESS: "yellow",
            OperationStatus.RETRYING: "orange",
        }
        color = color_map.get(status, "white")
        symbol_map = {
            OperationStatus.COMPLETED: "✓",
            OperationStatus.FAILED: "✗",
            OperationStatus.SKIPPED: "○",
            OperationStatus.IN_PROGRESS: "●",
            OperationStatus.RETRYING: "⟳",
        }
        symbol = symbol_map.get(status, "?")
        return f"[{color}]{symbol} {status.value.title()}[/{color}]"

    def export_results(
        self, format_type: str = "json", output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Export batch results to file.

        Args:
            format_type: Export format ('json' or 'csv')
            output_path: Optional custom output path

        Returns:
            Path to exported file or None if export failed
        """
        if not self.current_summary:
            self.console.print("[red]No results to export[/red]")
            return None

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_id_short = self.current_summary.batch_id[:8]
            filename = f"batch_results_{batch_id_short}_{timestamp}.{format_type}"
            output_path = Path.cwd() / filename

        try:
            if format_type.lower() == "json":
                return self._export_json(output_path)
            elif format_type.lower() == "csv":
                return self._export_csv(output_path)
            else:
                self.console.print(
                    f"[red]Unsupported export format: {format_type}[/red]"
                )
                return None
        except Exception as e:
            self.console.print(f"[red]Export failed: {e}[/red]")
            return None

    def _export_json(self, output_path: Path) -> Path:
        """Export results as JSON."""
        data = {
            "batch_summary": {
                "batch_id": self.current_summary.batch_id,
                "result": self.current_summary.result.value,
                "success_rate": self.current_summary.success_rate,
                "execution_time": (
                    str(self.current_summary.duration)
                    if self.current_summary.duration
                    else None
                ),
                "operations_summary": {
                    "total": self.current_summary.total_operations,
                    "successful": self.current_summary.successful_operations,
                    "failed": self.current_summary.failed_operations,
                    "skipped": self.current_summary.skipped_operations,
                    "retried": self.current_summary.retried_operations,
                },
            },
            "operation_results": [
                {
                    "id": result.operation_id,
                    "type": result.operation.type.value,
                    "status": result.status.value,
                    "description": result.operation.description,
                    "duration": (str(result.duration) if result.duration else None),
                    "retry_count": result.retry_count,
                    "error": result.error_message,
                }
                for result in self.operation_results.values()
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def _export_csv(self, output_path: Path) -> Path:
        """Export results as CSV."""
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Operation ID",
                    "Type",
                    "Description",
                    "Status",
                    "Duration (s)",
                    "Retry Count",
                    "Error Message",
                ]
            )

            # Write data rows
            for result in sorted(
                self.operation_results.values(), key=lambda x: x.operation_id
            ):
                writer.writerow(
                    [
                        result.operation_id + 1,
                        result.operation.type.value,
                        result.operation.description or "",
                        result.status.value,
                        (result.duration.total_seconds() if result.duration else ""),
                        result.retry_count,
                        result.error_message or "",
                    ]
                )

        return output_path


def create_batch_monitor(
    console: Optional[Console] = None, **kwargs
) -> BatchOperationMonitor:
    """
    Create a batch operation monitor with default configuration.

    Args:
        console: Rich console for display
        **kwargs: Additional monitor configuration options

    Returns:
        Configured BatchOperationMonitor instance
    """
    return BatchOperationMonitor(console=console, **kwargs)


async def simulate_operation_executor(operation: Operation) -> Dict[str, Any]:
    """
    Simulate operation execution for testing purposes.

    Args:
        operation: Operation to simulate

    Returns:
        Simulated execution result
    """
    # Simulate execution time
    await asyncio.sleep(0.1)

    # Simulate occasional failures for testing
    import random

    if random.random() < 0.1:  # 10% failure rate
        raise Exception(f"Simulated failure for {operation.type.value}")

    return {
        "simulated": True,
        "operation_type": operation.type.value,
        "timestamp": datetime.now().isoformat(),
    }
