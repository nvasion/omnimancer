"""
Cancellation handler for Omnimancer operations.

This module provides cancellation support for long-running operations
with user-friendly feedback and graceful operation termination.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

from rich.console import Console

logger = logging.getLogger(__name__)

# Global registry for active cancellation handler
_active_cancellation_handler: Optional["CancellationHandler"] = None


def get_active_cancellation_handler() -> Optional["CancellationHandler"]:
    """Get the currently active cancellation handler."""
    return _active_cancellation_handler


def set_active_cancellation_handler(handler: Optional["CancellationHandler"]):
    """Set the currently active cancellation handler."""
    global _active_cancellation_handler
    _active_cancellation_handler = handler


class CancellationHandler:
    """
    Handles operation cancellation with user feedback.

    This class provides a simple but effective cancellation system
    that integrates with the existing signal handling infrastructure.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the cancellation handler.

        Args:
            console: Rich console instance for user feedback
        """
        self.console = console or Console()
        self.active_operation: Optional[asyncio.Task] = None
        self.is_listening = False
        self.status_display = None
        self.is_paused = False
        logger.debug("CancellationHandler initialized")

    async def start_cancellable_operation(
        self,
        operation: Callable,
        status_message: str = "Processing...",
        cancellation_message: str = "Operation cancelled by user",
        signal_handler=None,
        status_callback: Optional[Callable[[], str]] = None,
    ) -> Any:
        """
        Start an operation with cancellation support.

        This method wraps an operation to provide cancellation feedback
        and integrates with the signal handler for Ctrl+C support.

        Args:
            operation: Async function to execute
            status_message: Message to show during operation
            cancellation_message: Message to show when cancelled
            status_callback: Optional callback that returns current status

        Returns:
            Result of the operation

        Raises:
            asyncio.CancelledError: If operation was cancelled
        """
        logger.debug(f"Starting cancellable operation: {status_message}")

        # Register this handler as active
        set_active_cancellation_handler(self)

        # Create the operation task
        self.active_operation = asyncio.create_task(operation())
        self.is_listening = True

        # Register with signal handler if provided
        if signal_handler:
            signal_handler.register_operation(self.active_operation)

        try:
            # Show status with cancellation instructions
            # Note: ESC key cancellation is not yet implemented, using Ctrl+C only
            # If we have a status callback, use dynamic status
            if status_callback:

                async def update_status():
                    while not self.active_operation.done():
                        current_status = status_callback()
                        self.console.status(
                            f"[bold green]{current_status} (Press Ctrl+C to cancel)",
                            spinner="dots",
                            refresh=True,
                        )
                        await asyncio.sleep(0.1)

                # Run status updates in parallel with operation
                status_task = asyncio.create_task(update_status())
                try:
                    result = await self.active_operation
                finally:
                    status_task.cancel()
                    try:
                        await status_task
                    except asyncio.CancelledError:
                        pass
            else:
                # Use static status message
                self.status_display = self.console.status(
                    f"[bold green]{status_message} (Press Ctrl+C to cancel)",
                    spinner="dots",
                )
                with self.status_display:
                    result = await self.active_operation

            logger.debug("Cancellable operation completed successfully")
            return result

        except asyncio.CancelledError:
            logger.info("Operation cancelled by user")
            self.console.print(f"[yellow]⚠️  {cancellation_message}[/yellow]")
            raise
        finally:
            self.is_listening = False
            self.active_operation = None
            self.status_display = None
            self.is_paused = False
            # Unregister this handler
            set_active_cancellation_handler(None)

    def is_operation_active(self) -> bool:
        """
        Check if an operation is currently active.

        Returns:
            True if an operation is running
        """
        return self.active_operation is not None and not self.active_operation.done()

    def cancel_active_operation(self) -> bool:
        """
        Cancel the currently active operation.

        Returns:
            True if an operation was cancelled, False otherwise
        """
        if self.is_operation_active():
            logger.info("Cancelling active operation")
            self.active_operation.cancel()
            return True
        return False

    def pause_status_display(self):
        """
        Pause the status display to allow for user interaction.

        This stops the spinner temporarily while preserving the operation.
        """
        if self.status_display and not self.is_paused:
            self.is_paused = True
            try:
                self.status_display.stop()
            except:
                pass
            logger.debug("Status display paused for user interaction")

    def resume_status_display(self):
        """
        Resume the status display after user interaction.
        """
        if self.status_display and self.is_paused and self.is_operation_active():
            self.is_paused = False
            try:
                self.status_display.start()
            except:
                pass
            logger.debug("Status display resumed after user interaction")


class EnhancedStatusDisplay:
    """
    Enhanced status display with better cancellation feedback.

    Provides improved visual feedback for operations that can be cancelled.
    """

    def __init__(self, console: Console):
        """
        Initialize the enhanced status display.

        Args:
            console: Rich console instance
        """
        self.console = console

    def show_thinking_with_cancel(self, message: str = "Thinking") -> None:
        """
        Show a thinking indicator with cancellation instructions.

        Args:
            message: Base message to display
        """
        enhanced_message = (
            f"[bold green]{message}... (Press Ctrl+C to cancel)[/bold green]"
        )
        return self.console.status(enhanced_message, spinner="dots")

    def show_cancellation_notice(self, message: str = "Operation cancelled") -> None:
        """
        Show a cancellation notice to the user.

        Args:
            message: Cancellation message to display
        """
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")

    def show_completion_notice(self, message: str = "Operation completed") -> None:
        """
        Show a completion notice to the user.

        Args:
            message: Completion message to display
        """
        self.console.print(f"[green]✅ {message}[/green]")
