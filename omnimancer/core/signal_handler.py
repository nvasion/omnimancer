"""
Signal handling for graceful shutdown of Omnimancer.

This module provides signal handling functionality to allow users to
exit Omnimancer gracefully with Ctrl+C, including cancellation of active
operations and proper resource cleanup.
"""

import asyncio
import logging
import os
import signal
from typing import Set

logger = logging.getLogger(__name__)


class SignalHandler:
    """
    Handles signal-based shutdown for Omnimancer application.

    This class manages SIGINT (Ctrl+C) and SIGTERM signals to provide
    graceful shutdown functionality including:
    - Cancellation of active operations
    - Cleanup of agent resources
    - Force exit on double Ctrl+C
    """

    def __init__(self, agent_engine=None):
        """
        Initialize the SignalHandler.

        Args:
            agent_engine: Optional agent engine instance for cleanup
        """
        self.agent_engine = agent_engine
        self.shutdown_event = asyncio.Event()
        self.active_operations: Set[asyncio.Task] = set()
        self.shutdown_in_progress = False
        logger.debug("SignalHandler initialized")

    def setup_signal_handlers(self) -> None:
        """
        Register SIGINT (Ctrl+C) and SIGTERM handlers.

        This method sets up the signal handlers that will be called
        when the user presses Ctrl+C or when the process receives
        a SIGTERM signal.
        """
        try:
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGTERM, self._handle_interrupt)
            logger.info("Signal handlers registered for SIGINT and SIGTERM")
        except ValueError as e:
            # This can happen if signal handlers are set up on a thread
            # other than the main thread
            logger.warning(f"Failed to register signal handlers: {e}")

    def _handle_interrupt(self, signum: int, frame) -> None:
        """
        Handle interrupt signals gracefully.

        This method is called when SIGINT or SIGTERM is received.
        On the first signal, it initiates graceful shutdown.
        On the second signal, it forces immediate exit.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.debug(f"Received {signal_name} signal")

        if self.shutdown_in_progress:
            # Force exit on second Ctrl+C
            print("\nForce exit requested. Terminating immediately...")
            logger.info("Force exit requested, terminating immediately")
            os._exit(1)

        self.shutdown_in_progress = True
        print("\nShutdown requested. Canceling operations...")
        logger.info("Graceful shutdown initiated")

        # Create shutdown task in the event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._graceful_shutdown())
            else:
                logger.warning("Event loop is not running, cannot create shutdown task")
        except RuntimeError:
            # No event loop running, exit immediately
            logger.warning("No event loop available, exiting immediately")
            os._exit(1)

    async def _graceful_shutdown(self) -> None:
        """
        Cancel active operations and shutdown gracefully.

        This method:
        1. Cancels all active operations
        2. Waits for operations to complete (with timeout)
        3. Cleans up agent resources
        4. Sets the shutdown event
        """
        logger.info("Starting graceful shutdown process")

        # Cancel all active operations
        cancelled_count = 0
        for task in self.active_operations.copy():
            if not task.done():
                task.cancel()
                cancelled_count += 1

        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} active operations")

        # Wait for operations to complete or timeout
        if self.active_operations:
            print(
                f"Waiting for {len(self.active_operations)} operations to complete..."
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_operations, return_exceptions=True),
                    timeout=5.0,
                )
                logger.info("All operations completed successfully")
            except asyncio.TimeoutError:
                print("Timeout waiting for operations to complete. Forcing shutdown...")
                logger.warning("Timeout during graceful shutdown, forcing exit")

        # Cleanup agent resources
        if self.agent_engine and hasattr(self.agent_engine, "cleanup"):
            try:
                await self.agent_engine.cleanup()
                logger.info("Agent engine cleanup completed")
            except Exception as e:
                logger.error(f"Error during agent engine cleanup: {e}")

        # Signal shutdown completion
        self.shutdown_event.set()
        logger.info("Graceful shutdown completed")

    def register_operation(self, task: asyncio.Task) -> None:
        """
        Register an active operation for tracking.

        Registered operations will be cancelled during shutdown
        and automatically removed when they complete.

        Args:
            task: Asyncio task to track
        """
        if task.done():
            logger.debug("Attempted to register already completed task")
            return

        self.active_operations.add(task)
        task.add_done_callback(self.active_operations.discard)
        logger.debug(
            f"Registered operation, total active: {len(self.active_operations)}"
        )

    async def wait_for_shutdown(self) -> None:
        """
        Wait for shutdown signal.

        This method blocks until a shutdown signal is received
        and the graceful shutdown process is complete.
        """
        logger.debug("Waiting for shutdown signal")
        await self.shutdown_event.wait()
        logger.info("Shutdown signal received")
