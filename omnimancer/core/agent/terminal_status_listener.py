"""
Terminal Status Listener for Omnimancer Agent Status Streaming.

This module provides a Rich-based terminal UI listener that displays real-time
agent status updates in the terminal using the async status stream system.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, Set

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from .status_core import (
    AgentEvent,
    AgentStatus,
    EventType,
    StatusStreamEvent,
    StatusStreamListener,
)

logger = logging.getLogger(__name__)


class TerminalStatusListener(StatusStreamListener):
    """
    Terminal UI listener for agent status updates using Rich library.

    Displays real-time status updates in a formatted terminal interface
    with live updating components.
    """

    def __init__(
        self,
        listener_id: str = "terminal_ui",
        event_types: Optional[Set[EventType]] = None,
        console: Optional[Console] = None,
        update_frequency: float = 0.5,
    ):
        """
        Initialize terminal status listener.

        Args:
            listener_id: Unique identifier for this listener
            event_types: Set of event types to listen for
            console: Rich console instance (optional)
            update_frequency: How often to update the display (seconds)
        """
        super().__init__(listener_id, event_types)

        self.console = console or Console()
        self.update_frequency = update_frequency

        # Status tracking
        self.agent_statuses: Dict[str, AgentStatus] = {}
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.recent_events: list = []
        self.max_recent_events = 10

        # Display state
        self.live_display: Optional[Live] = None
        self.last_update = 0.0
        self.stats = {
            "events_received": 0,
            "operations_tracked": 0,
            "agents_monitored": 0,
        }

    async def start_live_display(self) -> None:
        """Start the live display interface."""
        if self.live_display is None:
            self.live_display = Live(
                self._generate_display(),
                console=self.console,
                refresh_per_second=1 / self.update_frequency,
                auto_refresh=True,
            )
            self.live_display.start()
            logger.info("Terminal status display started")

    async def stop_live_display(self) -> None:
        """Stop the live display interface."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
            logger.info("Terminal status display stopped")

    async def _process_event(self, stream_event: StatusStreamEvent) -> None:
        """Process incoming status stream events."""
        try:
            event = stream_event.event
            self.stats["events_received"] += 1

            # Update recent events list
            self.recent_events.append(
                {
                    "timestamp": event.timestamp,
                    "type": event.event_type.value,
                    "agent_id": event.agent_id,
                    "priority": stream_event.priority.value,
                    "data": event.data,
                }
            )

            # Keep only recent events
            if len(self.recent_events) > self.max_recent_events:
                self.recent_events.pop(0)

            # Process specific event types
            await self._handle_specific_event(event)

            # Update display if enough time has passed
            current_time = time.time()
            if current_time - self.last_update >= self.update_frequency:
                await self._update_display()
                self.last_update = current_time

        except Exception as e:
            logger.error(f"Error processing event in terminal listener: {e}")

    async def _handle_specific_event(self, event: AgentEvent) -> None:
        """Handle specific event types with appropriate updates."""
        try:
            if event.event_type == EventType.AGENT_STATE_CHANGED:
                # Update agent status
                if event.agent_id and "new_status" in event.data:
                    try:
                        status_str = event.data["new_status"]
                        self.agent_statuses[event.agent_id] = AgentStatus(status_str)
                        self.stats["agents_monitored"] = len(
                            set(self.agent_statuses.keys())
                        )
                    except ValueError:
                        logger.warning(
                            f"Invalid agent status: {event.data['new_status']}"
                        )

            elif event.event_type == EventType.OPERATION_STARTED:
                # Track new operation
                if event.operation_id:
                    self.active_operations[event.operation_id] = {
                        "agent_id": event.agent_id,
                        "type": event.data.get("operation_type", "unknown"),
                        "description": event.data.get("description", "No description"),
                        "start_time": event.timestamp,
                        "progress": 0.0,
                        "status": "running",
                    }
                    self.stats["operations_tracked"] = len(self.active_operations)

            elif event.event_type == EventType.OPERATION_PROGRESS:
                # Update operation progress
                if event.operation_id and event.operation_id in self.active_operations:
                    self.active_operations[event.operation_id].update(
                        {
                            "progress": event.data.get("progress", 0.0),
                            "description": event.data.get(
                                "description",
                                self.active_operations[event.operation_id][
                                    "description"
                                ],
                            ),
                        }
                    )

            elif event.event_type in [
                EventType.OPERATION_COMPLETED,
                EventType.OPERATION_FAILED,
                EventType.OPERATION_CANCELLED,
            ]:
                # Complete operation
                if event.operation_id and event.operation_id in self.active_operations:
                    self.active_operations[event.operation_id].update(
                        {
                            "status": (
                                "completed"
                                if event.event_type == EventType.OPERATION_COMPLETED
                                else "failed"
                            ),
                            "end_time": event.timestamp,
                            "progress": (
                                100.0
                                if event.event_type == EventType.OPERATION_COMPLETED
                                else self.active_operations[event.operation_id][
                                    "progress"
                                ]
                            ),
                        }
                    )

                    # Remove completed operations after a delay (in a real implementation)
                    # For now, we'll keep them for display purposes

        except Exception as e:
            logger.error(f"Error handling specific event {event.event_type}: {e}")

    async def _update_display(self) -> None:
        """Update the live display with current status."""
        if self.live_display:
            try:
                self.live_display.update(self._generate_display())
            except Exception as e:
                logger.error(f"Error updating display: {e}")

    def _generate_display(self) -> Panel:
        """Generate the complete display panel."""
        try:
            # Create main layout
            display_content = []

            # Add system stats
            display_content.append(self._generate_stats_section())

            # Add agent status section
            if self.agent_statuses:
                display_content.append(self._generate_agents_section())

            # Add active operations section
            if self.active_operations:
                display_content.append(self._generate_operations_section())

            # Add recent events section
            if self.recent_events:
                display_content.append(self._generate_events_section())

            # Combine all sections
            content = "\n\n".join(display_content)

            return Panel(
                content,
                title="🤖 Agent Status Monitor",
                border_style="blue",
                title_align="left",
            )

        except Exception as e:
            logger.error(f"Error generating display: {e}")
            return Panel(f"Display Error: {e}", border_style="red")

    def _generate_stats_section(self) -> str:
        """Generate system statistics section."""
        return (
            f"[bold]System Statistics[/bold]\n"
            f"Events Received: {self.stats['events_received']}\n"
            f"Active Operations: {len([op for op in self.active_operations.values() if op['status'] == 'running'])}\n"
            f"Agents Monitored: {self.stats['agents_monitored']}\n"
            f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
        )

    def _generate_agents_section(self) -> str:
        """Generate agent status section."""
        content = "[bold]Agent Status[/bold]\n"

        for agent_id, status in self.agent_statuses.items():
            # Color coding based on status
            if status == AgentStatus.ENABLED:
                color = "green"
                emoji = "🟢"
            elif status == AgentStatus.RUNNING:
                color = "blue"
                emoji = "🔵"
            elif status == AgentStatus.ERROR:
                color = "red"
                emoji = "🔴"
            else:
                color = "yellow"
                emoji = "🟡"

            content += f"{emoji} [{color}]{agent_id}[/{color}]: {status.value}\n"

        return content.rstrip()

    def _generate_operations_section(self) -> str:
        """Generate active operations section."""
        content = "[bold]Active Operations[/bold]\n"

        active_ops = [
            op for op in self.active_operations.values() if op["status"] == "running"
        ]
        if not active_ops:
            content += "[dim]No active operations[/dim]"
        else:
            for op in active_ops[:5]:  # Show max 5 operations
                progress = op["progress"]
                agent_id = op["agent_id"] or "unknown"
                description = (
                    op["description"][:50] + "..."
                    if len(op["description"]) > 50
                    else op["description"]
                )

                # Progress bar representation
                bar_width = 20
                filled = int((progress / 100) * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)

                content += f"[cyan]{agent_id}[/cyan]: {description}\n"
                content += f"  [{progress:5.1f}%] {bar} \n"

        return content.rstrip()

    def _generate_events_section(self) -> str:
        """Generate recent events section."""
        content = "[bold]Recent Events[/bold]\n"

        if not self.recent_events:
            content += "[dim]No recent events[/dim]"
        else:
            for event in self.recent_events[-5:]:  # Show last 5 events
                timestamp = event["timestamp"].strftime("%H:%M:%S")
                event_type = event["type"].replace("_", " ").title()
                agent_id = event["agent_id"] or "system"
                priority = event["priority"]

                # Color coding by priority
                if priority == "critical":
                    color = "red"
                elif priority == "high":
                    color = "orange"
                elif priority == "normal":
                    color = "yellow"
                else:
                    color = "dim"

                content += f"[{color}]{timestamp}[/{color}] [{agent_id}] {event_type}\n"

        return content.rstrip()


# Convenience function to create and start a terminal status listener
async def create_terminal_status_display(
    status_stream: "AsyncStatusStream",
) -> TerminalStatusListener:
    """
    Create and start a terminal status display listener.

    Args:
        status_stream: The status stream to listen to

    Returns:
        The created and started terminal listener
    """
    listener = TerminalStatusListener()

    # Add listener to stream
    await status_stream.add_listener(listener)

    # Start live display
    await listener.start_live_display()

    return listener


# Example usage function
async def example_terminal_monitor():
    """Example of how to set up terminal monitoring."""
    try:
        from .status_manager import get_status_manager

        # Get status system components
        status_manager = get_status_manager()
        # The unified manager handles both status and streaming

        # Initialize the unified manager
        await status_manager.initialize()

        # Create terminal display
        terminal_display = await create_terminal_status_display(status_manager)

        print("Terminal status monitor running. Press Ctrl+C to stop.")

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down terminal monitor...")
        finally:
            await terminal_display.stop_live_display()
            await status_stream.stop()
            await status_manager.shutdown()

    except Exception as e:
        logger.error(f"Error in terminal monitor example: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_terminal_monitor())
