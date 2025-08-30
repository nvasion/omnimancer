"""
Unified Status Management System for Omnimancer Agents.

This module provides a consolidated status manager that combines operation tracking,
event management, and real-time streaming capabilities into a single cohesive system.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .status_core import (
    AgentEvent,
    AgentOperation,
    AgentStatus,
    EventListener,
    EventType,
    OperationStatus,
    StatusDisplayConfig,
    StatusStreamEvent,
    StatusStreamListener,
    StreamMetrics,
    StreamPriority,
)

logger = logging.getLogger(__name__)


class StatusUpdateCollector:
    """Batches status updates for efficient rendering."""

    def __init__(self, batch_size: int = 10, flush_interval: float = 0.1):
        """
        Initialize status update collector.

        Args:
            batch_size: Maximum number of updates to batch
            flush_interval: Maximum time to wait before flushing batch (seconds)
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.updates: List[StatusStreamEvent] = []
        self.last_flush = time.time()
        self._lock = asyncio.Lock()

    async def add_update(self, stream_event: StatusStreamEvent) -> bool:
        """
        Add update to batch.

        Args:
            stream_event: Status stream event to add

        Returns:
            True if batch should be flushed immediately
        """
        async with self._lock:
            self.updates.append(stream_event)

            # Check if we should flush
            now = time.time()
            should_flush = (
                len(self.updates) >= self.batch_size
                or (now - self.last_flush) >= self.flush_interval
                or stream_event.priority
                in [StreamPriority.HIGH, StreamPriority.CRITICAL]
            )

            return should_flush

    async def flush(self) -> List[StatusStreamEvent]:
        """
        Flush and return batched updates.

        Returns:
            List of batched status stream events
        """
        async with self._lock:
            updates = self.updates.copy()
            self.updates.clear()
            self.last_flush = time.time()
            return updates


class UnifiedStatusManager:
    """
    Unified status manager that combines operation tracking, event management,
    and real-time streaming capabilities.

    This class consolidates the functionality previously split between
    AgentStatusManager and AsyncStatusStream.
    """

    def __init__(self, config: Optional[StatusDisplayConfig] = None):
        """
        Initialize the unified status manager.

        Args:
            config: Configuration for the status system (uses defaults if None)
        """
        self.config = config or StatusDisplayConfig()

        # Agent status tracking
        self.agent_statuses: Dict[str, AgentStatus] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Operation tracking
        self.active_operations: Dict[str, AgentOperation] = {}
        self.operation_history: deque = deque(maxlen=self.config.max_operation_history)
        self.operations_by_agent: Dict[str, List[str]] = defaultdict(list)

        # Event system
        self.event_listeners: List[EventListener] = []
        self.event_history: deque = deque(maxlen=self.config.max_event_history)
        self.event_queue: asyncio.Queue = asyncio.Queue()

        # Streaming system
        self.stream_listeners: Dict[str, StatusStreamListener] = {}
        self.stream_event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self.priority_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size // 2
        )

        # Throttling and batching
        self.last_event_time = time.time()
        self.event_count = 0
        self.update_collector = StatusUpdateCollector(
            batch_size=self.config.batch_size,
            flush_interval=self.config.flush_interval,
        )

        # Metrics and monitoring
        self.metrics = StreamMetrics()

        # Statistics tracking
        self.stats = {
            "operations_started": 0,
            "operations_completed": 0,
            "operations_failed": 0,
            "operations_cancelled": 0,
            "events_emitted": 0,
        }

        # Control flags
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Background tasks
        self._event_processor_task: Optional[asyncio.Task] = None
        self._stream_processor_task: Optional[asyncio.Task] = None
        self._distributor_task: Optional[asyncio.Task] = None

        # Thread safety
        self._lock = asyncio.Lock()
        self._shutdown = False

    async def initialize(self) -> None:
        """Initialize the status manager and start all processing tasks."""
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()

        # Start background processing tasks
        self._event_processor_task = asyncio.create_task(self._process_events())
        self._stream_processor_task = asyncio.create_task(self._process_stream_events())
        self._distributor_task = asyncio.create_task(self._distribute_events())

        logger.info("UnifiedStatusManager initialized and started")

    async def shutdown(self) -> None:
        """Shutdown the status manager and cleanup resources."""
        if not self.running:
            return

        self._shutdown = True
        self.running = False
        self.shutdown_event.set()

        # Cancel all active operations
        async with self._lock:
            for operation in self.active_operations.values():
                if operation.is_active:
                    operation.cancel("System shutdown")

        # Stop background tasks
        tasks = [
            self._event_processor_task,
            self._stream_processor_task,
            self._distributor_task,
        ]

        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop all listeners
        for listener in self.event_listeners:
            listener.stop()

        for listener in self.stream_listeners.values():
            listener.stop()

        logger.info("UnifiedStatusManager shutdown complete")

    # Agent Status Management
    async def set_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set agent status and emit state change event.

        Args:
            agent_id: ID of the agent
            status: New status
            metadata: Optional metadata about the status change
        """
        async with self._lock:
            old_status = self.agent_statuses.get(agent_id)
            self.agent_statuses[agent_id] = status

            if metadata:
                self.agent_metadata[agent_id].update(metadata)

            # Emit state change event
            await self._emit_event(
                AgentEvent(
                    event_type=EventType.AGENT_STATE_CHANGED,
                    agent_id=agent_id,
                    data={
                        "old_status": old_status.value if old_status else None,
                        "new_status": status.value,
                        "metadata": metadata or {},
                    },
                    source="UnifiedStatusManager",
                )
            )

    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get current status of an agent."""
        return self.agent_statuses.get(agent_id)

    def get_agent_metadata(self, agent_id: str) -> Dict[str, Any]:
        """Get metadata for an agent."""
        return self.agent_metadata.get(agent_id, {}).copy()

    # Operation Management
    async def start_operation(self, operation: AgentOperation) -> str:
        """
        Start tracking a new operation.

        Args:
            operation: The operation to track

        Returns:
            The operation ID
        """
        async with self._lock:
            operation.status = OperationStatus.RUNNING
            operation.start_time = operation.start_time or time.time()

            self.active_operations[operation.operation_id] = operation

            if operation.agent_id:
                self.operations_by_agent[operation.agent_id].append(
                    operation.operation_id
                )

            self.stats["operations_started"] += 1

            # Emit operation started event
            await self._emit_event(
                AgentEvent(
                    event_type=EventType.OPERATION_STARTED,
                    agent_id=operation.agent_id,
                    operation_id=operation.operation_id,
                    data={
                        "operation_type": operation.operation_type.value,
                        "description": operation.description,
                        "metadata": operation.metadata,
                    },
                    source="UnifiedStatusManager",
                )
            )

            return operation.operation_id

    async def update_operation_progress(
        self,
        operation_id: str,
        progress: float,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update operation progress.

        Args:
            operation_id: ID of the operation
            progress: Progress percentage (0-100)
            description: Optional progress description
            metadata: Optional additional metadata
        """
        async with self._lock:
            operation = self.active_operations.get(operation_id)
            if not operation:
                return

            operation.update_progress(progress, description)
            if metadata:
                operation.metadata.update(metadata)

            # Emit progress event
            await self._emit_event(
                AgentEvent(
                    event_type=EventType.OPERATION_PROGRESS,
                    agent_id=operation.agent_id,
                    operation_id=operation_id,
                    data={
                        "progress": progress,
                        "description": description or operation.description,
                        "metadata": metadata or {},
                    },
                    source="UnifiedStatusManager",
                )
            )

    async def complete_operation(
        self,
        operation_id: str,
        result_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark operation as completed.

        Args:
            operation_id: ID of the operation
            result_metadata: Optional result metadata
        """
        async with self._lock:
            operation = self.active_operations.get(operation_id)
            if not operation:
                return

            operation.complete(result_metadata)
            self._move_to_history(operation)

            self.stats["operations_completed"] += 1

            # Emit completion event
            await self._emit_event(
                AgentEvent(
                    event_type=EventType.OPERATION_COMPLETED,
                    agent_id=operation.agent_id,
                    operation_id=operation_id,
                    data={
                        "duration": (
                            operation.duration.total_seconds()
                            if operation.duration
                            else 0
                        ),
                        "result_metadata": result_metadata or {},
                    },
                    source="UnifiedStatusManager",
                )
            )

    async def fail_operation(
        self,
        operation_id: str,
        error: str,
        error_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark operation as failed.

        Args:
            operation_id: ID of the operation
            error: Error description
            error_metadata: Optional error metadata
        """
        async with self._lock:
            operation = self.active_operations.get(operation_id)
            if not operation:
                return

            operation.fail(error, error_metadata)
            self._move_to_history(operation)

            self.stats["operations_failed"] += 1

            # Emit failure event
            await self._emit_event(
                AgentEvent(
                    event_type=EventType.OPERATION_FAILED,
                    agent_id=operation.agent_id,
                    operation_id=operation_id,
                    data={
                        "error": error,
                        "duration": (
                            operation.duration.total_seconds()
                            if operation.duration
                            else 0
                        ),
                        "error_metadata": error_metadata or {},
                    },
                    source="UnifiedStatusManager",
                )
            )

    async def cancel_operation(
        self, operation_id: str, reason: Optional[str] = None
    ) -> None:
        """
        Cancel an operation.

        Args:
            operation_id: ID of the operation
            reason: Optional cancellation reason
        """
        async with self._lock:
            operation = self.active_operations.get(operation_id)
            if not operation:
                return

            operation.cancel(reason)
            self._move_to_history(operation)

            self.stats["operations_cancelled"] += 1

            # Emit cancellation event
            await self._emit_event(
                AgentEvent(
                    event_type=EventType.OPERATION_CANCELLED,
                    agent_id=operation.agent_id,
                    operation_id=operation_id,
                    data={
                        "reason": reason or "No reason provided",
                        "duration": (
                            operation.duration.total_seconds()
                            if operation.duration
                            else 0
                        ),
                    },
                    source="UnifiedStatusManager",
                )
            )

    def get_operation(self, operation_id: str) -> Optional[AgentOperation]:
        """Get operation by ID."""
        return self.active_operations.get(operation_id)

    def get_active_operations(
        self, agent_id: Optional[str] = None
    ) -> List[AgentOperation]:
        """Get list of active operations, optionally filtered by agent."""
        operations = list(self.active_operations.values())
        if agent_id:
            operations = [op for op in operations if op.agent_id == agent_id]
        return operations

    def get_operation_history(
        self, agent_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[AgentOperation]:
        """Get operation history, optionally filtered by agent."""
        history = list(self.operation_history)
        if agent_id:
            history = [op for op in history if op.agent_id == agent_id]
        if limit:
            history = history[-limit:]
        return history

    # Event System
    def add_event_listener(self, listener: EventListener) -> None:
        """Add an event listener."""
        self.event_listeners.append(listener)

    def remove_event_listener(self, listener: EventListener) -> None:
        """Remove an event listener."""
        if listener in self.event_listeners:
            self.event_listeners.remove(listener)
            listener.stop()

    # Streaming System
    async def emit_stream_event(
        self,
        event: AgentEvent,
        priority: StreamPriority = StreamPriority.NORMAL,
    ) -> bool:
        """
        Emit an event to the status stream.

        Args:
            event: Agent event to emit
            priority: Priority level for the event

        Returns:
            True if event was queued successfully
        """
        if not self.running:
            return False

        try:
            stream_event = StatusStreamEvent(event=event, priority=priority)

            # Use priority queue for high/critical events
            if priority in [StreamPriority.HIGH, StreamPriority.CRITICAL]:
                await self.priority_queue.put(stream_event)
            else:
                await self.stream_event_queue.put(stream_event)

            return True

        except asyncio.QueueFull:
            self.metrics.events_dropped += 1
            logger.warning(
                f"Status stream queue full, dropping event: {event.event_type}"
            )
            return False
        except Exception as e:
            logger.error(f"Error emitting stream event: {e}")
            return False

    async def add_stream_listener(self, listener: StatusStreamListener) -> bool:
        """
        Add a listener to the status stream.

        Args:
            listener: Status stream listener to add

        Returns:
            True if listener was added successfully
        """
        if len(self.stream_listeners) >= self.config.max_listeners:
            logger.warning(f"Maximum listeners ({self.config.max_listeners}) reached")
            return False

        async with self._lock:
            if listener.listener_id in self.stream_listeners:
                logger.warning(f"Listener {listener.listener_id} already exists")
                return False

            self.stream_listeners[listener.listener_id] = listener
            self.metrics.listeners_count = len(self.stream_listeners)

            logger.debug(f"Added stream listener: {listener.listener_id}")
            return True

    async def remove_stream_listener(self, listener_id: str) -> bool:
        """
        Remove a listener from the status stream.

        Args:
            listener_id: ID of listener to remove

        Returns:
            True if listener was removed successfully
        """
        async with self._lock:
            if listener_id not in self.stream_listeners:
                return False

            # Stop the listener
            self.stream_listeners[listener_id].stop()

            # Clean up
            del self.stream_listeners[listener_id]
            self.metrics.listeners_count = len(self.stream_listeners)

            logger.debug(f"Removed stream listener: {listener_id}")
            return True

    # Statistics and Health
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self.stats,
            "active_operations": len(self.active_operations),
            "total_agents": len(self.agent_statuses),
            "enabled_agents": sum(
                1
                for status in self.agent_statuses.values()
                if status == AgentStatus.ENABLED
            ),
            "event_listeners": len(self.event_listeners),
            "stream_listeners": len(self.stream_listeners),
        }

    def get_metrics(self) -> StreamMetrics:
        """Get current stream metrics."""
        self.metrics.queue_size = (
            self.stream_event_queue.qsize() + self.priority_queue.qsize()
        )
        return self.metrics

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the status system."""
        metrics = self.get_metrics()

        return {
            "running": self.running,
            "queue_size": metrics.queue_size,
            "queue_full": metrics.queue_size >= self.config.max_queue_size * 0.9,
            "listeners_count": metrics.listeners_count,
            "events_processed": metrics.events_processed,
            "events_dropped": metrics.events_dropped,
            "avg_processing_time": metrics.avg_processing_time,
            "last_update": (
                metrics.last_update.isoformat() if metrics.last_update else None
            ),
            "healthy": (
                self.running
                and metrics.queue_size < self.config.max_queue_size * 0.9
                and metrics.avg_processing_time < 0.1
            ),
        }

    # Private Methods
    async def _emit_event(
        self,
        event: AgentEvent,
        stream_priority: Optional[StreamPriority] = None,
    ) -> None:
        """
        Emit an event to both event listeners and stream listeners.

        Args:
            event: The event to emit
            stream_priority: Priority level for status stream (if available)
        """
        self.stats["events_emitted"] += 1
        self.event_history.append(event)

        # Add to event processing queue
        await self.event_queue.put(event)

        # Emit to stream if priority is specified
        if stream_priority is not None:
            await self.emit_stream_event(event, stream_priority)
        else:
            # Determine priority based on event type
            if event.event_type in [
                EventType.ERROR_OCCURRED,
                EventType.OPERATION_FAILED,
            ]:
                stream_priority = StreamPriority.HIGH
            elif event.event_type in [
                EventType.AGENT_STATE_CHANGED,
                EventType.APPROVAL_REQUESTED,
            ]:
                stream_priority = StreamPriority.NORMAL
            else:
                stream_priority = StreamPriority.LOW

            await self.emit_stream_event(event, stream_priority)

    async def _process_events(self) -> None:
        """Background task to process events for event listeners."""
        logger.debug("Event processor started")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Wait for event with timeout to allow shutdown checking
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                # Process event for all listeners
                for listener in self.event_listeners[
                    :
                ]:  # Copy list to avoid modification during iteration
                    try:
                        await listener.handle_event(event)
                    except Exception as e:
                        logger.error(
                            f"Error in event listener {type(listener).__name__}: {e}"
                        )

            except asyncio.TimeoutError:
                continue  # Check shutdown flag
            except Exception as e:
                logger.error(f"Error processing event: {e}")

        logger.debug("Event processor stopped")

    async def _process_stream_events(self) -> None:
        """Background task to process stream events."""
        logger.debug("Stream event processor started")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Process priority events first
                try:
                    stream_event = await asyncio.wait_for(
                        self.priority_queue.get(), timeout=0.01
                    )
                    await self._handle_stream_event(stream_event)
                    continue
                except asyncio.TimeoutError:
                    pass

                # Process normal events with throttling
                try:
                    stream_event = await asyncio.wait_for(
                        self.stream_event_queue.get(), timeout=0.1
                    )

                    # Apply throttling for normal events
                    if self.config.throttle_rate > 0:
                        now = time.time()
                        time_since_last = now - self.last_event_time
                        min_interval = 1.0 / self.config.throttle_rate

                        if time_since_last < min_interval:
                            await asyncio.sleep(min_interval - time_since_last)

                    await self._handle_stream_event(stream_event)
                    self.last_event_time = time.time()

                except asyncio.TimeoutError:
                    # Timeout is normal, just continue
                    continue

            except Exception as e:
                logger.error(f"Error in stream event processor: {e}")
                await asyncio.sleep(0.1)

        logger.debug("Stream event processor stopped")

    async def _handle_stream_event(self, stream_event: StatusStreamEvent) -> None:
        """Handle a single stream event."""
        start_time = time.time()

        try:
            # Update metrics
            self.metrics.queue_size = (
                self.stream_event_queue.qsize() + self.priority_queue.qsize()
            )

            # Add to collector for batching
            should_flush = await self.update_collector.add_update(stream_event)

            # Distribute to listeners if we should flush
            if should_flush:
                updates = await self.update_collector.flush()
                for update in updates:
                    await self._distribute_to_stream_listeners(update)

            self.metrics.events_processed += 1

        except Exception as e:
            logger.error(f"Error handling stream event: {e}")
        finally:
            processing_time = time.time() - start_time
            self.metrics.update_processing_time(processing_time)
            self.metrics.last_update = datetime.now()

    async def _distribute_events(self) -> None:
        """Background task to distribute events to stream listeners."""
        logger.debug("Event distributor started")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Periodically flush any remaining updates
                await asyncio.sleep(0.05)  # 20 times per second

                updates = await self.update_collector.flush()
                for update in updates:
                    await self._distribute_to_stream_listeners(update)

            except Exception as e:
                logger.error(f"Error in event distributor: {e}")
                await asyncio.sleep(0.1)

        logger.debug("Event distributor stopped")

    async def _distribute_to_stream_listeners(
        self, stream_event: StatusStreamEvent
    ) -> None:
        """Distribute event to all relevant stream listeners."""
        if not self.stream_listeners:
            return

        # Create list of listeners to avoid modification during iteration
        listeners_list = list(self.stream_listeners.values())

        # Distribute to listeners concurrently
        tasks = []
        for listener in listeners_list:
            if listener.active:
                task = asyncio.create_task(listener.handle_event(stream_event))
                tasks.append(task)

        # Wait for all listeners to process (with timeout)
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=1.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some stream listeners timed out processing event")

    def _move_to_history(self, operation: AgentOperation) -> None:
        """Move completed operation from active to history."""
        if operation.operation_id in self.active_operations:
            del self.active_operations[operation.operation_id]

        self.operation_history.append(operation)

        # Clean up agent operation list
        if (
            operation.agent_id
            and operation.operation_id in self.operations_by_agent[operation.agent_id]
        ):
            self.operations_by_agent[operation.agent_id].remove(operation.operation_id)


# Global instance management
_global_status_manager: Optional[UnifiedStatusManager] = None


def get_status_manager() -> UnifiedStatusManager:
    """Get the global status manager instance."""
    global _global_status_manager
    if _global_status_manager is None:
        _global_status_manager = UnifiedStatusManager()
    return _global_status_manager


def set_status_manager(manager: UnifiedStatusManager) -> None:
    """Set the global status manager instance."""
    global _global_status_manager
    _global_status_manager = manager


async def initialize_status_system(
    config: Optional[StatusDisplayConfig] = None,
) -> UnifiedStatusManager:
    """
    Initialize the global status system.

    Args:
        config: Optional configuration for the status system

    Returns:
        The initialized status manager
    """
    if config:
        manager = UnifiedStatusManager(config)
        set_status_manager(manager)
    else:
        manager = get_status_manager()

    await manager.initialize()
    return manager


async def shutdown_status_system() -> None:
    """Shutdown the global status system."""
    global _global_status_manager
    if _global_status_manager:
        await _global_status_manager.shutdown()
        _global_status_manager = None
