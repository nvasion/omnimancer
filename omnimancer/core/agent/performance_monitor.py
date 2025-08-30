"""
Agent Performance Monitoring System for Omnimancer.

This module provides comprehensive performance monitoring for agent operations,
including token usage tracking, context efficiency measurement, operation timing,
and resource utilization metrics with optimization suggestions.
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


from .persona import PersonaManager, get_persona_manager
from .status_core import (
    AgentEvent,
    EventType,
    OperationType,
)
from .status_manager import UnifiedStatusManager as AgentStatusManager
from .status_manager import (
    get_status_manager,
)

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""

    TOKEN_USAGE = "token_usage"
    OPERATION_TIMING = "operation_timing"
    CONTEXT_EFFICIENCY = "context_efficiency"
    RESOURCE_UTILIZATION = "resource_utilization"
    SUCCESS_RATE = "success_rate"
    COST_TRACKING = "cost_tracking"
    PERSONA_PERFORMANCE = "persona_performance"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class OptimizationType(Enum):
    """Types of optimization suggestions."""

    PROVIDER_SWITCHING = "provider_switching"
    CONTEXT_REDUCTION = "context_reduction"
    PERSONA_OPTIMIZATION = "persona_optimization"
    RESOURCE_TUNING = "resource_tuning"
    COST_OPTIMIZATION = "cost_optimization"


@dataclass
class TokenUsageMetrics:
    """Token usage metrics for a specific operation or time period."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    provider_name: str = ""
    model_name: str = ""
    operation_type: OperationType = OperationType.API_CALL
    timestamp: datetime = field(default_factory=datetime.now)
    cost_estimate: float = 0.0

    @property
    def efficiency_ratio(self) -> float:
        """Calculate token efficiency ratio (output/total)."""
        return (
            (self.output_tokens / self.total_tokens) if self.total_tokens > 0 else 0.0
        )


@dataclass
class OperationTiming:
    """Timing metrics for agent operations."""

    operation_id: str
    operation_type: OperationType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    context_load_time_ms: float = 0.0
    api_call_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    queue_wait_time_ms: float = 0.0

    @property
    def is_complete(self) -> bool:
        """Check if timing is complete."""
        return self.end_time is not None

    def complete_timing(self, end_time: Optional[datetime] = None) -> None:
        """Complete the timing measurement."""
        if not end_time:
            end_time = datetime.now()
        self.end_time = end_time
        self.duration_ms = (end_time - self.start_time).total_seconds() * 1000


@dataclass
class ContextMetrics:
    """Context efficiency metrics."""

    context_size_chars: int = 0
    context_size_tokens: int = 0
    useful_context_ratio: float = 0.0  # % of context that was actually used
    context_compression_ratio: float = 0.0
    context_load_time_ms: float = 0.0
    context_type: str = ""  # "file", "conversation", "documentation", etc.

    @property
    def efficiency_score(self) -> float:
        """Calculate overall context efficiency score (0-100)."""
        base_score = self.useful_context_ratio * 100
        # Penalize slow loading
        time_penalty = min(10, self.context_load_time_ms / 100)
        return max(0, base_score - time_penalty)


@dataclass
class ResourceUtilization:
    """Resource utilization metrics."""

    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    network_requests: int = 0
    disk_reads: int = 0
    disk_writes: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def overall_utilization(self) -> float:
        """Calculate overall resource utilization score."""
        return (self.memory_mb / 1024) * 0.3 + self.cpu_percent * 0.7


@dataclass
class PerformanceAlert:
    """Performance monitoring alert."""

    alert_id: str
    level: AlertLevel
    title: str
    description: str
    metric_type: MetricType
    threshold_value: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    persona_name: Optional[str] = None
    provider_name: Optional[str] = None
    suggested_actions: List[str] = field(default_factory=list)
    auto_resolvable: bool = False


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion."""

    suggestion_id: str
    optimization_type: OptimizationType
    title: str
    description: str
    potential_improvement: str  # "20% faster", "30% cost reduction", etc.
    confidence: float  # 0-1
    implementation_difficulty: str  # "easy", "medium", "hard"
    estimated_impact: Dict[str, float] = field(default_factory=dict)
    required_actions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""

    timestamp: datetime = field(default_factory=datetime.now)
    token_usage: TokenUsageMetrics = field(default_factory=TokenUsageMetrics)
    context_metrics: ContextMetrics = field(default_factory=ContextMetrics)
    resource_utilization: ResourceUtilization = field(
        default_factory=ResourceUtilization
    )
    active_operations: int = 0
    success_rate: float = 100.0
    average_response_time_ms: float = 0.0
    persona_name: Optional[str] = None
    provider_name: Optional[str] = None


class PerformanceMetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, max_history_size: int = 10000):
        """Initialize the metrics collector."""
        self.max_history_size = max_history_size
        self.token_usage_history: deque = deque(maxlen=max_history_size)
        self.timing_history: deque = deque(maxlen=max_history_size)
        self.context_history: deque = deque(maxlen=max_history_size)
        self.resource_history: deque = deque(maxlen=max_history_size)
        self.active_timings: Dict[str, OperationTiming] = {}
        self._lock = threading.RLock()

    def record_token_usage(self, tokens: TokenUsageMetrics) -> None:
        """Record token usage metrics."""
        with self._lock:
            self.token_usage_history.append(tokens)
            logger.debug(
                f"Recorded token usage: {tokens.total_tokens} tokens for {tokens.provider_name}"
            )

    def start_operation_timing(
        self, operation_id: str, operation_type: OperationType
    ) -> OperationTiming:
        """Start timing an operation."""
        with self._lock:
            timing = OperationTiming(
                operation_id=operation_id,
                operation_type=operation_type,
                start_time=datetime.now(),
            )
            self.active_timings[operation_id] = timing
            return timing

    def complete_operation_timing(self, operation_id: str) -> Optional[OperationTiming]:
        """Complete timing measurement for an operation."""
        with self._lock:
            if operation_id in self.active_timings:
                timing = self.active_timings.pop(operation_id)
                timing.complete_timing()
                self.timing_history.append(timing)
                logger.debug(
                    f"Completed timing for {operation_id}: {timing.duration_ms}ms"
                )
                return timing
            return None

    def record_context_metrics(self, context: ContextMetrics) -> None:
        """Record context efficiency metrics."""
        with self._lock:
            self.context_history.append(context)
            logger.debug(
                f"Recorded context metrics: {context.efficiency_score:.1f} efficiency score"
            )

    def record_resource_utilization(self, resources: ResourceUtilization) -> None:
        """Record resource utilization metrics."""
        with self._lock:
            self.resource_history.append(resources)

    def get_token_usage_stats(
        self, time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get aggregated token usage statistics."""
        with self._lock:
            if not self.token_usage_history:
                return {"total_tokens": 0, "providers": {}, "operations": {}}

            # Filter by time window if provided
            cutoff_time = datetime.now() - time_window if time_window else None
            relevant_usage = [
                usage
                for usage in self.token_usage_history
                if not cutoff_time or usage.timestamp >= cutoff_time
            ]

            if not relevant_usage:
                return {"total_tokens": 0, "providers": {}, "operations": {}}

            # Aggregate statistics
            total_tokens = sum(usage.total_tokens for usage in relevant_usage)
            total_cost = sum(usage.cost_estimate for usage in relevant_usage)

            # Group by provider
            provider_stats = defaultdict(
                lambda: {"tokens": 0, "cost": 0.0, "requests": 0}
            )
            for usage in relevant_usage:
                provider_stats[usage.provider_name]["tokens"] += usage.total_tokens
                provider_stats[usage.provider_name]["cost"] += usage.cost_estimate
                provider_stats[usage.provider_name]["requests"] += 1

            # Group by operation type
            operation_stats = defaultdict(lambda: {"tokens": 0, "requests": 0})
            for usage in relevant_usage:
                operation_stats[usage.operation_type.value][
                    "tokens"
                ] += usage.total_tokens
                operation_stats[usage.operation_type.value]["requests"] += 1

            return {
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "total_requests": len(relevant_usage),
                "providers": dict(provider_stats),
                "operations": dict(operation_stats),
                "time_period": (
                    f"{time_window.total_seconds():.0f} seconds"
                    if time_window
                    else "all time"
                ),
            }

    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot."""
        with self._lock:
            # Calculate recent metrics (last 10 minutes)
            recent_window = timedelta(minutes=10)
            recent_tokens = [
                usage
                for usage in self.token_usage_history
                if (datetime.now() - usage.timestamp) <= recent_window
            ]
            recent_timings = [
                timing
                for timing in self.timing_history
                if timing.is_complete
                and (datetime.now() - timing.start_time) <= recent_window
            ]

            # Aggregate token usage
            token_metrics = TokenUsageMetrics()
            if recent_tokens:
                token_metrics.total_tokens = sum(t.total_tokens for t in recent_tokens)
                token_metrics.input_tokens = sum(t.input_tokens for t in recent_tokens)
                token_metrics.output_tokens = sum(
                    t.output_tokens for t in recent_tokens
                )
                token_metrics.cost_estimate = sum(
                    t.cost_estimate for t in recent_tokens
                )

            # Calculate average response time
            avg_response_time = 0.0
            if recent_timings:
                avg_response_time = statistics.mean(
                    t.duration_ms for t in recent_timings
                )

            # Calculate success rate
            total_ops = len(recent_timings)
            successful_ops = len(
                [t for t in recent_timings if t.duration_ms > 0]
            )  # Completed ops
            success_rate = (
                (successful_ops / total_ops * 100) if total_ops > 0 else 100.0
            )

            return PerformanceSnapshot(
                token_usage=token_metrics,
                active_operations=len(self.active_timings),
                success_rate=success_rate,
                average_response_time_ms=avg_response_time,
            )


class PerformanceMonitor:
    """Main performance monitoring system."""

    def __init__(
        self,
        status_manager: Optional[AgentStatusManager] = None,
        persona_manager: Optional[PersonaManager] = None,
        storage_path: Optional[Path] = None,
    ):
        """Initialize the performance monitor."""
        self.status_manager = status_manager or get_status_manager()
        self.persona_manager = persona_manager or get_persona_manager()
        self.storage_path = storage_path or Path.home() / ".omnimancer" / "performance"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.metrics_collector = PerformanceMetricsCollector()
        self.alerts: List[PerformanceAlert] = []
        self.suggestions: List[OptimizationSuggestion] = []
        self.alert_thresholds = self._initialize_thresholds()
        self.is_monitoring = False
        self._lock = threading.RLock()

        # Event listeners
        self._event_listeners: List[Callable] = []

        logger.info("Performance monitor initialized")

    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize default alert thresholds."""
        return {
            "token_usage": {
                "high_usage_per_minute": 10000,
                "excessive_usage_per_minute": 50000,
                "cost_threshold_per_hour": 10.0,
            },
            "timing": {
                "slow_response_ms": 5000,
                "very_slow_response_ms": 15000,
                "context_load_slow_ms": 2000,
            },
            "context": {
                "low_efficiency_threshold": 30.0,
                "very_low_efficiency_threshold": 10.0,
            },
            "resources": {"high_memory_mb": 1024, "high_cpu_percent": 80.0},
        }

    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring is already active")
            return

        self.is_monitoring = True

        # Register with status manager for events
        if self.status_manager:
            self.status_manager.add_event_listener(self._handle_agent_event)

        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        # Unregister from status manager
        if self.status_manager:
            self.status_manager.remove_event_listener(self._handle_agent_event)

        logger.info("Performance monitoring stopped")

    def record_api_call_metrics(
        self,
        provider_name: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        cost_estimate: float = 0.0,
        operation_type: OperationType = OperationType.API_CALL,
    ) -> None:
        """Record API call metrics."""
        metrics = TokenUsageMetrics(
            total_tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            provider_name=provider_name,
            model_name=model_name,
            operation_type=operation_type,
            cost_estimate=cost_estimate,
        )

        self.metrics_collector.record_token_usage(metrics)
        self._check_token_usage_alerts(metrics)

    def start_operation_tracking(
        self, operation_id: str, operation_type: OperationType
    ) -> str:
        """Start tracking an operation."""
        timing = self.metrics_collector.start_operation_timing(
            operation_id, operation_type
        )
        return timing.operation_id

    def complete_operation_tracking(
        self, operation_id: str
    ) -> Optional[OperationTiming]:
        """Complete tracking an operation."""
        timing = self.metrics_collector.complete_operation_timing(operation_id)
        if timing:
            self._check_timing_alerts(timing)
        return timing

    def record_context_usage(
        self,
        context_size_chars: int,
        context_size_tokens: int,
        useful_ratio: float,
        load_time_ms: float,
        context_type: str = "",
    ) -> None:
        """Record context usage metrics."""
        context_metrics = ContextMetrics(
            context_size_chars=context_size_chars,
            context_size_tokens=context_size_tokens,
            useful_context_ratio=useful_ratio,
            context_load_time_ms=load_time_ms,
            context_type=context_type,
        )

        self.metrics_collector.record_context_metrics(context_metrics)
        self._check_context_alerts(context_metrics)

    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard."""
        snapshot = self.metrics_collector.get_performance_snapshot()
        recent_stats = self.metrics_collector.get_token_usage_stats(timedelta(hours=1))

        return {
            "current_snapshot": asdict(snapshot),
            "recent_stats": recent_stats,
            "active_alerts": [
                asdict(alert) for alert in self.alerts[-10:]
            ],  # Last 10 alerts
            "suggestions": [
                asdict(suggestion) for suggestion in self.suggestions[-5:]
            ],  # Last 5 suggestions
            "monitoring_status": ("active" if self.is_monitoring else "inactive"),
        }

    def _handle_agent_event(self, event: AgentEvent) -> None:
        """Handle agent events for performance monitoring."""
        if not self.is_monitoring:
            return

        try:
            if event.event_type == EventType.OPERATION_STARTED:
                # Start tracking operation
                if event.operation:
                    self.start_operation_tracking(
                        event.operation.operation_id,
                        event.operation.operation_type,
                    )

            elif event.event_type == EventType.OPERATION_COMPLETED:
                # Complete tracking operation
                if event.operation:
                    self.complete_operation_tracking(event.operation.operation_id)

            elif event.event_type == EventType.OPERATION_FAILED:
                # Record failed operation
                if event.operation:
                    self.complete_operation_tracking(event.operation.operation_id)

        except Exception as e:
            logger.error(f"Error handling agent event: {e}")

    def _check_token_usage_alerts(self, metrics: TokenUsageMetrics) -> None:
        """Check for token usage alerts."""
        with self._lock:
            thresholds = self.alert_thresholds["token_usage"]

            # Check recent usage (last minute)
            recent_usage = self.metrics_collector.get_token_usage_stats(
                timedelta(minutes=1)
            )
            tokens_per_minute = recent_usage["total_tokens"]

            if tokens_per_minute > thresholds["excessive_usage_per_minute"]:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    "Excessive Token Usage",
                    f"Using {tokens_per_minute:,} tokens per minute (threshold: {thresholds['excessive_usage_per_minute']:,})",
                    MetricType.TOKEN_USAGE,
                    thresholds["excessive_usage_per_minute"],
                    tokens_per_minute,
                    provider_name=metrics.provider_name,
                    suggested_actions=[
                        "Consider switching to a more efficient model",
                        "Reduce context size",
                        "Implement response caching",
                    ],
                )
            elif tokens_per_minute > thresholds["high_usage_per_minute"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    "High Token Usage",
                    f"Using {tokens_per_minute:,} tokens per minute (threshold: {thresholds['high_usage_per_minute']:,})",
                    MetricType.TOKEN_USAGE,
                    thresholds["high_usage_per_minute"],
                    tokens_per_minute,
                    provider_name=metrics.provider_name,
                    suggested_actions=[
                        "Monitor usage patterns",
                        "Consider optimizing context",
                    ],
                )

    def _check_timing_alerts(self, timing: OperationTiming) -> None:
        """Check for timing-based alerts."""
        with self._lock:
            thresholds = self.alert_thresholds["timing"]

            if timing.duration_ms > thresholds["very_slow_response_ms"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    "Very Slow Response",
                    f"Operation took {timing.duration_ms:.0f}ms (threshold: {thresholds['very_slow_response_ms']:.0f}ms)",
                    MetricType.OPERATION_TIMING,
                    thresholds["very_slow_response_ms"],
                    timing.duration_ms,
                    suggested_actions=[
                        "Check network connectivity",
                        "Consider switching providers",
                    ],
                )

    def _check_context_alerts(self, context: ContextMetrics) -> None:
        """Check for context efficiency alerts."""
        with self._lock:
            thresholds = self.alert_thresholds["context"]
            efficiency_score = context.efficiency_score

            if efficiency_score < thresholds["very_low_efficiency_threshold"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    "Very Low Context Efficiency",
                    f"Context efficiency is {efficiency_score:.1f}% (threshold: {thresholds['very_low_efficiency_threshold']:.1f}%)",
                    MetricType.CONTEXT_EFFICIENCY,
                    thresholds["very_low_efficiency_threshold"],
                    efficiency_score,
                    suggested_actions=[
                        "Reduce context size",
                        "Improve context relevance",
                        "Consider context compression techniques",
                    ],
                )

    def _create_alert(
        self,
        level: AlertLevel,
        title: str,
        description: str,
        metric_type: MetricType,
        threshold_value: float,
        current_value: float,
        provider_name: Optional[str] = None,
        persona_name: Optional[str] = None,
        suggested_actions: Optional[List[str]] = None,
    ) -> PerformanceAlert:
        """Create a new performance alert."""
        alert = PerformanceAlert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            level=level,
            title=title,
            description=description,
            metric_type=metric_type,
            threshold_value=threshold_value,
            current_value=current_value,
            provider_name=provider_name,
            persona_name=persona_name,
            suggested_actions=suggested_actions or [],
        )

        self.alerts.append(alert)
        logger.warning(f"Performance alert created: {title}")

        # Notify listeners
        for listener in self._event_listeners:
            try:
                listener("performance_alert", alert)
            except Exception as e:
                logger.error(f"Error notifying alert listener: {e}")

        return alert

    def add_event_listener(self, listener: Callable) -> None:
        """Add an event listener for performance events."""
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable) -> None:
        """Remove an event listener."""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        with _monitor_lock:
            if _performance_monitor is None:
                _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def set_performance_monitor(monitor: PerformanceMonitor) -> None:
    """Set the global performance monitor instance."""
    global _performance_monitor
    with _monitor_lock:
        _performance_monitor = monitor


def initialize_performance_monitoring(
    auto_start: bool = True,
) -> PerformanceMonitor:
    """Initialize the performance monitoring system."""
    monitor = get_performance_monitor()
    if auto_start:
        monitor.start_monitoring()
    return monitor


def shutdown_performance_monitoring() -> None:
    """Shutdown the performance monitoring system."""
    global _performance_monitor
    if _performance_monitor:
        _performance_monitor.stop_monitoring()
        with _monitor_lock:
            _performance_monitor = None
