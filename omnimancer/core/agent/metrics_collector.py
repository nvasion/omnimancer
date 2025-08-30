"""
Performance Metrics Collection System for Omnimancer Agent Monitoring.

This module provides comprehensive metrics collection for agent operations,
including timing, resource usage, success rates, and operational efficiency.
"""

import json
import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from .persona import get_persona_manager
from .status_core import (
    AgentEvent,
    EventType,
    OperationType,
)
from .status_manager import (
    get_status_manager,
)
from .token_tracker import (
    get_token_tracker,
)

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categories of metrics being collected."""

    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    RESOURCE = "resource"
    COST = "cost"
    USER_EXPERIENCE = "user_experience"


class AggregationMethod(Enum):
    """Methods for aggregating metrics."""

    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"  # per minute/hour/day


@dataclass
class MetricValue:
    """Individual metric measurement."""

    name: str
    value: float
    unit: str = ""
    category: MetricCategory = MetricCategory.PERFORMANCE
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(
        default_factory=dict
    )  # For grouping (persona, provider, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric over time period."""

    name: str
    value: float
    unit: str
    aggregation_method: AggregationMethod
    time_window: timedelta
    sample_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)

    # Statistical information
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    std_deviation: Optional[float] = None


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""

    metric_name: str
    baseline_value: float
    confidence_interval: Tuple[float, float]  # (lower, upper)
    measurement_period: timedelta
    sample_size: int
    established_date: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemResourceSnapshot:
    """System resource usage snapshot."""

    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    disk_usage_gb: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    process_count: int = 0
    thread_count: int = 0
    open_files: int = 0

    @classmethod
    def capture_current(cls) -> "SystemResourceSnapshot":
        """Capture current system resource usage."""
        try:
            # Get current process
            process = psutil.Process()

            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()

            # Process-specific metrics
            process.memory_info()
            process.cpu_percent()

            return cls(
                cpu_percent=cpu_percent,
                memory_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                disk_usage_gb=disk.used / (1024 * 1024 * 1024),
                disk_usage_percent=(disk.used / disk.total) * 100,
                network_bytes_sent=network.bytes_sent if network else 0,
                network_bytes_recv=network.bytes_recv if network else 0,
                process_count=len(psutil.pids()),
                thread_count=process.num_threads(),
                open_files=len(process.open_files()),
            )
        except Exception as e:
            logger.warning(f"Failed to capture resource snapshot: {e}")
            return cls()


class MetricsBuffer:
    """Circular buffer for storing metrics with automatic aggregation."""

    def __init__(self, name: str, max_size: int = 10000):
        """Initialize metrics buffer."""
        self.name = name
        self.max_size = max_size
        self.values: deque = deque(maxlen=max_size)
        self.lock = threading.RLock()

    def add_value(self, value: MetricValue) -> None:
        """Add a metric value to the buffer."""
        with self.lock:
            self.values.append(value)

    def get_aggregated(
        self,
        time_window: timedelta,
        method: AggregationMethod = AggregationMethod.AVERAGE,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[AggregatedMetric]:
        """Get aggregated metric over time window."""
        with self.lock:
            if not self.values:
                return None

            # Filter by time window
            cutoff_time = datetime.now() - time_window
            relevant_values = [v for v in self.values if v.timestamp >= cutoff_time]

            # Filter by labels if provided
            if labels:
                relevant_values = [
                    v
                    for v in relevant_values
                    if all(v.labels.get(k) == v for k, v in labels.items())
                ]

            if not relevant_values:
                return None

            # Extract numeric values
            numeric_values = [v.value for v in relevant_values]

            # Calculate aggregated value based on method
            if method == AggregationMethod.SUM:
                agg_value = sum(numeric_values)
            elif method == AggregationMethod.AVERAGE:
                agg_value = statistics.mean(numeric_values)
            elif method == AggregationMethod.MEDIAN:
                agg_value = statistics.median(numeric_values)
            elif method == AggregationMethod.MIN:
                agg_value = min(numeric_values)
            elif method == AggregationMethod.MAX:
                agg_value = max(numeric_values)
            elif method == AggregationMethod.COUNT:
                agg_value = len(numeric_values)
            elif method == AggregationMethod.RATE:
                # Calculate rate per minute
                agg_value = len(numeric_values) / (time_window.total_seconds() / 60)
            else:
                agg_value = statistics.mean(numeric_values)

            # Calculate statistics
            min_value = min(numeric_values)
            max_value = max(numeric_values)
            std_dev = (
                statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0
            )

            # Get unit from first value (assuming all have same unit)
            unit = relevant_values[0].unit

            return AggregatedMetric(
                name=self.name,
                value=agg_value,
                unit=unit,
                aggregation_method=method,
                time_window=time_window,
                sample_count=len(numeric_values),
                labels=labels or {},
                min_value=min_value,
                max_value=max_value,
                std_deviation=std_dev,
            )

    def get_recent_values(self, count: int = 100) -> List[MetricValue]:
        """Get the most recent metric values."""
        with self.lock:
            return list(self.values)[-count:]


class PerformanceMetricsCollector:
    """Main metrics collection system."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        collection_interval: float = 10.0,  # seconds
        max_buffer_size: int = 50000,
    ):
        """Initialize the metrics collector."""
        self.storage_path = storage_path or Path.home() / ".omnimancer" / "metrics"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.collection_interval = collection_interval
        self.max_buffer_size = max_buffer_size

        # Metric buffers
        self.metric_buffers: Dict[str, MetricsBuffer] = {}
        self.resource_snapshots: deque = deque(maxlen=max_buffer_size)
        self.baselines: Dict[str, PerformanceBaseline] = {}

        # Collection state
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None

        # Dependencies
        self.status_manager = get_status_manager()
        self.token_tracker = get_token_tracker()
        self.persona_manager = get_persona_manager()

        # Metrics tracking
        self.operation_start_times: Dict[str, datetime] = {}
        self.lock = threading.RLock()

        logger.info("Performance metrics collector initialized")

    def start_collection(self) -> None:
        """Start metrics collection."""
        if self.is_collecting:
            logger.warning("Metrics collection is already running")
            return

        self.is_collecting = True

        # Start background collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop, name="MetricsCollection", daemon=True
        )
        self.collection_thread.start()

        # Register event listeners
        if self.status_manager:
            self.status_manager.add_event_listener(self._handle_agent_event)

        logger.info("Metrics collection started")

    def stop_collection(self) -> None:
        """Stop metrics collection."""
        if not self.is_collecting:
            return

        self.is_collecting = False

        # Unregister event listeners
        if self.status_manager:
            self.status_manager.remove_event_listener(self._handle_agent_event)

        # Wait for collection thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)

        logger.info("Metrics collection stopped")

    def _collection_loop(self) -> None:
        """Main collection loop running in background thread."""
        while self.is_collecting:
            try:
                self._collect_system_metrics()
                self._collect_agent_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            snapshot = SystemResourceSnapshot.capture_current()

            with self.lock:
                self.resource_snapshots.append(snapshot)

            # Convert to individual metrics
            self.record_metric("system.cpu_percent", snapshot.cpu_percent, "percent")
            self.record_metric("system.memory_mb", snapshot.memory_mb, "MB")
            self.record_metric(
                "system.memory_percent", snapshot.memory_percent, "percent"
            )
            self.record_metric("system.disk_usage_gb", snapshot.disk_usage_gb, "GB")
            self.record_metric("system.process_count", snapshot.process_count, "count")
            self.record_metric("system.thread_count", snapshot.thread_count, "count")

        except Exception as e:
            logger.debug(f"Failed to collect system metrics: {e}")

    def _collect_agent_metrics(self) -> None:
        """Collect agent-specific metrics."""
        try:
            # Collect token usage metrics
            if self.token_tracker:
                recent_summary = self.token_tracker.get_usage_summary(
                    timedelta(minutes=1)
                )

                self.record_metric(
                    "agent.tokens_per_minute",
                    recent_summary.get("total_tokens", 0),
                    "tokens",
                    category=MetricCategory.EFFICIENCY,
                )

                self.record_metric(
                    "agent.cost_per_minute",
                    recent_summary.get("total_cost", 0.0),
                    "USD",
                    category=MetricCategory.COST,
                )

                self.record_metric(
                    "agent.requests_per_minute",
                    recent_summary.get("total_requests", 0),
                    "requests",
                    category=MetricCategory.PERFORMANCE,
                )

            # Collect persona metrics
            if self.persona_manager:
                active_personas = self.persona_manager.get_active_personas()
                for persona in active_personas:
                    labels = {"persona": persona.name}

                    # Record persona-specific metrics
                    self.record_metric(
                        "persona.active",
                        1.0,
                        "boolean",
                        labels=labels,
                        category=MetricCategory.USER_EXPERIENCE,
                    )

        except Exception as e:
            logger.debug(f"Failed to collect agent metrics: {e}")

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
        category: MetricCategory = MetricCategory.PERFORMANCE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a metric value."""
        metric = MetricValue(
            name=name,
            value=value,
            unit=unit,
            category=category,
            labels=labels or {},
            metadata=metadata or {},
        )

        # Ensure metric buffer exists
        if name not in self.metric_buffers:
            self.metric_buffers[name] = MetricsBuffer(name, self.max_buffer_size)

        # Add to buffer
        self.metric_buffers[name].add_value(metric)

        logger.debug(f"Recorded metric {name}={value}{unit}")

    def record_operation_timing(
        self,
        operation_id: str,
        operation_type: OperationType,
        duration_ms: float,
        success: bool = True,
        persona_name: Optional[str] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        """Record operation timing metrics."""
        labels = {
            "operation_type": operation_type.value,
            "success": str(success).lower(),
        }

        if persona_name:
            labels["persona"] = persona_name
        if provider_name:
            labels["provider"] = provider_name

        # Record duration
        self.record_metric(
            "operation.duration_ms",
            duration_ms,
            "ms",
            labels=labels,
            category=MetricCategory.PERFORMANCE,
        )

        # Record success rate
        self.record_metric(
            "operation.success",
            1.0 if success else 0.0,
            "boolean",
            labels=labels,
            category=MetricCategory.RELIABILITY,
        )

        # Record operation count
        self.record_metric(
            "operation.count",
            1.0,
            "count",
            labels=labels,
            category=MetricCategory.PERFORMANCE,
        )

    def _handle_agent_event(self, event: AgentEvent) -> None:
        """Handle agent events for metrics collection."""
        try:
            if event.event_type == EventType.OPERATION_STARTED:
                if event.operation:
                    with self.lock:
                        self.operation_start_times[event.operation.operation_id] = (
                            datetime.now()
                        )

            elif event.event_type in [
                EventType.OPERATION_COMPLETED,
                EventType.OPERATION_FAILED,
            ]:
                if event.operation:
                    with self.lock:
                        start_time = self.operation_start_times.pop(
                            event.operation.operation_id, None
                        )

                    if start_time:
                        duration_ms = (
                            datetime.now() - start_time
                        ).total_seconds() * 1000
                        success = event.event_type == EventType.OPERATION_COMPLETED

                        self.record_operation_timing(
                            event.operation.operation_id,
                            event.operation.operation_type,
                            duration_ms,
                            success,
                            metadata={"event_type": event.event_type.value},
                        )

        except Exception as e:
            logger.error(f"Error handling agent event for metrics: {e}")

    def get_metric_summary(
        self,
        metric_name: str,
        time_window: timedelta = timedelta(hours=1),
        aggregation_method: AggregationMethod = AggregationMethod.AVERAGE,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[AggregatedMetric]:
        """Get aggregated metric summary."""
        if metric_name not in self.metric_buffers:
            return None

        return self.metric_buffers[metric_name].get_aggregated(
            time_window, aggregation_method, labels
        )

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics formatted for dashboard display."""
        metrics = {}

        # Performance metrics
        response_time = self.get_metric_summary(
            "operation.duration_ms", timedelta(minutes=10)
        )
        success_rate = self.get_metric_summary(
            "operation.success",
            timedelta(minutes=10),
            AggregationMethod.AVERAGE,
        )

        metrics["performance"] = {
            "avg_response_time_ms": (response_time.value if response_time else 0),
            "success_rate_percent": (
                (success_rate.value * 100) if success_rate else 100
            ),
        }

        # Resource metrics
        cpu_usage = self.get_metric_summary("system.cpu_percent", timedelta(minutes=5))
        memory_usage = self.get_metric_summary("system.memory_mb", timedelta(minutes=5))

        metrics["resources"] = {
            "cpu_percent": cpu_usage.value if cpu_usage else 0,
            "memory_mb": memory_usage.value if memory_usage else 0,
        }

        # Token usage metrics
        tokens_per_min = self.get_metric_summary(
            "agent.tokens_per_minute", timedelta(minutes=10)
        )
        cost_per_min = self.get_metric_summary(
            "agent.cost_per_minute", timedelta(minutes=10)
        )

        metrics["usage"] = {
            "tokens_per_minute": tokens_per_min.value if tokens_per_min else 0,
            "cost_per_minute": cost_per_min.value if cost_per_min else 0,
        }

        return metrics

    def establish_baseline(
        self,
        metric_name: str,
        measurement_period: timedelta = timedelta(days=7),
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[PerformanceBaseline]:
        """Establish performance baseline for a metric."""
        if metric_name not in self.metric_buffers:
            return None

        # Get historical data
        cutoff_time = datetime.now() - measurement_period
        with self.lock:
            values = [
                v
                for v in self.metric_buffers[metric_name].values
                if v.timestamp >= cutoff_time
            ]

            if labels:
                values = [
                    v
                    for v in values
                    if all(v.labels.get(k) == v for k, v in labels.items())
                ]

        if len(values) < 10:  # Need minimum samples
            return None

        # Calculate baseline statistics
        numeric_values = [v.value for v in values]
        mean_value = statistics.mean(numeric_values)
        std_dev = statistics.stdev(numeric_values)

        # 95% confidence interval (assuming normal distribution)
        confidence_interval = (
            mean_value - (1.96 * std_dev),
            mean_value + (1.96 * std_dev),
        )

        baseline = PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=mean_value,
            confidence_interval=confidence_interval,
            measurement_period=measurement_period,
            sample_size=len(values),
            labels=labels or {},
        )

        # Store baseline
        baseline_key = f"{metric_name}:{hash(str(sorted((labels or {}).items())))}"
        self.baselines[baseline_key] = baseline

        logger.info(f"Established baseline for {metric_name}: {mean_value:.2f}")
        return baseline

    def detect_anomalies(
        self, metric_name: str, labels: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics compared to baseline."""
        baseline_key = f"{metric_name}:{hash(str(sorted((labels or {}).items())))}"
        baseline = self.baselines.get(baseline_key)

        if not baseline:
            return []

        # Get recent values
        recent_values = self.get_metric_summary(
            metric_name,
            timedelta(minutes=10),
            AggregationMethod.AVERAGE,
            labels,
        )

        if not recent_values:
            return []

        anomalies = []
        current_value = recent_values.value

        # Check if value is outside confidence interval
        lower_bound, upper_bound = baseline.confidence_interval

        if current_value < lower_bound:
            anomalies.append(
                {
                    "type": "below_baseline",
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "baseline_value": baseline.baseline_value,
                    "deviation": baseline.baseline_value - current_value,
                    "severity": (
                        "high" if current_value < (lower_bound * 0.5) else "medium"
                    ),
                }
            )
        elif current_value > upper_bound:
            anomalies.append(
                {
                    "type": "above_baseline",
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "baseline_value": baseline.baseline_value,
                    "deviation": current_value - baseline.baseline_value,
                    "severity": (
                        "high" if current_value > (upper_bound * 1.5) else "medium"
                    ),
                }
            )

        return anomalies

    def export_metrics(
        self,
        time_window: Optional[timedelta] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Export metrics to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"

        filepath = self.storage_path / filename

        # Collect all metrics
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "time_window": str(time_window) if time_window else "all_time",
                "collector_version": "1.0",
            },
            "metrics": {},
            "baselines": {},
        }

        # Export metric buffers
        cutoff_time = datetime.now() - time_window if time_window else None

        for name, buffer in self.metric_buffers.items():
            with buffer.lock:
                values = list(buffer.values)
                if cutoff_time:
                    values = [v for v in values if v.timestamp >= cutoff_time]

                export_data["metrics"][name] = [asdict(v) for v in values]

        # Export baselines
        for key, baseline in self.baselines.items():
            export_data["baselines"][key] = asdict(baseline)

        # Write to file
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported metrics to {filepath}")
        return filepath


# Global metrics collector instance
_metrics_collector: Optional[PerformanceMetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> PerformanceMetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        with _collector_lock:
            if _metrics_collector is None:
                _metrics_collector = PerformanceMetricsCollector()
    return _metrics_collector


def set_metrics_collector(collector: PerformanceMetricsCollector) -> None:
    """Set the global metrics collector instance."""
    global _metrics_collector
    with _collector_lock:
        _metrics_collector = collector


def initialize_metrics_collection(
    auto_start: bool = True,
) -> PerformanceMetricsCollector:
    """Initialize metrics collection system."""
    collector = get_metrics_collector()
    if auto_start:
        collector.start_collection()
    return collector


def shutdown_metrics_collection() -> None:
    """Shutdown metrics collection system."""
    global _metrics_collector
    if _metrics_collector:
        _metrics_collector.stop_collection()
        with _collector_lock:
            _metrics_collector = None
