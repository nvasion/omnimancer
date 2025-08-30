"""
Optimization Engine and Alert System for Omnimancer Agent Performance.

This module provides intelligent optimization suggestions and proactive alerting
based on performance patterns, resource usage, and cost analysis.
"""

import logging
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .metrics_collector import (
    AggregationMethod,
    PerformanceMetricsCollector,
    get_metrics_collector,
)
from .performance_monitor import (
    AlertLevel,
    OptimizationSuggestion,
    OptimizationType,
    PerformanceAlert,
    PerformanceMonitor,
    get_performance_monitor,
)
from .persona import get_persona_manager
from .token_tracker import TokenUsageTracker, get_token_tracker

logger = logging.getLogger(__name__)


class OptimizationCategory(Enum):
    """Categories of optimizations."""

    COST_REDUCTION = "cost_reduction"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    RELIABILITY_ENHANCEMENT = "reliability_enhancement"
    USER_EXPERIENCE = "user_experience"


class AlertTrigger(Enum):
    """Types of alert triggers."""

    THRESHOLD_EXCEEDED = "threshold_exceeded"
    TREND_DETECTION = "trend_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    COST_SPIKE = "cost_spike"


@dataclass
class OptimizationRule:
    """Rule for generating optimization suggestions."""

    rule_id: str
    name: str
    category: OptimizationCategory
    description: str
    condition_check: Callable[[Dict[str, Any]], bool]
    suggestion_generator: Callable[[Dict[str, Any]], OptimizationSuggestion]
    priority: int = 50  # 0-100, higher is more important
    enabled: bool = True

    def evaluate(self, context: Dict[str, Any]) -> Optional[OptimizationSuggestion]:
        """Evaluate rule and generate suggestion if conditions are met."""
        if not self.enabled:
            return None

        try:
            if self.condition_check(context):
                return self.suggestion_generator(context)
        except Exception as e:
            logger.error(f"Error evaluating optimization rule {self.rule_id}: {e}")

        return None


@dataclass
class AlertRule:
    """Rule for generating performance alerts."""

    rule_id: str
    name: str
    metric_name: str
    trigger_type: AlertTrigger
    threshold_value: float
    comparison_operator: str  # "gt", "lt", "eq", "gte", "lte"
    alert_level: AlertLevel
    time_window: timedelta
    min_samples: int = 1
    enabled: bool = True
    cooldown_period: timedelta = timedelta(minutes=15)
    last_triggered: Optional[datetime] = None

    def check_condition(self, current_value: float, context: Dict[str, Any]) -> bool:
        """Check if alert condition is met."""
        if not self.enabled:
            return False

        # Check cooldown period
        if (
            self.last_triggered
            and datetime.now() - self.last_triggered < self.cooldown_period
        ):
            return False

        # Evaluate condition
        if self.comparison_operator == "gt":
            return current_value > self.threshold_value
        elif self.comparison_operator == "lt":
            return current_value < self.threshold_value
        elif self.comparison_operator == "eq":
            return abs(current_value - self.threshold_value) < 0.001
        elif self.comparison_operator == "gte":
            return current_value >= self.threshold_value
        elif self.comparison_operator == "lte":
            return current_value <= self.threshold_value

        return False

    def trigger_alert(
        self, current_value: float, context: Dict[str, Any]
    ) -> PerformanceAlert:
        """Generate alert when rule is triggered."""
        self.last_triggered = datetime.now()

        return PerformanceAlert(
            alert_id=f"alert_{self.rule_id}_{int(time.time() * 1000)}",
            level=self.alert_level,
            title=self.name,
            description=f"Metric '{self.metric_name}' {self.comparison_operator} {self.threshold_value}",
            metric_type=context.get("metric_type", "unknown"),
            threshold_value=self.threshold_value,
            current_value=current_value,
            provider_name=context.get("provider_name"),
            persona_name=context.get("persona_name"),
            suggested_actions=self._generate_suggested_actions(current_value, context),
        )

    def _generate_suggested_actions(
        self, current_value: float, context: Dict[str, Any]
    ) -> List[str]:
        """Generate suggested actions based on the alert."""
        actions = []

        if "token" in self.metric_name.lower():
            actions.extend(
                [
                    "Review recent token usage patterns",
                    "Consider switching to a more efficient model",
                    "Implement response caching for repeated queries",
                ]
            )
        elif "response_time" in self.metric_name.lower():
            actions.extend(
                [
                    "Check network connectivity",
                    "Consider switching to a faster provider",
                    "Review context size and complexity",
                ]
            )
        elif "cpu" in self.metric_name.lower() or "memory" in self.metric_name.lower():
            actions.extend(
                [
                    "Monitor system resource usage",
                    "Consider reducing concurrent operations",
                    "Review background processes",
                ]
            )

        return actions


class TrendAnalyzer:
    """Analyzes trends in performance metrics."""

    def __init__(self, min_data_points: int = 10):
        """Initialize trend analyzer."""
        self.min_data_points = min_data_points

    def detect_trend(
        self, values: List[float], timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Detect trend in time series data."""
        if len(values) < self.min_data_points:
            return {"trend": "insufficient_data", "confidence": 0.0}

        # Simple linear regression to detect trend
        try:
            # Convert timestamps to numeric values (seconds since first timestamp)
            time_numeric = [(t - timestamps[0]).total_seconds() for t in timestamps]

            # Calculate trend using linear regression
            n = len(values)
            sum_x = sum(time_numeric)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(time_numeric, values))
            sum_x_squared = sum(x * x for x in time_numeric)

            # Calculate slope (trend direction)
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)

            # Calculate correlation coefficient for confidence
            mean_x = sum_x / n
            mean_y = sum_y / n

            numerator = sum(
                (x - mean_x) * (y - mean_y) for x, y in zip(time_numeric, values)
            )
            denom_x = sum((x - mean_x) ** 2 for x in time_numeric)
            denom_y = sum((y - mean_y) ** 2 for y in values)

            correlation = (
                numerator / (denom_x * denom_y) ** 0.5 if denom_x * denom_y > 0 else 0
            )

            # Determine trend direction
            if abs(slope) < 0.001:  # Threshold for "no trend"
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            return {
                "trend": trend,
                "slope": slope,
                "confidence": abs(correlation),
                "correlation": correlation,
            }

        except (ValueError, ZeroDivisionError):
            return {"trend": "unknown", "confidence": 0.0}

    def detect_anomalies(
        self, values: List[float], sensitivity: float = 2.0
    ) -> List[int]:
        """Detect anomalies using statistical methods."""
        if len(values) < 3:
            return []

        # Use simple statistical outlier detection
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0

        if std_dev == 0:
            return []

        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_dev
            if z_score > sensitivity:
                anomalies.append(i)

        return anomalies


class OptimizationEngine:
    """Main optimization engine with rule-based suggestions and alerting."""

    def __init__(
        self,
        performance_monitor: Optional[PerformanceMonitor] = None,
        token_tracker: Optional[TokenUsageTracker] = None,
        metrics_collector: Optional[PerformanceMetricsCollector] = None,
    ):
        """Initialize optimization engine."""
        self.performance_monitor = performance_monitor or get_performance_monitor()
        self.token_tracker = token_tracker or get_token_tracker()
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.persona_manager = get_persona_manager()

        # Rule engines
        self.optimization_rules: Dict[str, OptimizationRule] = {}
        self.alert_rules: Dict[str, AlertRule] = {}

        # Analysis components
        self.trend_analyzer = TrendAnalyzer()

        # State
        self.is_running = False
        self.analysis_thread: Optional[threading.Thread] = None
        self.analysis_interval = 60.0  # seconds

        # Event handlers
        self.optimization_handlers: List[Callable] = []
        self.alert_handlers: List[Callable] = []

        self._initialize_default_rules()
        logger.info("Optimization engine initialized")

    def _initialize_default_rules(self) -> None:
        """Initialize default optimization and alert rules."""
        # Cost optimization rules
        self.add_optimization_rule(
            OptimizationRule(
                rule_id="high_cost_provider",
                name="High Cost Provider Usage",
                category=OptimizationCategory.COST_REDUCTION,
                description="Detects heavy usage of expensive providers",
                condition_check=self._check_high_cost_provider,
                suggestion_generator=self._suggest_cost_optimization,
                priority=80,
            )
        )

        self.add_optimization_rule(
            OptimizationRule(
                rule_id="inefficient_token_usage",
                name="Inefficient Token Usage",
                category=OptimizationCategory.COST_REDUCTION,
                description="Detects patterns of inefficient token usage",
                condition_check=self._check_inefficient_tokens,
                suggestion_generator=self._suggest_token_optimization,
                priority=70,
            )
        )

        # Performance optimization rules
        self.add_optimization_rule(
            OptimizationRule(
                rule_id="slow_response_times",
                name="Slow Response Times",
                category=OptimizationCategory.PERFORMANCE_IMPROVEMENT,
                description="Detects consistently slow response times",
                condition_check=self._check_slow_responses,
                suggestion_generator=self._suggest_performance_improvement,
                priority=75,
            )
        )

        # Alert rules
        self.add_alert_rule(
            AlertRule(
                rule_id="excessive_token_usage",
                name="Excessive Token Usage",
                metric_name="agent.tokens_per_minute",
                trigger_type=AlertTrigger.THRESHOLD_EXCEEDED,
                threshold_value=5000,
                comparison_operator="gt",
                alert_level=AlertLevel.WARNING,
                time_window=timedelta(minutes=5),
            )
        )

        self.add_alert_rule(
            AlertRule(
                rule_id="high_cost_rate",
                name="High Cost Rate",
                metric_name="agent.cost_per_minute",
                trigger_type=AlertTrigger.THRESHOLD_EXCEEDED,
                threshold_value=1.0,
                comparison_operator="gt",
                alert_level=AlertLevel.CRITICAL,
                time_window=timedelta(minutes=5),
            )
        )

        self.add_alert_rule(
            AlertRule(
                rule_id="slow_response_alert",
                name="Slow Response Times",
                metric_name="operation.duration_ms",
                trigger_type=AlertTrigger.THRESHOLD_EXCEEDED,
                threshold_value=10000,
                comparison_operator="gt",
                alert_level=AlertLevel.WARNING,
                time_window=timedelta(minutes=10),
            )
        )

    def add_optimization_rule(self, rule: OptimizationRule) -> None:
        """Add optimization rule."""
        self.optimization_rules[rule.rule_id] = rule
        logger.debug(f"Added optimization rule: {rule.name}")

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.debug(f"Added alert rule: {rule.name}")

    def start_analysis(self) -> None:
        """Start continuous optimization analysis."""
        if self.is_running:
            logger.warning("Optimization engine is already running")
            return

        self.is_running = True
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop,
            name="OptimizationAnalysis",
            daemon=True,
        )
        self.analysis_thread.start()
        logger.info("Optimization analysis started")

    def stop_analysis(self) -> None:
        """Stop continuous optimization analysis."""
        if not self.is_running:
            return

        self.is_running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5.0)
        logger.info("Optimization analysis stopped")

    def _analysis_loop(self) -> None:
        """Main analysis loop."""
        while self.is_running:
            try:
                self._run_analysis_cycle()
                time.sleep(self.analysis_interval)
            except Exception as e:
                logger.error(f"Error in optimization analysis loop: {e}")
                time.sleep(self.analysis_interval)

    def _run_analysis_cycle(self) -> None:
        """Run a single analysis cycle."""
        # Gather context data
        context = self._gather_analysis_context()

        # Check optimization rules
        suggestions = self._evaluate_optimization_rules(context)

        # Check alert rules
        alerts = self._evaluate_alert_rules(context)

        # Process suggestions
        for suggestion in suggestions:
            self._handle_optimization_suggestion(suggestion)

        # Process alerts
        for alert in alerts:
            self._handle_performance_alert(alert)

    def _gather_analysis_context(self) -> Dict[str, Any]:
        """Gather context data for analysis."""
        context = {}

        try:
            # Performance dashboard data
            if self.performance_monitor:
                context["dashboard_data"] = (
                    self.performance_monitor.get_performance_dashboard_data()
                )

            # Token usage data
            if self.token_tracker:
                context["token_summary_1h"] = self.token_tracker.get_usage_summary(
                    timedelta(hours=1)
                )
                context["token_summary_24h"] = self.token_tracker.get_usage_summary(
                    timedelta(hours=24)
                )
                context["usage_patterns"] = self.token_tracker.analyze_usage_patterns(7)

            # Metrics data
            if self.metrics_collector:
                context["metrics_data"] = self.metrics_collector.get_dashboard_metrics()

            # Current timestamp
            context["analysis_time"] = datetime.now()

        except Exception as e:
            logger.error(f"Error gathering analysis context: {e}")

        return context

    def _evaluate_optimization_rules(
        self, context: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Evaluate all optimization rules."""
        suggestions = []

        for rule in self.optimization_rules.values():
            try:
                suggestion = rule.evaluate(context)
                if suggestion:
                    suggestions.append(suggestion)
            except Exception as e:
                logger.error(f"Error evaluating optimization rule {rule.rule_id}: {e}")

        return suggestions

    def _evaluate_alert_rules(self, context: Dict[str, Any]) -> List[PerformanceAlert]:
        """Evaluate all alert rules."""
        alerts = []

        for rule in self.alert_rules.values():
            try:
                # Get current metric value
                metric_summary = self.metrics_collector.get_metric_summary(
                    rule.metric_name,
                    rule.time_window,
                    AggregationMethod.AVERAGE,
                )

                if metric_summary and rule.check_condition(
                    metric_summary.value, context
                ):
                    alert = rule.trigger_alert(metric_summary.value, context)
                    alerts.append(alert)

            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.rule_id}: {e}")

        return alerts

    def _handle_optimization_suggestion(
        self, suggestion: OptimizationSuggestion
    ) -> None:
        """Handle optimization suggestion."""
        # Add to performance monitor
        if self.performance_monitor:
            self.performance_monitor.suggestions.append(suggestion)

        # Notify handlers
        for handler in self.optimization_handlers:
            try:
                handler(suggestion)
            except Exception as e:
                logger.error(f"Error in optimization handler: {e}")

        logger.info(f"Generated optimization suggestion: {suggestion.title}")

    def _handle_performance_alert(self, alert: PerformanceAlert) -> None:
        """Handle performance alert."""
        # Add to performance monitor
        if self.performance_monitor:
            self.performance_monitor.alerts.append(alert)

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        logger.warning(f"Generated performance alert: {alert.title}")

    # Optimization rule condition checkers
    def _check_high_cost_provider(self, context: Dict[str, Any]) -> bool:
        """Check for high cost provider usage."""
        token_summary = context.get("token_summary_24h", {})
        provider_breakdown = token_summary.get("provider_breakdown", {})

        for provider, stats in provider_breakdown.items():
            if stats.get("cost", 0) > 5.0:  # More than $5 in 24 hours
                return True

        return False

    def _check_inefficient_tokens(self, context: Dict[str, Any]) -> bool:
        """Check for inefficient token usage patterns."""
        token_summary = context.get("token_summary_1h", {})
        avg_tokens_per_request = token_summary.get("average_tokens_per_request", 0)

        # Alert if average request uses more than 2000 tokens
        return avg_tokens_per_request > 2000

    def _check_slow_responses(self, context: Dict[str, Any]) -> bool:
        """Check for slow response times."""
        metrics_data = context.get("metrics_data", {})
        performance = metrics_data.get("performance", {})
        avg_response_time = performance.get("avg_response_time_ms", 0)

        # Alert if average response time is over 5 seconds
        return avg_response_time > 5000

    # Optimization suggestion generators
    def _suggest_cost_optimization(
        self, context: Dict[str, Any]
    ) -> OptimizationSuggestion:
        """Generate cost optimization suggestion."""
        token_summary = context.get("token_summary_24h", {})
        total_cost = token_summary.get("total_cost", 0)

        return OptimizationSuggestion(
            suggestion_id=f"cost_opt_{int(time.time())}",
            optimization_type=OptimizationType.COST_OPTIMIZATION,
            title="Reduce Provider Costs",
            description=f"High daily cost of ${total_cost:.2f} detected",
            potential_improvement="30-50% cost reduction",
            confidence=0.8,
            implementation_difficulty="medium",
            estimated_impact={"cost_savings": total_cost * 0.4},
            required_actions=[
                "Switch to more cost-effective models for routine tasks",
                "Implement response caching",
                "Optimize context sizes",
            ],
        )

    def _suggest_token_optimization(
        self, context: Dict[str, Any]
    ) -> OptimizationSuggestion:
        """Generate token usage optimization suggestion."""
        token_summary = context.get("token_summary_1h", {})
        avg_tokens = token_summary.get("average_tokens_per_request", 0)

        return OptimizationSuggestion(
            suggestion_id=f"token_opt_{int(time.time())}",
            optimization_type=OptimizationType.CONTEXT_REDUCTION,
            title="Optimize Token Usage",
            description=f"High average of {avg_tokens:.0f} tokens per request",
            potential_improvement="20-40% token reduction",
            confidence=0.75,
            implementation_difficulty="easy",
            estimated_impact={"token_savings": avg_tokens * 0.3},
            required_actions=[
                "Review and reduce context sizes",
                "Implement context compression",
                "Use more efficient prompting strategies",
            ],
        )

    def _suggest_performance_improvement(
        self, context: Dict[str, Any]
    ) -> OptimizationSuggestion:
        """Generate performance improvement suggestion."""
        metrics_data = context.get("metrics_data", {})
        performance = metrics_data.get("performance", {})
        response_time = performance.get("avg_response_time_ms", 0)

        return OptimizationSuggestion(
            suggestion_id=f"perf_opt_{int(time.time())}",
            optimization_type=OptimizationType.PROVIDER_SWITCHING,
            title="Improve Response Times",
            description=f"Slow average response time of {response_time:.0f}ms",
            potential_improvement="40-60% faster responses",
            confidence=0.7,
            implementation_difficulty="medium",
            estimated_impact={"response_time_improvement": response_time * 0.5},
            required_actions=[
                "Switch to faster providers",
                "Optimize network connectivity",
                "Reduce context complexity",
            ],
        )

    def add_optimization_handler(
        self, handler: Callable[[OptimizationSuggestion], None]
    ) -> None:
        """Add optimization suggestion handler."""
        self.optimization_handlers.append(handler)

    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]) -> None:
        """Add alert handler."""
        self.alert_handlers.append(handler)

    def get_current_suggestions(
        self, category: Optional[OptimizationCategory] = None
    ) -> List[OptimizationSuggestion]:
        """Get current optimization suggestions."""
        if not self.performance_monitor:
            return []

        suggestions = self.performance_monitor.suggestions

        if category:
            # Filter suggestions would require additional logic since OptimizationSuggestion
            # doesn't have category field in the original design
            pass

        # Return recent suggestions (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        return [s for s in suggestions if s.timestamp >= cutoff_time]

    def get_active_alerts(
        self, level: Optional[AlertLevel] = None
    ) -> List[PerformanceAlert]:
        """Get active performance alerts."""
        if not self.performance_monitor:
            return []

        alerts = self.performance_monitor.alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        # Return recent alerts (last 6 hours)
        cutoff_time = datetime.now() - timedelta(hours=6)
        return [a for a in alerts if a.timestamp >= cutoff_time]


# Global optimization engine instance
_optimization_engine: Optional[OptimizationEngine] = None
_engine_lock = threading.Lock()


def get_optimization_engine() -> OptimizationEngine:
    """Get the global optimization engine instance."""
    global _optimization_engine
    if _optimization_engine is None:
        with _engine_lock:
            if _optimization_engine is None:
                _optimization_engine = OptimizationEngine()
    return _optimization_engine


def set_optimization_engine(engine: OptimizationEngine) -> None:
    """Set the global optimization engine instance."""
    global _optimization_engine
    with _engine_lock:
        _optimization_engine = engine


def initialize_optimization_engine(
    auto_start: bool = True,
) -> OptimizationEngine:
    """Initialize optimization engine."""
    engine = get_optimization_engine()
    if auto_start:
        engine.start_analysis()
    return engine


def shutdown_optimization_engine() -> None:
    """Shutdown optimization engine."""
    global _optimization_engine
    if _optimization_engine:
        _optimization_engine.stop_analysis()
        with _engine_lock:
            _optimization_engine = None
