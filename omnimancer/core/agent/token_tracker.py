"""
Advanced Token Usage Tracking for Omnimancer Agent Performance Monitoring.

This module provides detailed token usage tracking with cost estimation,
provider-specific optimization, and usage pattern analysis.
"""

import json
import logging
import statistics
import threading
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class TokenCountMethod(Enum):
    """Methods for counting tokens."""

    ESTIMATED = "estimated"
    API_REPORTED = "api_reported"
    TIKTOKEN = "tiktoken"
    PROVIDER_SPECIFIC = "provider_specific"


class UsageCategory(Enum):
    """Categories of token usage."""

    INPUT_CONTEXT = "input_context"
    OUTPUT_GENERATION = "output_generation"
    SYSTEM_PROMPTS = "system_prompts"
    FUNCTION_CALLS = "function_calls"
    CACHED_CONTENT = "cached_content"
    RETRY_ATTEMPTS = "retry_attempts"


@dataclass
class TokenCost:
    """Token cost information for a specific provider and model."""

    provider_name: str
    model_name: str
    input_cost_per_1k: float  # Cost per 1,000 input tokens
    output_cost_per_1k: float  # Cost per 1,000 output tokens
    cached_cost_per_1k: float = 0.0  # Cost per 1,000 cached tokens (if different)
    currency: str = "USD"
    last_updated: datetime = field(default_factory=datetime.now)

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, cached_tokens: int = 0
    ) -> float:
        """Calculate total cost for token usage."""
        input_cost = (input_tokens / 1000.0) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * self.output_cost_per_1k
        cached_cost = (cached_tokens / 1000.0) * self.cached_cost_per_1k
        return input_cost + output_cost + cached_cost


@dataclass
class DetailedTokenUsage:
    """Detailed token usage breakdown."""

    total_tokens: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0

    # Detailed breakdown
    context_tokens: int = 0
    system_prompt_tokens: int = 0
    user_prompt_tokens: int = 0
    function_call_tokens: int = 0
    response_tokens: int = 0

    # Metadata
    provider_name: str = ""
    model_name: str = ""
    persona_name: str = ""
    operation_id: str = ""
    count_method: TokenCountMethod = TokenCountMethod.ESTIMATED
    timestamp: datetime = field(default_factory=datetime.now)

    # Cost information
    estimated_cost: float = 0.0
    cost_currency: str = "USD"

    # Efficiency metrics
    compression_ratio: float = 0.0  # How much input was compressed
    response_efficiency: float = 0.0  # Useful output / total output
    context_utilization: float = 0.0  # How much context was actually used


@dataclass
class UsagePattern:
    """Token usage pattern analysis."""

    time_period: timedelta
    total_requests: int
    total_tokens: int
    total_cost: float
    average_tokens_per_request: float
    average_cost_per_request: float
    peak_usage_time: datetime
    peak_tokens_per_minute: int

    # Efficiency metrics
    average_context_utilization: float = 0.0
    average_response_efficiency: float = 0.0
    cache_hit_rate: float = 0.0

    # Provider breakdown
    provider_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    persona_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class CostCalculator:
    """Calculates costs for different providers and models."""

    def __init__(self):
        """Initialize with default cost information."""
        self.cost_data: Dict[str, TokenCost] = {}
        self._initialize_default_costs()

    def _initialize_default_costs(self) -> None:
        """Initialize with known provider costs (as of 2024)."""
        # Claude costs (Anthropic)
        self.cost_data["claude:claude-3-5-sonnet-20241022"] = TokenCost(
            provider_name="claude",
            model_name="claude-3-5-sonnet-20241022",
            input_cost_per_1k=3.00,
            output_cost_per_1k=15.00,
        )
        self.cost_data["claude:claude-3-haiku-20240307"] = TokenCost(
            provider_name="claude",
            model_name="claude-3-haiku-20240307",
            input_cost_per_1k=0.25,
            output_cost_per_1k=1.25,
        )

        # OpenAI costs
        self.cost_data["openai:gpt-4o"] = TokenCost(
            provider_name="openai",
            model_name="gpt-4o",
            input_cost_per_1k=5.00,
            output_cost_per_1k=15.00,
            cached_cost_per_1k=2.50,  # OpenAI has cached pricing
        )
        self.cost_data["openai:gpt-4o-mini"] = TokenCost(
            provider_name="openai",
            model_name="gpt-4o-mini",
            input_cost_per_1k=0.15,
            output_cost_per_1k=0.60,
            cached_cost_per_1k=0.075,
        )

        # Google Gemini costs
        self.cost_data["gemini:gemini-1.5-pro"] = TokenCost(
            provider_name="gemini",
            model_name="gemini-1.5-pro",
            input_cost_per_1k=3.50,
            output_cost_per_1k=10.50,
        )

        # Default fallback for unknown models
        self.cost_data["default"] = TokenCost(
            provider_name="unknown",
            model_name="unknown",
            input_cost_per_1k=2.00,
            output_cost_per_1k=6.00,
        )

    def get_cost_info(self, provider_name: str, model_name: str) -> TokenCost:
        """Get cost information for a provider and model."""
        key = f"{provider_name}:{model_name}"
        return self.cost_data.get(key, self.cost_data["default"])

    def calculate_cost(
        self,
        provider_name: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate cost for token usage."""
        cost_info = self.get_cost_info(provider_name, model_name)
        return cost_info.calculate_cost(input_tokens, output_tokens, cached_tokens)

    def update_cost_data(
        self, provider_name: str, model_name: str, cost_info: TokenCost
    ) -> None:
        """Update cost information for a provider/model."""
        key = f"{provider_name}:{model_name}"
        self.cost_data[key] = cost_info
        logger.info(f"Updated cost data for {key}")


class TokenEstimator:
    """Estimates token counts using various methods."""

    def __init__(self):
        """Initialize token estimator."""
        self._tiktoken_encoding = None
        self._initialize_tiktoken()

    def _initialize_tiktoken(self) -> None:
        """Initialize tiktoken for OpenAI-style token counting."""
        try:
            import tiktoken

            self._tiktoken_encoding = tiktoken.get_encoding(
                "cl100k_base"
            )  # GPT-4 encoding
        except ImportError:
            logger.warning(
                "tiktoken not available, falling back to character-based estimation"
            )
            self._tiktoken_encoding = None

    def estimate_tokens(
        self, text: str, method: TokenCountMethod = TokenCountMethod.ESTIMATED
    ) -> int:
        """Estimate token count for text."""
        if method == TokenCountMethod.TIKTOKEN and self._tiktoken_encoding:
            return len(self._tiktoken_encoding.encode(text))
        elif method == TokenCountMethod.ESTIMATED:
            # Rough estimation: ~4 characters per token for English text
            return max(1, len(text) // 4)
        else:
            return max(1, len(text) // 4)  # Fallback to estimation

    def estimate_tokens_by_provider(
        self, text: str, provider_name: str, model_name: str
    ) -> int:
        """Estimate tokens using provider-specific logic."""
        # Provider-specific estimation logic can be added here
        base_estimate = self.estimate_tokens(text)

        # Adjust for provider differences
        if provider_name == "claude":
            # Claude tends to count slightly fewer tokens
            return int(base_estimate * 0.95)
        elif provider_name == "openai":
            # Use tiktoken if available
            if self._tiktoken_encoding:
                return len(self._tiktoken_encoding.encode(text))
        elif provider_name == "gemini":
            # Gemini uses different tokenization
            return int(base_estimate * 1.05)

        return base_estimate


class TokenUsageTracker:
    """Advanced token usage tracking with pattern analysis."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_history_size: int = 50000,
    ):
        """Initialize the token usage tracker."""
        self.storage_path = storage_path or Path.home() / ".omnimancer" / "token_usage"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.max_history_size = max_history_size
        self.usage_history: deque = deque(maxlen=max_history_size)
        self.cost_calculator = CostCalculator()
        self.token_estimator = TokenEstimator()

        # Current session tracking
        self.session_usage: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tokens": 0, "cost": 0.0, "requests": 0}
        )

        self._lock = threading.RLock()
        logger.info("Token usage tracker initialized")

    def record_usage(
        self,
        provider_name: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        persona_name: Optional[str] = None,
        operation_id: Optional[str] = None,
        context_utilization: float = 0.0,
        response_efficiency: float = 0.0,
    ) -> DetailedTokenUsage:
        """Record detailed token usage."""
        with self._lock:
            # Calculate cost
            estimated_cost = self.cost_calculator.calculate_cost(
                provider_name,
                model_name,
                input_tokens,
                output_tokens,
                cached_tokens,
            )

            # Create detailed usage record
            usage = DetailedTokenUsage(
                total_tokens=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                provider_name=provider_name,
                model_name=model_name,
                persona_name=persona_name or "unknown",
                operation_id=operation_id or "",
                estimated_cost=estimated_cost,
                context_utilization=context_utilization,
                response_efficiency=response_efficiency,
                count_method=TokenCountMethod.API_REPORTED,
            )

            # Store in history
            self.usage_history.append(usage)

            # Update session tracking
            session_key = f"{provider_name}:{model_name}"
            self.session_usage[session_key]["tokens"] += usage.total_tokens
            self.session_usage[session_key]["cost"] += estimated_cost
            self.session_usage[session_key]["requests"] += 1

            logger.debug(
                f"Recorded {usage.total_tokens} tokens for {provider_name}:{model_name} (${estimated_cost:.4f})"
            )
            return usage

    def estimate_and_record_usage(
        self,
        provider_name: str,
        model_name: str,
        input_text: str,
        output_text: str,
        persona_name: Optional[str] = None,
        operation_id: Optional[str] = None,
    ) -> DetailedTokenUsage:
        """Estimate and record token usage from text."""
        input_tokens = self.token_estimator.estimate_tokens_by_provider(
            input_text, provider_name, model_name
        )
        output_tokens = self.token_estimator.estimate_tokens_by_provider(
            output_text, provider_name, model_name
        )

        usage = self.record_usage(
            provider_name=provider_name,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            persona_name=persona_name,
            operation_id=operation_id,
        )
        usage.count_method = TokenCountMethod.ESTIMATED
        return usage

    def get_usage_summary(
        self, time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        with self._lock:
            if not self.usage_history:
                return {
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "total_requests": 0,
                }

            # Filter by time window
            cutoff_time = datetime.now() - time_window if time_window else None
            relevant_usage = [
                usage
                for usage in self.usage_history
                if not cutoff_time or usage.timestamp >= cutoff_time
            ]

            if not relevant_usage:
                return {
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "total_requests": 0,
                }

            # Calculate totals
            total_tokens = sum(u.total_tokens for u in relevant_usage)
            total_cost = sum(u.estimated_cost for u in relevant_usage)
            total_requests = len(relevant_usage)

            # Provider breakdown
            provider_stats = defaultdict(
                lambda: {
                    "tokens": 0,
                    "cost": 0.0,
                    "requests": 0,
                    "models": set(),
                }
            )
            for usage in relevant_usage:
                key = usage.provider_name
                provider_stats[key]["tokens"] += usage.total_tokens
                provider_stats[key]["cost"] += usage.estimated_cost
                provider_stats[key]["requests"] += 1
                provider_stats[key]["models"].add(usage.model_name)

            # Convert sets to lists for JSON serialization
            for stats in provider_stats.values():
                stats["models"] = list(stats["models"])

            # Persona breakdown
            persona_stats = defaultdict(
                lambda: {"tokens": 0, "cost": 0.0, "requests": 0}
            )
            for usage in relevant_usage:
                if usage.persona_name:
                    persona_stats[usage.persona_name]["tokens"] += usage.total_tokens
                    persona_stats[usage.persona_name]["cost"] += usage.estimated_cost
                    persona_stats[usage.persona_name]["requests"] += 1

            # Efficiency metrics
            efficiency_scores = [
                u.context_utilization
                for u in relevant_usage
                if u.context_utilization > 0
            ]
            response_scores = [
                u.response_efficiency
                for u in relevant_usage
                if u.response_efficiency > 0
            ]

            return {
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "total_requests": total_requests,
                "average_tokens_per_request": (
                    total_tokens / total_requests if total_requests > 0 else 0
                ),
                "average_cost_per_request": (
                    total_cost / total_requests if total_requests > 0 else 0
                ),
                "provider_breakdown": dict(provider_stats),
                "persona_breakdown": dict(persona_stats),
                "average_context_utilization": (
                    statistics.mean(efficiency_scores) if efficiency_scores else 0
                ),
                "average_response_efficiency": (
                    statistics.mean(response_scores) if response_scores else 0
                ),
                "time_period": (
                    f"{time_window.total_seconds():.0f} seconds"
                    if time_window
                    else "all time"
                ),
                "period_start": (cutoff_time.isoformat() if cutoff_time else None),
                "period_end": datetime.now().isoformat(),
            }

    def analyze_usage_patterns(self, days_back: int = 7) -> UsagePattern:
        """Analyze usage patterns over a period."""
        with self._lock:
            time_period = timedelta(days=days_back)
            cutoff_time = datetime.now() - time_period

            relevant_usage = [
                usage for usage in self.usage_history if usage.timestamp >= cutoff_time
            ]

            if not relevant_usage:
                return UsagePattern(
                    time_period=time_period,
                    total_requests=0,
                    total_tokens=0,
                    total_cost=0.0,
                    average_tokens_per_request=0,
                    average_cost_per_request=0,
                    peak_usage_time=datetime.now(),
                    peak_tokens_per_minute=0,
                )

            # Basic metrics
            total_tokens = sum(u.total_tokens for u in relevant_usage)
            total_cost = sum(u.estimated_cost for u in relevant_usage)
            total_requests = len(relevant_usage)

            # Find peak usage
            # Group by hour and find the hour with most tokens
            hourly_usage = defaultdict(int)
            for usage in relevant_usage:
                hour_key = usage.timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_usage[hour_key] += usage.total_tokens

            peak_hour = (
                max(hourly_usage.keys(), key=hourly_usage.get)
                if hourly_usage
                else datetime.now()
            )
            peak_tokens_per_hour = hourly_usage[peak_hour] if hourly_usage else 0

            # Provider breakdown
            provider_breakdown = defaultdict(
                lambda: {"tokens": 0, "cost": 0.0, "requests": 0}
            )
            for usage in relevant_usage:
                key = usage.provider_name
                provider_breakdown[key]["tokens"] += usage.total_tokens
                provider_breakdown[key]["cost"] += usage.estimated_cost
                provider_breakdown[key]["requests"] += 1

            # Persona breakdown
            persona_breakdown = defaultdict(
                lambda: {"tokens": 0, "cost": 0.0, "requests": 0}
            )
            for usage in relevant_usage:
                key = usage.persona_name or "unknown"
                persona_breakdown[key]["tokens"] += usage.total_tokens
                persona_breakdown[key]["cost"] += usage.estimated_cost
                persona_breakdown[key]["requests"] += 1

            # Efficiency metrics
            efficiency_scores = [
                u.context_utilization
                for u in relevant_usage
                if u.context_utilization > 0
            ]
            response_scores = [
                u.response_efficiency
                for u in relevant_usage
                if u.response_efficiency > 0
            ]
            cached_requests = len([u for u in relevant_usage if u.cached_tokens > 0])

            return UsagePattern(
                time_period=time_period,
                total_requests=total_requests,
                total_tokens=total_tokens,
                total_cost=total_cost,
                average_tokens_per_request=(
                    total_tokens / total_requests if total_requests > 0 else 0
                ),
                average_cost_per_request=(
                    total_cost / total_requests if total_requests > 0 else 0
                ),
                peak_usage_time=peak_hour,
                peak_tokens_per_minute=int(peak_tokens_per_hour / 60),
                average_context_utilization=(
                    statistics.mean(efficiency_scores) if efficiency_scores else 0
                ),
                average_response_efficiency=(
                    statistics.mean(response_scores) if response_scores else 0
                ),
                cache_hit_rate=(
                    (cached_requests / total_requests * 100)
                    if total_requests > 0
                    else 0
                ),
                provider_breakdown=dict(provider_breakdown),
                persona_breakdown=dict(persona_breakdown),
            )

    def get_cost_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get cost optimization suggestions based on usage patterns."""
        suggestions = []

        # Analyze recent usage (last 24 hours)
        recent_summary = self.get_usage_summary(timedelta(hours=24))

        # Suggest provider switching if using expensive models heavily
        provider_breakdown = recent_summary.get("provider_breakdown", {})
        for provider, stats in provider_breakdown.items():
            if stats["cost"] > 5.0:  # More than $5 in 24 hours
                suggestions.append(
                    {
                        "type": "provider_switching",
                        "title": f"High cost on {provider}",
                        "description": f"${stats['cost']:.2f} spent on {provider} in 24 hours",
                        "suggestion": "Consider switching to a more cost-effective model for routine tasks",
                        "potential_savings": "30-70%",
                    }
                )

        # Suggest caching for repeated patterns
        if recent_summary.get("total_requests", 0) > 100:
            avg_cost = recent_summary.get("average_cost_per_request", 0)
            if avg_cost > 0.01:  # More than 1 cent per request
                suggestions.append(
                    {
                        "type": "response_caching",
                        "title": "High request frequency detected",
                        "description": f"{recent_summary['total_requests']} requests with avg cost ${avg_cost:.3f}",
                        "suggestion": "Implement response caching for similar queries",
                        "potential_savings": "20-50%",
                    }
                )

        return suggestions

    def save_usage_history(self, filename: Optional[str] = None) -> None:
        """Save usage history to disk."""
        if not filename:
            filename = f"token_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.storage_path / filename

        with self._lock:
            usage_data = [asdict(usage) for usage in self.usage_history]

        with open(filepath, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "total_records": len(usage_data),
                        "tracker_version": "1.0",
                    },
                    "usage_history": usage_data,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"Saved {len(usage_data)} usage records to {filepath}")

    def load_usage_history(self, filename: str) -> None:
        """Load usage history from disk."""
        filepath = self.storage_path / filename

        if not filepath.exists():
            logger.error(f"Usage history file not found: {filepath}")
            return

        with open(filepath, "r") as f:
            data = json.load(f)

        with self._lock:
            for usage_dict in data.get("usage_history", []):
                # Convert timestamp back to datetime
                if isinstance(usage_dict.get("timestamp"), str):
                    usage_dict["timestamp"] = datetime.fromisoformat(
                        usage_dict["timestamp"]
                    )

                usage = DetailedTokenUsage(**usage_dict)
                self.usage_history.append(usage)

        logger.info(
            f"Loaded {len(data.get('usage_history', []))} usage records from {filepath}"
        )


# Global token tracker instance
_token_tracker: Optional[TokenUsageTracker] = None
_tracker_lock = threading.Lock()


def get_token_tracker() -> TokenUsageTracker:
    """Get the global token tracker instance."""
    global _token_tracker
    if _token_tracker is None:
        with _tracker_lock:
            if _token_tracker is None:
                _token_tracker = TokenUsageTracker()
    return _token_tracker


def set_token_tracker(tracker: TokenUsageTracker) -> None:
    """Set the global token tracker instance."""
    global _token_tracker
    with _tracker_lock:
        _token_tracker = tracker
