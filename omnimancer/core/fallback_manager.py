"""
Enhanced Provider Fallback and Retry Logic for Omnimancer.

This module provides intelligent fallback system with provider health monitoring,
context preservation, and seamless failover capabilities.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..utils.errors import (
    NetworkError,
    ProviderError,
    ProviderUnavailableError,
)
from .health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reasons for provider fallback."""

    PROVIDER_ERROR = "provider_error"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_UNAVAILABLE = "model_unavailable"
    TIMEOUT = "timeout"
    HEALTH_CHECK_FAILED = "health_check_failed"


class ProviderRank(Enum):
    """Provider ranking based on reliability and performance."""

    PRIMARY = 1  # Best performance, highest reliability
    SECONDARY = 2  # Good performance, good reliability
    TERTIARY = 3  # Acceptable performance, lower reliability
    EMERGENCY = 4  # Last resort, basic functionality


@dataclass
class FallbackAttempt:
    """Record of a fallback attempt."""

    provider_name: str
    reason: FallbackReason
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    success: bool = False
    response_time: Optional[float] = None


@dataclass
class ProviderStats:
    """Statistics for provider performance tracking."""

    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    rank: ProviderRank = ProviderRank.SECONDARY
    consecutive_failures: int = 0
    total_requests: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    @property
    def reliability_score(self) -> float:
        """Calculate reliability score (0.0 to 1.0)."""
        base_score = self.success_rate

        # Penalize consecutive failures
        failure_penalty = min(self.consecutive_failures * 0.1, 0.8)

        # Bonus for recent success
        recency_bonus = 0.0
        if self.last_success:
            time_since_success = time.time() - self.last_success
            if time_since_success < 300:  # 5 minutes
                recency_bonus = 0.1

        return max(0.0, base_score - failure_penalty + recency_bonus)


class EnhancedProviderFallback:
    """
    Enhanced provider fallback system with intelligent provider selection,
    health monitoring, and context preservation.
    """

    def __init__(self, core_engine, health_monitor: Optional[HealthMonitor] = None):
        """
        Initialize the enhanced fallback manager.

        Args:
            core_engine: Reference to CoreEngine instance
            health_monitor: Optional health monitor instance
        """
        self.core_engine = core_engine
        self.health_monitor = health_monitor or HealthMonitor()

        # Configuration
        self.max_retry_attempts = 3
        self.base_retry_delay = 1.0
        self.max_retry_delay = 32.0
        self.jitter_factor = 0.1
        self.health_check_interval = 300  # 5 minutes

        # Provider management
        self.fallback_providers: List[str] = []
        self.provider_rankings: Dict[str, ProviderRank] = {}
        self.provider_stats: Dict[str, ProviderStats] = {}
        self.circuit_breaker_threshold = 5  # failures before circuit break
        self.circuit_breaker_recovery_time = 600  # 10 minutes

        # Context preservation
        self.context_preservation_enabled = True
        self.preserved_context: Optional[Dict[str, Any]] = None

        # Fallback history
        self.fallback_history: List[FallbackAttempt] = []
        self.max_history_size = 100

        # Excluded providers (temporarily disabled)
        self.excluded_providers: Set[str] = set()
        self.exclusion_expiry: Dict[str, float] = {}

    def set_fallback_providers(
        self,
        providers: List[str],
        rankings: Optional[Dict[str, ProviderRank]] = None,
    ):
        """
        Set the list of fallback providers with optional rankings.

        Args:
            providers: List of provider names in priority order
            rankings: Optional dict mapping provider names to rankings
        """
        self.fallback_providers = providers.copy()

        if rankings:
            self.provider_rankings.update(rankings)
        else:
            # Auto-assign rankings based on order
            for i, provider in enumerate(providers):
                if i == 0:
                    self.provider_rankings[provider] = ProviderRank.PRIMARY
                elif i < 3:
                    self.provider_rankings[provider] = ProviderRank.SECONDARY
                elif i < 6:
                    self.provider_rankings[provider] = ProviderRank.TERTIARY
                else:
                    self.provider_rankings[provider] = ProviderRank.EMERGENCY

        # Initialize stats for new providers
        for provider in providers:
            if provider not in self.provider_stats:
                self.provider_stats[provider] = ProviderStats()

    async def execute_with_fallback(
        self,
        operation_func: Callable,
        *args,
        preserve_context: bool = True,
        **kwargs,
    ) -> Any:
        """
        Execute operation with intelligent fallback and retry logic.

        Args:
            operation_func: Function to execute
            *args: Arguments for the function
            preserve_context: Whether to preserve conversation context
            **kwargs: Keyword arguments for the function

        Returns:
            Result from successful operation execution

        Raises:
            Exception: If all providers and retries fail
        """
        original_provider = self.core_engine.current_provider
        original_provider_name = (
            original_provider.get_provider_name() if original_provider else None
        )

        # Preserve context if enabled
        if preserve_context and self.context_preservation_enabled:
            await self._preserve_context()

        # Get ranked list of providers to try
        providers_to_try = await self._get_ranked_providers()

        last_error = None
        fallback_attempts = []

        for provider_name in providers_to_try:
            # Skip excluded providers
            if self._is_provider_excluded(provider_name):
                continue

            try:
                # Switch to provider if needed
                if not original_provider or original_provider_name != provider_name:
                    switch_success = await self.core_engine.switch_model(provider_name)
                    if not switch_success:
                        continue

                # Execute with retries for this provider
                result = await self._execute_with_retries(
                    operation_func, provider_name, *args, **kwargs
                )

                # Record success
                self._record_success(provider_name, time.time())

                # Restore context if needed
                if preserve_context and self.preserved_context:
                    await self._restore_context()

                return result

            except Exception as e:
                last_error = e
                reason = self._classify_error(e)

                # Record failure
                self._record_failure(provider_name, reason, str(e))

                # Add to fallback attempts
                attempt = FallbackAttempt(
                    provider_name=provider_name,
                    reason=reason,
                    error=str(e),
                    success=False,
                )
                fallback_attempts.append(attempt)

                # Check if provider should be temporarily excluded
                if self._should_exclude_provider(provider_name):
                    self._exclude_provider_temporarily(provider_name)

                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        # All providers failed
        self._record_fallback_history(fallback_attempts)

        # Try to restore original provider
        if original_provider_name:
            try:
                await self.core_engine.switch_model(original_provider_name)
            except Exception:
                pass  # Best effort restore

        # Raise the last error
        if last_error:
            raise ProviderUnavailableError(
                f"All providers failed. Last error: {last_error}",
                fallback_providers=[a.provider_name for a in fallback_attempts],
            )
        else:
            raise ProviderError("No providers available for fallback")

    async def _execute_with_retries(
        self, operation_func: Callable, provider_name: str, *args, **kwargs
    ) -> Any:
        """Execute operation with exponential backoff retry logic."""
        last_error = None

        for attempt in range(self.max_retry_attempts):
            try:
                start_time = time.time()
                result = await operation_func(*args, **kwargs)

                # Record response time
                response_time = time.time() - start_time
                self._update_response_time(provider_name, response_time)

                return result

            except Exception as e:
                last_error = e

                # Don't retry certain types of errors
                if self._is_non_retryable_error(e):
                    raise

                # Calculate delay with exponential backoff and jitter
                if attempt < self.max_retry_attempts - 1:
                    delay = min(
                        self.base_retry_delay * (2**attempt),
                        self.max_retry_delay,
                    )
                    # Add jitter
                    jitter = delay * self.jitter_factor * (random.random() - 0.5)
                    delay += jitter

                    logger.debug(
                        f"Retrying {provider_name} in {delay:.2f}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)

        # All retries failed
        if last_error:
            raise last_error
        else:
            raise ProviderError(
                f"All retry attempts failed for provider {provider_name}"
            )

    async def _get_ranked_providers(self) -> List[str]:
        """Get list of providers ranked by health and performance."""
        available_providers = []

        # Start with current provider if available
        if self.core_engine.current_provider:
            current_name = self.core_engine.current_provider.get_provider_name()
            if current_name not in self.excluded_providers:
                available_providers.append(current_name)

        # Add fallback providers
        for provider in self.fallback_providers:
            if (
                provider not in available_providers
                and provider not in self.excluded_providers
            ):
                available_providers.append(provider)

        # Sort by reliability score and ranking
        def sort_key(provider_name: str) -> Tuple[float, int]:
            stats = self.provider_stats.get(provider_name, ProviderStats())
            ranking = self.provider_rankings.get(provider_name, ProviderRank.SECONDARY)
            return (-stats.reliability_score, ranking.value)

        available_providers.sort(key=sort_key)

        logger.debug(f"Ranked providers: {available_providers}")
        return available_providers

    def _classify_error(self, error: Exception) -> FallbackReason:
        """Classify error to determine appropriate fallback strategy."""
        error_str = str(error).lower()
        type(error).__name__

        if "rate limit" in error_str or "429" in error_str:
            return FallbackReason.RATE_LIMIT
        elif "quota" in error_str or "usage limit" in error_str:
            return FallbackReason.QUOTA_EXCEEDED
        elif "timeout" in error_str or "timed out" in error_str:
            return FallbackReason.TIMEOUT
        elif "network" in error_str or "connection" in error_str:
            return FallbackReason.NETWORK_ERROR
        elif "model" in error_str and (
            "not found" in error_str or "unavailable" in error_str
        ):
            return FallbackReason.MODEL_UNAVAILABLE
        elif isinstance(error, NetworkError):
            return FallbackReason.NETWORK_ERROR
        else:
            return FallbackReason.PROVIDER_ERROR

    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Check if error should not be retried."""
        # Don't retry authentication errors, quota exceeded, etc.
        error_str = str(error).lower()
        non_retryable_patterns = [
            "invalid api key",
            "authentication failed",
            "unauthorized",
            "quota exceeded",
            "model not found",
            "invalid model",
        ]

        return any(pattern in error_str for pattern in non_retryable_patterns)

    def _record_success(self, provider_name: str, response_time: float):
        """Record successful operation for provider."""
        if provider_name not in self.provider_stats:
            self.provider_stats[provider_name] = ProviderStats()

        stats = self.provider_stats[provider_name]
        stats.success_count += 1
        stats.total_requests += 1
        stats.last_success = time.time()
        stats.consecutive_failures = 0  # Reset failure counter

        logger.debug(f"Recorded success for {provider_name}")

    def _record_failure(
        self, provider_name: str, reason: FallbackReason, error_msg: str
    ):
        """Record failed operation for provider."""
        if provider_name not in self.provider_stats:
            self.provider_stats[provider_name] = ProviderStats()

        stats = self.provider_stats[provider_name]
        stats.failure_count += 1
        stats.total_requests += 1
        stats.last_failure = time.time()
        stats.consecutive_failures += 1

        logger.debug(f"Recorded failure for {provider_name}: {reason.value}")

    def _update_response_time(self, provider_name: str, response_time: float):
        """Update average response time for provider."""
        if provider_name not in self.provider_stats:
            self.provider_stats[provider_name] = ProviderStats()

        stats = self.provider_stats[provider_name]
        if stats.avg_response_time == 0:
            stats.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.3
            stats.avg_response_time = (
                alpha * response_time + (1 - alpha) * stats.avg_response_time
            )

    def _should_exclude_provider(self, provider_name: str) -> bool:
        """Check if provider should be temporarily excluded."""
        stats = self.provider_stats.get(provider_name)
        if not stats:
            return False

        return stats.consecutive_failures >= self.circuit_breaker_threshold

    def _exclude_provider_temporarily(self, provider_name: str):
        """Temporarily exclude provider from fallback list."""
        self.excluded_providers.add(provider_name)
        self.exclusion_expiry[provider_name] = (
            time.time() + self.circuit_breaker_recovery_time
        )

        logger.warning(
            f"Temporarily excluded provider {provider_name} due to consecutive failures"
        )

    def _is_provider_excluded(self, provider_name: str) -> bool:
        """Check if provider is currently excluded."""
        if provider_name not in self.excluded_providers:
            return False

        # Check if exclusion has expired
        if provider_name in self.exclusion_expiry:
            if time.time() > self.exclusion_expiry[provider_name]:
                self.excluded_providers.discard(provider_name)
                del self.exclusion_expiry[provider_name]
                logger.info(
                    f"Re-enabled provider {provider_name} after recovery period"
                )
                return False

        return True

    async def _preserve_context(self):
        """Preserve current conversation context."""
        try:
            if hasattr(self.core_engine, "chat_manager"):
                context = self.core_engine.chat_manager.get_current_context()
                self.preserved_context = {
                    "messages": (
                        context.messages.copy() if hasattr(context, "messages") else []
                    ),
                    "session_id": getattr(context, "session_id", None),
                    "metadata": getattr(context, "metadata", {}),
                }
                logger.debug("Preserved conversation context")
        except Exception as e:
            logger.warning(f"Failed to preserve context: {e}")

    async def _restore_context(self):
        """Restore preserved conversation context."""
        try:
            if self.preserved_context and hasattr(self.core_engine, "chat_manager"):
                # This would require implementation in ChatManager
                # For now, just log the attempt
                logger.debug("Context restoration would be implemented here")
        except Exception as e:
            logger.warning(f"Failed to restore context: {e}")

    def _record_fallback_history(self, attempts: List[FallbackAttempt]):
        """Record fallback attempts in history."""
        self.fallback_history.extend(attempts)

        # Trim history if too large
        if len(self.fallback_history) > self.max_history_size:
            self.fallback_history = self.fallback_history[-self.max_history_size :]

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all providers."""
        stats = {}
        for provider_name, provider_stats in self.provider_stats.items():
            stats[provider_name] = {
                "success_rate": provider_stats.success_rate,
                "reliability_score": provider_stats.reliability_score,
                "avg_response_time": provider_stats.avg_response_time,
                "total_requests": provider_stats.total_requests,
                "consecutive_failures": provider_stats.consecutive_failures,
                "rank": provider_stats.rank.name,
                "excluded": provider_name in self.excluded_providers,
            }
        return stats

    def get_fallback_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent fallback history."""
        recent_history = (
            self.fallback_history[-limit:] if limit else self.fallback_history
        )

        return [
            {
                "provider": attempt.provider_name,
                "reason": attempt.reason.value,
                "error": attempt.error,
                "timestamp": attempt.timestamp,
                "success": attempt.success,
            }
            for attempt in recent_history
        ]

    async def health_check_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all configured providers."""
        if not hasattr(self.core_engine, "config_manager"):
            return {}

        try:
            config = self.core_engine.config_manager.get_config()
            health_result = await self.health_monitor.check_all_providers_health(
                config.providers, force=True
            )
            return health_result
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {}

    def reset_provider_stats(self, provider_name: Optional[str] = None):
        """Reset statistics for specific provider or all providers."""
        if provider_name:
            if provider_name in self.provider_stats:
                self.provider_stats[provider_name] = ProviderStats()
                logger.info(f"Reset stats for provider {provider_name}")
        else:
            self.provider_stats.clear()
            logger.info("Reset stats for all providers")

    def configure_circuit_breaker(self, threshold: int = 5, recovery_time: int = 600):
        """Configure circuit breaker settings."""
        self.circuit_breaker_threshold = threshold
        self.circuit_breaker_recovery_time = recovery_time
        logger.info(
            f"Circuit breaker configured: threshold={threshold}, recovery_time={recovery_time}s"
        )
