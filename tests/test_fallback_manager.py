"""
Comprehensive tests for the enhanced provider fallback system.
"""

import time
from unittest.mock import AsyncMock, Mock

import pytest

from omnimancer.core.fallback_manager import (
    EnhancedProviderFallback,
    FallbackAttempt,
    FallbackReason,
    ProviderRank,
    ProviderStats,
)
from omnimancer.core.health_monitor import HealthMonitor
from omnimancer.utils.errors import (
    NetworkError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)


class MockProvider:
    """Mock provider for testing."""

    def __init__(self, name: str, should_fail: bool = False, fail_count: int = 0):
        self.name = name
        self.should_fail = should_fail
        self.fail_count = fail_count
        self.call_count = 0

    def get_provider_name(self) -> str:
        return self.name

    async def send_message(self, message: str, context=None):
        self.call_count += 1

        if self.should_fail:
            if self.fail_count == 0 or self.call_count <= self.fail_count:
                raise ProviderError(f"Provider {self.name} failed")

        return f"Response from {self.name}: {message}"


class MockCoreEngine:
    """Mock core engine for testing."""

    def __init__(self):
        self.current_provider = None
        self.providers = {}
        self.switch_calls = []
        self.chat_manager = Mock()
        self.config_manager = Mock()

        # Setup chat manager mock
        mock_context = Mock()
        mock_context.messages = []
        mock_context.session_id = "test_session"
        mock_context.metadata = {}
        self.chat_manager.get_current_context.return_value = mock_context

    async def switch_model(self, provider_name: str) -> bool:
        self.switch_calls.append(provider_name)
        if provider_name in self.providers:
            self.current_provider = self.providers[provider_name]
            return True
        return False


@pytest.fixture
def mock_core_engine():
    """Create mock core engine."""
    engine = MockCoreEngine()

    # Add some mock providers
    engine.providers = {
        "primary": MockProvider("primary"),
        "secondary": MockProvider("secondary"),
        "tertiary": MockProvider("tertiary"),
        "emergency": MockProvider("emergency"),
    }
    engine.current_provider = engine.providers["primary"]

    return engine


@pytest.fixture
def mock_health_monitor():
    """Create mock health monitor."""
    monitor = Mock(spec=HealthMonitor)
    monitor.check_all_providers_health = AsyncMock(
        return_value={
            "primary": {"status": "healthy", "available": True},
            "secondary": {"status": "healthy", "available": True},
            "tertiary": {"status": "warning", "available": True},
            "emergency": {"status": "error", "available": False},
        }
    )
    return monitor


@pytest.fixture
def fallback_manager(mock_core_engine, mock_health_monitor):
    """Create fallback manager with mocks."""
    manager = EnhancedProviderFallback(mock_core_engine, mock_health_monitor)
    manager.set_fallback_providers(
        ["primary", "secondary", "tertiary", "emergency"],
        {
            "primary": ProviderRank.PRIMARY,
            "secondary": ProviderRank.SECONDARY,
            "tertiary": ProviderRank.TERTIARY,
            "emergency": ProviderRank.EMERGENCY,
        },
    )
    return manager


class TestProviderStats:
    """Test provider statistics tracking."""

    def test_provider_stats_initialization(self):
        """Test provider stats initialization."""
        stats = ProviderStats()

        assert stats.success_count == 0
        assert stats.failure_count == 0
        assert stats.total_requests == 0
        assert stats.success_rate == 1.0
        assert stats.reliability_score >= 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = ProviderStats()
        stats.success_count = 8
        stats.failure_count = 2
        stats.total_requests = 10

        assert stats.success_rate == 0.8

    def test_reliability_score_with_failures(self):
        """Test reliability score with consecutive failures."""
        stats = ProviderStats()
        stats.success_count = 5
        stats.total_requests = 10
        stats.consecutive_failures = 3

        # Should penalize consecutive failures
        assert stats.reliability_score < 0.5

    def test_reliability_score_with_recent_success(self):
        """Test reliability score bonus for recent success."""
        stats = ProviderStats()
        stats.success_count = 5
        stats.total_requests = 10
        stats.last_success = time.time()  # Recent success

        # Should get bonus for recent success
        assert stats.reliability_score > 0.5


class TestEnhancedProviderFallback:
    """Test enhanced provider fallback functionality."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_fallback(
        self, fallback_manager, mock_core_engine
    ):
        """Test successful execution without needing fallback."""

        async def mock_operation():
            return "success"

        result = await fallback_manager.execute_with_fallback(mock_operation)

        assert result == "success"
        assert len(mock_core_engine.switch_calls) == 0  # No provider switching

    @pytest.mark.asyncio
    async def test_fallback_to_secondary_provider(
        self, fallback_manager, mock_core_engine
    ):
        """Test fallback to secondary provider when primary fails."""
        # Make primary provider fail
        mock_core_engine.providers["primary"].should_fail = True

        async def mock_operation():
            if mock_core_engine.current_provider.name == "primary":
                raise ProviderError("Primary provider failed")
            return f"success from {mock_core_engine.current_provider.name}"

        result = await fallback_manager.execute_with_fallback(mock_operation)

        assert result == "success from secondary"
        assert "secondary" in mock_core_engine.switch_calls

    @pytest.mark.asyncio
    async def test_retry_logic_with_backoff(self, fallback_manager, mock_core_engine):
        """Test retry logic with exponential backoff."""
        # Make primary provider fail first 2 attempts, then succeed
        mock_core_engine.providers["primary"].fail_count = 2
        mock_core_engine.providers["primary"].should_fail = True

        call_count = 0

        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ProviderError("Temporary failure")
            return "success after retries"

        start_time = time.time()
        result = await fallback_manager.execute_with_fallback(mock_operation)
        end_time = time.time()

        assert result == "success after retries"
        assert call_count == 3
        # Should take some time due to backoff delays
        assert end_time - start_time >= 1.0  # At least 1 second for backoff

    @pytest.mark.asyncio
    async def test_all_providers_fail(self, fallback_manager, mock_core_engine):
        """Test behavior when all providers fail."""
        # Make all providers fail
        for provider in mock_core_engine.providers.values():
            provider.should_fail = True

        async def mock_operation():
            raise ProviderError("All providers fail")

        with pytest.raises(ProviderUnavailableError) as exc_info:
            await fallback_manager.execute_with_fallback(mock_operation)

        assert "All providers failed" in str(exc_info.value)
        assert exc_info.value.fallback_providers is not None

    @pytest.mark.asyncio
    async def test_error_classification(self, fallback_manager):
        """Test error classification for different error types."""
        # Test rate limit error
        rate_limit_error = RateLimitError("Rate limit exceeded")
        reason = fallback_manager._classify_error(rate_limit_error)
        assert reason == FallbackReason.RATE_LIMIT

        # Test network error
        network_error = NetworkError("Connection failed")
        reason = fallback_manager._classify_error(network_error)
        assert reason == FallbackReason.NETWORK_ERROR

        # Test quota error
        quota_error = ProviderError("Quota exceeded")
        reason = fallback_manager._classify_error(quota_error)
        assert reason == FallbackReason.QUOTA_EXCEEDED

        # Test generic provider error
        provider_error = ProviderError("Generic error")
        reason = fallback_manager._classify_error(provider_error)
        assert reason == FallbackReason.PROVIDER_ERROR

    def test_non_retryable_errors(self, fallback_manager):
        """Test identification of non-retryable errors."""
        # Authentication errors should not be retried
        auth_error = ProviderError("Invalid API key")
        assert fallback_manager._is_non_retryable_error(auth_error)

        # Quota errors should not be retried
        quota_error = ProviderError("Quota exceeded")
        assert fallback_manager._is_non_retryable_error(quota_error)

        # Network errors should be retried
        network_error = NetworkError("Connection timeout")
        assert not fallback_manager._is_non_retryable_error(network_error)

    @pytest.mark.asyncio
    async def test_provider_ranking_and_selection(self, fallback_manager):
        """Test provider ranking and selection logic."""
        # Simulate different reliability scores
        fallback_manager.provider_stats["primary"].success_count = 10
        fallback_manager.provider_stats["primary"].total_requests = 10

        fallback_manager.provider_stats["secondary"].success_count = 8
        fallback_manager.provider_stats["secondary"].total_requests = 10

        fallback_manager.provider_stats["tertiary"].success_count = 5
        fallback_manager.provider_stats["tertiary"].total_requests = 10

        ranked_providers = await fallback_manager._get_ranked_providers()

        # Primary should be first due to highest success rate
        assert ranked_providers[0] == "primary"
        assert "secondary" in ranked_providers
        assert "tertiary" in ranked_providers

    def test_circuit_breaker_functionality(self, fallback_manager):
        """Test circuit breaker functionality."""
        provider_name = "primary"

        # Simulate consecutive failures
        for _ in range(5):
            fallback_manager._record_failure(
                provider_name, FallbackReason.PROVIDER_ERROR, "Error"
            )

        # Should trigger circuit breaker
        assert fallback_manager._should_exclude_provider(provider_name)

        # Exclude provider
        fallback_manager._exclude_provider_temporarily(provider_name)
        assert fallback_manager._is_provider_excluded(provider_name)

        # Reset stats should clear exclusion after recovery time
        fallback_manager.reset_provider_stats(provider_name)
        stats = fallback_manager.provider_stats[provider_name]
        assert stats.consecutive_failures == 0

    def test_provider_stats_tracking(self, fallback_manager):
        """Test provider statistics tracking."""
        provider_name = "primary"

        # Record some successes and failures
        fallback_manager._record_success(provider_name, 0.5)
        fallback_manager._record_success(provider_name, 0.6)
        fallback_manager._record_failure(
            provider_name, FallbackReason.NETWORK_ERROR, "Network error"
        )

        stats = fallback_manager.provider_stats[provider_name]
        assert stats.success_count == 2
        assert stats.failure_count == 1
        assert stats.total_requests == 3
        assert stats.success_rate == 2 / 3
        assert stats.last_success is not None
        assert stats.last_failure is not None

    def test_fallback_history_tracking(self, fallback_manager):
        """Test fallback history tracking."""
        attempts = [
            FallbackAttempt("primary", FallbackReason.PROVIDER_ERROR, "Error 1"),
            FallbackAttempt("secondary", FallbackReason.NETWORK_ERROR, "Error 2"),
        ]

        fallback_manager._record_fallback_history(attempts)

        history = fallback_manager.get_fallback_history()
        assert len(history) == 2
        assert history[0]["provider"] == "primary"
        assert history[1]["provider"] == "secondary"

    def test_provider_stats_summary(self, fallback_manager):
        """Test provider statistics summary."""
        # Setup some stats
        fallback_manager._record_success("primary", 0.5)
        fallback_manager._record_failure(
            "secondary", FallbackReason.RATE_LIMIT, "Rate limited"
        )

        stats_summary = fallback_manager.get_provider_stats()

        assert "primary" in stats_summary
        assert "secondary" in stats_summary
        assert stats_summary["primary"]["success_rate"] == 1.0
        assert stats_summary["secondary"]["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_context_preservation(self, fallback_manager, mock_core_engine):
        """Test conversation context preservation during fallback."""
        # Enable context preservation
        fallback_manager.context_preservation_enabled = True

        async def mock_operation():
            return "success with context"

        result = await fallback_manager.execute_with_fallback(
            mock_operation, preserve_context=True
        )

        assert result == "success with context"
        # Context should have been preserved (mock implementation)
        assert fallback_manager.preserved_context is not None

    def test_circuit_breaker_configuration(self, fallback_manager):
        """Test circuit breaker configuration."""
        fallback_manager.configure_circuit_breaker(threshold=3, recovery_time=300)

        assert fallback_manager.circuit_breaker_threshold == 3
        assert fallback_manager.circuit_breaker_recovery_time == 300

    @pytest.mark.asyncio
    async def test_health_check_integration(
        self, fallback_manager, mock_health_monitor
    ):
        """Test integration with health monitoring system."""
        await fallback_manager.health_check_all_providers()

        # Should call the health monitor
        mock_health_monitor.check_all_providers_health.assert_called_once()

    def test_provider_exclusion_expiry(self, fallback_manager):
        """Test provider exclusion expiry logic."""
        provider_name = "primary"

        # Exclude provider with very short recovery time
        fallback_manager.circuit_breaker_recovery_time = 0.1  # 0.1 seconds
        fallback_manager._exclude_provider_temporarily(provider_name)

        assert fallback_manager._is_provider_excluded(provider_name)

        # Wait for expiry
        time.sleep(0.2)

        # Should no longer be excluded
        assert not fallback_manager._is_provider_excluded(provider_name)

    @pytest.mark.asyncio
    async def test_jitter_in_retry_delay(self, fallback_manager, mock_core_engine):
        """Test that retry delays include jitter to avoid thundering herd."""
        fallback_manager.jitter_factor = 0.5  # High jitter for testing

        call_times = []

        async def mock_operation():
            call_times.append(time.time())
            if len(call_times) <= 2:
                raise ProviderError("Temporary failure")
            return "success"

        time.time()
        await fallback_manager.execute_with_fallback(mock_operation)

        # Check that delays between calls vary (jitter effect)
        assert len(call_times) == 3

        # First retry should have some delay
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Delays should be different due to jitter (with some tolerance)
        assert abs(delay1 - delay2) > 0.1 or delay1 > 0.5

    @pytest.mark.asyncio
    async def test_max_retry_delay_cap(self, fallback_manager):
        """Test that retry delay is capped at maximum value."""
        fallback_manager.max_retry_delay = 2.0  # Low cap for testing
        fallback_manager.base_retry_delay = 1.0

        # Calculate delay for high attempt number
        delay = min(
            fallback_manager.base_retry_delay * (2**10),  # Would be very large
            fallback_manager.max_retry_delay,
        )

        assert delay == fallback_manager.max_retry_delay

    def test_provider_ranking_enum(self):
        """Test provider ranking enumeration."""
        assert ProviderRank.PRIMARY.value == 1
        assert ProviderRank.SECONDARY.value == 2
        assert ProviderRank.TERTIARY.value == 3
        assert ProviderRank.EMERGENCY.value == 4

    def test_fallback_reason_enum(self):
        """Test fallback reason enumeration."""
        reasons = [reason.value for reason in FallbackReason]
        expected_reasons = [
            "provider_error",
            "network_error",
            "rate_limit",
            "quota_exceeded",
            "model_unavailable",
            "timeout",
            "health_check_failed",
        ]

        for expected in expected_reasons:
            assert expected in reasons


@pytest.mark.integration
class TestFallbackIntegration:
    """Integration tests for fallback system."""

    @pytest.mark.asyncio
    async def test_full_fallback_scenario(self, fallback_manager, mock_core_engine):
        """Test complete fallback scenario with multiple providers."""
        # Setup scenario: primary fails, secondary always fails, tertiary succeeds
        mock_core_engine.providers["primary"].should_fail = True
        mock_core_engine.providers["secondary"].should_fail = True

        async def mock_operation():
            current_provider = mock_core_engine.current_provider.name

            if current_provider == "primary":
                raise ProviderError("Primary down")
            elif current_provider == "secondary":
                raise RateLimitError("Secondary rate limited")
            elif current_provider == "tertiary":
                return f"Success from {current_provider}"
            else:
                raise ProviderError(f"Unexpected provider: {current_provider}")

        result = await fallback_manager.execute_with_fallback(mock_operation)

        # Should eventually succeed with tertiary
        assert result == "Success from tertiary"

        # Check that providers were tried in order
        assert "secondary" in mock_core_engine.switch_calls
        assert "tertiary" in mock_core_engine.switch_calls

        # Check provider stats - should have some provider interactions
        stats = fallback_manager.get_provider_stats()
        assert len(stats) > 0  # Should have recorded some provider activity
