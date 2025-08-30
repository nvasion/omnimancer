"""
Unit tests for the RetryHandler class.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from omnimancer.utils.errors import NetworkError, RateLimitError
from omnimancer.utils.retry import RetryHandler


class TestRetryHandler:
    """Test cases for RetryHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.retry_handler = RetryHandler()

    def test_init_default_values(self):
        """Test initialization with default values."""
        handler = RetryHandler()

        assert handler.max_retries == 3
        assert handler.base_delay == 1.0
        assert handler.max_delay == 60.0
        assert handler.exponential_base == 2.0
        assert handler.jitter is True

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        handler = RetryHandler(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
        )

        assert handler.max_retries == 5
        assert handler.base_delay == 0.5
        assert handler.max_delay == 30.0
        assert handler.exponential_base == 1.5
        assert handler.jitter is False

    def test_calculate_delay_no_jitter(self):
        """Test delay calculation without jitter."""
        handler = RetryHandler(
            base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=False
        )

        # Test exponential backoff
        assert handler.calculate_delay(0) == 1.0
        assert handler.calculate_delay(1) == 2.0
        assert handler.calculate_delay(2) == 4.0
        assert handler.calculate_delay(3) == 8.0

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        handler = RetryHandler(
            base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=True
        )

        # With jitter, delay should be within expected range
        delay = handler.calculate_delay(1)
        expected_base = 2.0
        # Jitter is 10% of delay, so range is 90% to 110% of expected
        assert 0.9 * expected_base <= delay <= 1.1 * expected_base

    def test_calculate_delay_max_limit(self):
        """Test delay calculation respects max limit."""
        handler = RetryHandler(
            base_delay=1.0, exponential_base=2.0, max_delay=5.0, jitter=False
        )

        # Should not exceed max_delay
        assert handler.calculate_delay(10) == 5.0

    def test_calculate_delay_with_rate_limit_retry_after(self):
        """Test delay calculation with rate limit retry_after."""
        handler = RetryHandler(jitter=False)  # Disable jitter for exact comparison

        # Should use the specific retry_after value
        delay = handler.calculate_delay(1, rate_limit_retry_after=10)
        assert delay == 10.0

    def test_should_retry_network_error(self):
        """Test retry decision for network errors."""
        assert (
            self.retry_handler.should_retry(NetworkError("Connection failed"), 0)
            is True
        )
        assert self.retry_handler.should_retry(NetworkError("Timeout"), 1) is True

    def test_should_retry_rate_limit_error(self):
        """Test retry decision for rate limit errors."""
        assert (
            self.retry_handler.should_retry(RateLimitError("Rate limited"), 0) is True
        )
        assert (
            self.retry_handler.should_retry(RateLimitError("Too many requests"), 1)
            is True
        )

    def test_should_retry_other_error(self):
        """Test retry decision for other errors."""
        assert self.retry_handler.should_retry(ValueError("Invalid value"), 0) is False
        assert self.retry_handler.should_retry(TypeError("Type error"), 1) is False

    def test_should_retry_max_attempts_reached(self):
        """Test retry decision when max attempts reached."""
        handler = RetryHandler(max_retries=2)

        assert handler.should_retry(NetworkError("Connection failed"), 2) is False
        assert handler.should_retry(RateLimitError("Rate limited"), 3) is False

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_attempt(self):
        """Test async retry with success on first attempt."""
        mock_func = AsyncMock(return_value="success")

        result = await self.retry_handler.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_retries(self):
        """Test async retry with success after retries."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            NetworkError("Connection failed"),
            NetworkError("Connection failed"),
            "success",
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await self.retry_handler.execute_with_retry(mock_func)

            assert result == "success"
            assert mock_func.call_count == 3
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_retry_max_retries_exceeded(self):
        """Test async retry when max retries exceeded."""
        mock_func = AsyncMock()
        mock_func.side_effect = NetworkError("Connection failed")

        handler = RetryHandler(max_retries=2)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NetworkError, match="Connection failed"):
                await handler.execute_with_retry(mock_func)

            assert mock_func.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retryable_error(self):
        """Test async retry with non-retryable error."""
        mock_func = AsyncMock()
        mock_func.side_effect = ValueError("Invalid value")

        with pytest.raises(ValueError, match="Invalid value"):
            await self.retry_handler.execute_with_retry(mock_func)

        assert mock_func.call_count == 1  # No retries for non-retryable errors

    @pytest.mark.asyncio
    async def test_execute_with_retry_with_args_kwargs(self):
        """Test async retry with function arguments."""
        mock_func = AsyncMock(return_value="success")

        result = await self.retry_handler.execute_with_retry(
            mock_func, "arg1", "arg2", kwarg1="value1", kwarg2="value2"
        )

        assert result == "success"
        mock_func.assert_called_once_with(
            "arg1", "arg2", kwarg1="value1", kwarg2="value2"
        )

    @pytest.mark.asyncio
    async def test_execute_with_retry_sync_function(self):
        """Test executing sync function with async retry."""
        mock_func = Mock(return_value="success")

        result = await self.retry_handler.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_rate_limit_error(self):
        """Test retry with rate limit error that has retry_after."""
        # Create a handler without jitter for exact comparison
        handler = RetryHandler(jitter=False)

        # Create a mock RateLimitError with retry_after attribute
        rate_limit_error = RateLimitError("Rate limited")
        rate_limit_error.retry_after = 5

        mock_func = AsyncMock()
        mock_func.side_effect = [rate_limit_error, "success"]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await handler.execute_with_retry(mock_func)

            assert result == "success"
            assert mock_func.call_count == 2
            # Should use the retry_after value
            mock_sleep.assert_called_once_with(5.0)

    def test_execute_sync_with_retry_success_first_attempt(self):
        """Test sync retry with success on first attempt."""
        mock_func = Mock(return_value="success")

        result = self.retry_handler.execute_sync_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_execute_sync_with_retry_success_after_retries(self):
        """Test sync retry with success after retries."""
        mock_func = Mock()
        mock_func.side_effect = [
            NetworkError("Connection failed"),
            NetworkError("Connection failed"),
            "success",
        ]

        with patch("time.sleep") as mock_sleep:
            result = self.retry_handler.execute_sync_with_retry(mock_func)

            assert result == "success"
            assert mock_func.call_count == 3
            assert mock_sleep.call_count == 2

    def test_execute_sync_with_retry_max_retries_exceeded(self):
        """Test sync retry when max retries exceeded."""
        mock_func = Mock()
        mock_func.side_effect = NetworkError("Connection failed")

        handler = RetryHandler(max_retries=2)

        with patch("time.sleep"):
            with pytest.raises(NetworkError, match="Connection failed"):
                handler.execute_sync_with_retry(mock_func)

            assert mock_func.call_count == 3  # Initial + 2 retries

    def test_execute_sync_with_retry_non_retryable_error(self):
        """Test sync retry with non-retryable error."""
        mock_func = Mock()
        mock_func.side_effect = ValueError("Invalid value")

        with pytest.raises(ValueError, match="Invalid value"):
            self.retry_handler.execute_sync_with_retry(mock_func)

        assert mock_func.call_count == 1  # No retries for non-retryable errors

    def test_execute_sync_with_retry_with_args_kwargs(self):
        """Test sync retry with function arguments."""
        mock_func = Mock(return_value="success")

        result = self.retry_handler.execute_sync_with_retry(
            mock_func, "arg1", "arg2", kwarg1="value1", kwarg2="value2"
        )

        assert result == "success"
        mock_func.assert_called_once_with(
            "arg1", "arg2", kwarg1="value1", kwarg2="value2"
        )

    def test_retry_convenience_method(self):
        """Test the convenience retry method."""
        mock_func = Mock(return_value="success")

        result = self.retry_handler.retry(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_calculate_delay_negative_result_protection(self):
        """Test that calculate_delay never returns negative values."""
        handler = RetryHandler(base_delay=0.1, jitter=True)

        # Even with jitter, delay should never be negative
        for attempt in range(10):
            delay = handler.calculate_delay(attempt)
            assert delay >= 0

    def test_should_retry_boundary_conditions(self):
        """Test should_retry at boundary conditions."""
        handler = RetryHandler(max_retries=3)

        # At max retries boundary
        assert (
            handler.should_retry(NetworkError("test"), 2) is True
        )  # Last allowed retry
        assert (
            handler.should_retry(NetworkError("test"), 3) is False
        )  # Exceeds max retries

        # At zero retries
        assert handler.should_retry(NetworkError("test"), 0) is True

    def test_calculate_delay_zero_attempt(self):
        """Test delay calculation for zero attempt."""
        handler = RetryHandler(base_delay=2.0, jitter=False)

        # First attempt (attempt 0) should use base delay
        delay = handler.calculate_delay(0)
        assert delay == 2.0

    def test_calculate_delay_large_attempt(self):
        """Test delay calculation for large attempt numbers."""
        handler = RetryHandler(
            base_delay=1.0, exponential_base=2.0, max_delay=100.0, jitter=False
        )

        # Very large attempt should be capped at max_delay
        delay = handler.calculate_delay(20)
        assert delay == 100.0
