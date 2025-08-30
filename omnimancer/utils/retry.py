"""
Retry logic utilities for Omnimancer.

This module provides retry mechanisms with exponential backoff
for handling transient errors in API calls and other operations.
"""

import asyncio
import logging
import random
from typing import Any, Callable, Optional

from .errors import NetworkError, RateLimitError

logger = logging.getLogger(__name__)


class RetryHandler:
    """
    Handles retry logic with exponential backoff.

    This class provides configurable retry behavior for operations
    that may fail due to transient errors like network issues or
    rate limiting.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize the retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if an operation should be retried.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if the operation should be retried, False otherwise
        """
        if attempt >= self.max_retries:
            return False

        # Always retry network errors
        if isinstance(error, NetworkError):
            return True

        # Retry rate limit errors
        if isinstance(error, RateLimitError):
            return True

        # Don't retry other types of errors by default
        return False

    def calculate_delay(
        self, attempt: int, rate_limit_retry_after: Optional[int] = None
    ) -> float:
        """
        Calculate the delay before the next retry attempt.

        Args:
            attempt: Current attempt number (0-indexed)
            rate_limit_retry_after: Specific delay for rate limit errors

        Returns:
            Delay in seconds
        """
        if rate_limit_retry_after is not None:
            # Use the specific delay provided by the API
            delay = float(rate_limit_retry_after)
        else:
            # Calculate exponential backoff delay
            delay = self.base_delay * (self.exponential_base**attempt)

        # Cap the delay at max_delay
        delay = min(delay, self.max_delay)

        # Add jitter to avoid thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    async def execute_with_retry(
        self, func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call

        Raises:
            The last exception if all retries are exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success - log if this was a retry
                if attempt > 0:
                    logger.info(f"Operation succeeded after {attempt} retries")

                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not self.should_retry(e, attempt):
                    logger.debug(f"Not retrying after attempt {attempt}: {e}")
                    break

                # Calculate delay
                retry_after = None
                if isinstance(e, RateLimitError):
                    retry_after = e.retry_after

                delay = self.calculate_delay(attempt, retry_after)

                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )

                # Wait before retrying
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            f"All {self.max_retries} retries exhausted. Last error: {last_exception}"
        )
        raise last_exception

    def execute_sync_with_retry(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Execute a synchronous function with retry logic.

        Args:
            func: The synchronous function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call

        Raises:
            The last exception if all retries are exhausted
        """
        import time

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)

                # Success - log if this was a retry
                if attempt > 0:
                    logger.info(f"Operation succeeded after {attempt} retries")

                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                if not self.should_retry(e, attempt):
                    logger.debug(f"Not retrying after attempt {attempt}: {e}")
                    break

                # Calculate delay
                retry_after = None
                if isinstance(e, RateLimitError):
                    retry_after = e.retry_after

                delay = self.calculate_delay(attempt, retry_after)

                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )

                # Wait before retrying
                time.sleep(delay)

        # All retries exhausted
        logger.error(
            f"All {self.max_retries} retries exhausted. Last error: {last_exception}"
        )
        raise last_exception

    def retry(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Convenience method for synchronous retry execution.

        Args:
            func: The function to execute with retry
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call
        """
        return self.execute_sync_with_retry(func, *args, **kwargs)
