"""
Utilities module for Omnimancer.

This module contains utility functions and classes for error handling,
retry logic, encryption, and other common functionality.
"""

from .errors import (
    AuthenticationError,
    ConfigurationError,
    OmnimancerError,
    ProviderError,
    RateLimitError,
)

__all__ = [
    "OmnimancerError",
    "ConfigurationError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
]
