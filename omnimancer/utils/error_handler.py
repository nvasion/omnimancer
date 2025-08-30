"""
Error handling utilities for Omnimancer.

This module provides comprehensive error handling, graceful degradation,
and user-friendly error messages with suggested solutions.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from .errors import (
    AuthenticationError,
    MCPConnectionError,
    MCPServerError,
    MCPTimeoutError,
    ModelNotFoundError,
    NetworkError,
    ProviderConfigurationError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Centralized error handling and recovery system.

    This class provides methods for handling different types of errors,
    suggesting solutions, and implementing graceful degradation.
    """

    def __init__(self):
        """Initialize the error handler."""
        self.error_history: List[Dict[str, Any]] = []
        self.provider_status: Dict[str, Dict[str, Any]] = {}
        self.mcp_server_status: Dict[str, Dict[str, Any]] = {}

    def handle_provider_error(
        self,
        error: Exception,
        provider_name: str,
        available_providers: List[str] = None,
    ) -> Tuple[str, List[str], bool]:
        """
        Handle provider-specific errors and suggest solutions.

        Args:
            error: The exception that occurred
            provider_name: Name of the provider that failed
            available_providers: List of available alternative providers

        Returns:
            Tuple of (error_message, suggestions, should_retry)
        """
        available_providers = available_providers or []
        suggestions = []
        should_retry = False

        # Record error in history
        self._record_error(error, provider_name, "provider")

        if isinstance(error, AuthenticationError):
            error_msg = f"Authentication failed for {provider_name}"
            suggestions.extend(
                [
                    f"Check your {provider_name} API key configuration",
                    f"Verify API key permissions for {provider_name}",
                    "Run '/config' to update your API key",
                ]
            )

            # Suggest alternative providers
            if available_providers:
                working_providers = self._get_working_providers(available_providers)
                if working_providers:
                    suggestions.append(
                        f"Try switching to: {', '.join(working_providers)}"
                    )

        elif isinstance(error, RateLimitError):
            error_msg = f"Rate limit exceeded for {provider_name}"
            should_retry = True

            if hasattr(error, "retry_after") and error.retry_after:
                suggestions.append(f"Wait {error.retry_after} seconds before retrying")
            else:
                suggestions.append("Wait a few minutes before retrying")

            # Suggest alternative providers for immediate use
            if available_providers:
                working_providers = self._get_working_providers(available_providers)
                if working_providers:
                    suggestions.append(
                        f"Use alternative provider: {', '.join(working_providers)}"
                    )

        elif isinstance(error, QuotaExceededError):
            error_msg = f"Usage quota exceeded for {provider_name}"

            if hasattr(error, "reset_date") and error.reset_date:
                suggestions.append(f"Quota resets on: {error.reset_date}")
            else:
                suggestions.append("Check your account usage limits")

            suggestions.append(f"Review {provider_name} billing and usage")

            # Suggest alternative providers
            if available_providers:
                working_providers = self._get_working_providers(available_providers)
                if working_providers:
                    suggestions.append(f"Switch to: {', '.join(working_providers)}")

        elif isinstance(error, ModelNotFoundError):
            error_msg = f"Model not found or unavailable on {provider_name}"

            if hasattr(error, "get_suggestions"):
                model_suggestions = error.get_suggestions()
                if model_suggestions:
                    suggestions.append(
                        f"Try these models: {', '.join(model_suggestions)}"
                    )

            suggestions.extend(
                [
                    f"Check available models with '/models {provider_name}'",
                    "Verify model name spelling and availability",
                ]
            )

        elif isinstance(error, NetworkError):
            error_msg = f"Network error connecting to {provider_name}"
            should_retry = True

            suggestions.extend(
                [
                    "Check your internet connection",
                    "Verify firewall settings",
                    "Try again in a few moments",
                ]
            )

            # For Ollama specifically
            if provider_name.lower() == "ollama":
                suggestions.extend(
                    [
                        "Make sure Ollama is running: 'ollama serve'",
                        "Check Ollama server URL in configuration",
                    ]
                )

        elif isinstance(error, ProviderUnavailableError):
            error_msg = f"{provider_name} is temporarily unavailable"
            should_retry = True

            if hasattr(error, "estimated_recovery") and error.estimated_recovery:
                suggestions.append(f"Estimated recovery: {error.estimated_recovery}")

            if hasattr(error, "fallback_providers") and error.fallback_providers:
                suggestions.append(
                    f"Use fallback: {', '.join(error.fallback_providers)}"
                )

        elif isinstance(error, ProviderConfigurationError):
            error_msg = f"Configuration error for {provider_name}"

            if hasattr(error, "suggested_fix") and error.suggested_fix:
                suggestions.append(error.suggested_fix)

            suggestions.extend(
                [
                    f"Review {provider_name} configuration",
                    "Run '/config' to update settings",
                ]
            )

        else:
            error_msg = f"Unexpected error with {provider_name}: {str(error)}"
            suggestions.extend(
                [
                    "Try again in a few moments",
                    "Check provider status and configuration",
                ]
            )

            if available_providers:
                working_providers = self._get_working_providers(available_providers)
                if working_providers:
                    suggestions.append(
                        f"Try alternative: {', '.join(working_providers)}"
                    )

        # Update provider status
        self._update_provider_status(provider_name, error)

        return error_msg, suggestions, should_retry

    def handle_mcp_error(
        self,
        error: Exception,
        server_name: str = None,
        available_servers: List[str] = None,
    ) -> Tuple[str, List[str], bool]:
        """
        Handle MCP-specific errors and suggest solutions.

        Args:
            error: The exception that occurred
            server_name: Name of the MCP server that failed
            available_servers: List of available alternative servers

        Returns:
            Tuple of (error_message, suggestions, should_retry)
        """
        available_servers = available_servers or []
        suggestions = []
        should_retry = False

        # Record error in history
        self._record_error(error, server_name or "unknown", "mcp")

        if isinstance(error, MCPConnectionError):
            error_msg = f"Failed to connect to MCP server"
            if server_name:
                error_msg += f": {server_name}"

            should_retry = True
            suggestions.extend(
                [
                    "Check if the MCP server command is correct",
                    "Verify server dependencies are installed",
                    "Check server logs for startup errors",
                ]
            )

            if hasattr(error, "retry_suggestion") and error.retry_suggestion:
                suggestions.append(error.retry_suggestion)

        elif isinstance(error, MCPTimeoutError):
            error_msg = f"MCP server operation timed out"
            if server_name:
                error_msg += f" on {server_name}"

            should_retry = True
            suggestions.extend(
                [
                    "The operation may be taking longer than expected",
                    "Try increasing timeout settings",
                    "Check if the server is overloaded",
                ]
            )

            if hasattr(error, "operation") and error.operation:
                suggestions.append(f"Operation '{error.operation}' timed out")

        elif isinstance(error, MCPServerError):
            error_msg = f"MCP server error"
            if server_name:
                error_msg += f" on {server_name}"

            suggestions.extend(
                [
                    "Check MCP server configuration",
                    "Verify server is properly installed",
                    "Review server logs for errors",
                ]
            )

            # Suggest disabling problematic server
            if server_name:
                suggestions.append(
                    f"Consider temporarily disabling '{server_name}' server"
                )

        else:
            error_msg = f"MCP error: {str(error)}"
            suggestions.extend(
                [
                    "Check MCP server configuration",
                    "Try restarting MCP servers with '/mcp reload'",
                ]
            )

        # Update MCP server status
        if server_name:
            self._update_mcp_status(server_name, error)

        return error_msg, suggestions, should_retry

    def get_graceful_degradation_options(
        self,
        failed_provider: str,
        available_providers: List[str],
        required_capabilities: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Get options for graceful degradation when a provider fails.

        Args:
            failed_provider: Name of the provider that failed
            available_providers: List of available providers
            required_capabilities: List of required capabilities (e.g., 'tools', 'multimodal')

        Returns:
            Dictionary with degradation options
        """
        required_capabilities = required_capabilities or []

        # Filter working providers
        working_providers = self._get_working_providers(available_providers)

        # Filter by capabilities if specified
        if required_capabilities:
            # This would need to be implemented with actual capability checking
            # For now, we'll assume all providers are capable
            capable_providers = working_providers
        else:
            capable_providers = working_providers

        degradation_options = {
            "fallback_providers": capable_providers,
            "degraded_functionality": [],
            "recommendations": [],
        }

        if not capable_providers:
            degradation_options["degraded_functionality"].extend(
                [
                    "No alternative providers available",
                    "Some features may be unavailable",
                ]
            )
            degradation_options["recommendations"].extend(
                [
                    "Check provider configurations",
                    "Verify API keys and connectivity",
                ]
            )
        else:
            degradation_options["recommendations"].extend(
                [
                    f"Switch to: {', '.join(capable_providers[:2])}",
                    "Configure backup providers for reliability",
                ]
            )

        return degradation_options

    def should_retry_operation(
        self, error: Exception, attempt_count: int, max_attempts: int = 3
    ) -> Tuple[bool, float]:
        """
        Determine if an operation should be retried and with what delay.

        Args:
            error: The exception that occurred
            attempt_count: Current attempt number
            max_attempts: Maximum number of attempts

        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        if attempt_count >= max_attempts:
            return False, 0.0

        # Exponential backoff base delay
        base_delay = 2 ** (attempt_count - 1)

        if isinstance(error, RateLimitError):
            # Use provider-specified retry delay if available
            if hasattr(error, "retry_after") and error.retry_after:
                return True, float(error.retry_after)
            else:
                # Default rate limit backoff
                return True, min(base_delay * 2, 60.0)

        elif isinstance(error, NetworkError):
            # Network errors are often transient
            return True, min(base_delay, 10.0)

        elif isinstance(error, ProviderUnavailableError):
            # Provider outages may resolve quickly
            return True, min(base_delay * 3, 30.0)

        elif isinstance(error, MCPTimeoutError):
            # MCP timeouts might resolve with retry
            return True, min(base_delay, 5.0)

        elif isinstance(error, MCPConnectionError):
            # Connection errors might be transient
            return True, min(base_delay, 15.0)

        elif isinstance(
            error,
            (AuthenticationError, QuotaExceededError, ModelNotFoundError),
        ):
            # These errors typically don't resolve with retry
            return False, 0.0

        else:
            # Generic retry for unknown errors
            return True, min(base_delay, 5.0)

    def get_error_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get a summary of recent errors for monitoring and debugging.

        Args:
            time_window_hours: Hours to look back for errors

        Returns:
            Dictionary with error summary
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_errors = [
            error for error in self.error_history if error["timestamp"] > cutoff_time
        ]

        # Group errors by type and component
        error_counts = {}
        provider_errors = {}
        mcp_errors = {}

        for error in recent_errors:
            error_type = error["error_type"]
            component = error["component"]
            component_name = error["component_name"]

            # Overall error counts
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

            # Provider-specific errors
            if component == "provider":
                if component_name not in provider_errors:
                    provider_errors[component_name] = {}
                provider_errors[component_name][error_type] = (
                    provider_errors[component_name].get(error_type, 0) + 1
                )

            # MCP-specific errors
            elif component == "mcp":
                if component_name not in mcp_errors:
                    mcp_errors[component_name] = {}
                mcp_errors[component_name][error_type] = (
                    mcp_errors[component_name].get(error_type, 0) + 1
                )

        return {
            "time_window_hours": time_window_hours,
            "total_errors": len(recent_errors),
            "error_counts": error_counts,
            "provider_errors": provider_errors,
            "mcp_errors": mcp_errors,
            "most_recent_error": recent_errors[-1] if recent_errors else None,
        }

    def _record_error(self, error: Exception, component_name: str, component_type: str):
        """Record an error in the error history."""
        error_record = {
            "timestamp": datetime.now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "component": component_type,
            "component_name": component_name,
        }

        self.error_history.append(error_record)

        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

        logger.warning(f"Error recorded: {error_record}")

    def _update_provider_status(self, provider_name: str, error: Exception):
        """Update provider status based on error."""
        if provider_name not in self.provider_status:
            self.provider_status[provider_name] = {
                "status": "unknown",
                "last_error": None,
                "error_count": 0,
                "last_success": None,
            }

        status = self.provider_status[provider_name]
        status["last_error"] = datetime.now()
        status["error_count"] += 1

        # Determine status based on error type
        if isinstance(error, (AuthenticationError, ProviderConfigurationError)):
            status["status"] = "configuration_error"
        elif isinstance(error, (RateLimitError, QuotaExceededError)):
            status["status"] = "quota_limited"
        elif isinstance(error, NetworkError):
            status["status"] = "network_error"
        elif isinstance(error, ProviderUnavailableError):
            status["status"] = "unavailable"
        else:
            status["status"] = "error"

    def _update_mcp_status(self, server_name: str, error: Exception):
        """Update MCP server status based on error."""
        if server_name not in self.mcp_server_status:
            self.mcp_server_status[server_name] = {
                "status": "unknown",
                "last_error": None,
                "error_count": 0,
                "last_success": None,
            }

        status = self.mcp_server_status[server_name]
        status["last_error"] = datetime.now()
        status["error_count"] += 1

        # Determine status based on error type
        if isinstance(error, MCPConnectionError):
            status["status"] = "connection_failed"
        elif isinstance(error, MCPTimeoutError):
            status["status"] = "timeout"
        else:
            status["status"] = "error"

    def _get_working_providers(self, available_providers: List[str]) -> List[str]:
        """Get list of providers that are likely working based on recent status."""
        working_providers = []

        for provider in available_providers:
            status = self.provider_status.get(provider, {})

            # Consider provider working if:
            # 1. No recent errors, or
            # 2. Recent success after errors, or
            # 3. Only transient errors (network, rate limit)
            if (
                not status.get("last_error")
                or (
                    status.get("last_success")
                    and status.get("last_success") > status.get("last_error")
                )
                or status.get("status") in ["network_error", "quota_limited"]
            ):
                working_providers.append(provider)

        # If no providers have good status, return all (benefit of doubt)
        if not working_providers:
            working_providers = available_providers

        return working_providers


# Global error handler instance
error_handler = ErrorHandler()


def handle_provider_error(
    error: Exception, provider_name: str, available_providers: List[str] = None
):
    """Convenience function for handling provider errors."""
    return error_handler.handle_provider_error(
        error, provider_name, available_providers
    )


def handle_mcp_error(
    error: Exception,
    server_name: str = None,
    available_servers: List[str] = None,
):
    """Convenience function for handling MCP errors."""
    return error_handler.handle_mcp_error(error, server_name, available_servers)


def should_retry_operation(error: Exception, attempt_count: int, max_attempts: int = 3):
    """Convenience function for retry logic."""
    return error_handler.should_retry_operation(error, attempt_count, max_attempts)


def get_graceful_degradation_options(
    failed_provider: str,
    available_providers: List[str],
    required_capabilities: List[str] = None,
):
    """Convenience function for graceful degradation."""
    return error_handler.get_graceful_degradation_options(
        failed_provider, available_providers, required_capabilities
    )
