"""
Error classes for Omnimancer.

This module defines the exception hierarchy used throughout the application
to provide clear error handling and user feedback.
"""

__all__ = [
    # Base errors
    "OmnimancerError",
    "AgentError",
    # Configuration errors
    "ConfigurationError",
    "ProviderConfigurationError",
    "MCPConfigurationError",
    # Provider errors
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "NetworkError",
    "ModelNotFoundError",
    "ProviderUnavailableError",
    "QuotaExceededError",
    "ToolExecutionError",
    # Agent errors
    "SecurityError",
    "PermissionError",
    "ExecutionError",
    "TimeoutError",
    # Application errors
    "ValidationError",
    "ConversationError",
    # MCP errors
    "MCPError",
    "MCPServerError",
    "MCPToolError",
    "MCPConnectionError",
    "MCPTimeoutError",
]


class OmnimancerError(Exception):
    """
    Base exception class for all Omnimancer errors.

    All custom exceptions in Omnimancer should inherit from this class
    to provide consistent error handling.
    """

    def __init__(self, message: str, details: str = None):
        """
        Initialize the error.

        Args:
            message: Human-readable error message
            details: Additional technical details (optional)
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ConfigurationError(OmnimancerError):
    """
    Raised when there are configuration-related errors.

    This includes missing configuration files, invalid settings,
    or problems with configuration file format.
    """

    pass


class ProviderError(OmnimancerError):
    """
    Base class for AI provider-related errors.

    This is the parent class for all errors that occur when
    communicating with AI provider APIs.
    """

    def __init__(self, message: str, provider: str = None, details: str = None):
        """
        Initialize the provider error.

        Args:
            message: Human-readable error message
            provider: Name of the provider that caused the error
            details: Additional technical details
        """
        super().__init__(message, details)
        self.provider = provider

    def __str__(self) -> str:
        """String representation including provider name."""
        base_msg = super().__str__()
        if self.provider:
            return f"[{self.provider}] {base_msg}"
        return base_msg


class AuthenticationError(ProviderError):
    """
    Raised when API authentication fails.

    This typically indicates invalid API keys, expired tokens,
    or insufficient permissions.
    """

    pass


class RateLimitError(ProviderError):
    """
    Raised when API rate limits are exceeded.

    This indicates that too many requests have been made in a
    given time period and the client should wait before retrying.
    """

    def __init__(
        self,
        message: str,
        provider: str = None,
        retry_after: int = None,
        details: str = None,
    ):
        """
        Initialize the rate limit error.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            retry_after: Seconds to wait before retrying (if known)
            details: Additional technical details
        """
        super().__init__(message, provider, details)
        self.retry_after = retry_after

    def __str__(self) -> str:
        """String representation including retry information."""
        base_msg = super().__str__()
        if self.retry_after:
            return f"{base_msg}\nRetry after: {self.retry_after} seconds"
        return base_msg


class NetworkError(ProviderError):
    """
    Raised when network-related errors occur.

    This includes connection timeouts, DNS resolution failures,
    and other network connectivity issues.
    """

    pass


class ValidationError(OmnimancerError):
    """
    Raised when input validation fails.

    This includes invalid command parameters, malformed input,
    or data that doesn't meet expected formats.
    """

    pass


class ModelNotFoundError(ProviderError):
    """
    Raised when a requested model is not available.

    This occurs when trying to switch to a model that doesn't exist
    or is not accessible with the current API key.
    """

    def __init__(
        self,
        message: str,
        provider: str = None,
        model_name: str = None,
        available_models: list = None,
        details: str = None,
    ):
        """
        Initialize the model not found error.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            model_name: Name of the model that wasn't found
            available_models: List of available models (for suggestions)
            details: Additional technical details
        """
        super().__init__(message, provider, details)
        self.model_name = model_name
        self.available_models = available_models or []

    def get_suggestions(self) -> list:
        """
        Get suggested alternative models.

        Returns:
            List of suggested model names
        """
        if not self.available_models or not self.model_name:
            return []

        # Simple fuzzy matching for suggestions
        suggestions = []
        model_lower = self.model_name.lower()

        for model in self.available_models:
            model_lower_check = model.lower()
            # Check for partial matches or similar names
            if (
                model_lower in model_lower_check
                or model_lower_check in model_lower
                or any(part in model_lower_check for part in model_lower.split("-"))
            ):
                suggestions.append(model)

        return suggestions[:3]  # Return top 3 suggestions

    def __str__(self) -> str:
        """String representation with suggestions."""
        base_msg = super().__str__()
        suggestions = self.get_suggestions()

        if suggestions:
            suggestion_text = ", ".join(suggestions)
            return f"{base_msg}\nSuggested alternatives: {suggestion_text}"
        elif self.available_models:
            return (
                f"{base_msg}\nAvailable models: {', '.join(self.available_models[:5])}"
            )

        return base_msg


class ConversationError(OmnimancerError):
    """
    Raised when conversation management operations fail.

    This includes errors saving/loading conversations, context
    management issues, or conversation file corruption.
    """

    pass


class MCPError(OmnimancerError):
    """
    Base class for MCP (Model Context Protocol) related errors.

    This is the parent class for all errors that occur when
    working with MCP servers and tools.
    """

    pass


class MCPServerError(MCPError):
    """
    Raised when MCP server connection or communication fails.

    This includes server startup failures, connection timeouts,
    and communication protocol errors.
    """

    def __init__(self, message: str, server_name: str = None, details: str = None):
        """
        Initialize the MCP server error.

        Args:
            message: Human-readable error message
            server_name: Name of the MCP server that caused the error
            details: Additional technical details
        """
        super().__init__(message, details)
        self.server_name = server_name

    def __str__(self) -> str:
        """String representation including server name."""
        base_msg = super().__str__()
        if self.server_name:
            return f"[MCP Server: {self.server_name}] {base_msg}"
        return f"[MCP Server] {base_msg}"


class MCPToolError(MCPError):
    """
    Raised when MCP tool execution fails.

    This includes tool not found errors, execution failures,
    and invalid tool parameters.
    """

    def __init__(self, message: str, tool_name: str = None, details: str = None):
        """
        Initialize the MCP tool error.

        Args:
            message: Human-readable error message
            tool_name: Name of the tool that caused the error
            details: Additional technical details
        """
        super().__init__(message, details)
        self.tool_name = tool_name

    def __str__(self) -> str:
        """String representation including tool name."""
        base_msg = super().__str__()
        if self.tool_name:
            return f"[MCP Tool: {self.tool_name}] {base_msg}"
        return f"[MCP Tool] {base_msg}"


class MCPConfigurationError(MCPError):
    """
    Raised when MCP configuration is invalid.

    This includes missing server configurations, invalid parameters,
    and configuration file format errors.
    """

    pass


class ProviderUnavailableError(ProviderError):
    """
    Raised when a provider is temporarily unavailable.

    This includes service outages, maintenance windows,
    or temporary connectivity issues.
    """

    def __init__(
        self,
        message: str,
        provider: str = None,
        fallback_providers: list = None,
        estimated_recovery: str = None,
        details: str = None,
    ):
        """
        Initialize the provider unavailable error.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            fallback_providers: List of alternative providers
            estimated_recovery: Estimated recovery time
            details: Additional technical details
        """
        super().__init__(message, provider, details)
        self.fallback_providers = fallback_providers or []
        self.estimated_recovery = estimated_recovery

    def __str__(self) -> str:
        """String representation with fallback suggestions."""
        base_msg = super().__str__()

        if self.fallback_providers:
            fallback_text = ", ".join(
                str(provider) for provider in self.fallback_providers
            )
            base_msg += f"\nTry switching to: {fallback_text}"

        if self.estimated_recovery:
            base_msg += f"\nEstimated recovery: {self.estimated_recovery}"

        return base_msg


class ToolExecutionError(ProviderError):
    """
    Raised when tool execution fails during AI conversation.

    This includes MCP tool failures, invalid tool parameters,
    or tool execution timeouts.
    """

    def __init__(
        self,
        message: str,
        provider: str = None,
        tool_name: str = None,
        tool_args: dict = None,
        details: str = None,
    ):
        """
        Initialize the tool execution error.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            tool_name: Name of the tool that failed
            tool_args: Arguments passed to the tool
            details: Additional technical details
        """
        super().__init__(message, provider, details)
        self.tool_name = tool_name
        self.tool_args = tool_args or {}

    def __str__(self) -> str:
        """String representation with tool information."""
        base_msg = super().__str__()

        if self.tool_name:
            base_msg = f"Tool '{self.tool_name}' failed: {base_msg}"

        return base_msg


class QuotaExceededError(ProviderError):
    """
    Raised when API quota or usage limits are exceeded.

    This is different from rate limiting and typically indicates
    monthly/daily usage limits have been reached.
    """

    def __init__(
        self,
        message: str,
        provider: str = None,
        quota_type: str = None,
        reset_date: str = None,
        details: str = None,
    ):
        """
        Initialize the quota exceeded error.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            quota_type: Type of quota exceeded (e.g., 'monthly', 'tokens')
            reset_date: When the quota resets
            details: Additional technical details
        """
        super().__init__(message, provider, details)
        self.quota_type = quota_type
        self.reset_date = reset_date

    def __str__(self) -> str:
        """String representation with quota information."""
        base_msg = super().__str__()

        if self.quota_type:
            base_msg += f"\nQuota type: {self.quota_type}"

        if self.reset_date:
            base_msg += f"\nQuota resets: {self.reset_date}"

        return base_msg


class ProviderConfigurationError(ProviderError):
    """
    Raised when provider configuration is invalid or incomplete.

    This includes missing API keys, invalid endpoints,
    or incompatible configuration options.
    """

    def __init__(
        self,
        message: str,
        provider: str = None,
        config_field: str = None,
        suggested_fix: str = None,
        details: str = None,
    ):
        """
        Initialize the provider configuration error.

        Args:
            message: Human-readable error message
            provider: Name of the provider
            config_field: Configuration field that's invalid
            suggested_fix: Suggested fix for the configuration
            details: Additional technical details
        """
        super().__init__(message, provider, details)
        self.config_field = config_field
        self.suggested_fix = suggested_fix

    def __str__(self) -> str:
        """String representation with configuration help."""
        base_msg = super().__str__()

        if self.config_field:
            base_msg += f"\nConfiguration field: {self.config_field}"

        if self.suggested_fix:
            base_msg += f"\nSuggested fix: {self.suggested_fix}"

        return base_msg


class MCPConnectionError(MCPServerError):
    """
    Raised when MCP server connection fails.

    This includes connection timeouts, refused connections,
    and server startup failures.
    """

    def __init__(
        self,
        message: str,
        server_name: str = None,
        connection_type: str = None,
        retry_suggestion: str = None,
        details: str = None,
    ):
        """
        Initialize the MCP connection error.

        Args:
            message: Human-readable error message
            server_name: Name of the MCP server
            connection_type: Type of connection (e.g., 'stdio', 'tcp')
            retry_suggestion: Suggestion for retry
            details: Additional technical details
        """
        super().__init__(message, server_name, details)
        self.connection_type = connection_type
        self.retry_suggestion = retry_suggestion

    def __str__(self) -> str:
        """String representation with connection help."""
        base_msg = super().__str__()

        if self.connection_type:
            base_msg += f"\nConnection type: {self.connection_type}"

        if self.retry_suggestion:
            base_msg += f"\nSuggestion: {self.retry_suggestion}"

        return base_msg


class MCPTimeoutError(MCPServerError):
    """
    Raised when MCP server operations timeout.

    This includes tool execution timeouts, connection timeouts,
    and response timeouts.
    """

    def __init__(
        self,
        message: str,
        server_name: str = None,
        timeout_duration: float = None,
        operation: str = None,
        details: str = None,
    ):
        """
        Initialize the MCP timeout error.

        Args:
            message: Human-readable error message
            server_name: Name of the MCP server
            timeout_duration: Duration of the timeout in seconds
            operation: Operation that timed out
            details: Additional technical details
        """
        super().__init__(message, server_name, details)
        self.timeout_duration = timeout_duration
        self.operation = operation

    def __str__(self) -> str:
        """String representation with timeout information."""
        base_msg = super().__str__()

        if self.operation:
            base_msg += f"\nOperation: {self.operation}"

        if self.timeout_duration:
            base_msg += f"\nTimeout after: {self.timeout_duration}s"

        return base_msg


class AgentError(OmnimancerError):
    """
    Base class for agent-related errors.

    This includes errors that occur during agent operations,
    such as file system operations, command execution, or
    web requests performed by the agent.
    """

    pass


class SecurityError(AgentError):
    """
    Raised when a security violation is detected.

    This includes attempts to access forbidden paths,
    execute unauthorized commands, or perform unsafe operations.
    """

    pass


class PermissionError(AgentError):
    """
    Raised when insufficient permissions are detected.

    This includes file permission issues, command execution
    restrictions, or other permission-related failures.
    """

    pass


class ExecutionError(AgentError):
    """
    Raised when command or program execution fails.

    This includes process execution failures, command not found errors,
    non-zero exit codes, and other execution-related failures.
    """

    def __init__(self, message: str, details: str = None):
        """
        Initialize the execution error.

        Args:
            message: Human-readable error message
            details: Additional technical details (optional)
        """
        super().__init__(message, details)


class TimeoutError(AgentError):
    """
    Raised when operations exceed their timeout duration.

    This includes command execution timeouts, network timeouts,
    and other time-based operation failures.

    Note: This is distinct from Python's built-in TimeoutError
    and follows Omnimancer's error handling patterns.
    """

    def __init__(self, message: str, details: str = None):
        """
        Initialize the timeout error.

        Args:
            message: Human-readable error message
            details: Additional technical details (optional)
        """
        super().__init__(message, details)
