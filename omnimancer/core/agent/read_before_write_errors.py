"""
Error handling for read-before-write functionality.

This module provides comprehensive error handling for read-before-write operations,
including custom exceptions, error recovery strategies, and error context management.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReadBeforeWriteErrorType(Enum):
    """Types of errors that can occur in read-before-write operations."""

    FILE_READ_ERROR = "file_read_error"
    FILE_WRITE_ERROR = "file_write_error"
    USER_INTERFACE_ERROR = "user_interface_error"
    PERMISSION_ERROR = "permission_error"
    ENCODING_ERROR = "encoding_error"
    CONTENT_VALIDATION_ERROR = "content_validation_error"
    CALLBACK_ERROR = "callback_error"
    DIFF_GENERATION_ERROR = "diff_generation_error"
    PREVIEW_ERROR = "preview_error"
    USER_REJECTION = "user_rejection"
    TIMEOUT_ERROR = "timeout_error"


class RecoveryStrategy(Enum):
    """Strategies for recovering from read-before-write errors."""

    RETRY = "retry"
    FALLBACK_TO_REGULAR_WRITE = "fallback_to_regular_write"
    SKIP_OPERATION = "skip_operation"
    PROMPT_USER = "prompt_user"
    LOG_AND_CONTINUE = "log_and_continue"
    ABORT_OPERATION = "abort_operation"


@dataclass
class ErrorContext:
    """Context information for read-before-write errors."""

    error_type: ReadBeforeWriteErrorType
    file_path: str
    operation: str  # 'create' or 'modify'
    encoding: str
    content_length: int
    timestamp: datetime
    original_exception: Optional[Exception] = None
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_success: Optional[bool] = None
    additional_context: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_context is None:
            self.additional_context = {}


class ReadBeforeWriteError(Exception):
    """Base exception for read-before-write operations."""

    def __init__(
        self,
        message: str,
        error_type: ReadBeforeWriteErrorType,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.context = context
        self.original_exception = original_exception

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "message": self.message,
            "error_type": self.error_type.value,
            "context": {
                "file_path": self.context.file_path if self.context else None,
                "operation": self.context.operation if self.context else None,
                "encoding": self.context.encoding if self.context else None,
                "timestamp": (
                    self.context.timestamp.isoformat() if self.context else None
                ),
                "recovery_attempted": (
                    self.context.recovery_attempted if self.context else False
                ),
                "recovery_strategy": (
                    self.context.recovery_strategy.value
                    if self.context and self.context.recovery_strategy
                    else None
                ),
                "recovery_success": (
                    self.context.recovery_success if self.context else None
                ),
                "additional_context": (
                    self.context.additional_context if self.context else {}
                ),
            },
            "original_exception": (
                str(self.original_exception) if self.original_exception else None
            ),
        }


class FileReadError(ReadBeforeWriteError):
    """Error reading existing file content."""

    def __init__(self, file_path: str, original_exception: Exception):
        context = ErrorContext(
            error_type=ReadBeforeWriteErrorType.FILE_READ_ERROR,
            file_path=file_path,
            operation="read",
            encoding="unknown",
            content_length=0,
            timestamp=datetime.now(),
            original_exception=original_exception,
        )
        super().__init__(
            f"Failed to read existing content from {file_path}: {str(original_exception)}",
            ReadBeforeWriteErrorType.FILE_READ_ERROR,
            context,
            original_exception,
        )


class FileWriteError(ReadBeforeWriteError):
    """Error writing file content after review."""

    def __init__(self, file_path: str, original_exception: Exception):
        context = ErrorContext(
            error_type=ReadBeforeWriteErrorType.FILE_WRITE_ERROR,
            file_path=file_path,
            operation="write",
            encoding="unknown",
            content_length=0,
            timestamp=datetime.now(),
            original_exception=original_exception,
        )
        super().__init__(
            f"Failed to write reviewed content to {file_path}: {str(original_exception)}",
            ReadBeforeWriteErrorType.FILE_WRITE_ERROR,
            context,
            original_exception,
        )


class UserInterfaceError(ReadBeforeWriteError):
    """Error in user interface interaction."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        context = ErrorContext(
            error_type=ReadBeforeWriteErrorType.USER_INTERFACE_ERROR,
            file_path="unknown",
            operation="ui_interaction",
            encoding="unknown",
            content_length=0,
            timestamp=datetime.now(),
            original_exception=original_exception,
        )
        super().__init__(
            f"User interface error: {message}",
            ReadBeforeWriteErrorType.USER_INTERFACE_ERROR,
            context,
            original_exception,
        )


class CallbackError(ReadBeforeWriteError):
    """Error in user review callback execution."""

    def __init__(self, file_path: str, original_exception: Exception):
        context = ErrorContext(
            error_type=ReadBeforeWriteErrorType.CALLBACK_ERROR,
            file_path=file_path,
            operation="callback",
            encoding="unknown",
            content_length=0,
            timestamp=datetime.now(),
            original_exception=original_exception,
        )
        super().__init__(
            f"Error in user review callback for {file_path}: {str(original_exception)}",
            ReadBeforeWriteErrorType.CALLBACK_ERROR,
            context,
            original_exception,
        )


class ContentValidationError(ReadBeforeWriteError):
    """Error validating file content."""

    def __init__(self, file_path: str, validation_issue: str):
        context = ErrorContext(
            error_type=ReadBeforeWriteErrorType.CONTENT_VALIDATION_ERROR,
            file_path=file_path,
            operation="validation",
            encoding="unknown",
            content_length=0,
            timestamp=datetime.now(),
            additional_context={"validation_issue": validation_issue},
        )
        super().__init__(
            f"Content validation failed for {file_path}: {validation_issue}",
            ReadBeforeWriteErrorType.CONTENT_VALIDATION_ERROR,
            context,
        )


class DiffGenerationError(ReadBeforeWriteError):
    """Error generating content diff."""

    def __init__(self, file_path: str, original_exception: Exception):
        context = ErrorContext(
            error_type=ReadBeforeWriteErrorType.DIFF_GENERATION_ERROR,
            file_path=file_path,
            operation="diff_generation",
            encoding="unknown",
            content_length=0,
            timestamp=datetime.now(),
            original_exception=original_exception,
        )
        super().__init__(
            f"Failed to generate diff for {file_path}: {str(original_exception)}",
            ReadBeforeWriteErrorType.DIFF_GENERATION_ERROR,
            context,
            original_exception,
        )


class UserRejectionError(ReadBeforeWriteError):
    """User rejected the file modification."""

    def __init__(self, file_path: str, reason: str):
        context = ErrorContext(
            error_type=ReadBeforeWriteErrorType.USER_REJECTION,
            file_path=file_path,
            operation="user_decision",
            encoding="unknown",
            content_length=0,
            timestamp=datetime.now(),
            additional_context={"rejection_reason": reason},
        )
        super().__init__(
            f"User rejected modification of {file_path}: {reason}",
            ReadBeforeWriteErrorType.USER_REJECTION,
            context,
        )


class ReadBeforeWriteErrorHandler:
    """
    Comprehensive error handler for read-before-write operations.

    Provides error recovery strategies, logging, and fallback mechanisms.
    """

    def __init__(self, enable_recovery: bool = True, log_errors: bool = True):
        """
        Initialize error handler.

        Args:
            enable_recovery: Whether to attempt error recovery
            log_errors: Whether to log errors
        """
        self.enable_recovery = enable_recovery
        self.log_errors = log_errors
        self.error_history: List[ReadBeforeWriteError] = []

        # Recovery strategy mappings
        self.recovery_strategies = {
            ReadBeforeWriteErrorType.FILE_READ_ERROR: RecoveryStrategy.FALLBACK_TO_REGULAR_WRITE,
            ReadBeforeWriteErrorType.FILE_WRITE_ERROR: RecoveryStrategy.RETRY,
            ReadBeforeWriteErrorType.USER_INTERFACE_ERROR: RecoveryStrategy.PROMPT_USER,
            ReadBeforeWriteErrorType.PERMISSION_ERROR: RecoveryStrategy.ABORT_OPERATION,
            ReadBeforeWriteErrorType.ENCODING_ERROR: RecoveryStrategy.PROMPT_USER,
            ReadBeforeWriteErrorType.CONTENT_VALIDATION_ERROR: RecoveryStrategy.PROMPT_USER,
            ReadBeforeWriteErrorType.CALLBACK_ERROR: RecoveryStrategy.FALLBACK_TO_REGULAR_WRITE,
            ReadBeforeWriteErrorType.DIFF_GENERATION_ERROR: RecoveryStrategy.LOG_AND_CONTINUE,
            ReadBeforeWriteErrorType.PREVIEW_ERROR: RecoveryStrategy.LOG_AND_CONTINUE,
            ReadBeforeWriteErrorType.USER_REJECTION: RecoveryStrategy.SKIP_OPERATION,
            ReadBeforeWriteErrorType.TIMEOUT_ERROR: RecoveryStrategy.PROMPT_USER,
        }

    def handle_error(
        self,
        error: ReadBeforeWriteError,
        retry_count: int = 0,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Handle a read-before-write error with appropriate recovery strategy.

        Args:
            error: The error to handle
            retry_count: Current retry attempt count
            max_retries: Maximum number of retries allowed

        Returns:
            Dict with error handling result and recovery information
        """
        # Record error in history
        self.error_history.append(error)

        # Log error if enabled
        if self.log_errors:
            self._log_error(error)

        # Get recovery strategy
        strategy = self.recovery_strategies.get(
            error.error_type, RecoveryStrategy.ABORT_OPERATION
        )

        # Apply recovery strategy
        recovery_result = self._apply_recovery_strategy(
            error, strategy, retry_count, max_retries
        )

        # Update error context with recovery information
        if error.context:
            error.context.recovery_attempted = True
            error.context.recovery_strategy = strategy
            error.context.recovery_success = recovery_result.get("success", False)

        return {
            "error": error.to_dict(),
            "recovery_strategy": strategy.value,
            "recovery_result": recovery_result,
            "should_retry": recovery_result.get("should_retry", False),
            "should_abort": recovery_result.get("should_abort", False),
            "fallback_action": recovery_result.get("fallback_action"),
        }

    def _apply_recovery_strategy(
        self,
        error: ReadBeforeWriteError,
        strategy: RecoveryStrategy,
        retry_count: int,
        max_retries: int,
    ) -> Dict[str, Any]:
        """Apply the specified recovery strategy."""
        if not self.enable_recovery:
            return {
                "success": False,
                "message": "Error recovery disabled",
                "should_abort": True,
            }

        if strategy == RecoveryStrategy.RETRY:
            if retry_count < max_retries:
                return {
                    "success": True,
                    "message": f"Retrying operation (attempt {retry_count + 1}/{max_retries})",
                    "should_retry": True,
                }
            else:
                return {
                    "success": False,
                    "message": f"Maximum retries ({max_retries}) exceeded",
                    "should_abort": True,
                }

        elif strategy == RecoveryStrategy.FALLBACK_TO_REGULAR_WRITE:
            return {
                "success": True,
                "message": "Falling back to regular write operation without review",
                "fallback_action": "regular_write",
            }

        elif strategy == RecoveryStrategy.SKIP_OPERATION:
            return {
                "success": True,
                "message": "Skipping operation due to user rejection or other reason",
                "fallback_action": "skip",
            }

        elif strategy == RecoveryStrategy.LOG_AND_CONTINUE:
            return {
                "success": True,
                "message": "Error logged, continuing with operation",
                "fallback_action": "continue_with_warning",
            }

        elif strategy == RecoveryStrategy.PROMPT_USER:
            return {
                "success": False,
                "message": "User intervention required",
                "fallback_action": "prompt_user",
            }

        elif strategy == RecoveryStrategy.ABORT_OPERATION:
            return {
                "success": False,
                "message": "Operation aborted due to unrecoverable error",
                "should_abort": True,
            }

        return {
            "success": False,
            "message": f"Unknown recovery strategy: {strategy}",
            "should_abort": True,
        }

    def _log_error(self, error: ReadBeforeWriteError):
        """Log error with appropriate severity level."""
        log_message = f"Read-before-write error: {error.message}"

        if error.context:
            log_message += f" [File: {error.context.file_path}, Operation: {error.context.operation}]"

        # Determine log level based on error type
        if error.error_type in [ReadBeforeWriteErrorType.USER_REJECTION]:
            logger.info(log_message)
        elif error.error_type in [
            ReadBeforeWriteErrorType.DIFF_GENERATION_ERROR,
            ReadBeforeWriteErrorType.PREVIEW_ERROR,
        ]:
            logger.warning(log_message)
        else:
            logger.error(log_message, exc_info=error.original_exception)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about handled errors."""
        if not self.error_history:
            return {"total_errors": 0}

        error_counts = {}
        recovery_counts = {}

        for error in self.error_history:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

            if error.context and error.context.recovery_strategy:
                recovery_strategy = error.context.recovery_strategy.value
                recovery_counts[recovery_strategy] = (
                    recovery_counts.get(recovery_strategy, 0) + 1
                )

        successful_recoveries = sum(
            1
            for error in self.error_history
            if error.context and error.context.recovery_success
        )

        return {
            "total_errors": len(self.error_history),
            "error_types": error_counts,
            "recovery_strategies_used": recovery_counts,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": (
                successful_recoveries / len(self.error_history)
                if self.error_history
                else 0
            ),
        }

    def clear_error_history(self):
        """Clear the error history."""
        self.error_history.clear()

    def set_recovery_strategy(
        self, error_type: ReadBeforeWriteErrorType, strategy: RecoveryStrategy
    ):
        """Set custom recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy
        logger.info(f"Set recovery strategy for {error_type.value}: {strategy.value}")


# Convenience function for creating error handler
def create_error_handler(
    enable_recovery: bool = True, log_errors: bool = True
) -> ReadBeforeWriteErrorHandler:
    """
    Create a configured error handler for read-before-write operations.

    Args:
        enable_recovery: Whether to enable error recovery
        log_errors: Whether to log errors

    Returns:
        Configured ReadBeforeWriteErrorHandler instance
    """
    return ReadBeforeWriteErrorHandler(enable_recovery, log_errors)
