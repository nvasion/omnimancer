"""
Tests for ExecutionError and TimeoutError classes.

These tests verify the missing error classes that are required by
the agent system, particularly program_executor.py.
"""

import pytest

from omnimancer.utils.errors import (
    AgentError,
    ExecutionError,
    OmnimancerError,
    TimeoutError,
)


class TestExecutionError:
    """Test ExecutionError class functionality."""

    def test_execution_error_inheritance(self):
        """Test that ExecutionError inherits from AgentError."""
        error = ExecutionError("Test execution error")
        assert isinstance(error, AgentError)
        assert isinstance(error, OmnimancerError)
        assert isinstance(error, Exception)

    def test_execution_error_basic_message(self):
        """Test ExecutionError with basic message."""
        message = "Command execution failed"
        error = ExecutionError(message)

        assert error.message == message
        assert str(error) == message

    def test_execution_error_with_details(self):
        """Test ExecutionError with details."""
        message = "Command execution failed"
        details = "Exit code: 1, stderr: Permission denied"
        error = ExecutionError(message, details)

        assert error.message == message
        assert error.details == details
        assert str(error) == f"{message}\nDetails: {details}"

    def test_execution_error_with_command_context(self):
        """Test ExecutionError with command context."""
        message = "Command execution failed"
        details = "Command: ls -la /forbidden"
        command = "ls -la /forbidden"
        exit_code = 1

        error = ExecutionError(message, details)

        # Test that we can add custom attributes
        error.command = command
        error.exit_code = exit_code

        assert error.command == command
        assert error.exit_code == exit_code

    def test_execution_error_empty_details(self):
        """Test ExecutionError with empty details."""
        message = "Command failed"
        error = ExecutionError(message, "")

        # Empty details should not affect string representation
        assert str(error) == message

    def test_execution_error_none_details(self):
        """Test ExecutionError with None details."""
        message = "Command failed"
        error = ExecutionError(message, None)

        assert error.details is None
        assert str(error) == message


class TestTimeoutError:
    """Test TimeoutError class functionality."""

    def test_timeout_error_inheritance(self):
        """Test that TimeoutError inherits from AgentError."""
        error = TimeoutError("Test timeout error")
        assert isinstance(error, AgentError)
        assert isinstance(error, OmnimancerError)
        assert isinstance(error, Exception)

    def test_timeout_error_basic_message(self):
        """Test TimeoutError with basic message."""
        message = "Operation timed out"
        error = TimeoutError(message)

        assert error.message == message
        assert str(error) == message

    def test_timeout_error_with_details(self):
        """Test TimeoutError with details."""
        message = "Command timed out"
        details = "Timeout after 30 seconds"
        error = TimeoutError(message, details)

        assert error.message == message
        assert error.details == details
        assert str(error) == f"{message}\nDetails: {details}"

    def test_timeout_error_with_duration_context(self):
        """Test TimeoutError with duration context."""
        message = "Process execution timed out"
        duration = 30.5
        operation = "file_processing"

        error = TimeoutError(message)
        error.duration = duration
        error.operation = operation

        assert error.duration == duration
        assert error.operation == operation

    def test_timeout_error_distinction_from_builtin(self):
        """Test that our TimeoutError is distinct from built-in TimeoutError."""
        # Import built-in TimeoutError for comparison
        import builtins

        builtin_timeout = getattr(builtins, "TimeoutError", None)

        our_error = TimeoutError("Test timeout")

        # Our error should be an Omnimancer error
        assert isinstance(our_error, OmnimancerError)

        # If built-in TimeoutError exists (Python 3.3+), ensure they're different
        if builtin_timeout:
            assert not isinstance(our_error, builtin_timeout)
            assert TimeoutError != builtin_timeout

    def test_timeout_error_empty_details(self):
        """Test TimeoutError with empty details."""
        message = "Timeout occurred"
        error = TimeoutError(message, "")

        assert str(error) == message

    def test_timeout_error_none_details(self):
        """Test TimeoutError with None details."""
        message = "Timeout occurred"
        error = TimeoutError(message, None)

        assert error.details is None
        assert str(error) == message


class TestErrorClassInteraction:
    """Test interaction between ExecutionError and TimeoutError."""

    def test_errors_are_distinct_classes(self):
        """Test that ExecutionError and TimeoutError are distinct."""
        execution_error = ExecutionError("Execution failed")
        timeout_error = TimeoutError("Timeout occurred")

        assert type(execution_error) != type(timeout_error)
        assert not isinstance(execution_error, TimeoutError)
        assert not isinstance(timeout_error, ExecutionError)

    def test_both_inherit_from_agent_error(self):
        """Test that both errors inherit from AgentError."""
        execution_error = ExecutionError("Execution failed")
        timeout_error = TimeoutError("Timeout occurred")

        assert isinstance(execution_error, AgentError)
        assert isinstance(timeout_error, AgentError)

    def test_error_catching_patterns(self):
        """Test common error catching patterns."""

        def raise_execution_error():
            raise ExecutionError("Command failed")

        def raise_timeout_error():
            raise TimeoutError("Command timed out")

        # Test catching specific errors
        with pytest.raises(ExecutionError):
            raise_execution_error()

        with pytest.raises(TimeoutError):
            raise_timeout_error()

        # Test catching as AgentError
        with pytest.raises(AgentError):
            raise_execution_error()

        with pytest.raises(AgentError):
            raise_timeout_error()

        # Test catching as OmnimancerError
        with pytest.raises(OmnimancerError):
            raise_execution_error()

        with pytest.raises(OmnimancerError):
            raise_timeout_error()


class TestErrorsExportability:
    """Test that error classes can be imported properly."""

    def test_errors_importable_from_utils_errors(self):
        """Test that errors can be imported from utils.errors module."""
        # This test verifies the import works (already done at top)
        # and that the classes are available
        assert ExecutionError is not None
        assert TimeoutError is not None

        # Test that they can be instantiated
        exec_error = ExecutionError("Test")
        timeout_error = TimeoutError("Test")

        assert exec_error is not None
        assert timeout_error is not None

    def test_errors_in_module_all_export(self):
        """Test that errors are in __all__ if it exists."""
        import omnimancer.utils.errors as errors_module

        # If __all__ exists, our errors should be in it
        if hasattr(errors_module, "__all__"):
            assert "ExecutionError" in errors_module.__all__
            assert "TimeoutError" in errors_module.__all__
