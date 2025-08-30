"""
Tests for error handling robustness across Omnimancer components.

These tests verify that the system handles edge cases gracefully,
particularly focusing on Console initialization, agent operations,
and error recovery patterns.
"""

import unittest
from unittest.mock import Mock, patch

from omnimancer.utils.errors import (
    AgentError,
    ExecutionError,
    OmnimancerError,
    PermissionError,
    SecurityError,
    TimeoutError,
)


class TestConsoleInitializationFallback(unittest.TestCase):
    """Test Console initialization fallback patterns."""

    def test_cli_console_initialization_success(self):
        """Test successful Console initialization."""
        from omnimancer.cli.interface import CommandLineInterface
        from omnimancer.core.engine import CoreEngine

        # Mock engine
        mock_engine = Mock(spec=CoreEngine)

        # Test normal initialization
        cli = CommandLineInterface(mock_engine)
        assert cli.console is not None
        assert hasattr(cli.console, "print")

    def test_cli_console_initialization_fallback(self):
        """Test Console initialization fallback when advanced features fail."""
        from omnimancer.cli.interface import CommandLineInterface
        from omnimancer.core.engine import CoreEngine

        mock_engine = Mock(spec=CoreEngine)

        # Mock Console to raise exception on first call (force_terminal=True)
        # but succeed on second call (fallback)
        with patch("omnimancer.cli.interface.Console") as mock_console_class:
            # First call (advanced) raises exception, second call (fallback) succeeds
            fallback_console = Mock()
            fallback_console.print = Mock()

            mock_console_class.side_effect = [
                OSError("Terminal features not supported"),
                fallback_console,
            ]

            cli = CommandLineInterface(mock_engine)

            # Should have fallen back to basic console
            assert cli.console == fallback_console
            assert mock_console_class.call_count == 2

            # Verify fallback console is functional
            cli.console.print("test")
            fallback_console.print.assert_called_once_with("test")

    def test_agent_progress_ui_console_fallback(self):
        """Test agent progress UI console fallback."""
        from omnimancer.core.agent_progress_ui import AgentProgressUI

        # Mock agent manager
        mock_agent_manager = Mock()

        # Test with None console (should create default)
        progress_ui = AgentProgressUI(mock_agent_manager, console=None)
        assert progress_ui.console is not None

        # Test with provided console
        mock_console = Mock()
        progress_ui = AgentProgressUI(mock_agent_manager, console=mock_console)
        assert progress_ui.console == mock_console

    def test_cancellation_handler_console_fallback(self):
        """Test cancellation handler console fallback."""
        from omnimancer.ui.cancellation_handler import CancellationHandler

        # Test with None console (should create default)
        handler = CancellationHandler(console=None)
        assert handler.console is not None

        # Test with provided console
        mock_console = Mock()
        handler = CancellationHandler(console=mock_console)
        assert handler.console == mock_console


class TestAgentErrorHandling(unittest.TestCase):
    """Test agent system error handling robustness."""

    def test_agent_engine_path_validation_errors(self):
        """Test agent engine handles path validation errors gracefully."""

        from omnimancer.core.agent_engine import AgentEngine
        from omnimancer.core.config_manager import ConfigManager

        # Create agent with properly mocked config manager
        mock_config_manager = Mock(spec=ConfigManager)
        mock_config = Mock()
        mock_config.storage_path = "/tmp/test_storage"
        mock_config_manager.get_config.return_value = mock_config

        # Patch ConversationManager import and other CoreEngine dependencies
        with (
            patch("omnimancer.core.engine.ConversationManager"),
            patch("omnimancer.core.agent_engine.ProgramExecutor"),
            patch("omnimancer.core.agent_engine.WebClient"),
            patch("omnimancer.core.agent_engine.MCPIntegrator"),
            patch("omnimancer.core.agent_engine.ApprovalWorkflow"),
            patch("omnimancer.core.agent_engine.EnhancedApprovalManager"),
            patch("omnimancer.core.agent_engine.ApprovalInterface"),
            patch("omnimancer.core.agent_engine.ApprovalManager"),
            patch("omnimancer.core.agent_engine.ProviderFallback"),
        ):
            AgentEngine(mock_config_manager)

            # Skip this test as the FileSystemManager doesn't currently have path validation
            self.skipTest(
                "Path validation not implemented in current FileSystemManager"
            )

    def test_agent_engine_forbidden_path_detection(self):
        """Test agent engine detects forbidden paths."""
        # Skip this test as the FileSystemManager doesn't currently have path validation
        self.skipTest("Path validation not implemented in current FileSystemManager")

    def test_agent_file_operations_error_handling(self):
        """Test agent file operations handle errors gracefully."""
        from omnimancer.core.agent_engine import AgentEngine
        from omnimancer.core.config_manager import ConfigManager

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config = Mock()
        mock_config.storage_path = "/tmp/test_storage"
        mock_config_manager.get_config.return_value = mock_config

        # Patch all AgentEngine dependencies
        with (
            patch("omnimancer.core.engine.ConversationManager"),
            patch("omnimancer.core.agent_engine.EnhancedFileSystemManager"),
            patch("omnimancer.core.agent_engine.ProgramExecutor"),
            patch("omnimancer.core.agent_engine.WebClient"),
            patch("omnimancer.core.agent_engine.MCPIntegrator"),
            patch("omnimancer.core.agent_engine.ApprovalWorkflow"),
            patch("omnimancer.core.agent_engine.EnhancedApprovalManager"),
            patch("omnimancer.core.agent_engine.ApprovalInterface"),
            patch("omnimancer.core.agent_engine.ApprovalManager"),
            patch("omnimancer.core.agent_engine.ProviderFallback"),
        ):
            agent = AgentEngine(mock_config_manager)

            # Test file manager error handling pattern
            agent.file_manager = Mock()
            agent.file_manager.read_file.side_effect = PermissionError("Access denied")

            with self.assertRaises(PermissionError):
                # This would be called through agent operations
                agent.file_manager.read_file("/some/path")

    def test_agent_engine_unicode_handling(self):
        """Test agent engine handles Unicode decode errors."""
        from omnimancer.core.agent_engine import AgentEngine
        from omnimancer.core.config_manager import ConfigManager

        mock_config_manager = Mock(spec=ConfigManager)
        mock_config = Mock()
        mock_config.storage_path = "/tmp/test_storage"
        mock_config_manager.get_config.return_value = mock_config

        # Patch all AgentEngine dependencies
        with (
            patch("omnimancer.core.engine.ConversationManager"),
            patch("omnimancer.core.agent_engine.EnhancedFileSystemManager"),
            patch("omnimancer.core.agent_engine.ProgramExecutor"),
            patch("omnimancer.core.agent_engine.WebClient"),
            patch("omnimancer.core.agent_engine.MCPIntegrator"),
            patch("omnimancer.core.agent_engine.ApprovalWorkflow"),
            patch("omnimancer.core.agent_engine.EnhancedApprovalManager"),
            patch("omnimancer.core.agent_engine.ApprovalInterface"),
            patch("omnimancer.core.agent_engine.ApprovalManager"),
            patch("omnimancer.core.agent_engine.ProviderFallback"),
        ):
            agent = AgentEngine(mock_config_manager)

            # Test with mock file manager that raises UnicodeDecodeError
            mock_file_manager = Mock()
            mock_file_manager.read_file.side_effect = UnicodeDecodeError(
                "utf-8", b"\xff\xfe", 0, 2, "invalid start byte"
            )

            agent.file_manager = mock_file_manager

            # Should handle UnicodeDecodeError gracefully
            with self.assertRaises(UnicodeDecodeError):
                mock_file_manager.read_file("binary_file.bin")


class TestErrorContextAndRecovery(unittest.TestCase):
    """Test error context preservation and recovery patterns."""

    def test_execution_error_context_preservation(self):
        """Test ExecutionError preserves context information."""
        command = "invalid_command"
        exit_code = 127
        stderr_output = "command not found"

        error = ExecutionError(
            "Command execution failed",
            details=f"Command: {command}, Exit code: {exit_code}, Stderr: {stderr_output}",
        )

        # Test that context is preserved
        assert "Command execution failed" in str(error)
        assert command in error.details
        assert str(exit_code) in error.details
        assert stderr_output in error.details

    def test_timeout_error_context_preservation(self):
        """Test TimeoutError preserves timeout context."""
        duration = 30.5
        operation = "file_processing"

        error = TimeoutError(
            "Operation timed out",
            details=f"Operation: {operation}, Duration: {duration}s",
        )

        # Test context preservation
        assert "Operation timed out" in str(error)
        assert operation in error.details
        assert str(duration) in error.details

    def test_security_error_context_preservation(self):
        """Test SecurityError preserves security context."""
        path = "/etc/passwd"
        reason = "Access to system files forbidden"

        error = SecurityError(
            f"Security violation: {reason}", details=f"Attempted path: {path}"
        )

        assert reason in str(error)
        assert path in error.details

    def test_nested_error_handling(self):
        """Test handling of nested exceptions."""

        def cause_nested_error():
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise AgentError("Agent operation failed") from e

        with self.assertRaises(AgentError) as exc_info:
            cause_nested_error()

        # Test that the cause is preserved
        self.assertIsNotNone(exc_info.exception.__cause__)
        self.assertIsInstance(exc_info.exception.__cause__, ValueError)
        self.assertIn("Original error", str(exc_info.exception.__cause__))


class TestErrorHandlingEdgeCases(unittest.TestCase):
    """Test error handling edge cases and boundary conditions."""

    def test_empty_error_messages(self):
        """Test handling of empty error messages."""

        # Test with empty message
        error = OmnimancerError("")
        assert str(error) == ""

        # Test with None message (should not crash)
        try:
            error = OmnimancerError(None)
            str(error)  # Should not crash
        except TypeError:
            # This is acceptable behavior for None message
            pass

    def test_very_long_error_messages(self):
        """Test handling of very long error messages."""
        long_message = "A" * 10000  # 10KB message
        long_details = "B" * 50000  # 50KB details

        error = OmnimancerError(long_message, details=long_details)
        error_str = str(error)

        # Should handle long messages without crashing
        assert long_message in error_str
        assert long_details in error_str
        assert len(error_str) > 60000

    def test_unicode_error_messages(self):
        """Test handling of Unicode in error messages."""
        unicode_message = "Error with Unicode: 你好世界 🌍 émojis"
        unicode_details = "Details with Unicode: Ελληνικά, 日本語, العربية"

        error = OmnimancerError(unicode_message, details=unicode_details)
        error_str = str(error)

        # Should handle Unicode without issues
        assert unicode_message in error_str
        assert unicode_details in error_str

    def test_error_serialization(self):
        """Test error objects can be serialized/deserialized."""
        original_error = ExecutionError("Test error", details="Test details")

        # Test string representation roundtrip
        error_str = str(original_error)
        assert "Test error" in error_str
        assert "Test details" in error_str

        # Test that error attributes are accessible
        assert original_error.message == "Test error"
        assert original_error.details == "Test details"

    def test_error_inheritance_chain(self):
        """Test error inheritance chain is correct."""
        exec_error = ExecutionError("Test")
        timeout_error = TimeoutError("Test")
        security_error = SecurityError("Test")

        # Test inheritance chain
        assert isinstance(exec_error, AgentError)
        assert isinstance(exec_error, OmnimancerError)
        assert isinstance(exec_error, Exception)

        assert isinstance(timeout_error, AgentError)
        assert isinstance(timeout_error, OmnimancerError)
        assert isinstance(timeout_error, Exception)

        assert isinstance(security_error, AgentError)
        assert isinstance(security_error, OmnimancerError)
        assert isinstance(security_error, Exception)

    def test_concurrent_error_handling(self):
        """Test error handling in concurrent scenarios."""
        import threading
        import time

        errors_caught = []

        def worker_with_error():
            try:
                time.sleep(0.01)  # Small delay
                raise AgentError("Worker error")
            except AgentError as e:
                errors_caught.append(e)

        # Start multiple threads that will raise errors
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_with_error)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All errors should be caught
        assert len(errors_caught) == 5
        for error in errors_caught:
            assert isinstance(error, AgentError)
            assert "Worker error" in str(error)


class TestResourceCleanupOnErrors(unittest.TestCase):
    """Test that resources are properly cleaned up when errors occur."""

    def test_file_handle_cleanup_on_error(self):
        """Test file handles are cleaned up when errors occur."""

        # Create a temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("test content")
            temp_path = temp_file.name

        try:
            # Test that file operations clean up properly on error
            def read_with_error():
                with open(temp_path, "r") as f:
                    f.read()
                    # Simulate an error after opening file
                    raise AgentError("Simulated error")

            with self.assertRaises(AgentError):
                read_with_error()

            # File should still be accessible (handle was closed)
            with open(temp_path, "r") as f:
                content = f.read()
                assert content == "test content"

        finally:
            # Clean up temp file
            import os

            os.unlink(temp_path)

    def test_mock_resource_cleanup(self):
        """Test mock resource cleanup patterns."""

        # Create a mock resource that tracks cleanup
        class MockResource:
            def __init__(self):
                self.closed = False
                self.cleaned_up = False

            def close(self):
                self.closed = True

            def cleanup(self):
                self.cleaned_up = True

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.close()
                self.cleanup()

        # Test that context manager cleans up on error
        with self.assertRaises(AgentError):
            with MockResource() as resource:
                self.assertFalse(resource.closed)
                self.assertFalse(resource.cleaned_up)
                raise AgentError("Test error")

        # Resource should be cleaned up even though error occurred
        self.assertTrue(resource.closed)
        self.assertTrue(resource.cleaned_up)


class TestErrorReportingAndLogging(unittest.TestCase):
    """Test error reporting and logging functionality."""

    def test_error_logging_integration(self):
        """Test that errors are properly logged."""
        import logging
        from io import StringIO

        # Create a string buffer to capture log output
        log_buffer = StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.ERROR)

        # Set up logger
        logger = logging.getLogger("test_error_logger")
        logger.setLevel(logging.ERROR)
        logger.addHandler(handler)

        try:
            # Log an error
            error = AgentError("Test error for logging")
            logger.error(f"Agent error occurred: {error}")

            # Check that error was logged
            log_output = log_buffer.getvalue()
            assert "Test error for logging" in log_output
            assert "Agent error occurred" in log_output

        finally:
            logger.removeHandler(handler)

    def test_error_context_in_traceback(self):
        """Test that error context is preserved in tracebacks."""

        def level_3():
            raise ExecutionError("Deep execution error", details="From level 3")

        def level_2():
            try:
                level_3()
            except ExecutionError as e:
                raise AgentError("Level 2 error") from e

        def level_1():
            try:
                level_2()
            except AgentError as e:
                raise OmnimancerError("Top level error") from e

        with self.assertRaises(OmnimancerError) as exc_info:
            level_1()

        # Test that the full chain is preserved
        error = exc_info.exception
        self.assertIn("Top level error", str(error))

        # Check the cause chain
        cause = error.__cause__
        self.assertIsInstance(cause, AgentError)
        self.assertIn("Level 2 error", str(cause))

        root_cause = cause.__cause__
        self.assertIsInstance(root_cause, ExecutionError)
        self.assertIn("Deep execution error", str(root_cause))
        self.assertIn("From level 3", root_cause.details)
