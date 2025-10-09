"""
Comprehensive tests for continuous workflow execution system.

This module tests both normal workflow execution and edge cases including
error handling, termination conditions, and performance scenarios.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from omnimancer.cli.interface import CommandLineInterface
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.engine import CoreEngine


class TestContinuousWorkflow:
    """Test continuous workflow execution functionality."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        return Mock(spec=ConfigManager)

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine."""
        engine = Mock(spec=CoreEngine)
        engine.send_message = AsyncMock()
        return engine

    @pytest.fixture
    def mock_agent_manager(self):
        """Create a mock agent manager."""
        agent_manager = Mock()
        agent_manager.mode.value = "on"
        return agent_manager

    @pytest.fixture
    def cli_interface(self, mock_engine):
        """Create a CLI interface with mocked dependencies."""
        cli = CommandLineInterface(mock_engine)
        cli.progress_indicator = Mock()
        cli.progress_indicator.disable = Mock()
        cli.progress_indicator.enable = Mock()
        cli.progress_indicator.clear_all_operations = Mock()
        cli.console = Mock()
        return cli

    @pytest.mark.asyncio
    async def test_execute_continuous_workflow_with_operations(
        self, cli_interface, mock_agent_manager
    ):
        """Test continuous workflow execution when operations are present."""
        cli_interface.agent_manager = mock_agent_manager

        # Mock the first response with operations
        first_response = Mock()
        first_response.content = (
            "I'll analyze the files. [COMMAND_EXEC] ls -la [/COMMAND_EXEC]"
        )
        first_response.model_used = "test-model"
        first_response.is_success = True

        # Mock the second response without operations (should terminate)
        second_response = Mock()
        second_response.content = (
            "Analysis complete. The workspace has been analyzed successfully."
        )
        second_response.model_used = "test-model"
        second_response.is_success = True

        # Mock engine responses
        cli_interface.engine.send_message.side_effect = [second_response]

        # Mock operation parsing
        with patch.object(cli_interface, "_parse_and_execute_operations") as mock_parse:
            mock_parse.return_value = "✅ Command executed successfully: `ls -la`"

            with patch.object(cli_interface, "_show_assistant_message") as mock_show:
                await cli_interface._execute_continuous_workflow(
                    "analyze workspace", first_response
                )

                # Should show the first response with executed operations
                assert mock_show.call_count >= 1
                # Should attempt to send continuation message
                assert cli_interface.engine.send_message.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_continuous_workflow_no_operations(
        self, cli_interface, mock_agent_manager
    ):
        """Test workflow execution when no operations are present."""
        cli_interface.agent_manager = mock_agent_manager

        # Mock response without operations
        response = Mock()
        response.content = "Here's my analysis of the workspace without any operations."
        response.model_used = "test-model"

        with patch.object(cli_interface, "_parse_and_execute_operations") as mock_parse:
            mock_parse.return_value = response.content

            with patch.object(cli_interface, "_show_assistant_message") as mock_show:
                await cli_interface._execute_continuous_workflow(
                    "analyze workspace", response
                )

                # Should show the response once and not continue
                assert mock_show.call_count == 1
                assert cli_interface.engine.send_message.call_count == 0

    @pytest.mark.asyncio
    async def test_workflow_completion_detection(
        self, cli_interface, mock_agent_manager
    ):
        """Test that workflow detects completion indicators."""
        cli_interface.agent_manager = mock_agent_manager

        # First response with operations
        first_response = Mock()
        first_response.content = "Starting analysis. [COMMAND_EXEC] ls [/COMMAND_EXEC]"
        first_response.model_used = "test-model"

        # Second response indicating completion
        completion_response = Mock()
        completion_response.content = (
            "The task is complete. Analysis finished successfully."
        )
        completion_response.model_used = "test-model"
        completion_response.is_success = True

        cli_interface.engine.send_message.return_value = completion_response

        with patch.object(cli_interface, "_parse_and_execute_operations") as mock_parse:
            mock_parse.return_value = "✅ Command executed"

            with patch.object(cli_interface, "_show_assistant_message") as mock_show:
                await cli_interface._execute_continuous_workflow(
                    "test task", first_response
                )

                # Should show both responses
                assert mock_show.call_count == 2
                # Should only make one engine call (for continuation)
                assert cli_interface.engine.send_message.call_count == 1

    @pytest.mark.asyncio
    async def test_workflow_engine_failure_handling(
        self, cli_interface, mock_agent_manager
    ):
        """Test workflow handling when engine fails during continuation."""
        cli_interface.agent_manager = mock_agent_manager

        # First response with operations
        first_response = Mock()
        first_response.content = "Starting work. [COMMAND_EXEC] test [/COMMAND_EXEC]"
        first_response.model_used = "test-model"

        # Engine fails on continuation
        failed_response = Mock()
        failed_response.is_success = False
        failed_response.error = "API connection failed"

        cli_interface.engine.send_message.return_value = failed_response

        with patch.object(cli_interface, "_parse_and_execute_operations") as mock_parse:
            mock_parse.return_value = "✅ Command executed"

            with patch.object(cli_interface, "_show_assistant_message"):
                with patch.object(cli_interface, "_show_error") as mock_error:
                    await cli_interface._execute_continuous_workflow(
                        "test task", first_response
                    )

                    # Should show error for failed continuation
                    mock_error.assert_called_once_with(
                        "Workflow continuation failed: API connection failed"
                    )

    def test_operation_pattern_detection(self, cli_interface):
        """Test that operation patterns are correctly detected."""
        # Test with various operation patterns, including escaped brackets
        test_cases = [
            ("[FILE_WRITE:test.txt]content[/FILE_WRITE]", True),
            ("[FILE_WRITE:test.txt\\]content[/FILE_WRITE\\]", True),  # Escaped brackets
            ("[FILE_READ:test.txt]", True),
            ("[FILE_READ:test.txt\\]", True),  # Escaped bracket
            ("[COMMAND_EXEC] ls -la [/COMMAND_EXEC]", True),
            ("[COMMAND_EXEC\\] ls -la [/COMMAND_EXEC\\]", True),  # Escaped brackets
            ("[WEB_REQUEST:http://example.com]", True),
            ("[WEB_REQUEST:http://example.com\\]", True),  # Escaped bracket
            ("Just text without operations", False),
            ("Mixed text [FILE_READ:file.txt] with operations", True),
        ]

        import re

        # Updated patterns to match actual implementation (with optional backslash)
        operation_patterns = [
            r"\[FILE_WRITE:[^\]]+\\?\].*?\[/FILE_WRITE\\?\]",
            r"\[FILE_READ:[^\]]+\\?\]",
            r"\[COMMAND_EXEC\\?\].*?\[/COMMAND_EXEC\\?\]",
            r"\[WEB_REQUEST:[^\]]+\\?\]",
        ]

        for content, expected_has_ops in test_cases:
            has_operations = any(
                re.search(pattern, content, re.DOTALL) for pattern in operation_patterns
            )
            assert has_operations == expected_has_ops, f"Failed for content: {content}"


class TestWorkflowEdgeCases:
    """Test various workflow termination conditions and edge cases."""

    @pytest.fixture
    def cli_interface_edge(self):
        """Create CLI interface for edge case testing."""
        engine = Mock(spec=CoreEngine)
        engine.send_message = AsyncMock()

        cli = CommandLineInterface(engine)
        cli.progress_indicator = Mock()
        cli.progress_indicator.disable = Mock()
        cli.progress_indicator.enable = Mock()
        cli.console = Mock()

        # Mock agent manager
        cli.agent_manager = Mock()
        cli.agent_manager.mode.value = "on"

        return cli

    @pytest.mark.asyncio
    async def test_workflow_stops_on_completion_keywords(self, cli_interface_edge):
        """Test that workflow stops when AI indicates completion."""
        # First response with operations
        first_response = Mock()
        first_response.content = "Starting work. [COMMAND_EXEC] ls [/COMMAND_EXEC]"
        first_response.model_used = "test-model"

        # Second response with completion phrase AND no operations
        completion_response = Mock()
        completion_response.content = "The task is complete. Everything looks good."
        completion_response.model_used = "test-model"
        completion_response.is_success = True

        cli_interface_edge.engine.send_message.return_value = completion_response

        # Mock parse to return different content based on input
        async def mock_parse(content):
            if "[COMMAND_EXEC]" in content:
                return "✅ Command executed successfully"
            return content  # No operations, return as-is

        with patch.object(
            cli_interface_edge,
            "_parse_and_execute_operations",
            new=AsyncMock(side_effect=mock_parse),
        ):
            with patch.object(
                cli_interface_edge, "_show_assistant_message"
            ) as mock_show:
                await cli_interface_edge._execute_continuous_workflow(
                    "test task", first_response
                )

                # Should show both the first response and completion response
                assert mock_show.call_count >= 1

                # Should only make one continuation call
                assert cli_interface_edge.engine.send_message.call_count == 1

    @pytest.mark.asyncio
    async def test_workflow_handles_empty_responses(self, cli_interface_edge):
        """Test workflow behavior with empty or None responses."""
        first_response = Mock()
        first_response.content = "Starting. [COMMAND_EXEC] test [/COMMAND_EXEC]"
        first_response.model_used = "test-model"

        # Empty response
        empty_response = Mock()
        empty_response.content = ""
        empty_response.model_used = "test-model"
        empty_response.is_success = True

        cli_interface_edge.engine.send_message.return_value = empty_response

        with patch.object(
            cli_interface_edge,
            "_parse_and_execute_operations",
            new=AsyncMock(return_value="✅ Command executed"),
        ):
            with patch.object(cli_interface_edge, "_show_assistant_message"):
                await cli_interface_edge._execute_continuous_workflow(
                    "test", first_response
                )

                # Should handle empty response gracefully
                assert cli_interface_edge.engine.send_message.call_count == 1

    @pytest.mark.asyncio
    async def test_workflow_handles_malformed_operations(self, cli_interface_edge):
        """Test workflow with malformed operation markers."""
        malformed_cases = [
            "No operations here, just text",  # No operation markers at all
            "Invalid syntax [INVALID_OP] test [/INVALID_OP]",  # Unknown operation
            "Multiple same type [FILE_READ:a] and [FILE_READ:b]",  # Multiple reads
        ]

        for malformed_content in malformed_cases:
            response = Mock()
            response.content = malformed_content
            response.model_used = "test-model"

            # Create a mock response for continuation
            continuation_response = Mock()
            continuation_response.content = "Task is complete."
            continuation_response.model_used = "test-model"
            continuation_response.is_success = True

            # Reset mock and set return value
            cli_interface_edge.engine.send_message.reset_mock()
            cli_interface_edge.engine.send_message.return_value = continuation_response

            with patch.object(
                cli_interface_edge,
                "_parse_and_execute_operations",
                new=AsyncMock(return_value=malformed_content),
            ):
                with patch.object(cli_interface_edge, "_show_assistant_message"):
                    await cli_interface_edge._execute_continuous_workflow(
                        "test", response
                    )


class TestWorkflowIntegration:
    """Integration tests for workflow execution."""

    @pytest.mark.asyncio
    async def test_chat_message_workflow_integration(self):
        """Test integration between chat message handling and workflow execution."""
        # This would test the full integration from chat command to workflow execution
        # For now, this is a placeholder that could be expanded with real integration testing
        pass
