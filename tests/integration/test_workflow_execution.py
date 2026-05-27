"""
Integration tests for end-to-end workflow execution - SIMPLIFIED VERSION.
"""

import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from omnimancer.cli.commands import Command
from omnimancer.cli.interface import CommandLineInterface
from omnimancer.core.engine import CoreEngine


class TestWorkflowExecutionIntegration:
    """Integration tests for complete workflow execution scenarios."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_files = {
                "README.md": "# Test Project\nThis is a test project.",
                "main.py": 'print("Hello, World!")',
                "config.json": '{"name": "test", "version": "1.0.0"}',
            }

            for filename, content in test_files.items():
                with open(os.path.join(temp_dir, filename), "w") as f:
                    f.write(content)

            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            yield temp_dir

            # Restore original directory
            os.chdir(original_cwd)

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine."""
        engine = Mock(spec=CoreEngine)
        return engine

    @pytest.fixture
    def cli_with_mocked_engine(self, mock_engine):
        """Create CLI interface with mocked engine - AGENT MODE OFF."""
        cli = CommandLineInterface(mock_engine)

        # Mock progress indicator
        cli.progress_indicator = Mock()
        cli.progress_indicator.disable = Mock()
        cli.progress_indicator.enable = Mock()
        cli.progress_indicator.clear_all_operations = Mock()

        # Mock console
        cli.console = Mock()
        cli.console.print = Mock()

        # Mock agent manager with mode OFF to avoid workflow loop complexity
        cli.agent_manager = Mock()
        cli.agent_manager.mode.value = "off"

        return cli

    @pytest.mark.asyncio
    async def test_simple_chat_message(
        self, temp_workspace, cli_with_mocked_engine
    ):
        """Test a simple chat message without agent mode."""
        cli = cli_with_mocked_engine

        async def mock_send_message(message):
            response = Mock()
            response.is_success = True
            response.model_used = "test-model"
            response.content = "I've analyzed the workspace. It contains a Python project with README, main.py, and config.json files."
            return response

        cli.engine.send_message = AsyncMock(side_effect=mock_send_message)

        command = Command.create_chat_message("analyze this workspace")
        await cli._handle_chat_message(command)

        # Verify engine was called
        assert cli.engine.send_message.call_count == 1
        # Verify console printed the response
        assert cli.console.print.called

    @pytest.mark.asyncio
    async def test_chat_with_parse_and_execute_operations(
        self, temp_workspace, cli_with_mocked_engine
    ):
        """Test that _parse_and_execute_operations is called when agent mode is on."""
        cli = cli_with_mocked_engine

        # Enable agent mode for this specific test
        cli.agent_manager.mode.value = "on"
        cli.engine.provider_supports_tools = Mock(return_value=False)

        # Mock _parse_and_execute_operations to track calls
        original_method = cli._parse_and_execute_operations
        parse_calls = []

        async def mock_parse(content):
            parse_calls.append(content)
            # Just return content unchanged - no actual operations
            return content

        cli._parse_and_execute_operations = mock_parse

        # Mock engine response with NO operations (so workflow exits immediately)
        async def mock_send_message(message):
            response = Mock()
            response.is_success = True
            response.model_used = "test-model"
            # Response with no operation markers - workflow should exit on first check
            response.content = "The workspace has been analyzed successfully. Task is complete."
            return response

        cli.engine.send_message = AsyncMock(side_effect=mock_send_message)

        command = Command.create_chat_message("analyze workspace")
        await cli._handle_chat_message(command)

        # Verify _parse_and_execute_operations was called
        assert len(parse_calls) > 0, "Operation parsing was not called"

    @pytest.mark.asyncio
    async def test_error_handling(
        self, temp_workspace, cli_with_mocked_engine
    ):
        """Test workflow behavior when engine returns error."""
        cli = cli_with_mocked_engine

        async def mock_send_message(message):
            response = Mock()
            response.is_success = False
            response.error = "API rate limit exceeded"
            return response

        cli.engine.send_message = AsyncMock(side_effect=mock_send_message)

        command = Command.create_chat_message("test error handling")
        await cli._handle_chat_message(command)

        # Verify engine was called
        assert cli.engine.send_message.call_count == 1
        # Error handling should have been triggered (console should show error)
        assert cli.console.print.called
