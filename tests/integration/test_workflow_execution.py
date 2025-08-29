"""
Integration tests for end-to-end workflow execution.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from omnimancer.cli.interface import CommandLineInterface
from omnimancer.core.engine import CoreEngine
from omnimancer.core.config_manager import ConfigManager
from omnimancer.cli.commands import Command


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
    def mock_engine_with_responses(self):
        """Create a mock engine that returns realistic responses."""
        engine = Mock(spec=CoreEngine)

        async def mock_send_message(message):
            response = Mock()
            response.is_success = True
            response.model_used = "test-model"

            # For any follow-up message, return the file creation response
            if (
                "I executed the operations" in message
                or "What should I do next" in message
            ):
                response.content = """Based on the analysis results, I can see this is a Python project. 

Let me create a summary of the workspace structure.

[FILE_WRITE:WORKSPACE_SUMMARY.md]# Workspace Analysis Summary

## Project Structure
- README.md: Main project documentation
- main.py: Main Python application file  
- config.json: Configuration file

## Analysis Complete
The workspace has been analyzed and documented successfully.[/FILE_WRITE]

The workspace analysis is now complete. I've created a summary file with the key findings. Task completed."""

            elif (
                "analyze" in message.lower() and "workspace" in message.lower()
            ):
                response.content = """I'll analyze the workspace structure.
                
[COMMAND_EXEC] ls -la [/COMMAND_EXEC]

Now let me check what type of project this is by looking at key files.

[FILE_READ:README.md]

Based on my analysis, I can see this is a Python project. Let me create a summary file.

[FILE_WRITE:WORKSPACE_SUMMARY.md]# Workspace Analysis Summary

## Project Structure
- README.md: Main project documentation
- main.py: Main Python application file  
- config.json: Configuration file

## Analysis Complete
The workspace has been analyzed and documented successfully.[/FILE_WRITE]

Task completed."""

            else:
                response.content = "Task completed successfully."

            return response

        engine.send_message = AsyncMock(side_effect=mock_send_message)
        return engine

    @pytest.fixture
    def cli_with_mocked_engine(self, mock_engine_with_responses):
        """Create CLI interface with mocked engine."""
        cli = CommandLineInterface(mock_engine_with_responses)

        # Mock progress indicator
        cli.progress_indicator = Mock()
        cli.progress_indicator.disable = Mock()
        cli.progress_indicator.enable = Mock()
        cli.progress_indicator.clear_all_operations = Mock()

        # Mock console
        cli.console = Mock()
        cli.console.print = Mock()

        # Mock agent manager
        cli.agent_manager = Mock()
        cli.agent_manager.mode.value = "on"

        return cli

    @pytest.mark.asyncio
    async def test_complete_workspace_analysis_workflow(
        self, temp_workspace, cli_with_mocked_engine
    ):
        """Test a complete workflow that analyzes workspace and creates summary."""
        cli = cli_with_mocked_engine

        # Mock file operations
        executed_commands = []
        created_files = {}

        async def mock_parse_and_execute_operations(content):
            """Mock operation execution that tracks what operations were performed."""
            import re

            # Track file reads
            file_read_pattern = r"\[FILE_READ:([^\]]+)\]"
            for match in re.finditer(file_read_pattern, content):
                filename = match.group(1).strip()
                if os.path.exists(filename):
                    with open(filename, "r") as f:
                        file_content = f.read()
                    content = content.replace(
                        match.group(0),
                        f"✅ File read successfully: {filename}\nContent: {file_content[:100]}...",
                    )
                else:
                    content = content.replace(
                        match.group(0), f"❌ File not found: {filename}"
                    )

            # Track command executions
            command_pattern = r"\[COMMAND_EXEC\]\s*(.*?)\s*\[/COMMAND_EXEC\]"
            for match in re.finditer(command_pattern, content, re.DOTALL):
                command = match.group(1).strip()
                executed_commands.append(command)

                # Simulate command execution
                if command.startswith("ls"):
                    result = "README.md\nmain.py\nconfig.json"
                else:
                    result = f"Executed: {command}"

                content = content.replace(
                    match.group(0),
                    f"✅ Command executed successfully: `{command}`\nOutput: {result}",
                )

            # Track file writes
            file_write_pattern = r"\[FILE_WRITE:([^\]]+)\](.*?)\[/FILE_WRITE\]"
            for match in re.finditer(file_write_pattern, content, re.DOTALL):
                filename = match.group(1).strip()
                file_content = match.group(2).strip()
                created_files[filename] = file_content

                # Actually create the file
                with open(filename, "w") as f:
                    f.write(file_content)

                content = content.replace(
                    match.group(0),
                    f"✅ Successfully created file '{filename}' ({len(file_content)} characters)",
                )

            return content

        # Patch the operation parsing
        with patch.object(
            cli,
            "_parse_and_execute_operations",
            side_effect=mock_parse_and_execute_operations,
        ):
            # Create a command to analyze workspace
            command = Command.create_chat_message(
                "analyze this workspace and create a summary file"
            )

            # Execute the workflow
            await cli._handle_chat_message(command)

            # Verify operations were executed
            assert len(executed_commands) > 0, "No commands were executed"
            assert any(
                "ls" in cmd for cmd in executed_commands
            ), "ls command was not executed"

            # Verify files were created
            assert len(created_files) > 0, "No files were created"
            assert (
                "WORKSPACE_SUMMARY.md" in created_files
            ), "Summary file was not created"

            # Verify the summary file exists and has content
            assert os.path.exists(
                "WORKSPACE_SUMMARY.md"
            ), "Summary file was not actually created"

            with open("WORKSPACE_SUMMARY.md", "r") as f:
                summary_content = f.read()

            assert len(summary_content) > 0, "Summary file is empty"
            assert (
                "Workspace Analysis Summary" in summary_content
            ), "Summary file doesn't contain expected header"

            # Verify engine was called multiple times (for continuation)
            assert (
                cli.engine.send_message.call_count >= 2
            ), "Workflow didn't continue as expected"

    @pytest.mark.asyncio
    async def test_workflow_with_command_execution(
        self, temp_workspace, cli_with_mocked_engine
    ):
        """Test workflow that executes various commands."""
        cli = cli_with_mocked_engine

        # Mock engine to return command execution workflows
        async def command_focused_responses(message):
            response = Mock()
            response.is_success = True
            response.model_used = "test-model"

            if "check git status" in message.lower():
                response.content = """I'll check the git status and directory listing.
                
[COMMAND_EXEC] git status [/COMMAND_EXEC]
[COMMAND_EXEC] ls -la [/COMMAND_EXEC]

Let me also check if there are any Python files and their content.

[COMMAND_EXEC] find . -name "*.py" -type f [/COMMAND_EXEC]"""

            elif "executed" in message.lower():
                response.content = "All commands have been executed successfully. The workspace status check is complete."

            else:
                response.content = "Task completed."

            return response

        cli.engine.send_message = AsyncMock(
            side_effect=command_focused_responses
        )

        executed_commands = []

        async def mock_command_execution(content):
            import re

            command_pattern = r"\[COMMAND_EXEC\]\s*(.*?)\s*\[/COMMAND_EXEC\]"
            for match in re.finditer(command_pattern, content, re.DOTALL):
                command = match.group(1).strip()
                executed_commands.append(command)

                # Mock command results
                if "git status" in command:
                    result = "fatal: not a git repository"
                elif "ls" in command:
                    result = "README.md  main.py  config.json"
                elif "find" in command and ".py" in command:
                    result = "./main.py"
                else:
                    result = f"Command output for: {command}"

                content = content.replace(
                    match.group(0),
                    f"✅ Command executed successfully: `{command}`\nOutput: {result}",
                )

            return content

        with patch.object(
            cli,
            "_parse_and_execute_operations",
            side_effect=mock_command_execution,
        ):
            command = Command.create_chat_message(
                "check git status and list all files"
            )

            await cli._handle_chat_message(command)

            # Verify multiple commands were executed
            assert (
                len(executed_commands) >= 2
            ), f"Expected multiple commands, got: {executed_commands}"
            assert any(
                "git status" in cmd for cmd in executed_commands
            ), "git status was not executed"
            assert any(
                "ls" in cmd for cmd in executed_commands
            ), "ls command was not executed"

    @pytest.mark.asyncio
    async def test_workflow_error_recovery(
        self, temp_workspace, cli_with_mocked_engine
    ):
        """Test workflow behavior when operations fail."""
        cli = cli_with_mocked_engine

        # Mock engine responses that lead to failed operations
        async def error_inducing_responses(message):
            response = Mock()
            response.is_success = True
            response.model_used = "test-model"

            if "cause errors" in message.lower():
                response.content = """I'll try some operations that might fail.
                
[FILE_READ:nonexistent_file.txt]
[COMMAND_EXEC] invalid_command_that_fails [/COMMAND_EXEC]
[FILE_WRITE:readonly_file.txt]Some content[/FILE_WRITE]"""

            else:
                response.content = (
                    "Operations completed with some failures as expected."
                )

            return response

        cli.engine.send_message = AsyncMock(
            side_effect=error_inducing_responses
        )

        error_count = 0

        async def mock_error_prone_execution(content):
            nonlocal error_count
            import re

            # Simulate file read failure
            file_read_pattern = r"\[FILE_READ:([^\]]+)\]"
            for match in re.finditer(file_read_pattern, content):
                filename = match.group(1).strip()
                if "nonexistent" in filename:
                    error_count += 1
                    content = content.replace(
                        match.group(0), f"❌ File not found: {filename}"
                    )

            # Simulate command execution failure
            command_pattern = r"\[COMMAND_EXEC\]\s*(.*?)\s*\[/COMMAND_EXEC\]"
            for match in re.finditer(command_pattern, content, re.DOTALL):
                command = match.group(1).strip()
                if "invalid_command" in command:
                    error_count += 1
                    content = content.replace(
                        match.group(0), f"❌ Command failed: {command}"
                    )

            return content

        with patch.object(
            cli,
            "_parse_and_execute_operations",
            side_effect=mock_error_prone_execution,
        ):
            command = Command.create_chat_message(
                "try operations that might cause errors"
            )

            await cli._handle_chat_message(command)

            # Verify that errors were handled gracefully
            assert error_count > 0, "No errors were simulated"
            assert (
                cli.engine.send_message.call_count >= 1
            ), "Workflow didn't continue after errors"
