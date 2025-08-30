"""
Enhanced Program Execution and Command Runner for Omnimancer Agent.

This module provides secure command execution with whitelisting, sandboxing,
real-time output streaming, resource limits, and comprehensive error handling.
"""

import asyncio
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)


from ...utils.errors import SecurityError
from ..security.approval_workflow import ApprovalWorkflow, RiskLevel
from ..security.sandbox_manager import (
    ResourceLimits,
    SandboxedProcess,
    SandboxManager,
)

logger = logging.getLogger(__name__)


class CommandCategory(Enum):
    """Categories of commands for security classification."""

    SAFE_READ = "safe_read"  # ls, cat, grep, find
    DEVELOPMENT = "development"  # git, npm, pip, make
    SYSTEM_INFO = "system_info"  # ps, df, uptime, which
    BUILD_TOOLS = "build_tools"  # docker, docker-compose
    PACKAGE_MANAGERS = "package_mgmt"  # npm, pip, yarn, composer
    DANGEROUS = "dangerous"  # rm, sudo, chmod, dd
    NETWORK = "network"  # curl, wget, ssh, nc


class ExecutionMode(Enum):
    """Execution modes with different security levels."""

    RESTRICTED = "restricted"  # Only safe read commands
    DEVELOPMENT = "development"  # Safe + development tools
    FULL_ACCESS = "full_access"  # All whitelisted commands
    SANDBOX = "sandbox"  # Isolated execution environment


@dataclass
class CommandResult:
    """Result of command execution with streaming support."""

    success: bool
    command: str
    args: List[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    sandbox_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    was_terminated: bool = False

    @property
    def full_command(self) -> str:
        """Get the full command string."""
        return f"{self.command} {' '.join(self.args)}" if self.args else self.command


@dataclass
class ExecutionConfig:
    """Configuration for command execution."""

    timeout_seconds: int = 30
    max_memory_mb: int = 512
    max_output_size: int = 10 * 1024 * 1024  # 10MB
    working_directory: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    execution_mode: ExecutionMode = ExecutionMode.DEVELOPMENT
    enable_streaming: bool = True
    stream_callback: Optional[Callable[[str, str], None]] = (
        None  # (stream_type, content)
    )
    require_approval: bool = True


class CommandValidator:
    """Validates commands against security policies."""

    def __init__(self):
        self.command_categories = {
            CommandCategory.SAFE_READ: {
                "ls",
                "cat",
                "head",
                "tail",
                "grep",
                "find",
                "wc",
                "sort",
                "uniq",
                "diff",
                "less",
                "more",
                "file",
                "stat",
                "du",
                "tree",
                "echo",
            },
            CommandCategory.DEVELOPMENT: {
                "git",
                "python",
                "python3",
                "node",
                "npm",
                "yarn",
                "pip",
                "pip3",
                "make",
                "cmake",
                "javac",
                "java",
                "gcc",
                "g++",
                "rustc",
                "cargo",
                "go",
                "mvn",
                "gradle",
                "composer",
            },
            CommandCategory.SYSTEM_INFO: {
                "ps",
                "top",
                "htop",
                "df",
                "free",
                "uptime",
                "whoami",
                "id",
                "uname",
                "which",
                "whereis",
                "env",
                "printenv",
                "pwd",
                "date",
                "sleep",
            },
            CommandCategory.BUILD_TOOLS: {
                "docker",
                "docker-compose",
                "podman",
                "buildah",
                "kubernetes",
                "kubectl",
                "helm",
                "terraform",
                "ansible",
            },
            CommandCategory.PACKAGE_MANAGERS: {
                "apt",
                "apt-get",
                "yum",
                "dnf",
                "pacman",
                "brew",
                "choco",
                "snap",
                "flatpak",
                "conda",
                "mamba",
            },
            CommandCategory.DANGEROUS: {
                "rm",
                "rmdir",
                "mv",
                "cp",
                "chmod",
                "chown",
                "chgrp",
                "sudo",
                "su",
                "passwd",
                "dd",
                "fdisk",
                "parted",
                "mkfs",
                "mount",
                "umount",
                "systemctl",
                "service",
                "kill",
                "killall",
                "pkill",
            },
            CommandCategory.NETWORK: {
                "curl",
                "wget",
                "ssh",
                "scp",
                "rsync",
                "nc",
                "netcat",
                "telnet",
                "ftp",
                "sftp",
                "ping",
                "traceroute",
                "nslookup",
                "dig",
            },
        }

        # Commands allowed by execution mode
        self.mode_permissions = {
            ExecutionMode.RESTRICTED: {CommandCategory.SAFE_READ},
            ExecutionMode.DEVELOPMENT: {
                CommandCategory.SAFE_READ,
                CommandCategory.DEVELOPMENT,
                CommandCategory.SYSTEM_INFO,
            },
            ExecutionMode.FULL_ACCESS: {
                CommandCategory.SAFE_READ,
                CommandCategory.DEVELOPMENT,
                CommandCategory.SYSTEM_INFO,
                CommandCategory.BUILD_TOOLS,
                CommandCategory.PACKAGE_MANAGERS,
                CommandCategory.NETWORK,
            },
            ExecutionMode.SANDBOX: {
                CommandCategory.SAFE_READ,
                CommandCategory.DEVELOPMENT,
                CommandCategory.SYSTEM_INFO,
            },
        }

        # Dangerous patterns in command arguments
        self.dangerous_patterns = [
            r"rm\s+-rf\s*/",  # rm -rf /
            r"sudo\s+rm",  # sudo rm
            r"chmod\s+777",  # chmod 777
            r">\s*/dev/",  # redirect to /dev
            r"\|\s*nc\s",  # pipe to netcat
            r"eval\s*\(",  # eval execution
            r"exec\s*\(",  # exec execution
            r"system\s*\(",  # system calls
            r"`[^`]*`",  # backtick execution
            r"\$\([^)]*\)",  # command substitution
        ]

    def get_command_category(self, command: str) -> Optional[CommandCategory]:
        """Get the category of a command."""
        command_lower = command.lower()
        for category, commands in self.command_categories.items():
            if command_lower in commands:
                return category
        return None

    def is_command_allowed(self, command: str, mode: ExecutionMode) -> bool:
        """Check if command is allowed in the given execution mode."""
        category = self.get_command_category(command)
        if not category:
            return False  # Unknown commands are not allowed

        allowed_categories = self.mode_permissions.get(mode, set())
        return category in allowed_categories

    def assess_command_risk(self, command: str, args: List[str]) -> RiskLevel:
        """Assess the risk level of a command."""
        category = self.get_command_category(command)

        if category == CommandCategory.DANGEROUS:
            return RiskLevel.CRITICAL

        full_command = f"{command} {' '.join(args)}"

        # Check for dangerous patterns
        import re

        for pattern in self.dangerous_patterns:
            if re.search(pattern, full_command, re.IGNORECASE):
                return RiskLevel.CRITICAL

        # Risk by category
        risk_mapping = {
            CommandCategory.SAFE_READ: RiskLevel.LOW,
            CommandCategory.SYSTEM_INFO: RiskLevel.LOW,
            CommandCategory.DEVELOPMENT: RiskLevel.MEDIUM,
            CommandCategory.BUILD_TOOLS: RiskLevel.MEDIUM,
            CommandCategory.PACKAGE_MANAGERS: RiskLevel.HIGH,
            CommandCategory.NETWORK: RiskLevel.HIGH,
        }

        return risk_mapping.get(category, RiskLevel.HIGH)

    def validate_command_args(self, command: str, args: List[str]) -> List[str]:
        """Validate and sanitize command arguments."""

        sanitized_args = []
        for arg in args:
            # Remove dangerous characters and patterns
            if any(char in arg for char in [";", "|", "&", "`", "$", ">"]):
                # For now, reject arguments with shell metacharacters
                # In production, this could be more sophisticated
                raise SecurityError(f"Argument contains dangerous characters: {arg}")

            # Ensure proper escaping
            sanitized_args.append(shlex.quote(arg))

        return sanitized_args


class StreamingExecutor:
    """Handles real-time output streaming from processes."""

    def __init__(self, callback: Optional[Callable[[str, str], None]] = None):
        self.callback = callback
        self.stdout_buffer = []
        self.stderr_buffer = []

    async def stream_process_output(self, process: asyncio.subprocess.Process) -> None:
        """Stream output from a running process."""

        async def read_stream(
            stream: asyncio.StreamReader, stream_type: str, buffer: List[str]
        ):
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break

                    line_str = line.decode("utf-8", errors="replace")
                    buffer.append(line_str)

                    if self.callback:
                        try:
                            self.callback(stream_type, line_str.rstrip())
                        except Exception as callback_error:
                            logger.debug(f"Stream callback error: {callback_error}")

            except Exception as e:
                logger.debug(f"Error reading {stream_type}: {e}")

        # Start streaming tasks
        tasks = []
        if process.stdout:
            tasks.append(read_stream(process.stdout, "stdout", self.stdout_buffer))
        if process.stderr:
            tasks.append(read_stream(process.stderr, "stderr", self.stderr_buffer))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_output(self) -> tuple[str, str]:
        """Get accumulated output."""
        stdout = "".join(self.stdout_buffer)
        stderr = "".join(self.stderr_buffer)
        return stdout, stderr


class EnhancedProgramExecutor:
    """
    Enhanced program executor with comprehensive security and monitoring.

    Features:
    - Command whitelisting and validation
    - Sandboxed execution with resource limits
    - Real-time output streaming
    - Comprehensive error handling and cleanup
    - Integration with approval workflow
    """

    def __init__(
        self,
        sandbox_manager: Optional[SandboxManager] = None,
        approval_workflow: Optional[ApprovalWorkflow] = None,
    ):
        self.validator = CommandValidator()
        self.sandbox_manager = sandbox_manager or SandboxManager()
        self.approval_workflow = approval_workflow
        self.active_processes: Dict[str, SandboxedProcess] = {}
        self.execution_history: List[CommandResult] = []

    async def execute_command(
        self,
        command: str,
        args: List[str] = None,
        config: ExecutionConfig = None,
    ) -> CommandResult:
        """
        Execute a command with full security controls.

        Args:
            command: Command to execute
            args: Command arguments
            config: Execution configuration

        Returns:
            CommandResult with execution details
        """
        if args is None:
            args = []
        if config is None:
            config = ExecutionConfig()

        execution_id = f"exec_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            # Validate command
            if not self.validator.is_command_allowed(command, config.execution_mode):
                raise SecurityError(
                    f"Command '{command}' not allowed in {config.execution_mode.value} mode"
                )

            # Assess risk and request approval if needed
            risk_level = self.validator.assess_command_risk(command, args)
            if config.require_approval and self.approval_workflow:
                approved = await self._request_approval(
                    command, args, risk_level, config
                )
                if not approved:
                    return CommandResult(
                        success=False,
                        command=command,
                        args=args,
                        exit_code=-1,
                        error_message="Command execution denied by user",
                    )

            # Sanitize arguments
            sanitized_args = self.validator.validate_command_args(command, args)

            # Execute based on mode
            if config.execution_mode == ExecutionMode.SANDBOX:
                result = await self._execute_sandboxed(
                    command, sanitized_args, config, execution_id
                )
            else:
                result = await self._execute_direct(command, sanitized_args, config)

            result.execution_time = time.time() - start_time
            self.execution_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return CommandResult(
                success=False,
                command=command,
                args=args,
                exit_code=-1,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )
        finally:
            # Cleanup active process tracking
            if execution_id in self.active_processes:
                del self.active_processes[execution_id]

    async def _execute_direct(
        self, command: str, args: List[str], config: ExecutionConfig
    ) -> CommandResult:
        """Execute command directly with basic controls."""
        try:
            # Prepare environment
            env = dict(os.environ)
            env.update(config.environment_vars)

            # Create process
            process = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=config.working_directory,
                env=env,
            )

            # Handle streaming if enabled
            if config.enable_streaming:
                streaming_executor = StreamingExecutor(config.stream_callback)
                stream_task = asyncio.create_task(
                    streaming_executor.stream_process_output(process)
                )

                # Wait for completion with timeout
                try:
                    await asyncio.wait_for(
                        process.wait(), timeout=config.timeout_seconds
                    )
                    await stream_task
                    stdout, stderr = streaming_executor.get_output()
                except asyncio.TimeoutError:
                    # Cancel the streaming task first
                    stream_task.cancel()
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()

                    # Clean up the streaming task
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass

                    stdout, stderr = streaming_executor.get_output()
                    return CommandResult(
                        success=False,
                        command=command,
                        args=args,
                        exit_code=-1,
                        stdout=stdout,
                        stderr=stderr,
                        was_terminated=True,
                        error_message=f"Command timed out after {config.timeout_seconds} seconds",
                    )
            else:
                # Simple execution without streaming
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=config.timeout_seconds
                    )
                    stdout = stdout.decode("utf-8", errors="replace")
                    stderr = stderr.decode("utf-8", errors="replace")
                except asyncio.TimeoutError:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()

                    return CommandResult(
                        success=False,
                        command=command,
                        args=args,
                        exit_code=-1,
                        was_terminated=True,
                        error_message=f"Command timed out after {config.timeout_seconds} seconds",
                    )

            return CommandResult(
                success=process.returncode == 0,
                command=command,
                args=args,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                error_message=stderr if process.returncode != 0 else None,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                command=command,
                args=args,
                exit_code=-1,
                error_message=f"Execution failed: {str(e)}",
            )

    async def _execute_sandboxed(
        self,
        command: str,
        args: List[str],
        config: ExecutionConfig,
        execution_id: str,
    ) -> CommandResult:
        """Execute command in sandbox with resource limits."""
        resource_limits = ResourceLimits(
            max_memory_mb=config.max_memory_mb,
            timeout_seconds=config.timeout_seconds,
            max_file_size_mb=100,
            max_open_files=100,
            max_processes=10,
        )

        try:
            with self.sandbox_manager.create_sandbox(resource_limits) as sandbox:
                # Execute in sandbox
                sandboxed_process = await sandbox.execute_command(
                    command,
                    args,
                    working_dir=config.working_directory,
                    env=config.environment_vars,
                    enable_streaming=config.enable_streaming,
                    stream_callback=config.stream_callback,
                )

                self.active_processes[execution_id] = sandboxed_process

                # Wait for completion
                result = await sandboxed_process.wait_for_completion()

                return CommandResult(
                    success=result.exit_code == 0,
                    command=command,
                    args=args,
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    resource_usage=result.resource_usage,
                    sandbox_info=result.sandbox_info,
                    was_terminated=result.was_terminated,
                    error_message=result.error_message,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                command=command,
                args=args,
                exit_code=-1,
                error_message=f"Sandboxed execution failed: {str(e)}",
            )

    async def _request_approval(
        self,
        command: str,
        args: List[str],
        risk_level: RiskLevel,
        config: ExecutionConfig,
    ) -> bool:
        """Request approval for command execution."""
        if not self.approval_workflow:
            return True

        full_command = f"{command} {' '.join(args)}"

        request = await self.approval_workflow.request_approval(
            operation_type="command_execute",
            description=f"Execute command: {full_command}",
            metadata={
                "command": command,
                "args": args,
                "working_dir": config.working_directory,
                "execution_mode": config.execution_mode.value,
                "risk_level": risk_level.value,
            },
        )

        return request.status.value == "approved"

    async def terminate_command(self, execution_id: str) -> bool:
        """Terminate a running command."""
        if execution_id in self.active_processes:
            process = self.active_processes[execution_id]
            process.terminate()
            return True
        return False

    def get_execution_history(self, limit: int = 50) -> List[CommandResult]:
        """Get recent execution history."""
        return self.execution_history[-limit:]

    def get_active_processes(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently running processes."""
        active = {}
        for exec_id, process in self.active_processes.items():
            if process.is_running():
                active[exec_id] = {
                    "command": f"{process.process.args[0]} {' '.join(process.process.args[1:])}",
                    "pid": process.process.pid,
                    "start_time": process.start_time,
                    "runtime": time.time() - process.start_time,
                }
        return active

    async def stream_command_output(
        self,
        command: str,
        args: List[str] = None,
        config: ExecutionConfig = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Execute command and yield real-time output.

        Yields:
            Tuples of (stream_type, content) where stream_type is 'stdout' or 'stderr'
        """
        if args is None:
            args = []
        if config is None:
            config = ExecutionConfig(enable_streaming=True)

        # Set up streaming callback to yield output
        output_queue = asyncio.Queue()

        def stream_callback(stream_type: str, content: str):
            try:
                # Use the current event loop to schedule the task safely
                loop = asyncio.get_running_loop()
                loop.create_task(output_queue.put((stream_type, content)))
            except RuntimeError:
                # Event loop might be closing, ignore this output
                pass
            except Exception:
                # Ignore other exceptions to prevent callback from crashing the program
                pass

        config.stream_callback = stream_callback

        # Start execution in background
        execution_task = asyncio.create_task(
            self.execute_command(command, args, config)
        )

        # Yield output as it becomes available
        try:
            while not execution_task.done():
                try:
                    stream_type, content = await asyncio.wait_for(
                        output_queue.get(), timeout=0.1
                    )
                    yield stream_type, content
                except asyncio.TimeoutError:
                    continue

            # Get any remaining output
            while not output_queue.empty():
                stream_type, content = await output_queue.get()
                yield stream_type, content

        finally:
            # Ensure execution completes
            if not execution_task.done():
                execution_task.cancel()
                try:
                    await execution_task
                except (asyncio.CancelledError, Exception):
                    # Ignore cancellation and other exceptions during cleanup
                    pass
