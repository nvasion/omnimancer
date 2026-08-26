"""Sandbox manager for isolating and limiting agent operations."""

import logging
import os
import resource
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

# How long to wait for a terminated process to flush its output pipes.
OUTPUT_DRAIN_TIMEOUT_SECONDS = 5

# How often the monitor thread samples a sandboxed process.
MONITOR_POLL_INTERVAL_SECONDS = 1.0

# How long to wait for the monitor thread to notice a finished process.
MONITOR_JOIN_TIMEOUT_SECONDS = 5


def _timeout_reason(timeout_seconds: int) -> str:
    """Build the termination reason used when a command exceeds its timeout."""
    return f"Command timed out after {timeout_seconds} seconds"


def _limit_reason(limit_name: str, actual: str, allowed: str) -> str:
    """Build the termination reason used when a resource limit is exceeded."""
    return f"Command exceeded {limit_name} limit ({actual} > {allowed})"


class ResourceLimits:
    """Resource limits for sandboxed operations."""

    def __init__(
        self,
        max_memory_mb: int = 512,
        max_cpu_seconds: int = 30,
        max_file_size_mb: int = 100,
        max_open_files: int = 100,
        max_processes: int = 10,
        timeout_seconds: int = 60,
    ):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        self.max_file_size_mb = max_file_size_mb
        self.max_open_files = max_open_files
        self.max_processes = max_processes
        self.timeout_seconds = timeout_seconds


class SandboxedProcess:
    """Represents a process running in a sandbox."""

    def __init__(
        self, process: subprocess.Popen, temp_dir: str, limits: ResourceLimits
    ):
        self.process = process
        self.temp_dir = temp_dir
        self.limits = limits
        self.start_time = time.time()
        self.monitor_thread: Optional[threading.Thread] = None
        self.terminated = False
        # Why the process was forcibly stopped, if it was. Used to explain a
        # non-zero exit code that would otherwise carry no diagnostic output.
        self.termination_reason: Optional[str] = None
        # Signals the monitor thread to stop sampling immediately instead of
        # sitting out the rest of its poll interval.
        self.stop_event = threading.Event()
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.process.poll() is None and not self.terminated

    def terminate(self, reason: Optional[str] = None) -> None:
        """Terminate the sandboxed process, recording why it was stopped.

        Safe to call from several threads: the first caller wins and its
        ``reason`` is the one reported back to the caller of
        :meth:`SandboxManager.execute_sandboxed_command`.
        """
        with self._lock:
            if self.terminated:
                return
            self.terminated = True
            if reason is not None and self.termination_reason is None:
                self.termination_reason = reason

        self.stop_event.set()

        try:
            if self.process.poll() is None:
                self.process.terminate()
                # Give it a moment to terminate gracefully
                try:
                    self.process.wait(timeout=OUTPUT_DRAIN_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        except (ProcessLookupError, OSError) as e:
            logger.debug(
                "Could not terminate sandboxed process %s: %s", self.process.pid, e
            )

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        self.terminate()
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except (OSError, PermissionError) as e:
                logger.warning("Could not remove sandbox directory: %s", e)


class SandboxManager:
    """Manages sandboxed execution of agent operations."""

    def __init__(self, default_limits: Optional[ResourceLimits] = None):
        self.default_limits = default_limits or ResourceLimits()
        self.active_processes: Dict[int, SandboxedProcess] = {}
        self.temp_base_dir = tempfile.gettempdir()

    def create_sandbox_environment(
        self, limits: Optional[ResourceLimits] = None
    ) -> str:
        """Create a temporary sandbox directory."""
        limits = limits or self.default_limits

        # Create temporary directory for sandbox
        temp_dir = tempfile.mkdtemp(prefix="omnimancer_sandbox_")

        # Set up basic directory structure
        os.makedirs(os.path.join(temp_dir, "workspace"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "output"), exist_ok=True)

        # Create sandbox info file
        info_file = os.path.join(temp_dir, "sandbox_info.txt")
        with open(info_file, "w") as f:
            f.write(f"Sandbox created at: {time.ctime()}\n")
            f.write(f"Memory limit: {limits.max_memory_mb} MB\n")
            f.write(f"CPU limit: {limits.max_cpu_seconds} seconds\n")
            f.write(f"Timeout: {limits.timeout_seconds} seconds\n")

        return temp_dir

    def execute_sandboxed_command(
        self,
        command: List[str],
        working_dir: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        limits: Optional[ResourceLimits] = None,
        input_data: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a command in a sandboxed environment."""

        limits = limits or self.default_limits
        sandbox_dir = self.create_sandbox_environment(limits)
        process: Optional[subprocess.Popen] = None
        sandboxed_proc: Optional[SandboxedProcess] = None

        try:
            # Prepare environment
            sandbox_env = os.environ.copy()
            if env_vars:
                sandbox_env.update(env_vars)

            # Restrict environment variables
            sandbox_env = self._filter_environment_variables(sandbox_env)

            # Set working directory
            work_dir = working_dir or os.path.join(sandbox_dir, "workspace")

            # Surface unenforceable limits before forking (see the helper).
            self._warn_about_unenforceable_limits(limits)

            # Prepare process arguments
            process_args = {
                "args": command,
                "cwd": work_dir,
                "env": sandbox_env,
                "stdin": subprocess.PIPE if input_data else None,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "preexec_fn": self._setup_process_limits(limits),
            }

            # Start process
            process = subprocess.Popen(**process_args)  # type: ignore[call-overload]
            sandboxed_proc = SandboxedProcess(process, sandbox_dir, limits)

            # Track the process
            self.active_processes[process.pid] = sandboxed_proc

            # Start monitoring
            sandboxed_proc.monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(sandboxed_proc,),
                daemon=True,
            )
            sandboxed_proc.monitor_thread.start()

            # Execute and wait for completion
            try:
                stdout, stderr = process.communicate(
                    input=input_data, timeout=limits.timeout_seconds
                )
                return_code = process.returncode

            except subprocess.TimeoutExpired:
                sandboxed_proc.terminate(_timeout_reason(limits.timeout_seconds))
                stdout, stderr = self._drain_output(process)
                # The deadline was missed, so this is a failure even if the
                # process happened to exit cleanly while being terminated.
                signalled_code = process.returncode
                return self._build_result(
                    return_code=(
                        signalled_code if signalled_code not in (None, 0) else -1
                    ),
                    stdout=stdout,
                    stderr=stderr,
                    sandbox_dir=sandbox_dir,
                    termination_reason=sandboxed_proc.termination_reason,
                )

            # The monitor thread may have killed the process before it could
            # write any diagnostics (for example when a limit was exceeded just
            # under the communicate() timeout). Surface that reason instead of
            # returning a bare non-zero exit code with empty output.
            return self._build_result(
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                sandbox_dir=sandbox_dir,
                termination_reason=sandboxed_proc.termination_reason,
            )

        except Exception as e:
            logger.exception("Sandboxed command execution failed")
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": f"Sandbox execution error: {str(e)}",
                "sandbox_dir": sandbox_dir,
            }

        finally:
            self._release_process(process, sandboxed_proc)

    @staticmethod
    def _build_result(
        return_code: Optional[int],
        stdout: Optional[str],
        stderr: Optional[str],
        sandbox_dir: str,
        termination_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble the result mapping returned to callers.

        A recorded ``termination_reason`` only overrides the outcome when the
        command did not already exit cleanly; a process that finished
        successfully in the same instant it was signalled is still a success.
        """
        code = -1 if return_code is None else return_code
        success = code == 0
        stderr_text = stderr or ""

        if termination_reason and not success:
            stderr_text = (
                f"{stderr_text.rstrip()}\n{termination_reason}".lstrip()
                if stderr_text.strip()
                else termination_reason
            )

        return {
            "success": success,
            "return_code": code,
            "stdout": stdout or "",
            "stderr": stderr_text,
            "sandbox_dir": sandbox_dir,
        }

    @staticmethod
    def _drain_output(process: subprocess.Popen) -> Tuple[str, str]:
        """Collect whatever a stopped process managed to write to its pipes."""
        for attempt in range(2):
            try:
                stdout, stderr = process.communicate(
                    timeout=OUTPUT_DRAIN_TIMEOUT_SECONDS
                )
                return stdout or "", stderr or ""
            except subprocess.TimeoutExpired:
                # A child of the sandboxed process may still hold the pipes
                # open; kill the group leader once and retry, then give up.
                if attempt == 0:
                    process.kill()
                    continue
                logger.warning(
                    "Timed out draining output of sandboxed process %s", process.pid
                )
            except (OSError, ValueError) as e:
                logger.warning(
                    "Could not drain output of sandboxed process %s: %s",
                    process.pid,
                    e,
                )
                break
        return "", ""

    def _release_process(
        self,
        process: Optional[subprocess.Popen],
        sandboxed_proc: Optional[SandboxedProcess],
    ) -> None:
        """Stop monitoring, clean up the sandbox and untrack the process."""
        if sandboxed_proc is None:
            return

        # cleanup() terminates the process, which wakes the monitor thread.
        sandboxed_proc.cleanup()

        if sandboxed_proc.monitor_thread is not None:
            sandboxed_proc.monitor_thread.join(timeout=MONITOR_JOIN_TIMEOUT_SECONDS)
            if sandboxed_proc.monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop within the join timeout")

        if process is not None:
            self.active_processes.pop(process.pid, None)

    @staticmethod
    def _resource_limit_values(limits: ResourceLimits) -> Dict[int, Tuple[int, int]]:
        """Map each ``RLIMIT_*`` constant to the soft/hard value to apply."""
        return {
            resource.RLIMIT_AS: (limits.max_memory_mb * 1024 * 1024,) * 2,
            resource.RLIMIT_CPU: (limits.max_cpu_seconds,) * 2,
            resource.RLIMIT_FSIZE: (limits.max_file_size_mb * 1024 * 1024,) * 2,
            resource.RLIMIT_NOFILE: (limits.max_open_files,) * 2,
            resource.RLIMIT_NPROC: (limits.max_processes,) * 2,
        }

    def _warn_about_unenforceable_limits(self, limits: ResourceLimits) -> None:
        """Warn, before forking, about limits the kernel will refuse to apply.

        ``setrlimit`` runs post-fork where logging is not safe, so any
        diagnostics have to be produced here in the parent. The monitor thread
        still enforces memory, CPU and runtime caps if a limit cannot be set.
        """
        for limit_id, (requested, _) in self._resource_limit_values(limits).items():
            try:
                _, hard = resource.getrlimit(limit_id)
            except (OSError, ValueError) as e:
                logger.warning("Could not read resource limit %s: %s", limit_id, e)
                continue

            if hard != resource.RLIM_INFINITY and requested > hard:
                logger.warning(
                    "Requested resource limit %s (%s) exceeds the hard limit (%s); "
                    "falling back to monitor-based enforcement",
                    limit_id,
                    requested,
                    hard,
                )

    def _setup_process_limits(self, limits: ResourceLimits) -> Callable:
        """Create a function to set up process resource limits."""

        limit_values = self._resource_limit_values(limits)

        def setup_limits() -> None:
            # Runs in the forked child before exec: only async-signal-safe work
            # is allowed here, so failures cannot be logged or reported. They
            # are non-fatal by design -- a limit the kernel refuses (common in
            # containers) is reported by _warn_about_unenforceable_limits() and
            # still enforced by the monitor thread.
            for limit_id, values in limit_values.items():
                try:
                    resource.setrlimit(limit_id, values)
                except (OSError, ValueError):
                    continue

        return setup_limits

    def _monitor_process(self, sandboxed_proc: SandboxedProcess) -> None:
        """Monitor a sandboxed process for resource violations."""

        limits = sandboxed_proc.limits
        pid = sandboxed_proc.process.pid

        while sandboxed_proc.is_running():
            try:
                # Check if process still exists
                if sandboxed_proc.process.poll() is not None:
                    break

                # Get process info
                try:
                    proc_info = psutil.Process(pid)

                    # Check memory usage
                    memory_mb = proc_info.memory_info().rss / (1024 * 1024)
                    if memory_mb > limits.max_memory_mb:
                        self._terminate_for_limit(
                            sandboxed_proc,
                            _limit_reason(
                                "memory",
                                f"{memory_mb:.1f} MB",
                                f"{limits.max_memory_mb} MB",
                            ),
                        )
                        break

                    # Check CPU time
                    cpu_times = proc_info.cpu_times()
                    cpu_time = cpu_times.user + cpu_times.system
                    if cpu_time > limits.max_cpu_seconds:
                        self._terminate_for_limit(
                            sandboxed_proc,
                            _limit_reason(
                                "CPU time",
                                f"{cpu_time:.1f}s",
                                f"{limits.max_cpu_seconds}s",
                            ),
                        )
                        break

                    # Check total runtime
                    runtime = time.time() - sandboxed_proc.start_time
                    if runtime > limits.timeout_seconds:
                        self._terminate_for_limit(
                            sandboxed_proc, _timeout_reason(limits.timeout_seconds)
                        )
                        break

                except psutil.NoSuchProcess:
                    # Process already terminated
                    break

                # Wait before the next check, waking early if the process is
                # terminated in the meantime.
                if sandboxed_proc.stop_event.wait(MONITOR_POLL_INTERVAL_SECONDS):
                    break

            except Exception as e:
                logger.warning("Error monitoring sandboxed process %s: %s", pid, e)
                break

    @staticmethod
    def _terminate_for_limit(sandboxed_proc: SandboxedProcess, reason: str) -> None:
        """Stop a process that violated a resource limit and record why."""
        logger.warning(
            "Terminating sandboxed process %s: %s", sandboxed_proc.process.pid, reason
        )
        sandboxed_proc.terminate(reason)

    def _filter_environment_variables(self, env: Dict[str, str]) -> Dict[str, str]:
        """Filter environment variables to remove sensitive ones."""

        # List of sensitive environment variable patterns
        sensitive_patterns = [
            "PASSWORD",
            "SECRET",
            "TOKEN",
            "KEY",
            "CREDENTIAL",
            "AWS_",
            "AZURE_",
            "GCP_",
            "GOOGLE_",
            "SSH_",
            "HOME",
            "USER",
            "USERNAME",
        ]

        filtered_env = {}
        for key, value in env.items():
            # Keep only safe environment variables
            if not any(pattern in key.upper() for pattern in sensitive_patterns):
                filtered_env[key] = value

        # Add minimal required variables
        filtered_env.update(
            {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
            }
        )

        return filtered_env

    @contextmanager  # type: ignore[arg-type]
    def sandbox_context(  # type: ignore[misc]
        self,
        limits: Optional[ResourceLimits] = None,
    ) -> None:
        """Context manager for sandbox operations."""

        sandbox_dir = self.create_sandbox_environment(limits)
        try:
            yield sandbox_dir
        finally:
            if os.path.exists(sandbox_dir):
                try:
                    shutil.rmtree(sandbox_dir)
                except (OSError, PermissionError):
                    pass

    def cleanup_all_sandboxes(self) -> None:
        """Clean up all active sandboxes."""

        for proc in list(self.active_processes.values()):
            proc.cleanup()
        self.active_processes.clear()

    def get_active_process_count(self) -> int:
        """Get number of active sandboxed processes."""
        return len(self.active_processes)

    def get_sandbox_info(self, process_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a sandboxed process."""

        if process_id not in self.active_processes:
            return None

        proc = self.active_processes[process_id]
        try:
            proc_info = psutil.Process(process_id)
            return {
                "pid": process_id,
                "is_running": proc.is_running(),
                "start_time": proc.start_time,
                "runtime": time.time() - proc.start_time,
                "memory_mb": proc_info.memory_info().rss / (1024 * 1024),
                "cpu_percent": proc_info.cpu_percent(),
                "sandbox_dir": proc.temp_dir,
                "limits": {
                    "max_memory_mb": proc.limits.max_memory_mb,
                    "max_cpu_seconds": proc.limits.max_cpu_seconds,
                    "timeout_seconds": proc.limits.timeout_seconds,
                },
            }
        except psutil.NoSuchProcess:
            return {
                "pid": process_id,
                "is_running": False,
                "sandbox_dir": proc.temp_dir,
            }
