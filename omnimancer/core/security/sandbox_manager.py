"""Sandbox manager for isolating and limiting agent operations."""

import os
import resource
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import psutil


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
        self.monitor_thread = None
        self.terminated = False

    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.process.poll() is None and not self.terminated

    def terminate(self) -> None:
        """Terminate the sandboxed process."""
        if not self.terminated:
            self.terminated = True
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                    # Give it a moment to terminate gracefully
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
            except (ProcessLookupError, OSError):
                pass

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        self.terminate()
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except (OSError, PermissionError):
                pass


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

        try:
            # Prepare environment
            sandbox_env = os.environ.copy()
            if env_vars:
                sandbox_env.update(env_vars)

            # Restrict environment variables
            sandbox_env = self._filter_environment_variables(sandbox_env)

            # Set working directory
            work_dir = working_dir or os.path.join(sandbox_dir, "workspace")

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
            process = subprocess.Popen(**process_args)
            sandboxed_proc = SandboxedProcess(process, sandbox_dir, limits)

            # Track the process
            self.active_processes[process.pid] = sandboxed_proc

            # Start monitoring
            sandboxed_proc.monitor_thread = threading.Thread(
                target=self._monitor_process, args=(sandboxed_proc,)
            )
            sandboxed_proc.monitor_thread.start()

            # Execute and wait for completion
            try:
                stdout, stderr = process.communicate(
                    input=input_data, timeout=limits.timeout_seconds
                )
                return_code = process.returncode

            except subprocess.TimeoutExpired:
                sandboxed_proc.terminate()
                return {
                    "success": False,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": "Command timed out",
                    "sandbox_dir": sandbox_dir,
                }

            return {
                "success": return_code == 0,
                "return_code": return_code,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "sandbox_dir": sandbox_dir,
            }

        except Exception as e:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": f"Sandbox execution error: {str(e)}",
                "sandbox_dir": sandbox_dir,
            }

        finally:
            # Clean up
            if process.pid in self.active_processes:
                self.active_processes[process.pid].cleanup()
                del self.active_processes[process.pid]

    def _setup_process_limits(self, limits: ResourceLimits) -> Callable:
        """Create a function to set up process resource limits."""

        def setup_limits():
            try:
                # Set memory limit (in bytes)
                memory_limit = limits.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

                # Set CPU time limit (in seconds)
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (limits.max_cpu_seconds, limits.max_cpu_seconds),
                )

                # Set file size limit (in bytes)
                file_size_limit = limits.max_file_size_mb * 1024 * 1024
                resource.setrlimit(
                    resource.RLIMIT_FSIZE, (file_size_limit, file_size_limit)
                )

                # Set maximum number of open files
                resource.setrlimit(
                    resource.RLIMIT_NOFILE,
                    (limits.max_open_files, limits.max_open_files),
                )

                # Set maximum number of processes
                resource.setrlimit(
                    resource.RLIMIT_NPROC,
                    (limits.max_processes, limits.max_processes),
                )

            except (OSError, ValueError) as e:
                # Log the error but don't fail the process creation
                print(f"Warning: Could not set all resource limits: {e}")

        return setup_limits

    def _monitor_process(self, sandboxed_proc: SandboxedProcess) -> None:
        """Monitor a sandboxed process for resource violations."""

        while sandboxed_proc.is_running():
            try:
                # Check if process still exists
                if sandboxed_proc.process.poll() is not None:
                    break

                # Get process info
                try:
                    proc_info = psutil.Process(sandboxed_proc.process.pid)

                    # Check memory usage
                    memory_mb = proc_info.memory_info().rss / (1024 * 1024)
                    if memory_mb > sandboxed_proc.limits.max_memory_mb:
                        print(
                            f"Process {sandboxed_proc.process.pid} exceeded memory limit"
                        )
                        sandboxed_proc.terminate()
                        break

                    # Check CPU time
                    cpu_time = proc_info.cpu_times().user + proc_info.cpu_times().system
                    if cpu_time > sandboxed_proc.limits.max_cpu_seconds:
                        print(
                            f"Process {sandboxed_proc.process.pid} exceeded CPU time limit"
                        )
                        sandboxed_proc.terminate()
                        break

                    # Check total runtime
                    runtime = time.time() - sandboxed_proc.start_time
                    if runtime > sandboxed_proc.limits.timeout_seconds:
                        print(
                            f"Process {sandboxed_proc.process.pid} exceeded runtime limit"
                        )
                        sandboxed_proc.terminate()
                        break

                except psutil.NoSuchProcess:
                    # Process already terminated
                    break

                # Sleep before next check
                time.sleep(1.0)

            except Exception as e:
                print(f"Error monitoring process: {e}")
                break

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

    @contextmanager
    def sandbox_context(self, limits: Optional[ResourceLimits] = None):
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
