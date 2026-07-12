"""Full-trust mode: --dangerously-skip-permissions must bypass the hard
security layers in headless mode, not just auto-approve.

Regression: headless runs with --dangerously-skip-permissions still had
operations denied by (1) the command argument sanitizer (pipes, &&,
redirects), (2) the forbidden-command list, (3) the sensitive filename
patterns (*key*, *token*, .env, ...), and (4) the 30s default command
timeout — none of which the approval callback covers. Full trust relaxes all
four while keeping hard-restricted system paths blocked.
"""

import os
from unittest.mock import MagicMock

import pytest

from omnimancer.core.agent.program_executor import CommandValidator
from omnimancer.core.agent_managers import ProgramExecutor, SecurityError
from omnimancer.core.security.permission_controller import PermissionController
from omnimancer.core.security.security_manager import SecurityManager


class TestCommandValidatorFullTrust:
    def test_default_rejects_shell_metacharacters(self):
        validator = CommandValidator()
        with pytest.raises(SecurityError):
            validator.validate_command_args("go", ["test", "./...", "&&", "make"])

    def test_full_trust_passes_shell_metacharacters_through(self):
        validator = CommandValidator()
        validator.full_trust = True
        args = ["test", "./...", "|", "tail", ">", "out.txt", "&&", "make"]
        assert validator.validate_command_args("go", args) == args

    def test_full_trust_defaults_off(self):
        assert CommandValidator().full_trust is False


class TestProgramExecutorFullTrust:
    def test_default_forbids_rm(self):
        executor = ProgramExecutor()
        with pytest.raises(SecurityError):
            executor._validate_command("rm -rf build/")

    def test_full_trust_allows_forbidden_commands(self):
        executor = ProgramExecutor()
        executor.set_full_trust(True)
        assert executor._validate_command("rm -rf build/") is True

    def test_full_trust_raises_default_timeout(self):
        executor = ProgramExecutor()
        assert executor.timeout_seconds == 30
        executor.set_full_trust(True)
        assert executor.timeout_seconds == 600
        assert executor.default_config.timeout_seconds == 600

    def test_full_trust_propagates_to_validator(self):
        executor = ProgramExecutor()
        executor.set_full_trust(True)
        assert executor.enhanced_executor.validator.full_trust is True


class TestPermissionControllerFullTrust:
    def test_default_denies_sensitive_filenames(self):
        controller = PermissionController()
        path = os.path.join(os.getcwd(), "internal", "key_manager.go")
        assert controller.validate_path_access(path, "read") is False

    def test_full_trust_allows_sensitive_filenames(self):
        controller = PermissionController()
        controller.full_trust = True
        path = os.path.join(os.getcwd(), "internal", "key_manager.go")
        assert controller.validate_path_access(path, "read") is True
        env_path = os.path.join(os.getcwd(), ".env")
        assert controller.validate_path_access(env_path, "write") is True

    def test_full_trust_keeps_restricted_paths_blocked(self):
        controller = PermissionController()
        controller.full_trust = True
        assert (
            controller.validate_path_access(os.path.expanduser("~/.ssh/id_rsa"), "read")
            is False
        )
        assert controller.validate_path_access("/etc/passwd", "write") is False

    def test_full_trust_allows_chained_and_unknown_commands(self):
        controller = PermissionController()
        assert controller.validate_command("cd x && make") is False
        controller.full_trust = True
        assert controller.validate_command("cd x && make") is True
        assert controller.validate_command("some-unknown-tool --flag") is True

    def test_security_manager_setter_propagates(self):
        manager = SecurityManager(
            enable_sandbox=False,
            enable_approval_workflow=False,
            enable_audit_logging=False,
        )
        assert manager.permissions.full_trust is False
        manager.set_full_trust(True)
        assert manager.permissions.full_trust is True


class TestHeadlessFullTrustWiring:
    def test_enable_full_trust_flips_engine_managers(self):
        from omnimancer.cli.headless import HeadlessRunner

        agent_engine = MagicMock()
        executor = ProgramExecutor()
        agent_engine.executor = executor
        file_system = MagicMock()
        agent_engine.file_system = file_system

        HeadlessRunner._enable_full_trust(agent_engine)

        assert executor.full_trust is True
        assert executor.timeout_seconds == 600
        file_system.set_full_trust.assert_called_once_with(True)

    def test_missing_managers_do_not_crash(self):
        from omnimancer.cli.headless import HeadlessRunner

        agent_engine = object()  # no executor / file_system attributes
        HeadlessRunner._enable_full_trust(agent_engine)
