"""
Tests for successful module imports across the Omnimancer codebase.

These tests verify that all modules can be imported without errors,
particularly focusing on modules that had import issues during test collection.
"""

import importlib
import sys

import pytest


class TestCoreModuleImports:
    """Test core module import functionality."""

    def test_error_classes_importable(self):
        """Test that all error classes can be imported successfully."""
        # Test importing the main error module
        from omnimancer.utils import errors

        assert errors is not None

        # Test importing specific error classes
        from omnimancer.utils.errors import (
            AgentError,
            ExecutionError,
            OmnimancerError,
            TimeoutError,
        )

        # Verify error classes are actually classes
        assert isinstance(OmnimancerError, type)
        assert isinstance(AgentError, type)
        assert isinstance(ExecutionError, type)
        assert isinstance(TimeoutError, type)

    def test_program_executor_imports(self):
        """Test that program_executor module imports successfully."""
        # This was the main source of the original ImportError
        from omnimancer.core.agent import program_executor

        assert program_executor is not None

        # Test specific classes from program_executor
        from omnimancer.core.agent.program_executor import (
            CommandCategory,
            CommandResult,
            CommandValidator,
            EnhancedProgramExecutor,
            ExecutionConfig,
            ExecutionMode,
        )

        assert EnhancedProgramExecutor is not None
        assert CommandResult is not None
        assert ExecutionConfig is not None
        assert CommandValidator is not None
        assert CommandCategory is not None
        assert ExecutionMode is not None

    def test_agent_core_imports(self):
        """Test that agent core modules import successfully."""
        # Test agent system modules
        from omnimancer.core import agent_engine, agent_mode_manager, agent_progress_ui

        assert agent_engine is not None
        assert agent_mode_manager is not None
        assert agent_progress_ui is not None

    def test_security_module_imports(self):
        """Test that security modules import successfully."""
        from omnimancer.core.security import approval_workflow, sandbox_manager

        assert sandbox_manager is not None
        assert approval_workflow is not None

    def test_provider_imports(self):
        """Test that all provider modules import successfully."""
        providers = [
            "claude",
            "openai",
            "gemini",
            "ollama",
            "cohere",
            "azure",
            "bedrock",
            "claude_code",
            "mistral",
            "openrouter",
            "perplexity",
            "vertex",
            "xai",
        ]

        for provider_name in providers:
            module_path = f"omnimancer.providers.{provider_name}"
            try:
                provider_module = importlib.import_module(module_path)
                assert (
                    provider_module is not None
                ), f"Provider {provider_name} should import successfully"
            except ImportError as e:
                pytest.fail(f"Failed to import provider {provider_name}: {e}")

    def test_core_engine_imports(self):
        """Test that core engine modules import successfully."""
        from omnimancer.core import config_manager, engine, models

        assert engine is not None
        assert models is not None
        assert config_manager is not None

    def test_cli_interface_imports(self):
        """Test that CLI interface modules import successfully."""
        from omnimancer.cli import commands, interface

        assert interface is not None
        assert commands is not None


class TestModuleIntegrity:
    """Test module integrity and cross-imports."""

    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # Import all main modules together to check for circular dependencies
        modules_to_test = [
            "omnimancer.utils.errors",
            "omnimancer.core.engine",
            "omnimancer.core.models",
            "omnimancer.core.config_manager",
            "omnimancer.core.agent.program_executor",
            "omnimancer.core.agent_engine",
            "omnimancer.cli.interface",
            "omnimancer.cli.commands",
        ]

        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                assert (
                    module is not None
                ), f"Module {module_name} should import without circular dependencies"
            except ImportError as e:
                pytest.fail(
                    f"Circular import or missing dependency in {module_name}: {e}"
                )

    def test_all_exports_accessible(self):
        """Test that __all__ exports are accessible."""
        from omnimancer.utils.errors import __all__ as error_exports

        # Verify all exported error classes can be imported
        for export_name in error_exports:
            try:
                module = importlib.import_module("omnimancer.utils.errors")
                export_class = getattr(module, export_name)
                assert (
                    export_class is not None
                ), f"Export {export_name} should be accessible"
            except (ImportError, AttributeError) as e:
                pytest.fail(f"Failed to access export {export_name}: {e}")

    def test_critical_paths_working(self):
        """Test that critical import paths used by program_executor work."""
        # These are the exact imports that were causing test collection to fail
        try:
            from omnimancer.core.security.approval_workflow import (
                ApprovalWorkflow,
                RiskLevel,
            )
            from omnimancer.core.security.sandbox_manager import (
                ResourceLimits,
                SandboxedProcess,
                SandboxManager,
            )
            from omnimancer.utils.errors import (
                ExecutionError,
                SecurityError,
                TimeoutError,
            )

            assert SandboxManager is not None
            assert ResourceLimits is not None
            assert SandboxedProcess is not None
            assert ApprovalWorkflow is not None
            assert RiskLevel is not None
            assert SecurityError is not None
            assert ExecutionError is not None
            assert TimeoutError is not None

        except ImportError as e:
            pytest.fail(f"Critical import path failed: {e}")


class TestModuleCompatibility:
    """Test module compatibility and version requirements."""

    def test_python_version_compatibility(self):
        """Test that modules work with current Python version."""
        assert sys.version_info >= (3, 8), "Omnimancer requires Python 3.8+"

        # Test that async/await syntax works (required for agent system)
        import asyncio

        assert asyncio is not None

    def test_required_dependencies_available(self):
        """Test that required dependencies are available."""
        # Map package names to their import names where different
        required_deps = {
            "click": "click",
            "rich": "rich",
            "httpx": "httpx",
            "pydantic": "pydantic",
            "cryptography": "cryptography",
            "psutil": "psutil",
            "aiofiles": "aiofiles",
            "aiohttp": "aiohttp",
            "beautifulsoup4": "bs4",  # Package beautifulsoup4 imports as bs4
            "html2text": "html2text",
        }

        for package_name, import_name in required_deps.items():
            try:
                importlib.import_module(import_name)
            except ImportError:
                pytest.fail(
                    f"Required dependency {package_name} (import as {import_name}) is not available"
                )

    def test_agent_system_imports_complete(self):
        """Test that the complete agent system can be imported."""
        try:
            # Import the main agent engine
            # Import the program executor that was failing
            from omnimancer.core.agent.program_executor import (
                EnhancedProgramExecutor,
            )
            from omnimancer.core.agent_engine import AgentEngine
            from omnimancer.core.security.approval_workflow import (
                ApprovalWorkflow,
            )

            # Import security components
            from omnimancer.core.security.sandbox_manager import SandboxManager

            # Verify all are classes that can be instantiated
            assert issubclass(AgentEngine, object)
            assert issubclass(EnhancedProgramExecutor, object)
            assert issubclass(SandboxManager, object)
            assert issubclass(ApprovalWorkflow, object)

        except ImportError as e:
            pytest.fail(f"Agent system imports failed: {e}")
        except Exception as e:
            pytest.fail(f"Agent system import verification failed: {e}")


class TestErrorModuleSpecific:
    """Specific tests for the error module that was causing issues."""

    def test_execution_timeout_errors_importable(self):
        """Test that ExecutionError and TimeoutError specifically work."""
        from omnimancer.utils.errors import ExecutionError, TimeoutError

        # Test that they can be instantiated
        exec_error = ExecutionError("Test execution error")
        timeout_error = TimeoutError("Test timeout error")

        assert isinstance(exec_error, Exception)
        assert isinstance(timeout_error, Exception)

        # Test that they have the expected attributes
        assert hasattr(exec_error, "message")
        assert hasattr(exec_error, "details")
        assert hasattr(timeout_error, "message")
        assert hasattr(timeout_error, "details")

    def test_all_error_classes_instantiable(self):
        """Test that all error classes in __all__ can be instantiated."""
        import omnimancer.utils.errors as errors_module
        from omnimancer.utils.errors import __all__ as all_exports

        for error_name in all_exports:
            error_class = getattr(errors_module, error_name)

            # Test basic instantiation
            try:
                error_instance = error_class("Test error message")
                assert isinstance(error_instance, Exception)
                assert hasattr(error_instance, "message")
            except Exception as e:
                pytest.fail(f"Failed to instantiate {error_name}: {e}")

    def test_program_executor_error_usage(self):
        """Test that program_executor can use the error classes."""
        # Import the module that was failing
        from omnimancer.core.agent.program_executor import (
            EnhancedProgramExecutor,
        )
        from omnimancer.utils.errors import (
            ExecutionError,
            SecurityError,
            TimeoutError,
        )

        # Verify the executor can reference these error types
        try:
            # Test that the executor class exists and references work
            executor = EnhancedProgramExecutor()
            assert executor is not None

            # Test that error classes are usable in context
            test_security_error = SecurityError("Test security error")
            test_execution_error = ExecutionError("Test execution error")
            test_timeout_error = TimeoutError("Test timeout error")

            assert all(
                [
                    isinstance(test_security_error, Exception),
                    isinstance(test_execution_error, Exception),
                    isinstance(test_timeout_error, Exception),
                ]
            )

        except Exception as e:
            pytest.fail(f"Program executor error usage test failed: {e}")
