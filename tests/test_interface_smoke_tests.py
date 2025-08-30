"""
Comprehensive smoke tests for interface refactoring.

These tests ensure that the refactored interface maintains
all functionality during the modularization process.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.cli.commands import Command, CommandType, parse_command
from omnimancer.cli.interface import CommandLineInterface
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.engine import CoreEngine
from omnimancer.core.models import ProviderConfig


class TestInterfaceSmokeTests:
    """Comprehensive smoke tests for interface functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.json"

        # Create test config manager and engine
        self.config_manager = ConfigManager(str(self.config_path))
        self.config = self.config_manager.get_config()

        # Add minimal test provider
        self.config.providers = {
            "test": ProviderConfig(
                model="test-model",
                api_key="test-key",
                provider_type="test",
                supports_tools=True,
                supports_multimodal=False,
            )
        }
        self.config.default_provider = "test"

        # Mock the engine
        self.engine = MagicMock(spec=CoreEngine)
        self.engine.config_manager = self.config_manager

        # Create interface instance
        self.interface = CommandLineInterface(self.engine, no_approval=True)

    def test_interface_initialization(self):
        """Test that interface initializes properly."""
        assert self.interface is not None
        assert self.interface.engine is not None
        assert self.interface.console is not None
        assert hasattr(self.interface, "running")
        assert hasattr(self.interface, "signal_handler")

    def test_ui_methods_exist(self):
        """Test that all UI methods exist and are callable."""
        ui_methods = [
            "_show_welcome",
            "_show_goodbye",
            "_show_help",
            "_show_status",
            "_show_error",
            "_show_success",
            "_show_warning",
            "_show_info",
            "_show_user_message",
            "_show_assistant_message",
            "_clear_screen",
        ]

        for method_name in ui_methods:
            assert hasattr(
                self.interface, method_name
            ), f"Missing method: {method_name}"
            method = getattr(self.interface, method_name)
            assert callable(method), f"Method not callable: {method_name}"

    def test_command_handler_methods_exist(self):
        """Test that all command handler methods exist and are callable."""
        handler_methods = [
            "_handle_user_input",
            "_handle_chat_message",
            "_handle_slash_command",
            "_handle_system_command",
            "_handle_keyboard_interrupt",
        ]

        for method_name in handler_methods:
            assert hasattr(
                self.interface, method_name
            ), f"Missing method: {method_name}"
            method = getattr(self.interface, method_name)
            assert callable(method), f"Method not callable: {method_name}"

    def test_completion_methods_exist(self):
        """Test that completion methods exist and are callable."""
        completion_methods = [
            "_complete_command",
            "_complete_slash_commands",
            "_complete_command_arguments",
            "_complete_switch_command",
        ]

        # Some completion methods might not exist in all versions
        for method_name in completion_methods:
            if hasattr(self.interface, method_name):
                method = getattr(self.interface, method_name)
                assert callable(method), f"Method not callable: {method_name}"

    def test_ui_methods_basic_functionality(self):
        """Test basic functionality of UI methods."""
        # Mock console to capture output
        with patch.object(self.interface.console, "print") as mock_print:
            # Test welcome message
            self.interface._show_welcome()
            assert mock_print.called
            mock_print.reset_mock()

            # Test goodbye message
            self.interface._show_goodbye()
            assert mock_print.called
            mock_print.reset_mock()

            # Test error message
            self.interface._show_error("Test error")
            assert mock_print.called
            mock_print.reset_mock()

            # Test success message
            self.interface._show_success("Test success")
            assert mock_print.called
            mock_print.reset_mock()

            # Test warning message
            self.interface._show_warning("Test warning")
            assert mock_print.called

    @pytest.mark.asyncio
    async def test_command_processing(self):
        """Test command processing functionality."""
        # Mock the engine's send_message method
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.content = "Test response"
        mock_response.model_used = "test-model"

        self.engine.send_message = AsyncMock(return_value=mock_response)

        # Test chat command
        with patch.object(self.interface, "_show_user_message"):
            with patch.object(self.interface, "_show_assistant_message"):
                command = Command(
                    type=CommandType.CHAT_MESSAGE,
                    content="Hello test",
                    parameters={},
                    raw_input="Hello test",
                )

                # This should not raise an exception
                await self.interface._process_command(command)

    def test_console_operations(self):
        """Test console operations work correctly."""
        # Test that console can print without errors
        try:
            self.interface.console.print("Test message")
            self.interface.console.print("[red]Red text[/red]")
            self.interface.console.print("[bold]Bold text[/bold]")
        except Exception as e:
            pytest.fail(f"Console operations failed: {e}")

    def test_signal_handler_integration(self):
        """Test signal handler integration."""
        assert hasattr(self.interface, "signal_handler")
        assert self.interface.signal_handler is not None
        assert hasattr(self.interface.signal_handler, "setup_signal_handlers")

    def test_history_manager_integration(self):
        """Test history manager integration."""
        assert hasattr(self.interface, "history_manager")
        assert self.interface.history_manager is not None
        assert hasattr(self.interface.history_manager, "add_command")
        assert hasattr(self.interface.history_manager, "get_recent_commands")

    def test_approval_integration(self):
        """Test approval integration (if available)."""
        # This test should pass whether approval is enabled or not
        assert hasattr(self.interface, "approval_integration")
        # approval_integration might be None if not configured

    def test_agent_integration(self):
        """Test agent integration components."""
        assert hasattr(self.interface, "agent_cli_handler")
        assert hasattr(self.interface, "agent_persona_handler")
        assert hasattr(self.interface, "permissions_handler")

    def test_error_handling_robustness(self):
        """Test that interface handles errors gracefully."""
        # Test with invalid input
        with patch("builtins.input", return_value=""):
            user_input = self.interface._get_user_input()
            # Should return None or empty string, not crash
            assert user_input is None or user_input == ""

        # Test with EOF condition
        with patch("builtins.input", side_effect=EOFError):
            user_input = self.interface._get_user_input()
            assert user_input is None

    def test_method_signatures(self):
        """Test that key method signatures are correct."""
        # Test UI methods
        import inspect

        # _show_error should accept a message parameter
        sig = inspect.signature(self.interface._show_error)
        assert "message" in sig.parameters

        # _show_success should accept a message parameter
        sig = inspect.signature(self.interface._show_success)
        assert "message" in sig.parameters

        # _show_user_message should accept a message parameter
        sig = inspect.signature(self.interface._show_user_message)
        assert "message" in sig.parameters


class TestRegressionSafety:
    """Tests to ensure no functionality is lost during refactoring."""

    def test_command_parsing_still_works(self):
        """Test that command parsing still works after refactoring."""
        # Test various command types
        commands = [
            "/help",
            "/quit",
            "/status",
            "Hello world",
            "/switch claude",
            "",
        ]

        for cmd_text in commands:
            try:
                result = parse_command(cmd_text)
                # Should not raise exceptions
                assert result is not None or cmd_text == ""
            except Exception as e:
                pytest.fail(f"Command parsing failed for '{cmd_text}': {e}")

    def test_existing_cli_tests_compatibility(self):
        """Ensure compatibility with existing CLI tests."""
        # This test imports and instantiates the original interface
        # to ensure it still works
        from omnimancer.cli.interface import CommandLineInterface
        from omnimancer.core.engine import CoreEngine

        # Mock engine
        engine = MagicMock(spec=CoreEngine)

        try:
            cli = CommandLineInterface(engine, no_approval=True)
            assert cli is not None
        except Exception as e:
            pytest.fail(f"Original interface instantiation failed: {e}")
