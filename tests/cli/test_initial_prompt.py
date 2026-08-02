"""Tests for interactive initial prompts and input selection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest

from omnimancer.cli.interface import CommandLineInterface, validate_prompt_options


async def test_initial_prompt_runs_once_before_input_loop() -> None:
    cli = CommandLineInterface.__new__(CommandLineInterface)
    cli.engine = MagicMock()
    cli.engine.runtime_identity.return_value = ("p", "test-model")
    cli.engine.initialize_providers = AsyncMock()
    cli.engine.initialize_mcp = AsyncMock()
    cli.engine.shutdown_mcp = AsyncMock()
    cli.engine.agent_engine = None
    cli.engine.config_manager = MagicMock()
    cli.signal_handler = MagicMock()
    cli.signal_handler.shutdown_event.is_set.return_value = True
    cli.signal_handler.shutdown_in_progress = False
    cli.signal_handler.setup_signal_handlers = MagicMock()
    cli._reset_terminal = MagicMock()
    cli._show_welcome = MagicMock()
    cli._show_goodbye = MagicMock()
    cli._show_error = MagicMock()
    cli._process_command = AsyncMock()
    cli.approval_integration = None
    cli.agent_manager = None
    cli.initial_prompt = "start here"
    cli.read_only = False
    cli.full_trust = False
    cli.turn_notifier = MagicMock(session_id="sess-initial-prompt")

    manager = MagicMock()
    manager.mode = SimpleNamespace(value="off")
    with patch("omnimancer.cli.interface.AgentModeManager", return_value=manager):
        await cli._async_start()

    cli._process_command.assert_awaited_once()
    command = cli._process_command.await_args.args[0]
    assert command.content == "start here"
    assert command.is_chat_message
    assert cli.initial_prompt is None


def test_initial_and_headless_prompts_are_mutually_exclusive() -> None:
    with pytest.raises(click.UsageError):
        validate_prompt_options("headless", "interactive")


def test_plain_input_environment_skips_prompt_toolkit(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OMNIMANCER_PLAIN_INPUT", "1")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(
        CommandLineInterface, "_setup_readline_history", lambda self: None
    )
    engine = MagicMock()
    engine.agent_engine = None
    engine.config_manager = MagicMock()

    cli = CommandLineInterface(engine)

    assert cli.prompt_input is None


def test_full_trust_configures_repl_agent_engine() -> None:
    cli = CommandLineInterface.__new__(CommandLineInterface)
    cli.engine = MagicMock()
    agent_engine = MagicMock()
    cli.engine.agent_engine = agent_engine
    cli.read_only = False
    cli.full_trust = True

    cli._configure_agent_engine_session()

    agent_engine.approval.set_approval_callback.assert_called_once()
    agent_engine.enhanced_approval.set_approval_callback.assert_called_once()
    agent_engine.executor.set_full_trust.assert_called_once_with(True)
    agent_engine.file_system.set_full_trust.assert_called_once_with(True)
