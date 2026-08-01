"""Tests for external turn-completion notifications."""

import asyncio
import json
import sys
import time
import uuid
from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.commands import Command
from omnimancer.cli.interface import CommandLineInterface
from omnimancer.cli.turn_notify import TurnNotifier
from omnimancer.core.models import ChatResponse


def _response() -> ChatResponse:
    return ChatResponse(
        content="finished",
        model_used="test",
        tokens_used=17,
        input_tokens=11,
        output_tokens=6,
        cost_estimate=0.125,
    )


async def test_payload_schema_and_final_argv(tmp_path) -> None:
    recorder = tmp_path / "recorder.py"
    output = tmp_path / "payload.json"
    recorder.write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[1]).write_text(sys.argv[-1])\n"
    )
    notifier = TurnNotifier(f"{sys.executable} {recorder} {output}", cwd=str(tmp_path))
    response = _response()
    notifier.record_assistant(response.content, response)

    returned = await notifier.fire()
    payload = json.loads(output.read_text())

    assert payload == returned
    assert set(payload) == {
        "type",
        "turn-id",
        "last-assistant-message",
        "session_id",
        "usage",
        "cwd",
    }
    assert payload["type"] == "agent-turn-complete"
    uuid.UUID(payload["turn-id"])
    uuid.UUID(payload["session_id"])
    assert payload["last-assistant-message"] == "finished"
    assert payload["usage"] == {
        "input_tokens": 11,
        "output_tokens": 6,
        "total_cost_usd": 0.125,
    }
    assert payload["cwd"] == str(tmp_path)


async def test_turn_identifier_changes_between_fires(tmp_path) -> None:
    notifier = TurnNotifier(None, cwd=str(tmp_path))
    first = await notifier.fire()
    second = await notifier.fire()
    assert first["turn-id"] != second["turn-id"]
    assert first["session_id"] == second["session_id"]


async def test_unset_command_is_noop(monkeypatch, tmp_path) -> None:
    spawn = AsyncMock()
    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    notifier = TurnNotifier(None, cwd=str(tmp_path))
    payload = await notifier.fire()
    spawn.assert_not_awaited()
    assert payload["last-assistant-message"] is None


async def test_bogus_command_never_raises(tmp_path) -> None:
    notifier = TurnNotifier("/definitely/not/a/real/command", cwd=str(tmp_path))
    payload = await notifier.fire()
    assert payload["type"] == "agent-turn-complete"


async def test_timeout_kills_process(monkeypatch, tmp_path) -> None:
    child = tmp_path / "child.py"
    parent = tmp_path / "parent.py"
    descendant_completed = tmp_path / "descendant-completed"
    child.write_text(
        "import pathlib, sys, time\n"
        "time.sleep(0.3)\n"
        "pathlib.Path(sys.argv[1]).write_text('completed')\n"
    )
    parent.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
        "time.sleep(5)\n"
    )
    monkeypatch.setattr("omnimancer.cli.turn_notify._NOTIFY_TIMEOUT_SECONDS", 0.05)
    notifier = TurnNotifier(
        f"{sys.executable} {parent} {child} {descendant_completed}",
        cwd=str(tmp_path),
    )

    started = time.monotonic()
    await notifier.fire()
    await asyncio.sleep(0.5)

    assert time.monotonic() - started < 2
    assert not descendant_completed.exists()


def test_display_capture_records_latest_response(tmp_path) -> None:
    from omnimancer.cli.display import DisplayMixin

    host = type("DisplayHost", (DisplayMixin,), {})()
    host.turn_notifier = TurnNotifier(None, cwd=str(tmp_path))
    host.usage = None
    host.console = MagicMock()
    response = _response()
    host._show_token_status(response)
    payload = host.turn_notifier.build_payload()
    assert payload["last-assistant-message"] == response.content
    assert payload["usage"]["input_tokens"] == 11


async def test_headless_notifier_and_hook_share_payload(tmp_path) -> None:
    from omnimancer.cli.headless import HeadlessRunner, OutputFormat

    recorder = tmp_path / "headless_recorder.py"
    output = tmp_path / "headless_payload.json"
    recorder.write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[1]).write_text(sys.argv[-1])\n"
    )
    response = ChatResponse(
        content="work complete\nDONE",
        model_used="test",
        tokens_used=7,
        input_tokens=4,
        output_tokens=3,
        cost_estimate=0.02,
    )
    engine = MagicMock()
    engine.agent_engine = MagicMock()
    engine.config_manager.get_config.return_value.default_provider = "test"
    engine.provider_supports_tools.return_value = True
    engine.send_message_with_tools = AsyncMock(return_value=response)
    engine._fire_hook = AsyncMock()
    runner = HeadlessRunner(
        engine,
        output_format=OutputFormat.TEXT,
        notify_cmd=f"{sys.executable} {recorder} {output}",
    )
    runner._emitter._stdout = StringIO()

    assert await runner.run("do work") == 0

    hook_payload = engine._fire_hook.await_args.args[1]
    assert engine._fire_hook.await_args.args[0] == "turn_complete"
    assert hook_payload == json.loads(output.read_text())


@pytest.mark.parametrize(
    "effect, expected_exception",
    [
        (None, None),
        (RuntimeError("preprocessing failed"), RuntimeError),
        (asyncio.CancelledError(), asyncio.CancelledError),
    ],
)
async def test_interactive_turn_finalizes_once_for_every_outcome(
    effect, expected_exception
) -> None:
    cli = CommandLineInterface.__new__(CommandLineInterface)
    cli.engine = MagicMock()
    cli.turn_notifier = MagicMock()
    cli._turn_seq = 0
    cli._handle_chat_message_turn = AsyncMock(side_effect=effect)
    command = Command.create_chat_message("hello")

    with pytest.MonkeyPatch.context() as monkeypatch:
        complete = AsyncMock()
        monkeypatch.setattr("omnimancer.cli.interface.fire_turn_complete", complete)
        if expected_exception is None:
            await cli._handle_chat_message(command)
        else:
            with pytest.raises(expected_exception):
                await cli._handle_chat_message(command)

    cli.turn_notifier.reset_turn.assert_called_once_with()
    complete.assert_awaited_once_with(cli.engine, cli.turn_notifier)
