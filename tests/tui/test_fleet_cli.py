"""omn fleet CLI mount + app skeleton tests (WU-B3).

Pins: the missing-textual install hint, the argv pre-dispatch (which must
not disturb the main `omn` command surface), and a Pilot smoke test of the
FleetApp shell.
"""

import sys

import pytest
from click.testing import CliRunner

from omnimancer.tui.fleet.cli import INSTALL_HINT, fleet_main


class TestHooksSnippet:
    def test_hooks_flag_prints_valid_settings_block(self):
        import json as json_module

        result = CliRunner().invoke(fleet_main, ["--hooks"])
        assert result.exit_code == 0
        block = json_module.loads(result.output)
        hooks = block["hooks"]
        assert set(hooks) == {
            "SessionStart",
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "Stop",
            "SessionEnd",
        }
        for event_name, entries in hooks.items():
            (entry,) = entries
            (hook,) = entry["hooks"]
            assert hook["command"] == "omn-fleet-hook"
            assert hook["async"] is True
            if event_name in {"PreToolUse", "PostToolUse", "PostToolUseFailure"}:
                assert entry["matcher"] == ".*"

    def test_hooks_flag_needs_no_textual(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "textual", None)
        result = CliRunner().invoke(fleet_main, ["--hooks"])
        assert result.exit_code == 0
        assert "omn-fleet-hook" in result.output

    def test_negative_budget_rejected(self):
        result = CliRunner().invoke(fleet_main, ["--budget-gb", "-1"])
        assert result.exit_code != 0
        assert "not in the range" in result.output or "Invalid" in result.output


class TestTextualGuard:
    def test_missing_textual_prints_install_hint(self, monkeypatch):
        # sys.modules[name] = None makes `import textual` raise ImportError.
        monkeypatch.setitem(sys.modules, "textual", None)
        result = CliRunner().invoke(fleet_main, [])
        assert result.exit_code != 0
        assert "omnimancer-cli[tui]" in result.output

    def test_hint_names_the_command(self):
        assert "omn fleet" in INSTALL_HINT


class TestArgvPredispatch:
    def test_fleet_help_dispatches(self, monkeypatch, capsys):
        from omnimancer.cli.interface import main

        monkeypatch.setattr(sys, "argv", ["omn", "fleet", "--help"])
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "Full-screen live dashboard" in out
        assert "--jobs-dir" in out

    def test_fleet_help_does_not_show_main_flags(self, monkeypatch, capsys):
        from omnimancer.cli.interface import main

        monkeypatch.setattr(sys, "argv", ["omn", "fleet", "--help"])
        with pytest.raises(SystemExit):
            main()
        out = capsys.readouterr().out
        # The main omn surface (e.g. --print/-p) must not bleed into fleet.
        assert "--print" not in out


class TestFleetAppSmoke:
    @pytest.fixture
    def app(self, tmp_path):
        pytest.importorskip("textual")
        from omnimancer.tui.fleet.app import FleetApp

        return FleetApp(
            jobs_dir=tmp_path / "jobs",
            events_dir=tmp_path / "events",
            project_dir=tmp_path,
        )

    async def test_shell_mounts_all_panels(self, app):
        from textual.widgets import DataTable, RichLog

        async with app.run_test() as pilot:
            table = app.query_one("#agents", DataTable)
            assert [str(col.label) for col in table.columns.values()] == [
                "id",
                "backend",
                "state",
                "model",
                "turns",
                "blocker",
                "usage",
                "age",
            ]
            assert app.query_one("#activity", RichLog) is not None
            assert app.query_one("#comms", RichLog) is not None
            await pilot.press("p")
            assert app.paused is True
            await pilot.press("p")
            assert app.paused is False

    async def test_once_mode_exits(self, tmp_path):
        pytest.importorskip("textual")
        from omnimancer.tui.fleet.app import FleetApp

        app = FleetApp(
            jobs_dir=tmp_path,
            events_dir=tmp_path,
            project_dir=tmp_path,
            once=True,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
        assert app.is_running is False
