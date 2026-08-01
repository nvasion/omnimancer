"""FleetApp data-wiring Pilot tests (WU-B5).

Seeds real files into tmp jobs/events dirs and drives the app with
Textual's Pilot: rows derive the right states, feeds populate from the
JSONL/ledger tails, live appends arrive, and row selection opens the
detail modal.
"""

import json

import pytest

pytest.importorskip("textual")

from omnimancer.tui.fleet.app import FleetApp, JobDetailScreen  # noqa: E402

JOB = {
    "id": "aabbccdd",
    "backend": "codex",
    "status": "running",
    "turnState": "idle",
    "processState": "running",
    "model": "gpt-5.6-sol",
    "turnsCompleted": 2,
    "cwd": "/tmp/elsewhere",
}

EVENT_SESSION_START = {
    "v": 1,
    "ts": "2026-08-01T03:14:34.686+00:00",
    "event": "session_start",
    "session_id": "11112222-3333",
    "agent_id": "main",
    "mode": "interactive",
    "cwd": "/tmp/proj",
    "data": {"provider": "gateway", "model": "qwen3-coder-30b"},
}

EVENT_TOOL_START = {
    "v": 1,
    "ts": "2026-08-01T03:14:35.000+00:00",
    "event": "tool_start",
    "session_id": "11112222-3333",
    "agent_id": "main",
    "mode": "interactive",
    "cwd": "/tmp/proj",
    "data": {"tool": "Bash", "target": "pytest -q"},
}


@pytest.fixture
def fleet_dirs(tmp_path):
    jobs = tmp_path / "jobs"
    events = tmp_path / "events"
    project = tmp_path / "proj"
    jobs.mkdir()
    events.mkdir()
    project.mkdir()
    (jobs / "aabbccdd.json").write_text(json.dumps(JOB))
    (jobs / "aabbccdd.turn-complete").write_text("{}")
    (events / "sess.jsonl").write_text(
        json.dumps(EVENT_SESSION_START) + "\n" + json.dumps(EVENT_TOOL_START) + "\n"
    )
    (project / "agents.log").write_text(
        "## Session: 2026-08-01 03:10\n"
        "### Spawned: aabbccdd - 03:11\n"
        "- VERDICT: PASS\n"
    )
    return jobs, events, project


def _app(fleet_dirs) -> FleetApp:
    jobs, events, project = fleet_dirs
    return FleetApp(jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05)


def _log_text(app, selector: str) -> str:
    from textual.widgets import RichLog

    log = app.query_one(selector, RichLog)
    return "\n".join(strip.text for strip in log.lines)


async def _settle(pilot, seconds: float = 0.4) -> None:
    await pilot.pause(seconds)
    await pilot.pause()


class TestFleetAppData:
    async def test_job_row_waiting_and_session_row(self, fleet_dirs):
        app = _app(fleet_dirs)
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 2
            first = [cell.plain for cell in table.get_row_at(0)]
            # idle + .turn-complete signal -> WAITING (codex stays "running")
            assert first[0] == "aabbccdd"
            assert first[2] == "waiting"
            second = [cell.plain for cell in table.get_row_at(1)]
            assert second[0] == "11112222"
            assert second[1] == "omn:interactive"
            assert second[3] == "qwen3-coder-30b"

    async def test_feeds_populate(self, fleet_dirs):
        app = _app(fleet_dirs)
        async with app.run_test() as pilot:
            await _settle(pilot)
            activity = _log_text(app, "#activity")
            comms = _log_text(app, "#comms")
            assert "Bash" in activity
            assert "pytest -q" in activity
            assert "VERDICT: PASS" in comms
            assert "spawned aabbccdd" in comms

    async def test_live_append_reaches_activity(self, fleet_dirs):
        jobs, events, project = fleet_dirs
        app = _app(fleet_dirs)
        async with app.run_test() as pilot:
            await _settle(pilot)
            before = _log_text(app, "#activity")
            new_event = dict(EVENT_TOOL_START)
            new_event["event"] = "tool_end"
            new_event["data"] = {"tool": "Bash", "success": True}
            with open(events / "sess.jsonl", "a") as handle:
                handle.write(json.dumps(new_event) + "\n")
            await _settle(pilot)
            after = _log_text(app, "#activity")
            assert len(after) > len(before)
            assert "✓ Bash" in after

    async def test_session_freshness_uses_event_ts_not_ingestion(self, tmp_path):
        """A dashboard started after a session died must show it STALE:
        freshness comes from the events' own timestamps, never from when
        the dashboard happened to read them."""
        from datetime import datetime, timezone

        jobs = tmp_path / "jobs"
        events = tmp_path / "events"
        project = tmp_path / "proj"
        for directory in (jobs, events, project):
            directory.mkdir()

        def _session_event(session_id: str, ts: str) -> str:
            event = dict(EVENT_SESSION_START)
            event["session_id"] = session_id
            event["ts"] = ts
            return json.dumps(event)

        fresh_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        (events / "fresh.jsonl").write_text(
            _session_event("fresh-sess-1111", fresh_ts) + "\n"
        )
        # The fixture timestamp is hours in the past relative to any test run.
        (events / "dead.jsonl").write_text(
            _session_event("dead-sess-2222", "2026-08-01T03:14:34.686+00:00") + "\n"
        )

        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            states = {}
            for row_index in range(table.row_count):
                cells = [cell.plain for cell in table.get_row_at(row_index)]
                states[cells[0]] = cells[2]
            assert states["fresh-se"] == "working"
            assert states["dead-ses"] == "stale"

    async def test_once_waits_for_event_replay(self, tmp_path):
        """--once must include session rows from the initial event replay:
        exiting after the jobs scan alone would drop them entirely."""
        jobs = tmp_path / "jobs"
        events = tmp_path / "events"
        project = tmp_path / "proj"
        for directory in (jobs, events, project):
            directory.mkdir()
        event = dict(EVENT_SESSION_START)
        event["session_id"] = "dead-sess-3333"
        event["ts"] = "2026-08-01T03:14:34.686+00:00"
        (events / "dead.jsonl").write_text(json.dumps(event) + "\n")

        app = FleetApp(
            jobs_dir=jobs,
            events_dir=events,
            project_dir=project,
            refresh=0.05,
            once=True,
        )
        # The app unmounts its widgets on exit, so snapshot the table at
        # the exact moment --once decides to leave.
        from textual.widgets import DataTable

        rows_at_exit = []
        original_exit = app.exit

        def capturing_exit(*args, **kwargs):
            table = app.query_one("#agents", DataTable)
            for row_index in range(table.row_count):
                rows_at_exit.append(
                    [cell.plain for cell in table.get_row_at(row_index)]
                )
            original_exit(*args, **kwargs)

        app.exit = capturing_exit  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await _settle(pilot, 0.6)

        assert len(rows_at_exit) == 1
        assert rows_at_exit[0][0] == "dead-ses"
        assert rows_at_exit[0][2] == "stale"

    async def test_claude_session_row_and_waiting_semantics(self, tmp_path):
        """A Claude Code session (from the hook adapter's events) renders
        with backend 'claude', its model, and flips to WAITING once its
        latest event is turn_end (idle at prompt)."""
        from datetime import datetime, timezone

        jobs = tmp_path / "jobs"
        events = tmp_path / "events"
        project = tmp_path / "proj"
        for directory in (jobs, events, project):
            directory.mkdir()

        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        claude_events = [
            {
                "v": 1,
                "ts": now_ts,
                "event": "session_start",
                "session_id": "cccc1111-2222",
                "agent_id": "main",
                "mode": "claude",
                "cwd": "/tmp/claudeproj",
                "seq": -1,
                "data": {"harness": "claude", "model": "claude-fable-5"},
            },
            {
                "v": 1,
                "ts": now_ts,
                "event": "tool_start",
                "session_id": "cccc1111-2222",
                "agent_id": "main",
                "mode": "claude",
                "cwd": "/tmp/claudeproj",
                "seq": -1,
                "data": {"tool": "Bash", "target": "ls", "invocation": "claude"},
            },
        ]
        (events / "omn-cccc1111-2222.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in claude_events)
        )

        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            cells = [cell.plain for cell in table.get_row_at(0)]
            assert cells[0] == "cccc1111"
            assert cells[1] == "claude"
            assert cells[2] == "working"
            assert cells[3] == "claude-fable-5"

            # Stop event arrives -> idle at prompt -> waiting.
            stop_event = dict(claude_events[0])
            stop_event["event"] = "turn_end"
            stop_event["data"] = {"last_message_preview": "done"}
            with open(events / "omn-cccc1111-2222.jsonl", "a") as handle:
                handle.write(json.dumps(stop_event) + "\n")
            await _settle(pilot)
            cells = [cell.plain for cell in table.get_row_at(0)]
            assert cells[2] == "waiting"

    async def test_once_deadline_exits_nonzero_on_wedged_source(self, tmp_path):
        """If an initial poll never lands, --once must exit non-zero with a
        warning — a partial snapshot must never masquerade as complete."""
        for name in ("jobs", "events", "proj"):
            (tmp_path / name).mkdir()
        app = FleetApp(
            jobs_dir=tmp_path / "jobs",
            events_dir=tmp_path / "events",
            project_dir=tmp_path / "proj",
            refresh=0.05,
            once=True,
            once_fallback_s=0.3,
        )
        # Wedge the jobs source: its snapshot message never arrives.
        app._scan_jobs = lambda: None  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await _settle(pilot, 0.8)
        assert app.return_code == 1

    async def test_row_selection_opens_detail_modal(self, fleet_dirs):
        app = _app(fleet_dirs)
        async with app.run_test() as pilot:
            await _settle(pilot)
            await pilot.press("enter")
            await _settle(pilot)
            assert isinstance(app.screen, JobDetailScreen)
            assert app.screen.job_id == "aabbccdd"

    async def test_sort_filter_and_cursor_stability(self, tmp_path):
        """Operator console behaviors: active work sorts to the top, `f`
        cycles htop-style state filters, and the 1s rescan never yanks the
        cursor back to the top (field-reported)."""
        jobs = tmp_path / "jobs"
        events = tmp_path / "events"
        project = tmp_path / "proj"
        for directory in (jobs, events, project):
            directory.mkdir()
        (jobs / "11111111.json").write_text(
            json.dumps({"id": "11111111", "backend": "codex", "status": "completed"})
        )
        (jobs / "22222222.json").write_text(
            json.dumps(
                {
                    "id": "22222222",
                    "backend": "codex",
                    "status": "running",
                    "turnState": "working",
                }
            )
        )
        (jobs / "aabbccdd.json").write_text(
            json.dumps(
                {
                    "id": "aabbccdd",
                    "backend": "codex",
                    "status": "running",
                    "turnState": "idle",
                }
            )
        )
        (jobs / "aabbccdd.turn-complete").write_text("{}")

        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            order = [table.get_row_at(i)[0].plain for i in range(table.row_count)]
            # working first, waiting next, completed last
            assert order == ["22222222", "aabbccdd", "11111111"]

            # Cursor stays put across rescans (several ticks pass here).
            table.move_cursor(row=2)
            key_before = app._cursor_row_key(table)
            assert key_before == "11111111"
            await _settle(pilot)
            assert app._cursor_row_key(table) == key_before

            # f cycles: all -> active -> attention -> done -> all
            await pilot.press("f")
            await pilot.pause()
            assert table.row_count == 1  # active: the working job
            assert "active" in str(table.border_title)
            await pilot.press("f")
            await pilot.pause()
            assert table.row_count == 1  # attention: the waiting job
            await pilot.press("f")
            await pilot.pause()
            assert table.row_count == 1  # done: the completed job
            await pilot.press("f")
            await pilot.pause()
            assert table.row_count == 3
            assert "all" in str(table.border_title)

    async def test_detail_modal_survives_ansi_log_tail(self, fleet_dirs):
        """Field-reported crash: tmux .log files carry raw ANSI escapes and
        bracket junk that exploded Textual's markup parser in the modal."""
        jobs, _events, _project = fleet_dirs
        (jobs / "aabbccdd.log").write_bytes(
            b"\x1b[31mred line\x1b[39m\x1b[49m\x1b[0m\n"
            b"[not markup] [bold nonsense [\n"
        )
        app = _app(fleet_dirs)
        async with app.run_test() as pilot:
            await _settle(pilot)
            await pilot.press("enter")
            await _settle(pilot)
            assert isinstance(app.screen, JobDetailScreen)
            assert "red line" in app.screen.log_tail
            # And it must survive being rendered + dismissed.
            await pilot.press("escape")
            await _settle(pilot)
            assert not isinstance(app.screen, JobDetailScreen)
