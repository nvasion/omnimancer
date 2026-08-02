"""FleetApp data-wiring Pilot tests (WU-B5).

Seeds real files into tmp jobs/events dirs and drives the app with
Textual's Pilot: rows derive the right states, feeds populate from the
JSONL/ledger tails, live appends arrive, and row selection opens the
detail modal.
"""

import json
import os
import time

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


def _overflow_jobs(tmp_path):
    """Seed 1 WORKING + 14 COMPLETED codex jobs with mtimes pinned 2h old.

    Fifteen rows overflow the 40%-height agents table at the Pilot
    default 80x24 size, and the stale mtimes keep the age column stable
    so the skip-rebuild guard is deterministic between content changes.
    """
    jobs = tmp_path / "jobs"
    events = tmp_path / "events"
    project = tmp_path / "proj"
    for directory in (jobs, events, project):
        directory.mkdir()
    old = time.time() - 7200
    working = jobs / "00working.json"
    working.write_text(
        json.dumps(
            {
                "id": "00working",
                "backend": "codex",
                "status": "running",
                "turnState": "working",
            }
        )
    )
    os.utime(working, (old, old))
    for n in range(14):
        path = jobs / f"c{n:07d}.json"
        path.write_text(
            json.dumps({"id": f"c{n:07d}", "backend": "codex", "status": "completed"})
        )
        os.utime(path, (old, old))
    return jobs, events, project


async def _wait_for(pilot, condition, timeout_s: float = 5.0) -> None:
    """Settle until condition() is truthy or the deadline passes.

    Fixed sleeps starve under CI load; polling keeps the stability tests
    honest without slowing the happy path. The caller asserts afterwards,
    so a timeout still fails with the caller's own message.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if condition():
            return
        await _settle(pilot, 0.1)


def _fleet_tmp_dirs(tmp_path):
    """Create the standard empty jobs/events/project directory triple."""
    jobs = tmp_path / "jobs"
    events = tmp_path / "events"
    project = tmp_path / "proj"
    for directory in (jobs, events, project):
        directory.mkdir()
    return jobs, events, project


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
            assert first[4] == "-"
            second = [cell.plain for cell in table.get_row_at(1)]
            assert second[0] == "11112222"
            assert second[1] == "omn:interactive"
            assert second[3] == "qwen3-coder-30b"
            assert second[4] == "gateway"

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

        jobs, events, project = _fleet_tmp_dirs(tmp_path)

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
        jobs, events, project = _fleet_tmp_dirs(tmp_path)
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

        jobs, events, project = _fleet_tmp_dirs(tmp_path)

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

    async def test_resumed_session_reappears(self, tmp_path):
        """claude --resume reuses the session id: session_end then a fresh
        session_start must bring the row back (ended cleared)."""
        from datetime import datetime, timezone

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        def event(name: str, data: dict) -> str:
            return json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": name,
                    "session_id": "dddd4444-5555",
                    "agent_id": "main",
                    "mode": "claude",
                    "cwd": "/tmp/x",
                    "seq": -1,
                    "data": data,
                }
            )

        (events / "s.jsonl").write_text(
            event("session_start", {"model": "claude-fable-5"})
            + "\n"
            + event("session_end", {"reason": "logout"})
            + "\n"
            + event("session_start", {"model": "claude-fable-5"})
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 1
            cells = [cell.plain for cell in table.get_row_at(0)]
            assert cells[0] == "dddd4444"
            assert cells[1] == "claude"

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
        jobs, events, project = _fleet_tmp_dirs(tmp_path)
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


class TestEndedSessionVisibility:
    """P5 — ended sessions render with correct state and respect retention."""

    async def test_ended_session_renders_completed_with_model(self, tmp_path):
        """An ended session with exit status 0 renders as completed,
        showing the model from session_start."""
        from datetime import datetime, timezone

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        (events / "sess.jsonl").write_text(
            json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "eeee1111-2222",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3.5-9b"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_end",
                    "session_id": "eeee1111-2222",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"reason": "exit", "status": 0},
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 1
            cells = [cell.plain for cell in table.get_row_at(0)]
            assert cells[2] == "completed"
            assert cells[3] == "qwen3.5-9b"

    async def test_ended_session_nonzero_status_renders_failed(self, tmp_path):
        """An ended session with a non-zero exit status renders as failed."""
        from datetime import datetime, timezone

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        (events / "sess.jsonl").write_text(
            json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "ffff3333-4444",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3.5-9b"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_end",
                    "session_id": "ffff3333-4444",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"reason": "exit", "status": 3},
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 1
            cells = [cell.plain for cell in table.get_row_at(0)]
            assert cells[2] == "failed"

    async def test_ended_session_older_than_retention_hidden(self, tmp_path):
        """A session whose last event is >24h old must not render at all."""
        from datetime import datetime, timezone

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        old_ts = (
            datetime.now(timezone.utc) - __import__("datetime").timedelta(hours=25)
        ).isoformat(timespec="milliseconds")
        (events / "sess.jsonl").write_text(
            json.dumps(
                {
                    "v": 1,
                    "ts": old_ts,
                    "event": "session_start",
                    "session_id": "aaaa5555-6666",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3.5-9b"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "v": 1,
                    "ts": old_ts,
                    "event": "session_end",
                    "session_id": "aaaa5555-6666",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"reason": "exit", "status": 0},
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 0


class TestJobSessionDedup:
    """P5 — job/session dedup only suppresses where actually deduplicating."""

    async def test_live_session_not_hidden_by_dead_job(self, tmp_path):
        """A terminal job and a live session in the same cwd must both render;
        the dedup only applies when the job is terminal AND the session is ended."""
        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        (jobs / "11111111.json").write_text(
            json.dumps(
                {
                    "id": "11111111",
                    "backend": "codex",
                    "status": "completed",
                    "cwd": "/tmp/proj",
                }
            )
        )
        from datetime import datetime, timezone

        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        (events / "sess.jsonl").write_text(
            json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "bbbb7777-8888",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3.5-9b"},
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 2
            keys = {table.get_row_at(i)[0].plain for i in range(table.row_count)}
            assert "11111111" in keys
            assert "bbbb7777" in keys

    async def test_live_session_hidden_by_running_job_same_cwd(self, tmp_path):
        """A live session in the same cwd as a running job must be hidden;
        the job row alone represents the run."""
        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        (jobs / "22222222.json").write_text(
            json.dumps(
                {
                    "id": "22222222",
                    "backend": "codex",
                    "status": "running",
                    "turnState": "working",
                    "cwd": "/tmp/proj",
                }
            )
        )
        from datetime import datetime, timezone

        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        (events / "sess.jsonl").write_text(
            json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "cccc9999-0000",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3.5-9b"},
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 1
            assert table.get_row_at(0)[0].plain == "22222222"

    async def test_ended_session_deduped_by_fresh_terminal_job(self, tmp_path):
        """A terminal job whose file mtime is now covers an ended session in
        the same cwd — the pair is one run seen through two data sources."""
        import os
        import time

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        # Write a terminal job; mtime will be set to now.
        job_data = {
            "id": "33333333",
            "backend": "codex",
            "status": "completed",
            "cwd": "/tmp/proj",
        }
        (jobs / "33333333.json").write_text(json.dumps(job_data))
        # Pin mtime to "now" so it covers the session.
        now = time.time()
        os.utime(jobs / "33333333.json", (now, now))
        from datetime import datetime, timezone

        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        (events / "sess.jsonl").write_text(
            json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "dddd1111-2222",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3.5-9b"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_end",
                    "session_id": "dddd1111-2222",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"reason": "exit", "status": 0},
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 1
            assert table.get_row_at(0)[0].plain == "33333333"

    async def test_ended_session_not_deduped_by_much_newer_job(self, tmp_path):
        """A terminal job written hours AFTER the session ended is a
        different run: the ended session must still render (the cover
        window is bounded on both sides)."""
        import os
        import time
        from datetime import datetime, timedelta, timezone

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        (jobs / "44444444.json").write_text(
            json.dumps(
                {
                    "id": "44444444",
                    "backend": "codex",
                    "status": "completed",
                    "cwd": "/tmp/proj",
                }
            )
        )
        # Job file written "now"; the session ended two hours earlier.
        now = time.time()
        os.utime(jobs / "44444444.json", (now, now))
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(
            timespec="milliseconds"
        )
        (events / "sess.jsonl").write_text(
            json.dumps(
                {
                    "v": 1,
                    "ts": old_ts,
                    "event": "session_start",
                    "session_id": "hhhh1111-2222",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3.5-9b"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "v": 1,
                    "ts": old_ts,
                    "event": "session_end",
                    "session_id": "hhhh1111-2222",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"reason": "exit", "status": 0},
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 2
            keys = {table.get_row_at(i)[0].plain for i in range(table.row_count)}
            assert "44444444" in keys
            assert "hhhh1111" in keys

    async def test_subagent_row_renders_without_clobbering_parent(self, tmp_path):
        """A subagent session_start must create its own row and NOT
        overwrite the parent main row's model."""
        from datetime import datetime, timezone

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        (events / "sess.jsonl").write_text(
            # Main session_start — no agent_id (or "main") matches real fixtures.
            json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "eeee5555-6666",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3-coder-30b"},
                }
            )
            + "\n"
            # Subagent session_start.
            + json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "eeee5555-6666",
                    "agent_id": "subagent-researcher-ab12cd34",
                    "parent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {
                        "provider": "gateway",
                        "model": "qwen3-4b",
                        "subagent": "researcher",
                    },
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 2
            # Gather rows by id column.
            rows = {}
            for i in range(table.row_count):
                cells = [cell.plain for cell in table.get_row_at(i)]
                rows[cells[0]] = cells
            # Main row: model must STILL be qwen3-coder-30b (anti-clobber).
            assert "eeee5555" in rows
            assert rows["eeee5555"][3] == "qwen3-coder-30b"
            # Subagent row: id starts with ↳researcher, backend "sub", model "qwen3-4b".
            sub_rows = [v for v in rows.values() if v[0].startswith("↳")]
            assert len(sub_rows) == 1
            sr = sub_rows[0]
            assert sr[0].startswith("↳researcher")
            assert sr[1] == "sub"
            assert sr[3] == "qwen3-4b"
            assert sr[4] == "gateway"

    async def test_subagent_end_flips_subagent_row_only(self, tmp_path):
        """A subagent session_end must flip the subagent row to completed
        while the main row stays non-ended."""
        from datetime import datetime, timezone

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        (events / "sess.jsonl").write_text(
            json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "ffff7777-8888",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3-coder-30b"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "ffff7777-8888",
                    "agent_id": "subagent-researcher-abcdef01",
                    "parent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {
                        "provider": "gateway",
                        "model": "qwen3-4b",
                        "subagent": "researcher",
                    },
                }
            )
            + "\n"
            + json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_end",
                    "session_id": "ffff7777-8888",
                    "agent_id": "subagent-researcher-abcdef01",
                    "parent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"reason": "subagent_complete"},
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 2
            states = {}
            for i in range(table.row_count):
                cells = [cell.plain for cell in table.get_row_at(i)]
                states[cells[0]] = cells[2]
            # Main row stays non-ended (WORKING).
            assert states["ffff7777"] == "working"
            # Subagent row is completed.
            sub_states = [v for k, v in states.items() if k.startswith("↳")]
            assert len(sub_states) == 1
            assert sub_states[0] == "completed"

    async def test_subagent_activity_keeps_parent_fresh(self, tmp_path):
        """A main session with an old-ish timestamp alone would render STALE,
        but a subagent event with a fresh timestamp must keep the parent
        row from being stale."""
        from datetime import datetime, timedelta, timezone

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        # 10 minutes ago for the main session.
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(
            timespec="milliseconds"
        )
        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        (events / "sess.jsonl").write_text(
            json.dumps(
                {
                    "v": 1,
                    "ts": old_ts,
                    "event": "session_start",
                    "session_id": "gggg9999-0000",
                    "agent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {"provider": "gateway", "model": "qwen3-coder-30b"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "v": 1,
                    "ts": now_ts,
                    "event": "session_start",
                    "session_id": "gggg9999-0000",
                    "agent_id": "subagent-researcher-11223344",
                    "parent_id": "main",
                    "mode": "interactive",
                    "cwd": "/tmp/proj",
                    "data": {
                        "provider": "gateway",
                        "model": "qwen3-4b",
                        "subagent": "researcher",
                    },
                }
            )
            + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            states = {}
            for i in range(table.row_count):
                cells = [cell.plain for cell in table.get_row_at(i)]
                states[cells[0]] = cells[2]
            # Main row must NOT be stale.
            assert states["gggg9999"] != "stale"
            assert states["gggg9999"] == "working"


class TestCursorScrollStability:
    """The 1s rescan must never move the operator's cursor or viewport."""

    async def test_scroll_and_cursor_survive_rebuild(self, tmp_path):
        jobs, events, project = _overflow_jobs(tmp_path)
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 15
            table.move_cursor(row=table.row_count - 1)
            await _settle(pilot)
            assert app._cursor_row_key(table) == "c0000013"
            scroll_before = table.scroll_y
            assert scroll_before > 0

            # Idle ticks: the skip-guard path must leave everything alone.
            await _settle(pilot, 0.8)
            assert app._cursor_row_key(table) == "c0000013"
            assert table.scroll_y == scroll_before

            # Content change: the full-rebuild path must restore both.
            state_before = app._last_render_state
            (jobs / "00working.json").write_text(
                json.dumps(
                    {
                        "id": "00working",
                        "backend": "codex",
                        "status": "running",
                        "turnState": "working",
                        "turnsCompleted": 5,
                    }
                )
            )
            await _wait_for(pilot, lambda: app._last_render_state is not state_before)
            assert app._last_render_state is not state_before
            assert app._cursor_row_key(table) == "c0000013"
            assert table.scroll_y == scroll_before

    async def test_skip_rebuild_leaves_table_untouched(self, tmp_path):
        jobs, events, project = _overflow_jobs(tmp_path)
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            assert table.row_count == 15

            calls = []
            original_clear = table.clear

            def counting_clear(*args, **kwargs):
                calls.append(1)
                return original_clear(*args, **kwargs)

            table.clear = counting_clear  # type: ignore[method-assign]

            # Idle fleet: stable content must skip the rebuild entirely.
            await _settle(pilot, 0.8)
            assert not calls

            # A real content change still rebuilds.
            (jobs / "00working.json").write_text(
                json.dumps(
                    {
                        "id": "00working",
                        "backend": "codex",
                        "status": "running",
                        "turnState": "working",
                        "turnsCompleted": 9,
                    }
                )
            )
            await _wait_for(pilot, lambda: bool(calls))
            assert calls
            assert "all (15/15)" in str(table.border_title)

    async def test_vanished_row_lands_on_neighbor_with_scroll(self, tmp_path):
        jobs, events, project = _overflow_jobs(tmp_path)
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            table.move_cursor(row=table.row_count - 1)
            await _settle(pilot)
            scroll_before = table.scroll_y
            assert scroll_before > 0

            (jobs / "c0000013.json").unlink()
            await _wait_for(pilot, lambda: table.row_count == 14)
            assert table.row_count == 14
            # Nearest surviving index, never a hard reset to the top.
            assert table.cursor_coordinate.row == 13
            assert app._cursor_row_key(table) == "c0000012"
            assert table.scroll_y > 0
            assert table.scroll_y >= scroll_before - 1

            # The row coming back must not steal the cursor either.
            (jobs / "c0000013.json").write_text(
                json.dumps(
                    {"id": "c0000013", "backend": "codex", "status": "completed"}
                )
            )
            await _wait_for(pilot, lambda: table.row_count == 15)
            assert table.row_count == 15
            assert app._cursor_row_key(table) == "c0000012"
            assert table.scroll_y > 0


class TestSessionDetail:
    """Enter on session/subagent rows opens the session detail modal."""

    async def test_enter_on_session_row_opens_session_detail(self, fleet_dirs):
        app = _app(fleet_dirs)
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            from omnimancer.tui.fleet.app import SessionDetailScreen

            table = app.query_one("#agents", DataTable)
            table.move_cursor(row=table.get_row_index("session-11112222-3333"))
            await pilot.press("enter")
            await _settle(pilot)
            assert isinstance(app.screen, SessionDetailScreen)
            assert app.screen.session_id == "11112222-3333"
            assert app.screen.agent_id == "main"
            assert any(
                event.get("event") == "tool_start"
                and (event.get("data") or {}).get("tool") == "Bash"
                for event in app.screen.events
            )
            await pilot.press("escape")
            await _settle(pilot)
            assert not isinstance(app.screen, SessionDetailScreen)

    async def test_enter_on_session_row_canonical_file(self, tmp_path):
        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        (events / "omn-11112222-3333.jsonl").write_text(
            json.dumps(EVENT_SESSION_START) + "\n" + json.dumps(EVENT_TOOL_START) + "\n"
        )
        decoy = dict(EVENT_SESSION_START)
        decoy["session_id"] = "99998888-7777"
        (events / "omn-99998888-7777.jsonl").write_text(json.dumps(decoy) + "\n")
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            from omnimancer.tui.fleet.app import SessionDetailScreen

            table = app.query_one("#agents", DataTable)
            table.move_cursor(row=table.get_row_index("session-11112222-3333"))
            await pilot.press("enter")
            await _settle(pilot)
            assert isinstance(app.screen, SessionDetailScreen)
            assert app.screen.session_id == "11112222-3333"
            assert app.screen.events
            assert all(
                event.get("session_id") == "11112222-3333"
                for event in app.screen.events
            )
            await pilot.press("escape")
            await _settle(pilot)

    async def test_enter_on_subagent_row_filters_events(self, tmp_path):
        from datetime import datetime, timezone

        jobs, events, project = _fleet_tmp_dirs(tmp_path)
        now_ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        main_start = {
            "v": 1,
            "ts": now_ts,
            "event": "session_start",
            "session_id": "eeee5555-6666",
            "agent_id": "main",
            "mode": "interactive",
            "cwd": "/tmp/proj",
            "data": {"provider": "gateway", "model": "qwen3-coder-30b"},
        }
        sub_start = {
            "v": 1,
            "ts": now_ts,
            "event": "session_start",
            "session_id": "eeee5555-6666",
            "agent_id": "subagent-researcher-ab12cd34",
            "parent_id": "main",
            "mode": "interactive",
            "cwd": "/tmp/proj",
            "data": {
                "provider": "gateway",
                "model": "qwen3-4b",
                "subagent": "researcher",
            },
        }
        sub_tool = {
            "v": 1,
            "ts": now_ts,
            "event": "tool_start",
            "session_id": "eeee5555-6666",
            "agent_id": "subagent-researcher-ab12cd34",
            "mode": "interactive",
            "cwd": "/tmp/proj",
            "data": {"tool": "Read", "target": "notes.md"},
        }
        (events / "sess.jsonl").write_text(
            "\n".join(json.dumps(e) for e in (main_start, sub_start, sub_tool)) + "\n"
        )
        app = FleetApp(
            jobs_dir=jobs, events_dir=events, project_dir=project, refresh=0.05
        )
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            from omnimancer.tui.fleet.app import SessionDetailScreen

            table = app.query_one("#agents", DataTable)
            row_key = "session-eeee5555-6666:subagent-researcher-ab12cd34"
            table.move_cursor(row=table.get_row_index(row_key))
            await pilot.press("enter")
            await _settle(pilot)
            assert isinstance(app.screen, SessionDetailScreen)
            assert app.screen.agent_id == "subagent-researcher-ab12cd34"
            assert app.screen.agent_filtered is True
            assert app.screen.events
            assert all(
                event.get("agent_id") == "subagent-researcher-ab12cd34"
                for event in app.screen.events
            )
            await pilot.press("escape")
            await _settle(pilot)

    async def test_job_row_enter_still_opens_job_detail(self, fleet_dirs):
        app = _app(fleet_dirs)
        async with app.run_test() as pilot:
            await _settle(pilot)
            from textual.widgets import DataTable

            table = app.query_one("#agents", DataTable)
            table.move_cursor(row=table.get_row_index("aabbccdd"))
            await pilot.press("enter")
            await _settle(pilot)
            assert isinstance(app.screen, JobDetailScreen)
            assert app.screen.job_id == "aabbccdd"
            await pilot.press("escape")
            await _settle(pilot)
