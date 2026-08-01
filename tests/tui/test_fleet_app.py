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

    async def test_row_selection_opens_detail_modal(self, fleet_dirs):
        app = _app(fleet_dirs)
        async with app.run_test() as pilot:
            await _settle(pilot)
            await pilot.press("enter")
            await _settle(pilot)
            assert isinstance(app.screen, JobDetailScreen)
            assert app.screen.job_id == "aabbccdd"
