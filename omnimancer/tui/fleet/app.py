"""FleetApp — the Textual application for the omn fleet dashboard.

Layout: agents table on top, activity feed and comms panel side by side
below. All disk I/O runs in thread workers on timers (exclusive per
source, so a slow disk skips ticks instead of queueing); the UI thread
only consumes their messages.

Data model notes:
- codex-agent jobs come from ``<jobs_dir>/*.json``; display state is
  derived per omnimancer.tui.fleet.models (WAITING needs the
  ``.turn-complete`` signal, omn jobs get an age-based STALE rule).
- Interactive omn sessions are discovered from the events JSONL feed and
  rendered as extra rows when no job matches their cwd.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from omnimancer.tui.fleet.models import (
    DisplayState,
    JobRecord,
    derive_state,
    parse_job,
)
from omnimancer.tui.fleet.sources import (
    AgentsLogParser,
    EventsTailer,
    JobsSnapshot,
    JobsSource,
)
from omnimancer.tui.fleet.widgets import comms_line, feed_line, job_row

AGENT_COLUMNS = (
    "id",
    "backend",
    "state",
    "model",
    "turns",
    "blocker",
    "usage",
    "age",
)

FEED_MAX_LINES = 2000
# Events routed to the comms panel; everything else is activity.
COMMS_EVENTS = {"turn_end", "session_start", "session_end"}

# Operator-first row order: running work at the top, things needing a human
# next, terminal states last. Ties break on the row key for stability.
STATE_PRIORITY: Dict[DisplayState, int] = {
    DisplayState.WORKING: 0,
    DisplayState.STARTING: 1,
    DisplayState.PENDING: 2,
    DisplayState.WAITING: 3,
    DisplayState.BLOCKED: 4,
    DisplayState.STALE: 5,
    DisplayState.FAILED: 6,
    DisplayState.CANCELLED: 7,
    DisplayState.COMPLETED: 8,
}

# htop-style filter groups, cycled with the `f` binding. "attention" is the
# human-on-the-loop view: answered/blocked/stalled agents that need eyes.
FILTERS: tuple = (
    ("all", None),
    (
        "active",
        {DisplayState.WORKING, DisplayState.STARTING, DisplayState.PENDING},
    ),
    (
        "attention",
        {DisplayState.WAITING, DisplayState.BLOCKED, DisplayState.STALE},
    ),
    (
        "done",
        {DisplayState.COMPLETED, DisplayState.FAILED, DisplayState.CANCELLED},
    ),
)


@dataclass
class SessionInfo:
    """Live view of one omn session discovered from the event feed."""

    session_id: str
    mode: str = ""
    cwd: str = ""
    model: str = ""
    provider: str = ""
    # 0.0, not now(): a session discovered from replayed history must age
    # from its own event timestamps, never from when the dashboard read it.
    last_seen: float = 0.0
    turns: int = 0
    ended: bool = False
    # Last lifecycle event name: a session whose latest event is turn_end
    # is idle at its prompt — WAITING in operator terms, not stale.
    last_event: str = ""


class JobsUpdated(Message):
    """A jobs-directory rescan finished (posted from a thread worker)."""

    def __init__(self, snapshot: JobsSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot


class FeedsUpdated(Message):
    """Event/ledger tails produced new entries (posted from a worker)."""

    def __init__(self, events: List[dict], ledger: List[dict]) -> None:
        super().__init__()
        self.events = events
        self.ledger = ledger


class JobDetailScreen(ModalScreen[None]):
    """Modal with the raw job record and a tail of its tmux log."""

    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, job_id: str, raw_job: dict, log_tail: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.raw_job = raw_job
        self.log_tail = log_tail

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="detail"):
            yield Static(f"[b]Job {self.job_id}[/b]", id="detail-title")
            # Text objects bypass markup parsing entirely: job JSON and
            # tmux log tails contain brackets and raw ANSI escapes that
            # crash the markup parser (field-reported). from_ansi renders
            # the log's own colors instead of choking on them.
            yield Static(Text(json.dumps(self.raw_job, indent=2, default=str)))
            if self.log_tail:
                yield Static(Text("log tail", style="bold"))
                yield Static(Text.from_ansi(self.log_tail))


class FleetApp(App[None]):
    """Full-screen dashboard over fleet agents, activity, and comms."""

    TITLE = "omn fleet"

    CSS = """
    #agents {
        height: 40%;
        border: solid $primary;
    }
    #feeds {
        height: 1fr;
    }
    #activity {
        width: 60%;
        border: solid $secondary;
    }
    #comms {
        width: 1fr;
        border: solid $secondary;
    }
    #detail {
        background: $surface;
        border: thick $primary;
        margin: 2 4;
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "toggle_pause", "Pause feeds"),
        Binding("f", "cycle_filter", "Filter"),
    ]

    def __init__(
        self,
        jobs_dir: Path,
        events_dir: Path,
        project_dir: Path,
        refresh: float = 1.0,
        once: bool = False,
        once_fallback_s: float = 10.0,
        budget_bytes: int = 50 * 1024**3,
    ) -> None:
        """Store data-source locations and refresh cadence.

        Args:
            jobs_dir: codex-agent jobs directory.
            events_dir: omnimancer JSONL events directory.
            project_dir: project root containing agents.log.
            refresh: jobs rescan interval seconds; tails poll at half this.
            once: render a single snapshot and exit (smoke/scripting mode).
            once_fallback_s: --once deadline for the initial polls; hitting
                it exits with return code 1 and an incompleteness warning
                instead of silently shipping a partial snapshot.
            budget_bytes: events-dir size budget the periodic sweep
                enforces (the audit store's cap while the board is open).
        """
        super().__init__()
        self.jobs_dir = jobs_dir
        self.events_dir = events_dir
        self.project_dir = project_dir
        self.refresh_interval = refresh
        self.once = once
        self.once_fallback_s = once_fallback_s
        self.budget_bytes = budget_bytes
        self.paused = False
        self._jobs_source = JobsSource(jobs_dir)
        self._events_tailer = EventsTailer(events_dir)
        self._ledger_parser = AgentsLogParser(project_dir / "agents.log")
        self._snapshot: JobsSnapshot = JobsSnapshot.empty()
        self._sessions: Dict[str, SessionInfo] = {}
        # --once exits only after BOTH initial polls have landed (jobs scan
        # AND event/ledger replay) — otherwise session rows would be absent
        # from the snapshot instead of correctly aged.
        self._once_seen = {"jobs": False, "feeds": False}
        self._once_finishing = False
        self._filter_index = 0

    def compose(self) -> ComposeResult:
        """Build the static widget tree."""
        yield Header(show_clock=True)
        yield DataTable(id="agents", cursor_type="row")
        with Horizontal(id="feeds"):
            yield RichLog(
                id="activity", markup=False, wrap=False, max_lines=FEED_MAX_LINES
            )
            yield RichLog(
                id="comms", markup=False, wrap=False, max_lines=FEED_MAX_LINES
            )
        yield Footer()

    def on_mount(self) -> None:
        """Set up table columns, timers, and once-mode exit."""
        table = self.query_one("#agents", DataTable)
        table.add_columns(*AGENT_COLUMNS)
        self.set_interval(self.refresh_interval, self.refresh_jobs)
        self.set_interval(self.refresh_interval / 2, self.refresh_feeds)
        # Keep the audit store bounded even when no omn session starts
        # (sessions sweep on their own writer threads; the board covers
        # the long-running-dashboard case).
        self.set_interval(600.0, self._budget_sweep)
        self.refresh_jobs()
        self.refresh_feeds()
        if self.once:
            # Snapshot mode exits after BOTH initial polls have been
            # applied (_maybe_finish_once); this deadline is the wedged-
            # source escape hatch, and it reports incompleteness instead
            # of silently shipping a partial snapshot.
            self.set_timer(self.once_fallback_s, self._once_deadline)

    def action_cycle_filter(self) -> None:
        """Cycle the agents-table filter: all -> active -> attention -> done."""
        self._filter_index = (self._filter_index + 1) % len(FILTERS)
        self._rebuild_table()

    def action_toggle_pause(self) -> None:
        """Toggle feed auto-scroll / polling."""
        self.paused = not self.paused
        for log_id in ("#activity", "#comms"):
            self.query_one(log_id, RichLog).auto_scroll = not self.paused

    # Data collection (thread workers) --------------------------------

    def refresh_jobs(self) -> None:
        """Rescan the jobs directory off the UI thread."""
        if self.paused:
            return
        self.run_worker(self._scan_jobs, thread=True, exclusive=True, group="jobs-scan")

    def refresh_feeds(self) -> None:
        """Tail event/ledger sources off the UI thread."""
        if self.paused:
            return
        self.run_worker(
            self._tail_feeds, thread=True, exclusive=True, group="feeds-tail"
        )

    def _scan_jobs(self) -> None:
        """Thread worker: scan jobs and post the snapshot."""
        self.post_message(JobsUpdated(self._jobs_source.scan()))

    def _budget_sweep(self) -> None:
        """Enforce the events-dir size budget off the UI thread."""
        self.run_worker(
            self._run_budget_sweep, thread=True, exclusive=True, group="budget"
        )

    def _run_budget_sweep(self) -> None:
        """Thread worker: oldest-first pruning down to budget_bytes."""
        from omnimancer.events.jsonl_writer import (
            SESSION_FILE_RE,
            enforce_size_budget,
        )

        enforce_size_budget(self.events_dir, self.budget_bytes, name_re=SESSION_FILE_RE)

    def _tail_feeds(self) -> None:
        """Thread worker: poll both tails and post new entries.

        Posts even when empty: --once completion and liveness both key off
        "a poll finished", not "a poll found something".
        """
        events = self._events_tailer.poll()
        ledger = self._ledger_parser.poll()
        self.post_message(FeedsUpdated(events, ledger))

    # UI-thread message handlers --------------------------------------

    def on_jobs_updated(self, message: JobsUpdated) -> None:
        """Rebuild the agents table from a fresh snapshot."""
        self._snapshot = message.snapshot
        self._rebuild_table()
        self._maybe_finish_once("jobs")

    def on_feeds_updated(self, message: FeedsUpdated) -> None:
        """Route new events/ledger entries into the feed panels."""
        activity = self.query_one("#activity", RichLog)
        comms = self.query_one("#comms", RichLog)
        for event in message.events:
            self._track_session(event)
            if event.get("event") in COMMS_EVENTS:
                comms.write(comms_line(event))
            else:
                activity.write(feed_line(event))
        for entry in message.ledger:
            comms.write(comms_line(entry))
        self._maybe_finish_once("feeds")

    def _maybe_finish_once(self, source: str) -> None:
        """In --once mode, exit after both initial polls have been applied.

        The final rebuild runs after the second poll lands so session rows
        discovered from the event replay are in the snapshot table.
        """
        if not self.once or self._once_finishing:
            return
        self._once_seen[source] = True
        if all(self._once_seen.values()):
            self._once_finishing = True
            self._rebuild_table()
            self.call_after_refresh(self.exit)

    def _once_deadline(self) -> None:
        """--once escape hatch for a wedged data source.

        Exits non-zero with a warning: a deadline exit means the snapshot
        is missing at least one initial poll and must not be mistaken for
        a complete render.
        """
        if self._once_finishing:
            return
        self._once_finishing = True
        missing = [name for name, seen in self._once_seen.items() if not seen]
        self._rebuild_table()
        self.exit(
            return_code=1,
            message=(
                "omn fleet --once: timed out waiting for initial data "
                f"({', '.join(missing)}); snapshot incomplete"
            ),
        )

    @staticmethod
    def _event_epoch(event: dict) -> float:
        """Epoch seconds of an event's own timestamp (ingestion time only
        as a fallback for unparsable ts). Using the event's clock means a
        dashboard started after a session died sees its true age instead
        of treating replayed history as fresh activity."""
        ts = event.get("ts")
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts).timestamp()
            except ValueError:
                pass
        return time.time()

    def _track_session(self, event: dict) -> None:
        """Maintain the omn-session registry from lifecycle events."""
        session_id = str(event.get("session_id") or "")
        if not session_id:
            return
        info = self._sessions.setdefault(session_id, SessionInfo(session_id))
        info.last_seen = max(info.last_seen, self._event_epoch(event))
        info.mode = str(event.get("mode") or info.mode)
        info.cwd = str(event.get("cwd") or info.cwd)
        data = event.get("data") or {}
        name = event.get("event")
        if isinstance(name, str) and name:
            info.last_event = name
        if name == "session_start":
            info.model = str(data.get("model") or "")
            info.provider = str(data.get("provider") or "")
        elif name == "turn_end":
            info.turns += 1
        elif name == "session_end":
            info.ended = True

    def _collect_rows(self) -> list:
        """Build (key, state, cells) for every job and loose session."""
        now = time.time()
        snapshot = self._snapshot
        rows = []
        job_cwds = set()
        for job_id, raw in sorted(snapshot.jobs.items()):
            job = parse_job(raw)
            job_cwds.add(job.cwd)
            age = self._job_age_s(job_id, now)
            state = derive_state(
                job,
                has_turn_complete=job_id in snapshot.turn_complete_ids,
                activity_age_s=age,
            )
            rows.append((job_id, state, job_row(job, state, age)))
        for session in sorted(self._sessions.values(), key=lambda s: s.session_id):
            if session.ended or session.cwd in job_cwds:
                continue
            if session.mode == "claude":
                backend = "claude"
            elif session.mode:
                backend = f"omn:{session.mode}"
            else:
                backend = "omn"
            record = JobRecord(
                job_id=session.session_id[:8],
                backend=backend,
                status="running",
                model=session.model,
                turns_completed=session.turns,
            )
            age = now - session.last_seen
            if session.last_event == "turn_end":
                # Idle at its prompt: answered and waiting for a human.
                state = DisplayState.WAITING
            elif age > 120.0:
                state = DisplayState.STALE
            else:
                state = DisplayState.WORKING
            rows.append(
                (f"session-{session.session_id}", state, job_row(record, state, age))
            )
        return rows

    def _cursor_row_key(self, table: DataTable) -> Optional[str]:
        """Row key under the cursor, or None."""
        try:
            if table.row_count == 0 or table.cursor_row is None:
                return None
            coordinate = table.cursor_coordinate
            cell_key = table.coordinate_to_cell_key(coordinate)
            return cell_key.row_key.value
        except Exception:
            return None

    def _rebuild_table(self) -> None:
        """Recompute the agents table: filtered, operator-sorted, and with
        the cursor kept on the same row across rescans (a 1s refresh must
        never yank the operator back to the top)."""
        table = self.query_one("#agents", DataTable)
        cursor_key = self._cursor_row_key(table)

        rows = self._collect_rows()
        total = len(rows)
        filter_name, allowed = FILTERS[self._filter_index]
        if allowed is not None:
            rows = [row for row in rows if row[1] in allowed]
        rows.sort(key=lambda row: (STATE_PRIORITY.get(row[1], 99), row[0]))

        table.clear()
        for key, _state, cells in rows:
            table.add_row(*cells, key=key)
        table.border_title = f"agents — {filter_name} ({len(rows)}/{total})"

        if cursor_key is not None:
            try:
                table.move_cursor(row=table.get_row_index(cursor_key))
            except Exception:
                # Row filtered out or gone; leave the cursor clamped.
                pass

    def _job_age_s(self, job_id: str, now: float) -> Optional[float]:
        """Age of the newest on-disk activity for a job, if known."""
        mtimes = [
            mtime
            for mtime in (
                self._snapshot.json_mtimes.get(job_id),
                self._snapshot.log_mtimes.get(job_id),
            )
            if mtime is not None
        ]
        if not mtimes:
            return None
        return now - max(mtimes)

    # Interactivity ----------------------------------------------------

    def on_data_table_row_selected(self, message: DataTable.RowSelected) -> None:
        """Open the detail modal for a selected job row."""
        row_key = message.row_key.value if message.row_key else None
        if not row_key or row_key.startswith("session-"):
            return
        raw = self._snapshot.jobs.get(row_key, {})
        self.run_worker(
            lambda: self._open_detail(row_key, raw),
            thread=True,
            exclusive=True,
            group="detail",
        )

    def _open_detail(self, job_id: str, raw: dict) -> None:
        """Thread worker: read the log tail, then push the modal."""
        log_tail = ""
        log_path = self.jobs_dir / f"{job_id}.log"
        try:
            text = log_path.read_text(errors="replace")
            log_tail = "\n".join(text.splitlines()[-30:])
        except OSError:
            pass
        self.call_from_thread(self.push_screen, JobDetailScreen(job_id, raw, log_tail))
