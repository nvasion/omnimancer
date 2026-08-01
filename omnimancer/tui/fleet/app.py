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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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


@dataclass
class SessionInfo:
    """Live view of one omn session discovered from the event feed."""

    session_id: str
    mode: str = ""
    cwd: str = ""
    model: str = ""
    provider: str = ""
    last_seen: float = field(default_factory=time.time)
    turns: int = 0
    ended: bool = False


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
            yield Static(json.dumps(self.raw_job, indent=2, default=str))
            if self.log_tail:
                yield Static(f"[b]log tail[/b]\n{self.log_tail}")


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
    ]

    def __init__(
        self,
        jobs_dir: Path,
        events_dir: Path,
        project_dir: Path,
        refresh: float = 1.0,
        once: bool = False,
    ) -> None:
        """Store data-source locations and refresh cadence.

        Args:
            jobs_dir: codex-agent jobs directory.
            events_dir: omnimancer JSONL events directory.
            project_dir: project root containing agents.log.
            refresh: jobs rescan interval seconds; tails poll at half this.
            once: render a single snapshot and exit (smoke/scripting mode).
        """
        super().__init__()
        self.jobs_dir = jobs_dir
        self.events_dir = events_dir
        self.project_dir = project_dir
        self.refresh_interval = refresh
        self.once = once
        self.paused = False
        self._jobs_source = JobsSource(jobs_dir)
        self._events_tailer = EventsTailer(events_dir)
        self._ledger_parser = AgentsLogParser(project_dir / "agents.log")
        self._snapshot: JobsSnapshot = JobsSnapshot.empty()
        self._sessions: Dict[str, SessionInfo] = {}

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
        self.refresh_jobs()
        self.refresh_feeds()
        if self.once:
            # Snapshot mode exits after the first populated table render
            # (see on_jobs_updated); this timer is the no-data fallback.
            self.set_timer(3.0, self.exit)

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

    def _tail_feeds(self) -> None:
        """Thread worker: poll both tails and post new entries."""
        events = self._events_tailer.poll()
        ledger = self._ledger_parser.poll()
        if events or ledger:
            self.post_message(FeedsUpdated(events, ledger))

    # UI-thread message handlers --------------------------------------

    def on_jobs_updated(self, message: JobsUpdated) -> None:
        """Rebuild the agents table from a fresh snapshot."""
        self._snapshot = message.snapshot
        self._rebuild_table()
        if self.once:
            self.call_after_refresh(self.exit)

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

    def _track_session(self, event: dict) -> None:
        """Maintain the omn-session registry from lifecycle events."""
        session_id = str(event.get("session_id") or "")
        if not session_id:
            return
        info = self._sessions.setdefault(session_id, SessionInfo(session_id))
        info.last_seen = time.time()
        info.mode = str(event.get("mode") or info.mode)
        info.cwd = str(event.get("cwd") or info.cwd)
        data = event.get("data") or {}
        name = event.get("event")
        if name == "session_start":
            info.model = str(data.get("model") or "")
            info.provider = str(data.get("provider") or "")
        elif name == "turn_end":
            info.turns += 1
        elif name == "session_end":
            info.ended = True

    def _rebuild_table(self) -> None:
        """Recompute every row: codex/omn jobs first, loose sessions after."""
        table = self.query_one("#agents", DataTable)
        table.clear()
        now = time.time()
        snapshot = self._snapshot
        job_cwds = set()
        for job_id, raw in sorted(snapshot.jobs.items()):
            job = parse_job(raw)
            job_cwds.add(job.cwd)
            state = derive_state(
                job,
                has_turn_complete=job_id in snapshot.turn_complete_ids,
                activity_age_s=self._job_age_s(job_id, now),
            )
            table.add_row(
                *job_row(job, state, self._job_age_s(job_id, now)), key=job_id
            )
        for session in sorted(self._sessions.values(), key=lambda s: s.session_id):
            if session.ended or session.cwd in job_cwds:
                continue
            record = JobRecord(
                job_id=session.session_id[:8],
                backend=f"omn:{session.mode}" if session.mode else "omn",
                status="running",
                model=session.model,
                turns_completed=session.turns,
            )
            age = now - session.last_seen
            state = DisplayState.STALE if age > 120.0 else DisplayState.WORKING
            table.add_row(
                *job_row(record, state, age), key=f"session-{session.session_id}"
            )

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
