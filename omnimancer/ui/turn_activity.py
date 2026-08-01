"""Live per-turn tool activity for the interactive REPL.

`TurnActivityLog` subscribes to the in-process event bus and keeps a small
bounded window of recent tool/approval rows. During streaming, the
existing single `StreamingDisplay` Live renders it above the assistant
panel via its ``activity_provider`` — no second Live region is ever
created, so the documented nested-Live hazards cannot occur.

Rows are formatted with the same renderer the fleet dashboard uses
(:func:`omnimancer.tui.fleet.widgets.feed_line`, rich-only — no Textual
import), so the REPL and `omn fleet` read identically.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from rich.table import Table

from ..core.agent.status_core import AgentEvent, EventListener, EventType
from ..events.emitter import _EVENT_NAME_BY_TYPE
from ..tui.fleet.widgets import feed_line

# Bus events shown in the REPL activity window.
_ACTIVITY_TYPES = {
    EventType.OPERATION_STARTED,
    EventType.OPERATION_COMPLETED,
    EventType.OPERATION_FAILED,
    EventType.OPERATION_CANCELLED,
    EventType.APPROVAL_REQUESTED,
    EventType.APPROVAL_GRANTED,
    EventType.APPROVAL_DENIED,
}

MAX_ROWS = 6


class TurnActivityLog(EventListener):
    """Bounded window of this turn's tool activity, rendered on demand."""

    def __init__(self, max_rows: int = MAX_ROWS) -> None:
        """Subscribe to tool/approval events only."""
        super().__init__(event_types=_ACTIVITY_TYPES)
        self._rows: Deque = deque(maxlen=max_rows)
        self._suspended = False

    def reset_turn(self) -> None:
        """Clear the window at a turn boundary."""
        self._rows.clear()

    def suspend(self) -> None:
        """Stop collecting rows (e.g. while an approval prompt owns stdin)."""
        self._suspended = True

    def resume(self) -> None:
        """Resume collecting rows."""
        self._suspended = False

    async def _process_event(self, event: AgentEvent) -> None:
        """Convert a bus event to a display row (never raises)."""
        if self._suspended:
            return
        try:
            name = _EVENT_NAME_BY_TYPE.get(event.event_type)
            if name is None:
                return
            data = dict(event.data or {})
            metadata = data.pop("metadata", None)
            if isinstance(metadata, dict):
                data.update(metadata)
            if event.event_type == EventType.OPERATION_COMPLETED:
                data.setdefault("success", True)
            elif event.event_type in (
                EventType.OPERATION_FAILED,
                EventType.OPERATION_CANCELLED,
            ):
                data.setdefault("success", False)
            self._rows.append(
                feed_line(
                    {
                        "ts": event.timestamp.astimezone().isoformat(),
                        "event": name,
                        "session_id": event.agent_id or "main",
                        "data": data,
                    }
                )
            )
        except Exception:  # display-only; never disturb the bus
            pass

    def render(self) -> Optional[Table]:
        """Return the activity table, or None when there is nothing to show."""
        if not self._rows:
            return None
        table = Table.grid(padding=(0, 1))
        table.add_column()
        for row in self._rows:
            table.add_row(row)
        return table
