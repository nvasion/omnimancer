"""Rendering functions for the TUI fleet view."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from rich.text import Text

from omnimancer.tui.fleet.models import DisplayState, JobRecord

STATE_STYLES: Dict[DisplayState, str] = {
    DisplayState.PENDING: "dim",
    DisplayState.STARTING: "yellow",
    DisplayState.WORKING: "green",
    DisplayState.WAITING: "cyan",
    DisplayState.BLOCKED: "bold red",
    DisplayState.STALE: "yellow",
    DisplayState.COMPLETED: "dim green",
    DisplayState.FAILED: "red",
    DisplayState.CANCELLED: "dim",
}


def format_age(age_s: Optional[float]) -> str:
    """Format age in seconds into a human-readable string.

    Args:
        age_s: Age in seconds, or None if unknown

    Returns:
        Formatted age string (e.g., "12s", "1m", "2h") or "-" if None
    """
    if age_s is None:
        return "-"
    if age_s < 60:
        return f"{int(age_s)}s"
    if age_s < 3600:
        return f"{int(age_s // 60)}m"
    return f"{int(age_s // 3600)}h"


def format_tokens(usage: Optional[dict]) -> str:
    """Format token usage into a human-readable string.

    Args:
        usage: Token usage dict with input_tokens and output_tokens keys.

    Returns:
        Formatted token string (e.g., "1.0k/1.5k") or "-" if None/invalid
    """
    if usage is None or "input_tokens" not in usage or "output_tokens" not in usage:
        return "-"
    # Usage values come from external JSON: guard the arithmetic so a
    # malformed None/string value renders "-" instead of crashing the feed.
    try:
        input_tokens = float(usage.get("input_tokens", 0))
        output_tokens = float(usage.get("output_tokens", 0))
    except (TypeError, ValueError):
        return "-"
    return f"{input_tokens/1000:.1f}k/{output_tokens/1000:.1f}k"


def _ts_hms(ts: object) -> str:
    """Extract HH:MM:SS from ISO timestamp string.

    Args:
        ts: Timestamp string in ISO format or other type

    Returns:
        Formatted time string (HH:MM:SS) or "--:--:--" if invalid
    """
    try:
        dt = datetime.fromisoformat(str(ts))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return "--:--:--"


def _sid8(session_id: object) -> str:
    """Get first 8 characters of session ID.

    Args:
        session_id: Session ID value

    Returns:
        First 8 characters of session ID string
    """
    return str(session_id)[:8]


FEED_SYMBOLS = {
    "tool_start": "▶",
    "tool_progress": "…",
    "approval_requested": "?",
    "approval_denied": "✗",
    "approval_granted": "✓",
    "session_start": "●",
    "session_end": "○",
    "turn_start": "→",
    "turn_end": "✉",
    "error": "!",
}


def job_row(job: JobRecord, state: DisplayState, age_s: Optional[float]) -> tuple:
    """Create a row for displaying job information.

    Args:
        job: Job record to display
        state: Current display state of the job
        age_s: Age of the job in seconds, or None if unknown

    Returns:
        Tuple of 8 rich Text objects representing the row columns
    """
    return (
        Text(job.job_id),
        Text(job.backend),
        Text(state.value, style=STATE_STYLES[state]),
        Text(job.model),
        Text(str(job.turns_completed)),
        Text(job.blocker_kind or "-"),
        Text(format_tokens(job.usage)),
        Text(format_age(age_s)),
    )


def feed_line(event: dict) -> Text:
    """Render a fleet event into a rich Text line for the feed.

    Args:
        event: Fleet event dictionary

    Returns:
        Rich Text object representing the formatted event line
    """
    data = event.get("data") or {}
    event_name = event.get("event", "")

    # Determine symbol based on event type
    if event_name == "tool_end":
        success = data.get("success", False)
        symbol = "✓" if success else "✗"
    else:
        symbol = FEED_SYMBOLS.get(event_name, "·")

    # Extract details - but handle approval events specially
    detail = data.get("error") or data.get("target") or data.get("description") or ""

    # For approval events, we should not include the event name as tool;
    # tool_end never falls back to the event name (a bare "tool_end" label
    # is noise).
    if event_name in ["approval_requested", "approval_denied"]:
        tool = ""
    elif event_name == "tool_end":
        tool = data.get("tool", "")
    else:
        tool = data.get("tool") or event_name

    # Build the text line
    if tool:
        return Text.assemble(
            (f"{_ts_hms(event.get('ts'))} ", "dim"),
            (f"{_sid8(event.get('session_id'))} ", "dim"),
            (symbol + " ", "red" if symbol == "✗" else "default"),
            (f"{tool}", "bold"),
            (f" {detail}" if detail else "", "default"),
        )
    else:
        return Text.assemble(
            (f"{_ts_hms(event.get('ts'))} ", "dim"),
            (f"{_sid8(event.get('session_id'))} ", "dim"),
            (symbol + " ", "red" if symbol == "✗" else "default"),
            (f"{detail}" if detail else "", "default"),
        )


def comms_line(entry: dict) -> Text:
    """Render a communication line (either a ledger event or a fleet event).

    Args:
        entry: Either a ledger entry (with "kind") or a fleet event

    Returns:
        Rich Text object representing the formatted line
    """
    # Handle ledger events
    if "kind" in entry:
        kind = entry.get("kind", "")
        job_id = entry.get("job_id", "")
        if kind == "verdict":
            verdict = entry.get("verdict", "")
            return Text.assemble(
                (
                    f"VERDICT: {verdict}",
                    "bold green" if verdict == "PASS" else "bold red",
                )
            )
        else:
            return Text.assemble((f"{kind} ", "cyan"), (str(job_id), "bold"))

    # Handle fleet events
    event_name = entry.get("event", "")
    if event_name == "turn_end":
        # Special handling for turn_end events
        ts = entry.get("ts", "")
        sid = entry.get("session_id", "")
        data = entry.get("data", {})
        turn = data.get("turn", "?")
        usage = data.get("usage", {})
        return Text.assemble(
            (f"{_ts_hms(ts)} ", "dim"),
            (f"{_sid8(sid)} ", "dim"),
            ("✉ ", "cyan"),
            (f"turn {turn} ", "default"),
            (format_tokens(usage), "dim"),
        )
    else:
        # Fall back to feed_line for other events
        return feed_line(entry)
