"""Tests for the TUI fleet widgets rendering functions."""

from omnimancer.tui.fleet.models import DisplayState, JobRecord
from omnimancer.tui.fleet.widgets import (
    STATE_STYLES,
    comms_line,
    feed_line,
    format_age,
    format_tokens,
    job_row,
)


def test_state_styles_cover_all_states():
    """Test that STATE_STYLES has an entry for every DisplayState member."""
    assert len(STATE_STYLES) == len(DisplayState)
    for state in DisplayState:
        assert state in STATE_STYLES


def test_job_row_columns():
    """Test job_row columns."""
    row = job_row(
        JobRecord(
            job_id="4553891c",
            backend="codex",
            model="gpt-5.6-sol",
            turns_completed=3,
            blocker_kind=None,
            usage={"input_tokens": 12345, "output_tokens": 678},
        ),
        DisplayState.WORKING,
        age_s=95.0,
    )
    assert len(row) == 8
    assert row[0].plain == "4553891c"
    assert row[1].plain == "codex"
    assert row[2].plain == "working"
    assert row[3].plain == "gpt-5.6-sol"
    assert row[4].plain == "3"
    assert row[5].plain == "-"
    assert row[6].plain == "12.3k/0.7k"
    assert row[7].plain == "1m"


def test_job_row_blocker_and_missing_usage():
    """Test job_row with blocker_kind and missing usage."""
    row = job_row(
        JobRecord(
            job_id="test",
            backend="test",
            model="test",
            turns_completed=0,
            blocker_kind="context_limit",
            usage=None,
        ),
        DisplayState.BLOCKED,
        age_s=None,
    )
    assert row[5].plain == "context_limit"
    assert row[6].plain == "-"
    assert row[7].plain == "-"


def test_format_age():
    """Test format_age function."""
    assert format_age(None) == "-"
    assert format_age(12.0) == "12s"
    assert format_age(95.0) == "1m"
    assert format_age(7300.0) == "2h"


def test_format_tokens():
    """Test format_tokens function."""
    assert format_tokens(None) == "-"
    assert format_tokens({}) == "-"
    assert format_tokens({"input_tokens": 999, "output_tokens": 1500}) == "1.0k/1.5k"
    assert (
        format_tokens({"input_tokens": 250000, "output_tokens": 4773}) == "250.0k/4.8k"
    )


def test_feed_line_tool_start():
    """Test feed_line with tool_start event."""
    line = feed_line(
        {
            "ts": "2026-08-01T03:14:34.686+00:00",
            "event": "tool_start",
            "session_id": "359346cf-aaaa",
            "agent_id": "main",
            "data": {"tool": "Bash", "target": "pytest -q"},
        }
    )
    assert line.plain == "03:14:34 359346cf ▶ Bash pytest -q"


def test_feed_line_tool_end_failure():
    """Test feed_line with tool_end event that failed."""
    line = feed_line(
        {
            "ts": "2026-08-01T03:14:34.686+00:00",
            "event": "tool_end",
            "session_id": "359346cf-aaaa",
            "agent_id": "main",
            "data": {"tool": "Write", "success": False, "error": "denied"},
        }
    )
    assert line.plain == "03:14:34 359346cf ✗ Write denied"
    assert "red" in line.markup


def test_feed_line_tool_end_success():
    """Test feed_line with tool_end event that succeeded."""
    line = feed_line(
        {
            "ts": "2026-08-01T03:14:34.686+00:00",
            "event": "tool_end",
            "session_id": "359346cf-aaaa",
            "agent_id": "main",
            "data": {"tool": "Read", "success": True, "target": "/x.py"},
        }
    )
    assert line.plain == "03:14:34 359346cf ✓ Read /x.py"


def test_feed_line_approval():
    """Test feed_line with approval events."""
    line = feed_line(
        {
            "ts": "2026-08-01T03:14:34.686+00:00",
            "event": "approval_requested",
            "session_id": "359346cf-aaaa",
            "agent_id": "main",
            "data": {},
        }
    )
    assert line.plain == "03:14:34 359346cf ? "

    line = feed_line(
        {
            "ts": "2026-08-01T03:14:34.686+00:00",
            "event": "approval_denied",
            "session_id": "359346cf-aaaa",
            "agent_id": "main",
            "data": {},
        }
    )
    assert line.plain == "03:14:34 359346cf ✗ "


def test_feed_line_bad_ts_and_unknown_event():
    """Test feed_line with bad timestamp and unknown event."""
    line = feed_line(
        {
            "ts": "garbage",
            "event": "something_new",
            "session_id": "359346cf-aaaa",
            "agent_id": "main",
            "data": {},
        }
    )
    assert line.plain.startswith("--:--:-- ")


def test_comms_line_turn_end():
    """Test comms_line with turn_end event."""
    line = comms_line(
        {
            "ts": "2026-08-01T03:14:34.686+00:00",
            "event": "turn_end",
            "session_id": "359346cf-aaaa",
            "agent_id": "main",
            "data": {"turn": 2, "usage": {"input_tokens": 1000, "output_tokens": 200}},
        }
    )
    assert line.plain == "03:14:34 359346cf ✉ turn 2 1.0k/0.2k"


def test_comms_line_verdict():
    """Test comms_line with verdict event."""
    line = comms_line(
        {
            "kind": "verdict",
            "verdict": "PASS",
            "line": "- VERDICT: PASS",
        }
    )
    assert "VERDICT: PASS" in line.plain
    assert "green" in line.markup

    line = comms_line(
        {
            "kind": "verdict",
            "verdict": "FAIL",
            "line": "- VERDICT: FAIL",
        }
    )
    assert "VERDICT: FAIL" in line.plain
    assert "red" in line.markup


def test_comms_line_ledger():
    """Test comms_line with ledger events."""
    line = comms_line(
        {
            "kind": "spawned",
            "job_id": "aabbccdd",
            "line": "### Spawned: aabbccdd - 03:11",
        }
    )
    assert "spawned aabbccdd" in line.plain

    line = comms_line(
        {
            "kind": "complete",
            "job_id": "aabbccdd",
            "line": "### Complete: aabbccdd - 03:11",
        }
    )
    assert "complete aabbccdd" in line.plain

    line = comms_line(
        {
            "kind": "died",
            "job_id": "aabbccdd",
            "line": "### Died: aabbccdd - 03:11",
        }
    )
    assert "died aabbccdd" in line.plain
