"""Tests for the fleet event schema."""

import json
import os

from omnimancer.events.schema import (
    ALL_EVENTS,
    EVENT_APPROVAL_DENIED,
    EVENT_APPROVAL_GRANTED,
    EVENT_APPROVAL_REQUESTED,
    EVENT_ERROR,
    EVENT_SESSION_END,
    EVENT_SESSION_START,
    EVENT_TOOL_END,
    EVENT_TOOL_PROGRESS,
    EVENT_TOOL_START,
    EVENT_TURN_END,
    EVENT_TURN_START,
    FleetEvent,
    translate_operation_type,
    truncate,
)


def test_to_json_line_single_line():
    """Test that to_json_line returns a single line without newlines."""
    event = FleetEvent(event="tool_start", session_id="s1", data={"tool": "Bash"})
    json_line = event.to_json_line()
    assert isinstance(json_line, str)
    assert "\n" not in json_line
    parsed = json.loads(json_line)
    assert set(parsed.keys()) == {
        "v",
        "ts",
        "event",
        "session_id",
        "agent_id",
        "parent_id",
        "pid",
        "seq",
        "mode",
        "cwd",
        "data",
    }


def test_envelope_defaults():
    """Test default values for envelope fields."""
    event = FleetEvent(event="tool_start", session_id="s1", data={"tool": "Bash"})
    parsed = json.loads(event.to_json_line())
    assert parsed["v"] == 1
    assert parsed["agent_id"] == "main"
    assert parsed["parent_id"] is None
    assert parsed["mode"] == "interactive"
    assert parsed["pid"] == os.getpid()
    assert parsed["seq"] == 0
    assert parsed["cwd"] == os.getcwd()
    assert parsed["data"] == {"tool": "Bash"}


def test_ts_utc_iso():
    """Test that timestamp is in proper UTC ISO format."""
    event = FleetEvent(event="tool_start", session_id="s1", data={"tool": "Bash"})
    parsed = json.loads(event.to_json_line())
    ts = parsed["ts"]
    assert ts.endswith("+00:00")
    # This should not raise an exception
    import datetime

    datetime.datetime.fromisoformat(ts)


def test_truncate_short_unchanged():
    """Test truncation of short text is unchanged."""
    result = truncate("abc", 10)
    assert result == "abc"


def test_truncate_long():
    """Test truncation of long text."""
    r = truncate("x" * 300, 200)
    assert len(r) == 200
    assert r.endswith("…")


def test_translate_all_members():
    """Every agent OperationType maps to the exact status-core member."""
    from omnimancer.core.agent.status_core import OperationType as StatusOperationType
    from omnimancer.core.agent.types import OperationType as AgentOperationType

    expected = {
        AgentOperationType.COMMAND_EXECUTE: StatusOperationType.COMMAND_EXECUTION,
        AgentOperationType.MCP_TOOL_CALL: StatusOperationType.API_CALL,
        AgentOperationType.WORKFLOW_STEP: StatusOperationType.API_CALL,
        AgentOperationType.FILE_READ: StatusOperationType.FILE_READ,
        AgentOperationType.FILE_WRITE: StatusOperationType.FILE_WRITE,
        AgentOperationType.FILE_DELETE: StatusOperationType.FILE_DELETE,
        AgentOperationType.DIRECTORY_CREATE: StatusOperationType.DIRECTORY_CREATE,
        AgentOperationType.DIRECTORY_DELETE: StatusOperationType.DIRECTORY_DELETE,
        AgentOperationType.WEB_REQUEST: StatusOperationType.WEB_REQUEST,
    }
    # Exhaustive: a new agent OperationType member without a mapping is a bug.
    assert set(expected) == set(AgentOperationType)
    for member, status_member in expected.items():
        assert translate_operation_type(member) is status_member


def test_event_name_constants():
    """Test that all event name constants are correctly defined."""
    assert EVENT_SESSION_START == "session_start"
    assert EVENT_SESSION_END == "session_end"
    assert EVENT_TURN_START == "turn_start"
    assert EVENT_TURN_END == "turn_end"
    assert EVENT_TOOL_START == "tool_start"
    assert EVENT_TOOL_PROGRESS == "tool_progress"
    assert EVENT_TOOL_END == "tool_end"
    assert EVENT_APPROVAL_REQUESTED == "approval_requested"
    assert EVENT_APPROVAL_GRANTED == "approval_granted"
    assert EVENT_APPROVAL_DENIED == "approval_denied"
    assert EVENT_ERROR == "error"

    expected_events = {
        EVENT_SESSION_START,
        EVENT_SESSION_END,
        EVENT_TURN_START,
        EVENT_TURN_END,
        EVENT_TOOL_START,
        EVENT_TOOL_PROGRESS,
        EVENT_TOOL_END,
        EVENT_APPROVAL_REQUESTED,
        EVENT_APPROVAL_GRANTED,
        EVENT_APPROVAL_DENIED,
        EVENT_ERROR,
    }
    assert ALL_EVENTS == expected_events
