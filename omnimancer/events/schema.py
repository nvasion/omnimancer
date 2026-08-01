"""
Event schema for Omnimancer session tracking.

This module defines the schema for fleet events used to track session activity
and agent interactions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from omnimancer.core.agent.status_core import OperationType as StatusOperationType
from omnimancer.core.agent.types import OperationType as AgentOperationType

SCHEMA_VERSION = 1
PREVIEW_CHARS = 200
MESSAGE_PREVIEW_CHARS = 500

EVENT_SESSION_START = "session_start"
EVENT_SESSION_END = "session_end"
EVENT_TURN_START = "turn_start"
EVENT_TURN_END = "turn_end"
EVENT_TOOL_START = "tool_start"
EVENT_TOOL_PROGRESS = "tool_progress"
EVENT_TOOL_END = "tool_end"
EVENT_APPROVAL_REQUESTED = "approval_requested"
EVENT_APPROVAL_GRANTED = "approval_granted"
EVENT_APPROVAL_DENIED = "approval_denied"
EVENT_ERROR = "error"

ALL_EVENTS = frozenset(
    [
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
    ]
)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format with milliseconds."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def truncate(text: str, limit: int) -> str:
    """
    Truncate text to the given limit, adding an ellipsis if truncated.

    Args:
        text: Text to truncate
        limit: Maximum length of result

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


@dataclass
class FleetEvent:
    """A fleet event representing an activity in the Omnimancer session."""

    event: str
    session_id: str
    agent_id: str = "main"
    parent_id: Optional[str] = None
    pid: int = field(default_factory=os.getpid)
    seq: int = 0
    mode: str = "interactive"
    cwd: str = field(default_factory=os.getcwd)
    ts: str = field(default_factory=utc_now_iso)
    data: Dict[str, Any] = field(default_factory=dict)
    v: int = SCHEMA_VERSION

    def to_json_line(self) -> str:
        """
        Convert the event to a JSON line format.

        Returns:
            JSON string representation of the event
        """
        return json.dumps(
            {
                "v": self.v,
                "ts": self.ts,
                "event": self.event,
                "session_id": self.session_id,
                "agent_id": self.agent_id,
                "parent_id": self.parent_id,
                "pid": self.pid,
                "seq": self.seq,
                "mode": self.mode,
                "cwd": self.cwd,
                "data": self.data,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


_OP_TYPE_MAP: Dict[AgentOperationType, StatusOperationType] = {
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


def translate_operation_type(op: AgentOperationType) -> StatusOperationType:
    """
    Translate an agent operation type to a status operation type.

    Args:
        op: Agent operation type to translate

    Returns:
        Corresponding status operation type. An unmapped member degrades
        to API_CALL instead of raising: this runs on the agent's tool
        path (emitter.start_tool_operation, outside its try/except), so
        a mapping gap must mislabel an event, never crash the turn.
        test_translate_all_members keeps the map exhaustive in CI.
    """
    return _OP_TYPE_MAP.get(op, StatusOperationType.API_CALL)
