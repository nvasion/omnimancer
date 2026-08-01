"""Claude Code hook adapter for the omnimancer fleet event feed.

Registered in ``~/.claude/settings.json`` (all events, ``async: true``) as
the ``omn-fleet-hook`` console script. Each invocation reads ONE hook
payload from stdin, maps it to an omn.event.v1 line, and appends it to
``~/.omnimancer/events/omn-<session_id>.jsonl`` — the same files the
``omn fleet`` dashboard tails, so Claude Code sessions appear on the board
next to omn and codex agents.

DELIBERATELY a top-level, stdlib-only module: importing the omnimancer
package pulls pydantic and friends, and this process starts once per hook
event. Schema constants are duplicated from ``omnimancer/events/schema.py``
(the owner of the omn.event.v1 vocabulary) — keep them in sync.

Never blocks, never fails: any error exits 0 silently (a monitoring hook
must not be able to break a Claude session). ``OMNIMANCER_EVENTS=0``
disables it; ``OMNIMANCER_EVENTS_DIR`` overrides the output directory.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Owner of these values: omnimancer/events/schema.py (omn.event.v1).
SCHEMA_VERSION = 1
PREVIEW_CHARS = 200
MESSAGE_PREVIEW_CHARS = 500

MAX_STDIN_BYTES = 1_000_000
SESSION_ID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")
TARGET_KEYS = ("command", "file_path", "path", "url", "pattern")

# Claude hook_event_name -> omn.event.v1 event name.
EVENT_MAP = {
    "SessionStart": "session_start",
    "UserPromptSubmit": "turn_start",
    "PreToolUse": "tool_start",
    "PostToolUse": "tool_end",
    "PostToolUseFailure": "tool_end",
    "Stop": "turn_end",
    "SessionEnd": "session_end",
}


def _truncate(text: str, limit: int) -> str:
    """Truncate to limit chars with a single-char ellipsis (schema rule)."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _utc_now_iso() -> str:
    """UTC ISO8601 with millisecond precision (schema rule)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _events_dir() -> str:
    """Resolve the events directory (env override for tests/relocation)."""
    override = os.environ.get("OMNIMANCER_EVENTS_DIR")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".omnimancer", "events")


def _target_from(tool_input: Any) -> Optional[str]:
    """First actionable string from a tool_input dict, clipped."""
    if not isinstance(tool_input, dict):
        return None
    for key in TARGET_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return _truncate(value, PREVIEW_CHARS)
    return None


def _build_data(payload: Dict[str, Any], event: str) -> Dict[str, Any]:
    """Per-event data payload (fields mirror the omn emitters)."""
    hook_event = payload.get("hook_event_name")
    if event == "session_start":
        return {
            "harness": "claude",
            "model": payload.get("model"),
            "source": payload.get("source"),
            "permission_mode": payload.get("permission_mode"),
        }
    if event == "turn_start":
        prompt = payload.get("prompt_text")
        return {
            "prompt_preview": (
                _truncate(prompt, PREVIEW_CHARS) if isinstance(prompt, str) else None
            ),
        }
    if event == "tool_start":
        return {
            "tool": payload.get("tool_name"),
            "target": _target_from(payload.get("tool_input")),
            "op_id": payload.get("tool_use_id"),
            "invocation": "claude",
        }
    if event == "tool_end":
        data: Dict[str, Any] = {
            "tool": payload.get("tool_name"),
            "op_id": payload.get("tool_use_id"),
            "success": hook_event == "PostToolUse",
        }
        if hook_event == "PostToolUseFailure":
            output = payload.get("tool_output")
            if isinstance(output, str):
                data["error"] = _truncate(output, MESSAGE_PREVIEW_CHARS)
        return data
    if event == "turn_end":
        message = payload.get("last_assistant_message")
        return {
            "last_message_preview": (
                _truncate(message, MESSAGE_PREVIEW_CHARS)
                if isinstance(message, str)
                else None
            ),
        }
    if event == "session_end":
        return {"reason": payload.get("reason")}
    return {}


def _agent_identity(payload: Dict[str, Any]) -> tuple:
    """(agent_id, parent_id): subagent payloads carry agent_id/agent_type."""
    agent_id = payload.get("agent_id")
    if isinstance(agent_id, str) and agent_id:
        agent_type = payload.get("agent_type") or "agent"
        return f"subagent-{agent_type}-{agent_id[:8]}", "main"
    return "main", None


def build_line(payload: Dict[str, Any]) -> Optional[str]:
    """Map one Claude hook payload to an omn.event.v1 JSON line, or None."""
    event = EVENT_MAP.get(str(payload.get("hook_event_name")))
    if event is None:
        return None
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not SESSION_ID_RE.match(session_id):
        return None
    agent_id, parent_id = _agent_identity(payload)
    envelope = {
        "v": SCHEMA_VERSION,
        "ts": _utc_now_iso(),
        "event": event,
        "session_id": session_id.lower(),
        "agent_id": agent_id,
        "parent_id": parent_id,
        "pid": os.getpid(),
        # -1: out-of-band producer (no shared counter across hook processes);
        # consumers order by file-append position and ts.
        "seq": -1,
        "mode": "claude",
        "cwd": payload.get("cwd"),
        "data": _build_data(payload, event),
    }
    return json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    """Read one payload from stdin, append one line; always exit 0."""
    try:
        if os.environ.get("OMNIMANCER_EVENTS", "") == "0":
            return 0
        raw = sys.stdin.buffer.read(MAX_STDIN_BYTES)
        payload = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return 0
        line = build_line(payload)
        if line is None:
            return 0
        directory = _events_dir()
        os.makedirs(directory, mode=0o700, exist_ok=True)
        path = os.path.join(directory, f"omn-{payload['session_id'].lower()}.jsonl")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        # A monitoring hook must never surface an error into the session.
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
