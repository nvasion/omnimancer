"""Durable checkpoints for headless runs (omn -p / omn --resume).

The headless agent loop retransmits the whole conversation each iteration, so
losing the process to a rate limit meant re-buying every token already spent.
The runner saves a checkpoint at the top of each iteration; `--resume
<session-id>` reloads it and continues from the pending message.

Unlike the interactive conversation store, this format is lossless for tool
exchanges: tool_calls, tool_results, and raw_content survive the round trip,
so native tool-history providers replay correctly after a resume.
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import ChatMessage, MessageRole, ToolCall, ToolResultRecord

logger = logging.getLogger(__name__)

CHECKPOINT_DIR_ENV = "OMNIMANCER_CHECKPOINT_DIR"
CHECKPOINT_VERSION = 1

# Session ids become file names; anything else is rejected outright.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass
class HeadlessCheckpoint:
    """Resumable state of a headless run, captured before a provider call."""

    session_id: str
    prompt: str
    # Completed loop iterations; the resumed loop starts at this index.
    iteration: int
    # The message queued for the next provider call ("" = native continuation).
    current_message: str
    # Serialized ChatMessage dicts (see message_to_dict).
    messages: List[Dict[str, Any]]
    tool_log: List[Dict[str, Any]]
    no_tool_nudges: int
    usage: Dict[str, Any]
    provider: str = ""
    model: str = ""
    version: int = CHECKPOINT_VERSION
    created_at: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


def message_to_dict(msg: ChatMessage) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "role": msg.role.value,
        "content": msg.content,
        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
        "model_used": msg.model_used,
    }
    if msg.tool_calls:
        data["tool_calls"] = [
            {
                "name": tc.name,
                "arguments": tc.arguments,
                "server_name": tc.server_name,
                "id": tc.id,
            }
            for tc in msg.tool_calls
        ]
    if msg.raw_content is not None:
        data["raw_content"] = msg.raw_content
    if msg.tool_results:
        data["tool_results"] = [
            {"tool_call_id": tr.tool_call_id, "content": tr.content}
            for tr in msg.tool_results
        ]
    return data


def message_from_dict(data: Dict[str, Any]) -> ChatMessage:
    tool_calls = None
    if data.get("tool_calls"):
        tool_calls = [
            ToolCall(
                name=tc.get("name", ""),
                arguments=tc.get("arguments") or {},
                server_name=tc.get("server_name"),
                id=tc.get("id"),
            )
            for tc in data["tool_calls"]
        ]
    tool_results = None
    if data.get("tool_results"):
        tool_results = [
            ToolResultRecord(
                tool_call_id=tr.get("tool_call_id", ""),
                content=tr.get("content", ""),
            )
            for tr in data["tool_results"]
        ]
    raw_ts = data.get("timestamp")
    try:
        timestamp = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now()
    except ValueError:
        timestamp = datetime.now()
    return ChatMessage(
        role=MessageRole(data["role"]),
        content=data.get("content", ""),
        timestamp=timestamp,
        model_used=data.get("model_used", ""),
        tool_calls=tool_calls,
        raw_content=data.get("raw_content"),
        tool_results=tool_results,
    )


def checkpoint_dir() -> Path:
    override = os.environ.get(CHECKPOINT_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".omnimancer" / "headless_checkpoints"


def checkpoint_path(session_id: str) -> Optional[Path]:
    """Path for a session's checkpoint, or None for unsafe ids."""
    if not _SESSION_ID_RE.match(session_id or ""):
        return None
    return checkpoint_dir() / f"{session_id}.json"


def save_checkpoint(checkpoint: HeadlessCheckpoint) -> Path:
    path = checkpoint_path(checkpoint.session_id)
    if path is None:
        raise ValueError(f"Invalid checkpoint session id: {checkpoint.session_id!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.created_at:
        checkpoint.created_at = checkpoint.updated_at
    # Write-then-rename so a crash mid-write never corrupts the checkpoint.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(checkpoint), default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_checkpoint(session_id: str) -> Optional[HeadlessCheckpoint]:
    path = checkpoint_path(session_id)
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return HeadlessCheckpoint(**data)
    except (ValueError, TypeError, OSError) as exc:
        logger.warning("Failed to load checkpoint %s: %s", path, exc)
        return None


def delete_checkpoint(session_id: str) -> None:
    path = checkpoint_path(session_id)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Failed to delete checkpoint %s: %s", path, exc)
