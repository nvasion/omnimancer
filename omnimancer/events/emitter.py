"""Emitter facade: wires the UnifiedStatusManager bus to the JSONL transport.

One `init_events()` call per process attaches a `JsonlEventListener` to the
global status manager; everything the bus emits that maps into the
omn.event.v1 vocabulary is appended to ``<events dir>/<session_id>.jsonl``
by a background writer thread. Emission never blocks a turn: the bus queues
are put_nowait with drop-on-full, and the writer thread owns all file I/O.

Agent identity (main vs subagent) travels via contextvars set with
`agent_context()` at spawn sites, and is captured at emit time — listener
callbacks run in the bus's processing task, whose context never changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from omnimancer.core.agent.status_core import (
    AgentEvent,
    AgentOperation,
    EventListener,
    EventType,
    StreamPriority,
)
from omnimancer.core.agent.status_manager import (
    UnifiedStatusManager,
    initialize_status_system,
    shutdown_status_system,
)
from omnimancer.core.agent.types import OperationType as AgentOperationType
from omnimancer.core.models import EventsConfig
from omnimancer.events.jsonl_writer import JsonlWriter, cleanup_old_files
from omnimancer.events.schema import (
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
    MESSAGE_PREVIEW_CHARS,
    PREVIEW_CHARS,
    FleetEvent,
    translate_operation_type,
    truncate,
)

logger = logging.getLogger(__name__)

ENV_KILL_SWITCH = "OMNIMANCER_EVENTS"

# Retention cleanup deletes ONLY files matching this (session uuid4 names),
# so a user-configured shared directory never loses unrelated .jsonl data.
SESSION_FILE_RE = (
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jsonl$"
)

# Payload fields clipped before serialization; everything else the gate
# sends is already bounded. Targets/descriptions can embed secrets-adjacent
# command text — same exposure class as hook payloads, but capped here.
_CLIP_LIMITS: Dict[str, int] = {
    "target": PREVIEW_CHARS,
    "description": PREVIEW_CHARS,
    "error": MESSAGE_PREVIEW_CHARS,
}


def _clip(data: Dict[str, Any]) -> Dict[str, Any]:
    """Truncate known free-text payload fields in place and return it."""
    for key, limit in _CLIP_LIMITS.items():
        value = data.get(key)
        if isinstance(value, str):
            data[key] = truncate(value, limit)
    return data


# Bus EventType -> omn.event.v1 event name. AGENT_STATE_CHANGED is
# intentionally unmapped: it is not part of the v1 schema.
_EVENT_NAME_BY_TYPE: Dict[EventType, str] = {
    EventType.OPERATION_STARTED: EVENT_TOOL_START,
    EventType.OPERATION_PROGRESS: EVENT_TOOL_PROGRESS,
    EventType.OPERATION_COMPLETED: EVENT_TOOL_END,
    EventType.OPERATION_FAILED: EVENT_TOOL_END,
    EventType.OPERATION_CANCELLED: EVENT_TOOL_END,
    EventType.ERROR_OCCURRED: EVENT_ERROR,
    EventType.APPROVAL_REQUESTED: EVENT_APPROVAL_REQUESTED,
    EventType.APPROVAL_GRANTED: EVENT_APPROVAL_GRANTED,
    EventType.APPROVAL_DENIED: EVENT_APPROVAL_DENIED,
    EventType.SESSION_START: EVENT_SESSION_START,
    EventType.SESSION_END: EVENT_SESSION_END,
    EventType.TURN_START: EVENT_TURN_START,
    EventType.TURN_END: EVENT_TURN_END,
}

_current_agent_id: ContextVar[str] = ContextVar("omn_agent_id", default="main")
_current_parent_id: ContextVar[Optional[str]] = ContextVar(
    "omn_parent_id", default=None
)


def current_agent_id() -> str:
    """Return the agent id for the current execution context."""
    return _current_agent_id.get()


def current_parent_id() -> Optional[str]:
    """Return the parent agent id for the current execution context."""
    return _current_parent_id.get()


@contextlib.contextmanager
def agent_context(agent_id: str, parent_id: Optional[str] = "main") -> Iterator[None]:
    """Scope subsequent emissions to a (sub)agent identity.

    Args:
        agent_id: Identity for events emitted inside the scope.
        parent_id: The spawning agent's id (None for top-level).
    """
    token_agent = _current_agent_id.set(agent_id)
    token_parent = _current_parent_id.set(parent_id)
    try:
        yield
    finally:
        _current_agent_id.reset(token_agent)
        _current_parent_id.reset(token_parent)


class JsonlEventListener(EventListener):
    """Bus listener that serializes mapped events to the JSONL writer."""

    def __init__(self, session_id: str, mode: str, writer: JsonlWriter) -> None:
        """Initialize with session identity and a running writer."""
        super().__init__()
        self.session_id = session_id
        self.mode = mode
        self.writer = writer
        self._seq = 0

    async def _process_event(self, event: AgentEvent) -> None:
        """Translate a bus event into one JSONL line (never raises)."""
        try:
            name = _EVENT_NAME_BY_TYPE.get(event.event_type)
            if name is None:
                return
            data = dict(event.data or {})
            metadata = data.pop("metadata", None)
            if isinstance(metadata, dict):
                data.update(metadata)
            parent_id = data.pop("parent_id", None)
            if event.event_type == EventType.OPERATION_COMPLETED:
                data.setdefault("success", True)
            elif event.event_type == EventType.OPERATION_FAILED:
                data.setdefault("success", False)
            elif event.event_type == EventType.OPERATION_CANCELLED:
                data.setdefault("success", False)
                data.setdefault("was_cancelled", True)
            if event.operation_id:
                data.setdefault("op_id", event.operation_id)
            fleet_event = FleetEvent(
                event=name,
                session_id=self.session_id,
                agent_id=event.agent_id or "main",
                parent_id=parent_id,
                seq=self._seq,
                mode=self.mode,
                data=data,
            )
            self._seq += 1
            self.writer.enqueue(fleet_event.to_json_line())
        except Exception as exc:
            logger.debug(f"Event serialization failed: {exc}")


@dataclass
class _EmitterState:
    """Process-global emitter wiring; a plain holder, not public API."""

    manager: Optional[UnifiedStatusManager] = None
    writer: Optional[JsonlWriter] = None
    listener: Optional[JsonlEventListener] = None
    loop: Optional[asyncio.AbstractEventLoop] = None


_state = _EmitterState()


def events_enabled() -> bool:
    """True when init_events() succeeded and the bus is running."""
    return _state.manager is not None and _state.manager.running


def register_listener(listener: EventListener) -> bool:
    """Attach an extra in-process listener (e.g. the REPL activity panel).

    Returns:
        True when attached; False when the pipeline is down.
    """
    if not events_enabled() or _state.manager is None:
        return False
    _state.manager.add_event_listener(listener)
    return True


def default_events_dir() -> Path:
    """Return the default events directory (~/.omnimancer/events)."""
    return Path.home() / ".omnimancer" / "events"


async def init_events(
    session_id: str, mode: str, config: Optional[EventsConfig] = None
) -> bool:
    """Initialize the event pipeline for this process.

    Args:
        session_id: The session uuid used for the JSONL filename and envelope.
        mode: "interactive" or "headless".
        config: Events settings; defaults apply when None.

    Returns:
        True when the pipeline is live; False when disabled or already up.
    """
    if _state.manager is not None:
        return False
    cfg = config or EventsConfig()
    # Strict type check: a mock or malformed config object must never reach
    # Path() below — MagicMock's __fspath__ yields a relative garbage path
    # that would be created for real in the process cwd.
    if not isinstance(cfg, EventsConfig):
        return False
    if not cfg.enabled or os.environ.get(ENV_KILL_SWITCH, "") == "0":
        return False
    try:
        directory = (
            Path(cfg.directory).expanduser() if cfg.directory else default_events_dir()
        )

        def _cap_notice() -> str:
            # Built at cap time so the timestamp is honest; seq=-1 marks it
            # as an out-of-band writer notice, not a bus-ordered event.
            return FleetEvent(
                event=EVENT_ERROR,
                session_id=session_id,
                mode=mode,
                seq=-1,
                data={"message": "event file size cap reached"},
            ).to_json_line()

        def _startup_cleanup() -> None:
            # Runs on the writer thread — init_events does no filesystem
            # I/O on the event loop.
            cleanup_old_files(directory, cfg.retention_days, name_re=SESSION_FILE_RE)

        writer = JsonlWriter(
            directory / f"{session_id}.jsonl",
            max_file_mb=cfg.max_file_mb,
            cap_notice=_cap_notice,
            on_start=_startup_cleanup,
        )
        manager = await initialize_status_system()
        listener = JsonlEventListener(session_id, mode, writer)
        manager.add_event_listener(listener)
        _state.manager = manager
        _state.writer = writer
        _state.listener = listener
        _state.loop = asyncio.get_running_loop()
        return True
    except Exception as exc:
        # Fail open: a broken event pipeline must never break the CLI.
        logger.debug(f"Event pipeline init failed: {exc}")
        return False


async def emit_event(
    event_type: EventType,
    data: Optional[Dict[str, Any]] = None,
    *,
    operation_id: Optional[str] = None,
    priority: Optional[StreamPriority] = None,
) -> None:
    """Emit one event onto the bus (no-op when the pipeline is down)."""
    if not events_enabled() or _state.manager is None:
        return
    payload = _clip(dict(data or {}))
    parent_id = current_parent_id()
    if parent_id is not None:
        payload["parent_id"] = parent_id
    event = AgentEvent(
        event_type=event_type,
        agent_id=current_agent_id(),
        operation_id=operation_id,
        data=payload,
        source="emitter",
    )
    try:
        await _state.manager.emit_event(event, priority)
    except Exception as exc:
        logger.debug(f"Event emission failed: {exc}")


def emit_event_threadsafe(
    event_type: EventType,
    data: Optional[Dict[str, Any]] = None,
    *,
    operation_id: Optional[str] = None,
) -> None:
    """Emit from a non-loop thread (e.g. process output readers)."""
    if not events_enabled() or _state.loop is None or _state.loop.is_closed():
        return
    payload = _clip(dict(data or {}))
    parent_id = current_parent_id()
    if parent_id is not None:
        payload["parent_id"] = parent_id
    event = AgentEvent(
        event_type=event_type,
        agent_id=current_agent_id(),
        operation_id=operation_id,
        data=payload,
        source="emitter",
    )
    manager = _state.manager

    def _schedule() -> None:
        if manager is not None and manager.running:
            asyncio.ensure_future(manager.emit_event(event))

    try:
        _state.loop.call_soon_threadsafe(_schedule)
    except RuntimeError:
        # Loop already closed during shutdown; drop silently.
        pass


def build_progress_callback(
    operation_id: Optional[str], max_per_sec: float = 4.0
) -> Optional[Any]:
    """Return a throttled sync callback emitting tool_progress, or None.

    None (pipeline down or no operation id) tells the caller to keep its
    non-streaming execution path — zero behavior change when the feed is
    off. The callback is thread-agnostic: it uses the threadsafe emit path.

    Args:
        operation_id: The bus operation to attach progress to.
        max_per_sec: Emission ceiling; intermediate chunks are coalesced
            into the running byte count.
    """
    if operation_id is None or not events_enabled():
        return None
    min_interval = 1.0 / max_per_sec
    state = {"last_emit": 0.0, "total_bytes": 0}

    def _callback(stream_type: str, content: str) -> None:
        state["total_bytes"] += len(content)
        now = time.monotonic()
        if now - state["last_emit"] < min_interval:
            return
        state["last_emit"] = now
        emit_event_threadsafe(
            EventType.OPERATION_PROGRESS,
            {
                "stream": stream_type,
                "bytes": state["total_bytes"],
                "preview": truncate(content, PREVIEW_CHARS),
            },
            operation_id=operation_id,
        )

    return _callback


async def start_tool_operation(
    operation_type: AgentOperationType,
    description: str,
    data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Begin a bus-tracked tool operation (emits tool_start).

    Returns:
        The operation id for the matching end_tool_operation() call,
        or None when the pipeline is disabled.
    """
    if not events_enabled() or _state.manager is None:
        return None
    metadata = _clip(dict(data or {}))
    parent_id = current_parent_id()
    if parent_id is not None:
        metadata["parent_id"] = parent_id
    operation = AgentOperation(
        operation_type=translate_operation_type(operation_type),
        description=description,
        agent_id=current_agent_id(),
        metadata=metadata,
    )
    try:
        return await _state.manager.start_operation(operation)
    except Exception as exc:
        logger.debug(f"start_tool_operation failed: {exc}")
        return None


async def end_tool_operation(
    operation_id: Optional[str],
    *,
    success: bool,
    error: Optional[str] = None,
    was_cancelled: bool = False,
) -> None:
    """Close a bus-tracked tool operation (emits tool_end)."""
    if operation_id is None or not events_enabled() or _state.manager is None:
        return
    clipped_error = (
        truncate(error, MESSAGE_PREVIEW_CHARS) if isinstance(error, str) else error
    )
    try:
        if was_cancelled:
            await _state.manager.cancel_operation(
                operation_id, clipped_error or "cancelled"
            )
        elif success:
            await _state.manager.complete_operation(operation_id)
        else:
            await _state.manager.fail_operation(operation_id, clipped_error or "failed")
    except Exception as exc:
        logger.debug(f"end_tool_operation failed: {exc}")


async def shutdown_events(timeout: float = 1.0) -> None:
    """Stop the bus (which closes out active operations) and the writer.

    Order matters: the manager's shutdown emits terminal
    OPERATION_CANCELLED events for anything still active and drains its
    queue while the listener is attached — only then is the writer flushed.
    """
    manager = _state.manager
    writer = _state.writer
    _state.manager = None
    _state.writer = None
    _state.listener = None
    _state.loop = None
    if manager is None:
        return
    try:
        await shutdown_status_system()
    except Exception as exc:
        logger.debug(f"Event pipeline shutdown error: {exc}")
    finally:
        if writer is not None:
            writer.flush(timeout)
            writer.shutdown(timeout)
