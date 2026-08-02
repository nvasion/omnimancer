"""Turn-completion notifications shared by interactive and headless modes."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shlex
import signal
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_NOTIFY_TIMEOUT_SECONDS = 10.0


@dataclass
class TurnUsage:
    """Token and cost values included in a turn-completion payload."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0

    @classmethod
    def from_response(cls, response: Any) -> "TurnUsage":
        """Create usage values from a provider response.

        Args:
            response: Object exposing ChatResponse-compatible usage fields.

        Returns:
            Normalized integer token counts and floating-point cost.
        """
        if response is None:
            return cls()
        input_tokens = getattr(response, "input_tokens", None)
        output_tokens = getattr(response, "output_tokens", None)
        cost = getattr(response, "cost_estimate", None)
        return cls(
            input_tokens=int(input_tokens) if isinstance(input_tokens, int) else 0,
            output_tokens=int(output_tokens) if isinstance(output_tokens, int) else 0,
            total_cost_usd=(float(cost) if isinstance(cost, (int, float)) else 0.0),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the exact usage mapping expected by notify consumers."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


class TurnNotifier:
    """Record the latest assistant response and notify an external command."""

    def __init__(self, notify_cmd: Optional[str], cwd: str) -> None:
        """Initialize a notifier for one Omnimancer session.

        Args:
            notify_cmd: Command invoked when a turn completes, or None to disable it.
            cwd: Working directory reported in notification payloads.
        """
        self.notify_cmd = notify_cmd
        self.cwd = cwd
        self.session_id = str(uuid.uuid4())
        self._last_assistant_message: Optional[str] = None
        self._usage = TurnUsage()

    def reset_turn(self) -> None:
        """Clear response state before processing a new turn."""
        self._last_assistant_message = None
        self._usage = TurnUsage()

    def record_assistant(self, content: Optional[str], response: Any) -> None:
        """Store the latest assistant text and its usage.

        Args:
            content: Assistant response text, or None when no text was produced.
            response: Object exposing ChatResponse-compatible usage fields.
        """
        self._last_assistant_message = content
        self._usage = TurnUsage.from_response(response)

    def build_payload(self) -> Dict[str, Any]:
        """Build a turn payload with a fresh per-turn identifier."""
        return {
            "type": "agent-turn-complete",
            "turn-id": str(uuid.uuid4()),
            "last-assistant-message": self._last_assistant_message,
            "session_id": self.session_id,
            "usage": self._usage.as_dict(),
            "cwd": self.cwd,
        }

    async def fire(self) -> Dict[str, Any]:
        """Invoke the configured notifier, swallowing every failure.

        Returns:
            The payload built for this turn, whether or not a command ran.
        """
        payload = self.build_payload()
        if not self.notify_cmd:
            return payload

        process: Optional[asyncio.subprocess.Process] = None
        try:
            argv = shlex.split(self.notify_cmd) + [json.dumps(payload)]
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            await asyncio.wait_for(
                process.communicate(), timeout=_NOTIFY_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.debug("Turn notification timed out; killing the command")
            if process is not None:
                await self._kill(process)
        except BaseException as exc:
            logger.debug("Turn notification failed: %s", exc, exc_info=True)
            if process is not None and process.returncode is None:
                await self._kill(process)
        return payload

    @staticmethod
    async def _kill(process: asyncio.subprocess.Process) -> None:
        """Kill and reap the notifier's complete process group without raising."""
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows fallback
                process.kill()
        except ProcessLookupError:
            pass
        except BaseException as exc:
            logger.debug("Could not kill turn notification process group: %s", exc)
        try:
            await process.wait()
        except BaseException as exc:
            logger.debug("Could not reap turn notification command: %s", exc)


async def fire_turn_complete(engine: Any, notifier: TurnNotifier) -> Dict[str, Any]:
    """Fire an external notification and the observe-only completion hooks.

    Args:
        engine: CoreEngine-compatible object exposing the existing hook method.
        notifier: Session notifier holding the final response state.

    Returns:
        The shared payload delivered to both notification mechanisms.
    """
    payload = await notifier.fire()
    try:
        fire_hook = getattr(engine, "_fire_hook", None)
        if callable(fire_hook):
            result = fire_hook("turn_complete", payload)
            if inspect.isawaitable(result):
                await result
    except Exception as exc:
        logger.debug("Turn-complete hook firing failed: %s", exc, exc_info=True)
    return payload
