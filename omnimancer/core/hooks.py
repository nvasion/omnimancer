"""Lifecycle hooks for Omnimancer.

A *hook* is a user-configured shell command that fires on a lifecycle event
(see :class:`~omnimancer.core.models.HooksConfig`). Hooks serve two purposes:

* **Observe** — react to events for logging, notifications, metrics, etc.
* **Gate** — a ``blocking`` hook that exits non-zero (or times out) vetoes the
  action that triggered it (e.g. refuse to send a message or run a tool).

The event payload is passed to the hook as JSON on **stdin**, and a few scalar
fields are also exported as ``OMNIMANCER_HOOK_*`` environment variables for
convenience in simple shell one-liners.

Design rules:

* Hooks must never crash the host. Every subprocess error is caught; only an
  explicit non-zero exit from a ``blocking`` hook affects control flow, and it
  does so through the returned :class:`HookOutcome`, never by raising.
* Firing an event with no configured (and enabled) hooks is a cheap no-op.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import HookCommand, HooksConfig

logger = logging.getLogger(__name__)


@dataclass
class HookResult:
    """Outcome of running a single hook command."""

    name: str
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: Optional[str] = None
    # True when this hook vetoed the action (blocking + failed).
    blocked: bool = False

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None


@dataclass
class HookOutcome:
    """Aggregate result of firing all hooks for an event."""

    event: str
    allowed: bool = True
    results: List[HookResult] = field(default_factory=list)

    @property
    def reason(self) -> str:
        """Human-readable explanation when the action was vetoed."""
        for r in self.results:
            if r.blocked:
                detail = (r.stderr or r.stdout or r.error or "").strip()
                suffix = f": {detail}" if detail else ""
                return f"hook '{r.name}'{suffix}"
        return ""


class HooksManager:
    """Runs configured hooks for lifecycle events."""

    def __init__(self, config: Optional[HooksConfig]) -> None:
        # Tolerate a missing/None config (e.g. partially-mocked engines) by
        # falling back to an empty, disabled-by-emptiness configuration.
        self._config = config if isinstance(config, HooksConfig) else HooksConfig()

    def _selected(self, event: str, match_target: str) -> List[HookCommand]:
        """Enabled hooks for ``event`` whose matcher accepts ``match_target``."""
        if not self._config.enabled:
            return []
        selected = []
        for hook in self._config.hooks_for(event):
            if not hook.enabled:
                continue
            if hook.matcher:
                try:
                    if not re.search(hook.matcher, match_target):
                        continue
                except re.error as e:
                    logger.warning(
                        "Hook '%s' has an invalid matcher %r: %s",
                        hook.name,
                        hook.matcher,
                        e,
                    )
                    continue
            selected.append(hook)
        return selected

    async def fire(
        self,
        event: str,
        context: Optional[Dict[str, Any]] = None,
        match_target: str = "",
    ) -> HookOutcome:
        """Run every matching hook for ``event``.

        Args:
            event: Lifecycle event name (e.g. ``"pre_send_message"``).
            context: JSON-serialisable payload passed to hooks on stdin.
            match_target: String tested against each hook's ``matcher`` regex.

        Returns:
            A :class:`HookOutcome`. ``allowed`` is False when a blocking hook
            failed, in which case the caller should abort the action.
        """
        outcome = HookOutcome(event=event)
        hooks = self._selected(event, match_target)
        if not hooks:
            return outcome

        payload = dict(context or {})
        payload.setdefault("event", event)
        stdin_data = self._safe_json(payload)
        env_extra = self._env_for(event, payload)

        for hook in hooks:
            result = await self._run_one(hook, stdin_data, env_extra)
            outcome.results.append(result)
            if result.blocked:
                outcome.allowed = False
        return outcome

    async def _run_one(
        self, hook: HookCommand, stdin_data: str, env_extra: Dict[str, str]
    ) -> HookResult:
        """Execute a single hook command, never raising."""
        import os

        result = HookResult(name=hook.name)
        env = {**os.environ, **env_extra}
        try:
            process = await asyncio.create_subprocess_shell(
                hook.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except Exception as e:  # pragma: no cover - spawn failure is rare
            result.error = f"failed to start hook: {e}"
            result.blocked = hook.blocking
            logger.warning("Hook '%s' failed to start: %s", hook.name, e)
            return result

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(input=stdin_data.encode()),
                timeout=hook.timeout,
            )
        except asyncio.TimeoutError:
            result.timed_out = True
            result.blocked = hook.blocking
            await self._kill(process)
            logger.warning("Hook '%s' timed out after %ss", hook.name, hook.timeout)
            return result
        except Exception as e:  # pragma: no cover - unexpected runtime failure
            result.error = str(e)
            result.blocked = hook.blocking
            logger.warning("Hook '%s' errored: %s", hook.name, e)
            return result

        result.returncode = process.returncode
        result.stdout = stdout_b.decode(errors="replace")
        result.stderr = stderr_b.decode(errors="replace")
        if not result.succeeded:
            result.blocked = hook.blocking
            logger.info(
                "Hook '%s' exited %s%s",
                hook.name,
                result.returncode,
                " (blocking)" if hook.blocking else "",
            )
        return result

    @staticmethod
    async def _kill(process: "asyncio.subprocess.Process") -> None:
        try:
            process.kill()
        except ProcessLookupError:  # pragma: no cover - already gone
            return
        try:
            # Drain the process within the running loop so its transport is
            # closed now, instead of being finalised after the loop shuts down
            # (which logs a spurious "Event loop is closed" warning).
            await process.wait()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    @staticmethod
    def _safe_json(payload: Dict[str, Any]) -> str:
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            return json.dumps({"event": payload.get("event", "")})

    @staticmethod
    def _env_for(event: str, payload: Dict[str, Any]) -> Dict[str, str]:
        """Export scalar payload fields as OMNIMANCER_HOOK_* env vars."""
        env = {"OMNIMANCER_HOOK_EVENT": event}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)):
                env[f"OMNIMANCER_HOOK_{key.upper()}"] = str(value)
        return env
