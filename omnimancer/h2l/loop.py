"""The isolated tool-calling loop shared by the planner, workers and judge.

One loop, three callers (PRD §10, NFR-3): talks to a provider instance
directly with its own :class:`ChatContext`, appends its own history (providers
never mutate the context), executes tool calls through a caller-supplied
executor, and stops on completion, cancellation, a cap, a repeat abort, an
error, or a designated *stop tool* (``submit_plan`` / ``submit_verdict``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterable, List, Optional

from ..cli.tool_handler import DUPLICATE_CALL_NUDGE, RepeatedCallTracker
from ..core.models import (
    ChatContext,
    ChatMessage,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultRecord,
    describe_tool_calls,
    parse_described_tool_calls,
)
from .models import ToolLogEntry, UsageTotals

logger = logging.getLogger(__name__)

ToolExecutor = Callable[[ToolCall], Awaitable[ToolResult]]
ToolObserver = Callable[[ToolCall, ToolResult], Awaitable[None]]
WaitObserver = Callable[[float], Awaitable[None]]

DEFAULT_HEARTBEAT_SECONDS = 30.0

# finish/stop reasons that mean "the output hit the token limit".
_TRUNCATED_REASONS = ("length", "max_tokens")
MAX_TRUNCATION_CONTINUES = 2
TRUNCATED_NOTE = (
    "[note] Your previous output was cut off at the output token limit, so a "
    "tool call may have arrived incomplete. Keep each tool call small: one "
    "item per call."
)
TRUNCATED_CONTINUE = (
    "Your previous output was cut off at the output token limit. Continue, and "
    "keep each tool call small: one item per call."
)

# Tool results are logged for the report; keep each entry bounded.
LOG_RESULT_CHARS = 2000


@dataclass
class LoopResult:
    content: str = ""
    iterations: int = 0
    usage: UsageTotals = field(default_factory=UsageTotals)
    # done | stopped | cancelled | max_iterations | repeat_abort | error
    stop: str = "done"
    error: Optional[str] = None
    tool_log: List[ToolLogEntry] = field(default_factory=list)
    # The model the provider reports having answered with (last non-empty).
    model_used: str = ""
    # How many responses were cut off at the output-token limit.
    truncations: int = 0


class IsolatedLoop:
    """Run a tool-calling conversation to completion on one provider."""

    def __init__(
        self,
        provider: Any,
        context: ChatContext,
        tools: List[ToolDefinition],
        execute: ToolExecutor,
        max_iterations: Optional[int],
        stop_tools: Iterable[str] = (),
        on_tool_call: Optional[ToolObserver] = None,
        on_wait: Optional[WaitObserver] = None,
        heartbeat: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        self.provider = provider
        self.context = context
        self.tools = tools
        self.execute = execute
        # None = unbounded: the loop ends only on completion, a stop tool,
        # cancellation, a repeat abort, or an error.
        self.max_iterations: Optional[int] = (
            None if max_iterations is None else max(1, int(max_iterations))
        )
        self.stop_tools = set(stop_tools)
        self.on_tool_call = on_tool_call
        # Heartbeat: while a provider call is in flight, ``on_wait(elapsed)``
        # fires every ``heartbeat`` seconds so a slow model never looks idle.
        self.on_wait = on_wait
        self.heartbeat = max(0.001, float(heartbeat))

    async def _call_provider(self, message: str) -> Any:
        """Await the provider, firing the heartbeat while it thinks."""
        if self.on_wait is None:
            return await self.provider.send_message_with_tools(
                message, self.context, self.tools
            )
        call = asyncio.ensure_future(
            self.provider.send_message_with_tools(message, self.context, self.tools)
        )
        started = time.monotonic()
        try:
            while True:
                done, _ = await asyncio.wait({call}, timeout=self.heartbeat)
                if done:
                    return call.result()
                try:
                    await self.on_wait(time.monotonic() - started)
                except Exception as exc:  # a display hiccup must not end the call
                    logger.debug("H2L wait heartbeat failed: %s", exc)
        except asyncio.CancelledError:
            call.cancel()
            raise

    def _native_history(self) -> bool:
        getter = getattr(self.provider, "supports_native_tool_history", None)
        if not callable(getter):
            return False
        try:
            return getter() is True
        except Exception:
            return False

    def _model_label(self) -> str:
        raw = getattr(self.provider, "model", "")
        return raw if isinstance(raw, str) else ""

    def _append(self, role: MessageRole, content: str, **extra: Any) -> None:
        self.context.add_message(
            ChatMessage(
                role=role,
                content=content,
                timestamp=datetime.now(),
                model_used=self._model_label(),
                **extra,
            )
        )

    async def run(self, message: str) -> LoopResult:
        result = LoopResult()
        tracker = RepeatedCallTracker()
        native = self._native_history()

        while self.max_iterations is None or result.iterations < self.max_iterations:
            result.iterations += 1
            try:
                response = await self._call_provider(message)
            except Exception as exc:
                result.stop, result.error = "error", str(exc)
                return result
            result.usage = result.usage.add(UsageTotals.from_response(response))
            served = getattr(response, "model_used", "")
            if isinstance(served, str) and served:
                result.model_used = served
            if not response.is_success:
                result.stop = "error"
                result.error = response.error or "provider call failed"
                return result

            # Providers never mutate the context — record both turns here.
            if message:
                self._append(MessageRole.USER, message)
            content = response.content or ""
            note = describe_tool_calls(response.tool_calls)
            recorded = f"{content}\n{note}".strip() if note else content
            self._append(
                MessageRole.ASSISTANT,
                recorded,
                tool_calls=response.tool_calls,
                raw_content=content,
            )
            if content:
                result.content = content

            truncated = str(getattr(response, "stop_reason", "") or "") in (
                _TRUNCATED_REASONS
            )
            if truncated:
                result.truncations += 1

            tool_calls = response.tool_calls or parse_described_tool_calls(content)
            if not tool_calls:
                if truncated and result.truncations <= MAX_TRUNCATION_CONTINUES:
                    # Cut off mid-thought: say so instead of ending the turn.
                    message = TRUNCATED_CONTINUE
                    continue
                result.stop = "done"
                return result

            tracker.record(tool_calls)
            offender = tracker.abort_offender(tool_calls)
            if offender is not None:
                result.stop = "repeat_abort"
                result.error = (
                    f"the model repeated {offender.name} "
                    f"{tracker.count(offender)} times despite warnings"
                )
                return result

            parts: List[str] = []
            records: List[ToolResultRecord] = []
            outcome = ""
            for idx, tc in enumerate(tool_calls):
                if tracker.is_duplicate(tc):
                    tool_result = ToolResult(content=DUPLICATE_CALL_NUDGE)
                else:
                    tool_result = await self.execute(tc)
                result.tool_log.append(
                    ToolLogEntry(
                        name=tc.name,
                        arguments=dict(tc.arguments or {}),
                        result=(tool_result.content or "")[:LOG_RESULT_CHARS],
                        error=tool_result.error,
                    )
                )
                if self.on_tool_call is not None:
                    await self.on_tool_call(tc, tool_result)
                label = f"{tc.name}({json.dumps(tc.arguments, default=str)})"
                if tool_result.error:
                    parts.append(f"[{label}] Error: {tool_result.error}")
                else:
                    parts.append(f"[{label}] Result: {tool_result.content}")
                records.append(
                    ToolResultRecord(
                        tool_call_id=tc.id or f"call_{idx}",
                        content=tool_result.error or tool_result.content,
                    )
                )
                if tool_result.cancelled:
                    outcome = "cancelled"
                    break
                if tc.name in self.stop_tools and not tool_result.error:
                    outcome = "stopped"
                    break

            if truncated:
                parts.append(TRUNCATED_NOTE)
            results_text = "Tool results:\n\n" + "\n\n".join(parts)
            if native:
                self._append(MessageRole.USER, results_text, tool_results=records)
                message = ""
            else:
                message = results_text

            if outcome:
                result.stop = outcome
                return result

        result.stop = "max_iterations"
        return result
