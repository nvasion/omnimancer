"""Headless pipe mode for Omnimancer (omn -p).

Runs a single prompt through the coding agent and outputs structured
results to stdout. No Rich console, no readline, no interactive UI.
"""

import json
import logging
import os
import sys
from enum import Enum
from typing import Any, Optional

from ..core.agent.status_core import EventType
from ..core.models import (
    ChatResponse,
    ToolResult,
    ToolResultRecord,
    parse_described_tool_calls,
)
from ..core.rate_limit_fallback import matches_rate_limit
from ..events import emitter as fleet_events
from ..events.schema import MESSAGE_PREVIEW_CHARS, PREVIEW_CHARS, truncate
from .headless_checkpoint import (
    HeadlessCheckpoint,
    delete_checkpoint,
    load_checkpoint,
    message_from_dict,
    message_to_dict,
    save_checkpoint,
)
from .session import apply_session_overrides
from .system_prompts import build_agent_prompt
from .tool_handler import (
    DUPLICATE_CALL_NUDGE,
    MAX_TOOL_ITERATIONS,
    RepeatedCallTracker,
    ToolHandler,
)
from .turn_notify import TurnNotifier, fire_turn_complete
from .usage import TokenAccumulator  # noqa: F401  (re-export; moved to usage.py)

logger = logging.getLogger(__name__)

# A turn with no tool calls used to end the run immediately. Models that
# narrate their plan ("Let me look into X...") without attaching a tool call
# made headless runs "complete" with zero work done, so the runner now nudges
# the model to either keep working or explicitly declare completion.
NO_TOOL_CALL_NUDGE = (
    "You did not call any tools. If the task is fully complete, reply with "
    "exactly DONE. Otherwise continue working now by calling tools — do not "
    "describe what you plan to do without doing it."
)

# Consecutive tool-less turns tolerated before the run is ended anyway.
MAX_NO_TOOL_NUDGES = 2

# Total characters of tool-result history kept verbatim in the conversation.
# The full history is retransmitted to the provider on every loop iteration,
# so unbounded accumulation of 16 KB tool results grows token cost
# quadratically with run length. Once the budget is exceeded, the OLDEST
# results are replaced with a short stub — the model can always re-run a tool
# if it needs elided output again. The newest result batch is never elided.
TOOL_RESULT_HISTORY_BUDGET = 60_000

# Stub that replaces an elided tool result. Also serves as the marker that a
# message has already been elided (so it is not re-processed every iteration).
ELIDED_RESULT_STUB = (
    "[Older tool output elided to save context — re-run the tool if you "
    "need it again.]"
)


def _resolve_tool_result_budget() -> int:
    """Resolve the tool-result history budget: the OMNIMANCER_TOOL_RESULT_BUDGET
    environment variable wins, then the default."""
    raw = os.environ.get("OMNIMANCER_TOOL_RESULT_BUDGET")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
            logger.warning(
                "Ignoring non-positive OMNIMANCER_TOOL_RESULT_BUDGET %r", raw
            )
        except ValueError:
            logger.warning("Ignoring invalid OMNIMANCER_TOOL_RESULT_BUDGET %r", raw)
    return TOOL_RESULT_HISTORY_BUDGET


def _message_tool_result_size(msg: Any) -> int:
    """Characters of tool-result payload a history message carries (0 if none)."""
    records = getattr(msg, "tool_results", None)
    if records:
        return sum(len(r.content or "") for r in records)
    content = getattr(msg, "content", "") or ""
    if getattr(msg, "role", None) is not None and msg.role.value == "user":
        if content.startswith("Tool results:"):
            return len(content)
    return 0


def _elide_stale_tool_results(context: Any, budget: int) -> None:
    """Replace old tool-result payloads in the conversation with a stub.

    Walks history newest-first, keeping tool results verbatim until `budget`
    characters are retained; everything older is elided in place. Covers both
    the native path (ToolResultRecord list on a user message) and the
    text-protocol path ("Tool results:" user messages). Already-elided
    messages are skipped via the stub marker.
    """
    kept = 0
    for msg in reversed(context.messages):
        size = _message_tool_result_size(msg)
        if size == 0:
            continue
        content = getattr(msg, "content", "") or ""
        if ELIDED_RESULT_STUB in content:
            continue
        # Always keep the newest result batch, then keep within budget.
        if kept == 0 or kept + size <= budget:
            kept += size
            continue
        records = getattr(msg, "tool_results", None)
        if records:
            for record in records:
                record.content = ELIDED_RESULT_STUB
        msg.content = ELIDED_RESULT_STUB


def _is_done_reply(content: Optional[str]) -> bool:
    """True when the model explicitly declares completion per the nudge."""
    if not content:
        return False
    lines = content.strip().splitlines()
    if not lines:
        return False
    return lines[-1].strip().rstrip(".!").upper() == "DONE"


def _resolve_max_iterations(explicit: Optional[int]) -> int:
    """Resolve the tool-iteration cap: explicit value wins, then the
    OMNIMANCER_MAX_ITERATIONS environment variable, then the default.

    The default (25) suits interactive use; orchestrators running real
    coding tasks inject a higher cap — exhausting it mid-task previously
    looked like a clean "completed" run with no changes.
    """
    if explicit:
        return explicit
    raw = os.environ.get("OMNIMANCER_MAX_ITERATIONS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
            logger.warning("Ignoring non-positive OMNIMANCER_MAX_ITERATIONS %r", raw)
        except ValueError:
            logger.warning("Ignoring invalid OMNIMANCER_MAX_ITERATIONS %r", raw)
    return MAX_TOOL_ITERATIONS


def _checkpointing_enabled() -> bool:
    """Checkpointing is on by default; OMNIMANCER_CHECKPOINT=0 disables it."""
    raw = os.environ.get("OMNIMANCER_CHECKPOINT", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


class HeadlessOutputEmitter:
    """Emits structured output to stdout in the configured format."""

    def __init__(self, fmt: OutputFormat, session_id: str, verbose: bool = False):
        self._format = fmt
        self._session_id = session_id
        self._verbose = verbose
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._last_content = ""

    def _write_json_line(self, data: dict) -> None:
        data["session_id"] = self._session_id
        self._stdout.write(json.dumps(data, default=str) + "\n")
        self._stdout.flush()

    def emit_init(self, model: str) -> None:
        if self._format == OutputFormat.STREAM_JSON:
            self._write_json_line(
                {
                    "type": "system",
                    "subtype": "init",
                    "model": model,
                }
            )

    def emit_assistant(
        self,
        content: str,
        model: str,
        stop_reason: Optional[str],
    ) -> None:
        self._last_content = content
        if self._format == OutputFormat.TEXT:
            self._stdout.write(content + "\n")
            self._stdout.flush()
        elif self._format == OutputFormat.STREAM_JSON:
            self._write_json_line(
                {
                    "type": "assistant",
                    "message": {
                        "model": model,
                        "content": content,
                        "stop_reason": stop_reason,
                    },
                }
            )

    def emit_tool_use(
        self,
        name: str,
        arguments: dict,
    ) -> None:
        if self._format == OutputFormat.TEXT and self._verbose:
            args_json = json.dumps(arguments)
            self._stdout.write(f"[tool] {name} {args_json}\n")
            self._stdout.flush()
        elif self._format == OutputFormat.STREAM_JSON:
            self._write_json_line(
                {
                    "type": "tool_use",
                    "tool": {"name": name, "arguments": arguments},
                }
            )

    def emit_tool_result(
        self,
        name: str,
        content: str,
        error: Optional[str],
    ) -> None:
        if self._format == OutputFormat.TEXT and self._verbose:
            status = f"error: {error}" if error else "ok"
            self._stdout.write(f"[result] {name}: {status}\n")
            self._stdout.flush()
        elif self._format == OutputFormat.STREAM_JSON:
            self._write_json_line(
                {
                    "type": "tool_result",
                    "tool": {
                        "name": name,
                        "content": content,
                        "error": error,
                    },
                }
            )

    def emit_result(
        self,
        content: str,
        model: str,
        usage: dict,
        cost: float,
        stop_reason: Optional[str],
        tool_calls: Optional[list] = None,
        num_turns: int = 0,
        stop_cause: Optional[str] = None,
        provider: str = "",
    ) -> None:
        if self._format == OutputFormat.TEXT:
            if content != self._last_content:
                self._stdout.write(content + "\n")
                self._stdout.flush()
            # stdout stays pure payload; truncation is a stderr concern.
            if stop_cause in ("max_iterations", "repeat_abort"):
                self._stderr.write(
                    f"Warning: run ended early (stop_cause={stop_cause}); "
                    "the result is partial\n"
                )
                self._stderr.flush()
        elif self._format == OutputFormat.JSON:
            blob = {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": content,
                "session_id": self._session_id,
                "model": model,
                "provider": provider,
                "num_turns": num_turns,
                "tool_calls": tool_calls or [],
                "usage": usage,
                "total_cost_usd": cost,
                "stop_reason": stop_reason,
                "stop_cause": stop_cause,
            }
            self._stdout.write(json.dumps(blob, default=str) + "\n")
            self._stdout.flush()
        elif self._format == OutputFormat.STREAM_JSON:
            self._write_json_line(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": content,
                    "model": model,
                    "provider": provider,
                    "num_turns": num_turns,
                    "usage": usage,
                    "total_cost_usd": cost,
                    "stop_reason": stop_reason,
                    "stop_cause": stop_cause,
                }
            )

    def emit_error(
        self,
        message: str,
        tool_calls: Optional[list] = None,
        model: str = "",
        provider: str = "",
        stop_cause: Optional[str] = None,
        resume_session_id: Optional[str] = None,
    ) -> None:
        # Always surface a human-readable line on stderr (stdout stays
        # reserved for the structured result in JSON modes).
        self._stderr.write(f"Error: {message}\n")
        self._stderr.flush()

        if self._format == OutputFormat.JSON:
            self._stdout.write(
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "error",
                        "is_error": True,
                        "error": message,
                        "session_id": self._session_id,
                        "model": model,
                        "provider": provider,
                        "tool_calls": tool_calls or [],
                        "stop_cause": stop_cause,
                        "resume_session_id": resume_session_id,
                    },
                    default=str,
                )
                + "\n"
            )
            self._stdout.flush()
        elif self._format == OutputFormat.STREAM_JSON:
            self._write_json_line(
                {
                    "type": "error",
                    "is_error": True,
                    "message": message,
                    "model": model,
                    "provider": provider,
                    "stop_cause": stop_cause,
                    "resume_session_id": resume_session_id,
                }
            )

    def emit_resume_hint(self, session_id: str) -> None:
        """Tell the caller (human or orchestrator log) how to continue."""
        self._stderr.write(f"Resume with: omn --resume {session_id}\n")
        self._stderr.flush()


class HeadlessRunner:
    """Executes a single prompt through the agent and emits structured output."""

    def __init__(
        self,
        engine: Any,
        output_format: OutputFormat = OutputFormat.TEXT,
        no_approval: bool = False,
        verbose: bool = False,
        max_iterations: Optional[int] = None,
        notify_cmd: Optional[str] = None,
        read_only: bool = False,
        resume_session_id: Optional[str] = None,
    ) -> None:
        self._engine = engine
        self._no_approval = no_approval
        self._verbose = verbose
        self._read_only = read_only
        # Headless runs are unattended, so a cap (not a check-in prompt) is
        # the safety net; --max-iterations sizes it to the job.
        self._max_iterations = _resolve_max_iterations(max_iterations)
        self._turn_notifier = TurnNotifier(notify_cmd=notify_cmd, cwd=os.getcwd())
        session_id = self._turn_notifier.session_id
        self._emitter = HeadlessOutputEmitter(output_format, session_id, verbose)
        self._tokens = TokenAccumulator()
        self._provider_name: str = ""
        self._model: str = ""
        # Checkpoint identity: a resumed run keeps writing to the checkpoint
        # it was resumed from, so repeated failures stay resumable under one
        # id instead of scattering files.
        self._resume_session_id = resume_session_id
        self._checkpoint_id = resume_session_id or session_id
        self._checkpoint_saved = False

    @staticmethod
    def _enable_auto_approval(agent_engine: Any) -> None:
        """Approve every operation, honoring --dangerously-skip-permissions.

        Headless runs are unattended: no approval callback is ever installed,
        and both approval managers deny by default without one, so a headless
        agent could never write a file or run a command regardless of the
        flag. Config-driven permission DENY rules and hooks still apply — they
        are checked before the approval step.
        """

        async def _approve(_request: Any) -> bool:
            return True

        approval = getattr(agent_engine, "approval", None)
        if approval is not None:
            approval.set_approval_callback(_approve)
        enhanced = getattr(agent_engine, "enhanced_approval", None)
        if enhanced is not None:
            enhanced.set_approval_callback(_approve)

    @staticmethod
    def _enable_full_trust(agent_engine: Any) -> None:
        """Relax the hard security layers for unattended full-trust runs.

        Auto-approval alone is not enough: the command argument sanitizer
        (pipes, &&, redirects), the forbidden-command list, the sensitive
        filename patterns (*key*, *token*, .env, ...), and the 30s default
        command timeout all block operations before/outside the approval
        step, so a --dangerously-skip-permissions run still could not do real
        work. The caller (e.g. a CI or agent container) is the security
        boundary; hard-restricted system paths (~/.ssh, /etc, ...) stay
        blocked.
        """
        executor = getattr(agent_engine, "executor", None)
        if executor is not None and hasattr(executor, "set_full_trust"):
            executor.set_full_trust(True)
        file_system = getattr(agent_engine, "file_system", None)
        if file_system is not None and hasattr(file_system, "set_full_trust"):
            file_system.set_full_trust(True)

    async def run(self, prompt: str) -> int:
        """Execute one prompt and always emit turn-completion signals.

        Args:
            prompt: User request to run through the coding agent.

        Returns:
            Process-style status code: zero for success, one for failure.
        """
        self._turn_notifier.reset_turn()
        config = self._engine.config_manager.get_config()
        await fleet_events.init_events(
            self._turn_notifier.session_id, "headless", config.events
        )
        self._provider_name, self._model = self._engine.runtime_identity()
        await fleet_events.emit_event(
            EventType.SESSION_START,
            {
                "provider": self._provider_name,
                "model": self._model,
                "read_only": self._read_only,
            },
        )
        await fleet_events.emit_event(
            EventType.TURN_START,
            {
                "turn": 1,
                "prompt_preview": truncate(prompt, PREVIEW_CHARS),
                "prompt_chars": len(prompt),
            },
        )
        status = 1
        try:
            status = await self._run(prompt)
            return status
        finally:
            await fire_turn_complete(self._engine, self._turn_notifier)
            # Guarded: a diagnostics emission inside a finally must never
            # mask the run's real exception.
            try:
                turn_payload = self._turn_notifier.build_payload()
                await fleet_events.emit_event(
                    EventType.TURN_END,
                    {
                        "turn": 1,
                        "usage": turn_payload.get("usage"),
                        "last_message_preview": truncate(
                            str(turn_payload.get("last-assistant-message") or ""),
                            MESSAGE_PREVIEW_CHARS,
                        ),
                    },
                )
                await fleet_events.emit_event(
                    EventType.SESSION_END, {"reason": "exit", "status": status}
                )
            except Exception as exc:
                logger.debug(f"session_end event emission failed: {exc}")
            await fleet_events.shutdown_events()

    async def _run(self, prompt: str) -> int:
        """Execute the headless agent loop without completion finalization."""
        agent_engine = getattr(self._engine, "agent_engine", None)
        if not agent_engine:
            self._emitter.emit_error(
                "Agent engine not available.",
                model=self._model,
                provider=self._provider_name,
            )
            return 1

        if self._read_only and hasattr(agent_engine, "set_read_only"):
            agent_engine.set_read_only(True)

        if self._no_approval:
            self._enable_auto_approval(agent_engine)
            self._enable_full_trust(agent_engine)

        model = self._model or "unknown"
        self._emitter.emit_init(model)

        supports_tools = self._engine.provider_supports_tools()
        agent_prompt = build_agent_prompt(supports_tools=supports_tools)

        tool_handler = ToolHandler(agent_engine)
        tools = tool_handler.get_tool_definitions()

        current_message = f"{agent_prompt}\n\nUser: {prompt}"
        last_content = ""
        prev_content = ""
        last_model = model
        last_stop_reason = "end_turn"
        # Record of the actions the agent took, surfaced in the JSON result.
        tool_log: list = []
        turns = 0
        # Identical repeated calls get skipped with a corrective nudge; the
        # run is aborted only if the model keeps repeating despite nudges.
        repeat_tracker = RepeatedCallTracker()
        no_tool_nudges = 0
        start_iteration = 0

        if self._resume_session_id:
            restored = load_checkpoint(self._resume_session_id)
            if restored is None:
                self._emitter.emit_error(
                    "No checkpoint found for session "
                    f"'{self._resume_session_id}' — nothing to resume.",
                    model=self._model,
                    provider=self._provider_name,
                )
                return 1
            self._restore_from_checkpoint(restored)
            prompt = prompt or restored.prompt
            current_message = restored.current_message
            tool_log = list(restored.tool_log)
            no_tool_nudges = restored.no_tool_nudges
            start_iteration = restored.iteration
            # A prior save exists on disk; keep resume hints available even
            # if this run fails before its first save.
            self._checkpoint_saved = True

        # Each break below overwrites this; only loop fall-through — the
        # iteration cap — leaves it standing.
        stop_cause = "max_iterations"

        result_budget = _resolve_tool_result_budget()

        for iteration in range(start_iteration, self._max_iterations):
            turns = iteration + 1
            # The whole history is retransmitted each iteration — elide old
            # tool results beyond the budget before sending.
            context = getattr(
                getattr(self._engine, "chat_manager", None), "current_context", None
            )
            if context is not None and isinstance(
                getattr(context, "messages", None), list
            ):
                _elide_stale_tool_results(context, result_budget)
            # Snapshot resumable state before spending tokens: on a failed or
            # killed run, --resume re-sends `current_message` on top of the
            # saved history instead of restarting the whole task.
            self._save_checkpoint(
                prompt, iteration, current_message, tool_log, no_tool_nudges
            )
            response = await self._engine.send_message_with_tools(
                current_message, tools
            )
            self._tokens.add(response)
            self._turn_notifier.record_assistant(response.content, response)

            if not response.is_success:
                # Re-resolve at emit time: rate-limit fallback may have
                # switched the engine's provider since the run started.
                provider_name, model_name = self._engine.runtime_identity()
                error_text = response.error or "Unknown error"
                # Rate limits are transient: the run is resumable, and exit
                # code 4 tells orchestrators to re-invoke with --resume
                # instead of restarting the task from zero.
                rate_limited = matches_rate_limit(error_text)
                self._emitter.emit_error(
                    error_text,
                    tool_calls=tool_log,
                    model=model_name or self._model,
                    provider=provider_name or self._provider_name,
                    stop_cause="rate_limited" if rate_limited else "error",
                    resume_session_id=(
                        self._checkpoint_id if self._checkpoint_saved else None
                    ),
                )
                if self._checkpoint_saved:
                    self._emitter.emit_resume_hint(self._checkpoint_id)
                return 4 if rate_limited else 1

            last_model = response.model_used or model
            last_stop_reason = response.stop_reason or "end_turn"

            if response.content:
                prev_content = last_content
                last_content = response.content
                self._emitter.emit_assistant(
                    response.content,
                    last_model,
                    last_stop_reason,
                )

            # The model may mimic the "[Called tools: ...]" history notation
            # as plain text instead of issuing native tool calls — recover
            # those so the run doesn't end with the work undone.
            if not response.tool_calls:
                recovered = parse_described_tool_calls(response.content)
                if recovered:
                    response.tool_calls = recovered
                else:
                    # No tool call: either the final answer or narration from
                    # a model that stopped without acting. End the run only on
                    # an explicit DONE or after repeated nudges — breaking
                    # immediately made agents "complete" with zero changes.
                    if (
                        _is_done_reply(response.content)
                        or no_tool_nudges >= MAX_NO_TOOL_NUDGES
                    ):
                        # A bare "DONE" acknowledgment must not replace the
                        # model's actual final summary in the result payload.
                        if (
                            response.content
                            and response.content.strip().rstrip(".!").upper() == "DONE"
                            and prev_content
                        ):
                            last_content = prev_content
                        if _is_done_reply(response.content):
                            stop_cause = "done"
                        else:
                            stop_cause = "nudge_exhausted"
                        break
                    no_tool_nudges += 1
                    current_message = NO_TOOL_CALL_NUDGE
                    continue
            no_tool_nudges = 0

            repeat_tracker.record(response.tool_calls)
            offender = repeat_tracker.abort_offender(response.tool_calls)
            if offender is not None:
                stop_cause = "repeat_abort"
                if not last_content:
                    last_content = (
                        f"Stopped: the model repeated the same tool call "
                        f"{repeat_tracker.count(offender)} times "
                        f"({offender.name}) despite duplicate warnings."
                    )
                break

            result_parts = []
            records = []
            for i, tc in enumerate(response.tool_calls):
                self._emitter.emit_tool_use(tc.name, tc.arguments)

                # Skip exact repeats with a nudge so the model can recover
                # instead of the run being killed.
                if repeat_tracker.is_duplicate(tc):
                    result = ToolResult(content=DUPLICATE_CALL_NUDGE)
                else:
                    result = await tool_handler.execute_tool_call(tc)
                self._emitter.emit_tool_result(
                    tc.name,
                    result.content or "",
                    result.error,
                )
                tool_log.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "error": result.error,
                    }
                )

                # Label results with the call's arguments so the model can
                # tell which call each result belongs to.
                label = f"{tc.name}({json.dumps(tc.arguments, default=str)})"
                if result.error:
                    result_parts.append(f"[{label}] Error: {result.error}")
                else:
                    result_parts.append(f"[{label}] Result: {result.content}")
                records.append(
                    ToolResultRecord(
                        tool_call_id=tc.id or f"call_{i}",
                        content=result.error or result.content,
                    )
                )

            # Feed results back without forcing the model to "continue".
            results_text = "Tool results:\n\n" + "\n\n".join(result_parts)
            if self._engine.provider_supports_native_tool_history() is True:
                # Native protocol: structured history + empty continuation
                # message (see interface._handle_tool_calling_flow).
                self._engine.record_tool_results(results_text, records)
                current_message = ""
            else:
                current_message = results_text

        usage = self._tokens.total
        aggregate_response = ChatResponse(
            content=last_content,
            model_used=last_model,
            tokens_used=usage["input_tokens"] + usage["output_tokens"],
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_estimate=usage["total_cost_usd"],
            stop_reason=last_stop_reason,
        )
        self._turn_notifier.record_assistant(last_content or None, aggregate_response)
        # Re-resolve at emit time: rate-limit fallback may have switched the
        # engine's provider mid-run, and the envelope must stay consistent
        # with the model that actually answered.
        provider_name, _ = self._engine.runtime_identity()
        self._emitter.emit_result(
            last_content,
            last_model,
            usage,
            usage["total_cost_usd"],
            last_stop_reason,
            stop_cause=stop_cause,
            tool_calls=tool_log,
            num_turns=turns,
            provider=provider_name or self._provider_name,
        )
        # 3 = the run ended without the model finishing its work (cap hit or
        # duplicate-call abort) even though a result was emitted; orchestrators
        # need that distinction and 2 is reserved by click for usage errors.
        if stop_cause in ("max_iterations", "repeat_abort"):
            # Partial run — keep the checkpoint so a re-invocation with a
            # higher cap continues instead of starting over.
            if self._checkpoint_saved:
                self._emitter.emit_resume_hint(self._checkpoint_id)
            return 3
        # Clean completion: the checkpoint has served its purpose.
        self._delete_checkpoint()
        return 0

    def _save_checkpoint(
        self,
        prompt: str,
        iteration: int,
        current_message: str,
        tool_log: list,
        no_tool_nudges: int,
    ) -> None:
        """Persist resumable state; never let a save failure break the run."""
        if not _checkpointing_enabled():
            return
        context = getattr(
            getattr(self._engine, "chat_manager", None), "current_context", None
        )
        messages = getattr(context, "messages", None)
        if not isinstance(messages, list):
            return
        try:
            save_checkpoint(
                HeadlessCheckpoint(
                    session_id=self._checkpoint_id,
                    prompt=prompt,
                    iteration=iteration,
                    current_message=current_message,
                    messages=[message_to_dict(m) for m in messages],
                    tool_log=tool_log,
                    no_tool_nudges=no_tool_nudges,
                    usage=self._tokens.total,
                    provider=self._provider_name,
                    model=self._model,
                )
            )
            self._checkpoint_saved = True
        except Exception as exc:
            logger.warning("Checkpoint save failed: %s", exc)

    def _restore_from_checkpoint(self, checkpoint: Any) -> None:
        """Rebuild the conversation and usage totals from a saved run."""
        context = getattr(
            getattr(self._engine, "chat_manager", None), "current_context", None
        )
        if context is not None:
            context.messages = [message_from_dict(m) for m in checkpoint.messages]
        self._tokens.restore(checkpoint.usage or {})

    def _delete_checkpoint(self) -> None:
        if not self._checkpoint_saved:
            return
        try:
            delete_checkpoint(self._checkpoint_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Checkpoint delete failed: %s", exc)


async def run_headless(
    prompt: str,
    output_format: str = "text",
    config_path: Optional[str] = None,
    no_approval: bool = False,
    verbose: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    max_iterations: Optional[int] = None,
    notify_cmd: Optional[str] = None,
    read_only: bool = False,
    resume: Optional[str] = None,
) -> int:
    from ..core.config_manager import ConfigManager
    from ..core.engine import CoreEngine

    config_manager = ConfigManager(config_path)
    apply_session_overrides(config_manager, provider, model, base_url)

    engine = CoreEngine(config_manager)
    await engine.initialize_providers()

    fmt = OutputFormat(output_format)
    runner = HeadlessRunner(
        engine=engine,
        output_format=fmt,
        no_approval=no_approval,
        verbose=verbose,
        max_iterations=max_iterations,
        notify_cmd=notify_cmd,
        read_only=read_only,
        resume_session_id=resume,
    )
    return await runner.run(prompt)
