"""Headless pipe mode for Omnimancer (omn -p).

Runs a single prompt through the coding agent and outputs structured
results to stdout. No Rich console, no readline, no interactive UI.
"""

import json
import logging
import os
import sys
import uuid
from enum import Enum
from typing import Any, Dict, Optional

from ..core.models import (
    ChatResponse,
    ToolResult,
    ToolResultRecord,
    parse_described_tool_calls,
)
from .system_prompts import build_agent_prompt
from .tool_handler import (
    DUPLICATE_CALL_NUDGE,
    MAX_TOOL_ITERATIONS,
    RepeatedCallTracker,
    ToolHandler,
)

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


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


class TokenAccumulator:
    """Tracks cumulative token usage across multiple API calls."""

    def __init__(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_cost = 0.0

    def add(self, response: ChatResponse) -> None:
        self._input_tokens += response.input_tokens or 0
        self._output_tokens += response.output_tokens or 0
        self._total_cost += response.cost_estimate or 0.0

    @property
    def total(self) -> Dict[str, Any]:
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_cost_usd": self._total_cost,
        }


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
    ) -> None:
        if self._format == OutputFormat.TEXT:
            if content != self._last_content:
                self._stdout.write(content + "\n")
                self._stdout.flush()
        elif self._format == OutputFormat.JSON:
            blob = {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": content,
                "session_id": self._session_id,
                "model": model,
                "num_turns": num_turns,
                "tool_calls": tool_calls or [],
                "usage": usage,
                "total_cost_usd": cost,
                "stop_reason": stop_reason,
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
                    "num_turns": num_turns,
                    "usage": usage,
                    "total_cost_usd": cost,
                    "stop_reason": stop_reason,
                }
            )

    def emit_error(self, message: str, tool_calls: Optional[list] = None) -> None:
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
                        "tool_calls": tool_calls or [],
                    },
                    default=str,
                )
                + "\n"
            )
            self._stdout.flush()
        elif self._format == OutputFormat.STREAM_JSON:
            self._write_json_line(
                {"type": "error", "is_error": True, "message": message}
            )


class HeadlessRunner:
    """Executes a single prompt through the agent and emits structured output."""

    def __init__(
        self,
        engine: Any,
        output_format: OutputFormat = OutputFormat.TEXT,
        no_approval: bool = False,
        verbose: bool = False,
        max_iterations: Optional[int] = None,
    ) -> None:
        self._engine = engine
        self._no_approval = no_approval
        self._verbose = verbose
        # Headless runs are unattended, so a cap (not a check-in prompt) is
        # the safety net; --max-iterations sizes it to the job.
        self._max_iterations = _resolve_max_iterations(max_iterations)
        session_id = str(uuid.uuid4())
        self._emitter = HeadlessOutputEmitter(output_format, session_id, verbose)
        self._tokens = TokenAccumulator()

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
        agent_engine = getattr(self._engine, "agent_engine", None)
        if not agent_engine:
            self._emitter.emit_error("Agent engine not available.")
            return 1

        if self._no_approval:
            self._enable_auto_approval(agent_engine)
            self._enable_full_trust(agent_engine)

        model = self._engine.config_manager.get_config().default_provider or "unknown"
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

        for iteration in range(self._max_iterations):
            turns = iteration + 1
            response = await self._engine.send_message_with_tools(
                current_message, tools
            )
            self._tokens.add(response)

            if not response.is_success:
                self._emitter.emit_error(
                    response.error or "Unknown error", tool_calls=tool_log
                )
                return 1

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
                        break
                    no_tool_nudges += 1
                    current_message = NO_TOOL_CALL_NUDGE
                    continue
            no_tool_nudges = 0

            repeat_tracker.record(response.tool_calls)
            offender = repeat_tracker.abort_offender(response.tool_calls)
            if offender is not None:
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
        self._emitter.emit_result(
            last_content,
            last_model,
            usage,
            usage["total_cost_usd"],
            last_stop_reason,
            tool_calls=tool_log,
            num_turns=turns,
        )
        return 0


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
) -> int:
    from ..core.config_manager import ConfigManager
    from ..core.engine import CoreEngine

    config_manager = ConfigManager(config_path)

    if provider:
        config = config_manager.get_config()
        config.default_provider = provider
        config_manager.save_config(config)

    engine = CoreEngine(config_manager)
    await engine.initialize_providers()

    current_provider_name = engine.config_manager.get_config().default_provider

    if model and current_provider_name and current_provider_name in engine.providers:
        engine.providers[current_provider_name].model = model

    # Ephemeral endpoint override for this run (does not touch saved config).
    # base_url is defined on the OpenAI-family/Claude providers but not on the
    # BaseProvider contract, so set it dynamically.
    if base_url and current_provider_name and current_provider_name in engine.providers:
        setattr(
            engine.providers[current_provider_name],
            "base_url",
            base_url.rstrip("/"),
        )

    fmt = OutputFormat(output_format)
    runner = HeadlessRunner(
        engine=engine,
        output_format=fmt,
        no_approval=no_approval,
        verbose=verbose,
        max_iterations=max_iterations,
    )
    return await runner.run(prompt)
