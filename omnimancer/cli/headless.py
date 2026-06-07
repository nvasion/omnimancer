"""Headless pipe mode for Omnimancer (omn -p).

Runs a single prompt through the coding agent and outputs structured
results to stdout. No Rich console, no readline, no interactive UI.
"""

import json
import sys
import uuid
from enum import Enum
from typing import Any, Dict, Optional

from ..core.models import ChatResponse
from .system_prompts import build_agent_prompt
from .tool_handler import MAX_TOOL_ITERATIONS, ToolHandler


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
    ) -> None:
        self._engine = engine
        self._no_approval = no_approval
        self._verbose = verbose
        session_id = str(uuid.uuid4())
        self._emitter = HeadlessOutputEmitter(output_format, session_id, verbose)
        self._tokens = TokenAccumulator()

    async def run(self, prompt: str) -> int:
        agent_engine = getattr(self._engine, "agent_engine", None)
        if not agent_engine:
            self._emitter.emit_error("Agent engine not available.")
            return 1

        model = self._engine.config_manager.get_config().default_provider or "unknown"
        self._emitter.emit_init(model)

        supports_tools = self._engine.provider_supports_tools()
        agent_prompt = build_agent_prompt(supports_tools=supports_tools)

        tool_handler = ToolHandler(agent_engine)
        tools = tool_handler.get_tool_definitions()

        current_message = f"{agent_prompt}\n\nUser: {prompt}"
        last_content = ""
        last_model = model
        last_stop_reason = "end_turn"
        # Record of the actions the agent took, surfaced in the JSON result.
        tool_log: list = []
        turns = 0
        # Detect runaway loops where a weak model repeats the same tool call.
        seen_calls: dict = {}

        for iteration in range(MAX_TOOL_ITERATIONS):
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
                last_content = response.content
                self._emitter.emit_assistant(
                    response.content,
                    last_model,
                    last_stop_reason,
                )

            if not response.tool_calls:
                break

            repeated = False
            for tc in response.tool_calls:
                sig = (
                    f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, default=str)}"
                )
                seen_calls[sig] = seen_calls.get(sig, 0) + 1
                if seen_calls[sig] >= 3:
                    repeated = True
            if repeated:
                if not last_content:
                    last_content = (
                        "Stopped: the model kept repeating the same tool call."
                    )
                break

            result_parts = []
            for tc in response.tool_calls:
                self._emitter.emit_tool_use(tc.name, tc.arguments)

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

                if result.error:
                    result_parts.append(f"[{tc.name}] Error: {result.error}")
                else:
                    result_parts.append(f"[{tc.name}] Result: {result.content}")

            # Feed results back without forcing the model to "continue".
            current_message = "Tool results:\n\n" + "\n\n".join(result_parts)

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
    )
    return await runner.run(prompt)
