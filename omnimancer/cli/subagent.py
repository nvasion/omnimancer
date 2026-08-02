"""Subagents: spawn scoped child agents that run isolated tool-calling loops.

A subagent (see :class:`~omnimancer.core.models.SubAgentDefinition`) handles a
focused task with its own system prompt, a restricted tool allowlist, and an
optional model override. It runs an isolated loop against the current provider
using its *own* conversation context, so the parent conversation is never
touched. This is the lightweight counterpart to Claude Code's AgentTool.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..core.agent.status_core import EventType
from ..core.models import ChatContext, SubAgentDefinition
from ..events import emitter as fleet_events
from .tool_handler import ToolHandler

logger = logging.getLogger(__name__)


@dataclass
class SubAgentResult:
    """Result of running a subagent."""

    name: str
    model: str = ""
    output: str = ""
    iterations: int = 0
    tool_calls: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class SubAgentRunner:
    """Runs a :class:`SubAgentDefinition` to completion against the engine."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def _scoped_tools(
        self, tool_handler: ToolHandler, definition: SubAgentDefinition
    ) -> List[Any]:
        all_tools = tool_handler.get_tool_definitions()
        if definition.tools is None:
            return all_tools
        allowed = set(definition.tools)
        return [t for t in all_tools if t.name in allowed]

    async def run(self, definition: SubAgentDefinition, task: str) -> SubAgentResult:
        """Execute the subagent's isolated tool-calling loop for ``task``."""
        provider = getattr(self.engine, "current_provider", None)
        if provider is None:
            return SubAgentResult(
                name=definition.name,
                success=False,
                error="No provider available to run the subagent.",
            )

        agent_engine = getattr(self.engine, "agent_engine", None) or self.engine
        tool_handler = ToolHandler(agent_engine)
        tools = self._scoped_tools(tool_handler, definition)

        original_model = getattr(provider, "model", None)
        if definition.model:
            provider.model = definition.model

        # Compute the provider/model identity for fleet lifecycle events.
        raw_model = getattr(provider, "model", "")
        model_used: str = raw_model if isinstance(raw_model, str) else ""
        provider_name = ""
        providers = getattr(self.engine, "providers", None)
        if isinstance(providers, dict):
            for name, instance in providers.items():
                if instance is provider:
                    provider_name = name
                    break
        if not provider_name:
            getter = getattr(provider, "get_provider_name", None)
            if callable(getter):
                raw_name = getter()
                if isinstance(raw_name, str):
                    provider_name = raw_name

        # Isolated context — the parent conversation is never modified. The
        # run id is unique per invocation so two runs of the same subagent
        # are distinguishable in the event feed.
        run_id = f"subagent-{definition.name}-{uuid.uuid4().hex[:8]}"
        context = ChatContext(
            messages=[],
            current_model=getattr(provider, "model", "") or "",
            session_id=run_id,
        )
        message = f"{definition.prompt}\n\n{task}" if definition.prompt else task
        tool_calls_made: List[str] = []
        output = ""
        iterations = 0
        # Reported on SESSION_END; only the clean-success path clears it.
        exit_status = 1

        # Every tool event inside the loop carries the subagent's identity;
        # parent is whatever agent spawned us (supports nested subagents).
        with fleet_events.agent_context(run_id, fleet_events.current_agent_id()):
            try:
                await fleet_events.emit_event(
                    EventType.SESSION_START,
                    {
                        "provider": provider_name,
                        "model": model_used,
                        "subagent": definition.name,
                    },
                )
                for i in range(max(1, definition.max_iterations)):
                    iterations = i + 1
                    response = await provider.send_message_with_tools(
                        message, context, tools
                    )
                    if not response.is_success:
                        return SubAgentResult(
                            name=definition.name,
                            model=model_used,
                            output=output,
                            iterations=iterations,
                            tool_calls=tool_calls_made,
                            success=False,
                            error=response.error or "Subagent provider call failed.",
                        )
                    if response.content:
                        output = response.content
                    if not response.tool_calls:
                        break

                    result_parts = []
                    for tc in response.tool_calls:
                        tool_calls_made.append(tc.name)
                        result = await tool_handler.execute_tool_call(tc)
                        if result.error:
                            result_parts.append(f"[{tc.name}] Error: {result.error}")
                        else:
                            result_parts.append(f"[{tc.name}] Result: {result.content}")
                    message = "Tool results:\n\n" + "\n\n".join(result_parts)

                exit_status = 0
                return SubAgentResult(
                    name=definition.name,
                    model=model_used,
                    output=output,
                    iterations=iterations,
                    tool_calls=tool_calls_made,
                    success=True,
                )
            except Exception as e:  # never let a subagent crash the parent
                logger.warning("Subagent '%s' failed: %s", definition.name, e)
                return SubAgentResult(
                    name=definition.name,
                    model=model_used,
                    output=output,
                    iterations=iterations,
                    tool_calls=tool_calls_made,
                    success=False,
                    error=str(e),
                )
            finally:
                # Restore the shared provider BEFORE any await: a cancellation
                # during the emit must never leave the parent conversation
                # running on the subagent's model.
                if definition.model and original_model is not None:
                    provider.model = original_model
                try:
                    await fleet_events.emit_event(
                        EventType.SESSION_END,
                        {"reason": "subagent_complete", "status": exit_status},
                    )
                except Exception as exc:
                    logger.debug(f"subagent session_end emission failed: {exc}")
