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

from ..core.models import ChatContext, SubAgentDefinition
from ..events import emitter as fleet_events
from .tool_handler import ToolHandler

logger = logging.getLogger(__name__)


@dataclass
class SubAgentResult:
    """Result of running a subagent."""

    name: str
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

        # Every tool event inside the loop carries the subagent's identity;
        # parent is whatever agent spawned us (supports nested subagents).
        with fleet_events.agent_context(run_id, fleet_events.current_agent_id()):
            try:
                for i in range(max(1, definition.max_iterations)):
                    iterations = i + 1
                    response = await provider.send_message_with_tools(
                        message, context, tools
                    )
                    if not response.is_success:
                        return SubAgentResult(
                            name=definition.name,
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

                return SubAgentResult(
                    name=definition.name,
                    output=output,
                    iterations=iterations,
                    tool_calls=tool_calls_made,
                    success=True,
                )
            except Exception as e:  # never let a subagent crash the parent
                logger.warning("Subagent '%s' failed: %s", definition.name, e)
                return SubAgentResult(
                    name=definition.name,
                    output=output,
                    iterations=iterations,
                    tool_calls=tool_calls_made,
                    success=False,
                    error=str(e),
                )
            finally:
                if definition.model and original_model is not None:
                    provider.model = original_model
