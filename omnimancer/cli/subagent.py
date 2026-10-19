"""Subagents: spawn scoped child agents that run isolated tool-calling loops.

A subagent (see :class:`~omnimancer.core.models.SubAgentDefinition`) handles a
focused task with its own system prompt, a restricted tool allowlist, and an
optional model override. It runs an isolated loop using its *own* conversation
context, so the parent conversation is never touched. A model override also
gets its own provider instance from
:mod:`omnimancer.core.provider_resolver` rather than reassigning the shared
session provider's model. This is the lightweight counterpart to Claude
Code's AgentTool.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..core.agent.status_core import EventType
from ..core.models import ChatContext, ProviderConfig, SubAgentDefinition
from ..core.provider_resolver import (
    ProviderResolutionError,
    resolve_entry,
    resolve_provider,
)
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

    def _provider_name(self, provider: Any) -> str:
        """The config-entry name for the running provider, if we can tell."""
        providers = getattr(self.engine, "providers", None)
        if isinstance(providers, dict):
            for name, instance in providers.items():
                if instance is provider:
                    return str(name)
        getter = getattr(provider, "get_provider_name", None)
        if callable(getter):
            raw_name = getter()
            if isinstance(raw_name, str):
                return str(raw_name)
        return ""

    def _isolated_provider(self, provider_name: str, model: str) -> Optional[Any]:
        """A provider instance of our own for a model override.

        Returns None when the entry cannot be resolved, so the caller falls
        back to the legacy mutate-and-restore behaviour rather than failing
        a subagent over a provider-resolution detail.
        """
        config_manager = getattr(self.engine, "config_manager", None)
        if not provider_name or config_manager is None:
            return None
        try:
            # Only a real entry is safe to build from: anything else means we
            # cannot tell what we would be constructing.
            entry = resolve_entry(provider_name, config_manager, self.engine, model)
            if not isinstance(entry, ProviderConfig):
                return None
            return resolve_provider(provider_name, config_manager, model, self.engine)
        except ProviderResolutionError as exc:
            logger.debug(
                "Subagent model override falling back to the shared provider: %s",
                exc,
            )
            return None
        except Exception as exc:
            logger.debug(
                "Subagent model override falling back to the shared provider: %s",
                exc,
            )
            return None

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

        # Identity is read before any override so the entry name is the
        # session's, not something derived from a swapped-in instance.
        provider_name = self._provider_name(provider)

        shared_provider = provider
        original_model = getattr(provider, "model", None)
        restore_model = False
        if definition.model:
            isolated = self._isolated_provider(provider_name, definition.model)
            if isolated is not None:
                provider = isolated
            else:
                # Last resort: the shared object, restored in the finally.
                provider.model = definition.model
                restore_model = True

        raw_model = getattr(provider, "model", "")
        model_used: str = raw_model if isinstance(raw_model, str) else ""

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
                # running on the subagent's model. Only the fallback path
                # touched it; a resolved instance leaves it alone entirely.
                if restore_model and original_model is not None:
                    shared_provider.model = original_model
                try:
                    await fleet_events.emit_event(
                        EventType.SESSION_END,
                        {"reason": "subagent_complete", "status": exit_status},
                    )
                except Exception as exc:
                    logger.debug(f"subagent session_end emission failed: {exc}")
