"""Planner: the high model reads the repo and submits a validated plan (PRD §3).

Stories arrive **one at a time** through ``add_story`` and are validated as
they arrive; ``submit_plan`` then closes the plan with its goal and design.
No single model output has to hold the whole plan, so no output-token cap can
truncate it (the first live runs lost whole plans to a 4096-token cap and the
model kept shrinking the plan until it fit), and a bad story costs one small
retry instead of a full regeneration. ``submit_plan`` still accepts inline
``stories`` for small plans.

Tool schemas are written out flat (no ``$defs``) so every tool-capable
provider accepts them. After the first accepted plan the planner runs one
bounded self-check turn; a resubmission, when it comes, wins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from ..cli.system_prompts import get_directory_context, load_project_instructions
from ..cli.tool_handler import ToolHandler
from ..core.agent.tool_definitions import CODING_AGENT_TOOLS
from ..core.models import ChatContext, ToolCall, ToolDefinition, ToolResult
from .loop import IsolatedLoop
from .models import (
    Plan,
    PlanValidationError,
    Story,
    ToolLogEntry,
    UsageTotals,
    normalize_story,
    validate_plan,
)
from .prompts import PLANNER_SELF_CHECK, PLANNER_SYSTEM

logger = logging.getLogger(__name__)

PLANNER_READ_TOOLS = {"Read", "Glob", "Grep", "Bash"}
_MUTATING_TOOLS = {"Write", "Edit", "file_write", "file_edit", "file_delete"}

_STORY_PROPERTIES: Dict[str, Any] = {
    "id": {"type": "string", "description": "S1, S2, ..."},
    "title": {"type": "string"},
    "instructions": {
        "type": "string",
        "description": (
            "Step-by-step instructions with exact paths, exact "
            "old_string/new_string pairs or full file content, and exact commands"
        ),
    },
    "files": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Absolute paths the worker may touch",
    },
    "acceptance": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Testable criteria checked against the diff",
    },
    "verify": {
        "type": "string",
        "description": "Shell command that exits 0 when done",
    },
    "depends_on": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Ids of stories that must pass first",
    },
    "max_iterations": {
        "type": "integer",
        "description": "Optional tool-call budget (max 30)",
    },
}
_STORY_REQUIRED = ["id", "title", "instructions", "files", "acceptance"]

ADD_STORY_TOOL = ToolDefinition(
    name="add_story",
    description=(
        "Record ONE story of the backlog. Call once per story, in execution "
        "order. Calling it again with an existing id replaces that story. The "
        "story must be executable from its text alone by a worker with no "
        "repository access."
    ),
    parameters={
        "type": "object",
        "properties": _STORY_PROPERTIES,
        "required": _STORY_REQUIRED,
    },
)

SUBMIT_PLAN_TOOL = ToolDefinition(
    name="submit_plan",
    description=(
        "Close the plan: give the goal and the design. The backlog is the "
        "stories you recorded with add_story (you may instead pass a short "
        "backlog inline as `stories`). Call after every story is recorded."
    ),
    parameters={
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "The goal, restated"},
            "design": {
                "type": "string",
                "description": "Short design note: what changes and why",
            },
            "stories": {
                "type": "array",
                "description": (
                    "Optional inline backlog for small plans; omit when the "
                    "stories were recorded with add_story"
                ),
                "items": {
                    "type": "object",
                    "properties": _STORY_PROPERTIES,
                    "required": _STORY_REQUIRED,
                },
            },
        },
        "required": ["goal", "design"],
    },
)

NO_SUBMIT_NUDGE = (
    "You have not closed the plan. Record each story with add_story (one call "
    "per story), then call submit_plan with the goal and design. If you still "
    "need to read something, do that first."
)

BUDGET_EXHAUSTED = (
    "Your exploration budget is used up. Write the plan now from what you have "
    "already read: record each story with add_story, then call submit_plan. "
    "No reading tool is available; do not ask to read more."
)

# Turns granted to the closing phase once an exploration budget is exhausted
# (stories arrive one per call, so this must cover a whole backlog).
CLOSING_TURNS = 24
# Turns granted to the self-check after a plan has been accepted.
SELF_CHECK_TURNS = 12
_REASON_CHARS = 500


@dataclass
class PlanResult:
    plan: Optional[Plan] = None
    usage: UsageTotals = field(default_factory=UsageTotals)
    error: Optional[str] = None
    iterations: int = 0
    tool_log: List[ToolLogEntry] = field(default_factory=list)
    model_used: str = ""
    truncations: int = 0

    def absorb(self, outcome: Any) -> None:
        """Fold one loop outcome into the running totals."""
        self.usage = self.usage.add(outcome.usage)
        self.iterations += outcome.iterations
        self.tool_log.extend(outcome.tool_log)
        self.truncations += int(getattr(outcome, "truncations", 0) or 0)
        if getattr(outcome, "model_used", ""):
            self.model_used = outcome.model_used


class Planner:
    def __init__(
        self,
        provider: Any,
        agent_engine: Any,
        cwd: Path,
        max_iterations: Optional[int] = None,
        include_project_instructions: bool = True,
        on_tool_call: Optional[Any] = None,
        on_wait: Optional[Any] = None,
        allow_bash: bool = True,
    ) -> None:
        self.provider = provider
        self.agent_engine = agent_engine
        self.cwd = Path(cwd)
        self.max_iterations = max_iterations
        self.include_project_instructions = include_project_instructions
        # Progress observers: the surfaces show each exploration call and a
        # heartbeat during slow model calls so planning never looks idle.
        self.on_tool_call = on_tool_call
        self.on_wait = on_wait
        # Headless without --dangerously-skip-permissions denies every Bash
        # call; offering the tool just burns planner turns on refusals.
        self.allow_bash = allow_bash
        self._tool_handler_factory: Callable[[Any], Any] = ToolHandler
        # Per-plan state, reset by plan().
        self._stories: Dict[str, Story] = {}
        self._candidate: List[Plan] = []
        self._rejections: List[str] = []

    # ------------------------------------------------------------ prompt

    def _first_message(self, goal: str) -> str:
        parts = [PLANNER_SYSTEM, get_directory_context()]
        if self.include_project_instructions:
            try:
                instructions = load_project_instructions()
            except Exception as exc:  # never let a bad file block planning
                logger.debug("project instructions unavailable: %s", exc)
                instructions = ""
            if instructions:
                parts.append(instructions)
        if self.max_iterations is not None:
            parts.append(
                f"Exploration budget: {self.max_iterations} turns. Read what you "
                "need, then record the stories and submit before it runs out."
            )
        parts.append(f"Goal: {goal}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------ run

    async def plan(self, goal: str) -> PlanResult:
        result = PlanResult()
        self._stories, self._candidate, self._rejections = {}, [], []
        handler = self._tool_handler_factory(self.agent_engine)
        offered = set(PLANNER_READ_TOOLS)
        if not self.allow_bash:
            offered.discard("Bash")
        plan_tools = [ADD_STORY_TOOL, SUBMIT_PLAN_TOOL]
        tools = [t for t in CODING_AGENT_TOOLS if t.name in offered] + plan_tools

        async def execute(tc: ToolCall) -> ToolResult:
            if tc.name in _MUTATING_TOOLS:
                return ToolResult(
                    content="",
                    error=f"H2L: the planner may not modify files ({tc.name})",
                )
            if tc.name == "add_story":
                return self._add_story(tc)
            if tc.name == "submit_plan":
                return self._submit(tc)
            if tc.name == "Bash" and not self.allow_bash:
                return ToolResult(
                    content="",
                    error=(
                        "H2L: Bash is not available to the planner in this run; "
                        "use Read, Glob and Grep"
                    ),
                )
            if tc.name in offered:
                out: ToolResult = await handler.execute_tool_call(tc)
                return out
            return ToolResult(content="", error=f"Unknown tool: {tc.name}")

        context = ChatContext(
            messages=[],
            current_model=getattr(self.provider, "model", "") or "",
            session_id="h2l-planner",
        )

        def loop(loop_tools: List[ToolDefinition], turns: Optional[int]) -> Any:
            return IsolatedLoop(
                self.provider,
                context,
                loop_tools,
                execute,
                turns,
                stop_tools={"submit_plan"},
                on_tool_call=self.on_tool_call,
                on_wait=self.on_wait,
            )

        # Phase 1: explore, record stories, submit.
        first = await loop(tools, self.max_iterations).run(self._first_message(goal))
        result.absorb(first)
        budget_left = (
            self.max_iterations is None or result.iterations < self.max_iterations
        )
        if first.stop == "done" and not self._candidate and budget_left:
            # One nudge for a model that narrated instead of submitting.
            turns = (
                None
                if self.max_iterations is None
                else max(1, self.max_iterations - result.iterations)
            )
            nudged = await loop(tools, turns).run(NO_SUBMIT_NUDGE)
            result.absorb(nudged)
            first = nudged
        if first.stop == "error":
            result.error = f"plan_failed: {first.error}"
            return result

        if not self._candidate:
            # Exploration ended (budget, repeat abort, or silence) without an
            # accepted plan: a closing phase with only the plan tools.
            closing = await loop(plan_tools, CLOSING_TURNS).run(BUDGET_EXHAUSTED)
            result.absorb(closing)
            if closing.stop == "error":
                result.error = f"plan_failed: {closing.error}"
                return result
        if not self._candidate:
            result.error = self._failure_reason(result)
            return result

        # Phase 2: self-check; a resubmission replaces the first plan. Always
        # bounded: an accepted plan already exists.
        second = await loop(tools, SELF_CHECK_TURNS).run(PLANNER_SELF_CHECK)
        result.absorb(second)
        if second.stop == "error":
            logger.warning("H2L planner self-check failed: %s", second.error)

        result.plan = self._candidate[-1]
        return result

    def _failure_reason(self, result: PlanResult) -> str:
        """Say what actually went wrong, not just that nothing was submitted."""
        if self._rejections:
            reason = (
                f"plan_failed: {len(self._rejections)} plan submission(s) "
                f"rejected; last reason: {self._rejections[-1][:_REASON_CHARS]}"
            )
        elif self._stories:
            reason = (
                f"plan_failed: {len(self._stories)} stories were recorded but "
                "the planner never called submit_plan"
            )
        else:
            reason = "plan_failed: the planner stopped without submitting a plan"
        if result.truncations:
            reason += (
                f" (the model's output was cut off at the token limit "
                f"{result.truncations} time(s); raise h2l.high_max_tokens)"
            )
        return reason

    # ------------------------------------------------------------ tools

    def _add_story(self, tc: ToolCall) -> ToolResult:
        args = tc.arguments if isinstance(tc.arguments, dict) else {}
        try:
            story = Story.model_validate(args)
        except ValidationError as exc:
            return ToolResult(content="", error=f"add_story rejected: {exc}")
        story, problems = normalize_story(story, self.cwd)
        if problems:
            return ToolResult(content="", error="H2L: " + "; ".join(problems))
        replaced = story.id in self._stories
        self._stories[story.id] = story
        verb = "replaced" if replaced else "recorded"
        return ToolResult(
            content=f"Story {story.id} {verb} ({len(self._stories)} in the backlog)."
        )

    def _submit(self, tc: ToolCall) -> ToolResult:
        args = dict(tc.arguments) if isinstance(tc.arguments, dict) else {}
        inline = args.get("stories")
        if inline is None or inline == []:
            # No inline backlog: the plan is what add_story recorded. Any
            # other malformed value falls through to the schema error.
            args["stories"] = [s.model_dump() for s in self._stories.values()]
        if args["stories"] == []:
            return self._reject(
                "H2L: plan has no stories — record each story with add_story "
                "(one call per story), then call submit_plan"
            )
        try:
            plan = Plan.model_validate(args)
        except ValidationError as exc:
            return self._reject(f"submit_plan rejected: {exc}")
        try:
            plan = validate_plan(plan, self.cwd)
        except PlanValidationError as exc:
            return self._reject(str(exc))
        self._candidate.append(plan)
        # Later add_story calls (self-check fixes) edit the accepted backlog.
        self._stories = {s.id: s for s in plan.stories}
        return ToolResult(content=f"Plan received ({len(plan.stories)} stories).")

    def _reject(self, message: str) -> ToolResult:
        self._rejections.append(message)
        return ToolResult(content="", error=message)
