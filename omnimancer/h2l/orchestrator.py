"""H2LRun: the one loop both surfaces call (PRD §6).

Plans with the high model, gates the plan through the presenter, then runs
stories in dependency order across the low-model pool: round-robin
assignment, judge feedback on retries, an escalation policy after the
retries are spent, serialized and labeled approval prompts, and per-story
identity on the event bus. Surfaces (TUI, headless) only supply a
:class:`Presenter`.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple

from ..core.agent.tool_definitions import CODING_AGENT_TOOLS
from ..core.rate_limit_fallback import matches_rate_limit
from ..events import emitter as fleet_events
from .judge import Judge
from .models import (
    H2LConfig,
    H2LModelRef,
    Plan,
    RunSummary,
    Story,
    StoryOutcome,
    StoryState,
    UsageTotals,
    Verdict,
    WorkerReport,
    validate_plan,
)
from .planner import Planner, PlanResult
from .refs import ResolvedTiers, validate_tiers
from .worker import WorkerRunner

logger = logging.getLogger(__name__)

_current_story: contextvars.ContextVar[str] = contextvars.ContextVar(
    "h2l_current_story", default=""
)

_TERMINAL_BAD = (StoryState.BLOCKED, StoryState.SKIPPED, StoryState.CANCELLED)


@dataclass
class PlanDecision:
    accepted: bool
    plan: Optional[Plan] = None


class Presenter(Protocol):
    """What a surface must provide. Every method is awaited."""

    async def approve_plan(self, plan: Plan) -> PlanDecision: ...

    async def on_story_start(
        self, story: Story, ref: H2LModelRef, attempt: int, agent_id: str
    ) -> None: ...

    async def on_report(self, story: Story, report: WorkerReport) -> None: ...

    async def on_verdict(self, story: Story, verdict: Verdict) -> None: ...

    async def on_escalation(self, story: Story, outcome: StoryOutcome) -> str: ...

    async def on_done(self, summary: RunSummary) -> None: ...


class H2LRun:
    def __init__(
        self,
        engine: Any,
        config: H2LConfig,
        presenter: Presenter,
        cwd: Path,
        session_id: str,
        tiers: Optional[ResolvedTiers] = None,
    ) -> None:
        self.engine = engine
        self.config = config
        self.presenter = presenter
        self.cwd = Path(cwd)
        self.session_id = session_id
        self.run_id = f"h2l-{session_id}"
        self.tiers = tiers
        self.plan: Optional[Plan] = None
        self.outcomes: Dict[str, StoryOutcome] = {}
        self.usage_by_tier: Dict[str, UsageTotals] = {
            "high": UsageTotals(),
            "low": UsageTotals(),
        }
        self._stop_cause = "done"
        self._stopped = False
        self._pool_index = 0
        # Surfaces turn this off when Bash would only be denied (headless
        # without --dangerously-skip-permissions).
        self.planner_bash = True
        # What the high provider reported answering with during planning —
        # the evidence that the tier ref was honoured.
        self.plan_model_used = ""
        # Factories are overridable for tests and future surfaces.
        self._planner_factory: Callable[..., Any] = Planner
        self._worker_factory: Callable[..., Any] = WorkerRunner
        self._judge_factory: Callable[..., Any] = Judge
        self._worker: Any = None
        self._judge: Any = None

    # ------------------------------------------------------------ setup

    @property
    def agent_engine(self) -> Any:
        return getattr(self.engine, "agent_engine", None) or self.engine

    def _ensure_tiers(self) -> ResolvedTiers:
        if self.tiers is None:
            self.tiers = validate_tiers(
                self.config, self.engine.config_manager, self.engine
            )
        return self.tiers

    def _components(self) -> Tuple[Any, Any]:
        tiers = self._ensure_tiers()
        if self._worker is None:
            self._worker = self._worker_factory(self.agent_engine, self.cwd)
        if self._judge is None:
            self._judge = self._judge_factory(
                tiers.high,
                self.config.pass_threshold,
                on_wait=self._wait_observer(f"judge {self.config.high.label}"),
            )
        return self._worker, self._judge

    def _high_context(self) -> Any:
        return fleet_events.agent_context(
            f"h2l-high-{uuid.uuid4().hex[:8]}", self.run_id
        )

    # ------------------------------------------------------------ planning

    def _wait_observer(self, label: str) -> Optional[Any]:
        """Heartbeat while a model call is in flight (PRD §7/§8 progress)."""
        hook = getattr(self.presenter, "on_wait", None)
        if hook is None:
            return None

        async def observe(seconds: float) -> None:
            try:
                await hook(label, seconds)
            except Exception as exc:
                logger.debug("H2L wait presenter hook failed: %s", exc)

        return observe

    def _planner_observer(self) -> Optional[Any]:
        hook = getattr(self.presenter, "on_planner_tool_call", None)
        if hook is None:
            return None

        async def observe(tool_call: Any, result: Any) -> None:
            try:
                await hook(tool_call, result)
            except Exception as exc:  # presentation must never break planning
                logger.debug("H2L planner progress hook failed: %s", exc)

        return observe

    async def plan_goal(self, goal: str) -> PlanResult:
        tiers = self._ensure_tiers()
        planner = self._planner_factory(
            tiers.high,
            self.agent_engine,
            self.cwd,
            on_tool_call=self._planner_observer(),
            on_wait=self._wait_observer(f"planner {self.config.high.label}"),
            allow_bash=self.planner_bash,
            max_iterations=self.config.planner_max_iterations,
        )
        with self._high_context():
            result: PlanResult = await planner.plan(goal)
        self.usage_by_tier["high"] = self.usage_by_tier["high"].add(result.usage)
        self.plan_model_used = getattr(result, "model_used", "") or ""
        return result

    async def run(self, goal: str) -> RunSummary:
        """Plan, gate through the presenter, execute (PRD §6–§8)."""
        result = await self.plan_goal(goal)
        if result.plan is None:
            summary = self._summary(goal, "plan_failed", error=result.error)
            await self.presenter.on_done(summary)
            return summary
        plan = result.plan
        on_plan = getattr(self.presenter, "on_plan", None)
        if on_plan is not None:
            await on_plan(plan, self.plan_model_used)
        if self.config.plan_approval:
            decision = await self.presenter.approve_plan(plan)
            if not decision.accepted:
                summary = self._summary(goal, "cancelled")
                await self.presenter.on_done(summary)
                return summary
            if decision.plan is not None:
                plan = decision.plan
        return await self.run_plan(plan)

    # ------------------------------------------------------------ execution

    async def run_plan(self, plan: Plan) -> RunSummary:
        self.plan = validate_plan(plan, self.cwd)
        stories = {s.id: s for s in self.plan.stories}
        self.outcomes = {
            s.id: StoryOutcome(id=s.id, title=s.title) for s in self.plan.stories
        }
        self._stop_cause = "done"
        self._stopped = False
        self._components()
        parent = fleet_events.current_agent_id()
        with (
            self._serialized_approvals(),
            fleet_events.agent_context(self.run_id, parent),
        ):
            await self._schedule(stories)
        summary = self._summary(self.plan.goal, self._stop_cause)
        await self.presenter.on_done(summary)
        return summary

    async def _schedule(self, stories: Dict[str, Story]) -> None:
        assert self.plan is not None
        order = [s.id for s in self.plan.stories]
        pending = set(order)
        running: Dict[str, asyncio.Task[None]] = {}
        limit = max(1, self.config.max_parallel)

        while pending or running:
            if self._stopped:
                break
            for sid in [i for i in order if i in pending]:
                story = stories[sid]
                bad = next(
                    (
                        d
                        for d in story.depends_on
                        if self.outcomes[d].state in _TERMINAL_BAD
                    ),
                    None,
                )
                if bad is not None:
                    outcome = self.outcomes[sid]
                    outcome.state = StoryState.SKIPPED
                    outcome.reason = (
                        f"dependency_{self.outcomes[bad].state.value}:{bad}"
                    )
                    pending.discard(sid)
                    continue
                ready = all(
                    self.outcomes[d].state == StoryState.PASSED
                    for d in story.depends_on
                )
                if ready and len(running) < limit:
                    pending.discard(sid)
                    running[sid] = asyncio.create_task(self._execute_story(story))
            if not running:
                if pending:
                    # Only reachable if a dependency is still pending but can
                    # never start — validation rejects cycles, so this is a
                    # defensive exit, not an expected path.
                    for sid in list(pending):
                        outcome = self.outcomes[sid]
                        outcome.state = StoryState.SKIPPED
                        outcome.reason = "unschedulable"
                        pending.discard(sid)
                break
            done, _ = await asyncio.wait(
                set(running.values()), return_when=asyncio.FIRST_COMPLETED
            )
            for sid, task in list(running.items()):
                if task in done:
                    del running[sid]
                    exc = task.exception()
                    if exc is not None:
                        logger.warning("H2L story %s crashed: %s", sid, exc)
                        outcome = self.outcomes[sid]
                        outcome.state = StoryState.BLOCKED
                        outcome.reason = f"error: {exc}"
        if running:
            await asyncio.gather(*running.values(), return_exceptions=True)

    async def _execute_story(self, story: Story) -> None:
        outcome = self.outcomes[story.id]
        outcome.state = StoryState.RUNNING
        max_attempts = self.config.max_retries + 1
        feedback: Optional[str] = None
        attempt = 0
        while True:
            attempt += 1
            ref, provider = self._next_ref()
            verdict = await self._attempt(story, ref, provider, attempt, feedback)
            if verdict is None:
                return  # cancelled or the run stopped
            if verdict.passed:
                outcome.state = StoryState.PASSED
                outcome.score = verdict.score
                outcome.model = ref.label
                return
            feedback = verdict.feedback
            if attempt < max_attempts:
                continue
            policy: str = self.config.escalation
            if policy == "ask":
                policy = await self.presenter.on_escalation(story, outcome)
            if policy == "retry":
                max_attempts += 1
                continue
            if policy == "high":
                await self._escalate_high(story, outcome, attempt + 1, feedback)
                return
            outcome.state = StoryState.BLOCKED
            outcome.reason = "retries_exhausted"
            return

    async def _attempt(
        self,
        story: Story,
        ref: H2LModelRef,
        provider: Any,
        attempt: int,
        feedback: Optional[str],
    ) -> Optional[Verdict]:
        worker, judge = self._components()
        outcome = self.outcomes[story.id]
        outcome.attempts = attempt
        agent_id = self._agent_id(story)
        await self.presenter.on_story_start(story, ref, attempt, agent_id)
        token = _current_story.set(f"{story.id} · {ref.label}")
        try:
            report: WorkerReport = await worker.run(
                story,
                ref,
                provider,
                worker_tools=list(self.config.worker_tools),
                allow_read=self.config.allow_worker_read,
                max_iterations=story.max_iterations
                or self.config.worker_max_iterations,
                run_id=self.run_id,
                feedback=feedback,
                attempt=attempt,
                agent_id=agent_id,
                on_tool_call=self._observer(story, agent_id),
                on_wait=self._wait_observer(f"{story.id} {ref.label}"),
            )
        finally:
            _current_story.reset(token)
        self._add_usage("low", report.usage)
        outcome.usage = outcome.usage.add(report.usage)
        await self.presenter.on_report(story, report)

        if report.cancelled:
            outcome.state = StoryState.CANCELLED
            outcome.reason = "cancelled"
            return None
        if not report.success and matches_rate_limit(report.error or ""):
            # Transient: stop the run, leave the story pending for --resume.
            self._stopped = True
            self._stop_cause = "rate_limited"
            outcome.state = StoryState.PENDING
            outcome.attempts = attempt - 1
            return None

        with self._high_context():
            verdict: Verdict = await judge.check(story, report)
        self._add_usage("high", verdict.usage)
        outcome.usage = outcome.usage.add(verdict.usage)
        outcome.verdicts.append(verdict)
        await self.presenter.on_verdict(story, verdict)
        return verdict

    async def _escalate_high(
        self,
        story: Story,
        outcome: StoryOutcome,
        attempt: int,
        feedback: Optional[str],
    ) -> None:
        worker, _ = self._components()
        tiers = self._ensure_tiers()
        ref = self.config.high
        outcome.attempts = attempt
        agent_id = self._agent_id(story)
        await self.presenter.on_story_start(story, ref, attempt, agent_id)
        token = _current_story.set(f"{story.id} · {ref.label}")
        try:
            with self._high_context():
                report: WorkerReport = await worker.run(
                    story,
                    ref,
                    tiers.high,
                    worker_tools=[t.name for t in CODING_AGENT_TOOLS],
                    allow_read=True,
                    max_iterations=30,
                    run_id=self.run_id,
                    feedback=feedback,
                    attempt=attempt,
                    agent_id=agent_id,
                    on_tool_call=self._observer(story, agent_id),
                    on_wait=self._wait_observer(f"{story.id} {ref.label}"),
                )
        finally:
            _current_story.reset(token)
        self._add_usage("high", report.usage)
        outcome.usage = outcome.usage.add(report.usage)
        await self.presenter.on_report(story, report)
        if report.cancelled:
            outcome.state = StoryState.CANCELLED
            outcome.reason = "cancelled"
            return
        verdict = Judge.deterministic(story, report)
        if verdict is None:
            outcome.state = StoryState.PASSED
            outcome.reason = "escalated_to_high"
            outcome.model = ref.label
            return
        outcome.verdicts.append(verdict)
        await self.presenter.on_verdict(story, verdict)
        outcome.state = StoryState.BLOCKED
        outcome.reason = "escalation_failed"

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _agent_id(story: Story) -> str:
        return f"h2l-{story.id}-{uuid.uuid4().hex[:8]}"

    def _observer(self, story: Story, agent_id: str) -> Optional[Any]:
        """Forward worker tool calls to the presenter when it wants them."""
        hook = getattr(self.presenter, "on_tool_call", None)
        if hook is None:
            return None

        async def observe(tool_call: Any, result: Any) -> None:
            try:
                await hook(story, tool_call, result, agent_id)
            except Exception as exc:  # presentation must never break a worker
                logger.debug("H2L on_tool_call presenter hook failed: %s", exc)

        return observe

    def _next_ref(self) -> Tuple[H2LModelRef, Any]:
        tiers = self._ensure_tiers()
        pool = list(zip(self.config.low, tiers.low))
        ref, provider = pool[self._pool_index % len(pool)]
        self._pool_index += 1
        return ref, provider

    def _add_usage(self, tier: str, usage: UsageTotals) -> None:
        self.usage_by_tier[tier] = self.usage_by_tier[tier].add(usage)

    def _summary(
        self, goal: str, stop_cause: str, error: Optional[str] = None
    ) -> RunSummary:
        stories: List[StoryOutcome] = []
        if self.plan is not None:
            stories = [self.outcomes[s.id] for s in self.plan.stories]
        if (
            stop_cause == "done"
            and stories
            and not all(s.state == StoryState.PASSED for s in stories)
        ):
            stop_cause = "partial"
        summary = RunSummary(
            goal=goal,
            stories=stories,
            usage_by_tier=dict(self.usage_by_tier),
            stop_cause=stop_cause,
        )
        if error:
            summary.error = error
        return summary

    @contextlib.contextmanager
    def _serialized_approvals(self) -> Iterator[None]:
        """Prompts arrive one at a time and say which story is asking (PRD §6)."""
        agent_engine = self.agent_engine
        managers = [
            m
            for m in (
                getattr(agent_engine, "approval", None),
                getattr(agent_engine, "enhanced_approval", None),
            )
            if m is not None
        ]
        lock = asyncio.Lock()
        originals: List[Tuple[Any, Any]] = []
        for manager in managers:
            original = getattr(manager, "approval_callback", None)
            if original is None or not callable(original):
                continue

            def _make(orig: Any) -> Any:
                async def wrapped(operation: Any) -> Any:
                    label = _current_story.get()
                    description = getattr(operation, "description", None)
                    if label and isinstance(description, str):
                        prefix = f"[H2L {label}] "
                        if not description.startswith(prefix):
                            operation.description = prefix + description
                    async with lock:
                        result = orig(operation)
                        if inspect.isawaitable(result):
                            result = await result
                        return result

                return wrapped

            manager.approval_callback = _make(original)
            originals.append((manager, original))
        try:
            yield
        finally:
            for manager, original in originals:
                manager.approval_callback = original
