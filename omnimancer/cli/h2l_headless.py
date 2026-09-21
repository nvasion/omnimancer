"""Headless H2L surface (PRD §8): flags → config, presenter → emitter.

``HeadlessRunner._run`` hands off here when ``--h2l`` is set. This module
never imports ``headless.py`` at module load (it would be circular); the
runner passes what the presenter needs.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import ChatResponse, Config
from ..h2l.models import (
    H2LConfig,
    H2LModelRef,
    Plan,
    RunSummary,
    Story,
    StoryOutcome,
    StoryState,
    Verdict,
    WorkerReport,
    parse_model_ref,
)
from ..h2l.orchestrator import H2LRun, PlanDecision
from ..h2l.refs import H2LConfigError, validate_tiers
from ..h2l.render import exit_code_for, render_plan_text, render_summary_text

logger = logging.getLogger(__name__)


@dataclass
class H2LOptions:
    """The ``--h2l*`` flags as parsed by ``cli_main``."""

    enabled: bool = False
    high: Optional[str] = None
    low: List[str] = field(default_factory=list)
    parallel: Optional[int] = None
    retries: Optional[int] = None
    threshold: Optional[int] = None
    escalation: Optional[str] = None
    plan_only: bool = False
    plan_iterations: Optional[int] = None


def resolve_h2l_config(config: Config, options: H2LOptions) -> H2LConfig:
    """Flags override the config block field by field; never persisted."""
    base: Dict[str, Any] = {}
    block = getattr(config, "h2l", None)
    if block is not None:
        base = block.model_dump()
    default_provider = getattr(config, "default_provider", "") or ""

    if options.high:
        base["high"] = parse_model_ref(options.high, default_provider)
    if options.low:
        base["low"] = [parse_model_ref(ref, default_provider) for ref in options.low]
    if options.parallel is not None:
        base["max_parallel"] = options.parallel
    if options.retries is not None:
        base["max_retries"] = options.retries
    if options.threshold is not None:
        base["pass_threshold"] = options.threshold
    if options.escalation:
        base["escalation"] = options.escalation
    if options.plan_iterations is not None:
        base["planner_max_iterations"] = options.plan_iterations

    if not base.get("high"):
        raise H2LConfigError(
            "H2L: no high model configured (set h2l.high or pass --high)"
        )
    if not base.get("low"):
        raise H2LConfigError(
            "H2L: no low models configured (set h2l.low or pass --low)"
        )

    # Headless is unattended: nobody can answer "ask", and the plan gate is
    # replaced by the plan-only mode (PRD §8).
    if base.get("escalation") == "ask":
        base["escalation"] = "high"
    base["plan_approval"] = False
    try:
        return H2LConfig(**base)
    except ValueError as exc:
        raise H2LConfigError(f"H2L: invalid configuration: {exc}") from exc


class HeadlessPresenter:
    """Turns run events into stream-json lines or stderr progress."""

    def __init__(self, emitter: Any, text_mode: bool) -> None:
        self.emitter = emitter
        self.text_mode = text_mode
        self.summary: Optional[RunSummary] = None
        self._stderr = getattr(emitter, "_stderr", sys.stderr)

    def _progress(self, line: str) -> None:
        if self.text_mode:
            self._stderr.write(line + "\n")
            self._stderr.flush()

    async def on_wait(self, label: str, seconds: float) -> None:
        self.emitter.emit_h2l("waiting", {"who": label, "seconds": int(seconds)})
        self._progress(f"waiting on {label} ({int(seconds)}s)")

    async def on_planner_tool_call(self, tool_call: Any, result: Any) -> None:
        arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        error = getattr(result, "error", None)
        if tool_call.name == "submit_plan":
            # The full plan arrives in the `plan` event; here a summary keeps
            # the line short and the rejection reason readable.
            stories = arguments.get("stories")
            arguments = {
                "goal": str(arguments.get("goal") or "")[:120],
                "stories": (
                    [str(s.get("id")) for s in stories if isinstance(s, dict)]
                    if isinstance(stories, list)
                    else []
                ),
            }
        # `error` precedes `arguments` so a truncated log line still shows why
        # a call failed.
        self.emitter.emit_h2l(
            "planner_tool",
            {"name": tool_call.name, "error": error, "arguments": arguments},
        )
        if tool_call.name == "submit_plan":
            target = f"{len(arguments['stories'])} stories"
        else:
            target = (
                arguments.get("file_path")
                or arguments.get("path")
                or arguments.get("pattern")
                or arguments.get("command")
                or ""
            )
        word = "rejected" if tool_call.name == "submit_plan" else "error"
        status = f" — {word}: {str(error)[:300]}" if error else ""
        self._progress(f"planner › {tool_call.name} {target}{status}".rstrip())

    async def on_plan(self, plan: Plan, model_used: str = "") -> None:
        # `model_used` first: it is the evidence the tier ref was honoured,
        # and the plan body can be long.
        self.emitter.emit_h2l(
            "plan", {"model_used": model_used, "plan": plan.model_dump()}
        )
        if model_used:
            self._progress(f"planned by {model_used}")
        self._progress(render_plan_text(plan))

    async def approve_plan(self, plan: Plan) -> PlanDecision:
        return PlanDecision(accepted=True, plan=plan)

    async def on_story_start(
        self, story: Story, ref: H2LModelRef, attempt: int, agent_id: str = ""
    ) -> None:
        self.emitter.emit_h2l(
            "story_start",
            {
                "story_id": story.id,
                "attempt": attempt,
                "provider": ref.provider,
                "model": ref.model,
                "agent_id": agent_id,
            },
        )
        self._progress(f"{story.id} attempt {attempt} on {ref.label}: {story.title}")

    async def on_tool_call(
        self, story: Story, tool_call: Any, result: Any, agent_id: str = ""
    ) -> None:
        self.emitter.emit_tool_use(
            tool_call.name, tool_call.arguments, story_id=story.id, agent_id=agent_id
        )
        self.emitter.emit_tool_result(
            tool_call.name,
            result.content or "",
            result.error,
            story_id=story.id,
            agent_id=agent_id,
        )

    async def on_report(self, story: Story, report: WorkerReport) -> None:
        self.emitter.emit_h2l(
            "story_report",
            {
                "story_id": story.id,
                "attempt": report.attempt,
                "success": report.success,
                "files_changed": report.files_changed,
                "verify_exit": report.verify_exit,
            },
        )

    async def on_verdict(self, story: Story, verdict: Verdict) -> None:
        self.emitter.emit_h2l(
            "story_verdict",
            {
                "story_id": story.id,
                "attempt": verdict.attempt,
                "score": verdict.score,
                "passed": verdict.passed,
                "failures": verdict.failures,
                "judge": verdict.judge,
            },
        )
        status = "passed" if verdict.passed else "failed"
        self._progress(
            f"{story.id} attempt {verdict.attempt}: judge {verdict.score} {status}"
            + (f" ({'; '.join(verdict.failures)})" if verdict.failures else "")
        )

    async def on_escalation(self, story: Story, outcome: StoryOutcome) -> str:
        return "high"

    async def on_done(self, summary: RunSummary) -> None:
        self.summary = summary
        for outcome in summary.stories:
            if outcome.state == StoryState.BLOCKED:
                self.emitter.emit_h2l(
                    "story_blocked",
                    {"story_id": outcome.id, "reason": outcome.reason or "blocked"},
                )
            elif outcome.state == StoryState.SKIPPED:
                self.emitter.emit_h2l(
                    "story_skipped",
                    {"story_id": outcome.id, "reason": outcome.reason or "skipped"},
                )
        self.emitter.emit_h2l(
            "done",
            {
                "passed": summary.passed,
                "blocked": summary.blocked,
                "skipped": summary.skipped,
                "cancelled": summary.cancelled,
            },
        )


def _aggregate_response(text: str, usage: Dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content=text,
        model_used="h2l",
        tokens_used=int(usage["input_tokens"]) + int(usage["output_tokens"]),
        input_tokens=int(usage["input_tokens"]),
        output_tokens=int(usage["output_tokens"]),
        cost_estimate=float(usage["total_cost_usd"]),
        stop_reason="end_turn",
    )


def _usage_dict(summary: RunSummary) -> Dict[str, Any]:
    total = summary.usage_by_tier.get("high", None)
    low = summary.usage_by_tier.get("low", None)
    if total is None or low is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost_usd": 0.0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
    combined = total.add(low)
    return {
        "input_tokens": combined.input_tokens,
        "output_tokens": combined.output_tokens,
        "total_cost_usd": combined.cost_usd,
        "cache_read_input_tokens": combined.cache_read_input_tokens,
        "cache_creation_input_tokens": combined.cache_creation_input_tokens,
    }


async def run_h2l_headless(
    runner: Any, prompt: str, options: H2LOptions, text_mode: bool
) -> int:
    """Drive one H2L run through the headless runner's emitter (PRD §8)."""
    engine = runner._engine
    emitter = runner._emitter
    model = runner._model or "unknown"
    provider_name = runner._provider_name
    config = engine.config_manager.get_config()

    try:
        h2l_config = resolve_h2l_config(config, options)
        tiers = validate_tiers(h2l_config, engine.config_manager, engine)
    except H2LConfigError as exc:
        emitter.emit_error(
            str(exc), model=model, provider=provider_name, stop_cause="error"
        )
        return 1

    # The init line (and every error blob after it) names the H2L high tier,
    # not the session model: the session model never takes part in an H2L run.
    model = h2l_config.high.label
    provider_name = h2l_config.high.provider
    emitter.emit_init(model)
    runner._turn_notifier.extra["h2l"] = True
    presenter = HeadlessPresenter(emitter, text_mode)
    run = H2LRun(
        engine,
        h2l_config,
        presenter,
        cwd=Path.cwd(),
        session_id=runner._turn_notifier.session_id,
        tiers=tiers,
    )
    # Without auto-approval every Bash call is denied; don't offer it.
    run.planner_bash = bool(runner._no_approval)

    plan_only = options.plan_only or not runner._no_approval
    if plan_only:
        result = await run.plan_goal(prompt)
        if result.plan is None:
            emitter.emit_error(
                result.error or "plan_failed",
                model=model,
                provider=provider_name,
                stop_cause="plan_failed",
            )
            return 1
        model_used = getattr(result, "model_used", "") or ""
        await presenter.on_plan(result.plan, model_used)
        text = render_plan_text(result.plan)
        plan_usage = _usage_dict(
            RunSummary(goal=prompt, usage_by_tier=run.usage_by_tier)
        )
        runner._turn_notifier.record_assistant(
            text, _aggregate_response(text, plan_usage)
        )
        emitter.emit_result(
            text,
            model,
            plan_usage,
            plan_usage["total_cost_usd"],
            "end_turn",
            stop_cause="plan_only",
            provider=provider_name,
            subtype="h2l_plan_only",
            h2l={
                "plan": result.plan.model_dump(),
                "model_used": model_used,
                "usage_by_tier": RunSummary(
                    goal=prompt, usage_by_tier=run.usage_by_tier
                ).to_result_dict()["usage_by_tier"],
            },
        )
        return 0

    summary = await run.run(prompt)
    if summary.stop_cause == "plan_failed":
        emitter.emit_error(
            summary.error or "plan_failed",
            model=model,
            provider=provider_name,
            stop_cause="plan_failed",
        )
        return 1

    text = render_summary_text(summary)
    usage = _usage_dict(summary)
    runner._turn_notifier.record_assistant(text, _aggregate_response(text, usage))
    emitter.emit_result(
        text,
        model,
        usage,
        usage["total_cost_usd"],
        "end_turn",
        stop_cause=summary.stop_cause,
        num_turns=len(summary.stories),
        provider=provider_name,
        h2l=summary.to_result_dict(),
    )
    return exit_code_for(summary)
