"""Plain-text renderings shared by the headless and TUI surfaces."""

from __future__ import annotations

from typing import List

from .models import Plan, RunSummary, StoryState


def render_plan_text(plan: Plan) -> str:
    lines: List[str] = [
        f"Goal: {plan.goal}",
        "",
        "Design:",
        plan.design or "(none)",
        "",
    ]
    lines.append("Stories:")
    for story in plan.stories:
        deps = ", ".join(story.depends_on) or "-"
        verify = story.verify or "-"
        lines.append(f"  {story.id}  {story.title}")
        lines.append(f"      files: {', '.join(story.files) or '-'}")
        lines.append(f"      depends: {deps}   verify: {verify}")
    return "\n".join(lines)


def render_summary_text(summary: RunSummary) -> str:
    lines: List[str] = []
    for outcome in summary.stories:
        score = f"judge {outcome.score}" if outcome.score is not None else ""
        model = outcome.model or ""
        reason = f"({outcome.reason})" if outcome.reason else ""
        state = outcome.state.value
        lines.append(
            f"  {outcome.id:<4} {state:<9} attempts {outcome.attempts}  "
            f"{model} {score} {reason}".rstrip()
        )
    counts = (
        f"Done: {summary.passed} passed, {summary.blocked} blocked, "
        f"{summary.skipped} skipped, {summary.cancelled} cancelled."
    )
    if summary.stop_cause not in ("done", "partial"):
        counts += f" Stopped: {summary.stop_cause}."
    high = summary.usage_by_tier.get("high")
    low = summary.usage_by_tier.get("low")
    usage = ""
    if high is not None and low is not None:
        usage = (
            f"Tokens  high {high.input_tokens} in / {high.output_tokens} out "
            f"(${high.cost_usd:.2f})   low {low.input_tokens} in / "
            f"{low.output_tokens} out (${low.cost_usd:.2f})"
        )
    return "\n".join(lines + [counts] + ([usage] if usage else []))


def exit_code_for(summary: RunSummary) -> int:
    """Exit-code contract (PRD §8): 0 all passed, 3 partial, 4 rate limited."""
    if summary.stop_cause == "rate_limited":
        return 4
    if summary.stop_cause == "plan_failed":
        return 1
    if summary.all_passed:
        return 0
    return 3


__all__ = ["render_plan_text", "render_summary_text", "exit_code_for", "StoryState"]
