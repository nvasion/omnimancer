"""Judge: deterministic checks, then the high model grades the diff (PRD §5).

The model never sees the worker's narrative or tool log: only the story, the
acceptance criteria, the diff, and the verify output. That is the direct
answer to judge bias, and it keeps judge tokens low.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..core.models import ChatContext, ToolCall, ToolDefinition, ToolResult
from .loop import IsolatedLoop
from .models import Story, UsageTotals, Verdict, WorkerReport
from .prompts import JUDGE_SYSTEM, render_story

logger = logging.getLogger(__name__)

DIFF_PROMPT_CHARS = 40_000
VERIFY_PROMPT_CHARS = 8_000

SUBMIT_VERDICT_TOOL = ToolDefinition(
    name="submit_verdict",
    description="Submit the review verdict for this story attempt. Call exactly once.",
    parameters={
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "description": "0-100; how completely the diff meets the criteria",
            },
            "passed": {
                "type": "boolean",
                "description": "True only when every acceptance criterion is met",
            },
            "failures": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One entry per failed criterion (empty when passed)",
            },
            "feedback": {
                "type": "string",
                "description": (
                    "Instructions a worker can act on without any other context"
                ),
            },
        },
        "required": ["score", "passed", "failures", "feedback"],
    },
)

NO_VERDICT_NUDGE = (
    "You did not call submit_verdict. Call it now with score, passed, "
    "failures and feedback."
)


class Judge:
    def __init__(
        self,
        provider: Any,
        pass_threshold: int,
        max_iterations: int = 3,
        on_wait: Optional[Any] = None,
    ):
        self.provider = provider
        self.pass_threshold = int(pass_threshold)
        self.max_iterations = max(1, max_iterations)
        self.on_wait = on_wait

    # ------------------------------------------------------------ deterministic

    @staticmethod
    def deterministic(story: Story, report: WorkerReport) -> Optional[Verdict]:
        """The free checks; the first hit fails the attempt (PRD §5 order)."""

        def fail(failure: str, feedback: str) -> Verdict:
            return Verdict(
                story_id=story.id,
                attempt=report.attempt,
                judge="deterministic",
                failures=[failure],
                feedback=feedback,
            )

        if report.cancelled:
            return fail(
                "worker cancelled",
                "The attempt was cancelled before completion. Start again.",
            )
        if not report.success:
            error = report.error or "unknown error"
            return fail(
                f"worker failed: {error}",
                f"The previous attempt failed before completing: {error}. "
                "Start again from the story's first step.",
            )
        if report.verify_exit not in (None, 0):
            output = (report.verify_output or "")[:2000]
            return fail(
                f"verify failed (exit {report.verify_exit})",
                f"The verify command `{story.verify}` exited {report.verify_exit}. "
                f"Output:\n{output}\nFix the changes until the command exits 0.",
            )
        if report.files_outside_scope:
            listed = ", ".join(report.files_outside_scope)
            allowed = ", ".join(story.files) or "(none)"
            return fail(
                f"modified files outside story scope: {listed}",
                f"You modified files outside the story's list: {listed}. "
                f"Revert those changes. Only touch: {allowed}.",
            )
        if not report.diff.strip() and story.files:
            return fail(
                "no changes made",
                "No changes were made to: "
                + ", ".join(story.files)
                + ". Apply the story's edits exactly as written.",
            )
        return None

    # ------------------------------------------------------------ model stage

    async def check(self, story: Story, report: WorkerReport) -> Verdict:
        early = self.deterministic(story, report)
        if early is not None:
            return early

        verdicts: List[Verdict] = []

        async def execute(tc: ToolCall) -> ToolResult:
            if tc.name != "submit_verdict":
                return ToolResult(content="", error=f"Unknown tool: {tc.name}")
            return self._accept(tc, story, report, verdicts)

        context = ChatContext(
            messages=[],
            current_model=getattr(self.provider, "model", "") or "",
            session_id=f"h2l-judge-{story.id}-{report.attempt}",
        )
        loop = IsolatedLoop(
            self.provider,
            context,
            [SUBMIT_VERDICT_TOOL],
            execute,
            self.max_iterations,
            stop_tools={"submit_verdict"},
            on_wait=self.on_wait,
        )
        outcome = await loop.run(self._message(story, report))
        usage = outcome.usage
        if (
            outcome.stop == "done"
            and not verdicts
            and outcome.iterations < self.max_iterations
        ):
            nudged = await IsolatedLoop(
                self.provider,
                context,
                [SUBMIT_VERDICT_TOOL],
                execute,
                max(1, self.max_iterations - outcome.iterations),
                stop_tools={"submit_verdict"},
                on_wait=self.on_wait,
            ).run(NO_VERDICT_NUDGE)
            usage = usage.add(nudged.usage)
            outcome.stop, outcome.error = nudged.stop, nudged.error

        if verdicts:
            verdict = verdicts[-1]
            verdict.usage = usage
            return verdict

        error = (
            f"judge error: {outcome.error}"
            if outcome.stop == "error"
            else "judge returned no verdict"
        )
        return Verdict(
            story_id=story.id,
            attempt=report.attempt,
            judge="model",
            failures=[error],
            feedback="The review could not be completed; the attempt will be retried.",
            error=error,
            usage=usage,
        )

    def _accept(
        self,
        tc: ToolCall,
        story: Story,
        report: WorkerReport,
        verdicts: List[Verdict],
    ) -> ToolResult:
        args = tc.arguments if isinstance(tc.arguments, dict) else {}
        try:
            score = int(args.get("score", 0))
            passed = bool(args.get("passed", False))
            failures = [str(f) for f in (args.get("failures") or [])]
            feedback = str(args.get("feedback") or "")
            verdict = Verdict(
                story_id=story.id,
                attempt=report.attempt,
                score=score,
                passed=passed,
                failures=failures,
                feedback=feedback,
                judge="model",
            )
        except (ValueError, TypeError) as exc:
            return ToolResult(content="", error=f"submit_verdict rejected: {exc}")
        if verdict.passed and verdict.score < self.pass_threshold:
            verdict.passed = False
            verdict.failures = verdict.failures + [
                f"score {verdict.score} below pass threshold {self.pass_threshold}"
            ]
        verdicts.append(verdict)
        return ToolResult(content="Verdict received.")

    def _message(self, story: Story, report: WorkerReport) -> str:
        diff = report.diff
        if len(diff) > DIFF_PROMPT_CHARS:
            omitted = len(diff) - DIFF_PROMPT_CHARS
            diff = (
                diff[:DIFF_PROMPT_CHARS] + f"\n[... {omitted} characters truncated ...]"
            )
        verify = report.verify_output or "(no verify command)"
        if len(verify) > VERIFY_PROMPT_CHARS:
            verify = verify[:VERIFY_PROMPT_CHARS] + "\n[... truncated ...]"
        return "\n\n".join(
            [
                JUDGE_SYSTEM,
                render_story(story),
                f"Attempt: {report.attempt}",
                f"Diff (scoped to the story's files):\n```diff\n{diff}\n```",
                f"Verify exit code: {report.verify_exit}\nVerify output:\n{verify}",
                "Call submit_verdict now.",
            ]
        )


def usage_of(verdict: Verdict) -> UsageTotals:
    return verdict.usage
