"""The ``/h2l`` slash command and its TUI presenter (PRD §7).

Kept out of ``command_dispatch.py`` (2.4k lines) on purpose; the dispatcher
delegates here. The presenter talks to the user through the host CLI's
console and an injectable ``_h2l_input`` coroutine so tests never touch stdin.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.panel import Panel
from rich.table import Table

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
)
from ..h2l.orchestrator import H2LRun, PlanDecision
from ..h2l.refs import H2LConfigError, validate_tiers
from ..h2l.render import render_summary_text

logger = logging.getLogger(__name__)

USAGE = (
    "Usage:\n"
    "  /h2l <goal>                 plan, review the stories, then run them\n"
    "  /h2l plan <goal>            plan only; keeps the plan pending\n"
    "  /h2l run                    run the pending plan\n"
    "  /h2l status                 show the pending plan\n"
    "  /h2l config                 show the effective h2l block\n"
    "  /h2l config set <field> <value> [<field> <value> ...]"
)

_SETTABLE = {
    "enabled",
    "high",
    "low",
    "max_parallel",
    "max_retries",
    "pass_threshold",
    "worker_tools",
    "allow_worker_read",
    "escalation",
    "plan_approval",
    "worker_max_iterations",
    "planner_max_iterations",
    "high_max_tokens",
    "low_max_tokens",
}
_LIST_FIELDS = {"low", "worker_tools"}
_INT_FIELDS = {
    "max_parallel",
    "max_retries",
    "pass_threshold",
    "worker_max_iterations",
    "planner_max_iterations",
    "high_max_tokens",
    "low_max_tokens",
}
# Optional integers: "off"/"none" clears them.
_OPTIONAL_INT_FIELDS = {"planner_max_iterations", "high_max_tokens", "low_max_tokens"}
_BOOL_FIELDS = {"enabled", "allow_worker_read", "plan_approval"}


# ---------------------------------------------------------------- presenter


class TUIPresenter:
    def __init__(self, cli: Any) -> None:
        self.cli = cli
        self.console = cli.console
        self.summary: Optional[RunSummary] = None

    async def _ask(self, prompt: str) -> str:
        asker = getattr(self.cli, "_h2l_input", None)
        if asker is not None:
            answer: str = await asker(prompt)
            return answer
        return await asyncio.to_thread(input, prompt)

    async def _edit_text(self, text: str) -> str:
        """Open ``$EDITOR`` on the text, or fall back to a multi-line prompt."""
        editor = os.environ.get("EDITOR")
        if editor:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".md", delete=False, encoding="utf-8"
            ) as handle:
                handle.write(text)
                path = handle.name
            try:
                await asyncio.to_thread(
                    subprocess.run, [*shlex.split(editor), path], check=False
                )
                return Path(path).read_text(encoding="utf-8")
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        self.console.print(
            "Enter the new instructions; finish with a line containing only '.'"
        )
        lines: List[str] = []
        while True:
            line = await self._ask("")
            if line.strip() == ".":
                break
            lines.append(line)
        return "\n".join(lines) or text

    def _plan_table(self, plan: Plan) -> Table:
        table = Table(title="H2L plan")
        table.add_column("ID", style="cyan")
        table.add_column("Title")
        table.add_column("Files", style="dim")
        table.add_column("Depends", style="magenta")
        table.add_column("Verify", style="dim")
        for story in plan.stories:
            table.add_row(
                story.id,
                story.title,
                "\n".join(story.files),
                ", ".join(story.depends_on) or "-",
                story.verify or "-",
            )
        return table

    async def on_plan(self, plan: Plan, model_used: str = "") -> None:
        if plan.design:
            title = f"Design (planned by {model_used})" if model_used else "Design"
            self.console.print(Panel(plan.design, title=title))
        self.console.print(self._plan_table(plan))

    async def approve_plan(self, plan: Plan) -> PlanDecision:
        stories = list(plan.stories)
        while True:
            answer = (
                await self._ask("[a]ccept  [d]rop <id>  [e]dit <id>  [c]ancel > ")
            ).strip()
            parts = answer.split(None, 1)
            verb = parts[0].lower() if parts else ""
            arg = parts[1].strip() if len(parts) > 1 else ""
            if verb in ("a", "accept"):
                if not stories:
                    self.cli._show_error("Nothing left to run.")
                    return PlanDecision(accepted=False)
                return PlanDecision(
                    accepted=True, plan=plan.model_copy(update={"stories": stories})
                )
            if verb in ("c", "cancel"):
                return PlanDecision(accepted=False)
            if verb in ("d", "drop", "e", "edit"):
                target = next((s for s in stories if s.id == arg), None)
                if target is None:
                    self.cli._show_error(f"No story '{arg}' in the plan.")
                    continue
                if verb in ("d", "drop"):
                    stories = [s for s in stories if s.id != arg]
                    for s in stories:
                        s.depends_on = [d for d in s.depends_on if d != arg]
                    self.cli._show_info(f"Dropped {arg}.")
                else:
                    new_text = await self._edit_text(target.instructions)
                    stories = [
                        (
                            s.model_copy(update={"instructions": new_text})
                            if s.id == arg
                            else s
                        )
                        for s in stories
                    ]
                    self.cli._show_info(f"Updated {arg}.")
                self.console.print(
                    self._plan_table(plan.model_copy(update={"stories": stories}))
                )
                continue
            self.cli._show_error("Answer a, d <id>, e <id>, or c.")

    async def on_story_start(
        self, story: Story, ref: H2LModelRef, attempt: int, agent_id: str = ""
    ) -> None:
        self.console.print(
            f"[cyan]{story.id}[/cyan] attempt {attempt} on [dim]{ref.label}[/dim]: "
            f"{story.title}"
        )

    async def on_tool_call(
        self, story: Story, tool_call: Any, result: Any, agent_id: str = ""
    ) -> None:
        status = "[red]error[/red]" if getattr(result, "error", None) else "ok"
        self.console.print(f"  [dim]{story.id} › {tool_call.name} {status}[/dim]")

    async def on_wait(self, label: str, seconds: float) -> None:
        self.console.print(f"  [dim]waiting on {label} ({int(seconds)}s)[/dim]")

    async def on_planner_tool_call(self, tool_call: Any, result: Any) -> None:
        arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        if tool_call.name == "submit_plan":
            stories = arguments.get("stories")
            count = len(stories) if isinstance(stories, list) else 0
            target = f"{count} stories"
        else:
            target = (
                arguments.get("file_path")
                or arguments.get("path")
                or arguments.get("pattern")
                or arguments.get("command")
                or ""
            )
        error = getattr(result, "error", None)
        word = "rejected" if tool_call.name == "submit_plan" else "error"
        status = f"[red]{word}: {str(error)[:300]}[/red]" if error else ""
        self.console.print(
            f"  [dim]planner › {tool_call.name} {target}[/dim] {status}".rstrip()
        )

    async def on_report(self, story: Story, report: WorkerReport) -> None:
        verify = (
            f"verify exit {report.verify_exit}"
            if report.verify_exit is not None
            else ""
        )
        files = ", ".join(report.files_changed) or "no files changed"
        self.console.print(f"  [dim]{story.id}: {files} {verify}[/dim]")

    async def on_verdict(self, story: Story, verdict: Verdict) -> None:
        if verdict.passed:
            self.console.print(
                f"  [green]{story.id} passed[/green] (judge {verdict.score})"
            )
        else:
            failures = "; ".join(verdict.failures)
            self.console.print(
                f"  [yellow]{story.id} failed[/yellow] (judge {verdict.score}, "
                f"{verdict.judge}): {failures}"
            )

    async def on_escalation(self, story: Story, outcome: StoryOutcome) -> str:
        while True:
            answer = (
                (
                    await self._ask(
                        f"{story.id} failed {outcome.attempts} attempts. "
                        "[h]igh model does it  [b]lock  [r]etry once more > "
                    )
                )
                .strip()
                .lower()
            )
            if answer in ("h", "high"):
                return "high"
            if answer in ("b", "block"):
                return "block"
            if answer in ("r", "retry"):
                return "retry"
            self.cli._show_error("Answer h, b, or r.")

    async def on_done(self, summary: RunSummary) -> None:
        self.summary = summary
        table = Table(title="H2L summary")
        table.add_column("ID", style="cyan")
        table.add_column("State")
        table.add_column("Attempts", justify="right")
        table.add_column("Model", style="dim")
        table.add_column("Score", justify="right")
        table.add_column("Reason", style="dim")
        styles = {
            StoryState.PASSED: "green",
            StoryState.BLOCKED: "red",
            StoryState.SKIPPED: "yellow",
            StoryState.CANCELLED: "yellow",
        }
        for outcome in summary.stories:
            style = styles.get(outcome.state, "")
            state = (
                f"[{style}]{outcome.state.value}[/{style}]"
                if style
                else outcome.state.value
            )
            table.add_row(
                outcome.id,
                state,
                str(outcome.attempts),
                outcome.model or "-",
                str(outcome.score) if outcome.score is not None else "-",
                outcome.reason or "",
            )
        self.console.print(table)
        high = summary.usage_by_tier.get("high")
        low = summary.usage_by_tier.get("low")
        if high is not None and low is not None:
            self.console.print(
                f"[dim]Tokens  high {high.input_tokens} in / {high.output_tokens} out "
                f"(${high.cost_usd:.2f})   low {low.input_tokens} in / "
                f"{low.output_tokens} out (${low.cost_usd:.2f})[/dim]"
            )


# ---------------------------------------------------------------- command


def _session_id(cli: Any) -> str:
    notifier = getattr(cli, "turn_notifier", None)
    session = getattr(notifier, "session_id", None)
    return str(session) if session else uuid.uuid4().hex


def _h2l_config(cli: Any) -> Optional[H2LConfig]:
    config = cli.engine.config_manager.get_config()
    block: Optional[H2LConfig] = getattr(config, "h2l", None)
    if block is None or not block.enabled:
        cli._show_error(
            "H2L is not configured. Add an 'h2l' block to config or run: "
            "/h2l config set high <provider:model> low <provider:model[,...]>"
        )
        return None
    return block


def _build_run(
    cli: Any, config: H2LConfig, presenter: TUIPresenter
) -> Optional[H2LRun]:
    try:
        tiers = validate_tiers(config, cli.engine.config_manager, cli.engine)
    except H2LConfigError as exc:
        cli._show_error(str(exc))
        return None
    return H2LRun(
        cli.engine,
        config,
        presenter,
        cwd=Path.cwd(),
        session_id=_session_id(cli),
        tiers=tiers,
    )


def _post_summary(cli: Any, summary: RunSummary) -> None:
    """One assistant message in the parent conversation, never the transcripts."""
    text = "H2L run summary:\n" + render_summary_text(summary)
    try:
        _, model = cli.engine.runtime_identity()
    except Exception:
        model = ""
    chat = getattr(cli.engine, "chat_manager", None)
    if chat is not None:
        chat.add_assistant_message(text, model or "h2l")


async def handle_h2l_command(cli: Any, command: Any) -> None:
    args: List[str] = list(command.args or [])
    if not args:
        cli._show_info(USAGE)
        return
    sub = args[0].lower()

    if sub == "config":
        await _handle_config(cli, args[1:])
        return
    if sub == "status":
        pending = getattr(cli, "_h2l_pending_plan", None)
        if pending is None:
            cli._show_info("No pending plan. Use '/h2l plan <goal>' to create one.")
        else:
            await TUIPresenter(cli).on_plan(pending)
            cli._show_info("Pending plan ready: '/h2l run' executes it.")
        return

    config = _h2l_config(cli)
    if config is None:
        return
    presenter = TUIPresenter(cli)

    if sub == "plan":
        goal = " ".join(args[1:]).strip()
        if not goal:
            cli._show_error("Usage: /h2l plan <goal>")
            return
        run = _build_run(cli, config, presenter)
        if run is None:
            return
        cli._show_info(f"Planning with {config.high.label}…")
        result = await run.plan_goal(goal)
        if result.plan is None:
            cli._show_error(result.error or "H2L: planning failed")
            return
        cli._h2l_pending_plan = result.plan
        await presenter.on_plan(result.plan)
        cli._show_info("Plan pending: '/h2l run' executes it, '/h2l status' shows it.")
        return

    if sub == "run":
        pending = getattr(cli, "_h2l_pending_plan", None)
        if pending is None:
            cli._show_error("No pending plan. Use '/h2l plan <goal>' first.")
            return
        run = _build_run(cli, config, presenter)
        if run is None:
            return
        summary = await run.run_plan(pending)
        cli._h2l_pending_plan = None
        _post_summary(cli, summary)
        return

    goal = " ".join(args).strip()
    run = _build_run(cli, config, presenter)
    if run is None:
        return
    cli._show_info(f"Planning with {config.high.label}…")
    summary = await run.run(goal)
    if summary.stop_cause == "plan_failed":
        cli._show_error(summary.error or "H2L: planning failed")
        return
    if summary.stop_cause == "cancelled":
        cli._show_info("H2L run cancelled.")
        return
    _post_summary(cli, summary)


async def _handle_config(cli: Any, args: List[str]) -> None:
    config = cli.engine.config_manager.get_config()
    block = getattr(config, "h2l", None)
    if not args or args[0].lower() == "show":
        if block is None:
            cli._show_info(
                "No h2l block configured. Set both tiers: "
                "/h2l config set high <ref> low <ref[,ref]>"
            )
            return
        table = Table(title="h2l config")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for key, current in block.model_dump().items():
            shown: str
            if key == "high":
                shown = block.high.label
            elif key == "low":
                shown = ", ".join(r.label for r in block.low)
            elif isinstance(current, list):
                shown = ", ".join(str(v) for v in current)
            else:
                shown = str(current)
            table.add_row(key, shown)
        cli.console.print(table)
        return

    if args[0].lower() != "set" or len(args) < 3 or (len(args) - 1) % 2 != 0:
        cli._show_error("Usage: /h2l config set <field> <value> [<field> <value> ...]")
        return

    data: Dict[str, Any] = block.model_dump() if block is not None else {}
    pairs = list(zip(args[1::2], args[2::2]))
    for key, raw in pairs:
        key = key.lower()
        if key not in _SETTABLE:
            cli._show_error(
                f"Unknown h2l field '{key}'. Fields: {', '.join(sorted(_SETTABLE))}"
            )
            return
        value: Any = raw
        if key in _LIST_FIELDS:
            value = [item.strip() for item in raw.split(",") if item.strip()]
        elif key in _OPTIONAL_INT_FIELDS and raw.lower() in ("off", "none"):
            value = None  # unbounded budget / provider's own output cap
        elif key in _INT_FIELDS:
            try:
                value = int(raw)
            except ValueError:
                cli._show_error(f"{key} must be an integer")
                return
        elif key in _BOOL_FIELDS:
            value = raw.lower() in ("1", "true", "yes", "on")
        data[key] = value

    if block is None and not (data.get("high") and data.get("low")):
        cli._show_error(
            "No h2l block yet: set both tiers at once, e.g. "
            "/h2l config set high claude:claude-opus-5 low gateway:qwen3-8b"
        )
        return
    try:
        new_block = H2LConfig(**data)
    except ValueError as exc:
        cli._show_error(f"Invalid h2l config: {exc}")
        return
    config.h2l = new_block
    try:
        cli.engine.config_manager.save_config(config)
    except Exception as exc:
        cli._show_error(f"Could not save config: {exc}")
        return
    cli._show_success("h2l config updated: " + ", ".join(f"{k}={v}" for k, v in pairs))
