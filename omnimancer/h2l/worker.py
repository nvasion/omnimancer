"""WorkerRunner: one story, one low model, own provider, own history (PRD §4).

Fixes the seven gaps of ``cli/subagent.py`` for H2L's purposes: the provider
instance is supplied by the caller (never the session's), history
accumulates, the tool allowlist is enforced at execution time, out-of-scope
paths are refused before the gate, ``cancelled`` aborts, and the report
carries the diff, verify output, and out-of-scope files the judge needs.
"""

from __future__ import annotations

import difflib
import logging
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..cli.tool_handler import ToolHandler
from ..core.agent.status_core import EventType
from ..core.agent.tool_definitions import CODING_AGENT_TOOLS
from ..core.models import ChatContext, ToolCall, ToolDefinition, ToolResult
from ..events import emitter as fleet_events
from .loop import IsolatedLoop
from .models import H2LModelRef, Story, WorkerReport
from .prompts import WORKER_SYSTEM, render_story

logger = logging.getLogger(__name__)

# Tools whose first argument names a file the story must list.
_PATH_TOOLS = {"Read", "Write", "Edit", "file_read", "file_write", "file_edit"}
_EXIT_CODE_RE = re.compile(r"Command exited with code (-?\d+)")
_VERIFY_OUTPUT_CHARS = 8000


class WorkerRunner:
    """Runs one story attempt to completion and reports what changed."""

    def __init__(self, agent_engine: Any, cwd: Path) -> None:
        self.agent_engine = agent_engine
        self.cwd = Path(cwd)
        # Overridable for tests; the real handler routes through the gate.
        self._tool_handler_factory: Callable[[Any], Any] = ToolHandler
        # Story id -> {path: content or None}, taken before its first attempt.
        self._baselines: Dict[str, Dict[str, Optional[str]]] = {}

    async def run(
        self,
        story: Story,
        ref: H2LModelRef,
        provider: Any,
        *,
        worker_tools: List[str],
        allow_read: bool,
        max_iterations: int,
        run_id: str,
        feedback: Optional[str] = None,
        attempt: int = 1,
        agent_id: Optional[str] = None,
        on_tool_call: Optional[Any] = None,
        on_wait: Optional[Any] = None,
    ) -> WorkerReport:
        report = WorkerReport(story_id=story.id, worker=ref, attempt=attempt)
        allowed: Set[str] = set(worker_tools)
        if allow_read:
            allowed.add("Read")
        tools = [t for t in CODING_AGENT_TOOLS if t.name in allowed]
        scope = set(story.files)
        handler = self._tool_handler_factory(self.agent_engine)

        async def execute(tc: ToolCall) -> ToolResult:
            if tc.name not in allowed:
                return ToolResult(
                    content="",
                    error=f"H2L: tool {tc.name} not permitted for this story",
                )
            if tc.name in _PATH_TOOLS:
                path = self._path_of(tc)
                if path is None or path not in scope:
                    return ToolResult(
                        content="",
                        error=(
                            f"H2L: {tc.name} on {path or '(no path)'} not permitted: "
                            "not in this story's files"
                        ),
                    )
            result: ToolResult = await handler.execute_tool_call(tc)
            return result

        context = ChatContext(
            messages=[],
            current_model=getattr(provider, "model", ref.model) or ref.model,
            session_id=f"h2l-{story.id}-{attempt}",
        )
        message = self._first_message(story, feedback, attempt)
        loop = IsolatedLoop(
            provider,
            context,
            tools,
            execute,
            max_iterations,
            on_tool_call=on_tool_call,
            on_wait=on_wait,
        )

        agent_id = agent_id or f"h2l-{story.id}-{uuid.uuid4().hex[:8]}"
        if story.id not in self._baselines:
            self._baselines[story.id] = {p: _read_text(p) for p in story.files}
        before = self._dirty_files()
        status = 1
        with fleet_events.agent_context(agent_id, run_id):
            try:
                await fleet_events.emit_event(
                    EventType.SESSION_START,
                    {
                        "provider": ref.provider,
                        "model": ref.model,
                        "story_id": story.id,
                        "attempt": attempt,
                        "h2l": "worker",
                    },
                )
                outcome = await loop.run(message)
                report.narrative = outcome.content
                report.tool_log = outcome.tool_log
                report.usage = outcome.usage
                report.iterations = outcome.iterations
                if outcome.stop == "cancelled":
                    report.cancelled = True
                    report.success = False
                    report.error = "cancelled at approval"
                elif outcome.stop == "error":
                    report.success = False
                    report.error = outcome.error
                elif outcome.stop in ("max_iterations", "repeat_abort"):
                    report.success = False
                    report.error = outcome.error or outcome.stop
                else:
                    report.success = True

                self._capture_changes(report, story, before)
                if story.verify and not report.cancelled:
                    await self._run_verify(report, story, handler)
                status = 0 if report.success else 1
                return report
            except Exception as exc:  # a worker must never take the run down
                logger.warning("H2L worker %s failed: %s", story.id, exc)
                report.success = False
                report.error = str(exc)
                return report
            finally:
                try:
                    await fleet_events.emit_event(
                        EventType.SESSION_END,
                        {"reason": "h2l_worker_complete", "status": status},
                    )
                except Exception as exc:
                    logger.debug("h2l worker session_end failed: %s", exc)

    # ------------------------------------------------------------ prompt

    @staticmethod
    def _first_message(story: Story, feedback: Optional[str], attempt: int) -> str:
        parts = [WORKER_SYSTEM, render_story(story)]
        if feedback:
            parts.append(
                f"Feedback from review of attempt {max(1, attempt - 1)}:\n{feedback}"
            )
        parts.append("Begin now.")
        return "\n\n".join(parts)

    @staticmethod
    def _path_of(tc: ToolCall) -> Optional[str]:
        args = tc.arguments if isinstance(tc.arguments, dict) else {}
        raw = args.get("file_path") or args.get("path")
        if not raw:
            return None
        path = Path(str(raw))
        if not path.is_absolute():
            return str(path)
        return str(path.resolve()) if path.exists() else str(path)

    # ------------------------------------------------------------ verify

    async def _run_verify(
        self, report: WorkerReport, story: Story, handler: Any
    ) -> None:
        call = ToolCall(name="Bash", arguments={"command": story.verify})
        try:
            result: ToolResult = await handler.execute_tool_call(call)
        except Exception as exc:
            report.verify_exit = 1
            report.verify_output = f"verify could not run: {exc}"
            return
        if result.cancelled:
            report.cancelled = True
            report.success = False
            report.verify_exit = None
            return
        if result.error:
            match = _EXIT_CODE_RE.search(result.error)
            report.verify_exit = int(match.group(1)) if match else 1
            report.verify_output = result.error[:_VERIFY_OUTPUT_CHARS]
        else:
            report.verify_exit = 0
            report.verify_output = (result.content or "")[:_VERIFY_OUTPUT_CHARS]

    # ------------------------------------------------------------ git capture

    def _git(self, *args: str) -> Optional[subprocess.CompletedProcess[str]]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=str(self.cwd),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("git unavailable for H2L capture: %s", exc)
            return None

    def _repo_root(self) -> Optional[Path]:
        proc = self._git("rev-parse", "--show-toplevel")
        if proc is None or proc.returncode != 0:
            return None
        return Path(proc.stdout.strip())

    def _dirty_files(self) -> Dict[str, str]:
        """Absolute path -> porcelain status for every changed/untracked file."""
        root = self._repo_root()
        if root is None:
            return {}
        proc = self._git(
            "-C", str(root), "status", "--porcelain", "--untracked-files=all"
        )
        if proc is None or proc.returncode != 0:
            return {}
        dirty: Dict[str, str] = {}
        for line in proc.stdout.splitlines():
            if len(line) < 4:
                continue
            code, rest = line[:2], line[3:]
            if " -> " in rest:  # rename: "R  old -> new"
                rest = rest.split(" -> ", 1)[1]
            rest = rest.strip().strip('"')
            dirty[str(root / rest)] = code
        return dirty

    def _capture_changes(
        self, report: WorkerReport, story: Story, before: Dict[str, str]
    ) -> None:
        if self._repo_root() is None:
            self._capture_from_baseline(report, story)
            return
        after = self._dirty_files()
        scope = set(story.files)
        new_or_changed = {p for p in after if p not in before or p in scope}
        report.files_changed = sorted(p for p in new_or_changed if p in scope)
        report.files_outside_scope = sorted(
            p for p in after if p not in before and p not in scope
        )
        chunks: List[str] = []
        for path in report.files_changed:
            chunks.append(self._diff_for(path, after.get(path, "")))
        report.diff = "\n".join(c for c in chunks if c)

    def _diff_for(self, path: str, code: str) -> str:
        if code.strip() == "??":
            proc = self._git("diff", "--no-index", "--", "/dev/null", path)
        else:
            proc = self._git("diff", "HEAD", "--", path)
        if proc is None:
            return f"[git diff unavailable for {path}]"
        return proc.stdout

    def _capture_from_baseline(self, report: WorkerReport, story: Story) -> None:
        """Without git, diff the story's files against its first-attempt snapshot.

        The snapshot stands in for HEAD, so retries accumulate as they do under
        git. Out-of-scope writes cannot be seen this way; the tool-level scope
        check is the guard there.
        """
        baseline = self._baselines.get(story.id, {})
        changed: List[str] = []
        chunks: List[str] = []
        for path in story.files:
            old, new = baseline.get(path), _read_text(path)
            if old != new:
                changed.append(path)
                chunks.append(_unified_diff(path, old, new))
        report.files_changed = changed
        report.diff = "\n".join(chunks)


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _unified_diff(path: str, old: Optional[str], new: Optional[str]) -> str:
    fromfile = path if old is not None else "/dev/null"
    tofile = path if new is not None else "/dev/null"
    lines = list(
        difflib.unified_diff(
            (old or "").splitlines(),
            (new or "").splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )
    # An empty file created or removed has no hunks; keep the header so the
    # change is still visible.
    return "\n".join(lines or [f"--- {fromfile}", f"+++ {tofile}"])


def worker_tool_names(tools: List[ToolDefinition]) -> List[str]:
    return [t.name for t in tools]
