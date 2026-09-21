"""H2L data models: model refs, config block, plan/story, reports, verdicts.

Pure pydantic — no provider or engine imports, so ``core.models`` can embed
:class:`H2LConfig` without a cycle.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class H2LModelRef(BaseModel):
    """A tier participant: a config provider entry plus a model id."""

    provider: str
    model: str

    @field_validator("provider", "model")
    @classmethod
    def _non_empty(cls, v: Any) -> Any:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("model ref parts cannot be empty")
        return v.strip()

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def __str__(self) -> str:
        return self.label


def parse_model_ref(text: str, default_provider: str) -> H2LModelRef:
    """Parse ``<entry>:<model>`` or a bare ``<model>``.

    Only the first colon separates the provider entry from the model id, so
    ids like ``qwen2.5-coder:14b`` survive when written in the two-part form.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("model ref cannot be empty")
    if ":" in raw:
        provider, _, model = raw.partition(":")
        if not provider.strip() or not model.strip():
            raise ValueError(
                f"malformed model ref {raw!r}: expected <provider-entry>:<model>"
            )
        return H2LModelRef(provider=provider, model=model)
    if not default_provider or not default_provider.strip():
        raise ValueError(
            f"model ref {raw!r} has no provider and no default provider is set"
        )
    return H2LModelRef(provider=default_provider, model=raw)


def _coerce_ref(value: Any) -> Any:
    """Config files write refs as strings; accept both forms."""
    if isinstance(value, str):
        # No default provider at this layer: config-file refs must be explicit.
        return parse_model_ref(value, default_provider="")
    return value


Escalation = Literal["high", "block", "ask"]


class H2LConfig(BaseModel):
    """The ``h2l`` config block (PRD §1). Absence on ``Config`` disables H2L."""

    enabled: bool = True
    high: H2LModelRef
    low: List[H2LModelRef] = Field(min_length=1)
    max_parallel: int = Field(default=2, ge=1)
    max_retries: int = Field(default=2, ge=0)
    pass_threshold: int = Field(default=80, ge=0, le=100)
    worker_tools: List[str] = Field(default_factory=lambda: ["Edit", "Write", "Bash"])
    allow_worker_read: bool = True
    escalation: Escalation = "ask"
    plan_approval: bool = True
    worker_max_iterations: int = Field(default=15, ge=1, le=30)
    # Optional exploration budget for the planner (turns before it must
    # submit). None = unbounded, like the interactive agent loop; opt in with
    # --h2l-plan-iterations or this field.
    planner_max_iterations: Optional[int] = Field(default=None, ge=1, le=500)
    # Output-token floors for the tiers. Providers default to 4096, which
    # truncates a plan (and, on thinking models, is mostly spent reasoning).
    # None leaves the provider entry's own max_tokens untouched.
    high_max_tokens: Optional[int] = Field(default=16384, ge=256)
    low_max_tokens: Optional[int] = Field(default=8192, ge=256)

    @field_validator("high", mode="before")
    @classmethod
    def _coerce_high(cls, v: Any) -> Any:
        return _coerce_ref(v)

    @field_validator("low", mode="before")
    @classmethod
    def _coerce_low(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [_coerce_ref(item) for item in v]
        return v


class Story(BaseModel):
    """One unit of delegated work, executable without exploration (PRD §2)."""

    id: str
    title: str
    instructions: str
    files: List[str] = Field(default_factory=list)
    acceptance: List[str] = Field(default_factory=list)
    verify: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    max_iterations: Optional[int] = Field(default=None, ge=1, le=30)

    @field_validator("id", "title", "instructions")
    @classmethod
    def _non_empty(cls, v: Any) -> Any:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("story id, title and instructions are required")
        return v.strip()


class Plan(BaseModel):
    goal: str
    design: str = ""
    stories: List[Story] = Field(default_factory=list)


class PlanValidationError(ValueError):
    """A plan that must not run: names the story and the rule it broke."""


def validate_plan(plan: Plan, cwd: Path) -> Plan:
    """Validate a plan and add implicit shared-file edges (PRD §2).

    Rules: at least one story; unique ids; every ``depends_on`` resolves; no
    cycles; every file path absolute, normalized, and under ``cwd``. Two
    stories that list the same file without a declared edge get an implicit
    edge in plan order so no two workers ever edit one file concurrently.
    Returns a new Plan; the input is not mutated.
    """
    if not plan.stories:
        raise PlanValidationError("H2L: plan has no stories")

    # Every problem is collected and reported together: each rejection costs
    # the planner a full plan regeneration, so first-error-only is expensive.
    problems: List[str] = []
    root = Path(cwd).resolve()
    seen: set[str] = set()
    for story in plan.stories:
        if story.id in seen:
            problems.append(f"duplicate story id {story.id}")
        seen.add(story.id)

    normalized: List[Story] = []
    for story in plan.stories:
        fixed, story_problems = normalize_story(story, root)
        problems.extend(story_problems)
        normalized.append(fixed)

    for story in normalized:
        for dep in story.depends_on:
            if dep not in seen:
                problems.append(f"story {story.id} depends on unknown story {dep}")
            elif dep == story.id:
                problems.append(f"story {story.id} depends on itself")

    if problems:
        raise PlanValidationError("H2L: " + "; ".join(problems))

    # Declared cycles are the planner's to fix; check before adding edges so
    # the message never blames an edge the planner did not write.
    _reject_cycles(normalized)

    # Implicit edges: two stories sharing a file must never run together. An
    # edge is added (later waits for earlier, in plan order) only when the
    # declared dependencies do not already order the pair in either direction
    # — otherwise a legitimate "S1 depends on S2" would become a false cycle.
    graph: Dict[str, List[str]] = {s.id: list(s.depends_on) for s in normalized}
    owners: Dict[str, str] = {}
    for story in normalized:
        for file_path in story.files:
            earlier = owners.get(file_path)
            if (
                earlier is not None
                and earlier != story.id
                and not _reaches(graph, story.id, earlier)
                and not _reaches(graph, earlier, story.id)
            ):
                graph[story.id].append(earlier)
            owners[file_path] = story.id
        story.depends_on = list(graph[story.id])

    return plan.model_copy(update={"stories": normalized})


def normalize_story(story: Story, cwd: Path) -> "tuple[Story, List[str]]":
    """Normalize one story's paths and list its path problems.

    Used both per story (the planner's ``add_story`` gives feedback as each
    story arrives) and for whole-plan validation.
    """
    root = Path(cwd).resolve()
    problems: List[str] = []
    files: List[str] = []
    for raw in story.files:
        path = Path(raw)
        if not path.is_absolute():
            problems.append(
                f"story {story.id} lists {raw!r}, which is not an absolute path"
            )
            continue
        resolved = _normalize(path)
        if not _is_under(resolved, root):
            problems.append(
                f"story {story.id} lists {raw}, outside the working directory "
                f"({root})"
            )
            continue
        text = str(resolved)
        if text not in files:
            files.append(text)
    return story.model_copy(update={"files": files}), problems


def _reaches(graph: Dict[str, List[str]], src: str, dst: str) -> bool:
    """True when ``src`` depends on ``dst``, directly or transitively."""
    stack, visited = list(graph.get(src, [])), set()
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(graph.get(node, []))
    return False


def _normalize(path: Path) -> Path:
    """Collapse ``..`` and ``.`` without requiring the file to exist."""
    parts: List[str] = []
    for part in path.parts:
        if part == "..":
            if len(parts) > 1:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return Path(*parts) if parts else path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_cycles(stories: List[Story]) -> None:
    graph = {s.id: list(s.depends_on) for s in stories}
    state: Dict[str, int] = {}  # 0 unvisited, 1 in progress, 2 done

    def visit(node: str, trail: List[str]) -> None:
        mark = state.get(node, 0)
        if mark == 2:
            return
        if mark == 1:
            loop = " -> ".join(trail[trail.index(node) :] + [node])
            raise PlanValidationError(f"H2L: dependency cycle: {loop}")
        state[node] = 1
        for dep in graph[node]:
            visit(dep, trail + [node])
        state[node] = 2

    for story_id in graph:
        visit(story_id, [])


def topological_order(stories: List[Story]) -> List[Story]:
    """Stories in dependency order, stable with respect to plan order."""
    by_id = {s.id: s for s in stories}
    done: List[str] = []
    seen: set[str] = set()

    def visit(story_id: str) -> None:
        if story_id in seen:
            return
        seen.add(story_id)
        for dep in by_id[story_id].depends_on:
            visit(dep)
        done.append(story_id)

    for story in stories:
        visit(story.id)
    return [by_id[i] for i in done]


class ToolLogEntry(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    error: Optional[str] = None


class UsageTotals(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: "UsageTotals") -> "UsageTotals":
        return UsageTotals(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cost_usd=self.cost_usd + other.cost_usd,
            calls=self.calls + other.calls,
        )

    @classmethod
    def from_response(cls, response: Any) -> "UsageTotals":
        return cls(
            input_tokens=int(getattr(response, "input_tokens", 0) or 0),
            output_tokens=int(getattr(response, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(
                getattr(response, "cache_read_input_tokens", 0) or 0
            ),
            cache_creation_input_tokens=int(
                getattr(response, "cache_creation_input_tokens", 0) or 0
            ),
            cost_usd=float(getattr(response, "cost_estimate", 0.0) or 0.0),
            calls=1,
        )


class WorkerReport(BaseModel):
    """What a worker produced for one attempt (PRD §4)."""

    story_id: str
    worker: H2LModelRef
    attempt: int = 1
    success: bool = True
    narrative: str = ""
    tool_log: List[ToolLogEntry] = Field(default_factory=list)
    diff: str = ""
    verify_output: Optional[str] = None
    verify_exit: Optional[int] = None
    files_outside_scope: List[str] = Field(default_factory=list)
    files_changed: List[str] = Field(default_factory=list)
    usage: UsageTotals = Field(default_factory=UsageTotals)
    iterations: int = 0
    cancelled: bool = False
    error: Optional[str] = None


class Verdict(BaseModel):
    """The judge's decision on one attempt (PRD §5)."""

    story_id: str
    attempt: int = 1
    score: int = Field(default=0, ge=0, le=100)
    passed: bool = False
    failures: List[str] = Field(default_factory=list)
    feedback: str = ""
    judge: Literal["deterministic", "model"] = "deterministic"
    usage: UsageTotals = Field(default_factory=UsageTotals)
    error: Optional[str] = None

    @model_validator(mode="after")
    def _fail_needs_reason(self) -> "Verdict":
        if not self.passed and not self.failures and not self.error:
            self.failures = ["did not meet the acceptance criteria"]
        return self


class StoryState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StoryOutcome(BaseModel):
    id: str
    title: str
    state: StoryState = StoryState.PENDING
    attempts: int = 0
    score: Optional[int] = None
    model: Optional[str] = None
    reason: Optional[str] = None
    usage: UsageTotals = Field(default_factory=UsageTotals)
    verdicts: List[Verdict] = Field(default_factory=list)


class RunSummary(BaseModel):
    goal: str
    stories: List[StoryOutcome] = Field(default_factory=list)
    usage_by_tier: Dict[str, UsageTotals] = Field(
        default_factory=lambda: {"high": UsageTotals(), "low": UsageTotals()}
    )
    # done | partial | plan_failed | cancelled | rate_limited
    stop_cause: str = "done"
    error: Optional[str] = None

    def count(self, state: StoryState) -> int:
        return sum(1 for s in self.stories if s.state == state)

    @property
    def passed(self) -> int:
        return self.count(StoryState.PASSED)

    @property
    def blocked(self) -> int:
        return self.count(StoryState.BLOCKED)

    @property
    def skipped(self) -> int:
        return self.count(StoryState.SKIPPED)

    @property
    def cancelled(self) -> int:
        return self.count(StoryState.CANCELLED)

    @property
    def all_passed(self) -> bool:
        return bool(self.stories) and self.passed == len(self.stories)

    def to_result_dict(self) -> Dict[str, Any]:
        """The ``result.h2l`` block for headless output (PRD §8)."""
        return {
            "goal": self.goal,
            "stop_cause": self.stop_cause,
            "passed": self.passed,
            "blocked": self.blocked,
            "skipped": self.skipped,
            "cancelled": self.cancelled,
            "stories": [
                {
                    "id": s.id,
                    "title": s.title,
                    "state": s.state.value,
                    "attempts": s.attempts,
                    "score": s.score,
                    "model": s.model,
                    "reason": s.reason,
                    "tokens": {
                        "input": s.usage.input_tokens,
                        "output": s.usage.output_tokens,
                    },
                }
                for s in self.stories
            ],
            "usage_by_tier": {
                tier: {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cache_read_input_tokens": u.cache_read_input_tokens,
                    "cost_usd": u.cost_usd,
                }
                for tier, u in self.usage_by_tier.items()
            },
        }
