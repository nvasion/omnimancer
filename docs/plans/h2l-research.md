# Research: High-to-Low (H2L) Planning and Execution

**Status:** Research (pre-PRD). Nothing here is implemented.
**Date:** 2026-09-20
**Companion:** `docs/plans/h2l-research.html` (same content, browsable)

## 1. What H2L is

H2L is a two-tier agent flow. A **high model** acts as the principal engineer: it
reads the repository, produces a design, and breaks the work into a backlog of
**stories**. Each story is written for a junior who cannot see the repo: absolute
paths, exact strings to replace, exact commands to run, and explicit acceptance
criteria. A pool of **low models** executes the stories in isolation, one story per
worker, using only the instructions they were handed. The high model then **judges**
each worker's result against the story's acceptance criteria and either accepts it,
sends it back with feedback, or escalates.

The three properties that make this different from "just use subagents":

1. **Workers are deliberately context-poor.** They do not explore the repo. The
   planner does all the reading, and the quality of the run is bounded by how
   explicit the stories are.
2. **Workers are a pool, not a single model.** The user supplies a list of low
   models; the scheduler assigns stories across them and can run them in parallel.
3. **There is a judge loop.** The high model scores every result before it counts
   as done. Failed stories are retried with the judge's feedback attached.

Both the TUI (`omn`) and headless (`omn -p`) must support it. Models are chosen
from config in the TUI and from flags in headless.

## 2. Prior art

| System | Shape | What H2L borrows | Where H2L differs |
|---|---|---|---|
| Aider architect/editor mode | One reasoning model proposes, one editor model turns the proposal into file edits, per turn | The core insight that "propose" and "edit" are separable and that weak editors do well with explicit instructions | Aider is one architect, one editor, one turn. No backlog, no pool, no judge |
| Claude Code orchestrator/worker subagents | A lead spawns subagents (optionally on cheaper models) that each run a full tool loop with repo access | Per-agent model override, isolated context, parallel fan-out, nested identity | Claude Code workers are context-rich explorers. H2L workers are deliberately blind |
| LLM-as-judge literature for code | A strong model grades patches, often after cheaper deterministic screening | Deterministic checks first, then the judge; grade the diff, not the narration; versioned rubrics | H2L's judge is the same model that wrote the story, so it grades against criteria it authored |

The judge literature carries a warning worth designing around: LLM judges are
biased toward verbose, confident output and toward outputs that resemble their own
style. The mitigation is to show the judge the **actual diff and test output**, not
the worker's narration, and to run cheap deterministic checks before the judge sees
anything.

## 3. Fit assessment

**Verdict: H2L fits well.** Roughly four fifths of the substrate already exists.
The remaining fifth is a real feature, not a wiring job, but nothing in the codebase
fights the design.

| H2L needs | What exists today | Gap |
|---|---|---|
| Isolated worker loop with its own history and a scoped tool set | `SubAgentRunner` in `cli/subagent.py` | Shares the parent's provider object, never accumulates its own history, serial only, not reachable from headless |
| A second provider instance on a different model, safely | `prompt_enhancer.enhance()` builds one via `ProviderFactory.create_provider` with a `model_copy`; the instance cache keys on model | Nothing generalizes this. `switch_model` and the subagent runner still mutate the shared instance |
| Parallel workers | Nothing. There is no `asyncio.gather` anywhere in the package | Scheduler, concurrency limit, and approval serialization are all new |
| Structured plan and verdict objects | Nothing. No task, story, plan, or backlog type exists. `WorkflowStep` in the dead workflow orchestrator is callable-based and unsuitable | New models |
| A role-based model config | `EnhancementConfig` names a provider and model for one job. `SubAgentDefinition.model` overrides per agent | An `h2l` config block following the enhancement precedent |
| Headless flags for models | `--provider`, `--model`, `--base-url` via `apply_session_overrides` | `--high` and repeatable `--low` flags, and an `--h2l` mode switch |
| A shared "run one turn" loop for TUI and headless | None. `interface.py` and `headless.py` each reimplement the loop | H2L's worker loop should be the first shared one |
| Per-worker observability | ContextVar-based `agent_context(agent_id, parent_id)` in `events/emitter.py`; `TurnActivityLog` groups by agent id; Fleet TUI renders child rows | A grouped per-worker view in the REPL is a display-layer change only |
| Resumable long runs | `HeadlessCheckpoint` snapshots the conversation each iteration | Needs a plan-level checkpoint so `--resume` restarts at story granularity |
| Enforcement of security and approvals | `AgentEngine.execute_with_approval` is the single gate for every operation on both paths | Unchanged. Workers inherit it for free |

## 4. Codebase findings

### 4.1 The subagent runner is the right seed, with seven known gaps

`cli/subagent.py` (176 lines) runs a `SubAgentDefinition` to completion: isolated
`ChatContext`, tool allowlist, optional model override, capped iterations, and fleet
lifecycle events with a unique run id and a parent id. Tests live in
`tests/cli/test_subagent.py`. It is the closest thing to an H2L worker.

Gaps, in order of how much they matter to H2L:

1. **Shared provider mutation.** The runner sets `provider.model` on the parent's
   live provider and restores it in `finally`. Two concurrent runs with different
   models would clobber each other. This blocks parallelism outright.
2. **No history accumulation.** `context.messages` stays empty for the whole run.
   Continuity is carried by rewriting the next message as "Tool results: ...". A
   worker that edits, runs a test, and edits again needs real history.
3. **Not model-invocable and not headless-reachable.** The only entry point is the
   `/subagents run` slash command. There is no Task tool and no marker.
4. **Results never re-enter a conversation.** The result is printed as a Rich panel.
   `SubAgentResult` carries only the last text and a list of tool names, no diff and
   no tool arguments. A judge needs more than that.
5. **Tool allowlist is advisory.** It filters the definitions sent to the model but
   `execute_tool_call` never re-checks. A worker that hallucinates a tool name still
   runs it.
6. **`ToolResult.cancelled` is ignored** inside the loop, so a user pressing quit at
   an approval prompt does not abort the worker.
7. **The definition prompt is folded into the first user message**, not sent as a
   system message.

Recommendation: do not extend `SubAgentRunner` in place. Build a new `WorkerRunner`
for H2L that fixes all seven, then consider re-basing `SubAgentRunner` on it later.

### 4.2 Provider and model binding: the safe pattern already exists

The model is a plain attribute on every provider instance, set at construction
(`BaseProvider.__init__`). `CoreEngine.providers` holds one live instance per config
entry, and `current_provider` is a pointer into that dict.

Two ways to run a different model exist today:

- **Mutate the shared instance.** `switch_model` and `SubAgentRunner` do this. It is
  fine for a single conversation and unsafe for concurrency.
- **Build a second instance.** `prompt_enhancer.enhance()` copies the provider's
  config with `model_copy(update={"model": ...})`, calls
  `ProviderFactory.create_provider`, and talks to the new instance with its own
  `ChatContext`. `ProviderInitializer._generate_cache_key` includes the model, so
  the same provider with two models yields two distinct cached instances.

H2L must use the second pattern for every worker and for the planner. This is the
single most important implementation constraint.

Alias entries matter here. A config entry named `gateway` with
`provider_type: "openai-compatible"` is a first-class provider name. That is how a
local vLLM or Ollama endpoint becomes a low model. Model tokens in H2L config and
flags should therefore be `<provider-entry>:<model>`, resolving the entry through
`Config.providers` and the class through `provider_type`.

### 4.3 Only tool-capable providers can be workers

Workers execute through native tool calls. Providers whose `supports_tools()` is
false cannot be workers today: `ollama`, `cohere`, `perplexity`, and `claude-code`.
The Ollama gap is the painful one, because local small models are the obvious low
tier. The practical workaround is already supported: point an `openai-compatible`
alias at Ollama's OpenAI-compatible endpoint, which does accept tools. The PRD should
either document that workaround or add native tool support to the Ollama provider
as a prerequisite story.

The planner runs on the high model and also needs tools, since it reads the repo to
design. Every frontier provider in the registry qualifies.

### 4.4 Two agent loops, no shared turn function

The interactive loop (`_handle_tool_calling_flow` in `cli/interface.py`) and the
headless loop (`HeadlessRunner._run` in `cli/headless.py`) are separate
implementations of the same idea. They differ in iteration cap, checkpointing,
history elision, streaming, and approval wiring. The shared pieces are `ToolHandler`,
`RepeatedCallTracker`, `build_agent_prompt`, `TokenAccumulator`, the fleet events,
and the `execute_with_approval` gate.

Consequence for H2L: the worker loop must not be written a third time inside each
surface. It should be one module that both the `/h2l` command and the headless runner
call. The H2L worker loop is small and bounded, so it is a good first shared loop.

A second consequence: headless has **no marker fallback**. A non-tool provider under
`omn -p` emits markers as dead text. H2L should reject non-tool providers for both
tiers at config validation time rather than discover it at runtime.

### 4.5 Config surface

There is no global `default_model`. The model lives on each provider entry
(`ProviderConfig.model`), and `Config.default_provider` picks the active entry. Three
role-like mechanisms exist and set the precedent for an `h2l` block:

- `EnhancementConfig` (`provider`, `model`, `temperature`, `enabled`) names a small
  model for one job and is opt-in by the block's absence.
- `SubAgentDefinition.model` overrides per agent.
- `FallbackConfig.fallback_order` is a list of provider names.

Config precedence is **env > CLI flags > stored config**, and it is a known gotcha:
`apply_env_overrides` runs inside `initialize_providers`, after session overrides,
and assigns unconditionally. `OMNIMANCER_<PROVIDER>_MODEL` silently beats `--model`.
H2L flags must resolve their model tokens *after* provider initialization, or the env
override will rewrite the high or low model out from under the run.

The CLI is a single click command (`cli_main` in `cli/interface.py`), deliberately
not a group because external orchestrators parse `omn -p --help`. New H2L flags must
be plain options on that command. Session flags never persist to disk, and a
regression test enforces the config file stays byte-identical.

### 4.6 The approval and security gate

Every operation from either path funnels through `AgentEngine.execute_with_approval`:
preview, permission rules (deny > ask > allow), the blocking `tool_use_request` hook,
the approval callback, execution, then the observe-only `post_tool` hook. Inside
execution, `SecurityManager` and `PermissionController` re-validate, and
hard-restricted paths have no bypass at all.

Three facts shape H2L's approval design:

1. **Approval prompts block stdin.** `approval_prompt.py` uses
   `asyncio.to_thread(input)`. Parallel workers that each hit a write prompt would
   interleave on the terminal. H2L needs an approval serializer, a lock around the
   callback so prompts arrive one at a time and each names the worker and story.
2. **Headless is deny-by-default.** Without `--dangerously-skip-permissions` no
   worker can write or run anything. H2L headless inherits that. Factory already
   passes the flag.
3. **Approval is per operation, not per story.** A plan-level approval gate in the
   TUI ("here are 6 stories, run them?") is a new concept. It fits the existing
   philosophy and is the natural place for the user to edit or drop stories.

### 4.7 Observability is nearly free

`events/emitter.py` keeps the current agent id and parent id in ContextVars, so
`agent_context(run_id, parent_id)` propagates correctly across `asyncio.gather`.
Every tool event already carries `(agent_id, parent_id)`. The Fleet TUI renders
subagent rows as indented children, and `TurnActivityLog` groups rows by agent id.
`StreamingDisplay` exposes an `activity_provider` callback as the one sanctioned place
to inject a renderable above the live panel.

A per-worker status view in the REPL is therefore a display change, not new
instrumentation. Headless gets it through the existing JSONL event feed plus new
stream-json event types.

### 4.8 Code to avoid

- `core/agent/workflow_orchestrator.py` is instantiated on every engine start and
  never driven. Its `WorkflowStep` wraps Python callables, not model instructions.
  Do not reuse it for stories.
- `AgentModeManager` has a `max_concurrent_operations` queue that the live tool
  loop does not use, and its `_execute_operation` is a stub.
- `core/agent/config.py` (`AgentConfig`) is a parallel config system with zero
  instantiations.

## 5. Proposed architecture

A new package, `omnimancer/h2l/`, with five modules. Both surfaces call into it.

### 5.1 Data model (`h2l/models.py`)

```python
class H2LModelRef(BaseModel):
    provider: str            # config entry name, e.g. "claude", "gateway"
    model: str               # model id on that entry

class H2LConfig(BaseModel):
    enabled: bool = True
    high: H2LModelRef
    low: List[H2LModelRef]   # the worker pool, ordered
    max_parallel: int = 2
    max_retries: int = 2     # judge send-backs per story before escalation
    pass_threshold: int = 80 # judge score 0..100
    worker_tools: List[str] = ["Edit", "Write", "Bash"]
    allow_worker_read: bool = True   # Read limited to story.files

class Story(BaseModel):
    id: str
    title: str
    instructions: str        # imperative, explicit, no exploration required
    files: List[str]         # absolute paths the worker may touch
    acceptance: List[str]    # criteria the judge scores against
    verify: Optional[str]    # shell command; exit 0 required
    depends_on: List[str] = []

class Plan(BaseModel):
    goal: str
    design: str              # the principal's short design note
    stories: List[Story]

class WorkerReport(BaseModel):
    story_id: str
    worker: H2LModelRef
    success: bool
    narrative: str           # worker's final text (shown, not graded)
    tool_log: List[dict]     # name + arguments + truncated result
    diff: str                # git diff limited to story.files
    verify_output: Optional[str]
    verify_exit: Optional[int]
    files_outside_scope: List[str]

class Verdict(BaseModel):
    story_id: str
    score: int
    passed: bool
    failures: List[str]
    feedback: str            # sent verbatim to the worker on retry
```

### 5.2 Planner (`h2l/planner.py`)

Runs the high model on a fresh provider instance with the full coding tool set, so
it can read the repo. Its system prompt says, in effect: you are writing for a
worker that cannot see the repository and will not explore; every story must contain
absolute paths, exact text for `Edit` (old string and new string) or full file
content for `Write`, exact commands, and testable acceptance criteria; stories
touching the same file must declare a dependency.

The plan is returned through a **`submit_plan` tool call** whose schema is the
`Plan` model, not through free-form JSON. Every tool-capable provider enforces tool
schemas, which makes this reliable across vendors and avoids JSON repair code.

### 5.3 Worker (`h2l/worker.py`)

`WorkerRunner.run(story, model_ref) -> WorkerReport`. This is the reworked subagent
loop:

- Fresh provider instance per worker via `create_provider` on a `model_copy`.
- Real `ChatContext` history that accumulates every turn.
- System prompt that reads like instructions to someone who follows steps literally:
  do exactly what the story says, do not explore, if an `Edit` old string does not
  match then stop and report it.
- Tool allowlist enforced at execution time, not just at definition time.
- `Read` allowed only for paths in `story.files`, so a worker can recover from a
  stale old string without wandering.
- Captures tool arguments and results, computes a `git diff` scoped to
  `story.files`, runs `story.verify`, and lists any file touched outside scope.
- Honors `ToolResult.cancelled`.
- Runs inside `agent_context(f"h2l-{story.id}-{hex}", parent=run_id)`.

### 5.4 Judge (`h2l/judge.py`)

Two stages. Deterministic checks first and they are free:

- verify command exit code non-zero: fail
- any file outside `story.files` modified: fail
- empty diff on a story that requires edits: fail

Only then does the high model see the story, the acceptance criteria, the diff, and
the verify output. It does **not** see the worker's narrative. It returns a `Verdict`
through a `submit_verdict` tool call. A score below `pass_threshold` sends the story
back with `feedback` prepended to the retry, up to `max_retries`, after which the
orchestrator escalates: either the high model does the story itself, or the run
marks it blocked, configurable.

### 5.5 Orchestrator (`h2l/orchestrator.py`)

- Builds a DAG from `depends_on` and adds implicit edges between stories that share
  a file, so concurrent edits to one file never happen.
- Runs ready stories with `asyncio.gather` under a semaphore of `max_parallel`.
- Assigns low models round-robin from the pool; a story retried after a failed
  verdict goes to the next model in the pool, which is a cheap form of diversity.
- Wraps the approval callback in an `asyncio.Lock` and stamps each prompt with the
  worker and story.
- Emits `h2l_plan`, `h2l_story_start`, `h2l_story_report`, `h2l_story_verdict`, and
  `h2l_done` events on the fleet bus and, in headless, as stream-json lines.
- Checkpoints the `Plan` plus per-story state under the headless checkpoint dir, so
  `omn --resume` restarts at the first unfinished story instead of iteration zero.

### 5.6 TUI surface

- `/h2l <goal>` runs plan, shows the stories in a table, asks for approval (edit,
  drop, or accept), then runs the pool with a per-worker activity panel through
  `StreamingDisplay.activity_provider`.
- `/h2l plan <goal>` stops after the plan. `/h2l run` executes the last plan.
- `/h2l status` and `/h2l config` for inspection. Config edits go through the
  existing `/config set` path.
- The `h2l` block in `~/.omnimancer/config.json` follows the `enhancement` precedent
  and is opt-in by presence.

### 5.7 Headless surface

```
omn -p "Add rate limiting to the webhook handler" \
  --h2l \
  --high claude:claude-opus-5 \
  --low gateway:qwen3-8b --low ollama-oai:qwen2.5-coder-14b \
  --h2l-parallel 2 \
  --output-format stream-json \
  --dangerously-skip-permissions
```

Flags override the config block. Missing flags fall back to config. If neither
supplies a high model the run exits 1 with a clear error. Exit codes reuse the
existing contract: 0 done, 1 error, 3 partial (some stories blocked), 4 rate-limited
and resumable. The final JSON result gains a `stories` array with per-story verdicts
and the aggregate usage split by tier, which is the number the user will care about.

## 6. Design decisions and rationale

1. **Never mutate a shared provider.** Every H2L participant gets its own instance
   from the factory. This is what makes parallelism safe and matches the enhancer.
2. **Structured outputs through tool schemas.** `submit_plan` and `submit_verdict`
   are tools. This works on every tool-capable provider without JSON repair.
3. **Workers get the smallest tool set that can do the job.** `Edit`, `Write`,
   `Bash` by default, `Read` scoped to the story's files. `Glob`, `Grep`, and
   `WebFetch` are off. If a worker needs to search, the story was under-specified
   and that is the planner's bug.
4. **Deterministic checks before the judge, and the judge grades the diff.** This is
   the direct answer to judge bias and keeps judge tokens low.
5. **File-conflict-aware scheduling now, worktrees later.** Implicit edges between
   stories sharing a file is cheap and removes the write race. Per-worker git
   worktrees are the v2 answer if stories want to run truly independently.
6. **One worker loop, two surfaces.** Build the loop once in `omnimancer/h2l/` and
   call it from both `command_dispatch.py` and `headless.py`. Do not copy it.
7. **Reject non-tool providers at validation time.** Both tiers require
   `supports_tools()`. Fail early with a message that names the alias workaround.
8. **Plan approval is a first-class gate in the TUI.** Headless skips it when
   `--dangerously-skip-permissions` is set, otherwise the run stops after the plan
   and prints it, which is a useful dry-run mode on its own.

## 7. Risks and open questions for the PRD

- **Story quality is the whole game.** If the planner writes vague stories, blind
  workers fail and the judge loop burns tokens. The planner prompt needs the same
  care as the PromptFoundry meta-prompts, and probably a self-check pass where the
  high model reviews its own stories for missing exact strings before submitting.
- **Stale `Edit` old strings.** Story N+1 may reference text that story N changed.
  The DAG edge on shared files handles ordering, but the planner still wrote both
  stories against the original file. Mitigation: workers may `Read` in-scope files,
  and the judge feedback loop catches the rest. Worth measuring.
- **Escalation policy.** When a story fails `max_retries` times, should the high
  model do it itself (expensive, reliable) or should the run stop (cheap, needs a
  human)? Recommend configurable with "escalate to high" as the default in headless
  and "ask" in the TUI.
- **Cost accounting.** `TokenAccumulator` totals per session. H2L should report per
  tier and per story, since the point of the feature is to move spend from the high
  tier to the low tier and the user needs to see that it worked.
- **Ollama native tools.** Either the alias workaround is documented or the provider
  gains tool support. The second is a small, well-bounded prerequisite.
- **Headless checkpoint schema.** `HeadlessCheckpoint` is versioned at 1. Adding
  plan state either bumps it or lives in a sibling file keyed by the same session id.
  Sibling file is less invasive.
- **Approval fatigue.** Six workers each writing three files is eighteen prompts. The
  plan approval gate plus the existing `/accept` modes and permission rules should be
  enough, but the PRD should say what the default is.

## 8. Suggested phasing

**Phase 1, serial MVP.** Models, planner, worker, judge, orchestrator with
`max_parallel=1`. `/h2l` in the TUI and `--h2l` plus `--high`/`--low` in headless.
Plan approval gate. Deterministic checks and the LLM judge. Stream-json events.
This proves the loop end to end without touching concurrency.

**Phase 2, parallel pool.** Per-worker provider instances are already in place from
phase 1, so this adds the semaphore, the file-conflict DAG, the approval lock, the
round-robin assignment, and the per-worker activity panel.

**Phase 3, resilience.** Plan-level checkpoint and `--resume` at story granularity,
per-tier cost reporting, escalation policy, optional worktrees.

## 9. Test plan hooks

Per the TDD rule in `CLAUDE.md`, each module lands with tests first:

- `tests/h2l/test_models.py`: validation, `<provider>:<model>` token parsing, DAG
  construction including implicit shared-file edges, cycle rejection.
- `tests/h2l/test_planner.py`: a mocked provider returning a `submit_plan` tool
  call; assert the planner never touches `chat_manager`; assert stories with shared
  files get dependencies.
- `tests/h2l/test_worker.py`: history accumulates; tool allowlist enforced at
  execution; `Read` outside `story.files` is refused; `cancelled` aborts; diff and
  verify captured; the parent provider's `.model` is never mutated.
- `tests/h2l/test_judge.py`: deterministic fails short-circuit the model call; the
  judge prompt excludes the narrative; threshold and retry feedback.
- `tests/h2l/test_orchestrator.py`: two providers with different models run
  concurrently without clobbering; approval lock serializes prompts; round-robin
  and retry reassignment; events carry the right parent id.
- `tests/cli/test_h2l_command.py` and additions to `tests/cli/test_headless.py`:
  flag parsing, config fallback, env-override ordering, exit codes, stream-json
  schema (which `tests/cli/test_headless_events.py` locks).
- Provider validation: a non-tool provider in either tier fails config validation
  with the alias hint.

## 10. Sources

- Aider chat modes, architect and editor models: https://aider.chat/docs/usage/modes.html
- Aider architect/editor pattern write-up: https://generaitelabs.com/aider-implements-new-architect-editor-approach-for-ai-assisted-coding/
- Multi-model orchestration issue in pydantic-ai-harness: https://github.com/pydantic/pydantic-ai-harness/issues/92
- Smart orchestrator with cheaper subagent models in Claude Code: https://www.mindstudio.ai/blog/smart-orchestrator-cheaper-sub-agent-models-claude-code
- Claude Code subagents and orchestration guide: https://hidekazu-konishi.com/entry/claude_code_subagents_and_orchestration_guide.html
- Sub-agent orchestration patterns: https://readysolutions.ai/blog/2026-04-18-sub-agent-orchestration-patterns-claude/
- Bias in the Loop, auditing LLM-as-a-judge for software engineering: https://arxiv.org/html/2604.16790
- LLM-as-a-Judge for software engineering, literature review: https://arxiv.org/pdf/2510.24367
- Improving code generation via small language model as judge: https://arxiv.org/html/2602.11911
- MCTS-Judge, test-time scaling for code correctness judging: https://arxiv.org/pdf/2502.12468
- Internal: `docs/CODEBASE_MAP.md`, `cli/subagent.py`, `core/prompt_enhancer.py`,
  `cli/headless.py`, `core/agent_engine.py`, `events/emitter.py`
