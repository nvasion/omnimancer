# PRD: H2L — a high model plans and judges, a pool of low models executes

2026-09-20 · Author: Claude (principal on the omnimancer H2L research) · Owner: Kellan
Repo: `omnimancer` — new package `omnimancer/h2l/`; touches `core/models.py`, `cli/interface.py`, `cli/headless.py`, `cli/command_dispatch.py`, `cli/completion.py`, `cli/display.py`, `cli/headless_checkpoint.py`
Research: `docs/plans/h2l-research.md` · Companion: `docs/plans/PRD-h2l.html`

**Status: Phase 1 CODE + TESTS GREEN 2026-09-20 on branch `feat/h2l` (UNCOMMITTED, not released). 167 new tests (`tests/h2l/`, `tests/cli/test_h2l_headless.py`, `tests/cli/test_h2l_command.py`); full suite 2485 passed; flake8/isort/black clean; mypy clean except pre-existing `core/security/sandbox_manager.py` errors. PRD written 2026-09-20 from the research memo, verified against the codebase at `d17b9f8` the same day.**

**Decisions taken (2026-09-20, PRD defaults adopted on Kellan's "lets start" — flag if wrong):**
Q1 workers do not see `CLAUDE.md` (planner does); Q2 retry goes to the next model in the pool;
Q3 a high-model escalation is judged deterministically only; Q4 worker `Bash` unrestricted;
Q5 headless escalation default `high` (`ask` → `high`); Q6 Ollama alias workaround documented in
the error message, native tools deferred; Q7 built here, not in Factory.

**Phase 1 deviations (2026-09-20):**
- Prompts are **prepended to the first user message**, not sent as a SYSTEM context message
  (§3–§5 said "system"): the Claude provider drops SYSTEM-role context messages when
  building its request, and both existing agent loops prepend for the same reason.
- Planner self-check: when the model does not resubmit after the self-check turn, the
  **first** accepted plan is used (the PRD said the second submission is the plan; a
  second submission still wins when it comes). One nudge is sent before `plan_failed`.
- `/h2l config set <field> <value> …` instead of `/config set h2l.<field>`: the existing
  `/config set` only knows `default_provider` and `providers.*`. With no block yet, both
  tiers must be set in one command.
- Workers' `Edit`/`Write` paths are scope-checked at execution too, not only `Read`
  (§4 named `Read`); out-of-scope writes are refused before the gate.
- `escalation: ask` offers a third answer, `retry` (one more attempt on the next model).
- Already landed from Phase 2 because the seams were cheap: `max_parallel > 1`
  (dynamic DAG scheduler), the serialized + `[H2L S1 · gateway:qwen3-8b]`-prefixed
  approval callback, next-model retry, and `/h2l` completion/help entries. Not landed:
  per-worker activity rows in the live display (§7), §9 checkpoint/resume, TUI
  checkpoints.
- Verified-only-by-unit: the full live acceptance (§Acceptance 9) has not been run yet;
  plan-only has (below).

**Live-run findings, 2026-09-20/21 (plan-only on `digitalocean:qwen3.8-max`), all fixed with tests:**
- *Env-only providers did not resolve.* `apply_env_overrides` returns a deep copy, so an
  entry materialized from `DIGITALOCEAN_INFERENCE_KEY` never reaches the stored config.
  `refs.resolve` now reads the same effective config the engine initializes from, with a
  live-instance fallback.
- *Plans were lost to the 4096 output-token cap.* OpenAI-family and Claude providers
  default `max_tokens` to 4096 and the non-streaming tool handler never surfaced
  `finish_reason`, so a truncated `submit_plan` looked like an empty plan; the model
  kept shrinking the plan until it fit (11 submissions, ~8 of 16 minutes). Fixes:
  **§3 changed — stories are recorded one per call with a new `add_story` tool**
  (validated as they arrive, same id replaces) and `submit_plan` closes the plan with
  goal + design; new `high_max_tokens` (16384) / `low_max_tokens` (8192) floors applied
  in `refs.resolve`; the OpenAI-family provider now reports `finish_reason` and the
  server-reported `model`; the loop tells the model when its output was cut off.
- *False cycle rejections.* The implicit shared-file edge was added blindly; a declared
  reverse order on a shared file closed a cycle. Now added only when the pair is not
  already ordered either way, and **all** validation problems are reported in one
  rejection.
- *Self-check forced a full regeneration.* It now says "reply OK if the plan stands",
  fixes go through `add_story` by id, and the phase is bounded (12 turns).
- *Planner budget.* Unbounded by default (Kellan: "most people don't care about
  iterations"); `--h2l-plan-iterations` / `planner_max_iterations` is opt-in, and failure
  messages name the real cause (rejections, truncation), never iterations.
- *Visibility.* `planner_tool` and `waiting` (30s heartbeat) stream-json events; `init`
  and error blobs name the high tier; `plan` carries `model_used`; `error` precedes
  `arguments` and `submit_plan` arguments are summarized so truncated logs stay useful.
  Planner `Bash` is withheld when approvals are off (it was only ever denied).
**Priority: High (makes local and cheap models useful for real work; moves frontier spend to planning and judging)**
**Affects: omnimancer only. No Factory API, MCP, or UI change; Factory consumes the new stream-json events and result fields when it wants to.**

Source: Kellan 2026-09-20 — "H2L is the process of high model planning to low model
execution. Think of it like a principal engineer is creating a design and a backlog of
stories and then delegating to juniors ... Lower models should only be working, they
don't need repository reads ... low models need to be talked to like they are stupid ...
You should be able to pass a list of low models to do the work ... scoring low models'
work by the high model ... the low models are the subagents and they return their output
where they are judged by the higher model if they meet criteria."

## Problem Statement

One model does everything in an omnimancer session today: design, exploration, edits,
verification. Three costs follow.

1. **Frontier tokens are spent on mechanical edits.** The interactive loop
   (`cli/interface.py::_handle_tool_calling_flow`) and the headless loop
   (`cli/headless.py::HeadlessRunner._run`) send every turn, including "replace this
   string in this file", to the session model.
2. **Small and local models are unusable for agent work.** They fail when asked to
   explore and design. They do well when handed exact instructions, and nothing in
   omnimancer produces exact instructions for them.
3. **Nothing checks the result.** A run ends when the model stops calling tools or says
   `DONE`. No criteria, no score, no second opinion.

The pieces for a two-tier flow already exist; nothing connects them, and one of them is
built the wrong way for concurrency (verified 2026-09-20, HEAD `d17b9f8`):

- **An isolated worker loop exists.** `cli/subagent.py::SubAgentRunner.run` runs a
  `SubAgentDefinition` with its own `ChatContext`, a tool allowlist, a model override,
  a capped loop, and fleet lifecycle events with a parent id. But it **mutates the shared
  provider's `model` attribute** (`subagent.py:66`, restored at `:168`), never appends
  to `context.messages`, runs only from `/subagents run`, and prints its result to a
  panel. Two concurrent runs on different models would clobber each other.
- **The concurrency-safe way to run a second model exists.**
  `core/prompt_enhancer.py:277-283` copies a `ProviderConfig` with
  `model_copy(update={"model": ...})`, calls `ProviderFactory.create_provider`, and talks
  to the new instance with an isolated `ChatContext`. `ProviderInitializer._generate_cache_key`
  (`provider_initializer.py:251`) includes the model, so one provider with two models
  yields two instances. Nothing generalizes this.
- **No plan, story, or task type exists.** `core/agent/workflow_orchestrator.py`
  (851 lines) is instantiated on every engine start and never driven; its `WorkflowStep`
  wraps Python callables, not model instructions.
- **No parallelism exists.** There is no `asyncio.gather` in `omnimancer/`. Approval
  prompts block stdin through `asyncio.to_thread(input)` (`approval_prompt.py:689`).
- **Two loops, no shared turn.** `docs/CODEBASE_MAP.md:165`: "`headless.py` reimplements
  the agent loop (changes usually land in both files)."
- **Only tool-capable providers can execute.** Headless has no marker fallback.
  `ollama`, `cohere`, `perplexity`, and `claude-code` report `supports_tools()` false;
  Ollama is the painful one because local models are the obvious low tier. An
  `openai-compatible` alias pointed at Ollama's OpenAI endpoint works today.
- **The gate is single and shared.** `core/agent_engine.py:190::execute_with_approval`
  is the only path to a write, a command, or a web call on both loops. Workers inherit
  permission rules, hooks, approval, and hard-restricted paths for free.
- **Observability is already per-agent.** `events/emitter.py:137::agent_context` is
  ContextVar-based, so it survives `asyncio.gather`; every tool event carries
  `(agent_id, parent_id)`; the Fleet TUI renders subagent rows as children.

Config precedence gotcha that will bite the flags: **env > CLI flags > stored config**.
`apply_env_overrides` runs inside `CoreEngine.initialize_providers` after
`apply_session_overrides` and assigns unconditionally, so `OMNIMANCER_<PROVIDER>_MODEL`
silently beats `--model` (`docs/CODEBASE_MAP.md` gotcha 1).

## What It Is

A run has three roles and one loop.

1. **Principal (high model).** Reads the repo with the full read tool set, writes a short
   design, and submits a backlog of **stories** through a `submit_plan` tool. A story is
   written for a worker that cannot see the repository: absolute paths, exact
   `old_string`/`new_string` pairs or full file content, exact commands, testable
   acceptance criteria, an optional `verify` command, and `depends_on`.
2. **Workers (low model pool).** One story per worker on its own provider instance with
   its own history. Tools are `Edit`, `Write`, `Bash`, plus `Read` scoped to the story's
   files. No `Glob`, `Grep`, or `WebFetch`. The worker returns a diff and the verify output.
3. **Judge (high model).** Deterministic checks first (verify exit code, out-of-scope
   files, empty diff). Then the high model scores the **diff and verify output** against
   the acceptance criteria through a `submit_verdict` tool. It never sees the worker's
   narration. Below threshold goes back to the next model in the pool with the judge's
   feedback, up to `max_retries`, then an escalation policy applies.

Both surfaces run the same package. TUI: `/h2l <goal>` with models from the `h2l` config
block. Headless: `omn -p "<goal>" --h2l --high <ref> --low <ref> [--low <ref> ...]`.

Relationship to `/subagents`: unchanged and kept. H2L's `WorkerRunner` fixes the subagent
runner's seven gaps (research §4.1); re-basing `SubAgentRunner` on it is a follow-up, not
part of this PRD.

## Goals

- A user runs a multi-file task where the high model plans, one or more low models
  execute, and the high model judges, from either surface, with the same package.
- Workers never explore. Story quality is the lever, and the planner is held to it.
- Every story is scored before it counts. The user sees per story: model, attempts,
  score, diff, and tokens; and totals by tier.
- Headless runs are resumable at story granularity and keep the existing exit-code and
  stream-json contracts.
- No change to security, permission, or approval behavior. No change at all when the
  feature is not enabled.

## Non-goals

Automatic model selection or routing (the user names the tiers). Nested H2L. Per-worker
git worktrees (stories sharing a file serialize instead; worktrees are a documented
follow-up). Replacing `/subagents`. Judging with any model other than the high one. A
marker-protocol fallback for non-tool providers. TUI checkpoints (headless only in this
PRD).

## Design

### 1. Config block and model refs

New optional `Config.h2l: Optional[H2LConfig]` in `core/models.py`, next to `enhancement`.
Absence disables the feature (same precedent). Passthrough in `config_migration.py`'s
v1→v2 key list so it survives migration (the 2026-07-31 regression class).

```python
class H2LModelRef(BaseModel):
    provider: str   # key of Config.providers, registered name or alias
    model: str

class H2LConfig(BaseModel):
    enabled: bool = True
    high: H2LModelRef
    low: List[H2LModelRef]                 # non-empty, ordered pool
    max_parallel: int = 2
    max_retries: int = 2                   # judge send-backs per story
    pass_threshold: int = 80               # 0..100
    worker_tools: List[str] = ["Edit", "Write", "Bash"]
    allow_worker_read: bool = True         # Read limited to story.files
    escalation: Literal["high", "block", "ask"] = "ask"
    plan_approval: bool = True
    worker_max_iterations: int = 15        # planner may raise per story, cap 30
```

Ref grammar, parsed by `h2l/refs.py::parse_model_ref(text) -> H2LModelRef`:
`<entry>:<model>`; a bare `<model>` resolves against `Config.default_provider`. The
two-part form is required when the model id itself contains a colon.

Resolution, `h2l/refs.py::resolve(ref, engine) -> BaseProvider`, runs **after**
`initialize_providers` so the env pass cannot rewrite a tier. It looks up
`engine.config_manager.config.providers[ref.provider]`, does
`model_copy(update={"model": ref.model})`, and calls `ProviderFactory.create_provider`.
It never touches `engine.providers` or `engine.current_provider`. Errors:

- unknown entry → `H2L: provider entry 'x' is not configured (see /providers)`
- non-tool provider → `H2L: provider 'ollama' does not support tool calling; use an
  openai-compatible alias pointed at its OpenAI endpoint (see docs/provider-setup.md)`

Both tiers are validated before any model call. Flags override config field by field and
never persist (same rule as `apply_session_overrides`, regression-tested byte-identical).

### 2. Plan and story model

```python
class Story(BaseModel):
    id: str                       # "S1"..; unique
    title: str
    instructions: str             # executable without exploration
    files: List[str]              # absolute paths the worker may touch
    acceptance: List[str]         # testable criteria
    verify: Optional[str] = None  # shell; exit 0 required when present
    depends_on: List[str] = []
    max_iterations: Optional[int] = None

class Plan(BaseModel):
    goal: str
    design: str
    stories: List[Story]
```

`h2l/orchestrator.py::validate_plan(plan, cwd)`: unique ids; every `depends_on` resolves;
no cycles; every path absolute and under `cwd` (`Path.resolve().is_relative_to`); stories
sharing a file without a declared edge get an **implicit edge in plan order** so two
workers never edit one file concurrently. Failure text names the story and the rule, e.g.
`H2L: story S3 lists /etc/hosts, outside the working directory`.

### 3. Planner

`h2l/planner.py::Planner(high_provider, cwd).plan(goal) -> Plan`. Fresh provider instance
(§1), isolated `ChatContext` seeded with a SYSTEM message, never `chat_manager`. Tools:
`Read`, `Glob`, `Grep`, `Bash` (through the normal gate, so writes are still approval-
gated, and the planner prompt forbids them) plus `submit_plan` whose schema is `Plan`.
`Write`/`Edit` are not in the planner's tool list and are refused at execution if called.

The system prompt (`h2l/prompts.py::PLANNER_SYSTEM`, sha256-pinned in tests like the
PromptFoundry meta-prompts) states: workers cannot see the repository and will not
explore; every story must be executable from its text alone; absolute paths; exact
`old_string`/`new_string` for edits, full content for new files; a `verify` command
whenever a test or lint can prove the story; `depends_on` for shared files; stories small
enough for a few tool calls; submit through `submit_plan` only. Project instructions
(`CLAUDE.md`/`OMNIMANCER.md`) are included through the existing
`load_project_instructions` path with its "not system authority" banner.

Self-check turn: after the first `submit_plan`, the planner receives its own plan back
with "re-read each story for missing exact strings and unresolved references; resubmit"
and must call `submit_plan` again. The second submission is the plan. Cap: 20 iterations;
no `submit_plan` by then → `stop_cause: "plan_failed"`, exit 1, no worker starts.

### 4. Worker

`h2l/worker.py::WorkerRunner(agent_engine).run(story, ref, feedback=None) -> WorkerReport`.

- Provider instance from §1; the parent provider's `model` is never read or written.
- `ChatContext` seeded with `WORKER_SYSTEM` plus the story as SYSTEM; every assistant and
  tool turn appended, native tool history when the provider supports it (same branch as
  both existing loops).
- Tool definitions: `worker_tools` + `Read` when `allow_worker_read`. Enforcement at
  execution: a call to any tool not in the set, or a `Read` of a path not in
  `story.files`, returns `ToolResult(error="H2L: tool X not permitted for this story")`
  and is never dispatched.
- Every dispatched call goes through `AgentEngine.execute_with_approval` unchanged.
- `ToolResult.cancelled` aborts the worker; the story becomes `cancelled`.
- Repeat guard reuses `RepeatedCallTracker`.
- On exit, builds the report: `git diff -- <story.files>` (untracked files via
  `git status --porcelain`), `story.verify` run through the gate as a `Bash` call with
  its stdout, stderr, and exit code captured, and `files_outside_scope` from
  `git status --porcelain` minus `story.files`.

```python
class WorkerReport(BaseModel):
    story_id: str; worker: H2LModelRef; attempt: int
    success: bool; narrative: str
    tool_log: List[ToolLogEntry]          # name, arguments, result[:2000]
    diff: str; verify_output: Optional[str]; verify_exit: Optional[int]
    files_outside_scope: List[str]
    usage: UsageTotals
```

`WORKER_SYSTEM` states: follow the story exactly; do not explore; do not read files you
were not given; if an `Edit` old string does not match, stop and say so; when finished,
run the verify command if given and report in one paragraph.

### 5. Judge

`h2l/judge.py::Judge(high_provider).check(story, report) -> Verdict`.

Deterministic stage, no model call, in this order; the first hit fails the attempt with
`judge: "deterministic"`:

1. `report.verify_exit not in (None, 0)` → `verify failed (exit N)`
2. `report.files_outside_scope` → `modified files outside story scope: ...`
3. `report.diff == ""` and `story.files` non-empty → `no changes made`

Model stage: fresh high instance, isolated context, input = story text, acceptance list,
diff, verify output. **Not** the narrative or tool log. Returns through `submit_verdict`:

```python
class Verdict(BaseModel):
    story_id: str; attempt: int
    score: int                 # 0..100
    passed: bool
    failures: List[str]        # one per failed criterion
    feedback: str              # instructions a worker can act on without context
    judge: Literal["deterministic", "model"]
```

Pass = `passed and score >= pass_threshold`. `JUDGE_SYSTEM` states: score the diff
against the acceptance criteria only; verify output is evidence; do not reward length or
explanation; list each failed criterion; write feedback as instructions.

### 6. Orchestrator

`h2l/orchestrator.py::H2LRun(engine, config, presenter).run(goal) -> RunSummary`.

- DAG from §2. Ready stories run under `asyncio.gather` with
  `asyncio.Semaphore(max_parallel)`. `max_parallel=1` is fully serial.
- Pool assignment round-robin in `low` order. A retry goes to the **next** model in the
  pool (same model when the pool has one entry) with `verdict.feedback` prepended to the
  story instructions as `Feedback from review of attempt N:`.
- After `max_retries` failures, `escalation`: `high` → the high model runs the story with
  the worker loop and the full tool set, judged deterministically only; `block` → story
  `blocked`, dependents `skipped` with reason `dependency_blocked:<id>`; `ask` → the TUI
  prompts (`[h]igh model / [b]lock / [r]etry once more`), headless treats `ask` as `high`.
- Approval: the run wraps `agent_engine.approval`'s callback in an `asyncio.Lock`, and
  the operation preview is prefixed `[H2L S3 · gateway:qwen3-8b]` so prompts arrive one
  at a time and say who is asking. Restored on exit.
- Identity: run id `h2l-<session_id>`. Planner and judge run inside
  `agent_context("h2l-high-<hex>", parent=run_id)`; each worker attempt inside
  `agent_context("h2l-<story_id>-<hex>", parent=run_id)`. `SESSION_START` carries
  `provider`, `model`, `story_id`, `attempt`; `SESSION_END` carries `status` and
  `verdict_score` when present. The Fleet TUI needs no change.
- Rate limit after the engine's retry handler is exhausted → stop, `stop_cause:
  "rate_limited"`, checkpoint kept (§9).
- Presenter protocol so the logic is written once:
  `approve_plan(plan) -> PlanDecision`, `on_story_start`, `on_report`, `on_verdict`,
  `on_escalation`, `on_done(summary)`.

```python
class StoryState(str, Enum): pending, running, passed, blocked, skipped, cancelled
class RunSummary(BaseModel):
    stories: List[StoryOutcome]   # id, title, state, attempts, score, model, tokens
    usage_by_tier: Dict[str, UsageTotals]   # "high", "low"
    passed: int; blocked: int; skipped: int; cancelled: int
```

### 7. TUI surface (`cli/command_dispatch.py`)

- `/h2l <goal>` → plan → design paragraph + story table (id, title, files, depends,
  verify) → prompt `[a]ccept  [d]rop <id>  [e]dit <id>  [c]ancel` → run → summary table.
  `edit` opens the story instructions in `$EDITOR` when set, else a multi-line prompt;
  the edited plan is re-validated (§2).
- `/h2l plan <goal>` stops after the plan and holds it as pending. `/h2l run` executes the
  pending plan. `/h2l status`, `/h2l config`. Config edits through the existing
  `/config set h2l.<field> <value>` path.
- While workers run, one row per active worker (story, model, current tool, elapsed)
  renders above the streaming panel via `StreamingDisplay.activity_provider`. Never a
  second `Live` region (the rule in `ui/streaming_display.py`).
- Verdicts and retries print inline as they happen.
- The parent conversation gets **one** assistant message: the summary. Worker transcripts
  never enter `chat_manager`.
- `completion.py` gets `"h2l": {0: ["plan", "run", "status", "config"]}`; `display.py`
  help gets a line.

### 8. Headless surface (`cli/headless.py`, `cli/interface.py::cli_main`)

New plain options on the single click command (it must stay a command, not a group;
orchestrators parse `omn -p --help`):

| Flag | Type | Overrides |
|---|---|---|
| `--h2l` | flag | enables the mode |
| `--high REF` | str | `h2l.high` |
| `--low REF` | str, repeatable | replaces `h2l.low` when given at least once |
| `--h2l-parallel N` | int | `h2l.max_parallel` |
| `--h2l-retries N` | int | `h2l.max_retries` |
| `--h2l-threshold N` | int | `h2l.pass_threshold` |
| `--h2l-escalation` | `high`\|`block` | `h2l.escalation` |
| `--h2l-plan-only` | flag | stop after the plan |

Any `--h2l-*`, `--high`, or `--low` without `--h2l` is a `click.UsageError`. `--h2l` with
no resolvable high or an empty pool → stderr `H2L: no high model configured (set h2l.high
or pass --high)`, exit 1.

Without `--dangerously-skip-permissions` workers cannot write (deny-by-default,
`docs/CODEBASE_MAP.md` gotcha 9), so the run prints the plan and exits 0 with result
`subtype: "h2l_plan_only"`. `--h2l-plan-only` does the same regardless. That is the
dry-run mode.

stream-json adds one event type with seven subtypes; `tool_use`/`tool_result` events
gain `story_id` and `agent_id` during an H2L run:

```json
{"type":"h2l","subtype":"plan","session_id":"…","plan":{…}}
{"type":"h2l","subtype":"story_start","session_id":"…","story_id":"S1","attempt":1,"provider":"gateway","model":"qwen3-8b","agent_id":"h2l-S1-a1b2c3d4"}
{"type":"h2l","subtype":"story_report","session_id":"…","story_id":"S1","attempt":1,"success":true,"files_changed":["/abs/p.py"],"verify_exit":0}
{"type":"h2l","subtype":"story_verdict","session_id":"…","story_id":"S1","attempt":1,"score":92,"passed":true,"failures":[],"judge":"model"}
{"type":"h2l","subtype":"story_blocked","session_id":"…","story_id":"S3","reason":"retries_exhausted"}
{"type":"h2l","subtype":"story_skipped","session_id":"…","story_id":"S4","reason":"dependency_blocked:S3"}
{"type":"h2l","subtype":"done","session_id":"…","passed":5,"blocked":1,"skipped":1,"cancelled":0}
```

The final `result` blob gains `h2l: {stories: [...], usage_by_tier: {high: {...}, low:
{...}}}` with the `RunSummary` fields; each tier carries `input_tokens`, `output_tokens`,
`cache_read_input_tokens`, `cost_usd`. Exit codes: 0 all passed; 1 error before or during
planning; 3 finished with any blocked, skipped, or cancelled story; 4 rate-limited and
resumable. `--notify-cmd` and the `turn_complete` hook fire once at the end with the
existing payload plus `"h2l": true`. `tests/cli/test_headless_events.py` locks the new
shapes.

### 9. Checkpoint and resume (headless)

Sibling file `<session_id>.h2l.json` in `checkpoint_dir()` (`cli/headless_checkpoint.py`),
same session-id regex guard and atomic replace. Holds the plan, per-story state,
attempts, last verdict, and usage by tier; rewritten after every verdict. `omn --resume
<id>` finds the sibling and continues from the first story not `passed`, `blocked`, or
`skipped`; a story that was `running` restarts from its last completed attempt. No
sibling → the ordinary headless resume path. Clean exit deletes both files; exit 3 and 4
keep them and print the existing `Resume with: omn --resume <id>` hint.

### 10. Package layout

```
omnimancer/h2l/
├── __init__.py
├── models.py        H2LModelRef, H2LConfig, Story, Plan, WorkerReport, Verdict,
│                    StoryState, StoryOutcome, RunSummary, UsageTotals
├── refs.py          parse_model_ref, resolve
├── prompts.py       PLANNER_SYSTEM, WORKER_SYSTEM, JUDGE_SYSTEM (sha256-pinned)
├── planner.py       Planner
├── worker.py        WorkerRunner
├── judge.py         Judge
├── orchestrator.py  H2LRun, validate_plan, Presenter protocol
└── checkpoint.py    H2LCheckpoint
```

No module in `omnimancer/h2l/` imports from `cli/interface.py` or `cli/headless.py`.
The two surfaces supply presenters and call `H2LRun`.

## Acceptance

1. **Unit, models and refs** (`tests/h2l/test_models.py`, `test_refs.py`): ref grammar
   incl. bare and colon-in-model forms; config defaults; `validate_plan` rejects cycles,
   duplicate ids, unresolved `depends_on`, paths outside cwd, and adds the implicit
   shared-file edge in plan order; `resolve` after a simulated env override still yields
   the flag's model; a non-tool provider fails with the exact alias message; the object
   returned by `resolve` is never the one in `engine.providers`.
2. **Unit, planner** (`test_planner.py`): mocked provider returns `submit_plan`; a `Write`
   call is refused; the self-check turn happens and the second submission wins; missing
   `submit_plan` → `plan_failed`; `engine.chat_manager.add_user_message` never called.
3. **Unit, worker** (`test_worker.py`): history accumulates across three turns; a
   hallucinated tool and an out-of-scope `Read` return the permitted-error and never reach
   `execute_with_approval`; `cancelled` aborts; diff scoped to `story.files`; verify
   captured; `files_outside_scope` populated; parent provider's `model` byte-identical
   before and after.
4. **Unit, judge** (`test_judge.py`): each deterministic rule short-circuits with no
   model call; the model prompt excludes `narrative` and `tool_log`; threshold boundary
   (79 fails, 80 passes with `passed=True`); feedback lands in the retry prompt.
5. **Unit, orchestrator** (`test_orchestrator.py`): DAG order with implicit edges; two
   providers on different models run concurrently under `max_parallel=2` without
   clobbering; approval callback serialized and prefixed; round-robin and next-model
   retry; all three escalation policies; skip on blocked dependency; events carry the
   right `parent_id`; checkpoint rewritten after each verdict.
6. **CLI** (`tests/cli/test_h2l_command.py`, `test_headless.py`, `test_headless_events.py`,
   `test_headless_checkpoint.py`): plan table and accept/drop/edit/cancel; summary is
   the only message added to the parent conversation; flags, config fallback, usage
   error without `--h2l`; `h2l_plan_only` exit 0; exit codes 0/1/3/4; the seven `h2l`
   subtypes and the `result.h2l` block; sibling checkpoint round-trip and story-granular
   resume; config file byte-identical after a flag run.
7. **Prompts** (`tests/test_system_prompts.py`): sha256 pins for the three prompts.
8. **Regression:** `pytest tests/` and the four lint gates green; `tests/cli/test_subagent.py`
   untouched and green.
9. **Live:** in this repository, `/h2l Add per-client rate limiting to the webhook
   handler` (or an equivalent three-to-five story task) with a frontier high model and a
   local 7B-class low model behind an `openai-compatible` alias: every story passes or
   is blocked with a named reason; no file outside any story's list is modified; the
   summary shows high-tier output tokens below the same task run single-model. Repeat
   headlessly with `--output-format stream-json --dangerously-skip-permissions`; kill it
   after the second verdict; `omn --resume <id>` finishes with exactly the remaining
   stories started.

## Rollout / risk

**Phase 1 — serial MVP** (`feat:` → 0.3.0). §1–§5, §6 with `max_parallel` forced to 1
and `escalation` limited to `high`/`block`, §7 without the activity rows, §8, §10.
Acceptance 1–4, 6–8, and the serial half of 9. Proves the loop end to end without
touching concurrency.

**Phase 2 — parallel pool.** `max_parallel > 1`, approval lock and prefix, next-model
retry, `escalation: ask`, per-worker activity rows, completion and help entries.
Acceptance 5 and the concurrent assertions in 9.

**Phase 3 — resilience.** §9 checkpoint and resume, usage-by-tier in the TUI summary,
Ollama native tool support if not already landed. The resume half of 9.

Risks:

- **Story quality is the whole game.** Vague stories mean blind workers fail and the
  judge loop burns tokens. Mitigated by the self-check turn, the pinned planner prompt,
  and `--h2l-plan-only` so a user can inspect before spending. Measure first-attempt pass
  rate on a ten-task benchmark in this repo before release; target ≥ 70 percent with a
  7B-class low model.
- **Stale `old_string`.** Story N+1 may cite text story N changed. The implicit edge
  orders them but both were written against the original file. Scoped `Read` plus the
  feedback loop catch it; if the benchmark shows it often, the planner prompt gains
  "write later stories against the post-edit file".
- **Judge bias.** The judge grades diffs, never narration, and deterministic checks run
  first. Hand-assess judge/user disagreement on the benchmark; target < 15 percent.
- **Approval fatigue.** Six workers writing three files each is eighteen prompts. The
  plan gate, existing `/accept` modes, and permission rules cover it; the prefix makes
  each prompt attributable.
- **Behaviour change surface.** None when `h2l` is absent and `--h2l` is not passed.
  Everything new is behind the block or the flag.
- Deploy: omnimancer only. Factory containers pick it up on the next pinned version bump
  (`docker/agent-omnimancer/Dockerfile`), no Factory change required to run it; consuming
  `result.h2l` is optional.

## Open questions for Kellan

| # | Question | Recommended default |
|---|---|---|
| Q1 | Should workers see `CLAUDE.md` project instructions? | No. Workers get the story only; the planner sees them and encodes what matters into stories. |
| Q2 | Retry on the next model or the same model with feedback? | Next model in the pool (cheap diversity); same model when the pool has one entry. |
| Q3 | Should a high-model escalation result also get a model verdict? | No, deterministic only; the high model judging its own work adds cost without signal. |
| Q4 | Worker `Bash`: unrestricted or verify-command only? | Unrestricted in Phase 1 (permission rules already gate it); revisit after the benchmark. |
| Q5 | Default `escalation` in headless: `high` or `block`? | `high`. Factory wants runs to finish; `block` is one flag away. |
| Q6 | Ollama: document the alias workaround or add native tools first? | Document now in `docs/provider-setup.md`; native tools as a Phase 3 item. |
| Q7 | Register this PRD in Factory Nexus (`create_prd`) and run it as a swarm, or build it here? | Build Phase 1 here (it is one package with one owner); consider Factory for Phase 2–3 once the loop is proven. |
