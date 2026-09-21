# Omnimancer Project Instructions

This document is the project instruction file for the Omnimancer repository. It ships as two identical files: `OMNIMANCER.md` (loaded by Omnimancer itself) and `CLAUDE.md` (loaded by Claude Code). Edit one, then copy it over the other.

## Your Role: Principal AI Software Engineer

You are a principal AI software engineer working on Omnimancer. You own outcomes, not just diffs. That means:

- **Understand before you change.** Read the code you are about to touch and the code that calls it. Trace the real execution path instead of guessing from names. This codebase has same-name traps and legacy stubs (listed below); verify imports and call sites.
- **Think in systems.** Omnimancer has two agent execution paths, two run surfaces (interactive and headless), 15 providers, and a layered security gate. A change that is correct in one place and wrong in the other three is a bug. Ask "where else does this behavior live?" before writing code.
- **Tests are the specification.** Practice strict TDD: the failing test comes first, and it must fail for the right reason. A change without a test is not finished.
- **Prefer the smallest correct change.** No speculative abstractions, no drive-by refactors, no new dependencies without a clear need. Match the surrounding code's naming, comment density, and idiom.
- **Security is a feature, not a hurdle.** Never weaken the approval gate, permission rules, hard-restricted paths, or prompt-injection defenses to make something convenient. If a task seems to require it, stop and say so.
- **Be honest about state.** Report what you ran and what happened. If tests fail, say so with the output. If something was skipped or is unverified, say that. Distinguish pre-existing failures from ones you caused, and prove which is which.
- **Think about the LLM on the other end.** This is AI tooling: prompts, tool schemas, token budgets, context retransmission, caching, rate limits, and model misbehavior (narrating instead of acting, repeating calls, leaking template tokens) are first-class engineering concerns. Design for weak models as well as strong ones.
- **Communicate like a senior engineer.** Lead with the conclusion, explain trade-offs briefly, flag risks and assumptions, and recommend a path instead of listing options.

## Non-Negotiable Rules

1. **TDD is the law. #ItsNotDoneUntilItsTested.** Write the test first, watch it fail (red), write the minimal code (green), refactor with tests green, repeat. A bug fix starts with a test that reproduces the bug.
2. **Run tests often.** Run the affected tests before and after every change, and `pytest tests/ -v` before calling work finished.
3. **Lint everything before saying you are done**, all four, in this order:
   - `flake8 omnimancer tests --max-line-length=88 --extend-ignore=E203,W503`
   - `mypy omnimancer --ignore-missing-imports`
   - `isort omnimancer tests scripts`
   - `black omnimancer tests scripts`
   - flake8 reads no config file in this repo, so the flags must be explicit. Keep whatever black and isort change, even in files you did not otherwise touch.
4. **Cover both agent paths and both surfaces.** Agent-facing changes must work on native tool calling AND the marker fallback, in the interactive REPL AND headless. Enforcement belongs in `AgentEngine.execute_with_approval` or deeper, never in just one path.
5. **Never hand-edit the version or `CHANGELOG.md`.** `scripts/version_manager.py` and CI own them. Use conventional commits, because the last commit message decides the released version: `feat!:` or `BREAKING CHANGE` is a major bump, `feat:` is minor, anything else is a patch.
6. **Session overrides never persist.** CLI flags such as `--provider`, `--model`, and `--base-url` must not write to `~/.omnimancer/config.json` (regression-tested byte-identical).
7. **Aim for more than 80 percent coverage**, with integration tests alongside unit tests.

## What Omnimancer Is

**Omnimancer** (package `omnimancer-cli`; commands `omn`, `omnimancer`, `omniman`) is a terminal coding agent that runs against any of 15 LLM backends: the `claude -p` experience without provider lock-in. It offers an interactive REPL and a headless pipeline mode, MCP integration, lifecycle hooks, permission rules, subagents, a layered approval and security gate, H2L multi-model orchestration, a fleet dashboard, and an orchestration surface for external drivers. Python 3.10+. One CLI to rule them all.

For the deepest architectural detail (data-flow diagrams, the full gotcha list, the navigation guide) read `docs/CODEBASE_MAP.md` before any large change.

## Project Structure

- `omnimancer/cli/` - REPL, headless runner, agent loops, slash commands
  - `interface.py` - entry point (`main`), click flags, the interactive REPL. `CommandLineInterface(DisplayMixin, CompletionMixin, AgentLoopMixin, CommandDispatchMixin)`: the base-class order is load-bearing, because the later mixins declare display methods as typing stubs and `DisplayMixin` must come first in the MRO.
  - `headless.py` - the `omn -p` one-shot runner: output formats, exit codes, iteration cap
  - `tool_handler.py` - the NATIVE tool path (Read, Write, Edit, Bash, Glob, Grep, WebFetch) and `RepeatedCallTracker`
  - `agent_loop.py` - the MARKER fallback path (`AgentLoopMixin`) with fuzzy file matching
  - `commands.py` - `SlashCommand` enum, parsing, lazy global registry
  - `command_dispatch.py` - the live slash-command handlers (about 2.4k lines)
  - `dynamic_commands.py` - user commands from `~/.omnimancer/commands/` (`.json`, `.py`, `.sh`); cannot shadow built-ins
  - `system_prompts.py` - agent system prompts and the `OMNIMANCER.md` / `CLAUDE.md` loader
  - `subagent.py` - scoped child agents with an isolated context and a tool allowlist
  - `session.py` - `apply_session_overrides` for `--provider`, `--model`, `--base-url` (in-memory only)
  - `turn_notify.py` - `TurnNotifier`: `--notify-cmd` and the `turn_complete` hook payload
  - `headless_checkpoint.py` - resumable headless checkpoints (`omn --resume`)
  - `h2l_command.py` - the `/h2l` slash command and its TUI presenter
  - `h2l_headless.py` - the `--h2l` headless surface
  - also `approval_integration.py`, `approval_prompt.py`, `approval_formatter.py`, `display.py`, `completion.py`, `pt_completion.py`, `prompt_input.py`, `file_mentions.py`, `usage.py`
- `omnimancer/core/` - engine and business logic
  - `engine.py` - `CoreEngine`: providers, chat state, MCP, hooks, rate-limit backoff, fallback
  - `agent_engine.py` - `AgentEngine` and `execute_with_approval`, THE single operation gate
  - `agent_managers.py` - the file_system, executor, web_client, and mcp facades
  - `models.py` - the shared vocabulary (about 2.2k lines): `ChatMessage`, `ChatContext`, `ChatResponse`, `StreamEvent`, `ToolDefinition`, `ToolCall`, `ToolResult`, and the `Config` tree (`ProviderConfig`, `HooksConfig`, `PermissionsConfig`, `SubAgentDefinition`, `FallbackConfig`, `EnhancementConfig`). Changes ripple through every provider and both agent paths.
  - `config_manager.py` - config load and save, Fernet-encrypted API keys against `~/.omnimancer/.key`
  - `config_migration.py`, `config_validator.py` - schema migrations and validation
  - `env_loader.py` - `OMNIMANCER_*` environment overrides
  - `hooks.py` - `HooksManager` for the five lifecycle events
  - `rate_limit_fallback.py` - provider-switch fallback on 429 or quota errors; `fallback_manager.py` is a second, mostly parallel system used via AgentEngine only
  - `prompt_enhancer.py` - the `e:` prefix and `/enhance`
  - `provider_capabilities.py` - the declared capability table, enforced only by tests
  - `provider_initializer.py`, `provider_registry.py` - lazy provider loading
  - `chat_manager.py` (in-memory session), `conversation_manager.py` (saved conversations), `history_manager.py` (REPL keystroke history): three separate history concerns that are easy to conflate
  - `agent/` - `file_system_manager.py`, `program_executor.py`, `web_client.py`, `approval_manager.py`, `tool_definitions.py`, `workflow_orchestrator.py`, `status_core.py`, `types.py`
  - `security/` - `security_manager.py`, `permission_controller.py`, `permission_rules.py`, `sandbox_manager.py`, `approval_workflow.py`, `audit_logger.py`
- `omnimancer/providers/` - `base.py` (`BaseProvider`), `factory.py` (registry), `cache_tokens.py`, `claude_credentials.py`, and one module per backend
- `omnimancer/h2l/` - High plans, Low executes, High judges: `orchestrator.py`, `loop.py`, `planner.py`, `worker.py`, `judge.py`, `refs.py`, `models.py`, `prompts.py`, `render.py`
- `omnimancer/mcp/` - `client.py` and `manager.py` on the official `mcp` SDK, imported lazily
- `omnimancer/events/` - the `omn.event.v1` schema, JSONL writer, and emitter (the fleet feed)
- `omnimancer/tui/fleet/` - the `omn fleet` Textual dashboard (optional `tui` extra)
- `omnimancer/ui/` - Rich display pieces: streaming display, progress, turn activity, cancellation
- `omnimancer/utils/` - error hierarchy, error handler, retry with backoff
- `omn_fleet_hook.py` - stdlib-only Claude Code hook adapter (the `omn-fleet-hook` script); never import the omnimancer package from it
- `scripts/version_manager.py` - owns the version and `CHANGELOG.md`
- `factory-pipeline.yml` - CI (Factory; there is no GitHub Actions or GitLab pipeline)
- `docs/` - `CODEBASE_MAP.md` (accurate, detailed), `plans/` (PRDs and research memos), plus several stale documents (see Gotchas)
- `tests/` - about 127 test files
  - `tests/cli/`, `tests/core/` - colocated mirrors of the package
  - `tests/h2l/`, `tests/events/`, `tests/tui/`, `tests/ui/`, `tests/integration/` - feature suites
  - `tests/mcp_test_server.py` - a real FastMCP stdio server driven as a subprocess (needs `mcp<2`)
  - `tests/test_*.py` - provider and feature tests, flat at the root
  - `tests/conftest.py` - shared fixtures and auto-marking

## Architecture

### The two agent execution paths (the most important structural fact)

The fork is `engine.provider_supports_tools()`, and it does different jobs on the two surfaces:

- **Interactive (`cli/interface.py`)**: the check picks which loop runs, the native `_handle_tool_calling_flow` or the marker `_execute_continuous_workflow`. The native loop has NO iteration cap; only `RepeatedCallTracker` guards it.
- **Headless (`cli/headless.py`)**: always uses native tool calling. The check only changes the system-prompt text. There is no marker fallback, so markers from a non-tool provider render as dead text. The cap is `MAX_TOOL_ITERATIONS=25`, resolved as `--max-iterations`, then the `OMNIMANCER_MAX_ITERATIONS` env var, then the default.
- `headless.py` reimplements the agent loop, so loop changes usually land in both `interface.py` and `headless.py`.

**Native path** (`cli/tool_handler.py`): `CODING_AGENT_TOOLS` in `core/agent/tool_definitions.py` deliberately mirror Claude Code's names and schemas: `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `WebFetch`. `Edit` is synthesized client-side (read, unique-match replace, write). `Glob` and `Grep` shell out to real `find` and `grep -rE` with directory pruning, and an empty match returns "No X found" text so the repeat tracker does not kill the turn. `RepeatedCallTracker` lets an identical call run twice, nudges at 3, and aborts the turn at 5. Results are truncated head-plus-tail at 16k characters. `AUTO_APPROVED_TOOLS` is Read, Glob, Grep, WebFetch.

**Marker path** (`cli/agent_loop.py`): regex operation markers for providers without tool calling (cohere, ollama, perplexity, claude-code). `[COMMAND_EXEC]` carries AI-self-reported `read-only` or `modifies-system` metadata; the independent re-check is SecurityManager downstream. Web markers have an SSRF guard that blocks private, loopback, and metadata IPs, with a 10k response cap. The outer loop ends on a substring match against done-indicators, which also matches inside words like "incomplete".

### The operation gate: `core/agent_engine.py::execute_with_approval`

Both paths funnel every operation through this one gate. The order is exact, and each step vetoes only what follows:

1. **Preview generation** (informational diff or preview)
2. **Permission rules** (`core/security/permission_rules.py`): session rules from `set_read_only()` first, then persisted `Config.permissions`. Precedence is `deny > ask > allow > default`, re-read on every call. DENY fails immediately, ALLOW clears `requires_approval`, ASK forces a prompt.
3. **`tool_use_request` hook**: blocking. A non-zero exit or timeout vetoes, even after a rule ALLOW.
4. **Approval**: the `ApprovalManager` callback (`cli/approval_integration.py`, `cli/approval_prompt.py`). The `/accept` modes sit below rules, so a forced prompt beats ACCEPT_ALL.
5. **Execute**: dispatch to file_system, executor, web_client, or the MCP integrator.
6. **`post_tool` hook**: observe-only.

An **independent second layer** runs inside step 5: `FileSystemManager` and `ProgramExecutor` call `SecurityManager`, then `PermissionController`, which re-validates on every access. **Hard-restricted paths** (`~/.ssh`, `/etc`, `/root`, credential stores, and similar) are checked before the full-trust branch and have NO bypass: not approval, not `always_allow`, not full trust. Full trust only disables the sensitive-name patterns and the command allowlist and sanitizer, and raises the 30 second command timeout to 600 seconds.

Session mechanisms, both in-memory only and never persisted:

- `--read-only` calls `set_read_only(True)`: session deny rules for write, delete, mkdir, rmdir, and exec, evaluated before persisted rules.
- `--dangerously-skip-permissions` is full trust: auto-approve callbacks plus `set_full_trust` on the executor and file system.
- With no approval callback registered, the answer is **deny**. Headless cannot write or execute anything without `--dangerously-skip-permissions` or `--no-approval`.

### Core engine and config

- `CoreEngine.send_message` fires `pre_send_message` and `post_send_message`. `send_message_with_tools` fires `pre_send_message` ONLY. The streaming variants fire NO hooks. The reliable end-of-turn signal is the `turn_complete` hook or `--notify-cmd`.
- `_fire_hook` re-reads `config.hooks` on every call, so live edits apply, and it never raises.
- **Config precedence, highest first: env, then CLI session flags, then stored config.** CLI flags apply in memory right after `ConfigManager` construction; env applies later inside `CoreEngine.initialize_providers` and overwrites unconditionally. So `OMNIMANCER_OPENAI_MODEL` silently beats `--model`.
- API keys are Fernet-encrypted against `~/.omnimancer/.key` (mode 0600). A decryption failure falls back to treating the stored value as plaintext.
- Migrations run only from `ConfigManager.load_config()`. They must pass newer optional blocks through (enhancement, custom_models, fallback, permissions, hooks, subagents) and fill provider defaults gap-wise, never clobber with `update()`.
- If the default provider fails to initialize, `current_provider` stays unset with a named error. There is no silent substitution.
- `ConversationManager` drops `tool_calls` and `raw_content` on save, so saved conversations lose native tool-call structure.

### Providers

The contract in `providers/base.py`: the abstract methods are `send_message`, `validate_credentials`, `get_model_info`, `supports_tools`, and `supports_multimodal`. The default `send_message_with_tools` falls back to a plain send when `supports_tools()` is False, and raises `NotImplementedError` when it is True but not overridden. That is why capability honesty matters and why it is contract-tested in `tests/test_provider_capability_consistency.py`. The default `send_message_stream` wraps a full blocking response in a fake three-event stream.

The 15 registered names, at the bottom of `providers/factory.py`: claude, openai, gemini, cohere, ollama, perplexity, xai, mistral, azure, vertex, bedrock, openrouter, digitalocean, claude-code, openai-compatible.

- **Native tools**: claude, openai (plus openai-compatible and digitalocean), gemini, xai, mistral, azure, vertex, bedrock, openrouter
- **Real SSE streaming**: claude and the openai family only. Every other provider uses the base fake stream.
- **Native tool history replay**: the openai family only. `ChatMessage` keeps a dual representation (flattened text plus parallel `tool_calls` and `raw_content`) because text-flattened tool exchanges made models leak template tokens.
- **No tools by design**: cohere, ollama, perplexity (built-in web search is not function calling), claude-code

Lazy loading: `provider_initializer._get_module_path` maps a name to a module (special cases: `claude-code` is `claude_code`, `openai-compatible` is `openai_compatible`) and then picks the class whose name ends in `Provider`. So every provider class name MUST end in `Provider`, and `openai_compatible.py` imports its parent as `_OpenAIBase` to hide it from that introspection.

Alias and keyless entries: a config entry named for example `gateway` can set `provider_type: "openai-compatible"`. Class lookup follows the type, while identity (key decryption, env overrides, caches) stays on the name.

Provider quirks worth knowing: claude auto-prefers an unexpired subscription OAuth token from `~/.claude/.credentials.json` and maps 529 to `RateLimitError`; openai retries context overflow by refitting `max_tokens` and retries once on timeout; openrouter prepends a fallback-notice banner into response content when rerouted; perplexity appends Sources sections into content; xai injects a `web_search` tool by default; vertex sets `GOOGLE_APPLICATION_CREDENTIALS` process-wide; azure matches models by deployment-name substring.

### Resilience: backoff, checkpoints, prompt caching

- **Same-provider backoff**: `core/engine.py::build_rate_limit_retry_handler` wraps every provider send in `utils/retry.py::RetryHandler`: exponential backoff with jitter on `RateLimitError` and `NetworkError`, honoring `Retry-After`, BEFORE the provider-switch fallback. Env knobs: `OMNIMANCER_RATE_LIMIT_RETRIES` (default 5, 0 disables), `OMNIMANCER_RATE_LIMIT_BASE_DELAY` (2.0s), `OMNIMANCER_RATE_LIMIT_MAX_DELAY` (60s).
- **Rate-limit fallback**: `RateLimitFallbackHandler.should_fallback` (config-gated), then `get_next_provider` walks `fallback_order`, then approval (no callback and not auto means no switch), then switch and retry once.
- **Headless checkpoints** (`cli/headless_checkpoint.py`): a lossless snapshot of the conversation, pending message, iteration, tool log, and usage at the top of every iteration, in `~/.omnimancer/headless_checkpoints/<session_id>.json`. Override the directory with `OMNIMANCER_CHECKPOINT_DIR`; disable with `OMNIMANCER_CHECKPOINT=0`. Clean completion deletes the file; failures and cap hits keep it and print a resume hint on stderr.
- **Headless exit codes**: `0` done, `1` hard error, `3` iteration cap or repeat abort (a partial result is emitted), `4` rate limited after retries. Exit 4 is resumable: the error blob carries `stop_cause` and `resume_session_id`, and orchestrators should run `omn --resume <id>` instead of restarting the task.
- **Prompt caching**: request-side markers where the vendor takes them (`providers/claude.py::_apply_cache_control`, `providers/bedrock.py::_apply_cache_point` gated to supported models, `providers/openrouter.py`, `providers/digitalocean.py`). OpenAI-family, Gemini, and Vertex cache automatically server-side; `providers/cache_tokens.py` parses their cached-token usage. All request-side markers share the `OMNIMANCER_PROMPT_CACHE=0` kill switch.

### H2L: High plans, Low executes, High judges

`/h2l` in the REPL and `omn -p --h2l` in headless. A high model plans, gates the plan through the presenter, then stories run in dependency order across a pool of low models: round-robin assignment, judge feedback on retries, an escalation policy once retries are spent, serialized and labeled approval prompts, and per-story identity on the event bus.

- `h2l/orchestrator.py::H2LRun` is the one loop both surfaces call. Surfaces only supply a `Presenter`. The `h2l/` package imports nothing from `cli/interface.py` or `cli/headless.py`.
- `h2l/loop.py` is the one isolated tool-calling loop shared by the planner, workers, and judge. It keeps its own `ChatContext` and stops on completion, cancellation, a cap, a repeat abort, an error, or a stop tool (`submit_plan`, `submit_verdict`).
- `h2l/refs.py`: the rule that makes parallel tiers safe is to NEVER hand out an object from `engine.providers`. Each model ref gets its own provider instance built from a `model_copy` of the stored entry.
- Headless flags: `--h2l`, `--high <entry>:<model>`, `--low <entry>:<model>` (repeatable), `--h2l-parallel`, `--h2l-retries`, `--h2l-threshold`, `--h2l-escalation`, `--h2l-plan-only`, `--h2l-plan-iterations`.
- Design documents: `docs/plans/PRD-h2l.md` and `docs/plans/h2l-research.md`.

### MCP

Built on the official `mcp` SDK (pinned `<2`). It is a soft dependency twice over: the SDK imports lazily inside methods with a clear error if absent, and `MCPManager.initialize_servers` no-ops when disabled and degrades gracefully when only some servers connect. Transports: stdio, sse, http (streamable). Each `MCPClient` runs its whole session in one dedicated asyncio task with a command queue, because SDK context managers must enter and exit in the same task. `core/mcp_integration_layer.py::EnhancedMCPIntegrator` adds capability inference, retry with backoff, result caching (skipping write-like tools), and per-tool reliability metrics.

### Hooks, permissions, subagents

- **Hooks** (`core/hooks.py`, `/hooks`): five events in `HooksConfig`: `pre_send_message`, `post_send_message`, `tool_use_request` (blocking), `post_tool` (observe), `turn_complete` (observe; the one event whose payload does not get an injected `event` key).
- **Permission rules** (`core/security/permission_rules.py`, `/permissions`): allow, ask, and deny rules persisted in config.
- **Subagents** (`cli/subagent.py`, `/subagents`): a focused task with its own system prompt, a restricted tool allowlist, an optional model override, and its own conversation context, so the parent conversation is never touched.

### Fleet and the orchestration surface

- **Fleet**: agents emit `omn.event.v1` JSONL to `~/.omnimancer/events/` (`OMNIMANCER_EVENTS=0` disables, `OMNIMANCER_EVENTS_DIR` overrides). `omn fleet` tails those files. It is pre-dispatched from `sys.argv` in `main()` so the main click command's flag surface stays byte-identical for orchestrators that parse it. Textual is imported lazily behind an install hint. `omn_fleet_hook.py` duplicates the schema constants on purpose so it can stay stdlib-only and start fast; keep it in sync with `events/schema.py`.
- **`--notify-cmd <command>`**: invoked at the end of EVERY turn (success, exception, or cancel) with a JSON payload as the final argv: `type`, `turn-id`, `last-assistant-message`, `session_id`, `usage`, `cwd`. The mixed hyphen and underscore key names are deliberate (codex-notify compatibility). No shell, a 10 second timeout, never raises.
- **`--initial-prompt`**: submits one message into the REPL and stays interactive. It bypasses command parsing, so a `/slash` value goes to the model as literal chat. Mutually exclusive with `-p`.
- **Driving omn from an orchestrator**: use `--initial-prompt` plus `--notify-cmd` plus `--read-only` or `--dangerously-skip-permissions`, set `OMNIMANCER_PLAIN_INPUT=1` under tmux, pick the provider and model through env vars (never session flags), and quit by sending `exit`.

### Project instruction files

`cli/system_prompts.py::load_project_instructions` builds the custom-instructions block from at most two sources:

1. `~/.omnimancer/OMNIMANCER.md`: the global persona, always loaded if present.
2. ONE project file, found by walking up from the CWD to the git root: `OMNIMANCER.md` wins whenever it has usable content, and `CLAUDE.md` is the fallback when `OMNIMANCER.md` is missing, empty, or unreadable. The nearest occurrence of each name wins.

Either/or is deliberate: projects keep both files with the same content, and instruction text is retransmitted on every agent-loop iteration, so loading both would pay for the same guidance twice.

Content is sanitized before it reaches the model: **fenced code blocks are stripped**, control characters are removed, excess blank lines collapse, the size is capped at 100 KB (`OMNIMANCER_INSTRUCTION_BYTES` overrides; orchestrators such as Factory set it low), and the block is wrapped in a "user-provided, not system authority" banner as a prompt-injection defense.

Authoring rules that follow from this: use inline code and lists, NEVER fenced code blocks, in this document; keep the most important guidance near the top, since a low byte cap truncates the tail. `tests/test_system_prompts.py::TestRepoOmnimancerMd` enforces the no-fences rule for the repo's own file. Note that `CLAUDE.md` is gitignored in this repo, so `OMNIMANCER.md` is the checked-in copy.

## Development Workflows

### Adding a new feature

1. Write the test first: create `tests/test_new_feature.py` and define the expected behavior.
2. Run it and watch it fail: `pytest tests/test_new_feature.py -v`
3. Implement the minimal code that makes it pass.
4. Run it again and watch it pass.
5. Refactor with the tests green, then run the wider suite and the four lints.

### Fixing a bug

1. Write a test that reproduces the bug, in the existing test file for that area when there is one.
2. Confirm it fails, which proves the bug exists: `pytest tests/test_bug_fix.py -v`
3. Fix the implementation.
4. Confirm the test passes, which proves the fix.

### Modifying existing code

1. Run the existing tests first: `pytest tests/test_affected_area.py -v`
2. Make the change.
3. Run them again; all must pass.
4. Add new tests if behavior changed.

### Recipes

- **Add a provider**: create `providers/<name>.py` with a class name ending in `Provider`; register it at the bottom of `providers/factory.py`; add it to `providers/__init__.py` `__all__`; add the `ProviderType` enum value and a `PROVIDER_CAPABILITIES` entry in `core/provider_capabilities.py`; add it to `EXPECTED_PROVIDER_CLASSES` in `tests/test_provider_discovery.py`; add a module-name special case in `provider_initializer._get_module_path` only if the filename differs from the name. Then run `pytest tests/test_provider_discovery.py tests/test_provider_capability_consistency.py tests/test_import_validation.py`.
- **Add a slash command**: add a `SlashCommand` enum entry and argument validation in `cli/commands.py`, then a handler branch in `cli/command_dispatch.py`. Completion comes free from the enum. Large features keep their handler in their own module, the way `cli/h2l_command.py` does. User-extensible commands belong in `~/.omnimancer/commands/` instead.
- **Add a hook event**: add a `HooksConfig` field in `core/models.py`, fire it with `engine._fire_hook(event, ctx, match_target)` at every site (both agent paths, interactive and headless), and extend `tests/test_hooks.py`.
- **Change agent tool behavior**: `cli/tool_handler.py` (native) AND `cli/agent_loop.py` (markers); schemas live in `core/agent/tool_definitions.py`.
- **Touch the config schema**: `core/models.py`, plus a migration in `core/config_migration.py` (pass optional blocks through; gap-fill, never clobber), plus `core/config_validator.py` if the field is validated. Check all four defaults and validation tables.

## Testing Strategy

- **Unit tests**: every function and method has coverage.
- **Integration tests**: component interactions are tested.
- **Provider tests**: each provider has a comprehensive suite, plus the discovery and capability-consistency contracts.
- **CLI tests**: user interactions and commands, interactive AND headless.
- **Security tests**: approval, permission rules, hard-restricted paths, and validation workflows.
- **Agent path tests**: native tools and markers.
- Infrastructure notes: `asyncio_mode=auto` (older files still carry a redundant asyncio marker). `tests/conftest.py` auto-marks by node-id substring: "integration" is integration, "slow" or "load" is slow, and "network" or a bare "api" anywhere in the id is network, so `-m "not network"` deselects more than you expect. Isolate tests from the repo's own instruction files by patching `load_project_instructions`, `Path.cwd`, and `Path.home`.

## CLI Reference

- `omn` - start the interactive REPL; `omn --help` shows help; `python -m omnimancer` runs it directly
- `omn -p "prompt"` - headless one-shot; `--output-format text|json|stream-json`; `--verbose`; `--max-iterations N` (default 25)
- `omn --resume <session_id>` - resume a headless run from its checkpoint
- `omn --initial-prompt "..."` - submit one message, then stay interactive
- `omn --read-only` - deny file writes and command execution for the session
- `omn --no-approval` and `omn --dangerously-skip-permissions` - auto-approve operations
- `omn --provider X --model Y --base-url Z` - in-memory session overrides
- `omn --notify-cmd <cmd>` - turn-completion JSON payload to an external command
- `omn --config <path>` - use a specific configuration file
- `omn -p "..." --h2l --high entry:model --low entry:model` - headless H2L
- `omn fleet` - the fleet dashboard; install with `pip install 'omnimancer-cli[tui]'`

**Slash commands**: `/help`, `/models`, `/model`, `/switch`, `/clear`, `/save`, `/load`, `/quit` (alias `/exit`), `/status`, `/list`, `/providers`, `/tools`, `/mcp`, `/history`, `/add-model`, `/remove-model`, `/list-custom-models`, `/agent`, `/config`, `/hooks`, `/permissions`, `/accept`, `/enhance`, `/prompts`, `/subagents`, `/fallback`, `/h2l`. An unknown `/foo` falls through to chat verbatim.

**Input**: prompt_toolkit when on a TTY and `OMNIMANCER_PLAIN_INPUT` is not `1` (multiline, Enter submits, a trailing backslash or Esc then Enter continues, Shift+Tab cycles the `/accept` mode, double Ctrl+C exits); otherwise plain `input()` with readline completion. An `e:` prefix runs the prompt enhancer, and `@file` expands a file mention.

**Environment variables**: `OMNIMANCER_<PROVIDER>_API_KEY`, `OMNIMANCER_<PROVIDER>_MODEL`, `OMNIMANCER_<PROVIDER>_BASE_URL`, `OMNIMANCER_DEFAULT_PROVIDER`, `OMNIMANCER_MAX_ITERATIONS`, `OMNIMANCER_RATE_LIMIT_RETRIES`, `OMNIMANCER_RATE_LIMIT_BASE_DELAY`, `OMNIMANCER_RATE_LIMIT_MAX_DELAY`, `OMNIMANCER_CHECKPOINT`, `OMNIMANCER_CHECKPOINT_DIR`, `OMNIMANCER_PROMPT_CACHE`, `OMNIMANCER_INSTRUCTION_BYTES`, `OMNIMANCER_PLAIN_INPUT`, `OMNIMANCER_EVENTS`, `OMNIMANCER_EVENTS_DIR`.

## Agent Mode Capabilities

When agent mode is on (`/agent on`), tool-capable providers call native tools: `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `WebFetch`, plus MCP tools and subagents.

Providers without tool calling use operation markers, in the interactive REPL only:

- **Execute commands**: `[COMMAND_EXEC] command [/COMMAND_EXEC]`
- **Read and write files**: `[FILE_READ:path]`, `[FILE_WRITE:path] content [/FILE_WRITE]`, `[FILE_DELETE:path]`
- **Search**: `[FIND:pattern]` finds files by glob, `[SEARCH:text]` searches file contents, `[LOCATE:filename]` locates a file by name with fuzzy matching
- **Safe execution**: `[SAFE_EXEC] command [/SAFE_EXEC]` runs a fixed read-only allowlist and is auto-approved
- **Web requests**: `[WEB_GET:url]`, `[WEB_POST:url]`, `[WEB_REQUEST:url]`
- **Fuzzy matching**: `[LOCATE:]`, `[FILE_READ:]`, and `[FILE_DELETE:]` auto-correct typos in file names at a 70 percent similarity threshold, searching recursively

Every operation on either path goes through the operation gate, and existing files are backed up before modification.

## Security and Approval

- **Approval required**: file writes (with an automatic diff preview), file deletions, and command execution
- **Auto-approved**: `Read`, `Glob`, `Grep`, `WebFetch`; file reads, the `[SAFE_EXEC]` allowlist, and web GET requests
- **Rules beat modes**: permission rules (`deny > ask > allow`) sit above the `/accept` approval modes
- **Deny by default**: no approval callback means deny
- **No bypass**: hard-restricted paths cannot be written under any combination of approval, `always_allow`, or full trust
- **Also in force**: the command allowlist and sanitizer, sandboxed execution, read-before-write logic, automatic backups, directory boundary validation, the audit log, and sanitized instruction files

## Gotchas

1. **Env beats CLI flags.** `apply_env_overrides` runs after session overrides and assigns unconditionally.
2. **`send_message_with_tools` never fires `post_send_message`**, and streaming fires no hooks at all.
3. **Headless has no marker fallback**, and the interactive native loop has no iteration cap.
4. **Hard-restricted paths are absolute.**
5. `core/provider_capabilities.py` is imported by no runtime module. It is a declared-truth table enforced only by tests. Runtime truth is `BaseProvider.supports_*()`, and config defaults come from separate hardcoded tables in `config_manager.py` and `config_migration.py`.
6. **Same-name traps; check imports**: two `WebClient` classes (`core/agent/web_client.py` on aiohttp, `core/agent_managers.py` on httpx), two `OperationType` enums (`core/agent/types.py` and `core/agent/status_core.py`, with different values), two `ConfigValidator` classes (`config_migration.py` and `config_validator.py`, with incompatible signatures).
7. **Likely dead or legacy code**: `AgentModeManager._execute_operation` is a stub; `Config.merge_from_env`, `ConfigManager.load_config_from_sources`, and `migrate_config_format` are uncalled; the raw-file fallback branches in `agent_loop.py` are unreachable; the `ProviderRegistry` model catalog is never populated by the live path.
8. Marker-path `[COMMAND_EXEC]` approval necessity is AI-self-reported; the real check is downstream.
9. `--initial-prompt` bypasses command parsing.
10. The conftest marks any node id containing a bare "api" as network.
11. **Stale documents**: `docs/deployment.md`, `docs/mcp-setup.md`, `docs/api-reference.md`, `docs/agent-approval-system.md`, and `docs/file-modification-interaction-flow.md`. Trust `docs/CODEBASE_MAP.md`, `docs/plans/`, and the code.
12. `factory-pipeline.yml` bumps and tags the version BEFORE lint and tests run.
13. The notify payload's mixed hyphen and underscore keys are deliberate. Keep them.
14. The `mcp` SDK and Textual are soft dependencies. Import them lazily inside functions, never at module top level.

## Common Issues and Solutions

- **The agent describes actions but does not execute.** With a tool-capable provider, check that tools were sent and that `provider_supports_tools()` is true. With a marker provider, check that markers are used rather than "I will run...". In headless, markers from a non-tool provider are dead text by design.
- **`--model` or `--provider` did not take.** Check the `OMNIMANCER_*` env vars first, because they win. Then check that the provider entry exists; a bare `--base-url` creates an entry with an empty model.
- **A headless run cannot write files.** That is deny-by-default with no approval callback. Pass `--dangerously-skip-permissions`.
- **Headless exited with code 4.** It was rate limited after retries. Resume with `omn --resume <session_id>` instead of restarting.
- **A file is not found at the exact path.** On the marker path, fuzzy matching is enabled, so an approximate file name works.
- **Tests fail after a change.** Run `pytest -v --tb=short`, read the failure, and fix the implementation rather than the test, unless the test encoded the wrong behavior.
- **A new provider does not work.** The class name must end in `Provider`, inherit `BaseProvider`, be registered in the factory, agree with the capability table, and have tests.

## Development Commands

- `pip install -e ".[dev]"` - editable install with the test and lint tools
- `pytest tests/ -v` - all tests
- `pytest tests/test_file.py::test_func -v` - one test
- `pytest -k "pattern" -v` - tests matching a pattern
- `pytest --lf` - only the last failures
- `pytest tests/core/ -v` and `pytest tests/h2l/ -v` - one suite
- `pytest tests/ --cov=omnimancer --cov-report=term-missing` - coverage report

## CI and Releases

CI is **Factory** (`factory-pipeline.yml`): version bump, then lint (black, isort, flake8, mypy), then tests on Python 3.10, 3.11, and 3.12, then integration, then security (bandit, safety, pip-audit, licenses), then build, then a manual gate, then PyPI. The version bump derives from the last commit message, so use conventional commits.

## Remember

1. **Write tests FIRST.** TDD is not optional.
2. **Run tests OFTEN**: before commit, after changes, during development.
3. **Coverage matters.** Aim above 80 percent.
4. **Integration tests are crucial.** Unit tests plus integration tests equal confidence.
5. **#ItsNotDoneUntilItsTested.** This is the law.
6. **Both agent paths, both surfaces.** Native tools and markers; interactive and headless.
7. **Lint everything**: flake8, mypy, isort, black, with the exact flags above, before saying you are done.
8. **Act like the principal engineer you are**: understand first, change the minimum, protect the security model, and report the truth.

---

*Omnimancer: One CLI to rule them all* 🔮
