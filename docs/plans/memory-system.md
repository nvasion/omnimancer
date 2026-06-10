# Plan: Memory System

**Status:** Proposed (plan only — not implemented)
**Motivation:** Typing `continue where I left off` today gives the model nothing to
work with — it starts blind-running `git log` / `ls` to rediscover context. Omnimancer
has no persistence between sessions beyond manually `/save`d conversations.

## Goals

1. **Session continuity** — a new session knows what the last one was doing, so
   "continue where I left off" resolves without the agent re-exploring the repo.
2. **Project memory** — durable, human-editable facts about a project (build
   commands, conventions, gotchas) injected into the agent's system prompt.
3. **User memory** — cross-project preferences (e.g. "always run black before
   committing", preferred provider/model).
4. **Agent-writable memory** — the agent can save a fact it learned, subject to
   the existing approval workflow (a memory write is a file write).

Non-goals (for v1): vector search / embeddings, cross-machine sync, automatic
fact extraction from every message.

## Design

### Storage layout

Two scopes, mirroring how config already splits global vs project:

```
~/.omnimancer/memory/           # global (user) scope
├── MEMORY.md                   # index + freeform user preferences
└── <slug>.md                   # one fact per file

<project>/.omnimancer/memory/   # project scope (committable, like CLAUDE.md)
├── MEMORY.md                   # index + project conventions
├── <slug>.md                   # one fact per file
└── last-session.md             # auto-written session summary (gitignored)
```

- Plain Markdown with a small YAML frontmatter (`name`, `description`, optional
  `type: user|project|session`). Human-editable, diffable, no database.
- `MEMORY.md` is the only file loaded wholesale; per-fact files are listed in it
  one line each so the model can ask to read them (cheap on tokens).
- `last-session.md` is overwritten at the end of each session and excluded from
  git via the existing config-dir handling.

### Components

| Piece | Where | Responsibility |
|-------|-------|----------------|
| `MemoryManager` | `core/memory.py` (new) | Load/save/list/delete memories in both scopes; enforce size budget |
| Prompt injection | `cli/system_prompts.py` | Prepend `MEMORY.md` (both scopes) + `last-session.md` to the agent prompt, under a token budget (~2k tokens, truncate oldest first) |
| Session summary writer | `cli/interface.py` exit path (and `/quit`) | On session end, ask the current provider for a 5–10 line summary of the conversation (skip if < 2 user messages); write `last-session.md` |
| `/memory` command | `commands.py` + `command_dispatch.py` + `completion.py` + `display.py` | `/memory [list \| show <name> \| add <name> <text> \| remove <name> \| clear-session]` — follows the `/hooks`-`/permissions` pattern |
| `memory_write` tool | `core/agent/tool_definitions.py` + `cli/tool_handler.py` | Lets the agent persist a fact; maps to a `FILE_WRITE` operation inside `.omnimancer/memory/`, so the normal approval + diff preview applies unchanged |
| Config | `core/models.py` (`MemoryConfig`) | `enabled`, `max_prompt_tokens`, `auto_session_summary` toggles |

### How "continue where I left off" works after this

1. Session ends → summary of what was being worked on lands in `last-session.md`.
2. New session starts → system prompt contains the summary + project `MEMORY.md`.
3. The model answers from context instead of issuing exploratory `git log`/`ls`
   tool calls (the exact behavior that triggered this plan).

### Interaction with existing systems

- **Approval/permissions:** memory writes are ordinary `file_write` operations,
  so permission rules (`/permissions allow file_write "\.omnimancer/memory/.*"`)
  and hooks apply with zero new security surface. The memory dir is inside the
  project boundary, so the low-level gate is already satisfied.
- **Headless mode:** memory is *read* (prompt injection) but the session summary
  is *not* written by default — `-p` runs are one-shot and would churn the file.
  Flag `--no-memory` disables injection for clean CI runs.
- **`/save`–`/load`:** unchanged. Conversations remain full transcripts; memory
  is the distilled layer above them.

## Phases (TDD each — tests first)

1. **Core storage** — `MemoryManager` CRUD on both scopes, frontmatter parsing,
   token-budget truncation. Tests: `tests/core/test_memory.py`.
2. **Prompt injection** — `build_agent_prompt` includes memory under budget;
   missing/empty memory dirs are a silent no-op. Tests: `tests/test_system_prompts.py`.
3. **`/memory` command** — subcommand parsing, dispatch, completion, help.
   Tests: `tests/cli/test_memory_command.py`.
4. **Session summary** — exit-path summary written via provider call; degraded
   gracefully when the provider errors (never block exit). Tests: mock provider.
5. **`memory_write` tool** — tool definition, handler mapping, approval flow.
   Tests: `tests/test_tool_handler.py` additions.

Phases 1–3 are independently shippable; 4–5 build on them.

## Open questions

- Should `last-session.md` keep a rolling history (last N sessions) instead of
  overwriting? Start with overwrite; revisit if users want `/memory history`.
- Auto-capture ("the agent silently remembers things") is deliberately out of
  v1 — every memory write is explicit (user command or approved tool call).
- Global-scope writes by the agent: allow, or project-scope only? Proposal:
  project-only for the tool; global only via `/memory add --global`.
