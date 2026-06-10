# Omnimancer

A multi-model coding agent for the terminal. One tool, any LLM.

Omnimancer works like `claude -p` but isn't locked to a single provider. Point it at Claude, OpenAI, Gemini, Bedrock, Ollama, or any of 13+ supported backends and get a coding agent that reads files, writes code, runs commands, and iterates autonomously — with streaming responses, token/cost tracking, and structured JSON output for pipeline integration.

Beyond the basics, it ships with:

- **MCP support** (stdio, SSE, and HTTP transports) for tools, resources, and prompts
- **Lifecycle hooks** — run shell commands on message/tool events, with blocking veto power
- **Permission rules** — declarative allow/deny/ask rules per tool with regex matchers
- **Subagents** — scoped child agents with their own prompt, tool whitelist, and model
- **Layered security** — approval workflow, sensitive-path protection, and project-boundary enforcement

## Install

```bash
pip install omnimancer-cli
```

## Usage

### Headless (pipeline mode)

```bash
# Single prompt, JSON output — like claude -p
omn -p "refactor auth.py to use dependency injection"

# Pipe context in
cat error.log | omn -p "diagnose this crash and suggest a fix"

# Use a specific provider and model.
# Note: the prompt must come right after -p; put other flags after it.
omn -p "write tests for src/api/routes.py" --provider claude --model claude-sonnet-4
omn -p "explain this codebase" --provider openai --model gpt-4o
omn -p "review this diff" --provider ollama < changes.patch

# Output formats
omn -p "summarize this repo"                                # plain text (default)
omn -p "summarize this repo" --output-format json           # structured JSON
omn -p "summarize this" --output-format stream-json         # streaming JSON

# Auto-approve all tool operations (CI/scripts)
omn -p "fix the failing tests" --dangerously-skip-permissions

# Verbose output / explicit config file
omn -p "audit the security module" --verbose --config ~/.omnimancer/config.json

# Raise the tool-iteration cap for big jobs (default: 25)
omn -p "migrate every test file to pytest fixtures" --max-iterations 100
```

The `--provider`, `--model`, and `--base-url` flags are **session overrides**: they
modify the in-memory config for this run only and are never written back to disk.

Headless mode with `--output-format json` emits a single structured result
object — including the tool calls the agent made along the way:

```json
{
  "type": "result",
  "subtype": "success",
  "is_error": false,
  "result": "Here's the refactored code...",
  "session_id": "…",
  "model": "claude-sonnet-4-20250514",
  "num_turns": 3,
  "tool_calls": [
    {"name": "file_read", "arguments": {"path": "src/auth.py"}, "error": null},
    {"name": "file_write", "arguments": {"path": "src/auth.py"}, "error": null}
  ],
  "usage": {"input_tokens": 1523, "output_tokens": 892, "total_cost_usd": 0.04},
  "total_cost_usd": 0.04,
  "stop_reason": "end_turn"
}
```

On failure it stays valid JSON (stdout), with the error and any tool calls made:

```json
{"type": "result", "subtype": "error", "is_error": true, "error": "…", "tool_calls": [...]}
```

For a live, line-by-line stream of what the agent is doing (assistant text,
each `tool_use`, each `tool_result`, then the final `result`), use
`--output-format stream-json`.

### Interactive mode

```bash
omn                                  # start interactive REPL
omn --provider openai                # start with a specific provider (session-only)
omn --no-approval                    # skip approval prompts
omn --dangerously-skip-permissions   # auto-approve all tool operations
```

Interactive mode gives you a REPL with streaming responses, token/cost display, and agent capabilities:

```
>>> read src/main.py and add error handling
[text streams in real-time as the model generates]
  tokens: 1523 in / 892 out | ~$0.0134

>>> /switch openai gpt-4o
>>> now review what we just changed
[switches to OpenAI, continues conversation]
```

### Streaming responses

Responses stream token-by-token as the model generates, so you see output immediately instead of waiting for the full response. After each response, a token/cost summary is displayed.

Streaming is automatic for providers that support it. Other providers fall back to displaying the full response once complete — no configuration needed.

| Provider | Streaming |
|----------|:---------:|
| Claude (Anthropic) | Yes |
| All others | Fallback (full response) |

Streaming works in both regular chat and agent mode (tool calling flow). The display uses a live-updating terminal panel that refreshes at 15fps.

### Agent mode

When agent mode is enabled (`/agent on`), the AI can autonomously:

- **Read and write files** with approval workflow
- **Execute shell commands** with security validation
- **Search codebases** with fuzzy file matching (70% similarity threshold)
- **Make HTTP requests** for API testing
- **Call MCP tools** from connected MCP servers

All destructive operations require explicit approval (or are governed by your [permission rules](#permission-rules)). Reads and searches are auto-approved.

Providers that support native tool calling (Claude, OpenAI, Gemini) use structured function calls. Others fall back to operation markers parsed from the response text.

```
>>> /agent on
>>> fix the failing test in tests/test_auth.py
[agent reads test file, reads source, edits code, runs pytest, iterates]
  tokens: 4210 in / 1893 out | ~$0.0412
```

### Subagents

Subagents are scoped child agents with their own system prompt, tool whitelist, model override, and iteration cap. Each run uses an isolated conversation context, so a subagent never pollutes your main session — and its errors are caught and reported instead of crashing the REPL.

Define them in `config.json`:

```json
"subagents": {
  "reviewer": {
    "description": "Reviews code for bugs and style issues",
    "prompt": "You are a meticulous code reviewer...",
    "tools": ["Read", "Grep", "Bash"],
    "model": null,
    "max_iterations": 10
  }
}
```

Tool names match the agent's toolset (`Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `WebFetch`). `tools: null` inherits all tools; `model: null` inherits the session model.

```
>>> /subagents                          # list configured subagents
>>> /subagents run reviewer check src/auth.py for security issues
```

## Supported Providers

| Provider | Tool Calling | Streaming | Notes |
|----------|:---:|:---:|-------|
| **Claude** (Anthropic) | Yes | Yes | Primary target. Best coding performance. |
| **OpenAI** | Yes | Fallback | GPT-4o, o1, etc. |
| **Gemini** (Google) | Yes | Fallback | Large context window. |
| **AWS Bedrock** | Yes | Fallback | Claude/Titan via AWS. |
| **Ollama** | No | Fallback | Local models. No API key needed. |
| **xAI** (Grok) | Yes | Fallback | |
| **Mistral** | No | Fallback | |
| **Perplexity** | No | Fallback | Web search built-in. |
| **Azure OpenAI** | Yes | Fallback | Enterprise Azure deployment. |
| **Vertex AI** | Yes | Fallback | Google Cloud deployment. |
| **OpenRouter** | No | Fallback | Access to 100+ models. |
| **DigitalOcean** | No | Fallback | OpenAI-compatible GenAI inference. Custom endpoint supported. |
| **Cohere** | No | Fallback | |

"Fallback" means the provider works but sends the full response at once instead of streaming token-by-token. The UI handles both modes transparently.

## Commands

| Command | Description |
|---------|-------------|
| `/help [command]` | Show help (optionally for a specific command) |
| `/quit` | Exit (also: `/exit`, Ctrl+D) |
| `/clear` | Clear terminal screen |
| `/switch <provider> [model]` | Switch provider or model (uncataloged models are registered on the fly) |
| `/models [filter]` | List available models (alias: `/model`) |
| `/providers` | List all providers with status |
| `/agent on\|off\|status` | Toggle agent mode |
| `/subagents [run <name> <task>]` | List or run scoped child agents |
| `/config show\|get\|set` | View or modify configuration |
| `/config set-provider <name> [--api-key …] [--base-url …] [--model …]` | Create/update a provider |
| `/config remove-provider <name>` | Remove a provider |
| `/hooks [list\|on\|off\|add\|remove]` | Manage lifecycle hooks |
| `/permissions [list\|on\|off\|allow\|deny\|ask\|remove]` | Manage permission rules |
| `/save [name]` | Save conversation |
| `/load [name]` | Load conversation |
| `/list` | List saved conversations |
| `/history [recent\|search\|clear\|export\|stats]` | Manage conversation history |
| `/tools` | Show available MCP tools |
| `/prompts [list\|<name> [key=value …]]` | List and render MCP prompts |
| `/mcp [status\|reload\|connect\|disconnect\|health] [server]` | MCP server management |
| `/add-model <name> <provider> [description]` | Register a custom model |
| `/remove-model <name> <provider>` | Unregister a custom model |
| `/list-custom-models` | List registered custom models |
| `/status` | System status |

You can also define your own slash commands: drop `.json` or `.py` command
definitions into `~/.omnimancer/commands/` and they're loaded at startup.

## MCP servers

Omnimancer's MCP layer is built on the official [`mcp` SDK](https://pypi.org/project/mcp/) and supports three transports — **stdio** (local subprocess), **SSE**, and **streamable HTTP** (remote) — and three MCP features: **tools** (exposed to the agent and listed via `/tools`), **resources**, and **prompts** (listed and rendered via `/prompts`).

Configure servers in `config.json`:

```json
"mcp": {
  "enabled": true,
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/project"],
      "env": {},
      "enabled": true,
      "auto_approve": ["read_file"],
      "timeout": 30
    },
    "remote-api": {
      "transport": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": {"Authorization": "Bearer ..."},
      "enabled": true
    }
  }
}
```

Stdio servers need `command` (plus optional `args`/`env`); SSE and HTTP servers need `url` (plus optional `headers` — e.g. a bearer token for authenticated servers). `auto_approve` lists tool names that skip the approval prompt for that server.

Manage servers at runtime with `/mcp status`, `/mcp connect [server]`, `/mcp disconnect [server]`, `/mcp reload [server]`, and `/mcp health [server]`.

MCP is optional: if the `mcp` package isn't installed, everything else still works — connecting to a server just tells you to `pip install mcp`.

## Hooks

Hooks run shell commands at lifecycle events — for logging, notifications, linting, or policy enforcement. Four events are available:

| Event | Fires | Can block? |
|-------|-------|:---:|
| `pre_send_message` | Before a message is sent to the provider | Yes |
| `post_send_message` | After a successful provider response | No |
| `tool_use_request` | Before an agent tool/operation runs | Yes |
| `post_tool` | After a tool/operation completes | No |

A **blocking** hook vetoes the action if it exits non-zero (or times out). Hooks receive the event payload as JSON on stdin, plus `OMNIMANCER_HOOK_*` environment variables (`OMNIMANCER_HOOK_EVENT`, `OMNIMANCER_HOOK_MESSAGE`, …) for shell one-liners. Hook errors never crash the app.

Manage from the CLI:

```
>>> /hooks                                   # list configured hooks
>>> /hooks add post_tool notify --timeout 10 notify-send "tool ran"
>>> /hooks add tool_use_request guard --matcher "rm -rf" --blocking exit 1
>>> /hooks remove post_tool notify
>>> /hooks off                               # disable all hooks globally
```

`--matcher` is a regex tested against the event's target (message text, file path, command) — the hook only fires on a match. Hooks are stored in `config.json` under `hooks`:

```json
"hooks": {
  "enabled": true,
  "tool_use_request": [
    {"name": "guard", "command": "exit 1", "matcher": "rm -rf", "blocking": true, "timeout": 30}
  ]
}
```

## Permission rules

Permission rules decide what the agent may do before the approval prompt is ever shown. Each rule matches a tool (an operation type like `file_write`, `command_execute`, `web_request` — or `*` for any) plus an optional regex on the target (file path, command, URL).

Precedence: **deny > ask > allow > default** (normal approval workflow).

```
>>> /permissions                             # list rules
>>> /permissions deny command_execute "rm -rf"
>>> /permissions ask file_write ".*prod.*"
>>> /permissions allow file_write "\.env$"   # authorize project-local .env writes
>>> /permissions remove deny 1               # remove rule by index
>>> /permissions off                         # disable rules globally
```

- `deny` rules refuse without prompting.
- `ask` rules force a prompt even if the operation was previously "remembered" as approved.
- `allow` rules auto-approve — and also authorize writes to sensitive-named files (like `.env`) inside the project.

Rules live in `config.json` under `permissions` (`always_deny` / `always_ask` / `always_allow` lists).

## Security model

Three layers govern every agent operation, in order:

1. **Permission rules** — your declarative deny/ask/allow rules (above).
2. **Approval workflow** — interactive y/n with diff preview; approvals can be "remembered" per session.
3. **Low-level security gate** — path and command validation that runs regardless of approval.

The security gate distinguishes:

- **Hard-restricted paths** — never writable, even with approval: system directories (`/etc`, `/sys`, `/proc`, `/boot`, `/usr/bin`, …), `~/.ssh`, `~/.aws/credentials`, `~/.config/gcloud`.
- **Sensitive name patterns** — denied by default but overridable by explicit approval or an `always_allow` rule: `.env*`, `*secret*`, `*key*`, `*token*`, `*password*`, `*credentials*`, `*.db`/`*.sqlite*`.
- **Project boundary** — writes must stay inside the directory `omn` was launched from (or a temp dir).

Plus: command whitelist/blacklist, sandboxed execution, read-before-write logic, and automatic backups of existing files before modification.

## Configuration

### API keys

The simplest setup is environment variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
export XAI_API_KEY="..."
omn
```

### Config file

Config is stored in `~/.omnimancer/config.json` (API keys are encrypted at rest).
You can edit it directly or, more conveniently, configure everything from the CLI:

```bash
omn
# Configure a provider in one step (api key is encrypted before storage)
>>> /config set-provider claude --api-key sk-ant-...
>>> /config set-provider openai --api-key sk-... --model gpt-4o
>>> /config set-provider openrouter --api-key sk-or-... --model anthropic/claude-3.5-sonnet

# Set or change individual fields
>>> /config set providers.openai.base_url https://my-proxy.example.com/v1
>>> /config set providers.openai.max_tokens 8192
>>> /config set default_provider claude

# Inspect / clean up
>>> /config show
>>> /config get default_provider
>>> /config remove-provider openai
```

### Multiple endpoints

Each of these is just a provider you can point anywhere via `base_url`, so you can run
several endpoints side by side and switch between them with `/switch <provider>`:

| Endpoint | Provider name | Default base URL |
|----------|---------------|------------------|
| Claude (direct) | `claude` | `https://api.anthropic.com/v1` |
| OpenAI | `openai` | `https://api.openai.com/v1` |
| OpenRouter | `openrouter` | `https://openrouter.ai/api/v1` |
| DigitalOcean inference | `digitalocean` | `https://inference.do-ai.run/v1` |

Any OpenAI-compatible service (local proxy, gateway, self-hosted model) works by
overriding `base_url` on the `openai` provider.

### Environment variable overrides

Environment variables take precedence over the saved config and are applied at
runtime only (never written back to disk). This makes them ideal for CI and for
testing endpoints without touching `config.json`.

```bash
# Conventional API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
export DIGITALOCEAN_INFERENCE_KEY="..."

# Per-provider overrides: OMNIMANCER_<PROVIDER>_{API_KEY,BASE_URL,MODEL}
export OMNIMANCER_OPENAI_BASE_URL="http://localhost:1234/v1"
export OMNIMANCER_DIGITALOCEAN_MODEL="llama3.3-70b-instruct"

# Pick the default provider for this run
export OMNIMANCER_DEFAULT_PROVIDER="digitalocean"
omn
```

You can also override the endpoint for a single headless run with `--base-url`:

```bash
omn -p "summarize README.md" --provider openai --base-url http://localhost:1234/v1
omn -p "explain this repo" --provider digitalocean --model llama3.3-70b-instruct
```

### Provider-specific setup

**Claude (Anthropic):**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Models: claude-sonnet-4, claude-opus-4, claude-3-5-sonnet
```

**OpenAI:**
```bash
export OPENAI_API_KEY="sk-..."
# Models: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
```

**Google Gemini:**
```bash
export GOOGLE_API_KEY="..."
# Models: gemini-1.5-pro, gemini-1.5-flash
```

**AWS Bedrock:**
```bash
# Uses AWS credentials (env vars, ~/.aws/credentials, or IAM role)
export AWS_DEFAULT_REGION="us-east-1"
# Models: anthropic.claude-3-5-sonnet, amazon.titan
```

**OpenRouter:**
```bash
export OPENROUTER_API_KEY="sk-or-..."
# Models: anthropic/claude-3.5-sonnet, openai/gpt-4o, and 100+ more
```

**DigitalOcean inference (OpenAI-compatible):**
```bash
export DIGITALOCEAN_INFERENCE_KEY="..."
# Default endpoint: https://inference.do-ai.run/v1
# Models: llama3.3-70b-instruct, llama3-8b-instruct, openai-gpt-4o
# Override the endpoint if needed:
export OMNIMANCER_DIGITALOCEAN_BASE_URL="https://inference.do-ai.run/v1"
```

**Ollama (local, no API key):**
```bash
ollama serve
ollama pull llama3.1
omn
>>> /switch ollama llama3.1
```

**Azure OpenAI:**
```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

## Architecture

```
omnimancer/
├── cli/                    # CLI interface (modular)
│   ├── interface.py       # Core REPL loop, streaming integration
│   ├── headless.py        # -p pipeline mode (text/json/stream-json)
│   ├── command_dispatch.py # Slash command handlers
│   ├── agent_loop.py      # Marker-based agent workflow
│   ├── tool_handler.py    # Native tool call execution
│   ├── subagent.py        # Scoped child agent runner
│   ├── system_prompts.py  # Prompt building
│   ├── display.py         # Terminal output & token status
│   └── completion.py      # Tab completion
├── core/                   # Engine & business logic
│   ├── engine.py          # Provider abstraction & streaming delegation
│   ├── agent_engine.py    # Autonomous agent capabilities
│   ├── agent_managers.py  # Executor/web/MCP/approval manager facades
│   ├── hooks.py           # Lifecycle hooks manager
│   ├── models.py          # Data models (ChatResponse, StreamEvent, etc.)
│   ├── agent/             # File ops, approval, tool definitions
│   └── security/          # Permission rules, path validation, sandboxing
├── providers/              # 13+ AI provider implementations
│   ├── base.py            # Provider interface (streaming fallback)
│   ├── claude.py          # Anthropic (native streaming & tool calling)
│   ├── openai.py          # OpenAI (native tool calling)
│   └── ...
├── ui/                     # Terminal UI components
│   └── streaming_display.py # Rich Live streaming display
└── mcp/                    # Model Context Protocol (official mcp SDK)
    ├── client.py          # Per-server lifecycle (stdio/SSE/HTTP)
    └── manager.py         # Multi-server orchestration
```

### Streaming architecture

Streaming uses async generators that flow through the full stack:

```
Provider (SSE parsing) → Engine (delegation) → Interface (display routing)
                                                  ↓
                                          StreamingDisplay (Rich Live panel)
```

Each layer yields `StreamEvent` objects. Providers that don't implement real streaming get an automatic fallback in `BaseProvider` that wraps the full response in the same event format, so the UI code works identically for all providers.

## Development

```bash
git clone https://gitlab.com/jite-ai/omnimancer
cd omnimancer
pip install -e ".[dev]"
pytest tests/ -v
```

Tests follow TDD. 1,460 tests across providers, CLI, streaming, agent operations, hooks, permissions, MCP, and integration scenarios (including a real MCP stdio server exercised end-to-end).

## License

MIT
