# Omnimancer

A multi-model coding agent for the terminal. One tool, any LLM.

Omnimancer works like `claude -p` but isn't locked to a single provider. Point it at Claude, OpenAI, Gemini, Bedrock, Ollama, or any of 13+ supported backends and get a coding agent that reads files, writes code, runs commands, and iterates autonomously — with streaming responses, token/cost tracking, and structured JSON output for pipeline integration.

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

# Use a specific provider and model
omn -p --provider claude --model claude-sonnet-4 "write tests for src/api/routes.py"
omn -p --provider openai --model gpt-4o "explain this codebase"
omn -p --provider ollama "review this diff" < changes.patch

# Output formats
omn -p "summarize this repo"                          # plain text (default)
omn -p --output-format json "summarize this repo"     # structured JSON
omn -p --output-format stream-json "summarize this"   # streaming JSON

# Auto-approve all tool operations (CI/scripts)
omn -p --dangerously-skip-permissions "fix the failing tests"
```

Headless mode with `--output-format json` outputs:

```json
{
  "response": "Here's the refactored code...",
  "model": "claude-sonnet-4-20250514",
  "tool_calls": [
    {"tool": "file_read", "args": {"path": "src/auth.py"}, "result": "..."},
    {"tool": "file_write", "args": {"path": "src/auth.py"}, "result": "success"}
  ],
  "tokens": {"input": 1523, "output": 892}
}
```

### Interactive mode

```bash
omn                        # start interactive REPL
omn --provider openai      # start with a specific provider
omn --no-approval          # skip approval prompts
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

All destructive operations require explicit approval. Reads and searches are auto-approved.

Providers that support native tool calling (Claude, OpenAI, Gemini) use structured function calls. Others fall back to operation markers parsed from the response text.

```
>>> /agent on
>>> fix the failing test in tests/test_auth.py
[agent reads test file, reads source, edits code, runs pytest, iterates]
  tokens: 4210 in / 1893 out | ~$0.0412
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
| **Cohere** | No | Fallback | |

"Fallback" means the provider works but sends the full response at once instead of streaming token-by-token. The UI handles both modes transparently.

## Commands

| Command | Description |
|---------|-------------|
| `/help [command]` | Show help (optionally for a specific command) |
| `/quit` | Exit (also: `/exit`, Ctrl+D) |
| `/clear` | Clear terminal screen |
| `/switch <provider> [model]` | Switch provider or model |
| `/models [filter]` | List available models |
| `/providers` | List all providers with status |
| `/agent on\|off\|status` | Toggle agent mode |
| `/config show\|set\|get` | View or modify configuration |
| `/validate [provider]` | Validate provider configurations |
| `/health [provider]` | Check provider health |
| `/save [name]` | Save conversation |
| `/load [name]` | Load conversation |
| `/list` | List saved conversations |
| `/history` | Manage conversation history |
| `/tools` | Show available MCP tools |
| `/mcp status\|health\|reload` | MCP server management |
| `/status` | System status |

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

Config is stored in `~/.omnimancer/config.json`. You can edit it directly or use the CLI:

```bash
omn
>>> /config set default_provider claude
>>> /config get default_provider
>>> /config validate                    # validate all provider configs
>>> /config validate claude             # validate specific provider
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
│   ├── command_dispatch.py # Slash command handlers
│   ├── agent_loop.py      # Marker-based agent workflow
│   ├── tool_handler.py    # Native tool call execution
│   ├── system_prompts.py  # Prompt building
│   ├── display.py         # Terminal output & token status
│   └── completion.py      # Tab completion
├── core/                   # Engine & business logic
│   ├── engine.py          # Provider abstraction & streaming delegation
│   ├── agent_engine.py    # Autonomous agent capabilities
│   ├── models.py          # Data models (ChatResponse, StreamEvent, etc.)
│   └── agent/             # File ops, approval, security
├── providers/              # 13+ AI provider implementations
│   ├── base.py            # Provider interface (streaming fallback)
│   ├── claude.py          # Anthropic (native streaming & tool calling)
│   ├── openai.py          # OpenAI (native tool calling)
│   └── ...
├── ui/                     # Terminal UI components
│   └── streaming_display.py # Rich Live streaming display
└── mcp/                    # Model Context Protocol
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

Tests follow TDD. 1,260 tests across providers, CLI, streaming, agent operations, and integration scenarios.

## License

MIT
