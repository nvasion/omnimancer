# Omnimancer CLI
## Just-In-Time Engineering, Inc.

A unified command-line interface for multiple AI providers - chat with Claude, OpenAI, Gemini, and 10+ other AI models through a single, intuitive tool.

## Quick Start

### Installation

**Using pipx (recommended):**
```bash
pipx install omnimancer-cli
```

**Using pip:**
```bash
pip install omnimancer-cli
```

### Available Commands

After installation, use any of these commands:
- **`omnimancer`** - Full command name
- **`omn`** - Quick alias ⚡
- **`omniman`** - Alternative alias

### First Run

```bash
omn  # or omnimancer, or omniman
```

On first run, you'll be guided through setup:

```
🚀 Omnimancer Setup Wizard

Select a provider to configure:
1. Claude (Anthropic)
2. OpenAI  
3. Google Gemini
4. Perplexity AI
5. Ollama (Local)
...

Choose [1]: 1

Enter your Claude API key: sk-ant-...
✅ Configuration complete!

>>> Hello! How can you help me today?
🤖 Claude: I'm Claude, an AI assistant created by Anthropic...
```

## Basic Usage

```bash
# Start Omnimancer
omn

# Start chatting
>>> What's the weather like?

# Switch models mid-conversation  
>>> /switch openai gpt-4o
>>> Now using GPT-4. How are you different?

# Check available providers and models
>>> /providers
>>> /models

# Save conversations
>>> /save my-chat

# Load previous conversations
>>> /load my-chat

# Get help
>>> /help
```

## Agent Mode & File Operations

Omnimancer includes advanced agent capabilities that allow AI models to perform file operations with your explicit approval:

### 🤖 **Autonomous Agent Features**
- **File Creation**: Create new files with AI-generated content
- **File Modification**: Edit existing files with intelligent changes
- **Code Refactoring**: Restructure and improve existing code
- **Documentation Generation**: Create comprehensive documentation
- **Project Setup**: Initialize new projects with proper structure

### 🔒 **Secure Approval System**
Every file operation requires your explicit approval with:

```bash
🔍 File Operation Approval Required
📄 Creating: data_analyzer.py
📊 Risk Level: Low | 🟢 
📏 Size: 1,247 bytes (45 lines)

[Y] Approve  [N] Deny  [D] View Details  [Q] Quit
```

### 🎨 **Rich Visual Interface**
- **Syntax Highlighting**: Code displayed with proper formatting
- **Diff Views**: See exactly what changes before approval
- **Risk Assessment**: Operations rated Low/Medium/High/Critical
- **Batch Operations**: Handle multiple files efficiently

### ⚡ **Quick Examples**

```bash 
# Ask AI to create files
>>> Create a Python script to analyze CSV data
🔍 Shows preview → [Y] to approve → ✅ File created

# Request code modifications  
>>> Add error handling to this function
🔍 Shows diff view → [Y] to approve → ✅ File updated

# Batch project setup
>>> Set up a Flask web application
🔍 Shows 8 files → [A] approve all → ✅ Project ready
```

[**📖 Full Documentation**](docs/agent-approval-system.md) | [**🛡️ Security Guide**](docs/security.md)

## Supported Providers

| Provider | API Key Required | Best For |
|----------|------------------|----------|
| **Claude** | [Anthropic Console](https://console.anthropic.com/) | Complex reasoning, analysis |
| **Claude Code** | Anthropic API key | IDE integration, coding |
| **OpenAI** | [OpenAI Platform](https://platform.openai.com/) | General purpose, coding |
| **Gemini** | [Google AI Studio](https://aistudio.google.com/) | Large context, research |
| **Perplexity** | [Perplexity](https://www.perplexity.ai/) | Real-time web search |
| **xAI (Grok)** | [xAI Console](https://console.x.ai/) | Creative tasks, real-time info |
| **Mistral** | [Mistral Platform](https://mistral.ai/) | Code generation, efficiency |
| **AWS Bedrock** | [AWS Console](https://console.aws.amazon.com/bedrock/) | AWS integration |
| **Ollama** | No API key (local) | Privacy, offline use |
| **Azure OpenAI** | Azure setup required | Enterprise |
| **Vertex AI** | Google Cloud setup | Enterprise |
| **OpenRouter** | [OpenRouter](https://openrouter.ai/) | 100+ models access |
| **Cohere** | [Cohere Platform](https://cohere.com/) | Multilingual, embeddings |

## Commands

### Core Commands
| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/setup` | Run interactive setup wizard |
| `/quit` | Exit Omnimancer |
| `/clear` | Clear screen |

### Model & Provider Management
| Command | Description |
|---------|-------------|
| `/models` | List available models |
| `/providers` | Show configured providers |
| `/switch [provider] [model]` | Change provider/model |
| `/validate [provider]` | Validate provider configurations |
| `/health [provider]` | Check provider health status |
| `/repair [provider]` | Repair provider issues |
| `/diagnose [provider]` | Run diagnostic tests |

### Conversation Management
| Command | Description |
|---------|-------------|
| `/save [name]` | Save current conversation |
| `/load [name]` | Load saved conversation |
| `/list` | List saved conversations |
| `/history` | Conversation history management |

### Agent & File Operations
| Command | Description |
|---------|-------------|
| `/agent` | Enable/disable agent mode |
| `/agents` | Manage agent configurations |
| `/approvals` | View/manage file operation approvals |
| `/permissions` | Configure security permissions |

### Tool Integration
| Command | Description |
|---------|-------------|
| `/tools` | Show available tools |
| `/mcp` | MCP server management |

### Model Management
| Command | Description |
|---------|-------------|
| `/add-model` | Add custom model |
| `/remove-model` | Remove custom model |
| `/list-custom-models` | List custom models |

### System
| Command | Description |
|---------|-------------|
| `/status` | Show system status |

## Configuration

Omnimancer stores encrypted configuration in `~/.omnimancer/config.json`.

### Manual Configuration

```bash
# Add a provider
omnimancer --config

# Or edit configuration interactively
>>> /config
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
# ... then run omnimancer
```

## Local AI with Ollama

For privacy and offline use:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Download a model
ollama pull llama3.1

# Configure Omnimancer
omn
>>> /switch ollama llama3.1
```

## Tool Integration (MCP)

Enable AI tool calling for file operations, web search, and more:

```bash
# Install UV for MCP servers
curl -LsSf https://astral.sh/uv/install.sh | sh

# Check tool status
>>> /tools
>>> /mcp status
```

Popular MCP tools:
- **Filesystem**: File operations
- **Web Search**: Real-time search  
- **Git**: Repository management

## Examples

### Basic Chat
```
>>> Explain quantum computing in simple terms
🤖 Claude: Quantum computing is like having a super-powered calculator...

>>> /switch openai gpt-4o  
>>> How would you explain it differently?
🤖 GPT-4: I'd compare quantum computing to exploring a maze...
```

### Code Generation
```
>>> Write a Python function to calculate fibonacci numbers
🤖 Claude: Here's an efficient implementation using memoization:

```python
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]
```

### Model Comparison
```
>>> /switch claude claude-3-5-sonnet
>>> What's 15 * 24?
🤖 Claude: 15 × 24 = 360

>>> /switch openai gpt-4o
>>> What's 15 * 24?  
🤖 GPT-4: 15 × 24 = 360
```

## Advanced Features

- **Conversation Management**: Save/load chat history
- **Model Switching**: Compare responses between providers
- **Tool Calling**: AI can execute code, search web, manage files
- **Health Monitoring**: Provider status and diagnostics
- **Configuration Templates**: Pre-configured setups for different use cases

## Development

```bash
git clone https://gitlab.com/jite-ai/omnimancer
cd omnimancer
pip install -e ".[dev]"
pytest
```

## Troubleshooting

### Common Issues

**"No providers configured"**
```bash
omn  # Run setup wizard
>>> /setup
```

**"Invalid API key"**
- Check key format (Claude: `sk-ant-`, OpenAI: `sk-`, etc.)
- Verify key at provider's website
- Use `/validate` command to test configuration

**"Ollama connection failed"**
```bash
ollama serve  # Start Ollama server
ollama pull llama3.1  # Download a model
```

**Check system health:**
```bash
omn
>>> /health  # Check all providers
>>> /diagnose  # Run diagnostics
>>> /validate  # Validate configurations
```

**Debug mode:**
```bash
export OMNIMANCER_DEBUG=1
omn
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Links

- [GitHub Repository]https://gitlab.com/jite-ai/omnimancer)
- [Issues](https://gitlab.com/jite-ai/omnimancer/issues)
- [Documentation](https://gitlab.com/jite-ai/omnimancer/docs)

---

**Omnimancer CLI** - One tool, multiple AI providers, endless possibilities.