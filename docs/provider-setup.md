# Provider Setup Guide

This comprehensive guide covers setting up all supported AI providers in Omnimancer CLI, including detailed configuration instructions, troubleshooting tips, and best practices.

## Overview

Omnimancer CLI supports 11+ major AI providers, each with unique capabilities and setup requirements:

| Provider | Type | Tool Support | Multimodal | Best For |
|----------|------|--------------|------------|----------|
| **Claude** | Cloud API | ✅ | ✅ | Complex reasoning, analysis |
| **OpenAI** | Cloud API | ✅ | ✅ | General purpose, coding |
| **Gemini** | Cloud API | ✅ | ✅ | Large context, research |
| **Perplexity** | Cloud API | ✅ | ❌ | Real-time search, research |
| **xAI (Grok)** | Cloud API | ✅ | ✅ | Creative tasks, conversation |
| **Mistral** | Cloud API | ✅ | ❌ | Code generation, multilingual |
| **Azure OpenAI** | Cloud API | ✅ | ✅ | Enterprise, compliance |
| **Vertex AI** | Cloud API | ✅ | ✅ | Google Cloud integration |
| **AWS Bedrock** | Cloud API | ✅ | ✅ | AWS integration, multi-model |
| **OpenRouter** | Cloud API | ✅ | ✅* | Model aggregation, cost optimization |
| **Claude-code** | Local | ❌ | ❌ | Free Claude access, privacy |
| **Cohere** | Cloud API | ❌ | ❌ | Conversation, multilingual |
| **Ollama** | Local | ❌* | ✅* | Privacy, offline usage |

*Depends on specific model

## Claude (Anthropic)

### Account Setup

1. **Create Account**: Visit [Anthropic Console](https://console.anthropic.com/)
2. **Verify Email**: Complete email verification process
3. **Add Payment Method**: Required for API access beyond free tier
4. **Generate API Key**: Navigate to API Keys section

### API Key Configuration

```bash
# Interactive setup
omn  # or omnimancer, omniman
>>> /config
Choose provider: claude
Enter API key: sk-ant-api03-...
Choose model: claude-3-5-sonnet-20241022
```

### Available Models

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `claude-3-5-sonnet-20241022` | 200K | Latest, most capable | Complex analysis, coding |
| `claude-3-5-haiku-20241022` | 200K | Fast, efficient | Quick tasks, summaries |
| `claude-3-opus-20240229` | 200K | Most powerful | Research, creative writing |
| `claude-3-sonnet-20240229` | 200K | Balanced | General purpose |
| `claude-3-haiku-20240307` | 200K | Speed optimized | Simple queries |

### Configuration Options

```json
{
  "providers": {
    "claude": {
      "api_key": "sk-ant-api03-...",
      "model": "claude-3-5-sonnet-20241022",
      "max_tokens": 4096,
      "temperature": 0.7
    }
  }
}
```

### Pricing (as of 2024)

- **Claude 3.5 Sonnet**: $3.00 / 1M input tokens, $15.00 / 1M output tokens
- **Claude 3.5 Haiku**: $0.25 / 1M input tokens, $1.25 / 1M output tokens
- **Claude 3 Opus**: $15.00 / 1M input tokens, $75.00 / 1M output tokens

### Troubleshooting

**"Invalid API key"**
- Verify key format starts with `sk-ant-api03-`
- Check key hasn't expired in console
- Ensure billing is set up

**"Rate limit exceeded"**
- Claude has usage limits based on tier
- Upgrade account tier for higher limits
- Implement retry logic (built into Omnimancer)

## OpenAI

### Account Setup

1. **Create Account**: Visit [OpenAI Platform](https://platform.openai.com/)
2. **Verify Phone**: Phone verification required
3. **Add Payment**: Credit card required for API access
4. **Generate API Key**: Create in API Keys section

### API Key Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: openai
Enter API key: sk-...
Choose model: gpt-4o
Organization ID (optional): org-...
```

### Available Models

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `gpt-4o` | 128K | Latest, multimodal | General purpose, vision |
| `gpt-4o-mini` | 128K | Fast, cost-effective | Quick tasks, summaries |
| `gpt-4-turbo` | 128K | Enhanced GPT-4 | Complex reasoning |
| `gpt-4` | 8K | Original GPT-4 | High-quality responses |
| `gpt-3.5-turbo` | 16K | Fast, affordable | Basic conversations |

### Configuration Options

```json
{
  "providers": {
    "openai": {
      "api_key": "sk-...",
      "model": "gpt-4o",
      "max_tokens": 4096,
      "temperature": 0.7,
      "organization": "org-...",
      "base_url": "https://api.openai.com/v1"
    }
  }
}
```

### Pricing (as of 2024)

- **GPT-4o**: $5.00 / 1M input tokens, $15.00 / 1M output tokens
- **GPT-4o Mini**: $0.15 / 1M input tokens, $0.60 / 1M output tokens
- **GPT-4 Turbo**: $10.00 / 1M input tokens, $30.00 / 1M output tokens

### Troubleshooting

**"Insufficient quota"**
- Add payment method to account
- Check billing limits and usage
- Upgrade to paid tier

**"Model not found"**
- Verify model name spelling
- Check if model is available in your region
- Some models require special access

## Google Gemini

### Account Setup

1. **Google Account**: Use existing or create new Google account
2. **AI Studio Access**: Visit [Google AI Studio](https://aistudio.google.com/)
3. **Generate API Key**: Create key in API Keys section
4. **Enable Billing**: For higher usage limits (optional)

### API Key Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: gemini
Enter API key: AIza...
Choose model: gemini-1.5-pro-latest
```

### Available Models

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `gemini-1.5-pro-latest` | 2M | Massive context | Long documents, research |
| `gemini-1.5-flash-latest` | 1M | Fast, efficient | General purpose |
| `gemini-1.5-flash-8b-latest` | 1M | Lightweight | Simple tasks |
| `gemini-1.0-pro` | 32K | Original model | Basic conversations |

### Configuration Options

```json
{
  "providers": {
    "gemini": {
      "api_key": "AIza...",
      "model": "gemini-1.5-pro-latest",
      "max_tokens": 8192,
      "temperature": 0.7,
      "project_id": "your-project-id"
    }
  }
}
```

### Pricing (as of 2024)

- **Gemini 1.5 Pro**: $3.50 / 1M input tokens, $10.50 / 1M output tokens
- **Gemini 1.5 Flash**: $0.35 / 1M input tokens, $1.05 / 1M output tokens
- **Free Tier**: 15 requests/minute, 1500 requests/day

### Troubleshooting

**"API key not valid"**
- Verify key format starts with `AIza`
- Check key restrictions in AI Studio
- Ensure API is enabled for your project

**"Quota exceeded"**
- Check daily/monthly quotas
- Enable billing for higher limits
- Monitor usage in Google Cloud Console

## Cohere

### Account Setup

1. **Create Account**: Visit [Cohere Platform](https://cohere.com/)
2. **Email Verification**: Complete verification process
3. **Generate API Key**: Create in Dashboard → API Keys
4. **Choose Plan**: Free tier available, paid plans for production

### API Key Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: cohere
Enter API key: co-...
Choose model: command-r-plus
```

### Available Models

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `command-r-plus` | 128K | Most advanced | Complex reasoning |
| `command-r` | 128K | Balanced | General conversations |
| `command-light` | 4K | Fast, affordable | Simple queries |
| `command` | 4K | Original | Basic conversations |

### Configuration Options

```json
{
  "providers": {
    "cohere": {
      "api_key": "co-...",
      "model": "command-r-plus",
      "max_tokens": 4096,
      "temperature": 0.7
    }
  }
}
```

### Pricing (as of 2024)

- **Command R+**: $3.00 / 1M input tokens, $15.00 / 1M output tokens
- **Command R**: $0.50 / 1M input tokens, $1.50 / 1M output tokens
- **Command Light**: $0.30 / 1M input tokens, $0.60 / 1M output tokens

### Troubleshooting

**"Invalid API key"**
- Verify key format starts with `co-`
- Check key status in dashboard
- Regenerate key if needed

**"Model not available"**
- Some models require special access
- Check model availability in your region
- Contact Cohere support for access

## Perplexity AI

### Account Setup

1. **Create Account**: Visit [Perplexity AI](https://www.perplexity.ai/)
2. **Verify Email**: Complete email verification process
3. **Subscribe to Pro**: Required for API access
4. **Generate API Key**: Navigate to Settings → API Keys

### API Key Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: perplexity
Enter API key: pplx-...
Choose model: sonar-pro
```

### Available Models

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `sonar-pro` | 127K | Advanced reasoning with web search | Research, analysis |
| `sonar` | 127K | Balanced performance with search | General queries |
| `llama-3.1-sonar-small-128k-online` | 128K | Cost-effective with search | Simple research |
| `llama-3.1-sonar-large-128k-online` | 128K | High performance with search | Complex research |

### Configuration Options

```json
{
  "providers": {
    "perplexity": {
      "api_key": "pplx-...",
      "model": "sonar-pro",
      "max_tokens": 4096,
      "temperature": 0.7,
      "search_enabled": true,
      "search_recency_filter": "month"
    }
  }
}
```

### Pricing (as of 2024)

- **Sonar Pro**: $3.00 / 1M input tokens, $15.00 / 1M output tokens
- **Sonar**: $1.00 / 1M input tokens, $3.00 / 1M output tokens
- **Llama Sonar Models**: $0.20 / 1M input tokens, $0.20 / 1M output tokens

### Troubleshooting

**"Invalid API key"**
- Verify key format starts with `pplx-`
- Ensure Pro subscription is active
- Check key permissions in settings

**"Search failed"**
- Check internet connectivity
- Verify search is enabled for model
- Try different search terms

## xAI (Grok)

### Account Setup

1. **Create Account**: Visit [xAI Console](https://console.x.ai/)
2. **Verify Account**: Complete verification process
3. **Add Payment Method**: Required for API access
4. **Generate API Key**: Create in API Keys section

### API Key Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: xai
Enter API key: xai-...
Choose model: grok-3
```

### Available Models

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `grok-3` | 131K | Latest with enhanced reasoning | Complex analysis, creative tasks |
| `grok-3-fast` | 131K | Faster responses | Quick queries, conversations |
| `grok-2` | 131K | Previous generation | General purpose |
| `grok-beta` | 131K | Beta features | Experimental use |

### Configuration Options

```json
{
  "providers": {
    "xai": {
      "api_key": "xai-...",
      "model": "grok-3",
      "max_tokens": 4096,
      "temperature": 0.7,
      "grok_mode": "balanced"
    }
  }
}
```

### Pricing (as of 2024)

- **Grok-3**: $3.00 / 1M input tokens, $15.00 / 1M output tokens
- **Grok-3 Fast**: $1.50 / 1M input tokens, $7.50 / 1M output tokens
- **Grok-2**: $2.00 / 1M input tokens, $10.00 / 1M output tokens

### Troubleshooting

**"Invalid API key"**
- Verify key format starts with `xai-`
- Check key hasn't expired
- Ensure billing is configured

**"Model not available"**
- Some models require special access
- Check model availability in console
- Contact xAI support for beta access

## Mistral AI

### Account Setup

1. **Create Account**: Visit [Mistral Console](https://console.mistral.ai/)
2. **Verify Email**: Complete email verification
3. **Add Payment Method**: Required for production usage
4. **Generate API Key**: Create in API Keys section

### API Key Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: mistral
Enter API key: mistral-...
Choose model: mistral-large-latest
```

### Available Models

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `mistral-large-latest` | 128K | Most capable | Complex reasoning, analysis |
| `mistral-small-latest` | 128K | Fast, efficient | General conversations |
| `codestral-latest` | 32K | Code-specialized | Programming, debugging |
| `mistral-nemo` | 128K | Lightweight | Simple tasks |

### Configuration Options

```json
{
  "providers": {
    "mistral": {
      "api_key": "mistral-...",
      "model": "mistral-large-latest",
      "max_tokens": 4096,
      "temperature": 0.7,
      "safe_prompt": true
    }
  }
}
```

### Pricing (as of 2024)

- **Mistral Large**: $4.00 / 1M input tokens, $12.00 / 1M output tokens
- **Mistral Small**: $1.00 / 1M input tokens, $3.00 / 1M output tokens
- **Codestral**: $1.00 / 1M input tokens, $3.00 / 1M output tokens

### Troubleshooting

**"Invalid API key"**
- Verify key format starts with `mistral-`
- Check key status in console
- Ensure account is verified

**"Safety filter triggered"**
- Rephrase potentially sensitive content
- Use safe_prompt: false (if appropriate)
- Check content guidelines

## Azure OpenAI

### Account Setup

1. **Azure Subscription**: Create or use existing Azure subscription
2. **Create Resource**: Create Azure OpenAI resource in Azure Portal
3. **Request Access**: Apply for Azure OpenAI access (if needed)
4. **Deploy Models**: Deploy models in Azure OpenAI Studio

### API Key Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: azure
Enter API key: your-azure-key
Enter endpoint: https://your-resource.openai.azure.com/
Enter deployment name: your-deployment
```

### Available Models (Deployed)

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `gpt-4o` | 128K | Latest GPT-4 Omni | General purpose, multimodal |
| `gpt-4-turbo` | 128K | Enhanced GPT-4 | Complex reasoning |
| `gpt-35-turbo` | 16K | Cost-effective | Basic conversations |
| `dall-e-3` | N/A | Image generation | Visual content creation |

### Configuration Options

```json
{
  "providers": {
    "azure": {
      "api_key": "your-azure-key",
      "azure_endpoint": "https://your-resource.openai.azure.com/",
      "azure_deployment": "your-deployment-name",
      "api_version": "2024-02-01",
      "model": "gpt-4o",
      "max_tokens": 4096,
      "temperature": 0.7
    }
  }
}
```

### Pricing

- Pricing varies by region and deployment
- Check Azure OpenAI pricing page for current rates
- Enterprise agreements may have different pricing

### Troubleshooting

**"Deployment not found"**
- Verify deployment name in Azure OpenAI Studio
- Ensure model is deployed and running
- Check deployment status

**"Authentication failed"**
- Verify API key and endpoint URL
- Check Azure resource permissions
- Ensure subscription is active

## Google Vertex AI

### Account Setup

1. **Google Cloud Account**: Create or use existing GCP account
2. **Create Project**: Create new GCP project or use existing
3. **Enable APIs**: Enable Vertex AI API
4. **Set up Authentication**: Configure service account or gcloud auth

### Authentication Setup

#### Option 1: Service Account (Recommended)
```bash
# Create service account
gcloud iam service-accounts create omnimancer-vertex

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:omnimancer-vertex@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create ~/vertex-key.json \
    --iam-account=omnimancer-vertex@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

#### Option 2: Application Default Credentials
```bash
# Install gcloud CLI
# Authenticate
gcloud auth application-default login
```

### API Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: vertex
Enter project ID: your-project-id
Enter location: us-central1
Enter credentials path: /path/to/vertex-key.json
```

### Available Models

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `gemini-1.5-pro` | 2M | Most capable | Long documents, research |
| `gemini-1.5-flash` | 1M | Fast, efficient | General purpose |
| `text-bison` | 8K | PaLM-based | Text generation |
| `chat-bison` | 8K | PaLM-based | Conversations |

### Configuration Options

```json
{
  "providers": {
    "vertex": {
      "vertex_project": "your-project-id",
      "vertex_location": "us-central1",
      "vertex_credentials_path": "/path/to/vertex-key.json",
      "model": "gemini-1.5-pro",
      "max_tokens": 8192,
      "temperature": 0.7
    }
  }
}
```

### Pricing (as of 2024)

- **Gemini 1.5 Pro**: $3.50 / 1M input tokens, $10.50 / 1M output tokens
- **Gemini 1.5 Flash**: $0.35 / 1M input tokens, $1.05 / 1M output tokens
- **Text/Chat Bison**: $1.00 / 1M input tokens, $2.00 / 1M output tokens

### Troubleshooting

**"Authentication failed"**
- Check service account permissions
- Verify credentials file path
- Ensure Vertex AI API is enabled

**"Project not found"**
- Verify project ID spelling
- Check project access permissions
- Ensure project exists and is active

## AWS Bedrock

### Account Setup

1. **AWS Account**: Create or use existing AWS account
2. **Enable Bedrock**: Enable AWS Bedrock in your region
3. **Request Model Access**: Request access to models in Bedrock console
4. **Configure Credentials**: Set up AWS credentials

### Credentials Setup

#### Option 1: AWS CLI
```bash
# Install AWS CLI
# Configure credentials
aws configure
```

#### Option 2: Environment Variables
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

#### Option 3: IAM Roles (EC2/Lambda)
- Attach appropriate IAM role to your compute resource

### API Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: bedrock
Enter AWS region: us-east-1
Choose model: anthropic.claude-3-5-sonnet-20241022-v2:0
```

### Available Models

| Model | Context | Strengths | Use Cases |
|-------|---------|-----------|-----------|
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | 200K | Latest Claude | Complex reasoning |
| `anthropic.claude-3-haiku-20240307-v1:0` | 200K | Fast Claude | Quick tasks |
| `amazon.titan-text-express-v1` | 8K | Amazon's model | General text |
| `ai21.j2-ultra-v1` | 8K | AI21 Labs | Text generation |

### Configuration Options

```json
{
  "providers": {
    "bedrock": {
      "aws_region": "us-east-1",
      "aws_access_key_id": "your-access-key",
      "aws_secret_access_key": "your-secret-key",
      "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
      "max_tokens": 4096,
      "temperature": 0.7
    }
  }
}
```

### Pricing

- Pay-per-use pricing varies by model
- Check AWS Bedrock pricing page for current rates
- No upfront costs or minimum fees

### Troubleshooting

**"Model access denied"**
- Request model access in Bedrock console
- Wait for access approval (can take time)
- Check model availability in your region

**"Authentication failed"**
- Verify AWS credentials
- Check IAM permissions
- Ensure Bedrock service permissions

## OpenRouter

### Account Setup

1. **Create Account**: Visit [OpenRouter](https://openrouter.ai/)
2. **Verify Email**: Complete email verification
3. **Add Credits**: Add credits to your account
4. **Generate API Key**: Create API key in Keys section

### API Key Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: openrouter
Enter API key: sk-or-...
Choose model: anthropic/claude-3.5-sonnet
```

### Available Models (Examples)

| Model | Provider | Context | Use Cases |
|-------|----------|---------|-----------|
| `anthropic/claude-3.5-sonnet` | Anthropic | 200K | Complex reasoning |
| `openai/gpt-4o` | OpenAI | 128K | General purpose |
| `google/gemini-pro` | Google | 32K | Large context |
| `meta-llama/llama-3.1-405b` | Meta | 32K | Open source |

### Configuration Options

```json
{
  "providers": {
    "openrouter": {
      "api_key": "sk-or-...",
      "model": "anthropic/claude-3.5-sonnet",
      "max_tokens": 4096,
      "temperature": 0.7,
      "openrouter_referrer": "https://your-app.com",
      "openrouter_title": "Your App Name"
    }
  }
}
```

### Pricing

- Varies by model and provider
- Generally competitive with direct provider pricing
- Check OpenRouter pricing page for current rates

### Troubleshooting

**"Invalid API key"**
- Verify key format starts with `sk-or-`
- Check key status in dashboard
- Ensure sufficient credits

**"Model not available"**
- Check model name spelling
- Verify model is supported
- Some models may have usage restrictions

## Claude-code (Local)

### Installation

#### macOS
```bash
# Install via Homebrew (if available)
brew install claude-code

# Or download from GitHub releases
# https://github.com/anthropics/claude-code/releases
```

#### Linux
```bash
# Download binary from GitHub releases
wget https://github.com/anthropics/claude-code/releases/latest/download/claude-code-linux
chmod +x claude-code-linux
sudo mv claude-code-linux /usr/local/bin/claude-code
```

#### Windows
```bash
# Download .exe from GitHub releases
# Add to PATH environment variable
```

### Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: claude-code
Choose mode: opus
```

### Available Modes

| Mode | Quality | Speed | Use Cases |
|------|---------|-------|-----------|
| `opus` | Highest | Slower | Complex analysis, research |
| `sonnet` | High | Balanced | General programming |
| `haiku` | Good | Fastest | Quick queries, simple tasks |

### Configuration Options

```json
{
  "providers": {
    "claude-code": {
      "claude_code_mode": "opus",
      "max_tokens": 4096,
      "temperature": 0.7
    }
  }
}
```

### Troubleshooting

**"claude-code not found"**
- Ensure claude-code is installed
- Check PATH environment variable
- Verify binary permissions

**"Model download failed"**
- Check internet connection
- Ensure sufficient disk space
- Try different mode (smaller model)

## Ollama (Local AI)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB (7B models) | 16GB+ (13B+) |
| **Storage** | 10GB free | 50GB+ |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | None | NVIDIA with CUDA |

### Installation

#### macOS
```bash
# Option 1: Download from website
# Visit https://ollama.ai and download installer

# Option 2: Homebrew
brew install ollama
```

#### Linux
```bash
# Official installer
curl -fsSL https://ollama.ai/install.sh | sh

# Manual installation
# Download binary from GitHub releases
```

#### Windows
```bash
# Download installer from https://ollama.ai
# Run the .exe installer
```

### Server Setup

```bash
# Start Ollama server
ollama serve

# Verify server is running
curl http://localhost:11434/api/version
```

### Model Management

```bash
# List available models online
ollama list

# Download models
ollama pull llama3.1          # Latest Llama 3.1 8B
ollama pull llama3.1:70b      # Larger 70B version
ollama pull codellama         # Code-specialized
ollama pull mistral           # Efficient general-purpose
ollama pull llava             # Vision-language model

# Remove models
ollama rm model-name

# Show model info
ollama show llama3.1
```

### Omnimancer Configuration

```bash
# Interactive setup
omnimancer
>>> /config
Choose provider: ollama
Server URL: http://localhost:11434
Choose model: llama3.1
```

### Popular Models

#### General Purpose
- **llama3.1** (8B) - Latest Meta model, excellent performance
- **llama3.1:70b** (70B) - Larger, more capable version
- **mistral** (7B) - Efficient, good reasoning
- **phi3** (3.8B) - Microsoft's compact model

#### Code-Specialized
- **codellama** (7B-34B) - Meta's code-focused model
- **deepseek-coder** (6.7B) - Specialized coding assistant
- **starcoder** (15B) - Code generation and completion

#### Multimodal
- **llava** (7B) - Vision-language model
- **bakllava** (7B) - Enhanced vision capabilities

### Configuration Options

```json
{
  "providers": {
    "ollama": {
      "base_url": "http://localhost:11434",
      "model": "llama3.1",
      "temperature": 0.7,
      "num_ctx": 4096,
      "num_predict": 2048
    }
  }
}
```

### Performance Optimization

#### GPU Acceleration
```bash
# Check GPU support
nvidia-smi

# Ollama automatically uses GPU if available
# Monitor GPU usage during inference
```

#### Memory Management
```bash
# Set memory limits (optional)
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=5m
export OLLAMA_MAX_LOADED_MODELS=1
```

### Troubleshooting

**"Connection refused"**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start server if not running
ollama serve

# Check port availability
netstat -an | grep 11434
```

**"Model not found"**
```bash
# List installed models
ollama list

# Download missing model
ollama pull model-name
```

**"Out of memory"**
- Use smaller models (7B instead of 70B)
- Close other applications
- Increase system swap space
- Consider cloud deployment

**"Slow performance"**
- Enable GPU acceleration
- Use smaller models for faster responses
- Increase system RAM
- Use SSD storage

## Multi-Provider Configuration

### Configuration File Structure

```json
{
  "default_provider": "claude",
  "providers": {
    "claude": {
      "api_key": "sk-ant-api03-...",
      "model": "claude-3-5-sonnet-20241022"
    },
    "openai": {
      "api_key": "sk-...",
      "model": "gpt-4o"
    },
    "gemini": {
      "api_key": "AIza...",
      "model": "gemini-1.5-pro-latest"
    },
    "cohere": {
      "api_key": "co-...",
      "model": "command-r-plus"
    },
    "ollama": {
      "base_url": "http://localhost:11434",
      "model": "llama3.1"
    }
  }
}
```

### Provider Switching

```bash
# Switch providers during conversation
/switch claude                    # Use default Claude model
/switch openai gpt-4o            # Specific OpenAI model
/switch ollama llama3.1          # Local Ollama model
/switch gemini                   # Use default Gemini model
```

### Best Practices

1. **Start with Free Tiers**: Test providers before committing to paid plans
2. **Monitor Usage**: Track token consumption and costs
3. **Use Appropriate Models**: Match model capabilities to task requirements
4. **Backup Configurations**: Keep configuration backups
5. **Security**: Never share API keys, use environment variables in production

## Environment Variables

For production deployments, use environment variables:

```bash
# Claude
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_ORG_ID="org-..."

# Gemini
export GOOGLE_API_KEY="AIza..."

# Cohere
export COHERE_API_KEY="co-..."

# Ollama
export OLLAMA_HOST="http://localhost:11434"
```

## Security Considerations

1. **API Key Storage**: Omnimancer encrypts keys locally
2. **Network Security**: All API calls use HTTPS
3. **Local Models**: Ollama provides complete privacy
4. **Key Rotation**: Regularly rotate API keys
5. **Access Control**: Limit API key permissions where possible

## Getting Help

- **Provider Documentation**: Check official provider docs
- **Omnimancer Issues**: [GitHub Issues](https://github.com/omnimancer-cli/omnimancer/issues)
- **Community Support**: [GitHub Discussions](https://github.com/omnimancer-cli/omnimancer/discussions)
- **Provider Support**: Contact provider support teams directly