"""Core data models and interfaces for Omnimancer CLI."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from ..utils.errors import MCPError


class MessageRole(Enum):
    """Message roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Represents a single chat message."""

    role: MessageRole
    content: str
    timestamp: datetime
    model_used: str


@dataclass
class ChatContext:
    """Maintains conversation context."""

    messages: List[ChatMessage]
    current_model: str
    session_id: str
    max_context_length: int = 4000

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the context."""
        self.messages.append(message)

        # Trim context if it gets too long
        if len(self.messages) > self.max_context_length:
            # Keep system messages and trim older user/assistant messages
            system_messages = [
                msg for msg in self.messages if msg.role == MessageRole.SYSTEM
            ]
            other_messages = [
                msg for msg in self.messages if msg.role != MessageRole.SYSTEM
            ]

            # Keep the most recent messages
            keep_count = self.max_context_length - len(system_messages)
            if keep_count > 0:
                other_messages = other_messages[-keep_count:]

            self.messages = system_messages + other_messages

    def clear(self) -> None:
        """Clear all messages from the context."""
        self.messages.clear()

    def get_context_for_api(self) -> List[Dict[str, str]]:
        """Get messages formatted for API calls."""
        return [
            {"role": msg.role.value, "content": msg.content} for msg in self.messages
        ]


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by AI providers."""

    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: Optional[str] = None
    auto_approved: bool = False


@dataclass
class ToolCall:
    """Represents a tool call made by an AI provider."""

    name: str
    arguments: Dict[str, Any]
    server_name: Optional[str] = None


@dataclass
class ToolResult:
    """Result of a tool execution."""

    content: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Response from an AI provider."""

    content: str
    model_used: str
    tokens_used: int
    cost_estimate: Optional[float] = None
    timestamp: Optional[datetime] = None
    error: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None

    @property
    def is_success(self) -> bool:
        """Check if the response was successful."""
        return self.error is None


@dataclass
class ModelInfo:
    """Information about an AI model."""

    name: str
    provider: str
    description: str
    max_tokens: int
    cost_per_token: float
    available: bool
    supports_tools: bool = False
    supports_multimodal: bool = False
    latest_version: bool = False
    deprecated: bool = False


@dataclass
class EnhancedModelInfo:
    """Extended model information with SWE scores and pricing."""

    name: str
    provider: str
    description: str
    max_tokens: int
    cost_per_million_input: float
    cost_per_million_output: float
    swe_score: Optional[float] = None
    swe_rating: Optional[str] = None  # ★★★, ★★☆, ★☆☆
    available: bool = True
    supports_tools: bool = False
    supports_multimodal: bool = False
    latest_version: bool = False
    deprecated: bool = False
    release_date: Optional[datetime] = None
    context_window: int = 4096
    is_free: bool = False

    def get_cost_display(self) -> str:
        """Get formatted cost display string."""
        if self.is_free:
            return "Free"
        return f"${self.cost_per_million_input:.2f} in, ${self.cost_per_million_output:.2f} out"

    def get_swe_display(self) -> str:
        """Get formatted SWE score display."""
        if self.swe_score is None:
            return "N/A"
        return f"{self.swe_score:.1f}% {self.swe_rating or ''}"

    def get_swe_rating(self) -> str:
        """Calculate SWE rating based on score."""
        if self.swe_score is None:
            return ""
        elif self.swe_score >= 60:
            return "★★★"
        elif self.swe_score >= 40:
            return "★★☆"
        else:
            return "★☆☆"

    def get_cost_tier(self) -> str:
        """Get cost tier indicator."""
        if self.is_free:
            return "Free"

        # Calculate average cost per million tokens
        avg_cost = (self.cost_per_million_input + self.cost_per_million_output) / 2

        if avg_cost <= 1.0:
            return "💰"
        elif avg_cost <= 10.0:
            return "💰💰"
        else:
            return "💰💰💰"

    def validate_pricing(self) -> bool:
        """Validate pricing data."""
        if self.is_free:
            return (
                self.cost_per_million_input == 0 and self.cost_per_million_output == 0
            )
        return self.cost_per_million_input >= 0 and self.cost_per_million_output >= 0

    def validate_swe_score(self) -> bool:
        """Validate SWE score data."""
        if self.swe_score is None:
            return True
        return 0 <= self.swe_score <= 100

    def update_swe_rating(self) -> None:
        """Update SWE rating based on current score."""
        self.swe_rating = self.get_swe_rating()

    def to_model_info(self) -> ModelInfo:
        """Convert to legacy ModelInfo for backward compatibility."""
        # Calculate legacy cost_per_token (average of input/output per million / 1M)
        if self.is_free:
            cost_per_token = 0.0
        else:
            cost_per_token = (
                (self.cost_per_million_input + self.cost_per_million_output)
                / 2
                / 1_000_000
            )

        return ModelInfo(
            name=self.name,
            provider=self.provider,
            description=self.description,
            max_tokens=self.max_tokens,
            cost_per_token=cost_per_token,
            available=self.available,
            supports_tools=self.supports_tools,
            supports_multimodal=self.supports_multimodal,
            latest_version=self.latest_version,
            deprecated=self.deprecated,
        )

    @classmethod
    def from_model_info(cls, model_info: ModelInfo, **kwargs) -> "EnhancedModelInfo":
        """Create EnhancedModelInfo from legacy ModelInfo."""
        # Convert legacy cost_per_token to per-million costs
        cost_per_million = model_info.cost_per_token * 1_000_000

        return cls(
            name=model_info.name,
            provider=model_info.provider,
            description=model_info.description,
            max_tokens=model_info.max_tokens,
            cost_per_million_input=cost_per_million,
            cost_per_million_output=cost_per_million,
            available=model_info.available,
            supports_tools=model_info.supports_tools,
            supports_multimodal=model_info.supports_multimodal,
            latest_version=model_info.latest_version,
            deprecated=model_info.deprecated,
            context_window=model_info.max_tokens,
            is_free=model_info.cost_per_token == 0,
            **kwargs,
        )


class ChatSettings(BaseModel):
    """Chat-specific settings."""

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    context_length: int = 4000
    save_history: bool = True


class ProviderConfig(BaseModel):
    """Configuration for a specific AI provider."""

    api_key: Optional[str] = (
        None  # Optional for providers like Ollama that don't need keys
    )
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    # Provider-specific settings
    base_url: Optional[str] = None  # For custom endpoints (OpenAI-compatible, Ollama)
    organization: Optional[str] = None  # For OpenAI organizations
    project_id: Optional[str] = None  # For Google Cloud projects
    timeout: Optional[float] = None  # Request timeout in seconds

    # Advanced settings
    top_p: Optional[float] = None  # Nucleus sampling parameter
    frequency_penalty: Optional[float] = None  # Frequency penalty (OpenAI)
    presence_penalty: Optional[float] = None  # Presence penalty (OpenAI)

    # Gemini-specific settings
    safety_settings: Optional[Dict[str, str]] = None  # Gemini safety settings
    generation_config: Optional[Dict[str, Any]] = None  # Gemini generation config
    vertex_ai_project: Optional[str] = None  # Vertex AI project ID
    vertex_ai_location: Optional[str] = None  # Vertex AI location
    service_account_path: Optional[str] = None  # Path to service account JSON

    # Cohere-specific settings
    max_input_tokens: Optional[int] = None  # Cohere input token limit
    connectors: Optional[List[Dict[str, Any]]] = None  # Cohere connectors for RAG
    chat_history: Optional[List[Dict[str, str]]] = None  # Cohere chat history
    documents: Optional[List[Dict[str, Any]]] = None  # Cohere documents for RAG
    preamble: Optional[str] = None  # Cohere preamble for conversation context
    k: Optional[int] = None  # Cohere k parameter for RAG

    # Ollama-specific settings
    num_predict: Optional[int] = None  # Number of tokens to predict
    num_ctx: Optional[int] = None  # Context window size
    repeat_penalty: Optional[float] = None  # Repetition penalty
    seed: Optional[int] = None  # Random seed for reproducibility
    stop: Optional[List[str]] = None  # Stop sequences
    mirostat: Optional[int] = None  # Mirostat sampling
    mirostat_eta: Optional[float] = None  # Mirostat learning rate
    mirostat_tau: Optional[float] = None  # Mirostat target entropy
    num_gpu: Optional[int] = None  # Number of GPU layers
    num_thread: Optional[int] = None  # Number of threads

    # Claude-specific settings
    anthropic_version: Optional[str] = None  # Anthropic API version
    system_prompt: Optional[str] = None  # System prompt for Claude

    # OpenAI-specific settings
    response_format: Optional[Dict[str, str]] = (
        None  # Response format (e.g., json_object)
    )
    logit_bias: Optional[Dict[int, float]] = None  # Token biasing

    # Provider capabilities
    supports_tools: bool = False  # Whether provider supports function calling
    supports_multimodal: bool = False  # Whether provider supports images/files
    supports_streaming: bool = True  # Whether provider supports streaming responses
    supports_system_messages: bool = True  # Whether provider supports system messages
    supports_function_calling: bool = False  # Alias for supports_tools for clarity
    supports_vision: bool = False  # Whether provider supports image analysis
    supports_json_mode: bool = False  # Whether provider supports JSON mode

    # Configuration metadata
    provider_type: Optional[str] = None  # Provider type identifier
    config_version: str = "2.0"  # Configuration schema version
    enabled: bool = True  # Whether this provider is enabled
    priority: int = 0  # Provider priority for fallback ordering

    # Rate limiting and retry settings
    rate_limit_requests_per_minute: Optional[int] = None
    rate_limit_tokens_per_minute: Optional[int] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True

    # Health check settings
    health_check_enabled: bool = True
    health_check_interval: int = 300  # seconds
    health_check_timeout: float = 10.0  # seconds

    # Custom headers and authentication
    custom_headers: Optional[Dict[str, str]] = None
    auth_type: str = "api_key"  # api_key, bearer, oauth, service_account, none
    oauth_config: Optional[Dict[str, Any]] = None  # OAuth configuration

    # Perplexity-specific settings
    search_enabled: Optional[bool] = None  # Enable web search for Perplexity
    search_recency_filter: Optional[str] = None  # hour, day, week, month, year
    return_citations: Optional[bool] = None  # Return citations in response
    return_images: Optional[bool] = None  # Return images in response
    return_related_questions: Optional[bool] = None  # Return related questions

    # xAI-specific settings
    grok_mode: Optional[str] = None  # balanced, creative, precise
    enable_web_search: Optional[bool] = None  # Enable web search for Grok
    enable_real_time: Optional[bool] = None  # Enable real-time information

    # Mistral-specific settings
    safe_prompt: Optional[bool] = None  # Enable safe prompt mode
    random_seed: Optional[int] = None  # Random seed for reproducibility
    top_k: Optional[int] = None  # Top-k sampling parameter

    # Azure-specific settings
    azure_endpoint: Optional[str] = None  # Azure OpenAI endpoint
    azure_deployment: Optional[str] = None  # Azure deployment name
    api_version: Optional[str] = None  # Azure API version

    # Vertex AI-specific settings
    vertex_project: Optional[str] = None  # Google Cloud project ID
    vertex_location: Optional[str] = None  # Vertex AI location
    vertex_credentials_path: Optional[str] = None  # Path to service account JSON

    # AWS Bedrock-specific settings
    aws_region: Optional[str] = None  # AWS region
    aws_access_key_id: Optional[str] = None  # AWS access key ID
    aws_secret_access_key: Optional[str] = None  # AWS secret access key
    aws_session_token: Optional[str] = None  # AWS session token

    # OpenRouter-specific settings
    openrouter_referrer: Optional[str] = None  # HTTP referrer for OpenRouter
    openrouter_title: Optional[str] = None  # Application title for OpenRouter
    enable_fallback: Optional[bool] = None  # Enable model fallback
    max_cost_per_token: Optional[float] = None  # Maximum cost per token
    prefer_cheaper_models: Optional[bool] = None  # Prefer cheaper models

    # Claude-code specific settings
    claude_code_mode: Optional[str] = None  # opus, sonnet
    claude_code_path: Optional[str] = None  # Path to claude-code executable
    working_directory: Optional[str] = None  # Working directory for claude-code

    # Provider-specific extra settings (for extensibility)
    extra_settings: Optional[Dict[str, Any]] = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("top_p must be between 0 and 1")
        return v

    @field_validator("frequency_penalty")
    @classmethod
    def validate_frequency_penalty(cls, v):
        if v is not None and (v < -2 or v > 2):
            raise ValueError("frequency_penalty must be between -2 and 2")
        return v

    @field_validator("presence_penalty")
    @classmethod
    def validate_presence_penalty(cls, v):
        if v is not None and (v < -2 or v > 2):
            raise ValueError("presence_penalty must be between -2 and 2")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("timeout must be positive")
        return v

    @field_validator("repeat_penalty")
    @classmethod
    def validate_repeat_penalty(cls, v):
        if v is not None and (v < 0.1 or v > 2.0):
            raise ValueError("repeat_penalty must be between 0.1 and 2.0")
        return v

    @field_validator("max_input_tokens")
    @classmethod
    def validate_max_input_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_input_tokens must be positive")
        return v

    @field_validator("num_predict")
    @classmethod
    def validate_num_predict(cls, v):
        if v is not None and v <= 0:
            raise ValueError("num_predict must be positive")
        return v

    @field_validator("num_ctx")
    @classmethod
    def validate_num_ctx(cls, v):
        if v is not None and v <= 0:
            raise ValueError("num_ctx must be positive")
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v):
        if v is not None and (v < 0 or v > 2**32 - 1):
            raise ValueError("seed must be between 0 and 2^32-1")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        if v < 0:
            raise ValueError("priority must be non-negative")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v

    @field_validator("retry_delay")
    @classmethod
    def validate_retry_delay(cls, v):
        if v < 0:
            raise ValueError("retry_delay must be non-negative")
        return v

    @field_validator("health_check_interval")
    @classmethod
    def validate_health_check_interval(cls, v):
        if v <= 0:
            raise ValueError("health_check_interval must be positive")
        return v

    @field_validator("health_check_timeout")
    @classmethod
    def validate_health_check_timeout(cls, v):
        if v <= 0:
            raise ValueError("health_check_timeout must be positive")
        return v

    @field_validator("rate_limit_requests_per_minute")
    @classmethod
    def validate_rate_limit_requests(cls, v):
        if v is not None and v <= 0:
            raise ValueError("rate_limit_requests_per_minute must be positive")
        return v

    @field_validator("rate_limit_tokens_per_minute")
    @classmethod
    def validate_rate_limit_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValueError("rate_limit_tokens_per_minute must be positive")
        return v

    @field_validator("auth_type")
    @classmethod
    def validate_auth_type(cls, v):
        valid_types = ["api_key", "bearer", "oauth", "service_account", "none"]
        if v not in valid_types:
            raise ValueError(f'auth_type must be one of: {", ".join(valid_types)}')
        return v

    @field_validator("mirostat")
    @classmethod
    def validate_mirostat(cls, v):
        if v is not None and v not in [0, 1, 2]:
            raise ValueError("mirostat must be 0, 1, or 2")
        return v

    @field_validator("mirostat_eta")
    @classmethod
    def validate_mirostat_eta(cls, v):
        if v is not None and (v <= 0 or v > 1):
            raise ValueError("mirostat_eta must be between 0 and 1")
        return v

    @field_validator("mirostat_tau")
    @classmethod
    def validate_mirostat_tau(cls, v):
        if v is not None and v <= 0:
            raise ValueError("mirostat_tau must be positive")
        return v

    @field_validator("num_gpu")
    @classmethod
    def validate_num_gpu(cls, v):
        if v is not None and v < 0:
            raise ValueError("num_gpu must be non-negative")
        return v

    @field_validator("num_thread")
    @classmethod
    def validate_num_thread(cls, v):
        if v is not None and v <= 0:
            raise ValueError("num_thread must be positive")
        return v

    @field_validator("search_recency_filter")
    @classmethod
    def validate_search_recency_filter(cls, v):
        if v is not None:
            valid_filters = ["hour", "day", "week", "month", "year"]
            if v not in valid_filters:
                raise ValueError(
                    f'search_recency_filter must be one of: {", ".join(valid_filters)}'
                )
        return v

    @field_validator("grok_mode")
    @classmethod
    def validate_grok_mode(cls, v):
        if v is not None:
            valid_modes = ["balanced", "creative", "precise"]
            if v not in valid_modes:
                raise ValueError(f'grok_mode must be one of: {", ".join(valid_modes)}')
        return v

    @field_validator("random_seed")
    @classmethod
    def validate_random_seed(cls, v):
        if v is not None and (v < 0 or v > 2**32 - 1):
            raise ValueError("random_seed must be between 0 and 2^32-1")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValueError("top_k must be positive")
        return v

    @field_validator("azure_endpoint")
    @classmethod
    def validate_azure_endpoint(cls, v):
        if v is not None and v.strip():
            # Basic URL validation
            if not (v.startswith("https://") or v.startswith("http://")):
                # Allow endpoints without protocol, we'll add it in the provider
                pass
        return v

    @field_validator("api_version")
    @classmethod
    def validate_api_version(cls, v):
        if v is not None and not v.strip():
            raise ValueError("api_version cannot be empty if provided")
        return v

    @field_validator("vertex_location")
    @classmethod
    def validate_vertex_location(cls, v):
        if v is not None and not v.strip():
            raise ValueError("vertex_location cannot be empty if provided")
        return v

    @field_validator("aws_region")
    @classmethod
    def validate_aws_region(cls, v):
        if v is not None and not v.strip():
            raise ValueError("aws_region cannot be empty if provided")
        return v

    @field_validator("max_cost_per_token")
    @classmethod
    def validate_max_cost_per_token(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_cost_per_token must be positive")
        return v

    @field_validator("claude_code_mode")
    @classmethod
    def validate_claude_code_mode(cls, v):
        if v is not None:
            valid_modes = ["opus", "sonnet"]
            if v not in valid_modes:
                raise ValueError(
                    f'claude_code_mode must be one of: {", ".join(valid_modes)}'
                )
        return v

    def get_provider_specific_config(self) -> Dict[str, Any]:
        """Get provider-specific configuration as a dictionary."""
        config = {}

        # Add non-None values to config
        for field_name, field_value in self.model_dump().items():
            if field_value is not None and field_name not in [
                "api_key",
                "model",
            ]:
                config[field_name] = field_value

        return config

    @classmethod
    def get_provider_config_template(cls, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration template for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary with provider-specific configuration template
        """
        templates = {
            "perplexity": {
                "model": "llama-3.1-sonar-large-128k-online",
                "api_key": "your-perplexity-api-key",
                "search_enabled": True,
                "search_recency_filter": "month",
                "return_citations": True,
                "return_images": False,
                "return_related_questions": False,
                "temperature": 0.2,
                "max_tokens": 4096,
            },
            "xai": {
                "model": "grok-beta",
                "api_key": "your-xai-api-key",
                "grok_mode": "balanced",
                "enable_web_search": True,
                "enable_real_time": True,
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "mistral": {
                "model": "mistral-large-latest",
                "api_key": "your-mistral-api-key",
                "safe_prompt": False,
                "temperature": 0.7,
                "top_p": 1.0,
                "max_tokens": 4096,
            },
            "azure": {
                "model": "gpt-4",
                "api_key": "your-azure-api-key",
                "azure_endpoint": "https://your-resource.openai.azure.com",
                "azure_deployment": "your-deployment-name",
                "api_version": "2024-02-15-preview",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "vertex": {
                "model": "gemini-1.5-pro",
                "vertex_project": "your-gcp-project-id",
                "vertex_location": "us-central1",
                "service_account_path": "/path/to/service-account.json",
                "temperature": 0.7,
                "max_tokens": 8192,
            },
            "bedrock": {
                "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                "aws_region": "us-east-1",
                "aws_access_key_id": "your-aws-access-key",
                "aws_secret_access_key": "your-aws-secret-key",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "openrouter": {
                "model": "anthropic/claude-3.5-sonnet",
                "api_key": "your-openrouter-api-key",
                "openrouter_referrer": "https://github.com/your-project",
                "openrouter_title": "Your Application",
                "enable_fallback": True,
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "claude-code": {
                "model": "claude-code-sonnet",
                "claude_code_mode": "sonnet",
                "claude_code_path": "claude-code",
                "working_directory": "/path/to/working/directory",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
        }

        return templates.get(
            provider_name,
            {
                "model": "default-model",
                "api_key": "your-api-key",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
        )

    @classmethod
    def get_all_provider_templates(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get configuration templates for all supported providers.

        Returns:
            Dictionary mapping provider names to their configuration templates
        """
        provider_names = [
            "claude",
            "openai",
            "gemini",
            "cohere",
            "ollama",
            "perplexity",
            "xai",
            "mistral",
            "azure",
            "vertex",
            "bedrock",
            "openrouter",
            "claude-code",
        ]

        templates = {}
        for provider_name in provider_names:
            templates[provider_name] = cls.get_provider_config_template(provider_name)

        return templates

    def __str__(self) -> str:
        """String representation with masked sensitive data."""
        data = self.model_dump()

        # Mask sensitive fields
        sensitive_fields = ["api_key", "service_account_path", "oauth_config"]
        for field in sensitive_fields:
            if field in data and data[field]:
                if isinstance(data[field], str):
                    data[field] = self._mask_string(data[field])
                elif isinstance(data[field], dict):
                    data[field] = {
                        k: (self._mask_string(str(v)) if isinstance(v, str) else v)
                        for k, v in data[field].items()
                    }

        return f"ProviderConfig(model={data.get('model')}, provider_type={data.get('provider_type')}, api_key={'***masked***' if self.api_key else None})"

    def __repr__(self) -> str:
        """Representation with masked sensitive data."""
        return self.__str__()

    @staticmethod
    def _mask_string(value: str) -> str:
        """Mask a string value for security."""
        if not value or len(value) < 8:
            return "***masked***"
        return f"{value[:4]}...{value[-4:]}"


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str
    command: str
    args: List[str] = []
    env: Dict[str, str] = {}
    enabled: bool = True
    auto_approve: List[str] = []
    timeout: int = 30

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("MCP server name cannot be empty")
        return v.strip()

    @field_validator("command")
    @classmethod
    def validate_command(cls, v):
        if not v or not v.strip():
            raise ValueError("MCP server command cannot be empty")
        return v.strip()


class MCPConfig(BaseModel):
    """Configuration for MCP integration."""

    enabled: bool = True
    servers: Dict[str, MCPServerConfig] = {}
    auto_approve_timeout: int = 30
    max_concurrent_servers: int = 10

    # Global MCP settings
    default_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Security settings
    allow_dangerous_tools: bool = False
    sandbox_mode: bool = True
    max_tool_execution_time: int = 300  # seconds

    # Logging and monitoring
    log_tool_calls: bool = True
    log_level: str = "INFO"
    metrics_enabled: bool = False

    # Tool discovery and management
    auto_discover_tools: bool = True
    tool_cache_ttl: int = 3600  # seconds
    refresh_tools_on_startup: bool = True

    @field_validator("auto_approve_timeout")
    @classmethod
    def validate_auto_approve_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("auto_approve_timeout must be positive")
        if v is not None and v > 120:  # 2 minutes max
            raise ValueError("auto_approve_timeout must be 120 seconds or less")
        return v

    @field_validator("max_concurrent_servers")
    @classmethod
    def validate_max_concurrent_servers(cls, v):
        if v <= 0:
            raise ValueError("max_concurrent_servers must be positive")
        return v

    @field_validator("default_timeout")
    @classmethod
    def validate_default_timeout(cls, v):
        if v <= 0:
            raise ValueError("default_timeout must be positive")
        return v

    @field_validator("retry_attempts")
    @classmethod
    def validate_retry_attempts(cls, v):
        if v < 0:
            raise ValueError("retry_attempts must be non-negative")
        return v

    @field_validator("retry_delay")
    @classmethod
    def validate_retry_delay(cls, v):
        if v < 0:
            raise ValueError("retry_delay must be non-negative")
        return v

    @field_validator("max_tool_execution_time")
    @classmethod
    def validate_max_tool_execution_time(cls, v):
        if v <= 0:
            raise ValueError("max_tool_execution_time must be positive")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of: {", ".join(valid_levels)}')
        return v.upper()

    @field_validator("tool_cache_ttl")
    @classmethod
    def validate_tool_cache_ttl(cls, v):
        if v <= 0:
            raise ValueError("tool_cache_ttl must be positive")
        return v

    def get_enabled_servers(self) -> Dict[str, MCPServerConfig]:
        """Get only enabled MCP servers."""
        return {name: config for name, config in self.servers.items() if config.enabled}

    def get_server_by_name(self, name: str) -> Optional[MCPServerConfig]:
        """Get a specific MCP server configuration by name."""
        return self.servers.get(name)

    def add_server(self, name: str, config: MCPServerConfig) -> None:
        """Add a new MCP server configuration."""
        self.servers[name] = config

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration."""
        if name in self.servers:
            del self.servers[name]
            return True
        return False

    def get_auto_approved_tools(self) -> Dict[str, List[str]]:
        """Get all auto-approved tools grouped by server."""
        auto_approved = {}
        for server_name, server_config in self.servers.items():
            if server_config.auto_approve:
                auto_approved[server_name] = server_config.auto_approve
        return auto_approved


class ConfigProfile(BaseModel):
    """Configuration profile for different environments/projects."""

    name: str
    description: Optional[str] = None
    default_provider: str
    providers: Dict[str, ProviderConfig]
    chat_settings: ChatSettings = ChatSettings()
    mcp: MCPConfig = MCPConfig()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("profile name cannot be empty")
        return v.strip()

    @field_validator("default_provider")
    @classmethod
    def validate_default_provider(cls, v):
        if not v or not v.strip():
            raise ValueError("default_provider cannot be empty")
        return v.strip()


class Config(BaseModel):
    """Main configuration model."""

    default_provider: str
    providers: Dict[str, ProviderConfig]
    chat_settings: ChatSettings = ChatSettings()
    storage_path: str
    mcp: MCPConfig = MCPConfig()

    # Profile management
    profiles: Dict[str, ConfigProfile] = {}
    active_profile: Optional[str] = None

    # Configuration metadata
    config_version: str = "2.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Configuration sources tracking
    config_sources: Dict[str, str] = {}  # Maps setting names to their sources

    # Global settings
    debug_mode: bool = False
    log_level: str = "INFO"
    auto_update_check: bool = True
    telemetry_enabled: bool = False

    # Provider management settings
    provider_fallback_enabled: bool = True
    provider_health_check_enabled: bool = True
    provider_timeout_default: float = 30.0

    # Security settings
    api_key_encryption_enabled: bool = True
    secure_storage_enabled: bool = True

    # Performance settings
    concurrent_requests_limit: int = 5
    request_cache_enabled: bool = True
    request_cache_ttl: int = 300  # seconds

    # Custom models - user-defined models that extend available options
    custom_models: List[EnhancedModelInfo] = []

    @field_validator("default_provider")
    @classmethod
    def validate_default_provider(cls, v):
        if not v or not v.strip():
            raise ValueError("default_provider cannot be empty")
        return v.strip()

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, v):
        if not v or not v.strip():
            raise ValueError("storage_path cannot be empty")
        return v.strip()

    @field_validator("active_profile")
    @classmethod
    def validate_active_profile(cls, v, info):
        if v is not None and "profiles" in info.data:
            profiles = info.data["profiles"]
            if v not in profiles:
                raise ValueError(f'active_profile "{v}" not found in profiles')
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of: {", ".join(valid_levels)}')
        return v.upper()

    @field_validator("provider_timeout_default")
    @classmethod
    def validate_provider_timeout_default(cls, v):
        if v <= 0:
            raise ValueError("provider_timeout_default must be positive")
        return v

    @field_validator("concurrent_requests_limit")
    @classmethod
    def validate_concurrent_requests_limit(cls, v):
        if v <= 0:
            raise ValueError("concurrent_requests_limit must be positive")
        return v

    @field_validator("request_cache_ttl")
    @classmethod
    def validate_request_cache_ttl(cls, v):
        if v <= 0:
            raise ValueError("request_cache_ttl must be positive")
        return v

    def get_active_config(self) -> "Config":
        """Get the active configuration (profile or main config)."""
        if self.active_profile and self.active_profile in self.profiles:
            profile = self.profiles[self.active_profile]
            # Create a new Config object with profile settings
            return Config(
                default_provider=profile.default_provider,
                providers=profile.providers,
                chat_settings=profile.chat_settings,
                storage_path=self.storage_path,  # Storage path is global
                mcp=profile.mcp,
                profiles=self.profiles,
                active_profile=self.active_profile,
                config_version=self.config_version,
                created_at=self.created_at,
                updated_at=self.updated_at,
                config_sources=self.config_sources,
            )
        return self

    def create_profile(
        self, name: str, description: Optional[str] = None
    ) -> ConfigProfile:
        """Create a new configuration profile based on current settings."""
        profile = ConfigProfile(
            name=name,
            description=description,
            default_provider=self.default_provider,
            providers=self.providers.copy(),
            chat_settings=self.chat_settings.model_copy(),
            mcp=self.mcp.model_copy(),
        )
        self.profiles[name] = profile
        return profile

    def switch_profile(self, profile_name: str) -> None:
        """Switch to a different configuration profile."""
        if profile_name not in self.profiles:
            raise ValueError(f'Profile "{profile_name}" not found')
        self.active_profile = profile_name

    def delete_profile(self, profile_name: str) -> bool:
        """Delete a configuration profile."""
        if profile_name in self.profiles:
            del self.profiles[profile_name]
            if self.active_profile == profile_name:
                self.active_profile = None
            return True
        return False

    def merge_from_env(self, env_vars: Dict[str, str]) -> None:
        """Merge configuration from environment variables."""
        # Map environment variables to configuration fields
        env_mapping = {
            "omnimancer_DEFAULT_PROVIDER": "default_provider",
            "omnimancer_STORAGE_PATH": "storage_path",
            "omnimancer_MAX_TOKENS": "chat_settings.max_tokens",
            "omnimancer_TEMPERATURE": "chat_settings.temperature",
            "omnimancer_CONTEXT_LENGTH": "chat_settings.context_length",
            "omnimancer_SAVE_HISTORY": "chat_settings.save_history",
            "omnimancer_MCP_ENABLED": "mcp.enabled",
            "omnimancer_MCP_AUTO_APPROVE_TIMEOUT": "mcp.auto_approve_timeout",
            "omnimancer_MCP_MAX_CONCURRENT_SERVERS": "mcp.max_concurrent_servers",
        }

        for env_var, config_path in env_mapping.items():
            if env_var in env_vars:
                value = env_vars[env_var]
                self._set_nested_value(config_path, value)
                self.config_sources[config_path] = f"env:{env_var}"

        # Handle provider-specific environment variables
        for env_var, value in env_vars.items():
            if env_var.startswith("omnimancer_") and "_API_KEY" in env_var:
                # Extract provider name from env var (e.g., omnimancer_CLAUDE_API_KEY -> claude)
                provider_name = (
                    env_var.replace("omnimancer_", "").replace("_API_KEY", "").lower()
                )
                if provider_name in self.providers:
                    # Don't store the actual API key in config_sources for security
                    self.config_sources[f"providers.{provider_name}.api_key"] = (
                        f"env:{env_var}"
                    )

    def _set_nested_value(self, path: str, value: str) -> None:
        """Set a nested configuration value from a dot-separated path."""
        parts = path.split(".")
        obj = self

        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return  # Path doesn't exist

        # Set the final value with appropriate type conversion
        final_key = parts[-1]
        if hasattr(obj, final_key):
            current_value = getattr(obj, final_key)
            try:
                if isinstance(current_value, bool):
                    converted_value = value.lower() in (
                        "true",
                        "1",
                        "yes",
                        "on",
                    )
                elif isinstance(current_value, int):
                    converted_value = int(value)
                elif isinstance(current_value, float):
                    converted_value = float(value)
                else:
                    converted_value = value
                setattr(obj, final_key, converted_value)
            except (ValueError, TypeError):
                # If conversion fails, keep as string
                setattr(obj, final_key, value)

    def validate_configuration(self) -> List[str]:
        """
        Validate the entire configuration and return a list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate basic configuration
        if not self.default_provider:
            errors.append("No default provider specified")
        elif self.default_provider not in self.providers:
            errors.append(
                f"Default provider '{self.default_provider}' not found in configured providers"
            )

        if not self.providers:
            errors.append("No providers configured")

        # Validate each provider
        for provider_name, provider_config in self.providers.items():
            provider_errors = self._validate_provider_config(
                provider_name, provider_config
            )
            errors.extend(provider_errors)

        # Validate MCP configuration
        mcp_errors = self._validate_mcp_config()
        errors.extend(mcp_errors)

        # Validate profiles
        for profile_name, profile in self.profiles.items():
            profile_errors = self._validate_profile_config(profile_name, profile)
            errors.extend(profile_errors)

        # Validate active profile
        if self.active_profile and self.active_profile not in self.profiles:
            errors.append(f"Active profile '{self.active_profile}' not found")

        return errors

    def __str__(self) -> str:
        """String representation with masked sensitive data."""
        # Create a safe representation without exposing API keys
        provider_summary = {}
        for name, config in self.providers.items():
            provider_summary[name] = {
                "model": config.model,
                "provider_type": config.provider_type,
                "api_key_configured": bool(config.api_key),
                "enabled": config.enabled,
            }

        return f"Config(default_provider={self.default_provider}, providers={list(self.providers.keys())}, mcp_enabled={self.mcp.enabled})"

    def __repr__(self) -> str:
        """Representation with masked sensitive data."""
        return self.__str__()

    def _validate_provider_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate a single provider configuration."""
        errors = []

        # Basic validation
        if not config.model:
            errors.append(f"Provider '{provider_name}' has no model specified")

        # Provider-specific validation
        if config.provider_type == "claude" or provider_name == "claude":
            errors.extend(self._validate_claude_config(provider_name, config))
        elif config.provider_type == "openai" or provider_name == "openai":
            errors.extend(self._validate_openai_config(provider_name, config))
        elif config.provider_type == "gemini" or provider_name == "gemini":
            errors.extend(self._validate_gemini_config(provider_name, config))
        elif config.provider_type == "cohere" or provider_name == "cohere":
            errors.extend(self._validate_cohere_config(provider_name, config))
        elif config.provider_type == "ollama" or provider_name == "ollama":
            errors.extend(self._validate_ollama_config(provider_name, config))

        return errors

    def _validate_claude_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate Claude-specific configuration."""
        errors = []
        if not config.api_key:
            errors.append(f"Claude provider '{provider_name}' requires an API key")
        return errors

    def _validate_openai_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate OpenAI-specific configuration."""
        errors = []
        if not config.api_key:
            errors.append(f"OpenAI provider '{provider_name}' requires an API key")
        return errors

    def _validate_gemini_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate Gemini-specific configuration."""
        errors = []
        if not config.api_key:
            errors.append(f"Gemini provider '{provider_name}' requires an API key")
        return errors

    def _validate_cohere_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate Cohere-specific configuration."""
        errors = []
        if not config.api_key:
            errors.append(f"Cohere provider '{provider_name}' requires an API key")
        return errors

    def _validate_ollama_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate Ollama-specific configuration."""
        errors = []
        # Ollama doesn't require API key but needs base_url
        if not config.base_url:
            config.base_url = "http://localhost:11434"  # Set default
        return errors

    def _validate_mcp_config(self) -> List[str]:
        """Validate MCP configuration."""
        errors = []

        if self.mcp.enabled:
            if not self.mcp.servers:
                errors.append("MCP is enabled but no servers are configured")

            for server_name, server_config in self.mcp.servers.items():
                if not server_config.command:
                    errors.append(
                        f"MCP server '{server_name}' has no command specified"
                    )

        return errors

    def _validate_profile_config(
        self, profile_name: str, profile: ConfigProfile
    ) -> List[str]:
        """Validate a configuration profile."""
        errors = []

        if profile.default_provider not in profile.providers:
            errors.append(
                f"Profile '{profile_name}' default provider '{profile.default_provider}' not found in profile providers"
            )

        # Validate each provider in the profile
        for provider_name, provider_config in profile.providers.items():
            provider_errors = self._validate_provider_config(
                provider_name, provider_config
            )
            errors.extend(
                [f"Profile '{profile_name}': {error}" for error in provider_errors]
            )

        return errors

    def _validate_claude_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate Claude provider configuration."""
        errors = []

        if not config.api_key:
            errors.append(f"Claude provider '{provider_name}' requires an API key")

        # Validate model
        valid_models = [
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
        ]
        if config.model not in valid_models:
            errors.append(
                f"Unknown Claude model '{config.model}' for provider '{provider_name}'. Valid models: {', '.join(valid_models)}"
            )

        return errors

    def _validate_openai_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate OpenAI provider configuration."""
        errors = []

        if not config.api_key:
            errors.append(f"OpenAI provider '{provider_name}' requires an API key")

        # Validate model
        valid_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
        if config.model not in valid_models:
            errors.append(
                f"Unknown OpenAI model '{config.model}' for provider '{provider_name}'. Valid models: {', '.join(valid_models)}"
            )

        return errors

    def _validate_gemini_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate Gemini provider configuration."""
        errors = []

        if not config.api_key and not config.service_account_path:
            errors.append(
                f"Gemini provider '{provider_name}' requires either an API key or service account"
            )

        # Validate model
        valid_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        if config.model not in valid_models:
            errors.append(
                f"Unknown Gemini model '{config.model}' for provider '{provider_name}'. Valid models: {', '.join(valid_models)}"
            )

        return errors

    def _validate_cohere_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate Cohere provider configuration."""
        errors = []

        if not config.api_key:
            errors.append(f"Cohere provider '{provider_name}' requires an API key")

        # Validate model
        valid_models = [
            "command-r",
            "command-r-plus",
            "command-light",
            "command",
        ]
        if config.model not in valid_models:
            errors.append(
                f"Unknown Cohere model '{config.model}' for provider '{provider_name}'. Valid models: {', '.join(valid_models)}"
            )

        return errors

    def _validate_ollama_config(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate Ollama provider configuration."""
        errors = []

        # Ollama doesn't require an API key (local server)
        # But we should validate the base_url if provided
        if config.base_url:
            if not config.base_url.startswith(("http://", "https://")):
                errors.append(
                    f"Ollama provider '{provider_name}' base_url must start with 'http://' or 'https://'"
                )

        # Model validation is difficult for Ollama since models are dynamic
        if not config.model:
            errors.append(f"Ollama provider '{provider_name}' requires a model name")

        return errors

    def _validate_mcp_config(self) -> List[str]:
        """Validate MCP configuration."""
        errors = []

        # Validate server configurations
        for server_name, server_config in self.mcp.servers.items():
            if not server_config.name:
                errors.append(f"MCP server '{server_name}' has no name")
            if not server_config.command:
                errors.append(f"MCP server '{server_name}' has no command")
            if server_config.timeout <= 0:
                errors.append(
                    f"MCP server '{server_name}' has invalid timeout: {server_config.timeout}"
                )

        return errors

    def _validate_profile_config(
        self, profile_name: str, profile: ConfigProfile
    ) -> List[str]:
        """Validate a configuration profile."""
        errors = []

        if not profile.default_provider:
            errors.append(f"Profile '{profile_name}' has no default provider")
        elif profile.default_provider not in profile.providers:
            errors.append(
                f"Profile '{profile_name}' default provider '{profile.default_provider}' not found in profile providers"
            )

        if not profile.providers:
            errors.append(f"Profile '{profile_name}' has no providers configured")

        # Validate each provider in the profile
        for provider_name, provider_config in profile.providers.items():
            provider_errors = self._validate_provider_config(
                f"{profile_name}.{provider_name}", provider_config
            )
            errors.extend(provider_errors)

        return errors

    def get_enabled_providers(self) -> Dict[str, ProviderConfig]:
        """Get all enabled provider configurations."""
        return {
            name: config for name, config in self.providers.items() if config.enabled
        }

    def get_providers_by_priority(self) -> List[tuple[str, ProviderConfig]]:
        """Get providers sorted by priority (highest first)."""
        enabled_providers = self.get_enabled_providers()
        return sorted(
            enabled_providers.items(),
            key=lambda x: x[1].priority,
            reverse=True,
        )

    def get_providers_with_tools(self) -> Dict[str, ProviderConfig]:
        """Get all providers that support tools."""
        return {
            name: config
            for name, config in self.providers.items()
            if config.enabled
            and (config.supports_tools or config.supports_function_calling)
        }

    def get_providers_with_multimodal(self) -> Dict[str, ProviderConfig]:
        """Get all providers that support multimodal input."""
        return {
            name: config
            for name, config in self.providers.items()
            if config.enabled and config.supports_multimodal
        }

    def update_provider_health_status(
        self, provider_name: str, is_healthy: bool
    ) -> None:
        """Update the health status of a provider (for runtime tracking)."""
        if provider_name in self.providers:
            # This would be tracked in runtime state, not persisted config
            pass

    def get_fallback_providers(self, exclude_provider: str = None) -> List[str]:
        """Get list of fallback providers in priority order."""
        providers_by_priority = self.get_providers_by_priority()
        fallback_providers = []

        for provider_name, config in providers_by_priority:
            if provider_name != exclude_provider and config.enabled:
                fallback_providers.append(provider_name)

        return fallback_providers

    def get_provider_timeout(self, provider_name: str) -> float:
        """Get timeout for a specific provider, with fallback to default."""
        if provider_name in self.providers:
            provider_config = self.providers[provider_name]
            return provider_config.timeout or self.provider_timeout_default
        return self.provider_timeout_default

    def is_provider_enabled(self, provider_name: str) -> bool:
        """Check if a provider is enabled."""
        if provider_name in self.providers:
            return self.providers[provider_name].enabled
        return False

    def get_provider_auth_type(self, provider_name: str) -> str:
        """Get authentication type for a provider."""
        if provider_name in self.providers:
            return self.providers[provider_name].auth_type
        return "api_key"  # Default auth type

    def get_provider_capabilities(self, provider_name: str) -> Dict[str, bool]:
        """Get capabilities for a specific provider."""
        if provider_name not in self.providers:
            return {}

        config = self.providers[provider_name]
        return {
            "supports_tools": config.supports_tools or config.supports_function_calling,
            "supports_multimodal": config.supports_multimodal,
            "supports_streaming": config.supports_streaming,
            "supports_system_messages": config.supports_system_messages,
        }

    def get_mcp_enabled_servers(self) -> Dict[str, MCPServerConfig]:
        """Get all enabled MCP servers."""
        return self.mcp.get_enabled_servers()

    def has_mcp_servers(self) -> bool:
        """Check if any MCP servers are configured and enabled."""
        return bool(self.get_mcp_enabled_servers())

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "config_version": self.config_version,
            "default_provider": self.default_provider,
            "active_profile": self.active_profile,
            "total_providers": len(self.providers),
            "enabled_providers": len(self.get_enabled_providers()),
            "providers_with_tools": len(self.get_providers_with_tools()),
            "providers_with_multimodal": len(self.get_providers_with_multimodal()),
            "total_profiles": len(self.profiles),
            "mcp_enabled": self.mcp.enabled,
            "mcp_servers": len(self.mcp.servers),
            "enabled_mcp_servers": len(self.get_mcp_enabled_servers()),
            "debug_mode": self.debug_mode,
            "log_level": self.log_level,
        }

    def get_provider_health_status(self, provider_name: str) -> Optional[bool]:
        """Get the health status of a provider."""
        if provider_name in self.providers:
            extra_settings = self.providers[provider_name].extra_settings
            if extra_settings and "is_healthy" in extra_settings:
                return extra_settings["is_healthy"]
        return None


# BaseProvider is defined in providers/base.py to avoid circular imports


class MCPServerError(MCPError):
    """MCP server connection or execution error."""

    pass


class MCPToolError(MCPError):
    """MCP tool execution error."""

    pass


# Minimal config template classes to replace removed config_templates.py
@dataclass
class ConfigTemplate:
    """Minimal configuration template definition."""

    name: str
    description: str
    use_case: str
    recommended_providers: List[str]
    recommended_models: Dict[str, str]
    settings: Dict[str, Any]
    mcp_tools: List[str] = None
    provider_configs: Dict[str, Any] = None
    mcp_servers: Dict[str, Any] = None

    def __post_init__(self):
        if self.mcp_tools is None:
            self.mcp_tools = []
        if self.provider_configs is None:
            self.provider_configs = {}
        if self.mcp_servers is None:
            self.mcp_servers = {}


class ConfigTemplateManager:
    """Minimal configuration template manager."""

    def __init__(self):
        self.templates = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize all available templates."""
        self.templates["coding"] = self._create_coding_template()
        self.templates["research"] = self._create_research_template()
        self.templates["creative"] = self._create_creative_template()
        self.templates["performance"] = self._create_performance_template()
        self.templates["general"] = self._create_general_template()

    def _create_coding_template(self) -> ConfigTemplate:
        """Create coding template."""
        return ConfigTemplate(
            name="coding",
            description="Optimized for software development and programming tasks",
            use_case="coding",
            recommended_providers=["claude", "openai", "claude_code"],
            recommended_models={
                "claude": "claude-3-5-sonnet-20241022",
                "openai": "gpt-4o",
                "claude_code": "claude-code-sonnet",
            },
            settings={"temperature": 0.2, "max_tokens": 4096},
            provider_configs={
                "claude": {
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "model": "claude-3-5-sonnet-20241022",
                },
                "openai": {
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "model": "gpt-4o",
                },
                "claude_code": {
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "model": "claude-code-sonnet",
                },
            },
            mcp_servers={
                "filesystem": {
                    "command": "uvx",
                    "args": ["mcp-server-filesystem"],
                    "enabled": False,
                },
                "git": {
                    "command": "uvx",
                    "args": ["mcp-server-git"],
                    "enabled": False,
                },
            },
        )

    def _create_research_template(self) -> ConfigTemplate:
        """Create research template."""
        return ConfigTemplate(
            name="research",
            description="Optimized for research and analysis tasks",
            use_case="research",
            recommended_providers=["perplexity", "claude", "openai"],
            recommended_models={
                "perplexity": "sonar-pro",
                "claude": "claude-3-5-sonnet-20241022",
                "openai": "gpt-4o",
            },
            settings={"temperature": 0.3, "max_tokens": 8192},
            provider_configs={
                "perplexity": {
                    "temperature": 0.3,
                    "max_tokens": 8192,
                    "model": "sonar-pro",
                },
                "claude": {
                    "temperature": 0.3,
                    "max_tokens": 8192,
                    "model": "claude-3-5-sonnet-20241022",
                },
                "openai": {
                    "temperature": 0.3,
                    "max_tokens": 8192,
                    "model": "gpt-4o",
                },
            },
            mcp_servers={
                "web-search": {
                    "command": "uvx",
                    "args": ["mcp-server-web-search"],
                    "enabled": False,
                },
                "knowledge": {
                    "command": "uvx",
                    "args": ["mcp-server-knowledge"],
                    "enabled": False,
                },
            },
        )

    def _create_creative_template(self) -> ConfigTemplate:
        """Create creative template."""
        return ConfigTemplate(
            name="creative",
            description="Optimized for creative writing and content generation",
            use_case="creative",
            recommended_providers=["openai", "claude", "gemini"],
            recommended_models={
                "openai": "gpt-4o",
                "claude": "claude-3-5-sonnet-20241022",
                "gemini": "gemini-1.5-pro",
            },
            settings={"temperature": 0.8, "max_tokens": 4096},
            provider_configs={
                "openai": {
                    "temperature": 0.8,
                    "max_tokens": 4096,
                    "model": "gpt-4o",
                },
                "claude": {
                    "temperature": 0.8,
                    "max_tokens": 4096,
                    "model": "claude-3-5-sonnet-20241022",
                },
                "gemini": {
                    "temperature": 0.8,
                    "max_tokens": 4096,
                    "model": "gemini-1.5-pro",
                },
            },
        )

    def _create_performance_template(self) -> ConfigTemplate:
        """Create performance template."""
        return ConfigTemplate(
            name="performance",
            description="Optimized for fast responses and efficiency",
            use_case="performance",
            recommended_providers=["openai", "claude", "gemini"],
            recommended_models={
                "openai": "gpt-4o-mini",
                "claude": "claude-3-haiku",
                "gemini": "gemini-1.5-flash",
            },
            settings={"temperature": 0.5, "max_tokens": 2048},
            provider_configs={
                "openai": {
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "model": "gpt-4o-mini",
                },
                "claude": {
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "model": "claude-3-haiku",
                },
                "gemini": {
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "model": "gemini-1.5-flash",
                },
            },
        )

    def _create_general_template(self) -> ConfigTemplate:
        """Create general template."""
        return ConfigTemplate(
            name="general",
            description="General purpose configuration",
            use_case="general",
            recommended_providers=["openai", "anthropic"],
            recommended_models={
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-haiku",
            },
            settings={"temperature": 0.7, "max_tokens": 2048},
            provider_configs={
                "openai": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "model": "gpt-4o-mini",
                },
                "anthropic": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "model": "claude-3-haiku",
                },
            },
        )

    def get_template(self, name: str) -> ConfigTemplate:
        """Get template by name. Raises KeyError if not found."""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")
        return self.templates[name]

    def list_templates(self) -> List[Dict[str, str]]:
        """List available templates with their information."""
        return [
            {
                "name": template.name,
                "description": template.description,
                "use_case": template.use_case,
            }
            for template in self.templates.values()
        ]
