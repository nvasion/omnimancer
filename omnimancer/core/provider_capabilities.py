"""
Provider capability definitions for Omnimancer configuration optimization.

This module defines the inherent capabilities and default settings for each provider,
eliminating the need to store null values in configuration files.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ProviderType(str, Enum):
    """Supported AI provider types."""

    OPENAI = "openai"
    CLAUDE = "claude"
    CLAUDE_CODE = "claude-code"
    GEMINI = "gemini"
    COHERE = "cohere"
    MISTRAL = "mistral"
    OLLAMA = "ollama"
    PERPLEXITY = "perplexity"
    XAI = "xai"
    AZURE = "azure"
    VERTEX = "vertex"
    BEDROCK = "bedrock"
    OPENROUTER = "openrouter"


@dataclass
class ProviderCapabilities:
    """Defines the capabilities and defaults for a provider."""

    # Capabilities (what the provider supports)
    supports_tools: bool = False
    supports_multimodal: bool = False
    supports_streaming: bool = True
    supports_system_messages: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False

    # Default settings
    default_max_tokens: Optional[int] = None
    default_temperature: Optional[float] = 0.7
    default_timeout: float = 30.0

    # Retry and reliability defaults
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    health_check_enabled: bool = True
    health_check_interval: int = 300
    health_check_timeout: float = 10.0

    # Security defaults
    auth_type: str = "api_key"
    requires_api_key: bool = True

    # Provider-specific settings that are commonly used
    common_settings: Dict[str, Any] = None

    def __post_init__(self):
        if self.common_settings is None:
            self.common_settings = {}


# Provider capability definitions
PROVIDER_CAPABILITIES: Dict[ProviderType, ProviderCapabilities] = {
    ProviderType.CLAUDE: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=True,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        default_max_tokens=4096,
        default_temperature=0.7,
        common_settings={"anthropic_version": "2023-06-01"},
    ),
    ProviderType.CLAUDE_CODE: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=True,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        default_max_tokens=8192,
        default_temperature=0.3,  # Lower for code
        requires_api_key=False,  # Local Claude Code
        auth_type="local",
        common_settings={"api_key": "local", "claude_code_mode": True},
    ),
    ProviderType.OPENAI: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=True,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        default_max_tokens=4096,
        default_temperature=0.7,
    ),
    ProviderType.GEMINI: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=True,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        default_max_tokens=8192,
        default_temperature=0.7,
        common_settings={
            "safety_settings": {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        },
    ),
    ProviderType.COHERE: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=False,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=False,
        supports_json_mode=True,
        default_max_tokens=4096,
        default_temperature=0.7,
    ),
    ProviderType.MISTRAL: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=False,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=False,
        supports_json_mode=True,
        default_max_tokens=4096,
        default_temperature=0.7,
        common_settings={"safe_prompt": False},
    ),
    ProviderType.OLLAMA: ProviderCapabilities(
        supports_tools=False,
        supports_multimodal=False,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=False,
        supports_vision=False,
        supports_json_mode=False,
        default_max_tokens=2048,
        default_temperature=0.7,
        requires_api_key=False,
        auth_type="none",
        common_settings={
            "base_url": "http://localhost:11434",
            "num_ctx": 4096,
            "repeat_penalty": 1.1,
        },
    ),
    ProviderType.PERPLEXITY: ProviderCapabilities(
        supports_tools=False,
        supports_multimodal=False,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=False,
        supports_vision=False,
        supports_json_mode=False,
        default_max_tokens=4096,
        default_temperature=0.2,  # Lower for research
        common_settings={
            "search_enabled": True,
            "return_citations": True,
            "search_recency_filter": "month",
        },
    ),
    ProviderType.XAI: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=False,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=False,
        supports_json_mode=True,
        default_max_tokens=4096,
        default_temperature=0.7,
        common_settings={
            "grok_mode": "balanced",
            "enable_web_search": True,
            "enable_real_time": True,
        },
    ),
    ProviderType.AZURE: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=True,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        default_max_tokens=4096,
        default_temperature=0.7,
        common_settings={"api_version": "2024-02-15-preview"},
    ),
    ProviderType.VERTEX: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=True,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        default_max_tokens=8192,
        default_temperature=0.7,
        common_settings={"vertex_location": "us-central1"},
    ),
    ProviderType.BEDROCK: ProviderCapabilities(
        supports_tools=True,
        supports_multimodal=True,
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=True,
        supports_vision=True,
        supports_json_mode=True,
        default_max_tokens=4096,
        default_temperature=0.7,
        common_settings={"aws_region": "us-east-1"},
    ),
    ProviderType.OPENROUTER: ProviderCapabilities(
        supports_tools=False,  # Depends on selected model
        supports_multimodal=False,  # Depends on selected model
        supports_streaming=True,
        supports_system_messages=True,
        supports_function_calling=False,  # Depends on selected model
        supports_vision=False,  # Depends on selected model
        supports_json_mode=False,  # Depends on selected model
        default_max_tokens=4096,
        default_temperature=0.7,
        common_settings={"base_url": "https://openrouter.ai/api/v1"},
    ),
}


def get_provider_capabilities(provider_type: str) -> ProviderCapabilities:
    """
    Get capabilities for a provider type.

    Args:
        provider_type: The provider type string

    Returns:
        ProviderCapabilities object with defaults and capabilities

    Raises:
        ValueError: If provider type is not supported
    """
    try:
        provider_enum = ProviderType(provider_type.lower().replace("-", "_"))
        return PROVIDER_CAPABILITIES[provider_enum]
    except (ValueError, KeyError):
        # Return basic capabilities for unknown providers
        return ProviderCapabilities()


def get_provider_defaults(provider_type: str) -> Dict[str, Any]:
    """
    Get default configuration values for a provider.

    Args:
        provider_type: The provider type string

    Returns:
        Dictionary of default configuration values
    """
    capabilities = get_provider_capabilities(provider_type)

    defaults = {
        # Capabilities
        "supports_tools": capabilities.supports_tools,
        "supports_multimodal": capabilities.supports_multimodal,
        "supports_streaming": capabilities.supports_streaming,
        "supports_system_messages": capabilities.supports_system_messages,
        "supports_function_calling": capabilities.supports_function_calling,
        "supports_vision": capabilities.supports_vision,
        "supports_json_mode": capabilities.supports_json_mode,
        # Default settings
        "max_tokens": capabilities.default_max_tokens,
        "temperature": capabilities.default_temperature,
        "timeout": capabilities.default_timeout,
        # Reliability settings
        "max_retries": capabilities.max_retries,
        "retry_delay": capabilities.retry_delay,
        "exponential_backoff": capabilities.exponential_backoff,
        "health_check_enabled": capabilities.health_check_enabled,
        "health_check_interval": capabilities.health_check_interval,
        "health_check_timeout": capabilities.health_check_timeout,
        # Security
        "auth_type": capabilities.auth_type,
        # Provider-specific defaults
        **capabilities.common_settings,
    }

    return defaults


def merge_user_config_with_defaults(
    provider_type: str, user_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge user configuration with provider defaults.

    Args:
        provider_type: The provider type string
        user_config: User-provided configuration

    Returns:
        Complete configuration with defaults applied
    """
    defaults = get_provider_defaults(provider_type)

    # Start with defaults
    merged_config = defaults.copy()

    # Override with user settings (only non-None values)
    for key, value in user_config.items():
        if value is not None:
            merged_config[key] = value

    return merged_config


def get_essential_config_fields() -> List[str]:
    """
    Get list of configuration fields that should be stored in config files.

    Returns:
        List of essential configuration field names
    """
    return [
        # Required for all providers
        "api_key",
        "model",
        # Common user preferences
        "max_tokens",
        "temperature",
        "timeout",
        "base_url",
        # Provider-specific essentials
        "organization",  # OpenAI
        "project_id",  # Google
        "azure_endpoint",  # Azure
        "azure_deployment",  # Azure
        "api_version",  # Azure
        "vertex_project",  # Vertex
        "vertex_location",  # Vertex
        "vertex_credentials_path",  # Vertex
        "aws_region",  # Bedrock
        "aws_access_key_id",  # Bedrock
        "aws_secret_access_key",  # Bedrock
        "service_account_path",  # Vertex/Google
        # Feature toggles that users might want to change
        "search_enabled",  # Perplexity
        "grok_mode",  # XAI
        "safe_prompt",  # Mistral
        # System settings
        "enabled",
        "priority",
    ]


def should_store_in_config(field_name: str, value: Any, provider_type: str) -> bool:
    """
    Determine if a configuration field should be stored in the config file.

    Args:
        field_name: Name of the configuration field
        value: Value of the field
        provider_type: Provider type

    Returns:
        True if the field should be stored, False otherwise
    """
    # Always store essential fields if they have values
    if field_name in get_essential_config_fields() and value is not None:
        return True

    # Don't store None values
    if value is None:
        return False

    # Don't store default values
    defaults = get_provider_defaults(provider_type)
    if field_name in defaults and defaults[field_name] == value:
        return False

    return True
