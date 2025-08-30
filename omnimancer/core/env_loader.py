"""
Environment variable loader for API keys.

This module provides functionality to load API keys from environment variables
and inject them into provider configurations.
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Mapping of provider names to environment variable names
ENV_VAR_MAPPING = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "azure": "AZURE_OPENAI_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def load_api_key_from_env(provider_name: str) -> Optional[str]:
    """
    Load API key from environment variable for a specific provider.

    Args:
        provider_name: Name of the provider

    Returns:
        API key from environment or None if not found
    """
    env_var = ENV_VAR_MAPPING.get(provider_name)
    if not env_var:
        return None

    api_key = os.environ.get(env_var)
    if api_key:
        logger.debug(
            f"Loaded API key for {provider_name} from environment variable {env_var}"
        )
    return api_key


def inject_env_api_keys(provider_configs: Dict) -> Dict:
    """
    Inject API keys from environment variables into provider configurations.

    This function checks each provider configuration and if the API key is missing
    or is a placeholder, it attempts to load it from the environment.

    Args:
        provider_configs: Dictionary of provider configurations

    Returns:
        Updated provider configurations with environment API keys
    """
    for provider_name, config in provider_configs.items():
        # Skip if provider doesn't need API key
        if provider_name in ["ollama", "claude-code"]:
            continue

        # Check if API key is missing or is a placeholder
        current_key = getattr(config, "api_key", None)
        if (
            not current_key
            or current_key.startswith("your-")
            or current_key.startswith("sk-your")
        ):
            # Try to load from environment
            env_key = load_api_key_from_env(provider_name)
            if env_key:
                config.api_key = env_key
                logger.info(f"Injected API key for {provider_name} from environment")
            else:
                logger.debug(f"No environment API key found for {provider_name}")

    return provider_configs
