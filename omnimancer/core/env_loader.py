"""
Environment variable loader for API keys.

This module provides functionality to load API keys from environment variables
and inject them into provider configurations.
"""

import logging
import os
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .models import Config

logger = logging.getLogger(__name__)

# Mapping of provider names to (conventional) environment variable names for API keys.
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
    "digitalocean": "DIGITALOCEAN_INFERENCE_KEY",
}

# Environment variable that overrides the default provider.
DEFAULT_PROVIDER_ENV_VAR = "OMNIMANCER_DEFAULT_PROVIDER"


def _omnimancer_env_prefix(provider_name: str) -> str:
    """Build the OMNIMANCER_<PROVIDER>_ env var prefix for a provider."""
    return "OMNIMANCER_" + provider_name.upper().replace("-", "_")


def load_provider_env_overrides(provider_name: str) -> Dict[str, str]:
    """
    Collect per-provider overrides from environment variables.

    Recognized variables (all optional):
        OMNIMANCER_<PROVIDER>_API_KEY   (also falls back to the conventional
                                         key in ENV_VAR_MAPPING)
        OMNIMANCER_<PROVIDER>_BASE_URL
        OMNIMANCER_<PROVIDER>_MODEL

    Returns:
        Dict with any of the keys "api_key", "base_url", "model" that were set.
    """
    prefix = _omnimancer_env_prefix(provider_name)
    overrides: Dict[str, str] = {}

    api_key = os.environ.get(f"{prefix}_API_KEY") or load_api_key_from_env(
        provider_name
    )
    if api_key:
        overrides["api_key"] = api_key

    base_url = os.environ.get(f"{prefix}_BASE_URL")
    if base_url:
        overrides["base_url"] = base_url

    model = os.environ.get(f"{prefix}_MODEL")
    if model:
        overrides["model"] = model

    return overrides


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


def load_claude_subscription_token() -> Optional[dict]:
    """
    Load Claude subscription OAuth token from ~/.claude/.credentials.json.

    Returns:
        Dict with 'access_token' and 'auth_type' keys, or None if not available.
    """
    try:
        from ..providers.claude_credentials import load_claude_credentials

        creds = load_claude_credentials()
        if creds and not creds.is_expired:
            logger.info("Using Claude subscription OAuth token")
            return {"access_token": creds.access_token, "auth_type": "bearer"}
        elif creds and creds.is_expired:
            logger.debug("Claude subscription token is expired, needs refresh")
            return {
                "access_token": creds.access_token,
                "auth_type": "bearer",
                "expired": True,
                "creds": creds,
            }
    except Exception as e:
        logger.debug(f"Could not load Claude subscription credentials: {e}")
    return None


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


def apply_env_overrides(config: "Config") -> "Config":
    """
    Return a copy of ``config`` with environment-variable overrides applied.

    Overrides are ephemeral (never written back to disk) and let environment
    variables take precedence over file-based configuration. Supported:

      - Per-provider API keys via OMNIMANCER_<PROVIDER>_API_KEY or the
        conventional key (e.g. OPENAI_API_KEY, DIGITALOCEAN_INFERENCE_KEY).
      - Per-provider endpoint via OMNIMANCER_<PROVIDER>_BASE_URL.
      - Per-provider model via OMNIMANCER_<PROVIDER>_MODEL.
      - Default provider via OMNIMANCER_DEFAULT_PROVIDER.

    A provider entry is created on the fly when env vars define one that is not
    already present in the config (requires at least an api_key or base_url).
    """
    from .models import ProviderConfig

    effective = config.model_copy(deep=True)

    # Consider providers already configured, those with a recognized API-key
    # env var, plus any provider named by OMNIMANCER_DEFAULT_PROVIDER.
    candidates = set(effective.providers.keys()) | set(ENV_VAR_MAPPING.keys())
    default_override = os.environ.get(DEFAULT_PROVIDER_ENV_VAR)
    if default_override:
        candidates.add(default_override)

    for name in candidates:
        overrides = load_provider_env_overrides(name)
        if not overrides:
            continue

        provider_config = effective.providers.get(name)
        if provider_config is None:
            # Only materialize a new provider when env gives enough to use it.
            if "api_key" not in overrides and "base_url" not in overrides:
                continue
            provider_config = ProviderConfig(
                api_key=overrides.get("api_key"),
                model=overrides.get("model", ""),
            )
            effective.providers[name] = provider_config
            logger.info(f"Created provider '{name}' from environment overrides")

        if "api_key" in overrides:
            provider_config.api_key = overrides["api_key"]
        if "base_url" in overrides:
            provider_config.base_url = overrides["base_url"]
        if "model" in overrides:
            provider_config.model = overrides["model"]

    if default_override:
        effective.default_provider = default_override
        logger.info(f"Default provider overridden from environment: {default_override}")

    return effective
