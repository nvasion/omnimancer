"""In-memory CLI session configuration overrides."""

from typing import Optional

from ..core.config_manager import ConfigManager
from ..core.models import ProviderConfig


def apply_session_overrides(
    config_manager: ConfigManager,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Apply provider overrides to the in-memory configuration only.

    Args:
        config_manager: Configuration manager whose loaded model is updated.
        provider: Optional provider selected for this process.
        model: Optional provider model selected for this process.
        base_url: Optional provider endpoint selected for this process.
    """
    if not (provider or model or base_url):
        return

    config = config_manager.get_config()
    if provider:
        config.default_provider = provider

    target = provider or config.default_provider
    if not target:
        return

    provider_config = config.providers.get(target)
    if provider_config is None:
        provider_config = ProviderConfig(model=model or "")
        config.providers[target] = provider_config
    if model:
        provider_config.model = model
    if base_url:
        provider_config.base_url = base_url.rstrip("/")
