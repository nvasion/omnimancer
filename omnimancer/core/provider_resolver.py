"""The single answer to "which provider entry is in force, and give me one".

Every surface needs the same two things: the *effective* configuration
(stored config with environment overrides applied, in that precedence) and an
isolated provider instance built from one entry of it. Before this module each
caller re-derived that independently: ``CoreEngine.initialize_providers``
applied the overrides to a local variable and threw it away, H2L re-applied
them per model ref, the prompt enhancer read the stored config and so never
saw them, and subagents skipped resolution entirely and mutated the shared
session provider's model. A provider configured purely through
``OMNIMANCER_<PROVIDER>_API_KEY`` therefore existed for the engine and for
nobody else.

Two rules hold for everything here:

* **Nothing is ever persisted.** :func:`effective_config` returns a deep copy
  carrying ephemeral environment values; saving it would write env secrets
  into ``~/.omnimancer/config.json``. Writers keep using
  ``ConfigManager.get_config()``.
* **Never hand out an object from ``engine.providers``.** Each resolution
  builds its own instance from a ``model_copy`` of the entry, so parallel
  work (H2L tiers, subagents) cannot interleave through a shared provider.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

from ..providers.factory import ProviderFactory
from .env_loader import apply_env_overrides
from .models import Config, ProviderConfig

logger = logging.getLogger(__name__)

# Either a ConfigManager or an already-loaded Config is accepted, because the
# callers have one or the other at hand and converting is this module's job.
ConfigSource = Union[Config, Any]


class ProviderResolutionError(ValueError):
    """Base class: a provider entry could not be turned into an instance."""


class EntryNotConfigured(ProviderResolutionError):
    """No config entry, no environment entry, and no live instance."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.env_var = "OMNIMANCER_" + provider.upper().replace("-", "_") + "_API_KEY"
        super().__init__(
            f"provider entry '{provider}' is not configured "
            f"(see /providers, or set {self.env_var})"
        )


class ProviderInitFailed(ProviderResolutionError):
    """The factory could not build the provider (bad credentials, bad model)."""

    def __init__(self, provider: str, model: str, cause: Exception) -> None:
        self.provider = provider
        self.model = model
        self.cause = cause
        super().__init__(f"could not initialize {provider}:{model}: {cause}")


class ToolsUnsupported(ProviderResolutionError):
    """The caller required native tool calling and this provider has none."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"provider '{provider}' does not support tool calling")


def _as_config(source: ConfigSource) -> Config:
    """Accept a ``Config``, a ``ConfigManager``, or any config-like object.

    Config-like (anything exposing ``providers``) is allowed because callers
    and tests pass lightweight stand-ins; environment overrides need a real
    pydantic ``Config`` and degrade to a no-op for the rest.
    """
    if isinstance(source, Config):
        return source
    getter = getattr(source, "get_config", None)
    if callable(getter):
        config = getter()
        if config is not None:
            return config  # type: ignore[no-any-return,return-value]
    if hasattr(source, "providers"):
        return source  # type: ignore[no-any-return,return-value]
    raise ProviderResolutionError(
        "provider resolution needs a Config or a ConfigManager"
    )


def effective_config(source: ConfigSource) -> Config:
    """Stored configuration with environment overrides applied.

    This is the configuration the engine actually runs with. Never save it:
    it carries ephemeral environment values, including API keys.
    """
    config = _as_config(source)
    try:
        return apply_env_overrides(config)
    except Exception as exc:  # overrides must never block a run
        logger.debug(f"Environment overrides unavailable: {exc}")
        return config


def effective_providers(source: ConfigSource) -> Dict[str, ProviderConfig]:
    """The provider entries in force, including env-materialized ones."""
    return effective_config(source).providers or {}


def _entry_from_live_instance(
    provider_name: str, engine: Optional[Any], model: str
) -> Optional[ProviderConfig]:
    """Rebuild an entry from a running instance (session flags, earlier init)."""
    live = (getattr(engine, "providers", None) or {}) if engine is not None else {}
    instance = live.get(provider_name) if isinstance(live, dict) else None
    if instance is None:
        return None

    api_key = getattr(instance, "api_key", "") or ""
    kwargs = {
        k: v
        for k, v in (getattr(instance, "config", None) or {}).items()
        if v is not None and k not in ("api_key", "model")
    }
    for candidate in (kwargs, {}):
        try:
            return ProviderConfig(api_key=api_key, model=model, **candidate)
        except Exception:
            continue
    return None


def resolve_entry(
    provider_name: str,
    source: ConfigSource,
    engine: Optional[Any] = None,
    model: str = "",
) -> ProviderConfig:
    """The entry for ``provider_name``, from the first source that has it.

    Environment-effective config first (what the engine initialized from),
    then the stored config, then a live instance in ``engine.providers``.

    Raises:
        EntryNotConfigured: when no source knows this name.
    """
    config = _as_config(source)
    stored = getattr(config, "providers", None) or {}
    entry = effective_providers(config).get(provider_name) or stored.get(provider_name)
    if entry is not None:
        return entry

    from_live = _entry_from_live_instance(provider_name, engine, model)
    if from_live is not None:
        return from_live

    raise EntryNotConfigured(provider_name)


def resolve_provider(
    provider_name: str,
    source: ConfigSource,
    model: Optional[str] = None,
    engine: Optional[Any] = None,
    *,
    min_output_tokens: Optional[int] = None,
    require_tools: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build a fresh provider instance for one entry.

    Args:
        provider_name: Config-entry name (may be an alias with provider_type).
        source: ConfigManager or Config.
        model: Model id to run; the entry's own model when omitted.
        engine: Live engine, for the last-resort instance lookup.
        min_output_tokens: Floor on ``max_tokens``; a larger configured value
            wins. Providers default to 4096, which truncates long output.
        require_tools: Reject a provider without native tool calling.
        overrides: Extra entry fields to set (e.g. the enhancer's temperature).

    Raises:
        EntryNotConfigured, ProviderInitFailed, ToolsUnsupported
    """
    entry = resolve_entry(provider_name, source, engine, model or "")

    update: Dict[str, Any] = dict(overrides or {})
    if model:
        update["model"] = model
    if min_output_tokens is not None:
        current = getattr(entry, "max_tokens", None) or 0
        update["max_tokens"] = max(int(current), int(min_output_tokens))
    effective = entry.model_copy(update=update) if update else entry

    config_manager = None if isinstance(source, Config) else source
    try:
        provider = ProviderFactory.create_provider(
            provider_name, effective, config_manager
        )
    except Exception as exc:
        raise ProviderInitFailed(provider_name, effective.model, exc) from exc

    if require_tools:
        try:
            supports_tools = bool(provider.supports_tools())
        except Exception:  # a provider that cannot even answer is not usable
            supports_tools = False
        if not supports_tools:
            raise ToolsUnsupported(provider_name)

    # Belt and braces: the factory caches on (name, model), so an unchanged
    # model can legitimately return the session's own object. Callers that
    # need isolation pass a distinct model; warn when they do not get it.
    shared = getattr(engine, "providers", None) if engine is not None else None
    if isinstance(shared, dict) and any(provider is p for p in shared.values()):
        logger.debug(
            "Resolved %s to the session's own provider instance; work may "
            "interleave with the main conversation",
            provider_name,
        )
    return provider
