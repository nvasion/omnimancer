"""Resolve H2L model refs into isolated provider instances (PRD §1).

The one rule that makes parallel tiers safe: never hand out an object from
``engine.providers``. Each ref gets its own instance built from a
``model_copy`` of the stored entry, the same way ``prompt_enhancer`` does it.
The instance cache in ``ProviderInitializer`` keys on the model, so two refs
on one entry with different models are two objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..core.env_loader import apply_env_overrides
from ..core.models import ProviderConfig
from ..providers.factory import ProviderFactory
from .models import H2LConfig, H2LModelRef

logger = logging.getLogger(__name__)


class H2LConfigError(ValueError):
    """A tier cannot run; the message says which one and why."""


@dataclass
class ResolvedTiers:
    high: Any
    low: List[Any] = field(default_factory=list)


def _entry_for(ref: H2LModelRef, config_manager: Any, engine: Optional[Any]) -> Any:
    """The provider entry the engine actually runs with, for ``ref.provider``.

    Three sources, in order: the stored config with environment overrides
    applied (the engine initializes from exactly this effective config, and
    ``apply_env_overrides`` returns a deep copy, so env-only entries such as
    ``DIGITALOCEAN_INFERENCE_KEY`` never appear in the stored config); the
    stored config itself; and finally a live instance in ``engine.providers``
    (an entry created by session flags or an earlier init).
    """
    config = config_manager.get_config()
    stored = getattr(config, "providers", None) or {}
    effective_providers = stored
    try:
        effective_providers = apply_env_overrides(config).providers or stored
    except Exception as exc:  # overrides must never block a run
        logger.debug("H2L: env overrides unavailable: %s", exc)
    entry = effective_providers.get(ref.provider) or stored.get(ref.provider)
    if entry is not None:
        return entry

    live = (getattr(engine, "providers", None) or {}) if engine is not None else {}
    instance = live.get(ref.provider) if isinstance(live, dict) else None
    if instance is not None:
        api_key = getattr(instance, "api_key", "") or ""
        kwargs = {
            k: v
            for k, v in (getattr(instance, "config", None) or {}).items()
            if v is not None and k not in ("api_key", "model")
        }
        for candidate in (kwargs, {}):
            try:
                return ProviderConfig(api_key=api_key, model=ref.model, **candidate)
            except Exception:
                continue

    prefix = "OMNIMANCER_" + ref.provider.upper().replace("-", "_")
    raise H2LConfigError(
        f"H2L: provider entry '{ref.provider}' is not configured "
        f"(see /providers, or set {prefix}_API_KEY)"
    )


def resolve(
    ref: H2LModelRef,
    config_manager: Any,
    engine: Optional[Any],
    min_output_tokens: Optional[int] = None,
) -> Any:
    """Build a fresh, tool-capable provider for ``ref``.

    ``min_output_tokens`` is a floor on the instance's ``max_tokens``: the
    providers default to 4096, which truncates plans and verdict feedback (the
    first live runs lost whole plans to it). A larger configured value wins.

    Raises :class:`H2LConfigError` when the entry is not configured, the
    factory cannot build it, or the provider reports no tool support.
    """
    entry = _entry_for(ref, config_manager, engine)
    update: dict = {"model": ref.model}
    if min_output_tokens is not None:
        current = getattr(entry, "max_tokens", None) or 0
        update["max_tokens"] = max(int(current), int(min_output_tokens))
    effective = entry.model_copy(update=update)
    try:
        provider = ProviderFactory.create_provider(
            ref.provider, effective, config_manager
        )
    except Exception as exc:
        raise H2LConfigError(f"H2L: could not initialize {ref.label}: {exc}") from exc

    try:
        supports_tools = bool(provider.supports_tools())
    except Exception:  # a provider that cannot even answer is not usable
        supports_tools = False
    if not supports_tools:
        raise H2LConfigError(
            f"H2L: provider '{ref.provider}' does not support tool calling; use "
            "an openai-compatible alias pointed at its OpenAI endpoint "
            "(see docs/provider-setup.md)"
        )
    # Belt and braces: the factory must never return the session's object.
    shared = getattr(engine, "providers", None) if engine is not None else None
    if isinstance(shared, dict) and any(provider is p for p in shared.values()):
        logger.warning(
            "H2L: factory returned the session provider for %s; results may "
            "interleave with the main conversation",
            ref.label,
        )
    return provider


def validate_tiers(
    config: H2LConfig, config_manager: Any, engine: Optional[Any]
) -> ResolvedTiers:
    """Resolve the high ref and every low ref before any model call (PRD §1)."""
    high = resolve(config.high, config_manager, engine, config.high_max_tokens)
    low = [
        resolve(ref, config_manager, engine, config.low_max_tokens)
        for ref in config.low
    ]
    return ResolvedTiers(high=high, low=low)
