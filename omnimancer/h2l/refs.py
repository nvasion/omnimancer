"""Resolve H2L model refs into isolated provider instances (PRD §1).

Entry lookup and instance building live in
:mod:`omnimancer.core.provider_resolver`, which every surface shares; this
module is the H2L-flavoured wrapper that turns a resolution failure into an
:class:`H2LConfigError` naming the tier. The rule that makes parallel tiers
safe is the resolver's: never hand out an object from ``engine.providers``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..core.provider_resolver import (
    EntryNotConfigured,
    ProviderInitFailed,
    ProviderResolutionError,
    ToolsUnsupported,
    resolve_provider,
)
from .models import H2LConfig, H2LModelRef

logger = logging.getLogger(__name__)


class H2LConfigError(ValueError):
    """A tier cannot run; the message says which one and why."""


@dataclass
class ResolvedTiers:
    high: Any
    low: List[Any] = field(default_factory=list)


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
    try:
        return resolve_provider(
            ref.provider,
            config_manager,
            ref.model,
            engine,
            min_output_tokens=min_output_tokens,
            require_tools=True,
        )
    except EntryNotConfigured as exc:
        raise H2LConfigError(f"H2L: {exc}") from exc
    except ToolsUnsupported as exc:
        raise H2LConfigError(
            f"H2L: {exc}; use an openai-compatible alias pointed at its "
            "OpenAI endpoint (see docs/provider-setup.md)"
        ) from exc
    except ProviderInitFailed as exc:
        raise H2LConfigError(
            f"H2L: could not initialize {ref.label}: {exc.cause}"
        ) from exc
    except ProviderResolutionError as exc:  # future resolver failure modes
        raise H2LConfigError(f"H2L: {exc}") from exc


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
