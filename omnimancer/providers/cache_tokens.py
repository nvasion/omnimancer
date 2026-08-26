"""Cross-provider prompt-cache helpers.

Vendors expose prompt caching differently:

- Anthropic API: explicit ``cache_control`` breakpoints (providers/claude.py).
- OpenAI-compatible APIs (OpenAI, Azure, xAI, OpenRouter, ...): caching is
  automatic server-side; the only integration is reading
  ``usage.prompt_tokens_details.cached_tokens``.
- Gemini / Vertex: implicit caching; ``usageMetadata.cachedContentTokenCount``.
- Bedrock Converse: explicit ``cachePoint`` blocks, model-gated
  (providers/bedrock.py).

``OMNIMANCER_PROMPT_CACHE=0`` disables every request-side marker; the
usage-parsing helpers are unconditional (reading what the API reports is
always safe).
"""

import os
from typing import Any, Dict, Optional


def prompt_cache_enabled() -> bool:
    raw = os.environ.get("OMNIMANCER_PROMPT_CACHE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def openai_cached_tokens(usage: Dict[str, Any]) -> Optional[int]:
    """Cached prompt tokens from an OpenAI-shape usage block, if reported.

    Falls back to a top-level ``cache_read_input_tokens`` — DigitalOcean's
    gateway reports Anthropic/OpenAI cache hits there instead of inside
    ``prompt_tokens_details``.
    """
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = _as_int(details.get("cached_tokens"))
        if cached is not None:
            return cached
    return _as_int(usage.get("cache_read_input_tokens"))


def gemini_cached_tokens(usage_metadata: Dict[str, Any]) -> Optional[int]:
    """Cached prompt tokens from a Gemini/Vertex usageMetadata block."""
    return _as_int(usage_metadata.get("cachedContentTokenCount"))
