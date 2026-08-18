"""Session token/cost accumulation, shared by headless and interactive modes."""

from typing import Any, Dict

from ..core.models import ChatResponse


class TokenAccumulator:
    """Tracks cumulative token usage across multiple API calls."""

    def __init__(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_cost = 0.0
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0

    def add(self, response: ChatResponse) -> None:
        self._input_tokens += response.input_tokens or 0
        self._output_tokens += response.output_tokens or 0
        self._total_cost += response.cost_estimate or 0.0
        # getattr: some callers feed response-shaped objects predating the
        # cache-token fields.
        self._cache_read_tokens += getattr(response, "cache_read_input_tokens", 0) or 0
        self._cache_creation_tokens += (
            getattr(response, "cache_creation_input_tokens", 0) or 0
        )

    def restore(self, totals: Dict[str, Any]) -> None:
        """Seed the counters from a saved `total` dict (checkpoint resume)."""
        self._input_tokens = int(totals.get("input_tokens") or 0)
        self._output_tokens = int(totals.get("output_tokens") or 0)
        self._total_cost = float(totals.get("total_cost_usd") or 0.0)
        self._cache_read_tokens = int(totals.get("cache_read_input_tokens") or 0)
        self._cache_creation_tokens = int(
            totals.get("cache_creation_input_tokens") or 0
        )

    @property
    def total(self) -> Dict[str, Any]:
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_cost_usd": self._total_cost,
            "cache_read_input_tokens": self._cache_read_tokens,
            "cache_creation_input_tokens": self._cache_creation_tokens,
        }
