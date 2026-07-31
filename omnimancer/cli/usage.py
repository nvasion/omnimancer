"""Session token/cost accumulation, shared by headless and interactive modes."""

from typing import Any, Dict

from ..core.models import ChatResponse


class TokenAccumulator:
    """Tracks cumulative token usage across multiple API calls."""

    def __init__(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_cost = 0.0

    def add(self, response: ChatResponse) -> None:
        self._input_tokens += response.input_tokens or 0
        self._output_tokens += response.output_tokens or 0
        self._total_cost += response.cost_estimate or 0.0

    @property
    def total(self) -> Dict[str, Any]:
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_cost_usd": self._total_cost,
        }
