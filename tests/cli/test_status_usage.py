"""Session usage totals (shared TokenAccumulator) and /status enrichment."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from rich.console import Console

from omnimancer.cli.display import DisplayMixin
from omnimancer.cli.usage import TokenAccumulator


def _response(input_tokens, output_tokens, cost):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_estimate=cost,
    )


class _Harness(DisplayMixin):
    def __init__(self):
        self.console = Console(record=True, width=100, force_terminal=False)
        self.usage = TokenAccumulator()
        self.engine = MagicMock()
        self.engine.get_conversation_summary.return_value = {
            "message_count": 3,
            "current_provider": "local",
            "current_model": "qwen3-coder-30b",
            "session_id": "s-1",
        }
        self.engine.get_current_model_info.return_value = {"name": "qwen3-coder-30b"}

    def _session_approval_mode_name(self):
        return "accept-edits"

    @property
    def text(self):
        return self.console.export_text(clear=False)


class TestSharedAccumulator:
    def test_moved_module_importable_from_both_homes(self):
        from omnimancer.cli.headless import TokenAccumulator as headless_acc
        from omnimancer.cli.usage import TokenAccumulator as usage_acc

        assert headless_acc is usage_acc

    def test_totals(self):
        acc = TokenAccumulator()
        acc.add(_response(10, 5, 0.01))
        acc.add(_response(20, 10, 0.02))
        totals = acc.total
        assert totals["input_tokens"] == 30
        assert totals["output_tokens"] == 15
        assert round(totals["total_cost_usd"], 4) == 0.03


class TestTokenStatusAccumulates:
    def test_show_token_status_feeds_session_totals(self):
        harness = _Harness()
        harness._show_token_status(_response(100, 50, 0.05))
        harness._show_token_status(_response(200, 100, 0.10))
        assert harness.usage.total["input_tokens"] == 300
        assert harness.usage.total["output_tokens"] == 150


class TestStatusEnrichment:
    def test_status_shows_session_totals_and_mode(self):
        harness = _Harness()
        harness._show_token_status(_response(100, 50, 0.05))
        harness._show_status()
        assert "100" in harness.text
        assert "50" in harness.text
        assert "accept-edits" in harness.text
