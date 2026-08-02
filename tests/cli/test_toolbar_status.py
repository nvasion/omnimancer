"""Toolbar status provider tests (field-reported: stale model after /switch).

The toolbar must reflect the LIVE session provider, not the stored config
defaults — /switch swaps engine.current_provider and the very next prompt
must show it.
"""

from unittest.mock import MagicMock

from omnimancer.cli.interface import CommandLineInterface
from omnimancer.cli.usage import TokenAccumulator


def _iface_with_provider(entry_name: str, model: str) -> CommandLineInterface:
    cli = CommandLineInterface.__new__(CommandLineInterface)
    provider = MagicMock()
    provider.model = model
    provider.get_provider_name.return_value = "fallback-name"
    cli.engine = MagicMock()
    cli.engine.current_provider = provider
    cli.engine.providers = {"other": MagicMock(), entry_name: provider}
    cli.usage = TokenAccumulator()
    cli.read_only = False
    return cli


class TestToolbarStatus:
    def test_shows_live_entry_name_and_model(self):
        cli = _iface_with_provider("gateway", "qwen3-coder-30b")
        assert cli._toolbar_status() == "gateway/qwen3-coder-30b · $0.00"

    def test_reflects_switch_immediately(self):
        cli = _iface_with_provider("gateway", "qwen3-coder-30b")
        switched = MagicMock()
        switched.model = "gpt-oss-120b"
        switched.get_provider_name.return_value = "fallback-name"
        cli.engine.providers["gateway2"] = switched
        cli.engine.current_provider = switched
        assert cli._toolbar_status() == "gateway2/gpt-oss-120b · $0.00"

    def test_read_only_badge(self):
        cli = _iface_with_provider("gateway", "m")
        cli.read_only = True
        assert cli._toolbar_status() == "gateway/m · $0.00 · read-only"

    def test_no_provider_returns_none(self):
        cli = _iface_with_provider("gateway", "m")
        cli.engine.current_provider = None
        assert cli._toolbar_status() is None

    def test_never_raises(self):
        cli = CommandLineInterface.__new__(CommandLineInterface)
        cli.engine = None  # attribute access will explode inside
        assert cli._toolbar_status() is None
