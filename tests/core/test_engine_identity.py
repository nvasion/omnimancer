"""Tests for CoreEngine.runtime_identity()."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from omnimancer.core.engine import CoreEngine


def _bare_engine() -> CoreEngine:
    engine = CoreEngine.__new__(CoreEngine)
    engine.config_manager = MagicMock()
    engine.providers = {}
    engine.current_provider = None
    return engine


class TestRuntimeIdentity:
    """Tests for CoreEngine.runtime_identity()."""

    def test_identity_from_live_provider(self):
        engine = _bare_engine()
        prov = SimpleNamespace(model="qwen3.5-9b")
        engine.providers = {"gateway": prov}
        engine.current_provider = prov
        assert engine.runtime_identity() == ("gateway", "qwen3.5-9b")

    def test_identity_config_fallback_when_no_provider(self):
        engine = _bare_engine()
        engine.current_provider = None
        engine.config_manager.get_config.return_value = SimpleNamespace(
            default_provider="p",
            providers={"p": SimpleNamespace(model="disk-model")},
        )
        assert engine.runtime_identity() == ("p", "disk-model")

    def test_identity_overlay_beats_disk_config(self):
        engine = _bare_engine()
        prov = SimpleNamespace(model="env-model")
        engine.providers = {"gateway": prov}
        engine.current_provider = prov
        engine.config_manager.get_config.return_value = SimpleNamespace(
            default_provider="gateway",
            providers={"gateway": SimpleNamespace(model="disk-model")},
        )
        assert engine.runtime_identity() == ("gateway", "env-model")
