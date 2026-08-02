"""Tests for CLI --provider/--model/--base-url overrides in interactive mode."""

import json
import os
import tempfile

import pytest

from omnimancer.cli.interface import apply_session_overrides
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.models import ProviderConfig


def _config_manager(default="digitalocean", model="llama3.3-70b-instruct"):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "config.json")
    json.dump(
        {
            "default_provider": default,
            "providers": {default: {"api_key": "k", "model": model}},
            "storage_path": tmp,
        },
        open(path, "w"),
    )
    return ConfigManager(path)


class TestApplySessionOverrides:
    def test_model_override_applied(self):
        cm = _config_manager()
        apply_session_overrides(cm, provider=None, model="arcee-trinity-large-thinking")
        assert cm.get_config().providers["digitalocean"].model == (
            "arcee-trinity-large-thinking"
        )

    def test_provider_and_model_override(self):
        cm = _config_manager()
        apply_session_overrides(cm, provider="openai", model="gpt-4o")
        cfg = cm.get_config()
        assert cfg.default_provider == "openai"
        assert cfg.providers["openai"].model == "gpt-4o"

    def test_base_url_override_strips_slash(self):
        cm = _config_manager()
        apply_session_overrides(cm, provider="digitalocean", base_url="http://x/v1/")
        assert cm.get_config().providers["digitalocean"].base_url == "http://x/v1"

    def test_no_overrides_is_noop(self):
        cm = _config_manager()
        apply_session_overrides(cm, provider=None, model=None, base_url=None)
        cfg = cm.get_config()
        assert cfg.default_provider == "digitalocean"
        assert cfg.providers["digitalocean"].model == "llama3.3-70b-instruct"

    def test_creates_provider_entry_when_missing(self):
        cm = _config_manager()
        apply_session_overrides(cm, provider="openrouter", model="anthropic/claude")
        cfg = cm.get_config()
        assert "openrouter" in cfg.providers
        assert cfg.providers["openrouter"].model == "anthropic/claude"


@pytest.mark.asyncio
async def test_headless_provider_override_does_not_persist(monkeypatch, tmp_path):
    import omnimancer.cli.headless as headless
    import omnimancer.core.engine as engine_module

    config_path = tmp_path / "config.json"
    config_manager = ConfigManager(str(config_path))
    config = config_manager.get_config()
    config.default_provider = "stored"
    config.providers = {"stored": ProviderConfig(api_key="k", model="m")}
    config.storage_path = str(tmp_path)
    config_manager.save_config(config)
    original = config_path.read_bytes()

    class FakeCoreEngine:
        def __init__(self, config_manager):
            self.config_manager = config_manager

        async def initialize_providers(self):
            return None

    async def fake_run(self, prompt):
        return 0

    monkeypatch.setattr(engine_module, "CoreEngine", FakeCoreEngine)
    monkeypatch.setattr(headless.HeadlessRunner, "run", fake_run)

    result = await headless.run_headless(
        "hello", config_path=str(config_path), provider="openai"
    )

    assert result == 0
    assert config_path.read_bytes() == original


class TestDangerouslySkipPermissionsWiring:
    """--dangerously-skip-permissions must skip approvals in interactive mode too.

    Regression: the flag was honoured in headless (-p) mode but silently ignored
    in the interactive REPL (only --no-approval worked there).
    """

    def _invoke(self, argv):
        import sys
        from unittest.mock import MagicMock

        import pytest

        import omnimancer.cli.interface as iface

        captured = {}

        class _FakeCLI:
            def __init__(
                self,
                engine,
                no_approval=False,
                full_trust=False,
                **kwargs,
            ):
                captured["no_approval"] = no_approval
                captured["full_trust"] = full_trust

            def start(self):
                return None

        orig = {
            "CommandLineInterface": iface.CommandLineInterface,
            "CoreEngine": iface.CoreEngine,
            "ConfigManager": iface.ConfigManager,
            "apply_session_overrides": iface.apply_session_overrides,
            "argv": sys.argv,
        }
        iface.CommandLineInterface = _FakeCLI
        iface.CoreEngine = lambda cm: MagicMock()
        iface.ConfigManager = lambda c: MagicMock()
        iface.apply_session_overrides = lambda *a, **k: None
        sys.argv = argv
        try:
            with pytest.raises(SystemExit):
                iface.main()
        finally:
            iface.CommandLineInterface = orig["CommandLineInterface"]
            iface.CoreEngine = orig["CoreEngine"]
            iface.ConfigManager = orig["ConfigManager"]
            iface.apply_session_overrides = orig["apply_session_overrides"]
            sys.argv = orig["argv"]
        return captured

    def test_flag_enables_no_approval_interactive(self):
        captured = self._invoke(["omn", "--dangerously-skip-permissions"])
        assert captured["no_approval"] is True
        assert captured["full_trust"] is True

    def test_no_approval_flag_enables_interactive(self):
        captured = self._invoke(["omn", "--no-approval"])
        assert captured["no_approval"] is True
        assert captured["full_trust"] is False

    def test_default_keeps_approvals_on(self):
        captured = self._invoke(["omn"])
        assert captured["no_approval"] is False
