"""Tests for CLI --provider/--model/--base-url overrides in interactive mode."""

import json
import os
import tempfile

from omnimancer.cli.interface import apply_session_overrides
from omnimancer.core.config_manager import ConfigManager


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
