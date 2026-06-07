"""Tests for the /config command handlers (set / set-provider / remove-provider)."""

import os
import tempfile

import pytest

from omnimancer.cli.command_dispatch import CommandDispatchMixin
from omnimancer.core.config_manager import ConfigManager


class _Engine:
    def __init__(self, config_manager):
        self.config_manager = config_manager


class _Harness(CommandDispatchMixin):
    """Minimal host exposing only what the config handlers need."""

    def __init__(self, config_manager):
        self.engine = _Engine(config_manager)
        self.messages = {"error": [], "info": [], "success": [], "warning": []}

    def _show_error(self, m):
        self.messages["error"].append(m)

    def _show_info(self, m):
        self.messages["info"].append(m)

    def _show_success(self, m):
        self.messages["success"].append(m)

    def _show_warning(self, m):
        self.messages["warning"].append(m)


@pytest.fixture
def harness():
    tmp = tempfile.mkdtemp()
    cm = ConfigManager(os.path.join(tmp, "config.json"))
    return _Harness(cm), cm


class TestSetProvider:
    async def test_creates_provider_with_all_fields(self, harness):
        h, cm = harness
        await h._handle_config_set_provider(
            [
                "digitalocean",
                "--api-key",
                "do-secret",
                "--base-url",
                "https://inference.do-ai.run/v1",
                "--model",
                "llama3.3-70b-instruct",
            ]
        )
        cfg = cm.get_provider_config("digitalocean")
        assert cfg.base_url == "https://inference.do-ai.run/v1"
        assert cfg.model == "llama3.3-70b-instruct"
        # API key is encrypted at rest but decrypts back to plaintext.
        assert cfg.api_key != "do-secret"
        assert cm.get_api_key("digitalocean") == "do-secret"
        assert h.messages["success"]

    async def test_requires_a_field(self, harness):
        h, cm = harness
        await h._handle_config_set_provider(["openai"])
        assert h.messages["error"]

    async def test_rejects_unknown_flag(self, harness):
        h, cm = harness
        await h._handle_config_set_provider(["openai", "--bogus", "x"])
        assert h.messages["error"]


class TestConfigSet:
    async def test_set_provider_field_persists_and_coerces(self, harness):
        h, cm = harness
        await h._handle_config_set("providers.openai.api_key", "sk-123")
        await h._handle_config_set("providers.openai.base_url", "https://proxy/v1")
        await h._handle_config_set("providers.openai.max_tokens", "8192")
        cfg = cm.get_provider_config("openai")
        assert cfg.base_url == "https://proxy/v1"
        assert cfg.max_tokens == 8192
        assert isinstance(cfg.max_tokens, int)

    async def test_set_default_provider(self, harness):
        h, cm = harness
        await h._handle_config_set("providers.claude.api_key", "sk-abc")
        await h._handle_config_set("default_provider", "claude")
        assert cm.get_config().default_provider == "claude"

    async def test_set_unknown_default_provider_errors(self, harness):
        h, cm = harness
        await h._handle_config_set("default_provider", "nope")
        assert h.messages["error"]

    async def test_unknown_field_errors(self, harness):
        h, cm = harness
        await h._handle_config_set("providers.openai.not_a_field", "x")
        assert h.messages["error"]

    async def test_unsupported_key_errors(self, harness):
        h, cm = harness
        await h._handle_config_set("random_key", "x")
        assert h.messages["error"]


class TestRemoveProvider:
    async def test_remove_repoints_default(self, harness):
        h, cm = harness
        await h._handle_config_set_provider(["openai", "--api-key", "a"])
        await h._handle_config_set_provider(["claude", "--api-key", "b"])
        await h._handle_config_set("default_provider", "openai")

        await h._handle_config_remove_provider(["openai"])
        cfg = cm.get_config()
        assert "openai" not in cfg.providers
        # Default repointed to the remaining provider.
        assert cfg.default_provider == "claude"

    async def test_remove_missing_provider_errors(self, harness):
        h, cm = harness
        await h._handle_config_remove_provider(["ghost"])
        assert h.messages["error"]
