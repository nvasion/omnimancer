"""The one provider-resolution primitive every surface shares.

Before this module each caller re-derived "which entry is in force" — the
engine applied environment overrides to a local variable and discarded it,
H2L re-applied them per ref, the prompt enhancer forgot them entirely, and
subagents mutated the shared session provider instead of resolving at all.
A provider configured purely through OMNIMANCER_<PROVIDER>_API_KEY therefore
existed for the engine and for nobody else.
"""

from unittest.mock import MagicMock, patch

import pytest

from omnimancer.core.models import Config, ProviderConfig
from omnimancer.core.provider_resolver import (
    EntryNotConfigured,
    ProviderInitFailed,
    ToolsUnsupported,
    effective_config,
    effective_providers,
    resolve_entry,
    resolve_provider,
)

_FACTORY = "omnimancer.core.provider_resolver.ProviderFactory.create_provider"


def _config():
    return Config(
        default_provider="claude",
        providers={
            "claude": ProviderConfig(api_key="k", model="claude-sonnet-5"),
            "gateway": ProviderConfig(
                api_key="", model="qwen3-8b", provider_type="openai-compatible"
            ),
        },
        storage_path="/tmp/omn-resolver",
    )


def _config_manager(config=None):
    cm = MagicMock()
    cm.get_config.return_value = config or _config()
    return cm


def _tool_provider(supports=True):
    provider = MagicMock()
    provider.supports_tools.return_value = supports
    return provider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for provider in ("BEDROCK", "CLAUDE", "GATEWAY", "LOCAL", "NOPE"):
        for suffix in ("API_KEY", "BASE_URL", "MODEL"):
            monkeypatch.delenv(f"OMNIMANCER_{provider}_{suffix}", raising=False)
    monkeypatch.delenv("OMNIMANCER_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    yield


class TestEffectiveConfig:
    def test_applies_env_overrides(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_BEDROCK_API_KEY", "bedrock-key")
        effective = effective_config(_config_manager())
        assert effective.providers["bedrock"].api_key == "bedrock-key"

    def test_accepts_a_config_directly(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_BEDROCK_API_KEY", "bedrock-key")
        effective = effective_config(_config())
        assert "bedrock" in effective.providers

    def test_does_not_mutate_the_stored_config(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_BEDROCK_API_KEY", "bedrock-key")
        cm = _config_manager()
        effective_config(cm)
        assert "bedrock" not in cm.get_config().providers

    def test_never_persists(self, monkeypatch):
        """Env values are ephemeral: saving them would write secrets to disk."""
        monkeypatch.setenv("OMNIMANCER_BEDROCK_API_KEY", "bedrock-key")
        cm = _config_manager()
        effective_config(cm)
        cm.save_config.assert_not_called()

    def test_env_default_provider_wins(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_DEFAULT_PROVIDER", "bedrock")
        monkeypatch.setenv("OMNIMANCER_BEDROCK_API_KEY", "bedrock-key")
        assert effective_config(_config_manager()).default_provider == "bedrock"

    def test_falls_back_to_stored_config_when_overrides_fail(self):
        """Overrides must never block a run."""
        cm = _config_manager()
        with patch(
            "omnimancer.core.provider_resolver.apply_env_overrides",
            side_effect=RuntimeError("boom"),
        ):
            effective = effective_config(cm)
        assert "claude" in effective.providers

    def test_effective_providers_returns_the_mapping(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_BEDROCK_API_KEY", "bedrock-key")
        names = set(effective_providers(_config_manager()))
        assert {"claude", "gateway", "bedrock"} <= names


class TestResolveEntry:
    def test_stored_entry(self):
        entry = resolve_entry("claude", _config_manager())
        assert entry.model == "claude-sonnet-5"

    def test_env_only_entry(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_BEDROCK_API_KEY", "bedrock-key")
        entry = resolve_entry("bedrock", _config_manager())
        assert entry.api_key == "bedrock-key"

    def test_env_key_applies_to_stored_entry(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_GATEWAY_BASE_URL", "http://env-host:8000/v1")
        entry = resolve_entry("gateway", _config_manager())
        assert entry.base_url == "http://env-host:8000/v1"

    def test_live_engine_instance_is_the_last_resort(self):
        live = MagicMock()
        live.api_key = "live-key"
        live.config = {"base_url": "http://live:9000/v1"}
        engine = MagicMock()
        engine.providers = {"local": live}
        entry = resolve_entry("local", _config_manager(), engine)
        assert entry.api_key == "live-key"

    def test_unknown_entry_names_the_env_var(self):
        with pytest.raises(EntryNotConfigured) as exc:
            resolve_entry("nope", _config_manager())
        assert exc.value.provider == "nope"
        assert exc.value.env_var == "OMNIMANCER_NOPE_API_KEY"


class TestResolveProvider:
    def test_model_override_does_not_touch_the_stored_entry(self):
        cm = _config_manager()
        with patch(_FACTORY, return_value=_tool_provider()) as create:
            resolve_provider("claude", cm, model="claude-opus-5")
        assert create.call_args[0][1].model == "claude-opus-5"
        assert cm.get_config().providers["claude"].model == "claude-sonnet-5"

    def test_min_output_tokens_is_a_floor(self):
        cm = _config_manager()
        with patch(_FACTORY, return_value=_tool_provider()) as create:
            resolve_provider("claude", cm, model="m", min_output_tokens=16384)
        assert create.call_args[0][1].max_tokens == 16384

    def test_larger_configured_max_tokens_wins(self):
        config = _config()
        config.providers["claude"].max_tokens = 32000
        cm = _config_manager(config)
        with patch(_FACTORY, return_value=_tool_provider()) as create:
            resolve_provider("claude", cm, model="m", min_output_tokens=16384)
        assert create.call_args[0][1].max_tokens == 32000

    def test_extra_overrides_are_applied(self):
        """The prompt enhancer needs its own temperature on the entry."""
        cm = _config_manager()
        with patch(_FACTORY, return_value=_tool_provider()) as create:
            resolve_provider("claude", cm, model="m", overrides={"temperature": 0.4})
        assert create.call_args[0][1].temperature == 0.4

    def test_model_is_optional(self):
        cm = _config_manager()
        with patch(_FACTORY, return_value=_tool_provider()) as create:
            resolve_provider("claude", cm)
        assert create.call_args[0][1].model == "claude-sonnet-5"

    def test_factory_failure_is_wrapped(self):
        with patch(_FACTORY, side_effect=RuntimeError("no credentials")):
            with pytest.raises(ProviderInitFailed) as exc:
                resolve_provider("claude", _config_manager(), model="m")
        assert "no credentials" in str(exc.value)

    def test_require_tools_rejects_a_text_only_provider(self):
        with patch(_FACTORY, return_value=_tool_provider(supports=False)):
            with pytest.raises(ToolsUnsupported):
                resolve_provider(
                    "claude", _config_manager(), model="m", require_tools=True
                )

    def test_require_tools_treats_an_unanswerable_provider_as_unusable(self):
        provider = MagicMock()
        provider.supports_tools.side_effect = RuntimeError("boom")
        with patch(_FACTORY, return_value=provider):
            with pytest.raises(ToolsUnsupported):
                resolve_provider(
                    "claude", _config_manager(), model="m", require_tools=True
                )

    def test_tool_support_not_checked_unless_required(self):
        with patch(_FACTORY, return_value=_tool_provider(supports=False)):
            assert resolve_provider("claude", _config_manager(), model="m") is not None

    def test_env_only_provider_resolves_end_to_end(self, monkeypatch):
        """The reported failure: a provider configured only via the env."""
        monkeypatch.setenv("OMNIMANCER_BEDROCK_API_KEY", "bedrock-key")
        cm = _config_manager()
        with patch(_FACTORY, return_value=_tool_provider()) as create:
            resolve_provider(
                "bedrock", cm, model="moonshotai.kimi-k2-instruct", require_tools=True
            )
        name, entry, passed_cm = create.call_args[0]
        assert name == "bedrock"
        assert entry.api_key == "bedrock-key"
        assert entry.model == "moonshotai.kimi-k2-instruct"
        assert passed_cm is cm
