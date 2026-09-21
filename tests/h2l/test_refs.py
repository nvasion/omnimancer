"""H2L model-ref resolution builds isolated provider instances (PRD-h2l §1)."""

from unittest.mock import MagicMock, patch

import pytest

from omnimancer.core.models import Config, ProviderConfig
from omnimancer.h2l.models import H2LConfig, H2LModelRef
from omnimancer.h2l.refs import H2LConfigError, resolve, validate_tiers


def _config_manager(providers=None, default="claude"):
    cm = MagicMock()
    cm.get_config.return_value = Config(
        default_provider=default,
        providers=providers
        or {
            "claude": ProviderConfig(api_key="k", model="claude-sonnet-5"),
            "gateway": ProviderConfig(
                api_key="", model="qwen3-8b", provider_type="openai-compatible"
            ),
        },
        storage_path="/tmp/omn-h2l-refs",
    )
    return cm


def _tool_provider(supports=True):
    provider = MagicMock()
    provider.supports_tools.return_value = supports
    return provider


class TestResolve:
    def test_builds_instance_from_model_copy_not_shared_config(self):
        cm = _config_manager()
        engine = MagicMock()
        engine.providers = {"claude": MagicMock()}
        fresh = _tool_provider()
        ref = H2LModelRef(provider="claude", model="claude-opus-5")
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider", return_value=fresh
        ) as create:
            provider = resolve(ref, cm, engine)
        assert provider is fresh
        assert provider is not engine.providers["claude"]
        name, effective, passed_cm = create.call_args[0]
        assert name == "claude"
        assert effective.model == "claude-opus-5"
        assert passed_cm is cm
        # The stored config entry is untouched (the env pass or a later
        # /switch must not see the H2L tier's model).
        assert cm.get_config().providers["claude"].model == "claude-sonnet-5"

    def test_ref_model_wins_over_config_model(self):
        # Simulates the env-override gotcha: even if the stored entry's model
        # was rewritten after flags were parsed, the ref's model is what runs.
        cm = _config_manager()
        cm.get_config().providers["gateway"].model = "rewritten-by-env"
        ref = H2LModelRef(provider="gateway", model="qwen3-8b")
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(),
        ) as create:
            resolve(ref, cm, MagicMock())
        assert create.call_args[0][1].model == "qwen3-8b"

    def test_output_token_floor_applied(self):
        # Providers default to max_tokens=4096; the live runs lost whole
        # plans to that cap. H2L tiers get a floor.
        cm = _config_manager()
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(),
        ) as create:
            resolve(
                H2LModelRef(provider="claude", model="m"),
                cm,
                None,
                min_output_tokens=16384,
            )
        assert create.call_args[0][1].max_tokens == 16384
        assert cm.get_config().providers["claude"].max_tokens is None

    def test_larger_configured_cap_wins_over_floor(self):
        cm = _config_manager()
        cm.get_config().providers["claude"].max_tokens = 64000
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(),
        ) as create:
            resolve(H2LModelRef(provider="claude", model="m"), cm, None, 16384)
        assert create.call_args[0][1].max_tokens == 64000

    def test_no_floor_leaves_cap_untouched(self):
        cm = _config_manager()
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(),
        ) as create:
            resolve(H2LModelRef(provider="claude", model="m"), cm, None, None)
        assert create.call_args[0][1].max_tokens is None

    def test_unknown_entry_names_it(self, monkeypatch):
        monkeypatch.delenv("OMNIMANCER_NOPE_API_KEY", raising=False)
        with pytest.raises(H2LConfigError) as exc:
            resolve(H2LModelRef(provider="nope", model="m"), _config_manager(), None)
        message = str(exc.value)
        assert "provider entry 'nope' is not configured" in message
        assert "OMNIMANCER_NOPE_API_KEY" in message

    def test_env_only_provider_resolves(self, monkeypatch):
        # Env-materialized entries never reach the stored config (the engine
        # works on a deep copy), so the resolver must see the same effective
        # config the engine used — the bug hit on the first live run.
        monkeypatch.setenv("DIGITALOCEAN_INFERENCE_KEY", "do-key")
        cm = _config_manager()  # no "digitalocean" entry on disk
        ref = H2LModelRef(provider="digitalocean", model="qwen3.5-397b-a17b")
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(),
        ) as create:
            resolve(ref, cm, MagicMock())
        name, effective, _ = create.call_args[0]
        assert name == "digitalocean"
        assert effective.model == "qwen3.5-397b-a17b"
        assert effective.api_key == "do-key"
        # The stored config is still untouched.
        assert "digitalocean" not in cm.get_config().providers

    def test_env_key_applies_to_stored_entry(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_GATEWAY_BASE_URL", "http://env-host:8000/v1")
        cm = _config_manager()
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(),
        ) as create:
            resolve(H2LModelRef(provider="gateway", model="qwen3-8b"), cm, None)
        assert create.call_args[0][1].base_url == "http://env-host:8000/v1"

    def test_live_engine_provider_used_when_no_entry(self, monkeypatch):
        monkeypatch.delenv("OMNIMANCER_LOCAL_API_KEY", raising=False)
        monkeypatch.delenv("OMNIMANCER_LOCAL_BASE_URL", raising=False)
        live = MagicMock()
        live.api_key = "live-key"
        live.config = {"base_url": "http://live:1234/v1", "provider_type": "openai"}
        engine = MagicMock()
        engine.providers = {"local": live}
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(),
        ) as create:
            resolve(H2LModelRef(provider="local", model="m"), _config_manager(), engine)
        name, effective, _ = create.call_args[0]
        assert name == "local"
        assert effective.base_url == "http://live:1234/v1"
        assert effective.api_key == "live-key"
        assert effective.model == "m"

    def test_non_tool_provider_rejected_with_alias_hint(self):
        cm = _config_manager(
            providers={"ollama": ProviderConfig(api_key="", model="qwen2.5")}
        )
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(supports=False),
        ):
            with pytest.raises(H2LConfigError) as exc:
                resolve(H2LModelRef(provider="ollama", model="qwen2.5"), cm, None)
        message = str(exc.value)
        assert "does not support tool calling" in message
        assert "openai-compatible alias" in message

    def test_factory_failure_surfaces_as_config_error(self):
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            side_effect=RuntimeError("no key"),
        ):
            with pytest.raises(H2LConfigError, match="no key"):
                resolve(
                    H2LModelRef(provider="claude", model="x"), _config_manager(), None
                )


class TestValidateTiers:
    def test_validates_high_and_every_low_before_any_model_call(self):
        cfg = H2LConfig(high="claude:a", low=["gateway:b", "gateway:c"])
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(),
        ) as create:
            resolved = validate_tiers(cfg, _config_manager(), None)
        assert create.call_count == 3
        assert resolved.high is not None
        assert len(resolved.low) == 2
        caps = [call[0][1].max_tokens for call in create.call_args_list]
        assert caps == [16384, 8192, 8192]  # high floor, then the low pool

    def test_bad_low_entry_fails_whole_validation(self):
        cfg = H2LConfig(high="claude:a", low=["gateway:b", "missing:c"])
        with patch(
            "omnimancer.h2l.refs.ProviderFactory.create_provider",
            return_value=_tool_provider(),
        ):
            with pytest.raises(H2LConfigError, match="missing"):
                validate_tiers(cfg, _config_manager(), None)
