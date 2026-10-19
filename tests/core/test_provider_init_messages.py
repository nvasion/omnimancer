"""Initialization reporting: no noise for incomplete entries, real causes.

An entry materialized from ``OMNIMANCER_<PROVIDER>_API_KEY`` carries no model
when the model comes from somewhere else (an H2L ``--high bedrock:<model>``
ref supplies its own). Treating that as a failed initialization produced an
ERROR line for a provider that was about to work fine, and the engine's
default-provider message guessed "missing or invalid API key?" even when the
entry did not exist at all.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from omnimancer.core.models import Config, ProviderConfig
from omnimancer.core.provider_initializer import ProviderInitializer


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for suffix in ("API_KEY", "BASE_URL", "MODEL"):
        monkeypatch.delenv(f"OMNIMANCER_BEDROCK_{suffix}", raising=False)
    monkeypatch.delenv("OMNIMANCER_DEFAULT_PROVIDER", raising=False)
    # The initializer caches instances per (name, model); a cache hit would
    # bypass the patched constructor and make these assertions meaningless.
    ProviderInitializer.clear_caches()
    yield
    ProviderInitializer.clear_caches()


class TestIncompleteEntriesAreNotFailures:
    @pytest.mark.asyncio
    async def test_entry_without_a_model_is_skipped_quietly(self, caplog):
        """No ERROR for an entry that simply has no model yet."""
        entries = {
            "bedrock": ProviderConfig(api_key="k", model=""),
            "claude": ProviderConfig(api_key="k", model="claude-sonnet-5"),
        }
        with patch.object(
            ProviderInitializer, "_create_provider_instance", return_value=MagicMock()
        ):
            with caplog.at_level(logging.DEBUG):
                providers = await ProviderInitializer.initialize_providers(entries)

        assert "claude" in providers
        assert "bedrock" not in providers
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, [r.getMessage() for r in errors]
        assert any("bedrock" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_real_failure_is_still_reported_with_its_cause(self, caplog):
        entries = {"claude": ProviderConfig(api_key="k", model="claude-sonnet-5")}
        with patch.object(
            ProviderInitializer,
            "_create_provider_instance",
            side_effect=RuntimeError("invalid api key"),
        ):
            with caplog.at_level(logging.ERROR):
                providers = await ProviderInitializer.initialize_providers(entries)

        assert providers == {}
        assert ProviderInitializer.get_last_init_errors()["claude"] == "invalid api key"
        assert any("invalid api key" in r.getMessage() for r in caplog.records)


class TestDefaultProviderMessage:
    """The message must say which of the two problems actually happened."""

    def _engine(self, default, providers):
        from omnimancer.core.engine import CoreEngine

        engine = CoreEngine.__new__(CoreEngine)
        engine.config_manager = MagicMock()
        engine.config_manager.get_config.return_value = Config(
            default_provider=default,
            providers=providers,
            storage_path="/tmp/omn-init-msg",
        )
        engine.providers = {}
        engine.current_provider = None
        engine._unavailable_default_provider = default
        return engine

    def test_unconfigured_default_says_so_and_names_the_env_var(self):
        engine = self._engine(
            "digitalocean", {"claude": ProviderConfig(api_key="k", model="m")}
        )
        message = engine._no_provider_error()
        assert "digitalocean" in message
        assert "not configured" in message
        assert "OMNIMANCER_DIGITALOCEAN_API_KEY" in message
        # It must not guess at a cause it has not established.
        assert "invalid API key" not in message

    def test_configured_default_reports_the_initialization_cause(self):
        engine = self._engine(
            "claude", {"claude": ProviderConfig(api_key="k", model="m")}
        )
        with patch.object(
            ProviderInitializer,
            "get_last_init_errors",
            return_value={"claude": "boom from the factory"},
        ):
            message = engine._no_provider_error()
        assert "claude" in message
        assert "boom from the factory" in message

    def test_entry_without_a_model_tells_the_user_to_set_one(self):
        engine = self._engine("bedrock", {"bedrock": ProviderConfig(model="")})
        message = engine._no_provider_error()
        assert "bedrock" in message
        assert "no model" in message.lower()
        assert "OMNIMANCER_BEDROCK_MODEL" in message
