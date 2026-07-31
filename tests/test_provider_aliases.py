"""Provider-alias resolution tests.

Named provider entries (e.g. ``gateway``, ``local``) are config-dict keys that
resolve to a registered provider class via ``ProviderConfig.provider_type``.
The config name stays the identity for API-key decryption, env overrides, and
instance caching; only the *class lookup* follows ``provider_type``.
"""

from unittest.mock import MagicMock, patch

import pytest

from omnimancer.core.models import ProviderConfig
from omnimancer.core.provider_initializer import ProviderInitializer
from omnimancer.providers.factory import ProviderFactory
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.utils.errors import ConfigurationError


@pytest.fixture(autouse=True)
def _clean_instance_caches():
    """Each test starts and ends with empty provider-instance caches."""
    ProviderInitializer.clear_caches()
    yield
    ProviderInitializer.clear_caches()


def _alias_config(**overrides) -> ProviderConfig:
    """A typical alias entry: openai-family type pointed at a custom endpoint."""
    data = {
        "model": "qwen3-coder-30b",
        "provider_type": "openai",
        "base_url": "http://localhost:8000/v1",
        "api_key": "test-key",
    }
    data.update(overrides)
    return ProviderConfig(**data)


class TestProviderTypeResolution:
    def test_alias_resolves_class_via_provider_type(self):
        provider = ProviderFactory.create_provider("gateway", _alias_config())
        assert isinstance(provider, OpenAIProvider)

    def test_registered_name_wins_over_provider_type(self):
        """A factory-registered dict key keeps its own class even if
        provider_type disagrees (migration back-fills make this reachable)."""
        config = _alias_config(provider_type="claude")
        provider = ProviderFactory.create_provider("openai", config)
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_name_without_type_raises(self):
        config = ProviderConfig(model="m", api_key="k")
        with pytest.raises(ConfigurationError) as exc:
            ProviderFactory.create_provider("mystery", config)
        assert "mystery" in str(exc.value)

    def test_unknown_type_error_names_alias_and_type(self):
        config = _alias_config(provider_type="nonsense")
        with pytest.raises(ConfigurationError) as exc:
            ProviderFactory.create_provider("gateway", config)
        message = str(exc.value)
        assert "gateway" in message
        assert "nonsense" in message

    def test_alias_base_url_reaches_instance(self):
        provider = ProviderFactory.create_provider(
            "gateway", _alias_config(base_url="http://alpha:8888/v1")
        )
        assert provider.base_url == "http://alpha:8888/v1"


class TestAliasIdentity:
    def test_two_aliases_same_type_get_distinct_instances(self):
        p_gateway = ProviderFactory.create_provider("gateway", _alias_config())
        p_local = ProviderFactory.create_provider("local", _alias_config())
        assert p_gateway is not p_local

    def test_api_key_decryption_keyed_by_alias_name(self):
        config_manager = MagicMock()
        config_manager.get_api_key.return_value = "decrypted-key"
        provider = ProviderFactory.create_provider(
            "gateway", _alias_config(api_key="encrypted-blob"), config_manager
        )
        config_manager.get_api_key.assert_called_with("gateway")
        assert provider.api_key == "decrypted-key"


class TestCacheKey:
    def test_cache_key_distinguishes_provider_type(self):
        k1 = ProviderInitializer._generate_cache_key(
            "gateway", _alias_config(provider_type="openai")
        )
        k2 = ProviderInitializer._generate_cache_key(
            "gateway", _alias_config(provider_type="ollama")
        )
        assert k1 != k2

    def test_cache_key_distinguishes_timeout(self):
        """Editing an alias's timeout must not serve a stale cached instance."""
        k1 = ProviderInitializer._generate_cache_key(
            "gateway", _alias_config(timeout=120)
        )
        k2 = ProviderInitializer._generate_cache_key(
            "gateway", _alias_config(timeout=360)
        )
        assert k1 != k2


class TestConfigWriteInvalidation:
    def test_set_provider_config_clears_instance_caches(self, tmp_path):
        from omnimancer.core.config_manager import ConfigManager

        manager = ConfigManager(config_path=tmp_path / "config.json")
        with patch.object(ProviderInitializer, "clear_caches") as clear:
            manager.set_provider_config("gateway", _alias_config())
        clear.assert_called_once()
