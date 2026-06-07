"""Tests for environment-variable based config overrides."""

import pytest

from omnimancer.core.env_loader import (
    ENV_VAR_MAPPING,
    apply_env_overrides,
    load_api_key_from_env,
    load_provider_env_overrides,
)
from omnimancer.core.models import Config, ProviderConfig

# Environment variables this module may touch; cleared before each test so the
# host/CI environment cannot leak into assertions.
_RELEVANT_ENV = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "DIGITALOCEAN_INFERENCE_KEY",
    "GOOGLE_API_KEY",
    "PERPLEXITY_API_KEY",
    "XAI_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "AZURE_OPENAI_KEY",
    "OMNIMANCER_DEFAULT_PROVIDER",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in _RELEVANT_ENV:
        monkeypatch.delenv(var, raising=False)
    # Clear any OMNIMANCER_<PROVIDER>_* vars for common providers.
    for provider in ("OPENAI", "CLAUDE", "DIGITALOCEAN", "OPENROUTER"):
        for suffix in ("API_KEY", "BASE_URL", "MODEL"):
            monkeypatch.delenv(f"OMNIMANCER_{provider}_{suffix}", raising=False)
    yield


def _config():
    return Config(
        default_provider="claude",
        providers={"openai": ProviderConfig(api_key="enc", model="gpt-4")},
        storage_path="/tmp/omni-test",
    )


class TestEnvMapping:
    def test_digitalocean_key_mapped(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_INFERENCE_KEY", "do-123")
        assert ENV_VAR_MAPPING["digitalocean"] == "DIGITALOCEAN_INFERENCE_KEY"
        assert load_api_key_from_env("digitalocean") == "do-123"


class TestProviderEnvOverrides:
    def test_namespaced_overrides(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_OPENAI_BASE_URL", "http://localhost:1234/v1")
        monkeypatch.setenv("OMNIMANCER_OPENAI_MODEL", "gpt-4o")
        monkeypatch.setenv("OMNIMANCER_OPENAI_API_KEY", "sk-namespaced")
        overrides = load_provider_env_overrides("openai")
        assert overrides == {
            "api_key": "sk-namespaced",
            "base_url": "http://localhost:1234/v1",
            "model": "gpt-4o",
        }

    def test_conventional_key_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-standard")
        assert load_provider_env_overrides("openai") == {"api_key": "sk-standard"}


class TestApplyEnvOverrides:
    def test_overrides_existing_provider_base_url(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_OPENAI_BASE_URL", "http://proxy/v1")
        effective = apply_env_overrides(_config())
        assert effective.providers["openai"].base_url == "http://proxy/v1"

    def test_default_provider_override(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_DEFAULT_PROVIDER", "digitalocean")
        monkeypatch.setenv("DIGITALOCEAN_INFERENCE_KEY", "do-123")
        effective = apply_env_overrides(_config())
        assert effective.default_provider == "digitalocean"

    def test_creates_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("DIGITALOCEAN_INFERENCE_KEY", "do-123")
        monkeypatch.setenv(
            "OMNIMANCER_DIGITALOCEAN_BASE_URL", "https://inference.do-ai.run/v1"
        )
        monkeypatch.setenv("OMNIMANCER_DIGITALOCEAN_MODEL", "llama3.3-70b-instruct")
        effective = apply_env_overrides(_config())
        do = effective.providers["digitalocean"]
        assert do.api_key == "do-123"
        assert do.base_url == "https://inference.do-ai.run/v1"
        assert do.model == "llama3.3-70b-instruct"

    def test_does_not_mutate_original(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_OPENAI_BASE_URL", "http://proxy/v1")
        monkeypatch.setenv("OMNIMANCER_DEFAULT_PROVIDER", "openai")
        original = _config()
        apply_env_overrides(original)
        assert original.providers["openai"].base_url is None
        assert original.default_provider == "claude"

    def test_no_env_is_noop(self):
        effective = apply_env_overrides(_config())
        assert effective.default_provider == "claude"
        assert effective.providers["openai"].base_url is None
