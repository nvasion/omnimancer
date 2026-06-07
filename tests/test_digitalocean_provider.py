"""Tests for the DigitalOcean inference provider and custom base_url support."""

import pytest

from omnimancer.core.models import ChatContext
from omnimancer.core.provider_initializer import ProviderInitializer
from omnimancer.providers.claude import ClaudeProvider
from omnimancer.providers.digitalocean import DigitalOceanProvider
from omnimancer.providers.factory import ProviderFactory
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.providers.openrouter import OpenRouterProvider
from omnimancer.utils.errors import AuthenticationError


class TestDigitalOceanProvider:
    def test_registered_in_factory(self):
        assert "digitalocean" in ProviderFactory.get_available_providers()

    def test_class_resolves_by_name(self):
        cls = ProviderInitializer.get_provider_class("digitalocean")
        assert cls is DigitalOceanProvider

    def test_default_base_url_and_model(self):
        provider = DigitalOceanProvider(api_key="k")
        assert provider.base_url == "https://inference.do-ai.run/v1"
        assert provider.model == "llama3.3-70b-instruct"

    def test_custom_base_url_override(self):
        provider = DigitalOceanProvider(
            api_key="k", base_url="https://gateway.example.com/v1/"
        )
        # Trailing slash is stripped.
        assert provider.base_url == "https://gateway.example.com/v1"

    def test_provider_name(self):
        assert DigitalOceanProvider(api_key="k").get_provider_name() == "digitalocean"

    def test_model_info_reports_digitalocean(self):
        info = DigitalOceanProvider(
            api_key="k", model="llama3-8b-instruct"
        ).get_model_info()
        assert info.provider == "digitalocean"

    def test_static_models_are_constructible(self):
        """The static fallback catalog must build without missing fields.

        Regression: ModelInfo requires cost_per_token, which was omitted and
        crashed _get_static_models() at runtime.
        """
        models = DigitalOceanProvider(api_key="k")._get_static_models()
        assert models, "expected a non-empty static model catalog"
        for m in models:
            assert m.provider == "digitalocean"
            assert isinstance(m.cost_per_token, float)

    @pytest.mark.parametrize("empty_key", ["", "   "])
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_clear_error(self, empty_key):
        """A missing key must raise an actionable AuthenticationError.

        Regression: an empty key was sent as the header ``Bearer `` and httpx
        raised a cryptic "Illegal header value" NetworkError that hid the real
        cause. The message must name the provider and how to set the key.
        """
        provider = DigitalOceanProvider(api_key=empty_key)
        ctx = ChatContext(messages=[], current_model="m", session_id="s")
        with pytest.raises(AuthenticationError) as exc:
            await provider.send_message("hi", ctx)
        msg = str(exc.value)
        assert "digitalocean" in msg
        assert "DIGITALOCEAN_INFERENCE_KEY" in msg
        # The misleading low-level header error must not surface.
        assert "Illegal header value" not in msg


@pytest.mark.parametrize(
    "provider_cls,default_url",
    [
        (OpenAIProvider, "https://api.openai.com/v1"),
        (OpenRouterProvider, "https://openrouter.ai/api/v1"),
        (ClaudeProvider, "https://api.anthropic.com/v1"),
    ],
)
class TestCustomBaseUrl:
    def test_default_base_url(self, provider_cls, default_url):
        assert provider_cls(api_key="k").base_url == default_url

    def test_base_url_override(self, provider_cls, default_url):
        provider = provider_cls(api_key="k", base_url="http://localhost:1234/v1")
        assert provider.base_url == "http://localhost:1234/v1"
        # The default is not used when an override is provided.
        assert provider.base_url != default_url
