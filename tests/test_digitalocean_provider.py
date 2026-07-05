"""Tests for the DigitalOcean inference provider and custom base_url support."""

from unittest.mock import MagicMock

import pytest

from omnimancer.core.models import ChatContext
from omnimancer.core.provider_initializer import ProviderInitializer
from omnimancer.providers.claude import ClaudeProvider
from omnimancer.providers.digitalocean import DigitalOceanProvider
from omnimancer.providers.factory import ProviderFactory
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.providers.openrouter import OpenRouterProvider
from omnimancer.utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)


def _response(status, json_body=None, text="boom"):
    resp = MagicMock()
    resp.status_code = status
    if json_body is None:
        resp.json = MagicMock(side_effect=ValueError("not json"))
    else:
        resp.json = MagicMock(return_value=json_body)
    resp.text = text
    return resp


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


class TestDigitalOceanErrorSurfacing:
    """DigitalOcean error bodies must surface their real message.

    Regression: DO returns errors as ``{"id": ..., "message": ...}`` rather
    than OpenAI's ``{"error": {"message": ...}}``, so every non-401/404/429
    failure was reported as the useless "OpenAI API error: Unknown error".
    """

    def _provider(self, model="qwen3.5-397b-a17b"):
        return DigitalOceanProvider(api_key="k", model=model)

    def test_do_shaped_error_message_surfaces(self):
        resp = _response(
            403,
            {"id": "forbidden", "message": "Model access denied for this key"},
        )
        with pytest.raises(ProviderError) as exc:
            self._provider()._handle_response(resp)
        msg = str(exc.value)
        assert "Model access denied for this key" in msg
        assert "Unknown error" not in msg

    def test_error_names_digitalocean_not_openai(self):
        resp = _response(400, {"id": "bad_request", "message": "nope"})
        with pytest.raises(ProviderError) as exc:
            self._provider()._handle_response(resp)
        msg = str(exc.value)
        assert "DigitalOcean" in msg
        assert "OpenAI" not in msg

    def test_openai_shaped_error_still_surfaces(self):
        resp = _response(400, {"error": {"message": "Invalid request"}})
        with pytest.raises(ProviderError, match="Invalid request"):
            self._provider()._handle_response(resp)

    def test_error_as_plain_string_surfaces(self):
        resp = _response(400, {"error": "model is overloaded"})
        with pytest.raises(ProviderError, match="model is overloaded"):
            self._provider()._handle_response(resp)

    def test_fastapi_detail_shape_surfaces(self):
        resp = _response(422, {"detail": "Invalid model name"})
        with pytest.raises(ProviderError, match="Invalid model name"):
            self._provider()._handle_response(resp)

    def test_non_json_body_reports_status(self):
        resp = _response(502, json_body=None)
        with pytest.raises(ProviderError, match="HTTP 502"):
            self._provider()._handle_response(resp)

    def test_404_includes_body_message_and_model(self):
        resp = _response(404, {"id": "not_found", "message": "model does not exist"})
        with pytest.raises(ModelNotFoundError) as exc:
            self._provider()._handle_response(resp)
        msg = str(exc.value)
        assert "qwen3.5-397b-a17b" in msg
        assert "model does not exist" in msg

    def test_401_includes_body_message(self):
        resp = _response(
            401, {"id": "Unauthorized", "message": "Unable to authenticate you"}
        )
        with pytest.raises(AuthenticationError) as exc:
            self._provider()._handle_response(resp)
        msg = str(exc.value)
        assert "DigitalOcean" in msg
        assert "Unable to authenticate you" in msg

    def test_429_names_provider(self):
        resp = _response(429, {"id": "rate_limited", "message": "slow down"})
        with pytest.raises(RateLimitError) as exc:
            self._provider()._handle_response(resp)
        assert "DigitalOcean" in str(exc.value)

    def test_openai_provider_keeps_openai_label(self):
        resp = _response(400, {"error": {"message": "bad"}})
        with pytest.raises(ProviderError, match="OpenAI API error: bad"):
            OpenAIProvider(api_key="k")._handle_response(resp)


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
