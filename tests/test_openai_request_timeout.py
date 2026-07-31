"""Tests for chat-completion request timeouts and timeout retry.

Covers the OpenAI provider and, by inheritance, the DigitalOcean and other
OpenAI-compatible providers. Large models on serverless backends (e.g.
DigitalOcean inference) exceed the old hardcoded 30s/60s timeouts, and the
backend also stalls sporadically — headless agent runs died with
"Request to OpenAI API timed out" and no output.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from omnimancer.core.models import ChatContext
from omnimancer.providers.digitalocean import DigitalOceanProvider
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.providers.openrouter import OpenRouterProvider
from omnimancer.utils.errors import NetworkError


def _ok_response(content="done"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(
        return_value={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )
    return resp


@pytest.fixture
def context():
    return ChatContext(messages=[], current_model="m", session_id="s")


class TestTimeoutResolution:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("OMNIMANCER_REQUEST_TIMEOUT", raising=False)
        provider = OpenAIProvider(api_key="k")
        assert provider.request_timeout == OpenAIProvider.DEFAULT_REQUEST_TIMEOUT

    def test_kwarg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_REQUEST_TIMEOUT", "300")
        provider = OpenAIProvider(api_key="k", request_timeout=45)
        assert provider.request_timeout == 45.0

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_REQUEST_TIMEOUT", "300")
        provider = DigitalOceanProvider(api_key="k", model="qwen-x")
        assert provider.request_timeout == 300.0

    @pytest.mark.parametrize("bad", ["not-a-number", "-5", "0"])
    def test_invalid_values_fall_back(self, monkeypatch, bad):
        monkeypatch.setenv("OMNIMANCER_REQUEST_TIMEOUT", bad)
        provider = OpenAIProvider(api_key="k")
        assert provider.request_timeout == OpenAIProvider.DEFAULT_REQUEST_TIMEOUT

    def test_openrouter_env_var(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_REQUEST_TIMEOUT", "300")
        provider = OpenRouterProvider(api_key="k")
        assert provider.request_timeout == 300.0

    def test_config_timeout_field_reaches_provider(self, monkeypatch):
        """ProviderConfig.timeout arrives as the `timeout` kwarg via the
        initializer's splat — it must configure the request timeout.
        (It used to be silently dropped: only `request_timeout`, which is
        not a ProviderConfig field, was read.)"""
        monkeypatch.delenv("OMNIMANCER_REQUEST_TIMEOUT", raising=False)
        provider = OpenAIProvider(api_key="k", timeout=360)
        assert provider.request_timeout == 360.0

    def test_request_timeout_kwarg_wins_over_timeout(self, monkeypatch):
        monkeypatch.delenv("OMNIMANCER_REQUEST_TIMEOUT", raising=False)
        provider = OpenAIProvider(api_key="k", timeout=360, request_timeout=45)
        assert provider.request_timeout == 45.0

    def test_config_timeout_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_REQUEST_TIMEOUT", "300")
        provider = OpenAIProvider(api_key="k", timeout=360)
        assert provider.request_timeout == 360.0

    def test_openrouter_config_timeout_field(self, monkeypatch):
        monkeypatch.delenv("OMNIMANCER_REQUEST_TIMEOUT", raising=False)
        provider = OpenRouterProvider(api_key="k", timeout=360)
        assert provider.request_timeout == 360.0


class TestOpenRouterTimeoutUsed:
    async def test_send_message_uses_configured_timeout(self, context):
        provider = OpenRouterProvider(api_key="k", request_timeout=250)
        seen = []

        async def fake_post(url, **kwargs):
            seen.append(kwargs["timeout"])
            return _ok_response()

        with patch("httpx.AsyncClient") as client:
            inst = client.return_value.__aenter__.return_value
            inst.post = AsyncMock(side_effect=fake_post)
            await provider.send_message("hi", context)

        assert seen == [250]


class TestTimeoutUsedOnRequests:
    async def test_send_message_uses_configured_timeout(self, context):
        provider = OpenAIProvider(api_key="k", request_timeout=222)
        seen = []

        async def fake_post(url, **kwargs):
            seen.append(kwargs["timeout"])
            return _ok_response()

        with patch("httpx.AsyncClient") as client:
            inst = client.return_value.__aenter__.return_value
            inst.post = AsyncMock(side_effect=fake_post)
            await provider.send_message("hi", context)

        assert seen == [222]


class TestTimeoutRetry:
    async def test_retries_once_on_timeout_then_succeeds(self, context):
        provider = DigitalOceanProvider(api_key="k", model="qwen-x")
        calls = []

        async def fake_post(url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                raise httpx.ReadTimeout("stalled")
            return _ok_response("recovered")

        with patch("httpx.AsyncClient") as client:
            inst = client.return_value.__aenter__.return_value
            inst.post = AsyncMock(side_effect=fake_post)
            result = await provider.send_message("hi", context)

        assert result.content == "recovered"
        assert len(calls) == 2

    async def test_second_timeout_raises_network_error(self, context):
        provider = DigitalOceanProvider(api_key="k", model="qwen-x")

        with patch("httpx.AsyncClient") as client:
            inst = client.return_value.__aenter__.return_value
            inst.post = AsyncMock(side_effect=httpx.ReadTimeout("stalled"))
            with pytest.raises(NetworkError):
                await provider.send_message("hi", context)

        assert inst.post.await_count == 2
