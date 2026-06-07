"""Tests for automatic max_tokens refitting on context-window overflow.

Covers the OpenAI provider and, by inheritance, the DigitalOcean provider.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core.models import ChatContext
from omnimancer.providers.digitalocean import DigitalOceanProvider
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.utils.errors import ProviderError

# The real error string returned by DigitalOcean / vLLM-style backends.
DO_ERROR = (
    "This model's maximum context length is 131072 tokens. However, you "
    "requested 4096 output tokens and your prompt contains at least 126977 "
    "input tokens, for a total of at least 131073 tokens. Please reduce the "
    "length of the input prompt or the number of requested output tokens."
)
# The canonical OpenAI / vLLM phrasing.
OPENAI_ERROR = (
    "This model's maximum context length is 8192 tokens. However, you requested "
    "6000 tokens (5000 in the messages, 1000 in the completion)."
)


def _error_response(message, status=400):
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value={"error": {"message": message}})
    return resp


def _ok_response(content="done"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(
        return_value={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        }
    )
    return resp


@pytest.fixture
def context():
    return ChatContext(messages=[], current_model="m", session_id="s")


class TestFitMaxTokensParsing:
    @pytest.mark.parametrize(
        "error,expected",
        [
            # 131072 - 126977 - 256 buffer = 3839
            (DO_ERROR, 3839),
            # 8192 - 5000 - 256 = 2936
            (OPENAI_ERROR, 2936),
        ],
    )
    def test_parses_and_fits(self, error, expected):
        provider = OpenAIProvider(api_key="k")
        fitted = provider._fit_max_tokens(_error_response(error), current_max=4096)
        assert fitted == expected

    def test_non_context_error_returns_none(self):
        provider = OpenAIProvider(api_key="k")
        assert provider._fit_max_tokens(_error_response("nope"), 4096) is None

    def test_prompt_larger_than_window_returns_zero(self):
        provider = OpenAIProvider(api_key="k")
        too_big = (
            "This model's maximum context length is 1000 tokens. However, you "
            "requested 2000 input tokens."
        )
        assert provider._fit_max_tokens(_error_response(too_big), 4096) == 0


class TestRetryBehavior:
    async def test_retries_with_fitted_tokens_then_succeeds(self, context):
        provider = DigitalOceanProvider(api_key="k", model="qwen-x")
        posts = []

        async def fake_post(url, **kwargs):
            posts.append(kwargs["json"]["max_tokens"])
            if len(posts) == 1:
                return _error_response(DO_ERROR)
            return _ok_response("hello")

        with patch("httpx.AsyncClient") as client:
            inst = client.return_value.__aenter__.return_value
            inst.post = AsyncMock(side_effect=fake_post)
            result = await provider.send_message("explain this repo", context)

        assert result.content == "hello"
        assert posts == [4096, 3839]  # original, then refit (131072-126977-256)

    async def test_refits_again_when_input_grows_between_attempts(self, context):
        """The exact reported scenario: the server's reported input size grows
        between attempts, so a single refit lands just over the limit and a
        second refit is needed."""
        provider = DigitalOceanProvider(api_key="k", model="qwen-x")
        # input grows 128000 -> 129000 across attempts, forcing two refits.
        errors = [
            "This model's maximum context length is 131072 tokens. However, you "
            "requested 4096 output tokens and your prompt contains at least "
            "128000 input tokens.",
            "This model's maximum context length is 131072 tokens. However, you "
            "requested 2816 output tokens and your prompt contains at least "
            "129000 input tokens.",
        ]
        posts = []

        async def fake_post(url, **kwargs):
            posts.append(kwargs["json"]["max_tokens"])
            if len(posts) <= len(errors):
                return _error_response(errors[len(posts) - 1])
            return _ok_response("ok")

        with patch("httpx.AsyncClient") as client:
            inst = client.return_value.__aenter__.return_value
            inst.post = AsyncMock(side_effect=fake_post)
            result = await provider.send_message("explain this repo", context)

        assert result.content == "ok"
        # 4096 -> (131072-128000-256)=2816 -> (131072-129000-256)=1816
        assert posts == [4096, 2816, 1816]

    async def test_prompt_too_large_raises_clear_error(self, context):
        provider = OpenAIProvider(api_key="k")
        too_big = (
            "This model's maximum context length is 1000 tokens. However, you "
            "requested 5000 input tokens."
        )

        async def fake_post(url, **kwargs):
            return _error_response(too_big)

        with patch("httpx.AsyncClient") as client:
            inst = client.return_value.__aenter__.return_value
            inst.post = AsyncMock(side_effect=fake_post)
            with pytest.raises(ProviderError, match="too large"):
                await provider.send_message("hi", context)

    async def test_non_context_error_not_retried(self, context):
        provider = OpenAIProvider(api_key="k")
        calls = []

        async def fake_post(url, **kwargs):
            calls.append(1)
            return _error_response("Some other error", status=400)

        with patch("httpx.AsyncClient") as client:
            inst = client.return_value.__aenter__.return_value
            inst.post = AsyncMock(side_effect=fake_post)
            with pytest.raises(ProviderError):
                await provider.send_message("hi", context)

        assert len(calls) == 1  # no retry
