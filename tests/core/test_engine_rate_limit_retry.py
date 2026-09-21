"""Tests for engine-level same-provider retry with backoff on rate limits.

A 429 used to kill the request immediately (and in headless mode the whole
process), forcing orchestrators to restart runs from scratch. The engine now
retries the same provider with exponential backoff — honoring Retry-After —
before falling through to the provider-switch fallback path.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core.chat_manager import ChatManager
from omnimancer.core.engine import CoreEngine
from omnimancer.core.models import ChatResponse
from omnimancer.utils.errors import RateLimitError


def _engine_with_provider(provider) -> CoreEngine:
    engine = CoreEngine.__new__(CoreEngine)
    engine.config_manager = MagicMock()
    engine.providers = {"claude": provider}
    engine.current_provider = provider
    engine.chat_manager = ChatManager()
    engine.configure_fallback = MagicMock()
    engine._fire_hook = AsyncMock(return_value=SimpleNamespace(allowed=True))
    # No fallback provider available — retry is the only mitigation.
    engine._apply_rate_limit_fallback = AsyncMock(return_value=None)
    return engine


def _provider(side_effect) -> MagicMock:
    provider = MagicMock()
    provider.model = "test-model"
    provider.get_provider_name.return_value = "claude"
    provider.send_message_with_tools = AsyncMock(side_effect=side_effect)
    return provider


_OK = ChatResponse(
    content="hi",
    model_used="test-model",
    tokens_used=5,
    input_tokens=3,
    output_tokens=2,
    stop_reason="end_turn",
)


class TestRateLimitRetry:
    @pytest.mark.asyncio
    async def test_retries_after_rate_limit_then_succeeds(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_RATE_LIMIT_RETRIES", "3")
        provider = _provider(
            [RateLimitError("429 rate limit"), RateLimitError("429 rate limit"), _OK]
        )
        engine = _engine_with_provider(provider)

        with patch(
            "omnimancer.utils.retry.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            response = await engine.send_message_with_tools("hello", [])

        assert response.is_success
        assert provider.send_message_with_tools.call_count == 3
        assert sleep.await_count == 2

    @pytest.mark.asyncio
    async def test_backoff_delays_grow_exponentially(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_RATE_LIMIT_RETRIES", "3")
        monkeypatch.setenv("OMNIMANCER_RATE_LIMIT_BASE_DELAY", "2.0")
        provider = _provider([RateLimitError("429"), RateLimitError("429"), _OK])
        engine = _engine_with_provider(provider)

        with patch(
            "omnimancer.utils.retry.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            await engine.send_message_with_tools("hello", [])

        delays = [call.args[0] for call in sleep.await_args_list]
        # base 2.0, exponential base 2 → ~2s then ~4s (±10% jitter)
        assert 1.8 <= delays[0] <= 2.2
        assert 3.6 <= delays[1] <= 4.4

    @pytest.mark.asyncio
    async def test_honors_retry_after_from_provider(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_RATE_LIMIT_RETRIES", "2")
        provider = _provider([RateLimitError("429", retry_after=30), _OK])
        engine = _engine_with_provider(provider)

        with patch(
            "omnimancer.utils.retry.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            await engine.send_message_with_tools("hello", [])

        delay = sleep.await_args_list[0].args[0]
        assert 27.0 <= delay <= 33.0  # 30s ±10% jitter

    @pytest.mark.asyncio
    async def test_exhausted_retries_return_error_response(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_RATE_LIMIT_RETRIES", "2")
        provider = _provider(RateLimitError("429 rate limit"))
        engine = _engine_with_provider(provider)

        with patch("omnimancer.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            response = await engine.send_message_with_tools("hello", [])

        assert not response.is_success
        assert "rate limit" in (response.error or "").lower()
        assert provider.send_message_with_tools.call_count == 3  # 1 + 2 retries

    @pytest.mark.asyncio
    async def test_retries_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_RATE_LIMIT_RETRIES", "0")
        provider = _provider(RateLimitError("429 rate limit"))
        engine = _engine_with_provider(provider)

        with patch(
            "omnimancer.utils.retry.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            response = await engine.send_message_with_tools("hello", [])

        assert not response.is_success
        assert provider.send_message_with_tools.call_count == 1
        assert sleep.await_count == 0

    @pytest.mark.asyncio
    async def test_non_rate_limit_errors_do_not_retry(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_RATE_LIMIT_RETRIES", "3")
        provider = _provider(ValueError("boom"))
        engine = _engine_with_provider(provider)

        with patch("omnimancer.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            response = await engine.send_message_with_tools("hello", [])

        assert not response.is_success
        assert provider.send_message_with_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_send_message_also_retries(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_RATE_LIMIT_RETRIES", "2")
        provider = MagicMock()
        provider.model = "test-model"
        provider.get_provider_name.return_value = "claude"
        provider.send_message = AsyncMock(side_effect=[RateLimitError("429"), _OK])
        engine = _engine_with_provider(provider)

        with patch("omnimancer.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            response = await engine.send_message("hello")

        assert response.is_success
        assert provider.send_message.call_count == 2
