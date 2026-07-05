"""
Tests for the rate-limit fallback system.

Covers:
  * RateLimitFallbackHandler — error classification, provider selection, approval
  * FallbackConfig model — validation
  * Engine integration — fallback triggered on 429, correct provider switch
"""

from __future__ import annotations

from datetime import datetime

import pytest

from omnimancer.core.models import ChatResponse, FallbackConfig
from omnimancer.core.rate_limit_fallback import RateLimitFallbackHandler
from omnimancer.utils.errors import RateLimitError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fallback_config(**kwargs) -> FallbackConfig:
    return FallbackConfig(**kwargs)


def make_rate_limit_error(msg: str = "Rate limit exceeded") -> RateLimitError:
    return RateLimitError(msg)


def make_chat_response(success: bool = True, error: str = "") -> ChatResponse:
    if success:
        return ChatResponse(
            content="Hello!",
            model_used="test-model",
            tokens_used=10,
            timestamp=datetime.now(),
        )
    return ChatResponse(
        content="",
        model_used="",
        tokens_used=0,
        error=error,
        timestamp=datetime.now(),
    )


# ---------------------------------------------------------------------------
# FallbackConfig — validation
# ---------------------------------------------------------------------------


class TestFallbackConfig:
    def test_defaults(self):
        cfg = FallbackConfig()
        assert cfg.fallback_order == []
        assert cfg.auto_fallback is False
        assert cfg.fallback_on_rate_limit is True
        assert cfg.fallback_on_quota is False

    def test_fallback_order_strips_whitespace(self):
        cfg = FallbackConfig(fallback_order=["  claude  ", "openai", "  "])
        assert cfg.fallback_order == ["claude", "openai"]

    def test_fallback_order_filters_empty(self):
        cfg = FallbackConfig(fallback_order=["", "openai", ""])
        assert cfg.fallback_order == ["openai"]

    def test_auto_fallback_flag(self):
        cfg = FallbackConfig(auto_fallback=True)
        assert cfg.auto_fallback is True


# ---------------------------------------------------------------------------
# RateLimitFallbackHandler — error classification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    @pytest.mark.parametrize(
        "error_str",
        [
            "Rate limit exceeded",
            "ratelimit error",
            "rate_limit hit",
            "429 Too Many Requests",
            "HTTP 429",
            "too many requests",
            "requests per minute exceeded",
            "tokens per minute limit reached",
            "capacity exceeded",
        ],
    )
    def test_is_rate_limit_error_positive(self, error_str: str):
        handler = RateLimitFallbackHandler()
        assert handler.is_rate_limit_error(
            error_str
        ), f"Expected rate-limit match for: {error_str}"

    @pytest.mark.parametrize(
        "error_str",
        [
            "Connection refused",
            "Invalid API key",
            "Model not found",
            "Internal server error",
            "",
        ],
    )
    def test_is_rate_limit_error_negative(self, error_str: str):
        handler = RateLimitFallbackHandler()
        assert not handler.is_rate_limit_error(
            error_str
        ), f"Unexpected rate-limit match for: {error_str}"

    @pytest.mark.parametrize(
        "error_str",
        [
            "quota exceeded",
            "quota_exceeded",
            "usage limit reached",
            "insufficient_quota",
            "billing issue",
            "payment required",
            "402 payment",
        ],
    )
    def test_is_quota_error_positive(self, error_str: str):
        handler = RateLimitFallbackHandler()
        assert handler.is_quota_error(error_str)

    def test_should_fallback_rate_limit_enabled(self):
        handler = RateLimitFallbackHandler(fallback_on_rate_limit=True)
        assert handler.should_fallback("Rate limit exceeded")

    def test_should_fallback_rate_limit_disabled(self):
        handler = RateLimitFallbackHandler(fallback_on_rate_limit=False)
        assert not handler.should_fallback("Rate limit exceeded")

    def test_should_fallback_quota_enabled(self):
        handler = RateLimitFallbackHandler(fallback_on_quota=True)
        assert handler.should_fallback("quota exceeded")

    def test_should_fallback_quota_disabled_by_default(self):
        handler = RateLimitFallbackHandler(fallback_on_quota=False)
        assert not handler.should_fallback("quota exceeded")

    def test_should_fallback_other_error_false(self):
        handler = RateLimitFallbackHandler()
        assert not handler.should_fallback("Invalid API key")


# ---------------------------------------------------------------------------
# RateLimitFallbackHandler — provider selection
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_no_fallback_order_picks_first_non_current(self):
        handler = RateLimitFallbackHandler()
        result = handler.get_next_provider("claude", ["claude", "openai", "gemini"])
        assert result == "openai"

    def test_no_fallback_order_only_current_returns_none(self):
        handler = RateLimitFallbackHandler()
        result = handler.get_next_provider("claude", ["claude"])
        assert result is None

    def test_no_fallback_order_empty_available_returns_none(self):
        handler = RateLimitFallbackHandler()
        result = handler.get_next_provider("claude", [])
        assert result is None

    def test_fallback_order_picks_next_in_order(self):
        handler = RateLimitFallbackHandler(
            fallback_order=["claude", "openai", "gemini"]
        )
        result = handler.get_next_provider("claude", ["claude", "openai", "gemini"])
        assert result == "openai"

    def test_fallback_order_skips_unavailable(self):
        handler = RateLimitFallbackHandler(
            fallback_order=["claude", "openai", "gemini"]
        )
        # openai not available
        result = handler.get_next_provider("claude", ["claude", "gemini"])
        assert result == "gemini"

    def test_fallback_order_wraps_around(self):
        handler = RateLimitFallbackHandler(
            fallback_order=["claude", "openai", "gemini"]
        )
        # current is last in order — should wrap to first available entry
        result = handler.get_next_provider("gemini", ["claude", "gemini"])
        assert result == "claude"

    def test_fallback_order_current_not_in_order(self):
        handler = RateLimitFallbackHandler(fallback_order=["openai", "gemini"])
        # "mistral" is not in fallback_order — should still return first order entry
        result = handler.get_next_provider("mistral", ["mistral", "openai", "gemini"])
        assert result == "openai"

    def test_fallback_order_none_available_uses_any(self):
        handler = RateLimitFallbackHandler(fallback_order=["openai", "gemini"])
        # Neither openai nor gemini is available — falls back to any candidate
        result = handler.get_next_provider("claude", ["claude", "mistral"])
        assert result == "mistral"


# ---------------------------------------------------------------------------
# RateLimitFallbackHandler — approval
# ---------------------------------------------------------------------------


class TestApproval:
    @pytest.mark.asyncio
    async def test_auto_fallback_returns_true(self):
        handler = RateLimitFallbackHandler(auto_fallback=True)
        result = await handler.request_approval("claude", "openai", "429")
        assert result is True

    @pytest.mark.asyncio
    async def test_auto_fallback_fires_notification_callback(self):
        notified: list = []

        async def on_switch(current, next_):
            notified.append((current, next_))

        handler = RateLimitFallbackHandler(auto_fallback=True)
        handler.set_switch_notification_callback(on_switch)
        result = await handler.request_approval("claude", "openai", "429")
        assert result is True
        assert notified == [("claude", "openai")]

    @pytest.mark.asyncio
    async def test_manual_approval_callback_true(self):
        async def approve(current, next_, error):
            return True

        handler = RateLimitFallbackHandler(auto_fallback=False)
        handler.set_approval_callback(approve)
        result = await handler.request_approval("claude", "openai", "429")
        assert result is True

    @pytest.mark.asyncio
    async def test_manual_approval_callback_false(self):
        async def deny(current, next_, error):
            return False

        handler = RateLimitFallbackHandler(auto_fallback=False)
        handler.set_approval_callback(deny)
        result = await handler.request_approval("claude", "openai", "429")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_callback_no_auto_returns_false(self):
        handler = RateLimitFallbackHandler(auto_fallback=False)
        result = await handler.request_approval("claude", "openai", "429")
        assert result is False

    @pytest.mark.asyncio
    async def test_callback_exception_returns_false(self):
        async def buggy(current, next_, error):
            raise RuntimeError("oops")

        handler = RateLimitFallbackHandler(auto_fallback=False)
        handler.set_approval_callback(buggy)
        result = await handler.request_approval("claude", "openai", "429")
        assert result is False


# ---------------------------------------------------------------------------
# RateLimitFallbackHandler — update_from_config
# ---------------------------------------------------------------------------


class TestUpdateFromConfig:
    def test_update_syncs_all_fields(self):
        handler = RateLimitFallbackHandler()
        cfg = FallbackConfig(
            fallback_order=["openai", "gemini"],
            auto_fallback=True,
            fallback_on_rate_limit=False,
            fallback_on_quota=True,
        )
        handler.update_from_config(cfg)
        assert handler.fallback_order == ["openai", "gemini"]
        assert handler.auto_fallback is True
        assert handler.fallback_on_rate_limit is False
        assert handler.fallback_on_quota is True

    def test_update_non_config_is_no_op(self):
        handler = RateLimitFallbackHandler(auto_fallback=True)
        handler.update_from_config("not-a-config")  # type: ignore[arg-type]
        assert handler.auto_fallback is True  # unchanged


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


class MockProvider:
    """Minimal provider stub for engine tests."""

    def __init__(self, name: str, model: str = "test-model"):
        self.name = name
        self.model = model
        self._calls: list = []

    def get_provider_name(self) -> str:
        return self.name

    def supports_streaming(self) -> bool:
        return False

    def supports_tools(self) -> bool:
        return False

    def supports_native_tool_history(self) -> bool:
        return False

    async def send_message(self, message, context):
        self._calls.append(message)
        raise NotImplementedError("MockProvider stub — override in test")

    async def send_message_with_tools(self, message, context, tools):
        self._calls.append(message)
        raise NotImplementedError("MockProvider stub — override in test")


def make_engine(providers: dict):
    """Build a minimal CoreEngine-like object for integration tests."""
    import json
    import os

    # Build a real ConfigManager backed by a tempfile so we don't need disk I/O.
    import tempfile

    from omnimancer.core.config_manager import ConfigManager
    from omnimancer.core.engine import CoreEngine

    first = next(iter(providers))

    tmpdir = tempfile.mkdtemp()
    cfg_path = os.path.join(tmpdir, "config.json")
    key_path = os.path.join(tmpdir, ".key")

    # Generate a Fernet key manually
    from cryptography.fernet import Fernet

    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)
    os.chmod(key_path, 0o600)

    # Write minimal config
    cfg_data = {
        "default_provider": first,
        "providers": {name: {"model": "test-model"} for name in providers},
        "storage_path": tmpdir,
        "config_version": "2.0",
        "fallback": {},
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg_data, f)

    config_manager = ConfigManager(cfg_path)
    engine = CoreEngine(config_manager)
    engine.providers = dict(providers)
    engine.current_provider = list(providers.values())[0]
    return engine


class TestEngineIntegration:
    @pytest.mark.asyncio
    async def test_no_fallback_config_returns_error(self):
        """Without fallback configured, rate-limit errors are passed through."""
        primary = MockProvider("claude")
        engine = make_engine({"claude": primary})

        async def raise_rl(msg, ctx):
            raise RateLimitError("429 Too Many Requests")

        primary.send_message = raise_rl

        response = await engine.send_message("hello")
        assert not response.is_success
        assert (
            "429" in (response.error or "").lower()
            or "rate limit" in (response.error or "").lower()
        )

    @pytest.mark.asyncio
    async def test_auto_fallback_switches_provider(self):
        """With auto_fallback=True, the engine switches to the next provider."""
        primary = MockProvider("claude")
        fallback_p = MockProvider("openai")

        engine = make_engine({"claude": primary, "openai": fallback_p})

        # Configure auto-fallback
        config = engine.config_manager.get_config()
        config.fallback = FallbackConfig(
            fallback_order=["claude", "openai"],
            auto_fallback=True,
            fallback_on_rate_limit=True,
        )
        engine.configure_fallback()

        async def raise_rl(msg, ctx):
            raise RateLimitError("429 Too Many Requests")

        async def ok_response(msg, ctx):
            return ChatResponse(
                content="Fallback response",
                model_used="openai-model",
                tokens_used=5,
                timestamp=datetime.now(),
            )

        primary.send_message = raise_rl
        fallback_p.send_message = ok_response

        response = await engine.send_message("hello")
        assert response.is_success
        assert response.content == "Fallback response"
        # Engine should have switched to openai
        assert engine.current_provider is fallback_p

    @pytest.mark.asyncio
    async def test_interactive_fallback_user_approves(self):
        """With auto_fallback=False and callback returning True, switch happens."""
        primary = MockProvider("claude")
        fallback_p = MockProvider("openai")

        engine = make_engine({"claude": primary, "openai": fallback_p})

        config = engine.config_manager.get_config()
        config.fallback = FallbackConfig(
            fallback_order=["claude", "openai"],
            auto_fallback=False,
            fallback_on_rate_limit=True,
        )
        engine.configure_fallback()

        # Register a callback that always approves
        async def approve(current, next_, error):
            return True

        engine.set_fallback_approval_callback(approve)

        async def raise_rl(msg, ctx):
            raise RateLimitError("429")

        async def ok_response(msg, ctx):
            return ChatResponse(
                content="openai answer",
                model_used="gpt-4o",
                tokens_used=7,
                timestamp=datetime.now(),
            )

        primary.send_message = raise_rl
        fallback_p.send_message = ok_response

        response = await engine.send_message("hello")
        assert response.is_success
        assert response.content == "openai answer"

    @pytest.mark.asyncio
    async def test_interactive_fallback_user_declines(self):
        """With auto_fallback=False and callback returning False, error is returned."""
        primary = MockProvider("claude")
        fallback_p = MockProvider("openai")

        engine = make_engine({"claude": primary, "openai": fallback_p})

        config = engine.config_manager.get_config()
        config.fallback = FallbackConfig(
            fallback_order=["claude", "openai"],
            auto_fallback=False,
            fallback_on_rate_limit=True,
        )
        engine.configure_fallback()

        async def deny(current, next_, error):
            return False

        engine.set_fallback_approval_callback(deny)

        async def raise_rl(msg, ctx):
            raise RateLimitError("429 rate limit")

        primary.send_message = raise_rl

        response = await engine.send_message("hello")
        assert not response.is_success
        # fallback provider should NOT have been called
        assert fallback_p._calls == []

    @pytest.mark.asyncio
    async def test_fallback_response_error_triggers_switch(self):
        """A rate-limit error in ChatResponse.error (not raised) also triggers
        fallback."""
        primary = MockProvider("claude")
        fallback_p = MockProvider("openai")

        engine = make_engine({"claude": primary, "openai": fallback_p})

        config = engine.config_manager.get_config()
        config.fallback = FallbackConfig(
            fallback_order=["claude", "openai"],
            auto_fallback=True,
            fallback_on_rate_limit=True,
        )
        engine.configure_fallback()

        async def return_rl_error(msg, ctx):
            return ChatResponse(
                content="",
                model_used="",
                tokens_used=0,
                error="HTTP 429: rate limit exceeded",
                timestamp=datetime.now(),
            )

        async def ok_response(msg, ctx):
            return ChatResponse(
                content="ok",
                model_used="gpt-4o",
                tokens_used=5,
                timestamp=datetime.now(),
            )

        primary.send_message = return_rl_error
        fallback_p.send_message = ok_response

        response = await engine.send_message("hello")
        assert response.is_success
        assert response.content == "ok"

    @pytest.mark.asyncio
    async def test_non_rate_limit_error_not_fallback(self):
        """Non-rate-limit errors are NOT retried on the fallback provider."""
        primary = MockProvider("claude")
        fallback_p = MockProvider("openai")

        engine = make_engine({"claude": primary, "openai": fallback_p})

        config = engine.config_manager.get_config()
        config.fallback = FallbackConfig(
            fallback_order=["claude", "openai"],
            auto_fallback=True,
            fallback_on_rate_limit=True,
        )
        engine.configure_fallback()

        async def return_auth_error(msg, ctx):
            return ChatResponse(
                content="",
                model_used="",
                tokens_used=0,
                error="Invalid API key",
                timestamp=datetime.now(),
            )

        primary.send_message = return_auth_error

        response = await engine.send_message("hello")
        assert not response.is_success
        assert "Invalid API key" in (response.error or "")
        assert fallback_p._calls == []
