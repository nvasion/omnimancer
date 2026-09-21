"""Tests for Anthropic prompt caching and rate-limit metadata in ClaudeProvider.

Every agent-loop iteration resends the same system prompt, tool schemas, and
transcript. cache_control breakpoints let the API serve that stable prefix
from cache (~90% cheaper) instead of re-billing it each iteration.
"""

import httpx
import pytest

from omnimancer.core.models import ChatResponse
from omnimancer.providers.claude import ClaudeProvider
from omnimancer.utils.errors import RateLimitError


@pytest.fixture
def provider():
    return ClaudeProvider(api_key="test-key", model="claude-sonnet-4-6")


class TestCacheControlInjection:
    def _body(self):
        return {
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "agent prompt + task"},
                {"role": "assistant", "content": "working"},
                {"role": "user", "content": "Tool results: ..."},
            ],
            "tools": [
                {"name": "a", "description": "", "input_schema": {}},
                {"name": "b", "description": "", "input_schema": {}},
            ],
        }

    def test_marks_last_tool_and_last_message(self, provider):
        body = self._body()
        provider._apply_cache_control(body)

        assert "cache_control" not in body["tools"][0]
        assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}

        # Earlier messages untouched (plain strings)…
        assert body["messages"][0]["content"] == "agent prompt + task"
        # …last message converted to a block carrying the breakpoint.
        last = body["messages"][-1]["content"]
        assert isinstance(last, list)
        assert last[-1]["text"] == "Tool results: ..."
        assert last[-1]["cache_control"] == {"type": "ephemeral"}

    def test_no_tools_is_fine(self, provider):
        body = self._body()
        del body["tools"]
        provider._apply_cache_control(body)
        last = body["messages"][-1]["content"]
        assert last[-1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_last_message_left_alone(self, provider):
        body = self._body()
        body["messages"][-1]["content"] = ""
        provider._apply_cache_control(body)
        assert body["messages"][-1]["content"] == ""

    def test_block_content_gets_breakpoint_in_place(self, provider):
        body = self._body()
        body["messages"][-1]["content"] = [{"type": "text", "text": "hi"}]
        provider._apply_cache_control(body)
        assert body["messages"][-1]["content"][-1]["cache_control"] == {
            "type": "ephemeral"
        }

    def test_disabled_via_env(self, provider, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_PROMPT_CACHE", "0")
        body = self._body()
        provider._apply_cache_control(body)
        assert "cache_control" not in body["tools"][-1]
        assert body["messages"][-1]["content"] == "Tool results: ..."


class TestCacheUsageParsing:
    def _response(self, usage):
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": usage,
            },
        )

    def test_cache_tokens_parsed_with_tools(self, provider):
        resp = provider._handle_response_with_tools(
            self._response(
                {
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 4000,
                    "cache_creation_input_tokens": 500,
                }
            )
        )
        assert resp.input_tokens == 12
        assert resp.cache_read_input_tokens == 4000
        assert resp.cache_creation_input_tokens == 500

    def test_cache_tokens_parsed_plain(self, provider):
        resp = provider._handle_response(
            self._response(
                {
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 0,
                }
            )
        )
        assert resp.cache_read_input_tokens == 100
        assert resp.cache_creation_input_tokens == 0

    def test_missing_cache_fields_default_none(self, provider):
        resp = provider._handle_response(
            self._response({"input_tokens": 1, "output_tokens": 1})
        )
        assert resp.cache_read_input_tokens is None
        assert resp.cache_creation_input_tokens is None


class TestRateLimitMetadata:
    def test_429_extracts_retry_after_header(self, provider):
        response = httpx.Response(
            429,
            json={"error": {"message": "rate limited"}},
            headers={"retry-after": "17"},
        )
        with pytest.raises(RateLimitError) as exc_info:
            provider._handle_response(response)
        assert exc_info.value.retry_after == 17

    def test_429_without_header_has_no_retry_after(self, provider):
        response = httpx.Response(429, json={"error": {"message": "rate limited"}})
        with pytest.raises(RateLimitError) as exc_info:
            provider._handle_response(response)
        assert exc_info.value.retry_after is None

    def test_529_overloaded_raises_rate_limit_error(self, provider):
        response = httpx.Response(
            529, json={"error": {"type": "overloaded_error", "message": "Overloaded"}}
        )
        with pytest.raises(RateLimitError):
            provider._handle_response(response)


class TestChatResponseCacheFields:
    def test_defaults(self):
        resp = ChatResponse(content="x", model_used="m", tokens_used=1)
        assert resp.cache_read_input_tokens is None
        assert resp.cache_creation_input_tokens is None
