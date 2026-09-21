"""Cross-provider prompt-cache integration tests.

Vendors expose caching differently: OpenAI-compatible APIs cache automatically
and only need their cached-token usage parsed; Gemini/Vertex likewise; Bedrock
takes explicit cachePoint blocks (model-gated); OpenRouter forwards Anthropic
cache_control. These tests pin down each shape — plus the input/output token
split several providers previously dropped entirely.
"""

import json

import httpx
import pytest

from omnimancer.providers.azure import AzureProvider
from omnimancer.providers.bedrock import BedrockProvider
from omnimancer.providers.digitalocean import DigitalOceanProvider
from omnimancer.providers.gemini import GeminiProvider
from omnimancer.providers.mistral import MistralProvider
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.providers.openrouter import OpenRouterProvider
from omnimancer.providers.vertex import VertexAIProvider
from omnimancer.providers.xai import XAIProvider


def _openai_shape_body(text="hello"):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
    }


class TestOpenAIFamilyCachedTokens:
    """OpenAI-compatible providers: parse automatic-cache usage fields."""

    @pytest.mark.parametrize(
        "provider_cls,model",
        [
            (OpenAIProvider, "gpt-4o"),
            (AzureProvider, "gpt-4"),
            (XAIProvider, "grok-beta"),
            (MistralProvider, "mistral-large-latest"),
            (OpenRouterProvider, "openai/gpt-4o"),
        ],
    )
    def test_handle_response_parses_cache_and_token_split(self, provider_cls, model):
        kwargs = (
            {"azure_endpoint": "https://unit.openai.azure.com"}
            if provider_cls is AzureProvider
            else {}
        )
        provider = provider_cls(api_key="k", model=model, **kwargs)
        resp = provider._handle_response(httpx.Response(200, json=_openai_shape_body()))
        assert resp.input_tokens == 120
        assert resp.output_tokens == 30
        assert resp.cache_read_input_tokens == 100

    def test_missing_details_defaults_none(self):
        provider = OpenAIProvider(api_key="k", model="gpt-4o")
        body = _openai_shape_body()
        del body["usage"]["prompt_tokens_details"]
        resp = provider._handle_response(httpx.Response(200, json=body))
        assert resp.cache_read_input_tokens is None

    def test_top_level_cache_read_field_is_a_fallback(self):
        # DigitalOcean reports Anthropic/OpenAI cache hits as a top-level
        # usage.cache_read_input_tokens instead of prompt_tokens_details.
        provider = DigitalOceanProvider(api_key="k", model="anthropic-claude-opus-4")
        body = _openai_shape_body()
        del body["usage"]["prompt_tokens_details"]
        body["usage"]["cache_read_input_tokens"] = 777
        resp = provider._handle_response(httpx.Response(200, json=body))
        assert resp.cache_read_input_tokens == 777


class TestDigitalOceanPromptCache:
    """DigitalOcean Gradient: opt-in differs per model family."""

    def _payload(self, model):
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "latest"},
            ],
            "max_tokens": 100,
        }

    def test_anthropic_model_gets_cache_control(self):
        provider = DigitalOceanProvider(api_key="k", model="anthropic-claude-opus-4")
        payload = self._payload(provider.model)
        provider._apply_prompt_cache(payload)
        assert payload["messages"][0]["content"] == "first"
        last = payload["messages"][-1]["content"]
        assert last[-1]["text"] == "latest"
        assert last[-1]["cache_control"] == {"type": "ephemeral"}
        assert "prompt_cache_retention" not in payload

    def test_openai_model_gets_prompt_cache_retention(self):
        provider = DigitalOceanProvider(api_key="k", model="openai-gpt-4o")
        payload = self._payload(provider.model)
        provider._apply_prompt_cache(payload)
        assert payload["prompt_cache_retention"] == "in_memory"
        assert payload["messages"][-1]["content"] == "latest"

    def test_open_source_model_untouched(self):
        # Open-source models cache automatically — no opt-in fields.
        provider = DigitalOceanProvider(api_key="k", model="llama3.3-70b-instruct")
        payload = self._payload(provider.model)
        provider._apply_prompt_cache(payload)
        assert payload["messages"][-1]["content"] == "latest"
        assert "prompt_cache_retention" not in payload

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_PROMPT_CACHE", "0")
        provider = DigitalOceanProvider(api_key="k", model="anthropic-claude-opus-4")
        payload = self._payload(provider.model)
        provider._apply_prompt_cache(payload)
        assert payload["messages"][-1]["content"] == "latest"
        assert "prompt_cache_retention" not in payload

    def test_base_openai_provider_hook_is_a_no_op(self):
        provider = OpenAIProvider(api_key="k", model="gpt-4o")
        payload = self._payload("gpt-4o")
        provider._apply_prompt_cache(payload)
        assert payload == self._payload("gpt-4o")


class TestGeminiFamilyCachedTokens:
    """Gemini/Vertex: implicit caching reports cachedContentTokenCount."""

    def _body(self):
        return {
            "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}}],
            "usageMetadata": {
                "promptTokenCount": 200,
                "candidatesTokenCount": 40,
                "totalTokenCount": 240,
                "cachedContentTokenCount": 150,
            },
        }

    def test_gemini_parses_cache_and_token_split(self):
        provider = GeminiProvider(api_key="k", model="gemini-1.5-pro")
        resp = provider._handle_response_with_tools(
            httpx.Response(200, json=self._body())
        )
        assert resp.input_tokens == 200
        assert resp.output_tokens == 40
        assert resp.cache_read_input_tokens == 150

    def test_vertex_parses_cache_and_token_split(self):
        provider = VertexAIProvider(
            api_key="k", model="gemini-1.5-pro", vertex_project="unit-project"
        )
        resp = provider._handle_response_with_tools(
            httpx.Response(200, json=self._body())
        )
        assert resp.input_tokens == 200
        assert resp.output_tokens == 40
        assert resp.cache_read_input_tokens == 150


class TestBedrockCachePoint:
    """Bedrock Converse: explicit cachePoint blocks, model-gated."""

    def _provider(self, model):
        return BedrockProvider(api_key="k", model=model, aws_region="us-east-1")

    def _context(self):
        from omnimancer.core.models import ChatContext

        return ChatContext(messages=[], current_model="", session_id="s")

    def test_supported_model_gets_cache_points(self):
        provider = self._provider("anthropic.claude-3-7-sonnet-20250219-v1:0")
        from omnimancer.core.models import ToolDefinition

        body = json.loads(
            provider._prepare_bedrock_request_with_tools(
                "hi",
                self._context(),
                [ToolDefinition(name="t", description="d", parameters={})],
            )
        )
        # Breakpoint after the tool list…
        assert body["toolConfig"]["tools"][-1] == {"cachePoint": {"type": "default"}}
        # …and after the last message content block.
        assert body["messages"][-1]["content"][-1] == {
            "cachePoint": {"type": "default"}
        }

    def test_unsupported_model_gets_no_cache_points(self):
        # Claude 3 Sonnet (v1) does not support Converse prompt caching —
        # a cachePoint there is a hard ValidationException.
        provider = self._provider("anthropic.claude-3-sonnet-20240229-v1:0")
        body = json.loads(provider._prepare_bedrock_request("hi", self._context()))
        assert body["messages"][-1]["content"] == [{"text": "hi"}]

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_PROMPT_CACHE", "0")
        provider = self._provider("anthropic.claude-3-7-sonnet-20250219-v1:0")
        body = json.loads(provider._prepare_bedrock_request("hi", self._context()))
        assert body["messages"][-1]["content"] == [{"text": "hi"}]

    def test_usage_parsing_includes_cache_and_split(self):
        provider = self._provider("anthropic.claude-3-7-sonnet-20250219-v1:0")
        resp = provider._handle_response(
            httpx.Response(
                200,
                json={
                    "output": {
                        "message": {"role": "assistant", "content": [{"text": "ok"}]}
                    },
                    "usage": {
                        "inputTokens": 90,
                        "outputTokens": 12,
                        "cacheReadInputTokens": 800,
                        "cacheWriteInputTokens": 60,
                    },
                },
            )
        )
        assert resp.input_tokens == 90
        assert resp.output_tokens == 12
        assert resp.cache_read_input_tokens == 800
        assert resp.cache_creation_input_tokens == 60


class TestOpenRouterAnthropicCacheControl:
    """OpenRouter forwards Anthropic cache_control on claude models."""

    def test_claude_model_last_message_gets_cache_control(self):
        provider = OpenRouterProvider(api_key="k", model="anthropic/claude-3.5-sonnet")
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "latest"},
        ]
        provider._apply_cache_control(messages)
        assert messages[0]["content"] == "first"
        last = messages[-1]["content"]
        assert last[-1]["text"] == "latest"
        assert last[-1]["cache_control"] == {"type": "ephemeral"}

    def test_non_anthropic_model_untouched(self):
        provider = OpenRouterProvider(api_key="k", model="openai/gpt-4o")
        messages = [{"role": "user", "content": "latest"}]
        provider._apply_cache_control(messages)
        assert messages[0]["content"] == "latest"

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("OMNIMANCER_PROMPT_CACHE", "0")
        provider = OpenRouterProvider(api_key="k", model="anthropic/claude-3.5-sonnet")
        messages = [{"role": "user", "content": "latest"}]
        provider._apply_cache_control(messages)
        assert messages[0]["content"] == "latest"
