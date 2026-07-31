"""Tests for the openai-compatible provider type (self-hosted vLLM etc.).

Covers the behaviors that differ from the real OpenAI provider: keyless
auth is the norm, the model catalog comes from the endpoint (no GPT-name
filtering), and timeouts carry a cold-start hint because self-hosted
backends may load a model on first request.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core.models import ChatContext, ChatMessage, MessageRole
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.providers.openai_compatible import OpenAICompatibleProvider
from omnimancer.utils.errors import NetworkError


@pytest.fixture
def keyless_provider():
    return OpenAICompatibleProvider(
        api_key="",
        model="qwen3-coder-30b",
        base_url="http://localhost:8000/v1",
    )


@pytest.fixture
def chat_context():
    return ChatContext(
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content="Hello",
                timestamp=datetime.now(),
                model_used="qwen3-coder-30b",
            ),
        ],
        current_model="qwen3-coder-30b",
        session_id="test-session",
    )


def _ok_chat_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "id": "cmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "model": "qwen3-coder-30b",
    }
    return response


VLLM_MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "qwen3-coder-30b",
            "object": "model",
            "owned_by": "vllm",
            "max_model_len": 131072,
        },
        {
            "id": "qwen3-8b",
            "object": "model",
            "owned_by": "vllm",
            "max_model_len": 32768,
        },
    ],
}


class TestKeylessAuth:
    @pytest.mark.asyncio
    async def test_keyless_send_omits_authorization_header(
        self, keyless_provider, chat_context
    ):
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=_ok_chat_response())
            mock_client.return_value.__aenter__.return_value.post = mock_post

            response = await keyless_provider.send_message("Hello", chat_context)

        assert response.content == "Hi there"
        headers = mock_post.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_key_present_sends_authorization_header(self, chat_context):
        provider = OpenAICompatibleProvider(
            api_key="secret",
            model="m",
            base_url="http://localhost:8000/v1",
        )
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=_ok_chat_response())
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await provider.send_message("Hello", chat_context)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret"

    @pytest.mark.asyncio
    async def test_base_openai_auth_type_none_skips_key_requirement(self, chat_context):
        """The real openai provider keeps failing fast on a missing key,
        except when the config explicitly opts out via auth_type='none'."""
        provider = OpenAIProvider(
            api_key="",
            model="m",
            base_url="http://localhost:1234/v1",
            auth_type="none",
        )
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=_ok_chat_response())
            mock_client.return_value.__aenter__.return_value.post = mock_post

            response = await provider.send_message("Hello", chat_context)

        assert response.content == "Hi there"
        assert "Authorization" not in mock_post.call_args.kwargs["headers"]


class TestModelCatalog:
    def test_static_models_empty(self, keyless_provider):
        """No GPT catalog noise for self-hosted endpoints."""
        assert keyless_provider.get_available_models() == []

    @pytest.mark.asyncio
    async def test_fetch_live_models_uses_endpoint_list(self, keyless_provider):
        models_response = MagicMock()
        models_response.status_code = 200
        models_response.json.return_value = VLLM_MODELS_PAYLOAD
        models_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=models_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            models = await keyless_provider.fetch_live_models()

        names = {m.name for m in models}
        assert names == {"qwen3-coder-30b", "qwen3-8b"}
        by_name = {m.name: m for m in models}
        # Served max_model_len is the context truth
        assert by_name["qwen3-coder-30b"].max_tokens == 131072
        assert by_name["qwen3-8b"].max_tokens == 32768
        assert all(m.supports_tools for m in models)
        assert all(m.cost_per_token == 0.0 for m in models)
        # Keyless GET must omit Authorization
        headers = mock_get.call_args.kwargs["headers"]
        assert "Authorization" not in headers


class TestColdStartTimeout:
    @pytest.mark.asyncio
    async def test_timeout_error_carries_cold_start_hint(
        self, keyless_provider, chat_context
    ):
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
            mock_client.return_value.__aenter__.return_value.post = mock_post

            with pytest.raises(NetworkError) as exc:
                await keyless_provider.send_message("Hello", chat_context)

        message = str(exc.value)
        assert "timed out" in message
        assert "providers.<name>.timeout" in message


class TestCredentialValidation:
    @pytest.mark.asyncio
    async def test_validate_credentials_uses_models_endpoint(self, keyless_provider):
        models_response = MagicMock()
        models_response.status_code = 200
        models_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=models_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            assert await keyless_provider.validate_credentials() is True

        url = mock_get.call_args.args[0]
        assert url.endswith("/models")
