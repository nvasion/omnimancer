"""Tests for Claude subscription OAuth credential loading."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.providers.claude_credentials import (
    ClaudeCredentials,
    load_claude_credentials,
    refresh_claude_token,
)


@pytest.fixture
def valid_credentials_data():
    return {
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-test-access-token",
            "refreshToken": "sk-ant-ort01-test-refresh-token",
            "expiresAt": int((time.time() + 3600) * 1000),
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": "max",
            "rateLimitTier": "default_claude_max_5x",
        }
    }


@pytest.fixture
def expired_credentials_data():
    return {
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat01-expired-token",
            "refreshToken": "sk-ant-ort01-test-refresh-token",
            "expiresAt": int((time.time() - 3600) * 1000),
            "scopes": ["user:inference"],
            "subscriptionType": "pro",
            "rateLimitTier": "default",
        }
    }


class TestClaudeCredentials:
    def test_from_dict(self, valid_credentials_data):
        creds = ClaudeCredentials.from_dict(valid_credentials_data)
        assert creds.access_token == "sk-ant-oat01-test-access-token"
        assert creds.refresh_token == "sk-ant-ort01-test-refresh-token"
        assert creds.subscription_type == "max"
        assert not creds.is_expired

    def test_is_expired_true(self, expired_credentials_data):
        creds = ClaudeCredentials.from_dict(expired_credentials_data)
        assert creds.is_expired

    def test_is_expired_false(self, valid_credentials_data):
        creds = ClaudeCredentials.from_dict(valid_credentials_data)
        assert not creds.is_expired

    def test_is_expired_with_buffer(self):
        data = {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-test",
                "refreshToken": "sk-ant-ort01-test",
                "expiresAt": int((time.time() + 30) * 1000),
                "scopes": [],
                "subscriptionType": "pro",
                "rateLimitTier": "default",
            }
        }
        creds = ClaudeCredentials.from_dict(data)
        assert creds.is_expired

    def test_from_dict_missing_oauth_key(self):
        with pytest.raises(KeyError):
            ClaudeCredentials.from_dict({"someOtherKey": {}})

    def test_from_dict_missing_fields(self):
        with pytest.raises(KeyError):
            ClaudeCredentials.from_dict({"claudeAiOauth": {"accessToken": "x"}})


class TestLoadClaudeCredentials:
    def test_load_from_file(self, tmp_path, valid_credentials_data):
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps(valid_credentials_data))
        creds = load_claude_credentials(str(creds_file))
        assert creds is not None
        assert creds.access_token == "sk-ant-oat01-test-access-token"

    def test_load_from_default_path(self, valid_credentials_data):
        creds_json = json.dumps(valid_credentials_data)
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=creds_json),
        ):
            creds = load_claude_credentials()
            assert creds is not None

    def test_load_file_not_found(self):
        creds = load_claude_credentials("/nonexistent/path/.credentials.json")
        assert creds is None

    def test_load_invalid_json(self, tmp_path):
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text("not json")
        creds = load_claude_credentials(str(creds_file))
        assert creds is None

    def test_load_missing_oauth_section(self, tmp_path):
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps({"other": "data"}))
        creds = load_claude_credentials(str(creds_file))
        assert creds is None


class TestRefreshClaudeToken:
    @pytest.mark.asyncio
    async def test_refresh_success(self, tmp_path, valid_credentials_data):
        creds_file = tmp_path / ".credentials.json"
        creds_file.write_text(json.dumps(valid_credentials_data))

        int((time.time() + 7200) * 1000)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "sk-ant-oat01-new-token",
            "refresh_token": "sk-ant-ort01-new-refresh",
            "expires_in": 7200,
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            old_creds = ClaudeCredentials.from_dict(valid_credentials_data)
            new_creds = await refresh_claude_token(old_creds, str(creds_file))

            assert new_creds is not None
            assert new_creds.access_token == "sk-ant-oat01-new-token"
            assert new_creds.refresh_token == "sk-ant-ort01-new-refresh"

    @pytest.mark.asyncio
    async def test_refresh_failure_returns_none(self, valid_credentials_data):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            old_creds = ClaudeCredentials.from_dict(valid_credentials_data)
            new_creds = await refresh_claude_token(old_creds)
            assert new_creds is None

    @pytest.mark.asyncio
    async def test_refresh_network_error_returns_none(self, valid_credentials_data):
        import httpx

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("fail"))
            mock_client_cls.return_value = mock_client

            old_creds = ClaudeCredentials.from_dict(valid_credentials_data)
            new_creds = await refresh_claude_token(old_creds)
            assert new_creds is None


class TestClaudeProviderBearerAuth:
    """Test that ClaudeProvider uses bearer auth when configured."""

    @pytest.mark.asyncio
    async def test_bearer_auth_header(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="sk-ant-oat01-bearer-token",
            model="claude-sonnet-4-6",
            auth_type="bearer",
        )
        assert provider.auth_type == "bearer"

    @pytest.mark.asyncio
    async def test_api_key_auth_default(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="sk-ant-api03-normal-key",
            model="claude-sonnet-4-6",
        )
        assert provider.auth_type == "api_key"

    @pytest.mark.asyncio
    async def test_bearer_sends_authorization_header(self):
        from omnimancer.core.models import ChatContext
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="sk-ant-oat01-bearer-token",
            model="claude-sonnet-4-6",
            auth_type="bearer",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }

        context = ChatContext(messages=[], current_model="test", session_id="test")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await provider.send_message("hi", context)

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer sk-ant-oat01-bearer-token"
            assert "x-api-key" not in headers

    @pytest.mark.asyncio
    async def test_api_key_sends_x_api_key_header(self):
        from omnimancer.core.models import ChatContext
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="sk-ant-api03-normal-key",
            model="claude-sonnet-4-6",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }

        context = ChatContext(messages=[], current_model="test", session_id="test")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await provider.send_message("hi", context)

            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers", {})
            assert "x-api-key" in headers
            assert headers["x-api-key"] == "sk-ant-api03-normal-key"
            assert "Authorization" not in headers


class TestClaudeProviderBetaHeaders:
    """Test beta header construction for different auth/model combos."""

    def test_bearer_sonnet_includes_claude_code_beta(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="token", model="claude-sonnet-4-6", auth_type="bearer"
        )
        headers = provider._build_headers()
        assert "claude-code-20250219" in headers["anthropic-beta"]
        assert "oauth-2025-04-20" in headers["anthropic-beta"]

    def test_bearer_haiku_omits_claude_code_beta(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="token", model="claude-haiku-4-5-20251001", auth_type="bearer"
        )
        headers = provider._build_headers()
        assert "oauth-2025-04-20" in headers["anthropic-beta"]
        assert "claude-code-20250219" not in headers["anthropic-beta"]

    def test_api_key_has_no_beta_header(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(api_key="sk-key", model="claude-sonnet-4-6")
        headers = provider._build_headers()
        assert "anthropic-beta" not in headers

    def test_bearer_opus_includes_claude_code_beta(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="token", model="claude-opus-4-6", auth_type="bearer"
        )
        headers = provider._build_headers()
        assert "claude-code-20250219" in headers["anthropic-beta"]


class TestSubscription429Detection:
    """Test detection of fake 429s from subscription token + non-Haiku models."""

    def test_detects_subscription_429_for_sonnet(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="token", model="claude-sonnet-4-6", auth_type="bearer"
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Error"},
        }
        mock_response.headers = {"content-type": "application/json"}
        assert provider._is_subscription_429(mock_response) is True

    def test_real_rate_limit_not_flagged(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="token", model="claude-sonnet-4-6", auth_type="bearer"
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"},
        }
        mock_response.headers = {
            "anthropic-ratelimit-unified-requests-limit": "100",
        }
        assert provider._is_subscription_429(mock_response) is False

    def test_haiku_429_not_flagged(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="token", model="claude-haiku-4-5-20251001", auth_type="bearer"
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Error"},
        }
        mock_response.headers = {}
        assert provider._is_subscription_429(mock_response) is False

    def test_api_key_429_not_flagged(self):
        from omnimancer.providers.claude import ClaudeProvider

        provider = ClaudeProvider(
            api_key="sk-key", model="claude-sonnet-4-6"
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Error"},
        }
        mock_response.headers = {}
        assert provider._is_subscription_429(mock_response) is False

    @pytest.mark.asyncio
    async def test_subscription_429_raises_provider_error_not_rate_limit(self):
        from omnimancer.core.models import ChatContext
        from omnimancer.providers.claude import ClaudeProvider
        from omnimancer.utils.errors import ProviderError

        provider = ClaudeProvider(
            api_key="token", model="claude-sonnet-4-6", auth_type="bearer"
        )

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Error"},
        }
        mock_response.headers = {"content-type": "application/json"}

        context = ChatContext(messages=[], current_model="test", session_id="test")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            with pytest.raises(
                ProviderError,
                match="subscription tokens only support Haiku",
            ):
                await provider.send_message("hi", context)
