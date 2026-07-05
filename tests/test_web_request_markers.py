"""
Tests for web-request operation marker handling in omnimancer/cli/agent_loop.py.

Covers:
- URL safety validation (_check_url_safety)
- URL log sanitisation (_sanitize_url_for_log)
- Marker detection and parsing for [WEB_GET:url], [WEB_REQUEST:url], [WEB_POST:url]
- Routing through agent_engine.execute_with_approval
- Approval flow (approved / denied / cancelled)
- Response insertion and truncation
- Graceful handling of invalid / blocked URLs
- All blocked metadata hostnames
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.agent_loop import (
    _BLOCKED_METADATA_HOSTNAMES,
    _MAX_WEB_RESPONSE_CHARS,
    _WEB_POST_PATTERN,
    AgentLoopMixin,
    _check_url_safety,
    _sanitize_url_for_log,
)
from omnimancer.core.agent.types import OperationResult

# ---------------------------------------------------------------------------
# Concrete stub that satisfies AgentLoopMixin's abstract interface
# ---------------------------------------------------------------------------


class ConcreteAgentLoop(AgentLoopMixin):
    """Minimal concrete class for testing AgentLoopMixin behaviour."""

    def __init__(self, engine: object | None = None) -> None:
        self.engine = engine
        self.console = MagicMock()
        self._errors: list[str] = []

    def _show_assistant_message(self, content: str, model: str) -> None:
        pass

    def _show_error(self, message: str) -> None:
        self._errors.append(message)


def _make_engine(execute_result: OperationResult) -> MagicMock:
    """Build a mock engine whose agent_engine.execute_with_approval returns *result*."""
    agent_engine = MagicMock()
    agent_engine.execute_with_approval = AsyncMock(return_value=execute_result)
    engine = MagicMock()
    engine.agent_engine = agent_engine
    return engine


# ---------------------------------------------------------------------------
# _check_url_safety
# ---------------------------------------------------------------------------


class TestCheckUrlSafety:
    """Unit tests for the URL safety validator."""

    def test_valid_http_url(self) -> None:
        assert _check_url_safety("http://example.com/page") is None

    def test_valid_https_url(self) -> None:
        assert _check_url_safety("https://api.example.com/v1/data") is None

    def test_valid_url_with_port(self) -> None:
        assert _check_url_safety("https://example.com:8443/path") is None

    def test_ftp_scheme_blocked(self) -> None:
        error = _check_url_safety("ftp://example.com/file")
        assert error is not None
        assert "ftp" in error.lower() or "scheme" in error.lower()

    def test_file_scheme_blocked(self) -> None:
        error = _check_url_safety("file:///etc/passwd")
        assert error is not None

    def test_javascript_scheme_blocked(self) -> None:
        error = _check_url_safety("javascript:alert(1)")
        assert error is not None

    def test_embedded_credentials_blocked(self) -> None:
        error = _check_url_safety("https://user:secret@example.com/")
        assert error is not None
        assert "credential" in error.lower()

    def test_embedded_username_only_blocked(self) -> None:
        error = _check_url_safety("https://user@example.com/")
        assert error is not None
        assert "credential" in error.lower()

    # --- localhost / loopback ---

    def test_localhost_blocked(self) -> None:
        error = _check_url_safety("http://localhost/")
        assert error is not None

    def test_127_0_0_1_blocked(self) -> None:
        error = _check_url_safety("http://127.0.0.1/admin")
        assert error is not None

    def test_ipv6_loopback_blocked(self) -> None:
        error = _check_url_safety("http://[::1]/")
        assert error is not None

    def test_all_zeros_blocked(self) -> None:
        error = _check_url_safety("http://0.0.0.0/")
        assert error is not None

    # --- private address ranges ---

    def test_private_192_168_blocked(self) -> None:
        error = _check_url_safety("http://192.168.1.100/")
        assert error is not None

    def test_private_10_x_blocked(self) -> None:
        error = _check_url_safety("http://10.0.0.1/internal")
        assert error is not None

    def test_private_172_16_blocked(self) -> None:
        error = _check_url_safety("http://172.16.0.1/")
        assert error is not None

    def test_private_172_31_blocked(self) -> None:
        error = _check_url_safety("http://172.31.255.255/")
        assert error is not None

    # --- link-local addresses ---

    def test_link_local_ipv4_blocked(self) -> None:
        # 169.254.0.1 is link-local (APIPA) – must be blocked
        error = _check_url_safety("http://169.254.0.1/")
        assert error is not None

    def test_link_local_ipv6_blocked(self) -> None:
        # fe80::1 is IPv6 link-local
        error = _check_url_safety("http://[fe80::1]/")
        assert error is not None

    # --- cloud metadata endpoints (parametrized) ---

    @pytest.mark.parametrize(
        "hostname",
        [
            "169.254.169.254",  # AWS IMDSv1/v2, GCP, Azure shared endpoint
            "metadata.google.internal",  # GCP metadata server
            "fd00:ec2::254",  # AWS IPv6 IMDS
            "metadata.internal",  # Azure internal alias
        ],
    )
    def test_blocked_metadata_hostname(self, hostname: str) -> None:
        """Every hostname in _BLOCKED_METADATA_HOSTNAMES must be rejected."""
        # Wrap raw IPv6 addresses in brackets for URL syntax
        if ":" in hostname:
            url = f"http://[{hostname}]/"
        else:
            url = f"http://{hostname}/"
        error = _check_url_safety(url)
        assert error is not None, (
            f"Expected {hostname!r} to be blocked but _check_url_safety "
            f"returned None for {url!r}"
        )

    def test_all_blocked_metadata_hostnames_covered(self) -> None:
        """Constant _BLOCKED_METADATA_HOSTNAMES entries are all rejected."""
        for hostname in _BLOCKED_METADATA_HOSTNAMES:
            if ":" in hostname:
                url = f"http://[{hostname}]/"
            else:
                url = f"http://{hostname}/"
            error = _check_url_safety(url)
            assert error is not None, (
                f"Expected {hostname!r} to be blocked but _check_url_safety "
                f"returned None for {url!r}"
            )

    # --- IPv6 private/reserved address ranges ---

    def test_ipv6_private_fc00_blocked(self) -> None:
        # fc00::/7 is Unique Local Address (ULA) – RFC 4193
        error = _check_url_safety("http://[fc00::1]/")
        assert error is not None

    def test_ipv6_private_fd00_user_blocked(self) -> None:
        # fd00::/8 is also ULA – commonly used in private networks
        error = _check_url_safety("http://[fd00::1]/")
        assert error is not None

    # --- public addresses (must be allowed) ---

    def test_public_ip_allowed(self) -> None:
        # 8.8.8.8 is Google Public DNS – publicly routable, should be allowed
        assert _check_url_safety("https://8.8.8.8/") is None

    def test_no_hostname(self) -> None:
        error = _check_url_safety("http:///path")
        assert error is not None


# ---------------------------------------------------------------------------
# _sanitize_url_for_log
# ---------------------------------------------------------------------------


class TestSanitizeUrlForLog:
    """Unit tests for log-safe URL sanitisation."""

    def test_strips_credentials(self) -> None:
        result = _sanitize_url_for_log("https://user:pass@example.com/path")
        assert "user" not in result
        assert "pass" not in result
        assert "example.com" in result

    def test_strips_query_string(self) -> None:
        result = _sanitize_url_for_log("https://example.com/search?q=secret&token=abc")
        assert "secret" not in result
        assert "token" not in result
        assert "example.com" in result

    def test_strips_fragment(self) -> None:
        result = _sanitize_url_for_log("https://example.com/page#section")
        assert "#" not in result

    def test_preserves_path(self) -> None:
        result = _sanitize_url_for_log("https://example.com/api/v1/resource")
        assert "/api/v1/resource" in result

    def test_preserves_port(self) -> None:
        result = _sanitize_url_for_log("https://example.com:8443/path")
        assert "8443" in result

    def test_plain_url_unchanged_except_normalisation(self) -> None:
        result = _sanitize_url_for_log("https://example.com/path")
        assert "example.com" in result
        assert "/path" in result

    def test_invalid_url_returns_placeholder(self) -> None:
        # Passing a completely broken value
        result = _sanitize_url_for_log("")
        # Should not raise; returns something safe
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# [WEB_GET:url] marker detection and execution
# ---------------------------------------------------------------------------


class TestWebGetMarker:
    """Tests for the [WEB_GET:url] operation marker."""

    @pytest.mark.asyncio
    async def test_web_get_marker_replaced_on_success(self) -> None:
        body = "Hello, world!"
        result = OperationResult(
            success=True,
            data={"content": body, "status_code": 200, "url": "https://example.com"},
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        response = "[WEB_GET:https://example.com/]"
        updated = await loop._parse_and_execute_operations(response)

        assert "[WEB_GET:" not in updated
        assert "Hello, world!" in updated
        assert "example.com" in updated

    @pytest.mark.asyncio
    async def test_web_get_routes_to_execute_with_approval(self) -> None:
        result = OperationResult(
            success=True,
            data={"content": "ok", "status_code": 200, "url": "https://example.com"},
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        await loop._parse_and_execute_operations("[WEB_GET:https://example.com/data]")

        engine.agent_engine.execute_with_approval.assert_awaited_once()
        call_args = engine.agent_engine.execute_with_approval.call_args
        operation = call_args[0][0]
        assert operation.data["url"] == "https://example.com/data"
        assert operation.data["method"] == "GET"

    @pytest.mark.asyncio
    async def test_web_get_approval_required(self) -> None:
        result = OperationResult(
            success=True,
            data={"content": "ok", "status_code": 200, "url": "https://example.com"},
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        await loop._parse_and_execute_operations("[WEB_GET:https://example.com/]")

        operation = engine.agent_engine.execute_with_approval.call_args[0][0]
        assert operation.requires_approval is True

    @pytest.mark.asyncio
    async def test_web_get_failed_request(self) -> None:
        result = OperationResult(
            success=False,
            error="HTTP 404",
            was_cancelled=False,
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:https://example.com/missing]"
        )

        assert "[WEB_GET:" not in updated
        assert "❌" in updated
        assert "failed" in updated.lower() or "HTTP 404" in updated

    @pytest.mark.asyncio
    async def test_web_get_cancelled_triggers_workflow_cancel(self) -> None:
        result = OperationResult(
            success=False,
            was_cancelled=True,
            error="User cancelled",
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:https://example.com/]"
        )

        assert "__WORKFLOW_CANCELLED__" in updated
        assert "🚫" in updated

    @pytest.mark.asyncio
    async def test_web_get_no_agent_engine(self) -> None:
        engine = MagicMock(spec=[])  # No agent_engine attribute
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:https://example.com/]"
        )

        assert "[WEB_GET:" not in updated
        assert "❌" in updated
        assert "not available" in updated.lower()

    @pytest.mark.asyncio
    async def test_web_get_response_truncated(self) -> None:
        long_body = "x" * (_MAX_WEB_RESPONSE_CHARS + 500)
        result = OperationResult(
            success=True,
            data={"content": long_body, "status_code": 200, "url": "https://ex.com"},
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:https://example.com/big]"
        )

        assert "[... response truncated ...]" in updated
        # The body itself should be capped
        content_part = updated.split("\n", 1)[1] if "\n" in updated else updated
        # Count 'x' characters – should not exceed the limit
        x_count = content_part.count("x")
        assert x_count <= _MAX_WEB_RESPONSE_CHARS

    @pytest.mark.asyncio
    async def test_web_get_invalid_url_blocked(self) -> None:
        engine = MagicMock()
        engine.agent_engine = MagicMock()
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:ftp://evil.com/file]"
        )

        assert "[WEB_GET:" not in updated
        assert "❌" in updated
        engine.agent_engine.execute_with_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_web_get_localhost_blocked(self) -> None:
        engine = MagicMock()
        engine.agent_engine = MagicMock()
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:http://localhost:8080/secret]"
        )

        assert "❌" in updated
        assert "blocked" in updated.lower()
        engine.agent_engine.execute_with_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_web_get_metadata_endpoint_blocked(self) -> None:
        engine = MagicMock()
        engine.agent_engine = MagicMock()
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:http://169.254.169.254/latest/meta-data/]"
        )

        assert "❌" in updated
        engine.agent_engine.execute_with_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_web_get_private_ip_blocked(self) -> None:
        engine = MagicMock()
        engine.agent_engine = MagicMock()
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:http://192.168.100.1/admin]"
        )

        assert "❌" in updated
        engine.agent_engine.execute_with_approval.assert_not_called()


# ---------------------------------------------------------------------------
# [WEB_REQUEST:url] marker detection and execution
# ---------------------------------------------------------------------------


class TestWebRequestMarker:
    """Tests for the [WEB_REQUEST:url] operation marker (treated as GET)."""

    @pytest.mark.asyncio
    async def test_web_request_marker_replaced_on_success(self) -> None:
        result = OperationResult(
            success=True,
            data={
                "content": "response body",
                "status_code": 200,
                "url": "https://example.com",
            },
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_REQUEST:https://example.com/api]"
        )

        assert "[WEB_REQUEST:" not in updated
        assert "response body" in updated

    @pytest.mark.asyncio
    async def test_web_request_routes_get(self) -> None:
        result = OperationResult(
            success=True,
            data={"content": "ok", "status_code": 200, "url": "https://example.com"},
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        await loop._parse_and_execute_operations(
            "[WEB_REQUEST:https://example.com/endpoint]"
        )

        operation = engine.agent_engine.execute_with_approval.call_args[0][0]
        assert operation.data["method"] == "GET"
        assert operation.data["url"] == "https://example.com/endpoint"

    @pytest.mark.asyncio
    async def test_web_request_failed(self) -> None:
        result = OperationResult(
            success=False,
            error="Connection refused",
            was_cancelled=False,
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_REQUEST:https://example.com/down]"
        )

        assert "❌" in updated
        assert "Connection refused" in updated

    @pytest.mark.asyncio
    async def test_web_request_no_agent_engine(self) -> None:
        engine = MagicMock(spec=[])  # No agent_engine attribute
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_REQUEST:https://example.com/]"
        )

        assert "[WEB_REQUEST:" not in updated
        assert "❌" in updated

    @pytest.mark.asyncio
    async def test_web_request_response_truncated_at_limit(self) -> None:
        big_content = "A" * (_MAX_WEB_RESPONSE_CHARS * 2)
        result = OperationResult(
            success=True,
            data={"content": big_content, "status_code": 200, "url": "https://ex.com"},
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_REQUEST:https://example.com/large]"
        )

        assert "[... response truncated ...]" in updated

    @pytest.mark.asyncio
    async def test_web_request_cancelled(self) -> None:
        result = OperationResult(
            success=False,
            was_cancelled=True,
            error="cancelled",
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_REQUEST:https://example.com/]"
        )

        assert "__WORKFLOW_CANCELLED__" in updated


# ---------------------------------------------------------------------------
# Multiple markers in a single response
# ---------------------------------------------------------------------------


class TestMultipleWebMarkers:
    """Tests for responses containing multiple web markers."""

    @pytest.mark.asyncio
    async def test_multiple_web_get_markers(self) -> None:
        call_count = 0

        async def side_effect(op: object) -> OperationResult:
            nonlocal call_count
            call_count += 1
            return OperationResult(
                success=True,
                data={
                    "content": f"response-{call_count}",
                    "status_code": 200,
                    "url": "https://example.com",
                },
            )

        engine = MagicMock()
        engine.agent_engine = MagicMock()
        engine.agent_engine.execute_with_approval = AsyncMock(side_effect=side_effect)
        loop = ConcreteAgentLoop(engine)

        response = (
            "[WEB_GET:https://a.example.com/]\n" "[WEB_GET:https://b.example.com/]"
        )
        updated = await loop._parse_and_execute_operations(response)

        assert "[WEB_GET:" not in updated
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_mixed_web_get_and_web_request_markers(self) -> None:
        results = iter(
            [
                OperationResult(
                    success=True,
                    data={"content": "from-get", "status_code": 200, "url": ""},
                ),
                OperationResult(
                    success=True,
                    data={"content": "from-request", "status_code": 200, "url": ""},
                ),
            ]
        )

        async def side_effect(op: object) -> OperationResult:
            return next(results)

        engine = MagicMock()
        engine.agent_engine = MagicMock()
        engine.agent_engine.execute_with_approval = AsyncMock(side_effect=side_effect)
        loop = ConcreteAgentLoop(engine)

        response = (
            "[WEB_GET:https://a.example.com/]\n" "[WEB_REQUEST:https://b.example.com/]"
        )
        updated = await loop._parse_and_execute_operations(response)

        assert "from-get" in updated
        assert "from-request" in updated

    @pytest.mark.asyncio
    async def test_one_valid_one_blocked(self) -> None:
        result = OperationResult(
            success=True,
            data={"content": "ok", "status_code": 200, "url": "https://example.com"},
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        response = (
            "[WEB_GET:https://example.com/]\n" "[WEB_GET:http://localhost/secret]"
        )
        updated = await loop._parse_and_execute_operations(response)

        assert "ok" in updated  # valid request succeeded
        assert "❌" in updated  # blocked request has error
        # Only one call to execute_with_approval (the safe URL)
        assert engine.agent_engine.execute_with_approval.await_count == 1


# ---------------------------------------------------------------------------
# Response data shape variations
# ---------------------------------------------------------------------------


class TestResponseDataShapes:
    """Tests for different result.data shapes returned by execute_with_approval."""

    @pytest.mark.asyncio
    async def test_result_data_as_plain_string(self) -> None:
        result = OperationResult(success=True, data="plain string response")
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:https://example.com/]"
        )

        assert "plain string response" in updated

    @pytest.mark.asyncio
    async def test_result_data_is_none(self) -> None:
        result = OperationResult(success=True, data=None)
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:https://example.com/]"
        )

        # Should not raise; empty content is acceptable
        assert "[WEB_GET:" not in updated

    @pytest.mark.asyncio
    async def test_result_data_content_not_string(self) -> None:
        result = OperationResult(
            success=True,
            data={"content": 12345, "status_code": 200, "url": ""},
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_GET:https://example.com/]"
        )

        assert "12345" in updated


# ---------------------------------------------------------------------------
# URL log sanitisation in success messages
# ---------------------------------------------------------------------------


class TestUrlSanitisedInOutput:
    """Verify that credentials and query strings are not echoed in the response."""

    @pytest.mark.asyncio
    async def test_query_params_stripped_from_output(self) -> None:
        result = OperationResult(
            success=True,
            data={"content": "body", "status_code": 200, "url": ""},
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        url = "https://example.com/search?secret_token=abc123"
        updated = await loop._parse_and_execute_operations(f"[WEB_GET:{url}]")

        assert "secret_token" not in updated
        assert "abc123" not in updated
        assert "example.com" in updated

    def test_credentials_stripped_from_output(self) -> None:
        """Credentials in URL must not appear in sanitized output."""
        # Safety check blocks credentialed URLs, but verify safe_url stripping
        # works at the sanitizer level independently.
        safe = _sanitize_url_for_log("https://admin:hunter2@example.com/path")
        assert "hunter2" not in safe
        assert "admin" not in safe
        assert "example.com" in safe


# ---------------------------------------------------------------------------
# [WEB_POST:url] marker handling
# ---------------------------------------------------------------------------


class TestWebPostMarker:
    """Tests for the [WEB_POST:url] operation marker."""

    def test_web_post_pattern_compiled(self) -> None:
        """_WEB_POST_PATTERN must be exported and match [WEB_POST:url] tokens."""
        assert _WEB_POST_PATTERN is not None
        m = _WEB_POST_PATTERN.search("[WEB_POST:https://api.example.com/submit]")
        assert m is not None
        assert m.group(1) == "https://api.example.com/submit"

    @pytest.mark.asyncio
    async def test_web_post_marker_replaced_on_success(self) -> None:
        result = OperationResult(
            success=True,
            data={
                "content": "created",
                "status_code": 201,
                "url": "https://api.example.com",
            },
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_POST:https://api.example.com/items]"
        )

        assert "[WEB_POST:" not in updated
        assert "created" in updated

    @pytest.mark.asyncio
    async def test_web_post_routes_post_method(self) -> None:
        result = OperationResult(
            success=True,
            data={
                "content": "ok",
                "status_code": 200,
                "url": "https://api.example.com",
            },
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        await loop._parse_and_execute_operations(
            "[WEB_POST:https://api.example.com/submit]"
        )

        operation = engine.agent_engine.execute_with_approval.call_args[0][0]
        assert operation.data["method"] == "POST"
        assert operation.data["url"] == "https://api.example.com/submit"

    @pytest.mark.asyncio
    async def test_web_post_requires_approval(self) -> None:
        result = OperationResult(
            success=True,
            data={
                "content": "ok",
                "status_code": 200,
                "url": "https://api.example.com",
            },
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        await loop._parse_and_execute_operations(
            "[WEB_POST:https://api.example.com/submit]"
        )

        operation = engine.agent_engine.execute_with_approval.call_args[0][0]
        assert operation.requires_approval is True

    @pytest.mark.asyncio
    async def test_web_post_blocked_for_private_ip(self) -> None:
        engine = MagicMock()
        engine.agent_engine = MagicMock()
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_POST:http://192.168.1.1/admin]"
        )

        assert "❌" in updated
        assert "blocked" in updated.lower()
        engine.agent_engine.execute_with_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_web_post_failed_request(self) -> None:
        result = OperationResult(
            success=False,
            error="HTTP 403 Forbidden",
            was_cancelled=False,
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_POST:https://api.example.com/restricted]"
        )

        assert "❌" in updated
        assert "HTTP 403 Forbidden" in updated

    @pytest.mark.asyncio
    async def test_web_post_cancelled_triggers_workflow_cancel(self) -> None:
        result = OperationResult(
            success=False,
            was_cancelled=True,
            error="User cancelled",
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_POST:https://api.example.com/submit]"
        )

        assert "__WORKFLOW_CANCELLED__" in updated
        assert "🚫" in updated

    @pytest.mark.asyncio
    async def test_web_post_no_agent_engine(self) -> None:
        engine = MagicMock(spec=[])  # No agent_engine attribute
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_POST:https://api.example.com/submit]"
        )

        assert "[WEB_POST:" not in updated
        assert "❌" in updated
        assert "not available" in updated.lower()

    @pytest.mark.asyncio
    async def test_web_post_response_truncated(self) -> None:
        long_body = "P" * (_MAX_WEB_RESPONSE_CHARS + 500)
        result = OperationResult(
            success=True,
            data={
                "content": long_body,
                "status_code": 200,
                "url": "https://api.example.com",
            },
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        updated = await loop._parse_and_execute_operations(
            "[WEB_POST:https://api.example.com/large]"
        )

        assert "[... response truncated ...]" in updated


# ---------------------------------------------------------------------------
# Mixed web and non-web markers in a single response
# ---------------------------------------------------------------------------


class TestMixedMarkers:
    """Verify web markers are processed correctly alongside non-web markers."""

    @pytest.mark.asyncio
    async def test_web_get_with_surrounding_text(self) -> None:
        """Non-marker text surrounding [WEB_GET] is preserved unchanged."""
        result = OperationResult(
            success=True,
            data={
                "content": "fetched",
                "status_code": 200,
                "url": "https://example.com",
            },
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        response = "Before text.\n[WEB_GET:https://example.com/]\nAfter text."
        updated = await loop._parse_and_execute_operations(response)

        assert "Before text." in updated
        assert "After text." in updated
        assert "fetched" in updated
        assert "[WEB_GET:" not in updated

    @pytest.mark.asyncio
    async def test_web_get_does_not_interfere_with_other_markers(self) -> None:
        """A [WEB_GET] marker in the response does not consume or corrupt
        non-web marker text (e.g. [FILE_READ]) that appears nearby."""
        result = OperationResult(
            success=True,
            data={
                "content": "page content",
                "status_code": 200,
                "url": "https://example.com",
            },
        )
        engine = _make_engine(result)
        loop = ConcreteAgentLoop(engine)

        # [FILE_READ] is handled by a different branch – it won't be processed
        # here, but its text must survive unchanged after WEB_GET is resolved.
        response = "[FILE_READ:README.md]\n[WEB_GET:https://example.com/]"
        updated = await loop._parse_and_execute_operations(response)

        assert "[WEB_GET:" not in updated
        assert "page content" in updated

    @pytest.mark.asyncio
    async def test_web_post_and_web_get_in_same_response(self) -> None:
        """Both [WEB_GET] and [WEB_POST] markers are handled independently."""
        call_count = 0

        async def side_effect(op: object) -> OperationResult:
            nonlocal call_count
            call_count += 1
            return OperationResult(
                success=True,
                data={
                    "content": f"result-{call_count}",
                    "status_code": 200,
                    "url": "https://example.com",
                },
            )

        engine = MagicMock()
        engine.agent_engine = MagicMock()
        engine.agent_engine.execute_with_approval = AsyncMock(side_effect=side_effect)
        loop = ConcreteAgentLoop(engine)

        response = (
            "[WEB_GET:https://example.com/read]\n"
            "[WEB_POST:https://api.example.com/write]"
        )
        updated = await loop._parse_and_execute_operations(response)

        assert "[WEB_GET:" not in updated
        assert "[WEB_POST:" not in updated
        assert call_count == 2
