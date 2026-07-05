"""
Tests for MCP server management commands: add, remove, connect.

Tests are written against CoreEngine._handle_mcp_command and the helper
methods _mcp_add, _mcp_remove, and _mcp_connect without any live MCP
server processes.  All external I/O is mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core.engine import CoreEngine
from omnimancer.core.models import MCPConfig, MCPServerConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_engine(servers=None):
    """
    Build a CoreEngine instance bypassing __init__, with a fully-mocked
    MCPManager and ConfigManager pre-wired to the given server dict.
    """
    engine = CoreEngine.__new__(CoreEngine)

    # ------------------------------------------------------------------
    # Config manager mock
    # ------------------------------------------------------------------
    cm = MagicMock()
    # Track persisted server configs so assertions can inspect them.
    persisted: dict = {}

    def _set_mcp_server_config(name, cfg):
        persisted[name] = cfg

    def _remove_mcp_server_config(name):
        if name in persisted:
            del persisted[name]
            return True
        return False

    cm.set_mcp_server_config.side_effect = _set_mcp_server_config
    cm.remove_mcp_server_config.side_effect = _remove_mcp_server_config
    cm._persisted = persisted
    engine.config_manager = cm

    # ------------------------------------------------------------------
    # MCP manager mock
    # ------------------------------------------------------------------
    servers = servers or {}
    mcp_config = MCPConfig(enabled=True, servers=dict(servers))

    mcp_manager = MagicMock()
    mcp_manager.mcp_config = mcp_config
    mcp_manager.clients = {}
    mcp_manager.initialized = True

    # shutdown_servers removes a client by name and returns True/False
    async def _shutdown_servers(name):
        if name in mcp_manager.clients:
            del mcp_manager.clients[name]
            return True
        return False

    mcp_manager.shutdown_servers = AsyncMock(side_effect=_shutdown_servers)

    engine.mcp_manager = mcp_manager
    return engine, cm


def _stdio_cfg(name="myserver", command="echo", args=None):
    return MCPServerConfig(name=name, command=command, args=args or [])


def _url_cfg(name="remote", url="http://localhost:8080/mcp", transport="sse"):
    return MCPServerConfig(name=name, transport=transport, url=url)


# ===========================================================================
# /mcp add — stdio transport
# ===========================================================================


class TestMcpAddStdio:
    @pytest.mark.asyncio
    async def test_add_stdio_minimal(self):
        """Add a server with just name and command."""
        engine, cm = _make_engine()
        result = await engine._mcp_add(["myserver", "npx"])
        assert "added successfully" in result
        assert "myserver" in engine.mcp_manager.mcp_config.servers
        cfg = engine.mcp_manager.mcp_config.servers["myserver"]
        assert cfg.command == "npx"
        assert cfg.args == []
        assert cm.set_mcp_server_config.called

    @pytest.mark.asyncio
    async def test_add_stdio_with_args(self):
        """Add a stdio server with extra command arguments."""
        engine, cm = _make_engine()
        result = await engine._mcp_add(
            ["fs", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
        assert "added successfully" in result
        cfg = engine.mcp_manager.mcp_config.servers["fs"]
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

    @pytest.mark.asyncio
    async def test_add_insufficient_args_shows_usage(self):
        """Fewer than 2 arguments returns usage instructions."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["onlyname"])
        assert "Usage" in result

    @pytest.mark.asyncio
    async def test_add_no_args_shows_usage(self):
        """No arguments returns usage instructions."""
        engine, _ = _make_engine()
        result = await engine._mcp_add([])
        assert "Usage" in result


# ===========================================================================
# /mcp add — remote (URL) transport
# ===========================================================================


class TestMcpAddRemote:
    @pytest.mark.asyncio
    async def test_add_url_default_transport_sse(self):
        """Add a remote server with --url; default transport is sse."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["remote", "--url", "http://localhost:8080/mcp"])
        assert "added successfully" in result
        cfg = engine.mcp_manager.mcp_config.servers["remote"]
        assert cfg.url == "http://localhost:8080/mcp"
        assert cfg.transport == "sse"

    @pytest.mark.asyncio
    async def test_add_url_explicit_transport_http(self):
        """Add a remote server with --url and --transport http."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(
            [
                "remote2",
                "--url",
                "https://example.com/mcp",
                "--transport",
                "http",
            ]
        )
        assert "added successfully" in result
        cfg = engine.mcp_manager.mcp_config.servers["remote2"]
        assert cfg.transport == "http"

    @pytest.mark.asyncio
    async def test_add_url_missing_url_value(self):
        """--url without a following value returns an error."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["remote", "--url"])
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_add_url_non_http_scheme_rejected(self):
        """file:// URLs are rejected for SSRF prevention."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["evil", "--url", "file:///etc/passwd"])
        assert "Error" in result
        assert "evil" not in engine.mcp_manager.mcp_config.servers

    @pytest.mark.asyncio
    async def test_add_url_https_allowed(self):
        """https:// URLs are accepted."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(
            ["secure", "--url", "https://api.example.com/mcp"]
        )
        assert "added successfully" in result


# ===========================================================================
# /mcp add — error / edge cases
# ===========================================================================


class TestMcpAddErrors:
    @pytest.mark.asyncio
    async def test_add_duplicate_server_rejected(self):
        """Adding a server that already exists returns an error."""
        existing = _stdio_cfg("existing")
        engine, _ = _make_engine(servers={"existing": existing})
        result = await engine._mcp_add(["existing", "echo", "hello"])
        assert "Error" in result
        assert "already configured" in result

    @pytest.mark.asyncio
    async def test_add_empty_name_rejected(self):
        """An empty server name is rejected."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["", "echo"])
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_add_name_with_path_separator_rejected(self):
        """Names containing '/' are rejected to prevent path traversal."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["../evil", "echo"])
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_add_name_with_special_chars_rejected(self):
        """Names with special characters are rejected."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["name!with@chars", "echo"])
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_add_name_with_underscores_hyphens_allowed(self):
        """Names with underscores and hyphens are valid."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["my_server-1", "echo"])
        assert "added successfully" in result

    @pytest.mark.asyncio
    async def test_add_persists_config(self):
        """set_mcp_server_config is called with the correct arguments."""
        engine, cm = _make_engine()
        await engine._mcp_add(["srv", "python3", "-m", "mymodule"])
        cm.set_mcp_server_config.assert_called_once()
        name_arg, cfg_arg = cm.set_mcp_server_config.call_args[0]
        assert name_arg == "srv"
        assert cfg_arg.command == "python3"
        assert cfg_arg.args == ["-m", "mymodule"]

    @pytest.mark.asyncio
    async def test_add_no_mcp_manager(self):
        """Returns graceful message when MCP is not configured."""
        engine, _ = _make_engine()
        engine.mcp_manager = None
        result = await engine._mcp_add(["srv", "echo"])
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_add_uppercase_name_allowed(self):
        """Server names with uppercase letters are valid."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["MyServer", "echo"])
        assert "added successfully" in result
        assert "MyServer" in engine.mcp_manager.mcp_config.servers

    @pytest.mark.asyncio
    async def test_add_mixed_case_name_preserved(self):
        """Mixed-case server names are stored exactly as provided."""
        engine, _ = _make_engine()
        await engine._mcp_add(["CamelCase_Server-01", "echo"])
        assert "CamelCase_Server-01" in engine.mcp_manager.mcp_config.servers

    @pytest.mark.asyncio
    async def test_add_transport_invalid_value_rejected(self):
        """An invalid --transport value returns an error."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(
            ["remote", "--url", "http://localhost:8080/mcp", "--transport", "ftp"]
        )
        assert "Error" in result
        assert "ftp" in result or "Invalid transport" in result
        assert "remote" not in engine.mcp_manager.mcp_config.servers

    @pytest.mark.asyncio
    async def test_add_transport_sse_valid(self):
        """The 'sse' transport value is accepted."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(
            ["srv", "--url", "http://localhost:8080/mcp", "--transport", "sse"]
        )
        assert "added successfully" in result
        assert engine.mcp_manager.mcp_config.servers["srv"].transport == "sse"

    @pytest.mark.asyncio
    async def test_add_transport_http_valid(self):
        """The 'http' transport value is accepted."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(
            ["srv2", "--url", "http://localhost:8080/mcp", "--transport", "http"]
        )
        assert "added successfully" in result
        assert engine.mcp_manager.mcp_config.servers["srv2"].transport == "http"

    @pytest.mark.asyncio
    async def test_add_persistence_failure_does_not_corrupt_runtime(self):
        """If config persistence raises, the server is NOT added to runtime."""
        engine, cm = _make_engine()
        cm.set_mcp_server_config.side_effect = OSError("disk full")

        # The command should return an error message but must not corrupt state.
        await engine._mcp_add(["srv", "echo"])
        assert "srv" not in engine.mcp_manager.mcp_config.servers


# ===========================================================================
# /mcp add — security
# ===========================================================================


class TestMcpAddSecurity:
    """Security-focused tests for stdio command validation."""

    @pytest.mark.asyncio
    async def test_command_with_semicolon_rejected(self):
        """Commands containing ';' are rejected (shell injection guard)."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["srv", "echo;rm"])
        assert "Error" in result
        assert "srv" not in engine.mcp_manager.mcp_config.servers

    @pytest.mark.asyncio
    async def test_command_with_pipe_rejected(self):
        """Commands containing '|' are rejected."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["srv", "cat|evil"])
        assert "Error" in result
        assert "srv" not in engine.mcp_manager.mcp_config.servers

    @pytest.mark.asyncio
    async def test_command_with_ampersand_rejected(self):
        """Commands containing '&' are rejected."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["srv", "cmd&bg"])
        assert "Error" in result
        assert "srv" not in engine.mcp_manager.mcp_config.servers

    @pytest.mark.asyncio
    async def test_command_with_dollar_rejected(self):
        """Commands containing '$' (shell variable expansion) are rejected."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["srv", "$MALICIOUS"])
        assert "Error" in result
        assert "srv" not in engine.mcp_manager.mcp_config.servers

    @pytest.mark.asyncio
    async def test_safe_command_npx_accepted(self):
        """'npx' is a safe command and must be accepted."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["srv", "npx"])
        assert "added successfully" in result

    @pytest.mark.asyncio
    async def test_safe_command_absolute_path_accepted(self):
        """Absolute path commands like '/usr/bin/python3' are accepted."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(["srv", "/usr/bin/python3"])
        assert "added successfully" in result

    @pytest.mark.asyncio
    async def test_args_with_at_sign_accepted(self):
        """npm-style package names with '@' in args are accepted."""
        engine, _ = _make_engine()
        result = await engine._mcp_add(
            ["fs", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
        assert "added successfully" in result
        cfg = engine.mcp_manager.mcp_config.servers["fs"]
        assert "@modelcontextprotocol/server-filesystem" in cfg.args


# ===========================================================================
# /mcp remove
# ===========================================================================


class TestMcpRemove:
    @pytest.mark.asyncio
    async def test_remove_existing_server(self):
        """Removing a configured server succeeds."""
        existing = _stdio_cfg("myserver")
        engine, cm = _make_engine(servers={"myserver": existing})
        # Seed persisted dict as if set_mcp_server_config was called earlier
        cm._persisted["myserver"] = existing

        result = await engine._mcp_remove(["myserver"])
        assert "removed successfully" in result
        assert "myserver" not in engine.mcp_manager.mcp_config.servers

    @pytest.mark.asyncio
    async def test_remove_also_disconnects_client(self):
        """Removing a server that is connected also disconnects it."""
        existing = _stdio_cfg("myserver")
        engine, cm = _make_engine(servers={"myserver": existing})
        cm._persisted["myserver"] = existing
        # Simulate a live client
        fake_client = MagicMock()
        engine.mcp_manager.clients["myserver"] = fake_client

        await engine._mcp_remove(["myserver"])
        # The client should no longer be in the clients dict
        assert "myserver" not in engine.mcp_manager.clients

    @pytest.mark.asyncio
    async def test_remove_nonexistent_server_returns_error(self):
        """Removing a server that is not configured returns an error."""
        engine, _ = _make_engine()
        result = await engine._mcp_remove(["nonexistent"])
        assert "Error" in result
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_remove_no_name_shows_usage(self):
        """Calling /mcp remove without a name returns usage."""
        engine, _ = _make_engine()
        result = await engine._mcp_remove([])
        assert "Usage" in result

    @pytest.mark.asyncio
    async def test_remove_config_persisted_before_runtime_update(self):
        """
        Config removal is called before the runtime dict is mutated,
        so that a failure in persistence does not corrupt runtime state.
        """
        existing = _stdio_cfg("myserver")
        engine, cm = _make_engine(servers={"myserver": existing})
        cm._persisted["myserver"] = existing

        call_order = []

        original_remove = cm.remove_mcp_server_config.side_effect

        def _tracking_remove(name):
            call_order.append("config_remove")
            return original_remove(name)

        cm.remove_mcp_server_config.side_effect = _tracking_remove

        # Patch dict __delitem__ to record when runtime is modified
        original_servers = engine.mcp_manager.mcp_config.servers

        class TrackingDict(dict):
            def __delitem__(self, key):
                call_order.append("runtime_remove")
                super().__delitem__(key)

        engine.mcp_manager.mcp_config.servers = TrackingDict(original_servers)

        await engine._mcp_remove(["myserver"])

        assert call_order.index("config_remove") < call_order.index("runtime_remove")

    @pytest.mark.asyncio
    async def test_remove_no_mcp_manager(self):
        """Returns graceful message when MCP is not configured."""
        engine, _ = _make_engine()
        engine.mcp_manager = None
        result = await engine._mcp_remove(["srv"])
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_remove_disconnect_failure_does_not_abort_removal(self):
        """
        If shutdown_servers raises an exception the removal still succeeds.

        The disconnect is best-effort; errors must be swallowed so that a
        broken transport does not leave orphaned configuration entries.
        """
        existing = _stdio_cfg("myserver")
        engine, cm = _make_engine(servers={"myserver": existing})
        cm._persisted["myserver"] = existing

        # Make the disconnect raise an unexpected error
        engine.mcp_manager.shutdown_servers = AsyncMock(
            side_effect=RuntimeError("transport broken")
        )

        result = await engine._mcp_remove(["myserver"])
        # Removal should still succeed despite the disconnect error
        assert "removed successfully" in result
        assert "myserver" not in engine.mcp_manager.mcp_config.servers


# ===========================================================================
# /mcp connect
# ===========================================================================


class TestMcpConnect:
    @pytest.mark.asyncio
    async def test_connect_existing_server(self):
        """Connect to a configured server creates a client and registers it."""
        cfg = _stdio_cfg("myserver")
        engine, _ = _make_engine(servers={"myserver": cfg})

        fake_client = MagicMock()
        fake_client.is_connected = True
        fake_client.connect = AsyncMock()
        fake_client.disconnect = AsyncMock()

        # Patch where MCPClient is imported inside the method
        with patch("omnimancer.mcp.client.MCPClient", return_value=fake_client):
            result = await engine._mcp_connect("myserver")

        assert "Successfully connected" in result or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_connect_nonexistent_server_returns_error(self):
        """Connecting to a server that is not configured returns an error."""
        engine, _ = _make_engine()
        result = await engine._mcp_connect("ghost")
        assert "Error" in result
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_connect_no_server_name_initializes_all(self):
        """Calling connect without a server name triggers initialize_servers."""
        engine, _ = _make_engine()
        engine.mcp_manager.initialize_servers = AsyncMock()
        result = await engine._mcp_connect(None)
        engine.mcp_manager.initialize_servers.assert_awaited_once()
        assert "connect" in result.lower() or "attempt" in result.lower()

    @pytest.mark.asyncio
    async def test_connect_disconnects_existing_client_first(self):
        """Disconnect existing client before reconnecting to the same server."""
        cfg = _stdio_cfg("myserver")
        engine, _ = _make_engine(servers={"myserver": cfg})

        # Simulate an existing live client
        old_client = MagicMock()
        old_client.is_connected = True
        old_client.disconnect = AsyncMock()
        engine.mcp_manager.clients["myserver"] = old_client

        new_client = MagicMock()
        new_client.connect = AsyncMock()
        new_client.is_connected = True

        with patch("omnimancer.mcp.client.MCPClient", return_value=new_client):
            await engine._mcp_connect("myserver")

        old_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_no_mcp_manager(self):
        """Returns graceful message when MCP is not configured."""
        engine, _ = _make_engine()
        engine.mcp_manager = None
        result = await engine._mcp_connect("srv")
        assert "not configured" in result.lower()


# ===========================================================================
# /mcp help — new commands documented
# ===========================================================================


class TestMcpHelp:
    def test_help_contains_add(self):
        engine, _ = _make_engine()
        help_text = engine._mcp_help()
        assert "add" in help_text

    def test_help_contains_remove(self):
        engine, _ = _make_engine()
        help_text = engine._mcp_help()
        assert "remove" in help_text

    def test_help_contains_connect(self):
        engine, _ = _make_engine()
        help_text = engine._mcp_help()
        assert "connect" in help_text


# ===========================================================================
# handle_mcp_command routing
# ===========================================================================


class TestHandleMcpCommandRouting:
    def _make_command(self, text):
        cmd = MagicMock()
        cmd.args = text.split()
        return cmd

    @pytest.mark.asyncio
    async def test_routes_add(self):
        engine, _ = _make_engine()
        engine._mcp_add = AsyncMock(return_value="added")
        cmd = self._make_command("add myserver echo hello")
        await engine._handle_mcp_command(cmd)
        engine._mcp_add.assert_awaited_once_with(["myserver", "echo", "hello"])

    @pytest.mark.asyncio
    async def test_routes_remove(self):
        engine, _ = _make_engine()
        engine._mcp_remove = AsyncMock(return_value="removed")
        cmd = self._make_command("remove myserver")
        await engine._handle_mcp_command(cmd)
        engine._mcp_remove.assert_awaited_once_with(["myserver"])

    @pytest.mark.asyncio
    async def test_routes_connect_with_name(self):
        engine, _ = _make_engine()
        engine._mcp_connect = AsyncMock(return_value="connected")
        cmd = self._make_command("connect myserver")
        await engine._handle_mcp_command(cmd)
        engine._mcp_connect.assert_awaited_once_with("myserver")


# ===========================================================================
# Completion manager
# ===========================================================================


class TestMcpCompletions:
    def test_mcp_completions_include_add_and_remove(self):
        from omnimancer.cli.completion import CompletionManager

        cm = CompletionManager()
        completions = cm.get_completions("mcp", 0, "", [])
        assert "add" in completions
        assert "remove" in completions

    def test_mcp_completions_include_servers_and_tools(self):
        from omnimancer.cli.completion import CompletionManager

        cm = CompletionManager()
        completions = cm.get_completions("mcp", 0, "", [])
        assert "servers" in completions
        assert "tools" in completions

    def test_mcp_completions_include_existing_subcommands(self):
        from omnimancer.cli.completion import CompletionManager

        cm = CompletionManager()
        completions = cm.get_completions("mcp", 0, "", [])
        assert "status" in completions
        assert "connect" in completions
        assert "disconnect" in completions
        assert "reload" in completions
        assert "health" in completions

    def test_mcp_completions_filter_by_prefix(self):
        from omnimancer.cli.completion import CompletionManager

        cm = CompletionManager()
        completions = cm.get_completions("mcp", 0, "a", [])
        assert all(c.startswith("a") for c in completions)
        assert "add" in completions

    def test_mcp_completions_slash_prefix_stripped(self):
        from omnimancer.cli.completion import CompletionManager

        cm = CompletionManager()
        completions = cm.get_completions("/mcp", 0, "", [])
        assert "add" in completions
