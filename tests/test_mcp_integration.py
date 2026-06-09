"""
Integration tests for MCP (Model Context Protocol) functionality.

This module tests the MCP client, manager, and integration with providers
including server connection, tool discovery, tool execution, and error handling.
"""

import asyncio
import os
import subprocess
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from omnimancer.core.models import (
    MCPConfig,
    MCPError,
    MCPServerConfig,
    ToolDefinition,
    ToolResult,
)
from omnimancer.mcp.client import MCPClient
from omnimancer.mcp.manager import MCPManager
from omnimancer.utils.errors import MCPConnectionError, MCPTimeoutError


@pytest.fixture
def sample_mcp_server_config():
    """Create a sample MCP server configuration."""
    return MCPServerConfig(
        name="test_server",
        command="python3",
        args=["-m", "test_mcp_server"],
        env={"TEST_ENV": "test_value"},
        enabled=True,
        auto_approve=["safe_tool"],
        timeout=10.0,
    )


@pytest.fixture
def sample_mcp_config(sample_mcp_server_config):
    """Create a sample MCP configuration."""
    return MCPConfig(
        enabled=True,
        servers={"test_server": sample_mcp_server_config},
        max_concurrent_servers=5,
    )


@pytest.fixture
def mock_mcp_process():
    """Create a mock MCP server process."""
    process = MagicMock(spec=subprocess.Popen)
    process.poll.return_value = None  # Process is running
    process.stdin = MagicMock()
    process.stdout = MagicMock()
    process.stderr = MagicMock()
    process.wait.return_value = 0
    return process


@pytest.fixture
def sample_tool_definitions():
    """Create sample tool definitions."""
    return [
        ToolDefinition(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
            server_name="test_server",
            auto_approved=False,
        ),
        ToolDefinition(
            name="safe_tool",
            description="A safe tool that's auto-approved",
            parameters={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input parameter",
                    }
                },
                "required": ["input"],
            },
            server_name="test_server",
            auto_approved=True,
        ),
    ]


TEST_SERVER = os.path.join(os.path.dirname(__file__), "mcp_test_server.py")


@pytest.fixture
def stdio_config():
    """Config pointing at the real FastMCP test server (tests/mcp_test_server.py)."""
    return MCPServerConfig(
        name="test_server",
        command=sys.executable,
        args=[TEST_SERVER],
        auto_approve=["add"],
        timeout=30,
    )


@pytest_asyncio.fixture
async def connected_client(stdio_config):
    """A live MCPClient connected to the stdio test server."""
    client = MCPClient(stdio_config)
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


class TestMCPClient:
    """Test the SDK-based MCP client against a real stdio server."""

    def test_client_initialization(self, stdio_config):
        client = MCPClient(stdio_config)
        assert client.server_config == stdio_config
        assert client.connected is False
        assert client.is_connected is False
        assert client.tools == {}

    def test_transport_selection_builds_without_error(self):
        # stdio
        c = MCPClient(MCPServerConfig(name="s", command="echo hi", transport="stdio"))
        c._build_transport()
        # sse / http build their context managers from a url
        for transport in ("sse", "http"):
            cfg = MCPServerConfig(
                name="r",
                transport=transport,
                url="https://example.com/mcp",
                headers={"Authorization": "Bearer x"},
            )
            MCPClient(cfg)._build_transport()

    @pytest.mark.asyncio
    async def test_connect_discovers_tools(self, connected_client):
        assert connected_client.is_connected is True
        assert "add" in connected_client.tools
        tool = connected_client.get_tool_definition("add")
        assert tool is not None
        assert tool.server_name == "test_server"
        # auto_approve from config is honoured.
        assert tool.auto_approved is True

    @pytest.mark.asyncio
    async def test_list_tools(self, connected_client):
        tools = await connected_client.list_tools()
        assert any(t.name == "add" for t in tools)

    @pytest.mark.asyncio
    async def test_call_tool(self, connected_client):
        result = await connected_client.call_tool("add", {"a": 2, "b": 3})
        assert isinstance(result, ToolResult)
        assert result.error is None
        assert "5" in result.content

    @pytest.mark.asyncio
    async def test_list_and_read_resources(self, connected_client):
        resources = await connected_client.list_resources()
        assert any(r["uri"] == "test://greeting" for r in resources)
        content = await connected_client.read_resource("test://greeting")
        assert "hello from resource" in content

    @pytest.mark.asyncio
    async def test_list_and_get_prompts(self, connected_client):
        prompts = await connected_client.list_prompts()
        names = {p["name"] for p in prompts}
        assert "greet" in names
        rendered = await connected_client.get_prompt("greet", {"name": "Kellan"})
        assert "Hello, Kellan!" in rendered

    @pytest.mark.asyncio
    async def test_health_check_connected(self, connected_client):
        assert await connected_client.health_check() is True

    @pytest.mark.asyncio
    async def test_disconnect(self, stdio_config):
        client = MCPClient(stdio_config)
        await client.connect()
        assert client.is_connected is True
        await client.disconnect()
        assert client.is_connected is False
        assert client.connected is False

    @pytest.mark.asyncio
    async def test_call_tool_not_connected_raises(self, stdio_config):
        client = MCPClient(stdio_config)
        with pytest.raises(MCPConnectionError):
            await client.call_tool("add", {"a": 1, "b": 2})

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, stdio_config):
        client = MCPClient(stdio_config)
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_connect_bad_command_raises(self):
        client = MCPClient(
            MCPServerConfig(
                name="broken",
                command="this-binary-does-not-exist-xyz",
                timeout=10,
            )
        )
        with pytest.raises(MCPConnectionError):
            await client.connect()
        assert client.is_connected is False


class TestMCPManager:
    """Test MCP manager functionality."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self, sample_mcp_config):
        """Test MCP manager initialization."""
        manager = MCPManager(sample_mcp_config)

        assert manager.mcp_config == sample_mcp_config
        assert len(manager.clients) == 0
        assert manager.initialized is False

    @pytest.mark.asyncio
    async def test_manager_initialize_servers_disabled(self):
        """Test manager initialization when MCP is disabled."""
        config = MCPConfig(enabled=False, servers={})
        manager = MCPManager(config)

        await manager.initialize_servers()

        assert manager.initialized is True
        assert len(manager.clients) == 0

    @pytest.mark.asyncio
    async def test_manager_initialize_with_no_enabled_servers(self):
        """Test manager initialization when all servers are disabled."""
        # Create config with servers but all disabled
        disabled_server = MCPServerConfig(
            name="disabled_server",
            command="test_command",
            args=["arg1"],
            enabled=False,  # Explicitly disabled
        )
        config = MCPConfig(enabled=True, servers={"disabled_server": disabled_server})
        manager = MCPManager(config)

        # Mock logging to verify disabled servers are logged
        with patch("omnimancer.mcp.manager.logger") as mock_logger:
            await manager.initialize_servers()

            # Should still be initialized successfully
            assert manager.initialized is True
            assert len(manager.clients) == 0

            # Should log debug message about disabled servers
            mock_logger.debug.assert_called_once()
            debug_call_args = mock_logger.debug.call_args[0][0]
            assert "Disabled MCP servers" in debug_call_args
            assert "disabled_server" in debug_call_args

    @pytest.mark.asyncio
    async def test_manager_initialize_servers_success(self, sample_mcp_config):
        """Test successful server initialization."""
        manager = MCPManager(sample_mcp_config)

        # Mock successful client connection
        mock_client = AsyncMock(spec=MCPClient)
        mock_client.connect = AsyncMock()

        with patch("omnimancer.mcp.manager.MCPClient", return_value=mock_client):
            await manager.initialize_servers()

        assert manager.initialized is True
        assert len(manager.clients) == 1
        assert "test_server" in manager.clients
        mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_initialize_servers_partial_failure(self, sample_mcp_config):
        """Test server initialization with partial failures."""
        # Add another server that will fail
        failing_config = MCPServerConfig(
            name="failing_server",
            command="nonexistent_command",
            args=[],
            enabled=True,
        )
        sample_mcp_config.servers["failing_server"] = failing_config

        manager = MCPManager(sample_mcp_config)

        # Mock one successful and one failing client
        def mock_client_factory(config):
            client = AsyncMock(spec=MCPClient)
            if config.name == "test_server":
                client.connect = AsyncMock()
            else:
                client.connect = AsyncMock(side_effect=Exception("Connection failed"))
            return client

        with patch("omnimancer.mcp.manager.MCPClient", side_effect=mock_client_factory):
            await manager.initialize_servers()

        assert manager.initialized is True
        assert len(manager.clients) == 1  # Only successful connection
        assert "test_server" in manager.clients
        assert "failing_server" not in manager.clients

    @pytest.mark.asyncio
    async def test_manager_shutdown(self, sample_mcp_config):
        """Test manager shutdown."""
        manager = MCPManager(sample_mcp_config)

        # Add mock clients
        mock_client1 = AsyncMock(spec=MCPClient)
        mock_client2 = AsyncMock(spec=MCPClient)
        manager.clients = {"server1": mock_client1, "server2": mock_client2}
        manager.initialized = True

        await manager.shutdown()

        assert manager.initialized is False
        assert len(manager.clients) == 0
        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_get_available_tools(
        self, sample_mcp_config, sample_tool_definitions
    ):
        """Test getting available tools from all servers."""
        manager = MCPManager(sample_mcp_config)
        manager.initialized = True

        # Mock client with tools
        mock_client = AsyncMock(spec=MCPClient)
        mock_client.is_connected = True
        mock_client.list_tools = AsyncMock(return_value=sample_tool_definitions)
        manager.clients = {"test_server": mock_client}

        tools = await manager.get_available_tools()

        assert len(tools) == 2
        assert any(tool.name == "calculate" for tool in tools)
        assert any(tool.name == "safe_tool" for tool in tools)
        mock_client.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_get_available_tools_not_initialized(self, sample_mcp_config):
        """Test getting tools when manager is not initialized."""
        manager = MCPManager(sample_mcp_config)
        manager.initialized = False

        tools = await manager.get_available_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_manager_execute_tool_success(self, sample_mcp_config):
        """Test successful tool execution."""
        manager = MCPManager(sample_mcp_config)
        manager.initialized = True

        # Mock client with tool
        mock_client = AsyncMock(spec=MCPClient)
        mock_client.is_connected = True
        mock_client.get_tool_definition.return_value = ToolDefinition(
            name="calculate",
            description="Test tool",
            parameters={},
            server_name="test_server",
        )
        mock_result = ToolResult(content="42", error=None, metadata={})
        mock_client.call_tool_with_retry = AsyncMock(return_value=mock_result)
        manager.clients = {"test_server": mock_client}

        result = await manager.execute_tool("calculate", {"expression": "6 * 7"})

        assert result == mock_result
        mock_client.call_tool_with_retry.assert_called_once_with(
            "calculate", {"expression": "6 * 7"}
        )

    @pytest.mark.asyncio
    async def test_manager_execute_tool_not_found(self, sample_mcp_config):
        """Test executing a tool that doesn't exist."""
        manager = MCPManager(sample_mcp_config)
        manager.initialized = True

        # Mock client without the tool
        mock_client = AsyncMock(spec=MCPClient)
        mock_client.is_connected = True
        mock_client.get_tool_definition.return_value = None
        mock_client.tools = {}  # Add missing tools attribute
        manager.clients = {"test_server": mock_client}

        with pytest.raises(MCPError, match="Tool 'nonexistent' not found"):
            await manager.execute_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_manager_execute_tool_not_initialized(self, sample_mcp_config):
        """Test executing tool when manager is not initialized."""
        manager = MCPManager(sample_mcp_config)
        manager.initialized = False

        with pytest.raises(MCPError, match="MCP manager not initialized"):
            await manager.execute_tool("any_tool", {})

    @pytest.mark.asyncio
    async def test_manager_reload_servers(self, sample_mcp_config):
        """Test server configuration reload."""
        manager = MCPManager(sample_mcp_config)

        # Mock the shutdown and initialize methods
        manager.shutdown = AsyncMock()
        manager.initialize_servers = AsyncMock()

        await manager.reload_servers()

        manager.shutdown.assert_called_once()
        manager.initialize_servers.assert_called_once()

    def test_manager_get_server_status(self, sample_mcp_config):
        """Test getting server status information."""
        manager = MCPManager(sample_mcp_config)

        # Mock connected client
        mock_client = MagicMock(spec=MCPClient)
        mock_client.is_connected = True
        mock_client.tools = {"tool1": MagicMock(), "tool2": MagicMock()}
        manager.clients = {"test_server": mock_client}

        status = manager.get_server_status()

        assert "test_server" in status
        server_status = status["test_server"]
        assert server_status["enabled"] is True
        assert server_status["connected"] is True
        assert server_status["tool_count"] == 2
        assert server_status["command"] == "python3"
        assert server_status["args"] == ["-m", "test_mcp_server"]

    @pytest.mark.asyncio
    async def test_manager_health_check(self, sample_mcp_config):
        """Test health check on all servers."""
        manager = MCPManager(sample_mcp_config)

        # Mock clients with different health statuses
        healthy_client = AsyncMock(spec=MCPClient)
        healthy_client.is_connected = True
        healthy_client.health_check = AsyncMock(return_value=True)

        unhealthy_client = AsyncMock(spec=MCPClient)
        unhealthy_client.is_connected = True
        unhealthy_client.health_check = AsyncMock(return_value=False)

        manager.clients = {
            "healthy_server": healthy_client,
            "unhealthy_server": unhealthy_client,
        }

        # Add a server config for the unhealthy server
        unhealthy_config = MCPServerConfig(
            name="unhealthy_server", command="test", args=[], enabled=True
        )
        manager.mcp_config.servers["unhealthy_server"] = unhealthy_config

        health_status = await manager.health_check()

        assert health_status["healthy_server"] is True
        assert health_status["unhealthy_server"] is False

    def test_manager_utility_methods(self, sample_mcp_config):
        """Test manager utility methods."""
        manager = MCPManager(sample_mcp_config)

        # Test is_enabled property
        assert manager.is_enabled is True

        # Test connected_server_count with no clients
        assert manager.connected_server_count == 0

        # Test total_tool_count with no clients
        assert manager.total_tool_count == 0

        # Add mock clients
        mock_client1 = MagicMock(spec=MCPClient)
        mock_client1.is_connected = True
        mock_client1.tools = {"tool1": MagicMock(), "tool2": MagicMock()}

        mock_client2 = MagicMock(spec=MCPClient)
        mock_client2.is_connected = False
        mock_client2.tools = {"tool3": MagicMock()}

        manager.clients = {"server1": mock_client1, "server2": mock_client2}

        # Test counts with clients
        assert manager.connected_server_count == 1  # Only server1 is connected
        assert manager.total_tool_count == 2  # Only tools from connected servers

    def test_manager_get_tool_by_name(self, sample_mcp_config, sample_tool_definitions):
        """Test getting a tool by name."""
        manager = MCPManager(sample_mcp_config)

        # Mock client with tools
        mock_client = MagicMock(spec=MCPClient)
        mock_client.is_connected = True
        mock_client.get_tool_definition.side_effect = lambda name: (
            sample_tool_definitions[0] if name == "calculate" else None
        )
        manager.clients = {"test_server": mock_client}

        # Test finding existing tool
        tool = manager.get_tool_by_name("calculate")
        assert tool == sample_tool_definitions[0]

        # Test tool not found
        tool = manager.get_tool_by_name("nonexistent")
        assert tool is None

    def test_manager_get_tools_by_server(
        self, sample_mcp_config, sample_tool_definitions
    ):
        """Test getting tools from a specific server."""
        manager = MCPManager(sample_mcp_config)

        # Mock client with tools
        mock_client = MagicMock(spec=MCPClient)
        mock_client.is_connected = True
        mock_client.tools = {tool.name: tool for tool in sample_tool_definitions}
        manager.clients = {"test_server": mock_client}

        # Test getting tools from existing server
        tools = manager.get_tools_by_server("test_server")
        assert len(tools) == 2

        # Test getting tools from non-existent server
        tools = manager.get_tools_by_server("nonexistent")
        assert tools == []


class TestMCPIntegrationScenarios:
    """Test end-to-end MCP integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_mcp_workflow(self, sample_mcp_config):
        """Test complete MCP workflow from initialization to tool execution."""
        manager = MCPManager(sample_mcp_config)

        # Mock successful initialization
        mock_client = AsyncMock(spec=MCPClient)
        mock_client.connect = AsyncMock()
        mock_client.is_connected = True

        # Mock tool discovery
        test_tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            server_name="test_server",
        )
        mock_client.list_tools = AsyncMock(return_value=[test_tool])
        mock_client.get_tool_definition.return_value = test_tool

        # Mock tool execution
        expected_result = ToolResult(
            content="Tool executed successfully",
            error=None,
            metadata={"execution_time": 0.1},
        )
        mock_client.call_tool_with_retry = AsyncMock(return_value=expected_result)

        with patch("omnimancer.mcp.manager.MCPClient", return_value=mock_client):
            # Initialize servers
            await manager.initialize_servers()
            assert manager.initialized is True
            assert len(manager.clients) == 1

            # Get available tools
            tools = await manager.get_available_tools()
            assert len(tools) == 1
            assert tools[0].name == "test_tool"

            # Execute tool
            result = await manager.execute_tool("test_tool", {"input": "test"})
            assert result == expected_result

            # Shutdown
            await manager.shutdown()
            assert manager.initialized is False
            assert len(manager.clients) == 0

    @pytest.mark.asyncio
    async def test_multiple_servers_workflow(self):
        """Test workflow with multiple MCP servers."""
        # Create config with multiple servers
        server1_config = MCPServerConfig(
            name="server1",
            command="python3",
            args=["-m", "server1"],
            enabled=True,
        )
        server2_config = MCPServerConfig(
            name="server2",
            command="python3",
            args=["-m", "server2"],
            enabled=True,
        )

        config = MCPConfig(
            enabled=True,
            servers={"server1": server1_config, "server2": server2_config},
        )

        manager = MCPManager(config)

        # Mock clients for both servers
        def mock_client_factory(server_config):
            client = AsyncMock(spec=MCPClient)
            client.connect = AsyncMock()
            client.is_connected = True

            if server_config.name == "server1":
                tool1 = ToolDefinition(
                    name="tool1",
                    description="Tool from server 1",
                    parameters={},
                    server_name="server1",
                )
                client.list_tools = AsyncMock(return_value=[tool1])
                client.get_tool_definition.side_effect = lambda name: (
                    tool1 if name == "tool1" else None
                )
                client.tools = {"tool1": tool1}
            else:
                tool2 = ToolDefinition(
                    name="tool2",
                    description="Tool from server 2",
                    parameters={},
                    server_name="server2",
                )
                client.list_tools = AsyncMock(return_value=[tool2])
                client.get_tool_definition.side_effect = lambda name: (
                    tool2 if name == "tool2" else None
                )
                client.tools = {"tool2": tool2}

            return client

        with patch("omnimancer.mcp.manager.MCPClient", side_effect=mock_client_factory):
            await manager.initialize_servers()

            # Should have both servers connected
            assert len(manager.clients) == 2
            assert "server1" in manager.clients
            assert "server2" in manager.clients

            # Should have tools from both servers
            tools = await manager.get_available_tools()
            assert len(tools) == 2
            tool_names = {tool.name for tool in tools}
            assert "tool1" in tool_names
            assert "tool2" in tool_names

            # Should be able to find tools by server
            server1_tools = manager.get_tools_by_server("server1")
            assert len(server1_tools) == 1
            assert server1_tools[0].name == "tool1"

            server2_tools = manager.get_tools_by_server("server2")
            assert len(server2_tools) == 1
            assert server2_tools[0].name == "tool2"

    @pytest.mark.asyncio
    async def test_server_failure_and_recovery(self, sample_mcp_config):
        """Test handling of server failures and recovery."""
        manager = MCPManager(sample_mcp_config)

        # Mock client that initially fails, then succeeds
        connection_attempts = 0

        def mock_client_factory(server_config):
            client = AsyncMock(spec=MCPClient)

            async def mock_connect():
                nonlocal connection_attempts
                connection_attempts += 1
                if connection_attempts == 1:
                    raise Exception("Connection failed")
                # Second attempt succeeds
                client.is_connected = True

            client.connect = mock_connect
            client.list_tools = AsyncMock(return_value=[])
            return client

        with patch("omnimancer.mcp.manager.MCPClient", side_effect=mock_client_factory):
            # First initialization should fail
            await manager.initialize_servers()
            assert len(manager.clients) == 0  # No successful connections

            # Reload should succeed
            await manager.reload_servers()
            assert len(manager.clients) == 1  # Successful connection on retry

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, sample_mcp_config):
        """Test concurrent execution of multiple tools."""
        manager = MCPManager(sample_mcp_config)

        # Mock client with multiple tools
        mock_client = AsyncMock(spec=MCPClient)
        mock_client.connect = AsyncMock()
        mock_client.is_connected = True

        tools = [
            ToolDefinition(
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters={},
                server_name="test_server",
            )
            for i in range(3)
        ]

        mock_client.list_tools = AsyncMock(return_value=tools)
        mock_client.get_tool_definition.side_effect = lambda name: next(
            (tool for tool in tools if tool.name == name), None
        )

        # Mock tool execution with delays to test concurrency
        async def mock_call_tool_with_retry(name, args):
            await asyncio.sleep(0.1)  # Simulate processing time
            return ToolResult(
                content=f"Result from {name}",
                error=None,
                metadata={"tool": name},
            )

        mock_client.call_tool_with_retry = mock_call_tool_with_retry

        with patch("omnimancer.mcp.manager.MCPClient", return_value=mock_client):
            await manager.initialize_servers()

            # Execute multiple tools concurrently
            tasks = [
                manager.execute_tool(f"tool_{i}", {"input": f"test_{i}"})
                for i in range(3)
            ]

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time

            # Should complete in ~0.1s (concurrent)
            # rather than 0.3s (sequential)
            assert execution_time < 0.2

            # Verify all results
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.content == f"Result from tool_{i}"
                assert result.metadata["tool"] == f"tool_{i}"


class TestMCPErrorHandling:
    """Test MCP error handling scenarios."""

    @pytest.mark.asyncio
    async def test_server_process_crash_during_operation(
        self, sample_mcp_server_config
    ):
        """Test handling of server process crash during operation."""
        client = MCPClient(sample_mcp_server_config)

        # Mock process that crashes
        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process has crashed (non-zero exit code)
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()

        client.process = mock_process
        client.connected = True
        client.tools = {
            "test_tool": ToolDefinition(
                name="test_tool",
                description="Test",
                parameters={},
                server_name="test_server",
            )
        }

        # Health check should detect the crash
        is_healthy = await client.health_check()
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self):
        """An unknown tool yields an error ToolResult (server-reported)."""
        server = os.path.join(os.path.dirname(__file__), "mcp_test_server.py")
        client = MCPClient(
            MCPServerConfig(name="t", command=sys.executable, args=[server])
        )
        await client.connect()
        try:
            result = await client.call_tool("nonexistent_tool", {})
            assert result.error is not None
        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_connect_timeout(self):
        """A non-MCP process that never completes the handshake times out."""
        client = MCPClient(
            MCPServerConfig(
                name="hang",
                command=sys.executable,
                args=["-c", "import time; time.sleep(30)"],
                timeout=2,
            )
        )
        with pytest.raises((MCPConnectionError, MCPTimeoutError)):
            await client.connect()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_manager_partial_server_failure(self):
        """Test manager behavior when some servers fail."""
        # Create config with multiple servers
        config = MCPConfig(
            enabled=True,
            servers={
                "working_server": MCPServerConfig(
                    name="working_server",
                    command="python3",
                    args=["-m", "working"],
                    enabled=True,
                ),
                "failing_server": MCPServerConfig(
                    name="failing_server",
                    command="nonexistent",
                    args=[],
                    enabled=True,
                ),
            },
        )

        manager = MCPManager(config)

        def mock_client_factory(server_config):
            client = AsyncMock(spec=MCPClient)
            if server_config.name == "working_server":
                working_tool = ToolDefinition(
                    name="working_tool",
                    description="A working tool",
                    parameters={},
                    server_name="working_server",
                )
                client.connect = AsyncMock()
                client.is_connected = True
                client.list_tools = AsyncMock(return_value=[working_tool])
                client.tools = {"working_tool": working_tool}
            else:
                client.connect = AsyncMock(side_effect=Exception("Failed to connect"))
                client.is_connected = False
                client.tools = {}
            return client

        with patch("omnimancer.mcp.manager.MCPClient", side_effect=mock_client_factory):
            await manager.initialize_servers()

            # Should have one working server
            assert len(manager.clients) == 1
            assert "working_server" in manager.clients
            assert "failing_server" not in manager.clients

            # Should still be able to get tools from working server
            tools = await manager.get_available_tools()
            assert len(tools) == 1
            assert tools[0].name == "working_tool"

            # Status should show both servers
            status = manager.get_server_status()
            assert len(status) == 2
            assert status["working_server"]["connected"] is True
            assert status["failing_server"]["connected"] is False


class TestMCPOptionalSDK:
    """The app must import without the optional mcp SDK; connecting errors clearly."""

    def test_require_sdk_raises_clear_error_when_missing(self):
        import sys
        from unittest.mock import patch

        from omnimancer.mcp.client import _require_sdk

        # Simulate the SDK not being installed.
        with patch.dict(sys.modules, {"mcp": None}):
            with pytest.raises(MCPConnectionError, match="pip install mcp"):
                _require_sdk()

    @pytest.mark.asyncio
    async def test_connect_without_sdk_raises_clear_error(self):
        import sys
        from unittest.mock import patch

        client = MCPClient(MCPServerConfig(name="x", command="echo hi", timeout=5))
        with patch.dict(sys.modules, {"mcp": None}):
            with pytest.raises(MCPConnectionError, match="pip install mcp"):
                await client.connect()
