"""
MCP Manager for managing multiple MCP server connections.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from ..core.models import (
    MCPConfig,
    MCPError,
    MCPServerConfig,
    ToolDefinition,
    ToolResult,
)
from ..utils.error_handler import handle_mcp_error
from ..utils.errors import (
    MCPConfigurationError,
    MCPConnectionError,
    MCPTimeoutError,
)
from .client import MCPClient

logger = logging.getLogger(__name__)


class MCPManager:
    """
    Manager for multiple MCP server connections.

    This class handles the lifecycle of multiple MCP servers,
    aggregates tools from all servers, and routes tool calls
    to the appropriate server.
    """

    def __init__(self, mcp_config: MCPConfig):
        """
        Initialize MCP manager.

        Args:
            mcp_config: MCP configuration
        """
        self.mcp_config = mcp_config
        self.clients: Dict[str, MCPClient] = {}
        self.initialized = False

    async def initialize_servers(
        self, new_server_config: Optional[Dict[str, MCPServerConfig]] = None
    ) -> None:
        """
        Initialize all configured MCP servers with comprehensive error handling.

        This method attempts to connect to all enabled servers and provides
        graceful degradation when some servers fail to connect.
        """
        if not self.mcp_config.enabled:
            logger.info("MCP integration is disabled")
            self.initialized = True  # Successfully initialized with no servers
            return

        logger.info("Initializing MCP servers...")

        # Log disabled servers for transparency
        disabled_servers = {
            name: config
            for name, config in self.mcp_config.servers.items()
            if not config.enabled
        }
        if disabled_servers:
            logger.debug(f"Disabled MCP servers: {', '.join(disabled_servers.keys())}")

        enabled_servers = self.mcp_config.get_enabled_servers()
        if not enabled_servers:
            logger.info("No enabled MCP servers found - MCP initialization complete")
            self.initialized = True  # Successfully initialized with no servers
            return

        # Track initialization results
        successful_connections = 0
        failed_connections = []

        # Limit concurrent server connections
        max_concurrent = min(
            len(enabled_servers),
            getattr(self.mcp_config, "max_concurrent_servers", 3),
        )

        # Create semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(max_concurrent)

        async def connect_server(
            server_name: str, server_config: MCPServerConfig
        ) -> None:
            nonlocal successful_connections

            async with semaphore:
                try:
                    logger.debug(f"Attempting to connect to MCP server: {server_name}")
                    client = MCPClient(server_config)
                    await client.connect()
                    self.clients[server_name] = client
                    successful_connections += 1
                    logger.info(f"Successfully initialized MCP server: {server_name}")

                except MCPConfigurationError as e:
                    error_msg, suggestions, _ = handle_mcp_error(e, server_name)
                    logger.error(
                        f"Configuration error for MCP server '{server_name}': {error_msg}"
                    )
                    for suggestion in suggestions:
                        logger.info(f"  Suggestion: {suggestion}")
                    failed_connections.append(
                        (server_name, "configuration_error", str(e))
                    )

                except MCPConnectionError as e:
                    error_msg, suggestions, should_retry = handle_mcp_error(
                        e, server_name
                    )
                    logger.warning(
                        f"Connection failed for MCP server '{server_name}': {error_msg}"
                    )

                    # Attempt one retry for connection errors
                    if should_retry:
                        try:
                            logger.info(
                                f"Retrying connection to MCP server: {server_name}"
                            )
                            await asyncio.sleep(2.0)  # Brief delay before retry
                            client = MCPClient(server_config)
                            await client.connect()
                            self.clients[server_name] = client
                            successful_connections += 1
                            logger.info(
                                f"Successfully connected to MCP server on retry: {server_name}"
                            )
                        except Exception as retry_error:
                            logger.error(
                                f"Retry failed for MCP server '{server_name}': {retry_error}"
                            )
                            failed_connections.append(
                                (server_name, "connection_failed", str(e))
                            )
                    else:
                        failed_connections.append(
                            (server_name, "connection_failed", str(e))
                        )

                except MCPTimeoutError as e:
                    error_msg, suggestions, _ = handle_mcp_error(e, server_name)
                    logger.warning(
                        f"Timeout connecting to MCP server '{server_name}': {error_msg}"
                    )
                    failed_connections.append((server_name, "timeout", str(e)))

                except Exception as e:
                    error_msg, suggestions, _ = handle_mcp_error(e, server_name)
                    logger.error(
                        f"Unexpected error initializing MCP server '{server_name}': {error_msg}"
                    )
                    failed_connections.append((server_name, "unknown_error", str(e)))

        # Connect to all servers concurrently
        tasks = [
            connect_server(name, config) for name, config in enabled_servers.items()
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Only mark as initialized if at least one server connected successfully
        self.initialized = successful_connections > 0

        # Log initialization summary
        total_servers = len(enabled_servers)
        logger.info(
            f"MCP initialization complete. Connected to {successful_connections}/{total_servers} servers."
        )

        if failed_connections:
            logger.warning(
                f"Failed to connect to {len(failed_connections)} MCP servers:"
            )
            for server_name, error_type, error_msg in failed_connections:
                logger.warning(f"  - {server_name}: {error_type}")

        # Provide graceful degradation information
        if successful_connections == 0 and total_servers > 0:
            logger.warning(
                "No MCP servers connected. Tool functionality will be unavailable."
            )
        elif successful_connections < total_servers:
            logger.info(
                f"Partial MCP connectivity: {successful_connections} servers available for tool execution."
            )

    async def shutdown(self) -> None:
        """Shutdown all MCP server connections."""
        logger.info("Shutting down MCP servers...")

        # Disconnect all clients
        disconnect_tasks = [client.disconnect() for client in self.clients.values()]

        await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        self.clients.clear()
        self.initialized = False
        logger.info("MCP shutdown complete")

    async def get_available_tools(self) -> List[ToolDefinition]:
        """
        Get all available tools from connected MCP servers.

        Returns:
            List of available tool definitions
        """
        if not self.initialized:
            return []

        all_tools = []

        for client in self.clients.values():
            if client.is_connected:
                try:
                    tools = await client.list_tools()
                    all_tools.extend(tools)
                except Exception as e:
                    logger.warning(
                        f"Failed to get tools from server '{client.server_name}': {e}"
                    )

        return all_tools

    async def execute_tool(self, name: str, arguments: Dict) -> ToolResult:
        """
        Execute a tool by name with given arguments, with comprehensive error handling.

        Args:
            name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Result of the tool execution

        Raises:
            MCPError: If tool execution fails
        """
        if not self.initialized:
            raise MCPError("MCP manager not initialized")

        # Find the server that has this tool
        target_client = None
        for client in self.clients.values():
            if client.is_connected and client.get_tool_definition(name):
                target_client = client
                break

        if not target_client:
            # Provide helpful error message with available tools
            available_tools = []
            for client in self.clients.values():
                if client.is_connected:
                    available_tools.extend(
                        [tool.name for tool in client.tools.values()]
                    )

            if available_tools:
                raise MCPError(
                    f"Tool '{name}' not found. Available tools: {', '.join(available_tools[:5])}"
                )
            else:
                raise MCPError(
                    f"Tool '{name}' not found. No MCP servers are connected."
                )

        try:
            # Use the enhanced call_tool_with_retry method
            return await target_client.call_tool_with_retry(name, arguments)

        except Exception as e:
            error_msg, suggestions, should_retry = handle_mcp_error(
                e, target_client.server_name
            )

            # Log the error with suggestions
            logger.error(f"Tool execution failed: {error_msg}")
            for suggestion in suggestions:
                logger.info(f"  Suggestion: {suggestion}")

            # If the client is no longer connected, try to find an alternative
            if not target_client.is_connected:
                logger.info(
                    f"Server {target_client.server_name} disconnected, looking for alternatives..."
                )

                # Try to find another server with the same tool
                alternative_client = None
                for client in self.clients.values():
                    if (
                        client != target_client
                        and client.is_connected
                        and client.get_tool_definition(name)
                    ):
                        alternative_client = client
                        break

                if alternative_client:
                    logger.info(
                        f"Found alternative server: {alternative_client.server_name}"
                    )
                    try:
                        return await alternative_client.call_tool_with_retry(
                            name, arguments
                        )
                    except Exception as alt_error:
                        logger.error(f"Alternative server also failed: {alt_error}")

            raise MCPError(f"Failed to execute tool '{name}': {e}")

    async def reload_servers(self, server_name: Optional[str] = None) -> bool:
        """
        Reload server configurations and reconnect.

        This method shuts down existing connections and reinitializes
        servers based on the current configuration.
        """
        logger.info("Reloading MCP servers...")

        # Shutdown existing connections
        await self.shutdown()

        # Reinitialize with current configuration
        await self.initialize_servers()

        logger.info("MCP server reload complete")

    def get_server_status(
        self, server_name: Optional[str] = None
    ) -> Dict[str, Dict[str, any]]:
        """
        Get status information for all configured servers.

        Returns:
            Dictionary with server status information
        """
        status = {}

        for server_name, server_config in self.mcp_config.servers.items():
            client = self.clients.get(server_name)

            status[server_name] = {
                "enabled": server_config.enabled,
                "connected": client.is_connected if client else False,
                "tool_count": len(client.tools) if client else 0,
                "command": server_config.command,
                "args": server_config.args,
            }

        return status

    async def health_check(self) -> Dict[str, any]:
        """
        Perform health checks on all connected servers.

        Returns:
            Dictionary with server health status and overall health
        """
        health_status = {}

        health_tasks = []
        server_names = []

        for server_name, client in self.clients.items():
            if client.is_connected:
                health_tasks.append(client.health_check())
                server_names.append(server_name)

        if health_tasks:
            results = await asyncio.gather(*health_tasks, return_exceptions=True)

            for server_name, result in zip(server_names, results):
                if isinstance(result, Exception):
                    health_status[server_name] = False
                else:
                    health_status[server_name] = result

        # Add disconnected servers as unhealthy
        for server_name in self.mcp_config.servers:
            if server_name not in health_status:
                health_status[server_name] = False

        # Calculate overall health
        all_healthy = all(health_status.values()) if health_status else False
        health_status["overall_healthy"] = all_healthy

        return health_status

    def get_tool_by_name(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get a tool definition by name.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolDefinition or None if not found
        """
        for client in self.clients.values():
            if client.is_connected:
                tool_def = client.get_tool_definition(tool_name)
                if tool_def:
                    return tool_def

        return None

    def get_tools_by_server(self, server_name: str) -> List[ToolDefinition]:
        """
        Get all tools from a specific server.

        Args:
            server_name: Name of the server

        Returns:
            List of tool definitions from the server
        """
        client = self.clients.get(server_name)
        if client and client.is_connected:
            return list(client.tools.values())

        return []

    @property
    def is_enabled(self) -> bool:
        """Check if MCP integration is enabled."""
        return self.mcp_config.enabled

    @property
    def connected_server_count(self) -> int:
        """Get the number of connected servers."""
        return sum(1 for client in self.clients.values() if client.is_connected)

    @property
    def total_tool_count(self) -> int:
        """Get the total number of available tools across all servers."""
        return sum(
            len(client.tools) for client in self.clients.values() if client.is_connected
        )

    async def recover_failed_servers(self) -> Dict[str, bool]:
        """
        Attempt to recover failed or disconnected servers.

        Returns:
            Dictionary mapping server names to recovery success status
        """
        recovery_results = {}

        for server_name, server_config in self.mcp_config.servers.items():
            if not server_config.enabled:
                continue

            client = self.clients.get(server_name)

            # Check if server needs recovery
            needs_recovery = (
                client is None
                or not client.is_connected
                or not await client.health_check()
            )

            if needs_recovery:
                logger.info(f"Attempting to recover MCP server: {server_name}")

                try:
                    # Remove old client if exists
                    if client:
                        await client.disconnect()

                    # Create new client and connect
                    new_client = MCPClient(server_config)
                    await new_client.connect()
                    self.clients[server_name] = new_client

                    recovery_results[server_name] = True
                    logger.info(f"Successfully recovered MCP server: {server_name}")

                except Exception as e:
                    recovery_results[server_name] = False
                    logger.error(f"Failed to recover MCP server '{server_name}': {e}")
            else:
                recovery_results[server_name] = True  # Already healthy

        return recovery_results

    async def monitor_and_recover(self, check_interval: float = 30.0) -> None:
        """
        Continuously monitor server health and attempt recovery.

        Args:
            check_interval: Seconds between health checks
        """
        logger.info(
            f"Starting MCP server monitoring (check interval: {check_interval}s)"
        )

        while self.initialized:
            try:
                # Perform health checks
                health_status = await self.health_check()

                # Identify unhealthy servers
                unhealthy_servers = [
                    name for name, healthy in health_status.items() if not healthy
                ]

                if unhealthy_servers:
                    logger.warning(
                        f"Unhealthy MCP servers detected: {unhealthy_servers}"
                    )

                    # Attempt recovery
                    recovery_results = await self.recover_failed_servers()

                    # Log recovery results
                    for server_name, recovered in recovery_results.items():
                        if server_name in unhealthy_servers:
                            if recovered:
                                logger.info(
                                    f"Successfully recovered server: {server_name}"
                                )
                            else:
                                logger.error(f"Failed to recover server: {server_name}")

                # Wait before next check
                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error during MCP monitoring: {e}")
                await asyncio.sleep(check_interval)

    def get_degradation_status(self) -> Dict[str, any]:
        """
        Get information about current degradation status.

        Returns:
            Dictionary with degradation information
        """
        total_configured = len(self.mcp_config.servers)
        enabled_servers = len(self.mcp_config.get_enabled_servers())
        connected_servers = self.connected_server_count

        degradation_level = "none"
        if connected_servers == 0 and enabled_servers > 0:
            degradation_level = "complete"
        elif connected_servers < enabled_servers:
            degradation_level = "partial"

        return {
            "degradation_level": degradation_level,
            "total_configured": total_configured,
            "enabled_servers": enabled_servers,
            "connected_servers": connected_servers,
            "available_tools": self.total_tool_count,
            "functionality_impact": self._get_functionality_impact(degradation_level),
        }

    def _get_functionality_impact(self, degradation_level: str) -> List[str]:
        """
        Get list of functionality impacts based on degradation level.

        Args:
            degradation_level: Level of degradation

        Returns:
            List of impact descriptions
        """
        if degradation_level == "complete":
            return [
                "All MCP tool functionality unavailable",
                "AI conversations limited to basic text generation",
                "No external tool integration available",
            ]
        elif degradation_level == "partial":
            return [
                "Some MCP tools may be unavailable",
                "Reduced tool functionality",
                "Some integrations may not work",
            ]
        else:
            return ["All MCP functionality available"]

    async def execute_tool_with_fallback(
        self, name: str, arguments: Dict, fallback_tools: List[str] = None
    ) -> ToolResult:
        """
        Execute a tool with fallback to alternative tools if the primary fails.

        Args:
            name: Primary tool name to execute
            arguments: Arguments to pass to the tool
            fallback_tools: List of alternative tool names to try

        Returns:
            Result of the tool execution

        Raises:
            MCPError: If all tools fail
        """
        fallback_tools = fallback_tools or []

        # Try primary tool first
        try:
            return await self.execute_tool(name, arguments)
        except MCPError as primary_error:
            logger.warning(f"Primary tool '{name}' failed: {primary_error}")

            # Try fallback tools
            for fallback_tool in fallback_tools:
                try:
                    logger.info(f"Trying fallback tool: {fallback_tool}")
                    return await self.execute_tool(fallback_tool, arguments)
                except MCPError as fallback_error:
                    logger.warning(
                        f"Fallback tool '{fallback_tool}' failed: {fallback_error}"
                    )
                    continue

            # If all tools failed, raise the original error
            raise primary_error

    def get_available_tools_by_server(self, server_name: str) -> List[ToolDefinition]:
        """Get tools available from a specific server."""
        return self.get_tools_by_server(server_name)

    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        """Find which server has a specific tool."""
        for server_name, client in self.clients.items():
            if client.is_connected and client.get_tool_definition(tool_name):
                return server_name
        return None

    async def shutdown_servers(self, server_name: str) -> bool:
        """Shutdown a specific server."""
        if server_name in self.clients:
            client = self.clients[server_name]
            await client.disconnect()
            del self.clients[server_name]
            return True
        return False
