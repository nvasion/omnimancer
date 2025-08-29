"""
MCP (Model Context Protocol) integration for Omnimancer.

This module provides MCP server integration capabilities including
server connection management, tool discovery, and tool execution.
"""

from .client import MCPClient
from .manager import MCPManager

__all__ = ["MCPClient", "MCPManager"]
