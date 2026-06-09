"""A minimal FastMCP server used by MCP client integration tests.

Run over stdio (the default) so tests can connect to it as a real subprocess.
Exposes one tool, one resource, and one prompt to exercise the full surface.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("omnimancer-test-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.resource("test://greeting")
def greeting() -> str:
    """A static greeting resource."""
    return "hello from resource"


@mcp.prompt()
def greet(name: str) -> str:
    """A simple greeting prompt."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run()
