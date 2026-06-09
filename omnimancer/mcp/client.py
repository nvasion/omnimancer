"""MCP client built on the official ``mcp`` SDK.

Supports all three client transports (stdio, SSE, streamable HTTP) selected from
:class:`~omnimancer.core.models.MCPServerConfig`, and the full protocol surface
this project uses: tools, resources, and prompts.

Lifecycle note: the SDK's transports and ``ClientSession`` are async context
managers whose anyio cancel scopes must be entered and exited *in the same task*.
The manager, however, connects servers concurrently and shuts them down later
(different tasks). To stay correct, each client runs its session inside a single
dedicated task and all session I/O is funnelled to that task through a command
queue — so connect, every request, and disconnect all happen in the owning task.
"""

import asyncio
import logging
import shlex
from contextlib import AsyncExitStack
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..core.models import MCPServerConfig, ToolCall, ToolDefinition, ToolResult
from ..utils.errors import MCPConnectionError, MCPTimeoutError

logger = logging.getLogger(__name__)

# The official ``mcp`` SDK is imported lazily (inside the methods that need it)
# so that the rest of Omnimancer still imports and runs even when the optional
# SDK is not installed — only actually connecting to an MCP server needs it.


def _require_sdk() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the SDK
        raise MCPConnectionError(
            "The 'mcp' package is required for MCP servers. "
            "Install it with: pip install mcp"
        ) from e


class MCPClient:
    """Client for a single MCP server (any transport)."""

    def __init__(self, server_config: MCPServerConfig):
        self.server_config = server_config
        self.connected = False
        self.tools: Dict[str, ToolDefinition] = {}
        self.capabilities: Optional[Any] = None

        self._session: Optional[Any] = None
        self._task: Optional[asyncio.Task] = None
        self._stop: Optional[asyncio.Event] = None
        self._ready: Optional[asyncio.Future] = None
        self._cmd_queue: Optional[asyncio.Queue] = None

    # ----------------------------------------------------------- lifecycle

    @property
    def server_name(self) -> str:
        return self.server_config.name

    @property
    def is_connected(self) -> bool:
        return self.connected and self._task is not None and not self._task.done()

    def _build_transport(self, http_client: Any = None) -> Any:
        """Return the SDK transport async-context-manager for this config.

        ``http_client`` (an httpx.AsyncClient carrying auth headers) is used for
        the streamable-HTTP transport; it is created and lifetime-managed by the
        caller (see ``_run_session``).
        """
        from mcp.client.sse import sse_client
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.streamable_http import streamable_http_client

        cfg = self.server_config
        transport = cfg.transport or "stdio"
        if transport == "stdio":
            parts = shlex.split(cfg.command)
            if not parts:
                raise MCPConnectionError(f"Empty command for MCP server {cfg.name}")
            command, args = parts[0], parts[1:] + list(cfg.args)
            params = StdioServerParameters(
                command=command,
                args=args,
                env={**cfg.env} if cfg.env else None,
            )
            return stdio_client(params)
        if transport in ("sse", "http"):
            # The config validator guarantees url is set for remote transports.
            assert cfg.url is not None
            if transport == "sse":
                return sse_client(cfg.url, headers=cfg.headers or None)
            return streamable_http_client(cfg.url, http_client=http_client)
        raise MCPConnectionError(f"Unknown transport '{transport}' for {cfg.name}")

    async def connect(self, retry_count: int = 0, max_retries: int = 3) -> None:
        """Open the session (in a dedicated task) and discover tools."""
        if self.is_connected:
            return
        _require_sdk()
        loop = asyncio.get_event_loop()
        self._stop = asyncio.Event()
        self._ready = loop.create_future()
        self._cmd_queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run_session())
        try:
            await asyncio.wait_for(self._ready, timeout=self.server_config.timeout)
        except asyncio.TimeoutError:
            await self.disconnect()
            raise MCPTimeoutError(
                f"Timed out connecting to MCP server '{self.server_name}'"
            )
        except Exception as e:
            await self.disconnect()
            if retry_count < max_retries and self._is_retryable_error(e):
                await asyncio.sleep(0.1 * (2**retry_count))
                return await self.connect(retry_count + 1, max_retries)
            raise MCPConnectionError(
                f"Failed to connect to MCP server '{self.server_name}': {e}"
            )

    async def _run_session(self) -> None:
        """Own the session lifetime + serve queued commands in one task."""
        assert self._ready is not None and self._stop is not None
        from mcp import ClientSession

        try:
            async with AsyncExitStack() as stack:
                cfg = self.server_config
                http_client = None
                if (cfg.transport or "stdio") == "http" and cfg.headers:
                    import httpx

                    http_client = await stack.enter_async_context(
                        httpx.AsyncClient(headers=cfg.headers)
                    )
                streams = await stack.enter_async_context(
                    self._build_transport(http_client=http_client)
                )
                # streamable HTTP yields a 3-tuple (read, write, get_session_id).
                read, write = streams[0], streams[1]
                session = await stack.enter_async_context(ClientSession(read, write))
                init_result = await session.initialize()
                self.capabilities = getattr(init_result, "capabilities", None)
                self._session = session
                await self._discover_tools()
                self.connected = True
                if not self._ready.done():
                    self._ready.set_result(True)
                await self._serve_commands()
        except Exception as e:
            logger.warning("MCP session '%s' ended: %s", self.server_name, e)
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(e)
        finally:
            self.connected = False
            self._session = None

    async def _serve_commands(self) -> None:
        """Process queued session calls until disconnect is requested."""
        assert self._cmd_queue is not None and self._stop is not None
        while not self._stop.is_set():
            getter = asyncio.ensure_future(self._cmd_queue.get())
            stopper = asyncio.ensure_future(self._stop.wait())
            done, _ = await asyncio.wait(
                {getter, stopper}, return_when=asyncio.FIRST_COMPLETED
            )
            if getter in done:
                fn, fut = getter.result()
                if not fut.cancelled():
                    try:
                        fut.set_result(await fn(self._session))
                    except Exception as e:  # surface to the caller's future
                        fut.set_exception(e)
            else:
                getter.cancel()
            if stopper in done:
                break
            stopper.cancel()

    async def _call(self, fn: Callable[[Any], Awaitable[Any]]) -> Any:
        """Run ``fn(session)`` inside the owning task and return its result."""
        if not self.is_connected or self._cmd_queue is None:
            raise MCPConnectionError(
                f"MCP server '{self.server_name}' is not connected"
            )
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._cmd_queue.put((fn, fut))
        return await fut

    async def disconnect(self) -> None:
        """Signal the session task to close and wait for clean teardown."""
        if self._stop is not None:
            self._stop.set()
        task = self._task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
            except Exception:
                pass
        self._task = None
        self.connected = False

    async def reconnect(self) -> bool:
        await self.disconnect()
        try:
            await self.connect()
            return True
        except Exception as e:
            logger.warning("Reconnect to '%s' failed: %s", self.server_name, e)
            return False

    # --------------------------------------------------------------- tools

    async def _discover_tools(self) -> None:
        result = await self._session.list_tools()  # type: ignore[union-attr]
        self.tools = {
            tool.name: self._to_tool_definition(tool) for tool in result.tools
        }

    def _to_tool_definition(self, tool: Any) -> ToolDefinition:
        return ToolDefinition(
            name=tool.name,
            description=tool.description or "",
            parameters=getattr(tool, "inputSchema", None) or {},
            server_name=self.server_name,
            auto_approved=tool.name in self.server_config.auto_approve,
        )

    async def list_tools(self) -> List[ToolDefinition]:
        result = await self._call(lambda s: s.list_tools())
        self.tools = {t.name: self._to_tool_definition(t) for t in result.tools}
        return list(self.tools.values())

    def get_available_tools(self) -> List[ToolDefinition]:
        return list(self.tools.values())

    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        return self.tools.get(tool_name)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        result = await self._call(lambda s: s.call_tool(name, arguments))
        return self._to_tool_result(result)

    async def call_tool_with_retry(
        self,
        name: str,
        arguments: Dict[str, Any],
        max_retries: int = 2,
    ) -> ToolResult:
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return await self.call_tool(name, arguments)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
        return ToolResult(content="", error=f"Tool call failed: {last_error}")

    @staticmethod
    def _to_tool_result(result: Any) -> ToolResult:
        text_parts = []
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if text is not None:
                text_parts.append(text)
        content = "\n".join(text_parts)
        is_error = bool(getattr(result, "isError", False))
        return ToolResult(
            content=content,
            error=content if is_error else None,
            metadata={"structured": getattr(result, "structuredContent", None)},
        )

    def _validate_tool_call(self, tool_call: ToolCall) -> None:
        if tool_call.name not in self.tools:
            from ..core.models import MCPToolError

            raise MCPToolError(f"Unknown tool '{tool_call.name}'")

    # ----------------------------------------------------------- resources

    async def list_resources(self) -> List[Dict[str, Any]]:
        if not self._supports("resources"):
            return []
        result = await self._call(lambda s: s.list_resources())
        return [
            {
                "uri": str(r.uri),
                "name": r.name,
                "description": r.description,
                "mimeType": r.mimeType,
                "server": self.server_name,
            }
            for r in result.resources
        ]

    async def read_resource(self, uri: str) -> str:
        result = await self._call(lambda s: s.read_resource(uri))
        parts = []
        for item in getattr(result, "contents", None) or []:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
        return "\n".join(parts)

    # ------------------------------------------------------------- prompts

    async def list_prompts(self) -> List[Dict[str, Any]]:
        if not self._supports("prompts"):
            return []
        result = await self._call(lambda s: s.list_prompts())
        return [
            {
                "name": p.name,
                "description": p.description,
                "arguments": [
                    {
                        "name": a.name,
                        "description": a.description,
                        "required": a.required,
                    }
                    for a in (p.arguments or [])
                ],
                "server": self.server_name,
            }
            for p in result.prompts
        ]

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        result = await self._call(lambda s: s.get_prompt(name, arguments or {}))
        parts = []
        for msg in getattr(result, "messages", None) or []:
            text = getattr(getattr(msg, "content", None), "text", None)
            if text is not None:
                parts.append(text)
        return "\n".join(parts)

    # ------------------------------------------------------------- health

    async def health_check(self) -> bool:
        if not self.is_connected:
            return False
        try:
            await self._call(lambda s: s.send_ping())
            return True
        except Exception:
            self.connected = False
            return False

    def get_server_info(self) -> Dict[str, Any]:
        return {
            "name": self.server_name,
            "transport": self.server_config.transport,
            "connected": self.is_connected,
            "tool_count": len(self.tools),
        }

    # ------------------------------------------------------------ helpers

    def _supports(self, capability: str) -> bool:
        """Best-effort check of negotiated server capabilities."""
        caps = self.capabilities
        if caps is None:
            return True  # unknown — attempt and let the call fail gracefully
        return getattr(caps, capability, None) is not None

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        message = str(error).lower()
        retryable = ("timeout", "connection", "refused", "reset", "broken pipe")
        return any(marker in message for marker in retryable)
