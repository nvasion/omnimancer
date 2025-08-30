"""
Enhanced MCP Server Integration Layer for Omnimancer Agent.

This module provides advanced MCP server integration with capability matching,
tool discovery, result processing, and context-aware execution.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.models import ToolDefinition
from ..mcp.manager import MCPManager

logger = logging.getLogger(__name__)


class ToolCapability(Enum):
    """Tool capability categories for matching."""

    FILE_OPERATIONS = "file_operations"
    WEB_REQUESTS = "web_requests"
    DATA_PROCESSING = "data_processing"
    SYSTEM_INFO = "system_info"
    CODE_EXECUTION = "code_execution"
    DATABASE = "database"
    API_INTEGRATION = "api_integration"
    SEARCH = "search"
    COMMUNICATION = "communication"
    MULTIMEDIA = "multimedia"
    UNKNOWN = "unknown"


class ExecutionPriority(Enum):
    """Priority levels for tool execution."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ToolExecutionContext:
    """Context information for tool execution."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    task_context: Optional[str] = None
    execution_priority: ExecutionPriority = ExecutionPriority.NORMAL
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """Enhanced result from tool execution."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    server_name: Optional[str] = None
    tool_name: Optional[str] = None
    attempt_count: int = 1
    context_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMetrics:
    """Metrics for tool performance tracking."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    last_execution_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    error_patterns: Dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 1.0
        return self.successful_executions / self.total_executions

    @property
    def reliability_score(self) -> float:
        """Calculate reliability score (0.0 to 1.0)."""
        base_score = self.success_rate

        # Penalize consecutive failures
        failure_penalty = min(self.consecutive_failures * 0.1, 0.5)

        # Bonus for recent success
        recency_bonus = 0.0
        if self.last_success_time and time.time() - self.last_success_time < 300:
            recency_bonus = 0.1

        return max(0.0, min(1.0, base_score - failure_penalty + recency_bonus))


class EnhancedMCPIntegrator:
    """
    Enhanced MCP integration layer with advanced capabilities including
    tool discovery, capability matching, context-aware execution, and performance monitoring.
    """

    def __init__(self, mcp_manager: Optional[MCPManager] = None):
        """
        Initialize the enhanced MCP integrator.

        Args:
            mcp_manager: Optional MCP manager instance
        """
        self.mcp_manager = mcp_manager

        # Tool discovery and caching
        self.discovered_tools: Dict[str, ToolDefinition] = {}
        self.tools_by_capability: Dict[ToolCapability, List[str]] = defaultdict(list)
        self.tools_by_server: Dict[str, List[str]] = defaultdict(list)
        self.tool_capabilities: Dict[str, Set[ToolCapability]] = {}

        # Performance monitoring
        self.tool_metrics: Dict[str, ToolMetrics] = {}
        self.server_metrics: Dict[str, ToolMetrics] = {}

        # Execution management
        self.execution_queue: Dict[ExecutionPriority, List[Dict[str, Any]]] = (
            defaultdict(list)
        )
        self.concurrent_executions: Dict[str, asyncio.Task] = {}
        self.max_concurrent_executions = 5

        # Caching
        self.result_cache: Dict[str, Tuple[ToolExecutionResult, float]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_enabled = True

        # Last discovery time
        self.last_discovery_time = 0
        self.discovery_interval = 60  # 1 minute

    async def initialize(self) -> bool:
        """
        Initialize the MCP integrator and discover available tools.

        Returns:
            True if initialization was successful
        """
        if not self.mcp_manager or not self.mcp_manager.initialized:
            logger.warning("MCP manager not available or not initialized")
            return False

        try:
            await self.discover_tools()
            logger.info(
                f"MCP integrator initialized with {len(self.discovered_tools)} tools"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MCP integrator: {e}")
            return False

    async def discover_tools(
        self, force_refresh: bool = False
    ) -> Dict[str, ToolDefinition]:
        """
        Discover and categorize available tools from all MCP servers.

        Args:
            force_refresh: Force rediscovery even if recently done

        Returns:
            Dictionary of discovered tools
        """
        current_time = time.time()

        # Check if discovery is needed
        if (
            not force_refresh
            and current_time - self.last_discovery_time < self.discovery_interval
            and self.discovered_tools
        ):
            return self.discovered_tools

        if not self.mcp_manager or not self.mcp_manager.initialized:
            logger.warning("Cannot discover tools: MCP manager not available")
            return {}

        try:
            logger.debug("Discovering MCP tools...")

            # Get all available tools
            available_tools = await self.mcp_manager.get_available_tools()

            # Clear previous discoveries
            self.discovered_tools.clear()
            self.tools_by_capability.clear()
            self.tools_by_server.clear()
            self.tool_capabilities.clear()

            # Process each tool
            for tool in available_tools:
                # Store tool definition
                self.discovered_tools[tool.name] = tool

                # Determine capabilities
                capabilities = self._analyze_tool_capabilities(tool)
                self.tool_capabilities[tool.name] = capabilities

                # Categorize by capability
                for capability in capabilities:
                    self.tools_by_capability[capability].append(tool.name)

                # Find server for this tool
                server_name = self._find_tool_server(tool.name)
                if server_name:
                    self.tools_by_server[server_name].append(tool.name)

                # Initialize metrics if not exists
                if tool.name not in self.tool_metrics:
                    self.tool_metrics[tool.name] = ToolMetrics()

            self.last_discovery_time = current_time

            logger.info(
                f"Discovered {len(self.discovered_tools)} tools across "
                f"{len(self.tools_by_server)} servers"
            )

            # Log capability distribution
            for capability, tools in self.tools_by_capability.items():
                if tools:
                    logger.debug(f"{capability.value}: {len(tools)} tools")

            return self.discovered_tools

        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            return {}

    def _analyze_tool_capabilities(self, tool: ToolDefinition) -> Set[ToolCapability]:
        """
        Analyze a tool's capabilities based on its name, description, and parameters.

        Args:
            tool: Tool definition to analyze

        Returns:
            Set of identified capabilities
        """
        capabilities = set()

        # Analyze based on tool name and description
        text_to_analyze = f"{tool.name} {getattr(tool, 'description', '')}"
        text_lower = text_to_analyze.lower()

        # File operations
        if any(
            keyword in text_lower
            for keyword in [
                "file",
                "read",
                "write",
                "directory",
                "folder",
                "path",
                "save",
                "load",
            ]
        ):
            capabilities.add(ToolCapability.FILE_OPERATIONS)

        # Web requests
        if any(
            keyword in text_lower
            for keyword in [
                "http",
                "web",
                "request",
                "api",
                "url",
                "fetch",
                "download",
                "upload",
            ]
        ):
            capabilities.add(ToolCapability.WEB_REQUESTS)

        # Data processing
        if any(
            keyword in text_lower
            for keyword in [
                "data",
                "process",
                "parse",
                "transform",
                "convert",
                "analyze",
                "filter",
            ]
        ):
            capabilities.add(ToolCapability.DATA_PROCESSING)

        # System info
        if any(
            keyword in text_lower
            for keyword in [
                "system",
                "info",
                "status",
                "monitor",
                "health",
                "performance",
            ]
        ):
            capabilities.add(ToolCapability.SYSTEM_INFO)

        # Code execution
        if any(
            keyword in text_lower
            for keyword in [
                "execute",
                "run",
                "code",
                "script",
                "command",
                "shell",
                "bash",
            ]
        ):
            capabilities.add(ToolCapability.CODE_EXECUTION)

        # Database
        if any(
            keyword in text_lower
            for keyword in [
                "database",
                "db",
                "sql",
                "query",
                "table",
                "record",
            ]
        ):
            capabilities.add(ToolCapability.DATABASE)

        # API integration
        if any(
            keyword in text_lower
            for keyword in [
                "api",
                "integration",
                "service",
                "endpoint",
                "rest",
                "graphql",
            ]
        ):
            capabilities.add(ToolCapability.API_INTEGRATION)

        # Search
        if any(
            keyword in text_lower
            for keyword in ["search", "find", "lookup", "query", "index"]
        ):
            capabilities.add(ToolCapability.SEARCH)

        # Communication
        if any(
            keyword in text_lower
            for keyword in [
                "email",
                "message",
                "notification",
                "send",
                "communicate",
            ]
        ):
            capabilities.add(ToolCapability.COMMUNICATION)

        # Multimedia
        if any(
            keyword in text_lower
            for keyword in [
                "image",
                "video",
                "audio",
                "media",
                "picture",
                "photo",
            ]
        ):
            capabilities.add(ToolCapability.MULTIMEDIA)

        # Default to unknown if no capabilities identified
        if not capabilities:
            capabilities.add(ToolCapability.UNKNOWN)

        return capabilities

    def _find_tool_server(self, tool_name: str) -> Optional[str]:
        """Find which server provides a specific tool."""
        if not self.mcp_manager:
            return None

        return self.mcp_manager._find_tool_server(tool_name)

    async def execute_tool_with_context(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolExecutionResult:
        """
        Execute a tool with context awareness and enhanced error handling.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            context: Execution context

        Returns:
            Enhanced execution result
        """
        context = context or ToolExecutionContext()
        start_time = time.time()

        # Check cache first
        if self.cache_enabled:
            cached_result = self._get_cached_result(tool_name, arguments)
            if cached_result:
                logger.debug(f"Using cached result for tool: {tool_name}")
                return cached_result

        # Ensure tool is discovered
        if tool_name not in self.discovered_tools:
            await self.discover_tools()

        if tool_name not in self.discovered_tools:
            return ToolExecutionResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
                execution_time=time.time() - start_time,
                tool_name=tool_name,
            )

        # Get server name
        server_name = self._find_tool_server(tool_name)

        # Execute with retries
        last_error = None
        attempt_count = 0

        for attempt in range(context.retry_attempts):
            attempt_count += 1

            try:
                # Add context to arguments if applicable
                enhanced_args = self._enhance_arguments_with_context(arguments, context)

                # Execute the tool
                execution_start = time.time()

                # Use timeout
                result = await asyncio.wait_for(
                    self.mcp_manager.execute_tool(tool_name, enhanced_args),
                    timeout=context.timeout_seconds,
                )

                execution_time = time.time() - execution_start

                # Create success result
                success_result = ToolExecutionResult(
                    success=True,
                    data=(result.content if hasattr(result, "content") else result),
                    execution_time=execution_time,
                    server_name=server_name,
                    tool_name=tool_name,
                    attempt_count=attempt_count,
                    context_used=bool(enhanced_args != arguments),
                    metadata={
                        "server_response_time": execution_time,
                        "context_priority": context.execution_priority.name,
                        "retry_attempt": attempt_count,
                    },
                )

                # Record success metrics
                self._record_execution_metrics(
                    tool_name, server_name, True, execution_time
                )

                # Cache result if applicable
                if self.cache_enabled and self._is_cacheable(tool_name, arguments):
                    self._cache_result(tool_name, arguments, success_result)

                return success_result

            except asyncio.TimeoutError:
                last_error = (
                    f"Tool execution timed out after {context.timeout_seconds}s"
                )
                logger.warning(f"Tool {tool_name} timed out on attempt {attempt + 1}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tool {tool_name} failed on attempt {attempt + 1}: {e}")

                # Don't retry for certain errors
                if self._is_non_retryable_error(e):
                    break

                # Wait before retry with exponential backoff
                if attempt < context.retry_attempts - 1:
                    wait_time = min(2**attempt, 10)
                    await asyncio.sleep(wait_time)

        # All attempts failed
        total_execution_time = time.time() - start_time

        error_result = ToolExecutionResult(
            success=False,
            error=last_error,
            execution_time=total_execution_time,
            server_name=server_name,
            tool_name=tool_name,
            attempt_count=attempt_count,
            metadata={
                "max_retries_reached": True,
                "context_priority": context.execution_priority.name,
                "final_error": last_error,
            },
        )

        # Record failure metrics
        self._record_execution_metrics(
            tool_name, server_name, False, total_execution_time, last_error
        )

        return error_result

    def _enhance_arguments_with_context(
        self, arguments: Dict[str, Any], context: ToolExecutionContext
    ) -> Dict[str, Any]:
        """
        Enhance tool arguments with context information if applicable.

        Args:
            arguments: Original arguments
            context: Execution context

        Returns:
            Enhanced arguments
        """
        enhanced_args = arguments.copy()

        # Add context fields that tools might use
        if context.user_id:
            enhanced_args.setdefault("user_id", context.user_id)

        if context.session_id:
            enhanced_args.setdefault("session_id", context.session_id)

        if context.task_context:
            enhanced_args.setdefault("task_context", context.task_context)

        # Add relevant conversation history for context-aware tools
        if context.conversation_history and len(context.conversation_history) > 0:
            # Only add recent history to avoid overwhelming the tool
            recent_history = context.conversation_history[-5:]
            enhanced_args.setdefault("conversation_context", recent_history)

        return enhanced_args

    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Check if an error should not be retried."""
        error_str = str(error).lower()
        non_retryable_patterns = [
            "invalid arguments",
            "permission denied",
            "not found",
            "authentication failed",
            "tool not found",
        ]

        return any(pattern in error_str for pattern in non_retryable_patterns)

    def _is_cacheable(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Check if a tool result should be cached."""
        # Don't cache tools that might have side effects
        side_effect_tools = ["write", "delete", "send", "execute", "run"]
        if any(keyword in tool_name.lower() for keyword in side_effect_tools):
            return False

        # Don't cache if arguments contain dynamic data
        if any(key in arguments for key in ["timestamp", "random", "session_id"]):
            return False

        return True

    def _get_cached_result(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[ToolExecutionResult]:
        """Get cached result if available and not expired."""
        cache_key = self._generate_cache_key(tool_name, arguments)

        if cache_key in self.result_cache:
            result, timestamp = self.result_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                # Remove expired cache entry
                del self.result_cache[cache_key]

        return None

    def _cache_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: ToolExecutionResult,
    ):
        """Cache a tool execution result."""
        cache_key = self._generate_cache_key(tool_name, arguments)
        self.result_cache[cache_key] = (result, time.time())

        # Limit cache size
        if len(self.result_cache) > 100:
            # Remove oldest entries
            sorted_cache = sorted(self.result_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:20]:  # Remove 20 oldest entries
                del self.result_cache[key]

    def _generate_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate a cache key for tool execution."""
        # Sort arguments for consistent key generation
        sorted_args = json.dumps(arguments, sort_keys=True)
        return f"{tool_name}:{hash(sorted_args)}"

    def _record_execution_metrics(
        self,
        tool_name: str,
        server_name: Optional[str],
        success: bool,
        execution_time: float,
        error: Optional[str] = None,
    ):
        """Record execution metrics for monitoring."""
        current_time = time.time()

        # Update tool metrics
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolMetrics()

        tool_metrics = self.tool_metrics[tool_name]
        tool_metrics.total_executions += 1
        tool_metrics.last_execution_time = current_time

        # Update average execution time
        if tool_metrics.average_execution_time == 0:
            tool_metrics.average_execution_time = execution_time
        else:
            tool_metrics.average_execution_time = (
                tool_metrics.average_execution_time * 0.8 + execution_time * 0.2
            )

        if success:
            tool_metrics.successful_executions += 1
            tool_metrics.last_success_time = current_time
            tool_metrics.consecutive_failures = 0
        else:
            tool_metrics.failed_executions += 1
            tool_metrics.last_failure_time = current_time
            tool_metrics.consecutive_failures += 1

            # Track error patterns
            if error:
                error_pattern = error[:50]  # First 50 chars
                tool_metrics.error_patterns[error_pattern] = (
                    tool_metrics.error_patterns.get(error_pattern, 0) + 1
                )

        # Update server metrics
        if server_name:
            if server_name not in self.server_metrics:
                self.server_metrics[server_name] = ToolMetrics()

            server_metrics = self.server_metrics[server_name]
            server_metrics.total_executions += 1
            server_metrics.last_execution_time = current_time

            if success:
                server_metrics.successful_executions += 1
                server_metrics.last_success_time = current_time
                server_metrics.consecutive_failures = 0
            else:
                server_metrics.failed_executions += 1
                server_metrics.last_failure_time = current_time
                server_metrics.consecutive_failures += 1

    def find_tools_by_capability(self, capability: ToolCapability) -> List[str]:
        """Find tools that have a specific capability."""
        return self.tools_by_capability.get(capability, [])

    def find_best_tool_for_task(
        self,
        task_description: str,
        capability: Optional[ToolCapability] = None,
    ) -> Optional[str]:
        """
        Find the best tool for a given task based on description and capability.

        Args:
            task_description: Description of the task
            capability: Optional capability requirement

        Returns:
            Name of the best matching tool or None
        """
        candidates = []

        if capability:
            # Filter by capability first
            candidates = self.find_tools_by_capability(capability)
        else:
            # Use all available tools
            candidates = list(self.discovered_tools.keys())

        if not candidates:
            return None

        # Score tools based on task description
        task_lower = task_description.lower()
        scored_tools = []

        for tool_name in candidates:
            tool = self.discovered_tools[tool_name]
            score = 0

            # Score based on name similarity
            if any(word in tool_name.lower() for word in task_lower.split()):
                score += 2

            # Score based on description similarity
            description = getattr(tool, "description", "")
            if description:
                desc_lower = description.lower()
                common_words = set(task_lower.split()) & set(desc_lower.split())
                score += len(common_words)

            # Boost score based on reliability
            metrics = self.tool_metrics.get(tool_name)
            if metrics:
                score += metrics.reliability_score * 5

                # Penalize if tool has been failing recently
                if metrics.consecutive_failures > 2:
                    score -= metrics.consecutive_failures

            scored_tools.append((tool_name, score))

        # Return the highest scoring tool
        if scored_tools:
            scored_tools.sort(key=lambda x: x[1], reverse=True)
            return scored_tools[0][0]

        return None

    def get_tool_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for tools."""
        if tool_name:
            metrics = self.tool_metrics.get(tool_name)
            if metrics:
                return {
                    "tool_name": tool_name,
                    "total_executions": metrics.total_executions,
                    "success_rate": metrics.success_rate,
                    "reliability_score": metrics.reliability_score,
                    "average_execution_time": metrics.average_execution_time,
                    "consecutive_failures": metrics.consecutive_failures,
                    "error_patterns": dict(metrics.error_patterns),
                }
            return {}
        else:
            # Return summary for all tools
            summary = {}
            for name, metrics in self.tool_metrics.items():
                summary[name] = {
                    "success_rate": metrics.success_rate,
                    "reliability_score": metrics.reliability_score,
                    "total_executions": metrics.total_executions,
                }
            return summary

    def get_server_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for servers."""
        summary = {}
        for server_name, metrics in self.server_metrics.items():
            summary[server_name] = {
                "success_rate": metrics.success_rate,
                "reliability_score": metrics.reliability_score,
                "total_executions": metrics.total_executions,
                "tool_count": len(self.tools_by_server.get(server_name, [])),
            }
        return summary

    def get_capability_summary(self) -> Dict[str, int]:
        """Get summary of tools by capability."""
        return {
            capability.value: len(tools)
            for capability, tools in self.tools_by_capability.items()
            if tools
        }

    def clear_cache(self):
        """Clear the result cache."""
        self.result_cache.clear()
        logger.info("Tool result cache cleared")

    def reset_metrics(self, tool_name: Optional[str] = None):
        """Reset performance metrics."""
        if tool_name:
            if tool_name in self.tool_metrics:
                self.tool_metrics[tool_name] = ToolMetrics()
                logger.info(f"Reset metrics for tool: {tool_name}")
        else:
            self.tool_metrics.clear()
            self.server_metrics.clear()
            logger.info("Reset all tool metrics")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the MCP integration layer."""
        health_status = {
            "integrator_healthy": True,
            "mcp_manager_available": bool(self.mcp_manager),
            "mcp_manager_initialized": bool(
                self.mcp_manager and self.mcp_manager.initialized
            ),
            "discovered_tools_count": len(self.discovered_tools),
            "servers_count": len(self.tools_by_server),
            "cache_size": len(self.result_cache),
            "metrics_tracked_tools": len(self.tool_metrics),
            "last_discovery_time": self.last_discovery_time,
        }

        # Check if discovery is stale
        if time.time() - self.last_discovery_time > self.discovery_interval * 2:
            health_status["discovery_stale"] = True
            health_status["integrator_healthy"] = False

        # Check for tools with high failure rates
        problematic_tools = []
        for tool_name, metrics in self.tool_metrics.items():
            if metrics.reliability_score < 0.5 and metrics.total_executions >= 5:
                problematic_tools.append(tool_name)

        if problematic_tools:
            health_status["problematic_tools"] = problematic_tools
            if (
                len(problematic_tools) > len(self.discovered_tools) * 0.3
            ):  # More than 30% problematic
                health_status["integrator_healthy"] = False

        return health_status
