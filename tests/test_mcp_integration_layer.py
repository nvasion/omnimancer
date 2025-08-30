"""
Comprehensive tests for the enhanced MCP integration layer.
"""

import asyncio
import time
from typing import Any, Dict, List

import pytest

from omnimancer.core.mcp_integration_layer import (
    EnhancedMCPIntegrator,
    ExecutionPriority,
    ToolCapability,
    ToolExecutionContext,
    ToolMetrics,
)
from omnimancer.core.models import ToolDefinition
from omnimancer.utils.errors import MCPError


class MockToolDefinition:
    """Mock tool definition for testing."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class MockMCPManager:
    """Mock MCP manager for testing."""

    def __init__(self):
        self.initialized = True
        self.tools = {}
        self.servers = {}
        self.execution_results = {}
        self.execution_delays = {}
        self.execution_errors = {}

    async def get_available_tools(self) -> List[ToolDefinition]:
        """Return mock available tools."""
        return [
            MockToolDefinition("file_read", "Read a file from the filesystem"),
            MockToolDefinition("web_fetch", "Fetch content from a web URL"),
            MockToolDefinition("data_parse", "Parse and analyze data"),
            MockToolDefinition("system_info", "Get system information"),
            MockToolDefinition("code_execute", "Execute code in a sandbox"),
            MockToolDefinition("db_query", "Query a database"),
            MockToolDefinition("api_call", "Make an API call"),
            MockToolDefinition("search_web", "Search the web for information"),
            MockToolDefinition("send_email", "Send an email message"),
            MockToolDefinition("image_process", "Process an image file"),
        ]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Mock tool execution."""
        # Simulate execution delay if configured
        if tool_name in self.execution_delays:
            await asyncio.sleep(self.execution_delays[tool_name])

        # Simulate error if configured
        if tool_name in self.execution_errors:
            raise MCPError(self.execution_errors[tool_name])

        # Return mock result
        if tool_name in self.execution_results:
            return self.execution_results[tool_name]

        # Default mock result
        return {
            "content": f"Mock result from {tool_name}",
            "arguments_received": arguments,
        }

    def _find_tool_server(self, tool_name: str) -> str:
        """Mock server finding."""
        return self.servers.get(tool_name, "mock-server")

    def set_tool_result(self, tool_name: str, result: Any):
        """Set mock result for a tool."""
        self.execution_results[tool_name] = result

    def set_tool_delay(self, tool_name: str, delay: float):
        """Set execution delay for a tool."""
        self.execution_delays[tool_name] = delay

    def set_tool_error(self, tool_name: str, error: str):
        """Set error for a tool."""
        self.execution_errors[tool_name] = error


@pytest.fixture
def mock_mcp_manager():
    """Create mock MCP manager."""
    return MockMCPManager()


@pytest.fixture
def mcp_integrator(mock_mcp_manager):
    """Create MCP integrator with mock manager."""
    integrator = EnhancedMCPIntegrator(mock_mcp_manager)
    return integrator


class TestToolCapabilityAnalysis:
    """Test tool capability analysis."""

    def test_file_operations_capability(self, mcp_integrator):
        """Test file operations capability detection."""
        tool = MockToolDefinition("file_read", "Read a file from disk")
        capabilities = mcp_integrator._analyze_tool_capabilities(tool)

        assert ToolCapability.FILE_OPERATIONS in capabilities

    def test_web_request_capability(self, mcp_integrator):
        """Test web request capability detection."""
        tool = MockToolDefinition("web_fetch", "Fetch content from a URL")
        capabilities = mcp_integrator._analyze_tool_capabilities(tool)

        assert ToolCapability.WEB_REQUESTS in capabilities

    def test_data_processing_capability(self, mcp_integrator):
        """Test data processing capability detection."""
        tool = MockToolDefinition("data_parse", "Parse and analyze JSON data")
        capabilities = mcp_integrator._analyze_tool_capabilities(tool)

        assert ToolCapability.DATA_PROCESSING in capabilities

    def test_system_info_capability(self, mcp_integrator):
        """Test system info capability detection."""
        tool = MockToolDefinition("system_status", "Get system health status")
        capabilities = mcp_integrator._analyze_tool_capabilities(tool)

        assert ToolCapability.SYSTEM_INFO in capabilities

    def test_code_execution_capability(self, mcp_integrator):
        """Test code execution capability detection."""
        tool = MockToolDefinition("code_run", "Execute Python code")
        capabilities = mcp_integrator._analyze_tool_capabilities(tool)

        assert ToolCapability.CODE_EXECUTION in capabilities

    def test_multiple_capabilities(self, mcp_integrator):
        """Test tool with multiple capabilities."""
        tool = MockToolDefinition("file_api", "Read file and send data via API")
        capabilities = mcp_integrator._analyze_tool_capabilities(tool)

        assert ToolCapability.FILE_OPERATIONS in capabilities
        assert ToolCapability.API_INTEGRATION in capabilities

    def test_unknown_capability(self, mcp_integrator):
        """Test tool with unknown capability."""
        tool = MockToolDefinition("mystery_tool", "Does something mysterious")
        capabilities = mcp_integrator._analyze_tool_capabilities(tool)

        assert ToolCapability.UNKNOWN in capabilities


class TestToolDiscovery:
    """Test tool discovery functionality."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, mcp_integrator, mock_mcp_manager):
        """Test successful initialization."""
        result = await mcp_integrator.initialize()

        assert result is True
        assert len(mcp_integrator.discovered_tools) > 0
        assert len(mcp_integrator.tools_by_capability) > 0

    @pytest.mark.asyncio
    async def test_initialize_no_manager(self):
        """Test initialization without MCP manager."""
        integrator = EnhancedMCPIntegrator(None)
        result = await integrator.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_discover_tools(self, mcp_integrator):
        """Test tool discovery."""
        await mcp_integrator.initialize()
        tools = await mcp_integrator.discover_tools()

        assert len(tools) == 10  # Based on mock tools
        assert "file_read" in tools
        assert "web_fetch" in tools

        # Check categorization
        file_tools = mcp_integrator.find_tools_by_capability(
            ToolCapability.FILE_OPERATIONS
        )
        assert "file_read" in file_tools

        web_tools = mcp_integrator.find_tools_by_capability(ToolCapability.WEB_REQUESTS)
        assert "web_fetch" in web_tools

    @pytest.mark.asyncio
    async def test_discover_tools_caching(self, mcp_integrator):
        """Test tool discovery caching."""
        await mcp_integrator.initialize()

        # First discovery
        time.time()
        tools1 = await mcp_integrator.discover_tools()
        first_discovery_time = mcp_integrator.last_discovery_time

        # Second discovery (should use cache)
        tools2 = await mcp_integrator.discover_tools()
        second_discovery_time = mcp_integrator.last_discovery_time

        assert tools1 == tools2
        assert first_discovery_time == second_discovery_time

        # Force refresh
        tools3 = await mcp_integrator.discover_tools(force_refresh=True)
        third_discovery_time = mcp_integrator.last_discovery_time

        assert tools3 == tools1
        assert third_discovery_time > second_discovery_time


class TestToolExecution:
    """Test tool execution functionality."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mcp_integrator, mock_mcp_manager):
        """Test successful tool execution."""
        await mcp_integrator.initialize()

        # Set up mock result
        mock_result = {"data": "test result"}
        mock_mcp_manager.set_tool_result("file_read", mock_result)

        context = ToolExecutionContext(task_context="Test task")
        result = await mcp_integrator.execute_tool_with_context(
            "file_read", {"path": "/test/file.txt"}, context
        )

        assert result.success is True
        assert result.data == mock_result
        assert result.tool_name == "file_read"
        assert result.attempt_count == 1

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mcp_integrator):
        """Test execution of non-existent tool."""
        await mcp_integrator.initialize()

        result = await mcp_integrator.execute_tool_with_context("nonexistent_tool", {})

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_with_retries(self, mcp_integrator, mock_mcp_manager):
        """Test tool execution with retries."""
        await mcp_integrator.initialize()

        # Set up tool to fail first two attempts
        call_count = 0

        async def failing_execute(tool_name, arguments):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise MCPError("Temporary failure")
            return {"data": "success after retries"}

        mock_mcp_manager.execute_tool = failing_execute

        context = ToolExecutionContext(retry_attempts=3)
        result = await mcp_integrator.execute_tool_with_context(
            "file_read", {}, context
        )

        assert result.success is True
        assert result.attempt_count == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self, mcp_integrator, mock_mcp_manager):
        """Test tool execution timeout."""
        await mcp_integrator.initialize()

        # Set up tool with long delay
        mock_mcp_manager.set_tool_delay("file_read", 2.0)

        context = ToolExecutionContext(timeout_seconds=0.5)
        result = await mcp_integrator.execute_tool_with_context(
            "file_read", {}, context
        )

        assert result.success is False
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_with_context_enhancement(
        self, mcp_integrator, mock_mcp_manager
    ):
        """Test tool execution with context enhancement."""
        await mcp_integrator.initialize()

        # Track arguments passed to tool
        captured_args = {}

        async def capture_execute(tool_name, arguments):
            captured_args.update(arguments)
            return {"data": "success"}

        mock_mcp_manager.execute_tool = capture_execute

        context = ToolExecutionContext(
            user_id="test_user",
            session_id="test_session",
            task_context="Test task",
            conversation_history=[{"role": "user", "content": "Hello"}],
        )

        result = await mcp_integrator.execute_tool_with_context(
            "file_read", {"path": "/test"}, context
        )

        assert result.success is True
        assert result.context_used is True
        assert captured_args["user_id"] == "test_user"
        assert captured_args["session_id"] == "test_session"
        assert captured_args["task_context"] == "Test task"
        assert "conversation_context" in captured_args


class TestToolMetrics:
    """Test tool metrics functionality."""

    @pytest.mark.asyncio
    async def test_metrics_recording_success(self, mcp_integrator, mock_mcp_manager):
        """Test metrics recording for successful executions."""
        await mcp_integrator.initialize()

        # Execute tool successfully
        result = await mcp_integrator.execute_tool_with_context("file_read", {})

        assert result.success is True

        # Check metrics
        metrics = mcp_integrator.tool_metrics["file_read"]
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.success_rate == 1.0
        assert metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_metrics_recording_failure(self, mcp_integrator, mock_mcp_manager):
        """Test metrics recording for failed executions."""
        await mcp_integrator.initialize()

        # Set up tool to fail
        mock_mcp_manager.set_tool_error("file_read", "Test error")

        result = await mcp_integrator.execute_tool_with_context("file_read", {})

        assert result.success is False

        # Check metrics
        metrics = mcp_integrator.tool_metrics["file_read"]
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 1
        assert metrics.success_rate == 0.0
        assert metrics.consecutive_failures == 1
        assert len(metrics.error_patterns) == 1

    @pytest.mark.asyncio
    async def test_metrics_reliability_score(self, mcp_integrator, mock_mcp_manager):
        """Test reliability score calculation with basic executions."""
        await mcp_integrator.initialize()

        # Disable caching for this test
        mcp_integrator.cache_enabled = False

        # Execute tool successfully first
        result1 = await mcp_integrator.execute_tool_with_context(
            "file_read", {"path": "/unique/path1"}
        )
        assert result1.success is True

        # Set up for failure and execute with different arguments to avoid any issues
        mock_mcp_manager.set_tool_error("file_read", "Test error")
        result2 = await mcp_integrator.execute_tool_with_context(
            "file_read", {"path": "/unique/path2"}
        )
        assert result2.success is False

        # Check that metrics are being tracked
        metrics = mcp_integrator.tool_metrics["file_read"]
        assert metrics.total_executions >= 2
        assert 0.0 <= metrics.reliability_score <= 1.0

    def test_tool_metrics_properties(self):
        """Test ToolMetrics properties."""
        metrics = ToolMetrics()

        # Initial state
        assert metrics.success_rate == 1.0
        assert metrics.reliability_score >= 0.0

        # With data
        metrics.total_executions = 10
        metrics.successful_executions = 8
        metrics.failed_executions = 2
        metrics.consecutive_failures = 1

        assert metrics.success_rate == 0.8
        assert metrics.reliability_score <= 0.8  # Should be penalized for failure


class TestToolCaching:
    """Test result caching functionality."""

    @pytest.mark.asyncio
    async def test_result_caching(self, mcp_integrator, mock_mcp_manager):
        """Test result caching for appropriate tools."""
        await mcp_integrator.initialize()

        # Set up cacheable result
        mock_result = {"data": "cached result"}
        mock_mcp_manager.set_tool_result("file_read", mock_result)

        # First execution
        result1 = await mcp_integrator.execute_tool_with_context(
            "file_read", {"path": "/test"}
        )

        # Modify mock to return different result
        mock_mcp_manager.set_tool_result("file_read", {"data": "new result"})

        # Second execution should use cache
        result2 = await mcp_integrator.execute_tool_with_context(
            "file_read", {"path": "/test"}
        )

        assert result1.success is True
        assert result2.success is True
        assert result1.data == result2.data == mock_result

    @pytest.mark.asyncio
    async def test_cache_expiry(self, mcp_integrator, mock_mcp_manager):
        """Test cache expiry."""
        await mcp_integrator.initialize()
        mcp_integrator.cache_ttl = 0.1  # Very short TTL for testing

        # First execution
        result1 = await mcp_integrator.execute_tool_with_context(
            "file_read", {"path": "/test"}
        )

        # Wait for cache to expire
        await asyncio.sleep(0.2)

        # Change mock result
        mock_mcp_manager.set_tool_result(
            "file_read", {"data": "new result after expiry"}
        )

        # Second execution should not use cache
        result2 = await mcp_integrator.execute_tool_with_context(
            "file_read", {"path": "/test"}
        )

        assert result1.data != result2.data

    def test_cache_key_generation(self, mcp_integrator):
        """Test cache key generation."""
        key1 = mcp_integrator._generate_cache_key("tool1", {"a": 1, "b": 2})
        key2 = mcp_integrator._generate_cache_key(
            "tool1", {"b": 2, "a": 1}
        )  # Different order
        key3 = mcp_integrator._generate_cache_key(
            "tool1", {"a": 1, "b": 3}
        )  # Different value
        key4 = mcp_integrator._generate_cache_key(
            "tool2", {"a": 1, "b": 2}
        )  # Different tool

        assert key1 == key2  # Order shouldn't matter
        assert key1 != key3  # Different values should produce different keys
        assert key1 != key4  # Different tools should produce different keys

    def test_cacheable_detection(self, mcp_integrator):
        """Test detection of cacheable operations."""
        # Read operations should be cacheable
        assert mcp_integrator._is_cacheable("file_read", {"path": "/test"})

        # Write operations should not be cacheable
        assert not mcp_integrator._is_cacheable(
            "file_write", {"path": "/test", "content": "data"}
        )

        # Operations with dynamic data should not be cacheable
        assert not mcp_integrator._is_cacheable(
            "file_read", {"path": "/test", "timestamp": time.time()}
        )


class TestToolSelection:
    """Test intelligent tool selection."""

    @pytest.mark.asyncio
    async def test_find_best_tool_for_task(self, mcp_integrator):
        """Test finding the best tool for a task."""
        await mcp_integrator.initialize()

        # Test file-related task
        best_tool = mcp_integrator.find_best_tool_for_task("read a configuration file")
        assert best_tool == "file_read"

        # Test web-related task
        best_tool = mcp_integrator.find_best_tool_for_task("fetch data from a website")
        assert best_tool == "web_fetch"

        # Test search-related task
        best_tool = mcp_integrator.find_best_tool_for_task(
            "search for information online"
        )
        assert best_tool == "search_web"

    def test_find_tools_by_capability(self, mcp_integrator):
        """Test finding tools by capability."""
        # Mock some capability mappings
        mcp_integrator.tools_by_capability[ToolCapability.FILE_OPERATIONS] = [
            "file_read",
            "file_write",
        ]
        mcp_integrator.tools_by_capability[ToolCapability.WEB_REQUESTS] = [
            "web_fetch",
            "api_call",
        ]

        file_tools = mcp_integrator.find_tools_by_capability(
            ToolCapability.FILE_OPERATIONS
        )
        assert "file_read" in file_tools
        assert "file_write" in file_tools

        web_tools = mcp_integrator.find_tools_by_capability(ToolCapability.WEB_REQUESTS)
        assert "web_fetch" in web_tools
        assert "api_call" in web_tools

        # Non-existent capability
        empty_tools = mcp_integrator.find_tools_by_capability(ToolCapability.MULTIMEDIA)
        assert len(empty_tools) == 0


class TestHealthChecking:
    """Test health checking functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mcp_integrator, mock_mcp_manager):
        """Test health check when everything is healthy."""
        await mcp_integrator.initialize()

        health = await mcp_integrator.health_check()

        assert health["integrator_healthy"] is True
        assert health["mcp_manager_available"] is True
        assert health["mcp_manager_initialized"] is True
        assert health["discovered_tools_count"] > 0

    @pytest.mark.asyncio
    async def test_health_check_stale_discovery(self, mcp_integrator, mock_mcp_manager):
        """Test health check with stale discovery."""
        await mcp_integrator.initialize()

        # Make discovery appear stale
        mcp_integrator.last_discovery_time = time.time() - 200  # Old timestamp
        mcp_integrator.discovery_interval = 60

        health = await mcp_integrator.health_check()

        assert health["discovery_stale"] is True
        assert health["integrator_healthy"] is False

    @pytest.mark.asyncio
    async def test_health_check_problematic_tools(
        self, mcp_integrator, mock_mcp_manager
    ):
        """Test health check with problematic tools."""
        await mcp_integrator.initialize()

        # Create metrics for problematic tools
        for tool_name in ["tool1", "tool2", "tool3"]:
            metrics = ToolMetrics()
            metrics.total_executions = 10
            metrics.successful_executions = 2  # Low success rate
            metrics.failed_executions = 8
            metrics.consecutive_failures = 5
            mcp_integrator.tool_metrics[tool_name] = metrics

        # Add some good tools
        mcp_integrator.discovered_tools = {f"tool{i}": None for i in range(1, 6)}

        health = await mcp_integrator.health_check()

        assert "problematic_tools" in health
        assert len(health["problematic_tools"]) == 3
        assert health["integrator_healthy"] is False  # More than 30% problematic


class TestExecutionContext:
    """Test execution context functionality."""

    def test_execution_context_creation(self):
        """Test creation of execution context."""
        context = ToolExecutionContext(
            user_id="test_user",
            session_id="test_session",
            task_context="Test task",
            execution_priority=ExecutionPriority.HIGH,
            timeout_seconds=60.0,
            retry_attempts=5,
        )

        assert context.user_id == "test_user"
        assert context.session_id == "test_session"
        assert context.task_context == "Test task"
        assert context.execution_priority == ExecutionPriority.HIGH
        assert context.timeout_seconds == 60.0
        assert context.retry_attempts == 5

    def test_execution_context_defaults(self):
        """Test default values in execution context."""
        context = ToolExecutionContext()

        assert context.user_id is None
        assert context.session_id is None
        assert context.execution_priority == ExecutionPriority.NORMAL
        assert context.timeout_seconds == 30.0
        assert context.retry_attempts == 3
        assert isinstance(context.metadata, dict)


class TestIntegrationScenarios:
    """Test integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, mcp_integrator, mock_mcp_manager):
        """Test complete workflow from initialization to execution."""
        # Initialize
        init_success = await mcp_integrator.initialize()
        assert init_success is True

        # Discover tools
        tools = await mcp_integrator.discover_tools()
        assert len(tools) > 0

        # Find tool by capability
        file_tools = mcp_integrator.find_tools_by_capability(
            ToolCapability.FILE_OPERATIONS
        )
        assert len(file_tools) > 0

        # Find best tool for task
        best_tool = mcp_integrator.find_best_tool_for_task("read a file")
        assert best_tool is not None

        # Execute tool
        result = await mcp_integrator.execute_tool_with_context(
            best_tool, {"path": "/test/file.txt"}
        )
        assert result.success is True

        # Check metrics
        metrics = mcp_integrator.get_tool_metrics(best_tool)
        assert metrics["total_executions"] == 1
        assert metrics["success_rate"] == 1.0

        # Health check
        health = await mcp_integrator.health_check()
        assert health["integrator_healthy"] is True

    @pytest.mark.asyncio
    async def test_fallback_execution(self, mcp_integrator, mock_mcp_manager):
        """Test execution with fallback when primary tool fails."""
        await mcp_integrator.initialize()

        # Set up primary tool to fail
        mock_mcp_manager.set_tool_error("file_read", "Primary tool failed")

        # Execute - should handle the failure gracefully
        result = await mcp_integrator.execute_tool_with_context("file_read", {})

        assert result.success is False
        assert "Primary tool failed" in result.error

        # Metrics should reflect the failure
        metrics = mcp_integrator.tool_metrics["file_read"]
        assert metrics.failed_executions == 1
        assert metrics.consecutive_failures == 1
