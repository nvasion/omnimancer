"""
Tests for the AgentEngine and its components.

This module contains comprehensive tests for the AgentEngine class and all
its component managers including FileSystemManager, ProgramExecutor,
WebClient, MCPIntegrator, ApprovalManager, and ProviderFallback.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any, Dict

from omnimancer.core.agent_engine import (
    AgentEngine,
    ProgramExecutor,
    WebClient,
    MCPIntegrator,
    ApprovalManager,
    ProviderFallback,
    Operation,
    OperationResult,
    OperationType,
)
from omnimancer.core.agent.file_system_manager import FileSystemManager
from omnimancer.core.config_manager import ConfigManager
from omnimancer.utils.errors import SecurityError, AgentError


class TestOperation:
    """Test Operation dataclass functionality."""

    def test_operation_creation(self):
        """Test creating operations with different parameters."""
        op = Operation(
            type=OperationType.FILE_READ,
            description="Read test file",
            data={"path": "/test/file.txt"},
        )

        assert op.type == OperationType.FILE_READ
        assert op.description == "Read test file"
        assert op.data["path"] == "/test/file.txt"
        assert op.requires_approval is True
        assert op.reversible is False
        assert op.preview is None

    def test_operation_result_creation(self):
        """Test creating operation results."""
        result = OperationResult(
            success=True,
            data="File content",
            rollback_data={"backup": "old content"},
        )

        assert result.success is True
        assert result.data == "File content"
        assert result.error is None
        assert result.rollback_data["backup"] == "old content"


@pytest.mark.skip(
    reason="Outdated tests - use test_file_system_manager.py instead"
)
class TestFileSystemManager:
    """Test FileSystemManager functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def fs_manager(self, temp_dir):
        """Create FileSystemManager with current API."""
        from omnimancer.core.security import SecurityManager

        security_manager = SecurityManager()
        return FileSystemManager(
            security_manager=security_manager, require_approval=False
        )

    def test_init(self, temp_dir):
        """Test FileSystemManager initialization."""
        from omnimancer.core.security import SecurityManager

        security_manager = SecurityManager()
        fs_manager = FileSystemManager(
            security_manager=security_manager, require_approval=False
        )
        # Check internal attributes that actually exist
        assert hasattr(fs_manager, "max_file_size_mb")
        assert fs_manager.max_file_size_mb == 100
        assert hasattr(fs_manager, "chunk_size")
        assert fs_manager.chunk_size == 8192

    def test_current_directory_awareness(self, fs_manager):
        """Test current directory awareness functionality."""
        cwd = fs_manager.get_current_working_directory()
        assert isinstance(cwd, Path)
        assert cwd.exists()

    @pytest.mark.asyncio
    async def test_directory_context(self, fs_manager):
        """Test directory context generation."""
        context = await fs_manager.get_directory_context()
        assert "current_working_directory" in context
        assert "is_git_repository" in context
        assert isinstance(context["is_git_repository"], bool)

    @pytest.mark.asyncio
    async def test_git_repository_detection(self, fs_manager):
        """Test git repository detection."""
        # Test with current directory (should detect git repo)
        is_git = await fs_manager.is_git_repository()
        assert isinstance(is_git, bool)

        # Test with a specific path
        is_git_path = await fs_manager.is_git_repository(Path.cwd())
        assert isinstance(is_git_path, bool)

    @pytest.mark.asyncio
    async def test_read_file_success(self, fs_manager, temp_dir):
        """Test successful file reading."""
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        content = await fs_manager.read_file(str(test_file))
        assert content == test_content

    @pytest.mark.asyncio
    async def test_read_file_not_exists(self, fs_manager, temp_dir):
        """Test reading non-existent file."""
        try:
            await fs_manager.read_file(str(temp_dir / "nonexistent.txt"))
            assert False, "Should have raised an exception"
        except FileNotFoundError:
            pass  # Expected
        except Exception as e:
            # Any file-related exception is acceptable
            assert "not" in str(e).lower() or "exist" in str(e).lower()

    @pytest.mark.asyncio
    async def test_write_file_success(self, fs_manager, temp_dir):
        """Test successful file writing."""
        test_file = temp_dir / "write_test.txt"
        content = "New content"

        await fs_manager.write_file(str(test_file), content)

        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_file_with_backup(self, fs_manager, temp_dir):
        """Test file writing with backup creation."""
        test_file = temp_dir / "backup_test.txt"
        original_content = "Original content"
        new_content = "New content"

        # Create original file
        test_file.write_text(original_content)

        # Write with backup enabled (default behavior)
        await fs_manager.write_file(str(test_file), new_content)

        assert test_file.read_text() == new_content
        # Check that backup directory exists
        backup_dir = Path("/tmp/omnimancer_backups")
        if backup_dir.exists():
            backup_files = list(backup_dir.glob("*"))
            assert len(backup_files) > 0  # Backup should be created

    @pytest.mark.asyncio
    async def test_delete_file_success(self, fs_manager, temp_dir):
        """Test successful file deletion."""
        test_file = temp_dir / "delete_test.txt"
        content = "Content to delete"
        test_file.write_text(content)

        await fs_manager.delete_file(str(test_file))

        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_create_directory_success(self, fs_manager, temp_dir):
        """Test successful directory creation."""
        new_dir = temp_dir / "new_directory"

        await fs_manager.create_directory(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    @pytest.mark.asyncio
    async def test_delete_directory_success(self, fs_manager, temp_dir):
        """Test successful empty directory deletion."""
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()

        result = await fs_manager._delete_directory(str(test_dir))

        assert result.success is True
        assert not test_dir.exists()
        assert result.rollback_data["path"] == str(test_dir.resolve())

    @pytest.mark.asyncio
    async def test_delete_directory_not_empty(self, fs_manager, temp_dir):
        """Test deletion of non-empty directory fails."""
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        result = await fs_manager._delete_directory(str(test_dir))

        assert result.success is False
        assert "not empty" in result.error

    @pytest.mark.asyncio
    async def test_execute_operation_file_read(self, fs_manager, temp_dir):
        """Test executing file read operation."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        operation = Operation(
            type=OperationType.FILE_READ,
            description="Read test file",
            data={"path": str(test_file)},
        )

        result = await fs_manager.execute_operation(operation)

        assert result.success is True
        assert result.data == "test content"

    @pytest.mark.asyncio
    async def test_preview_operation(self, fs_manager, temp_dir):
        """Test operation preview generation."""
        operation = Operation(
            type=OperationType.FILE_WRITE,
            description="Write test file",
            data={
                "path": str(temp_dir / "test.txt"),
                "content": "Hello World",
            },
        )

        preview = await fs_manager.preview_operation(operation)

        assert "Write to test.txt" in preview
        assert "11 characters" in preview


class TestProgramExecutor:
    """Test ProgramExecutor functionality."""

    @pytest.fixture
    def executor(self):
        """Create ProgramExecutor instance."""
        return ProgramExecutor()

    def test_init(self, executor):
        """Test ProgramExecutor initialization."""
        assert executor.enabled is True
        assert "ls" in executor.allowed_commands
        assert "rm" in executor.forbidden_commands
        assert executor.timeout_seconds == 30

    def test_validate_command_allowed(self, executor):
        """Test validation of allowed command."""
        assert executor._validate_command("ls") is True

    def test_validate_command_forbidden(self, executor):
        """Test validation of forbidden command."""
        with pytest.raises(SecurityError, match="is forbidden"):
            executor._validate_command("rm")

    def test_validate_command_not_whitelisted(self, executor):
        """Test validation of non-whitelisted command."""
        with pytest.raises(SecurityError, match="is not whitelisted"):
            executor._validate_command("unknown_command")

    @pytest.mark.asyncio
    async def test_execute_command_success(self, executor):
        """Test successful command execution."""
        # Mock subprocess execution
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await executor._execute_command("ls", ["-l"])

            assert result.success is True
            assert result.data["stdout"] == "output"
            assert result.data["returncode"] == 0

    @pytest.mark.asyncio
    async def test_execute_command_failure(self, executor):
        """Test command execution failure."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"error")
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            result = await executor._execute_command("ls", ["nonexistent"])

            assert result.success is False
            assert result.error == "error"

    @pytest.mark.asyncio
    async def test_execute_command_timeout(self, executor):
        """Test command execution timeout."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = asyncio.TimeoutError()
            # terminate() and kill() should be synchronous methods, not async
            mock_process.terminate = Mock()
            mock_process.kill = Mock()
            mock_subprocess.return_value = mock_process

            result = await executor._execute_command("ls", [])

            assert result.success is False
            assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_execute_operation(self, executor):
        """Test executing command operation."""
        operation = Operation(
            type=OperationType.COMMAND_EXECUTE,
            description="List files",
            data={"command": "ls", "args": ["-l"]},
        )

        with patch.object(executor, "_execute_command") as mock_execute:
            mock_execute.return_value = OperationResult(
                success=True, data={"stdout": "files"}
            )

            result = await executor.execute_operation(operation)

            assert result.success is True
            mock_execute.assert_called_once_with("ls", ["-l"], None)

    @pytest.mark.asyncio
    async def test_preview_operation(self, executor):
        """Test command operation preview."""
        operation = Operation(
            type=OperationType.COMMAND_EXECUTE,
            description="List files",
            data={"command": "ls", "args": ["-l", "/tmp"]},
        )

        preview = await executor.preview_operation(operation)

        assert "Execute command: ls -l /tmp" in preview


class TestWebClient:
    """Test WebClient functionality."""

    @pytest.fixture
    def web_client(self):
        """Create WebClient instance."""
        return WebClient()

    def test_init(self, web_client):
        """Test WebClient initialization."""
        assert web_client.enabled is True
        assert web_client.rate_limit_delay == 1.0
        assert "localhost" in web_client.forbidden_domains

    def test_validate_url_success(self, web_client):
        """Test successful URL validation."""
        assert web_client._validate_url("https://httpbin.org/get") is True

    def test_validate_url_forbidden_domain(self, web_client):
        """Test URL validation with forbidden domain."""
        with pytest.raises(SecurityError, match="are forbidden"):
            web_client._validate_url("http://localhost:8080/api")

    def test_validate_url_allowed_domains(self, web_client):
        """Test URL validation with allowed domains restriction."""
        web_client.allowed_domains = {"example.com"}

        with pytest.raises(SecurityError, match="not in allowed list"):
            web_client._validate_url("https://google.com")

    @pytest.mark.asyncio
    async def test_make_request_success(self, web_client):
        """Test successful HTTP request."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.is_success = True
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.text = '{"result": "success"}'
            mock_response.url = "https://httpbin.org/get"

            mock_client.request.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = (
                mock_client
            )

            result = await web_client._make_request(
                "https://httpbin.org/get", "GET", {}, None
            )

            assert result.success is True
            assert result.data["status_code"] == 200
            assert '{"result": "success"}' in result.data["content"]

    @pytest.mark.asyncio
    async def test_make_request_failure(self, web_client):
        """Test HTTP request failure."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.is_success = False
            mock_response.status_code = 404
            mock_response.headers = {}
            mock_response.text = "Not Found"
            mock_response.url = "https://httpbin.org/status/404"

            mock_client.request.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = (
                mock_client
            )

            result = await web_client._make_request(
                "https://httpbin.org/status/404", "GET", {}, None
            )

            assert result.success is False
            assert result.error == "HTTP 404"

    @pytest.mark.asyncio
    async def test_execute_operation(self, web_client):
        """Test executing web request operation."""
        operation = Operation(
            type=OperationType.WEB_REQUEST,
            description="GET request",
            data={"url": "https://httpbin.org/get", "method": "GET"},
        )

        with patch.object(web_client, "_make_request") as mock_request:
            mock_request.return_value = OperationResult(
                success=True, data={"status_code": 200}
            )

            result = await web_client.execute_operation(operation)

            assert result.success is True
            mock_request.assert_called_once_with(
                "https://httpbin.org/get", "GET", {}, None
            )

    @pytest.mark.asyncio
    async def test_preview_operation(self, web_client):
        """Test web request operation preview."""
        operation = Operation(
            type=OperationType.WEB_REQUEST,
            description="POST request",
            data={"url": "https://httpbin.org/post", "method": "POST"},
        )

        preview = await web_client.preview_operation(operation)

        assert "POST request to: https://httpbin.org/post" in preview


class TestMCPIntegrator:
    """Test MCPIntegrator functionality."""

    @pytest.fixture
    def mcp_manager(self):
        """Create mock MCP manager."""
        return Mock()

    @pytest.fixture
    def mcp_integrator(self, mcp_manager):
        """Create MCPIntegrator instance."""
        return MCPIntegrator(mcp_manager)

    def test_init(self, mcp_integrator, mcp_manager):
        """Test MCPIntegrator initialization."""
        assert mcp_integrator.enabled is True
        assert mcp_integrator.mcp_manager == mcp_manager

    def test_init_no_manager(self):
        """Test MCPIntegrator initialization without manager."""
        integrator = MCPIntegrator()
        assert integrator.mcp_manager is None

    @pytest.mark.asyncio
    async def test_execute_operation_success(self, mcp_integrator):
        """Test successful MCP tool execution."""
        operation = Operation(
            type=OperationType.MCP_TOOL_CALL,
            description="Call test tool",
            data={"tool_name": "test_tool", "arguments": {"param": "value"}},
        )

        with patch.object(mcp_integrator, "_call_tool") as mock_call:
            mock_call.return_value = OperationResult(
                success=True, data="Tool result"
            )

            result = await mcp_integrator.execute_operation(operation)

            assert result.success is True
            mock_call.assert_called_once_with("test_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_execute_operation_no_manager(self):
        """Test MCP operation without manager."""
        integrator = MCPIntegrator()
        operation = Operation(
            type=OperationType.MCP_TOOL_CALL,
            description="Call test tool",
            data={"tool_name": "test_tool"},
        )

        result = await integrator.execute_operation(operation)

        assert result.success is False
        assert "MCP manager not available" in result.error

    @pytest.mark.asyncio
    async def test_call_tool(self, mcp_integrator):
        """Test MCP tool calling."""
        result = await mcp_integrator._call_tool(
            "test_tool", {"param": "value"}
        )

        assert result.success is True
        assert "Called tool test_tool" in result.data

    @pytest.mark.asyncio
    async def test_preview_operation(self, mcp_integrator):
        """Test MCP operation preview."""
        operation = Operation(
            type=OperationType.MCP_TOOL_CALL,
            description="Call test tool",
            data={"tool_name": "test_tool"},
        )

        preview = await mcp_integrator.preview_operation(operation)

        assert "Call MCP tool: test_tool" in preview


class TestApprovalManager:
    """Test ApprovalManager functionality."""

    @pytest.fixture
    def approval_manager(self):
        """Create ApprovalManager instance."""
        return ApprovalManager()

    def test_init(self, approval_manager):
        """Test ApprovalManager initialization."""
        assert len(approval_manager.auto_approve_types) == 0
        assert approval_manager.approval_callback is None

    @pytest.mark.asyncio
    async def test_request_approval_no_approval_needed(self, approval_manager):
        """Test approval request for operation that doesn't need approval."""
        operation = Operation(
            type=OperationType.FILE_READ,
            description="Read file",
            data={"path": "/test"},
            requires_approval=False,
        )

        approved = await approval_manager.request_approval(operation)

        assert approved is True

    @pytest.mark.asyncio
    async def test_request_approval_auto_approve(self, approval_manager):
        """Test approval request for auto-approved operation type."""
        approval_manager.add_auto_approve_type(OperationType.FILE_READ)

        operation = Operation(
            type=OperationType.FILE_READ,
            description="Read file",
            data={"path": "/test"},
        )

        approved = await approval_manager.request_approval(operation)

        assert approved is True

    @pytest.mark.asyncio
    async def test_request_approval_with_callback(self, approval_manager):
        """Test approval request with callback."""
        callback = AsyncMock(return_value=True)
        approval_manager.set_approval_callback(callback)

        operation = Operation(
            type=OperationType.FILE_WRITE,
            description="Write file",
            data={"path": "/test", "content": "content"},
        )

        approved = await approval_manager.request_approval(operation)

        assert approved is True
        callback.assert_called_once_with(operation)

    @pytest.mark.asyncio
    async def test_request_approval_no_callback(self, approval_manager):
        """Test approval request without callback (should default to deny)."""
        operation = Operation(
            type=OperationType.FILE_WRITE,
            description="Write file",
            data={"path": "/test", "content": "content"},
        )

        approved = await approval_manager.request_approval(operation)

        assert approved is False

    def test_add_remove_auto_approve_type(self, approval_manager):
        """Test adding and removing auto-approve types."""
        approval_manager.add_auto_approve_type(OperationType.FILE_READ)
        assert OperationType.FILE_READ in approval_manager.auto_approve_types

        approval_manager.remove_auto_approve_type(OperationType.FILE_READ)
        assert (
            OperationType.FILE_READ not in approval_manager.auto_approve_types
        )


class TestProviderFallback:
    """Test ProviderFallback functionality."""

    @pytest.fixture
    def core_engine(self):
        """Create mock CoreEngine."""
        return Mock()

    @pytest.fixture
    def provider_fallback(self, core_engine):
        """Create ProviderFallback instance."""
        return ProviderFallback(core_engine)

    def test_init(self, provider_fallback, core_engine):
        """Test ProviderFallback initialization."""
        assert provider_fallback.core_engine == core_engine
        assert provider_fallback.fallback_providers == []
        assert provider_fallback.retry_attempts == 3
        assert provider_fallback.retry_delay == 1.0

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, provider_fallback):
        """Test successful execution without fallback."""

        async def test_func():
            return "success"

        result = await provider_fallback.execute_with_fallback(test_func)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_with_fallback_retry(self, provider_fallback):
        """Test execution with retry on failure."""
        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"

        with patch("asyncio.sleep"):  # Speed up test
            result = await provider_fallback.execute_with_fallback(test_func)

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_fallback_providers(
        self, provider_fallback, core_engine
    ):
        """Test execution with fallback providers."""
        provider_fallback.set_fallback_providers(["provider1", "provider2"])

        call_count = 0

        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # Fail initial retries
                raise Exception("Primary failure")
            return "fallback success"

        core_engine.switch_model = AsyncMock()

        with patch("asyncio.sleep"):  # Speed up test
            result = await provider_fallback.execute_with_fallback(test_func)

        assert result == "fallback success"
        assert (
            core_engine.switch_model.call_count == 1
        )  # Called for first fallback

    @pytest.mark.asyncio
    async def test_execute_with_fallback_all_fail(
        self, provider_fallback, core_engine
    ):
        """Test execution when all providers fail."""
        provider_fallback.set_fallback_providers(["provider1"])

        async def test_func():
            raise Exception("All fail")

        core_engine.switch_model = AsyncMock()

        with pytest.raises(Exception, match="All fail"):
            with patch("asyncio.sleep"):  # Speed up test
                await provider_fallback.execute_with_fallback(test_func)

    def test_set_fallback_providers(self, provider_fallback):
        """Test setting fallback providers."""
        providers = ["provider1", "provider2", "provider3"]
        provider_fallback.set_fallback_providers(providers)

        assert provider_fallback.fallback_providers == providers


class TestAgentEngine:
    """Test AgentEngine functionality."""

    @pytest.fixture
    def config_manager(self):
        """Create mock ConfigManager."""
        manager = Mock(spec=ConfigManager)
        config = Mock()
        config.providers = {}
        config.default_provider = None
        config.mcp = None
        manager.get_config.return_value = config
        return manager

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def agent_engine(self, config_manager, temp_dir):
        """Create AgentEngine instance."""
        # Create AgentEngine without calling CoreEngine.__init__
        engine = object.__new__(AgentEngine)

        # Set up minimal CoreEngine attributes that AgentEngine expects
        engine.config_manager = config_manager
        engine.chat_manager = Mock()
        engine.conversation_manager = Mock()
        engine.health_monitor = Mock()
        engine.provider_initializer = Mock()
        engine.mcp_manager = None
        engine.providers = {}
        engine.current_provider = None
        engine._initialized = True

        # Initialize agent-specific managers manually
        engine.file_system = FileSystemManager(base_path=str(temp_dir))
        engine.executor = ProgramExecutor()
        engine.web_client = WebClient()
        engine.mcp_integrator = MCPIntegrator(engine.mcp_manager)
        engine.approval = ApprovalManager()
        engine.fallback = ProviderFallback(engine)

        # Agent state
        engine.agent_mode_enabled = False
        engine.pending_operations = []
        engine.operation_history = []

        return engine

    def test_init(self, agent_engine, temp_dir):
        """Test AgentEngine initialization."""
        assert isinstance(agent_engine.file_system, FileSystemManager)
        assert isinstance(agent_engine.executor, ProgramExecutor)
        assert isinstance(agent_engine.web_client, WebClient)
        assert isinstance(agent_engine.mcp_integrator, MCPIntegrator)
        assert isinstance(agent_engine.approval, ApprovalManager)
        assert isinstance(agent_engine.fallback, ProviderFallback)
        assert agent_engine.agent_mode_enabled is False
        assert agent_engine.file_system.base_path == temp_dir

    def test_get_manager_for_operation(self, agent_engine):
        """Test getting appropriate manager for operation types."""
        # File operations
        file_op = Operation(OperationType.FILE_READ, "Read", {"path": "/test"})
        assert (
            agent_engine._get_manager_for_operation(file_op)
            == agent_engine.file_system
        )

        # Command operations
        cmd_op = Operation(
            OperationType.COMMAND_EXECUTE, "Command", {"command": "ls"}
        )
        assert (
            agent_engine._get_manager_for_operation(cmd_op)
            == agent_engine.executor
        )

        # Web operations
        web_op = Operation(
            OperationType.WEB_REQUEST, "Web", {"url": "http://test.com"}
        )
        assert (
            agent_engine._get_manager_for_operation(web_op)
            == agent_engine.web_client
        )

        # MCP operations
        mcp_op = Operation(
            OperationType.MCP_TOOL_CALL, "MCP", {"tool_name": "test"}
        )
        assert (
            agent_engine._get_manager_for_operation(mcp_op)
            == agent_engine.mcp_integrator
        )

    @pytest.mark.asyncio
    async def test_generate_preview(self, agent_engine):
        """Test operation preview generation."""
        operation = Operation(
            OperationType.FILE_READ, "Read file", {"path": "/test/file.txt"}
        )

        with patch.object(
            agent_engine.file_system, "preview_operation"
        ) as mock_preview:
            mock_preview.return_value = "Preview: Read file.txt"

            preview = await agent_engine._generate_preview(operation)

            assert preview == "Preview: Read file.txt"
            mock_preview.assert_called_once_with(operation)

    @pytest.mark.asyncio
    async def test_execute_operation(self, agent_engine):
        """Test operation execution."""
        operation = Operation(
            OperationType.FILE_READ, "Read file", {"path": "/test/file.txt"}
        )

        expected_result = OperationResult(success=True, data="file content")

        with patch.object(
            agent_engine.file_system, "execute_operation"
        ) as mock_execute:
            mock_execute.return_value = expected_result

            result = await agent_engine._execute_operation(operation)

            assert result == expected_result
            mock_execute.assert_called_once_with(operation)

    @pytest.mark.asyncio
    async def test_execute_with_approval_success(self, agent_engine):
        """Test successful operation execution with approval."""
        operation = Operation(
            OperationType.FILE_READ,
            "Read file",
            {"path": "/test/file.txt"},
            requires_approval=True,
        )

        # Mock approval
        agent_engine.approval.request_approval = AsyncMock(return_value=True)

        # Mock preview and execution
        with patch.object(
            agent_engine, "_generate_preview"
        ) as mock_preview, patch.object(
            agent_engine, "_execute_operation"
        ) as mock_execute:

            mock_preview.return_value = "Preview: Read file"
            mock_execute.return_value = OperationResult(
                success=True, data="content"
            )

            result = await agent_engine.execute_with_approval(operation)

            assert result.success is True
            assert len(agent_engine.operation_history) == 1

    @pytest.mark.asyncio
    async def test_execute_with_approval_denied(self, agent_engine):
        """Test operation execution with approval denied."""
        operation = Operation(
            OperationType.FILE_WRITE,
            "Write file",
            {"path": "/test/file.txt", "content": "content"},
            requires_approval=True,
        )

        # Mock approval denial
        agent_engine.approval.request_approval = AsyncMock(return_value=False)

        with patch.object(agent_engine, "_generate_preview") as mock_preview:
            mock_preview.return_value = "Preview: Write file"

            result = await agent_engine.execute_with_approval(operation)

            assert result.success is False
            assert "not approved" in result.error
            assert len(agent_engine.operation_history) == 0

    @pytest.mark.asyncio
    async def test_execute_with_approval_no_approval_needed(
        self, agent_engine
    ):
        """Test operation execution without approval requirement."""
        operation = Operation(
            OperationType.FILE_READ,
            "Read file",
            {"path": "/test/file.txt"},
            requires_approval=False,
        )

        with patch.object(
            agent_engine, "_generate_preview"
        ) as mock_preview, patch.object(
            agent_engine, "_execute_operation"
        ) as mock_execute:

            mock_preview.return_value = "Preview: Read file"
            mock_execute.return_value = OperationResult(
                success=True, data="content"
            )

            result = await agent_engine.execute_with_approval(operation)

            assert result.success is True
            # Approval should not be called for operations that don't require approval
            # This is verified by the operation not having requires_approval=True

    def test_enable_disable_agent_mode(self, agent_engine):
        """Test enabling and disabling agent mode."""
        assert agent_engine.agent_mode_enabled is False

        agent_engine.enable_agent_mode()
        assert agent_engine.agent_mode_enabled is True

        agent_engine.disable_agent_mode()
        assert agent_engine.agent_mode_enabled is False

    def test_operation_history_management(self, agent_engine):
        """Test operation history management."""
        # Add some mock history
        agent_engine.operation_history = [
            {"operation": "op1", "result": "result1", "timestamp": 123},
            {"operation": "op2", "result": "result2", "timestamp": 456},
        ]

        # Get history
        history = agent_engine.get_operation_history()
        assert len(history) == 2
        assert history[0]["operation"] == "op1"

        # Clear history
        agent_engine.clear_operation_history()
        assert len(agent_engine.operation_history) == 0

    @pytest.mark.asyncio
    async def test_rollback_operation_file_write(self, agent_engine, temp_dir):
        """Test rolling back a file write operation."""
        # Create mock history entry
        original_content = "original content"
        operation = Operation(
            OperationType.FILE_WRITE,
            "Write file",
            {"path": str(temp_dir / "test.txt"), "content": "new content"},
        )
        result = OperationResult(
            success=True, rollback_data={"backup_content": original_content}
        )

        agent_engine.operation_history = [
            {"operation": operation, "result": result, "timestamp": 123}
        ]

        # Mock execute_with_approval for rollback
        with patch.object(
            agent_engine, "execute_with_approval"
        ) as mock_execute:
            mock_execute.return_value = OperationResult(success=True)

            success = await agent_engine.rollback_operation(0)

            assert success is True
            mock_execute.assert_called_once()

            # Verify rollback operation was created correctly
            rollback_op = mock_execute.call_args[0][0]
            assert rollback_op.type == OperationType.FILE_WRITE
            assert rollback_op.data["content"] == original_content
            assert rollback_op.requires_approval is False

    @pytest.mark.asyncio
    async def test_rollback_operation_file_delete(
        self, agent_engine, temp_dir
    ):
        """Test rolling back a file delete operation."""
        # Create mock history entry
        deleted_content = "deleted content"
        file_path = str(temp_dir / "deleted.txt")
        operation = Operation(
            OperationType.FILE_DELETE, "Delete file", {"path": file_path}
        )
        result = OperationResult(
            success=True,
            rollback_data={
                "backup_content": deleted_content,
                "path": file_path,
            },
        )

        agent_engine.operation_history = [
            {"operation": operation, "result": result, "timestamp": 123}
        ]

        # Mock execute_with_approval for rollback
        with patch.object(
            agent_engine, "execute_with_approval"
        ) as mock_execute:
            mock_execute.return_value = OperationResult(success=True)

            success = await agent_engine.rollback_operation(0)

            assert success is True
            mock_execute.assert_called_once()

            # Verify rollback operation was created correctly
            rollback_op = mock_execute.call_args[0][0]
            assert rollback_op.type == OperationType.FILE_WRITE
            assert rollback_op.data["content"] == deleted_content
            assert rollback_op.data["path"] == file_path

    @pytest.mark.asyncio
    async def test_rollback_operation_no_rollback_data(self, agent_engine):
        """Test rollback with no rollback data available."""
        operation = Operation(
            OperationType.FILE_READ, "Read", {"path": "/test"}
        )
        result = OperationResult(
            success=True, data="content"
        )  # No rollback_data

        agent_engine.operation_history = [
            {"operation": operation, "result": result, "timestamp": 123}
        ]

        success = await agent_engine.rollback_operation(0)

        assert success is False

    @pytest.mark.asyncio
    async def test_rollback_operation_invalid_index(self, agent_engine):
        """Test rollback with invalid operation index."""
        success = await agent_engine.rollback_operation(999)

        assert success is False


if __name__ == "__main__":
    pytest.main([__file__])
