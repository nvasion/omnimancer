"""Tests for the native tool call handler."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.tool_handler import (
    AUTO_APPROVED_TOOLS,
    MAX_TOOL_ITERATIONS,
    TOOL_TO_OPERATION,
    ToolHandler,
)
from omnimancer.core.agent.tool_definitions import CODING_AGENT_TOOLS
from omnimancer.core.agent.types import Operation, OperationResult, OperationType
from omnimancer.core.models import ToolCall, ToolResult


@pytest.fixture
def mock_agent_engine():
    engine = MagicMock()
    engine.execute_with_approval = AsyncMock(
        return_value=OperationResult(success=True, data="file contents here")
    )
    return engine


@pytest.fixture
def tool_handler(mock_agent_engine):
    return ToolHandler(mock_agent_engine)


class TestToolHandlerMapping:
    """Test tool call to operation conversion."""

    def test_file_read_mapping(self, tool_handler):
        tc = ToolCall(name="file_read", arguments={"path": "/src/main.py"})
        op = tool_handler._tool_call_to_operation(tc)

        assert op is not None
        assert op.type == OperationType.FILE_READ
        assert op.data["path"] == "/src/main.py"
        assert op.requires_approval is False

    def test_file_write_mapping(self, tool_handler):
        tc = ToolCall(name="file_write", arguments={"path": "/src/new.py", "content": "print('hello')"})
        op = tool_handler._tool_call_to_operation(tc)

        assert op.type == OperationType.FILE_WRITE
        assert op.data["path"] == "/src/new.py"
        assert op.data["content"] == "print('hello')"
        assert op.requires_approval is True

    def test_file_delete_mapping(self, tool_handler):
        tc = ToolCall(name="file_delete", arguments={"path": "/tmp/old.py"})
        op = tool_handler._tool_call_to_operation(tc)

        assert op.type == OperationType.FILE_DELETE
        assert op.requires_approval is True

    def test_command_exec_mapping(self, tool_handler):
        tc = ToolCall(name="command_exec", arguments={"command": "ls -la"})
        op = tool_handler._tool_call_to_operation(tc)

        assert op.type == OperationType.COMMAND_EXECUTE
        assert op.data["command"] == "ls -la"
        assert op.requires_approval is True

    def test_find_files_mapping(self, tool_handler):
        tc = ToolCall(name="find_files", arguments={"pattern": "*.py"})
        op = tool_handler._tool_call_to_operation(tc)

        assert op.type == OperationType.COMMAND_EXECUTE
        assert "*.py" in op.data["command"]
        assert op.requires_approval is False

    def test_search_text_mapping(self, tool_handler):
        tc = ToolCall(name="search_text", arguments={"pattern": "def main", "file_pattern": "*.py"})
        op = tool_handler._tool_call_to_operation(tc)

        assert op.type == OperationType.COMMAND_EXECUTE
        assert "grep" in op.data["command"]
        assert "--include='*.py'" in op.data["command"]
        assert op.requires_approval is False

    def test_web_request_get_mapping(self, tool_handler):
        tc = ToolCall(name="web_request", arguments={"url": "https://example.com"})
        op = tool_handler._tool_call_to_operation(tc)

        assert op.type == OperationType.WEB_REQUEST
        assert op.data["url"] == "https://example.com"
        assert op.requires_approval is False

    def test_web_request_post_mapping(self, tool_handler):
        tc = ToolCall(name="web_request", arguments={"url": "https://api.example.com", "method": "POST"})
        op = tool_handler._tool_call_to_operation(tc)

        assert op.requires_approval is True

    def test_unknown_tool_returns_none(self, tool_handler):
        tc = ToolCall(name="nonexistent_tool", arguments={})
        op = tool_handler._tool_call_to_operation(tc)

        assert op is None


class TestToolHandlerExecution:
    """Test tool call execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_call_success(self, tool_handler, mock_agent_engine):
        tc = ToolCall(name="file_read", arguments={"path": "/src/main.py"})
        result = await tool_handler.execute_tool_call(tc)

        assert result.error is None
        assert result.content == "file contents here"
        mock_agent_engine.execute_with_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_call_failure(self, tool_handler, mock_agent_engine):
        mock_agent_engine.execute_with_approval.return_value = OperationResult(
            success=False, error="Permission denied"
        )

        tc = ToolCall(name="file_write", arguments={"path": "/etc/passwd", "content": "bad"})
        result = await tool_handler.execute_tool_call(tc)

        assert result.error == "Permission denied"

    @pytest.mark.asyncio
    async def test_execute_tool_call_cancelled(self, tool_handler, mock_agent_engine):
        mock_agent_engine.execute_with_approval.return_value = OperationResult(
            success=False, error="User rejected", was_cancelled=True
        )

        tc = ToolCall(name="command_exec", arguments={"command": "rm -rf /"})
        result = await tool_handler.execute_tool_call(tc)

        assert result.error == "Operation cancelled by user"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, tool_handler):
        tc = ToolCall(name="unknown", arguments={})
        result = await tool_handler.execute_tool_call(tc)

        assert result.error == "Unknown tool: unknown"

    @pytest.mark.asyncio
    async def test_execute_multiple_tool_calls(self, tool_handler, mock_agent_engine):
        tool_calls = [
            ToolCall(name="file_read", arguments={"path": "/a.py"}),
            ToolCall(name="file_read", arguments={"path": "/b.py"}),
        ]

        results = await tool_handler.execute_tool_calls(tool_calls)

        assert len(results) == 2
        assert mock_agent_engine.execute_with_approval.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_tool_call_exception(self, tool_handler, mock_agent_engine):
        mock_agent_engine.execute_with_approval.side_effect = RuntimeError("engine crash")

        tc = ToolCall(name="file_read", arguments={"path": "/a.py"})
        result = await tool_handler.execute_tool_call(tc)

        assert result.error == "engine crash"


class TestToolDefinitions:
    """Test the coding agent tool definitions."""

    def test_all_tools_defined(self):
        tool_names = {t.name for t in CODING_AGENT_TOOLS}
        expected = {"file_read", "file_write", "file_delete", "command_exec", "find_files", "search_text", "web_request"}
        assert tool_names == expected

    def test_tool_definitions_have_required_fields(self):
        for tool in CODING_AGENT_TOOLS:
            assert tool.name
            assert tool.description
            assert tool.parameters
            assert "properties" in tool.parameters

    def test_auto_approved_tools_match(self):
        auto_approved_from_defs = {t.name for t in CODING_AGENT_TOOLS if t.auto_approved}
        assert auto_approved_from_defs == AUTO_APPROVED_TOOLS

    def test_get_tool_definitions(self, tool_handler):
        tools = tool_handler.get_tool_definitions()
        assert len(tools) == len(CODING_AGENT_TOOLS)

    def test_max_iterations_constant(self):
        assert MAX_TOOL_ITERATIONS == 25

    def test_all_tools_have_operation_mapping(self):
        for tool in CODING_AGENT_TOOLS:
            assert tool.name in TOOL_TO_OPERATION, f"No operation mapping for tool: {tool.name}"
