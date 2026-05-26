"""Native tool call handler — bridges provider tool calls to AgentEngine execution."""

import json
import logging
from typing import Any, Dict, List, Optional

from ..core.agent.tool_definitions import CODING_AGENT_TOOLS
from ..core.agent.types import Operation, OperationResult, OperationType
from ..core.models import ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 25

TOOL_TO_OPERATION = {
    "file_read": OperationType.FILE_READ,
    "file_write": OperationType.FILE_WRITE,
    "file_delete": OperationType.FILE_DELETE,
    "command_exec": OperationType.COMMAND_EXECUTE,
    "find_files": OperationType.COMMAND_EXECUTE,
    "search_text": OperationType.COMMAND_EXECUTE,
    "web_request": OperationType.WEB_REQUEST,
}

AUTO_APPROVED_TOOLS = {"file_read", "find_files", "search_text"}


class ToolHandler:
    """Executes tool calls from providers via the AgentEngine."""

    def __init__(self, agent_engine):
        self.agent_engine = agent_engine

    def get_tool_definitions(self) -> List[ToolDefinition]:
        return list(CODING_AGENT_TOOLS)

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        operation = self._tool_call_to_operation(tool_call)
        if operation is None:
            return ToolResult(
                content="",
                error=f"Unknown tool: {tool_call.name}",
            )

        try:
            result = await self.agent_engine.execute_with_approval(operation)
            return self._operation_result_to_tool_result(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_call.name}: {e}")
            return ToolResult(content="", error=str(e))

    async def execute_tool_calls(
        self, tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        results = []
        for tc in tool_calls:
            result = await self.execute_tool_call(tc)
            results.append(result)
        return results

    def _tool_call_to_operation(self, tool_call: ToolCall) -> Optional[Operation]:
        op_type = TOOL_TO_OPERATION.get(tool_call.name)
        if op_type is None:
            return None

        args = tool_call.arguments
        auto_approve = tool_call.name in AUTO_APPROVED_TOOLS

        if tool_call.name == "file_read":
            return Operation(
                type=OperationType.FILE_READ,
                description=f"Read file: {args.get('path', '')}",
                data={"path": args.get("path", "")},
                requires_approval=False,
            )

        if tool_call.name == "file_write":
            return Operation(
                type=OperationType.FILE_WRITE,
                description=f"Write file: {args.get('path', '')}",
                data={
                    "path": args.get("path", ""),
                    "content": args.get("content", ""),
                },
                requires_approval=True,
            )

        if tool_call.name == "file_delete":
            return Operation(
                type=OperationType.FILE_DELETE,
                description=f"Delete file: {args.get('path', '')}",
                data={"path": args.get("path", "")},
                requires_approval=True,
            )

        if tool_call.name == "command_exec":
            return Operation(
                type=OperationType.COMMAND_EXECUTE,
                description=f"Execute: {args.get('command', '')}",
                data={
                    "command": args.get("command", ""),
                    "working_dir": args.get("working_dir"),
                },
                requires_approval=True,
            )

        if tool_call.name == "find_files":
            pattern = args.get("pattern", "*")
            directory = args.get("directory", ".")
            return Operation(
                type=OperationType.COMMAND_EXECUTE,
                description=f"Find files: {pattern}",
                data={"command": f"find {directory} -name '{pattern}' -type f"},
                requires_approval=False,
            )

        if tool_call.name == "search_text":
            pattern = args.get("pattern", "")
            directory = args.get("directory", ".")
            file_pattern = args.get("file_pattern", "")
            cmd = f"grep -rn '{pattern}' {directory}"
            if file_pattern:
                cmd += f" --include='{file_pattern}'"
            return Operation(
                type=OperationType.COMMAND_EXECUTE,
                description=f"Search: {pattern}",
                data={"command": cmd},
                requires_approval=False,
            )

        if tool_call.name == "web_request":
            return Operation(
                type=OperationType.WEB_REQUEST,
                description=f"HTTP {args.get('method', 'GET')} {args.get('url', '')}",
                data={
                    "url": args.get("url", ""),
                    "method": args.get("method", "GET"),
                    "body": args.get("body"),
                },
                requires_approval=args.get("method", "GET") != "GET",
            )

        return None

    def _operation_result_to_tool_result(
        self, result: OperationResult
    ) -> ToolResult:
        if result.success:
            content = result.data if isinstance(result.data, str) else json.dumps(result.data, default=str)
            return ToolResult(content=content or "OK")
        else:
            error_msg = result.error or "Operation failed"
            if result.was_cancelled:
                error_msg = "Operation cancelled by user"
            return ToolResult(content="", error=error_msg)
