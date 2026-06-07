"""Native tool call handler — bridges provider tool calls to AgentEngine execution."""

import json
import logging
from typing import Any, List, Optional

from ..core.agent.tool_definitions import CODING_AGENT_TOOLS
from ..core.agent.types import Operation, OperationResult, OperationType
from ..core.models import ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 25

# Cap the size of any single tool result fed back into the model context.
# Unbounded results (e.g. `find`/`grep` walking .venv, or reading a huge file)
# would otherwise balloon the conversation until it exceeds the model's context
# window. ~16k chars is roughly 4k tokens — generous for code, bounded for runaways.
MAX_TOOL_RESULT_CHARS = 16000


def _truncate_tool_output(content: str) -> str:
    """Truncate oversized tool output, keeping head and tail with a marker."""
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content
    head = MAX_TOOL_RESULT_CHARS * 3 // 4
    tail = MAX_TOOL_RESULT_CHARS - head
    omitted = len(content) - MAX_TOOL_RESULT_CHARS
    return (
        content[:head] + f"\n\n[... {omitted} characters truncated "
        "(narrow your search or read a specific file/range) ...]\n\n" + content[-tail:]
    )


# Tool names follow Claude Code conventions (Read/Write/Edit/Bash/Glob/Grep/
# WebFetch). The previous snake_case names are kept as aliases for compatibility.
TOOL_TO_OPERATION = {
    # Claude Code names
    "Read": OperationType.FILE_READ,
    "Write": OperationType.FILE_WRITE,
    "Edit": OperationType.FILE_WRITE,  # implemented as read + replace + write
    "Bash": OperationType.COMMAND_EXECUTE,
    "Glob": OperationType.COMMAND_EXECUTE,
    "Grep": OperationType.COMMAND_EXECUTE,
    "WebFetch": OperationType.WEB_REQUEST,
    # Legacy aliases
    "file_read": OperationType.FILE_READ,
    "file_write": OperationType.FILE_WRITE,
    "file_edit": OperationType.FILE_WRITE,
    "file_delete": OperationType.FILE_DELETE,
    "command_exec": OperationType.COMMAND_EXECUTE,
    "find_files": OperationType.COMMAND_EXECUTE,
    "search_text": OperationType.COMMAND_EXECUTE,
    "web_request": OperationType.WEB_REQUEST,
}

# Tool-name groups (Claude Code name + legacy alias) for dispatch.
_READ_TOOLS = {"Read", "file_read"}
_WRITE_TOOLS = {"Write", "file_write"}
_EDIT_TOOLS = {"Edit", "file_edit"}
_DELETE_TOOLS = {"file_delete"}
_BASH_TOOLS = {"Bash", "command_exec"}
_GLOB_TOOLS = {"Glob", "find_files"}
_GREP_TOOLS = {"Grep", "search_text"}
_WEB_TOOLS = {"WebFetch", "web_request"}

AUTO_APPROVED_TOOLS = {
    "Read",
    "Glob",
    "Grep",
    "WebFetch",
    "file_read",
    "find_files",
    "search_text",
    "web_request",
}

# Heavy directories the agent should not waste its context budget walking.
_EXCLUDED_DIRS = (".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache")
# Suffix for `find`: prune the excluded directories.
_FIND_PRUNE = "".join(f" -not -path '*/{d}/*'" for d in _EXCLUDED_DIRS)
# Suffix for `grep -r`: skip the excluded directories.
_GREP_EXCLUDES = "".join(f" --exclude-dir={d}" for d in _EXCLUDED_DIRS)


class ToolHandler:
    """Executes tool calls from providers via the AgentEngine."""

    def __init__(self, agent_engine: Any) -> None:
        self.agent_engine = agent_engine

    def get_tool_definitions(self) -> List[ToolDefinition]:
        return list(CODING_AGENT_TOOLS)

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        try:
            # Read and Edit need post-processing / multi-step logic.
            if tool_call.name in _EDIT_TOOLS:
                return await self._execute_edit(tool_call)
            if tool_call.name in _READ_TOOLS:
                return await self._execute_read(tool_call)

            operation = self._tool_call_to_operation(tool_call)
            if operation is None:
                return ToolResult(
                    content="",
                    error=f"Unknown tool: {tool_call.name}",
                )

            result = await self.agent_engine.execute_with_approval(operation)
            return self._operation_result_to_tool_result(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_call.name}: {e}")
            return ToolResult(content="", error=str(e))

    @staticmethod
    def _path_arg(args: dict) -> str:
        """Accept Claude Code's ``file_path``/``path`` or legacy ``path``."""
        return args.get("file_path") or args.get("path") or ""

    async def _execute_read(self, tool_call: ToolCall) -> ToolResult:
        """Read a file, honoring optional offset/limit line slicing."""
        args = tool_call.arguments
        path = self._path_arg(args)
        operation = Operation(
            type=OperationType.FILE_READ,
            description=f"Read file: {path}",
            data={"path": path},
            requires_approval=False,
        )
        result = await self.agent_engine.execute_with_approval(operation)
        if not result.success:
            return self._operation_result_to_tool_result(result)

        content = (
            result.data
            if isinstance(result.data, str)
            else json.dumps(result.data, default=str)
        )
        offset = args.get("offset")
        limit = args.get("limit")
        if offset or limit:
            lines = content.splitlines()
            start = max((int(offset) if offset else 1) - 1, 0)
            end = start + int(limit) if limit else len(lines)
            content = "\n".join(lines[start:end])
        return ToolResult(content=_truncate_tool_output(content or "OK"))

    async def _execute_edit(self, tool_call: ToolCall) -> ToolResult:
        """Exact string replacement: read the file, replace, write it back."""
        args = tool_call.arguments
        path = self._path_arg(args)
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        replace_all = bool(args.get("replace_all", False))

        if old_string == new_string:
            return ToolResult(
                content="", error="old_string and new_string are identical"
            )

        read_op = Operation(
            type=OperationType.FILE_READ,
            description=f"Read file: {path}",
            data={"path": path},
            requires_approval=False,
        )
        read_result = await self.agent_engine.execute_with_approval(read_op)
        if not read_result.success:
            return ToolResult(
                content="",
                error=read_result.error or f"Could not read {path} for editing",
            )

        content = (
            read_result.data
            if isinstance(read_result.data, str)
            else str(read_result.data)
        )
        occurrences = content.count(old_string)
        if occurrences == 0:
            return ToolResult(content="", error="old_string not found in file")
        if occurrences > 1 and not replace_all:
            return ToolResult(
                content="",
                error=(
                    f"old_string is not unique ({occurrences} matches); add more "
                    "context or set replace_all=true"
                ),
            )

        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        write_op = Operation(
            type=OperationType.FILE_WRITE,
            description=f"Edit file: {path}",
            data={"path": path, "content": new_content},
            requires_approval=True,
        )
        write_result = await self.agent_engine.execute_with_approval(write_op)
        if not write_result.success:
            return self._operation_result_to_tool_result(write_result)

        replaced = occurrences if replace_all else 1
        plural = "s" if replaced != 1 else ""
        return ToolResult(content=f"Edited {path} ({replaced} replacement{plural})")

    def _build_grep_command(self, args: dict) -> str:
        """Build a `grep` command from Claude Code-style Grep arguments."""
        pattern = args.get("pattern", "")
        directory = args.get("path") or args.get("directory") or "."
        glob = args.get("glob") or args.get("file_pattern")
        output_mode = args.get("output_mode", "files_with_matches")
        case_insensitive = args.get("-i") or args.get("ignore_case")
        show_line_numbers = args.get("-n", True)
        context = args.get("-C") or args.get("context")
        head_limit = args.get("head_limit")

        flags = ["-r"]
        if case_insensitive:
            flags.append("-i")
        if output_mode == "files_with_matches":
            flags.append("-l")
        elif output_mode == "count":
            flags.append("-c")
        else:  # content
            if show_line_numbers is not False:
                flags.append("-n")
            if context:
                flags.append(f"-C {int(context)}")

        cmd = f"grep {' '.join(flags)} '{pattern}' {directory}{_GREP_EXCLUDES}"
        if glob:
            cmd += f" --include='{glob}'"
        if head_limit:
            cmd += f" | head -n {int(head_limit)}"
        return cmd

    async def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        results = []
        for tc in tool_calls:
            result = await self.execute_tool_call(tc)
            results.append(result)
        return results

    def _tool_call_to_operation(self, tool_call: ToolCall) -> Optional[Operation]:
        name = tool_call.name
        if name not in TOOL_TO_OPERATION:
            return None

        args = tool_call.arguments

        if name in _READ_TOOLS:
            path = self._path_arg(args)
            return Operation(
                type=OperationType.FILE_READ,
                description=f"Read file: {path}",
                data={"path": path},
                requires_approval=False,
            )

        if name in _WRITE_TOOLS:
            path = self._path_arg(args)
            return Operation(
                type=OperationType.FILE_WRITE,
                description=f"Write file: {path}",
                data={"path": path, "content": args.get("content", "")},
                requires_approval=True,
            )

        if name in _DELETE_TOOLS:
            path = self._path_arg(args)
            return Operation(
                type=OperationType.FILE_DELETE,
                description=f"Delete file: {path}",
                data={"path": path},
                requires_approval=True,
            )

        if name in _BASH_TOOLS:
            return Operation(
                type=OperationType.COMMAND_EXECUTE,
                description=f"Execute: {args.get('command', '')}",
                data={
                    "command": args.get("command", ""),
                    "working_dir": args.get("working_dir"),
                },
                requires_approval=True,
            )

        if name in _GLOB_TOOLS:
            pattern = args.get("pattern", "*")
            directory = args.get("path") or args.get("directory") or "."
            return Operation(
                type=OperationType.COMMAND_EXECUTE,
                description=f"Find files: {pattern}",
                data={
                    "command": (
                        f"find {directory} -type f -name '{pattern}'{_FIND_PRUNE}"
                    )
                },
                requires_approval=False,
            )

        if name in _GREP_TOOLS:
            return Operation(
                type=OperationType.COMMAND_EXECUTE,
                description=f"Search: {args.get('pattern', '')}",
                data={"command": self._build_grep_command(args)},
                requires_approval=False,
            )

        if name in _WEB_TOOLS:
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

    def _operation_result_to_tool_result(self, result: OperationResult) -> ToolResult:
        if result.success:
            content = (
                result.data
                if isinstance(result.data, str)
                else json.dumps(result.data, default=str)
            )
            return ToolResult(content=_truncate_tool_output(content or "OK"))
        else:
            error_msg = result.error or "Operation failed"
            if result.was_cancelled:
                error_msg = "Operation cancelled by user"
            return ToolResult(content="", error=error_msg)
