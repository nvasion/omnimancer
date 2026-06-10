"""Native tool call handler — bridges provider tool calls to AgentEngine execution."""

import json
import logging
import re
import shlex
from typing import Any, List, Optional

from ..core.agent.tool_definitions import CODING_AGENT_TOOLS
from ..core.agent.types import Operation, OperationResult, OperationType
from ..core.models import ToolCall, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 25

# Identical-call repetition policy: the 1st and 2nd occurrences execute
# normally (re-running a call can be legitimate — e.g. re-Read after an Edit),
# later occurrences are skipped and replaced with a corrective nudge, and the
# turn is aborted only if the model keeps repeating despite the nudges.
DUPLICATE_NUDGE_THRESHOLD = 3
DUPLICATE_ABORT_THRESHOLD = 5

DUPLICATE_CALL_NUDGE = (
    "Duplicate call skipped: you already made this exact call and received "
    "its result earlier in this conversation. Use that result, change the "
    "arguments, or give your final answer — do not repeat the call."
)


class RepeatedCallTracker:
    """Counts identical tool calls within one agent turn.

    A repeated identical call usually means the model lost track of an
    earlier result. Skipping it with a nudge lets the model recover;
    aborting is the last resort once nudges are ignored.
    """

    def __init__(self) -> None:
        self._counts: dict = {}

    @staticmethod
    def _signature(tool_call: ToolCall) -> str:
        return (
            f"{tool_call.name}:"
            f"{json.dumps(tool_call.arguments, sort_keys=True, default=str)}"
        )

    def record(self, tool_calls: List[ToolCall]) -> None:
        """Count one occurrence of each call in a response."""
        for tc in tool_calls:
            sig = self._signature(tc)
            self._counts[sig] = self._counts.get(sig, 0) + 1

    def count(self, tool_call: ToolCall) -> int:
        return self._counts.get(self._signature(tool_call), 0)

    def is_duplicate(self, tool_call: ToolCall) -> bool:
        """True when this occurrence should be skipped and nudged."""
        return self.count(tool_call) >= DUPLICATE_NUDGE_THRESHOLD

    def abort_offender(self, tool_calls: List[ToolCall]) -> Optional[ToolCall]:
        """The call that exhausted its nudges, if any — abort the turn."""
        worst: Optional[ToolCall] = None
        for tc in tool_calls:
            if self.count(tc) >= DUPLICATE_ABORT_THRESHOLD:
                if worst is None or self.count(tc) > self.count(worst):
                    worst = tc
        return worst

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
            if tool_call.name in _GLOB_TOOLS or tool_call.name in _GREP_TOOLS:
                return await self._execute_search(tool_call)

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

    @staticmethod
    def _translate_regex(pattern: str) -> str:
        """Translate PCRE-only character classes to POSIX equivalents.

        Models write ripgrep/PCRE-style regexes; `grep -E` (ERE) covers most
        of that syntax but has no \\d or \\D.
        """
        return pattern.replace(r"\d", "[0-9]").replace(r"\D", "[^0-9]")

    @staticmethod
    def _expand_braces(glob: str) -> List[str]:
        """Expand '*.{ts,tsx}' into ['*.ts', '*.tsx'] — fnmatch (used by
        grep --include) has no brace expansion."""
        match = re.search(r"\{([^{}]*)\}", glob)
        if not match:
            return [glob]
        head, tail = glob[: match.start()], glob[match.end() :]
        expanded: List[str] = []
        for alternative in match.group(1).split(","):
            expanded.extend(ToolHandler._expand_braces(head + alternative + tail))
        return expanded

    @staticmethod
    def _build_find_command(pattern: str, directory: str) -> str:
        """Translate a glob pattern into a `find` command.

        `find -name` matches basenames only, so a pattern containing `/`
        (e.g. '**/*.py') would silently match nothing. Patterns with interior
        slashes use `-path`, where `*` also crosses directory separators.
        """
        if "/" not in pattern:
            predicate = f"-name {shlex.quote(pattern)}"
        elif pattern.startswith("**/") and "/" not in pattern[3:]:
            # find already recurses; '**/' adds nothing for a basename match.
            predicate = f"-name {shlex.quote(pattern[3:])}"
        else:
            full = f"{directory.rstrip('/')}/{pattern}"
            # In -path's fnmatch, '*' crosses '/', so '**' collapses to '*'.
            while "**" in full:
                full = full.replace("**", "*")
            predicate = f"-path {shlex.quote(full)}"
        return f"find {shlex.quote(directory)} -type f {predicate}{_FIND_PRUNE}"

    async def _execute_search(self, tool_call: ToolCall) -> ToolResult:
        """Run Glob/Grep and make empty results explicit.

        An empty search must read as "found nothing" — a silent/blank result
        (or grep's exit code 1 surfacing as a failure) gives the model no
        signal, so it retries the identical call until the loop detector
        kills the turn.
        """
        operation = self._tool_call_to_operation(tool_call)
        result = await self.agent_engine.execute_with_approval(operation)

        data = result.data if isinstance(result.data, dict) else {}
        stdout = (
            data.get("stdout")
            if data
            else (result.data if isinstance(result.data, str) else "")
        ) or ""
        returncode = data.get("returncode")

        # grep exits 1 when nothing matched — a valid empty result, not an error.
        no_match_exit = tool_call.name in _GREP_TOOLS and returncode == 1
        if result.success or no_match_exit:
            if stdout.strip():
                return ToolResult(content=_truncate_tool_output(stdout.strip()))
            pattern = tool_call.arguments.get("pattern", "")
            noun = "files found" if tool_call.name in _GLOB_TOOLS else "matches found"
            return ToolResult(
                content=(
                    f"No {noun} matching pattern '{pattern}'. "
                    "Try a different pattern or location."
                )
            )
        return self._operation_result_to_tool_result(result)

    def _build_grep_command(self, args: dict) -> str:
        """Build a `grep` command from Claude Code-style Grep arguments."""
        pattern = self._translate_regex(args.get("pattern", ""))
        directory = args.get("path") or args.get("directory") or "."
        glob = args.get("glob") or args.get("file_pattern")
        output_mode = args.get("output_mode", "files_with_matches")
        case_insensitive = args.get("-i") or args.get("ignore_case")
        show_line_numbers = args.get("-n", True)
        context = args.get("-C") or args.get("context")
        head_limit = args.get("head_limit")

        # -E: models write extended-regex syntax (alternation, +, groups),
        # which plain grep treats as literal characters.
        flags = ["-r", "-E"]
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

        cmd = (
            f"grep {' '.join(flags)} {shlex.quote(pattern)} "
            f"{shlex.quote(directory)}{_GREP_EXCLUDES}"
        )
        if glob:
            for expanded in self._expand_braces(glob):
                cmd += f" --include={shlex.quote(expanded)}"
        if head_limit:
            cmd += f" | head -n {int(head_limit)}"
        return cmd

    async def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        results = []
        for tc in tool_calls:
            result = await self.execute_tool_call(tc)
            results.append(result)
            # "q" at the approval prompt cancels the turn — don't keep
            # prompting for the remaining calls in this batch.
            if result.cancelled:
                break
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
                data={"command": self._build_find_command(pattern, directory)},
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
        if isinstance(result.data, dict) and "stdout" in result.data:
            return self._command_result_to_tool_result(result)
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
            return ToolResult(
                content="", error=error_msg, cancelled=result.was_cancelled
            )

    @staticmethod
    def _command_result_to_tool_result(result: OperationResult) -> ToolResult:
        """Convert a command OperationResult, preserving stdout on failure.

        Tools like pytest report their failures on stdout with a nonzero
        exit code — dropping stdout there leaves the model blind to why the
        command failed.
        """
        if result.was_cancelled:
            return ToolResult(
                content="", error="Operation cancelled by user", cancelled=True
            )

        data = result.data
        stdout = (data.get("stdout") or "").strip()
        stderr = (data.get("stderr") or "").strip()
        returncode = data.get("returncode")

        if result.success:
            content = stdout or stderr or "OK (command produced no output)"
            return ToolResult(content=_truncate_tool_output(content))

        parts = [f"Command exited with code {returncode}."]
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if len(parts) == 1 and result.error:
            parts.append(result.error)
        return ToolResult(content="", error=_truncate_tool_output("\n".join(parts)))
