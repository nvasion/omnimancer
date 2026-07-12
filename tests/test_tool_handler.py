"""Tests for the native tool call handler."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.tool_handler import (
    AUTO_APPROVED_TOOLS,
    MAX_TOOL_ITERATIONS,
    MAX_TOOL_RESULT_CHARS,
    TOOL_TO_OPERATION,
    ToolHandler,
)
from omnimancer.core.agent.tool_definitions import CODING_AGENT_TOOLS
from omnimancer.core.agent.types import OperationResult, OperationType
from omnimancer.core.models import ToolCall


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
        tc = ToolCall(
            name="file_write",
            arguments={
                "path": "/src/new.py",
                "content": "print('hello')",
            },
        )
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

    def test_claude_code_names_map(self, tool_handler):
        # The Claude Code tool names resolve to operations.
        for name in ("Read", "Write", "Bash", "Glob", "Grep", "WebFetch"):
            op = tool_handler._tool_call_to_operation(ToolCall(name=name, arguments={}))
            assert op is not None, name

    def test_read_accepts_file_path(self, tool_handler):
        op = tool_handler._tool_call_to_operation(
            ToolCall(name="Read", arguments={"file_path": "/src/main.py"})
        )
        assert op.type == OperationType.FILE_READ
        assert op.data["path"] == "/src/main.py"

    def test_glob_uses_path_argument(self, tool_handler):
        op = tool_handler._tool_call_to_operation(
            ToolCall(name="Glob", arguments={"pattern": "*.py", "path": "src"})
        )
        assert "find src -type f -name '*.py'" in op.data["command"]

    def test_grep_default_is_files_with_matches(self, tool_handler):
        op = tool_handler._tool_call_to_operation(
            ToolCall(name="Grep", arguments={"pattern": "X"})
        )
        assert "-l" in op.data["command"]

    def test_grep_content_mode_with_options(self, tool_handler):
        op = tool_handler._tool_call_to_operation(
            ToolCall(
                name="Grep",
                arguments={
                    "pattern": "X",
                    "output_mode": "content",
                    "-i": True,
                    "-C": 2,
                    "glob": "*.py",
                    "head_limit": 50,
                },
            )
        )
        cmd = op.data["command"]
        assert "-i" in cmd and "-n" in cmd and "-C 2" in cmd
        assert "--include='*.py'" in cmd
        assert "| head -n 50" in cmd

    @pytest.mark.asyncio
    async def test_read_offset_limit_slices(self, tool_handler, mock_agent_engine):
        mock_agent_engine.execute_with_approval.return_value = OperationResult(
            success=True, data="l1\nl2\nl3\nl4\nl5"
        )
        result = await tool_handler.execute_tool_call(
            ToolCall(
                name="Read", arguments={"file_path": "/f", "offset": 2, "limit": 2}
            )
        )
        assert result.content == "l2\nl3"

    @pytest.mark.asyncio
    async def test_edit_replaces_unique_string(self, tool_handler, mock_agent_engine):
        calls = []

        async def fake(op):
            calls.append(op.type)
            if op.type == OperationType.FILE_READ:
                return OperationResult(success=True, data="alpha beta gamma")
            return OperationResult(success=True, data="written")

        mock_agent_engine.execute_with_approval = AsyncMock(side_effect=fake)
        result = await tool_handler.execute_tool_call(
            ToolCall(
                name="Edit",
                arguments={
                    "file_path": "/f",
                    "old_string": "beta",
                    "new_string": "BETA",
                },
            )
        )
        assert result.error is None
        assert "Edited /f" in result.content
        assert OperationType.FILE_READ in calls and OperationType.FILE_WRITE in calls

    @pytest.mark.asyncio
    async def test_edit_non_unique_requires_replace_all(
        self, tool_handler, mock_agent_engine
    ):
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="x x x")
        )
        result = await tool_handler.execute_tool_call(
            ToolCall(
                name="Edit",
                arguments={"file_path": "/f", "old_string": "x", "new_string": "y"},
            )
        )
        assert result.error is not None
        assert "not unique" in result.error

    @pytest.mark.asyncio
    async def test_edit_missing_old_string_errors(
        self, tool_handler, mock_agent_engine
    ):
        mock_agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="hello")
        )
        result = await tool_handler.execute_tool_call(
            ToolCall(
                name="Edit",
                arguments={"file_path": "/f", "old_string": "zzz", "new_string": "y"},
            )
        )
        assert "not found" in result.error

    def test_find_files_excludes_heavy_dirs(self, tool_handler):
        tc = ToolCall(name="find_files", arguments={"pattern": "*.py"})
        cmd = tool_handler._tool_call_to_operation(tc).data["command"]
        assert "-not -path '*/.venv/*'" in cmd
        assert "-not -path '*/node_modules/*'" in cmd
        assert "-not -path '*/.git/*'" in cmd

    def test_search_text_excludes_heavy_dirs(self, tool_handler):
        tc = ToolCall(name="search_text", arguments={"pattern": "import"})
        cmd = tool_handler._tool_call_to_operation(tc).data["command"]
        assert "--exclude-dir=.venv" in cmd
        assert "--exclude-dir=node_modules" in cmd

    def test_search_text_mapping(self, tool_handler):
        tc = ToolCall(
            name="search_text",
            arguments={
                "pattern": "def main",
                "file_pattern": "*.py",
            },
        )
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
        tc = ToolCall(
            name="web_request",
            arguments={
                "url": "https://api.example.com",
                "method": "POST",
            },
        )
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
    async def test_small_output_not_truncated(self, tool_handler, mock_agent_engine):
        mock_agent_engine.execute_with_approval.return_value = OperationResult(
            success=True, data="x" * 1000
        )
        tc = ToolCall(name="file_read", arguments={"path": "/a"})
        result = await tool_handler.execute_tool_call(tc)

        assert result.content == "x" * 1000
        assert "truncated" not in result.content

    @pytest.mark.asyncio
    async def test_large_output_truncated(self, tool_handler, mock_agent_engine):
        # Simulates `find`/`grep` dumping the whole tree (incl. .venv).
        huge = "y" * 500_000
        mock_agent_engine.execute_with_approval.return_value = OperationResult(
            success=True, data=huge
        )
        tc = ToolCall(name="search_text", arguments={"pattern": "import"})
        result = await tool_handler.execute_tool_call(tc)

        assert len(result.content) < len(huge)
        assert len(result.content) <= MAX_TOOL_RESULT_CHARS + 200  # marker overhead
        assert "characters truncated" in result.content
        # Keeps both head and tail of the output.
        assert result.content.startswith("y")
        assert result.content.endswith("y")

    @pytest.mark.asyncio
    async def test_execute_tool_call_failure(self, tool_handler, mock_agent_engine):
        mock_agent_engine.execute_with_approval.return_value = OperationResult(
            success=False, error="Permission denied"
        )

        tc = ToolCall(
            name="file_write",
            arguments={"path": "/etc/passwd", "content": "bad"},
        )
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
        assert result.cancelled is True

    @pytest.mark.asyncio
    async def test_denied_result_not_marked_cancelled(
        self, tool_handler, mock_agent_engine
    ):
        """'n' (deny) only rejects one operation — the turn continues."""
        mock_agent_engine.execute_with_approval.return_value = OperationResult(
            success=False, error="Operation not approved by user"
        )

        tc = ToolCall(name="command_exec", arguments={"command": "ls"})
        result = await tool_handler.execute_tool_call(tc)

        assert result.error
        assert result.cancelled is False

    @pytest.mark.asyncio
    async def test_execute_tool_calls_stops_after_cancellation(
        self, tool_handler, mock_agent_engine
    ):
        """'q' (quit) must not prompt for the remaining calls in the batch."""
        mock_agent_engine.execute_with_approval.return_value = OperationResult(
            success=False, error="User rejected", was_cancelled=True
        )

        tool_calls = [
            ToolCall(name="command_exec", arguments={"command": "ls"}),
            ToolCall(name="command_exec", arguments={"command": "pwd"}),
        ]
        results = await tool_handler.execute_tool_calls(tool_calls)

        assert mock_agent_engine.execute_with_approval.call_count == 1
        assert results[-1].cancelled is True

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, tool_handler):
        tc = ToolCall(name="unknown", arguments={})
        result = await tool_handler.execute_tool_call(tc)

        assert result.error == "Unknown tool: unknown"

    @pytest.mark.asyncio
    async def test_non_dict_arguments_return_corrective_error(
        self, tool_handler, mock_agent_engine
    ):
        """A provider that ships unparseable arguments must yield a clear
        error the model can act on — not "'str' object has no attribute
        'get'" from inside the handler."""
        tc = ToolCall(name="Glob", arguments="**/*.py")
        result = await tool_handler.execute_tool_call(tc)

        assert result.error is not None
        assert "JSON object" in result.error
        assert "attribute" not in result.error
        mock_agent_engine.execute_with_approval.assert_not_called()

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
        mock_agent_engine.execute_with_approval.side_effect = RuntimeError(
            "engine crash"
        )

        tc = ToolCall(name="file_read", arguments={"path": "/a.py"})
        result = await tool_handler.execute_tool_call(tc)

        assert result.error == "engine crash"


class TestGlobPatternTranslation:
    """Glob patterns must translate to `find` predicates that can match.

    `find -name` matches basenames only — a pattern containing `/` (like the
    '**/*.py' the tool schema suggests) silently matches nothing.
    """

    def _cmd(self, tool_handler, pattern, path=None):
        args = {"pattern": pattern}
        if path:
            args["path"] = path
        return tool_handler._tool_call_to_operation(
            ToolCall(name="Glob", arguments=args)
        ).data["command"]

    def test_leading_recursive_glob_uses_name(self, tool_handler):
        cmd = self._cmd(tool_handler, "**/Dockerfile*")
        assert "-name 'Dockerfile*'" in cmd
        assert "**" not in cmd

    def test_interior_slash_uses_path_predicate(self, tool_handler):
        cmd = self._cmd(tool_handler, "src/**/*.ts")
        assert "-path" in cmd
        assert "**" not in cmd

    def test_simple_pattern_still_uses_name(self, tool_handler):
        cmd = self._cmd(tool_handler, "*.py")
        assert "-name '*.py'" in cmd

    def test_pattern_with_single_quote_is_escaped(self, tool_handler):
        cmd = self._cmd(tool_handler, "it's*")
        # The quote must not terminate the shell string (injection vector
        # for an auto-approved tool).
        assert "'; " not in cmd
        assert "it" in cmd

    def test_injection_attempt_stays_inside_quotes(self, tool_handler):
        cmd = self._cmd(tool_handler, "'; rm -rf ~; '")
        # shlex-quoted: the malicious payload must remain a single shell word.
        import shlex

        words = shlex.split(cmd.split(" -name ", 1)[1].split(" -not", 1)[0])
        assert words == ["'; rm -rf ~; '"]

    def test_directory_with_space_is_quoted(self, tool_handler):
        cmd = self._cmd(tool_handler, "*.py", path="my dir")
        assert "'my dir'" in cmd


class TestGrepCommandTranslation:
    """Grep must use ERE, escape model input, and expand brace globs."""

    def _cmd(self, tool_handler, **args):
        return tool_handler._tool_call_to_operation(
            ToolCall(name="Grep", arguments=args)
        ).data["command"]

    def test_uses_extended_regex(self, tool_handler):
        cmd = self._cmd(tool_handler, pattern="foo|bar")
        assert "-E" in cmd.split()

    def test_translates_pcre_digit_class(self, tool_handler):
        cmd = self._cmd(tool_handler, pattern=r"v\d+")
        assert "[0-9]" in cmd
        assert r"\d" not in cmd

    def test_pattern_with_single_quote_is_escaped(self, tool_handler):
        import shlex

        cmd = self._cmd(tool_handler, pattern="it's")
        # Must parse as valid shell with the pattern as one word.
        assert "it's" in shlex.split(cmd.split("|")[0])

    def test_brace_glob_expands_to_multiple_includes(self, tool_handler):
        cmd = self._cmd(tool_handler, pattern="X", glob="*.{ts,tsx}")
        assert "--include='*.ts'" in cmd
        assert "--include='*.tsx'" in cmd
        assert "{" not in cmd

    def test_directory_with_space_is_quoted(self, tool_handler):
        cmd = self._cmd(tool_handler, pattern="X", path="my dir")
        assert "'my dir'" in cmd


def _command_result(stdout="", stderr="", returncode=0, success=None):
    if success is None:
        success = returncode == 0
    return OperationResult(
        success=success,
        data={"stdout": stdout, "stderr": stderr, "returncode": returncode},
        error=(stderr or None) if not success else None,
    )


class TestSearchResultSemantics:
    """Empty search results must be explicit, not silent or fake failures."""

    @pytest.mark.asyncio
    async def test_glob_empty_result_says_no_files(
        self, tool_handler, mock_agent_engine
    ):
        mock_agent_engine.execute_with_approval.return_value = _command_result()
        result = await tool_handler.execute_tool_call(
            ToolCall(name="Glob", arguments={"pattern": "**/Dockerfile*"})
        )
        assert result.error is None
        assert "No files found" in result.content
        assert "**/Dockerfile*" in result.content

    @pytest.mark.asyncio
    async def test_glob_returns_stdout_not_json(self, tool_handler, mock_agent_engine):
        mock_agent_engine.execute_with_approval.return_value = _command_result(
            stdout="./a/Dockerfile\n./b/Dockerfile.dev\n"
        )
        result = await tool_handler.execute_tool_call(
            ToolCall(name="Glob", arguments={"pattern": "**/Dockerfile*"})
        )
        assert result.error is None
        assert result.content == "./a/Dockerfile\n./b/Dockerfile.dev"

    @pytest.mark.asyncio
    async def test_grep_no_match_exit_1_is_not_an_error(
        self, tool_handler, mock_agent_engine
    ):
        # grep exits 1 when nothing matches — a valid empty result.
        mock_agent_engine.execute_with_approval.return_value = _command_result(
            returncode=1
        )
        result = await tool_handler.execute_tool_call(
            ToolCall(name="Grep", arguments={"pattern": "NoSuchThing"})
        )
        assert result.error is None
        assert "No matches found" in result.content

    @pytest.mark.asyncio
    async def test_grep_real_failure_still_errors(
        self, tool_handler, mock_agent_engine
    ):
        mock_agent_engine.execute_with_approval.return_value = _command_result(
            stderr="grep: bad regex", returncode=2
        )
        result = await tool_handler.execute_tool_call(
            ToolCall(name="Grep", arguments={"pattern": "("})
        )
        assert result.error is not None
        assert "bad regex" in result.error


class TestCommandResultFidelity:
    """Bash results must carry stdout even on failure (pytest writes there)."""

    @pytest.mark.asyncio
    async def test_failed_command_keeps_stdout(self, tool_handler, mock_agent_engine):
        mock_agent_engine.execute_with_approval.return_value = _command_result(
            stdout="FAILED tests/test_x.py::test_y - AssertionError",
            stderr="",
            returncode=1,
        )
        result = await tool_handler.execute_tool_call(
            ToolCall(name="Bash", arguments={"command": "pytest"})
        )
        assert result.error is not None
        assert "FAILED tests/test_x.py::test_y" in result.error
        assert "exit" in result.error.lower() or "code 1" in result.error

    @pytest.mark.asyncio
    async def test_successful_command_returns_stdout_not_json(
        self, tool_handler, mock_agent_engine
    ):
        mock_agent_engine.execute_with_approval.return_value = _command_result(
            stdout="hello\n"
        )
        result = await tool_handler.execute_tool_call(
            ToolCall(name="Bash", arguments={"command": "echo hello"})
        )
        assert result.error is None
        assert result.content == "hello"

    @pytest.mark.asyncio
    async def test_successful_command_with_no_output_is_explicit(
        self, tool_handler, mock_agent_engine
    ):
        mock_agent_engine.execute_with_approval.return_value = _command_result()
        result = await tool_handler.execute_tool_call(
            ToolCall(name="Bash", arguments={"command": "true"})
        )
        assert result.error is None
        assert result.content  # never empty — the model needs a signal


class TestRepeatedCallTracker:
    """Identical repeated calls are nudged before the turn is aborted.

    Aborting on the 3rd identical call punished the user for model confusion
    (and killed turns where a re-run was legitimate, e.g. re-Read after Edit).
    """

    def _tracker(self):
        from omnimancer.cli.tool_handler import RepeatedCallTracker

        return RepeatedCallTracker()

    def _call(self, name="Read", **args):
        return ToolCall(name=name, arguments=args)

    def test_first_two_occurrences_execute_normally(self):
        tracker = self._tracker()
        tc = self._call(path="/a.py")
        tracker.record([tc])
        assert not tracker.is_duplicate(tc)
        tracker.record([tc])
        assert not tracker.is_duplicate(tc)

    def test_third_occurrence_is_duplicate_not_abort(self):
        tracker = self._tracker()
        tc = self._call(path="/a.py")
        for _ in range(3):
            tracker.record([tc])
        assert tracker.is_duplicate(tc)
        assert tracker.abort_offender([tc]) is None

    def test_fifth_occurrence_aborts_and_names_offender(self):
        tracker = self._tracker()
        tc = self._call(path="/a.py")
        for _ in range(5):
            tracker.record([tc])
        offender = tracker.abort_offender([tc])
        assert offender is tc
        assert tracker.count(tc) == 5

    def test_different_arguments_are_not_duplicates(self):
        tracker = self._tracker()
        for offset in range(10):
            tc = self._call(path="/a.py", offset=offset)
            tracker.record([tc])
            assert not tracker.is_duplicate(tc)


class TestToolDefinitions:
    """Test the coding agent tool definitions."""

    def test_all_tools_defined(self):
        tool_names = {t.name for t in CODING_AGENT_TOOLS}
        expected = {
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "WebFetch",
        }
        assert tool_names == expected

    def test_tool_definitions_have_required_fields(self):
        for tool in CODING_AGENT_TOOLS:
            assert tool.name
            assert tool.description
            assert tool.parameters
            assert "properties" in tool.parameters

    def test_auto_approved_tools_match(self):
        auto_approved_from_defs = {
            t.name for t in CODING_AGENT_TOOLS if t.auto_approved
        }
        # Every auto-approved tool definition must be recognized as auto-approved
        # (AUTO_APPROVED_TOOLS also contains legacy aliases).
        assert auto_approved_from_defs == {"Read", "Glob", "Grep", "WebFetch"}
        assert auto_approved_from_defs <= AUTO_APPROVED_TOOLS

    def test_get_tool_definitions(self, tool_handler):
        tools = tool_handler.get_tool_definitions()
        assert len(tools) == len(CODING_AGENT_TOOLS)

    def test_max_iterations_constant(self):
        assert MAX_TOOL_ITERATIONS == 25

    def test_all_tools_have_operation_mapping(self):
        for tool in CODING_AGENT_TOOLS:
            assert tool.name in TOOL_TO_OPERATION, (
                f"No operation mapping for tool: " f"{tool.name}"
            )
