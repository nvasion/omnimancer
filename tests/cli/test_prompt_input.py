"""PromptInput — the prompt_toolkit input layer for the interactive REPL.

Driven headlessly through prompt_toolkit's pipe input + DummyOutput.
Key contract: Enter submits; Esc+Enter and trailing-backslash insert
newlines; bracketed paste lands as an editable block; Ctrl+C clears the
buffer first and exits on a second press at an empty prompt; history
migrates one-time from the old readline file.
"""

from contextlib import ExitStack

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from omnimancer.cli.prompt_input import PromptInput


@pytest.fixture
def make_prompt(tmp_path):
    """Factory: PromptInput wired to a pipe input we can feed keys into."""
    with ExitStack() as stack:

        def _make(feed: str, history_dir=None):
            pipe = stack.enter_context(create_pipe_input())
            pipe.send_text(feed)
            return PromptInput(
                history_dir=history_dir or tmp_path,
                input=pipe,
                output=DummyOutput(),
            )

        yield _make


class TestSubmission:
    @pytest.mark.asyncio
    async def test_enter_submits(self, make_prompt):
        prompt = make_prompt("hello\r")
        assert await prompt.prompt_async() == "hello"

    @pytest.mark.asyncio
    async def test_escape_enter_inserts_newline(self, make_prompt):
        prompt = make_prompt("line1\x1b\rline2\r")
        assert await prompt.prompt_async() == "line1\nline2"

    @pytest.mark.asyncio
    async def test_trailing_backslash_continues(self, make_prompt):
        prompt = make_prompt("line1\\\rline2\r")
        assert await prompt.prompt_async() == "line1\nline2"

    @pytest.mark.asyncio
    async def test_bracketed_paste_is_one_editable_block(self, make_prompt):
        prompt = make_prompt("\x1b[200~a\nb\x1b[201~\r")
        assert await prompt.prompt_async() == "a\nb"


class TestCtrlC:
    @pytest.mark.asyncio
    async def test_ctrl_c_clears_buffer_then_input_continues(self, make_prompt):
        prompt = make_prompt("abc\x03hello\r")
        assert await prompt.prompt_async() == "hello"

    @pytest.mark.asyncio
    async def test_double_ctrl_c_on_empty_prompt_exits(self, make_prompt):
        prompt = make_prompt("\x03\x03")
        with pytest.raises(KeyboardInterrupt):
            await prompt.prompt_async()

    @pytest.mark.asyncio
    async def test_ctrl_d_on_empty_prompt_is_eof(self, make_prompt):
        prompt = make_prompt("\x04")
        with pytest.raises(EOFError):
            await prompt.prompt_async()


class TestHistoryMigration:
    def test_readline_history_migrates_once(self, tmp_path, make_prompt):
        readline_file = tmp_path / "readline_history"
        readline_file.write_text("first command\nsecond command\n")

        make_prompt("x\r", history_dir=tmp_path)

        prompt_history = tmp_path / "prompt_history"
        assert prompt_history.exists()
        content = prompt_history.read_text()
        assert "+first command" in content
        assert "+second command" in content
        # Original stays for the readline fallback path
        assert readline_file.exists()

    def test_no_migration_when_prompt_history_exists(self, tmp_path, make_prompt):
        (tmp_path / "readline_history").write_text("old\n")
        (tmp_path / "prompt_history").write_text("\n# 2026-01-01\n+kept\n")

        make_prompt("x\r", history_dir=tmp_path)

        content = (tmp_path / "prompt_history").read_text()
        assert "+kept" in content
        assert "+old" not in content


class TestStatusToolbar:
    def test_toolbar_uses_status_provider(self, make_prompt):
        """Test that toolbar uses status provider when provided."""

        def status_provider():
            return "gateway/qwen3-coder-30b · $0.03 · read-only"

        prompt = make_prompt("test\r", history_dir=None)
        # Manually set the status provider since we're not using the factory
        prompt._status_provider = status_provider

        # Access the toolbar rendering directly
        toolbar_text = prompt._render_toolbar()
        assert toolbar_text is not None
        assert "gateway/qwen3-coder-30b" in toolbar_text
        assert "read-only" in toolbar_text

    def test_toolbar_combines_status_and_approval_mode(self, make_prompt):
        """Test that toolbar combines status and approval mode when both present."""

        def status_provider():
            return "m1 · $0.00"

        def mode_provider():
            return "ask"

        prompt = make_prompt("test\r", history_dir=None)
        prompt._status_provider = status_provider
        prompt._mode_provider = mode_provider

        toolbar_text = prompt._render_toolbar()
        assert toolbar_text is not None
        assert "m1 · $0.00" in toolbar_text
        assert "approval: ask" in toolbar_text
        assert " | " in toolbar_text  # Should be combined with separator

    def test_toolbar_none_without_provider_or_mode(self, make_prompt):
        """Test that toolbar is None when no status provider and normal mode."""

        def mode_provider():
            return "normal"

        prompt = make_prompt("test\r", history_dir=None)
        prompt._mode_provider = mode_provider
        # No status_provider set

        toolbar_text = prompt._render_toolbar()
        assert toolbar_text is None  # Should be None like original behavior

    def test_toolbar_provider_exception_safe(self, make_prompt):
        """Test that toolbar handles exceptions in status provider gracefully."""

        def failing_status_provider():
            raise RuntimeError("Test exception")

        def mode_provider():
            return "ask"

        prompt = make_prompt("test\r", history_dir=None)
        prompt._status_provider = failing_status_provider
        prompt._mode_provider = mode_provider

        # Should fall back to mode-only behavior
        toolbar_text = prompt._render_toolbar()
        assert toolbar_text is not None  # Should still show approval mode
        assert "approval: ask" in toolbar_text
