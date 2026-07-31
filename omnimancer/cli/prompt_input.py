"""prompt_toolkit input layer for the interactive REPL.

Replaces builtin ``input()`` (which blocked the entire asyncio event loop
while waiting at the prompt) with ``PromptSession.prompt_async``. Key
behavior:

* Enter submits; Esc+Enter (Alt+Enter) inserts a newline; a trailing
  backslash before Enter continues on the next line (claude-code parity).
* Bracketed paste is native: a multi-line paste lands in the buffer as one
  editable block and submits on Enter — this replaces the old
  termios/select paste-drain hack.
* Ctrl+C clears a non-empty buffer; at an empty prompt the first press
  shows an exit hint and the second consecutive press exits. Ctrl+D at an
  empty prompt is EOF (exit), as before.
* History persists to ``prompt_history`` (prompt_toolkit FileHistory
  format) with a one-time migration from the old readline history file,
  which is left untouched for the non-TTY readline fallback path.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

logger = logging.getLogger(__name__)

#: Old readline history filename (owned by the non-TTY fallback path).
READLINE_HISTORY_FILENAME = "readline_history"
#: prompt_toolkit FileHistory filename.
PROMPT_HISTORY_FILENAME = "prompt_history"


def _migrate_readline_history(history_dir: Path) -> Path:
    """One-time migration: readline history lines → FileHistory format.

    Runs only when ``prompt_history`` does not exist yet. The readline file
    is never modified or removed — the readline fallback still uses it.
    """
    prompt_history = history_dir / PROMPT_HISTORY_FILENAME
    readline_history = history_dir / READLINE_HISTORY_FILENAME

    if prompt_history.exists() or not readline_history.exists():
        return prompt_history

    try:
        history = FileHistory(str(prompt_history))
        for line in readline_history.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.strip():
                history.append_string(line)
        logger.info("Migrated readline history to %s", prompt_history)
    except Exception as e:
        logger.warning("Readline history migration failed: %s", e)
    return prompt_history


class PromptInput:
    """Owns the PromptSession and its key bindings for the REPL prompt."""

    def __init__(
        self,
        history_dir: Path,
        completer: Optional[Any] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
    ) -> None:
        """
        Args:
            history_dir: Directory holding the prompt history files.
            completer: Optional prompt_toolkit Completer.
            input/output: Test seams (pipe input / DummyOutput); production
                omits both and prompt_toolkit binds the real terminal.
        """
        history_dir = Path(history_dir)
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = _migrate_readline_history(history_dir)

        # Armed after a Ctrl+C at an empty prompt; the next consecutive
        # Ctrl+C exits. Any submitted input disarms.
        self._exit_armed = False

        session_kwargs: dict = {
            "history": FileHistory(str(history_path)),
            "auto_suggest": AutoSuggestFromHistory(),
            "multiline": True,
            "key_bindings": self._build_key_bindings(),
            "prompt_continuation": "... ",
        }
        if completer is not None:
            session_kwargs["completer"] = completer
            session_kwargs["complete_while_typing"] = True
        if input is not None:
            session_kwargs["input"] = input
        if output is not None:
            session_kwargs["output"] = output

        self.session: PromptSession = PromptSession(**session_kwargs)

    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def _submit_or_continue(event: Any) -> None:
            buffer = event.current_buffer
            document = buffer.document
            # Trailing backslash: strip it and keep editing on a new line.
            if document.text_before_cursor.endswith(
                "\\"
            ) and document.cursor_position == len(document.text):
                buffer.delete_before_cursor(1)
                buffer.insert_text("\n")
                return
            buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        def _insert_newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-c")
        def _ctrl_c(event: Any) -> None:
            buffer = event.current_buffer
            if buffer.text:
                buffer.reset()
                self._exit_armed = False
                return
            if self._exit_armed:
                event.app.exit(exception=KeyboardInterrupt())
                return
            self._exit_armed = True
            # Transient hint in the toolbar area rather than corrupting
            # the scrollback.
            event.app.output.bell()

        return bindings

    @property
    def exit_armed(self) -> bool:
        """True after one Ctrl+C at an empty prompt (exit hint state)."""
        return self._exit_armed

    async def prompt_async(self, message: str = ">>> ") -> str:
        """Read one submission from the user.

        Raises:
            EOFError: Ctrl+D at an empty prompt.
            KeyboardInterrupt: second consecutive Ctrl+C at an empty prompt.
        """
        with patch_stdout():
            text: str = await self.session.prompt_async(message)
        self._exit_armed = False
        return text
