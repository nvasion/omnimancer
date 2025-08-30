"""
Enhanced input handler with arrow key navigation and command history.

This module provides advanced terminal input capabilities including:
- Arrow key navigation through command history
- Tab completion
- Search functionality
- Multi-line input support
- Rich formatting and display
"""

import logging
import os
import select
import sys
import termios
import tty
from typing import Callable, List, Optional

from rich.console import Console

from .history_manager import HistoryManager

logger = logging.getLogger(__name__)


class KeyCodes:
    """Terminal key codes for cross-platform compatibility."""

    # Control characters
    CTRL_C = "\x03"
    CTRL_D = "\x04"
    CTRL_L = "\x0c"
    CTRL_R = "\x12"
    CTRL_U = "\x15"
    TAB = "\t"
    ENTER = "\r"
    BACKSPACE = "\x7f"
    ESC = "\x1b"

    # Arrow keys (escape sequences)
    ARROW_UP = "\x1b[A"
    ARROW_DOWN = "\x1b[B"
    ARROW_RIGHT = "\x1b[C"
    ARROW_LEFT = "\x1b[D"

    # Other keys
    HOME = "\x1b[H"
    END = "\x1b[F"
    DELETE = "\x1b[3~"
    PAGE_UP = "\x1b[5~"
    PAGE_DOWN = "\x1b[6~"


class EnhancedInput:
    """
    Enhanced input handler with history navigation and advanced editing.

    Features:
    - Arrow key navigation through command history
    - Basic line editing (cursor movement, backspace, delete)
    - Search functionality (Ctrl+R)
    - Multi-line support
    - Rich display integration
    - Tab completion support
    """

    def __init__(
        self,
        history_manager: HistoryManager,
        console: Console,
        prompt: str = ">>> ",
        completion_callback: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initialize the enhanced input handler.

        Args:
            history_manager: History manager instance
            console: Rich console for output
            prompt: Input prompt string
            completion_callback: Optional tab completion callback
        """
        self.history_manager = history_manager
        self.console = console
        self.prompt = prompt
        self.completion_callback = completion_callback

        # Input state
        self.current_line = ""
        self.cursor_pos = 0
        self.in_search_mode = False
        self.search_query = ""

        # Terminal state
        self.old_termios = None
        self.is_tty = sys.stdin.isatty()

        # For non-TTY fallback
        self.fallback_mode = not self.is_tty

    def get_input(self) -> Optional[str]:
        """
        Get input from user with enhanced features.

        Returns:
            User input string or None for EOF
        """
        if self.fallback_mode:
            return self._fallback_input()

        try:
            return self._enhanced_input()
        except Exception as e:
            logger.warning(f"Enhanced input failed, falling back to basic input: {e}")
            return self._fallback_input()

    def _fallback_input(self) -> Optional[str]:
        """Fallback to basic input() for non-TTY environments."""
        try:
            return input(self.prompt)
        except (EOFError, KeyboardInterrupt):
            return None

    def _enhanced_input(self) -> Optional[str]:
        """Enhanced input with arrow key support."""
        if not self.is_tty or self.fallback_mode:
            return self._fallback_input()

        # Set up raw terminal mode
        try:
            self._setup_terminal()

            # Check if terminal setup failed
            if self.fallback_mode:
                return self._fallback_input()

            # Display initial prompt
            self._display_prompt()

            while True:
                try:
                    key = self._read_key()

                    if key is None:  # EOF
                        return None

                    # Handle different key types
                    result = self._handle_key(key)

                    if result is not None:
                        return result

                except KeyboardInterrupt:
                    # Ctrl+C pressed
                    self._restore_terminal()
                    raise

        finally:
            self._restore_terminal()

    def _setup_terminal(self) -> None:
        """Set up terminal for raw key input."""
        if not self.is_tty:
            return

        try:
            self.old_termios = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        except Exception as e:
            logger.warning(f"Could not set up terminal: {e}")
            self.fallback_mode = True

    def _restore_terminal(self) -> None:
        """Restore terminal to original state."""
        if self.old_termios is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_termios)
            except Exception as e:
                logger.warning(f"Could not restore terminal: {e}")

    def _read_key(self) -> Optional[str]:
        """
        Read a single key or key sequence from stdin.

        Returns:
            Key string or None for EOF
        """
        try:
            # Check if data is available
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                return ""  # No input available, continue

            # Read first character
            char = sys.stdin.read(1)

            if not char:  # EOF
                return None

            # Handle escape sequences
            if char == KeyCodes.ESC:
                # Read additional characters for escape sequences
                try:
                    # Set a short timeout for escape sequences
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        seq = char + sys.stdin.read(2)

                        # Check for longer sequences
                        if (
                            seq.endswith("[")
                            and select.select([sys.stdin], [], [], 0.1)[0]
                        ):
                            seq += sys.stdin.read(1)
                            if (
                                seq[-1] in "0123456789"
                                and select.select([sys.stdin], [], [], 0.1)[0]
                            ):
                                seq += sys.stdin.read(1)

                        return seq
                    else:
                        return char  # Just ESC
                except:
                    return char

            return char

        except Exception as e:
            logger.debug(f"Error reading key: {e}")
            return None

    def _handle_key(self, key: str) -> Optional[str]:
        """
        Handle a key press and return command if complete.

        Args:
            key: Key string from _read_key

        Returns:
            Complete command string or None to continue
        """
        if not key:  # No input
            return None

        # Handle control characters
        if key == KeyCodes.CTRL_C:
            raise KeyboardInterrupt()

        elif key == KeyCodes.CTRL_D:
            if not self.current_line:
                return None  # EOF
            else:
                # Delete character at cursor
                if self.cursor_pos < len(self.current_line):
                    self.current_line = (
                        self.current_line[: self.cursor_pos]
                        + self.current_line[self.cursor_pos + 1 :]
                    )
                    self._redraw_line()

        elif key == KeyCodes.CTRL_L:
            # Clear screen
            self.console.clear()
            self._display_prompt()
            self._redraw_line()

        elif key == KeyCodes.CTRL_R:
            # Start reverse search
            self._start_search_mode()

        elif key == KeyCodes.CTRL_U:
            # Clear line
            self.current_line = ""
            self.cursor_pos = 0
            self._redraw_line()

        elif key in (KeyCodes.ENTER, "\n"):
            # Submit command
            command = self.current_line
            self._finish_input()
            return command

        elif key == KeyCodes.BACKSPACE:
            # Backspace
            if self.cursor_pos > 0:
                self.current_line = (
                    self.current_line[: self.cursor_pos - 1]
                    + self.current_line[self.cursor_pos :]
                )
                self.cursor_pos -= 1
                self._redraw_line()

        elif key == KeyCodes.TAB:
            # Tab completion
            self._handle_tab_completion()

        # Handle arrow keys
        elif key == KeyCodes.ARROW_UP:
            self._handle_history_up()

        elif key == KeyCodes.ARROW_DOWN:
            self._handle_history_down()

        elif key == KeyCodes.ARROW_LEFT:
            # Move cursor left
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
                self._update_cursor()

        elif key == KeyCodes.ARROW_RIGHT:
            # Move cursor right
            if self.cursor_pos < len(self.current_line):
                self.cursor_pos += 1
                self._update_cursor()

        elif key == KeyCodes.HOME:
            # Move to beginning of line
            self.cursor_pos = 0
            self._update_cursor()

        elif key == KeyCodes.END:
            # Move to end of line
            self.cursor_pos = len(self.current_line)
            self._update_cursor()

        # Handle printable characters
        elif len(key) == 1 and key.isprintable():
            # Insert character at cursor
            self.current_line = (
                self.current_line[: self.cursor_pos]
                + key
                + self.current_line[self.cursor_pos :]
            )
            self.cursor_pos += 1
            self._redraw_line()

        return None  # Continue input

    def _handle_history_up(self) -> None:
        """Handle up arrow key - previous command in history."""
        if self.in_search_mode:
            return

        prev_command = self.history_manager.get_previous_command()
        if prev_command is not None:
            self.current_line = prev_command
            self.cursor_pos = len(self.current_line)
            self._redraw_line()

    def _handle_history_down(self) -> None:
        """Handle down arrow key - next command in history."""
        if self.in_search_mode:
            return

        next_command = self.history_manager.get_next_command()
        if next_command is not None:
            self.current_line = next_command
            self.cursor_pos = len(self.current_line)
            self._redraw_line()

    def _handle_tab_completion(self) -> None:
        """Handle tab completion."""
        if not self.completion_callback:
            return

        try:
            # Get completion text (word before cursor)
            words = self.current_line[: self.cursor_pos].split()
            if words:
                current_word = words[-1]
            else:
                current_word = ""

            # Get completions
            completions = self.completion_callback(current_word)

            if not completions:
                return

            if len(completions) == 1:
                # Single completion - insert it
                completion = completions[0]
                if words:
                    # Replace last word
                    prefix = self.current_line[: self.cursor_pos - len(current_word)]
                    suffix = self.current_line[self.cursor_pos :]
                    self.current_line = prefix + completion + suffix
                    self.cursor_pos = len(prefix + completion)
                else:
                    # Insert at cursor
                    self.current_line = (
                        completion + self.current_line[self.cursor_pos :]
                    )
                    self.cursor_pos = len(completion)

                self._redraw_line()
            else:
                # Multiple completions - show them
                self._show_completions(completions)

        except Exception as e:
            logger.debug(f"Tab completion error: {e}")

    def _start_search_mode(self) -> None:
        """Start reverse search mode."""
        self.in_search_mode = True
        self.search_query = ""
        # TODO: Implement search mode display
        # For now, just exit search mode
        self.in_search_mode = False

    def _show_completions(self, completions: List[str]) -> None:
        """Show available completions."""
        # Move to new line
        sys.stdout.write("\n")

        # Display completions in columns
        max_width = max(len(c) for c in completions) if completions else 0
        term_width = os.get_terminal_size().columns
        cols = max(1, term_width // (max_width + 2))

        for i, completion in enumerate(completions):
            if i > 0 and i % cols == 0:
                sys.stdout.write("\n")
            sys.stdout.write(f"{completion:<{max_width + 2}}")

        sys.stdout.write("\n")
        sys.stdout.flush()

        # Redraw prompt and current line
        self._display_prompt()
        self._redraw_line()

    def _display_prompt(self) -> None:
        """Display the input prompt."""
        sys.stdout.write(self.prompt)
        sys.stdout.flush()

    def _redraw_line(self) -> None:
        """Redraw the current input line."""
        # Clear from cursor to end of line
        sys.stdout.write("\033[K")
        # Write current line
        sys.stdout.write(self.current_line)
        # Move cursor to correct position
        if self.cursor_pos < len(self.current_line):
            sys.stdout.write(f"\033[{len(self.current_line) - self.cursor_pos}D")
        sys.stdout.flush()

    def _update_cursor(self) -> None:
        """Update cursor position without redrawing line."""
        # This is a simplified version - in practice you'd calculate
        # the exact cursor movement needed
        self._redraw_line()

    def _finish_input(self) -> None:
        """Finish input and prepare for next line."""
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Reset state
        self.current_line = ""
        self.cursor_pos = 0
        self.in_search_mode = False
        self.search_query = ""


def create_completion_callback(engine) -> Callable[[str], List[str]]:
    """
    Create a tab completion callback for Omnimancer commands.

    Args:
        engine: Omnimancer engine instance

    Returns:
        Completion callback function
    """

    def completion_callback(text: str) -> List[str]:
        """Generate completions for the given text."""
        completions = []

        # Slash command completions
        if text.startswith("/"):
            from ..cli.commands import SlashCommand

            slash_commands = [cmd.value for cmd in SlashCommand]
            completions.extend([cmd for cmd in slash_commands if cmd.startswith(text)])

        # File path completions
        elif "/" in text or text.startswith(".") or text.startswith("~"):
            try:
                import glob

                pattern = text + "*"
                matches = glob.glob(pattern)
                # Return only the basename for display
                completions.extend(
                    [
                        (
                            os.path.basename(match)
                            if os.path.isfile(match)
                            else os.path.basename(match) + "/"
                        )
                        for match in matches[:20]
                    ]
                )  # Limit results
            except Exception:
                pass

        # Provider name completions (if engine available)
        try:
            if hasattr(engine, "config_manager") and engine.config_manager:
                provider_names = list(engine.config_manager.providers.keys())
                completions.extend(
                    [name for name in provider_names if name.startswith(text)]
                )
        except Exception:
            pass

        return sorted(list(set(completions)))

    return completion_callback
