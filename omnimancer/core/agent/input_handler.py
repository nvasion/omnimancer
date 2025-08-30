"""
Interactive Input Handler for Omnimancer Agent Approval System.

This module provides keyboard input handling, navigation controls, and
interactive features for the approval dialog system.
"""

import asyncio
import logging
import sys
import termios
import tty
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class InputMode(Enum):
    """Input modes for the interactive handler."""

    NORMAL = "normal"
    NAVIGATION = "navigation"
    SEARCH = "search"
    HELP = "help"


class KeyAction(Enum):
    """Available key actions."""

    APPROVE = "approve"
    DENY = "deny"
    QUIT = "quit"
    HELP = "help"
    NAVIGATE_UP = "navigate_up"
    NAVIGATE_DOWN = "navigate_down"
    NAVIGATE_LEFT = "navigate_left"
    NAVIGATE_RIGHT = "navigate_right"
    PAGE_UP = "page_up"
    PAGE_DOWN = "page_down"
    HOME = "home"
    END = "end"
    TOGGLE_DIFF = "toggle_diff"
    TOGGLE_DETAILS = "toggle_details"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    SEARCH = "search"
    COPY = "copy"
    EXPORT = "export"


@dataclass
class KeyBinding:
    """Represents a key binding configuration."""

    key: str
    action: KeyAction
    description: str
    mode: InputMode = InputMode.NORMAL
    requires_modifier: bool = False
    modifier: Optional[str] = None


@dataclass
class InputState:
    """Current state of input handling."""

    mode: InputMode = InputMode.NORMAL
    current_section: str = "diff"
    scroll_position: int = 0
    zoom_level: float = 1.0
    search_query: str = ""
    help_visible: bool = False
    diff_expanded: bool = False
    details_expanded: bool = False
    last_input: Optional[str] = None
    input_history: List[str] = field(default_factory=list)


class InteractiveInputHandler:
    """
    Handles keyboard input and interactive controls for approval dialogs.

    Provides comprehensive keyboard shortcuts, navigation controls,
    and interactive features for the approval interface.
    """

    def __init__(self, timeout_seconds: Optional[int] = None):
        """
        Initialize interactive input handler.

        Args:
            timeout_seconds: Optional timeout for input operations
        """
        self.timeout_seconds = timeout_seconds
        self.state = InputState()

        # Default key bindings
        self.key_bindings: Dict[str, KeyBinding] = {
            # Approval actions
            "y": KeyBinding("y", KeyAction.APPROVE, "Approve operation"),
            "Y": KeyBinding("Y", KeyAction.APPROVE, "Approve operation"),
            "n": KeyBinding("n", KeyAction.DENY, "Deny operation"),
            "N": KeyBinding("N", KeyAction.DENY, "Deny operation"),
            "q": KeyBinding("q", KeyAction.QUIT, "Quit/Cancel"),
            "Q": KeyBinding("Q", KeyAction.QUIT, "Quit/Cancel"),
            # Navigation
            "k": KeyBinding("k", KeyAction.NAVIGATE_UP, "Navigate up"),
            "j": KeyBinding("j", KeyAction.NAVIGATE_DOWN, "Navigate down"),
            "h": KeyBinding("h", KeyAction.NAVIGATE_LEFT, "Navigate left"),
            "l": KeyBinding("l", KeyAction.NAVIGATE_RIGHT, "Navigate right"),
            "\x1b[A": KeyBinding("\x1b[A", KeyAction.NAVIGATE_UP, "Up arrow"),
            "\x1b[B": KeyBinding("\x1b[B", KeyAction.NAVIGATE_DOWN, "Down arrow"),
            "\x1b[C": KeyBinding("\x1b[C", KeyAction.NAVIGATE_RIGHT, "Right arrow"),
            "\x1b[D": KeyBinding("\x1b[D", KeyAction.NAVIGATE_LEFT, "Left arrow"),
            # Page navigation
            "\x1b[5~": KeyBinding("\x1b[5~", KeyAction.PAGE_UP, "Page up"),
            "\x1b[6~": KeyBinding("\x1b[6~", KeyAction.PAGE_DOWN, "Page down"),
            "\x1b[H": KeyBinding("\x1b[H", KeyAction.HOME, "Home"),
            "\x1b[F": KeyBinding("\x1b[F", KeyAction.END, "End"),
            # Display controls
            "d": KeyBinding("d", KeyAction.TOGGLE_DIFF, "Toggle diff view"),
            "D": KeyBinding("D", KeyAction.TOGGLE_DETAILS, "Toggle details view"),
            "+": KeyBinding("+", KeyAction.ZOOM_IN, "Zoom in"),
            "=": KeyBinding("=", KeyAction.ZOOM_IN, "Zoom in"),
            "-": KeyBinding("-", KeyAction.ZOOM_OUT, "Zoom out"),
            "_": KeyBinding("_", KeyAction.ZOOM_OUT, "Zoom out"),
            # Utility functions
            "?": KeyBinding("?", KeyAction.HELP, "Show help"),
            "/": KeyBinding("/", KeyAction.SEARCH, "Search"),
            "c": KeyBinding("c", KeyAction.COPY, "Copy content"),
            "e": KeyBinding("e", KeyAction.EXPORT, "Export details"),
        }

        # Action handlers
        self.action_handlers: Dict[KeyAction, Callable] = {
            KeyAction.APPROVE: self._handle_approve,
            KeyAction.DENY: self._handle_deny,
            KeyAction.QUIT: self._handle_quit,
            KeyAction.HELP: self._handle_help,
            KeyAction.NAVIGATE_UP: self._handle_navigate_up,
            KeyAction.NAVIGATE_DOWN: self._handle_navigate_down,
            KeyAction.NAVIGATE_LEFT: self._handle_navigate_left,
            KeyAction.NAVIGATE_RIGHT: self._handle_navigate_right,
            KeyAction.PAGE_UP: self._handle_page_up,
            KeyAction.PAGE_DOWN: self._handle_page_down,
            KeyAction.HOME: self._handle_home,
            KeyAction.END: self._handle_end,
            KeyAction.TOGGLE_DIFF: self._handle_toggle_diff,
            KeyAction.TOGGLE_DETAILS: self._handle_toggle_details,
            KeyAction.ZOOM_IN: self._handle_zoom_in,
            KeyAction.ZOOM_OUT: self._handle_zoom_out,
            KeyAction.SEARCH: self._handle_search,
            KeyAction.COPY: self._handle_copy,
            KeyAction.EXPORT: self._handle_export,
        }

        # Callbacks for external integration
        self.approval_callback: Optional[Callable] = None
        self.denial_callback: Optional[Callable] = None
        self.quit_callback: Optional[Callable] = None
        self.display_update_callback: Optional[Callable] = None

        # Terminal state management
        self.original_terminal_settings: Optional[List] = None
        self.terminal_setup: bool = False

    def set_callbacks(
        self,
        approval_callback: Optional[Callable] = None,
        denial_callback: Optional[Callable] = None,
        quit_callback: Optional[Callable] = None,
        display_update_callback: Optional[Callable] = None,
    ):
        """Set callback functions for various actions."""
        if approval_callback:
            self.approval_callback = approval_callback
        if denial_callback:
            self.denial_callback = denial_callback
        if quit_callback:
            self.quit_callback = quit_callback
        if display_update_callback:
            self.display_update_callback = display_update_callback

    def add_key_binding(
        self,
        key: str,
        action: KeyAction,
        description: str,
        mode: InputMode = InputMode.NORMAL,
    ):
        """Add or modify a key binding."""
        self.key_bindings[key] = KeyBinding(key, action, description, mode)

    def remove_key_binding(self, key: str):
        """Remove a key binding."""
        if key in self.key_bindings:
            del self.key_bindings[key]

    def get_key_bindings_by_mode(self, mode: InputMode) -> Dict[str, KeyBinding]:
        """Get key bindings for a specific mode."""
        return {k: v for k, v in self.key_bindings.items() if v.mode == mode}

    async def setup_terminal(self):
        """Setup terminal for raw input mode."""
        if self.terminal_setup:
            return

        try:
            # Save original terminal settings
            self.original_terminal_settings = termios.tcgetattr(sys.stdin)

            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
            self.terminal_setup = True

        except (termios.error, OSError) as e:
            logger.warning(f"Could not setup terminal for raw input: {e}")
            # Fall back to line-based input if raw mode fails

    async def restore_terminal(self):
        """Restore original terminal settings."""
        if not self.terminal_setup or not self.original_terminal_settings:
            return

        try:
            termios.tcsetattr(
                sys.stdin, termios.TCSADRAIN, self.original_terminal_settings
            )
            self.terminal_setup = False
            self.original_terminal_settings = None
        except (termios.error, OSError) as e:
            logger.warning(f"Could not restore terminal settings: {e}")

    async def read_key(self) -> Optional[str]:
        """
        Read a single key from input.

        Returns:
            The key pressed, or None if timeout/error
        """
        try:
            if not self.terminal_setup:
                # Fallback to line input
                if self.timeout_seconds:
                    try:
                        line = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(None, input),
                            timeout=self.timeout_seconds,
                        )
                        return line.strip().lower()
                    except asyncio.TimeoutError:
                        return None
                else:
                    return input().strip().lower()

            # Raw character input
            def _read_char():
                char = sys.stdin.read(1)
                # Handle escape sequences
                if char == "\x1b":
                    char += sys.stdin.read(2)  # Read next 2 chars for arrow keys
                    if char.endswith("["):
                        char += sys.stdin.read(1)  # Read the final character
                return char

            if self.timeout_seconds:
                try:
                    key = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, _read_char),
                        timeout=self.timeout_seconds,
                    )
                    return key
                except asyncio.TimeoutError:
                    return None
            else:
                return await asyncio.get_event_loop().run_in_executor(None, _read_char)

        except (KeyboardInterrupt, EOFError):
            return "q"  # Treat as quit
        except Exception as e:
            logger.error(f"Error reading key input: {e}")
            return None

    async def handle_input_loop(self) -> Optional[KeyAction]:
        """
        Main input handling loop.

        Returns:
            The final action that should terminate the loop
        """
        await self.setup_terminal()

        try:
            while True:
                key = await self.read_key()

                if key is None:
                    # Timeout or error - return quit
                    return KeyAction.QUIT

                # Handle the key
                action = await self.handle_key(key)

                # Check if this action should terminate the loop
                if action in [
                    KeyAction.APPROVE,
                    KeyAction.DENY,
                    KeyAction.QUIT,
                ]:
                    return action

                # Update display if callback is available
                if self.display_update_callback:
                    await self.display_update_callback(self.state)

        finally:
            await self.restore_terminal()

    async def handle_key(self, key: str) -> Optional[KeyAction]:
        """
        Handle a single key press.

        Args:
            key: The key that was pressed

        Returns:
            The action that was triggered, if any
        """
        # Record input history
        self.state.last_input = key
        self.state.input_history.append(key)

        # Keep history limited
        if len(self.state.input_history) > 100:
            self.state.input_history = self.state.input_history[-100:]

        # Look up key binding
        binding = self.key_bindings.get(key)

        if not binding:
            # Unknown key - ignore or show help
            if key not in ["\r", "\n", " "]:  # Ignore common non-action keys
                logger.debug(f"Unknown key pressed: {repr(key)}")
            return None

        # Check if binding is valid for current mode
        if binding.mode != InputMode.NORMAL and binding.mode != self.state.mode:
            return None

        # Get the action handler
        handler = self.action_handlers.get(binding.action)

        if not handler:
            logger.warning(f"No handler for action: {binding.action}")
            return None

        # Execute the handler
        try:
            await handler()
            return binding.action
        except Exception as e:
            logger.error(f"Error executing action {binding.action}: {e}")
            return None

    # Action handlers
    async def _handle_approve(self) -> bool:
        """Handle approval action."""
        if self.approval_callback:
            return await self.approval_callback()
        return True

    async def _handle_deny(self) -> bool:
        """Handle denial action."""
        if self.denial_callback:
            return await self.denial_callback()
        return True

    async def _handle_quit(self) -> bool:
        """Handle quit action."""
        if self.quit_callback:
            return await self.quit_callback()
        return True

    async def _handle_help(self) -> bool:
        """Handle help display toggle."""
        self.state.help_visible = not self.state.help_visible
        self.state.mode = (
            InputMode.HELP if self.state.help_visible else InputMode.NORMAL
        )
        return True

    async def _handle_navigate_up(self) -> bool:
        """Handle upward navigation."""
        if self.state.scroll_position > 0:
            self.state.scroll_position -= 1
        return True

    async def _handle_navigate_down(self) -> bool:
        """Handle downward navigation."""
        self.state.scroll_position += 1
        return True

    async def _handle_navigate_left(self) -> bool:
        """Handle leftward navigation."""
        sections = ["diff", "details", "operation", "risk"]
        try:
            current_idx = sections.index(self.state.current_section)
            if current_idx > 0:
                self.state.current_section = sections[current_idx - 1]
        except ValueError:
            self.state.current_section = sections[0]
        return True

    async def _handle_navigate_right(self) -> bool:
        """Handle rightward navigation."""
        sections = ["diff", "details", "operation", "risk"]
        try:
            current_idx = sections.index(self.state.current_section)
            if current_idx < len(sections) - 1:
                self.state.current_section = sections[current_idx + 1]
        except ValueError:
            self.state.current_section = sections[0]
        return True

    async def _handle_page_up(self) -> bool:
        """Handle page up navigation."""
        self.state.scroll_position = max(0, self.state.scroll_position - 10)
        return True

    async def _handle_page_down(self) -> bool:
        """Handle page down navigation."""
        self.state.scroll_position += 10
        return True

    async def _handle_home(self) -> bool:
        """Handle home navigation."""
        self.state.scroll_position = 0
        return True

    async def _handle_end(self) -> bool:
        """Handle end navigation."""
        # Would need access to content length to implement properly
        self.state.scroll_position = 1000  # Large number as approximation
        return True

    async def _handle_toggle_diff(self) -> bool:
        """Handle diff view toggle."""
        self.state.diff_expanded = not self.state.diff_expanded
        return True

    async def _handle_toggle_details(self) -> bool:
        """Handle details view toggle."""
        self.state.details_expanded = not self.state.details_expanded
        return True

    async def _handle_zoom_in(self) -> bool:
        """Handle zoom in."""
        self.state.zoom_level = min(2.0, self.state.zoom_level + 0.1)
        return True

    async def _handle_zoom_out(self) -> bool:
        """Handle zoom out."""
        self.state.zoom_level = max(0.5, self.state.zoom_level - 0.1)
        return True

    async def _handle_search(self) -> bool:
        """Handle search mode activation."""
        self.state.mode = InputMode.SEARCH
        # Would implement search input handling here
        return True

    async def _handle_copy(self) -> bool:
        """Handle copy to clipboard."""
        # Would implement clipboard functionality here
        logger.info("Copy functionality not yet implemented")
        return True

    async def _handle_export(self) -> bool:
        """Handle export functionality."""
        # Would implement export functionality here
        logger.info("Export functionality not yet implemented")
        return True

    def get_help_text(self) -> str:
        """Generate help text for current mode."""
        mode_bindings = self.get_key_bindings_by_mode(self.state.mode)

        help_lines = [
            f"=== Help - {self.state.mode.value.title()} Mode ===",
            "",
        ]

        # Group bindings by category
        categories = {
            "Approval Actions": [
                KeyAction.APPROVE,
                KeyAction.DENY,
                KeyAction.QUIT,
            ],
            "Navigation": [
                KeyAction.NAVIGATE_UP,
                KeyAction.NAVIGATE_DOWN,
                KeyAction.NAVIGATE_LEFT,
                KeyAction.NAVIGATE_RIGHT,
                KeyAction.PAGE_UP,
                KeyAction.PAGE_DOWN,
                KeyAction.HOME,
                KeyAction.END,
            ],
            "Display Control": [
                KeyAction.TOGGLE_DIFF,
                KeyAction.TOGGLE_DETAILS,
                KeyAction.ZOOM_IN,
                KeyAction.ZOOM_OUT,
            ],
            "Utility": [
                KeyAction.HELP,
                KeyAction.SEARCH,
                KeyAction.COPY,
                KeyAction.EXPORT,
            ],
        }

        for category, actions in categories.items():
            help_lines.append(f"{category}:")
            for key, binding in mode_bindings.items():
                if binding.action in actions:
                    help_lines.append(f"  {key:<8} - {binding.description}")
            help_lines.append("")

        help_lines.extend(
            ["Press '?' again to close help", "Press 'q' to quit the dialog"]
        )

        return "\n".join(help_lines)

    def get_status_info(self) -> Dict[str, Any]:
        """Get current input handler status information."""
        return {
            "mode": self.state.mode.value,
            "current_section": self.state.current_section,
            "scroll_position": self.state.scroll_position,
            "zoom_level": self.state.zoom_level,
            "help_visible": self.state.help_visible,
            "diff_expanded": self.state.diff_expanded,
            "details_expanded": self.state.details_expanded,
            "last_input": self.state.last_input,
            "input_count": len(self.state.input_history),
        }


# Utility functions


def create_input_handler(
    timeout_seconds: Optional[int] = None,
) -> InteractiveInputHandler:
    """Create an input handler with default configuration."""
    return InteractiveInputHandler(timeout_seconds=timeout_seconds)


def create_custom_input_handler(
    key_bindings: Optional[Dict[str, KeyBinding]] = None,
    timeout_seconds: Optional[int] = None,
) -> InteractiveInputHandler:
    """Create an input handler with custom key bindings."""
    handler = InteractiveInputHandler(timeout_seconds=timeout_seconds)

    if key_bindings:
        # Clear default bindings and add custom ones
        handler.key_bindings.clear()
        handler.key_bindings.update(key_bindings)

    return handler


# Predefined key binding sets

MINIMAL_KEY_BINDINGS = {
    "y": KeyBinding("y", KeyAction.APPROVE, "Approve"),
    "n": KeyBinding("n", KeyAction.DENY, "Deny"),
    "q": KeyBinding("q", KeyAction.QUIT, "Quit"),
    "?": KeyBinding("?", KeyAction.HELP, "Help"),
}

VIM_STYLE_KEY_BINDINGS = {
    "y": KeyBinding("y", KeyAction.APPROVE, "Approve (yes)"),
    "n": KeyBinding("n", KeyAction.DENY, "Deny (no)"),
    "q": KeyBinding("q", KeyAction.QUIT, "Quit"),
    "j": KeyBinding("j", KeyAction.NAVIGATE_DOWN, "Down"),
    "k": KeyBinding("k", KeyAction.NAVIGATE_UP, "Up"),
    "h": KeyBinding("h", KeyAction.NAVIGATE_LEFT, "Left"),
    "l": KeyBinding("l", KeyAction.NAVIGATE_RIGHT, "Right"),
    "g": KeyBinding("g", KeyAction.HOME, "Go to top"),
    "G": KeyBinding("G", KeyAction.END, "Go to bottom"),
    "d": KeyBinding("d", KeyAction.TOGGLE_DIFF, "Toggle diff"),
    "/": KeyBinding("/", KeyAction.SEARCH, "Search"),
    "?": KeyBinding("?", KeyAction.HELP, "Help"),
}

ARROW_KEY_BINDINGS = {
    "y": KeyBinding("y", KeyAction.APPROVE, "Approve"),
    "n": KeyBinding("n", KeyAction.DENY, "Deny"),
    "q": KeyBinding("q", KeyAction.QUIT, "Quit"),
    "\x1b[A": KeyBinding("\x1b[A", KeyAction.NAVIGATE_UP, "Up arrow"),
    "\x1b[B": KeyBinding("\x1b[B", KeyAction.NAVIGATE_DOWN, "Down arrow"),
    "\x1b[C": KeyBinding("\x1b[C", KeyAction.NAVIGATE_RIGHT, "Right arrow"),
    "\x1b[D": KeyBinding("\x1b[D", KeyAction.NAVIGATE_LEFT, "Left arrow"),
    "\x1b[5~": KeyBinding("\x1b[5~", KeyAction.PAGE_UP, "Page Up"),
    "\x1b[6~": KeyBinding("\x1b[6~", KeyAction.PAGE_DOWN, "Page Down"),
    "?": KeyBinding("?", KeyAction.HELP, "Help"),
}
