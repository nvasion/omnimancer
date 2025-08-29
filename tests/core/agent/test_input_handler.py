"""
Tests for the Interactive Input Handler System.

This module tests keyboard input handling, navigation controls,
interactive features, and integration with the approval dialog.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from omnimancer.core.agent.input_handler import (
    InteractiveInputHandler,
    InputMode,
    KeyAction,
    KeyBinding,
    InputState,
    create_input_handler,
    create_custom_input_handler,
    MINIMAL_KEY_BINDINGS,
    VIM_STYLE_KEY_BINDINGS,
    ARROW_KEY_BINDINGS,
)


class TestKeyBinding:
    """Test KeyBinding dataclass."""

    def test_key_binding_creation(self):
        """Test creating key bindings."""
        binding = KeyBinding(
            key="y",
            action=KeyAction.APPROVE,
            description="Approve operation",
            mode=InputMode.NORMAL,
        )

        assert binding.key == "y"
        assert binding.action == KeyAction.APPROVE
        assert binding.description == "Approve operation"
        assert binding.mode == InputMode.NORMAL
        assert binding.requires_modifier is False
        assert binding.modifier is None

    def test_key_binding_with_modifier(self):
        """Test key binding with modifier."""
        binding = KeyBinding(
            key="s",
            action=KeyAction.SEARCH,
            description="Search with Ctrl",
            requires_modifier=True,
            modifier="ctrl",
        )

        assert binding.requires_modifier is True
        assert binding.modifier == "ctrl"


class TestInputState:
    """Test InputState functionality."""

    def test_default_input_state(self):
        """Test default input state."""
        state = InputState()

        assert state.mode == InputMode.NORMAL
        assert state.current_section == "diff"
        assert state.scroll_position == 0
        assert state.zoom_level == 1.0
        assert state.search_query == ""
        assert state.help_visible is False
        assert state.diff_expanded is False
        assert state.details_expanded is False
        assert state.last_input is None
        assert state.input_history == []

    def test_input_state_modifications(self):
        """Test modifying input state."""
        state = InputState()

        state.mode = InputMode.NAVIGATION
        state.current_section = "details"
        state.scroll_position = 5
        state.help_visible = True

        assert state.mode == InputMode.NAVIGATION
        assert state.current_section == "details"
        assert state.scroll_position == 5
        assert state.help_visible is True


class TestInteractiveInputHandler:
    """Test InteractiveInputHandler functionality."""

    def test_handler_initialization(self):
        """Test input handler initialization."""
        handler = InteractiveInputHandler()

        assert handler.timeout_seconds is None
        assert isinstance(handler.state, InputState)
        assert len(handler.key_bindings) > 0
        assert len(handler.action_handlers) > 0
        assert handler.terminal_setup is False

    def test_handler_with_timeout(self):
        """Test handler with timeout."""
        handler = InteractiveInputHandler(timeout_seconds=30)

        assert handler.timeout_seconds == 30

    def test_callback_setting(self):
        """Test setting callbacks."""
        handler = InteractiveInputHandler()

        approval_callback = AsyncMock()
        denial_callback = AsyncMock()
        quit_callback = AsyncMock()
        display_callback = AsyncMock()

        handler.set_callbacks(
            approval_callback=approval_callback,
            denial_callback=denial_callback,
            quit_callback=quit_callback,
            display_update_callback=display_callback,
        )

        assert handler.approval_callback == approval_callback
        assert handler.denial_callback == denial_callback
        assert handler.quit_callback == quit_callback
        assert handler.display_update_callback == display_callback

    def test_key_binding_management(self):
        """Test adding and removing key bindings."""
        handler = InteractiveInputHandler()
        original_count = len(handler.key_bindings)

        # Add new binding
        handler.add_key_binding("x", KeyAction.EXPORT, "Export data")
        assert len(handler.key_bindings) == original_count + 1
        assert handler.key_bindings["x"].action == KeyAction.EXPORT

        # Remove binding
        handler.remove_key_binding("x")
        assert len(handler.key_bindings) == original_count
        assert "x" not in handler.key_bindings

    def test_get_key_bindings_by_mode(self):
        """Test filtering key bindings by mode."""
        handler = InteractiveInputHandler()

        # Add some mode-specific bindings
        handler.add_key_binding("s", KeyAction.SEARCH, "Search", InputMode.SEARCH)
        handler.add_key_binding("h", KeyAction.HELP, "Help", InputMode.HELP)

        normal_bindings = handler.get_key_bindings_by_mode(InputMode.NORMAL)
        search_bindings = handler.get_key_bindings_by_mode(InputMode.SEARCH)
        help_bindings = handler.get_key_bindings_by_mode(InputMode.HELP)

        # Most bindings should be normal mode
        assert len(normal_bindings) > len(search_bindings)
        assert len(normal_bindings) > len(help_bindings)

        # Check specific mode bindings exist
        assert any(b.action == KeyAction.SEARCH for b in search_bindings.values())
        assert any(b.action == KeyAction.HELP for b in help_bindings.values())

    @pytest.mark.asyncio
    async def test_handle_key_approve(self):
        """Test handling approval key."""
        handler = InteractiveInputHandler()
        approval_mock = AsyncMock(return_value=True)
        handler.set_callbacks(approval_callback=approval_mock)

        action = await handler.handle_key("y")

        assert action == KeyAction.APPROVE
        approval_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_key_deny(self):
        """Test handling denial key."""
        handler = InteractiveInputHandler()
        denial_mock = AsyncMock(return_value=True)
        handler.set_callbacks(denial_callback=denial_mock)

        action = await handler.handle_key("n")

        assert action == KeyAction.DENY
        denial_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_key_quit(self):
        """Test handling quit key."""
        handler = InteractiveInputHandler()
        quit_mock = AsyncMock(return_value=True)
        handler.set_callbacks(quit_callback=quit_mock)

        action = await handler.handle_key("q")

        assert action == KeyAction.QUIT
        quit_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_key_navigation(self):
        """Test handling navigation keys."""
        handler = InteractiveInputHandler()
        initial_position = handler.state.scroll_position

        # Test navigation up
        await handler.handle_key("k")
        # Position should stay at 0 (can't go negative)
        assert handler.state.scroll_position == initial_position

        # Move down first, then up
        await handler.handle_key("j")
        assert handler.state.scroll_position == initial_position + 1

        await handler.handle_key("k")
        assert handler.state.scroll_position == initial_position

    @pytest.mark.asyncio
    async def test_handle_key_unknown(self):
        """Test handling unknown keys."""
        handler = InteractiveInputHandler()

        action = await handler.handle_key("z")  # Not bound by default

        assert action is None

    @pytest.mark.asyncio
    async def test_handle_key_help_toggle(self):
        """Test help toggle functionality."""
        handler = InteractiveInputHandler()

        assert handler.state.help_visible is False
        assert handler.state.mode == InputMode.NORMAL

        # Toggle help on
        await handler.handle_key("?")
        assert handler.state.help_visible is True
        assert handler.state.mode == InputMode.HELP

        # Toggle help off
        await handler.handle_key("?")
        assert handler.state.help_visible is False
        assert handler.state.mode == InputMode.NORMAL

    @pytest.mark.asyncio
    async def test_handle_key_section_navigation(self):
        """Test section navigation."""
        handler = InteractiveInputHandler()

        assert handler.state.current_section == "diff"

        # Navigate right
        await handler.handle_key("l")
        assert handler.state.current_section == "details"

        # Navigate left
        await handler.handle_key("h")
        assert handler.state.current_section == "diff"

    @pytest.mark.asyncio
    async def test_handle_key_page_navigation(self):
        """Test page navigation."""
        handler = InteractiveInputHandler()

        # Page down
        await handler.handle_key("\x1b[6~")  # Page Down escape sequence
        assert handler.state.scroll_position == 10

        # Page up
        await handler.handle_key("\x1b[5~")  # Page Up escape sequence
        assert handler.state.scroll_position == 0

        # Home
        handler.state.scroll_position = 50
        await handler.handle_key("\x1b[H")  # Home escape sequence
        assert handler.state.scroll_position == 0

    @pytest.mark.asyncio
    async def test_handle_key_zoom_controls(self):
        """Test zoom in/out functionality."""
        handler = InteractiveInputHandler()

        assert handler.state.zoom_level == 1.0

        # Zoom in
        await handler.handle_key("+")
        assert handler.state.zoom_level == 1.1

        # Zoom out
        await handler.handle_key("-")
        assert handler.state.zoom_level == 1.0

        # Test zoom limits
        for _ in range(20):  # Try to zoom in beyond limit
            await handler.handle_key("+")
        assert handler.state.zoom_level == 2.0  # Should cap at 2.0

        for _ in range(30):  # Try to zoom out beyond limit
            await handler.handle_key("-")
        assert handler.state.zoom_level == 0.5  # Should cap at 0.5

    @pytest.mark.asyncio
    async def test_handle_key_display_toggles(self):
        """Test display toggle functionality."""
        handler = InteractiveInputHandler()

        assert handler.state.diff_expanded is False
        assert handler.state.details_expanded is False

        # Toggle diff
        await handler.handle_key("d")
        assert handler.state.diff_expanded is True

        # Toggle details
        await handler.handle_key("D")
        assert handler.state.details_expanded is True

        # Toggle again
        await handler.handle_key("d")
        assert handler.state.diff_expanded is False

    def test_get_help_text(self):
        """Test help text generation."""
        handler = InteractiveInputHandler()

        help_text = handler.get_help_text()

        assert "Help - Normal Mode" in help_text
        assert "Approval Actions:" in help_text
        assert "Navigation:" in help_text
        assert "y" in help_text  # Should contain approval key
        assert "n" in help_text  # Should contain deny key
        assert "Approve" in help_text
        assert "Deny" in help_text

    def test_get_status_info(self):
        """Test status information retrieval."""
        handler = InteractiveInputHandler()
        handler.state.scroll_position = 5
        handler.state.help_visible = True
        handler.state.last_input = "k"

        status = handler.get_status_info()

        assert status["mode"] == "normal"
        assert status["scroll_position"] == 5
        assert status["help_visible"] is True
        assert status["last_input"] == "k"
        assert "zoom_level" in status
        assert "current_section" in status
        assert "input_count" in status

    @pytest.mark.asyncio
    @patch("sys.stdin")
    @patch("termios.tcgetattr")
    @patch("termios.tcsetattr")
    @patch("tty.setraw")
    async def test_terminal_setup_and_restore(
        self, mock_setraw, mock_tcsetattr, mock_tcgetattr, mock_stdin
    ):
        """Test terminal setup and restoration."""
        handler = InteractiveInputHandler()
        mock_tcgetattr.return_value = ["mock", "settings"]

        # Test setup
        await handler.setup_terminal()
        assert handler.terminal_setup is True
        assert handler.original_terminal_settings == ["mock", "settings"]
        mock_setraw.assert_called_once()

        # Test restore
        await handler.restore_terminal()
        assert handler.terminal_setup is False
        assert handler.original_terminal_settings is None
        mock_tcsetattr.assert_called_once()

    @pytest.mark.asyncio
    @patch("sys.stdin")
    @patch("termios.tcgetattr", side_effect=OSError("No terminal"))
    async def test_terminal_setup_fallback(self, mock_tcgetattr, mock_stdin):
        """Test terminal setup fallback when raw mode fails."""
        handler = InteractiveInputHandler()

        # Should not raise exception, just log warning
        await handler.setup_terminal()
        assert handler.terminal_setup is False
        assert handler.original_terminal_settings is None


class TestActionHandlers:
    """Test individual action handlers."""

    @pytest.mark.asyncio
    async def test_approve_handler(self):
        """Test approval handler."""
        handler = InteractiveInputHandler()

        result = await handler._handle_approve()
        assert result is True

    @pytest.mark.asyncio
    async def test_deny_handler(self):
        """Test denial handler."""
        handler = InteractiveInputHandler()

        result = await handler._handle_deny()
        assert result is True

    @pytest.mark.asyncio
    async def test_quit_handler(self):
        """Test quit handler."""
        handler = InteractiveInputHandler()

        result = await handler._handle_quit()
        assert result is True

    @pytest.mark.asyncio
    async def test_navigation_handlers(self):
        """Test navigation handlers."""
        handler = InteractiveInputHandler()
        initial_position = handler.state.scroll_position

        # Test up navigation
        handler.state.scroll_position = 5
        result = await handler._handle_navigate_up()
        assert result is True
        assert handler.state.scroll_position == 4

        # Test down navigation
        result = await handler._handle_navigate_down()
        assert result is True
        assert handler.state.scroll_position == 5

    @pytest.mark.asyncio
    async def test_section_navigation_handlers(self):
        """Test section navigation handlers."""
        handler = InteractiveInputHandler()
        handler.state.current_section = "details"

        # Test left navigation
        result = await handler._handle_navigate_left()
        assert result is True
        assert handler.state.current_section == "diff"

        # Test right navigation
        result = await handler._handle_navigate_right()
        assert result is True
        assert handler.state.current_section == "details"

    @pytest.mark.asyncio
    async def test_page_navigation_handlers(self):
        """Test page navigation handlers."""
        handler = InteractiveInputHandler()
        handler.state.scroll_position = 15

        # Test page up
        result = await handler._handle_page_up()
        assert result is True
        assert handler.state.scroll_position == 5

        # Test page down
        result = await handler._handle_page_down()
        assert result is True
        assert handler.state.scroll_position == 15

        # Test home
        result = await handler._handle_home()
        assert result is True
        assert handler.state.scroll_position == 0

    @pytest.mark.asyncio
    async def test_toggle_handlers(self):
        """Test toggle handlers."""
        handler = InteractiveInputHandler()

        # Test diff toggle
        assert handler.state.diff_expanded is False
        result = await handler._handle_toggle_diff()
        assert result is True
        assert handler.state.diff_expanded is True

        # Test details toggle
        assert handler.state.details_expanded is False
        result = await handler._handle_toggle_details()
        assert result is True
        assert handler.state.details_expanded is True

    @pytest.mark.asyncio
    async def test_zoom_handlers(self):
        """Test zoom handlers."""
        handler = InteractiveInputHandler()

        # Test zoom in
        result = await handler._handle_zoom_in()
        assert result is True
        assert handler.state.zoom_level == 1.1

        # Test zoom out
        result = await handler._handle_zoom_out()
        assert result is True
        assert handler.state.zoom_level == 1.0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_input_handler(self):
        """Test default input handler creation."""
        handler = create_input_handler()

        assert isinstance(handler, InteractiveInputHandler)
        assert handler.timeout_seconds is None

    def test_create_input_handler_with_timeout(self):
        """Test input handler creation with timeout."""
        handler = create_input_handler(timeout_seconds=30)

        assert handler.timeout_seconds == 30

    def test_create_custom_input_handler(self):
        """Test custom input handler creation."""
        custom_bindings = {
            "a": KeyBinding("a", KeyAction.APPROVE, "Approve"),
            "r": KeyBinding("r", KeyAction.DENY, "Reject"),
        }

        handler = create_custom_input_handler(key_bindings=custom_bindings)

        assert len(handler.key_bindings) == 2
        assert "a" in handler.key_bindings
        assert "r" in handler.key_bindings
        assert handler.key_bindings["a"].action == KeyAction.APPROVE
        assert handler.key_bindings["r"].action == KeyAction.DENY


class TestPredefinedBindings:
    """Test predefined key binding sets."""

    def test_minimal_key_bindings(self):
        """Test minimal key bindings."""
        bindings = MINIMAL_KEY_BINDINGS

        assert len(bindings) == 4  # y, n, q, ?
        assert bindings["y"].action == KeyAction.APPROVE
        assert bindings["n"].action == KeyAction.DENY
        assert bindings["q"].action == KeyAction.QUIT
        assert bindings["?"].action == KeyAction.HELP

    def test_vim_style_key_bindings(self):
        """Test Vim-style key bindings."""
        bindings = VIM_STYLE_KEY_BINDINGS

        # Should have navigation keys
        assert bindings["j"].action == KeyAction.NAVIGATE_DOWN
        assert bindings["k"].action == KeyAction.NAVIGATE_UP
        assert bindings["h"].action == KeyAction.NAVIGATE_LEFT
        assert bindings["l"].action == KeyAction.NAVIGATE_RIGHT

        # Should have Vim-style home/end
        assert bindings["g"].action == KeyAction.HOME
        assert bindings["G"].action == KeyAction.END

        # Should have diff and search
        assert bindings["d"].action == KeyAction.TOGGLE_DIFF
        assert bindings["/"].action == KeyAction.SEARCH

    def test_arrow_key_bindings(self):
        """Test arrow key bindings."""
        bindings = ARROW_KEY_BINDINGS

        # Should have arrow key sequences
        assert bindings["\x1b[A"].action == KeyAction.NAVIGATE_UP
        assert bindings["\x1b[B"].action == KeyAction.NAVIGATE_DOWN
        assert bindings["\x1b[C"].action == KeyAction.NAVIGATE_RIGHT
        assert bindings["\x1b[D"].action == KeyAction.NAVIGATE_LEFT

        # Should have page navigation
        assert bindings["\x1b[5~"].action == KeyAction.PAGE_UP
        assert bindings["\x1b[6~"].action == KeyAction.PAGE_DOWN


class TestInputIntegration:
    """Test input handler integration scenarios."""

    @pytest.mark.asyncio
    async def test_input_history_tracking(self):
        """Test that input history is tracked correctly."""
        handler = InteractiveInputHandler()

        # Simulate some key presses
        await handler.handle_key("k")
        await handler.handle_key("j")
        await handler.handle_key("?")

        assert len(handler.state.input_history) == 3
        assert handler.state.input_history[-1] == "?"
        assert handler.state.last_input == "?"

    @pytest.mark.asyncio
    async def test_mode_switching(self):
        """Test switching between input modes."""
        handler = InteractiveInputHandler()

        assert handler.state.mode == InputMode.NORMAL

        # Switch to help mode
        await handler.handle_key("?")
        assert handler.state.mode == InputMode.HELP

        # Switch back to normal
        await handler.handle_key("?")
        assert handler.state.mode == InputMode.NORMAL

    @pytest.mark.asyncio
    async def test_callback_integration(self):
        """Test callback integration with display updates."""
        handler = InteractiveInputHandler()
        display_callback = AsyncMock()
        handler.set_callbacks(display_update_callback=display_callback)

        # Simulate key that triggers display update
        await handler.handle_key("d")  # Toggle diff

        # Note: This would be called in the full input loop
        # Here we just verify the callback is set correctly
        assert handler.display_update_callback == display_callback

    @pytest.mark.asyncio
    async def test_error_handling_in_handlers(self):
        """Test error handling in action handlers."""
        handler = InteractiveInputHandler()

        # Mock a handler that raises an exception
        async def failing_handler():
            raise ValueError("Test error")

        handler.action_handlers[KeyAction.APPROVE] = failing_handler

        # Should not raise exception, just return None
        result = await handler.handle_key("y")
        assert result is None

    def test_complex_help_text_generation(self):
        """Test help text generation with multiple modes."""
        handler = InteractiveInputHandler()

        # Add some mode-specific bindings
        handler.add_key_binding("s", KeyAction.SEARCH, "Search", InputMode.SEARCH)
        handler.state.mode = InputMode.SEARCH

        help_text = handler.get_help_text()

        assert "Search Mode" in help_text
        assert "Search" in help_text
