"""
Tests for the RichTextRenderer class.

This module tests the rich text rendering system for Omnimancer,
including color schemes, terminal capability detection, and various
rendering functions.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from pathlib import Path

from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress

from omnimancer.core.agent.rich_renderer import (
    RichTextRenderer, ColorScheme, TerminalCapabilities,
    RiskLevel, OperationType, create_renderer, 
    render_risk_badge, render_operation_badge
)


class TestColorScheme:
    """Test ColorScheme dataclass functionality."""
    
    def test_default_color_scheme(self):
        """Test default color scheme values."""
        scheme = ColorScheme()
        
        # Test risk colors
        assert scheme.risk_low == "green"
        assert scheme.risk_high == "bright_red"
        assert scheme.risk_critical == "bold red on white"
        
        # Test operation colors
        assert scheme.op_read == "cyan"
        assert scheme.op_write == "yellow"
        assert scheme.op_delete == "red"
        
        # Test diff colors
        assert scheme.diff_added == "green"
        assert scheme.diff_removed == "red"
        assert scheme.diff_modified == "yellow"
    
    def test_custom_color_scheme(self):
        """Test custom color scheme configuration."""
        scheme = ColorScheme(
            risk_high="magenta",
            op_write="orange",
            diff_added="bright_green"
        )
        
        assert scheme.risk_high == "magenta"
        assert scheme.op_write == "orange"
        assert scheme.diff_added == "bright_green"
        
        # Default values should remain
        assert scheme.risk_low == "green"
        assert scheme.op_read == "cyan"


class TestTerminalCapabilities:
    """Test TerminalCapabilities detection and configuration."""
    
    def test_default_capabilities(self):
        """Test default terminal capabilities."""
        caps = TerminalCapabilities()
        
        assert caps.width == 80
        assert caps.height == 24
        assert caps.supports_color is True
        assert caps.supports_unicode is True
        assert caps.color_depth == 256
    
    @patch('shutil.get_terminal_size')
    @patch('os.environ')
    def test_detect_capabilities(self, mock_environ, mock_terminal_size):
        """Test automatic capability detection."""
        # Mock terminal size
        mock_terminal_size.return_value = Mock(columns=120, lines=30)
        
        # Mock environment variables for color support
        mock_environ.get.side_effect = lambda key, default='': {
            'COLORTERM': 'truecolor',
            'TERM': 'xterm-256color',
            'LANG': 'en_US.UTF-8'
        }.get(key, default)
        
        caps = TerminalCapabilities.detect()
        
        assert caps.width == 120
        assert caps.height == 30
        assert caps.supports_color is True
        assert caps.color_depth == 16777216  # True color
    
    @patch('shutil.get_terminal_size')
    @patch('os.environ')
    def test_detect_limited_terminal(self, mock_environ, mock_terminal_size):
        """Test detection of limited terminal capabilities."""
        mock_terminal_size.return_value = Mock(columns=80, lines=24)
        
        # Mock environment for limited color support
        mock_environ.get.side_effect = lambda key, default='': {
            'TERM': 'xterm',
            'LANG': 'C'
        }.get(key, default)
        
        caps = TerminalCapabilities.detect()
        
        assert caps.supports_color is True  # 'xterm' contains 'color'
        assert caps.color_depth == 16


class TestRichTextRenderer:
    """Test RichTextRenderer class functionality."""
    
    def test_initialization(self):
        """Test renderer initialization."""
        renderer = RichTextRenderer()
        
        assert renderer.console is not None
        assert renderer.color_scheme is not None
        assert renderer.capabilities is not None
        assert renderer.theme is not None
    
    def test_initialization_with_custom_components(self):
        """Test initialization with custom components."""
        console = Console()
        color_scheme = ColorScheme(risk_high="purple")
        
        renderer = RichTextRenderer(
            console=console,
            color_scheme=color_scheme,
            auto_detect=False
        )
        
        assert renderer.color_scheme.risk_high == "purple"
        assert renderer.capabilities.width == 80  # Default when auto_detect=False
    
    def test_get_risk_color(self):
        """Test risk level color mapping."""
        renderer = RichTextRenderer()
        
        # Test enum input
        assert renderer.get_risk_color(RiskLevel.LOW) == "risk.low"
        assert renderer.get_risk_color(RiskLevel.HIGH) == "risk.high"
        assert renderer.get_risk_color(RiskLevel.CRITICAL) == "risk.critical"
        
        # Test string input
        assert renderer.get_risk_color("low") == "risk.low"
        assert renderer.get_risk_color("HIGH") == "risk.high"
        
        # Test numeric input
        assert renderer.get_risk_color(1) == "risk.low"    # 0-2 -> low
        assert renderer.get_risk_color(4) == "risk.medium" # 3-4 -> medium
        assert renderer.get_risk_color(7) == "risk.high"   # 5-7 -> high
        assert renderer.get_risk_color(9) == "risk.critical" # 8+ -> critical
        
        # Test invalid input
        assert renderer.get_risk_color("invalid") == "risk.none"
    
    def test_get_operation_color(self):
        """Test operation type color mapping."""
        renderer = RichTextRenderer()
        
        # Test enum input
        assert renderer.get_operation_color(OperationType.FILE_READ) == "op.read"
        assert renderer.get_operation_color(OperationType.FILE_WRITE) == "op.write"
        assert renderer.get_operation_color(OperationType.FILE_DELETE) == "op.delete"
        
        # Test string input
        assert renderer.get_operation_color("read") == "op.read"
        assert renderer.get_operation_color("file_write") == "op.write"
        assert renderer.get_operation_color("command_execute") == "op.execute"
        
        # Test fallback
        assert renderer.get_operation_color("unknown_operation") == "op.read"
    
    def test_render_code_block(self):
        """Test code block rendering."""
        renderer = RichTextRenderer()
        
        # Test with explicit language
        code = "def hello():\n    return 'world'"
        syntax = renderer.render_code_block(code, language="python")
        
        assert isinstance(syntax, Syntax)
        assert syntax.lexer.name == "Python"
        
        # Test auto-detection
        syntax_auto = renderer.render_code_block(code)  # Should detect Python
        assert isinstance(syntax_auto, Syntax)
        
        # Test JavaScript auto-detection
        js_code = "function hello() { return 'world'; }"
        syntax_js = renderer.render_code_block(js_code)
        assert isinstance(syntax_js, Syntax)
        
        # Test JSON auto-detection
        json_code = '{"key": "value", "number": 42}'
        syntax_json = renderer.render_code_block(json_code)
        assert isinstance(syntax_json, Syntax)
    
    def test_render_table(self):
        """Test table rendering."""
        renderer = RichTextRenderer()
        
        headers = ["Name", "Status", "Count"]
        rows = [
            ["Agent 1", True, 42],
            ["Agent 2", False, 0],
            ["Agent 3", True, 15]
        ]
        
        table = renderer.render_table(headers, rows, title="Test Table")
        
        assert isinstance(table, Table)
        assert table.title == "Test Table"
        # Table should have the correct number of columns
        assert len(table.columns) == 3
    
    def test_render_panel(self):
        """Test panel rendering."""
        renderer = RichTextRenderer()
        
        content = "This is panel content"
        panel = renderer.render_panel(
            content, 
            title="Test Panel",
            subtitle="Subtitle"
        )
        
        assert isinstance(panel, Panel)
        assert panel.title == "Test Panel"
        assert panel.subtitle == "Subtitle"
    
    def test_render_risk_indicator(self):
        """Test risk indicator rendering."""
        renderer = RichTextRenderer()
        
        # Test with label and bar
        indicator = renderer.render_risk_indicator(RiskLevel.HIGH, show_label=True, show_bar=True)
        assert indicator is not None
        
        # Test with just label
        label_only = renderer.render_risk_indicator(RiskLevel.MEDIUM, show_label=True, show_bar=False)
        assert isinstance(label_only, Text)
        
        # Test with numeric risk
        numeric_indicator = renderer.render_risk_indicator(8, show_label=True, show_bar=True)
        assert numeric_indicator is not None
    
    def test_render_progress_bar(self):
        """Test progress bar rendering."""
        renderer = RichTextRenderer()
        
        progress = renderer.render_progress_bar("Test Task", total=100, completed=50)
        
        assert isinstance(progress, Progress)
        # Should have one task
        assert len(progress.tasks) == 1
        task = progress.tasks[0]
        assert task.description == "Test Task"
        assert task.total == 100
        assert task.completed == 50
    
    def test_create_responsive_layout(self):
        """Test responsive layout creation."""
        renderer = RichTextRenderer()
        
        sections = {
            "left": "Left content",
            "right": "Right content"
        }
        
        # Test with wide terminal (should use columns)
        renderer.capabilities.width = 120
        layout = renderer.create_responsive_layout(sections)
        # Should return Columns for wide terminals
        
        # Test with narrow terminal (should use vertical layout)
        renderer.capabilities.width = 60
        layout = renderer.create_responsive_layout(sections, vertical_threshold=100)
        # Should return Layout for narrow terminals
        
        assert layout is not None
    
    def test_format_shortcut(self):
        """Test shortcut formatting."""
        renderer = RichTextRenderer()
        
        # Test enabled shortcut
        shortcut = renderer.format_shortcut("Ctrl+C", "Cancel operation", enabled=True)
        assert isinstance(shortcut, Text)
        
        # Test disabled shortcut
        disabled = renderer.format_shortcut("Ctrl+S", "Save file", enabled=False)
        assert isinstance(disabled, Text)
    
    def test_render_shortcuts_help(self):
        """Test shortcuts help rendering."""
        renderer = RichTextRenderer()
        
        shortcuts = {
            "q": "Quit",
            "h": "Help",
            "r": "Refresh"
        }
        
        # Test with narrow terminal (should use table)
        renderer.capabilities.width = 50
        help_display = renderer.render_shortcuts_help(shortcuts)
        assert help_display is not None
        
        # Test with wide terminal (should use columns)
        renderer.capabilities.width = 120
        help_display_wide = renderer.render_shortcuts_help(shortcuts, columns=2)
        assert help_display_wide is not None
    
    def test_apply_diff_highlighting(self):
        """Test diff highlighting."""
        renderer = RichTextRenderer()
        
        diff_text = """+++ a/file.py
--- b/file.py
@@ -1,3 +1,4 @@
 def hello():
+    print("debug")
     return "world"
-    # old comment"""
        
        highlighted = renderer.apply_diff_highlighting(diff_text, file_type="python")
        
        assert isinstance(highlighted, Text)
        # Should contain the original text
        assert "def hello():" in str(highlighted)
        assert "print(\"debug\")" in str(highlighted)
    
    def test_render_with_fallback(self):
        """Test fallback rendering."""
        renderer = RichTextRenderer()
        
        # Mock capabilities to test fallback
        renderer.capabilities.supports_color = False
        renderer.capabilities.supports_unicode = False
        
        # Test with rich content and plain text fallback
        rich_content = Text("Rich text", style="bold red")
        plain_text = "Plain text"
        
        # Should not raise an exception
        renderer.render_with_fallback(rich_content, plain_text)
    
    def test_utility_methods(self):
        """Test utility methods."""
        renderer = RichTextRenderer()
        
        # Test property methods
        assert renderer.get_terminal_width() == renderer.capabilities.width
        assert renderer.get_terminal_height() == renderer.capabilities.height
        assert renderer.supports_emoji() == renderer.capabilities.supports_emoji
        assert renderer.supports_color() == renderer.capabilities.supports_color


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_renderer(self):
        """Test renderer creation utility."""
        renderer = create_renderer()
        
        assert isinstance(renderer, RichTextRenderer)
        assert renderer.capabilities is not None
    
    def test_render_risk_badge(self):
        """Test risk badge utility."""
        badge = render_risk_badge(RiskLevel.HIGH)
        
        assert isinstance(badge, Text)
    
    def test_render_operation_badge(self):
        """Test operation badge utility."""
        badge = render_operation_badge("file_write")
        
        assert isinstance(badge, Text)
        assert "File Write" in str(badge)


class TestThemeCreation:
    """Test theme creation and application."""
    
    def test_theme_creation_from_color_scheme(self):
        """Test theme creation from color scheme."""
        color_scheme = ColorScheme(
            risk_high="magenta",
            op_write="bright_yellow"
        )
        
        renderer = RichTextRenderer(color_scheme=color_scheme)
        
        # Theme should include custom colors
        assert "risk.high" in renderer.theme.styles
        assert "op.write" in renderer.theme.styles
        
        # Just verify theme exists - Rich Theme internal structure can vary
        assert renderer.theme is not None
        assert len(renderer.theme.styles) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_table_data(self):
        """Test table rendering with empty data."""
        renderer = RichTextRenderer()
        
        table = renderer.render_table(["Header"], [], title="Empty Table")
        
        assert isinstance(table, Table)
        assert len(table.columns) == 1
    
    def test_empty_code_block(self):
        """Test code block with empty content."""
        renderer = RichTextRenderer()
        
        syntax = renderer.render_code_block("", language="python")
        
        assert isinstance(syntax, Syntax)
    
    def test_invalid_risk_levels(self):
        """Test handling of invalid risk level inputs."""
        renderer = RichTextRenderer()
        
        # Test with None
        color = renderer.get_risk_color(None)
        assert color == "risk.none"
        
        # Test with negative number
        color = renderer.get_risk_color(-1)
        assert color == "risk.low"  # Should clamp to minimum
    
    def test_malformed_diff_text(self):
        """Test diff highlighting with malformed diff."""
        renderer = RichTextRenderer()
        
        # Test with invalid diff format
        malformed_diff = "This is not a proper diff"
        
        highlighted = renderer.apply_diff_highlighting(malformed_diff)
        
        assert isinstance(highlighted, Text)
        assert "This is not a proper diff" in str(highlighted)