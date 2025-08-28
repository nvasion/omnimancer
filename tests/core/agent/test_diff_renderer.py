"""
Tests for the EnhancedDiffRenderer class.

This module tests the enhanced diff display and code highlighting system
for the Omnimancer approval interface.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

from omnimancer.core.agent.diff_renderer import (
    EnhancedDiffRenderer, DiffType, FileChangeType, FileChange, DiffChunk,
    create_diff_renderer, render_git_diff, compare_files
)
from omnimancer.core.agent.rich_renderer import RichTextRenderer, create_renderer


class TestFileChange:
    """Test FileChange dataclass."""
    
    def test_file_change_creation(self):
        """Test FileChange creation with default values."""
        change = FileChange(
            file_path="test.py",
            change_type=FileChangeType.MODIFIED
        )
        
        assert change.file_path == "test.py"
        assert change.change_type == FileChangeType.MODIFIED
        assert change.old_path is None
        assert change.new_path is None
        assert change.lines_added == 0
        assert change.lines_removed == 0
    
    def test_file_change_with_all_fields(self):
        """Test FileChange with all fields populated."""
        change = FileChange(
            file_path="new_test.py",
            change_type=FileChangeType.RENAMED,
            old_path="old_test.py",
            new_path="new_test.py",
            old_content="old content",
            new_content="new content",
            diff_text="@@ -1 +1 @@\n-old content\n+new content",
            language="python",
            lines_added=1,
            lines_removed=1
        )
        
        assert change.file_path == "new_test.py"
        assert change.change_type == FileChangeType.RENAMED
        assert change.old_path == "old_test.py"
        assert change.new_path == "new_test.py"
        assert change.language == "python"
        assert change.lines_added == 1
        assert change.lines_removed == 1


class TestDiffChunk:
    """Test DiffChunk dataclass."""
    
    def test_diff_chunk_creation(self):
        """Test DiffChunk creation."""
        chunk = DiffChunk(
            old_start=10,
            old_count=5,
            new_start=10,
            new_count=6,
            context_before=["line1", "line2"],
            changes=[("add", "new line")],
            context_after=["line3"]
        )
        
        assert chunk.old_start == 10
        assert chunk.old_count == 5
        assert chunk.new_start == 10
        assert chunk.new_count == 6
        assert len(chunk.context_before) == 2
        assert len(chunk.changes) == 1
        assert len(chunk.context_after) == 1


class TestEnhancedDiffRenderer:
    """Test EnhancedDiffRenderer class functionality."""
    
    def test_initialization(self):
        """Test renderer initialization."""
        renderer = EnhancedDiffRenderer()
        
        assert renderer.renderer is not None
        assert renderer.console is not None
        assert renderer.max_line_length == 120
        assert renderer.context_lines == 3
        assert renderer.show_line_numbers is True
        assert renderer.highlight_syntax is True
        assert renderer.word_level_diff is True
    
    def test_initialization_with_custom_components(self):
        """Test initialization with custom components."""
        rich_renderer = create_renderer()
        console = Console()
        
        diff_renderer = EnhancedDiffRenderer(
            renderer=rich_renderer,
            console=console
        )
        
        assert diff_renderer.renderer == rich_renderer
        assert diff_renderer.console == console
    
    def test_detect_language(self):
        """Test programming language detection."""
        renderer = EnhancedDiffRenderer()
        
        # Test various file extensions
        assert renderer.detect_language("test.py") == "python"
        assert renderer.detect_language("app.js") == "javascript"
        assert renderer.detect_language("component.tsx") == "tsx"
        assert renderer.detect_language("config.json") == "json"
        assert renderer.detect_language("README.md") == "markdown"
        assert renderer.detect_language("script.sh") == "bash"
        assert renderer.detect_language("main.go") == "go"
        assert renderer.detect_language("lib.rs") == "rust"
        
        # Test case insensitive matching
        assert renderer.detect_language("TEST.PY") == "python"
        assert renderer.detect_language("App.JS") == "javascript"
        
        # Test unknown extension
        assert renderer.detect_language("unknown.xyz") is None
        assert renderer.detect_language("no_extension") is None
    
    def test_parse_unified_diff_simple(self):
        """Test parsing simple unified diff."""
        diff_text = """diff --git a/test.py b/test.py
index 1234567..abcdefg 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def hello():
+    print("debug")
     return "world"
-    # old comment"""
        
        renderer = EnhancedDiffRenderer()
        changes = renderer.parse_unified_diff(diff_text)
        
        assert len(changes) == 1
        
        change = changes[0]
        assert change.file_path == "test.py"
        assert change.change_type == FileChangeType.MODIFIED
        assert change.old_path == "test.py"
        assert change.new_path == "test.py"
        assert change.language == "python"
        assert change.lines_added == 1
        assert change.lines_removed == 1
        assert change.diff_text is not None
    
    def test_parse_unified_diff_new_file(self):
        """Test parsing diff with new file."""
        diff_text = """diff --git a/new_file.js b/new_file.js
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new_file.js
@@ -0,0 +1,3 @@
+function test() {
+    return 42;
+}"""
        
        renderer = EnhancedDiffRenderer()
        changes = renderer.parse_unified_diff(diff_text)
        
        assert len(changes) == 1
        
        change = changes[0]
        assert change.file_path == "new_file.js"
        assert change.change_type == FileChangeType.ADDED
        assert change.language == "javascript"
        assert change.lines_added == 3
        assert change.lines_removed == 0
    
    def test_parse_unified_diff_deleted_file(self):
        """Test parsing diff with deleted file."""
        diff_text = """diff --git a/old_file.py b/old_file.py
deleted file mode 100644
index 1234567..0000000
--- a/old_file.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old_function():
-    pass
-    # deprecated"""
        
        renderer = EnhancedDiffRenderer()
        changes = renderer.parse_unified_diff(diff_text)
        
        assert len(changes) == 1
        
        change = changes[0]
        assert change.file_path == "old_file.py"
        assert change.change_type == FileChangeType.DELETED
        assert change.language == "python"
        assert change.lines_added == 0
        assert change.lines_removed == 3
    
    def test_parse_unified_diff_renamed_file(self):
        """Test parsing diff with renamed file."""
        diff_text = """diff --git a/old_name.py b/new_name.py
similarity index 100%
rename from old_name.py
rename to new_name.py"""
        
        renderer = EnhancedDiffRenderer()
        changes = renderer.parse_unified_diff(diff_text)
        
        assert len(changes) == 1
        
        change = changes[0]
        assert change.file_path == "new_name.py"
        assert change.change_type == FileChangeType.RENAMED
        assert change.old_path == "old_name.py"
        assert change.new_path == "new_name.py"
        assert change.language == "python"
    
    def test_parse_unified_diff_multiple_files(self):
        """Test parsing diff with multiple files."""
        diff_text = """diff --git a/file1.py b/file1.py
index 1234567..abcdefg 100644
--- a/file1.py
+++ b/file1.py
@@ -1 +1,2 @@
 print("hello")
+print("world")
diff --git a/file2.js b/file2.js
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/file2.js
@@ -0,0 +1,2 @@
+console.log("hello");
+console.log("world");"""
        
        renderer = EnhancedDiffRenderer()
        changes = renderer.parse_unified_diff(diff_text)
        
        assert len(changes) == 2
        
        # First file - modified
        change1 = changes[0]
        assert change1.file_path == "file1.py"
        assert change1.change_type == FileChangeType.MODIFIED
        assert change1.language == "python"
        assert change1.lines_added == 1
        assert change1.lines_removed == 0
        
        # Second file - new
        change2 = changes[1]
        assert change2.file_path == "file2.js"
        assert change2.change_type == FileChangeType.ADDED
        assert change2.language == "javascript"
        assert change2.lines_added == 2
        assert change2.lines_removed == 0
    
    def test_render_file_tree(self):
        """Test file tree rendering."""
        renderer = EnhancedDiffRenderer()
        
        changes = [
            FileChange("src/main.py", FileChangeType.MODIFIED, lines_added=5, lines_removed=2),
            FileChange("tests/test_main.py", FileChangeType.ADDED, lines_added=10, lines_removed=0),
            FileChange("README.md", FileChangeType.MODIFIED, lines_added=1, lines_removed=1),
            FileChange("old_file.py", FileChangeType.DELETED, lines_added=0, lines_removed=20)
        ]
        
        tree = renderer.render_file_tree(changes)
        
        assert isinstance(tree, Tree)
        assert tree.label == "📁 Changed Files"
        # Tree should have nodes for different directories
        assert len(tree.children) > 0
    
    def test_render_unified_diff(self):
        """Test unified diff rendering."""
        renderer = EnhancedDiffRenderer()
        
        change = FileChange(
            file_path="test.py",
            change_type=FileChangeType.MODIFIED,
            diff_text="@@ -1,3 +1,4 @@\n def hello():\n+    print('debug')\n     return 'world'",
            language="python",
            lines_added=1,
            lines_removed=0
        )
        
        panel = renderer.render_unified_diff(change)
        
        assert isinstance(panel, Panel)
        assert "test.py" in str(panel.title)
    
    def test_render_unified_diff_no_changes(self):
        """Test unified diff rendering with no changes."""
        renderer = EnhancedDiffRenderer()
        
        change = FileChange(
            file_path="empty.py",
            change_type=FileChangeType.MODIFIED,
            diff_text=None
        )
        
        panel = renderer.render_unified_diff(change)
        
        assert isinstance(panel, Panel)
        assert "No changes" in str(panel.renderable)
    
    def test_render_side_by_side_diff(self):
        """Test side-by-side diff rendering."""
        renderer = EnhancedDiffRenderer()
        
        change = FileChange(
            file_path="test.py",
            change_type=FileChangeType.MODIFIED,
            old_content="def hello():\n    return 'world'",
            new_content="def hello():\n    print('debug')\n    return 'world'",
            language="python",
            lines_added=1,
            lines_removed=0
        )
        
        panel = renderer.render_side_by_side_diff(change)
        
        assert isinstance(panel, Panel)
        assert isinstance(panel.renderable, Table)
    
    def test_render_side_by_side_diff_fallback(self):
        """Test side-by-side diff fallback to unified."""
        renderer = EnhancedDiffRenderer()
        
        change = FileChange(
            file_path="test.py",
            change_type=FileChangeType.MODIFIED,
            old_content=None,  # Missing content should fallback
            new_content=None,
            diff_text="@@ -1 +1,2 @@\n print('hello')\n+print('world')"
        )
        
        panel = renderer.render_side_by_side_diff(change)
        
        assert isinstance(panel, Panel)
        # Should fallback to unified diff rendering
    
    def test_render_word_diff(self):
        """Test word-level diff highlighting."""
        renderer = EnhancedDiffRenderer()
        
        old_line = "Hello world how are you"
        new_line = "Hello beautiful world how are you today"
        
        old_formatted, new_formatted = renderer.render_word_diff(old_line, new_line)
        
        assert isinstance(old_formatted, Text)
        assert isinstance(new_formatted, Text)
        assert len(str(old_formatted)) > 0
        assert len(str(new_formatted)) > 0
    
    def test_render_word_diff_disabled(self):
        """Test word-level diff when disabled."""
        renderer = EnhancedDiffRenderer()
        renderer.word_level_diff = False
        
        old_line = "Hello world"
        new_line = "Hello beautiful world"
        
        old_formatted, new_formatted = renderer.render_word_diff(old_line, new_line)
        
        assert isinstance(old_formatted, Text)
        assert isinstance(new_formatted, Text)
        assert str(old_formatted) == old_line
        assert str(new_formatted) == new_line
    
    def test_render_diff_summary(self):
        """Test diff summary table rendering."""
        renderer = EnhancedDiffRenderer()
        
        changes = [
            FileChange("file1.py", FileChangeType.MODIFIED, language="python", lines_added=5, lines_removed=2),
            FileChange("file2.js", FileChangeType.ADDED, language="javascript", lines_added=10, lines_removed=0),
            FileChange("file3.md", FileChangeType.DELETED, language="markdown", lines_added=0, lines_removed=8)
        ]
        
        table = renderer.render_diff_summary(changes)
        
        assert isinstance(table, Table)
        assert table.title == "📊 Diff Summary"
        # Should have header + data rows + separator + totals = 6 rows
        assert len(table.rows) > 0
    
    def test_render_diff_set_simple(self):
        """Test rendering a complete diff set."""
        renderer = EnhancedDiffRenderer()
        
        diff_text = """diff --git a/test.py b/test.py
index 1234567..abcdefg 100644
--- a/test.py
+++ b/test.py
@@ -1,2 +1,3 @@
 def hello():
+    print("debug")
     return "world\""""
        
        renderables = renderer.render_diff_set(diff_text)
        
        assert len(renderables) > 0
        # Should include summary, file tree, and file diff
        assert any(isinstance(r, Table) for r in renderables)  # Summary table
        assert any(isinstance(r, Tree) for r in renderables)   # File tree
        assert any(isinstance(r, Panel) for r in renderables)  # File diff
    
    def test_render_diff_set_no_changes(self):
        """Test rendering diff set with no changes."""
        renderer = EnhancedDiffRenderer()
        
        diff_text = "No changes detected"  # Invalid diff format
        
        renderables = renderer.render_diff_set(diff_text)
        
        assert len(renderables) == 1
        assert isinstance(renderables[0], Panel)
        assert "No changes" in str(renderables[0].renderable)
    
    def test_render_diff_set_side_by_side(self):
        """Test rendering diff set with side-by-side view."""
        renderer = EnhancedDiffRenderer()
        
        diff_text = """diff --git a/test.py b/test.py
index 1234567..abcdefg 100644
--- a/test.py
+++ b/test.py
@@ -1,2 +1,3 @@
 def hello():
+    print("debug")
     return "world\""""
        
        renderables = renderer.render_diff_set(
            diff_text, 
            diff_type=DiffType.SIDE_BY_SIDE
        )
        
        assert len(renderables) > 0
        # Should include side-by-side panels (though may fallback to unified)
        assert any(isinstance(r, Panel) for r in renderables)
    
    def test_render_diff_set_minimal(self):
        """Test rendering diff set with minimal options."""
        renderer = EnhancedDiffRenderer()
        
        diff_text = """diff --git a/test.py b/test.py
index 1234567..abcdefg 100644
--- a/test.py
+++ b/test.py
@@ -1,2 +1,3 @@
 def hello():
+    print("debug")
     return "world\""""
        
        renderables = renderer.render_diff_set(
            diff_text,
            show_summary=False,
            show_file_tree=False
        )
        
        # Should only include file diffs, no summary or tree
        assert len(renderables) >= 1
        # Should have file diff and rule separators
        assert any(isinstance(r, Panel) for r in renderables)
    
    def test_render_file_content_comparison(self):
        """Test file content comparison rendering."""
        renderer = EnhancedDiffRenderer()
        
        old_content = "def hello():\n    return 'world'"
        new_content = "def hello():\n    print('debug')\n    return 'world'"
        
        panel = renderer.render_file_content_comparison(
            old_content,
            new_content,
            "test.py",
            language="python"
        )
        
        assert isinstance(panel, Panel)
        assert "test.py" in str(panel.title)
    
    def test_render_file_content_comparison_auto_detect(self):
        """Test file content comparison with language auto-detection."""
        renderer = EnhancedDiffRenderer()
        
        old_content = "console.log('hello');"
        new_content = "console.log('hello');\nconsole.log('world');"
        
        panel = renderer.render_file_content_comparison(
            old_content,
            new_content,
            "test.js"  # Should auto-detect JavaScript
        )
        
        assert isinstance(panel, Panel)
        assert "test.js" in str(panel.title)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_diff_renderer(self):
        """Test diff renderer creation utility."""
        renderer = create_diff_renderer()
        
        assert isinstance(renderer, EnhancedDiffRenderer)
        assert renderer.renderer is not None
        assert renderer.console is not None
    
    def test_create_diff_renderer_with_custom(self):
        """Test diff renderer creation with custom renderer."""
        rich_renderer = create_renderer()
        diff_renderer = create_diff_renderer(renderer=rich_renderer)
        
        assert isinstance(diff_renderer, EnhancedDiffRenderer)
        assert diff_renderer.renderer == rich_renderer
    
    @patch('omnimancer.core.agent.diff_renderer.EnhancedDiffRenderer')
    def test_render_git_diff(self, mock_renderer_class):
        """Test git diff rendering utility."""
        mock_renderer = Mock()
        mock_renderer.console = Mock()
        mock_renderer.render_diff_set.return_value = [Panel("Test diff")]
        mock_renderer_class.return_value = mock_renderer
        
        diff_output = "diff --git a/test.py b/test.py..."
        
        render_git_diff(diff_output, DiffType.UNIFIED)
        
        mock_renderer_class.assert_called_once()
        mock_renderer.render_diff_set.assert_called_once_with(diff_output, DiffType.UNIFIED)
        mock_renderer.console.print.assert_called()
    
    @patch('omnimancer.core.agent.diff_renderer.Path')
    @patch('omnimancer.core.agent.diff_renderer.EnhancedDiffRenderer')
    def test_compare_files_with_paths(self, mock_renderer_class, mock_path_class):
        """Test file comparison utility with file paths."""
        # Mock Path constructor to return mock objects
        mock_old_path = Mock()
        mock_old_path.exists.return_value = True
        mock_old_path.read_text.return_value = "old content"
        
        mock_new_path = Mock()
        mock_new_path.exists.return_value = True
        mock_new_path.read_text.return_value = "new content"
        
        # Make Path() return different mocks for old and new files
        def path_side_effect(path):
            if "old.py" in str(path):
                return mock_old_path
            else:  # new.py
                return mock_new_path
        
        mock_path_class.side_effect = path_side_effect
        
        # Mock renderer
        mock_renderer = Mock()
        mock_renderer.console = Mock()
        mock_renderer.render_file_content_comparison.return_value = Panel("Comparison")
        mock_renderer_class.return_value = mock_renderer
        
        compare_files("old.py", "new.py", "test.py")
        
        mock_renderer.render_file_content_comparison.assert_called_once_with(
            "old content", "new content", "test.py"
        )
        mock_renderer.console.print.assert_called_once()
    
    @patch('omnimancer.core.agent.diff_renderer.Path')
    @patch('omnimancer.core.agent.diff_renderer.EnhancedDiffRenderer')
    def test_compare_files_with_content(self, mock_renderer_class, mock_path):
        """Test file comparison utility with content strings."""
        # Mock files don't exist
        mock_path_obj = Mock()
        mock_path_obj.exists.return_value = False
        mock_path.return_value = mock_path_obj
        
        # Mock renderer
        mock_renderer = Mock()
        mock_renderer.console = Mock()
        mock_renderer.render_file_content_comparison.return_value = Panel("Comparison")
        mock_renderer_class.return_value = mock_renderer
        
        old_content = "old content"
        new_content = "new content"
        
        compare_files(old_content, new_content, "test.py")
        
        mock_renderer.render_file_content_comparison.assert_called_once_with(
            old_content, new_content, "test.py"
        )


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_diff_text(self):
        """Test parsing empty diff text."""
        renderer = EnhancedDiffRenderer()
        
        changes = renderer.parse_unified_diff("")
        
        assert changes == []
    
    def test_malformed_diff_text(self):
        """Test parsing malformed diff text."""
        renderer = EnhancedDiffRenderer()
        
        malformed_diff = "This is not a proper diff format"
        changes = renderer.parse_unified_diff(malformed_diff)
        
        # Should handle gracefully and return empty list
        assert isinstance(changes, list)
    
    def test_diff_with_no_file_path(self):
        """Test diff parsing with missing file paths."""
        renderer = EnhancedDiffRenderer()
        
        incomplete_diff = """diff --git 
--- a/unknown
+++ b/unknown
@@ -1 +1,2 @@
 line1
+line2"""
        
        changes = renderer.parse_unified_diff(incomplete_diff)
        
        # Should handle gracefully
        assert isinstance(changes, list)
    
    def test_render_diff_summary_empty(self):
        """Test diff summary with empty changes."""
        renderer = EnhancedDiffRenderer()
        
        table = renderer.render_diff_summary([])
        
        assert isinstance(table, Table)
        # Should still have totals row
        assert len(table.rows) >= 1
    
    def test_render_file_tree_empty(self):
        """Test file tree with empty changes."""
        renderer = EnhancedDiffRenderer()
        
        tree = renderer.render_file_tree([])
        
        assert isinstance(tree, Tree)
        assert tree.label == "📁 Changed Files"
        # Should have no children
        assert len(tree.children) == 0