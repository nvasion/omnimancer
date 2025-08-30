"""
Tests for the Unified File Content Display module.
"""

from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from omnimancer.core.agent.diff_renderer import DiffType
from omnimancer.core.agent.file_content_display import (
    DisplayMode,
    FileDisplayConfig,
    UnifiedFileContentDisplay,
    create_file_content_display,
)


class TestUnifiedFileContentDisplay:
    """Test suite for UnifiedFileContentDisplay."""

    def setup_method(self):
        """Set up test fixtures."""
        self.console = Mock(spec=Console)
        self.config = FileDisplayConfig(
            display_mode=DisplayMode.FULL_CONTENT,
            syntax_highlighting=True,
            show_line_numbers=True,
            max_preview_lines=50,
            diff_type=DiffType.UNIFIED,
        )
        self.display = UnifiedFileContentDisplay(
            console=self.console, config=self.config
        )

    @pytest.mark.asyncio
    async def test_display_file_creation_interactive(self):
        """Test interactive file creation display."""
        file_path = "/test/file.py"
        content = "def hello():\n    print('Hello, World!')"
        operation_context = {"interactive": True}

        with patch.object(
            self.display.unified_approval_ui,
            "prompt_for_file_modification_approval",
        ) as mock_review:
            mock_review.return_value = {
                "approved": True,
                "modified_content": None,
            }

            result = await self.display.display_file_creation(
                file_path, content, operation_context
            )

            assert mock_review.called
            call_args = mock_review.call_args[0][0]
            assert call_args["file_path"] == file_path
            assert call_args["new_content"] == content
            assert call_args["operation"] == "create"
            assert call_args["file_exists"] is False
            assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_display_file_creation_non_interactive(self):
        """Test non-interactive file creation display."""
        file_path = "/test/file.py"
        content = "def hello():\n    print('Hello, World!')"
        operation_context = {"interactive": False}

        result = await self.display.display_file_creation(
            file_path, content, operation_context
        )

        assert result["displayed"] is True
        assert result["interactive"] is False
        assert self.console.print.called

    @pytest.mark.asyncio
    async def test_display_file_creation_truncation(self):
        """Test file content truncation for large files."""
        file_path = "/test/large_file.txt"
        content = "x" * (2 * 1024 * 1024)  # 2MB content
        operation_context = {"interactive": False}

        result = await self.display.display_file_creation(
            file_path, content, operation_context
        )

        assert result["displayed"] is True
        # Content should be truncated to max_content_size
        assert len(content) > self.config.max_content_size

    @pytest.mark.asyncio
    async def test_display_file_modification_with_diff(self):
        """Test file modification display with diff generation."""
        file_path = "/test/file.py"
        current_content = "def hello():\n    print('Hello')"
        new_content = "def hello():\n    print('Hello, World!')"
        operation_context = {"interactive": True}

        with patch.object(
            self.display.diff_renderer, "render_unified_diff"
        ) as mock_diff:
            mock_diff.return_value = "--- a/test/file.py\n+++ b/test/file.py\n..."

            with patch.object(
                self.display.unified_approval_ui,
                "prompt_for_file_modification_approval",
            ) as mock_review:
                mock_review.return_value = {"approved": True}

                result = await self.display.display_file_modification(
                    file_path, current_content, new_content, operation_context
                )

                assert mock_diff.called
                assert mock_review.called
                assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_display_file_deletion_with_warning(self):
        """Test file deletion display with warning panel."""
        file_path = "/test/file_to_delete.py"
        content = "# This file will be deleted"
        operation_context = {"interactive": False}

        result = await self.display.display_file_deletion(
            file_path, content, operation_context
        )

        assert result["displayed"] is True
        assert result["interactive"] is False
        # Check that warning panel was printed
        assert self.console.print.called
        # Check that at least 2 panels were printed (warning and content)
        assert self.console.print.call_count >= 2

    @pytest.mark.asyncio
    async def test_display_file_deletion_interactive_confirm(self):
        """Test interactive file deletion with confirmation."""
        file_path = "/test/file_to_delete.py"
        content = "# This file will be deleted"
        operation_context = {"interactive": True}

        with patch("rich.prompt.Confirm") as mock_confirm:
            mock_confirm.ask.return_value = True

            result = await self.display.display_file_deletion(
                file_path, content, operation_context
            )

            assert result["approved"] is True
            assert result["interactive"] is True
            assert mock_confirm.ask.called

    def test_display_batch_operations(self):
        """Test batch operations display."""
        operations = [
            {
                "type": "create",
                "file_path": "/test/file1.py",
                "risk_level": "low",
            },
            {
                "type": "modify",
                "file_path": "/test/file2.py",
                "risk_level": "medium",
            },
            {
                "type": "delete",
                "file_path": "/test/file3.py",
                "risk_level": "high",
            },
            {
                "type": "create",
                "file_path": "/test/file4.py",
                "risk_level": "low",
            },
        ]

        result = self.display.display_batch_operations(operations)

        assert result["displayed"] is True
        assert result["operation_count"] == 4
        assert "create" in result["types"]
        assert "modify" in result["types"]
        assert "delete" in result["types"]
        assert self.console.print.called

    def test_display_batch_operations_large_batch(self):
        """Test batch operations display with many operations."""
        operations = [
            {
                "type": "create",
                "file_path": f"/test/file{i}.py",
                "risk_level": "low",
            }
            for i in range(20)
        ]

        result = self.display.display_batch_operations(operations)

        assert result["displayed"] is True
        assert result["operation_count"] == 20
        # Should only show first 10 operations in detail
        assert self.console.print.called

    def test_detect_language(self):
        """Test language detection from file extensions."""
        test_cases = [
            ("/test/file.py", "python"),
            ("/test/file.js", "javascript"),
            ("/test/file.ts", "typescript"),
            ("/test/file.java", "java"),
            ("/test/file.cpp", "cpp"),
            ("/test/file.md", "markdown"),
            ("/test/file.json", "json"),
            ("/test/file.yaml", "yaml"),
            ("/test/file.unknown", "text"),
        ]

        for file_path, expected_language in test_cases:
            detected = self.display._detect_language(file_path)
            assert detected == expected_language

    def test_get_operation_icon(self):
        """Test operation icon mapping."""
        test_cases = [
            ("create", "📄"),
            ("modify", "✏️"),
            ("delete", "🗑️"),
            ("move", "📦"),
            ("copy", "📋"),
            ("unknown", "❓"),
            ("invalid", "❓"),
        ]

        for op_type, expected_icon in test_cases:
            icon = self.display._get_operation_icon(op_type)
            assert icon == expected_icon

    def test_get_risk_color(self):
        """Test risk level color mapping."""
        test_cases = [
            ("low", "green"),
            ("medium", "yellow"),
            ("high", "orange1"),
            ("critical", "red"),
            ("unknown", "white"),
        ]

        for risk_level, expected_color in test_cases:
            color = self.display._get_risk_color(risk_level)
            assert color == expected_color

    def test_get_risk_summary(self):
        """Test risk summary generation."""
        operations = [
            {"type": "create", "risk_level": "low"},
            {"type": "create", "risk_level": "low"},
            {"type": "create", "risk_level": "medium"},
            {"type": "modify", "risk_level": "high"},
            {"type": "modify", "risk_level": "critical"},
        ]

        summary = self.display._get_risk_summary(operations, "create")
        assert "low: 2" in summary
        assert "medium: 1" in summary

        summary = self.display._get_risk_summary(operations, "modify")
        assert "critical: 1" in summary
        assert "high: 1" in summary

        summary = self.display._get_risk_summary(operations, "delete")
        assert summary == "N/A"

    def test_create_file_content_display_factory(self):
        """Test factory function for creating display instances."""
        display = create_file_content_display()
        assert isinstance(display, UnifiedFileContentDisplay)
        assert display.console is not None
        assert display.config is not None

        custom_console = Mock(spec=Console)
        custom_config = FileDisplayConfig(display_mode=DisplayMode.PREVIEW)
        display = create_file_content_display(
            console=custom_console, config=custom_config
        )
        assert display.console == custom_console
        assert display.config == custom_config

    @pytest.mark.asyncio
    async def test_error_handling_file_creation(self):
        """Test error handling in file creation display."""
        file_path = "/test/file.py"
        content = "test content"
        operation_context = {"interactive": True}

        with patch.object(
            self.display.unified_approval_ui,
            "prompt_for_file_modification_approval",
        ) as mock_review:
            mock_review.side_effect = Exception("Test error")

            result = await self.display.display_file_creation(
                file_path, content, operation_context
            )

            assert "error" in result
            assert result["displayed"] is False
            assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_error_handling_file_modification(self):
        """Test error handling in file modification display."""
        file_path = "/test/file.py"
        current_content = "old"
        new_content = "new"
        operation_context = {"interactive": True}

        with patch.object(
            self.display.diff_renderer, "render_unified_diff"
        ) as mock_diff:
            mock_diff.side_effect = Exception("Diff error")

            result = await self.display.display_file_modification(
                file_path, current_content, new_content, operation_context
            )

            assert "error" in result
            assert result["displayed"] is False
            assert "Diff error" in result["error"]

    def test_preview_mode_configuration(self):
        """Test preview mode configuration."""
        config = FileDisplayConfig(
            display_mode=DisplayMode.PREVIEW, max_preview_lines=10
        )
        display = UnifiedFileContentDisplay(config=config)

        assert display.config.display_mode == DisplayMode.PREVIEW
        assert display.config.max_preview_lines == 10
        assert display.dialog_options.show_diff is True
        assert display.dialog_options.syntax_highlighting is True
