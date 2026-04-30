"""
Tests for [LOCATE:] operation marker in agent mode.

This module tests the file location functionality that uses fuzzy matching
to find files by name.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from omnimancer.cli.interface import CommandLineInterface
from omnimancer.core.engine import CoreEngine


@pytest.fixture
def temp_dir():
    """Create temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create test files
        (tmp_path / "test_file.txt").write_text("test content")
        (tmp_path / "My Document.docx").write_text("doc content")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested_file.py").write_text("# python code")
        (tmp_path / "Resume 2024.pdf").write_text("resume content")

        yield tmp_path


@pytest.fixture
def mock_engine():
    """Create mock CoreEngine."""
    engine = Mock(spec=CoreEngine)
    engine.send_message = AsyncMock(return_value="mock response")
    engine.current_provider = Mock()
    engine.current_provider.name = "test-provider"
    return engine


@pytest.fixture
def cli_interface(mock_engine, temp_dir):
    """Create CLI interface with mocked engine."""
    cli = CommandLineInterface(engine=mock_engine)
    # Set working directory to temp dir
    cli._current_dir = str(temp_dir)
    return cli


class TestLocateOperation:
    """Test suite for [LOCATE:filename] operation."""

    @pytest.mark.asyncio
    async def test_locate_exact_match(self, cli_interface, temp_dir):
        """Test locating a file with exact filename match."""
        # Change to temp directory
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            response = "Looking for file: [LOCATE:test_file.txt]"
            result = await cli_interface._parse_and_execute_operations(response)

            assert "📍 Located:" in result
            assert "test_file.txt" in result
            assert "[LOCATE:" not in result  # Should be replaced
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_locate_case_insensitive(self, cli_interface, temp_dir):
        """Test locating a file with case-insensitive match."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            response = "Looking for: [LOCATE:TEST_FILE.TXT]"
            result = await cli_interface._parse_and_execute_operations(response)

            assert "📍 Located:" in result
            assert "test_file.txt" in result
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_locate_with_spaces(self, cli_interface, temp_dir):
        """Test locating a file with spaces in the filename."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            response = "Find document: [LOCATE:My Document.docx]"
            result = await cli_interface._parse_and_execute_operations(response)

            assert "📍 Located:" in result
            assert "My Document.docx" in result
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_locate_fuzzy_match_typo(self, cli_interface, temp_dir):
        """Test locating a file with typo using fuzzy matching."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Typo: "Resme" instead of "Resume"
            response = "Looking for: [LOCATE:Resme 2024.pdf]"
            result = await cli_interface._parse_and_execute_operations(response)

            # Should find Resume 2024.pdf with fuzzy matching (>70% similarity)
            assert "📍 Located:" in result
            assert "Resume 2024.pdf" in result
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_locate_nested_file(self, cli_interface, temp_dir):
        """Test locating a file in subdirectory."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            response = "Find Python file: [LOCATE:nested_file.py]"
            result = await cli_interface._parse_and_execute_operations(response)

            assert "📍 Located:" in result
            assert "nested_file.py" in result
            assert "subdir" in result  # Should show path with subdirectory
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_locate_file_not_found(self, cli_interface, temp_dir):
        """Test locating a non-existent file."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Use a filename very different from anything in temp_dir
            response = "Where is: [LOCATE:completely_unique_zzz_999.xyz]"
            result = await cli_interface._parse_and_execute_operations(response)

            assert "❌ Could not locate file similar to" in result
            assert "completely_unique_zzz_999.xyz" in result
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_locate_below_similarity_threshold(self, cli_interface, temp_dir):
        """Test that files below 70% similarity threshold are not matched."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Very different filename - should not match any existing files
            response = "Find: [LOCATE:completely_different_name_123456789.xyz]"
            result = await cli_interface._parse_and_execute_operations(response)

            assert "❌ Could not locate file similar to" in result
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_locate_multiple_operations(self, cli_interface, temp_dir):
        """Test multiple [LOCATE:] operations in one response."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            response = """
            First file: [LOCATE:test_file.txt]
            Second file: [LOCATE:My Document.docx]
            Third file: [LOCATE:nonexistent.file]
            """
            result = await cli_interface._parse_and_execute_operations(response)

            # Should handle all three operations
            assert result.count("📍 Located:") == 2
            assert result.count("❌ Could not locate") == 1
            assert "test_file.txt" in result
            assert "My Document.docx" in result
            assert "nonexistent.file" in result
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_locate_with_other_operations(self, cli_interface, temp_dir):
        """Test [LOCATE:] mixed with other operation markers."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            response = """
            Finding file: [LOCATE:test_file.txt]
            Then read it: [FILE_READ:test_file.txt]
            """
            result = await cli_interface._parse_and_execute_operations(response)

            # Should process LOCATE first
            assert "📍 Located:" in result
            assert "test_file.txt" in result
        finally:
            os.chdir(original_cwd)


class TestFuzzyFindFileMethod:
    """Test the _fuzzy_find_file method directly."""

    def test_fuzzy_find_exact_match(self, cli_interface, temp_dir):
        """Test exact filename match."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            result = cli_interface._fuzzy_find_file("test_file.txt")
            assert result is not None
            assert "test_file.txt" in result
        finally:
            os.chdir(original_cwd)

    def test_fuzzy_find_case_insensitive(self, cli_interface, temp_dir):
        """Test case-insensitive matching."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            result = cli_interface._fuzzy_find_file("TEST_FILE.TXT")
            assert result is not None
            assert "test_file.txt" in result
        finally:
            os.chdir(original_cwd)

    def test_fuzzy_find_similarity_matching(self, cli_interface, temp_dir):
        """Test fuzzy similarity matching."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Slight typo - should still match
            result = cli_interface._fuzzy_find_file("test_fiel.txt")
            assert result is not None
            assert "test_file.txt" in result
        finally:
            os.chdir(original_cwd)

    def test_fuzzy_find_not_found(self, cli_interface, temp_dir):
        """Test when no match is found."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            result = cli_interface._fuzzy_find_file("completely_different.xyz")
            assert result is None
        finally:
            os.chdir(original_cwd)

    def test_fuzzy_find_nested_file(self, cli_interface, temp_dir):
        """Test finding files in subdirectories."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            result = cli_interface._fuzzy_find_file("nested_file.py")
            assert result is not None
            assert "nested_file.py" in result
            assert "subdir" in result
        finally:
            os.chdir(original_cwd)

    def test_fuzzy_find_threshold(self, cli_interface, temp_dir):
        """Test that similarity threshold (70%) is enforced."""
        import os
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Very different - should not match (below 70% threshold)
            result = cli_interface._fuzzy_find_file("xyz123abc456.file")
            assert result is None
        finally:
            os.chdir(original_cwd)
