"""
Unit tests for system prompt logic and safety parameters.

This module tests the system prompt generation and ensures safety and directory
awareness parameters are properly included.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from omnimancer.cli.interface import CommandLineInterface
from omnimancer.core.agent.file_system_manager import FileSystemManager
from omnimancer.core.security import SecurityManager
from omnimancer.core.engine import CoreEngine


class TestSystemPromptLogic:
    """Test system prompt logic and safety parameters."""

    @pytest.fixture
    def interface(self):
        """Create CommandLineInterface for testing."""
        # Mock the necessary components
        mock_engine = Mock(spec=CoreEngine)
        interface = CommandLineInterface(mock_engine, no_approval=True)
        interface.agent_manager = Mock()
        interface.agent_manager.mode = Mock()
        return interface

    @pytest.fixture
    def file_system_manager(self):
        """Create FileSystemManager for testing."""
        return FileSystemManager(
            security_manager=SecurityManager(), require_approval=False
        )

    def test_get_agent_capabilities_prompt_content(self, interface):
        """Test that agent capabilities prompt contains expected content."""
        prompt = interface._get_agent_capabilities_prompt()

        # Test basic structure
        assert "SYSTEM: You are an autonomous AI agent" in prompt
        assert "FILE OPERATIONS:" in prompt
        assert "COMMAND EXECUTION:" in prompt
        assert "WEB OPERATIONS:" in prompt
        assert "SECURITY FEATURES:" in prompt

        # Test security features mentioned
        assert "security validation" in prompt
        assert "User approval required for high-risk operations" in prompt
        assert "Sandboxed execution environment" in prompt

        # Test operation markers
        assert "[FILE_WRITE:filename]" in prompt
        assert "[FILE_READ:filename]" in prompt
        assert "[COMMAND_EXEC]" in prompt
        assert "[WEB_REQUEST:url]" in prompt

    def test_agent_capabilities_prompt_immutability(self, interface):
        """Test that the prompt content is consistent."""
        prompt1 = interface._get_agent_capabilities_prompt()
        prompt2 = interface._get_agent_capabilities_prompt()

        assert prompt1 == prompt2
        assert len(prompt1) > 1000  # Should be substantial

    def test_directory_awareness_methods(self, file_system_manager):
        """Test directory awareness functionality."""
        # Test get_current_working_directory
        cwd = file_system_manager.get_current_working_directory()
        assert isinstance(cwd, Path)
        assert cwd.exists()

        # Test that it returns the actual current directory
        assert cwd == Path.cwd()

    @pytest.mark.asyncio
    async def test_directory_context_generation(self, file_system_manager):
        """Test directory context generation."""
        context = await file_system_manager.get_directory_context()

        # Test required fields
        assert "current_working_directory" in context
        assert "is_git_repository" in context
        assert "git_repository_root" in context
        assert "relative_to_repo_root" in context

        # Test that current working directory is a string path
        assert isinstance(context["current_working_directory"], str)
        assert os.path.exists(context["current_working_directory"])

        # Test git repository detection
        assert isinstance(context["is_git_repository"], bool)

    def test_enhanced_prompt_includes_directory_awareness(self, interface):
        """Test that enhanced prompt includes directory information."""
        prompt = interface._get_agent_capabilities_prompt()

        # These SHOULD be in the enhanced prompt
        assert "working directory" in prompt.lower()
        assert "git repository" in prompt.lower()
        assert str(Path.cwd()) in prompt
        assert "📍 CURRENT ENVIRONMENT:" in prompt

    def test_enhanced_prompt_includes_safety_details(self, interface):
        """Test enhanced prompt includes specific safety implementation details."""
        prompt = interface._get_agent_capabilities_prompt()

        # These specific safety features should now be present
        assert "read-before-write" in prompt.lower()
        assert "file existence checking" in prompt.lower()
        assert "user confirmation for file overwrites" in prompt.lower()
        assert "automatic backup creation" in prompt.lower()
        assert "SAFETY PROTOCOLS (CANNOT BE OVERRIDDEN)" in prompt

    @pytest.mark.asyncio
    async def test_agent_mode_prompt_injection(self, interface):
        """Test that agent prompt is properly injected when agent mode is on."""
        # Mock agent manager in 'on' state
        interface.agent_manager.mode.value = "on"
        interface.engine.send_message = AsyncMock()
        interface.engine.send_message.return_value = Mock(
            is_success=True, content="Test response", model_used="test-model"
        )
        interface._parse_and_execute_operations = AsyncMock(
            return_value="Executed response"
        )
        interface._show_assistant_message = Mock()
        interface._show_user_message = Mock()
        interface.progress_indicator = Mock()
        interface.progress_indicator.disable = Mock()

        # Mock command
        command = Mock()
        command.content = "Test user message"

        # Test that agent prompt is included
        await interface._handle_chat_message(command)

        # Verify send_message was called with agent prompt
        interface.engine.send_message.assert_called_once()
        call_args = interface.engine.send_message.call_args[0][0]

        # Should contain both agent prompt and user message
        assert "SYSTEM: You are an autonomous AI agent" in call_args
        assert "User: Test user message" in call_args

    @pytest.mark.asyncio
    async def test_agent_mode_disabled_no_prompt_injection(self, interface):
        """Test that agent prompt is NOT injected when agent mode is off."""
        # Mock agent manager in 'off' state
        interface.agent_manager.mode.value = "off"
        interface.engine.send_message = AsyncMock()
        interface.engine.send_message.return_value = Mock(
            is_success=True, content="Test response", model_used="test-model"
        )
        interface._show_assistant_message = Mock()
        interface._show_user_message = Mock()
        interface.progress_indicator = Mock()
        interface.progress_indicator.disable = Mock()

        # Mock command
        command = Mock()
        command.content = "Test user message"

        # Test that agent prompt is NOT included
        await interface._handle_chat_message(command)

        # Verify send_message was called with just user message
        interface.engine.send_message.assert_called_once()
        call_args = interface.engine.send_message.call_args[0][0]

        # Should only contain user message
        assert call_args == "Test user message"
        assert "SYSTEM: You are an autonomous AI agent" not in call_args


class TestSystemPromptSafetyRequirements:
    """Test what safety features should be added to the system prompt."""

    def test_safety_features_to_add(self):
        """Document what safety features should be added to the system prompt."""
        required_safety_features = [
            "Directory awareness with current working directory",
            "Git repository detection and context",
            "Read-before-write functionality for file modifications",
            "File existence checking before creation/overwrite",
            "User confirmation for file overwrites",
            "Automatic backup creation for file modifications",
            "Relative path awareness within git repositories",
        ]

        # This test documents what we need to implement
        assert len(required_safety_features) == 7

        # These features should be reflected in the updated prompt
        for feature in required_safety_features:
            assert isinstance(feature, str)
            assert len(feature) > 10  # Non-trivial descriptions
