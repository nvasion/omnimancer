"""
Tests for agent persona CLI handler.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from rich.console import Console

from omnimancer.cli.handlers.agent_persona_handler import AgentPersonaHandler
from omnimancer.cli.commands import Command, CommandType, SlashCommand
from omnimancer.core.agent.persona import (
    PersonaManager,
    AgentPersona,
    PersonaStatus,
    PersonaCategory,
    CodingPersona,
    ResearchPersona,
    GeneralPersona,
)
from omnimancer.core.agent.agent_switcher import (
    AgentSwitcher,
    SessionState,
    SwitchState,
)


class TestAgentPersonaHandler:
    """Test AgentPersonaHandler class."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console."""
        console = Mock(spec=Console)
        console.print = Mock()
        return console

    @pytest.fixture
    def mock_persona_manager(self):
        """Create a mock PersonaManager."""
        manager = Mock(spec=PersonaManager)

        # Create mock personas
        coding_persona = Mock(spec=AgentPersona)
        coding_persona.id = "coding"
        coding_persona.name = "Coding Agent"
        coding_persona.icon = "💻"
        coding_persona.category = PersonaCategory.DEVELOPMENT
        coding_persona.status = PersonaStatus.AVAILABLE
        coding_persona.description = "Optimized for software development"
        coding_persona.capabilities = set()
        coding_persona.configuration = Mock()
        coding_persona.configuration.template_id = "coding"
        coding_persona.configuration.primary_provider = "claude"
        coding_persona.configuration.fallback_providers = ["openai", "gemini"]
        coding_persona.configuration.tools_enabled = True
        coding_persona.configuration.web_search_enabled = False
        coding_persona.configuration.file_operations_enabled = True
        coding_persona.configuration.approval_required = True
        # Create a simple metadata object without Mock complications
        coding_persona.metadata = type(
            "MockMetadata",
            (),
            {
                "usage_count": 5,
                "last_used": None,
                "version": "1.0.0",
                "author": "Omnimancer",
                "is_builtin": True,
                "created_at": None,
                "tags": [],
                "__dict__": {
                    "usage_count": 5,
                    "last_used": None,
                    "version": "1.0.0",
                    "author": "Omnimancer",
                    "is_builtin": True,
                    "created_at": None,
                    "tags": [],
                },
            },
        )()

        research_persona = Mock(spec=AgentPersona)
        research_persona.id = "research"
        research_persona.name = "Research Agent"
        research_persona.icon = "🔍"
        research_persona.category = PersonaCategory.RESEARCH
        research_persona.status = PersonaStatus.AVAILABLE
        research_persona.description = "Configured for research with web search"
        research_persona.capabilities = set()
        research_persona.configuration = Mock()
        # Create a simple metadata object for research persona too
        research_persona.metadata = type(
            "MockMetadata",
            (),
            {
                "usage_count": 2,
                "last_used": None,
                "version": "1.0.0",
                "author": "Omnimancer",
                "is_builtin": True,
                "created_at": None,
                "tags": [],
                "__dict__": {
                    "usage_count": 2,
                    "last_used": None,
                    "version": "1.0.0",
                    "author": "Omnimancer",
                    "is_builtin": True,
                    "created_at": None,
                    "tags": [],
                },
            },
        )()

        manager.get_all_personas = Mock(
            return_value={"coding": coding_persona, "research": research_persona}
        )

        manager.get_persona = Mock(
            side_effect=lambda pid: {
                "coding": coding_persona,
                "research": research_persona,
            }.get(pid)
        )

        manager.active_persona = None
        manager.get_stats = Mock(
            return_value={
                "total_personas": 2,
                "builtin_personas": 2,
                "custom_personas": 0,
                "active_persona": None,
                "available_personas": 2,
            }
        )

        return manager

    @pytest.fixture
    def mock_agent_switcher(self, mock_persona_manager):
        """Create a mock AgentSwitcher."""
        switcher = Mock(spec=AgentSwitcher)
        switcher.persona_manager = mock_persona_manager
        switcher.switch_persona = Mock(return_value=(True, "Switched successfully"))
        switcher.get_current_state = Mock(return_value=SwitchState.IDLE)
        switcher.current_session_state = SessionState()
        switcher.get_switch_history = Mock(return_value=[])
        return switcher

    @pytest.fixture
    def handler(self, mock_console, mock_persona_manager, mock_agent_switcher):
        """Create handler with mocks."""
        with patch(
            "omnimancer.cli.handlers.agent_persona_handler.get_persona_manager",
            return_value=mock_persona_manager,
        ), patch(
            "omnimancer.cli.handlers.agent_persona_handler.get_agent_switcher",
            return_value=mock_agent_switcher,
        ):
            handler = AgentPersonaHandler(mock_console)
            handler.persona_manager = mock_persona_manager
            handler.agent_switcher = mock_agent_switcher
            return handler

    @pytest.mark.asyncio
    async def test_handle_list_personas(self, handler, mock_console):
        """Test listing personas."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["list"]},
            raw_input="/agent list",
        )

        await handler.handle_agent_command(command)

        # Verify console output was called
        assert mock_console.print.called
        # Verify personas were retrieved
        assert handler.persona_manager.get_all_personas.called

    @pytest.mark.asyncio
    async def test_handle_list_personas_default(self, handler, mock_console):
        """Test that list is default when no subcommand."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": []},
            raw_input="/agent",
        )

        await handler.handle_agent_command(command)

        # Should default to list
        assert handler.persona_manager.get_all_personas.called

    @pytest.mark.asyncio
    async def test_handle_use_persona_success(self, handler, mock_agent_switcher):
        """Test successful persona switch."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["use", "coding"]},
            raw_input="/agent use coding",
        )

        await handler.handle_agent_command(command)

        # Verify switch was attempted
        mock_agent_switcher.switch_persona.assert_called_with(
            "coding", reason="User requested switch via CLI"
        )

    @pytest.mark.asyncio
    async def test_handle_use_persona_not_found(self, handler, mock_console):
        """Test using non-existent persona."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["use", "nonexistent"]},
            raw_input="/agent use nonexistent",
        )

        await handler.handle_agent_command(command)

        # Should show error
        error_calls = [
            call
            for call in mock_console.print.call_args_list
            if "not found" in str(call).lower()
        ]
        assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_use_persona_already_active(
        self, handler, mock_persona_manager, mock_console
    ):
        """Test switching to already active persona."""
        coding_persona = mock_persona_manager.get_persona("coding")
        mock_persona_manager.active_persona = coding_persona

        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["use", "coding"]},
            raw_input="/agent use coding",
        )

        await handler.handle_agent_command(command)

        # Should show already using message
        info_calls = [
            call
            for call in mock_console.print.call_args_list
            if "already using" in str(call).lower()
        ]
        assert len(info_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_current_persona_active(
        self, handler, mock_persona_manager, mock_console
    ):
        """Test showing current active persona."""
        coding_persona = mock_persona_manager.get_persona("coding")
        mock_persona_manager.active_persona = coding_persona

        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["current"]},
            raw_input="/agent current",
        )

        await handler.handle_agent_command(command)

        # Should display current persona info
        assert mock_console.print.called
        # Check that a Panel was printed (current persona display uses Panel)
        call_args = mock_console.print.call_args_list
        assert len(call_args) > 0
        # Verify it's displaying a panel (for current persona)
        from rich.panel import Panel

        assert any(isinstance(call[0][0], Panel) for call in call_args if call[0])

    @pytest.mark.asyncio
    async def test_handle_current_persona_none(self, handler, mock_console):
        """Test showing current when no persona active."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["current"]},
            raw_input="/agent current",
        )

        await handler.handle_agent_command(command)

        # Should show no active persona message
        info_calls = [
            call
            for call in mock_console.print.call_args_list
            if "no agent persona" in str(call).lower()
        ]
        assert len(info_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_persona_info(self, handler, mock_console):
        """Test showing detailed persona info."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["info", "coding"]},
            raw_input="/agent info coding",
        )

        await handler.handle_agent_command(command)

        # Should display detailed info
        assert mock_console.print.called
        handler.persona_manager.get_persona.assert_called_with("coding")

    @pytest.mark.asyncio
    async def test_handle_persona_info_not_found(self, handler, mock_console):
        """Test info for non-existent persona."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["info", "nonexistent"]},
            raw_input="/agent info nonexistent",
        )

        await handler.handle_agent_command(command)

        # Should show error
        error_calls = [
            call
            for call in mock_console.print.call_args_list
            if "not found" in str(call).lower()
        ]
        assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_persona_status(self, handler, mock_console):
        """Test showing detailed status."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["status"]},
            raw_input="/agent status",
        )

        await handler.handle_agent_command(command)

        # Should display status
        assert mock_console.print.called
        assert handler.persona_manager.get_stats.called
        assert handler.agent_switcher.get_current_state.called

    @pytest.mark.asyncio
    async def test_handle_switch_history(
        self, handler, mock_agent_switcher, mock_console
    ):
        """Test showing switch history."""
        # Create mock history
        from datetime import datetime

        mock_context = Mock()
        mock_context.from_persona = None
        mock_context.to_persona = Mock()
        mock_context.to_persona.name = "Coding Agent"
        mock_context.timestamp = datetime.now()
        mock_context.switch_reason = "Test switch"
        mock_context.error_message = None

        mock_agent_switcher.get_switch_history.return_value = [mock_context]

        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["history"]},
            raw_input="/agent history",
        )

        await handler.handle_agent_command(command)

        # Should display history
        assert mock_console.print.called
        assert mock_agent_switcher.get_switch_history.called

    @pytest.mark.asyncio
    async def test_handle_switch_history_empty(self, handler, mock_console):
        """Test showing empty switch history."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["history"]},
            raw_input="/agent history",
        )

        await handler.handle_agent_command(command)

        # Should show no history message
        info_calls = [
            call
            for call in mock_console.print.call_args_list
            if "no persona switch history" in str(call).lower()
        ]
        assert len(info_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_help_command(self, handler, mock_console):
        """Test showing help."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["help"]},
            raw_input="/agent help",
        )

        await handler.handle_agent_command(command)

        # Should display help
        assert mock_console.print.called
        # Check that a Panel was printed (help uses Panel)
        call_args = mock_console.print.call_args_list
        assert len(call_args) > 0
        from rich.panel import Panel

        assert any(isinstance(call[0][0], Panel) for call in call_args if call[0])

    @pytest.mark.asyncio
    async def test_handle_unknown_subcommand(self, handler, mock_console):
        """Test unknown subcommand."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["unknown"]},
            raw_input="/agent unknown",
        )

        await handler.handle_agent_command(command)

        # Should show error
        error_calls = [
            call
            for call in mock_console.print.call_args_list
            if "unknown" in str(call).lower()
        ]
        assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_missing_args_for_use(self, handler, mock_console):
        """Test use command without persona argument."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["use"]},
            raw_input="/agent use",
        )

        await handler.handle_agent_command(command)

        # Should show usage error
        error_calls = [
            call
            for call in mock_console.print.call_args_list
            if "usage:" in str(call).lower()
        ]
        assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_missing_args_for_info(self, handler, mock_console):
        """Test info command without persona argument."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["info"]},
            raw_input="/agent info",
        )

        await handler.handle_agent_command(command)

        # Should show usage error
        error_calls = [
            call
            for call in mock_console.print.call_args_list
            if "usage:" in str(call).lower()
        ]
        assert len(error_calls) > 0

    def test_format_status(self, handler):
        """Test status formatting."""
        assert "Available" in handler._format_status(PersonaStatus.AVAILABLE)
        assert "Active" in handler._format_status(PersonaStatus.ACTIVE)
        assert "Disabled" in handler._format_status(PersonaStatus.DISABLED)
        assert "Error" in handler._format_status(PersonaStatus.ERROR)
        assert "Loading" in handler._format_status(PersonaStatus.LOADING)

    def test_format_switch_state(self, handler):
        """Test switch state formatting."""
        assert "Idle" in handler._format_switch_state(SwitchState.IDLE)
        assert "Preparing" in handler._format_switch_state(SwitchState.PREPARING)
        assert "Switching" in handler._format_switch_state(SwitchState.SWITCHING)
        assert "Complete" in handler._format_switch_state(SwitchState.COMPLETE)
        assert "Error" in handler._format_switch_state(SwitchState.ERROR)

    def test_suggest_personas(self, handler, mock_console):
        """Test persona suggestions."""
        handler._suggest_personas("cod")

        # Should suggest coding persona
        suggest_calls = [
            call
            for call in mock_console.print.call_args_list
            if "coding" in str(call).lower()
        ]
        assert len(suggest_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_recommend_persona_no_query(self, handler, mock_console):
        """Test persona recommendations without query."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["recommend"]},
            raw_input="/agent recommend",
        )

        await handler.handle_agent_command(command)

        # Should show general recommendations
        assert mock_console.print.called
        recommend_calls = [
            call
            for call in mock_console.print.call_args_list
            if "recommendations" in str(call).lower()
        ]
        assert len(recommend_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_recommend_persona_with_query(self, handler, mock_console):
        """Test persona recommendations with specific query."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["recommend", "debug", "python", "code"]},
            raw_input="/agent recommend debug python code",
        )

        await handler.handle_agent_command(command)

        # Should show coding recommendations
        assert mock_console.print.called
        coding_calls = [
            call
            for call in mock_console.print.call_args_list
            if "coding" in str(call).lower() or "programming" in str(call).lower()
        ]
        assert len(coding_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_compare_personas(self, handler, mock_console):
        """Test persona capability comparison."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["compare"]},
            raw_input="/agent compare",
        )

        await handler.handle_agent_command(command)

        # Should show comparison table
        assert mock_console.print.called
        # Check that a Table was printed (comparison uses Table)
        call_args = mock_console.print.call_args_list
        from rich.table import Table
        from rich.panel import Panel

        table_calls = [
            call for call in call_args if call[0] and isinstance(call[0][0], Table)
        ]
        panel_calls = [
            call for call in call_args if call[0] and isinstance(call[0][0], Panel)
        ]
        assert (
            len(table_calls) > 0 or len(panel_calls) > 0
        )  # Either table or legend panel

    @pytest.mark.asyncio
    async def test_handle_preview_persona_success(self, handler, mock_console):
        """Test successful persona preview."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["preview", "coding"]},
            raw_input="/agent preview coding",
        )

        await handler.handle_agent_command(command)

        # Should show preview panels
        assert mock_console.print.called
        preview_calls = [
            call
            for call in mock_console.print.call_args_list
            if "preview" in str(call).lower()
        ]
        assert len(preview_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_preview_persona_not_found(self, handler, mock_console):
        """Test preview for non-existent persona."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["preview", "nonexistent"]},
            raw_input="/agent preview nonexistent",
        )

        await handler.handle_agent_command(command)

        # Should show error
        error_calls = [
            call
            for call in mock_console.print.call_args_list
            if "not found" in str(call).lower()
        ]
        assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_handle_discover_personas(self, handler, mock_console):
        """Test interactive persona discovery."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["discover"]},
            raw_input="/agent discover",
        )

        await handler.handle_agent_command(command)

        # Should show discovery interface
        assert mock_console.print.called
        discover_calls = [
            call
            for call in mock_console.print.call_args_list
            if "discovery" in str(call).lower()
        ]
        assert len(discover_calls) > 0

    @pytest.mark.asyncio
    async def test_missing_args_for_preview(self, handler, mock_console):
        """Test preview command without persona argument."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["preview"]},
            raw_input="/agent preview",
        )

        await handler.handle_agent_command(command)

        # Should show usage error
        error_calls = [
            call
            for call in mock_console.print.call_args_list
            if "usage:" in str(call).lower()
        ]
        assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_recommend_with_research_query(self, handler, mock_console):
        """Test recommendations for research-related query."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["recommend", "research", "find", "information"]},
            raw_input="/agent recommend research find information",
        )

        await handler.handle_agent_command(command)

        # Should recommend research persona
        assert mock_console.print.called
        research_calls = [
            call
            for call in mock_console.print.call_args_list
            if "research" in str(call).lower()
        ]
        assert len(research_calls) > 0

    @pytest.mark.asyncio
    async def test_recommend_with_creative_query(self, handler, mock_console):
        """Test recommendations for creative writing query."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["recommend", "write", "story", "creative"]},
            raw_input="/agent recommend write story creative",
        )

        await handler.handle_agent_command(command)

        # Should recommend creative persona if available
        assert mock_console.print.called
        # At minimum should show some recommendation
        recommend_calls = [
            call
            for call in mock_console.print.call_args_list
            if any(
                word in str(call).lower() for word in ["creative", "writing", "general"]
            )
        ]
        assert len(recommend_calls) > 0

    @pytest.mark.asyncio
    async def test_recommend_with_performance_query(self, handler, mock_console):
        """Test recommendations for speed/performance query."""
        command = Command(
            type=CommandType.SLASH_COMMAND,
            content="/agent",
            parameters={"args": ["recommend", "fast", "quick", "urgent"]},
            raw_input="/agent recommend fast quick urgent",
        )

        await handler.handle_agent_command(command)

        # Should recommend performance persona if available
        assert mock_console.print.called
        # At minimum should show some recommendation
        recommend_calls = [
            call
            for call in mock_console.print.call_args_list
            if any(
                word in str(call).lower()
                for word in ["performance", "speed", "fast", "general"]
            )
        ]
        assert len(recommend_calls) > 0
