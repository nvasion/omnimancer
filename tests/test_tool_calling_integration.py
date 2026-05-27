"""Integration tests for the native tool calling flow.

Tests the full path: CLI -> engine -> provider -> tool handler -> agent engine -> back.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.cli.interface import CommandLineInterface
from omnimancer.cli.tool_handler import MAX_TOOL_ITERATIONS, ToolHandler
from omnimancer.core.agent.types import Operation, OperationResult, OperationType
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.engine import CoreEngine
from omnimancer.core.models import (
    ChatResponse,
    ProviderConfig,
    ToolCall,
    ToolDefinition,
)


@pytest.fixture
def mock_engine():
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "config.json"

    config_manager = ConfigManager(str(config_path))
    config = config_manager.get_config()
    config.providers = {
        "test": ProviderConfig(
            model="test-model",
            api_key="test-key",
            provider_type="test",
            supports_tools=True,
            supports_multimodal=False,
        )
    }
    config.default_provider = "test"

    engine = MagicMock(spec=CoreEngine)
    engine.config_manager = config_manager
    engine.current_provider = None
    return engine


@pytest.fixture
def interface(mock_engine):
    return CommandLineInterface(mock_engine, no_approval=True)


class TestToolCallingFlowIntegration:
    """Test the _handle_tool_calling_flow method."""

    @pytest.mark.asyncio
    async def test_simple_tool_call_round_trip(self, interface, mock_engine):
        """Provider returns a tool call, we execute it, send result back, provider finishes."""
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="print('hello world')")
        )
        mock_engine.agent_engine = agent_engine

        # First call: provider returns a tool call
        first_response = ChatResponse(
            content="I'll read the file for you.",
            model_used="claude-sonnet-4",
            tokens_used=50,
            timestamp=datetime.now(),
            tool_calls=[
                ToolCall(name="file_read", arguments={"path": "/src/main.py"})
            ],
        )

        # Second call: provider finishes (no more tool calls)
        second_response = ChatResponse(
            content="The file contains a hello world program.",
            model_used="claude-sonnet-4",
            tokens_used=30,
            timestamp=datetime.now(),
            tool_calls=None,
        )

        mock_engine.send_message_with_tools = AsyncMock(
            side_effect=[first_response, second_response]
        )
        mock_engine.provider_supports_tools = MagicMock(return_value=True)

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("Read main.py")

        assert mock_engine.send_message_with_tools.call_count == 2
        agent_engine.execute_with_approval.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_only_response_no_loop(self, interface, mock_engine):
        """Provider returns text with no tool calls — no loop, just display."""
        mock_engine.agent_engine = MagicMock()

        response = ChatResponse(
            content="Python is a programming language.",
            model_used="claude-sonnet-4",
            tokens_used=20,
            timestamp=datetime.now(),
            tool_calls=None,
        )

        mock_engine.send_message_with_tools = AsyncMock(return_value=response)

        with patch.object(interface, "_show_assistant_message") as mock_show:
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("What is Python?")

        mock_show.assert_called_once_with(
            "Python is a programming language.", "claude-sonnet-4"
        )
        assert mock_engine.send_message_with_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_error_response_stops_flow(self, interface, mock_engine):
        """Engine returns an error — show error and stop."""
        mock_engine.agent_engine = MagicMock()

        error_response = ChatResponse(
            content="",
            model_used="",
            tokens_used=0,
            error="Rate limit exceeded",
        )

        mock_engine.send_message_with_tools = AsyncMock(return_value=error_response)

        with patch.object(interface, "_show_error") as mock_error:
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("Do something")

        mock_error.assert_called_once()
        assert "Rate limit" in mock_error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_response(self, interface, mock_engine):
        """Provider returns multiple tool calls at once."""
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="OK")
        )
        mock_engine.agent_engine = agent_engine

        first_response = ChatResponse(
            content="Reading files and running tests.",
            model_used="claude-sonnet-4",
            tokens_used=60,
            timestamp=datetime.now(),
            tool_calls=[
                ToolCall(name="file_read", arguments={"path": "/a.py"}),
                ToolCall(name="command_exec", arguments={"command": "pytest"}),
            ],
        )

        final_response = ChatResponse(
            content="All done!",
            model_used="claude-sonnet-4",
            tokens_used=10,
            timestamp=datetime.now(),
            tool_calls=None,
        )

        mock_engine.send_message_with_tools = AsyncMock(
            side_effect=[first_response, final_response]
        )

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("Read files and test")

        assert agent_engine.execute_with_approval.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_call_failure_reported_back(self, interface, mock_engine):
        """When a tool execution fails, the error is sent back to the provider."""
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=False, error="Permission denied")
        )
        mock_engine.agent_engine = agent_engine

        first_response = ChatResponse(
            content="Deleting the file.",
            model_used="claude-sonnet-4",
            tokens_used=20,
            timestamp=datetime.now(),
            tool_calls=[
                ToolCall(name="file_delete", arguments={"path": "/etc/passwd"})
            ],
        )

        final_response = ChatResponse(
            content="I couldn't delete that file.",
            model_used="claude-sonnet-4",
            tokens_used=15,
            timestamp=datetime.now(),
            tool_calls=None,
        )

        mock_engine.send_message_with_tools = AsyncMock(
            side_effect=[first_response, final_response]
        )

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("Delete /etc/passwd")

        # Second call should contain the error in the message
        second_call_msg = mock_engine.send_message_with_tools.call_args_list[1][0][0]
        assert "Permission denied" in second_call_msg

    @pytest.mark.asyncio
    async def test_max_iterations_guard(self, interface, mock_engine):
        """Flow stops after MAX_TOOL_ITERATIONS even if provider keeps requesting tools."""
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="OK")
        )
        mock_engine.agent_engine = agent_engine

        infinite_response = ChatResponse(
            content="Still working...",
            model_used="claude-sonnet-4",
            tokens_used=10,
            timestamp=datetime.now(),
            tool_calls=[
                ToolCall(name="file_read", arguments={"path": "/loop.py"})
            ],
        )

        mock_engine.send_message_with_tools = AsyncMock(
            return_value=infinite_response
        )

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface, "_show_warning") as mock_warn:
                with patch.object(interface.console, "print"):
                    await interface._handle_tool_calling_flow("Loop forever")

        assert mock_engine.send_message_with_tools.call_count == MAX_TOOL_ITERATIONS
        mock_warn.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_agent_engine_shows_error(self, interface, mock_engine):
        """When agent_engine is not available, show error."""
        mock_engine.agent_engine = None
        del mock_engine.agent_engine

        with patch.object(interface, "_show_error") as mock_error:
            await interface._handle_tool_calling_flow("Do something")

        mock_error.assert_called_once()
        assert "not available" in mock_error.call_args[0][0]


class TestChatMessageRouting:
    """Test that _handle_chat_message routes correctly between tool calling and markers."""

    @pytest.mark.asyncio
    async def test_agent_mode_with_tools_uses_tool_flow(self, interface, mock_engine):
        """When agent mode is on and provider supports tools, use tool calling."""
        from omnimancer.cli.commands import Command, CommandType

        interface.agent_manager = MagicMock()
        interface.agent_manager.mode.value = "on"
        mock_engine.provider_supports_tools = MagicMock(return_value=True)

        command = Command(
            type=CommandType.CHAT_MESSAGE,
            content="Read main.py",
            parameters={},
            raw_input="Read main.py",
        )

        with patch.object(
            interface, "_handle_tool_calling_flow", new_callable=AsyncMock
        ) as mock_tool_flow:
            with patch.object(
                interface, "_complete_approval_integration_setup", new_callable=AsyncMock
            ):
                with patch.object(interface, "_show_user_message"):
                    with patch.object(interface.cancellation_handler, "start_cancellable_operation") as mock_cancel:
                        # Make the cancellation handler just run the operation directly
                        async def run_op(**kwargs):
                            await kwargs["operation"]()
                        mock_cancel.side_effect = run_op

                        await interface._handle_chat_message(command)

        mock_tool_flow.assert_called_once_with("Read main.py")

    @pytest.mark.asyncio
    async def test_agent_mode_without_tools_uses_markers(self, interface, mock_engine):
        """When agent mode is on but provider doesn't support tools, use marker flow."""
        from omnimancer.cli.commands import Command, CommandType

        interface.agent_manager = MagicMock()
        interface.agent_manager.mode.value = "on"
        mock_engine.provider_supports_tools = MagicMock(return_value=False)

        command = Command(
            type=CommandType.CHAT_MESSAGE,
            content="Read main.py",
            parameters={},
            raw_input="Read main.py",
        )

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.content = "Here's the file content"
        mock_response.model_used = "test-model"
        mock_engine.send_message = AsyncMock(return_value=mock_response)

        with patch.object(
            interface, "_handle_tool_calling_flow", new_callable=AsyncMock
        ) as mock_tool_flow:
            with patch.object(
                interface, "_execute_continuous_workflow", new_callable=AsyncMock
            ) as mock_workflow:
                with patch.object(
                    interface, "_complete_approval_integration_setup", new_callable=AsyncMock
                ):
                    with patch.object(interface, "_show_user_message"):
                        with patch.object(interface.cancellation_handler, "start_cancellable_operation") as mock_cancel:
                            async def run_op(**kwargs):
                                await kwargs["operation"]()
                            mock_cancel.side_effect = run_op

                            await interface._handle_chat_message(command)

        mock_tool_flow.assert_not_called()
        mock_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_agent_mode_sends_plain_message(self, interface, mock_engine):
        """When agent mode is off, just send the message normally."""
        from omnimancer.cli.commands import Command, CommandType

        interface.agent_manager = None

        command = Command(
            type=CommandType.CHAT_MESSAGE,
            content="Hello",
            parameters={},
            raw_input="Hello",
        )

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.content = "Hi there!"
        mock_response.model_used = "test-model"
        mock_engine.send_message = AsyncMock(return_value=mock_response)

        with patch.object(interface, "_show_user_message"):
            with patch.object(interface, "_show_assistant_message") as mock_show:
                with patch.object(interface, "_show_token_status"):
                    with patch.object(interface.cancellation_handler, "start_cancellable_operation") as mock_cancel:
                        async def run_op(**kwargs):
                            await kwargs["operation"]()
                        mock_cancel.side_effect = run_op

                        await interface._handle_chat_message(command)

        mock_show.assert_called_once_with("Hi there!", "test-model")
