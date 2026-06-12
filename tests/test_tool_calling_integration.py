"""Integration tests for the native tool calling flow.

Tests the full path: CLI -> engine -> provider -> tool handler -> agent engine -> back.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.cli.interface import CommandLineInterface
from omnimancer.cli.tool_handler import MAX_TOOL_ITERATIONS
from omnimancer.core.agent.types import OperationResult
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.engine import CoreEngine
from omnimancer.core.models import ChatResponse, ProviderConfig, ToolCall


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
        """Provider returns tool call; we execute and finish."""
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
            tool_calls=[ToolCall(name="file_read", arguments={"path": "/src/main.py"})],
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
    async def test_tool_results_labeled_with_arguments(self, interface, mock_engine):
        """Results fed back to the model identify the call they belong to.

        A bare '[Read] Result: ...' gives the model no way to tell which file
        a result came from (or that it already read it), so weak models
        re-issue the same call until the loop detector kills the turn.
        """
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="file contents")
        )
        mock_engine.agent_engine = agent_engine

        first_response = ChatResponse(
            content="",
            model_used="m",
            tokens_used=10,
            timestamp=datetime.now(),
            tool_calls=[ToolCall(name="file_read", arguments={"path": "/src/main.py"})],
        )
        second_response = ChatResponse(
            content="Done.",
            model_used="m",
            tokens_used=5,
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

        results_message = mock_engine.send_message_with_tools.call_args_list[1][0][0]
        assert "/src/main.py" in results_message
        assert "file contents" in results_message

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
    async def test_cancellation_drops_back_to_prompt(self, interface, mock_engine):
        """'q' at the approval prompt ends the whole turn.

        Regression: cancellation was fed back to the model as an ordinary
        tool error, so the model just tried another command and the user
        got prompted again instead of getting their prompt back.
        """
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(
                success=False, error="User rejected", was_cancelled=True
            )
        )
        mock_engine.agent_engine = agent_engine

        response = ChatResponse(
            content="",
            model_used="m",
            tokens_used=10,
            timestamp=datetime.now(),
            tool_calls=[ToolCall(name="command_exec", arguments={"command": "ls"})],
        )
        mock_engine.send_message_with_tools = AsyncMock(side_effect=[response])

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("list files")

        # No second round trip: the cancellation is not sent back to the model.
        assert mock_engine.send_message_with_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_native_tool_history_branch(self, interface, mock_engine):
        """Native providers get structured results, not flattened text.

        The continuation request must carry tool results as recorded
        structured history (assistant.tool_calls + role:"tool") with an
        empty user message — flattened "Tool results:" text violates the
        chat template and makes models leak template tokens as text.
        """
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="auth code")
        )
        mock_engine.agent_engine = agent_engine

        first = ChatResponse(
            content="",
            model_used="m",
            tokens_used=10,
            timestamp=datetime.now(),
            tool_calls=[
                ToolCall(name="file_read", arguments={"path": "/a"}, id="call_9")
            ],
        )
        final = ChatResponse(
            content="Done.",
            model_used="m",
            tokens_used=5,
            timestamp=datetime.now(),
            tool_calls=None,
        )
        mock_engine.send_message_with_tools = AsyncMock(side_effect=[first, final])
        mock_engine.provider_supports_tools = MagicMock(return_value=True)
        mock_engine.provider_supports_native_tool_history = MagicMock(return_value=True)
        mock_engine.record_tool_results = MagicMock()

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("read a")

        # Results recorded structurally, paired to the call's id.
        mock_engine.record_tool_results.assert_called_once()
        text, records = mock_engine.record_tool_results.call_args[0]
        assert "Tool results" in text
        assert records[0].tool_call_id == "call_9"
        assert records[0].content == "auth code"
        # Continuation request sends no flattened results text.
        assert mock_engine.send_message_with_tools.call_args_list[1][0][0] == ""

    @pytest.mark.asyncio
    async def test_text_mimicked_tool_call_is_recovered(self, interface, mock_engine):
        """A '[Called tools: ...]' emitted as TEXT still executes.

        Regression: weak models mimic the history notation instead of issuing
        native tool calls; the turn ended silently at the prompt instead of
        running the call the model clearly intended.
        """
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="src/auth.ts")
        )
        mock_engine.agent_engine = agent_engine

        mimicked = ChatResponse(
            content='[Called tools: Grep({"pattern": "auth", "glob": "*.ts"})]',
            model_used="qwen",
            tokens_used=10,
            timestamp=datetime.now(),
            tool_calls=None,
        )
        final = ChatResponse(
            content="Found it in src/auth.ts.",
            model_used="qwen",
            tokens_used=5,
            timestamp=datetime.now(),
            tool_calls=None,
        )
        mock_engine.send_message_with_tools = AsyncMock(side_effect=[mimicked, final])
        mock_engine.provider_supports_tools = MagicMock(return_value=True)

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface.console, "print"):
                await interface._handle_tool_calling_flow("find the auth code")

        agent_engine.execute_with_approval.assert_called_once()
        assert mock_engine.send_message_with_tools.call_count == 2

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

    def _endless_tool_responses(self, mock_engine):
        """Distinct tool calls forever, so only the iteration logic stops us."""
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="OK")
        )
        mock_engine.agent_engine = agent_engine

        def make_response(*_args, **_kwargs):
            make_response.n += 1
            return ChatResponse(
                content="Still working...",
                model_used="claude-sonnet-4",
                tokens_used=10,
                timestamp=datetime.now(),
                tool_calls=[
                    ToolCall(
                        name="Read",
                        arguments={"file_path": f"/loop{make_response.n}.py"},
                    )
                ],
            )

        make_response.n = 0
        mock_engine.send_message_with_tools = AsyncMock(side_effect=make_response)

    @pytest.mark.asyncio
    async def test_no_iteration_checkin_in_interactive(self, interface, mock_engine):
        """Long turns run past MAX_TOOL_ITERATIONS without prompting the user.

        Interactive mode has Ctrl+C as the escape hatch and the repeat
        tracker for actual runaway loops — a periodic "keep going?" prompt
        only interrupts legitimate long-running work. Headless mode keeps
        its own hard cap.
        """
        total = MAX_TOOL_ITERATIONS + 5
        self._endless_tool_responses(mock_engine)
        endless = mock_engine.send_message_with_tools.side_effect

        def finite_responses(*args, **kwargs):
            response = endless(*args, **kwargs)
            if mock_engine.send_message_with_tools.call_count >= total:
                response.tool_calls = []
            return response

        mock_engine.send_message_with_tools = AsyncMock(side_effect=finite_responses)

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface, "_show_warning"):
                with patch.object(interface.console, "print"):
                    with patch.object(interface.console, "input") as console_input:
                        await interface._handle_tool_calling_flow("Loop a while")

        console_input.assert_not_called()
        assert mock_engine.send_message_with_tools.call_count == total

    @pytest.mark.asyncio
    async def test_repeated_tool_call_aborts_after_ignored_nudges(
        self, interface, mock_engine
    ):
        """A call repeated forever is nudged twice, then the turn aborts.

        Occurrences 1-2 execute (re-runs can be legitimate), 3-4 are skipped
        with a corrective nudge, and the 5th aborts with a warning naming
        the offending call.
        """
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="OK")
        )
        mock_engine.agent_engine = agent_engine

        repeated = ChatResponse(
            content="",
            model_used="claude-sonnet-4",
            tokens_used=10,
            timestamp=datetime.now(),
            tool_calls=[ToolCall(name="Write", arguments={"file_path": "/same.txt"})],
        )
        mock_engine.send_message_with_tools = AsyncMock(return_value=repeated)

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface, "_show_warning") as mock_warn:
                with patch.object(interface.console, "print"):
                    await interface._handle_tool_calling_flow("loop")

        assert mock_engine.send_message_with_tools.call_count == 5
        # Only the first two occurrences actually executed.
        assert agent_engine.execute_with_approval.call_count == 2
        mock_warn.assert_called_once()
        warning = mock_warn.call_args[0][0]
        assert "Write" in warning

    @pytest.mark.asyncio
    async def test_duplicate_call_nudged_then_model_recovers(
        self, interface, mock_engine
    ):
        """The 3rd identical call is skipped with a nudge, not executed —
        and the turn continues so the model can still finish."""
        agent_engine = MagicMock()
        agent_engine.execute_with_approval = AsyncMock(
            return_value=OperationResult(success=True, data="OK")
        )
        mock_engine.agent_engine = agent_engine

        same_call = ChatResponse(
            content="",
            model_used="m",
            tokens_used=10,
            timestamp=datetime.now(),
            tool_calls=[ToolCall(name="Read", arguments={"file_path": "/a.py"})],
        )
        final = ChatResponse(
            content="Here is my answer.",
            model_used="m",
            tokens_used=5,
            timestamp=datetime.now(),
            tool_calls=None,
        )
        mock_engine.send_message_with_tools = AsyncMock(
            side_effect=[same_call, same_call, same_call, final]
        )

        with patch.object(interface, "_show_assistant_message"):
            with patch.object(interface, "_show_warning") as mock_warn:
                with patch.object(interface.console, "print"):
                    await interface._handle_tool_calling_flow("loop")

        # 3rd occurrence skipped: only 2 executions, no abort, turn completed.
        assert agent_engine.execute_with_approval.call_count == 2
        assert mock_engine.send_message_with_tools.call_count == 4
        mock_warn.assert_not_called()
        # The model was told the call was a duplicate.
        nudge_message = mock_engine.send_message_with_tools.call_args_list[3][0][0]
        assert "Duplicate" in nudge_message

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
    """Test _handle_chat_message routes between tool calling and markers."""

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
            interface,
            "_handle_tool_calling_flow",
            new_callable=AsyncMock,
        ) as mock_tool_flow:
            with patch.object(
                interface,
                "_complete_approval_integration_setup",
                new_callable=AsyncMock,
            ):
                with patch.object(interface, "_show_user_message"):
                    cancel_handler = interface.cancellation_handler
                    with patch.object(
                        cancel_handler,
                        "start_cancellable_operation",
                    ) as mock_cancel:

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
            interface,
            "_handle_tool_calling_flow",
            new_callable=AsyncMock,
        ) as mock_tool_flow:
            with patch.object(
                interface,
                "_execute_continuous_workflow",
                new_callable=AsyncMock,
            ) as mock_workflow:
                with patch.object(
                    interface,
                    "_complete_approval_integration_setup",
                    new_callable=AsyncMock,
                ):
                    with patch.object(interface, "_show_user_message"):
                        cancel_handler = interface.cancellation_handler
                        with patch.object(
                            cancel_handler,
                            "start_cancellable_operation",
                        ) as mock_cancel:

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
                    cancel_handler = interface.cancellation_handler
                    with patch.object(
                        cancel_handler,
                        "start_cancellable_operation",
                    ) as mock_cancel:

                        async def run_op(**kwargs):
                            await kwargs["operation"]()

                        mock_cancel.side_effect = run_op

                        await interface._handle_chat_message(command)

        mock_show.assert_called_once_with("Hi there!", "test-model")
