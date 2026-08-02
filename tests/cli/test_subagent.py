"""Tests for the subagent runner (scoped child agents)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.cli.subagent import SubAgentRunner
from omnimancer.core.agent.status_core import EventType
from omnimancer.core.models import (
    ChatResponse,
    SubAgentDefinition,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from omnimancer.events import emitter as fleet_events


def _response(content="", tool_calls=None, error=None):
    return ChatResponse(
        content=content,
        model_used="m",
        tokens_used=0,
        tool_calls=tool_calls,
        error=error,
    )


def _engine(provider):
    engine = MagicMock()
    engine.current_provider = provider
    engine.agent_engine = MagicMock()
    return engine


def _fake_tool_handler(tools, exec_result=None):
    th = MagicMock()
    th.get_tool_definitions.return_value = tools
    th.execute_tool_call = AsyncMock(
        return_value=exec_result or ToolResult(content="ok")
    )
    return th


TOOLS = [
    ToolDefinition(name="Read", description="", parameters={}),
    ToolDefinition(name="Bash", description="", parameters={}),
]


class TestSubAgentRunner:
    @pytest.mark.asyncio
    async def test_no_provider_returns_error(self):
        runner = SubAgentRunner(_engine(None))
        result = await runner.run(SubAgentDefinition(name="x"), "task")
        assert result.success is False
        assert "provider" in result.error.lower()

    @pytest.mark.asyncio
    async def test_returns_final_output(self):
        provider = MagicMock()
        provider.model = "base"
        provider.send_message_with_tools = AsyncMock(
            return_value=_response(content="done")
        )
        runner = SubAgentRunner(_engine(provider))
        with patch(
            "omnimancer.cli.subagent.ToolHandler",
            return_value=_fake_tool_handler(TOOLS),
        ):
            result = await runner.run(SubAgentDefinition(name="x"), "task")
        assert result.success is True
        assert result.output == "done"
        assert result.iterations == 1
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_tool_allowlist_scopes_tools(self):
        provider = MagicMock()
        provider.model = "base"
        provider.send_message_with_tools = AsyncMock(return_value=_response("ok"))
        runner = SubAgentRunner(_engine(provider))
        defn = SubAgentDefinition(name="x", tools=["Read"])
        with patch(
            "omnimancer.cli.subagent.ToolHandler",
            return_value=_fake_tool_handler(TOOLS),
        ):
            await runner.run(defn, "task")
        # The tools passed to the provider are scoped to the allowlist.
        passed_tools = provider.send_message_with_tools.call_args[0][2]
        assert [t.name for t in passed_tools] == ["Read"]

    @pytest.mark.asyncio
    async def test_tool_call_loop_and_isolation(self):
        provider = MagicMock()
        provider.model = "base"
        provider.send_message_with_tools = AsyncMock(
            side_effect=[
                _response(tool_calls=[ToolCall(name="Read", arguments={})]),
                _response(content="final answer"),
            ]
        )
        engine = _engine(provider)
        runner = SubAgentRunner(engine)
        with patch(
            "omnimancer.cli.subagent.ToolHandler",
            return_value=_fake_tool_handler(TOOLS, ToolResult(content="data")),
        ):
            result = await runner.run(SubAgentDefinition(name="x"), "task")
        assert result.success is True
        assert result.output == "final answer"
        assert result.tool_calls == ["Read"]
        assert result.iterations == 2
        # Isolation: the parent engine's conversation path is untouched.
        engine.send_message_with_tools.assert_not_called()
        engine.chat_manager.add_user_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_override_restored(self):
        provider = MagicMock()
        provider.model = "base"
        provider.send_message_with_tools = AsyncMock(return_value=_response("ok"))
        runner = SubAgentRunner(_engine(provider))
        defn = SubAgentDefinition(name="x", model="special")
        with patch(
            "omnimancer.cli.subagent.ToolHandler",
            return_value=_fake_tool_handler(TOOLS),
        ):
            await runner.run(defn, "task")
        assert provider.model == "base"  # restored after run

    @pytest.mark.asyncio
    async def test_max_iterations_cap(self):
        provider = MagicMock()
        provider.model = "base"
        # Always asks for another tool call → should stop at max_iterations.
        provider.send_message_with_tools = AsyncMock(
            return_value=_response(tool_calls=[ToolCall(name="Read", arguments={})])
        )
        runner = SubAgentRunner(_engine(provider))
        defn = SubAgentDefinition(name="loop", max_iterations=3)
        with patch(
            "omnimancer.cli.subagent.ToolHandler",
            return_value=_fake_tool_handler(TOOLS, ToolResult(content="x")),
        ):
            result = await runner.run(defn, "task")
        assert result.iterations == 3
        assert len(result.tool_calls) == 3

    @pytest.mark.asyncio
    async def test_provider_error_surfaces(self):
        provider = MagicMock()
        provider.model = "base"
        provider.send_message_with_tools = AsyncMock(
            return_value=_response(error="boom")
        )
        runner = SubAgentRunner(_engine(provider))
        with patch(
            "omnimancer.cli.subagent.ToolHandler",
            return_value=_fake_tool_handler(TOOLS),
        ):
            result = await runner.run(SubAgentDefinition(name="x"), "task")
        assert result.success is False
        assert result.error == "boom"

    @pytest.mark.asyncio
    async def test_subagent_emits_lifecycle_events_with_model(self):
        provider = MagicMock()
        provider.model = "base"
        provider.send_message_with_tools = AsyncMock(
            return_value=_response(content="done")
        )
        engine = _engine(provider)
        engine.providers = {"openai": provider}
        runner = SubAgentRunner(engine)
        defn = SubAgentDefinition(name="researcher", model="qwen3-4b")

        calls = []

        async def _spy(event_type, data, *args, **kwargs):
            calls.append(
                (
                    event_type,
                    data,
                    fleet_events.current_agent_id(),
                    fleet_events.current_parent_id(),
                )
            )

        with patch(
            "omnimancer.cli.subagent.ToolHandler",
            return_value=_fake_tool_handler(TOOLS),
        ):
            with patch.object(fleet_events, "emit_event", _spy):
                result = await runner.run(defn, "task")

        start_calls = [c for c in calls if c[0] == EventType.SESSION_START]
        end_calls = [c for c in calls if c[0] == EventType.SESSION_END]
        assert len(start_calls) == 1
        assert start_calls[0][1]["model"] == "qwen3-4b"
        assert start_calls[0][1]["subagent"] == "researcher"
        assert "provider" in start_calls[0][1]
        assert start_calls[0][2].startswith("subagent-researcher-")
        assert len(end_calls) == 1
        assert end_calls[0][1]["reason"] == "subagent_complete"
        assert end_calls[0][1]["status"] == 0
        assert result.model == "qwen3-4b"

    @pytest.mark.asyncio
    async def test_subagent_emits_session_end_on_provider_error(self):
        provider = MagicMock()
        provider.model = "base"
        provider.send_message_with_tools = AsyncMock(
            return_value=_response(error="boom")
        )
        engine = _engine(provider)
        engine.providers = {"openai": provider}
        runner = SubAgentRunner(engine)

        calls = []

        async def _spy(event_type, data, *args, **kwargs):
            calls.append((event_type, data))

        with patch(
            "omnimancer.cli.subagent.ToolHandler",
            return_value=_fake_tool_handler(TOOLS),
        ):
            with patch.object(fleet_events, "emit_event", _spy):
                result = await runner.run(
                    SubAgentDefinition(name="x", model="qwen3-4b"), "task"
                )

        end_calls = [c for c in calls if c[0] == EventType.SESSION_END]
        assert len(end_calls) == 1
        assert end_calls[0][1]["reason"] == "subagent_complete"
        assert end_calls[0][1]["status"] == 1
        assert result.success is False
        assert result.model == "qwen3-4b"

    @pytest.mark.asyncio
    async def test_subagent_emits_session_end_on_exception(self):
        provider = MagicMock()
        provider.model = "base"
        provider.send_message_with_tools = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )
        engine = _engine(provider)
        engine.providers = {"openai": provider}
        runner = SubAgentRunner(engine)

        calls = []

        async def _spy(event_type, data, *args, **kwargs):
            calls.append((event_type, data))

        with patch(
            "omnimancer.cli.subagent.ToolHandler",
            return_value=_fake_tool_handler(TOOLS),
        ):
            with patch.object(fleet_events, "emit_event", _spy):
                result = await runner.run(
                    SubAgentDefinition(name="x", model="qwen3-4b"), "task"
                )

        end_calls = [c for c in calls if c[0] == EventType.SESSION_END]
        assert len(end_calls) == 1
        assert end_calls[0][1]["reason"] == "subagent_complete"
        assert end_calls[0][1]["status"] == 1
        assert result.success is False


class TestSubAgentsCommand:
    """The /subagents CLI command (list + run)."""

    def _disp(self, subagents):
        from omnimancer.cli.command_dispatch import CommandDispatchMixin
        from omnimancer.cli.commands import Command, SlashCommand
        from omnimancer.core.models import Config, ProviderConfig

        config = Config(
            default_provider="openai",
            providers={"openai": ProviderConfig(api_key="k", model="gpt-4")},
            storage_path="/tmp/omnimancer_subagent_test",
            subagents=subagents,
        )

        class _D(CommandDispatchMixin):
            def __init__(self):
                self.engine = MagicMock()
                self.engine.config_manager.get_config.return_value = config
                self.console = MagicMock()
                self.errors = []
                self.infos = []

            def _show_error(self, m):
                self.errors.append(m)

            def _show_info(self, m):
                self.infos.append(m)

            def _show_success(self, m):
                pass

        return _D(), Command, SlashCommand

    @pytest.mark.asyncio
    async def test_list_empty(self):
        d, Command, SlashCommand = self._disp({})
        await d._handle_subagents_command(
            Command.create_slash_command(SlashCommand.SUBAGENTS, [], "x")
        )
        assert d.infos

    @pytest.mark.asyncio
    async def test_run_unknown(self):
        d, Command, SlashCommand = self._disp({})
        await d._handle_subagents_command(
            Command.create_slash_command(
                SlashCommand.SUBAGENTS, ["run", "missing", "do", "it"], "x"
            )
        )
        assert d.errors

    @pytest.mark.asyncio
    async def test_run_invokes_runner(self):
        defn = SubAgentDefinition(name="researcher", description="d")
        d, Command, SlashCommand = self._disp({"researcher": defn})
        with patch("omnimancer.cli.subagent.SubAgentRunner") as RunnerCls:
            RunnerCls.return_value.run = AsyncMock(
                return_value=__import__(
                    "omnimancer.cli.subagent", fromlist=["SubAgentResult"]
                ).SubAgentResult(name="researcher", output="result", success=True)
            )
            await d._handle_subagents_command(
                Command.create_slash_command(
                    SlashCommand.SUBAGENTS, ["run", "researcher", "find", "x"], "x"
                )
            )
        d.console.print.assert_called()
