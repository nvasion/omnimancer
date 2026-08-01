"""REPL turn-activity window tests (WU-C2).

The activity log is display-only: it filters bus events, keeps a bounded
row window, and renders inside the streaming display's existing Live via
the activity_provider — asserted here to compose a Group, never a second
Live.
"""

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from omnimancer.core.agent.status_core import AgentEvent, EventType
from omnimancer.ui.streaming_display import StreamingDisplay
from omnimancer.ui.turn_activity import TurnActivityLog


def _tool_event(
    event_type: EventType = EventType.OPERATION_STARTED, **metadata
) -> AgentEvent:
    return AgentEvent(
        event_type=event_type,
        agent_id="main",
        operation_id="op-1",
        data={"metadata": {"tool": "Bash", "target": "pytest -q", **metadata}},
    )


class TestTurnActivityLog:
    async def test_collects_and_renders_then_resets(self):
        log = TurnActivityLog()
        assert log.render() is None
        await log.handle_event(_tool_event())
        table = log.render()
        assert isinstance(table, Table)
        log.reset_turn()
        assert log.render() is None

    async def test_window_is_bounded(self):
        log = TurnActivityLog(max_rows=3)
        for _ in range(10):
            await log.handle_event(_tool_event())
        assert len(log._rows) == 3

    async def test_suspend_stops_collection(self):
        log = TurnActivityLog()
        log.suspend()
        await log.handle_event(_tool_event())
        assert log.render() is None
        log.resume()
        await log.handle_event(_tool_event())
        assert log.render() is not None

    async def test_non_activity_events_ignored(self):
        log = TurnActivityLog()
        await log.handle_event(AgentEvent(event_type=EventType.TURN_START))
        await log.handle_event(AgentEvent(event_type=EventType.SESSION_START))
        assert log.render() is None


class TestStreamingDisplayComposition:
    def _console(self):
        from io import StringIO

        from rich.console import Console

        return Console(file=StringIO(), force_terminal=True)

    def test_render_is_panel_without_provider(self):
        display = StreamingDisplay(self._console())
        display.accumulated_text = "hello"
        assert isinstance(display._render(), Panel)

    def test_render_groups_activity_above_panel(self):
        table = Table.grid()
        display = StreamingDisplay(self._console(), activity_provider=lambda: table)
        display.accumulated_text = "hello"
        rendered = display._render()
        assert isinstance(rendered, Group)
        assert rendered.renderables[0] is table
        assert isinstance(rendered.renderables[1], Panel)
        # Load-bearing: the agent loop reads accumulated_text back.
        assert display.accumulated_text == "hello"

    def test_render_falls_back_when_provider_empty_or_raises(self):
        display = StreamingDisplay(self._console(), activity_provider=lambda: None)
        assert isinstance(display._render(), Panel)

        def _boom():
            raise RuntimeError("provider broke")

        display = StreamingDisplay(self._console(), activity_provider=_boom)
        assert isinstance(display._render(), Panel)
