"""Streaming response display for Omnimancer interactive mode."""

import json
from typing import Callable, List, Optional, Union

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from ..core.models import StreamEvent, StreamEventType, ToolCall


class StreamingDisplay:
    """Manages live-updating display of a streaming API response.

    ``activity_provider`` (optional) supplies a renderable of recent tool
    activity composed ABOVE the assistant panel inside this display's own
    Live — never a second Live region.
    """

    def __init__(
        self,
        console: Console,
        model: str = "",
        activity_provider: Optional[Callable[[], Optional[RenderableType]]] = None,
    ):
        self.console = console
        self.model = model
        self.accumulated_text = ""
        self.tool_calls: List[ToolCall] = []
        self._activity_provider = activity_provider
        self._current_tool_json = ""
        self._current_tool_name = ""
        self._live: Optional[Live] = None

    def start(self) -> None:
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=15,
            vertical_overflow="visible",
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.update(self._render())
            self._live.stop()
            self._live = None

    def handle_event(self, event: StreamEvent) -> None:
        if event.type == StreamEventType.MESSAGE_START:
            self.model = event.model
        elif event.type == StreamEventType.TEXT_DELTA:
            self.accumulated_text += event.text
            self._update()
        elif event.type == StreamEventType.TOOL_USE_START:
            self._current_tool_name = event.tool_name
            self._current_tool_json = ""
        elif event.type == StreamEventType.TOOL_USE_DELTA:
            self._current_tool_json += event.partial_json
        elif event.type == StreamEventType.TOOL_USE_END:
            try:
                args = (
                    json.loads(self._current_tool_json)
                    if self._current_tool_json
                    else {}
                )
            except json.JSONDecodeError:
                args = {}
            self.tool_calls.append(
                ToolCall(
                    name=self._current_tool_name,
                    arguments=args,
                )
            )
            self._current_tool_name = ""
            self._current_tool_json = ""

    def _update(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _render(self) -> RenderableType:
        # Progressive markdown: re-parsing a few KB at 15fps is cheap, and
        # an unclosed fence mid-stream just renders as code-so-far.
        content: Union[str, Markdown]
        if not self.accumulated_text:
            content = "..."
        else:
            try:
                content = Markdown(self.accumulated_text, code_theme="monokai")
            except Exception:
                content = self.accumulated_text
        panel = Panel(content, title=f"Assistant ({self.model})", border_style="blue")
        if self._activity_provider is not None:
            try:
                activity = self._activity_provider()
            except Exception:
                activity = None
            if activity is not None:
                return Group(activity, panel)
        return panel
