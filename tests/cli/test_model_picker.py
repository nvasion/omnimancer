"""/model with no args — interactive picker over the current provider's
models (same candidate source as tab completion)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.command_dispatch import CommandDispatchMixin
from omnimancer.cli.completion import CompletionManager


class _Harness(CommandDispatchMixin):
    def __init__(self, choices, selection):
        self.console = MagicMock()
        self.engine = MagicMock()
        self.engine.get_conversation_summary.return_value = {
            "current_provider": "local"
        }
        self.engine.switch_model = AsyncMock(return_value=True)
        self.completion_manager = MagicMock(spec=CompletionManager)
        self.completion_manager.model_names.return_value = choices
        self._selection = selection
        self.messages = []

    async def _prompt_model_selection(self):
        return self._selection

    def _show_error(self, message):
        self.messages.append(("error", message))

    def _show_info(self, message):
        self.messages.append(("info", message))

    def _show_success(self, message):
        self.messages.append(("success", message))


CHOICES = ["qwen3-coder-30b", "gpt-oss-120b", "qwen3-8b"]


class TestModelPicker:
    @pytest.mark.asyncio
    async def test_pick_by_number(self):
        harness = _Harness(CHOICES, "2")
        await harness._handle_model_picker()
        harness.engine.switch_model.assert_awaited_once_with("local", "gpt-oss-120b")

    @pytest.mark.asyncio
    async def test_pick_by_name(self):
        harness = _Harness(CHOICES, "qwen3-8b")
        await harness._handle_model_picker()
        harness.engine.switch_model.assert_awaited_once_with("local", "qwen3-8b")

    @pytest.mark.asyncio
    async def test_empty_selection_cancels(self):
        harness = _Harness(CHOICES, "")
        await harness._handle_model_picker()
        harness.engine.switch_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_selection_errors(self):
        harness = _Harness(CHOICES, "99")
        await harness._handle_model_picker()
        harness.engine.switch_model.assert_not_awaited()
        assert any(kind == "error" for kind, _ in harness.messages)

    @pytest.mark.asyncio
    async def test_no_models_hints_refresh(self):
        harness = _Harness([], "1")
        await harness._handle_model_picker()
        harness.engine.switch_model.assert_not_awaited()
        assert any("refresh" in message for _, message in harness.messages)
