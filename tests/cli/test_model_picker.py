"""/model — bare opens the picker; with a name sets it on the current
provider. The current provider is resolved by IDENTITY against
engine.providers (engine.get_conversation_summary() has no provider key —
reading one there was the original 'No active provider' bug)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.command_dispatch import CommandDispatchMixin
from omnimancer.cli.completion import CompletionManager


class _Harness(CommandDispatchMixin):
    def __init__(self, choices, selection, with_current=True):
        self.console = MagicMock()
        self.engine = MagicMock()
        current = MagicMock()
        self.engine.providers = {"gateway": MagicMock(), "local": current}
        self.engine.current_provider = current if with_current else None
        self.engine.switch_model = AsyncMock(return_value=True)
        # Mirrors reality: the summary carries NO provider information.
        self.engine.get_conversation_summary.return_value = {
            "message_count": 0,
            "current_model": "qwen3-coder-30b",
            "session_id": "s",
        }
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


class TestCurrentProviderResolution:
    def test_identity_lookup_finds_alias_name(self):
        harness = _Harness(CHOICES, "")
        assert harness._current_provider_key() == "local"

    def test_none_when_no_current_provider(self):
        harness = _Harness(CHOICES, "", with_current=False)
        assert harness._current_provider_key() is None


class TestModelPicker:
    @pytest.mark.asyncio
    async def test_pick_by_number(self):
        harness = _Harness(CHOICES, "2")
        await harness._handle_model_command([])
        harness.engine.switch_model.assert_awaited_once_with("local", "gpt-oss-120b")

    @pytest.mark.asyncio
    async def test_pick_by_name(self):
        harness = _Harness(CHOICES, "qwen3-8b")
        await harness._handle_model_command([])
        harness.engine.switch_model.assert_awaited_once_with("local", "qwen3-8b")

    @pytest.mark.asyncio
    async def test_empty_selection_cancels(self):
        harness = _Harness(CHOICES, "")
        await harness._handle_model_command([])
        harness.engine.switch_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_selection_errors(self):
        harness = _Harness(CHOICES, "99")
        await harness._handle_model_command([])
        harness.engine.switch_model.assert_not_awaited()
        assert any(kind == "error" for kind, _ in harness.messages)

    @pytest.mark.asyncio
    async def test_no_models_hints_refresh(self):
        harness = _Harness([], "1")
        await harness._handle_model_command([])
        harness.engine.switch_model.assert_not_awaited()
        assert any("refresh" in message for _, message in harness.messages)

    @pytest.mark.asyncio
    async def test_no_current_provider_errors(self):
        harness = _Harness(CHOICES, "1", with_current=False)
        await harness._handle_model_command([])
        harness.engine.switch_model.assert_not_awaited()
        assert any("switch" in m.lower() for _, m in harness.messages)


class TestModelDirectSet:
    @pytest.mark.asyncio
    async def test_model_with_name_sets_on_current_provider(self):
        harness = _Harness(CHOICES, "unused")
        await harness._handle_model_command(["gpt-oss-120b"])
        harness.engine.switch_model.assert_awaited_once_with("local", "gpt-oss-120b")

    @pytest.mark.asyncio
    async def test_unknown_model_errors_with_switch_hint(self):
        harness = _Harness(CHOICES, "unused")
        await harness._handle_model_command(["nonexistent-model"])
        harness.engine.switch_model.assert_not_awaited()
        assert any("/switch" in message for _, message in harness.messages)
