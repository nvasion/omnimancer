"""Adding models on the fly.

Static per-provider catalogs are forever stale — OpenAI-compatible endpoints
accept any model string, so `/switch <provider> <new-model>` must work
without a separate registration step, and custom models must show up in
the `/models` display.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.command_dispatch import CommandDispatchMixin
from omnimancer.cli.commands import Command, SlashCommand
from omnimancer.core.models import EnhancedModelInfo, ModelInfo


def _static_model(name="llama3.3-70b-instruct", provider="digitalocean"):
    return ModelInfo(
        name=name,
        provider=provider,
        description="static",
        max_tokens=4096,
        cost_per_token=0.000001,
        available=True,
    )


class _Harness(CommandDispatchMixin):
    """Minimal host for the switch handler."""

    def __init__(self):
        provider = MagicMock()
        provider.get_available_models.return_value = [_static_model()]
        provider.model = "llama3.3-70b-instruct"
        provider.supports_tools.return_value = True
        provider.supports_multimodal.return_value = False
        provider.get_model_info.return_value = _static_model()

        self.engine = MagicMock()
        self.engine.providers = {"digitalocean": provider}
        self.engine.current_provider = provider
        self.engine.switch_model = AsyncMock(return_value=True)
        self.engine.config_manager.get_custom_models.return_value = []

        self.console = MagicMock()
        self.messages = {"error": [], "info": [], "success": [], "warning": []}

    def _show_error(self, m):
        self.messages["error"].append(m)

    def _show_info(self, m):
        self.messages["info"].append(m)

    def _show_success(self, m):
        self.messages["success"].append(m)

    def _show_warning(self, m):
        self.messages["warning"].append(m)


def _switch_command(provider, model):
    return Command.create_slash_command(
        SlashCommand.SWITCH, [provider, model], f"/switch {provider} {model}"
    )


class TestModelsAddHint:
    def test_models_add_points_to_add_model(self):
        """'/model add ...' must point at /add-model, not a filter error."""
        from omnimancer.cli.commands import _validate_command_args

        with pytest.raises(ValueError, match="/add-model"):
            _validate_command_args(SlashCommand.MODELS, ["add"])


class TestSwitchToUnknownModel:
    @pytest.mark.asyncio
    async def test_unknown_model_registered_and_switched(self):
        """/switch <provider> <unknown-model> registers it and proceeds."""
        h = _Harness()

        await h._handle_switch_command(
            _switch_command("digitalocean", "qwen3-coder-flash")
        )

        assert not h.messages["error"]
        h.engine.config_manager.add_custom_model.assert_called_once()
        registered = h.engine.config_manager.add_custom_model.call_args[0][0]
        assert registered.name == "qwen3-coder-flash"
        assert registered.provider == "digitalocean"
        h.engine.switch_model.assert_awaited_once_with(
            "digitalocean", "qwen3-coder-flash"
        )
        assert any("qwen3-coder-flash" in w for w in h.messages["warning"])

    @pytest.mark.asyncio
    async def test_known_model_not_reregistered(self):
        """A model already in the catalog switches without registration."""
        h = _Harness()

        await h._handle_switch_command(
            _switch_command("digitalocean", "llama3.3-70b-instruct")
        )

        h.engine.config_manager.add_custom_model.assert_not_called()
        h.engine.switch_model.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_model_not_reregistered(self):
        """An already-registered custom model switches without duplication."""
        h = _Harness()
        h.engine.config_manager.get_custom_models.return_value = [
            EnhancedModelInfo(
                name="qwen3.5-397b-a17b",
                provider="digitalocean",
                description="custom",
                max_tokens=4096,
                cost_per_million_input=1.0,
                cost_per_million_output=3.0,
                swe_score=50.0,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                latest_version=False,
                deprecated=False,
                release_date=datetime.now(),
                context_window=4096,
                is_free=False,
            )
        ]

        await h._handle_switch_command(
            _switch_command("digitalocean", "qwen3.5-397b-a17b")
        )

        h.engine.config_manager.add_custom_model.assert_not_called()
        h.engine.switch_model.assert_awaited_once()
