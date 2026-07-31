"""/models refresh — pull live model catalogs from provider endpoints.

The drift-proof path for served context sizes (vLLM max_model_len) vs
client expectations: refresh assigns each provider's fetch_enhanced_models
result to its _catalog_models, which get_available_models prefers.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimancer.cli.command_dispatch import CommandDispatchMixin
from omnimancer.cli.commands import Command, SlashCommand


class _RecordingConsole:
    def __init__(self):
        self.output = []

    def print(self, *args, **kwargs):
        self.output.append(" ".join(str(a) for a in args))


class _Harness(CommandDispatchMixin):
    def __init__(self, providers):
        self.console = _RecordingConsole()
        self.engine = MagicMock()
        self.engine.providers = providers

    def _show_error(self, message: str) -> None:
        self.console.print(message)

    def _show_info(self, message: str) -> None:
        self.console.print(message)

    def _show_success(self, message: str) -> None:
        self.console.print(message)

    @property
    def text(self):
        return "\n".join(self.console.output)


def _provider_with_models(models):
    provider = MagicMock()
    provider.fetch_enhanced_models = AsyncMock(return_value=models)
    # A plain attribute assignment target, like a real BaseProvider
    provider._catalog_models = None
    return provider


def _refresh_command(args):
    return Command.create_slash_command(
        SlashCommand.MODELS, args, "/models " + " ".join(args)
    )


class TestModelsRefresh:
    @pytest.mark.asyncio
    async def test_refresh_all_assigns_catalogs(self):
        gateway_models = [MagicMock(), MagicMock(), MagicMock()]
        local_models = [MagicMock()]
        providers = {
            "gateway": _provider_with_models(gateway_models),
            "local": _provider_with_models(local_models),
        }
        harness = _Harness(providers)

        await harness._show_models(_refresh_command(["refresh"]))

        assert providers["gateway"]._catalog_models == gateway_models
        assert providers["local"]._catalog_models == local_models
        assert "gateway" in harness.text
        assert "local" in harness.text
        assert "3" in harness.text

    @pytest.mark.asyncio
    async def test_refresh_single_provider(self):
        providers = {
            "gateway": _provider_with_models([MagicMock()]),
            "local": _provider_with_models([MagicMock()]),
        }
        harness = _Harness(providers)

        await harness._show_models(_refresh_command(["refresh", "gateway"]))

        assert providers["gateway"]._catalog_models is not None
        assert providers["local"]._catalog_models is None
        providers["local"].fetch_enhanced_models.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_refresh_unknown_provider_errors(self):
        harness = _Harness({"local": _provider_with_models([])})

        await harness._show_models(_refresh_command(["refresh", "nope"]))

        assert "nope" in harness.text
        assert "not" in harness.text.lower()

    @pytest.mark.asyncio
    async def test_refresh_failure_keeps_going(self):
        failing = MagicMock()
        failing.fetch_enhanced_models = AsyncMock(side_effect=RuntimeError("down"))
        failing._catalog_models = None
        ok_models = [MagicMock()]
        providers = {"gateway": failing, "local": _provider_with_models(ok_models)}
        harness = _Harness(providers)

        await harness._show_models(_refresh_command(["refresh"]))

        assert providers["local"]._catalog_models == ok_models
        assert failing._catalog_models is None
        assert "gateway" in harness.text  # failure is reported, not swallowed

    @pytest.mark.asyncio
    async def test_refresh_empty_result_does_not_clobber(self):
        provider = _provider_with_models([])
        provider._catalog_models = ["existing"]
        harness = _Harness({"gateway": provider})

        await harness._show_models(_refresh_command(["refresh"]))

        assert provider._catalog_models == ["existing"]
