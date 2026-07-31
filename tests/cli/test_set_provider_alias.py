"""Alias-aware /config set-provider, /switch resolution, migration back-fill,
and validator leniency for keyless self-hosted endpoints."""

import json
from unittest.mock import MagicMock

import pytest

from omnimancer.cli.command_dispatch import CommandDispatchMixin
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.models import Config, ProviderConfig


class _RecordingConsole:
    def __init__(self):
        self.output = []

    def print(self, *args, **kwargs):
        self.output.append(" ".join(str(a) for a in args))


class _Harness(CommandDispatchMixin):
    """CommandDispatchMixin declares _show_* as typing stubs (DisplayMixin
    provides the real ones), so the harness must implement them to observe
    user-facing messages."""

    def __init__(self, config_manager):
        self.console = _RecordingConsole()
        self.engine = MagicMock()
        self.engine.config_manager = config_manager

    def _show_error(self, message: str) -> None:
        self.console.print(message)

    def _show_info(self, message: str) -> None:
        self.console.print(message)

    def _show_success(self, message: str) -> None:
        self.console.print(message)

    @property
    def text(self):
        return "\n".join(self.console.output)


@pytest.fixture
def config_manager(tmp_path):
    return ConfigManager(config_path=tmp_path / "config.json")


@pytest.fixture
def harness(config_manager):
    return _Harness(config_manager)


class TestSetProviderAlias:
    @pytest.mark.asyncio
    async def test_alias_with_type_persists_provider_type(
        self, harness, config_manager
    ):
        await harness._handle_config_set_provider(
            [
                "gateway",
                "--type",
                "openai-compatible",
                "--base-url",
                "http://alpha:8888/v1",
                "--model",
                "qwen3-coder-30b",
            ]
        )
        stored = config_manager.get_provider_config("gateway")
        assert stored is not None
        assert stored.provider_type == "openai-compatible"
        assert stored.base_url == "http://alpha:8888/v1"
        assert stored.model == "qwen3-coder-30b"

    @pytest.mark.asyncio
    async def test_unknown_name_without_type_errors_and_persists_nothing(
        self, harness, config_manager
    ):
        await harness._handle_config_set_provider(
            ["mystery", "--base-url", "http://x/v1", "--model", "m"]
        )
        assert config_manager.get_provider_config("mystery") is None
        assert "--type" in harness.text
        assert "openai-compatible" in harness.text  # lists registered types

    @pytest.mark.asyncio
    async def test_unregistered_type_errors(self, harness, config_manager):
        await harness._handle_config_set_provider(
            ["gateway", "--type", "nonsense", "--model", "m"]
        )
        assert config_manager.get_provider_config("gateway") is None
        assert "nonsense" in harness.text

    @pytest.mark.asyncio
    async def test_new_entry_requires_model(self, harness, config_manager):
        await harness._handle_config_set_provider(
            ["gateway", "--type", "openai-compatible", "--base-url", "http://x/v1"]
        )
        assert config_manager.get_provider_config("gateway") is None
        assert "--model" in harness.text

    @pytest.mark.asyncio
    async def test_update_keeps_existing_model(self, harness, config_manager):
        config_manager.set_provider_config(
            "gateway",
            ProviderConfig(
                model="qwen3-coder-30b",
                provider_type="openai-compatible",
                base_url="http://old/v1",
            ),
        )
        await harness._handle_config_set_provider(
            ["gateway", "--base-url", "http://new/v1"]
        )
        stored = config_manager.get_provider_config("gateway")
        assert stored.model == "qwen3-coder-30b"
        assert stored.base_url == "http://new/v1"
        assert stored.provider_type == "openai-compatible"


class TestSwitchParsing:
    """Parse-layer rules for /switch: bare invocation reaches the handler
    (which shows usage + providers) instead of raising, and hyphenated
    provider names (openai-compatible, claude-code) are valid."""

    def test_bare_switch_parses_with_empty_args(self):
        from omnimancer.cli.commands import CommandType, parse_command

        command = parse_command("/switch")
        assert command.type == CommandType.SLASH_COMMAND
        assert command.args == []

    def test_hyphenated_provider_names_accepted(self):
        from omnimancer.cli.commands import parse_command

        assert parse_command("/switch openai-compatible").args == ["openai-compatible"]
        assert parse_command("/switch claude-code").args == ["claude-code"]


class TestSwitchResolution:
    def test_case_insensitive_match_against_configured_providers(self, harness):
        harness.engine.providers = {"MyGateway": object(), "local": object()}
        assert harness._resolve_provider_key("mygateway") == "MyGateway"
        assert harness._resolve_provider_key("LOCAL") == "local"

    def test_unknown_name_falls_back_to_lowercase(self, harness):
        harness.engine.providers = {"local": object()}
        assert harness._resolve_provider_key("Nope") == "nope"


class TestMigrationBackfill:
    def test_backfill_only_for_registered_names(self, tmp_path):
        config_path = tmp_path / "config.json"
        raw = {
            "default_provider": "openai",
            "storage_path": str(tmp_path),
            "providers": {
                "openai": {"model": "gpt-4", "api_key": "k"},
                "gateway": {
                    "model": "qwen3-coder-30b",
                    "base_url": "http://alpha:8888/v1",
                },
            },
        }
        config_path.write_text(json.dumps(raw))

        manager = ConfigManager(config_path=config_path)
        manager.migrate_config_format()

        stored = json.loads(config_path.read_text())
        assert stored["providers"]["openai"]["provider_type"] == "openai"
        # Aliases must NOT be back-filled with a useless self-referential type
        assert "provider_type" not in stored["providers"]["gateway"] or stored[
            "providers"
        ]["gateway"]["provider_type"] not in ("gateway",)


class TestValidatorLeniency:
    def _base_config(self, providers):
        return Config(
            default_provider=next(iter(providers)),
            storage_path="/tmp/omnimancer-test",
            providers=providers,
        )

    def test_openai_compatible_alias_keyless_is_clean(self):
        config = self._base_config(
            {
                "gateway": ProviderConfig(
                    model="qwen3-coder-30b",
                    provider_type="openai-compatible",
                    base_url="http://alpha:8888/v1",
                    auth_type="none",
                )
            }
        )
        errors = config._validate_provider_config(
            "gateway", config.providers["gateway"]
        )
        assert errors == []

    def test_openai_type_with_base_url_skips_model_allowlist(self):
        config = self._base_config(
            {
                "proxy": ProviderConfig(
                    model="llama3.3-70b-instruct",
                    provider_type="openai",
                    base_url="http://localhost:1234/v1",
                    api_key="k",
                )
            }
        )
        errors = config._validate_provider_config("proxy", config.providers["proxy"])
        assert errors == []

    def test_openai_type_auth_none_skips_key_requirement(self):
        config = self._base_config(
            {
                "proxy": ProviderConfig(
                    model="gpt-4",
                    provider_type="openai",
                    base_url="http://localhost:1234/v1",
                    auth_type="none",
                )
            }
        )
        errors = config._validate_provider_config("proxy", config.providers["proxy"])
        assert errors == []

    def test_real_openai_still_validated(self):
        config = self._base_config(
            {"openai": ProviderConfig(model="not-a-gpt-model", api_key="")}
        )
        errors = config._validate_provider_config("openai", config.providers["openai"])
        assert any("API key" in e for e in errors)
        assert any("Unknown OpenAI model" in e for e in errors)

    def test_config_validator_generic_branch_honors_auth_none(self, tmp_path):
        from omnimancer.core.config_validator import ConfigValidator

        validator = ConfigValidator()
        provider_config = ProviderConfig(
            model="qwen3-coder-30b",
            provider_type="openai-compatible",
            base_url="http://localhost:8000/v1",
            auth_type="none",
        )
        errors = validator.validate_provider_config("gateway", provider_config)
        assert errors == []
