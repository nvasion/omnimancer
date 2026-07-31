"""PromptFoundry port: meta-prompt fidelity, e:-prefix routing, the
enhance() call, and the /enhance command flow."""

import hashlib
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core import prompt_enhancer
from omnimancer.core.models import ChatResponse
from omnimancer.core.prompt_enhancer import enhance, split_enhance_prefix

# sha256 of the four meta-prompts as extracted VERBATIM from
# promptfoundry/background.js on 2026-07-31. If these fail, someone edited
# the ported texts — re-sync with the extension instead of hand-tuning.
EXPECTED_HASHES = {
    "DEFAULT_META_PROMPT": (
        "4ad0628c32707dab63e4117e5809b86c06c2b96bfc6d88201407bf8fb741fe88"
    ),
    "CODE_META_PROMPT": (
        "c20636093712c75c96b7dd5b32ffb8a0a2fdcb754f7639aa9e0adfe029264205"
    ),
    "IMAGE_META_PROMPT": (
        "c50722108a410830e57230b0d53bbb7ca103d9ab6afbbee6ab06cbcedd8a7fbc"
    ),
    "RESEARCH_META_PROMPT": (
        "ffe2fa95248a4dec9741ef5b650ecfbfc67af13a5306d384a83a2837b89e6a32"
    ),
}


class TestMetaPromptFidelity:
    @pytest.mark.parametrize("name,expected", sorted(EXPECTED_HASHES.items()))
    def test_verbatim(self, name, expected):
        text = getattr(prompt_enhancer, name)
        assert hashlib.sha256(text.encode()).hexdigest() == expected

    def test_profile_map(self):
        assert set(prompt_enhancer.META_PROMPTS) == {
            "chat",
            "code",
            "image",
            "research",
        }


class TestPrefixSplit:
    def test_basic(self):
        assert split_enhance_prefix("e: fix the tests") == "fix the tests"

    def test_case_insensitive(self):
        assert split_enhance_prefix("E: do a thing") == "do a thing"

    def test_leading_whitespace(self):
        assert split_enhance_prefix("  e: draft") == "draft"

    def test_no_prefix(self):
        assert split_enhance_prefix("explain e: notation") is None

    def test_empty_draft(self):
        assert split_enhance_prefix("e:") is None
        assert split_enhance_prefix("e:   ") is None

    def test_word_starting_with_e_is_not_trigger(self):
        assert split_enhance_prefix("everyone likes tests") is None


def _config_manager(providers=None, enhancement=None):
    from omnimancer.core.models import ProviderConfig

    providers = providers or {
        "gateway": ProviderConfig(
            model="qwen3-coder-30b",
            provider_type="openai-compatible",
            base_url="http://alpha:8888/v1",
            auth_type="none",
        )
    }
    config = SimpleNamespace(providers=providers, enhancement=enhancement)
    manager = MagicMock()
    manager.get_config.return_value = config
    return manager


class TestEnhance:
    @pytest.mark.asyncio
    async def test_success_returns_rewrite(self):
        provider = MagicMock()
        provider.send_message = AsyncMock(
            return_value=ChatResponse(
                content="REWRITTEN",
                model_used="qwen3-8b",
                tokens_used=1,
                timestamp=datetime.now(),
            )
        )
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            return_value=provider,
        ):
            text, ok = await enhance("draft", "code", _config_manager())
        assert (text, ok) == ("REWRITTEN", True)
        message = provider.send_message.call_args.args[0]
        assert message == "Draft prompt:\n\ndraft"
        context = provider.send_message.call_args.args[1]
        assert context.messages[0].content == prompt_enhancer.CODE_META_PROMPT

    @pytest.mark.asyncio
    async def test_provider_error_fails_open(self):
        provider = MagicMock()
        provider.send_message = AsyncMock(side_effect=RuntimeError("down"))
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            return_value=provider,
        ):
            text, ok = await enhance("my draft", "code", _config_manager())
        assert (text, ok) == ("my draft", False)

    @pytest.mark.asyncio
    async def test_unknown_profile_fails_open(self):
        text, ok = await enhance("draft", "bogus", _config_manager())
        assert (text, ok) == ("draft", False)

    @pytest.mark.asyncio
    async def test_missing_provider_fails_open(self):
        manager = _config_manager(providers={})
        text, ok = await enhance("draft", "code", manager)
        assert (text, ok) == ("draft", False)


class TestEnhancementConfig:
    def test_defaults_on_config_model(self):
        from omnimancer.core.models import EnhancementConfig

        settings = EnhancementConfig()
        assert settings.provider == "gateway"
        assert settings.model == "qwen3-8b"
        assert settings.temperature == 0.4
        assert settings.default_profile == "code"

    def test_config_carries_enhancement_block(self):
        from omnimancer.core.models import Config

        config = Config(
            default_provider="local",
            storage_path="/tmp/omni-test",
            providers={},
        )
        assert config.enhancement.provider == "gateway"


class TestEnhanceCommand:
    @pytest.fixture
    def harness(self):
        from omnimancer.cli.command_dispatch import CommandDispatchMixin

        class _Harness(CommandDispatchMixin):
            def __init__(self):
                self.console = MagicMock()
                self.engine = MagicMock()
                self.messages = []
                self.sent = []
                self._confirm = "y"

            def _show_error(self, message):
                self.messages.append(("error", message))

            def _show_info(self, message):
                self.messages.append(("info", message))

            def _show_success(self, message):
                self.messages.append(("success", message))

            async def _prompt_enhance_confirm(self):
                return self._confirm

            async def _handle_chat_message(self, command):
                self.sent.append(command.content)

        return _Harness()

    @pytest.mark.asyncio
    async def test_confirm_sends_enhanced(self, harness):
        with patch(
            "omnimancer.cli.command_dispatch.enhance_prompt",
            new=AsyncMock(return_value=("BETTER", True)),
        ):
            await harness._handle_enhance_command(["fix", "the", "tests"])
        assert harness.sent == ["BETTER"]

    @pytest.mark.asyncio
    async def test_decline_sends_nothing(self, harness):
        harness._confirm = "n"
        with patch(
            "omnimancer.cli.command_dispatch.enhance_prompt",
            new=AsyncMock(return_value=("BETTER", True)),
        ):
            await harness._handle_enhance_command(["fix", "it"])
        assert harness.sent == []

    @pytest.mark.asyncio
    async def test_explicit_profile(self, harness):
        mock = AsyncMock(return_value=("OUT", True))
        with patch("omnimancer.cli.command_dispatch.enhance_prompt", new=mock):
            await harness._handle_enhance_command(["research", "compare", "x"])
        assert mock.call_args.args[0] == "compare x"
        assert mock.call_args.args[1] == "research"

    @pytest.mark.asyncio
    async def test_no_args_errors(self, harness):
        await harness._handle_enhance_command([])
        assert any(kind == "error" for kind, _ in harness.messages)
