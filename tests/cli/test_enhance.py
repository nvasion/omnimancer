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


_NO_BLOCK = object()  # sentinel: distinguish "default block" from explicit None


def _config_manager(providers=None, enhancement=_NO_BLOCK):
    from omnimancer.core.models import EnhancementConfig, ProviderConfig

    # `is None` (not falsy-or): passing an explicit empty dict must yield a
    # config with NO providers. `{} or default` silently substituted the real
    # gateway entry, and the missing-provider test then hit the live network —
    # green only while the gateway was broken.
    if providers is None:
        providers = {
            "gateway": ProviderConfig(
                model="qwen3-coder-30b",
                provider_type="openai-compatible",
                base_url="http://alpha:8888/v1",
                auth_type="none",
            )
        }
    # Default = a real enhancement block (the configured state most tests
    # exercise); pass enhancement=None to model an unconfigured install.
    if enhancement is _NO_BLOCK:
        enhancement = EnhancementConfig()
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
    async def test_reasoning_think_block_is_stripped(self):
        """qwen3-8b wraps output in <think>…</think>; only the rewrite
        after the block may reach the user (observed live 2026-07-31)."""
        provider = MagicMock()
        provider.send_message = AsyncMock(
            return_value=ChatResponse(
                content="<think>\nchain of thought here\n</think>\n\nGoal: fix it.",
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
        assert (text, ok) == ("Goal: fix it.", True)

    @pytest.mark.asyncio
    async def test_only_think_block_fails_open(self):
        """All-reasoning output (nothing after </think>) is a failed
        enhancement — fall back to the original draft."""
        provider = MagicMock()
        provider.send_message = AsyncMock(
            return_value=ChatResponse(
                content="<think>reasoning but no rewrite</think>",
                model_used="qwen3-8b",
                tokens_used=1,
                timestamp=datetime.now(),
            )
        )
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            return_value=provider,
        ):
            text, ok = await enhance("my draft", "code", _config_manager())
        assert (text, ok) == ("my draft", False)

    @pytest.mark.asyncio
    async def test_unclosed_think_block_fails_open(self):
        """A truncated response can end mid-reasoning with no closing tag;
        everything from <think> on is reasoning, so nothing usable remains."""
        provider = MagicMock()
        provider.send_message = AsyncMock(
            return_value=ChatResponse(
                content="<think>ran out of tokens mid-thought",
                model_used="qwen3-8b",
                tokens_used=1,
                timestamp=datetime.now(),
            )
        )
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            return_value=provider,
        ):
            text, ok = await enhance("my draft", "code", _config_manager())
        assert (text, ok) == ("my draft", False)

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
        assert settings.enabled is True

    def test_config_without_block_means_feature_off(self):
        """Enhancement is opt-in: a config with no `enhancement` block has
        the feature disabled instead of silently assuming a 'gateway'
        provider that most installs won't have."""
        from omnimancer.core.models import Config

        config = Config(
            default_provider="local",
            storage_path="/tmp/omni-test",
            providers={},
        )
        assert config.enhancement is None

    def test_config_parses_enhancement_block(self):
        from omnimancer.core.models import Config

        config = Config(
            default_provider="local",
            storage_path="/tmp/omni-test",
            providers={},
            enhancement={"provider": "local", "model": "qwen3-coder-30b"},
        )
        assert config.enhancement.provider == "local"
        assert config.enhancement.enabled is True


class TestEnhancementEnabled:
    def _config(self, enhancement):
        return SimpleNamespace(enhancement=enhancement)

    def test_no_block_is_disabled(self):
        from omnimancer.core.prompt_enhancer import enhancement_enabled

        assert enhancement_enabled(self._config(None)) is False

    def test_block_present_is_enabled(self):
        from omnimancer.core.models import EnhancementConfig
        from omnimancer.core.prompt_enhancer import enhancement_enabled

        assert enhancement_enabled(self._config(EnhancementConfig())) is True

    def test_explicit_disable_wins(self):
        from omnimancer.core.models import EnhancementConfig
        from omnimancer.core.prompt_enhancer import enhancement_enabled

        settings = EnhancementConfig(enabled=False)
        assert enhancement_enabled(self._config(settings)) is False

    @pytest.mark.asyncio
    async def test_enhance_without_block_fails_open(self):
        manager = _config_manager(enhancement=None)
        text, ok = await enhance("draft", "code", manager)
        assert (text, ok) == ("draft", False)


def _provider_returning(content):
    provider = MagicMock()
    provider.send_message = AsyncMock(
        return_value=ChatResponse(
            content=content,
            model_used="m",
            tokens_used=1,
            timestamp=datetime.now(),
        )
    )
    return provider


def _provider_raising():
    provider = MagicMock()
    provider.send_message = AsyncMock(side_effect=RuntimeError("down"))
    return provider


def _two_provider_manager():
    from omnimancer.core.models import ProviderConfig

    return _config_manager(
        providers={
            "gateway": ProviderConfig(
                model="qwen3-coder-30b",
                provider_type="openai-compatible",
                base_url="http://alpha:8888/v1",
                auth_type="none",
            ),
            "local": ProviderConfig(
                model="qwen3-coder-30b",
                provider_type="openai-compatible",
                base_url="http://localhost:8000/v1",
                auth_type="none",
            ),
        }
    )


class TestEnhanceFallback:
    """Failsafe: configured enhancement model first, then the caller's
    current session model, then fail-open — never block the prompt."""

    @pytest.mark.asyncio
    async def test_primary_success_skips_fallback(self):
        factory = MagicMock(return_value=_provider_returning("PRIMARY"))
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            new=factory,
        ):
            text, ok = await enhance(
                "draft",
                "code",
                _two_provider_manager(),
                fallback_model=("local", "qwen3-coder-30b"),
            )
        assert (text, ok) == ("PRIMARY", True)
        assert factory.call_count == 1
        assert factory.call_args.args[0] == "gateway"

    @pytest.mark.asyncio
    async def test_primary_failure_uses_session_model(self):
        factory = MagicMock(
            side_effect=[_provider_raising(), _provider_returning("FROM FALLBACK")]
        )
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            new=factory,
        ):
            text, ok = await enhance(
                "draft",
                "code",
                _two_provider_manager(),
                fallback_model=("local", "qwen3-coder-30b"),
            )
        assert (text, ok) == ("FROM FALLBACK", True)
        assert factory.call_count == 2
        assert factory.call_args.args[0] == "local"
        # The fallback call must target the session model, not the
        # enhancement model.
        assert factory.call_args.args[1].model == "qwen3-coder-30b"

    @pytest.mark.asyncio
    async def test_all_candidates_fail_opens_with_draft(self):
        factory = MagicMock(side_effect=[_provider_raising(), _provider_raising()])
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            new=factory,
        ):
            text, ok = await enhance(
                "my draft",
                "code",
                _two_provider_manager(),
                fallback_model=("local", "qwen3-coder-30b"),
            )
        assert (text, ok) == ("my draft", False)
        assert factory.call_count == 2

    @pytest.mark.asyncio
    async def test_identical_fallback_attempted_once(self):
        factory = MagicMock(return_value=_provider_raising())
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            new=factory,
        ):
            text, ok = await enhance(
                "my draft",
                "code",
                _two_provider_manager(),
                fallback_model=("gateway", "qwen3-8b"),
            )
        assert (text, ok) == ("my draft", False)
        assert factory.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_provider_missing_is_skipped(self):
        factory = MagicMock(return_value=_provider_raising())
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            new=factory,
        ):
            text, ok = await enhance(
                "my draft",
                "code",
                _two_provider_manager(),
                fallback_model=("nonexistent", "m"),
            )
        assert (text, ok) == ("my draft", False)
        assert factory.call_count == 1

    @pytest.mark.asyncio
    async def test_think_stripped_on_fallback_path(self):
        factory = MagicMock(
            side_effect=[
                _provider_raising(),
                _provider_returning("<think>reasoning</think>\n\nGoal: x."),
            ]
        )
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            new=factory,
        ):
            text, ok = await enhance(
                "draft",
                "code",
                _two_provider_manager(),
                fallback_model=("local", "qwen3-coder-30b"),
            )
        assert (text, ok) == ("Goal: x.", True)

    @pytest.mark.asyncio
    async def test_think_only_primary_tries_fallback(self):
        """All-reasoning output from the primary counts as a failure and
        falls through to the session model."""
        factory = MagicMock(
            side_effect=[
                _provider_returning("<think>only reasoning</think>"),
                _provider_returning("REAL REWRITE"),
            ]
        )
        with patch(
            "omnimancer.providers.factory.ProviderFactory.create_provider",
            new=factory,
        ):
            text, ok = await enhance(
                "draft",
                "code",
                _two_provider_manager(),
                fallback_model=("local", "qwen3-coder-30b"),
            )
        assert (text, ok) == ("REAL REWRITE", True)


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

    @pytest.mark.asyncio
    async def test_disabled_hints_and_sends_nothing(self, harness):
        harness.engine.config_manager.get_config.return_value = SimpleNamespace(
            enhancement=None
        )
        mock = AsyncMock(return_value=("BETTER", True))
        with patch("omnimancer.cli.command_dispatch.enhance_prompt", new=mock):
            await harness._handle_enhance_command(["fix", "it"])
        mock.assert_not_awaited()
        assert harness.sent == []
        assert any("enhancement" in m.lower() for _, m in harness.messages)
