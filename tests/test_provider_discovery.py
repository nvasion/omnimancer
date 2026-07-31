"""Discoverability tests: every registered provider must be visible to users.

These guard against the user-facing surfaces (package exports, ``/config``
help, ``/providers``) drifting out of sync with the providers actually
registered in :class:`ProviderFactory`.
"""

from omnimancer import providers as providers_pkg
from omnimancer.cli.command_dispatch import CommandDispatchMixin
from omnimancer.core.engine import CoreEngine
from omnimancer.providers.factory import ProviderFactory

# Class names we expect the package to export, one per registered provider.
EXPECTED_PROVIDER_CLASSES = {
    "ClaudeProvider",
    "ClaudeCodeProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "CohereProvider",
    "OllamaProvider",
    "PerplexityProvider",
    "XAIProvider",
    "MistralProvider",
    "AzureProvider",
    "VertexAIProvider",
    "BedrockProvider",
    "OpenRouterProvider",
    "DigitalOceanProvider",
    "OpenAICompatibleProvider",
}


class _RecordingConsole:
    def __init__(self):
        self.output = []

    def print(self, *args, **kwargs):
        self.output.append(" ".join(str(a) for a in args))


class _HelpHarness(CommandDispatchMixin):
    def __init__(self):
        self.console = _RecordingConsole()


def test_init_exports_every_registered_provider_class():
    """``providers/__init__.py`` must export a class for every provider."""
    exported = set(providers_pkg.__all__)
    assert EXPECTED_PROVIDER_CLASSES.issubset(exported)
    for name in EXPECTED_PROVIDER_CLASSES:
        assert hasattr(providers_pkg, name), f"{name} not importable from package"


def test_config_help_lists_all_registered_providers():
    """``/config`` help must advertise every registered provider, not a subset."""
    harness = _HelpHarness()
    harness._show_config_help()
    text = "\n".join(harness.console.output)
    for name in ProviderFactory.get_available_providers():
        assert name in text, f"/config help omits provider '{name}'"


def test_providers_list_surfaces_available_unconfigured_providers():
    """``/providers`` must reveal registered-but-unconfigured providers."""
    engine = CoreEngine.__new__(CoreEngine)
    engine.providers = {}
    engine.current_provider = None

    out = engine._get_providers_list()

    for name in ProviderFactory.get_available_providers():
        assert name in out, f"/providers hides available provider '{name}'"


def test_providers_list_distinguishes_configured_from_available():
    """Configured providers are shown separately from merely-available ones."""

    class _StubProvider:
        model = "some-model"

        def get_provider_name(self):
            return "claude"

    engine = CoreEngine.__new__(CoreEngine)
    engine.providers = {"claude": _StubProvider()}
    engine.current_provider = None

    out = engine._get_providers_list()

    # Configured provider appears with its model.
    assert "claude" in out
    assert "some-model" in out
    # An unconfigured-but-registered provider still surfaces for discovery.
    assert "bedrock" in out
