"""Cross-provider capability-consistency contract tests.

These tests enforce that a provider never *advertises* a capability it cannot
actually deliver. The motivating bug: a provider whose ``supports_tools()``
returned ``True`` without implementing ``send_message_with_tools`` would raise
``NotImplementedError`` at runtime from ``BaseProvider.send_message_with_tools``.

Two sources of capability truth coexist by design:

* ``PROVIDER_CAPABILITIES`` (``core/provider_capabilities.py``) — provider-level
  truth: "is this integration *capable* of X". Drives config defaults.
* ``BaseProvider.supports_*()`` instance methods — model-aware runtime truth:
  "does the *currently selected model* support X".

The contract below keeps the two coherent and guards the registry name lookup.
"""

import pytest

from omnimancer.core.provider_capabilities import (
    PROVIDER_CAPABILITIES,
    get_provider_capabilities,
)
from omnimancer.core.provider_initializer import ProviderInitializer
from omnimancer.providers.base import BaseProvider
from omnimancer.providers.claude import ClaudeProvider
from omnimancer.providers.factory import ProviderFactory
from omnimancer.providers.perplexity import PerplexityProvider

# All registered provider names, resolved once.
PROVIDER_NAMES = ProviderFactory.get_available_providers()

# Minimal kwargs needed to construct a default instance of providers that
# validate configuration eagerly. Providers not listed construct with just an
# api_key. Providers that need real credentials/endpoints are still fully
# covered by the class-level invariant (which needs no instance).
CONSTRUCT_KWARGS = {
    "azure": {"azure_endpoint": "https://example.openai.azure.com", "model": "gpt-4o"},
    "vertex": {"vertex_project": "test-project", "model": "gemini-1.5-pro"},
    "bedrock": {"model": "anthropic.claude-3-sonnet-20240229-v1:0"},
}


def _implements_tools(cls: type) -> bool:
    """True if the provider class overrides send_message_with_tools."""
    return cls.send_message_with_tools is not BaseProvider.send_message_with_tools


def _implements_streaming(cls: type) -> bool:
    """True if the provider class overrides send_message_stream.

    Uses identity comparison on the unbound method objects so that the check
    works correctly for subclasses.  Note: this assumes no intermediate abstract
    base class redefines ``send_message_stream`` as a pass-through; if such a
    class were added, the identity check would give a false negative.
    """
    return cls.send_message_stream is not BaseProvider.send_message_stream


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_registry_resolves_without_silent_fallback(name):
    """Every registered provider name resolves to its real registry entry.

    Guards the name-normalization bug where e.g. ``"claude-code"`` silently
    fell through to an empty ``ProviderCapabilities()`` default.
    """
    caps = get_provider_capabilities(name)
    assert any(
        caps is entry for entry in PROVIDER_CAPABILITIES.values()
    ), f"{name!r} fell through to an empty default instead of its registry entry"


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_registry_tools_match_implementation(name):
    """Registry must advertise tool support iff the provider wires it.

    This is the core contract that prevents a runtime NotImplementedError: the
    registry's ``supports_tools`` is the single provider-level signal, and it
    must agree with whether ``send_message_with_tools`` is actually implemented.
    """
    cls = ProviderInitializer.get_provider_class(name)
    assert cls is not None, f"could not resolve provider class for {name!r}"
    caps = get_provider_capabilities(name)
    implements = _implements_tools(cls)
    assert caps.supports_tools == implements, (
        f"{name!r}: registry supports_tools={caps.supports_tools} but "
        f"send_message_with_tools implemented={implements}. These must agree."
    )


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_registry_streaming_match_implementation(name):
    """Registry must advertise streaming support iff the provider wires it.

    Analogous to the tools contract: ``supports_streaming`` in the registry
    must match whether the provider class actually overrides
    ``send_message_stream``.  Currently only ``ClaudeProvider`` does so.
    """
    cls = ProviderInitializer.get_provider_class(name)
    assert cls is not None, f"could not resolve provider class for {name!r}"
    caps = get_provider_capabilities(name)
    implements = _implements_streaming(cls)
    assert caps.supports_streaming == implements, (
        f"{name!r}: registry supports_streaming={caps.supports_streaming} but "
        f"send_message_stream implemented={implements}. These must agree — "
        f"only providers that override send_message_stream() should be True."
    )


def test_claude_implements_streaming():
    """Positive guard: ClaudeProvider must implement send_message_stream.

    This test validates that the ``_implements_streaming`` helper correctly
    identifies the one provider that *does* implement streaming, ensuring
    the helper is not vacuously True or False for all providers.
    """
    assert _implements_streaming(ClaudeProvider), (
        "ClaudeProvider must override send_message_stream() — "
        "it is the only provider with a real streaming implementation."
    )


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_instance_tools_never_overclaim(name):
    """A constructible default instance must not claim tools it cannot deliver.

    Catches model-aware ``supports_tools()`` logic that returns True for some
    models without an implementation (the original Perplexity bug).
    """
    cls = ProviderInitializer.get_provider_class(name)
    kwargs = CONSTRUCT_KWARGS.get(name, {})
    try:
        instance = cls(api_key="test-key", **kwargs)
    except Exception:
        pytest.skip(f"{name!r} needs real credentials to construct")
    if instance.supports_tools():
        assert _implements_tools(cls), (
            f"{name!r}: supports_tools() is True but send_message_with_tools "
            f"is not implemented — this raises NotImplementedError at runtime."
        )


# ---------------------------------------------------------------------------
# Perplexity-specific tests
# ---------------------------------------------------------------------------


def test_perplexity_get_model_info_tools_is_false():
    """get_model_info() must never report supports_tools=True for any model.

    Perplexity's built-in web search (on -online models) is not the same as
    user-provided function calling; advertising tools support would cause a
    NotImplementedError at runtime since send_message_with_tools is not
    implemented.

    Online model names are fetched dynamically from get_available_models() to
    stay in sync when the model list changes.
    """
    provider = PerplexityProvider(api_key="test-key")

    # Dynamically find online models rather than hardcoding names, so this
    # test stays correct when the model list changes.
    online_model_names = [
        m.name for m in provider.get_available_models() if "online" in m.name
    ]
    assert (
        online_model_names
    ), "Expected at least one online model in get_available_models()"

    for model_name in online_model_names:
        provider_with_model = PerplexityProvider(api_key="test-key", model=model_name)
        info = provider_with_model.get_model_info()
        assert info.supports_tools is False, (
            f"get_model_info() for {model_name!r} reports supports_tools=True. "
            "Built-in web search is not user-provided function calling — "
            "send_message_with_tools is not implemented on PerplexityProvider."
        )


def test_perplexity_get_available_models_tools_is_false():
    """get_available_models() must never report supports_tools=True for any model.

    Validates both online and chat models in the full catalog returned by
    get_available_models().
    """
    provider = PerplexityProvider(api_key="test-key")
    models = provider.get_available_models()
    assert models, "Expected at least one model from get_available_models()"

    for model in models:
        assert model.supports_tools is False, (
            f"get_available_models() reports supports_tools=True for {model.name!r}. "
            "PerplexityProvider does not implement send_message_with_tools — "
            "Built-in web search is not user-provided function calling."
        )
