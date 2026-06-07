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
from omnimancer.providers.factory import ProviderFactory

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
