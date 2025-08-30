#!/usr/bin/env python3
"""Test script for ProviderRegistry class."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))


from omnimancer.core.models import (
    ChatContext,
    ChatResponse,
    EnhancedModelInfo,
    ModelInfo,
)
from omnimancer.core.provider_registry import ProviderRegistry
from omnimancer.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(api_key, model, **kwargs)

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        return ChatResponse(
            content="Mock response", model_used=self.model, tokens_used=10
        )

    async def validate_credentials(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            provider="mock",
            description="Mock model",
            max_tokens=4096,
            cost_per_token=0.001,
            available=True,
        )

    def get_available_models(self) -> list[ModelInfo]:
        return [self.get_model_info()]

    def supports_tools(self) -> bool:
        return True

    def supports_multimodal(self) -> bool:
        return False


def test_provider_registry():
    """Test ProviderRegistry functionality."""
    print("Testing ProviderRegistry class...")

    # Create registry
    registry = ProviderRegistry()

    # Test initial state
    assert len(registry.get_registered_providers()) == 0
    assert not registry.is_provider_registered("mock")

    # Test provider registration
    registry.register_provider("mock", MockProvider)
    assert registry.is_provider_registered("mock")
    assert "mock" in registry.get_registered_providers()

    # Test getting provider class
    provider_class = registry.get_provider_class("mock")
    assert provider_class == MockProvider

    # Test model catalog operations
    test_models = [
        EnhancedModelInfo(
            name="mock-model-1",
            provider="mock",
            description="First mock model",
            max_tokens=4096,
            cost_per_million_input=1.0,
            cost_per_million_output=2.0,
            swe_score=75.0,
            supports_tools=True,
            latest_version=True,
        ),
        EnhancedModelInfo(
            name="mock-model-2",
            provider="mock",
            description="Second mock model",
            max_tokens=8192,
            cost_per_million_input=5.0,
            cost_per_million_output=10.0,
            swe_score=45.0,
            supports_multimodal=True,
            is_free=False,
        ),
        EnhancedModelInfo(
            name="mock-free-model",
            provider="mock",
            description="Free mock model",
            max_tokens=2048,
            cost_per_million_input=0.0,
            cost_per_million_output=0.0,
            is_free=True,
        ),
    ]

    # Update provider models
    registry.update_provider_models("mock", test_models)

    # Test getting models
    models = registry.get_models_for_provider("mock")
    assert len(models) == 3
    assert models[0].name == "mock-model-1"

    # Test getting all models
    all_models = registry.get_all_models_with_details()
    assert "mock" in all_models
    assert len(all_models["mock"]) == 3

    # Test capability filtering
    tool_models = registry.get_models_by_capability("tools")
    assert "mock" in tool_models
    assert len(tool_models["mock"]) == 1
    assert tool_models["mock"][0].name == "mock-model-1"

    multimodal_models = registry.get_models_by_capability("multimodal")
    assert "mock" in multimodal_models
    assert len(multimodal_models["mock"]) == 1
    assert multimodal_models["mock"][0].name == "mock-model-2"

    latest_models = registry.get_latest_models()
    assert "mock" in latest_models
    assert len(latest_models["mock"]) == 1
    assert latest_models["mock"][0].name == "mock-model-1"

    free_models = registry.get_free_models()
    assert "mock" in free_models
    assert len(free_models["mock"]) == 1
    assert free_models["mock"][0].name == "mock-free-model"

    # Test pricing filter
    budget_models = registry.get_models_by_pricing(5.0)
    assert "mock" in budget_models
    # Should include mock-model-1 (avg 1.5) and free model
    assert len(budget_models["mock"]) == 2

    # Test performance filter
    high_perf_models = registry.get_models_by_performance(60.0)
    assert "mock" in high_perf_models
    assert len(high_perf_models["mock"]) == 1
    assert high_perf_models["mock"][0].name == "mock-model-1"

    # Test search
    search_results = registry.search_models("first")
    assert "mock" in search_results
    assert len(search_results["mock"]) == 1
    assert search_results["mock"][0].name == "mock-model-1"

    # Test get model by name
    specific_model = registry.get_model_by_name("mock", "mock-model-2")
    assert specific_model is not None
    assert specific_model.name == "mock-model-2"

    # Test catalog stats
    stats = registry.get_catalog_stats()
    assert stats["total_providers"] == 1
    assert stats["total_models"] == 3
    assert stats["tool_capable_models"] == 1
    assert stats["multimodal_models"] == 1
    assert stats["free_models"] == 1
    assert stats["latest_models"] == 1

    # Test provider summary
    summary = registry.get_provider_summary("mock")
    assert summary["name"] == "mock"
    assert summary["registered"] == True
    assert summary["total_models"] == 3
    assert summary["available_models"] == 3
    assert summary["tool_capable_models"] == 1
    assert summary["multimodal_models"] == 1
    assert summary["free_models"] == 1

    # Test validation
    errors = registry.validate_catalog()
    assert len(errors) == 0  # Should be no errors

    # Test unregistration
    assert registry.unregister_provider("mock") == True
    assert not registry.is_provider_registered("mock")
    assert registry.unregister_provider("nonexistent") == False

    # Test string representations
    registry.register_provider("mock", MockProvider)
    registry.update_provider_models("mock", test_models)

    str_repr = str(registry)
    assert "ProviderRegistry" in str_repr
    assert "1 providers" in str_repr
    assert "3 models" in str_repr

    print("✅ All ProviderRegistry tests passed!")


if __name__ == "__main__":
    test_provider_registry()
