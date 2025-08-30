"""
Provider registry for comprehensive model catalog management.

This module provides a centralized registry for managing AI providers
and their model catalogs with enhanced information.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from ..providers.base import BaseProvider
from ..utils.errors import ConfigurationError
from .models import EnhancedModelInfo

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Enhanced provider registry with comprehensive model catalog."""

    def __init__(self):
        """Initialize the provider registry."""
        self.providers: Dict[str, Type[BaseProvider]] = {}
        self.model_catalog: Dict[str, List[EnhancedModelInfo]] = {}
        self.last_update: Optional[datetime] = None
        self._provider_instances: Dict[str, BaseProvider] = {}

    def register_provider(self, name: str, provider_class: Type[BaseProvider]) -> None:
        """
        Register a provider with the registry.

        Args:
            name: Provider name (e.g., "claude", "openai")
            provider_class: Provider class to register
        """
        logger.info(f"Registering provider: {name}")
        self.providers[name] = provider_class

        # Initialize empty model catalog for this provider
        if name not in self.model_catalog:
            self.model_catalog[name] = []

    def unregister_provider(self, name: str) -> bool:
        """
        Unregister a provider from the registry.

        Args:
            name: Provider name to unregister

        Returns:
            True if provider was unregistered, False if not found
        """
        if name in self.providers:
            logger.info(f"Unregistering provider: {name}")
            del self.providers[name]

            # Clean up model catalog and instances
            if name in self.model_catalog:
                del self.model_catalog[name]
            if name in self._provider_instances:
                del self._provider_instances[name]

            return True
        return False

    def get_registered_providers(self) -> List[str]:
        """
        Get list of registered provider names.

        Returns:
            List of provider names
        """
        return list(self.providers.keys())

    def is_provider_registered(self, name: str) -> bool:
        """
        Check if a provider is registered.

        Args:
            name: Provider name

        Returns:
            True if provider is registered
        """
        return name in self.providers

    def get_provider_class(self, name: str) -> Type[BaseProvider]:
        """
        Get provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ConfigurationError: If provider is not registered
        """
        if name not in self.providers:
            raise ConfigurationError(f"Provider '{name}' is not registered")
        return self.providers[name]

    def get_all_models_with_details(
        self,
    ) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get comprehensive model catalog with all details.

        Returns:
            Dictionary mapping provider names to their enhanced model lists
        """
        return self.model_catalog.copy()

    def get_models_for_provider(self, provider_name: str) -> List[EnhancedModelInfo]:
        """
        Get models for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            List of enhanced model info for the provider
        """
        return self.model_catalog.get(provider_name, [])

    def update_provider_models(
        self, provider_name: str, models: List[EnhancedModelInfo]
    ) -> None:
        """
        Update model catalog for a specific provider.

        Args:
            provider_name: Name of the provider
            models: List of enhanced model info
        """
        logger.info(
            f"Updating model catalog for provider: {provider_name} ({len(models)} models)"
        )
        self.model_catalog[provider_name] = models
        self.last_update = datetime.now()

    def get_models_by_capability(
        self, capability: str
    ) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get models filtered by capability (tools, multimodal, etc.).

        Args:
            capability: Capability to filter by ('tools', 'multimodal', 'latest', 'free')

        Returns:
            Dictionary mapping provider names to filtered models
        """
        filtered_models = {}

        for provider_name, models in self.model_catalog.items():
            if capability == "tools":
                filtered = [m for m in models if m.supports_tools]
            elif capability == "multimodal":
                filtered = [m for m in models if m.supports_multimodal]
            elif capability == "latest":
                filtered = [m for m in models if m.latest_version]
            elif capability == "free":
                filtered = [m for m in models if m.is_free]
            elif capability == "available":
                filtered = [m for m in models if m.available and not m.deprecated]
            else:
                filtered = models

            if filtered:
                filtered_models[provider_name] = filtered

        return filtered_models

    def get_models_by_pricing(
        self, max_cost_per_million: float
    ) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get models filtered by maximum cost per million tokens.

        Args:
            max_cost_per_million: Maximum average cost per million tokens

        Returns:
            Dictionary mapping provider names to models within budget
        """
        filtered_models = {}

        for provider_name, models in self.model_catalog.items():
            budget_models = []
            for model in models:
                if model.is_free:
                    budget_models.append(model)
                else:
                    avg_cost = (
                        model.cost_per_million_input + model.cost_per_million_output
                    ) / 2
                    if avg_cost <= max_cost_per_million:
                        budget_models.append(model)

            if budget_models:
                filtered_models[provider_name] = budget_models

        return filtered_models

    def get_models_by_performance(
        self, min_swe_score: float
    ) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get models filtered by minimum SWE score.

        Args:
            min_swe_score: Minimum SWE score (0-100)

        Returns:
            Dictionary mapping provider names to high-performance models
        """
        filtered_models = {}

        for provider_name, models in self.model_catalog.items():
            high_perf_models = [
                m
                for m in models
                if m.swe_score is not None and m.swe_score >= min_swe_score
            ]

            if high_perf_models:
                filtered_models[provider_name] = high_perf_models

        return filtered_models

    def search_models(self, query: str) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Search models by name or description.

        Args:
            query: Search query string

        Returns:
            Dictionary mapping provider names to matching models
        """
        query_lower = query.lower()
        matching_models = {}

        for provider_name, models in self.model_catalog.items():
            matches = [
                m
                for m in models
                if query_lower in m.name.lower() or query_lower in m.description.lower()
            ]

            if matches:
                matching_models[provider_name] = matches

        return matching_models

    def get_model_by_name(
        self, provider_name: str, model_name: str
    ) -> Optional[EnhancedModelInfo]:
        """
        Get a specific model by provider and name.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model

        Returns:
            Enhanced model info if found, None otherwise
        """
        models = self.model_catalog.get(provider_name, [])
        for model in models:
            if model.name == model_name:
                return model
        return None

    def get_latest_models(self) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get the latest models from all providers.

        Returns:
            Dictionary mapping provider names to their latest models
        """
        return self.get_models_by_capability("latest")

    def get_free_models(self) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get all free models from all providers.

        Returns:
            Dictionary mapping provider names to their free models
        """
        return self.get_models_by_capability("free")

    def get_tool_capable_models(self) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get all models that support tool calling.

        Returns:
            Dictionary mapping provider names to their tool-capable models
        """
        return self.get_models_by_capability("tools")

    def get_multimodal_models(self) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get all models that support multimodal inputs.

        Returns:
            Dictionary mapping provider names to their multimodal models
        """
        return self.get_models_by_capability("multimodal")

    def get_catalog_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model catalog.

        Returns:
            Dictionary with catalog statistics
        """
        total_models = sum(len(models) for models in self.model_catalog.values())
        total_providers = len(self.providers)

        # Count models by capability
        tool_models = sum(
            len(models) for models in self.get_tool_capable_models().values()
        )
        multimodal_models = sum(
            len(models) for models in self.get_multimodal_models().values()
        )
        free_models = sum(len(models) for models in self.get_free_models().values())
        latest_models = sum(len(models) for models in self.get_latest_models().values())

        # Count models by performance tier
        high_perf = sum(
            len(models) for models in self.get_models_by_performance(60.0).values()
        )
        mid_perf = (
            sum(len(models) for models in self.get_models_by_performance(40.0).values())
            - high_perf
        )

        return {
            "total_providers": total_providers,
            "total_models": total_models,
            "tool_capable_models": tool_models,
            "multimodal_models": multimodal_models,
            "free_models": free_models,
            "latest_models": latest_models,
            "high_performance_models": high_perf,
            "mid_performance_models": mid_perf,
            "last_update": self.last_update,
            "providers": list(self.providers.keys()),
        }

    def validate_catalog(self) -> List[str]:
        """
        Validate the model catalog and return any issues found.

        Returns:
            List of validation error messages
        """
        errors = []

        for provider_name, models in self.model_catalog.items():
            if provider_name not in self.providers:
                errors.append(
                    f"Models exist for unregistered provider: {provider_name}"
                )
                continue

            for model in models:
                # Validate model data
                if not model.validate_pricing():
                    errors.append(f"Invalid pricing for {provider_name}:{model.name}")

                if not model.validate_swe_score():
                    errors.append(f"Invalid SWE score for {provider_name}:{model.name}")

                # Check for duplicate models
                duplicates = [m for m in models if m.name == model.name]
                if len(duplicates) > 1:
                    errors.append(
                        f"Duplicate model {model.name} in provider {provider_name}"
                    )

        return errors

    def clear_catalog(self) -> None:
        """Clear all model catalog data."""
        logger.info("Clearing model catalog")
        self.model_catalog.clear()
        self.last_update = None

    def get_provider_summary(self, provider_name: str) -> Dict[str, Any]:
        """
        Get summary information for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary with provider summary
        """
        if provider_name not in self.providers:
            return {"error": f"Provider {provider_name} not registered"}

        models = self.model_catalog.get(provider_name, [])

        return {
            "name": provider_name,
            "registered": True,
            "total_models": len(models),
            "available_models": len(
                [m for m in models if m.available and not m.deprecated]
            ),
            "tool_capable_models": len([m for m in models if m.supports_tools]),
            "multimodal_models": len([m for m in models if m.supports_multimodal]),
            "free_models": len([m for m in models if m.is_free]),
            "latest_models": len([m for m in models if m.latest_version]),
            "deprecated_models": len([m for m in models if m.deprecated]),
            "models": [m.name for m in models if m.available and not m.deprecated],
        }

    def __str__(self) -> str:
        """String representation of the registry."""
        stats = self.get_catalog_stats()
        return f"ProviderRegistry({stats['total_providers']} providers, {stats['total_models']} models)"

    def __repr__(self) -> str:
        """Detailed string representation of the registry."""
        return f"ProviderRegistry(providers={list(self.providers.keys())}, last_update={self.last_update})"
