"""
Provider factory for Omnimancer.

This module provides a factory for creating AI provider instances
based on configuration and provider type.
"""

from typing import Dict, List, Type

from ..core.models import EnhancedModelInfo, ModelInfo, ProviderConfig
from ..utils.errors import ConfigurationError
from .azure import AzureProvider
from .base import BaseProvider
from .bedrock import BedrockProvider
from .claude import ClaudeProvider
from .claude_code import ClaudeCodeProvider
from .cohere import CohereProvider
from .gemini import GeminiProvider
from .mistral import MistralProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider
from .perplexity import PerplexityProvider
from .vertex import VertexAIProvider
from .xai import XAIProvider


class ProviderFactory:
    """
    Factory for creating AI provider instances.

    This class manages the registration and creation of different
    AI provider implementations with lazy loading and caching.
    """

    _providers: Dict[str, Type[BaseProvider]] = {}
    _provider_instances: Dict[str, BaseProvider] = {}
    _model_cache: Dict[str, List[ModelInfo]] = {}
    _enhanced_model_cache: Dict[str, List[EnhancedModelInfo]] = {}
    _cache_timestamps: Dict[str, float] = {}
    _cache_ttl: float = 3600.0  # 1 hour cache TTL
    _lazy_loading_enabled: bool = True

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """
        Register a provider class.

        Args:
            name: Provider name (e.g., "claude", "openai")
            provider_class: Provider class to register
        """
        cls._providers[name] = provider_class
        # Clear cached data for this provider when re-registering
        cls._clear_provider_cache(name)

    @classmethod
    def _clear_provider_cache(cls, provider_name: str) -> None:
        """Clear cached data for a specific provider."""

        if provider_name in cls._model_cache:
            del cls._model_cache[provider_name]
        if provider_name in cls._enhanced_model_cache:
            del cls._enhanced_model_cache[provider_name]
        if provider_name in cls._cache_timestamps:
            del cls._cache_timestamps[provider_name]
        if provider_name in cls._provider_instances:
            del cls._provider_instances[provider_name]

    @classmethod
    def _is_cache_valid(cls, provider_name: str) -> bool:
        """Check if cached data for a provider is still valid."""
        import time

        if provider_name not in cls._cache_timestamps:
            return False
        return (time.time() - cls._cache_timestamps[provider_name]) < cls._cache_ttl

    @classmethod
    def _get_cached_provider_instance(
        cls, provider_name: str, config: ProviderConfig
    ) -> BaseProvider:
        """Get or create a cached provider instance."""
        cache_key = f"{provider_name}:{config.model}:{hash(str(config.dict()))}"

        if cache_key in cls._provider_instances and cls._is_cache_valid(cache_key):
            return cls._provider_instances[cache_key]

        # Create new instance
        if provider_name not in cls._providers:
            raise ConfigurationError(f"Unknown provider: {provider_name}")

        provider_class = cls._providers[provider_name]
        instance = provider_class(
            api_key=config.api_key,
            model=config.model,
            **config.dict(exclude={"api_key", "model"}),
        )

        # Cache the instance
        import time

        cls._provider_instances[cache_key] = instance
        cls._cache_timestamps[cache_key] = time.time()

        return instance

    @classmethod
    def _get_cached_models(cls, provider_name: str, enhanced: bool = False) -> List:
        """Get cached model information for a provider."""
        import time

        cache_dict = cls._enhanced_model_cache if enhanced else cls._model_cache

        if provider_name in cache_dict and cls._is_cache_valid(provider_name):
            return cache_dict[provider_name]

        # Cache miss - need to fetch models
        if provider_name not in cls._providers:
            return []

        try:
            provider_class = cls._providers[provider_name]
            temp_provider = provider_class(api_key="dummy", model="dummy")
            models = temp_provider.get_available_models()

            if enhanced:
                # Convert to EnhancedModelInfo if needed
                if models and isinstance(models[0], ModelInfo):
                    models = [
                        EnhancedModelInfo.from_model_info(model) for model in models
                    ]
                cls._enhanced_model_cache[provider_name] = models
            else:
                # Convert to ModelInfo if needed
                if models and isinstance(models[0], EnhancedModelInfo):
                    models = [model.to_model_info() for model in models]
                cls._model_cache[provider_name] = models

            cls._cache_timestamps[provider_name] = time.time()
            return models

        except Exception:
            # Return empty list on error
            cache_dict[provider_name] = []
            cls._cache_timestamps[provider_name] = time.time()
            return []

    @classmethod
    def clear_all_caches(cls) -> None:
        """Clear all cached data."""
        cls._model_cache.clear()
        cls._enhanced_model_cache.clear()
        cls._cache_timestamps.clear()
        cls._provider_instances.clear()

    @classmethod
    def set_cache_ttl(cls, ttl_seconds: float) -> None:
        """Set the cache TTL (time to live) in seconds."""
        cls._cache_ttl = ttl_seconds

    @classmethod
    def enable_lazy_loading(cls, enabled: bool = True) -> None:
        """Enable or disable lazy loading."""
        cls._lazy_loading_enabled = enabled
        if not enabled:
            # If disabling lazy loading, clear caches to force immediate loading
            cls.clear_all_caches()

    @classmethod
    def create_provider(
        cls,
        name: str,
        config: ProviderConfig,
        config_manager: "ConfigManager" = None,
    ) -> BaseProvider:
        """
        Create a provider instance.

        Args:
            name: Provider name
            config: Provider configuration
            config_manager: Optional config manager for API key decryption

        Returns:
            Provider instance

        Raises:
            ConfigurationError: If provider is not registered
        """
        if name not in cls._providers:
            raise ConfigurationError(f"Unknown provider: {name}")

        # Use the optimized provider initializer for lazy loading and caching
        from ..core.provider_initializer import ProviderInitializer

        return ProviderInitializer.get_provider_instance(name, config, config_manager)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Get list of available provider names.

        Returns:
            List of registered provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def is_provider_available(cls, name: str) -> bool:
        """
        Check if a provider is available.

        Args:
            name: Provider name

        Returns:
            True if provider is registered, False otherwise
        """
        return name in cls._providers

    @classmethod
    def get_all_models(cls) -> Dict[str, List[ModelInfo]]:
        """
        Get all available models from all registered providers.

        Returns:
            Dictionary mapping provider names to their available models (legacy format)
        """
        all_models = {}

        for provider_name, provider_class in cls._providers.items():
            try:
                # Create a temporary instance to get model info
                # We'll use a dummy config since we only need model info
                temp_provider = provider_class(api_key="dummy", model="dummy")
                models = temp_provider.get_available_models()

                # Convert EnhancedModelInfo to ModelInfo for backward compatibility
                if models and isinstance(models[0], EnhancedModelInfo):
                    all_models[provider_name] = [
                        model.to_model_info() for model in models
                    ]
                else:
                    all_models[provider_name] = models
            except Exception:
                # If we can't create the provider, skip it
                all_models[provider_name] = []

        return all_models

    @classmethod
    def get_all_enhanced_models(cls) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get all available models with enhanced information from all registered providers.

        Returns:
            Dictionary mapping provider names to their enhanced model information
        """
        all_models = {}

        for provider_name, provider_class in cls._providers.items():
            try:
                # Create a temporary instance to get model info
                temp_provider = provider_class(api_key="dummy", model="dummy")
                models = temp_provider.get_available_models()

                # Convert ModelInfo to EnhancedModelInfo if needed
                if models and isinstance(models[0], ModelInfo):
                    all_models[provider_name] = [
                        EnhancedModelInfo.from_model_info(model) for model in models
                    ]
                else:
                    all_models[provider_name] = models
            except Exception:
                # If we can't create the provider, skip it
                all_models[provider_name] = []

        return all_models

    @classmethod
    def get_all_models_with_capabilities(cls) -> Dict[str, Dict[str, any]]:
        """
        Get all available models with their capabilities and provider info.

        Returns:
            Dictionary with provider info and models with capabilities
        """
        result = {}

        for provider_name, provider_class in cls._providers.items():
            try:
                # Create a temporary instance to get model info
                temp_provider = provider_class(api_key="dummy", model="dummy")
                models = temp_provider.get_available_models()

                # Get provider capabilities
                provider_supports_tools = temp_provider.supports_tools()
                provider_supports_multimodal = temp_provider.supports_multimodal()

                result[provider_name] = {
                    "models": models,
                    "provider_capabilities": {
                        "supports_tools": provider_supports_tools,
                        "supports_multimodal": provider_supports_multimodal,
                        "supports_streaming": temp_provider.supports_streaming(),
                    },
                    "latest_models": [
                        m for m in models if getattr(m, "latest_version", False)
                    ],
                    "tool_capable_models": [
                        m for m in models if getattr(m, "supports_tools", False)
                    ],
                    "multimodal_models": [
                        m for m in models if getattr(m, "supports_multimodal", False)
                    ],
                }
            except Exception:
                # If we can't create the provider, provide minimal info
                result[provider_name] = {
                    "models": [],
                    "provider_capabilities": {
                        "supports_tools": False,
                        "supports_multimodal": False,
                        "supports_streaming": False,
                    },
                    "latest_models": [],
                    "tool_capable_models": [],
                    "multimodal_models": [],
                }

        return result

    @classmethod
    def get_models_for_provider(cls, provider_name: str) -> List[ModelInfo]:
        """
        Get available models for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            List of ModelInfo objects for the provider

        Raises:
            ConfigurationError: If provider is not registered
        """
        if provider_name not in cls._providers:
            raise ConfigurationError(f"Unknown provider: {provider_name}")

        provider_class = cls._providers[provider_name]
        try:
            # Create a temporary instance to get model info
            temp_provider = provider_class(api_key="dummy", model="dummy")
            return temp_provider.get_available_models()
        except Exception as e:
            raise ConfigurationError(
                f"Failed to get models for provider {provider_name}: {e}"
            )

    @classmethod
    def validate_model_for_provider(cls, provider_name: str, model_name: str) -> bool:
        """
        Validate that a model is available for a specific provider.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model

        Returns:
            True if model is available for the provider
        """
        try:
            models = cls.get_models_for_provider(provider_name)
            return any(model.name == model_name for model in models)
        except ConfigurationError:
            return False

    @classmethod
    def get_latest_models(cls) -> Dict[str, List[ModelInfo]]:
        """
        Get the latest models from all providers.

        Returns:
            Dictionary mapping provider names to their latest models
        """
        latest_models = {}

        for provider_name, provider_class in cls._providers.items():
            try:
                temp_provider = provider_class(api_key="dummy", model="dummy")
                all_models = temp_provider.get_available_models()
                latest_models[provider_name] = [
                    model
                    for model in all_models
                    if getattr(model, "latest_version", False)
                ]
            except Exception:
                latest_models[provider_name] = []

        return latest_models

    @classmethod
    def get_provider_capabilities(cls, provider_name: str) -> Dict[str, bool]:
        """
        Get capabilities for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary with provider capabilities

        Raises:
            ConfigurationError: If provider is not registered
        """
        if provider_name not in cls._providers:
            raise ConfigurationError(f"Unknown provider: {provider_name}")

        try:
            provider_class = cls._providers[provider_name]
            temp_provider = provider_class(api_key="dummy", model="dummy")

            return {
                "supports_tools": temp_provider.supports_tools(),
                "supports_multimodal": temp_provider.supports_multimodal(),
                "supports_streaming": temp_provider.supports_streaming(),
            }
        except Exception as e:
            raise ConfigurationError(
                f"Failed to get capabilities for provider {provider_name}: {e}"
            )

    @classmethod
    async def check_provider_health(
        cls, provider_name: str, config: ProviderConfig
    ) -> Dict[str, any]:
        """
        Check the health status of a provider.

        Args:
            provider_name: Name of the provider
            config: Provider configuration

        Returns:
            Dictionary with health status information
        """
        if provider_name not in cls._providers:
            return {
                "status": "error",
                "message": f"Unknown provider: {provider_name}",
                "available": False,
                "credentials_valid": False,
            }

        try:
            # Create provider instance
            provider = cls.create_provider(provider_name, config)

            # Check credentials
            credentials_valid = await provider.validate_credentials()

            # Get model info to check if model is available
            try:
                model_info = provider.get_model_info()
                model_available = model_info.available
            except Exception:
                model_available = False

            # Determine overall status
            if credentials_valid and model_available:
                status = "healthy"
                message = "Provider is working correctly"
            elif credentials_valid:
                status = "warning"
                message = f'Provider accessible but model "{config.model}" may not be available'
            else:
                status = "error"
                message = "Invalid credentials or provider not accessible"

            return {
                "status": status,
                "message": message,
                "available": True,
                "credentials_valid": credentials_valid,
                "model_available": model_available,
                "provider_capabilities": cls.get_provider_capabilities(provider_name),
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "available": False,
                "credentials_valid": False,
                "model_available": False,
            }

    @classmethod
    async def get_all_provider_health(
        cls, configs: Dict[str, ProviderConfig]
    ) -> Dict[str, Dict[str, any]]:
        """
        Check health status for all configured providers.

        Args:
            configs: Dictionary mapping provider names to their configurations

        Returns:
            Dictionary mapping provider names to their health status
        """
        health_status = {}

        for provider_name in cls._providers.keys():
            if provider_name in configs:
                health_status[provider_name] = await cls.check_provider_health(
                    provider_name, configs[provider_name]
                )
            else:
                health_status[provider_name] = {
                    "status": "not_configured",
                    "message": "Provider not configured",
                    "available": True,  # Provider class is available
                    "credentials_valid": False,
                    "configured": False,
                }

        return health_status

    @classmethod
    def get_models_by_capability(cls, capability: str) -> Dict[str, List[ModelInfo]]:
        """
        Get models that support a specific capability.

        Args:
            capability: Capability to filter by ('tools', 'multimodal', 'streaming')

        Returns:
            Dictionary mapping provider names to models with the capability
        """
        capability_models = {}

        for provider_name, provider_class in cls._providers.items():
            try:
                temp_provider = provider_class(api_key="dummy", model="dummy")
                all_models = temp_provider.get_available_models()

                if capability == "tools":
                    filtered_models = [
                        m for m in all_models if getattr(m, "supports_tools", False)
                    ]
                elif capability == "multimodal":
                    filtered_models = [
                        m
                        for m in all_models
                        if getattr(m, "supports_multimodal", False)
                    ]
                elif capability == "streaming":
                    # For streaming, check provider capability since it's not model-specific
                    if temp_provider.supports_streaming():
                        filtered_models = all_models
                    else:
                        filtered_models = []
                else:
                    filtered_models = []

                # Convert to legacy format if needed
                if filtered_models and isinstance(
                    filtered_models[0], EnhancedModelInfo
                ):
                    capability_models[provider_name] = [
                        model.to_model_info() for model in filtered_models
                    ]
                else:
                    capability_models[provider_name] = filtered_models
            except Exception:
                capability_models[provider_name] = []

        return capability_models

    @classmethod
    def get_enhanced_models_by_capability(
        cls, capability: str
    ) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get enhanced models that support a specific capability.

        Args:
            capability: Capability to filter by ('tools', 'multimodal', 'streaming', 'latest', 'free')

        Returns:
            Dictionary mapping provider names to enhanced models with the capability
        """
        capability_models = {}

        for provider_name, provider_class in cls._providers.items():
            try:
                temp_provider = provider_class(api_key="dummy", model="dummy")
                all_models = temp_provider.get_available_models()

                # Convert to EnhancedModelInfo if needed
                if all_models and isinstance(all_models[0], ModelInfo):
                    all_models = [
                        EnhancedModelInfo.from_model_info(model) for model in all_models
                    ]

                if capability == "tools":
                    filtered_models = [m for m in all_models if m.supports_tools]
                elif capability == "multimodal":
                    filtered_models = [m for m in all_models if m.supports_multimodal]
                elif capability == "streaming":
                    # For streaming, check provider capability since it's not model-specific
                    if temp_provider.supports_streaming():
                        filtered_models = all_models
                    else:
                        filtered_models = []
                elif capability == "latest":
                    filtered_models = [m for m in all_models if m.latest_version]
                elif capability == "free":
                    filtered_models = [m for m in all_models if m.is_free]
                elif capability == "available":
                    filtered_models = [
                        m for m in all_models if m.available and not m.deprecated
                    ]
                else:
                    filtered_models = []

                if filtered_models:
                    capability_models[provider_name] = filtered_models
            except Exception:
                capability_models[provider_name] = []

        return capability_models

    @classmethod
    def get_latest_models_enhanced(cls) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get the latest models from all providers with enhanced information.

        Returns:
            Dictionary mapping provider names to their latest enhanced models
        """
        return cls.get_enhanced_models_by_capability("latest")

    @classmethod
    def get_models_by_performance(
        cls, min_swe_score: float
    ) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get models filtered by minimum SWE score.

        Args:
            min_swe_score: Minimum SWE score (0-100)

        Returns:
            Dictionary mapping provider names to high-performance models
        """
        performance_models = {}

        for provider_name, provider_class in cls._providers.items():
            try:
                temp_provider = provider_class(api_key="dummy", model="dummy")
                all_models = temp_provider.get_available_models()

                # Convert to EnhancedModelInfo if needed
                if all_models and isinstance(all_models[0], ModelInfo):
                    all_models = [
                        EnhancedModelInfo.from_model_info(model) for model in all_models
                    ]

                high_perf_models = [
                    m
                    for m in all_models
                    if m.swe_score is not None and m.swe_score >= min_swe_score
                ]

                if high_perf_models:
                    performance_models[provider_name] = high_perf_models
            except Exception:
                performance_models[provider_name] = []

        return performance_models

    @classmethod
    def get_models_by_pricing(
        cls, max_cost_per_million: float
    ) -> Dict[str, List[EnhancedModelInfo]]:
        """
        Get models filtered by maximum cost per million tokens.

        Args:
            max_cost_per_million: Maximum average cost per million tokens

        Returns:
            Dictionary mapping provider names to models within budget
        """
        budget_models = {}

        for provider_name, provider_class in cls._providers.items():
            try:
                temp_provider = provider_class(api_key="dummy", model="dummy")
                all_models = temp_provider.get_available_models()

                # Convert to EnhancedModelInfo if needed
                if all_models and isinstance(all_models[0], ModelInfo):
                    all_models = [
                        EnhancedModelInfo.from_model_info(model) for model in all_models
                    ]

                affordable_models = []
                for model in all_models:
                    if model.is_free:
                        affordable_models.append(model)
                    else:
                        avg_cost = (
                            model.cost_per_million_input + model.cost_per_million_output
                        ) / 2
                        if avg_cost <= max_cost_per_million:
                            affordable_models.append(model)

                if affordable_models:
                    budget_models[provider_name] = affordable_models
            except Exception:
                budget_models[provider_name] = []

        return budget_models

    @classmethod
    def get_provider_model_summary(cls, provider_name: str) -> Dict[str, any]:
        """
        Get comprehensive model summary for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary with provider model summary and capabilities
        """
        if provider_name not in cls._providers:
            return {"error": f"Provider {provider_name} not registered"}

        try:
            provider_class = cls._providers[provider_name]
            temp_provider = provider_class(api_key="dummy", model="dummy")
            models = temp_provider.get_available_models()

            # Convert to EnhancedModelInfo if needed
            if models and isinstance(models[0], ModelInfo):
                models = [EnhancedModelInfo.from_model_info(model) for model in models]

            # Calculate statistics
            total_models = len(models)
            available_models = len(
                [m for m in models if m.available and not m.deprecated]
            )
            tool_capable = len([m for m in models if m.supports_tools])
            multimodal = len([m for m in models if m.supports_multimodal])
            free_models = len([m for m in models if m.is_free])
            latest_models = len([m for m in models if m.latest_version])
            deprecated_models = len([m for m in models if m.deprecated])

            # Get provider capabilities
            provider_caps = cls.get_provider_capabilities(provider_name)

            # Find best performing model
            best_model = None
            if models:
                scored_models = [m for m in models if m.swe_score is not None]
                if scored_models:
                    best_model = max(scored_models, key=lambda x: x.swe_score)

            # Find cheapest model
            cheapest_model = None
            if models:
                paid_models = [m for m in models if not m.is_free]
                if paid_models:
                    cheapest_model = min(
                        paid_models,
                        key=lambda x: (
                            x.cost_per_million_input + x.cost_per_million_output
                        )
                        / 2,
                    )

            return {
                "provider_name": provider_name,
                "total_models": total_models,
                "available_models": available_models,
                "tool_capable_models": tool_capable,
                "multimodal_models": multimodal,
                "free_models": free_models,
                "latest_models": latest_models,
                "deprecated_models": deprecated_models,
                "provider_capabilities": provider_caps,
                "best_performing_model": (
                    {
                        "name": best_model.name,
                        "swe_score": best_model.swe_score,
                        "swe_rating": best_model.swe_rating,
                    }
                    if best_model
                    else None
                ),
                "cheapest_model": (
                    {
                        "name": cheapest_model.name,
                        "cost_display": cheapest_model.get_cost_display(),
                    }
                    if cheapest_model
                    else None
                ),
                "model_names": [
                    m.name for m in models if m.available and not m.deprecated
                ],
            }

        except Exception as e:
            return {
                "error": f"Failed to get summary for provider {provider_name}: {str(e)}"
            }

    @classmethod
    def get_all_provider_summaries(cls) -> Dict[str, Dict[str, any]]:
        """
        Get comprehensive summaries for all registered providers.

        Returns:
            Dictionary mapping provider names to their summaries
        """
        summaries = {}

        for provider_name in cls._providers.keys():
            summaries[provider_name] = cls.get_provider_model_summary(provider_name)

        return summaries

    @classmethod
    def detect_latest_models(cls) -> Dict[str, List[str]]:
        """
        Detect and flag latest models across all providers.

        Returns:
            Dictionary mapping provider names to their latest model names
        """
        latest_models = {}

        for provider_name, provider_class in cls._providers.items():
            try:
                temp_provider = provider_class(api_key="dummy", model="dummy")
                models = temp_provider.get_available_models()

                # Convert to EnhancedModelInfo if needed
                if models and isinstance(models[0], ModelInfo):
                    models = [
                        EnhancedModelInfo.from_model_info(model) for model in models
                    ]

                # Find latest models
                latest = [m.name for m in models if m.latest_version]
                if latest:
                    latest_models[provider_name] = latest

            except Exception:
                latest_models[provider_name] = []

        return latest_models

    @classmethod
    def get_capability_matrix(cls) -> Dict[str, Dict[str, bool]]:
        """
        Get capability matrix showing which providers support which features.

        Returns:
            Dictionary with provider capabilities matrix
        """
        matrix = {}

        for provider_name in cls._providers.keys():
            try:
                caps = cls.get_provider_capabilities(provider_name)
                matrix[provider_name] = caps
            except Exception:
                matrix[provider_name] = {
                    "supports_tools": False,
                    "supports_multimodal": False,
                    "supports_streaming": False,
                }

        return matrix

    @classmethod
    async def batch_health_check(
        cls, configs: Dict[str, ProviderConfig], timeout: float = 30.0
    ) -> Dict[str, Dict[str, any]]:
        """
        Perform batch health checks on multiple providers with timeout.

        Args:
            configs: Dictionary mapping provider names to their configurations
            timeout: Timeout for each health check in seconds

        Returns:
            Dictionary mapping provider names to their health status
        """
        import asyncio

        async def check_single_provider(
            provider_name: str, config: ProviderConfig
        ) -> tuple[str, Dict[str, any]]:
            try:
                # Add timeout to individual health check
                health_result = await asyncio.wait_for(
                    cls.check_provider_health(provider_name, config),
                    timeout=timeout,
                )
                return provider_name, health_result
            except asyncio.TimeoutError:
                return provider_name, {
                    "status": "timeout",
                    "message": f"Health check timed out after {timeout} seconds",
                    "available": True,
                    "credentials_valid": False,
                    "model_available": False,
                }
            except Exception as e:
                return provider_name, {
                    "status": "error",
                    "message": f"Health check failed: {str(e)}",
                    "available": False,
                    "credentials_valid": False,
                    "model_available": False,
                }

        # Create tasks for all configured providers
        tasks = []
        for provider_name, config in configs.items():
            if provider_name in cls._providers:
                tasks.append(check_single_provider(provider_name, config))

        # Execute all health checks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        health_status = {}
        for result in results:
            if isinstance(result, tuple):
                provider_name, status = result
                health_status[provider_name] = status
            else:
                # Handle exceptions
                health_status["unknown"] = {
                    "status": "error",
                    "message": f"Unexpected error: {str(result)}",
                    "available": False,
                    "credentials_valid": False,
                    "model_available": False,
                }

        # Add unconfigured providers
        for provider_name in cls._providers.keys():
            if provider_name not in health_status:
                health_status[provider_name] = {
                    "status": "not_configured",
                    "message": "Provider not configured",
                    "available": True,
                    "credentials_valid": False,
                    "configured": False,
                }

        return health_status


# Register built-in providers
ProviderFactory.register_provider("claude", ClaudeProvider)
ProviderFactory.register_provider("openai", OpenAIProvider)
ProviderFactory.register_provider("gemini", GeminiProvider)
ProviderFactory.register_provider("cohere", CohereProvider)
ProviderFactory.register_provider("ollama", OllamaProvider)

# Register new providers
ProviderFactory.register_provider("perplexity", PerplexityProvider)
ProviderFactory.register_provider("xai", XAIProvider)
ProviderFactory.register_provider("mistral", MistralProvider)
ProviderFactory.register_provider("azure", AzureProvider)
ProviderFactory.register_provider("vertex", VertexAIProvider)
ProviderFactory.register_provider("bedrock", BedrockProvider)
ProviderFactory.register_provider("openrouter", OpenRouterProvider)
ProviderFactory.register_provider("claude-code", ClaudeCodeProvider)
