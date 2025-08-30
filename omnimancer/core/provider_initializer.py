"""
Provider initialization optimization module.

This module provides optimized initialization for providers with
lazy loading and caching mechanisms to improve performance.
"""

import importlib
import inspect
import logging
import threading
import time
from typing import Dict, List, Type

from ..providers.base import BaseProvider
from .models import EnhancedModelInfo, ProviderConfig

logger = logging.getLogger(__name__)


class ProviderInitializer:
    """
    Optimized provider initialization with lazy loading and caching.

    This class handles lazy loading of provider classes and caching
    of provider instances and model information to improve performance.
    """

    # Class-level caches
    _provider_classes: Dict[str, Type[BaseProvider]] = {}
    _provider_instances: Dict[str, BaseProvider] = {}
    _model_cache: Dict[str, List[EnhancedModelInfo]] = {}
    _cache_timestamps: Dict[str, float] = {}
    _cache_ttl: float = 3600.0  # 1 hour cache TTL by default

    # Locks for thread safety
    _class_lock = threading.RLock()
    _instance_lock = threading.RLock()
    _model_lock = threading.RLock()

    # Configuration
    _lazy_loading_enabled: bool = True
    _cache_enabled: bool = True

    @classmethod
    def get_provider_class(cls, provider_name: str) -> Type[BaseProvider]:
        """
        Get provider class with lazy loading.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider class

        Raises:
            ImportError: If provider module cannot be imported
            ValueError: If provider class cannot be found
        """
        with cls._class_lock:
            # Check if class is already loaded
            if provider_name in cls._provider_classes:
                return cls._provider_classes[provider_name]

            # Lazy load provider class
            try:
                # Map provider name to module path
                module_path = cls._get_module_path(provider_name)

                # Import module
                module = importlib.import_module(module_path)

                # Find provider class in module
                provider_class = None
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, BaseProvider)
                        and obj != BaseProvider
                        and name.lower().endswith("provider")
                    ):
                        provider_class = obj
                        break

                if not provider_class:
                    raise ValueError(f"Could not find provider class in {module_path}")

                # Cache provider class
                cls._provider_classes[provider_name] = provider_class
                logger.debug(f"Lazy loaded provider class for {provider_name}")

                return provider_class

            except ImportError as e:
                logger.error(
                    f"Failed to import provider module for {provider_name}: {e}"
                )
                raise
            except Exception as e:
                logger.error(f"Error loading provider class for {provider_name}: {e}")
                raise ValueError(
                    f"Failed to load provider class for {provider_name}: {e}"
                )

    @classmethod
    def _get_module_path(cls, provider_name: str) -> str:
        """
        Map provider name to module path.

        Args:
            provider_name: Name of the provider

        Returns:
            Module path
        """
        # Special case for claude-code provider
        if provider_name == "claude-code":
            return "omnimancer.providers.claude_code"

        # Standard mapping
        return f"omnimancer.providers.{provider_name}"

    @classmethod
    def get_provider_instance(
        cls,
        provider_name: str,
        config: ProviderConfig,
        config_manager: "ConfigManager" = None,
    ) -> BaseProvider:
        """
        Get or create cached provider instance.

        Args:
            provider_name: Name of the provider
            config: Provider configuration
            config_manager: Optional config manager for API key decryption

        Returns:
            Provider instance
        """
        if not cls._cache_enabled:
            # If caching is disabled, create new instance every time
            return cls._create_provider_instance(provider_name, config, config_manager)

        # Generate cache key based on provider name and config
        cache_key = cls._generate_cache_key(provider_name, config)

        with cls._instance_lock:
            # Check if instance is already cached and valid
            if cache_key in cls._provider_instances and cls._is_cache_valid(cache_key):
                logger.debug(f"Using cached provider instance for {provider_name}")
                return cls._provider_instances[cache_key]

            # Create new instance
            instance = cls._create_provider_instance(
                provider_name, config, config_manager
            )

            # Cache instance
            cls._provider_instances[cache_key] = instance
            cls._cache_timestamps[cache_key] = time.time()

            return instance

    @classmethod
    def _create_provider_instance(
        cls,
        provider_name: str,
        config: ProviderConfig,
        config_manager: "ConfigManager" = None,
    ) -> BaseProvider:
        """
        Create a new provider instance.

        Args:
            provider_name: Name of the provider
            config: Provider configuration
            config_manager: Optional config manager for API key decryption

        Returns:
            Provider instance
        """
        # Get provider class
        provider_class = cls.get_provider_class(provider_name)

        # Get decrypted API key
        api_key = config.api_key
        if config_manager and api_key:
            try:
                # Use config manager to get properly decrypted API key
                decrypted_key = config_manager.get_api_key(provider_name)
                if decrypted_key:
                    api_key = decrypted_key
            except Exception as e:
                logger.warning(f"Failed to decrypt API key for {provider_name}: {e}")
                # Fall back to using the key as-is
                pass

        # If still no valid API key, try environment variables
        if not api_key or api_key.startswith("your-") or api_key.startswith("sk-your"):
            from .env_loader import load_api_key_from_env

            env_key = load_api_key_from_env(provider_name)
            if env_key:
                api_key = env_key
                logger.info(f"Using API key from environment for {provider_name}")

        # Create instance
        # Filter out None values to allow provider defaults to work
        kwargs = {
            k: v
            for k, v in config.model_dump(exclude={"api_key", "model"}).items()
            if v is not None
        }
        instance = provider_class(api_key=api_key, model=config.model, **kwargs)

        return instance

    @classmethod
    def _generate_cache_key(cls, provider_name: str, config: ProviderConfig) -> str:
        """
        Generate cache key for provider instance.

        Args:
            provider_name: Name of the provider
            config: Provider configuration

        Returns:
            Cache key
        """
        # Include key configuration parameters in cache key
        key_params = {
            "model": config.model,
            "base_url": getattr(config, "base_url", None),
            "organization": getattr(config, "organization", None),
            "project_id": getattr(config, "project_id", None),
            "azure_endpoint": getattr(config, "azure_endpoint", None),
            "azure_deployment": getattr(config, "azure_deployment", None),
        }

        # Filter out None values
        key_params = {k: v for k, v in key_params.items() if v is not None}

        # Generate key
        key_str = f"{provider_name}:{config.model}:{hash(str(key_params))}"
        return key_str

    @classmethod
    def _is_cache_valid(cls, cache_key: str) -> bool:
        """
        Check if cached data is still valid.

        Args:
            cache_key: Cache key

        Returns:
            True if cache is valid
        """
        if cache_key not in cls._cache_timestamps:
            return False

        return (time.time() - cls._cache_timestamps[cache_key]) < cls._cache_ttl

    @classmethod
    def get_model_info(
        cls, provider_name: str, enhanced: bool = True
    ) -> List[EnhancedModelInfo]:
        """
        Get cached model information for a provider.

        Args:
            provider_name: Name of the provider
            enhanced: Whether to return enhanced model info

        Returns:
            List of model info objects
        """
        if not cls._cache_enabled:
            # If caching is disabled, fetch models every time
            return cls._fetch_model_info(provider_name, enhanced)

        with cls._model_lock:
            cache_key = f"{provider_name}:{'enhanced' if enhanced else 'basic'}"

            # Check if models are already cached and valid
            if cache_key in cls._model_cache and cls._is_cache_valid(cache_key):
                logger.debug(f"Using cached model info for {provider_name}")
                return cls._model_cache[cache_key]

            # Fetch models
            models = cls._fetch_model_info(provider_name, enhanced)

            # Cache models
            cls._model_cache[cache_key] = models
            cls._cache_timestamps[cache_key] = time.time()

            return models

    @classmethod
    def _fetch_model_info(
        cls, provider_name: str, enhanced: bool
    ) -> List[EnhancedModelInfo]:
        """
        Fetch model information from provider.

        Args:
            provider_name: Name of the provider
            enhanced: Whether to return enhanced model info

        Returns:
            List of model info objects
        """
        try:
            # Get provider class
            provider_class = cls.get_provider_class(provider_name)

            # Create temporary instance with dummy credentials
            temp_provider = provider_class(api_key="dummy", model="dummy")

            # Get models
            models = temp_provider.get_available_models()

            # Convert to EnhancedModelInfo if needed
            from .models import ModelInfo

            if enhanced and models and isinstance(models[0], ModelInfo):
                models = [EnhancedModelInfo.from_model_info(model) for model in models]

            return models

        except Exception as e:
            logger.error(f"Error fetching model info for {provider_name}: {e}")
            return []

    @classmethod
    def clear_caches(cls) -> None:
        """Clear all caches."""
        with cls._class_lock:
            cls._provider_classes.clear()

        with cls._instance_lock:
            cls._provider_instances.clear()

        with cls._model_lock:
            cls._model_cache.clear()

        cls._cache_timestamps.clear()
        logger.info("Cleared all provider caches")

    @classmethod
    def set_cache_ttl(cls, ttl_seconds: float) -> None:
        """
        Set cache TTL (time to live).

        Args:
            ttl_seconds: TTL in seconds
        """
        cls._cache_ttl = ttl_seconds
        logger.info(f"Set provider cache TTL to {ttl_seconds} seconds")

    @classmethod
    def enable_lazy_loading(cls, enabled: bool = True) -> None:
        """
        Enable or disable lazy loading.

        Args:
            enabled: Whether to enable lazy loading
        """
        cls._lazy_loading_enabled = enabled
        logger.info(f"{'Enabled' if enabled else 'Disabled'} provider lazy loading")

    @classmethod
    def enable_caching(cls, enabled: bool = True) -> None:
        """
        Enable or disable caching.

        Args:
            enabled: Whether to enable caching
        """
        cls._cache_enabled = enabled
        if not enabled:
            cls.clear_caches()
        logger.info(f"{'Enabled' if enabled else 'Disabled'} provider caching")

    @classmethod
    def preload_provider_classes(cls, provider_names: List[str]) -> None:
        """
        Preload provider classes.

        Args:
            provider_names: List of provider names to preload
        """
        for provider_name in provider_names:
            try:
                cls.get_provider_class(provider_name)
                logger.debug(f"Preloaded provider class for {provider_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to preload provider class for {provider_name}: {e}"
                )

    @classmethod
    async def initialize_providers(
        cls,
        provider_configs: Dict[str, ProviderConfig],
        config_manager: "ConfigManager" = None,
    ) -> Dict[str, BaseProvider]:
        """
        Initialize all providers from configuration.

        Args:
            provider_configs: Dictionary of provider configurations
            config_manager: Optional config manager for API key decryption

        Returns:
            Dictionary of initialized provider instances
        """
        from ..providers.factory import ProviderFactory

        providers = {}

        for provider_name, config in provider_configs.items():
            try:
                # Use ProviderFactory instead of direct class loading
                provider = ProviderFactory.create_provider(
                    provider_name, config, config_manager
                )
                providers[provider_name] = provider
                logger.info(f"Initialized provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")
                # Continue with other providers instead of failing completely
                continue

        return providers

    @classmethod
    def preload_model_info(cls, provider_names: List[str]) -> None:
        """
        Preload model information.

        Args:
            provider_names: List of provider names to preload
        """
        for provider_name in provider_names:
            try:
                cls.get_model_info(provider_name)
                logger.debug(f"Preloaded model info for {provider_name}")
            except Exception as e:
                logger.warning(f"Failed to preload model info for {provider_name}: {e}")
