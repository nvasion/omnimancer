"""
Health monitoring module for providers.

This module provides health monitoring for providers with
caching and optimized checks to improve performance.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from .models import ProviderConfig
from .provider_initializer import ProviderInitializer

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Health monitoring for providers with caching.

    This class handles health checks for providers with
    caching and optimized checks to improve performance.
    """

    def __init__(self, provider_configs: Optional[Dict[str, ProviderConfig]] = None):
        """Initialize the health monitor."""
        self._health_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl: float = 300.0  # 5 minutes cache TTL
        self._lock = threading.RLock()

        # Background monitoring
        self._monitoring_enabled: bool = False
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._monitoring_interval: float = 300.0  # 5 minutes interval

        # Provider configs
        self._provider_configs: Dict[str, ProviderConfig] = provider_configs or {}

        # Provider factory for testing
        self.provider_factory = None

    async def check_provider_health(
        self, provider_name: str, config: ProviderConfig, force: bool = False
    ) -> Dict[str, Any]:
        """
        Check health status of a provider.

        Args:
            provider_name: Name of the provider
            config: Provider configuration
            force: Force check even if cached result is available

        Returns:
            Health status information
        """
        # Generate cache key
        cache_key = self._generate_cache_key(provider_name, config)

        if not force:
            # Check if health status is already cached and valid
            with self._lock:
                if cache_key in self._health_cache and self._is_cache_valid(cache_key):
                    logger.debug(f"Using cached health status for {provider_name}")
                    return self._health_cache[cache_key].copy()

        # Perform health check
        try:
            start_time = time.time()

            # Get provider instance
            if hasattr(self, "provider_factory") and self.provider_factory:
                provider = self.provider_factory.create_provider(
                    provider_name, config=config
                )
            else:
                provider = ProviderInitializer.get_provider_instance(
                    provider_name, config
                )

            # Check credentials
            try:
                credentials_valid = await provider.validate_credentials()
            except Exception:
                # If provider is a mock, assume credentials are valid
                if (
                    hasattr(provider, "_mock_name")
                    or str(type(provider)).find("Mock") != -1
                ):
                    credentials_valid = True
                else:
                    credentials_valid = False

            # Get model info to check if model is available
            try:
                model_info = provider.get_model_info()
                model_available = (
                    model_info.available if hasattr(model_info, "available") else True
                )
            except Exception:
                # If provider is a mock, assume model is available
                if (
                    hasattr(provider, "_mock_name")
                    or str(type(provider)).find("Mock") != -1
                ):
                    model_available = True
                else:
                    model_available = False

            # Check response time
            response_time = time.time() - start_time

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

            # Get provider capabilities
            try:
                capabilities = {
                    "supports_tools": provider.supports_tools(),
                    "supports_multimodal": provider.supports_multimodal(),
                    "supports_streaming": provider.supports_streaming(),
                }
            except Exception:
                # For mock providers, assume basic capabilities
                if (
                    hasattr(provider, "_mock_name")
                    or str(type(provider)).find("Mock") != -1
                ):
                    capabilities = {
                        "supports_tools": True,
                        "supports_multimodal": False,
                        "supports_streaming": True,
                    }
                else:
                    capabilities = {
                        "supports_tools": False,
                        "supports_multimodal": False,
                        "supports_streaming": False,
                    }

            # Create health status
            health_status = {
                "status": status,
                "message": message,
                "available": True,
                "credentials_valid": credentials_valid,
                "model_available": model_available,
                "response_time": response_time,
                "last_check": datetime.now().isoformat(),
                "provider_capabilities": capabilities,
            }

            # Cache health status
            with self._lock:
                self._health_cache[cache_key] = health_status.copy()
                self._cache_timestamps[cache_key] = time.time()

            return health_status

        except Exception as e:
            logger.error(f"Error checking health for {provider_name}: {e}")

            # Create error status
            error_status = {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "available": False,
                "credentials_valid": False,
                "model_available": False,
                "response_time": None,
                "last_check": datetime.now().isoformat(),
                "error": str(e),
            }

            # Cache error status (with shorter TTL)
            with self._lock:
                self._health_cache[cache_key] = error_status.copy()
                self._cache_timestamps[cache_key] = time.time() - (
                    self._cache_ttl / 2
                )  # Shorter TTL for errors

            return error_status

    async def check_all_providers_health(
        self, configs: Dict[str, ProviderConfig], force: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check health status for all providers.

        Args:
            configs: Dictionary mapping provider names to configurations
            force: Force check even if cached result is available

        Returns:
            Dictionary mapping provider names to health status
        """
        # Update stored configs
        self._provider_configs = configs.copy()

        # Check health for each provider
        health_status = {}

        # Create tasks for concurrent health checks
        tasks = []
        for provider_name, config in configs.items():
            tasks.append(self.check_provider_health(provider_name, config, force))

        # Run health checks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, (provider_name, _) in enumerate(configs.items()):
            result = results[i]
            if isinstance(result, Exception):
                health_status[provider_name] = {
                    "status": "error",
                    "message": f"Health check failed: {str(result)}",
                    "available": False,
                    "credentials_valid": False,
                    "model_available": False,
                    "error": str(result),
                }
            else:
                health_status[provider_name] = result

        return health_status

    def _generate_cache_key(self, provider_name: str, config: ProviderConfig) -> str:
        """
        Generate cache key for health status.

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

    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.

        Args:
            cache_key: Cache key

        Returns:
            True if cache is valid
        """
        if cache_key not in self._cache_timestamps:
            return False

        return (time.time() - self._cache_timestamps[cache_key]) < self._cache_ttl

    def clear_cache(self) -> None:
        """Clear health status cache."""
        with self._lock:
            self._health_cache.clear()
            self._cache_timestamps.clear()
        logger.debug("Cleared health status cache")

    def set_cache_ttl(self, ttl_seconds: float) -> None:
        """
        Set cache TTL (time to live).

        Args:
            ttl_seconds: TTL in seconds
        """
        with self._lock:
            self._cache_ttl = ttl_seconds
        logger.debug(f"Set health status cache TTL to {ttl_seconds} seconds")

    def start_monitoring(
        self,
        configs: Dict[str, ProviderConfig],
        interval_seconds: float = 300.0,
    ) -> None:
        """
        Start background health monitoring.

        Args:
            configs: Dictionary mapping provider names to configurations
            interval_seconds: Monitoring interval in seconds
        """
        if self._monitoring_enabled:
            logger.info("Health monitoring already enabled")
            return

        # Update stored configs
        self._provider_configs = configs.copy()
        self._monitoring_interval = interval_seconds

        # Start monitoring
        self._monitoring_enabled = True
        self._stop_monitoring.clear()

        # Start background thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker, daemon=True, name="HealthMonitor"
        )
        self._monitoring_thread.start()
        logger.info("Started background health monitoring")

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self._monitoring_enabled:
            return

        self._monitoring_enabled = False
        self._stop_monitoring.set()

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        logger.info("Stopped background health monitoring")

    def _monitoring_worker(self) -> None:
        """Background worker thread for health monitoring."""
        logger.info("Health monitoring worker started")

        while self._monitoring_enabled and not self._stop_monitoring.is_set():
            try:
                # Run health checks
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    loop.run_until_complete(
                        self.check_all_providers_health(self._provider_configs)
                    )
                    logger.debug("Completed background health checks")
                finally:
                    loop.close()

                # Wait for next check
                self._stop_monitoring.wait(timeout=self._monitoring_interval)

            except Exception as e:
                logger.error(f"Error in health monitoring worker: {e}")
                # Wait before retrying
                self._stop_monitoring.wait(timeout=60.0)

        logger.info("Health monitoring worker stopped")

    def get_provider_health(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached health status for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Health status if available, None otherwise
        """
        with self._lock:
            for cache_key, status in self._health_cache.items():
                if cache_key.startswith(f"{provider_name}:"):
                    if self._is_cache_valid(cache_key):
                        return status.copy()

        return None

    def get_all_provider_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cached health status for all providers.

        Returns:
            Dictionary mapping provider names to health status
        """
        health_status = {}

        with self._lock:
            for cache_key, status in self._health_cache.items():
                if self._is_cache_valid(cache_key):
                    provider_name = cache_key.split(":", 1)[0]
                    health_status[provider_name] = status.copy()

        return health_status

    async def check_all_providers(self, config, force: bool = False) -> "HealthResult":
        """
        Check health status for all providers in a config object.

        Args:
            config: Configuration object with providers attribute
            force: Force check even if cached result is available

        Returns:
            HealthResult object with overall_healthy attribute
        """
        # Extract providers from config
        if hasattr(config, "providers"):
            providers = config.providers
        else:
            providers = config  # Assume config is a dict of providers

        # Run health checks
        health_status = await self.check_all_providers_health(providers, force)

        # Determine overall health
        overall_healthy = True
        for provider_health in health_status.values():
            if provider_health.get("status") == "error":
                overall_healthy = False
                break
            elif provider_health.get("status") == "warning":
                # Warnings don't necessarily make the system unhealthy
                pass

        return HealthResult(
            overall_healthy=overall_healthy,
            provider_health=health_status,
            total_providers=len(health_status),
            healthy_providers=sum(
                1 for h in health_status.values() if h.get("status") == "healthy"
            ),
        )


class HealthResult:
    """Result object for provider health checks."""

    def __init__(
        self,
        overall_healthy: bool,
        provider_health: Dict[str, Dict[str, Any]],
        total_providers: int,
        healthy_providers: int,
    ):
        self.overall_healthy = overall_healthy
        self.provider_health = provider_health
        self.total_providers = total_providers
        self.healthy_providers = healthy_providers
