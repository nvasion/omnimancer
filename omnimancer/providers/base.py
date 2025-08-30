"""
Base provider interface for Omnimancer.

This module defines the abstract base class that all AI providers must implement
to ensure consistent behavior across different AI services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from ..core.models import ChatContext, ChatResponse, ModelInfo, ToolDefinition


class BaseProvider(ABC):
    """
    Abstract base class for AI providers.

    All AI provider implementations must inherit from this class and implement
    the required abstract methods to ensure consistent behavior.
    """

    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize the provider.

        Args:
            api_key: API key for authentication
            model: Model name to use
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @abstractmethod
    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to the AI provider and get a response.

        Args:
            message: The user's message
            context: Current conversation context

        Returns:
            ChatResponse with the AI's reply

        Raises:
            ProviderError: If the API call fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limits are exceeded
        """
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Validate that the API credentials are working.

        Returns:
            True if credentials are valid, False otherwise
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current model.

        Returns:
            ModelInfo object with model details
        """
        pass

    def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available models for this provider.
        Uses catalog models if available, otherwise returns static models.

        Returns:
            List of ModelInfo objects for available models
        """
        # Check if we have catalog models (updated from API)
        if hasattr(self, "_catalog_models") and self._catalog_models:
            # Convert EnhancedModelInfo to ModelInfo if needed
            result = []
            for model in self._catalog_models:
                if hasattr(model, "to_model_info"):
                    result.append(model.to_model_info())
                else:
                    result.append(model)
            return result

        # Otherwise, use the provider's static models
        return self._get_static_models()

    def _get_static_models(self) -> List[ModelInfo]:
        """
        Get static list of models (when catalog is not available).
        Subclasses should override this to return their default models.

        Returns:
            List of ModelInfo objects for available models
        """
        # Default implementation returns empty list
        # Subclasses should override this method
        return []

    async def fetch_live_models(self) -> List[ModelInfo]:
        """
        Fetch live model list from provider API.

        This method should be implemented by providers that support
        real-time model discovery via their APIs.

        Returns:
            List of ModelInfo objects from API, or empty list if not supported
        """
        # Default implementation returns static models
        return self.get_available_models()

    async def fetch_enhanced_models(self) -> List["EnhancedModelInfo"]:
        """
        Fetch enhanced model information from provider API.

        This method fetches detailed model information including pricing,
        capabilities, and performance metrics when available.

        Returns:
            List of EnhancedModelInfo objects with detailed model information
        """
        from ..core.models import EnhancedModelInfo

        # Try to get live models first
        models = await self.fetch_live_models()

        # Convert to enhanced model info
        enhanced_models = []
        for model in models:
            if isinstance(model, EnhancedModelInfo):
                enhanced_models.append(model)
            else:
                # Convert basic ModelInfo to EnhancedModelInfo
                enhanced = EnhancedModelInfo.from_model_info(model)
                enhanced_models.append(enhanced)

        return enhanced_models

    @abstractmethod
    def supports_tools(self) -> bool:
        """
        Check if this provider supports tool calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        pass

    @abstractmethod
    def supports_multimodal(self) -> bool:
        """
        Check if this provider supports multimodal inputs (images, audio, etc.).

        Returns:
            True if multimodal inputs are supported, False otherwise
        """
        pass

    async def send_message_with_tools(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> ChatResponse:
        """
        Send a message with available tools for the AI to use.

        This method should be overridden by providers that support tool calling.
        Default implementation falls back to regular message sending.

        Args:
            message: The user's message
            context: Current conversation context
            available_tools: List of tools available for the AI to use

        Returns:
            ChatResponse with the AI's reply and any tool calls

        Raises:
            ProviderError: If the API call fails
            AuthenticationError: If authentication fails
            RateLimitError: If rate limits are exceeded
        """
        # Default implementation for providers that don't support tools
        if not self.supports_tools():
            return await self.send_message(message, context)

        # This should be overridden by tool-supporting providers
        raise NotImplementedError(
            f"{self.__class__.__name__} supports tools but hasn't implemented send_message_with_tools"
        )

    def get_provider_name(self) -> str:
        """
        Get the name of this provider.

        Returns:
            Provider name (e.g., "claude", "openai")
        """
        return self.__class__.__name__.lower().replace("provider", "")

    def supports_streaming(self) -> bool:
        """
        Check if this provider supports streaming responses.

        Returns:
            True if streaming is supported, False otherwise
        """
        return False

    def get_max_tokens(self) -> int:
        """
        Get the maximum number of tokens supported by the current model.

        Returns:
            Maximum token count
        """
        model_info = self.get_model_info()
        return model_info.max_tokens

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost of a request based on token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        model_info = self.get_model_info()
        return (input_tokens + output_tokens) * model_info.cost_per_token

    def prepare_context(self, context: ChatContext) -> List[Dict[str, str]]:
        """
        Prepare conversation context for API call.

        Args:
            context: Current conversation context

        Returns:
            List of message dictionaries formatted for the API
        """
        return context.get_context_for_api()

    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.get_provider_name()}:{self.model}"

    def __repr__(self) -> str:
        """Detailed string representation of the provider."""
        return f"{self.__class__.__name__}(model='{self.model}')"
