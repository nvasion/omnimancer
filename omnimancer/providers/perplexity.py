"""
Perplexity AI provider implementation for Omnimancer.

This module provides the Perplexity AI provider implementation using Perplexity's API
with support for search-enabled conversations and recency filtering.
"""

from datetime import datetime
from typing import Dict, List

import httpx

from ..core.models import ChatContext, ChatResponse, EnhancedModelInfo
from ..utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from .base import BaseProvider


class PerplexityProvider(BaseProvider):
    """
    Perplexity AI provider implementation using Perplexity's API.

    Supports search-enabled conversations with real-time web information
    and recency filtering for up-to-date responses.
    """

    BASE_URL = "https://api.perplexity.ai"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize Perplexity provider.

        Args:
            api_key: Perplexity API key
            model: Perplexity model to use (e.g., 'llama-3.1-sonar-small-128k-online', 'llama-3.1-sonar-large-128k-online')
            **kwargs: Additional configuration including search settings
        """
        super().__init__(
            api_key, model or "llama-3.1-sonar-small-128k-online", **kwargs
        )
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.2)
        self.search_enabled = kwargs.get("search_enabled", True)
        self.search_recency_filter = kwargs.get(
            "search_recency_filter", "month"
        )  # hour, day, week, month, year
        self.return_citations = kwargs.get("return_citations", False)
        self.return_images = kwargs.get("return_images", False)
        self.return_related_questions = kwargs.get("return_related_questions", False)

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to Perplexity API with search capabilities.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Perplexity's reply including search results
        """
        try:
            # Prepare messages for Perplexity API
            messages = self._prepare_messages(message, context)

            # Build request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "return_citations": self.return_citations,
                "return_images": self.return_images,
                "return_related_questions": self.return_related_questions,
            }

            # Add search-specific parameters for online models
            if self._is_online_model():
                payload["search_recency_filter"] = self.search_recency_filter

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=60.0,  # Longer timeout for search operations
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Perplexity API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except (
            AuthenticationError,
            RateLimitError,
            ModelNotFoundError,
            ProviderError,
        ):
            # Re-raise provider-specific errors without wrapping
            raise
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    async def validate_credentials(self) -> bool:
        """
        Validate Perplexity API credentials by making a test request.

        Returns:
            True if credentials are valid
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 10,
                    },
                    timeout=10.0,
                )

            return response.status_code == 200

        except Exception:
            return False

    def _prepare_messages(
        self, message: str, context: ChatContext
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for Perplexity API format.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            List of messages formatted for Perplexity API
        """
        messages = []

        # Add context messages
        for msg in context.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Perplexity API response.

        Args:
            response: HTTP response from Perplexity API

        Returns:
            ChatResponse object

        Raises:
            Various provider errors based on response status
        """
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])

            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                usage = data.get("usage", {})

                # Extract citations and additional metadata
                citations = data.get("citations", [])
                images = data.get("images", [])
                related_questions = data.get("related_questions", [])

                # Format content with citations if available
                formatted_content = self._format_content_with_metadata(
                    content, citations, images, related_questions
                )

                return ChatResponse(
                    content=formatted_content,
                    model_used=self.model,
                    tokens_used=usage.get("total_tokens", 0),
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty response from Perplexity API")

        elif response.status_code == 401:
            raise AuthenticationError("Invalid Perplexity API key")
        elif response.status_code == 429:
            raise RateLimitError("Perplexity API rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError(f"Perplexity model '{self.model}' not found")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"Perplexity API error: {error_msg}")

    def _format_content_with_metadata(
        self,
        content: str,
        citations: List[Dict],
        images: List[Dict],
        related_questions: List[str],
    ) -> str:
        """
        Format response content with citations and metadata.

        Args:
            content: Main response content
            citations: List of citation objects
            images: List of image objects
            related_questions: List of related questions

        Returns:
            Formatted content string
        """
        formatted = content

        # Add citations if available
        if citations and self.return_citations:
            formatted += "\n\n**Sources:**\n"
            for i, citation in enumerate(citations, 1):
                title = citation.get("title", "Unknown")
                url = citation.get("url", "")
                formatted += f"{i}. [{title}]({url})\n"

        # Add related questions if available
        if related_questions and self.return_related_questions:
            formatted += "\n\n**Related Questions:**\n"
            for question in related_questions:
                formatted += f"• {question}\n"

        return formatted

    def _is_online_model(self) -> bool:
        """
        Check if the current model is an online/search-enabled model.

        Returns:
            True if model supports online search
        """
        online_models = [
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
        ]
        return self.model in online_models

    def get_model_info(self) -> EnhancedModelInfo:
        """
        Get information about the current Perplexity model.
        """
        model_configs = {
            "llama-3.1-sonar-small-128k-online": {
                "description": "Llama 3.1 Sonar Small with real-time web search",
                "max_tokens": 127072,
                "cost_per_million_input": 0.2,
                "cost_per_million_output": 0.2,
                "swe_score": 45.2,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "llama-3.1-sonar-large-128k-online": {
                "description": "Llama 3.1 Sonar Large with real-time web search",
                "max_tokens": 127072,
                "cost_per_million_input": 1.0,
                "cost_per_million_output": 1.0,
                "swe_score": 52.4,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "llama-3.1-sonar-huge-128k-online": {
                "description": "Llama 3.1 Sonar Huge with real-time web search",
                "max_tokens": 127072,
                "cost_per_million_input": 5.0,
                "cost_per_million_output": 5.0,
                "swe_score": 58.1,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "llama-3.1-sonar-small-128k-chat": {
                "description": "Llama 3.1 Sonar Small for chat without search",
                "max_tokens": 127072,
                "cost_per_million_input": 0.2,
                "cost_per_million_output": 0.2,
                "swe_score": 45.2,
                "supports_tools": False,
                "supports_multimodal": False,
            },
            "llama-3.1-sonar-large-128k-chat": {
                "description": "Llama 3.1 Sonar Large for chat without search",
                "max_tokens": 127072,
                "cost_per_million_input": 1.0,
                "cost_per_million_output": 1.0,
                "swe_score": 52.4,
                "supports_tools": False,
                "supports_multimodal": False,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"Perplexity model {self.model}",
                "max_tokens": 127072,
                "cost_per_million_input": 1.0,
                "cost_per_million_output": 1.0,
                "swe_score": 50.0,
                "supports_tools": False,
                "supports_multimodal": False,
            },
        )

        enhanced_info = EnhancedModelInfo(
            name=self.model,
            provider="perplexity",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_million_input=config["cost_per_million_input"],
            cost_per_million_output=config["cost_per_million_output"],
            swe_score=config["swe_score"],
            available=True,
            supports_tools=config["supports_tools"],
            supports_multimodal=config["supports_multimodal"],
            latest_version=self.model == "llama-3.1-sonar-huge-128k-online",
            context_window=config["max_tokens"],
            is_free=False,
            release_date=datetime(2024, 7, 1),  # Approximate release date
        )

        # Update SWE rating based on score
        enhanced_info.update_swe_rating()

        return enhanced_info

    def get_available_models(self) -> List[EnhancedModelInfo]:
        """
        Get list of available Perplexity models.
        """
        models = [
            EnhancedModelInfo(
                name="llama-3.1-sonar-small-128k-online",
                provider="perplexity",
                description="Llama 3.1 Sonar Small with real-time web search",
                max_tokens=127072,
                cost_per_million_input=0.2,
                cost_per_million_output=0.2,
                swe_score=45.2,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=127072,
                is_free=False,
                release_date=datetime(2024, 7, 1),
            ),
            EnhancedModelInfo(
                name="llama-3.1-sonar-large-128k-online",
                provider="perplexity",
                description="Llama 3.1 Sonar Large with real-time web search",
                max_tokens=127072,
                cost_per_million_input=1.0,
                cost_per_million_output=1.0,
                swe_score=52.4,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=127072,
                is_free=False,
                release_date=datetime(2024, 7, 1),
            ),
            EnhancedModelInfo(
                name="llama-3.1-sonar-huge-128k-online",
                provider="perplexity",
                description="Llama 3.1 Sonar Huge with real-time web search",
                max_tokens=127072,
                cost_per_million_input=5.0,
                cost_per_million_output=5.0,
                swe_score=58.1,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                latest_version=True,
                context_window=127072,
                is_free=False,
                release_date=datetime(2024, 7, 1),
            ),
            EnhancedModelInfo(
                name="llama-3.1-sonar-small-128k-chat",
                provider="perplexity",
                description="Llama 3.1 Sonar Small for chat without search",
                max_tokens=127072,
                cost_per_million_input=0.2,
                cost_per_million_output=0.2,
                swe_score=45.2,
                available=True,
                supports_tools=False,
                supports_multimodal=False,
                context_window=127072,
                is_free=False,
                release_date=datetime(2024, 7, 1),
            ),
            EnhancedModelInfo(
                name="llama-3.1-sonar-large-128k-chat",
                provider="perplexity",
                description="Llama 3.1 Sonar Large for chat without search",
                max_tokens=127072,
                cost_per_million_input=1.0,
                cost_per_million_output=1.0,
                swe_score=52.4,
                available=True,
                supports_tools=False,
                supports_multimodal=False,
                context_window=127072,
                is_free=False,
                release_date=datetime(2024, 7, 1),
            ),
        ]

        # Update SWE ratings for all models
        for model in models:
            model.update_swe_rating()

        return models

    def supports_tools(self) -> bool:
        """
        Check if Perplexity provider supports tool calling.

        Returns:
            True for online models that support search tools
        """
        return self._is_online_model()

    def supports_multimodal(self) -> bool:
        """
        Check if Perplexity provider supports multimodal inputs.

        Returns:
            False - Current Perplexity models don't support multimodal inputs
        """
        return False

    def supports_streaming(self) -> bool:
        """
        Check if Perplexity provider supports streaming responses.

        Returns:
            True - Perplexity supports streaming
        """
        return True
