"""
OpenAI provider implementation for Omnimancer.

This module provides the OpenAI API provider implementation using OpenAI's API.
"""

from datetime import datetime
from typing import Dict, List

import httpx

from ..core.models import ChatContext, ChatResponse, ModelInfo
from ..utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    OpenAI API provider implementation using OpenAI's API.
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (e.g., 'gpt-4', 'gpt-4o', 'gpt-3.5-turbo')
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model or "gpt-4", **kwargs)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to OpenAI API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with OpenAI's reply
        """
        try:
            # Prepare messages for OpenAI API
            messages = self._prepare_messages(message, context)

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    timeout=30.0,
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to OpenAI API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    async def validate_credentials(self) -> bool:
        """
        Validate OpenAI API credentials by making a test request.

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
        Prepare messages for OpenAI API format.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            List of messages formatted for OpenAI API
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
        Handle OpenAI API response.

        Args:
            response: HTTP response from OpenAI API

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

                return ChatResponse(
                    content=content,
                    model_used=self.model,
                    tokens_used=usage.get("total_tokens", 0),
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty response from OpenAI API")

        elif response.status_code == 401:
            raise AuthenticationError("Invalid OpenAI API key")
        elif response.status_code == 429:
            raise RateLimitError("OpenAI API rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError(f"OpenAI model '{self.model}' not found")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"OpenAI API error: {error_msg}")

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current OpenAI model.
        """
        model_configs = {
            "gpt-4": {
                "description": "GPT-4 - Most capable model",
                "max_tokens": 8192,
                "cost_per_token": 0.00003,
            },
            "gpt-4-turbo": {
                "description": "GPT-4 Turbo - Enhanced performance",
                "max_tokens": 128000,
                "cost_per_token": 0.00001,
            },
            "gpt-3.5-turbo": {
                "description": "GPT-3.5 Turbo - Fast and efficient",
                "max_tokens": 4096,
                "cost_per_token": 0.000002,
            },
            "gpt-3.5-turbo-16k": {
                "description": "GPT-3.5 Turbo with 16K context",
                "max_tokens": 16384,
                "cost_per_token": 0.000004,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"OpenAI model {self.model}",
                "max_tokens": 4096,
                "cost_per_token": 0.00002,
            },
        )

        return ModelInfo(
            name=self.model,
            provider="openai",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_token=config["cost_per_token"],
            available=True,
            supports_tools=self.supports_tools(),
            supports_multimodal=self.supports_multimodal(),
            latest_version=self.model == "gpt-4-turbo",
        )

    def _get_static_models(self) -> List[ModelInfo]:
        """
        Get static list of available OpenAI models.
        """
        return [
            ModelInfo(
                name="gpt-4",
                provider="openai",
                description="GPT-4 - Most capable model",
                max_tokens=8192,
                cost_per_token=0.00003,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
            ),
            ModelInfo(
                name="gpt-4-turbo",
                provider="openai",
                description="GPT-4 Turbo - Enhanced performance",
                max_tokens=128000,
                cost_per_token=0.00001,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
            ),
            ModelInfo(
                name="gpt-3.5-turbo",
                provider="openai",
                description="GPT-3.5 Turbo - Fast and efficient",
                max_tokens=4096,
                cost_per_token=0.000002,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
            ),
            ModelInfo(
                name="gpt-3.5-turbo-16k",
                provider="openai",
                description="GPT-3.5 Turbo with 16K context",
                max_tokens=16384,
                cost_per_token=0.000004,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
            ),
        ]

    def supports_tools(self) -> bool:
        """
        Check if OpenAI provider supports tool calling.

        Returns:
            True - OpenAI supports function calling/tools
        """
        return True

    def supports_multimodal(self) -> bool:
        """
        Check if OpenAI provider supports multimodal inputs.

        Returns:
            True for GPT-4 models that support vision, False for others
        """
        # GPT-4 models with vision support
        vision_models = ["gpt-4-vision-preview", "gpt-4-turbo", "gpt-4o"]
        return any(model in self.model for model in vision_models)

    async def fetch_live_models(self) -> List[ModelInfo]:
        """
        Fetch live model list from OpenAI API.

        Returns:
            List of ModelInfo objects from OpenAI API
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/models", headers=headers, timeout=30.0
                )
                response.raise_for_status()

                data = response.json()
                models = []

                # Filter and convert to ModelInfo objects
                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "")

                    # Filter for chat models (exclude fine-tuned and other model types)
                    if any(prefix in model_id for prefix in ["gpt-3.5", "gpt-4"]):
                        # Determine model capabilities
                        supports_tools = (
                            "gpt-3.5-turbo" in model_id or "gpt-4" in model_id
                        )
                        supports_multimodal = any(
                            vision in model_id
                            for vision in ["gpt-4-turbo", "gpt-4o", "vision"]
                        )

                        # Estimate max tokens based on model
                        max_tokens = 4096  # default
                        if "gpt-4-turbo" in model_id or "gpt-4o" in model_id:
                            max_tokens = 128000
                        elif "gpt-4" in model_id:
                            max_tokens = 8192
                        elif "gpt-3.5-turbo" in model_id:
                            max_tokens = 16384

                        models.append(
                            ModelInfo(
                                name=model_id,
                                provider="openai",
                                description=f"OpenAI {model_id}",
                                max_tokens=max_tokens,
                                cost_per_token=0.00001,  # Approximate
                                available=True,
                                supports_tools=supports_tools,
                                supports_multimodal=supports_multimodal,
                            )
                        )

                return models

        except Exception:
            # Fall back to static model list if API call fails
            return self.get_available_models()
