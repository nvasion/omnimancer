"""
Cohere provider implementation for Omnimancer.

This module provides the Cohere AI provider implementation using Cohere's API.
"""

from datetime import datetime
from typing import Dict, List

import httpx

from ..core.models import ChatContext, ChatResponse, ModelInfo
from ..utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderConfigurationError,
    ProviderError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from .base import BaseProvider


class CohereProvider(BaseProvider):
    """
    Cohere AI provider implementation using Cohere's API.
    """

    BASE_URL = "https://api.cohere.ai/v1"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize Cohere provider.

        Args:
            api_key: Cohere API key
            model: Cohere model to use (e.g., 'command-r', 'command-r-plus', 'command')
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model or "command-r", **kwargs)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to Cohere API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Cohere's reply
        """
        try:
            # Prepare chat history for Cohere API
            chat_history = self._prepare_chat_history(context)

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json={
                        "model": self.model,
                        "message": message,
                        "chat_history": chat_history,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    timeout=30.0,
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Cohere API timed out")
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
        Validate Cohere API credentials by making a test request.

        Returns:
            True if credentials are valid
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json={
                        "model": self.model,
                        "message": "Hi",
                        "max_tokens": 10,
                    },
                    timeout=10.0,
                )

            return response.status_code == 200

        except Exception:
            return False

    def _prepare_chat_history(self, context: ChatContext) -> List[Dict[str, str]]:
        """
        Prepare chat history for Cohere API format.

        Args:
            context: Conversation context

        Returns:
            List of messages formatted for Cohere API
        """
        chat_history = []

        # Convert context messages to Cohere format
        # Cohere uses "USER" and "CHATBOT" roles, and excludes system messages
        for msg in context.messages:
            if msg.role.value == "user":
                chat_history.append({"role": "USER", "message": msg.content})
            elif msg.role.value == "assistant":
                chat_history.append({"role": "CHATBOT", "message": msg.content})
            # Skip system messages as Cohere doesn't support them in chat history

        return chat_history

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Cohere API response with comprehensive error handling.

        Args:
            response: HTTP response from Cohere API

        Returns:
            ChatResponse object

        Raises:
            Various provider errors based on response status
        """
        if response.status_code == 200:
            try:
                data = response.json()
                text = data.get("text", "")

                if text:
                    # Extract token usage if available
                    meta = data.get("meta", {})
                    billed_units = meta.get("billed_units", {})
                    tokens_used = billed_units.get("output_tokens", 0)

                    return ChatResponse(
                        content=text,
                        model_used=self.model,
                        tokens_used=tokens_used,
                        timestamp=datetime.now(),
                    )
                else:
                    raise ProviderError(
                        "Empty response from Cohere API", provider="cohere"
                    )
            except (ValueError, KeyError) as e:
                raise ProviderError(
                    f"Invalid JSON response from Cohere API: {e}",
                    provider="cohere",
                )

        elif response.status_code == 400:
            try:
                error_data = response.json()
                error_msg = error_data.get("message", "Bad request")

                if "model" in error_msg.lower() or "not found" in error_msg.lower():
                    available_models = [m.name for m in self.get_available_models()]
                    raise ModelNotFoundError(
                        f"Cohere model '{self.model}' not found or not accessible",
                        provider="cohere",
                        model_name=self.model,
                        available_models=available_models,
                    )
                elif "token" in error_msg.lower() and "limit" in error_msg.lower():
                    raise ProviderError(
                        f"Token limit exceeded: {error_msg}. Try reducing message length.",
                        provider="cohere",
                    )
                elif (
                    "invalid" in error_msg.lower() and "parameter" in error_msg.lower()
                ):
                    raise ProviderConfigurationError(
                        f"Invalid parameter in request: {error_msg}",
                        provider="cohere",
                        suggested_fix="Check model parameters and configuration",
                    )
                else:
                    raise ProviderError(
                        f"Cohere API error: {error_msg}", provider="cohere"
                    )
            except (ValueError, KeyError):
                raise ProviderError("Bad request to Cohere API", provider="cohere")

        elif response.status_code == 401:
            raise AuthenticationError(
                "Invalid Cohere API key. Check your API key configuration.",
                provider="cohere",
            )

        elif response.status_code == 403:
            try:
                error_data = response.json()
                error_msg = error_data.get("message", "")
                if "billing" in error_msg.lower() or "payment" in error_msg.lower():
                    raise QuotaExceededError(
                        "Cohere API access restricted due to billing issues",
                        provider="cohere",
                    )
                else:
                    raise AuthenticationError(
                        "Cohere API access forbidden. Check API key permissions.",
                        provider="cohere",
                    )
            except (ValueError, KeyError):
                raise AuthenticationError(
                    "Cohere API access forbidden. Check API key permissions.",
                    provider="cohere",
                )

        elif response.status_code == 429:
            # Extract rate limit information if available
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    retry_seconds = int(retry_after)
                    raise RateLimitError(
                        "Cohere API rate limit exceeded",
                        provider="cohere",
                        retry_after=retry_seconds,
                    )
                except ValueError:
                    pass

            # Check if it's a quota issue
            try:
                error_data = response.json()
                error_msg = error_data.get("message", "")
                if "quota" in error_msg.lower() or "usage" in error_msg.lower():
                    raise QuotaExceededError(
                        "Cohere API usage quota exceeded", provider="cohere"
                    )
            except (ValueError, KeyError):
                pass

            raise RateLimitError(
                "Cohere API rate limit exceeded. Wait before retrying.",
                provider="cohere",
            )

        elif response.status_code == 500:
            raise ProviderUnavailableError(
                "Cohere API server error. Try again later.",
                provider="cohere",
                estimated_recovery="a few minutes",
            )

        elif response.status_code == 502 or response.status_code == 503:
            raise ProviderUnavailableError(
                "Cohere API temporarily unavailable. Service may be under maintenance.",
                provider="cohere",
                estimated_recovery="10-30 minutes",
            )

        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"Cohere API error: {error_msg}", provider="cohere")

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Cohere model.
        """
        model_configs = {
            "command-r": {
                "description": "Command R - Cohere's flagship conversational model",
                "max_tokens": 128000,
                "cost_per_token": 0.0000005,
            },
            "command-r-plus": {
                "description": "Command R+ - Enhanced version with improved capabilities",
                "max_tokens": 128000,
                "cost_per_token": 0.000003,
            },
            "command-light": {
                "description": "Command Light - Fast and efficient model for simple tasks",
                "max_tokens": 4096,
                "cost_per_token": 0.0000003,
            },
            "command": {
                "description": "Command - Previous generation conversational model",
                "max_tokens": 4096,
                "cost_per_token": 0.000001,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"Cohere model {self.model}",
                "max_tokens": 4096,
                "cost_per_token": 0.000001,
            },
        )

        return ModelInfo(
            name=self.model,
            provider="cohere",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_token=config["cost_per_token"],
            available=True,
            supports_tools=False,  # Cohere doesn't support tool calling in the same way
            supports_multimodal=False,  # Cohere is primarily text-based
            latest_version=self.model == "command-r-plus",
        )

    def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available Cohere models.
        """
        return [
            ModelInfo(
                name="command-r",
                provider="cohere",
                description="Command R - Cohere's flagship conversational model",
                max_tokens=128000,
                cost_per_token=0.0000005,
                available=True,
                supports_tools=False,
                supports_multimodal=False,
            ),
            ModelInfo(
                name="command-r-plus",
                provider="cohere",
                description="Command R+ - Enhanced version with improved capabilities",
                max_tokens=128000,
                cost_per_token=0.000003,
                available=True,
                supports_tools=False,
                supports_multimodal=False,
                latest_version=True,
            ),
            ModelInfo(
                name="command-light",
                provider="cohere",
                description="Command Light - Fast and efficient model for simple tasks",
                max_tokens=4096,
                cost_per_token=0.0000003,
                available=True,
                supports_tools=False,
                supports_multimodal=False,
            ),
            ModelInfo(
                name="command",
                provider="cohere",
                description="Command - Previous generation conversational model",
                max_tokens=4096,
                cost_per_token=0.000001,
                available=True,
                supports_tools=False,
                supports_multimodal=False,
            ),
        ]

    def supports_tools(self) -> bool:
        """
        Check if Cohere provider supports tool calling.

        Returns:
            False - Cohere doesn't support tool calling in the same way as OpenAI/Claude
        """
        return False

    def supports_multimodal(self) -> bool:
        """
        Check if Cohere provider supports multimodal inputs.

        Returns:
            False - Cohere is primarily text-based
        """
        return False
