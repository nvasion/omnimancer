"""
Mistral AI provider implementation for Omnimancer.

This module provides the Mistral AI provider implementation using Mistral's API
with support for safety settings and advanced parameters.
"""

from datetime import datetime
from typing import Dict, List

import httpx

from ..core.models import (
    ChatContext,
    ChatResponse,
    EnhancedModelInfo,
    ToolCall,
    ToolDefinition,
)
from ..utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from .base import BaseProvider


class MistralProvider(BaseProvider):
    """
    Mistral AI provider implementation using Mistral's API.

    Supports Mistral models with safety settings, function calling,
    and advanced conversation parameters.
    """

    BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize Mistral provider.

        Args:
            api_key: Mistral API key
            model: Mistral model to use (e.g., 'mistral-large-latest', 'mistral-small-latest', 'open-mistral-7b')
            **kwargs: Additional configuration including Mistral-specific settings
        """
        super().__init__(api_key, model or "mistral-large-latest", **kwargs)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.safe_prompt = kwargs.get("safe_prompt", False)
        self.random_seed = kwargs.get("random_seed", None)
        self.response_format = kwargs.get("response_format", None)  # For JSON mode

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to Mistral API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Mistral's reply
        """
        try:
            # Prepare messages for Mistral API
            messages = self._prepare_messages(message, context)

            # Build request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "safe_prompt": self.safe_prompt,
                "stream": False,
            }

            # Add optional parameters
            if self.random_seed is not None:
                payload["random_seed"] = self.random_seed

            if self.response_format:
                payload["response_format"] = self.response_format

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=30.0,
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Mistral API timed out")
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

    async def send_message_with_tools(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> ChatResponse:
        """
        Send a message with available tools for Mistral to use.

        Args:
            message: User message
            context: Conversation context
            available_tools: List of tools available for Mistral to use

        Returns:
            ChatResponse with Mistral's reply and any tool calls
        """
        try:
            # Prepare messages for Mistral API
            messages = self._prepare_messages(message, context)

            # Convert tools to Mistral format
            tools = self._convert_tools_to_mistral_format(available_tools)

            # Build request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "safe_prompt": self.safe_prompt,
                "tools": tools,
                "tool_choice": "auto",
                "stream": False,
            }

            # Add optional parameters
            if self.random_seed is not None:
                payload["random_seed"] = self.random_seed

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=30.0,
                )

            # Handle response with tool calls
            return self._handle_response_with_tools(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Mistral API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    async def validate_credentials(self) -> bool:
        """
        Validate Mistral API credentials by making a test request.

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
        Prepare messages for Mistral API format.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            List of messages formatted for Mistral API
        """
        messages = []

        # Add context messages
        for msg in context.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    def _convert_tools_to_mistral_format(
        self, tools: List[ToolDefinition]
    ) -> List[Dict]:
        """
        Convert tool definitions to Mistral API format.

        Args:
            tools: List of tool definitions

        Returns:
            List of tools formatted for Mistral API
        """
        mistral_tools = []

        for tool in tools:
            mistral_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            mistral_tools.append(mistral_tool)

        return mistral_tools

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Mistral API response.

        Args:
            response: HTTP response from Mistral API

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
                raise ProviderError("Empty response from Mistral API")

        elif response.status_code == 401:
            raise AuthenticationError("Invalid Mistral API key")
        elif response.status_code == 429:
            raise RateLimitError("Mistral API rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError(f"Mistral model '{self.model}' not found")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"Mistral API error: {error_msg}")

    def _handle_response_with_tools(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Mistral API response with tool calls.

        Args:
            response: HTTP response from Mistral API

        Returns:
            ChatResponse object with tool calls
        """
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])

            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                usage = data.get("usage", {})

                # Extract tool calls if present
                tool_calls = []
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if tool_call.get("type") == "function":
                            function = tool_call.get("function", {})
                            tool_calls.append(
                                ToolCall(
                                    name=function.get("name", ""),
                                    arguments=function.get("arguments", {}),
                                )
                            )

                return ChatResponse(
                    content=content,
                    model_used=self.model,
                    tokens_used=usage.get("total_tokens", 0),
                    tool_calls=tool_calls if tool_calls else None,
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty response from Mistral API")
        else:
            return self._handle_response(response)

    def get_model_info(self) -> EnhancedModelInfo:
        """
        Get information about the current Mistral model.
        """
        model_configs = {
            "mistral-large-latest": {
                "description": "Mistral Large - Most capable model for complex tasks",
                "max_tokens": 128000,
                "cost_per_million_input": 2.0,
                "cost_per_million_output": 6.0,
                "swe_score": 64.6,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "mistral-medium-latest": {
                "description": "Mistral Medium - Balanced performance and efficiency",
                "max_tokens": 32000,
                "cost_per_million_input": 2.7,
                "cost_per_million_output": 8.1,
                "swe_score": 55.8,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "mistral-small-latest": {
                "description": "Mistral Small - Fast and cost-effective",
                "max_tokens": 32000,
                "cost_per_million_input": 0.2,
                "cost_per_million_output": 0.6,
                "swe_score": 48.2,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "mistral-tiny": {
                "description": "Mistral Tiny - Ultra-fast for simple tasks",
                "max_tokens": 32000,
                "cost_per_million_input": 0.14,
                "cost_per_million_output": 0.42,
                "swe_score": 35.1,
                "supports_tools": False,
                "supports_multimodal": False,
            },
            "codestral-latest": {
                "description": "Codestral - Specialized for code generation and analysis",
                "max_tokens": 32000,
                "cost_per_million_input": 0.2,
                "cost_per_million_output": 0.6,
                "swe_score": 78.2,
                "supports_tools": True,
                "supports_multimodal": False,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"Mistral model {self.model}",
                "max_tokens": 32000,
                "cost_per_million_input": 2.0,
                "cost_per_million_output": 6.0,
                "swe_score": 50.0,
                "supports_tools": True,
                "supports_multimodal": False,
            },
        )

        enhanced_info = EnhancedModelInfo(
            name=self.model,
            provider="mistral",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_million_input=config["cost_per_million_input"],
            cost_per_million_output=config["cost_per_million_output"],
            swe_score=config["swe_score"],
            available=True,
            supports_tools=config["supports_tools"],
            supports_multimodal=config["supports_multimodal"],
            latest_version=self.model == "mistral-large-latest",
            context_window=config["max_tokens"],
            is_free=False,
            release_date=datetime(2024, 9, 1),  # Approximate release date
        )

        # Update SWE rating based on score
        enhanced_info.update_swe_rating()

        return enhanced_info

    def get_available_models(self) -> List[EnhancedModelInfo]:
        """
        Get list of available Mistral models.
        """
        models = [
            EnhancedModelInfo(
                name="mistral-large-latest",
                provider="mistral",
                description="Mistral Large - Most capable model for complex tasks",
                max_tokens=128000,
                cost_per_million_input=2.0,
                cost_per_million_output=6.0,
                swe_score=64.6,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                latest_version=True,
                context_window=128000,
                is_free=False,
                release_date=datetime(2024, 9, 1),
            ),
            EnhancedModelInfo(
                name="mistral-medium-latest",
                provider="mistral",
                description="Mistral Medium - Balanced performance and efficiency",
                max_tokens=32000,
                cost_per_million_input=2.7,
                cost_per_million_output=8.1,
                swe_score=55.8,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=32000,
                is_free=False,
                release_date=datetime(2024, 9, 1),
            ),
            EnhancedModelInfo(
                name="mistral-small-latest",
                provider="mistral",
                description="Mistral Small - Fast and cost-effective",
                max_tokens=32000,
                cost_per_million_input=0.2,
                cost_per_million_output=0.6,
                swe_score=48.2,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=32000,
                is_free=False,
                release_date=datetime(2024, 9, 1),
            ),
            EnhancedModelInfo(
                name="mistral-tiny",
                provider="mistral",
                description="Mistral Tiny - Ultra-fast for simple tasks",
                max_tokens=32000,
                cost_per_million_input=0.14,
                cost_per_million_output=0.42,
                swe_score=35.1,
                available=True,
                supports_tools=False,
                supports_multimodal=False,
                context_window=32000,
                is_free=False,
                release_date=datetime(2024, 9, 1),
            ),
            EnhancedModelInfo(
                name="codestral-latest",
                provider="mistral",
                description="Codestral - Specialized for code generation and analysis",
                max_tokens=32000,
                cost_per_million_input=0.2,
                cost_per_million_output=0.6,
                swe_score=78.2,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=32000,
                is_free=False,
                release_date=datetime(2024, 9, 1),
            ),
        ]

        # Update SWE ratings for all models
        for model in models:
            model.update_swe_rating()

        return models

    def supports_tools(self) -> bool:
        """
        Check if Mistral provider supports tool calling.

        Returns:
            True for most models except tiny variants
        """
        non_tool_models = ["mistral-tiny"]
        return self.model not in non_tool_models

    def supports_multimodal(self) -> bool:
        """
        Check if Mistral provider supports multimodal inputs.

        Returns:
            False - Current Mistral models don't support multimodal inputs
        """
        return False

    def supports_streaming(self) -> bool:
        """
        Check if Mistral provider supports streaming responses.

        Returns:
            True - Mistral supports streaming
        """
        return True
