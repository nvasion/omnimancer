"""
xAI (Grok) provider implementation for Omnimancer.

This module provides the xAI provider implementation using xAI's API
with support for Grok models, multimodal capabilities, and tool calling.
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


class XAIProvider(BaseProvider):
    """
    xAI (Grok) provider implementation using xAI's API.

    Supports Grok models with advanced reasoning, multimodal capabilities,
    and tool calling functionality.
    """

    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize xAI provider.

        Args:
            api_key: xAI API key
            model: xAI model to use (e.g., 'grok-beta', 'grok-vision-beta')
            **kwargs: Additional configuration including Grok-specific settings
        """
        super().__init__(api_key, model or "grok-beta", **kwargs)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        self.grok_mode = kwargs.get(
            "grok_mode", "balanced"
        )  # balanced, creative, precise
        self.system_prompt = kwargs.get("system_prompt", None)
        self.enable_web_search = kwargs.get("enable_web_search", True)
        self.enable_real_time = kwargs.get("enable_real_time", True)

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to xAI API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Grok's reply
        """
        try:
            # Prepare messages for xAI API
            messages = self._prepare_messages(message, context)

            # Build request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self._get_temperature_for_mode(),
                "stream": False,
            }

            # Add Grok-specific parameters
            if self.enable_web_search:
                payload["tools"] = [{"type": "web_search"}]

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=60.0,  # Longer timeout for complex reasoning
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to xAI API timed out")
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
        Send a message with available tools for Grok to use.

        Args:
            message: User message
            context: Conversation context
            available_tools: List of tools available for Grok to use

        Returns:
            ChatResponse with Grok's reply and any tool calls
        """
        try:
            # Prepare messages for xAI API
            messages = self._prepare_messages(message, context)

            # Convert tools to xAI format
            tools = self._convert_tools_to_xai_format(available_tools)

            # Build request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self._get_temperature_for_mode(),
                "tools": tools,
                "tool_choice": "auto",
                "stream": False,
            }

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=60.0,
                )

            # Handle response with tool calls
            return self._handle_response_with_tools(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to xAI API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    async def validate_credentials(self) -> bool:
        """
        Validate xAI API credentials by making a test request.

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
        Prepare messages for xAI API format.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            List of messages formatted for xAI API
        """
        messages = []

        # Add system prompt if specified
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add context messages
        for msg in context.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    def _get_temperature_for_mode(self) -> float:
        """
        Get temperature setting based on Grok mode.

        Returns:
            Temperature value for the current mode
        """
        mode_temperatures = {"precise": 0.1, "balanced": 0.7, "creative": 1.0}
        return mode_temperatures.get(self.grok_mode, self.temperature)

    def _convert_tools_to_xai_format(self, tools: List[ToolDefinition]) -> List[Dict]:
        """
        Convert tool definitions to xAI API format.

        Args:
            tools: List of tool definitions

        Returns:
            List of tools formatted for xAI API
        """
        xai_tools = []

        for tool in tools:
            xai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            xai_tools.append(xai_tool)

        return xai_tools

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle xAI API response.

        Args:
            response: HTTP response from xAI API

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
                raise ProviderError("Empty response from xAI API")

        elif response.status_code == 401:
            raise AuthenticationError("Invalid xAI API key")
        elif response.status_code == 429:
            raise RateLimitError("xAI API rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError(f"xAI model '{self.model}' not found")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"xAI API error: {error_msg}")

    def _handle_response_with_tools(self, response: httpx.Response) -> ChatResponse:
        """
        Handle xAI API response with tool calls.

        Args:
            response: HTTP response from xAI API

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
                raise ProviderError("Empty response from xAI API")
        else:
            return self._handle_response(response)

    def get_model_info(self) -> EnhancedModelInfo:
        """
        Get information about the current xAI model.
        """
        model_configs = {
            "grok-beta": {
                "description": "Grok Beta - Advanced reasoning with real-time information",
                "max_tokens": 131072,
                "cost_per_million_input": 5.0,
                "cost_per_million_output": 15.0,
                "swe_score": 70.1,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "grok-vision-beta": {
                "description": "Grok Vision Beta - Multimodal model with image understanding",
                "max_tokens": 8192,
                "cost_per_million_input": 5.0,
                "cost_per_million_output": 15.0,
                "swe_score": 65.3,
                "supports_tools": True,
                "supports_multimodal": True,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"xAI model {self.model}",
                "max_tokens": 131072,
                "cost_per_million_input": 5.0,
                "cost_per_million_output": 15.0,
                "swe_score": 65.0,
                "supports_tools": True,
                "supports_multimodal": True,
            },
        )

        enhanced_info = EnhancedModelInfo(
            name=self.model,
            provider="xai",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_million_input=config["cost_per_million_input"],
            cost_per_million_output=config["cost_per_million_output"],
            swe_score=config["swe_score"],
            available=True,
            supports_tools=config["supports_tools"],
            supports_multimodal=config["supports_multimodal"],
            latest_version=self.model == "grok-beta",
            context_window=config["max_tokens"],
            is_free=False,
            release_date=datetime(2024, 12, 1),  # Approximate release date
        )

        # Update SWE rating based on score
        enhanced_info.update_swe_rating()

        return enhanced_info

    def get_available_models(self) -> List[EnhancedModelInfo]:
        """
        Get list of available xAI models.
        """
        models = [
            EnhancedModelInfo(
                name="grok-beta",
                provider="xai",
                description="Grok Beta - Advanced reasoning with real-time information",
                max_tokens=131072,
                cost_per_million_input=5.0,
                cost_per_million_output=15.0,
                swe_score=70.1,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
                context_window=131072,
                is_free=False,
                release_date=datetime(2024, 12, 1),
            ),
            EnhancedModelInfo(
                name="grok-vision-beta",
                provider="xai",
                description="Grok Vision Beta - Multimodal model with image understanding",
                max_tokens=8192,
                cost_per_million_input=5.0,
                cost_per_million_output=15.0,
                swe_score=65.3,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                context_window=8192,
                is_free=False,
                release_date=datetime(2024, 12, 1),
            ),
        ]

        # Update SWE ratings for all models
        for model in models:
            model.update_swe_rating()

        return models

    def supports_tools(self) -> bool:
        """
        Check if xAI provider supports tool calling.

        Returns:
            True - xAI supports function calling and tools
        """
        return True

    def supports_multimodal(self) -> bool:
        """
        Check if xAI provider supports multimodal inputs.

        Returns:
            True - xAI models support images and multimodal inputs
        """
        return True

    def supports_streaming(self) -> bool:
        """
        Check if xAI provider supports streaming responses.

        Returns:
            True - xAI supports streaming
        """
        return True
