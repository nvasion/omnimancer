"""
OpenRouter provider implementation for Omnimancer.

This module provides the OpenRouter provider implementation using OpenRouter's API
with support for model aggregation, cost optimization, and access to multiple models.
"""

from datetime import datetime
from typing import Dict, List

import certifi
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


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter provider implementation using OpenRouter's API.

    Provides access to multiple models through OpenRouter's aggregation service
    with cost optimization features and model selection logic.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            model: OpenRouter model ID to use (e.g., 'anthropic/claude-3.5-sonnet', 'openai/gpt-4', 'meta-llama/llama-3.1-70b-instruct')
            **kwargs: Additional configuration including OpenRouter-specific settings
        """
        super().__init__(api_key, model or "anthropic/claude-3.5-sonnet", **kwargs)

        # OpenRouter-specific configuration
        self.openrouter_referrer = kwargs.get(
            "openrouter_referrer", "https://github.com/omnimancer-cli"
        )
        self.openrouter_title = kwargs.get("openrouter_title", "Omnimancer CLI")

        # Standard parameters
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        self.presence_penalty = kwargs.get("presence_penalty", 0.0)

        # Cost optimization settings
        self.enable_fallback = kwargs.get("enable_fallback", True)
        self.max_cost_per_token = kwargs.get("max_cost_per_token", None)
        self.prefer_cheaper_models = kwargs.get("prefer_cheaper_models", False)

        # Fallback notification settings
        self.show_fallback_warnings = kwargs.get("show_fallback_warnings", True)

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to OpenRouter API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with OpenRouter's reply
        """
        # Prepare messages for OpenRouter API
        messages = self._prepare_messages(message, context)

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": False,
        }

        # Add cost optimization parameters
        if self.enable_fallback:
            payload["route"] = "fallback"

        # Try with SSL verification first, then fall back if needed
        for ssl_verify in [True, certifi.where(), False]:
            try:
                async with httpx.AsyncClient(verify=ssl_verify) as client:
                    response = await client.post(
                        f"{self.BASE_URL}/chat/completions",
                        headers=self._get_headers(),
                        json=payload,
                        timeout=30.0,
                    )

                # If we get here, the request succeeded
                return self._handle_response(response)

            except httpx.ConnectError as e:
                if "SSL" in str(e) or "certificate" in str(e):
                    # SSL error, try next verification method
                    continue
                else:
                    # Non-SSL connection error, don't retry
                    raise NetworkError(f"Connection error: {e}")
            except httpx.TimeoutException:
                raise NetworkError("Request to OpenRouter API timed out")
            except httpx.RequestError as e:
                if "SSL" not in str(e) and "certificate" not in str(e):
                    # Non-SSL request error, don't retry
                    raise NetworkError(f"Network error: {e}")
                # SSL-related error, try next verification method
                continue
            except (
                AuthenticationError,
                RateLimitError,
                ModelNotFoundError,
                ProviderError,
            ):
                # Known provider errors, don't retry
                raise
            except Exception as e:
                # Unknown error, don't retry
                raise ProviderError(f"Unexpected error: {e}")

        # If we get here, all SSL methods failed
        raise NetworkError("Failed to establish SSL connection to OpenRouter API")

    async def send_message_with_tools(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> ChatResponse:
        """
        Send a message with available tools for OpenRouter to use.

        Args:
            message: User message
            context: Conversation context
            available_tools: List of tools available for the model to use

        Returns:
            ChatResponse with OpenRouter's reply and any tool calls
        """
        # Prepare messages for OpenRouter API
        messages = self._prepare_messages(message, context)

        # Convert tools to OpenRouter format
        tools = self._convert_tools_to_openrouter_format(available_tools)

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
        }

        # Add cost optimization parameters
        if self.enable_fallback:
            payload["route"] = "fallback"

        # Try with SSL verification first, then fall back if needed
        for ssl_verify in [True, certifi.where(), False]:
            try:
                async with httpx.AsyncClient(verify=ssl_verify) as client:
                    response = await client.post(
                        f"{self.BASE_URL}/chat/completions",
                        headers=self._get_headers(),
                        json=payload,
                        timeout=30.0,
                    )

                # If we get here, the request succeeded
                return self._handle_response_with_tools(response)

            except httpx.ConnectError as e:
                if "SSL" in str(e) or "certificate" in str(e):
                    # SSL error, try next verification method
                    continue
                else:
                    # Non-SSL connection error, don't retry
                    raise NetworkError(f"Connection error: {e}")
            except httpx.TimeoutException:
                raise NetworkError("Request to OpenRouter API timed out")
            except httpx.RequestError as e:
                if "SSL" not in str(e) and "certificate" not in str(e):
                    # Non-SSL request error, don't retry
                    raise NetworkError(f"Network error: {e}")
                # SSL-related error, try next verification method
                continue
            except (
                AuthenticationError,
                RateLimitError,
                ModelNotFoundError,
                ProviderError,
            ):
                # Known provider errors, don't retry
                raise
            except Exception as e:
                # Unknown error, don't retry
                raise ProviderError(f"Unexpected error: {e}")

        # If we get here, all SSL methods failed
        raise NetworkError("Failed to establish SSL connection to OpenRouter API")

    async def validate_credentials(self) -> bool:
        """
        Validate OpenRouter API credentials by making a test request.

        Returns:
            True if credentials are valid
        """
        # Try different SSL verification methods
        for ssl_verify in [True, certifi.where(), False]:
            try:
                async with httpx.AsyncClient(verify=ssl_verify) as client:
                    response = await client.post(
                        f"{self.BASE_URL}/chat/completions",
                        headers=self._get_headers(),
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 10,
                        },
                        timeout=10.0,
                    )

                # If we get here, connection worked
                return response.status_code == 200

            except httpx.ConnectError as e:
                if "SSL" in str(e) or "certificate" in str(e):
                    # SSL error, try next verification method
                    continue
                else:
                    # Non-SSL connection error
                    return False
            except httpx.RequestError as e:
                if "SSL" not in str(e) and "certificate" not in str(e):
                    # Non-SSL request error
                    return False
                # SSL-related error, try next verification method
                continue
            except Exception as e:
                # Other error (timeout, etc.) - if it's not SSL related, stop trying
                if "SSL" not in str(e) and "certificate" not in str(e):
                    return False
                # SSL-related error, continue to next method
                continue

        # If we get here, all SSL methods failed
        return False

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for OpenRouter API requests.

        Returns:
            Dictionary of headers
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.openrouter_referrer,
            "X-Title": self.openrouter_title,
        }

        return headers

    def _prepare_messages(
        self, message: str, context: ChatContext
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for OpenRouter API format.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            List of messages formatted for OpenRouter API
        """
        messages = []

        # Add context messages
        for msg in context.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    def _convert_tools_to_openrouter_format(
        self, tools: List[ToolDefinition]
    ) -> List[Dict]:
        """
        Convert tool definitions to OpenRouter API format.

        Args:
            tools: List of tool definitions

        Returns:
            List of tools formatted for OpenRouter API
        """
        openrouter_tools = []

        for tool in tools:
            openrouter_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            openrouter_tools.append(openrouter_tool)

        return openrouter_tools

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle OpenRouter API response.

        Args:
            response: HTTP response from OpenRouter API

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

                # Extract model used (OpenRouter may route to different models)
                model_used = data.get("model", self.model)

                # Check for model fallback and add warning if needed
                final_content = content
                if model_used != self.model:
                    # Always log the fallback for debugging
                    logger = __import__("logging").getLogger(__name__)
                    logger.warning(
                        f"OpenRouter model fallback: requested '{self.model}' but got '{model_used}'"
                    )

                    # Add user warning if enabled
                    if self.show_fallback_warnings:
                        fallback_warning = f"⚠️  **Model Fallback Notice**: Requested '{self.model}' but OpenRouter routed to '{model_used}' (model may be unavailable or overloaded)\n\n"
                        final_content = fallback_warning + content

                return ChatResponse(
                    content=final_content,
                    model_used=model_used,
                    tokens_used=usage.get("total_tokens", 0),
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty response from OpenRouter API")

        elif response.status_code == 401:
            raise AuthenticationError("Invalid OpenRouter API key")
        elif response.status_code == 429:
            raise RateLimitError("OpenRouter API rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError(f"OpenRouter model '{self.model}' not found")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"OpenRouter API error: {error_msg}")

    def _handle_response_with_tools(self, response: httpx.Response) -> ChatResponse:
        """
        Handle OpenRouter API response with tool calls.

        Args:
            response: HTTP response from OpenRouter API

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

                # Extract model used
                model_used = data.get("model", self.model)

                # Check for model fallback and add warning if needed
                final_content = content
                if model_used != self.model:
                    # Always log the fallback for debugging
                    logger = __import__("logging").getLogger(__name__)
                    logger.warning(
                        f"OpenRouter model fallback: requested '{self.model}' but got '{model_used}'"
                    )

                    # Add user warning if enabled
                    if self.show_fallback_warnings:
                        fallback_warning = f"⚠️  **Model Fallback Notice**: Requested '{self.model}' but OpenRouter routed to '{model_used}' (model may be unavailable or overloaded)\n\n"
                        final_content = fallback_warning + content

                return ChatResponse(
                    content=final_content,
                    model_used=model_used,
                    tokens_used=usage.get("total_tokens", 0),
                    tool_calls=tool_calls if tool_calls else None,
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty response from OpenRouter API")
        else:
            return self._handle_response(response)

    def get_model_info(self) -> EnhancedModelInfo:
        """
        Get information about the current OpenRouter model.
        """
        # Common OpenRouter model configurations
        model_configs = {
            "anthropic/claude-3.5-sonnet": {
                "description": "Claude 3.5 Sonnet via OpenRouter - Most capable",
                "max_tokens": 200000,
                "cost_per_million_input": 3.0,
                "cost_per_million_output": 15.0,
                "swe_score": 88.7,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "anthropic/claude-3-opus": {
                "description": "Claude 3 Opus via OpenRouter - Highest quality",
                "max_tokens": 200000,
                "cost_per_million_input": 15.0,
                "cost_per_million_output": 75.0,
                "swe_score": 84.9,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "openai/gpt-4o": {
                "description": "GPT-4o via OpenRouter - Fast and capable",
                "max_tokens": 128000,
                "cost_per_million_input": 5.0,
                "cost_per_million_output": 15.0,
                "swe_score": 71.2,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "google/gemini-pro-1.5": {
                "description": "Gemini Pro 1.5 via OpenRouter - Long context",
                "max_tokens": 2097152,
                "cost_per_million_input": 1.25,
                "cost_per_million_output": 5.0,
                "swe_score": 71.9,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "meta-llama/llama-3.1-405b-instruct": {
                "description": "Llama 3.1 405B via OpenRouter - Open source flagship",
                "max_tokens": 131072,
                "cost_per_million_input": 5.0,
                "cost_per_million_output": 15.0,
                "swe_score": 69.4,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "qwen/qwen-2.5-72b-instruct": {
                "description": "Qwen 2.5 72B via OpenRouter - Efficient and capable",
                "max_tokens": 131072,
                "cost_per_million_input": 0.4,
                "cost_per_million_output": 1.2,
                "swe_score": 65.9,
                "supports_tools": True,
                "supports_multimodal": False,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"OpenRouter model {self.model}",
                "max_tokens": 32768,
                "cost_per_million_input": 1.0,
                "cost_per_million_output": 3.0,
                "swe_score": 50.0,
                "supports_tools": True,
                "supports_multimodal": False,
            },
        )

        enhanced_info = EnhancedModelInfo(
            name=self.model,
            provider="openrouter",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_million_input=config["cost_per_million_input"],
            cost_per_million_output=config["cost_per_million_output"],
            swe_score=config["swe_score"],
            available=True,
            supports_tools=config["supports_tools"],
            supports_multimodal=config["supports_multimodal"],
            latest_version="claude-3.5-sonnet" in self.model,
            context_window=config["max_tokens"],
            is_free=False,
            release_date=datetime(2024, 6, 1),  # Approximate OpenRouter availability
        )

        # Update SWE rating based on score
        enhanced_info.update_swe_rating()

        return enhanced_info

    def get_available_models(self) -> List[EnhancedModelInfo]:
        """
        Get list of popular OpenRouter models.

        Note: OpenRouter has hundreds of models. This returns a curated list
        of the most popular and capable models. Any valid model name from
        https://openrouter.ai/models will work, not just the ones listed here.
        """
        models = [
            EnhancedModelInfo(
                name="anthropic/claude-3.5-sonnet",
                provider="openrouter",
                description="Claude 3.5 Sonnet via OpenRouter - Most capable",
                max_tokens=200000,
                cost_per_million_input=3.0,
                cost_per_million_output=15.0,
                swe_score=88.7,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
                context_window=200000,
                is_free=False,
                release_date=datetime(2024, 6, 1),
            ),
            EnhancedModelInfo(
                name="anthropic/claude-3-opus",
                provider="openrouter",
                description="Claude 3 Opus via OpenRouter - Highest quality",
                max_tokens=200000,
                cost_per_million_input=15.0,
                cost_per_million_output=75.0,
                swe_score=84.9,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                context_window=200000,
                is_free=False,
                release_date=datetime(2024, 6, 1),
            ),
            EnhancedModelInfo(
                name="openai/gpt-4o",
                provider="openrouter",
                description="GPT-4o via OpenRouter - Fast and capable",
                max_tokens=128000,
                cost_per_million_input=5.0,
                cost_per_million_output=15.0,
                swe_score=71.2,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                context_window=128000,
                is_free=False,
                release_date=datetime(2024, 6, 1),
            ),
            EnhancedModelInfo(
                name="google/gemini-pro-1.5",
                provider="openrouter",
                description="Gemini Pro 1.5 via OpenRouter - Long context",
                max_tokens=2097152,
                cost_per_million_input=1.25,
                cost_per_million_output=5.0,
                swe_score=71.9,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                context_window=2097152,
                is_free=False,
                release_date=datetime(2024, 6, 1),
            ),
            EnhancedModelInfo(
                name="meta-llama/llama-3.1-405b-instruct",
                provider="openrouter",
                description="Llama 3.1 405B via OpenRouter - Open source flagship",
                max_tokens=131072,
                cost_per_million_input=5.0,
                cost_per_million_output=15.0,
                swe_score=69.4,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=131072,
                is_free=False,
                release_date=datetime(2024, 6, 1),
            ),
            EnhancedModelInfo(
                name="qwen/qwen-2.5-72b-instruct",
                provider="openrouter",
                description="Qwen 2.5 72B via OpenRouter - Efficient and capable",
                max_tokens=131072,
                cost_per_million_input=0.4,
                cost_per_million_output=1.2,
                swe_score=65.9,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=131072,
                is_free=False,
                release_date=datetime(2024, 6, 1),
            ),
        ]

        # Update SWE ratings for all models
        for model in models:
            model.update_swe_rating()

        return models

    def supports_tools(self) -> bool:
        """
        Check if OpenRouter provider supports tool calling.

        Returns:
            True - Most models on OpenRouter support tool calling
        """
        # Most modern models on OpenRouter support tools
        return True

    def supports_multimodal(self) -> bool:
        """
        Check if OpenRouter provider supports multimodal inputs.

        Returns:
            True for models that support vision/multimodal
        """
        multimodal_models = [
            "claude-3",
            "claude-3.5",
            "gpt-4o",
            "gpt-4-vision",
            "gemini-pro",
            "gemini-1.5",
        ]
        return any(model in self.model.lower() for model in multimodal_models)

    def supports_streaming(self) -> bool:
        """
        Check if OpenRouter provider supports streaming responses.

        Returns:
            True - OpenRouter supports streaming
        """
        return True
