"""
Google Gemini provider implementation for Omnimancer.

This module provides the Google Gemini AI provider implementation using Google AI Studio API.
"""

from datetime import datetime
from typing import Dict, List

import httpx

from ..core.models import (
    ChatContext,
    ChatResponse,
    ModelInfo,
    ToolCall,
    ToolDefinition,
)
from ..utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    ProviderError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """
    Google Gemini AI provider implementation using Google AI Studio API.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google AI Studio API key
            model: Gemini model to use (e.g., 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash-exp')
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model or "gemini-1.5-pro", **kwargs)
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.temperature = kwargs.get("temperature", 0.7)

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to Gemini API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Gemini's reply
        """
        try:
            # Prepare messages for Gemini API
            contents = self._prepare_contents(message, context)

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/models/{self.model}:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": self.api_key},
                    json={
                        "contents": contents,
                        "generationConfig": {
                            "maxOutputTokens": self.max_tokens,
                            "temperature": self.temperature,
                        },
                    },
                    timeout=30.0,
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Gemini API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except (
            AuthenticationError,
            RateLimitError,
            ModelNotFoundError,
            ProviderError,
        ):
            # Re-raise our custom exceptions from _handle_response
            raise
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    async def validate_credentials(self) -> bool:
        """
        Validate Gemini API credentials by making a test request.

        Returns:
            True if credentials are valid

        Raises:
            AuthenticationError: If API key is invalid
            NetworkError: If network request fails
            ProviderError: For other validation errors
        """
        if not self.api_key:
            raise AuthenticationError("Gemini API key is required")

        if not self.api_key.startswith("AIza"):
            raise AuthenticationError(
                "Invalid Gemini API key format - should start with 'AIza'"
            )

        try:
            # Test with a minimal request to validate credentials
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}/models/{self.model}:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": self.api_key},
                    json={
                        "contents": [{"parts": [{"text": "test"}]}],
                        "generationConfig": {
                            "maxOutputTokens": 1,
                            "temperature": 0,
                        },
                    },
                    timeout=10.0,
                )

            if response.status_code == 200:
                return True
            elif response.status_code == 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "")
                    if "API_KEY" in error_msg.upper() or "INVALID" in error_msg.upper():
                        raise AuthenticationError(
                            f"Invalid Gemini API key: {error_msg}"
                        )
                    else:
                        raise ProviderError(f"Gemini API validation error: {error_msg}")
                except (ValueError, KeyError):
                    raise ProviderError(
                        "Invalid response from Gemini API during validation"
                    )
            elif response.status_code == 403:
                raise AuthenticationError(
                    "Gemini API access forbidden - check API key permissions"
                )
            elif response.status_code == 404:
                raise ModelNotFoundError(
                    f"Gemini model '{self.model}' not found or not accessible"
                )
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get(
                        "message", f"HTTP {response.status_code}"
                    )
                except:
                    error_msg = f"HTTP {response.status_code}"
                raise ProviderError(f"Gemini API validation failed: {error_msg}")

        except httpx.TimeoutException:
            raise NetworkError("Gemini API validation request timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error during Gemini API validation: {e}")
        except (
            AuthenticationError,
            NetworkError,
            ProviderError,
            ModelNotFoundError,
        ):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise ProviderError(f"Unexpected error during Gemini API validation: {e}")

    def _prepare_contents(self, message: str, context: ChatContext) -> List[Dict]:
        """
        Prepare contents for Gemini API format.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            List of contents formatted for Gemini API
        """
        contents = []

        # Add context messages
        for msg in context.messages:
            role = "user" if msg.role.value == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        # Add current message
        contents.append({"role": "user", "parts": [{"text": message}]})

        return contents

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Gemini API response with comprehensive error handling.

        Args:
            response: HTTP response from Gemini API

        Returns:
            ChatResponse object

        Raises:
            Various provider errors based on response status
        """
        if response.status_code == 200:
            try:
                data = response.json()
                candidates = data.get("candidates", [])

                if candidates and len(candidates) > 0:
                    candidate = candidates[0]

                    # Check for safety filtering or other blocking
                    finish_reason = candidate.get("finishReason")
                    if finish_reason and finish_reason != "STOP":
                        if finish_reason == "SAFETY":
                            raise ProviderError(
                                "Response blocked by Gemini safety filters. Try rephrasing your request.",
                                provider="gemini",
                            )
                        elif finish_reason == "MAX_TOKENS":
                            raise ProviderError(
                                "Response truncated due to token limit. Consider reducing context or max_tokens.",
                                provider="gemini",
                            )
                        else:
                            raise ProviderError(
                                f"Response generation stopped: {finish_reason}",
                                provider="gemini",
                            )

                    content = candidate.get("content", {})
                    parts = content.get("parts", [])

                    if parts and len(parts) > 0:
                        text_content = parts[0].get("text", "")
                        usage_metadata = data.get("usageMetadata", {})

                        return ChatResponse(
                            content=text_content,
                            model_used=self.model,
                            tokens_used=usage_metadata.get("totalTokenCount", 0),
                            timestamp=datetime.now(),
                        )
                    else:
                        raise ProviderError(
                            "Empty parts in Gemini API response",
                            provider="gemini",
                        )
                else:
                    raise ProviderError(
                        "Empty candidates in Gemini API response",
                        provider="gemini",
                    )
            except (ValueError, KeyError) as e:
                raise ProviderError(
                    f"Invalid JSON response from Gemini API: {e}",
                    provider="gemini",
                )

        elif response.status_code == 400:
            try:
                error_data = response.json()
                error_info = error_data.get("error", {})
                error_msg = error_info.get("message", "Bad request")
                error_code = error_info.get("code")

                if (
                    "API_KEY" in error_msg.upper()
                    or "INVALID_API_KEY" in error_msg.upper()
                ):
                    raise AuthenticationError(
                        "Invalid Gemini API key. Check your Google AI Studio API key.",
                        provider="gemini",
                    )
                elif "QUOTA" in error_msg.upper() or error_code == 429:
                    raise QuotaExceededError(
                        "Gemini API quota exceeded. Check your usage limits.",
                        provider="gemini",
                    )
                elif "MODEL_NOT_FOUND" in error_msg.upper():
                    available_models = [m.name for m in self.get_available_models()]
                    raise ModelNotFoundError(
                        f"Gemini model '{self.model}' not found",
                        provider="gemini",
                        model_name=self.model,
                        available_models=available_models,
                    )
                else:
                    raise ProviderError(
                        f"Gemini API error: {error_msg}", provider="gemini"
                    )
            except (ValueError, KeyError):
                raise ProviderError("Bad request to Gemini API", provider="gemini")

        elif response.status_code == 401:
            raise AuthenticationError(
                "Gemini API authentication failed. Verify your API key.",
                provider="gemini",
            )

        elif response.status_code == 403:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "")
                if "PERMISSION_DENIED" in error_msg:
                    raise AuthenticationError(
                        "Gemini API access denied. Check API key permissions and billing.",
                        provider="gemini",
                    )
                else:
                    raise AuthenticationError(
                        "Gemini API access forbidden - check API key permissions",
                        provider="gemini",
                    )
            except (ValueError, KeyError):
                raise AuthenticationError(
                    "Gemini API access forbidden - check API key permissions",
                    provider="gemini",
                )

        elif response.status_code == 429:
            # Extract retry-after header if available
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    retry_seconds = int(retry_after)
                    raise RateLimitError(
                        "Gemini API rate limit exceeded",
                        provider="gemini",
                        retry_after=retry_seconds,
                    )
                except ValueError:
                    pass

            raise RateLimitError(
                "Gemini API rate limit exceeded. Wait before retrying.",
                provider="gemini",
            )

        elif response.status_code == 404:
            available_models = [m.name for m in self.get_available_models()]
            raise ModelNotFoundError(
                f"Gemini model '{self.model}' not found or not accessible",
                provider="gemini",
                model_name=self.model,
                available_models=available_models,
            )

        elif response.status_code == 500:
            raise ProviderUnavailableError(
                "Gemini API server error. Try again later.",
                provider="gemini",
                estimated_recovery="a few minutes",
            )

        elif response.status_code == 502 or response.status_code == 503:
            raise ProviderUnavailableError(
                "Gemini API temporarily unavailable. Service may be under maintenance.",
                provider="gemini",
                estimated_recovery="10-30 minutes",
            )

        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"Gemini API error: {error_msg}", provider="gemini")

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Gemini model.
        """
        model_configs = {
            "gemini-1.5-pro": {
                "description": "Gemini 1.5 Pro - Most capable multimodal model",
                "max_tokens": 2097152,  # 2M tokens context
                "cost_per_token": 0.0000035,  # Approximate cost per token
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "gemini-1.5-flash": {
                "description": "Gemini 1.5 Flash - Fast and efficient multimodal model",
                "max_tokens": 1048576,  # 1M tokens context
                "cost_per_token": 0.00000035,  # Approximate cost per token
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "gemini-1.0-pro": {
                "description": "Gemini 1.0 Pro - Reliable text generation model",
                "max_tokens": 32768,
                "cost_per_token": 0.0000005,  # Approximate cost per token
                "supports_tools": False,
                "supports_multimodal": False,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"Gemini model {self.model}",
                "max_tokens": 32768,
                "cost_per_token": 0.0000035,
                "supports_tools": True,
                "supports_multimodal": True,
            },
        )

        return ModelInfo(
            name=self.model,
            provider="gemini",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_token=config["cost_per_token"],
            available=True,
            supports_tools=config["supports_tools"],
            supports_multimodal=config["supports_multimodal"],
            latest_version=self.model == "gemini-1.5-pro",
        )

    def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available Gemini models.
        """
        return [
            ModelInfo(
                name="gemini-1.5-pro",
                provider="gemini",
                description="Gemini 1.5 Pro - Most capable multimodal model",
                max_tokens=2097152,
                cost_per_token=0.0000035,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
            ),
            ModelInfo(
                name="gemini-1.5-flash",
                provider="gemini",
                description="Gemini 1.5 Flash - Fast and efficient multimodal model",
                max_tokens=1048576,
                cost_per_token=0.00000035,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
            ModelInfo(
                name="gemini-1.0-pro",
                provider="gemini",
                description="Gemini 1.0 Pro - Reliable text generation model",
                max_tokens=32768,
                cost_per_token=0.0000005,
                available=True,
                supports_tools=False,
                supports_multimodal=False,
            ),
        ]

    async def fetch_live_models(self) -> List[ModelInfo]:
        """
        Fetch live model list from Google AI API.

        Returns:
            List of ModelInfo objects from Google AI API
        """
        try:
            # Use the models.list endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/models",
                    params={"key": self.api_key},
                    timeout=30.0,
                )
                response.raise_for_status()

                data = response.json()
                models = []

                # Parse models from API response
                for model_data in data.get("models", []):
                    model_name = model_data.get("name", "").replace("models/", "")

                    # Only include generative models
                    if "gemini" in model_name.lower():
                        # Extract model capabilities
                        supported_methods = model_data.get(
                            "supportedGenerationMethods", []
                        )
                        supports_tools = "generateContent" in supported_methods

                        # Determine if multimodal based on model name
                        supports_multimodal = (
                            "vision" in model_name or "1.5" in model_name
                        )

                        # Get token limits
                        model_data.get("inputTokenLimit", 32768)
                        output_limit = model_data.get("outputTokenLimit", 8192)

                        # Estimate costs (these are approximate)
                        cost_per_token = 0.0000005  # Default
                        if "pro" in model_name.lower():
                            cost_per_token = 0.0000035
                        elif "flash" in model_name.lower():
                            cost_per_token = 0.00000035

                        models.append(
                            ModelInfo(
                                name=model_name,
                                provider="gemini",
                                description=model_data.get("displayName", model_name),
                                max_tokens=output_limit,
                                cost_per_token=cost_per_token,
                                available=True,
                                supports_tools=supports_tools,
                                supports_multimodal=supports_multimodal,
                                latest_version="latest"
                                in model_data.get("version", ""),
                            )
                        )

                return models if models else self.get_available_models()

        except Exception:
            # Fall back to static model list if API call fails
            return self.get_available_models()

    def supports_tools(self) -> bool:
        """
        Check if Gemini provider supports tool calling.

        Returns:
            True for newer models, False for gemini-1.0-pro
        """
        # Only newer Gemini models support function calling
        return self.model != "gemini-1.0-pro"

    def supports_multimodal(self) -> bool:
        """
        Check if Gemini provider supports multimodal inputs.

        Returns:
            True for newer models, False for gemini-1.0-pro
        """
        # Only newer Gemini models support multimodal inputs
        return self.model != "gemini-1.0-pro"

    async def send_message_with_tools(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> ChatResponse:
        """
        Send a message with available tools for Gemini to use.

        Args:
            message: The user's message
            context: Current conversation context
            available_tools: List of tools available for the AI to use

        Returns:
            ChatResponse with the AI's reply and any tool calls
        """
        # If model doesn't support tools, fall back to regular message sending
        if not self.supports_tools():
            return await self.send_message(message, context)

        try:
            # Prepare messages for Gemini API
            contents = self._prepare_contents(message, context)

            # Convert tools to Gemini function format
            tools = self._convert_tools_to_gemini_format(available_tools)

            # Make API request with tools
            async with httpx.AsyncClient() as client:
                request_data = {
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                }

                # Add tools if available
                if tools:
                    request_data["tools"] = tools

                response = await client.post(
                    f"{self.BASE_URL}/models/{self.model}:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": self.api_key},
                    json=request_data,
                    timeout=30.0,
                )

            # Handle response with potential tool calls
            return self._handle_response_with_tools(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Gemini API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    def _convert_tools_to_gemini_format(
        self, tools: List[ToolDefinition]
    ) -> List[Dict]:
        """
        Convert ToolDefinition objects to Gemini function calling format.

        Args:
            tools: List of tool definitions

        Returns:
            List of tools formatted for Gemini API
        """
        if not tools:
            return []

        gemini_tools = []
        function_declarations = []

        for tool in tools:
            function_declaration = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            function_declarations.append(function_declaration)

        if function_declarations:
            gemini_tools.append({"function_declarations": function_declarations})

        return gemini_tools

    def _handle_response_with_tools(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Gemini API response that may contain tool calls.

        Args:
            response: HTTP response from Gemini API

        Returns:
            ChatResponse object with potential tool calls
        """
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])

            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                text_content = ""
                tool_calls = []

                # Process all parts in the response
                for part in parts:
                    if "text" in part:
                        text_content += part["text"]
                    elif "functionCall" in part:
                        # Extract function call information
                        function_call = part["functionCall"]
                        tool_call = ToolCall(
                            name=function_call.get("name", ""),
                            arguments=function_call.get("args", {}),
                            server_name=None,  # Will be set by MCP manager
                        )
                        tool_calls.append(tool_call)

                usage_metadata = data.get("usageMetadata", {})

                return ChatResponse(
                    content=text_content,
                    model_used=self.model,
                    tokens_used=usage_metadata.get("totalTokenCount", 0),
                    timestamp=datetime.now(),
                    tool_calls=tool_calls if tool_calls else None,
                )
            else:
                raise ProviderError("Empty candidates in Gemini API response")
        else:
            # Use the existing error handling for non-200 responses
            return self._handle_response(response)
