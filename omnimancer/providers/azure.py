"""
Azure OpenAI provider implementation for Omnimancer.

This module provides the Azure OpenAI provider implementation using Azure's OpenAI Service
with support for deployment names, API versioning, and Azure-specific authentication.
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


class AzureProvider(BaseProvider):
    """
    Azure OpenAI provider implementation using Azure OpenAI Service.

    Supports Azure-specific endpoint configuration, deployment names,
    API versioning, and Azure authentication methods.
    """

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize Azure OpenAI provider.

        Args:
            api_key: Azure OpenAI API key
            model: Model deployment name in Azure (e.g., 'gpt-4', 'gpt-35-turbo', custom deployment name)
            **kwargs: Additional configuration including Azure-specific settings
        """
        super().__init__(api_key, model or "gpt-4", **kwargs)

        # Azure-specific configuration
        self.azure_endpoint = kwargs.get("azure_endpoint")
        self.azure_deployment = kwargs.get("azure_deployment", model)
        self.api_version = kwargs.get("api_version", "2024-02-15-preview")

        # Standard OpenAI parameters
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        self.presence_penalty = kwargs.get("presence_penalty", 0.0)

        # Validate required Azure configuration
        if not self.azure_endpoint:
            raise ValueError("azure_endpoint is required for Azure OpenAI provider")

        # Ensure endpoint format is correct
        if not self.azure_endpoint.startswith("https://"):
            self.azure_endpoint = f"https://{self.azure_endpoint}"

        if not self.azure_endpoint.endswith(".openai.azure.com"):
            if not self.azure_endpoint.endswith(".openai.azure.com/"):
                self.azure_endpoint = (
                    f"{self.azure_endpoint.rstrip('/')}.openai.azure.com"
                )

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to Azure OpenAI API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Azure OpenAI's reply
        """
        try:
            # Prepare messages for Azure OpenAI API
            messages = self._prepare_messages(message, context)

            # Build request payload
            payload = {
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "stream": False,
            }

            # Build URL for Azure OpenAI
            url = self._build_azure_url("chat/completions")

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "api-key": self.api_key,
                    },
                    json=payload,
                    timeout=30.0,
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Azure OpenAI API timed out")
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
        Send a message with available tools for Azure OpenAI to use.

        Args:
            message: User message
            context: Conversation context
            available_tools: List of tools available for the model to use

        Returns:
            ChatResponse with Azure OpenAI's reply and any tool calls
        """
        try:
            # Prepare messages for Azure OpenAI API
            messages = self._prepare_messages(message, context)

            # Convert tools to Azure OpenAI format
            tools = self._convert_tools_to_azure_format(available_tools)

            # Build request payload
            payload = {
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

            # Build URL for Azure OpenAI
            url = self._build_azure_url("chat/completions")

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "api-key": self.api_key,
                    },
                    json=payload,
                    timeout=30.0,
                )

            # Handle response with tool calls
            return self._handle_response_with_tools(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Azure OpenAI API timed out")
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
        Validate Azure OpenAI API credentials by making a test request.

        Returns:
            True if credentials are valid
        """
        try:
            url = self._build_azure_url("chat/completions")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "api-key": self.api_key,
                    },
                    json={
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 10,
                    },
                    timeout=10.0,
                )

            return response.status_code == 200

        except Exception:
            return False

    def _build_azure_url(self, endpoint: str) -> str:
        """
        Build Azure OpenAI API URL.

        Args:
            endpoint: API endpoint (e.g., "chat/completions")

        Returns:
            Complete Azure OpenAI API URL
        """
        base_url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}"
        return f"{base_url}/{endpoint}?api-version={self.api_version}"

    def _prepare_messages(
        self, message: str, context: ChatContext
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for Azure OpenAI API format.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            List of messages formatted for Azure OpenAI API
        """
        messages = []

        # Add context messages
        for msg in context.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    def _convert_tools_to_azure_format(self, tools: List[ToolDefinition]) -> List[Dict]:
        """
        Convert tool definitions to Azure OpenAI API format.

        Args:
            tools: List of tool definitions

        Returns:
            List of tools formatted for Azure OpenAI API
        """
        azure_tools = []

        for tool in tools:
            azure_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            azure_tools.append(azure_tool)

        return azure_tools

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Azure OpenAI API response.

        Args:
            response: HTTP response from Azure OpenAI API

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

                # Use model name instead of deployment name for consistency
                model_name = self.model if self.model else self.azure_deployment
                return ChatResponse(
                    content=content,
                    model_used=model_name,
                    tokens_used=usage.get("total_tokens", 0),
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty response from Azure OpenAI API")

        elif response.status_code == 401:
            raise AuthenticationError(
                "Invalid Azure OpenAI API key or insufficient permissions"
            )
        elif response.status_code == 429:
            raise RateLimitError("Azure OpenAI API rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError(
                f"Azure deployment '{self.azure_deployment}' not found"
            )
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"Azure OpenAI API error: {error_msg}")

    def _handle_response_with_tools(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Azure OpenAI API response with tool calls.

        Args:
            response: HTTP response from Azure OpenAI API

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

                # Use model name instead of deployment name for consistency
                model_name = self.model if self.model else self.azure_deployment
                return ChatResponse(
                    content=content,
                    model_used=model_name,
                    tokens_used=usage.get("total_tokens", 0),
                    tool_calls=tool_calls if tool_calls else None,
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty response from Azure OpenAI API")
        else:
            return self._handle_response(response)

    def get_model_info(self) -> EnhancedModelInfo:
        """
        Get information about the current Azure OpenAI model.
        """
        # Map common Azure deployment names to model info
        model_configs = {
            "gpt-4": {
                "description": "GPT-4 via Azure OpenAI Service",
                "max_tokens": 8192,
                "cost_per_million_input": 30.0,
                "cost_per_million_output": 60.0,
                "swe_score": 67.0,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "gpt-4-turbo": {
                "description": "GPT-4 Turbo on Azure - Enhanced performance",
                "max_tokens": 128000,
                "cost_per_million_input": 10.0,
                "cost_per_million_output": 30.0,
                "swe_score": 69.1,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "gpt-4o": {
                "description": "GPT-4o on Azure - Optimized for speed and efficiency",
                "max_tokens": 128000,
                "cost_per_million_input": 5.0,
                "cost_per_million_output": 15.0,
                "swe_score": 71.2,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "gpt-35-turbo": {
                "description": "GPT-3.5 Turbo via Azure OpenAI Service",
                "max_tokens": 4096,
                "cost_per_million_input": 0.5,
                "cost_per_million_output": 1.5,
                "swe_score": 48.1,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "gpt-35-turbo-16k": {
                "description": "GPT-3.5 Turbo 16K on Azure - Extended context",
                "max_tokens": 16384,
                "cost_per_million_input": 3.0,
                "cost_per_million_output": 4.0,
                "swe_score": 48.1,
                "supports_tools": True,
                "supports_multimodal": False,
            },
        }

        # Try to match deployment name to known models
        config = None
        for model_name, model_config in model_configs.items():
            if model_name in self.azure_deployment.lower():
                config = model_config
                break

        # Default configuration if no match found
        if not config:
            model_name_for_desc = self.model if self.model else self.azure_deployment
            config = {
                "description": f"Azure OpenAI model {model_name_for_desc}",
                "max_tokens": 4096,
                "cost_per_million_input": 10.0,
                "cost_per_million_output": 30.0,
                "swe_score": 50.0,
                "supports_tools": True,
                "supports_multimodal": False,
            }

        # Use model name instead of deployment name for consistency
        model_name = self.model if self.model else self.azure_deployment
        enhanced_info = EnhancedModelInfo(
            name=model_name,
            provider="azure",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_million_input=config["cost_per_million_input"],
            cost_per_million_output=config["cost_per_million_output"],
            swe_score=config["swe_score"],
            available=True,
            supports_tools=config["supports_tools"],
            supports_multimodal=config["supports_multimodal"],
            latest_version="gpt-4o" in self.azure_deployment.lower(),
            context_window=config["max_tokens"],
            is_free=False,
            release_date=datetime(2024, 5, 1),  # Approximate Azure availability
        )

        # Update SWE rating based on score
        enhanced_info.update_swe_rating()

        return enhanced_info

    def get_available_models(self) -> List[EnhancedModelInfo]:
        """
        Get list of available Azure OpenAI models.

        Note: This returns common Azure deployment configurations.
        Actual availability depends on your Azure subscription and region.
        """
        models = [
            EnhancedModelInfo(
                name="gpt-4",
                provider="azure",
                description="GPT-4 on Azure - Most capable model",
                max_tokens=8192,
                cost_per_million_input=30.0,
                cost_per_million_output=60.0,
                swe_score=67.0,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=8192,
                is_free=False,
                release_date=datetime(2024, 5, 1),
            ),
            EnhancedModelInfo(
                name="gpt-4-turbo",
                provider="azure",
                description="GPT-4 Turbo on Azure - Enhanced performance",
                max_tokens=128000,
                cost_per_million_input=10.0,
                cost_per_million_output=30.0,
                swe_score=69.1,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                context_window=128000,
                is_free=False,
                release_date=datetime(2024, 5, 1),
            ),
            EnhancedModelInfo(
                name="gpt-4o",
                provider="azure",
                description="GPT-4o on Azure - Optimized for speed and efficiency",
                max_tokens=128000,
                cost_per_million_input=5.0,
                cost_per_million_output=15.0,
                swe_score=71.2,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
                context_window=128000,
                is_free=False,
                release_date=datetime(2024, 5, 1),
            ),
            EnhancedModelInfo(
                name="gpt-35-turbo",
                provider="azure",
                description="GPT-3.5 Turbo on Azure - Fast and efficient",
                max_tokens=4096,
                cost_per_million_input=0.5,
                cost_per_million_output=1.5,
                swe_score=48.1,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=4096,
                is_free=False,
                release_date=datetime(2024, 5, 1),
            ),
            EnhancedModelInfo(
                name="gpt-35-turbo-16k",
                provider="azure",
                description="GPT-3.5 Turbo 16K on Azure - Extended context",
                max_tokens=16384,
                cost_per_million_input=3.0,
                cost_per_million_output=4.0,
                swe_score=48.1,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=16384,
                is_free=False,
                release_date=datetime(2024, 5, 1),
            ),
        ]

        # Update SWE ratings for all models
        for model in models:
            model.update_swe_rating()

        return models

    def supports_tools(self) -> bool:
        """
        Check if Azure OpenAI provider supports tool calling.

        Returns:
            True - Azure OpenAI supports function calling
        """
        return True

    def supports_multimodal(self) -> bool:
        """
        Check if Azure OpenAI provider supports multimodal inputs.

        Returns:
            True for GPT-4 models with vision support
        """
        vision_models = ["gpt-4-turbo", "gpt-4o", "gpt-4-vision"]
        return any(model in self.azure_deployment.lower() for model in vision_models)

    def supports_streaming(self) -> bool:
        """
        Check if Azure OpenAI provider supports streaming responses.

        Returns:
            True - Azure OpenAI supports streaming
        """
        return True
