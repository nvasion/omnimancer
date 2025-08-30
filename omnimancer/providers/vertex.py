"""
Google Vertex AI provider implementation for Omnimancer.

This module provides the Google Vertex AI provider implementation using Vertex AI's API
with support for Google Cloud project configuration, location settings, and service account authentication.
"""

import os
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


class VertexAIProvider(BaseProvider):
    """
    Google Vertex AI provider implementation using Vertex AI's API.

    Supports Google Cloud project and location configuration,
    service account authentication, and Gemini models.
    """

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize Vertex AI provider.

        Args:
            api_key: Google Cloud API key or service account key
            model: Vertex AI model to use (e.g., 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro')
            **kwargs: Additional configuration including Vertex AI-specific settings
        """
        super().__init__(api_key, model or "gemini-1.5-pro", **kwargs)

        # Vertex AI-specific configuration
        self.vertex_project = kwargs.get("vertex_project")
        self.vertex_location = kwargs.get("vertex_location", "us-central1")
        self.service_account_path = kwargs.get("service_account_path")

        # Standard parameters
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.95)
        self.top_k = kwargs.get("top_k", 40)

        # Safety settings
        self.safety_settings = kwargs.get("safety_settings", {})

        # Validate required configuration
        if not self.vertex_project:
            raise ValueError("vertex_project is required for Vertex AI provider")

        # Set up authentication
        self._setup_authentication()

        # Build base URL
        self.base_url = f"https://{self.vertex_location}-aiplatform.googleapis.com/v1/projects/{self.vertex_project}/locations/{self.vertex_location}/publishers/google/models"

    def _setup_authentication(self):
        """Set up Google Cloud authentication."""
        if self.service_account_path:
            # Use service account file
            if os.path.exists(self.service_account_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.service_account_path
            else:
                raise ValueError(
                    f"Service account file not found: {self.service_account_path}"
                )
        elif self.api_key and self.api_key != "":
            # Use API key (for testing or specific configurations)
            self.auth_token = self.api_key
        else:
            # Try to use default credentials
            try:
                # This would normally use Google's auth libraries
                # For now, we'll require explicit configuration
                raise ValueError(
                    "Either service_account_path or api_key must be provided for Vertex AI"
                )
            except Exception as e:
                raise ValueError(f"Failed to set up Vertex AI authentication: {e}")

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to Vertex AI API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Vertex AI's reply
        """
        try:
            # Prepare request for Vertex AI
            request_data = self._prepare_vertex_request(message, context)

            # Build URL for the specific model
            url = f"{self.base_url}/{self.model}:generateContent"

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_auth_headers(),
                    json=request_data,
                    timeout=30.0,
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Vertex AI API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    async def send_message_with_tools(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> ChatResponse:
        """
        Send a message with available tools for Vertex AI to use.

        Args:
            message: User message
            context: Conversation context
            available_tools: List of tools available for the model to use

        Returns:
            ChatResponse with Vertex AI's reply and any tool calls
        """
        try:
            # Prepare request for Vertex AI with tools
            request_data = self._prepare_vertex_request_with_tools(
                message, context, available_tools
            )

            # Build URL for the specific model
            url = f"{self.base_url}/{self.model}:generateContent"

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_auth_headers(),
                    json=request_data,
                    timeout=30.0,
                )

            # Handle response with tool calls
            return self._handle_response_with_tools(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to Vertex AI API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    async def validate_credentials(self) -> bool:
        """
        Validate Vertex AI credentials by making a test request.

        Returns:
            True if credentials are valid
        """
        try:
            url = f"{self.base_url}/{self.model}:generateContent"

            test_request = {
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
                "generationConfig": {
                    "maxOutputTokens": 10,
                    "temperature": 0.1,
                },
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=self._get_auth_headers(),
                    json=test_request,
                    timeout=10.0,
                )

            return response.status_code == 200

        except Exception:
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for Vertex AI API.

        Returns:
            Dictionary of headers
        """
        headers = {"Content-Type": "application/json"}

        if hasattr(self, "auth_token") and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        return headers

    def _prepare_vertex_request(self, message: str, context: ChatContext) -> Dict:
        """
        Prepare request data for Vertex AI API.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            Request data formatted for Vertex AI API
        """
        contents = []

        # Add context messages
        for msg in context.messages:
            role = "user" if msg.role.value == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        # Add current message
        contents.append({"role": "user", "parts": [{"text": message}]})

        request_data = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
                "topK": self.top_k,
            },
        }

        # Add safety settings if configured
        if self.safety_settings:
            request_data["safetySettings"] = self._format_safety_settings()

        return request_data

    def _prepare_vertex_request_with_tools(
        self, message: str, context: ChatContext, tools: List[ToolDefinition]
    ) -> Dict:
        """
        Prepare request data for Vertex AI API with tools.

        Args:
            message: Current user message
            context: Conversation context
            tools: List of available tools

        Returns:
            Request data formatted for Vertex AI API with tools
        """
        request_data = self._prepare_vertex_request(message, context)

        # Add tools in Vertex AI format
        if tools:
            request_data["tools"] = [
                self._convert_tool_to_vertex_format(tool) for tool in tools
            ]

        return request_data

    def _convert_tool_to_vertex_format(self, tool: ToolDefinition) -> Dict:
        """
        Convert tool definition to Vertex AI format.

        Args:
            tool: Tool definition

        Returns:
            Tool formatted for Vertex AI API
        """
        return {
            "functionDeclarations": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            ]
        }

    def _format_safety_settings(self) -> List[Dict]:
        """
        Format safety settings for Vertex AI API.

        Returns:
            List of safety settings
        """
        safety_settings = []

        # Default safety categories
        categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]

        for category in categories:
            threshold = self.safety_settings.get(category, "BLOCK_MEDIUM_AND_ABOVE")
            safety_settings.append({"category": category, "threshold": threshold})

        return safety_settings

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Vertex AI API response.

        Args:
            response: HTTP response from Vertex AI API

        Returns:
            ChatResponse object

        Raises:
            Various provider errors based on response status
        """
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])

            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                if parts and len(parts) > 0:
                    text = parts[0].get("text", "")

                    # Extract usage information
                    usage_metadata = data.get("usageMetadata", {})
                    total_tokens = usage_metadata.get("totalTokenCount", 0)

                    return ChatResponse(
                        content=text,
                        model_used=self.model,
                        tokens_used=total_tokens,
                        timestamp=datetime.now(),
                    )
                else:
                    raise ProviderError("Empty parts in Vertex AI response")
            else:
                raise ProviderError("Empty candidates in Vertex AI response")

        elif response.status_code == 401:
            raise AuthenticationError("Invalid Vertex AI credentials")
        elif response.status_code == 429:
            raise RateLimitError("Vertex AI API rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError(f"Vertex AI model '{self.model}' not found")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"Vertex AI API error: {error_msg}")

    def _handle_response_with_tools(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Vertex AI API response with tool calls.

        Args:
            response: HTTP response from Vertex AI API

        Returns:
            ChatResponse object with tool calls
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

                for part in parts:
                    if "text" in part:
                        text_content += part["text"]
                    elif "functionCall" in part:
                        func_call = part["functionCall"]
                        tool_calls.append(
                            ToolCall(
                                name=func_call.get("name", ""),
                                arguments=func_call.get("args", {}),
                            )
                        )

                # Extract usage information
                usage_metadata = data.get("usageMetadata", {})
                total_tokens = usage_metadata.get("totalTokenCount", 0)

                return ChatResponse(
                    content=text_content,
                    model_used=self.model,
                    tokens_used=total_tokens,
                    tool_calls=tool_calls if tool_calls else None,
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty candidates in Vertex AI response")
        else:
            return self._handle_response(response)

    def get_model_info(self) -> EnhancedModelInfo:
        """
        Get information about the current Vertex AI model.
        """
        model_configs = {
            "gemini-1.5-pro": {
                "description": "Gemini 1.5 Pro - Advanced reasoning and long context",
                "max_tokens": 2097152,
                "cost_per_million_input": 1.25,
                "cost_per_million_output": 5.0,
                "swe_score": 71.9,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "gemini-1.5-flash": {
                "description": "Gemini 1.5 Flash - Fast and efficient",
                "max_tokens": 1048576,
                "cost_per_million_input": 0.075,
                "cost_per_million_output": 0.3,
                "swe_score": 61.5,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "gemini-1.0-pro": {
                "description": "Gemini 1.0 Pro - Balanced performance",
                "max_tokens": 32768,
                "cost_per_million_input": 0.5,
                "cost_per_million_output": 1.5,
                "swe_score": 53.2,
                "supports_tools": True,
                "supports_multimodal": False,
            },
            "gemini-1.0-pro-vision": {
                "description": "Gemini 1.0 Pro Vision - Multimodal capabilities",
                "max_tokens": 16384,
                "cost_per_million_input": 0.25,
                "cost_per_million_output": 0.5,
                "swe_score": 53.2,
                "supports_tools": False,
                "supports_multimodal": True,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"Vertex AI model {self.model}",
                "max_tokens": 32768,
                "cost_per_million_input": 1.0,
                "cost_per_million_output": 3.0,
                "swe_score": 55.0,
                "supports_tools": True,
                "supports_multimodal": True,
            },
        )

        enhanced_info = EnhancedModelInfo(
            name=self.model,
            provider="vertex",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_million_input=config["cost_per_million_input"],
            cost_per_million_output=config["cost_per_million_output"],
            swe_score=config["swe_score"],
            available=True,
            supports_tools=config["supports_tools"],
            supports_multimodal=config["supports_multimodal"],
            latest_version=self.model == "gemini-1.5-pro",
            context_window=config["max_tokens"],
            is_free=False,
            release_date=datetime(2024, 5, 1),  # Approximate release date
        )

        # Update SWE rating based on score
        enhanced_info.update_swe_rating()

        return enhanced_info

    def get_available_models(self) -> List[EnhancedModelInfo]:
        """
        Get list of available Vertex AI models.
        """
        models = [
            EnhancedModelInfo(
                name="gemini-1.5-pro",
                provider="vertex",
                description="Gemini 1.5 Pro - Advanced reasoning and long context",
                max_tokens=2097152,
                cost_per_million_input=1.25,
                cost_per_million_output=5.0,
                swe_score=71.9,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
                context_window=2097152,
                is_free=False,
                release_date=datetime(2024, 5, 1),
            ),
            EnhancedModelInfo(
                name="gemini-1.5-flash",
                provider="vertex",
                description="Gemini 1.5 Flash - Fast and efficient",
                max_tokens=1048576,
                cost_per_million_input=0.075,
                cost_per_million_output=0.3,
                swe_score=61.5,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                context_window=1048576,
                is_free=False,
                release_date=datetime(2024, 5, 1),
            ),
            EnhancedModelInfo(
                name="gemini-1.0-pro",
                provider="vertex",
                description="Gemini 1.0 Pro - Balanced performance",
                max_tokens=32768,
                cost_per_million_input=0.5,
                cost_per_million_output=1.5,
                swe_score=53.2,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
                context_window=32768,
                is_free=False,
                release_date=datetime(2024, 2, 1),
            ),
            EnhancedModelInfo(
                name="gemini-1.0-pro-vision",
                provider="vertex",
                description="Gemini 1.0 Pro Vision - Multimodal capabilities",
                max_tokens=16384,
                cost_per_million_input=0.25,
                cost_per_million_output=0.5,
                swe_score=53.2,
                available=True,
                supports_tools=False,
                supports_multimodal=True,
                context_window=16384,
                is_free=False,
                release_date=datetime(2024, 2, 1),
            ),
        ]

        # Update SWE ratings for all models
        for model in models:
            model.update_swe_rating()

        return models

    def supports_tools(self) -> bool:
        """
        Check if Vertex AI provider supports tool calling.

        Returns:
            True for most models except vision-only variants
        """
        non_tool_models = ["gemini-1.0-pro-vision"]
        return self.model not in non_tool_models

    def supports_multimodal(self) -> bool:
        """
        Check if Vertex AI provider supports multimodal inputs.

        Returns:
            True for Gemini 1.5 models and vision variants
        """
        multimodal_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro-vision",
        ]
        return any(model in self.model for model in multimodal_models)

    def supports_streaming(self) -> bool:
        """
        Check if Vertex AI provider supports streaming responses.

        Returns:
            True - Vertex AI supports streaming
        """
        return True
