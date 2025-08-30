"""
AWS Bedrock provider implementation for Omnimancer.

This module provides the AWS Bedrock provider implementation using AWS Bedrock's Converse API
with support for Claude models, API key authentication, and region configuration.
"""

import json
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


class BedrockProvider(BaseProvider):
    """
    AWS Bedrock provider implementation using AWS Bedrock's Converse API.

    Supports Claude models through Bedrock, API key authentication,
    and region configuration.
    """

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize AWS Bedrock provider.

        Args:
            api_key: AWS Bedrock API key (30-day key from AWS console)
            model: Bedrock model ID or ARN to use (e.g., 'anthropic.claude-3-sonnet-20240229-v1:0'
                   or 'arn:aws:bedrock:us-west-2::foundation-model/meta.llama4-maverick-17b-instruct-v1:0')
            **kwargs: Additional configuration including region settings
        """
        # Validate API key is provided
        if not api_key:
            raise ValueError("API key is required for AWS Bedrock provider")

        # Validate model is provided (Bedrock requires explicit model specification)
        if not model:
            raise ValueError(
                "Model ID is required for AWS Bedrock provider. "
                "Use a model ID (e.g., 'anthropic.claude-3-sonnet-20240229-v1:0') "
                "or inference profile (e.g., 'us.anthropic.claude-3-sonnet-20240229-v1:0')"
            )

        super().__init__(api_key, model, **kwargs)

        # AWS-specific configuration
        self.aws_region = kwargs.get("aws_region", "us-east-1")

        # Standard parameters
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 1.0)
        self.top_k = kwargs.get("top_k", 250)

        # Build base URL for Bedrock API key authentication
        self.base_url = f"https://bedrock-runtime.{self.aws_region}.amazonaws.com"

    def _is_arn(self, model_id: str) -> bool:
        """Check if the model ID is in ARN format."""
        return model_id.startswith("arn:aws:bedrock:")

    def _extract_model_id_from_arn(self, arn: str) -> str:
        """Extract the model ID from an ARN."""
        if self._is_arn(arn):
            # ARN format: arn:aws:bedrock:region::foundation-model/model-id
            # or: arn:aws:bedrock:region::inference-profile/profile-id
            return arn.split("/")[-1]
        return arn

    def _convert_to_arn(
        self, model_id: str, model_type: str = "foundation-model"
    ) -> str:
        """Convert a model ID to ARN format for the current region."""
        if self._is_arn(model_id):
            return model_id
        return f"arn:aws:bedrock:{self.aws_region}::{model_type}/{model_id}"

    def _suggest_arn_conversion(self, model_id: str, error_message: str) -> dict:
        """Suggest ARN conversion when permission denied."""
        if not self._is_arn(model_id):
            foundation_arn = self._convert_to_arn(model_id, "foundation-model")
            inference_arn = (
                self._convert_to_arn(f"us.{model_id}", "inference-profile")
                if not model_id.startswith("us.")
                else self._convert_to_arn(model_id, "inference-profile")
            )

            return {
                "success": False,
                "message": error_message,
                "suggestion": f"Try using ARN format to specify the exact region. Suggestions:\n"
                f"  Foundation model: {foundation_arn}\n"
                f"  Inference profile: {inference_arn}",
                "suggested_arns": {
                    "foundation_model": foundation_arn,
                    "inference_profile": inference_arn,
                },
            }
        return {"success": False, "message": error_message}

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to AWS Bedrock API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Bedrock's reply
        """
        try:
            # Prepare request for Bedrock
            request_body = self._prepare_bedrock_request(message, context)

            # Build URL for the specific model
            url = f"{self.base_url}/model/{self.model}/converse"

            # Create headers with Bearer token authentication
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, headers=headers, content=request_body, timeout=30.0
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to AWS Bedrock API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except Exception as e:
            raise ProviderError(f"AWS Bedrock API error: {str(e)}")

    async def send_message_with_tools(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> ChatResponse:
        """
        Send a message with available tools for Bedrock to use.

        Args:
            message: User message
            context: Conversation context
            available_tools: List of tools available for the model to use

        Returns:
            ChatResponse with Bedrock's reply and any tool calls
        """
        try:
            # Prepare request for Bedrock with tools
            request_body = self._prepare_bedrock_request_with_tools(
                message, context, available_tools
            )

            # Build URL for the specific model - use appropriate endpoint based on model format
            if self._is_arn(self.model):
                # For ARN format, extract the actual model ID for the URL
                model_for_url = self._extract_model_id_from_arn(self.model)
                url = f"{self.base_url}/model/{model_for_url}/converse"
            else:
                url = f"{self.base_url}/model/{self.model}/converse"

            # Create headers with Bearer token authentication
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, headers=headers, content=request_body, timeout=30.0
                )

            # Handle response with tool calls
            return self._handle_response_with_tools(response)

        except httpx.TimeoutException:
            raise NetworkError("Request to AWS Bedrock API timed out")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except Exception as e:
            raise ProviderError(f"AWS Bedrock API error: {str(e)}")

    async def validate_credentials(self) -> bool:
        """
        Validate AWS Bedrock API key by checking API access (not model-specific).

        Returns:
            True if API key is valid (can access Bedrock API)
        """
        try:
            # Test API key by listing foundation models (not model-specific)
            url = f"https://bedrock.{self.aws_region}.amazonaws.com/foundation-models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10.0)

            # API key is valid if we can access the foundation models endpoint
            return response.status_code == 200

        except Exception:
            return False

    async def validate_model_access(self) -> dict:
        """
        Test access to the specific model configured.

        Returns:
            Dict with 'success', 'message', and optional 'suggestion' keys
        """
        try:
            # Test the specific model with a minimal request
            test_request = {
                "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
                "inferenceConfig": {"maxTokens": 10, "temperature": 0.1},
            }

            # If using ARN format, include the model ARN in the request
            if self._is_arn(self.model):
                test_request["modelId"] = self.model

            test_body = json.dumps(test_request)

            # Build URL for validation - use appropriate endpoint based on model format
            if self._is_arn(self.model):
                model_for_url = self._extract_model_id_from_arn(self.model)
                url = f"{self.base_url}/model/{model_for_url}/converse"
            else:
                url = f"{self.base_url}/model/{self.model}/converse"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, headers=headers, content=test_body, timeout=10.0
                )

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"Model {self.model} is accessible",
                }
            else:
                try:
                    error_data = response.json()
                    error_msg = error_data.get(
                        "message", error_data.get("Message", "Unknown error")
                    )

                    # Parse different error types and provide intelligent suggestions
                    if "not authorized" in error_msg or "explicit deny" in error_msg:
                        return self._suggest_arn_conversion(
                            self.model,
                            f"Access denied to model {self.model}. This may be due to SCP restrictions or regional access issues.",
                        )
                    elif (
                        "not supported" in error_msg or "inference profile" in error_msg
                    ):
                        if not self._is_arn(self.model):
                            # Suggest ARN format for inference profile access
                            inference_arn = (
                                self._convert_to_arn(
                                    f"us.{self.model}", "inference-profile"
                                )
                                if not self.model.startswith("us.")
                                else self._convert_to_arn(
                                    self.model, "inference-profile"
                                )
                            )
                            return {
                                "success": False,
                                "message": f"Model {self.model} requires inference profile access",
                                "suggestion": f"Try using inference profile ARN: {inference_arn}",
                            }
                        else:
                            return {
                                "success": False,
                                "message": f"Model {self.model} requires inference profile access",
                                "suggestion": "Some models require using inference profiles instead of direct model IDs.",
                            }
                    elif "don't have access" in error_msg:
                        return self._suggest_arn_conversion(
                            self.model,
                            f"No access to model {self.model}. You may need to request access in the AWS Bedrock console.",
                        )
                    else:
                        return self._suggest_arn_conversion(
                            self.model, f"Model test failed: {error_msg}"
                        )
                except:
                    return {
                        "success": False,
                        "message": f"Model test failed with HTTP {response.status_code}",
                    }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to test model: {str(e)}",
            }

    def _prepare_bedrock_request(self, message: str, context: ChatContext) -> str:
        """
        Prepare request body for AWS Bedrock Converse API.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            JSON string formatted for Bedrock Converse API
        """
        messages = []

        # Add context messages
        for msg in context.messages:
            messages.append(
                {"role": msg.role.value, "content": [{"text": msg.content}]}
            )

        # Add current message
        messages.append({"role": "user", "content": [{"text": message}]})

        request_data = {
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }

        # If using ARN format, include the model ARN in the request
        if self._is_arn(self.model):
            request_data["modelId"] = self.model

        return json.dumps(request_data)

    def _prepare_bedrock_request_with_tools(
        self, message: str, context: ChatContext, tools: List[ToolDefinition]
    ) -> str:
        """
        Prepare request body for AWS Bedrock Converse API with tools.

        Args:
            message: Current user message
            context: Conversation context
            tools: List of available tools

        Returns:
            JSON string formatted for Bedrock Converse API with tools
        """
        messages = []

        # Add context messages
        for msg in context.messages:
            messages.append(
                {"role": msg.role.value, "content": [{"text": msg.content}]}
            )

        # Add current message
        messages.append({"role": "user", "content": [{"text": message}]})

        request_data = {
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }

        # Add tools in Bedrock Converse format
        if tools:
            request_data["toolConfig"] = {
                "tools": [self._convert_tool_to_bedrock_format(tool) for tool in tools]
            }

        return json.dumps(request_data)

    def _convert_tool_to_bedrock_format(self, tool: ToolDefinition) -> Dict:
        """
        Convert tool definition to Bedrock format.

        Args:
            tool: Tool definition

        Returns:
            Tool formatted for Bedrock API
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle AWS Bedrock Converse API response.

        Args:
            response: HTTP response from Bedrock Converse API

        Returns:
            ChatResponse object

        Raises:
            Various provider errors based on response status
        """
        if response.status_code == 200:
            data = response.json()

            # Handle Bedrock Converse API response format
            if "output" in data:
                output = data.get("output", {})
                message = output.get("message", {})
                content_blocks = message.get("content", [])

                if content_blocks and len(content_blocks) > 0:
                    text = content_blocks[0].get("text", "")
                    usage = data.get("usage", {})

                    return ChatResponse(
                        content=text,
                        model_used=self.model,
                        tokens_used=usage.get("outputTokens", 0),
                        timestamp=datetime.now(),
                    )
                else:
                    raise ProviderError("Empty content in Bedrock response")
            else:
                raise ProviderError("Invalid response format from Bedrock Converse API")

        elif response.status_code == 401:
            raise AuthenticationError("Invalid API key for Bedrock")
        elif response.status_code == 429:
            raise RateLimitError("AWS Bedrock API rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError(f"Bedrock model '{self.model}' not found")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get(
                    "message", error_data.get("Message", "Unknown error")
                )
                # Include more error details for debugging
                error_code = error_data.get("__type", error_data.get("code", ""))
                if error_code:
                    error_msg = f"{error_code}: {error_msg}"
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:200] if response.text else 'No response body'}"

            raise ProviderError(f"AWS Bedrock API error: {error_msg}")

    def _handle_response_with_tools(self, response: httpx.Response) -> ChatResponse:
        """
        Handle AWS Bedrock Converse API response with tool calls.

        Args:
            response: HTTP response from Bedrock Converse API

        Returns:
            ChatResponse object with tool calls
        """
        if response.status_code == 200:
            data = response.json()

            if "output" in data:
                output = data.get("output", {})
                message = output.get("message", {})
                content_blocks = message.get("content", [])
                text_content = ""
                tool_calls = []

                for block in content_blocks:
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
                    elif block.get("type") == "toolUse":
                        tool_calls.append(
                            ToolCall(
                                name=block.get("name", ""),
                                arguments=block.get("input", {}),
                            )
                        )

                usage = data.get("usage", {})

                return ChatResponse(
                    content=text_content,
                    model_used=self.model,
                    tokens_used=usage.get("outputTokens", 0),
                    tool_calls=tool_calls if tool_calls else None,
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Invalid response format from Bedrock Converse API")
        else:
            return self._handle_response(response)

    def get_model_info(self) -> EnhancedModelInfo:
        """
        Get information about the current Bedrock model.
        """
        model_configs = {
            "anthropic.claude-3-sonnet-20240229-v1:0": {
                "description": "Claude 3 Sonnet on AWS Bedrock - Balanced performance",
                "max_tokens": 200000,
                "cost_per_million_input": 3.0,
                "cost_per_million_output": 15.0,
                "swe_score": 73.0,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "anthropic.claude-3-haiku-20240307-v1:0": {
                "description": "Claude 3 Haiku on AWS Bedrock - Fast and efficient",
                "max_tokens": 200000,
                "cost_per_million_input": 0.25,
                "cost_per_million_output": 1.25,
                "swe_score": 75.9,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "anthropic.claude-3-opus-20240229-v1:0": {
                "description": "Claude 3 Opus on AWS Bedrock - Most capable",
                "max_tokens": 200000,
                "cost_per_million_input": 15.0,
                "cost_per_million_output": 75.0,
                "swe_score": 84.9,
                "supports_tools": True,
                "supports_multimodal": True,
            },
            "anthropic.claude-3-5-sonnet-20241022-v2:0": {
                "description": "Claude 3.5 Sonnet on AWS Bedrock - Latest and most capable",
                "max_tokens": 200000,
                "cost_per_million_input": 3.0,
                "cost_per_million_output": 15.0,
                "swe_score": 88.7,
                "supports_tools": True,
                "supports_multimodal": True,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"AWS Bedrock model {self.model}",
                "max_tokens": 200000,
                "cost_per_million_input": 3.0,
                "cost_per_million_output": 15.0,
                "swe_score": 70.0,
                "supports_tools": True,
                "supports_multimodal": True,
            },
        )

        enhanced_info = EnhancedModelInfo(
            name=self.model,
            provider="bedrock",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_million_input=config["cost_per_million_input"],
            cost_per_million_output=config["cost_per_million_output"],
            swe_score=config["swe_score"],
            available=True,
            supports_tools=config["supports_tools"],
            supports_multimodal=config["supports_multimodal"],
            latest_version="claude-3-5-sonnet" in self.model,
            context_window=config["max_tokens"],
            is_free=False,
            release_date=datetime(2024, 6, 1),  # Approximate Bedrock availability
        )

        # Update SWE rating based on score
        enhanced_info.update_swe_rating()

        return enhanced_info

    def get_available_models(self) -> List[EnhancedModelInfo]:
        """
        Get list of available AWS Bedrock models.
        """
        models = [
            EnhancedModelInfo(
                name="anthropic.claude-3-5-sonnet-20241022-v2:0",
                provider="bedrock",
                description="Claude 3.5 Sonnet on AWS Bedrock - Latest and most capable",
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
                release_date=datetime(2024, 10, 1),
            ),
            EnhancedModelInfo(
                name="anthropic.claude-3-opus-20240229-v1:0",
                provider="bedrock",
                description="Claude 3 Opus on AWS Bedrock - Most capable",
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
                name="anthropic.claude-3-sonnet-20240229-v1:0",
                provider="bedrock",
                description="Claude 3 Sonnet on AWS Bedrock - Balanced performance",
                max_tokens=200000,
                cost_per_million_input=3.0,
                cost_per_million_output=15.0,
                swe_score=73.0,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                context_window=200000,
                is_free=False,
                release_date=datetime(2024, 6, 1),
            ),
            EnhancedModelInfo(
                name="anthropic.claude-3-haiku-20240307-v1:0",
                provider="bedrock",
                description="Claude 3 Haiku on AWS Bedrock - Fast and efficient",
                max_tokens=200000,
                cost_per_million_input=0.25,
                cost_per_million_output=1.25,
                swe_score=75.9,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                context_window=200000,
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
        Check if Bedrock provider supports tool calling.

        Returns:
            True - Claude models on Bedrock support tool calling
        """
        return True

    def supports_multimodal(self) -> bool:
        """
        Check if Bedrock provider supports multimodal inputs.

        Returns:
            True - Claude 3 models support multimodal inputs
        """
        return True

    def supports_streaming(self) -> bool:
        """
        Check if Bedrock provider supports streaming responses.

        Returns:
            True - Bedrock supports streaming
        """
        return True
