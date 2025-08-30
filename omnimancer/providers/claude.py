"""
Claude provider implementation for Omnimancer.

This module provides the Claude AI provider implementation using Anthropic's API.
"""

from datetime import datetime
from typing import Dict, List

import certifi
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


class ClaudeProvider(BaseProvider):
    """
    Claude AI provider implementation using Anthropic's API.
    """

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str, model: str = "", **kwargs):
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key
            model: Claude model to use (e.g., 'claude-sonnet-4-20250514', 'claude-3-5-sonnet-20241022')
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model or "claude-sonnet-4-20250514", **kwargs)
        self.max_tokens = kwargs.get("max_tokens") or 4096
        self.temperature = kwargs.get("temperature") or 0.7

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to Claude API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Claude's reply
        """
        # Prepare messages for Claude API
        messages = self._prepare_messages(message, context)

        # Try with SSL verification first, then fall back if needed
        for ssl_verify in [True, certifi.where(), False]:
            try:
                async with httpx.AsyncClient(verify=ssl_verify) as client:
                    response = await client.post(
                        f"{self.BASE_URL}/messages",
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                        },
                        json={
                            "model": self.model,
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                            "messages": messages,
                        },
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
                raise NetworkError("Request to Claude API timed out")
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
        raise NetworkError("Failed to establish SSL connection to Claude API")

    async def validate_credentials(self) -> bool:
        """
        Validate Claude API credentials by making a test request.

        Returns:
            True if credentials are valid
        """
        # Try different SSL verification methods
        for ssl_verify in [True, certifi.where(), False]:
            try:
                async with httpx.AsyncClient(verify=ssl_verify) as client:
                    response = await client.post(
                        f"{self.BASE_URL}/messages",
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                        },
                        json={
                            "model": self.model,
                            "max_tokens": 10,
                            "messages": [{"role": "user", "content": "Hi"}],
                        },
                        timeout=10.0,
                    )

                # If we get here, connection worked
                # Check for specific status codes
                if response.status_code == 200:
                    return True
                elif response.status_code == 401:
                    # Authentication failed - invalid API key
                    return False
                elif response.status_code == 400:
                    # Bad request - but API key might be valid, could be model issue
                    try:
                        error_data = response.json()
                        error_type = error_data.get("error", {}).get("type", "")
                        # If it's an authentication error, API key is invalid
                        if error_type == "authentication_error":
                            return False
                        # If it's other errors (like invalid model), API key is probably valid
                        return True
                    except:
                        # Can't parse error, assume API key is valid if we got a response
                        return True
                else:
                    # Other status codes - if we got a response, API key is probably valid
                    return True

            except httpx.ConnectError as e:
                if "SSL" in str(e) or "certificate" in str(e):
                    # SSL error, try next verification method
                    continue
                else:
                    # Non-SSL connection error
                    return False
            except httpx.TimeoutException:
                # Timeout might mean network issues, try next SSL method
                continue
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

    def _prepare_messages(
        self, message: str, context: ChatContext
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for Claude API format.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            List of messages formatted for Claude API
        """
        messages = []

        # Add context messages (excluding system messages for Claude)
        for msg in context.messages:
            if msg.role.value != "system":
                messages.append({"role": msg.role.value, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle Claude API response.

        Args:
            response: HTTP response from Claude API

        Returns:
            ChatResponse object

        Raises:
            Various provider errors based on response status
        """
        if response.status_code == 200:
            data = response.json()
            content = data.get("content", [])

            if content and len(content) > 0:
                text_content = content[0].get("text", "")
                usage = data.get("usage", {})

                return ChatResponse(
                    content=text_content,
                    model_used=self.model,
                    tokens_used=usage.get("output_tokens", 0),
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty response from Claude API")

        elif response.status_code == 401:
            raise AuthenticationError("Invalid Claude API key")
        elif response.status_code == 429:
            raise RateLimitError("Claude API rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError(f"Claude model '{self.model}' not found")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"Claude API error: {error_msg}")

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Claude model.
        """
        model_configs = {
            "claude-sonnet-4-20250514": {
                "description": "Claude Sonnet 4 - Latest and most capable model",
                "max_tokens": 200000,
                "cost_per_token": 0.000015,
            },
            "claude-opus-4-20250514": {
                "description": "Claude Opus 4 - Most powerful model for complex tasks",
                "max_tokens": 200000,
                "cost_per_token": 0.000075,
            },
            "claude-3-5-sonnet-20241022": {
                "description": "Claude 3.5 Sonnet - Enhanced reasoning and analysis",
                "max_tokens": 200000,
                "cost_per_token": 0.000015,
            },
            # Legacy models for backward compatibility
            "claude-3-sonnet-20240229": {
                "description": "Claude 3 Sonnet - Balanced performance and speed",
                "max_tokens": 200000,
                "cost_per_token": 0.000015,
            },
            "claude-3-haiku-20240307": {
                "description": "Claude 3 Haiku - Fast and efficient",
                "max_tokens": 200000,
                "cost_per_token": 0.00000025,
            },
            "claude-3-opus-20240229": {
                "description": "Claude 3 Opus - Most capable model",
                "max_tokens": 200000,
                "cost_per_token": 0.000075,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"Claude model {self.model}",
                "max_tokens": 200000,
                "cost_per_token": 0.000015,
            },
        )

        return ModelInfo(
            name=self.model,
            provider="claude",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_token=config["cost_per_token"],
            available=True,
            supports_tools=True,
            supports_multimodal=True,
            latest_version=self.model == "claude-sonnet-4-20250514",
        )

    def _get_static_models(self) -> List[ModelInfo]:
        """
        Get static list of available Claude models.
        """
        return [
            ModelInfo(
                name="claude-sonnet-4-20250514",
                provider="claude",
                description="Claude Sonnet 4 - Latest and most capable model",
                max_tokens=200000,
                cost_per_token=0.000015,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
            ),
            ModelInfo(
                name="claude-opus-4-20250514",
                provider="claude",
                description="Claude Opus 4 - Most powerful model for complex tasks",
                max_tokens=200000,
                cost_per_token=0.000075,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
            ModelInfo(
                name="claude-3-5-sonnet-20241022",
                provider="claude",
                description="Claude 3.5 Sonnet - Enhanced reasoning and analysis",
                max_tokens=200000,
                cost_per_token=0.000015,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
        ]

    async def fetch_live_models(self) -> List[ModelInfo]:
        """
        Fetch live model list from Anthropic API.

        Note: Anthropic doesn't have a public models list endpoint,
        so we return an enhanced static list with current pricing.

        Returns:
            List of ModelInfo objects for Claude models
        """
        # Anthropic doesn't provide a models list endpoint
        # Return the latest known models with updated information
        return [
            ModelInfo(
                name="claude-3-5-sonnet-20241022",
                provider="claude",
                description="Most intelligent model with enhanced coding capabilities",
                max_tokens=8192,
                cost_per_token=0.000003,  # $3 per million input tokens
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
            ),
            ModelInfo(
                name="claude-3-5-haiku-20241022",
                provider="claude",
                description="Fast model optimized for speed and efficiency",
                max_tokens=8192,
                cost_per_token=0.00000025,  # $0.25 per million input tokens
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
            ModelInfo(
                name="claude-3-opus-20240229",
                provider="claude",
                description="Previous generation powerful model for complex tasks",
                max_tokens=4096,
                cost_per_token=0.000015,  # $15 per million input tokens
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
            ModelInfo(
                name="claude-3-haiku-20240307",
                provider="claude",
                description="Previous generation fast and lightweight model",
                max_tokens=4096,
                cost_per_token=0.00000025,  # $0.25 per million input tokens
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
        ]

    def supports_tools(self) -> bool:
        """
        Check if Claude provider supports tool calling.

        Returns:
            True - Claude supports tool calling
        """
        return True

    def supports_multimodal(self) -> bool:
        """
        Check if Claude provider supports multimodal inputs.

        Returns:
            True - Claude supports images and other multimodal inputs
        """
        return True
