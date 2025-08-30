"""
Ollama provider implementation for Omnimancer.

This module provides the Ollama provider implementation for local AI inference
using the Ollama server API.
"""

from datetime import datetime
from typing import Dict, List

import httpx

from ..core.models import ChatContext, ChatResponse, ModelInfo
from ..utils.errors import (
    ModelNotFoundError,
    NetworkError,
    ProviderConfigurationError,
    ProviderError,
    ProviderUnavailableError,
)
from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """
    Ollama provider implementation for local AI inference.

    This provider connects to a local Ollama server to run AI models
    locally without requiring external API keys.
    """

    def __init__(self, api_key: str = "", model: str = "", **kwargs):
        """
        Initialize Ollama provider.

        Args:
            api_key: Not used for Ollama (local server)
            model: Ollama model to use (e.g., 'llama2', 'llama3', 'mistral', 'codellama')
            **kwargs: Additional configuration including base_url
        """
        # Ollama doesn't require an API key, but we maintain the interface
        # If no model specified, default to empty string for user to set later
        super().__init__(api_key or "local", model or "", **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        self.timeout = kwargs.get("timeout", 60.0)  # Ollama can be slower

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to Ollama server.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Ollama's reply
        """
        try:
            # Check if server is available first
            await self._check_server_availability()

            # Prepare messages for Ollama API
            messages = self._prepare_messages(message, context)

            # Make API request to Ollama
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        },
                    },
                    timeout=self.timeout,
                )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise NetworkError(
                f"Request to Ollama server timed out after {self.timeout}s. "
                "Local inference can take longer than cloud APIs.",
                provider="ollama",
            )
        except httpx.ConnectError:
            raise NetworkError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Make sure Ollama is running with 'ollama serve'.",
                provider="ollama",
            )
        except httpx.RequestError as e:
            raise NetworkError(
                f"Network error connecting to Ollama: {e}", provider="ollama"
            )
        except (NetworkError, ModelNotFoundError, ProviderError):
            # Re-raise our custom errors without wrapping them
            raise
        except Exception as e:
            raise ProviderError(f"Unexpected error with Ollama: {e}", provider="ollama")

    async def validate_credentials(self) -> bool:
        """
        Validate Ollama server connection.

        Since Ollama doesn't use API keys, this checks if the server
        is running and the model is available.

        Returns:
            True if server is accessible and model is available
        """
        try:
            # Check if server is running
            await self._check_server_availability()

            # Check if the model is available
            available_models = await self._get_server_models()
            model_names = [model.get("name", "") for model in available_models]

            return self.model in model_names

        except Exception:
            return False

    async def _check_server_availability(self) -> None:
        """
        Check if Ollama server is running and accessible.

        Raises:
            NetworkError: If server is not accessible
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=5.0)

                if response.status_code != 200:
                    raise NetworkError(
                        f"Ollama server returned status {response.status_code}",
                        provider="ollama",
                    )

        except httpx.ConnectError:
            raise NetworkError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Make sure Ollama is running with 'ollama serve'.",
                provider="ollama",
            )
        except httpx.TimeoutException:
            raise NetworkError(
                "Ollama server is not responding. Check if it's running.",
                provider="ollama",
            )

    async def _get_server_models(self) -> List[Dict]:
        """
        Get list of models from Ollama server.

        Returns:
            List of model dictionaries from Ollama API

        Raises:
            NetworkError: If unable to fetch models
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=10.0)

                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
                else:
                    raise NetworkError(
                        f"Failed to fetch models from Ollama server: HTTP {response.status_code}",
                        provider="ollama",
                    )

        except httpx.RequestError as e:
            raise NetworkError(
                f"Error fetching models from Ollama: {e}", provider="ollama"
            )

    def _prepare_messages(
        self, message: str, context: ChatContext
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for Ollama API format.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            List of messages formatted for Ollama API
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
        Handle Ollama API response with comprehensive error handling.

        Args:
            response: HTTP response from Ollama API

        Returns:
            ChatResponse object

        Raises:
            Various provider errors based on response status
        """
        if response.status_code == 200:
            try:
                data = response.json()

                # Check for errors in successful response
                if "error" in data:
                    error_msg = data["error"]
                    if "not found" in error_msg.lower():
                        raise ModelNotFoundError(
                            f"Model '{self.model}' not found. Use 'ollama pull {self.model}' to download it.",
                            provider="ollama",
                            model_name=self.model,
                        )
                    else:
                        raise ProviderError(
                            f"Ollama error: {error_msg}", provider="ollama"
                        )

                # Ollama response format
                message = data.get("message", {})
                content = message.get("content", "")

                if not content:
                    # Check if model is still loading
                    if data.get("done", True) is False:
                        raise ProviderError(
                            f"Model '{self.model}' is still loading. Please wait and try again.",
                            provider="ollama",
                        )
                    else:
                        raise ProviderError(
                            "Empty response from Ollama", provider="ollama"
                        )

                # Check for completion status
                done = data.get("done", True)
                if not done:
                    raise ProviderError(
                        "Incomplete response from Ollama. The model may be overloaded.",
                        provider="ollama",
                    )

                # Ollama doesn't provide detailed token usage, estimate it
                estimated_tokens = len(content.split()) * 1.3  # Rough estimate

                return ChatResponse(
                    content=content,
                    model_used=self.model,
                    tokens_used=int(estimated_tokens),
                    timestamp=datetime.now(),
                )

            except (KeyError, ValueError) as e:
                raise ProviderError(
                    f"Invalid response format from Ollama: {e}",
                    provider="ollama",
                )

        elif response.status_code == 400:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", "Bad request")

                if "model" in error_msg.lower() and "not found" in error_msg.lower():
                    raise ModelNotFoundError(
                        f"Model '{self.model}' not found. Use 'ollama pull {self.model}' to download it.",
                        provider="ollama",
                        model_name=self.model,
                    )
                elif (
                    "invalid" in error_msg.lower() and "parameter" in error_msg.lower()
                ):
                    raise ProviderConfigurationError(
                        f"Invalid parameter: {error_msg}",
                        provider="ollama",
                        suggested_fix="Check model parameters and options",
                    )
                elif (
                    "context length" in error_msg.lower()
                    or "token" in error_msg.lower()
                ):
                    raise ProviderError(
                        f"Context too long: {error_msg}. Try reducing conversation history.",
                        provider="ollama",
                    )
                else:
                    raise ProviderError(
                        f"Ollama API error: {error_msg}", provider="ollama"
                    )
            except (ValueError, KeyError):
                raise ProviderError("Bad request to Ollama API", provider="ollama")

        elif response.status_code == 404:
            # Model not found
            try:
                error_data = response.json()
                error_msg = error_data.get("error", f"Model '{self.model}' not found")
            except:
                error_msg = f"Model '{self.model}' not found"

            raise ModelNotFoundError(
                f"{error_msg}. Use 'ollama pull {self.model}' to download it.",
                provider="ollama",
                model_name=self.model,
            )

        elif response.status_code == 500:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", "Internal server error")

                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    raise ProviderError(
                        "Ollama server out of memory. Try using a smaller model or restart Ollama.",
                        provider="ollama",
                    )
                elif "model" in error_msg.lower() and "loading" in error_msg.lower():
                    raise ProviderError(
                        f"Model '{self.model}' failed to load. Check if model is corrupted.",
                        provider="ollama",
                    )
                else:
                    raise ProviderError(f"Ollama API error: {error_msg}")
            except (ValueError, KeyError):
                raise ProviderError("Ollama API error: HTTP 500")

        elif response.status_code == 503:
            raise ProviderUnavailableError(
                "Ollama server is overloaded or starting up. Please wait and try again.",
                provider="ollama",
                estimated_recovery="1-2 minutes",
            )

        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", "Unknown error")
            except:
                error_msg = f"HTTP {response.status_code}"

            if response.status_code >= 500:
                raise ProviderUnavailableError(
                    f"Ollama server error: {error_msg}",
                    provider="ollama",
                    estimated_recovery="a few minutes",
                )
            else:
                raise ProviderError(f"Ollama API error: {error_msg}", provider="ollama")

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Ollama model.

        Note: Ollama model info is limited compared to cloud providers.
        """
        return ModelInfo(
            name=self.model,
            provider="ollama",
            description=f"Ollama model {self.model} (local inference)",
            max_tokens=self.max_tokens,
            cost_per_token=0.0,  # Local models are free
            available=True,  # Assume available if configured
            supports_tools=False,  # Ollama doesn't support tool calling yet
            supports_multimodal=self._model_supports_multimodal(),
            latest_version=False,  # Can't determine without version info
        )

    def get_available_models(self) -> List[ModelInfo]:
        """
        Get list of available Ollama models.

        This method returns models that are currently available on the
        local Ollama server. It requires the server to be running.
        """
        try:
            # This is a synchronous method but we need async call
            # We'll return a basic list and let the async version handle server queries
            return [
                ModelInfo(
                    name=self.model,
                    provider="ollama",
                    description=f"Ollama model {self.model} (local inference)",
                    max_tokens=self.max_tokens,
                    cost_per_token=0.0,
                    available=True,
                    supports_tools=False,
                    supports_multimodal=self._model_supports_multimodal(),
                )
            ]
        except Exception:
            # Return empty list if we can't connect to server
            return []

    async def get_available_models_async(self) -> List[ModelInfo]:
        """
        Async version to get available models from Ollama server.

        Returns:
            List of ModelInfo objects for models available on the server
        """
        try:
            server_models = await self._get_server_models()
            model_infos = []

            for model_data in server_models:
                model_name = model_data.get("name", "")
                if not model_name:
                    continue

                # Extract size info if available
                size_info = model_data.get("size", 0)
                size_gb = round(size_info / (1024**3), 1) if size_info else 0

                description = f"Ollama model {model_name} (local inference"
                if size_gb > 0:
                    description += f", {size_gb}GB"
                description += ")"

                model_infos.append(
                    ModelInfo(
                        name=model_name,
                        provider="ollama",
                        description=description,
                        max_tokens=4096,  # Default, varies by model
                        cost_per_token=0.0,  # Local models are free
                        available=True,
                        supports_tools=False,
                        supports_multimodal=self._model_supports_multimodal(model_name),
                    )
                )

            return model_infos

        except Exception:
            # Return current model if we can't fetch from server
            return self.get_available_models()

    def _model_supports_multimodal(self, model_name: str = None) -> bool:
        """
        Check if a model supports multimodal inputs.

        Args:
            model_name: Model name to check, defaults to current model

        Returns:
            True if model supports multimodal inputs
        """
        check_model = model_name or self.model

        # Known multimodal Ollama models
        multimodal_models = [
            "llava",
            "bakllava",
            "moondream",
            "llava-llama3",
            "llava-phi3",
        ]

        return any(mm_model in check_model.lower() for mm_model in multimodal_models)

    def supports_tools(self) -> bool:
        """
        Check if Ollama provider supports tool calling.

        Returns:
            False - Ollama doesn't currently support tool calling
        """
        return False

    def supports_multimodal(self) -> bool:
        """
        Check if current Ollama model supports multimodal inputs.

        Returns:
            True if current model supports multimodal inputs
        """
        return self._model_supports_multimodal()

    def supports_streaming(self) -> bool:
        """
        Check if Ollama provider supports streaming responses.

        Returns:
            True - Ollama supports streaming
        """
        return True
