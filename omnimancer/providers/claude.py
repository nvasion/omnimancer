"""
Claude provider implementation for Omnimancer.

This module provides the Claude AI provider implementation using Anthropic's API.
"""

import json as json_module
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Union

import certifi
import httpx

from ..core.models import (
    ChatContext,
    ChatResponse,
    ModelInfo,
    StreamEvent,
    StreamEventType,
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
from .cache_tokens import prompt_cache_enabled


class ClaudeProvider(BaseProvider):
    """
    Claude AI provider implementation using Anthropic's API.
    """

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str, model: str = "", **kwargs: Any) -> None:
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key or OAuth bearer token
            model: Claude model to use
                (e.g., 'claude-sonnet-4-20250514')
            **kwargs: Additional configuration (auth_type="bearer" for OAuth)
        """
        super().__init__(api_key, model or "claude-sonnet-4-6", **kwargs)
        # Allow overriding the API endpoint (proxies, gateways, Bedrock-compatible)
        self.base_url = (kwargs.get("base_url") or self.BASE_URL).rstrip("/")
        self.max_tokens = kwargs.get("max_tokens") or 4096
        self.temperature = kwargs.get("temperature") or 0.7
        self.auth_type = kwargs.get("auth_type", "api_key")

    def _is_subscription_token(self) -> bool:
        return self.auth_type == "bearer"  # type: ignore[no-any-return]

    @staticmethod
    def _prompt_cache_enabled() -> bool:
        return prompt_cache_enabled()

    def _apply_cache_control(self, request_body: Dict[str, Any]) -> None:
        """Add prompt-cache breakpoints to the outgoing request.

        The agent loop resends an identical prefix (tools, agent prompt,
        transcript) on every iteration. A breakpoint on the last tool and on
        the last message lets the API serve that prefix from cache instead of
        re-billing it each call. Prefixes below the model's cacheable minimum
        are silently uncached — the markers are harmless there.
        """
        if not self._prompt_cache_enabled():
            return
        marker = {"type": "ephemeral"}
        tools = request_body.get("tools")
        if tools:
            tools[-1]["cache_control"] = marker
        messages = request_body.get("messages")
        if not messages:
            return
        content = messages[-1].get("content")
        if isinstance(content, str) and content:
            messages[-1]["content"] = [
                {"type": "text", "text": content, "cache_control": marker}
            ]
        elif isinstance(content, list) and content:
            content[-1]["cache_control"] = marker

    def _is_haiku(self) -> bool:
        return "haiku" in self.model.lower()

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self._is_subscription_token():
            headers["Authorization"] = f"Bearer {self.api_key}"
            betas = ["oauth-2025-04-20"]
            if not self._is_haiku():
                betas.insert(0, "claude-code-20250219")
            headers["anthropic-beta"] = ",".join(betas)
        else:
            headers["x-api-key"] = self.api_key
        return headers

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
        request_body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        self._apply_cache_control(request_body)

        # Try with SSL verification first, then fall back if needed
        ssl_options: List[Union[bool, str]] = [True, certifi.where(), False]
        for ssl_verify in ssl_options:
            try:
                async with httpx.AsyncClient(verify=ssl_verify) as client:
                    response = await client.post(
                        f"{self.base_url}/messages",
                        headers=self._build_headers(),
                        json=request_body,
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
        ssl_options: List[Union[bool, str]] = [True, certifi.where(), False]
        for ssl_verify in ssl_options:
            try:
                async with httpx.AsyncClient(verify=ssl_verify) as client:
                    response = await client.post(
                        f"{self.base_url}/messages",
                        headers=self._build_headers(),
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
                        # Other errors (like invalid model),
                        # API key is probably valid
                        return True
                    except Exception:
                        # Can't parse error, assume valid
                        # if we got a response
                        return True
                else:
                    # Other status codes - if we got a response,
                    # API key is probably valid
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
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    stop_reason=data.get("stop_reason", "end_turn"),
                    cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                    cache_creation_input_tokens=usage.get(
                        "cache_creation_input_tokens"
                    ),
                )
            else:
                raise ProviderError("Empty response from Claude API")

        elif response.status_code == 401:
            raise AuthenticationError("Invalid Claude API key")
        elif response.status_code == 429:
            if self._is_subscription_429(response):
                raise ProviderError(
                    "Claude subscription tokens only "
                    "support Haiku for direct API "
                    f"access. Model '{self.model}' "
                    "requires an Anthropic API key. "
                    "Set ANTHROPIC_API_KEY or configure"
                    " an API key in Omnimancer."
                )
            raise RateLimitError(
                "Claude API rate limit exceeded",
                retry_after=self._parse_retry_after(response),
            )
        elif response.status_code == 529:
            # Overloaded — transient like a 429; the engine's backoff applies.
            raise RateLimitError(
                "Claude API overloaded (529)",
                retry_after=self._parse_retry_after(response),
            )
        elif response.status_code == 404:
            raise ModelNotFoundError(f"Claude model '{self.model}' not found")
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except Exception:
                error_msg = f"HTTP {response.status_code}"

            raise ProviderError(f"Claude API error: {error_msg}")

    async def send_message_with_tools(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> ChatResponse:
        messages = self._prepare_messages(message, context)
        tools = self._convert_tools_to_claude_format(available_tools)

        request_body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if tools:
            request_body["tools"] = tools
        self._apply_cache_control(request_body)

        ssl_options: List[Union[bool, str]] = [True, certifi.where(), False]
        for ssl_verify in ssl_options:
            try:
                async with httpx.AsyncClient(verify=ssl_verify) as client:
                    response = await client.post(
                        f"{self.base_url}/messages",
                        headers=self._build_headers(),
                        json=request_body,
                        timeout=60.0,
                    )

                return self._handle_response_with_tools(response)

            except httpx.ConnectError as e:
                if "SSL" in str(e) or "certificate" in str(e):
                    continue
                raise NetworkError(f"Connection error: {e}")
            except httpx.TimeoutException:
                raise NetworkError("Request to Claude API timed out")
            except httpx.RequestError as e:
                if "SSL" not in str(e) and "certificate" not in str(e):
                    raise NetworkError(f"Network error: {e}")
                continue
            except (
                AuthenticationError,
                RateLimitError,
                ModelNotFoundError,
                ProviderError,
            ):
                raise
            except Exception as e:
                raise ProviderError(f"Unexpected error: {e}")

        raise NetworkError("Failed to establish SSL connection to Claude API")

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> Union[int, None]:
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return int(float(raw))
        except ValueError:
            return None

    def _is_subscription_429(self, response: httpx.Response) -> bool:
        if not self._is_subscription_token() or self._is_haiku():
            return False
        try:
            data = response.json()
            msg = data.get("error", {}).get("message", "")
            has_rate_headers = any(
                k.startswith("anthropic-ratelimit-") for k in response.headers
            )
            return msg == "Error" and not has_rate_headers
        except Exception:
            return False

    def _convert_tools_to_claude_format(
        self, tools: List[ToolDefinition]
    ) -> List[Dict]:
        if not tools:
            return []

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _parse_response_content(self, content_blocks: List[Dict]) -> tuple:
        text_parts = []
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        return "".join(text_parts), tool_calls

    def _handle_response_with_tools(self, response: httpx.Response) -> ChatResponse:
        if response.status_code == 200:
            data = response.json()
            content_blocks = data.get("content", [])
            text_content, tool_calls = self._parse_response_content(content_blocks)
            usage = data.get("usage", {})

            return ChatResponse(
                content=text_content,
                model_used=self.model,
                tokens_used=usage.get("output_tokens", 0),
                timestamp=datetime.now(),
                tool_calls=tool_calls if tool_calls else None,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                stop_reason=data.get("stop_reason", "end_turn"),
                cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            )

        return self._handle_response(response)

    def supports_streaming(self) -> bool:
        return True

    async def send_message_stream(
        self, message: str, context: ChatContext
    ) -> AsyncIterator[StreamEvent]:
        messages = self._prepare_messages(message, context)
        request_body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "stream": True,
        }
        self._apply_cache_control(request_body)
        async for event in self._stream_request(request_body):
            yield event

    async def send_message_with_tools_stream(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> AsyncIterator[StreamEvent]:
        messages = self._prepare_messages(message, context)
        tools = self._convert_tools_to_claude_format(available_tools)
        request_body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "stream": True,
        }
        if tools:
            request_body["tools"] = tools
        self._apply_cache_control(request_body)
        async for event in self._stream_request(request_body):
            yield event

    async def _stream_request(self, request_body: dict) -> AsyncIterator[StreamEvent]:
        ssl_options: List[Union[bool, str]] = [True, certifi.where(), False]
        for ssl_verify in ssl_options:
            try:
                async with httpx.AsyncClient(verify=ssl_verify) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/messages",
                        headers=self._build_headers(),
                        json=request_body,
                        timeout=120.0,
                    ) as response:
                        if response.status_code != 200:
                            await response.aread()
                            self._handle_response(response)
                            return

                        async for event in self._parse_sse_stream(response):
                            yield event
                        return

            except httpx.ConnectError as e:
                if "SSL" in str(e) or "certificate" in str(e):
                    continue
                raise NetworkError(f"Connection error: {e}")
            except httpx.TimeoutException:
                raise NetworkError("Request to Claude API timed out")
            except httpx.RequestError as e:
                if "SSL" not in str(e) and "certificate" not in str(e):
                    raise NetworkError(f"Network error: {e}")
                continue
            except (
                AuthenticationError,
                RateLimitError,
                ModelNotFoundError,
                ProviderError,
            ):
                raise

        raise NetworkError("Failed to establish SSL connection " "to Claude API")

    async def _parse_sse_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[StreamEvent]:
        accumulated_text = ""
        current_tool_name = ""
        current_tool_id = ""
        current_tool_json = ""
        input_tokens = 0
        output_tokens = 0
        stop_reason = None
        model = self.model
        tool_calls = []

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                data = json_module.loads(data_str)
            except json_module.JSONDecodeError:
                continue

            event_type = data.get("type")

            if event_type == "message_start":
                msg = data.get("message", {})
                model = msg.get("model", self.model)
                usage = msg.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                yield StreamEvent(type=StreamEventType.MESSAGE_START, model=model)

            elif event_type == "content_block_start":
                block = data.get("content_block", {})
                if block.get("type") == "tool_use":
                    current_tool_name = block.get("name", "")
                    current_tool_id = block.get("id", "")
                    current_tool_json = ""
                    yield StreamEvent(
                        type=StreamEventType.TOOL_USE_START,
                        tool_name=current_tool_name,
                        tool_id=current_tool_id,
                    )

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    accumulated_text += text
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=text)
                elif delta.get("type") == "input_json_delta":
                    partial = delta.get("partial_json", "")
                    current_tool_json += partial
                    yield StreamEvent(
                        type=StreamEventType.TOOL_USE_DELTA, partial_json=partial
                    )

            elif event_type == "content_block_stop":
                if current_tool_name:
                    try:
                        args = (
                            json_module.loads(current_tool_json)
                            if current_tool_json
                            else {}
                        )
                    except json_module.JSONDecodeError:
                        args = {}
                    tool_calls.append(
                        ToolCall(
                            name=current_tool_name,
                            arguments=args,
                        )
                    )
                    yield StreamEvent(type=StreamEventType.TOOL_USE_END)
                    current_tool_name = ""
                    current_tool_id = ""
                    current_tool_json = ""

            elif event_type == "message_delta":
                delta = data.get("delta", {})
                stop_reason = delta.get("stop_reason")
                usage = data.get("usage", {})
                output_tokens = usage.get("output_tokens", output_tokens)

        final_response = ChatResponse(
            content=accumulated_text,
            model_used=model,
            tokens_used=output_tokens,
            timestamp=datetime.now(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason or "end_turn",
            tool_calls=tool_calls if tool_calls else None,
        )
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            response=final_response,
        )

    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current Claude model.
        """
        model_configs = {
            "claude-sonnet-4-6": {
                "description": "Claude Sonnet 4.6 - Latest balanced model",
                "max_tokens": 200000,
                "cost_per_token": 0.000003,
            },
            "claude-opus-4-6": {
                "description": "Claude Opus 4.6 - Most powerful for complex tasks",
                "max_tokens": 200000,
                "cost_per_token": 0.000015,
            },
            "claude-haiku-4-5-20251001": {
                "description": "Claude Haiku 4.5 - Fast and efficient",
                "max_tokens": 200000,
                "cost_per_token": 0.00000025,
            },
            "claude-sonnet-4-20250514": {
                "description": "Claude Sonnet 4 - Previous generation",
                "max_tokens": 200000,
                "cost_per_token": 0.000003,
            },
        }

        config: Dict[str, Any] = model_configs.get(
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
            latest_version=self.model == "claude-sonnet-4-6",
        )

    def _get_static_models(self) -> List[ModelInfo]:  # type: ignore[override]
        """
        Get static list of available Claude models.
        """
        return [
            ModelInfo(
                name="claude-sonnet-4-6",
                provider="claude",
                description="Claude Sonnet 4.6 - Latest balanced model",
                max_tokens=200000,
                cost_per_token=0.000003,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
            ),
            ModelInfo(
                name="claude-opus-4-6",
                provider="claude",
                description="Claude Opus 4.6 - Most powerful for complex tasks",
                max_tokens=200000,
                cost_per_token=0.000015,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
            ModelInfo(
                name="claude-haiku-4-5-20251001",
                provider="claude",
                description="Claude Haiku 4.5 - Fast and efficient",
                max_tokens=200000,
                cost_per_token=0.00000025,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
        ]

    async def fetch_live_models(self) -> List[ModelInfo]:  # type: ignore[override]
        """
        Fetch live model list from Anthropic API.

        Note: Anthropic doesn't have a public models list endpoint,
        so we return an enhanced static list with current pricing.

        Returns:
            List of ModelInfo objects for Claude models
        """
        return [
            ModelInfo(
                name="claude-sonnet-4-6",
                provider="claude",
                description="Claude Sonnet 4.6 - Latest balanced model",
                max_tokens=200000,
                cost_per_token=0.000003,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
            ),
            ModelInfo(
                name="claude-opus-4-6",
                provider="claude",
                description="Claude Opus 4.6 - Most powerful for complex tasks",
                max_tokens=200000,
                cost_per_token=0.000015,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
            ),
            ModelInfo(
                name="claude-haiku-4-5-20251001",
                provider="claude",
                description="Claude Haiku 4.5 - Fast and efficient",
                max_tokens=200000,
                cost_per_token=0.00000025,
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
