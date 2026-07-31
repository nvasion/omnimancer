"""
OpenAI provider implementation for Omnimancer.

This module provides the OpenAI API provider implementation using OpenAI's API.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

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

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI API provider implementation using OpenAI's API.
    """

    BASE_URL = "https://api.openai.com/v1"

    # Human-readable provider name used in error messages. Subclasses for
    # OpenAI-compatible services (e.g. DigitalOcean) override this so errors
    # name the service the user actually configured.
    PROVIDER_LABEL = "OpenAI"

    # Patterns used to recover the model context window and prompt size from
    # context-length errors returned by OpenAI-compatible servers (OpenAI,
    # vLLM, DigitalOcean inference, etc.).
    _CONTEXT_LIMIT_RE = re.compile(r"maximum context length is (\d+)", re.IGNORECASE)
    _INPUT_TOKENS_RE = re.compile(
        r"(\d+)\s+(?:input tokens|in the (?:messages|prompt))", re.IGNORECASE
    )
    # Tokens left as headroom when refitting max_tokens to the context window.
    # The server's reported input size is a lower-bound estimate that can drift
    # by a few dozen tokens between requests, so keep a generous margin.
    _CONTEXT_FIT_BUFFER = 256
    # How many times to refit max_tokens against successive overflow errors.
    _CONTEXT_FIT_MAX_RETRIES = 3

    def __init__(self, api_key: str, model: str = "", **kwargs: Any) -> None:
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (e.g., 'gpt-4', 'gpt-4o', 'gpt-3.5-turbo')
            **kwargs: Additional configuration. Supports ``request_timeout``
                (seconds) for chat completions; the ``OMNIMANCER_REQUEST_TIMEOUT``
                environment variable is used when the kwarg is not given.
        """
        super().__init__(api_key, model or "gpt-4", **kwargs)
        # Allow overriding the API endpoint for OpenAI-compatible services
        self.base_url = (kwargs.get("base_url") or self.BASE_URL).rstrip("/")
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        # ProviderConfig's field is `timeout`; `request_timeout` is kept as
        # the historical kwarg and wins when both are given.
        self.request_timeout = self._resolve_request_timeout(
            kwargs.get("request_timeout") or kwargs.get("timeout")
        )

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to OpenAI API.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with OpenAI's reply
        """
        try:
            # Prepare messages for OpenAI API
            messages = self._prepare_messages(message, context)

            # Make API request (auto-fits max_tokens on context overflow)
            response = await self._post_chat(
                {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=self.request_timeout,
            )

            # Handle response
            return self._handle_response(response)

        except httpx.TimeoutException:
            raise self._timeout_network_error()
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except (
            AuthenticationError,
            RateLimitError,
            ModelNotFoundError,
            ProviderError,
        ):
            raise
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    async def validate_credentials(self) -> bool:
        """
        Validate OpenAI API credentials by making a test request.

        Returns:
            True if credentials are valid
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
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

    def supports_native_tool_history(self) -> bool:
        return True

    def _prepare_messages(
        self, message: str, context: ChatContext
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for OpenAI API format.

        Tool exchanges recorded with structured data are serialized in the
        native protocol (assistant.tool_calls + role:"tool" messages) —
        flattened text violates the chat template and makes models leak
        template tokens (<tool_call>, <function=...) as plain text.

        Args:
            message: Current user message ("" for a continuation request
                whose tool results are already in the context)
            context: Conversation context

        Returns:
            List of messages formatted for OpenAI API
        """
        import json

        messages: List[Dict[str, Any]] = []

        for msg in context.messages:
            tool_calls = getattr(msg, "tool_calls", None)
            tool_results = getattr(msg, "tool_results", None)
            if tool_calls:
                raw = getattr(msg, "raw_content", None)
                content = raw if raw is not None else msg.content
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": tc.id or f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments, default=str),
                                },
                            }
                            for i, tc in enumerate(tool_calls)
                        ],
                    }
                )
            elif tool_results:
                for record in tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": record.tool_call_id,
                            "content": record.content,
                        }
                    )
            else:
                messages.append({"role": msg.role.value, "content": msg.content})

        if message:
            messages.append({"role": "user", "content": message})

        return messages

    def _fit_max_tokens(
        self, response: httpx.Response, current_max: Optional[int]
    ) -> Optional[int]:
        """
        Inspect an error response for a context-length overflow and compute a
        ``max_tokens`` value that would fit the model's context window.

        Returns:
            A positive int that fits, ``0`` if the prompt alone already exceeds
            the window (no output budget possible), or ``None`` if the error is
            not a context-length error.
        """
        error_msg = self._extract_error_message(response) or ""

        limit_match = self._CONTEXT_LIMIT_RE.search(error_msg)
        input_match = self._INPUT_TOKENS_RE.search(error_msg)
        if not (limit_match and input_match):
            return None

        context_limit = int(limit_match.group(1))
        input_tokens = int(input_match.group(1))
        return max(context_limit - input_tokens - self._CONTEXT_FIT_BUFFER, 0)

    def _build_headers(self) -> Dict[str, str]:
        """Request headers; Authorization only when a key is configured.

        Keyless endpoints (self-hosted vLLM, local proxies) reject or ignore
        an empty ``Bearer `` header, so it is omitted entirely.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _timeout_network_error(self) -> NetworkError:
        """The NetworkError raised when a chat completion times out."""
        return NetworkError(f"Request to {self.PROVIDER_LABEL} API timed out")

    def _require_api_key(self) -> None:
        """Fail fast with an actionable error when no API key is configured.

        Without this guard an empty key is sent as the header ``Bearer `` and
        httpx raises a cryptic ``Illegal header value`` error that gives the user
        no idea the real problem is a missing key (common for OpenAI-compatible
        providers like DigitalOcean whose key lives only in an env var).

        Configs may opt out explicitly with ``auth_type: "none"`` (keyless
        self-hosted endpoints).
        """
        if self.config.get("auth_type") == "none":
            return
        if self.api_key and self.api_key.strip():
            return

        provider = self.get_provider_name()
        # Local import avoids any import cycle between providers and core.
        from ..core.env_loader import ENV_VAR_MAPPING, _omnimancer_env_prefix

        env_var = ENV_VAR_MAPPING.get(provider)
        hint = f"Set {env_var}" if env_var else "Configure an API key"
        hint += f" (or {_omnimancer_env_prefix(provider)}_API_KEY)"
        raise AuthenticationError(
            f"No API key configured for '{provider}'. {hint}, or add one with "
            f"'/config set-provider {provider}'.",
            provider=provider,
        )

    async def _post_once(self, body: Dict[str, Any], timeout: float) -> httpx.Response:
        """Single POST to the chat completions endpoint."""
        async with httpx.AsyncClient() as client:
            return await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._build_headers(),
                json=body,
                timeout=timeout,
            )

    async def _post_chat(
        self, request_body: Dict[str, Any], timeout: float
    ) -> httpx.Response:
        """
        POST to the chat completions endpoint, retrying once with a reduced
        ``max_tokens`` if the server rejects the request because the prompt plus
        requested output exceeds the model's context window.

        This is common with large agent contexts on OpenAI-compatible backends
        (vLLM, DigitalOcean inference). Reducing the output budget to fit is far
        more useful than failing outright.
        """
        self._require_api_key()

        async def _do_post(body: Dict[str, Any]) -> httpx.Response:
            # Serverless backends (notably DigitalOcean inference) sporadically
            # time out on an otherwise-fine request; one retry absorbs those
            # transient stalls instead of failing the whole agent run.
            try:
                return await self._post_once(body, timeout)
            except httpx.TimeoutException:
                logger.warning(
                    "%s chat completion timed out after %.0fs — retrying once",
                    self.PROVIDER_LABEL,
                    timeout,
                )
                return await self._post_once(body, timeout)

        body = request_body
        response = await _do_post(body)

        # Refit max_tokens against successive overflow errors. The server's
        # reported input size can drift slightly between requests, so a single
        # refit may still land just over the limit; recompute from each error.
        for _ in range(self._CONTEXT_FIT_MAX_RETRIES):
            if response.status_code == 200:
                return response

            current_max = body.get("max_tokens")
            fitted = self._fit_max_tokens(response, current_max)
            if fitted is None:
                # Not a context-length error; let the handler report it.
                return response

            if fitted <= 0:
                raise ProviderError(
                    "The prompt is too large for this model's context window "
                    "even with no room left for a response. Start a new "
                    "conversation, reduce the input, or switch to a model with a "
                    "larger context window."
                )

            if current_max is None or fitted >= current_max:
                # Nothing to gain by retrying with the same-or-larger budget.
                return response

            logger.info(
                "Reducing max_tokens from %s to %s to fit the model context window",
                current_max,
                fitted,
            )
            body = {**body, "max_tokens": fitted}
            response = await _do_post(body)

        return response

    def _handle_response(self, response: httpx.Response) -> ChatResponse:
        """
        Handle OpenAI API response.

        Args:
            response: HTTP response from OpenAI API

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
                # Tool-call responses carry "content": null — coalesce to "".
                content = message.get("content") or ""
                usage = data.get("usage", {})

                return ChatResponse(
                    content=content,
                    model_used=self.model,
                    tokens_used=usage.get("total_tokens", 0),
                    timestamp=datetime.now(),
                )
            else:
                raise ProviderError("Empty response from OpenAI API")

        label = self.PROVIDER_LABEL
        detail = self._extract_error_message(response)
        suffix = f": {detail}" if detail else ""

        if response.status_code == 401:
            raise AuthenticationError(f"Invalid {label} API key{suffix}")
        elif response.status_code == 429:
            raise RateLimitError(f"{label} API rate limit exceeded{suffix}")
        elif response.status_code == 404:
            raise ModelNotFoundError(f"{label} model '{self.model}' not found{suffix}")
        else:
            error_msg = detail or f"HTTP {response.status_code}"
            raise ProviderError(f"{label} API error: {error_msg}")

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> Optional[str]:
        """Pull a human-readable message out of an error response body.

        OpenAI-compatible backends disagree on the error shape: OpenAI nests
        it under ``error.message``, DigitalOcean returns a top-level
        ``message`` (``{"id": ..., "message": ...}``), FastAPI gateways use
        ``detail``, and some servers return ``error`` as a plain string.
        Returns None when no message can be recovered (e.g. non-JSON body).
        """
        try:
            data = response.json()
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        error = data.get("error")
        if isinstance(error, dict):
            msg = error.get("message")
            if msg:
                return str(msg)
        elif isinstance(error, str) and error:
            return error

        for key in ("message", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
            if value:
                return str(value)
        return None

    async def send_message_with_tools(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> ChatResponse:
        if not self.supports_tools():
            return await self.send_message(message, context)

        try:
            messages = self._prepare_messages(message, context)
            tools = self._convert_tools_to_openai_format(available_tools)

            request_body = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if tools:
                request_body["tools"] = tools

            response = await self._post_chat(request_body, timeout=self.request_timeout)

            return self._handle_response_with_tools(response)

        except httpx.TimeoutException:
            raise self._timeout_network_error()
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")
        except (
            AuthenticationError,
            RateLimitError,
            ModelNotFoundError,
            ProviderError,
        ):
            raise
        except Exception as e:
            raise ProviderError(f"Unexpected error: {e}")

    def _convert_tools_to_openai_format(
        self, tools: List[ToolDefinition]
    ) -> List[Dict]:
        if not tools:
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _handle_response_with_tools(self, response: httpx.Response) -> ChatResponse:
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])

            if choices and len(choices) > 0:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""
                usage = data.get("usage", {})

                tool_calls = None
                raw_tool_calls = msg.get("tool_calls")
                if raw_tool_calls:
                    tool_calls = []
                    for i, tc in enumerate(raw_tool_calls):
                        func = tc.get("function", {})
                        tool_calls.append(
                            ToolCall(
                                # ToolCall normalizes JSON-string arguments
                                # (including double-encoded ones) on
                                # construction.
                                name=func.get("name", ""),
                                arguments=func.get("arguments", {}),
                                # Some OpenAI-compatible servers omit ids;
                                # synthesize one so results can pair to calls.
                                id=tc.get("id") or f"call_{i}",
                            )
                        )

                return ChatResponse(
                    content=content,
                    model_used=self.model,
                    tokens_used=usage.get("total_tokens", 0),
                    timestamp=datetime.now(),
                    tool_calls=tool_calls if tool_calls else None,
                )
            else:
                raise ProviderError("Empty response from OpenAI API")

        return self._handle_response(response)

    def supports_streaming(self) -> bool:
        """Real SSE streaming over /chat/completions (OpenAI dialect).

        Also inherited by DigitalOcean and openai-compatible providers;
        their PROVIDER_CAPABILITIES entries flip together with this one
        (enforced by tests/test_provider_capability_consistency.py).
        """
        return True

    async def send_message_stream(
        self, message: str, context: ChatContext
    ) -> AsyncIterator[StreamEvent]:
        messages = self._prepare_messages(message, context)
        request_body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        async for event in self._stream_request(request_body):
            yield event

    async def send_message_with_tools_stream(
        self,
        message: str,
        context: ChatContext,
        available_tools: List[ToolDefinition],
    ) -> AsyncIterator[StreamEvent]:
        if not self.supports_tools():
            async for event in self.send_message_stream(message, context):
                yield event
            return

        messages = self._prepare_messages(message, context)
        tools = self._convert_tools_to_openai_format(available_tools)
        request_body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            request_body["tools"] = tools
        async for event in self._stream_request(request_body):
            yield event

    async def _stream_request(
        self, request_body: Dict[str, Any]
    ) -> AsyncIterator[StreamEvent]:
        """POST with stream=true and yield parsed StreamEvents.

        ``stream_options.include_usage`` asks for a final usage chunk
        (supported by OpenAI and vLLM); servers that reject the field with
        a 400 get one retry without it.
        """
        self._require_api_key()

        body = dict(request_body)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

        tried_without_usage = False
        try:
            while True:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=self._build_headers(),
                        json=body,
                        timeout=self.request_timeout,
                    ) as response:
                        if response.status_code != 200:
                            await response.aread()
                            error_msg = self._extract_error_message(response) or ""
                            if (
                                not tried_without_usage
                                and response.status_code == 400
                                and "stream_options" in error_msg
                            ):
                                tried_without_usage = True
                                body.pop("stream_options", None)
                                continue
                            # Raises the appropriate typed error.
                            self._handle_response(response)
                            return
                        async for event in self._parse_openai_sse(response):
                            yield event
                        return
        except httpx.TimeoutException:
            raise self._timeout_network_error()
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {e}")

    async def _parse_openai_sse(
        self, response: httpx.Response
    ) -> AsyncIterator[StreamEvent]:
        """Parse chat.completion.chunk SSE lines into StreamEvents.

        Tool-call fragments are accumulated by their ``index`` field: the
        first fragment of an index carries the name (and usually the id),
        later fragments append to ``function.arguments``.
        """
        accumulated_text = ""
        model = self.model
        stop_reason: Optional[str] = None
        input_tokens = 0
        output_tokens = 0
        started = False

        tool_order: List[int] = []
        tool_names: Dict[int, str] = {}
        tool_ids: Dict[int, str] = {}
        tool_args: Dict[int, str] = {}
        open_index: Optional[int] = None

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if not started:
                model = chunk.get("model") or self.model
                started = True
                yield StreamEvent(type=StreamEventType.MESSAGE_START, model=model)

            usage = chunk.get("usage")
            if usage:
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                stop_reason = choice["finish_reason"]

            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                accumulated_text += text
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=text)

            for fragment in delta.get("tool_calls") or []:
                index = fragment.get("index", 0)
                func = fragment.get("function") or {}
                if index not in tool_names:
                    if open_index is not None:
                        yield StreamEvent(type=StreamEventType.TOOL_USE_END)
                    tool_order.append(index)
                    tool_names[index] = func.get("name", "")
                    # Some OpenAI-compatible servers omit ids; synthesize
                    # one so results can pair to calls (matches the
                    # non-stream handler's convention).
                    tool_ids[index] = fragment.get("id") or f"call_{index}"
                    tool_args[index] = ""
                    open_index = index
                    yield StreamEvent(
                        type=StreamEventType.TOOL_USE_START,
                        tool_name=tool_names[index],
                        tool_id=tool_ids[index],
                    )
                else:
                    if func.get("name") and not tool_names[index]:
                        tool_names[index] = func["name"]
                    if fragment.get("id"):
                        tool_ids[index] = fragment["id"]
                partial = func.get("arguments") or ""
                if partial:
                    tool_args[index] += partial
                    yield StreamEvent(
                        type=StreamEventType.TOOL_USE_DELTA,
                        partial_json=partial,
                    )

        if open_index is not None:
            yield StreamEvent(type=StreamEventType.TOOL_USE_END)

        tool_calls = []
        for index in tool_order:
            try:
                arguments = json.loads(tool_args[index]) if tool_args[index] else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    name=tool_names[index],
                    arguments=arguments,
                    id=tool_ids[index],
                )
            )

        final_response = ChatResponse(
            content=accumulated_text,
            model_used=model,
            tokens_used=input_tokens + output_tokens,
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
        Get information about the current OpenAI model.
        """
        model_configs = {
            "gpt-4": {
                "description": "GPT-4 - Most capable model",
                "max_tokens": 8192,
                "cost_per_token": 0.00003,
            },
            "gpt-4-turbo": {
                "description": "GPT-4 Turbo - Enhanced performance",
                "max_tokens": 128000,
                "cost_per_token": 0.00001,
            },
            "gpt-3.5-turbo": {
                "description": "GPT-3.5 Turbo - Fast and efficient",
                "max_tokens": 4096,
                "cost_per_token": 0.000002,
            },
            "gpt-3.5-turbo-16k": {
                "description": "GPT-3.5 Turbo with 16K context",
                "max_tokens": 16384,
                "cost_per_token": 0.000004,
            },
        }

        config: Dict[str, Any] = model_configs.get(
            self.model,
            {
                "description": f"OpenAI model {self.model}",
                "max_tokens": 4096,
                "cost_per_token": 0.00002,
            },
        )

        return ModelInfo(
            name=self.model,
            provider="openai",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_token=config["cost_per_token"],
            available=True,
            supports_tools=self.supports_tools(),
            supports_multimodal=self.supports_multimodal(),
            latest_version=self.model == "gpt-4-turbo",
        )

    def _get_static_models(self) -> List[ModelInfo]:  # type: ignore[override]
        """
        Get static list of available OpenAI models.
        """
        return [
            ModelInfo(
                name="gpt-4",
                provider="openai",
                description="GPT-4 - Most capable model",
                max_tokens=8192,
                cost_per_token=0.00003,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
            ),
            ModelInfo(
                name="gpt-4-turbo",
                provider="openai",
                description="GPT-4 Turbo - Enhanced performance",
                max_tokens=128000,
                cost_per_token=0.00001,
                available=True,
                supports_tools=True,
                supports_multimodal=True,
                latest_version=True,
            ),
            ModelInfo(
                name="gpt-3.5-turbo",
                provider="openai",
                description="GPT-3.5 Turbo - Fast and efficient",
                max_tokens=4096,
                cost_per_token=0.000002,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
            ),
            ModelInfo(
                name="gpt-3.5-turbo-16k",
                provider="openai",
                description="GPT-3.5 Turbo with 16K context",
                max_tokens=16384,
                cost_per_token=0.000004,
                available=True,
                supports_tools=True,
                supports_multimodal=False,
            ),
        ]

    def supports_tools(self) -> bool:
        """
        Check if OpenAI provider supports tool calling.

        Returns:
            True - OpenAI supports function calling/tools
        """
        return True

    def supports_multimodal(self) -> bool:
        """
        Check if OpenAI provider supports multimodal inputs.

        Returns:
            True for GPT-4 models that support vision, False for others
        """
        # GPT-4 models with vision support
        vision_models = ["gpt-4-vision-preview", "gpt-4-turbo", "gpt-4o"]
        return any(model in self.model for model in vision_models)

    async def fetch_live_models(self) -> List[ModelInfo]:  # type: ignore[override]
        """
        Fetch live model list from OpenAI API.

        Returns:
            List of ModelInfo objects from OpenAI API
        """
        try:
            headers = self._build_headers()

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models", headers=headers, timeout=30.0
                )
                response.raise_for_status()

                data = response.json()
                models = []

                # Filter and convert to ModelInfo objects
                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "")

                    # Filter for chat models (exclude fine-tuned and other model types)
                    if any(prefix in model_id for prefix in ["gpt-3.5", "gpt-4"]):
                        # Determine model capabilities
                        supports_tools = (
                            "gpt-3.5-turbo" in model_id or "gpt-4" in model_id
                        )
                        supports_multimodal = any(
                            vision in model_id
                            for vision in ["gpt-4-turbo", "gpt-4o", "vision"]
                        )

                        # Estimate max tokens based on model
                        max_tokens = 4096  # default
                        if "gpt-4-turbo" in model_id or "gpt-4o" in model_id:
                            max_tokens = 128000
                        elif "gpt-4" in model_id:
                            max_tokens = 8192
                        elif "gpt-3.5-turbo" in model_id:
                            max_tokens = 16384

                        models.append(
                            ModelInfo(
                                name=model_id,
                                provider="openai",
                                description=f"OpenAI {model_id}",
                                max_tokens=max_tokens,
                                cost_per_token=0.00001,  # Approximate
                                available=True,
                                supports_tools=supports_tools,
                                supports_multimodal=supports_multimodal,
                            )
                        )

                return models

        except Exception:
            # Fall back to static model list if API call fails
            return self.get_available_models()  # type: ignore[return-value]
