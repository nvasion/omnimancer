"""
Core engine for Omnimancer CLI.

This module provides the main engine class that coordinates between
providers, configuration, and chat management.

Version: 1.0.0
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from ..core.models import (
    ChatResponse,
    EnhancedModelInfo,
    ModelInfo,
    StreamEvent,
    StreamEventType,
    ToolDefinition,
    describe_tool_calls,
)
from ..providers.base import BaseProvider
from ..ui.progress_indicator import OperationType, get_progress_indicator
from ..utils.errors import ConfigurationError, RateLimitError
from .chat_manager import ChatManager
from .config_manager import ConfigManager
from .conversation_manager import ConversationManager
from .health_monitor import HealthMonitor
from .hooks import HookOutcome, HooksManager
from .provider_initializer import ProviderInitializer
from .provider_registry import ProviderRegistry
from .rate_limit_fallback import ApprovalCallback, RateLimitFallbackHandler
from .security.permission_rules import PermissionDecision, PermissionRuleEngine

logger = logging.getLogger(__name__)


class CoreEngine:
    """
    Core engine that coordinates all Omnimancer functionality.

    This class manages providers, configuration, chat sessions,
    and provides the main interface for the CLI.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the core engine.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.chat_manager = ChatManager()

        # Get storage path from config or use default
        config = config_manager.get_config()
        storage_path = getattr(config, "storage_path", "~/.omnimancer")
        self.conversation_manager = ConversationManager(storage_path)

        self.health_monitor = HealthMonitor()
        self.provider_initializer = ProviderInitializer()

        # Initialize provider registry
        self.provider_registry = ProviderRegistry()

        # Initialize MCP manager
        mcp_config = getattr(config, "mcp", None)
        if mcp_config:
            from ..mcp.manager import MCPManager

            self.mcp_manager = MCPManager(mcp_config)
        else:
            self.mcp_manager = None  # type: ignore[assignment]

        self.providers: Dict[str, BaseProvider] = {}
        self.current_provider: Optional[BaseProvider] = None
        self._initialized = False

        # Initialize agent engine for autonomous operations
        self.agent_engine = None

        # Rate-limit fallback handler — configured from Config.fallback on
        # every send, so runtime edits via /fallback take effect immediately.
        self._fallback_handler = RateLimitFallbackHandler()

    async def initialize_providers(self) -> None:
        """Initialize all configured providers."""
        try:
            config = self.config_manager.get_config()

            # Apply ephemeral environment-variable overrides (API keys,
            # base_url, model, default provider). Never persisted to disk.
            # Overrides must never prevent startup, so fall back on any error.
            try:
                from .env_loader import apply_env_overrides

                config = apply_env_overrides(config)
            except Exception as e:
                logger.warning(f"Skipping environment overrides: {e}")

            # Use the optimized provider initializer with
            # config_manager for API key decryption
            self.providers = await self.provider_initializer.initialize_providers(
                config.providers, self.config_manager
            )

            # Register providers with the provider registry for catalog management
            from ..providers.factory import ProviderFactory

            factory = ProviderFactory()
            available_providers = factory.get_available_providers()

            # Register all available providers (not just
            # configured ones) for catalog management
            for provider_name in available_providers:
                try:
                    # Register the provider name; the registry
                    # will handle class loading
                    self.provider_registry.register_provider(
                        provider_name,
                        None,  # type: ignore[arg-type]
                    )
                except Exception as e:
                    logger.warning(f"Failed to register provider {provider_name}: {e}")

            # Set default provider
            self._unavailable_default_provider = None
            if config.default_provider:
                if config.default_provider in self.providers:
                    self.current_provider = self.providers[config.default_provider]
                else:
                    # The configured provider failed to initialize (most often a
                    # missing/invalid API key). Do NOT silently fall back to a
                    # different provider — that would send the request with the
                    # wrong provider's credentials (e.g. an Anthropic key for a
                    # DigitalOcean request). Leave it unset so the user gets a
                    # clear error naming the provider.
                    self._unavailable_default_provider = config.default_provider
                    logger.error(
                        "Default provider '%s' is configured but failed to "
                        "initialize (missing or invalid API key?). Not falling "
                        "back to another provider.",
                        config.default_provider,
                    )
            elif self.providers:
                # No default configured — use first available provider.
                self.current_provider = next(iter(self.providers.values()))

            # Initialize agent engine after providers are ready
            self._initialize_agent_engine()

            self._initialized = True
            logger.info(f"Initialized {len(self.providers)} providers")

        except Exception as e:
            logger.error(f"Failed to initialize providers: {e}")
            raise ConfigurationError(f"Provider initialization failed: {e}")

    async def switch_model(
        self, provider_name: str, model_name: Optional[str] = None
    ) -> bool:
        """
        Switch to a different provider/model.

        Args:
            provider_name: Name of the provider to switch to
            model_name: Optional model name to use

        Returns:
            True if switch was successful, False otherwise
        """
        try:
            if provider_name not in self.providers:
                raise ConfigurationError(f"Provider '{provider_name}' is not available")

            provider = self.providers[provider_name]

            # Switch model if specified
            if model_name:
                # Validate model is available for this provider
                available_models = provider.get_available_models()
                model_names = [m.name for m in available_models]

                # Also check custom models for this provider
                custom_models = self.config_manager.get_custom_models()
                custom_model_names = [
                    m.name for m in custom_models if m.provider == provider_name
                ]

                # Combine both lists
                all_model_names = model_names + custom_model_names

                if model_name not in all_model_names:
                    raise ConfigurationError(
                        f"Model '{model_name}' not available"
                        f" for provider '{provider_name}'"
                    )

                provider.model = model_name

            # Switch to the provider
            self.current_provider = provider

            # Update chat manager with new model
            self.chat_manager.set_current_model(provider.model)

            logger.info(
                f"Switched to provider: {provider_name}, model: {provider.model}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False

    def _no_provider_error(self) -> str:
        """Build a clear error for when there is no usable current provider.

        Names the configured provider that failed to initialize (if any) so the
        user isn't left guessing — e.g. a DigitalOcean default whose API key is
        missing, rather than a generic "no provider" message.
        """
        unavailable = getattr(self, "_unavailable_default_provider", None)
        if unavailable:
            return (
                f"Provider '{unavailable}' is selected but failed to initialize "
                f"— most likely a missing or invalid API key. Configure it (e.g. "
                f"'/config set-provider {unavailable}' or the provider's API-key "
                f"environment variable) and try again."
            )
        return "No provider available. Please configure a provider first."

    # ------------------------------------------------------------------
    # Fallback wiring
    # ------------------------------------------------------------------

    def set_fallback_approval_callback(self, callback: ApprovalCallback) -> None:
        """Register the interactive approval callback (typically set by the CLI).

        The callback signature is::

            async def callback(current: str, next_: str, error: str) -> bool

        Return ``True`` to proceed with the switch, ``False`` to abort.
        """
        self._fallback_handler.set_approval_callback(callback)

    def configure_fallback(self) -> None:
        """Sync fallback settings from the current Config.

        Called automatically before every send so that runtime changes made
        via ``/fallback`` are picked up without restarting.
        """
        try:
            config = self.config_manager.get_config()
            fallback_cfg = getattr(config, "fallback", None)
            if fallback_cfg is not None:
                self._fallback_handler.update_from_config(fallback_cfg)
        except Exception as exc:
            logger.debug("configure_fallback skipped: %s", exc)

    async def _apply_rate_limit_fallback(
        self,
        error_str: str,
    ) -> Optional[str]:
        """Try to obtain approval for a provider switch.

        Returns the name of the approved next provider, or ``None`` if no
        fallback should be attempted (not configured, no candidate, or user
        declined).
        """
        if not self.current_provider:
            return None
        if not self._fallback_handler.should_fallback(error_str):
            return None

        current_name = self.current_provider.get_provider_name()
        available = list(self.providers.keys())
        next_name = self._fallback_handler.get_next_provider(current_name, available)

        if not next_name:
            logger.debug("Rate-limit fallback: no alternative provider available.")
            return None

        approved = await self._fallback_handler.request_approval(
            current_name, next_name, error_str
        )
        if not approved:
            return None

        return next_name

    async def _do_provider_switch(self, next_name: str) -> None:
        """Switch current_provider to *next_name* and update chat state."""
        self.current_provider = self.providers[next_name]
        self.chat_manager.set_current_model(self.current_provider.model)
        logger.info("Rate-limit fallback: switched to provider '%s'.", next_name)

    # ------------------------------------------------------------------

    async def _fire_hook(
        self,
        event: str,
        context: Optional[Dict[str, Any]] = None,
        match_target: str = "",
    ) -> HookOutcome:
        """Fire configured lifecycle hooks for ``event``.

        Reads the current config so hook changes take effect without restart.
        Never raises — a misbehaving hook can only veto via the returned
        outcome, never break message sending.
        """
        try:
            config = self.config_manager.get_config()
            hooks_cfg = getattr(config, "hooks", None)
            return await HooksManager(hooks_cfg).fire(event, context, match_target)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Hook firing for '%s' failed: %s", event, e)
            return HookOutcome(event=event)

    def _permission_decision(self, tool: str, target: str = "") -> PermissionDecision:
        """Evaluate config-driven permission rules for an operation.

        Reads the current config so rule edits take effect without restart.
        Returns DEFAULT (normal approval workflow) on any error.
        """
        try:
            config = self.config_manager.get_config()
            perms = getattr(config, "permissions", None)
            return PermissionRuleEngine(perms).evaluate(tool, target)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Permission rule evaluation failed: %s", e)
            return PermissionDecision.DEFAULT

    async def send_message(self, message: str) -> ChatResponse:
        """
        Send a message using the current provider.

        Args:
            message: Message to send

        Returns:
            Chat response from the provider
        """
        if not self.current_provider:
            return ChatResponse(
                content="",
                model_used="",
                tokens_used=0,
                error=self._no_provider_error(),
            )

        # Sync fallback config so runtime changes via /fallback take effect.
        self.configure_fallback()

        try:
            # Fire pre-send hooks; a blocking hook can veto the send.
            hook_ctx = {
                "message": message,
                "provider": self.current_provider.get_provider_name(),
                "model": self.current_provider.model,
            }
            outcome = await self._fire_hook(
                "pre_send_message", hook_ctx, match_target=message
            )
            if not outcome.allowed:
                return ChatResponse(
                    content="",
                    model_used="",
                    tokens_used=0,
                    error=f"Message blocked by {outcome.reason}.",
                )

            # Get progress indicator
            progress = get_progress_indicator()

            # Get current chat context
            if progress and progress.enabled:
                progress.start_operation(
                    "engine_context",
                    OperationType.ANALYZE,
                    "Getting chat context",
                )
            context = self.chat_manager.get_current_context()
            if progress and progress.enabled:
                progress.complete_operation("engine_context", "completed")

            # Send message to provider (catch RateLimitError explicitly so we
            # can offer a fallback before falling through to the generic handler)
            if progress and progress.enabled:
                progress.start_operation(
                    "engine_provider",
                    OperationType.NETWORK,
                    f"Sending to {self.current_provider.get_provider_name()}",
                )
            try:
                response = await self.current_provider.send_message(message, context)
            except RateLimitError as exc:
                # Convert to a response so we can run the same fallback path.
                response = ChatResponse(
                    content="",
                    model_used="",
                    tokens_used=0,
                    error=str(exc),
                )

            # ---- Rate-limit fallback ----------------------------------------
            if not response.is_success:
                error_str = response.error or ""
                next_name = await self._apply_rate_limit_fallback(error_str)
                if next_name:
                    await self._do_provider_switch(next_name)
                    if progress and progress.enabled:
                        progress.start_operation(
                            "engine_provider_fallback",
                            OperationType.NETWORK,
                            "Retrying with "
                            f"{self.current_provider.get_provider_name()}",
                        )
                    try:
                        fb_context = self.chat_manager.get_current_context()
                        response = await self.current_provider.send_message(
                            message, fb_context
                        )
                    except Exception as exc:
                        response = ChatResponse(
                            content="",
                            model_used="",
                            tokens_used=0,
                            error=f"Fallback provider failed: {exc}",
                        )
                    if progress and progress.enabled:
                        progress.complete_operation(
                            "engine_provider_fallback",
                            "completed" if response.is_success else "failed",
                        )
            # -----------------------------------------------------------------

            if progress and progress.enabled:
                progress.complete_operation(
                    "engine_provider",
                    "completed" if response.is_success else "failed",
                )

            # Always add user message to chat history
            self.chat_manager.add_user_message(message)

            # Only add assistant message if response was successful
            if response.is_success:
                if progress and progress.enabled:
                    progress.start_operation(
                        "engine_history",
                        OperationType.WRITE,
                        "Updating chat history",
                    )
                self.chat_manager.add_assistant_message(
                    response.content, self.current_provider.model
                )
                if progress and progress.enabled:
                    progress.complete_operation("engine_history", "completed")

                # Observe-only post-send hooks.
                await self._fire_hook(
                    "post_send_message",
                    {
                        "message": message,
                        "response": response.content,
                        "provider": self.current_provider.get_provider_name(),
                        "model": self.current_provider.model,
                    },
                    match_target=response.content,
                )

            return response

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return ChatResponse(
                content="",
                model_used="",
                tokens_used=0,
                error=f"Failed to send message: {str(e)}",
            )

    async def send_message_with_tools(
        self, message: str, tools: List[ToolDefinition]
    ) -> ChatResponse:
        if not self.current_provider:
            return ChatResponse(
                content="",
                model_used="",
                tokens_used=0,
                error=self._no_provider_error(),
            )

        # Sync fallback config so runtime changes via /fallback take effect.
        self.configure_fallback()

        try:
            outcome = await self._fire_hook(
                "pre_send_message",
                {
                    "message": message,
                    "provider": self.current_provider.get_provider_name(),
                    "model": self.current_provider.model,
                    "with_tools": True,
                },
                match_target=message,
            )
            if not outcome.allowed:
                return ChatResponse(
                    content="",
                    model_used="",
                    tokens_used=0,
                    error=f"Message blocked by {outcome.reason}.",
                )

            context = self.chat_manager.get_current_context()

            try:
                response = await self.current_provider.send_message_with_tools(
                    message, context, tools
                )
            except RateLimitError as exc:
                response = ChatResponse(
                    content="",
                    model_used="",
                    tokens_used=0,
                    error=str(exc),
                )

            # ---- Rate-limit fallback ----------------------------------------
            if not response.is_success:
                error_str = response.error or ""
                next_name = await self._apply_rate_limit_fallback(error_str)
                if next_name:
                    await self._do_provider_switch(next_name)
                    try:
                        fb_context = self.chat_manager.get_current_context()
                        response = await self.current_provider.send_message_with_tools(
                            message, fb_context, tools
                        )
                    except Exception as exc:
                        response = ChatResponse(
                            content="",
                            model_used="",
                            tokens_used=0,
                            error=f"Fallback provider failed: {exc}",
                        )
            # -----------------------------------------------------------------

            # A continuation request ("" after recorded tool results) adds
            # nothing the model needs to see as a user turn.
            if message:
                self.chat_manager.add_user_message(message)
            if response.is_success:
                # Record the tool calls alongside the text — tool results come
                # back as plain text, so without this the model can't see which
                # calls it already made and repeats them. The structured form
                # rides along for providers that replay history natively.
                recorded = response.content or ""
                calls_note = describe_tool_calls(response.tool_calls)
                if calls_note:
                    recorded = f"{recorded}\n{calls_note}".strip()
                self.chat_manager.add_assistant_message(
                    recorded,
                    self.current_provider.model,
                    tool_calls=response.tool_calls,
                    raw_content=response.content,
                )

            return response

        except Exception as e:
            logger.error(f"Failed to send message with tools: {e}")
            return ChatResponse(
                content="",
                model_used="",
                tokens_used=0,
                error=f"Failed to send message: {str(e)}",
            )

    async def send_message_stream(self, message: str) -> AsyncIterator[StreamEvent]:
        if not self.current_provider:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error="No provider available.",
            )
            return

        # Sync fallback config on every call.
        self.configure_fallback()

        context = self.chat_manager.get_current_context()
        rate_limit_error: Optional[str] = None

        async for event in self.current_provider.send_message_stream(message, context):
            if event.type == StreamEventType.ERROR:
                err = event.error or ""
                if self._fallback_handler.should_fallback(err):
                    # Hold the error; don't yield it yet — try fallback first.
                    rate_limit_error = err
                    break
            yield event

        if rate_limit_error:
            next_name = await self._apply_rate_limit_fallback(rate_limit_error)
            if next_name:
                await self._do_provider_switch(next_name)
                fb_context = self.chat_manager.get_current_context()
                async for event in self.current_provider.send_message_stream(
                    message, fb_context
                ):
                    yield event
            else:
                # No fallback available / user declined — surface original error.
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=rate_limit_error,
                )

    async def send_message_with_tools_stream(
        self, message: str, tools: List[ToolDefinition]
    ) -> AsyncIterator[StreamEvent]:
        if not self.current_provider:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error="No provider available.",
            )
            return

        # Sync fallback config on every call.
        self.configure_fallback()

        context = self.chat_manager.get_current_context()
        rate_limit_error: Optional[str] = None

        async for event in self.current_provider.send_message_with_tools_stream(
            message, context, tools
        ):
            if event.type == StreamEventType.ERROR:
                err = event.error or ""
                if self._fallback_handler.should_fallback(err):
                    rate_limit_error = err
                    break
            yield event

        if rate_limit_error:
            next_name = await self._apply_rate_limit_fallback(rate_limit_error)
            if next_name:
                await self._do_provider_switch(next_name)
                fb_context = self.chat_manager.get_current_context()
                async for event in self.current_provider.send_message_with_tools_stream(
                    message, fb_context, tools
                ):
                    yield event
            else:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error=rate_limit_error,
                )

    def provider_supports_tools(self) -> bool:
        if not self.current_provider:
            return False
        return self.current_provider.supports_tools()

    def provider_supports_native_tool_history(self) -> bool:
        if not self.current_provider:
            return False
        return bool(self.current_provider.supports_native_tool_history())

    def record_tool_results(self, content: str, records: list) -> None:
        """Record tool results as a user turn with structured pairing.

        `content` is the flattened text every provider can read; `records`
        (ToolResultRecord) lets native providers replay them as role:"tool"
        messages. The next request is then sent with an empty message.
        """
        self.chat_manager.add_user_message(content, tool_results=records)

    def get_available_models(self) -> List[ModelInfo]:
        """Get all available models from all providers."""
        all_models = []

        for provider in self.providers.values():
            try:
                models = provider.get_available_models()
                # Convert EnhancedModelInfo to ModelInfo if needed
                for model in models:
                    if isinstance(model, EnhancedModelInfo):
                        all_models.append(model.to_model_info())
                    else:
                        all_models.append(model)
            except Exception as e:
                provider_name = provider.get_provider_name()
                logger.warning(
                    f"Failed to get models from" f" provider {provider_name}: {e}"
                )

        # Custom models (added via /add-model or /switch) live in config,
        # not in any provider's static catalog.
        seen = {(m.provider, m.name) for m in all_models}
        try:
            for custom in self.config_manager.get_custom_models():
                if (custom.provider, custom.name) in seen:
                    continue
                if isinstance(custom, EnhancedModelInfo):
                    all_models.append(custom.to_model_info())
                else:
                    all_models.append(custom)
                seen.add((custom.provider, custom.name))
        except Exception as e:
            logger.warning(f"Failed to merge custom models: {e}")

        return all_models

    def get_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all models organized by provider (for CLI display)."""
        result = {}  # type: ignore[var-annotated]

        for provider_name, provider in self.providers.items():
            try:
                models = provider.get_available_models()
                result[provider_name] = []

                for model in models:
                    model_dict = {
                        "name": model.name,
                        "provider": model.provider,
                        "supports_tools": getattr(model, "supports_tools", False),
                        "supports_multimodal": getattr(
                            model, "supports_multimodal", False
                        ),
                        "available": getattr(model, "available", True),
                    }

                    # Add enhanced info if available
                    if isinstance(model, EnhancedModelInfo):
                        model_dict.update(
                            {
                                "swe_score": model.swe_score,
                                "cost_display": model.get_cost_display(),
                                "latest_version": model.latest_version,
                            }
                        )

                    result[provider_name].append(model_dict)

            except Exception as e:
                logger.warning(
                    f"Failed to get models from provider {provider_name}: {e}"
                )
                result[provider_name] = []

        return result

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration for display."""
        try:
            config = self.config_manager.get_config()
            return {
                "default_provider": config.default_provider,
                "providers": {
                    name: {
                        "model": provider_config.model,
                        "api_key": (
                            f"{provider_config.api_key[:8]}***"
                            if provider_config.api_key
                            else "Not set"
                        ),
                    }
                    for name, provider_config in config.providers.items()
                },
                "current_provider": (
                    self.current_provider.get_provider_name()
                    if self.current_provider
                    else None
                ),
                "current_model": (
                    self.current_provider.model if self.current_provider else None
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get current config: {e}")
            return {"error": str(e)}

    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current model."""
        if not self.current_provider:
            return None

        try:
            model_info = self.current_provider.get_model_info()
            return {
                "name": model_info.name,
                "provider": model_info.provider,
                "supports_tools": getattr(model_info, "supports_tools", False),
                "supports_multimodal": getattr(
                    model_info, "supports_multimodal", False
                ),
                "available": getattr(model_info, "available", True),
            }
        except Exception as e:
            logger.warning(f"Failed to get current model info: {e}")
            return None

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        try:
            context = self.chat_manager.get_current_context()
            return {
                "message_count": len(context.messages),
                "current_model": (
                    self.current_provider.model if self.current_provider else None
                ),
                "session_id": context.session_id,
            }
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return {"error": str(e)}

    async def validate_current_provider(self) -> bool:
        """Validate that the current provider is working."""
        if not self.current_provider:
            return False

        try:
            return await self.current_provider.validate_credentials()
        except Exception as e:
            logger.error(f"Provider validation failed: {e}")
            return False

    async def check_provider_health(
        self, provider_name: Optional[str] = None, force: bool = False
    ) -> Dict[str, Any]:
        """
        Check health status of providers using the optimized health monitor.

        Args:
            provider_name: Name of specific provider to check, or None for all
            force: Force check even if cached result is available

        Returns:
            Dictionary with health status information
        """
        try:
            config = self.config_manager.get_config()

            if provider_name:
                # Check specific provider
                if provider_name not in config.providers:
                    return {
                        provider_name: {
                            "status": "error",
                            "message": f"Provider {provider_name} not configured",
                            "available": False,
                            "credentials_valid": False,
                        }
                    }

                provider_config = config.providers[provider_name]
                status = await self.health_monitor.check_provider_health(
                    provider_name, provider_config, force=force
                )
                return {provider_name: status}
            else:
                # Check all providers
                return await self.health_monitor.check_all_providers_health(
                    config.providers, force=force
                )

        except Exception as e:
            logger.error(f"Error checking provider health: {e}")
            if provider_name:
                return {
                    provider_name: {
                        "status": "error",
                        "message": f"Health check failed: {str(e)}",
                        "available": False,
                        "credentials_valid": False,
                    }
                }
            else:
                return {"error": f"Health check failed: {str(e)}"}

    def save_conversation(self, name: str) -> str:
        """Save current conversation."""
        try:
            context = self.chat_manager.get_current_context()
            return self.conversation_manager.save_conversation(context, name)
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List saved conversations."""
        try:
            return self.conversation_manager.list_conversations()
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []

    def load_conversation(self, filename: str) -> bool:
        """Load a saved conversation."""
        try:
            context = self.conversation_manager.load_conversation(filename)
            self.chat_manager.current_context = context
            return True
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    async def initialize_mcp(self) -> None:
        """Initialize MCP (Model Context Protocol) servers."""
        try:
            if self.mcp_manager:
                await self.mcp_manager.initialize_servers()
                logger.info("MCP servers initialized successfully")
            else:
                logger.info(
                    "MCP manager not configured" " - skipping MCP initialization"
                )
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            raise

    async def shutdown_mcp(self) -> None:
        """Shutdown MCP servers gracefully."""
        try:
            if self.mcp_manager:
                await self.mcp_manager.shutdown()
                logger.info("MCP servers shutdown successfully")
            else:
                logger.info("MCP manager not configured - skipping MCP shutdown")
        except Exception as e:
            logger.error(f"Failed to shutdown MCP: {e}")
            # Don't raise during shutdown

    def _get_models_list(self) -> str:
        """Get a basic models list as fallback."""
        models = self.get_available_models()
        if not models:
            return "No models available."

        lines = []
        for model in models:
            lines.append(f"- {model.name} ({model.provider})")

        return "\n".join(lines)

    def _get_providers_list(self) -> str:
        """Get a formatted list of configured and available providers."""
        from ..providers.factory import ProviderFactory

        current_name = (
            self.current_provider.get_provider_name() if self.current_provider else None
        )

        lines = []
        if self.providers:
            lines.append("Configured:")
            for name in sorted(self.providers):
                provider = self.providers[name]
                marker = " (current)" if name == current_name else ""
                try:
                    model = getattr(provider, "model", None)
                except Exception:
                    model = None
                model_text = f" - model: {model}" if model else ""
                lines.append(f"  - {name}{marker}{model_text}")

        # Surface registered providers the user has not configured yet so they
        # remain discoverable (configure with /config set-provider <name>).
        available = sorted(
            name
            for name in ProviderFactory.get_available_providers()
            if name not in self.providers
        )
        if available:
            if lines:
                lines.append("")
            lines.append("Available (not configured):")
            lines.append("  " + ", ".join(available))

        return "\n".join(lines) if lines else "No providers available."

    async def get_available_tools(self) -> List[Any]:
        """Return the list of available MCP tools (empty if MCP is unavailable)."""
        if not self.mcp_manager or not getattr(self.mcp_manager, "initialized", False):
            return []

        try:
            return await self.mcp_manager.get_available_tools()
        except Exception as e:
            logger.warning(f"Failed to get available tools: {e}")
            return []

    async def _get_tools_list(self) -> str:
        """Get formatted list of available MCP tools."""
        if not self.mcp_manager or not self.mcp_manager.initialized:
            return "MCP is not initialized. No tools available."

        try:
            # Get available tools from MCP manager
            tools = await self.mcp_manager.get_available_tools()

            if not tools:
                return "No MCP tools available."

            # Group tools by server
            tools_by_server = {}  # type: ignore[var-annotated]
            for tool in tools:
                server_name = getattr(tool, "server_name", "Unknown")
                if server_name not in tools_by_server:
                    tools_by_server[server_name] = []
                tools_by_server[server_name].append(tool)

            # Format output
            lines = []
            lines.append("Available MCP Tools:")
            lines.append("=" * 50)

            for server_name, server_tools in tools_by_server.items():
                lines.append(f"\n📡 {server_name} ({len(server_tools)} tools)")
                lines.append("-" * 30)

                for tool in server_tools:
                    name = getattr(tool, "name", "Unknown")
                    description = getattr(tool, "description", "No description")
                    lines.append(f"  🔧 {name}")
                    if description and description != "No description":
                        lines.append(f"     {description}")

            lines.append(
                f"\nTotal: {len(tools)} tools across {len(tools_by_server)} servers"
            )
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error getting tools list: {e}")
            return f"Error retrieving tools list: {str(e)}"

    async def _handle_mcp_command(self, command_obj: Any) -> str:
        """Handle MCP management commands."""
        if not self.mcp_manager:
            return "MCP is not configured for this installation."

        # Extract arguments from Command object or treat as string
        if hasattr(command_obj, "args") and command_obj.args:
            args = " ".join(command_obj.args)
        elif isinstance(command_obj, str):
            args = command_obj
        else:
            args = ""

        args = args.strip()
        if not args:
            args = "status"

        command_parts = args.split()
        command = command_parts[0].lower()

        try:
            if command == "status":
                return await self._mcp_status()
            elif command == "reload":
                return await self._mcp_reload()
            elif command == "connect":
                server_name = command_parts[1] if len(command_parts) > 1 else None
                return await self._mcp_connect(server_name)
            elif command == "disconnect":
                server_name = command_parts[1] if len(command_parts) > 1 else None
                return await self._mcp_disconnect(server_name)
            elif command == "health":
                return await self._mcp_health()
            elif command == "servers":
                return self._mcp_servers()
            elif command == "tools":
                server_name = command_parts[1] if len(command_parts) > 1 else None
                return await self._mcp_tools(server_name)
            else:
                return self._mcp_help()

        except Exception as e:
            logger.error(f"Error handling MCP command '{command}': {e}")
            return f"Error executing MCP command: {str(e)}"

    async def _mcp_status(self) -> str:
        """Get MCP system status."""
        if not self.mcp_manager:
            return "MCP is not configured."

        status_info = []
        status_info.append("MCP System Status")
        status_info.append("=" * 40)

        # Basic status
        enabled = "Yes" if self.mcp_manager.is_enabled else "No"
        status_info.append(f"Enabled: {enabled}")
        status_info.append(
            f"Initialized: {'Yes' if self.mcp_manager.initialized else 'No'}"
        )
        status_info.append(
            f"Connected Servers: {self.mcp_manager.connected_server_count}"
        )
        status_info.append(f"Total Tools: {self.mcp_manager.total_tool_count}")

        # Degradation status
        degradation = self.mcp_manager.get_degradation_status()
        status_info.append(f"Degradation Level: {degradation['degradation_level']}")

        if degradation["functionality_impact"]:
            status_info.append("\nFunctionality Impact:")
            impacts = degradation["functionality_impact"]
            for impact in impacts:  # type: ignore[attr-defined]
                status_info.append(f"  • {impact}")

        return "\n".join(status_info)

    async def _mcp_reload(self) -> str:
        """Reload MCP servers."""
        if not self.mcp_manager:
            return "MCP is not configured."

        try:
            await self.mcp_manager.reload_servers()
            return "MCP servers reloaded successfully."
        except Exception as e:
            return f"Error reloading MCP servers: {str(e)}"

    async def _mcp_connect(self, server_name: Optional[str] = None) -> str:
        """Connect to MCP server(s)."""
        if not self.mcp_manager:
            return "MCP is not configured."

        try:
            if server_name:
                # For specific server, we'd need a method to connect individual servers
                return (
                    f"Connecting to specific server"
                    f" '{server_name}' is not yet"
                    " implemented. Use reload to"
                    " reconnect all servers."
                )
            else:
                await self.mcp_manager.initialize_servers()
                return "Attempted to connect to all MCP servers."
        except Exception as e:
            return f"Error connecting to MCP servers: {str(e)}"

    async def _mcp_disconnect(self, server_name: Optional[str] = None) -> str:
        """Disconnect from MCP server(s)."""
        if not self.mcp_manager:
            return "MCP is not configured."

        try:
            if server_name:
                success = await self.mcp_manager.shutdown_servers(server_name)
                if success:
                    return f"Disconnected from server '{server_name}'."
                else:
                    return (
                        f"Server '{server_name}' was not connected or does not exist."
                    )
            else:
                await self.mcp_manager.shutdown()
                return "Disconnected from all MCP servers."
        except Exception as e:
            return f"Error disconnecting from MCP servers: {str(e)}"

    async def _mcp_health(self) -> str:
        """Get MCP health status."""
        if not self.mcp_manager:
            return "MCP is not configured."

        try:
            # Get both server status and health check
            server_status = self.mcp_manager.get_server_status()
            health_status = await self.mcp_manager.health_check()

            health_info = []
            health_info.append("MCP Server Health")
            health_info.append("=" * 40)

            for server_name, status in server_status.items():
                health_check = health_status.get(server_name, False)
                health_icon = "🟢" if health_check else "🔴"

                health_info.append(f"\n{health_icon} {server_name}")
                health_info.append(
                    f"   Enabled: {'Yes' if status['enabled'] else 'No'}"
                )
                health_info.append(
                    f"   Connected: {'Yes' if status['connected'] else 'No'}"
                )
                health_info.append(f"   Healthy: {'Yes' if health_check else 'No'}")
                health_info.append(f"   Tools: {status['tool_count']}")
                health_info.append(f"   Command: {status['command']}")

            overall_health = health_status.get("overall_healthy", False)
            health_text = "🟢 Healthy" if overall_health else "🔴 Issues detected"
            health_info.append(f"\nOverall Health: {health_text}")

            return "\n".join(health_info)

        except Exception as e:
            return f"Error getting health status: {str(e)}"

    def _mcp_servers(self) -> str:
        """List MCP servers."""
        if not self.mcp_manager:
            return "MCP is not configured."

        try:
            server_status = self.mcp_manager.get_server_status()

            servers_info = []
            servers_info.append("MCP Servers")
            servers_info.append("=" * 30)

            for server_name, status in server_status.items():
                icon = "🟢" if status["connected"] else "🔴"
                servers_info.append(f"{icon} {server_name}")
                servers_info.append(f"   Tools: {status['tool_count']}")
                if status["args"]:
                    servers_info.append(f"   Args: {' '.join(status['args'])}")

            return "\n".join(servers_info)

        except Exception as e:
            return f"Error listing servers: {str(e)}"

    async def _mcp_tools(self, server_name: Optional[str] = None) -> str:
        """List tools from specific server or all servers."""
        if not self.mcp_manager:
            return "MCP is not configured."

        try:
            if server_name:
                tools = self.mcp_manager.get_tools_by_server(server_name)
                if not tools:
                    return (
                        f"No tools found for server"
                        f" '{server_name}' or server"
                        " not connected."
                    )

                tools_info = []
                tools_info.append(f"Tools from {server_name}")
                tools_info.append("=" * 40)

                for tool in tools:
                    name = getattr(tool, "name", "Unknown")
                    description = getattr(tool, "description", "No description")
                    tools_info.append(f"🔧 {name}")
                    if description:
                        tools_info.append(f"   {description}")

                return "\n".join(tools_info)
            else:
                # Return summary of all tools
                return await self._get_tools_list()

        except Exception as e:
            return f"Error listing tools: {str(e)}"

    def _mcp_help(self) -> str:
        """Show MCP command help."""
        help_text = """
MCP Commands:
=============

/mcp status     - Show MCP system status
/mcp health     - Show server health status
/mcp servers    - List all configured servers
/mcp tools      - List all available tools
/mcp tools <server> - List tools from specific server
/mcp reload     - Reload MCP configuration
/mcp connect [server] - Connect to server(s)
/mcp disconnect [server] - Disconnect from server(s)

Examples:
  /mcp status
  /mcp tools filesystem
  /mcp health
"""
        return help_text.strip()

    def _initialize_agent_engine(self) -> None:
        """Initialize the agent engine for autonomous operations."""
        try:
            from .agent_engine import AgentEngine

            self.agent_engine = AgentEngine(  # type: ignore[assignment]
                self.config_manager
            )
            logger.info("Agent engine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize agent engine: {e}")
            # Don't fail completely if agent engine can't be initialized
            self.agent_engine = None
