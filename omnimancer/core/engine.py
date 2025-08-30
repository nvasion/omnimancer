"""
Core engine for Omnimancer CLI.

This module provides the main engine class that coordinates between
providers, configuration, and chat management.

Version: 1.0.0
"""

import logging
from typing import Any, Dict, List, Optional

from ..core.models import ChatResponse, EnhancedModelInfo, ModelInfo
from ..providers.base import BaseProvider
from ..ui.progress_indicator import OperationType, get_progress_indicator
from ..utils.errors import ConfigurationError
from .chat_manager import ChatManager
from .config_manager import ConfigManager
from .conversation_manager import ConversationManager
from .health_monitor import HealthMonitor
from .provider_initializer import ProviderInitializer
from .provider_registry import ProviderRegistry

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
            self.mcp_manager = None

        self.providers: Dict[str, BaseProvider] = {}
        self.current_provider: Optional[BaseProvider] = None
        self._initialized = False

        # Initialize agent engine for autonomous operations
        self.agent_engine = None

    async def initialize_providers(self) -> None:
        """Initialize all configured providers."""
        try:
            config = self.config_manager.get_config()

            # Use the optimized provider initializer with config_manager for API key decryption
            self.providers = await self.provider_initializer.initialize_providers(
                config.providers, self.config_manager
            )

            # Register providers with the provider registry for catalog management
            from ..providers.factory import ProviderFactory

            factory = ProviderFactory()
            available_providers = factory.get_available_providers()

            # Register all available providers (not just configured ones) for catalog management
            for provider_name in available_providers:
                try:
                    # Just register the provider name, the registry will handle class loading
                    self.provider_registry.register_provider(provider_name, None)
                except Exception as e:
                    logger.warning(f"Failed to register provider {provider_name}: {e}")

            # Set default provider
            if config.default_provider and config.default_provider in self.providers:
                self.current_provider = self.providers[config.default_provider]
            elif self.providers:
                # Use first available provider as default
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
                        f"Model '{model_name}' not available for provider '{provider_name}'"
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
                error="No provider available. Please configure a provider first.",
            )

        try:
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

            # Send message to provider
            if progress and progress.enabled:
                progress.start_operation(
                    "engine_provider",
                    OperationType.NETWORK,
                    f"Sending to {self.current_provider.get_provider_name()}",
                )
            response = await self.current_provider.send_message(message, context)
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

            return response

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return ChatResponse(
                content="",
                model_used="",
                tokens_used=0,
                error=f"Failed to send message: {str(e)}",
            )

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
                logger.warning(
                    f"Failed to get models from provider {provider.get_provider_name()}: {e}"
                )

        return all_models

    def get_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all models organized by provider (for CLI display)."""
        result = {}

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
                logger.info("MCP manager not configured - skipping MCP initialization")
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
            tools_by_server = {}
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

    async def _handle_mcp_command(self, command_obj) -> str:
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
        status_info.append(f"Enabled: {'Yes' if self.mcp_manager.is_enabled else 'No'}")
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
            for impact in degradation["functionality_impact"]:
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

    async def _mcp_connect(self, server_name: str = None) -> str:
        """Connect to MCP server(s)."""
        if not self.mcp_manager:
            return "MCP is not configured."

        try:
            if server_name:
                # For specific server, we'd need a method to connect individual servers
                return f"Connecting to specific server '{server_name}' is not yet implemented. Use reload to reconnect all servers."
            else:
                await self.mcp_manager.initialize_servers()
                return "Attempted to connect to all MCP servers."
        except Exception as e:
            return f"Error connecting to MCP servers: {str(e)}"

    async def _mcp_disconnect(self, server_name: str = None) -> str:
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
            health_info.append(
                f"\nOverall Health: {'🟢 Healthy' if overall_health else '🔴 Issues detected'}"
            )

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

    async def _mcp_tools(self, server_name: str = None) -> str:
        """List tools from specific server or all servers."""
        if not self.mcp_manager:
            return "MCP is not configured."

        try:
            if server_name:
                tools = self.mcp_manager.get_tools_by_server(server_name)
                if not tools:
                    return f"No tools found for server '{server_name}' or server not connected."

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

            self.agent_engine = AgentEngine(self.config_manager)
            logger.info("Agent engine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize agent engine: {e}")
            # Don't fail completely if agent engine can't be initialized
            self.agent_engine = None
