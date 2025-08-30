"""
Configuration manager for Omnimancer.

This module handles loading, saving, and managing configuration
including provider settings, API keys, and user preferences.
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cryptography.fernet import Fernet

from ..utils.errors import ConfigurationError, MCPConfigurationError
from .config_migration import ConfigMigration, ConfigValidator
from .models import (
    Config,
    ConfigProfile,
    MCPConfig,
    MCPServerConfig,
    ProviderConfig,
)


class ConfigManager:
    """
    Manages Omnimancer configuration.

    This class handles loading and saving configuration files,
    managing API keys, and providing access to settings.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to configuration file (optional)
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path.home() / ".omnimancer" / "config.json"

        self.config: Optional[Config] = None
        self._key_file = self.config_path.parent / ".key"
        self._ensure_config_dir()
        self._cipher = self._get_cipher()

    def _ensure_config_dir(self) -> None:
        """Ensure the configuration directory exists."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_cipher(self) -> Fernet:
        """
        Get or create encryption cipher.

        Returns:
            Fernet cipher instance
        """
        if self._key_file.exists():
            with open(self._key_file, "rb") as f:
                key = f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(self._key_file, "wb") as f:
                f.write(key)
            # Set restrictive permissions on key file
            os.chmod(self._key_file, 0o600)

        return Fernet(key)

    def _encrypt_api_key(self, api_key: str) -> str:
        """
        Encrypt an API key.

        Args:
            api_key: Plain text API key

        Returns:
            Encrypted API key as base64 string
        """
        encrypted = self._cipher.encrypt(api_key.encode())
        return base64.b64encode(encrypted).decode()

    def _decrypt_api_key(self, encrypted_key: str) -> str:
        """
        Decrypt an API key.

        Args:
            encrypted_key: Encrypted API key as base64 string

        Returns:
            Plain text API key

        Raises:
            ConfigurationError: If decryption fails
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_key.encode())
            decrypted = self._cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise ConfigurationError(f"Failed to decrypt API key: {e}")

    def load_config(self) -> Config:
        """
        Load configuration from file.

        Returns:
            Loaded Config object

        Raises:
            ConfigurationError: If config file is invalid or missing
        """
        if not self.config_path.exists() or self.config_path.stat().st_size == 0:
            # Create default config for missing or empty files
            self.config = self._create_default_config()
            self.save_config()
            return self.config

        # Check if migration is needed
        migration = ConfigMigration(self.config_path)
        if migration.needs_migration():
            success, messages = migration.migrate_config()
            if not success:
                raise ConfigurationError(
                    f"Configuration migration failed: {'; '.join(messages)}"
                )

        try:
            with open(self.config_path, "r") as f:
                content = f.read().strip()
                if not content:
                    # Handle empty file
                    self.config = self._create_default_config()
                    self.save_config()
                    return self.config

                config_data = json.loads(content)

            self.config = Config(**config_data)
            return self.config

        except (json.JSONDecodeError, ValueError):
            # If JSON is invalid, create default config
            self.config = self._create_default_config()
            self.save_config()
            return self.config
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {e}", details=str(e)
            )

    def save_config(self, config: Optional[Config] = None) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save (optional, uses current config if not provided)

        Raises:
            ConfigurationError: If saving fails
        """
        config_to_save = config or self.config
        if not config_to_save:
            raise ConfigurationError("No configuration to save")

        # Update current config if a new one was provided
        if config:
            self.config = config

        try:
            with open(self.config_path, "w") as f:
                # Use model_dump with mode='json' to properly serialize datetime objects
                config_dict = config_to_save.model_dump(mode="json")
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {e}", details=str(e)
            )

    def get_config(self) -> Config:
        """
        Get the current configuration.

        Returns:
            Current Config object
        """
        if not self.config:
            self.config = self.load_config()
        return self.config

    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderConfig or None if not found
        """
        config = self.get_config()
        return config.providers.get(provider_name)

    def set_provider_config(
        self, provider_name: str, provider_config: ProviderConfig
    ) -> None:
        """
        Set configuration for a provider.

        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration
        """
        config = self.get_config()
        config.providers[provider_name] = provider_config
        self.save_config()

    def set_default_provider(self, provider_name: str) -> None:
        """
        Set the default provider.

        Args:
            provider_name: Name of the provider

        Raises:
            ConfigurationError: If provider is not configured
        """
        config = self.get_config()
        if provider_name not in config.providers:
            raise ConfigurationError(f"Provider '{provider_name}' is not configured")

        config.default_provider = provider_name
        self.save_config()

    def get_api_key(self, provider_name: str) -> Optional[str]:
        """
        Get API key for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Decrypted API key or None if not found
        """
        provider_config = self.get_provider_config(provider_name)
        if provider_config and provider_config.api_key:
            try:
                return self._decrypt_api_key(provider_config.api_key)
            except ConfigurationError:
                # If decryption fails, assume it's a plain text key (for backward compatibility)
                return provider_config.api_key
        return None

    def set_api_key(self, provider_name: str, api_key: str) -> None:
        """
        Set API key for a provider.

        Args:
            provider_name: Name of the provider
            api_key: API key to set (will be encrypted before storage)
        """
        config = self.get_config()
        encrypted_key = self._encrypt_api_key(api_key)

        if provider_name in config.providers:
            config.providers[provider_name].api_key = encrypted_key
        else:
            # Create new provider config with minimal settings
            config.providers[provider_name] = ProviderConfig(
                api_key=encrypted_key,
                model="default",  # Will be updated when provider is initialized
            )

        self.save_config()

    def add_custom_model(self, model_info: "EnhancedModelInfo") -> None:
        """
        Add a custom model to the configuration.

        Args:
            model_info: Enhanced model information
        """
        config = self.get_config()

        # Check if model already exists (by name and provider)
        existing_model = None
        for i, model in enumerate(config.custom_models):
            if model.name == model_info.name and model.provider == model_info.provider:
                existing_model = i
                break

        if existing_model is not None:
            # Update existing model
            config.custom_models[existing_model] = model_info
        else:
            # Add new model
            config.custom_models.append(model_info)

        self.save_config()

    def remove_custom_model(self, model_name: str, provider: str) -> bool:
        """
        Remove a custom model from the configuration.

        Args:
            model_name: Name of the model to remove
            provider: Provider of the model

        Returns:
            True if model was removed, False if not found
        """
        config = self.get_config()

        for i, model in enumerate(config.custom_models):
            if model.name == model_name and model.provider == provider:
                config.custom_models.pop(i)
                self.save_config()
                return True

        return False

    def get_custom_models(self) -> List["EnhancedModelInfo"]:
        """
        Get list of custom models from configuration.

        Returns:
            List of custom model information
        """
        config = self.get_config()
        return config.custom_models.copy()

    def list_custom_models(self) -> List[Dict[str, Any]]:
        """
        Get custom models as a list of dictionaries for display.

        Returns:
            List of model dictionaries
        """
        custom_models = self.get_custom_models()
        return [
            {
                "name": model.name,
                "provider": model.provider,
                "description": model.description,
                "max_tokens": model.max_tokens,
                "cost_per_million_input": model.cost_per_million_input,
                "cost_per_million_output": model.cost_per_million_output,
                "swe_score": model.swe_score,
                "supports_tools": model.supports_tools,
                "supports_multimodal": model.supports_multimodal,
                "latest_version": model.latest_version,
                "is_free": model.is_free,
            }
            for model in custom_models
        ]

    def _create_default_config(self) -> Config:
        """
        Create a default configuration.

        Returns:
            Default Config object
        """
        return Config(
            default_provider="claude",
            providers={},
            storage_path=str(Path.home() / ".omnimancer"),
        )

    def get_storage_path(self) -> Path:
        """
        Get the storage path for Omnimancer data.

        Returns:
            Path object for storage directory
        """
        config = self.get_config()
        storage_path = Path(config.storage_path).expanduser()
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    def is_first_run(self) -> bool:
        """
        Check if this is the first run (no config file exists).

        Returns:
            True if this is the first run
        """
        return not self.config_path.exists()

    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        try:
            config = self.get_config()
        except ConfigurationError as e:
            return [f"Configuration file error: {e}"]

        # Use optimized config validator with caching
        from .config_validator import ConfigValidator

        validator = ConfigValidator()

        return validator.validate_config(config)

    def setup_initial_config(
        self,
        provider_configs: Dict[str, Dict[str, str]],
        default_provider: str,
    ) -> None:
        """
        Set up initial configuration with provider details.

        Args:
            provider_configs: Dict mapping provider names to their config (api_key, model)
            default_provider: Name of the default provider

        Raises:
            ConfigurationError: If setup fails
        """
        if not provider_configs:
            raise ConfigurationError("At least one provider must be configured")

        if default_provider not in provider_configs:
            raise ConfigurationError(
                f"Default provider '{default_provider}' not in provider configs"
            )

        # Create provider configurations
        providers = {}
        for provider_name, config_data in provider_configs.items():
            if "api_key" not in config_data or "model" not in config_data:
                raise ConfigurationError(
                    f"Provider '{provider_name}' missing api_key or model"
                )

            providers[provider_name] = ProviderConfig(
                api_key=self._encrypt_api_key(config_data["api_key"]),
                model=config_data["model"],
                max_tokens=config_data.get("max_tokens"),
                temperature=config_data.get("temperature"),
            )

        # Create and save configuration
        self.config = Config(
            default_provider=default_provider,
            providers=providers,
            storage_path=str(Path.home() / ".omnimancer"),
        )

        self.save_config()

    def update_provider_settings(self, provider_name: str, **kwargs) -> None:
        """
        Update settings for a specific provider.

        Args:
            provider_name: Name of the provider
            **kwargs: Settings to update (model, max_tokens, temperature)
        """
        config = self.get_config()

        if provider_name not in config.providers:
            raise ConfigurationError(f"Provider '{provider_name}' is not configured")

        provider_config = config.providers[provider_name]

        if "model" in kwargs:
            provider_config.model = kwargs["model"]
        if "max_tokens" in kwargs:
            provider_config.max_tokens = kwargs["max_tokens"]
        if "temperature" in kwargs:
            provider_config.temperature = kwargs["temperature"]

        self.save_config()

    def get_mcp_config(self) -> MCPConfig:
        """
        Get MCP configuration.

        Returns:
            MCPConfig object
        """
        config = self.get_config()
        return config.mcp

    def set_mcp_config(self, mcp_config: MCPConfig) -> None:
        """
        Set MCP configuration.

        Args:
            mcp_config: MCP configuration to set
        """
        config = self.get_config()
        config.mcp = mcp_config
        self.save_config()

    def get_mcp_server_config(self, server_name: str) -> Optional[MCPServerConfig]:
        """
        Get configuration for a specific MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            MCPServerConfig or None if not found
        """
        mcp_config = self.get_mcp_config()
        return mcp_config.servers.get(server_name)

    def set_mcp_server_config(
        self, server_name: str, server_config: MCPServerConfig
    ) -> None:
        """
        Set configuration for an MCP server.

        Args:
            server_name: Name of the MCP server
            server_config: Server configuration
        """
        config = self.get_config()
        config.mcp.servers[server_name] = server_config
        self.save_config()

    def remove_mcp_server_config(self, server_name: str) -> bool:
        """
        Remove configuration for an MCP server.

        Args:
            server_name: Name of the MCP server to remove

        Returns:
            True if server was removed, False if not found
        """
        config = self.get_config()
        if server_name in config.mcp.servers:
            del config.mcp.servers[server_name]
            self.save_config()
            return True
        return False

    def get_enabled_mcp_servers(self) -> Dict[str, MCPServerConfig]:
        """
        Get all enabled MCP server configurations.

        Returns:
            Dictionary of enabled MCP server configurations
        """
        mcp_config = self.get_mcp_config()
        return mcp_config.get_enabled_servers()

    def enable_mcp_server(self, server_name: str) -> None:
        """
        Enable an MCP server.

        Args:
            server_name: Name of the MCP server

        Raises:
            MCPConfigurationError: If server is not configured
        """
        server_config = self.get_mcp_server_config(server_name)
        if not server_config:
            raise MCPConfigurationError(f"MCP server '{server_name}' is not configured")

        server_config.enabled = True
        self.save_config()

    def disable_mcp_server(self, server_name: str) -> None:
        """
        Disable an MCP server.

        Args:
            server_name: Name of the MCP server

        Raises:
            MCPConfigurationError: If server is not configured
        """
        server_config = self.get_mcp_server_config(server_name)
        if not server_config:
            raise MCPConfigurationError(f"MCP server '{server_name}' is not configured")

        server_config.enabled = False
        self.save_config()

    def validate_mcp_config(self) -> List[str]:
        """
        Validate MCP configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            mcp_config = self.get_mcp_config()
        except Exception as e:
            return [f"MCP configuration error: {e}"]

        # Validate server configurations
        for server_name, server_config in mcp_config.servers.items():
            if not server_config.name:
                errors.append(f"MCP server '{server_name}' has no name")
            if not server_config.command:
                errors.append(f"MCP server '{server_name}' has no command")
            if server_config.timeout <= 0:
                errors.append(
                    f"MCP server '{server_name}' has invalid timeout: {server_config.timeout}"
                )

        # Validate global MCP settings
        if mcp_config.auto_approve_timeout <= 0:
            errors.append(
                f"Invalid MCP auto_approve_timeout: {mcp_config.auto_approve_timeout}"
            )
        if mcp_config.max_concurrent_servers <= 0:
            errors.append(
                f"Invalid MCP max_concurrent_servers: {mcp_config.max_concurrent_servers}"
            )

        return errors

    def load_mcp_config_from_file(self, mcp_config_path: str) -> None:
        """
        Load MCP configuration from a separate file.

        Args:
            mcp_config_path: Path to MCP configuration file

        Raises:
            MCPConfigurationError: If loading fails
        """
        try:
            config_path = Path(mcp_config_path).expanduser()
            if not config_path.exists():
                raise MCPConfigurationError(f"MCP config file not found: {config_path}")

            with open(config_path, "r") as f:
                mcp_data = json.load(f)

            # Validate and create MCP config
            mcp_config = MCPConfig(**mcp_data)

            # Update main configuration
            config = self.get_config()
            config.mcp = mcp_config
            self.save_config()

        except json.JSONDecodeError as e:
            raise MCPConfigurationError(f"Invalid JSON in MCP config file: {e}")
        except Exception as e:
            raise MCPConfigurationError(f"Failed to load MCP configuration: {e}")

    def export_mcp_config_to_file(self, mcp_config_path: str) -> None:
        """
        Export MCP configuration to a separate file.

        Args:
            mcp_config_path: Path to export MCP configuration

        Raises:
            MCPConfigurationError: If export fails
        """
        try:
            config_path = Path(mcp_config_path).expanduser()
            config_path.parent.mkdir(parents=True, exist_ok=True)

            mcp_config = self.get_mcp_config()

            with open(config_path, "w") as f:
                json.dump(mcp_config.model_dump(mode="json"), f, indent=2)

        except Exception as e:
            raise MCPConfigurationError(f"Failed to export MCP configuration: {e}")

    def validate_provider_config(
        self, provider_name: str, provider_config: ProviderConfig
    ) -> List[str]:
        """
        Validate configuration for a specific provider.

        Args:
            provider_name: Name of the provider
            provider_config: Provider configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Basic validation
        if not provider_config.model:
            errors.append(f"Provider '{provider_name}' has no model specified")

        # Provider-specific validation
        if provider_name == "claude":
            errors.extend(self._validate_claude_config(provider_config))
        elif provider_name == "openai":
            errors.extend(self._validate_openai_config(provider_config))
        elif provider_name == "gemini":
            errors.extend(self._validate_gemini_config(provider_config))
        elif provider_name == "cohere":
            errors.extend(self._validate_cohere_config(provider_config))
        elif provider_name == "ollama":
            errors.extend(self._validate_ollama_config(provider_config))
        else:
            # Generic validation for unknown providers
            if not provider_config.api_key:
                errors.append(f"Provider '{provider_name}' has no API key")

        return errors

    def _validate_claude_config(self, config: ProviderConfig) -> List[str]:
        """Validate Claude provider configuration."""
        errors = []

        if not config.api_key:
            errors.append("Claude provider requires an API key")
        elif not config.api_key.startswith("sk-ant-"):
            # Check if it's encrypted (base64) or plain text
            try:
                # Try to decrypt - if successful, it's encrypted
                self._decrypt_api_key(config.api_key)
            except:
                # If decryption fails and doesn't start with sk-ant-, it's likely invalid
                if not config.api_key.startswith("sk-ant-"):
                    errors.append("Claude API key should start with 'sk-ant-'")

        # Validate model
        valid_models = [
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
        ]
        if config.model not in valid_models:
            errors.append(
                f"Unknown Claude model '{config.model}'. Valid models: {', '.join(valid_models)}"
            )

        return errors

    def _validate_openai_config(self, config: ProviderConfig) -> List[str]:
        """Validate OpenAI provider configuration."""
        errors = []

        if not config.api_key:
            errors.append("OpenAI provider requires an API key")
        elif not config.api_key.startswith("sk-"):
            # Check if it's encrypted
            try:
                decrypted = self._decrypt_api_key(config.api_key)
                if not decrypted.startswith("sk-"):
                    errors.append("OpenAI API key should start with 'sk-'")
            except:
                errors.append("OpenAI API key should start with 'sk-'")

        # Validate model
        valid_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]
        if config.model not in valid_models:
            errors.append(
                f"Unknown OpenAI model '{config.model}'. Valid models: {', '.join(valid_models)}"
            )

        # Validate OpenAI-specific settings
        if config.organization and not isinstance(config.organization, str):
            errors.append("OpenAI organization must be a string")

        return errors

    def _validate_gemini_config(self, config: ProviderConfig) -> List[str]:
        """Validate Gemini provider configuration."""
        errors = []

        if not config.api_key:
            errors.append("Gemini provider requires an API key")
        elif not config.api_key.startswith("AIza"):
            # Check if it's encrypted
            try:
                decrypted = self._decrypt_api_key(config.api_key)
                if not decrypted.startswith("AIza"):
                    errors.append("Gemini API key should start with 'AIza'")
            except:
                errors.append("Gemini API key should start with 'AIza'")

        # Validate model
        valid_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        if config.model not in valid_models:
            errors.append(
                f"Unknown Gemini model '{config.model}'. Valid models: {', '.join(valid_models)}"
            )

        # Validate Google-specific settings
        if config.project_id and not isinstance(config.project_id, str):
            errors.append("Google Cloud project_id must be a string")

        return errors

    def _validate_cohere_config(self, config: ProviderConfig) -> List[str]:
        """Validate Cohere provider configuration."""
        errors = []

        if not config.api_key:
            errors.append("Cohere provider requires an API key")
        # Cohere API keys don't have a standard prefix, so we can't validate format

        # Validate model
        valid_models = [
            "command-r",
            "command-r-plus",
            "command-light",
            "command",
        ]
        if config.model not in valid_models:
            errors.append(
                f"Unknown Cohere model '{config.model}'. Valid models: {', '.join(valid_models)}"
            )

        return errors

    def _validate_ollama_config(self, config: ProviderConfig) -> List[str]:
        """Validate Ollama provider configuration."""
        errors = []

        # Ollama doesn't require an API key (local server)
        # But we should validate the base_url if provided
        if config.base_url:
            if not config.base_url.startswith(("http://", "https://")):
                errors.append("Ollama base_url must start with 'http://' or 'https://'")

        # Model validation is difficult for Ollama since models are dynamic
        # We'll just ensure it's not empty
        if not config.model:
            errors.append("Ollama provider requires a model name")

        # Validate timeout for Ollama (can be slower than cloud providers)
        if config.timeout and config.timeout < 10:
            errors.append(
                "Ollama timeout should be at least 10 seconds for local inference"
            )

        return errors

    def get_provider_defaults(self, provider_name: str) -> Dict[str, any]:
        """
        Get default configuration values for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary of default configuration values
        """
        defaults = {
            "claude": {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 4096,
                "temperature": 0.7,
                "provider_type": "claude",
                "supports_tools": True,
                "supports_multimodal": True,
                "supports_streaming": True,
                "supports_function_calling": True,
                "supports_system_messages": True,
                "enabled": True,
                "priority": 10,
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True,
                "health_check_enabled": True,
                "health_check_interval": 300,
                "health_check_timeout": 10.0,
                "auth_type": "api_key",
            },
            "openai": {
                "model": "gpt-4",
                "max_tokens": 4096,
                "temperature": 0.7,
                "provider_type": "openai",
                "supports_tools": True,
                "supports_multimodal": True,
                "supports_streaming": True,
                "supports_function_calling": True,
                "supports_system_messages": True,
                "enabled": True,
                "priority": 9,
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True,
                "health_check_enabled": True,
                "health_check_interval": 300,
                "health_check_timeout": 10.0,
                "auth_type": "api_key",
            },
            "gemini": {
                "model": "gemini-1.5-pro",
                "max_tokens": 8192,
                "temperature": 0.7,
                "provider_type": "gemini",
                "supports_tools": True,
                "supports_multimodal": True,
                "supports_streaming": True,
                "supports_function_calling": True,
                "supports_system_messages": True,
                "enabled": True,
                "priority": 8,
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True,
                "health_check_enabled": True,
                "health_check_interval": 300,
                "health_check_timeout": 10.0,
                "auth_type": "api_key",
                "safety_settings": {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
                },
            },
            "cohere": {
                "model": "command-r",
                "max_tokens": 4096,
                "temperature": 0.7,
                "provider_type": "cohere",
                "supports_tools": False,
                "supports_multimodal": False,
                "supports_streaming": True,
                "supports_function_calling": False,
                "supports_system_messages": True,
                "enabled": True,
                "priority": 7,
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True,
                "health_check_enabled": True,
                "health_check_interval": 300,
                "health_check_timeout": 10.0,
                "auth_type": "api_key",
            },
            "ollama": {
                "model": "llama2",
                "max_tokens": 4096,
                "temperature": 0.7,
                "base_url": "http://localhost:11434",
                "timeout": 60.0,
                "provider_type": "ollama",
                "supports_tools": False,
                "supports_multimodal": False,
                "supports_streaming": True,
                "supports_function_calling": False,
                "supports_system_messages": True,
                "enabled": True,
                "priority": 5,
                "max_retries": 3,
                "retry_delay": 2.0,
                "exponential_backoff": True,
                "health_check_enabled": True,
                "health_check_interval": 600,
                "health_check_timeout": 15.0,
                "auth_type": "none",
                "num_ctx": 4096,
                "repeat_penalty": 1.1,
            },
        }

        return defaults.get(
            provider_name,
            {
                "max_tokens": 4096,
                "temperature": 0.7,
                "supports_tools": False,
                "supports_multimodal": False,
                "supports_streaming": True,
                "supports_function_calling": False,
                "supports_system_messages": True,
                "enabled": True,
                "priority": 0,
                "max_retries": 3,
                "retry_delay": 1.0,
                "exponential_backoff": True,
                "health_check_enabled": True,
                "health_check_interval": 300,
                "health_check_timeout": 10.0,
                "auth_type": "api_key",
            },
        )

    def create_provider_config(
        self, provider_name: str, api_key: str = None, **kwargs
    ) -> ProviderConfig:
        """
        Create a provider configuration with defaults.

        Args:
            provider_name: Name of the provider
            api_key: API key (optional for providers like Ollama)
            **kwargs: Additional configuration options

        Returns:
            ProviderConfig with defaults applied
        """
        defaults = self.get_provider_defaults(provider_name)

        # Merge defaults with provided kwargs
        config_data = {**defaults, **kwargs}

        # Add API key if provided
        if api_key:
            config_data["api_key"] = self._encrypt_api_key(api_key)
        elif provider_name != "ollama":  # Ollama doesn't need API key
            config_data["api_key"] = None

        return ProviderConfig(**config_data)

    def load_config_from_sources(
        self,
        env_vars: Optional[Dict[str, str]] = None,
        cli_args: Optional[Dict[str, Any]] = None,
    ) -> Config:
        """
        Load configuration from multiple sources with precedence.

        Args:
            env_vars: Environment variables to consider
            cli_args: Command line arguments to consider

        Returns:
            Merged configuration
        """
        # Start with file-based configuration
        config = self.load_config()

        # Merge environment variables
        if env_vars:
            config.merge_from_env(env_vars)

        # Merge CLI arguments (highest precedence)
        if cli_args:
            self._merge_cli_args(config, cli_args)

        return config

    def _merge_cli_args(self, config: Config, cli_args: Dict[str, Any]) -> None:
        """Merge CLI arguments into configuration."""
        # Map CLI arguments to configuration fields
        cli_mapping = {
            "provider": "default_provider",
            "model": lambda: self._set_provider_model(config, cli_args.get("model")),
            "temperature": lambda: self._set_provider_temperature(
                config, cli_args.get("temperature")
            ),
            "max_tokens": lambda: self._set_provider_max_tokens(
                config, cli_args.get("max_tokens")
            ),
            "debug": "debug_mode",
            "log_level": "log_level",
        }

        for cli_arg, config_path in cli_mapping.items():
            if cli_arg in cli_args and cli_args[cli_arg] is not None:
                if callable(config_path):
                    config_path()
                else:
                    setattr(config, config_path, cli_args[cli_arg])
                    config.config_sources[config_path] = f"cli:{cli_arg}"

    def _set_provider_model(self, config: Config, model: str) -> None:
        """Set model for the default provider."""
        if model and config.default_provider in config.providers:
            config.providers[config.default_provider].model = model
            config.config_sources[f"providers.{config.default_provider}.model"] = (
                "cli:model"
            )

    def _set_provider_temperature(self, config: Config, temperature: float) -> None:
        """Set temperature for the default provider."""
        if temperature is not None and config.default_provider in config.providers:
            config.providers[config.default_provider].temperature = temperature
            config.config_sources[
                f"providers.{config.default_provider}.temperature"
            ] = "cli:temperature"

    def _set_provider_max_tokens(self, config: Config, max_tokens: int) -> None:
        """Set max_tokens for the default provider."""
        if max_tokens is not None and config.default_provider in config.providers:
            config.providers[config.default_provider].max_tokens = max_tokens
            config.config_sources[f"providers.{config.default_provider}.max_tokens"] = (
                "cli:max_tokens"
            )

    def get_effective_config(self) -> Config:
        """
        Get the effective configuration (considering active profile).

        Returns:
            The active configuration
        """
        config = self.get_config()
        return config.get_active_config()

    def create_profile(
        self,
        name: str,
        description: Optional[str] = None,
        copy_from_current: bool = True,
    ) -> ConfigProfile:
        """
        Create a new configuration profile.

        Args:
            name: Profile name
            description: Optional profile description
            copy_from_current: Whether to copy current settings

        Returns:
            Created ConfigProfile
        """
        config = self.get_config()

        if copy_from_current:
            profile = config.create_profile(name, description)
        else:
            # Create empty profile with minimal settings
            profile = ConfigProfile(
                name=name,
                description=description,
                default_provider="claude",
                providers={},
            )
            config.profiles[name] = profile

        self.save_config()
        return profile

    def switch_profile(self, profile_name: str) -> None:
        """
        Switch to a different configuration profile.

        Args:
            profile_name: Name of the profile to switch to
        """
        config = self.get_config()
        config.switch_profile(profile_name)
        self.save_config()

    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a configuration profile.

        Args:
            profile_name: Name of the profile to delete

        Returns:
            True if profile was deleted
        """
        config = self.get_config()
        result = config.delete_profile(profile_name)
        if result:
            self.save_config()
        return result

    def list_profiles(self) -> List[str]:
        """
        List all available configuration profiles.

        Returns:
            List of profile names
        """
        config = self.get_config()
        return list(config.profiles.keys())

    def get_profile(self, profile_name: str) -> Optional[ConfigProfile]:
        """
        Get a specific configuration profile.

        Args:
            profile_name: Name of the profile

        Returns:
            ConfigProfile or None if not found
        """
        config = self.get_config()
        return config.profiles.get(profile_name)

    def backup_config(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the current configuration.

        Args:
            backup_path: Optional path for backup file

        Returns:
            Path to the backup file
        """
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(
                self.config_path.parent / f"config_backup_{timestamp}.json"
            )

        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy current config to backup location
        if self.config_path.exists():
            import shutil

            shutil.copy2(self.config_path, backup_path)

        return str(backup_path)

    def restore_config(self, backup_path: str) -> None:
        """
        Restore configuration from a backup.

        Args:
            backup_path: Path to the backup file

        Raises:
            ConfigurationError: If restore fails
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise ConfigurationError(f"Backup file not found: {backup_path}")

        try:
            # Validate backup file first
            with open(backup_path, "r") as f:
                backup_data = json.load(f)

            # Try to create Config object to validate
            Config(**backup_data)

            # If validation passes, restore the backup
            import shutil

            shutil.copy2(backup_path, self.config_path)

            # Reload configuration
            self.config = None
            self.load_config()

        except Exception as e:
            raise ConfigurationError(f"Failed to restore configuration: {e}")

    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about the current configuration.

        Returns:
            Dictionary with configuration metadata
        """
        config = self.get_config()

        return {
            "config_path": str(self.config_path),
            "config_version": config.config_version,
            "created_at": (
                config.created_at.isoformat() if config.created_at else None
            ),
            "updated_at": (
                config.updated_at.isoformat() if config.updated_at else None
            ),
            "providers_count": len(config.providers),
            "enabled_providers": len(config.get_enabled_providers()),
            "profiles_count": len(config.profiles),
            "active_profile": config.active_profile,
            "mcp_servers_count": len(config.mcp.servers),
            "enabled_mcp_servers": len(config.mcp.get_enabled_servers()),
            "default_provider": config.default_provider,
            "storage_path": config.storage_path,
            "config_sources": config.config_sources,
        }
        """
        Load configuration from multiple sources (file, environment, CLI args).
        
        Args:
            env_vars: Environment variables to merge
            cli_args: CLI arguments to merge
            
        Returns:
            Loaded Config object with merged settings
        """
        # Start with file-based configuration
        config = self.load_config()

        # Merge environment variables
        if env_vars:
            config.merge_from_env(env_vars)

        # Merge CLI arguments (highest priority)
        if cli_args:
            self._merge_cli_args(config, cli_args)

        # Update timestamps
        from datetime import datetime

        if not config.created_at:
            config.created_at = datetime.now()
        config.updated_at = datetime.now()

        self.config = config
        return config

    def _merge_cli_args(self, config: Config, cli_args: Dict[str, Any]) -> None:
        """
        Merge CLI arguments into configuration.

        Args:
            config: Configuration object to update
            cli_args: CLI arguments to merge
        """
        # Map CLI arguments to configuration fields
        cli_mapping = {
            "provider": "default_provider",
            "model": "providers.{provider}.model",
            "max_tokens": "providers.{provider}.max_tokens",
            "temperature": "providers.{provider}.temperature",
            "storage_path": "storage_path",
            "profile": "active_profile",
        }

        for cli_arg, config_path in cli_mapping.items():
            if cli_arg in cli_args and cli_args[cli_arg] is not None:
                value = cli_args[cli_arg]

                # Handle provider-specific settings
                if "{provider}" in config_path:
                    provider = cli_args.get("provider", config.default_provider)
                    config_path = config_path.format(provider=provider)

                    # Ensure provider exists
                    if provider not in config.providers:
                        config.providers[provider] = self.create_provider_config(
                            provider
                        )

                # Set the value and track source
                self._set_nested_value_in_config(config, config_path, value)
                config.config_sources[config_path] = "cli"

    def _set_nested_value_in_config(
        self, config: Config, path: str, value: Any
    ) -> None:
        """
        Set a nested configuration value from a dot-separated path.

        Args:
            config: Configuration object to update
            path: Dot-separated path to the setting
            value: Value to set
        """
        parts = path.split(".")
        obj = config

        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return  # Path doesn't exist

        # Set the final value with type conversion
        final_key = parts[-1]
        if hasattr(obj, final_key):
            current_value = getattr(obj, final_key)
            try:
                # Convert string values to appropriate types
                if isinstance(current_value, bool):
                    converted_value = str(value).lower() in (
                        "true",
                        "1",
                        "yes",
                        "on",
                    )
                elif isinstance(current_value, int):
                    converted_value = int(value)
                elif isinstance(current_value, float):
                    converted_value = float(value)
                else:
                    converted_value = value
                setattr(obj, final_key, converted_value)
            except (ValueError, TypeError):
                # If conversion fails, keep as string
                setattr(obj, final_key, value)
        elif isinstance(obj, dict):
            obj[final_key] = value

    def create_profile(
        self, name: str, description: Optional[str] = None
    ) -> ConfigProfile:
        """
        Create a new configuration profile.

        Args:
            name: Profile name
            description: Optional profile description

        Returns:
            Created ConfigProfile
        """
        config = self.get_config()
        profile = config.create_profile(name, description)
        self.save_config()
        return profile

    def list_profiles(self) -> Dict[str, str]:
        """
        List all configuration profiles.

        Returns:
            Dictionary mapping profile names to descriptions
        """
        config = self.get_config()
        return {
            name: profile.description or "" for name, profile in config.profiles.items()
        }

    def switch_profile(self, profile_name: str) -> None:
        """
        Switch to a different configuration profile.

        Args:
            profile_name: Name of the profile to switch to

        Raises:
            ConfigurationError: If profile doesn't exist
        """
        config = self.get_config()
        if profile_name not in config.profiles:
            raise ConfigurationError(f"Profile '{profile_name}' not found")

        config.switch_profile(profile_name)
        self.save_config()

    def get_active_profile_name(self) -> Optional[str]:
        """
        Get the name of the active profile.

        Returns:
            Active profile name or None if no profile is active
        """
        config = self.get_config()
        return config.active_profile

    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a configuration profile.

        Args:
            profile_name: Name of the profile to delete

        Returns:
            True if profile was deleted, False if not found
        """
        config = self.get_config()
        result = config.delete_profile(profile_name)
        if result:
            self.save_config()
        return result

    def update_profile_provider_settings(
        self, profile_name: str, provider_name: str, **kwargs
    ) -> None:
        """
        Update provider settings in a specific profile.

        Args:
            profile_name: Name of the profile
            provider_name: Name of the provider
            **kwargs: Settings to update

        Raises:
            ConfigurationError: If profile doesn't exist
        """
        config = self.get_config()
        if profile_name not in config.profiles:
            raise ConfigurationError(f"Profile '{profile_name}' not found")

        profile = config.profiles[profile_name]

        if provider_name not in profile.providers:
            profile.providers[provider_name] = self.create_provider_config(
                provider_name
            )

        provider_config = profile.providers[provider_name]

        # Update settings
        for key, value in kwargs.items():
            if hasattr(provider_config, key):
                setattr(provider_config, key, value)

        self.save_config()

    def get_effective_config(self) -> Config:
        """
        Get the effective configuration (considering active profile).

        Returns:
            Effective Config object
        """
        config = self.get_config()
        return config.get_active_config()

    def migrate_config_format(self) -> bool:
        """
        Migrate configuration from older format to current format.

        Returns:
            True if migration was performed, False if not needed
        """
        if not self.config_path.exists():
            return False

        try:
            with open(self.config_path, "r") as f:
                config_data = json.load(f)

            # Check if migration is needed
            if config_data.get("config_version") == "2.0":
                return False  # Already current version

            # Backup original config
            self.backup_config()

            # Migrate provider configurations
            if "providers" in config_data:
                for provider_name, provider_data in config_data["providers"].items():
                    # Add provider_type if missing
                    if "provider_type" not in provider_data:
                        provider_data["provider_type"] = provider_name

                    # Add capability flags based on provider type
                    defaults = self.get_provider_defaults(provider_name)
                    for key in [
                        "supports_tools",
                        "supports_multimodal",
                        "supports_streaming",
                    ]:
                        if key not in provider_data:
                            provider_data[key] = defaults.get(key, False)

            # Add new fields
            config_data["config_version"] = "2.0"
            config_data["profiles"] = config_data.get("profiles", {})
            config_data["active_profile"] = config_data.get("active_profile")
            config_data["config_sources"] = config_data.get("config_sources", {})

            from datetime import datetime

            config_data["created_at"] = config_data.get(
                "created_at", datetime.now().isoformat()
            )
            config_data["updated_at"] = datetime.now().isoformat()

            # Save migrated config
            with open(self.config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            # Reload config
            self.config = None
            self.load_config()

            return True

        except Exception as e:
            raise ConfigurationError(f"Failed to migrate configuration: {e}")

    def backup_config(self) -> str:
        """
        Create a backup of the current configuration.

        Returns:
            Path to the backup file
        """
        if not self.config_path.exists():
            raise ConfigurationError("No configuration file to backup")

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.config_path.parent / f"config_backup_{timestamp}.json"

        import shutil

        shutil.copy2(self.config_path, backup_path)

        return str(backup_path)

    def restore_config(self, backup_path: str) -> None:
        """
        Restore configuration from a backup file.

        Args:
            backup_path: Path to the backup file

        Raises:
            ConfigurationError: If restore fails
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            raise ConfigurationError(f"Backup file not found: {backup_path}")

        try:
            import shutil

            shutil.copy2(backup_file, self.config_path)

            # Clear cached config and reload
            self.config = None
            self.load_config()

        except Exception as e:
            raise ConfigurationError(f"Failed to restore configuration: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.

        Returns:
            Dictionary with configuration summary
        """
        config = self.get_config()

        # Provider capabilities
        provider_capabilities = {}
        for provider_name, provider_config in config.providers.items():
            provider_capabilities[provider_name] = {
                "supports_tools": getattr(provider_config, "supports_tools", False),
                "supports_multimodal": getattr(
                    provider_config, "supports_multimodal", False
                ),
                "supports_streaming": getattr(
                    provider_config, "supports_streaming", True
                ),
                "provider_type": getattr(
                    provider_config, "provider_type", provider_name
                ),
            }

        return {
            "default_provider": config.default_provider,
            "providers": list(config.providers.keys()),
            "profiles": list(config.profiles.keys()),
            "active_profile": config.active_profile,
            "mcp_enabled": config.mcp.enabled,
            "mcp_servers": len(config.mcp.servers),
            "provider_capabilities": provider_capabilities,
            "config_version": config.config_version,
            "storage_path": config.storage_path,
        }

    def _set_nested_config_value(self, config: Config, path: str, value: Any) -> None:
        """
        Set a nested configuration value from a dot-separated path.

        Args:
            config: Configuration object to update
            path: Dot-separated path to the setting
            value: Value to set
        """
        parts = path.split(".")
        obj = config

        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return  # Path doesn't exist

        # Set the final value
        final_key = parts[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
        elif isinstance(obj, dict):
            obj[final_key] = value

    def create_profile(
        self,
        name: str,
        description: Optional[str] = None,
        copy_from_current: bool = True,
    ) -> None:
        """
        Create a new configuration profile.

        Args:
            name: Name of the profile
            description: Optional description
            copy_from_current: Whether to copy current settings to the new profile
        """
        config = self.get_config()

        if name in config.profiles:
            raise ConfigurationError(f"Profile '{name}' already exists")

        if copy_from_current:
            profile = config.create_profile(name, description)
        else:
            # Create empty profile with defaults
            profile = ConfigProfile(
                name=name,
                description=description,
                default_provider="claude",
                providers={},
            )
            config.profiles[name] = profile

        self.save_config()

    def switch_profile(self, profile_name: str) -> None:
        """
        Switch to a different configuration profile.

        Args:
            profile_name: Name of the profile to switch to
        """
        config = self.get_config()
        config.switch_profile(profile_name)
        self.save_config()

    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a configuration profile.

        Args:
            profile_name: Name of the profile to delete

        Returns:
            True if profile was deleted, False if not found
        """
        config = self.get_config()
        result = config.delete_profile(profile_name)
        if result:
            self.save_config()
        return result

    def list_profiles(self) -> Dict[str, str]:
        """
        List all available configuration profiles.

        Returns:
            Dictionary mapping profile names to descriptions
        """
        config = self.get_config()
        return {
            name: profile.description or "No description"
            for name, profile in config.profiles.items()
        }

    def get_active_profile_name(self) -> Optional[str]:
        """
        Get the name of the currently active profile.

        Returns:
            Active profile name or None if using main config
        """
        config = self.get_config()
        return config.active_profile

    def get_effective_config(self) -> Config:
        """
        Get the effective configuration (active profile or main config).

        Returns:
            The configuration that should be used for operations
        """
        config = self.get_config()
        return config.get_active_config()

    def update_profile_provider_settings(
        self, profile_name: str, provider_name: str, **kwargs
    ) -> None:
        """
        Update provider settings for a specific profile.

        Args:
            profile_name: Name of the profile
            provider_name: Name of the provider
            **kwargs: Settings to update
        """
        config = self.get_config()

        if profile_name not in config.profiles:
            raise ConfigurationError(f"Profile '{profile_name}' not found")

        profile = config.profiles[profile_name]

        if provider_name not in profile.providers:
            # Create provider config with defaults
            profile.providers[provider_name] = self.create_provider_config(
                provider_name
            )

        provider_config = profile.providers[provider_name]

        # Update settings
        for key, value in kwargs.items():
            if hasattr(provider_config, key):
                setattr(provider_config, key, value)

        self.save_config()

    def migrate_config_format(self) -> bool:
        """
        Migrate configuration from older format to current format.

        Returns:
            True if migration was performed, False if no migration needed
        """
        # Load the raw config data first
        if not self.config_path.exists():
            return False

        try:
            with open(self.config_path, "r") as f:
                raw_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return False

        # Check if migration is needed
        if raw_config.get("config_version") == "2.0":
            return False  # Already current version

        # Perform migration
        migrated = False

        # Add missing fields with defaults
        if "profiles" not in raw_config:
            raw_config["profiles"] = {}
            migrated = True

        if "active_profile" not in raw_config:
            raw_config["active_profile"] = None
            migrated = True

        if "config_version" not in raw_config:
            raw_config["config_version"] = "2.0"
            migrated = True

        if "config_sources" not in raw_config:
            raw_config["config_sources"] = {}
            migrated = True

        # Migrate provider configurations
        if "providers" in raw_config:
            for provider_name, provider_config in raw_config["providers"].items():
                if "provider_type" not in provider_config:
                    provider_config["provider_type"] = provider_name
                    migrated = True

                if "supports_tools" not in provider_config:
                    defaults = self.get_provider_defaults(provider_name)
                    provider_config["supports_tools"] = defaults.get(
                        "supports_tools", False
                    )
                    provider_config["supports_multimodal"] = defaults.get(
                        "supports_multimodal", False
                    )
                    provider_config["supports_streaming"] = defaults.get(
                        "supports_streaming", True
                    )
                    migrated = True

        # Add timestamps
        from datetime import datetime

        if "created_at" not in raw_config or not raw_config["created_at"]:
            raw_config["created_at"] = datetime.now().isoformat()
            migrated = True

        if "updated_at" not in raw_config:
            raw_config["updated_at"] = datetime.now().isoformat()
            migrated = True

        if migrated:
            # Save the migrated config
            with open(self.config_path, "w") as f:
                json.dump(raw_config, f, indent=2)

            # Reload the config
            self.config = None
            self.load_config()

        return migrated

    def backup_config(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the current configuration.

        Args:
            backup_path: Optional path for backup file

        Returns:
            Path to the backup file
        """
        if not backup_path:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(
                self.config_path.parent / f"config_backup_{timestamp}.json"
            )

        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy current config to backup location
        if self.config_path.exists():
            import shutil

            shutil.copy2(self.config_path, backup_path)

        return str(backup_path)

    def restore_config(self, backup_path: str) -> None:
        """
        Restore configuration from a backup file.

        Args:
            backup_path: Path to the backup file

        Raises:
            ConfigurationError: If restore fails
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise ConfigurationError(f"Backup file not found: {backup_path}")

        try:
            # Validate backup file
            with open(backup_path, "r") as f:
                backup_data = json.load(f)

            # Test that we can create a Config object from backup data
            Config(**backup_data)

            # Create backup of current config before restore
            if self.config_path.exists():
                import time
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Add milliseconds to avoid filename collision
                millis = int(time.time() * 1000) % 1000
                current_backup_path = (
                    self.config_path.parent
                    / f"config_pre_restore_{timestamp}_{millis}.json"
                )
                import shutil

                shutil.copy2(self.config_path, current_backup_path)

            # Copy backup to current config location
            import shutil

            shutil.copy2(backup_path, self.config_path)

            # Clear cached config and reload
            self.config = None
            self.load_config()

        except Exception as e:
            raise ConfigurationError(f"Failed to restore configuration: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.

        Returns:
            Dictionary with configuration summary
        """
        config = self.get_effective_config()

        summary = {
            "config_version": getattr(config, "config_version", "1.0"),
            "default_provider": config.default_provider,
            "providers": list(config.providers.keys()),
            "active_profile": getattr(config, "active_profile", None),
            "profiles": list(getattr(config, "profiles", {}).keys()),
            "mcp_enabled": config.mcp.enabled,
            "mcp_servers": len(config.mcp.servers),
            "enabled_mcp_servers": len(config.mcp.get_enabled_servers()),
            "storage_path": config.storage_path,
            "created_at": getattr(config, "created_at", None),
            "updated_at": getattr(config, "updated_at", None),
        }

        # Add provider capabilities summary
        provider_capabilities = {}
        for name, provider_config in config.providers.items():
            provider_capabilities[name] = {
                "model": provider_config.model,
                "supports_tools": getattr(provider_config, "supports_tools", False),
                "supports_multimodal": getattr(
                    provider_config, "supports_multimodal", False
                ),
                "supports_streaming": getattr(
                    provider_config, "supports_streaming", True
                ),
                "provider_type": getattr(provider_config, "provider_type", name),
            }

        summary["provider_capabilities"] = provider_capabilities

        return summary

    def validate_config_comprehensive(self) -> Dict[str, List[str]]:
        """
        Perform comprehensive configuration validation using ConfigValidator.

        Returns:
            Dictionary mapping validation categories to error lists
        """
        try:
            config = self.get_config()
            validator = ConfigValidator(config)
            return validator.validate_full_config()
        except Exception as e:
            return {"general": [f"Configuration validation failed: {e}"]}

    def create_config_backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the current configuration.

        Args:
            backup_path: Optional path for backup file

        Returns:
            Path to the backup file

        Raises:
            ConfigurationError: If backup creation fails
        """
        migration = ConfigMigration(self.config_path)
        try:
            if backup_path:
                backup_path_obj = Path(backup_path)
                backup_path_obj.parent.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(self.config_path, backup_path_obj)
                return str(backup_path_obj)
            else:
                return str(migration.create_backup())
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration backup: {e}")

    def restore_config_backup(self, backup_path: str) -> None:
        """
        Restore configuration from a backup.

        Args:
            backup_path: Path to the backup file

        Raises:
            ConfigurationError: If restore fails
        """
        migration = ConfigMigration(self.config_path)
        try:
            migration.restore_backup(Path(backup_path))
            # Reload configuration after restore
            self.config = None
            self.load_config()
        except Exception as e:
            raise ConfigurationError(f"Failed to restore configuration backup: {e}")

    def list_config_backups(self) -> List[Dict[str, Any]]:
        """
        List available configuration backups.

        Returns:
            List of backup information dictionaries
        """
        migration = ConfigMigration(self.config_path)
        return migration.list_backups()

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """
        Clean up old backup files, keeping only the most recent ones.

        Args:
            keep_count: Number of backups to keep

        Returns:
            Number of backups deleted
        """
        migration = ConfigMigration(self.config_path)
        return migration.cleanup_old_backups(keep_count)

    def migrate_config_manually(self) -> Tuple[bool, List[str]]:
        """
        Manually trigger configuration migration.

        Returns:
            Tuple of (success, list of migration messages)
        """
        migration = ConfigMigration(self.config_path)
        success, messages = migration.migrate_config()
        if success:
            # Reload configuration after migration
            self.config = None
            self.load_config()
        return success, messages

    def check_migration_needed(self) -> bool:
        """
        Check if configuration migration is needed.

        Returns:
            True if migration is needed
        """
        migration = ConfigMigration(self.config_path)
        return migration.needs_migration()

    def get_config_health_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive health report of the configuration.

        Returns:
            Dictionary with configuration health information
        """
        try:
            config = self.get_config()
            validation_results = self.validate_config_comprehensive()

            # Count errors by category
            error_counts = {}
            total_errors = 0
            for category, errors in validation_results.items():
                if isinstance(errors, list):
                    error_counts[category] = len(errors)
                    total_errors += len(errors)
                elif isinstance(errors, dict):
                    # For nested categories like providers
                    category_total = sum(
                        len(provider_errors) for provider_errors in errors.values()
                    )
                    error_counts[category] = category_total
                    total_errors += category_total

            # Get provider health status
            provider_health = {}
            for provider_name in config.providers.keys():
                provider_health[provider_name] = {
                    "enabled": config.providers[provider_name].enabled,
                    "has_api_key": bool(config.providers[provider_name].api_key),
                    "health_check_enabled": config.providers[
                        provider_name
                    ].health_check_enabled,
                }

            return {
                "overall_health": ("healthy" if total_errors == 0 else "issues_found"),
                "total_errors": total_errors,
                "error_counts_by_category": error_counts,
                "validation_results": validation_results,
                "config_version": config.config_version,
                "migration_needed": self.check_migration_needed(),
                "provider_health": provider_health,
                "mcp_status": {
                    "enabled": config.mcp.enabled,
                    "servers_configured": len(config.mcp.servers),
                    "servers_enabled": len(config.mcp.get_enabled_servers()),
                },
                "backup_info": {"available_backups": len(self.list_config_backups())},
            }
        except Exception as e:
            return {
                "overall_health": "error",
                "error": str(e),
                "total_errors": 1,
                "validation_results": {"general": [f"Health check failed: {e}"]},
            }
