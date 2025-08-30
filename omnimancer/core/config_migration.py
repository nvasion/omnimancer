"""
Configuration migration system for Omnimancer.

This module handles automatic migration from old configuration formats
to new ones, including backup and rollback functionality.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..utils.errors import ConfigurationError
from .models import (
    Config,
    ProviderConfig,
)


class ConfigMigration:
    """Handles configuration migration between versions."""

    def __init__(self, config_path):
        """
        Initialize the migration system.

        Args:
            config_path: Path to the configuration file (str or Path)
        """
        if isinstance(config_path, str):
            self.config_path = Path(config_path)
        elif hasattr(config_path, "__fspath__") or isinstance(config_path, Path):
            self.config_path = Path(config_path)
        else:
            # Handle other path-like objects
            self.config_path = Path(str(config_path))

        self.backup_dir = self.config_path.parent / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def needs_migration(self) -> bool:
        """
        Check if the configuration needs migration.

        Returns:
            True if migration is needed
        """
        if not self.config_path.exists():
            return False

        try:
            with open(self.config_path, "r") as f:
                config_data = json.load(f)

            # Check version
            current_version = config_data.get("config_version", "1.0")
            return current_version != "2.0"

        except (json.JSONDecodeError, KeyError):
            # If we can't read the config or it's malformed, assume migration needed
            return True

    def migrate_config(self) -> Tuple[bool, List[str]]:
        """
        Migrate configuration to the latest version.

        Returns:
            Tuple of (success, list of migration messages)
        """
        messages = []

        if not self.config_path.exists():
            messages.append("No configuration file found, nothing to migrate")
            return True, messages

        try:
            # Create backup first
            backup_path = self.create_backup()
            messages.append(f"Created backup at: {backup_path}")

            # Load old configuration
            with open(self.config_path, "r") as f:
                old_config = json.load(f)

            # Determine migration path
            old_version = old_config.get("config_version", "1.0")
            messages.append(f"Migrating from version {old_version} to 2.0")

            # Perform migration based on version
            if old_version == "1.0" or "config_version" not in old_config:
                new_config = self._migrate_from_v1(old_config)
                messages.extend(self._get_v1_migration_messages())
            else:
                # Future migrations would go here
                new_config = old_config
                messages.append(f"No migration needed for version {old_version}")

            # Validate migrated configuration
            try:
                config_obj = Config(**new_config)
                validation_errors = config_obj.validate_configuration()
                if validation_errors:
                    messages.append(
                        "Warning: Migrated configuration has validation errors:"
                    )
                    messages.extend([f"  - {error}" for error in validation_errors])
            except Exception as e:
                messages.append(
                    f"Warning: Could not validate migrated configuration: {e}"
                )

            # Save migrated configuration
            with open(self.config_path, "w") as f:
                json.dump(new_config, f, indent=2)

            messages.append("Configuration migration completed successfully")
            return True, messages

        except Exception as e:
            messages.append(f"Migration failed: {e}")
            # Try to restore backup
            try:
                self.restore_backup(backup_path)
                messages.append("Restored backup due to migration failure")
            except Exception as restore_error:
                messages.append(f"Failed to restore backup: {restore_error}")

            return False, messages

    def create_backup(self) -> Path:
        """
        Create a backup of the current configuration.

        Returns:
            Path to the backup file

        Raises:
            ConfigurationError: If backup creation fails
        """
        if not self.config_path.exists():
            raise ConfigurationError("No configuration file to backup")

        # Create unique timestamp with microseconds to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_filename = f"config_backup_{timestamp}.json"
        backup_path = self.backup_dir / backup_filename

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(self.config_path, backup_path)
            return backup_path
        except Exception as e:
            raise ConfigurationError(f"Failed to create backup: {e}")

    def restore_backup(self, backup_path: Path) -> None:
        """
        Restore configuration from a backup.

        Args:
            backup_path: Path to the backup file

        Raises:
            ConfigurationError: If restore fails
        """
        if not backup_path.exists():
            raise ConfigurationError(f"Backup file not found: {backup_path}")

        try:
            shutil.copy2(backup_path, self.config_path)
        except Exception as e:
            raise ConfigurationError(f"Failed to restore backup: {e}")

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List available configuration backups.

        Returns:
            List of backup information dictionaries
        """
        backups = []

        if not self.backup_dir.exists():
            return backups

        for backup_file in self.backup_dir.glob("config_backup_*.json"):
            try:
                stat = backup_file.stat()
                backups.append(
                    {
                        "path": backup_file,
                        "filename": backup_file.name,
                        "created": datetime.fromtimestamp(stat.st_ctime),
                        "size": stat.st_size,
                    }
                )
            except Exception:
                # Skip files we can't read
                continue

        # Sort by creation time, newest first
        backups.sort(key=lambda x: x["created"], reverse=True)
        return backups

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """
        Clean up old backup files, keeping only the most recent ones.

        Args:
            keep_count: Number of backups to keep

        Returns:
            Number of backups deleted
        """
        backups = self.list_backups()

        if len(backups) <= keep_count:
            return 0

        deleted_count = 0
        # Keep the first keep_count backups (newest), delete the rest (oldest)
        for backup in backups[keep_count:]:
            try:
                backup["path"].unlink()
                deleted_count += 1
            except Exception:
                # Skip files we can't delete
                continue

        return deleted_count

    def _migrate_from_v1(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate from version 1.0 configuration format.

        Args:
            old_config: Old configuration dictionary

        Returns:
            New configuration dictionary
        """
        new_config = {
            "config_version": "2.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "default_provider": old_config.get("default_provider", "claude"),
            "storage_path": old_config.get(
                "storage_path", str(Path.home() / ".omnimancer")
            ),
            "providers": {},
            "chat_settings": {
                "max_tokens": old_config.get("max_tokens"),
                "temperature": old_config.get("temperature"),
                "context_length": old_config.get("context_length", 4000),
                "save_history": old_config.get("save_history", True),
            },
            "mcp": {
                "enabled": True,
                "servers": {},
                "auto_approve_timeout": 30,
                "max_concurrent_servers": 10,
                "default_timeout": 30,
                "retry_attempts": 3,
                "retry_delay": 1.0,
                "allow_dangerous_tools": False,
                "sandbox_mode": True,
                "max_tool_execution_time": 300,
                "log_tool_calls": True,
                "log_level": "INFO",
                "metrics_enabled": False,
                "auto_discover_tools": True,
                "tool_cache_ttl": 3600,
                "refresh_tools_on_startup": True,
            },
            "profiles": {},
            "active_profile": None,
            "config_sources": {},
            "debug_mode": old_config.get("debug_mode", False),
            "log_level": old_config.get("log_level", "INFO"),
            "auto_update_check": True,
            "telemetry_enabled": False,
            "provider_fallback_enabled": True,
            "provider_health_check_enabled": True,
            "provider_timeout_default": 30.0,
            "api_key_encryption_enabled": True,
            "secure_storage_enabled": True,
            "concurrent_requests_limit": 5,
            "request_cache_enabled": True,
            "request_cache_ttl": 300,
        }

        # Migrate provider configurations
        old_providers = old_config.get("providers", {})
        for provider_name, provider_config in old_providers.items():
            new_provider_config = self._migrate_provider_config_v1(
                provider_name, provider_config
            )
            new_config["providers"][provider_name] = new_provider_config

        # Migrate MCP configuration if it exists
        if "mcp" in old_config:
            old_mcp = old_config["mcp"]
            new_config["mcp"].update(
                {
                    "enabled": old_mcp.get("enabled", True),
                    "servers": self._migrate_mcp_servers_v1(old_mcp.get("servers", {})),
                    "auto_approve_timeout": old_mcp.get("auto_approve_timeout", 30),
                    "max_concurrent_servers": old_mcp.get("max_concurrent_servers", 10),
                }
            )

        return new_config

    def _migrate_provider_config_v1(
        self, provider_name: str, old_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Migrate a single provider configuration from v1 format.

        Args:
            provider_name: Name of the provider
            old_config: Old provider configuration

        Returns:
            New provider configuration dictionary
        """
        # Get provider defaults
        provider_defaults = self._get_provider_defaults(provider_name)

        new_config = {
            "api_key": old_config.get("api_key"),
            "model": old_config.get("model", provider_defaults.get("model", "default")),
            "max_tokens": old_config.get("max_tokens"),
            "temperature": old_config.get("temperature"),
            "base_url": old_config.get("base_url"),
            "organization": old_config.get("organization"),
            "project_id": old_config.get("project_id"),
            "timeout": old_config.get("timeout"),
            "top_p": old_config.get("top_p"),
            "frequency_penalty": old_config.get("frequency_penalty"),
            "presence_penalty": old_config.get("presence_penalty"),
            "config_version": "2.0",
            "enabled": old_config.get("enabled", True),
            "priority": old_config.get(
                "priority", provider_defaults.get("priority", 0)
            ),
            "max_retries": old_config.get("max_retries", 3),
            "retry_delay": old_config.get("retry_delay", 1.0),
            "exponential_backoff": old_config.get("exponential_backoff", True),
            "health_check_enabled": old_config.get("health_check_enabled", True),
            "health_check_interval": old_config.get("health_check_interval", 300),
            "health_check_timeout": old_config.get("health_check_timeout", 10.0),
            "auth_type": old_config.get("auth_type", "api_key"),
            "custom_headers": old_config.get("custom_headers"),
            "oauth_config": old_config.get("oauth_config"),
            "extra_settings": old_config.get("extra_settings"),
        }

        # Add provider-specific defaults
        new_config.update(provider_defaults)

        # Remove None values
        return {k: v for k, v in new_config.items() if v is not None}

    def _migrate_mcp_servers_v1(self, old_servers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate MCP server configurations from v1 format.

        Args:
            old_servers: Old MCP server configurations

        Returns:
            New MCP server configurations dictionary
        """
        new_servers = {}

        for server_name, server_config in old_servers.items():
            new_servers[server_name] = {
                "name": server_config.get("name", server_name),
                "command": server_config.get("command", ""),
                "args": server_config.get("args", []),
                "env": server_config.get("env", {}),
                "enabled": server_config.get("enabled", True),
                "auto_approve": server_config.get("auto_approve", []),
                "timeout": server_config.get("timeout", 30),
            }

        return new_servers

    def _get_provider_defaults(self, provider_name: str) -> Dict[str, Any]:
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
                "provider_type": "claude",
                "supports_tools": True,
                "supports_multimodal": True,
                "supports_streaming": True,
                "supports_function_calling": True,
                "supports_system_messages": True,
                "priority": 10,
                "auth_type": "api_key",
            },
            "openai": {
                "model": "gpt-4",
                "provider_type": "openai",
                "supports_tools": True,
                "supports_multimodal": True,
                "supports_streaming": True,
                "supports_function_calling": True,
                "supports_system_messages": True,
                "priority": 9,
                "auth_type": "api_key",
            },
            "gemini": {
                "model": "gemini-1.5-pro",
                "provider_type": "gemini",
                "supports_tools": True,
                "supports_multimodal": True,
                "supports_streaming": True,
                "supports_function_calling": True,
                "supports_system_messages": True,
                "priority": 8,
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
                "provider_type": "cohere",
                "supports_tools": False,
                "supports_multimodal": False,
                "supports_streaming": True,
                "supports_function_calling": False,
                "supports_system_messages": True,
                "priority": 7,
                "auth_type": "api_key",
            },
            "ollama": {
                "model": "llama2",
                "provider_type": "ollama",
                "supports_tools": False,
                "supports_multimodal": False,
                "supports_streaming": True,
                "supports_function_calling": False,
                "supports_system_messages": True,
                "priority": 6,
                "auth_type": "none",
                "base_url": "http://localhost:11434",
            },
        }

        return defaults.get(
            provider_name,
            {
                "provider_type": provider_name,
                "supports_tools": False,
                "supports_multimodal": False,
                "supports_streaming": True,
                "supports_function_calling": False,
                "supports_system_messages": True,
                "priority": 5,
                "auth_type": "api_key",
            },
        )

    def _get_v1_migration_messages(self) -> List[str]:
        """
        Get informational messages about v1 migration.

        Returns:
            List of migration messages
        """
        return [
            "Added new provider capability flags (supports_tools, supports_multimodal, etc.)",
            "Added provider priority and health check settings",
            "Added MCP (Model Context Protocol) configuration section",
            "Added configuration profiles support",
            "Added enhanced security and performance settings",
            "Added configuration source tracking",
            "Updated configuration schema to version 2.0",
        ]


class ConfigValidator:
    """Validates configuration and provides detailed error reporting."""

    def __init__(self, config: Config):
        """
        Initialize the validator.

        Args:
            config: Configuration to validate
        """
        self.config = config

    def validate_config(self, config: Config) -> List[str]:
        """
        Validate configuration and return list of errors.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors
        """
        self.config = config
        errors = []

        # General validation
        general_errors = self._validate_general()
        errors.extend(general_errors)

        # Provider validation
        for provider_name, provider_config in config.providers.items():
            provider_errors = self._validate_provider(provider_name, provider_config)
            errors.extend(provider_errors)

        # MCP validation
        mcp_errors = self._validate_mcp()
        errors.extend(mcp_errors)

        return errors

    def validate_full_config(self) -> Dict[str, List[str]]:
        """
        Perform comprehensive configuration validation.

        Returns:
            Dictionary mapping validation categories to error lists
        """
        validation_results = {
            "general": [],
            "providers": {},
            "mcp": [],
            "profiles": {},
            "security": [],
            "performance": [],
        }

        # General validation
        validation_results["general"] = self._validate_general()

        # Provider validation
        for provider_name, provider_config in self.config.providers.items():
            provider_errors = self._validate_provider(provider_name, provider_config)
            if provider_errors:
                validation_results["providers"][provider_name] = provider_errors

        # MCP validation
        validation_results["mcp"] = self._validate_mcp()

        # Profile validation
        for profile_name, profile in self.config.profiles.items():
            profile_errors = self._validate_profile(profile_name, profile)
            if profile_errors:
                validation_results["profiles"][profile_name] = profile_errors

        # Security validation
        validation_results["security"] = self._validate_security()

        # Performance validation
        validation_results["performance"] = self._validate_performance()

        # Remove empty categories
        return {k: v for k, v in validation_results.items() if v}

    def _validate_general(self) -> List[str]:
        """Validate general configuration settings."""
        errors = []

        if not self.config.default_provider:
            errors.append("No default provider specified")
        elif self.config.default_provider not in self.config.providers:
            errors.append(
                f"Default provider '{self.config.default_provider}' not configured"
            )

        if not self.config.providers:
            errors.append("No providers configured")

        if not self.config.storage_path:
            errors.append("No storage path specified")
        else:
            try:
                storage_path = Path(self.config.storage_path).expanduser()
                if not storage_path.parent.exists():
                    errors.append(
                        f"Storage path parent directory does not exist: {storage_path.parent}"
                    )
            except Exception as e:
                errors.append(f"Invalid storage path: {e}")

        return errors

    def _validate_provider(
        self, provider_name: str, config: ProviderConfig
    ) -> List[str]:
        """Validate a single provider configuration."""
        errors = []

        if not config.model:
            errors.append("No model specified")

        if config.auth_type != "none" and not config.api_key:
            errors.append("API key required but not provided")

        # Provider-specific validation
        if config.provider_type == "claude" or provider_name == "claude":
            errors.extend(self._validate_claude_provider(config))
        elif config.provider_type == "openai" or provider_name == "openai":
            errors.extend(self._validate_openai_provider(config))
        elif config.provider_type == "gemini" or provider_name == "gemini":
            errors.extend(self._validate_gemini_provider(config))
        elif config.provider_type == "cohere" or provider_name == "cohere":
            errors.extend(self._validate_cohere_provider(config))
        elif config.provider_type == "ollama" or provider_name == "ollama":
            errors.extend(self._validate_ollama_provider(config))

        return errors

    def _validate_claude_provider(self, config: ProviderConfig) -> List[str]:
        """Validate Claude provider configuration."""
        errors = []

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

    def _validate_openai_provider(self, config: ProviderConfig) -> List[str]:
        """Validate OpenAI provider configuration."""
        errors = []

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

        return errors

    def _validate_gemini_provider(self, config: ProviderConfig) -> List[str]:
        """Validate Gemini provider configuration."""
        errors = []

        valid_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        if config.model not in valid_models:
            errors.append(
                f"Unknown Gemini model '{config.model}'. Valid models: {', '.join(valid_models)}"
            )

        return errors

    def _validate_cohere_provider(self, config: ProviderConfig) -> List[str]:
        """Validate Cohere provider configuration."""
        errors = []

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

    def _validate_ollama_provider(self, config: ProviderConfig) -> List[str]:
        """Validate Ollama provider configuration."""
        errors = []

        if config.base_url and not config.base_url.startswith(("http://", "https://")):
            errors.append("Ollama base_url must start with 'http://' or 'https://'")

        return errors

    def _validate_mcp(self) -> List[str]:
        """Validate MCP configuration."""
        errors = []

        for server_name, server_config in self.config.mcp.servers.items():
            if not server_config.name:
                errors.append(f"MCP server '{server_name}' has no name")
            if not server_config.command:
                errors.append(f"MCP server '{server_name}' has no command")

        return errors

    def _validate_profile(
        self, profile_name: str, profile: "ConfigProfile"
    ) -> List[str]:
        """Validate a configuration profile."""
        errors = []

        if not profile.default_provider:
            errors.append("No default provider specified")
        elif profile.default_provider not in profile.providers:
            errors.append(
                f"Default provider '{profile.default_provider}' not configured"
            )

        if not profile.providers:
            errors.append("No providers configured")

        return errors

    def _validate_security(self) -> List[str]:
        """Validate security-related settings."""
        errors = []

        # Check for insecure configurations
        if not self.config.api_key_encryption_enabled:
            errors.append("API key encryption is disabled - this is a security risk")

        if not self.config.secure_storage_enabled:
            errors.append("Secure storage is disabled - this is a security risk")

        # Check MCP security settings
        if self.config.mcp.allow_dangerous_tools:
            errors.append("MCP dangerous tools are allowed - this is a security risk")

        if not self.config.mcp.sandbox_mode:
            errors.append("MCP sandbox mode is disabled - this is a security risk")

        return errors

    def _validate_performance(self) -> List[str]:
        """Validate performance-related settings."""
        errors = []

        if self.config.concurrent_requests_limit > 20:
            errors.append(
                "High concurrent request limit may cause rate limiting issues"
            )

        if self.config.request_cache_ttl > 3600:
            errors.append("Long cache TTL may cause stale data issues")

        return errors
