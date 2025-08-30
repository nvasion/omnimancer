"""Agent configuration and settings management system."""

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import toml
import yaml
from jsonschema import ValidationError, validate


class ConfigFormat(Enum):
    """Supported configuration file formats."""

    JSON = "json"
    TOML = "toml"
    YAML = "yaml"


class ProviderType(Enum):
    """Types of Omnimancer providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    BEDROCK = "bedrock"
    MISTRAL = "mistral"
    PERPLEXITY = "perplexity"
    XAI = "xai"
    OPENROUTER = "openrouter"
    CLAUDE_CODE = "claude_code"
    VERTEX = "vertex"


@dataclass
class SecuritySettings:
    """Security-related configuration settings."""

    enabled: bool = True
    auto_approve_safe_operations: bool = False
    backup_before_changes: bool = True
    max_file_size_mb: int = 10
    allowed_commands: List[str] = field(
        default_factory=lambda: [
            "git",
            "npm",
            "pip",
            "python",
            "node",
            "make",
            "cargo",
            "go",
        ]
    )
    restricted_paths: List[str] = field(
        default_factory=lambda: [
            ".ssh",
            ".env",
            ".aws",
            ".config/gcloud",
            "/etc",
            "/System",
        ]
    )
    enable_sandboxing: bool = True
    enable_audit_logging: bool = True
    audit_log_level: str = "INFO"


@dataclass
class AgentSettings:
    """Core agent behavior settings."""

    enabled: bool = True
    max_concurrent_operations: int = 5
    default_timeout_seconds: int = 300
    enable_learning: bool = True
    remember_user_preferences: bool = True
    auto_backup_interval_minutes: int = 60
    verbose_logging: bool = False
    enable_progress_reporting: bool = True


@dataclass
class FileSystemSettings:
    """File system operation settings."""

    enable_atomic_operations: bool = True
    backup_directory: str = "/tmp/omnimancer_backups"
    max_backup_age_days: int = 30
    enable_version_control: bool = True
    git_auto_add: bool = False
    large_file_threshold_mb: int = 10
    enable_streaming: bool = True


@dataclass
class WebClientSettings:
    """Web client configuration settings."""

    enabled: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size_mb: int = 100
    enable_rate_limiting: bool = True
    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    requests_per_hour: float = 1000.0
    timeout_seconds: int = 30
    max_redirects: int = 10
    user_agent: str = "Omnimancer-Agent/1.0"
    blacklisted_domains: List[str] = field(
        default_factory=lambda: ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
    )
    whitelisted_domains: List[str] = field(default_factory=list)


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""

    provider_type: ProviderType
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    custom_headers: Dict[str, str] = field(default_factory=dict)
    provider_specific: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreferences:
    """User preferences and learned behaviors."""

    preferred_approval_method: str = "interactive"  # interactive, auto, batch
    preferred_diff_format: str = "unified"  # unified, context, side_by_side
    auto_approve_types: List[str] = field(
        default_factory=lambda: ["file_read", "system_info"]
    )
    remember_choices: bool = True
    notification_level: str = "normal"  # quiet, normal, verbose
    preferred_editor: str = "auto"
    theme: str = "auto"  # auto, light, dark
    language: str = "en"
    timezone: str = "UTC"


class ConfigurationError(Exception):
    """Configuration-related error."""

    pass


class ConfigValidator:
    """Validates configuration using JSON schema."""

    def __init__(self):
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict[str, Dict]:
        """Load JSON schemas for configuration validation."""

        security_schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "auto_approve_safe_operations": {"type": "boolean"},
                "backup_before_changes": {"type": "boolean"},
                "max_file_size_mb": {"type": "integer", "minimum": 1},
                "allowed_commands": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "restricted_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "enable_sandboxing": {"type": "boolean"},
                "enable_audit_logging": {"type": "boolean"},
                "audit_log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                },
            },
            "additionalProperties": False,
        }

        agent_schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "max_concurrent_operations": {"type": "integer", "minimum": 1},
                "default_timeout_seconds": {"type": "integer", "minimum": 1},
                "enable_learning": {"type": "boolean"},
                "remember_user_preferences": {"type": "boolean"},
                "auto_backup_interval_minutes": {
                    "type": "integer",
                    "minimum": 1,
                },
                "verbose_logging": {"type": "boolean"},
                "enable_progress_reporting": {"type": "boolean"},
            },
            "additionalProperties": False,
        }

        provider_schema = {
            "type": "object",
            "properties": {
                "provider_type": {
                    "type": "string",
                    "enum": [p.value for p in ProviderType],
                },
                "enabled": {"type": "boolean"},
                "api_key": {"type": ["string", "null"]},
                "base_url": {"type": ["string", "null"]},
                "model": {"type": ["string", "null"]},
                "max_tokens": {"type": ["integer", "null"], "minimum": 1},
                "temperature": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "maximum": 2,
                },
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "retry_attempts": {"type": "integer", "minimum": 0},
                "custom_headers": {"type": "object"},
                "provider_specific": {"type": "object"},
            },
            "required": ["provider_type"],
            "additionalProperties": False,
        }

        return {
            "security": security_schema,
            "agent": agent_schema,
            "provider": provider_schema,
        }

    def validate_security_settings(self, settings: Dict[str, Any]) -> None:
        """Validate security settings."""
        try:
            validate(instance=settings, schema=self.schemas["security"])
        except ValidationError as e:
            raise ConfigurationError(f"Invalid security settings: {e.message}")

    def validate_agent_settings(self, settings: Dict[str, Any]) -> None:
        """Validate agent settings."""
        try:
            validate(instance=settings, schema=self.schemas["agent"])
        except ValidationError as e:
            raise ConfigurationError(f"Invalid agent settings: {e.message}")

    def validate_provider_config(self, config: Dict[str, Any]) -> None:
        """Validate provider configuration."""
        try:
            validate(instance=config, schema=self.schemas["provider"])
        except ValidationError as e:
            raise ConfigurationError(f"Invalid provider config: {e.message}")


class AgentConfig:
    """Comprehensive configuration management for Omnimancer agents."""

    CONFIG_VERSION = "1.0"
    DEFAULT_CONFIG_DIR = Path.home() / ".omnimancer"

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        auto_save: bool = True,
        watch_files: bool = True,
    ):

        self.config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save = auto_save
        self.watch_files = watch_files

        # Configuration components
        self.security = SecuritySettings()
        self.agent = AgentSettings()
        self.filesystem = FileSystemSettings()
        self.web_client = WebClientSettings()
        self.user_preferences = UserPreferences()

        # Provider configurations
        self.providers: Dict[str, ProviderConfig] = {}

        # Configuration metadata
        self.version = self.CONFIG_VERSION
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # Validation and change tracking
        self.validator = ConfigValidator()
        self.change_callbacks: List[Callable[[str, Any, Any], None]] = []
        self.dirty_sections: Set[str] = set()

        # File watchers (if enabled)
        self._file_watchers: List[Any] = []

        # Load existing configuration
        self._load_all_configs()

        # Set up file watching
        if self.watch_files:
            self._setup_file_watchers()

    def _get_config_path(
        self, section: str, format: ConfigFormat = ConfigFormat.JSON
    ) -> Path:
        """Get path for configuration file."""
        filename = f"{section}.{format.value}"
        return self.config_dir / filename

    def _load_config_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        if not path.exists():
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix == ".json":
                    return json.load(f)
                elif path.suffix == ".toml":
                    return toml.load(f)
                elif path.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config format: {path.suffix}"
                    )
        except Exception as e:
            logging.warning(f"Failed to load config file {path}: {e}")
            return {}

    def _save_config_file(
        self,
        path: Path,
        data: Dict[str, Any],
        format: ConfigFormat = ConfigFormat.JSON,
    ) -> None:
        """Save configuration to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w", encoding="utf-8") as f:
                if format == ConfigFormat.JSON:
                    json.dump(data, f, indent=2, default=str, ensure_ascii=False)
                elif format == ConfigFormat.TOML:
                    toml.dump(data, f)
                elif format == ConfigFormat.YAML:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            raise ConfigurationError(f"Failed to save config file {path}: {e}")

    def _load_all_configs(self) -> None:
        """Load all configuration files."""

        # Load security settings
        security_path = self._get_config_path("security")
        security_data = self._load_config_file(security_path)
        if security_data:
            self.validator.validate_security_settings(security_data)
            self.security = SecuritySettings(**security_data)

        # Load agent settings
        agent_path = self._get_config_path("agent")
        agent_data = self._load_config_file(agent_path)
        if agent_data:
            self.validator.validate_agent_settings(agent_data)
            self.agent = AgentSettings(**agent_data)

        # Load filesystem settings
        filesystem_path = self._get_config_path("filesystem")
        filesystem_data = self._load_config_file(filesystem_path)
        if filesystem_data:
            self.filesystem = FileSystemSettings(**filesystem_data)

        # Load web client settings
        webclient_path = self._get_config_path("webclient")
        webclient_data = self._load_config_file(webclient_path)
        if webclient_data:
            self.web_client = WebClientSettings(**webclient_data)

        # Load user preferences
        preferences_path = self._get_config_path("preferences")
        preferences_data = self._load_config_file(preferences_path)
        if preferences_data:
            self.user_preferences = UserPreferences(**preferences_data)

        # Load provider configurations
        providers_path = self._get_config_path("providers")
        providers_data = self._load_config_file(providers_path)
        if providers_data:
            for provider_id, provider_data in providers_data.items():
                self.validator.validate_provider_config(provider_data)
                provider_type = ProviderType(provider_data["provider_type"])
                self.providers[provider_id] = ProviderConfig(
                    provider_type=provider_type,
                    **{k: v for k, v in provider_data.items() if k != "provider_type"},
                )

    def _save_all_configs(self) -> None:
        """Save all configuration files."""

        # Save security settings
        if (
            "security" in self.dirty_sections
            or not self._get_config_path("security").exists()
        ):
            security_data = asdict(self.security)
            self._save_config_file(self._get_config_path("security"), security_data)

        # Save agent settings
        if (
            "agent" in self.dirty_sections
            or not self._get_config_path("agent").exists()
        ):
            agent_data = asdict(self.agent)
            self._save_config_file(self._get_config_path("agent"), agent_data)

        # Save filesystem settings
        if (
            "filesystem" in self.dirty_sections
            or not self._get_config_path("filesystem").exists()
        ):
            filesystem_data = asdict(self.filesystem)
            self._save_config_file(self._get_config_path("filesystem"), filesystem_data)

        # Save web client settings
        if (
            "webclient" in self.dirty_sections
            or not self._get_config_path("webclient").exists()
        ):
            webclient_data = asdict(self.web_client)
            self._save_config_file(self._get_config_path("webclient"), webclient_data)

        # Save user preferences
        if (
            "preferences" in self.dirty_sections
            or not self._get_config_path("preferences").exists()
        ):
            preferences_data = asdict(self.user_preferences)
            self._save_config_file(
                self._get_config_path("preferences"), preferences_data
            )

        # Save provider configurations
        if (
            "providers" in self.dirty_sections
            or not self._get_config_path("providers").exists()
        ):
            providers_data = {}
            for provider_id, provider_config in self.providers.items():
                provider_dict = asdict(provider_config)
                provider_dict["provider_type"] = provider_config.provider_type.value
                providers_data[provider_id] = provider_dict
            self._save_config_file(self._get_config_path("providers"), providers_data)

        # Clear dirty flags and update timestamp
        self.dirty_sections.clear()
        self.updated_at = datetime.now()

    def _setup_file_watchers(self) -> None:
        """Set up file system watchers for configuration files."""
        # This would use a library like watchdog in a real implementation
        # For now, we'll skip the actual file watching setup
        pass

    def _mark_dirty(self, section: str) -> None:
        """Mark a configuration section as dirty (needs saving)."""
        self.dirty_sections.add(section)

        if self.auto_save:
            self._save_all_configs()

    def _notify_change(self, section: str, old_value: Any, new_value: Any) -> None:
        """Notify registered callbacks of configuration changes."""
        for callback in self.change_callbacks:
            try:
                callback(section, old_value, new_value)
            except Exception as e:
                logging.warning(f"Configuration change callback failed: {e}")

    # Configuration access and modification methods

    def get_security_settings(self) -> SecuritySettings:
        """Get security settings."""
        return self.security

    def update_security_settings(self, **kwargs) -> None:
        """Update security settings."""
        old_settings = asdict(self.security)

        for key, value in kwargs.items():
            if hasattr(self.security, key):
                setattr(self.security, key, value)
            else:
                raise ConfigurationError(f"Unknown security setting: {key}")

        # Validate updated settings
        new_settings = asdict(self.security)
        self.validator.validate_security_settings(new_settings)

        self._mark_dirty("security")
        self._notify_change("security", old_settings, new_settings)

    def get_agent_settings(self) -> AgentSettings:
        """Get agent settings."""
        return self.agent

    def update_agent_settings(self, **kwargs) -> None:
        """Update agent settings."""
        old_settings = asdict(self.agent)

        for key, value in kwargs.items():
            if hasattr(self.agent, key):
                setattr(self.agent, key, value)
            else:
                raise ConfigurationError(f"Unknown agent setting: {key}")

        # Validate updated settings
        new_settings = asdict(self.agent)
        self.validator.validate_agent_settings(new_settings)

        self._mark_dirty("agent")
        self._notify_change("agent", old_settings, new_settings)

    def get_provider_config(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get provider configuration."""
        return self.providers.get(provider_id)

    def add_provider_config(self, provider_id: str, config: ProviderConfig) -> None:
        """Add or update provider configuration."""
        # Validate configuration
        config_dict = asdict(config)
        config_dict["provider_type"] = config.provider_type.value
        self.validator.validate_provider_config(config_dict)

        old_config = self.providers.get(provider_id)
        self.providers[provider_id] = config

        self._mark_dirty("providers")
        self._notify_change(f"providers.{provider_id}", old_config, config)

    def remove_provider_config(self, provider_id: str) -> bool:
        """Remove provider configuration."""
        if provider_id in self.providers:
            old_config = self.providers.pop(provider_id)
            self._mark_dirty("providers")
            self._notify_change(f"providers.{provider_id}", old_config, None)
            return True
        return False

    def get_all_providers(self) -> Dict[str, ProviderConfig]:
        """Get all provider configurations."""
        return self.providers.copy()

    def get_enabled_providers(self) -> Dict[str, ProviderConfig]:
        """Get all enabled provider configurations."""
        return {pid: config for pid, config in self.providers.items() if config.enabled}

    def update_user_preferences(self, **kwargs) -> None:
        """Update user preferences."""
        old_preferences = asdict(self.user_preferences)

        for key, value in kwargs.items():
            if hasattr(self.user_preferences, key):
                setattr(self.user_preferences, key, value)
            else:
                raise ConfigurationError(f"Unknown user preference: {key}")

        new_preferences = asdict(self.user_preferences)
        self._mark_dirty("preferences")
        self._notify_change("preferences", old_preferences, new_preferences)

    def learn_preference(self, preference_type: str, value: Any) -> None:
        """Learn and store a user preference."""
        if not self.user_preferences.remember_choices:
            return

        # This could be expanded with ML-based preference learning
        if preference_type == "approval_method":
            self.user_preferences.preferred_approval_method = value
        elif preference_type == "diff_format":
            self.user_preferences.preferred_diff_format = value
        elif preference_type == "auto_approve_type":
            if value not in self.user_preferences.auto_approve_types:
                self.user_preferences.auto_approve_types.append(value)

        self._mark_dirty("preferences")

    def add_change_callback(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Add callback for configuration changes."""
        self.change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Remove configuration change callback."""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)

    def save(self) -> None:
        """Manually save all configurations."""
        self._save_all_configs()

    def reload(self) -> None:
        """Reload all configurations from files."""
        self._load_all_configs()
        self.dirty_sections.clear()

    def reset_to_defaults(self, section: Optional[str] = None) -> None:
        """Reset configuration section to defaults."""
        if section == "security" or section is None:
            self.security = SecuritySettings()
            self._mark_dirty("security")

        if section == "agent" or section is None:
            self.agent = AgentSettings()
            self._mark_dirty("agent")

        if section == "filesystem" or section is None:
            self.filesystem = FileSystemSettings()
            self._mark_dirty("filesystem")

        if section == "webclient" or section is None:
            self.web_client = WebClientSettings()
            self._mark_dirty("webclient")

        if section == "preferences" or section is None:
            self.user_preferences = UserPreferences()
            self._mark_dirty("preferences")

        if section == "providers" or section is None:
            self.providers.clear()
            self._mark_dirty("providers")

    def export_config(
        self,
        export_path: Path,
        format: ConfigFormat = ConfigFormat.JSON,
        include_sensitive: bool = False,
    ) -> None:
        """Export complete configuration to file."""
        config_data = {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "security": asdict(self.security),
            "agent": asdict(self.agent),
            "filesystem": asdict(self.filesystem),
            "web_client": asdict(self.web_client),
            "user_preferences": asdict(self.user_preferences),
            "providers": {},
        }

        # Handle providers
        for provider_id, provider_config in self.providers.items():
            provider_dict = asdict(provider_config)
            provider_dict["provider_type"] = provider_config.provider_type.value

            # Remove sensitive data if requested
            if not include_sensitive:
                provider_dict.pop("api_key", None)
                provider_dict["custom_headers"] = {
                    k: (
                        "[REDACTED]"
                        if any(
                            sensitive in k.lower()
                            for sensitive in [
                                "key",
                                "token",
                                "authorization",
                                "secret",
                            ]
                        )
                        else v
                    )
                    for k, v in provider_dict.get("custom_headers", {}).items()
                }

            config_data["providers"][provider_id] = provider_dict

        self._save_config_file(export_path, config_data, format)

    def import_config(self, import_path: Path, merge: bool = True) -> None:
        """Import configuration from file."""
        imported_data = self._load_config_file(import_path)

        if not imported_data:
            raise ConfigurationError(f"No data found in {import_path}")

        # Version compatibility check
        imported_version = imported_data.get("version", "1.0")
        if imported_version != self.CONFIG_VERSION:
            logging.warning(
                f"Configuration version mismatch: {imported_version} vs {self.CONFIG_VERSION}"
            )

        if not merge:
            self.reset_to_defaults()

        # Import each section
        if "security" in imported_data:
            self.validator.validate_security_settings(imported_data["security"])
            self.security = SecuritySettings(**imported_data["security"])
            self._mark_dirty("security")

        if "agent" in imported_data:
            self.validator.validate_agent_settings(imported_data["agent"])
            self.agent = AgentSettings(**imported_data["agent"])
            self._mark_dirty("agent")

        if "filesystem" in imported_data:
            self.filesystem = FileSystemSettings(**imported_data["filesystem"])
            self._mark_dirty("filesystem")

        if "web_client" in imported_data:
            self.web_client = WebClientSettings(**imported_data["web_client"])
            self._mark_dirty("webclient")

        if "user_preferences" in imported_data:
            self.user_preferences = UserPreferences(**imported_data["user_preferences"])
            self._mark_dirty("preferences")

        if "providers" in imported_data:
            for provider_id, provider_data in imported_data["providers"].items():
                self.validator.validate_provider_config(provider_data)
                provider_type = ProviderType(provider_data["provider_type"])
                self.providers[provider_id] = ProviderConfig(
                    provider_type=provider_type,
                    **{k: v for k, v in provider_data.items() if k != "provider_type"},
                )
            self._mark_dirty("providers")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "version": self.version,
            "config_dir": str(self.config_dir),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "sections": {
                "security": {
                    "enabled": self.security.enabled,
                    "sandboxing": self.security.enable_sandboxing,
                    "audit_logging": self.security.enable_audit_logging,
                    "allowed_commands_count": len(self.security.allowed_commands),
                    "restricted_paths_count": len(self.security.restricted_paths),
                },
                "agent": {
                    "enabled": self.agent.enabled,
                    "learning_enabled": self.agent.enable_learning,
                    "max_concurrent_ops": self.agent.max_concurrent_operations,
                    "timeout_seconds": self.agent.default_timeout_seconds,
                },
                "providers": {
                    "total_count": len(self.providers),
                    "enabled_count": len(self.get_enabled_providers()),
                    "types": list(
                        set(p.provider_type.value for p in self.providers.values())
                    ),
                },
            },
        }

    @asynccontextmanager
    async def config_transaction(self):
        """Context manager for atomic configuration changes."""
        old_auto_save = self.auto_save
        old_dirty_sections = self.dirty_sections.copy()

        # Save current state for rollback
        checkpoint_data = {
            "security": asdict(self.security),
            "agent": asdict(self.agent),
            "filesystem": asdict(self.filesystem),
            "web_client": asdict(self.web_client),
            "user_preferences": asdict(self.user_preferences),
            "providers": {k: asdict(v) for k, v in self.providers.items()},
        }

        # Disable auto-save during transaction
        self.auto_save = False

        try:
            yield self
            # Save all changes at once
            self._save_all_configs()
        except Exception:
            # Rollback changes to checkpoint state
            self.security = SecuritySettings(**checkpoint_data["security"])
            self.agent = AgentSettings(**checkpoint_data["agent"])
            self.filesystem = FileSystemSettings(**checkpoint_data["filesystem"])
            self.web_client = WebClientSettings(**checkpoint_data["web_client"])
            self.user_preferences = UserPreferences(
                **checkpoint_data["user_preferences"]
            )

            self.providers.clear()
            for provider_id, provider_data in checkpoint_data["providers"].items():
                # Convert provider_type string back to enum
                provider_data["provider_type"] = ProviderType(
                    provider_data["provider_type"]
                )
                self.providers[provider_id] = ProviderConfig(**provider_data)

            self.dirty_sections = old_dirty_sections
            raise
        finally:
            self.auto_save = old_auto_save

    def cleanup(self) -> None:
        """Clean up resources."""
        # Save any pending changes
        if self.dirty_sections and self.auto_save:
            self._save_all_configs()

        # Clean up file watchers
        for watcher in self._file_watchers:
            try:
                watcher.stop()
            except:
                pass
        self._file_watchers.clear()

        # Clear callbacks
        self.change_callbacks.clear()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
