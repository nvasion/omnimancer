"""
Custom Agent Configuration Schema and Storage Layer.

This module defines the data model for custom agent configurations and provides
persistent storage functionality for user-created agent profiles.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from jsonschema import ValidationError, validate

from omnimancer.core.models import ConfigTemplateManager

from .config import (
    ProviderType,
)
from .persona import PersonaCapability, PersonaCategory, PersonaMetadata

logger = logging.getLogger(__name__)


class CustomAgentStatus(Enum):
    """Status of custom agent configurations."""

    DRAFT = "draft"
    ACTIVE = "active"
    DISABLED = "disabled"
    ARCHIVED = "archived"
    ERROR = "error"


class AgentTool(Enum):
    """Available tools that can be enabled for custom agents."""

    FILE_OPERATIONS = "file_operations"
    PROGRAM_EXECUTOR = "program_executor"
    WEB_CLIENT = "web_client"
    MCP_TOOLS = "mcp_tools"
    SEARCH = "search"
    CODE_ANALYSIS = "code_analysis"
    SYSTEM_INFO = "system_info"


@dataclass
class ModelSettings:
    """Model configuration settings for custom agents."""

    provider_type: ProviderType
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate model settings after initialization."""
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if self.top_p is not None and not (0.0 <= self.top_p <= 1.0):
            raise ValueError("Top-p must be between 0.0 and 1.0")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("Max tokens must be positive")


@dataclass
class ContextParameters:
    """Context and behavior parameters for custom agents."""

    system_prompt: str = ""
    context_window_size: int = 32000
    conversation_memory: bool = True
    memory_limit: int = 50  # Number of messages to remember
    user_context: str = ""  # Additional user-specific context
    response_format: str = "default"  # default, json, markdown

    def __post_init__(self):
        """Validate context parameters after initialization."""
        if self.context_window_size < 1000:
            raise ValueError("Context window size must be at least 1000")
        if self.memory_limit < 1:
            raise ValueError("Memory limit must be positive")


@dataclass
class BehaviorRules:
    """Behavior rules and constraints for custom agents."""

    max_response_length: Optional[int] = None
    require_confirmation_for: List[str] = field(
        default_factory=lambda: ["file_delete", "system_commands"]
    )
    auto_approve_patterns: List[str] = field(default_factory=list)
    forbidden_operations: List[str] = field(default_factory=list)
    custom_restrictions: List[str] = field(default_factory=list)
    reasoning_style: str = "balanced"  # concise, balanced, verbose, step_by_step
    creativity_level: str = "medium"  # low, medium, high
    safety_level: str = "standard"  # strict, standard, permissive


@dataclass
class CustomAgentConfig:
    """Complete configuration for a custom agent."""

    # Identity and metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Agent"
    description: str = ""
    category: PersonaCategory = PersonaCategory.CUSTOM

    # Base template information
    base_template_id: Optional[str] = None  # Template to inherit from

    # Configuration components
    model_settings: ModelSettings = field(
        default_factory=lambda: ModelSettings(
            provider_type=ProviderType.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
        )
    )
    context_parameters: ContextParameters = field(default_factory=ContextParameters)
    behavior_rules: BehaviorRules = field(default_factory=BehaviorRules)

    # Capabilities and tools
    capabilities: Set[PersonaCapability] = field(
        default_factory=lambda: {PersonaCapability.GENERAL_PURPOSE}
    )
    enabled_tools: Set[AgentTool] = field(
        default_factory=lambda: {
            AgentTool.FILE_OPERATIONS,
            AgentTool.PROGRAM_EXECUTOR,
        }
    )

    # Security and permissions (inherit from base if not overridden)
    security_overrides: Optional[Dict[str, Any]] = None
    agent_overrides: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: PersonaMetadata = field(
        default_factory=lambda: PersonaMetadata(is_builtin=False, is_custom=True)
    )
    status: CustomAgentStatus = CustomAgentStatus.DRAFT

    # Version and tracking
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

        # Ensure metadata is properly configured for custom agents
        self.metadata.is_custom = True
        self.metadata.is_builtin = False
        if not self.metadata.author:
            self.metadata.author = "User"

    def _validate_config(self):
        """Validate the complete configuration."""
        if not self.name.strip():
            raise ValueError("Agent name cannot be empty")

        if len(self.name) > 100:
            raise ValueError("Agent name too long (max 100 characters)")

        if len(self.description) > 500:
            raise ValueError("Agent description too long (max 500 characters)")

        # Validate tool dependencies
        if AgentTool.MCP_TOOLS in self.enabled_tools:
            # MCP tools require web client for most operations
            if AgentTool.WEB_CLIENT not in self.enabled_tools:
                logger.warning("MCP tools work best with web client enabled")

        # Validate capability consistency
        if PersonaCapability.WEB_SEARCH in self.capabilities:
            if AgentTool.WEB_CLIENT not in self.enabled_tools:
                raise ValueError("Web search capability requires web client tool")

        if PersonaCapability.FILE_OPERATIONS in self.capabilities:
            if AgentTool.FILE_OPERATIONS not in self.enabled_tools:
                raise ValueError(
                    "File operations capability requires file operations tool"
                )

    def inherit_from_template(self, template_manager: ConfigTemplateManager) -> None:
        """Inherit settings from a base template."""
        if not self.base_template_id:
            return

        try:
            template = template_manager.get_template(self.base_template_id)

            # Inherit capabilities if none specified
            if not self.capabilities or self.capabilities == {
                PersonaCapability.GENERAL_PURPOSE
            }:
                # Map template capabilities to persona capabilities
                template_caps = template.metadata.get("capabilities", [])
                self.capabilities = set()
                for cap_name in template_caps:
                    try:
                        self.capabilities.add(PersonaCapability(cap_name))
                    except ValueError:
                        logger.warning(f"Unknown capability in template: {cap_name}")

            # Inherit model settings if using default
            if (
                self.model_settings.provider_type == ProviderType.ANTHROPIC
                and self.model_settings.model_name == "claude-3-haiku-20240307"
            ):

                # Use template's recommended model
                primary_provider = template.metadata.get("primary_provider", "claude")
                if primary_provider in template.recommended_models:
                    self.model_settings.model_name = template.recommended_models[
                        primary_provider
                    ]

                # Use template's provider configuration
                if primary_provider in template.provider_configs:
                    provider_config = template.provider_configs[primary_provider]
                    self.model_settings.temperature = provider_config.get(
                        "temperature", 0.7
                    )
                    self.model_settings.max_tokens = provider_config.get("max_tokens")

        except Exception as e:
            logger.warning(
                f"Failed to inherit from template {self.base_template_id}: {e}"
            )

    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now()
        self.metadata.updated_at = self.updated_at

    def increment_usage(self):
        """Increment usage statistics."""
        self.metadata.update_usage()
        self.update_timestamp()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)

        # Convert enums to string values
        data["category"] = self.category.value
        data["status"] = self.status.value
        data["model_settings"][
            "provider_type"
        ] = self.model_settings.provider_type.value
        data["capabilities"] = [cap.value for cap in self.capabilities]
        data["enabled_tools"] = [tool.value for tool in self.enabled_tools]

        # Handle datetime serialization
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["metadata"]["created_at"] = self.metadata.created_at.isoformat()
        data["metadata"]["updated_at"] = self.metadata.updated_at.isoformat()
        if self.metadata.last_used:
            data["metadata"]["last_used"] = self.metadata.last_used.isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomAgentConfig":
        """Create instance from dictionary."""
        # Convert string values back to enums
        data = data.copy()  # Don't modify original

        # Handle PersonaCategory
        if "category" in data:
            data["category"] = PersonaCategory(data["category"])

        # Handle CustomAgentStatus
        if "status" in data:
            data["status"] = CustomAgentStatus(data["status"])

        # Handle ModelSettings
        if "model_settings" in data:
            model_data = data["model_settings"].copy()
            model_data["provider_type"] = ProviderType(model_data["provider_type"])
            data["model_settings"] = ModelSettings(**model_data)

        # Handle ContextParameters
        if "context_parameters" in data:
            data["context_parameters"] = ContextParameters(**data["context_parameters"])

        # Handle BehaviorRules
        if "behavior_rules" in data:
            data["behavior_rules"] = BehaviorRules(**data["behavior_rules"])

        # Handle capabilities set
        if "capabilities" in data:
            data["capabilities"] = {
                PersonaCapability(cap) for cap in data["capabilities"]
            }

        # Handle enabled_tools set
        if "enabled_tools" in data:
            data["enabled_tools"] = {AgentTool(tool) for tool in data["enabled_tools"]}

        # Handle datetime fields
        for field_name in ["created_at", "updated_at"]:
            if field_name in data:
                data[field_name] = datetime.fromisoformat(data[field_name])

        # Handle metadata
        if "metadata" in data:
            metadata_data = data["metadata"].copy()
            for field_name in ["created_at", "updated_at"]:
                if field_name in metadata_data:
                    metadata_data[field_name] = datetime.fromisoformat(
                        metadata_data[field_name]
                    )
            if "last_used" in metadata_data and metadata_data["last_used"]:
                metadata_data["last_used"] = datetime.fromisoformat(
                    metadata_data["last_used"]
                )
            data["metadata"] = PersonaMetadata(**metadata_data)

        return cls(**data)

    def get_display_summary(self) -> Dict[str, Any]:
        """Get summary for display in listings."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description[:100]
            + ("..." if len(self.description) > 100 else ""),
            "category": self.category.value,
            "status": self.status.value,
            "model": f"{self.model_settings.provider_type.value}:{self.model_settings.model_name}",
            "capabilities_count": len(self.capabilities),
            "tools_count": len(self.enabled_tools),
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
            "last_used": (
                self.metadata.last_used.strftime("%Y-%m-%d %H:%M")
                if self.metadata.last_used
                else "Never"
            ),
            "usage_count": self.metadata.usage_count,
        }


class CustomAgentValidator:
    """Validates custom agent configurations."""

    def __init__(self):
        self.schema = self._build_schema()

    def _build_schema(self) -> Dict[str, Any]:
        """Build JSON schema for validation."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "pattern": r"^[a-f0-9\-]{36}$"},
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
                "description": {"type": "string", "maxLength": 500},
                "category": {
                    "type": "string",
                    "enum": [cat.value for cat in PersonaCategory],
                },
                "base_template_id": {"type": ["string", "null"]},
                "model_settings": {
                    "type": "object",
                    "properties": {
                        "provider_type": {
                            "type": "string",
                            "enum": [pt.value for pt in ProviderType],
                        },
                        "model_name": {"type": "string", "minLength": 1},
                        "temperature": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 2.0,
                        },
                        "max_tokens": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                        },
                        "top_p": {
                            "type": ["number", "null"],
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["provider_type", "model_name", "temperature"],
                    "additionalProperties": True,
                },
                "context_parameters": {
                    "type": "object",
                    "properties": {
                        "context_window_size": {
                            "type": "integer",
                            "minimum": 1000,
                        },
                        "memory_limit": {"type": "integer", "minimum": 1},
                        "system_prompt": {"type": "string"},
                        "response_format": {
                            "type": "string",
                            "enum": ["default", "json", "markdown"],
                        },
                    },
                    "additionalProperties": True,
                },
                "capabilities": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [cap.value for cap in PersonaCapability],
                    },
                },
                "enabled_tools": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [tool.value for tool in AgentTool],
                    },
                },
                "status": {
                    "type": "string",
                    "enum": [status.value for status in CustomAgentStatus],
                },
            },
            "required": [
                "id",
                "name",
                "model_settings",
                "capabilities",
                "enabled_tools",
            ],
            "additionalProperties": True,
        }

    def validate(self, config_data: Dict[str, Any]) -> None:
        """Validate configuration data."""
        try:
            validate(instance=config_data, schema=self.schema)
        except ValidationError as e:
            raise ValueError(f"Invalid agent configuration: {e.message}")

    def validate_config(self, config: CustomAgentConfig) -> None:
        """Validate a CustomAgentConfig instance."""
        config_dict = config.to_dict()
        self.validate(config_dict)


class AgentRepository:
    """Repository for CRUD operations on custom agent configurations."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize repository.

        Args:
            storage_dir: Directory to store agent configs (defaults to ~/.omnimancer/custom_agents)
        """
        self.storage_dir = storage_dir or (
            Path.home() / ".omnimancer" / "custom_agents"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.validator = CustomAgentValidator()
        self._cache: Dict[str, CustomAgentConfig] = {}
        self._cache_loaded = False

    def _get_config_path(self, agent_id: str) -> Path:
        """Get file path for agent configuration."""
        return self.storage_dir / f"{agent_id}.json"

    def _load_cache(self) -> None:
        """Load all configurations into cache."""
        if self._cache_loaded:
            return

        self._cache.clear()

        for config_file in self.storage_dir.glob("*.json"):
            try:
                agent_config = self._load_from_file(config_file)
                self._cache[agent_config.id] = agent_config
            except Exception as e:
                logger.warning(f"Failed to load agent config from {config_file}: {e}")

        self._cache_loaded = True

    def _load_from_file(self, file_path: Path) -> CustomAgentConfig:
        """Load configuration from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate data structure
        self.validator.validate(data)

        return CustomAgentConfig.from_dict(data)

    def _save_to_file(self, config: CustomAgentConfig) -> None:
        """Save configuration to file."""
        # Validate before saving
        self.validator.validate_config(config)

        config_path = self._get_config_path(config.id)
        config_data = config.to_dict()

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        # Update cache
        self._cache[config.id] = config

    def create(self, config: CustomAgentConfig) -> CustomAgentConfig:
        """Create a new agent configuration."""
        self._load_cache()

        if config.id in self._cache:
            raise ValueError(f"Agent with ID {config.id} already exists")

        # Ensure timestamps are set
        config.created_at = datetime.now()
        config.updated_at = datetime.now()
        config.metadata.created_at = config.created_at
        config.metadata.updated_at = config.updated_at

        self._save_to_file(config)
        logger.info(f"Created custom agent: {config.name} ({config.id})")

        return config

    def get(self, agent_id: str) -> Optional[CustomAgentConfig]:
        """Get agent configuration by ID."""
        self._load_cache()
        return self._cache.get(agent_id)

    def get_by_name(self, name: str) -> Optional[CustomAgentConfig]:
        """Get agent configuration by name."""
        self._load_cache()
        for config in self._cache.values():
            if config.name == name:
                return config
        return None

    def list_all(self) -> List[CustomAgentConfig]:
        """Get all agent configurations."""
        self._load_cache()
        return list(self._cache.values())

    def list_by_category(self, category: PersonaCategory) -> List[CustomAgentConfig]:
        """Get agent configurations by category."""
        self._load_cache()
        return [
            config for config in self._cache.values() if config.category == category
        ]

    def list_by_status(self, status: CustomAgentStatus) -> List[CustomAgentConfig]:
        """Get agent configurations by status."""
        self._load_cache()
        return [config for config in self._cache.values() if config.status == status]

    def update(self, config: CustomAgentConfig) -> CustomAgentConfig:
        """Update existing agent configuration."""
        self._load_cache()

        if config.id not in self._cache:
            raise ValueError(f"Agent with ID {config.id} does not exist")

        config.update_timestamp()
        self._save_to_file(config)
        logger.info(f"Updated custom agent: {config.name} ({config.id})")

        return config

    def delete(self, agent_id: str) -> bool:
        """Delete agent configuration."""
        self._load_cache()

        if agent_id not in self._cache:
            return False

        config_path = self._get_config_path(agent_id)
        if config_path.exists():
            config_path.unlink()

        config = self._cache.pop(agent_id)
        logger.info(f"Deleted custom agent: {config.name} ({agent_id})")

        return True

    def search(
        self, query: str, fields: Optional[List[str]] = None
    ) -> List[CustomAgentConfig]:
        """Search agent configurations by query."""
        self._load_cache()

        if not fields:
            fields = ["name", "description"]

        query_lower = query.lower()
        results = []

        for config in self._cache.values():
            match = False

            if "name" in fields and query_lower in config.name.lower():
                match = True
            if "description" in fields and query_lower in config.description.lower():
                match = True
            if "capabilities" in fields:
                for cap in config.capabilities:
                    if query_lower in cap.value.lower():
                        match = True
                        break

            if match:
                results.append(config)

        return results

    def export_config(self, agent_id: str, export_path: Path) -> bool:
        """Export agent configuration to file."""
        config = self.get(agent_id)
        if not config:
            return False

        export_data = config.to_dict()

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return True

    def import_config(
        self, import_path: Path, new_id: bool = True
    ) -> Optional[CustomAgentConfig]:
        """Import agent configuration from file."""
        with open(import_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        self.validator.validate(data)

        config = CustomAgentConfig.from_dict(data)

        # Optionally generate new ID to avoid conflicts
        if new_id:
            config.id = str(uuid.uuid4())
            config.created_at = datetime.now()
            config.updated_at = datetime.now()

        return self.create(config)

    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        self._load_cache()

        total_count = len(self._cache)
        status_counts = {}
        category_counts = {}
        provider_counts = {}

        for config in self._cache.values():
            # Count by status
            status = config.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            # Count by category
            category = config.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

            # Count by provider
            provider = config.model_settings.provider_type.value
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        return {
            "total_agents": total_count,
            "status_distribution": status_counts,
            "category_distribution": category_counts,
            "provider_distribution": provider_counts,
            "storage_directory": str(self.storage_dir),
            "last_updated": (
                max([c.updated_at for c in self._cache.values()]).isoformat()
                if self._cache
                else None
            ),
        }

    def cleanup_orphaned_files(self) -> int:
        """Clean up orphaned configuration files."""
        self._load_cache()

        valid_ids = set(self._cache.keys())
        removed_count = 0

        for config_file in self.storage_dir.glob("*.json"):
            file_id = config_file.stem
            if file_id not in valid_ids:
                try:
                    config_file.unlink()
                    removed_count += 1
                    logger.info(f"Removed orphaned config file: {config_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove orphaned file {config_file}: {e}")

        return removed_count
