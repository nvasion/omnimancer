"""
Agent Persona Architecture for Omnimancer.

This module provides the architectural design for mapping configuration templates
to agent personas with clear interfaces and data models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from omnimancer.core.models import ConfigTemplate, ConfigTemplateManager

from .config import AgentConfig, ProviderConfig, ProviderType
from .status_core import AgentStatus
from .status_manager import UnifiedStatusManager as AgentStatusManager
from .status_manager import (
    get_status_manager,
)

logger = logging.getLogger(__name__)


class PersonaCapability(Enum):
    """Capabilities that agent personas can have."""

    TOOL_CALLING = "tool_calling"
    CODE_GENERATION = "code_generation"
    FILE_OPERATIONS = "file_operations"
    WEB_SEARCH = "web_search"
    LARGE_CONTEXT = "large_context"
    RESEARCH = "research"
    CREATIVE_WRITING = "creative_writing"
    HIGH_TEMPERATURE = "high_temperature"
    FAST_RESPONSE = "fast_response"
    COST_EFFICIENT = "cost_efficient"
    BALANCED = "balanced"
    GENERAL_PURPOSE = "general_purpose"
    MULTIMODAL = "multimodal"
    FUNCTION_CALLING = "function_calling"
    REASONING = "reasoning"
    MATH = "math"
    ANALYSIS = "analysis"


class PersonaCategory(Enum):
    """Categories for organizing agent personas."""

    DEVELOPMENT = "development"
    RESEARCH = "research"
    CREATIVE = "creative"
    PRODUCTIVITY = "productivity"
    ANALYSIS = "analysis"
    GENERAL = "general"
    CUSTOM = "custom"


class PersonaStatus(Enum):
    """Status of agent personas."""

    AVAILABLE = "available"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class PersonaMetadata:
    """Metadata structure for UI display and management."""

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    author: str = "Omnimancer"
    tags: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    rating: Optional[float] = None
    user_notes: str = ""
    is_builtin: bool = True
    is_custom: bool = False

    def update_usage(self) -> None:
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.now()
        self.updated_at = datetime.now()


@dataclass
class PersonaConfiguration:
    """Configuration mapping for agent personas."""

    template_id: str  # Maps to ConfigTemplate
    primary_provider: str  # Primary provider from template
    fallback_providers: List[str] = field(default_factory=list)  # Fallback providers

    # Override settings (optional)
    temperature_override: Optional[float] = None
    max_tokens_override: Optional[int] = None
    timeout_override: Optional[int] = None

    # Omnimancer-specific settings
    tools_enabled: bool = True
    web_search_enabled: bool = False
    file_operations_enabled: bool = True
    approval_required: bool = True
    auto_approve_safe_operations: bool = False

    def get_template(self, template_manager: ConfigTemplateManager) -> ConfigTemplate:
        """Get the underlying configuration template."""
        return template_manager.get_template(self.template_id)

    def get_primary_provider_config(
        self, template_manager: ConfigTemplateManager
    ) -> Dict[str, Any]:
        """Get provider configuration for the primary provider."""
        template = self.get_template(template_manager)
        if self.primary_provider not in template.provider_configs:
            raise ValueError(
                f"Provider {self.primary_provider} not found in template {self.template_id}"
            )

        config = template.provider_configs[self.primary_provider].copy()

        # Apply overrides
        if self.temperature_override is not None:
            config["temperature"] = self.temperature_override
        if self.max_tokens_override is not None:
            config["max_tokens"] = self.max_tokens_override
        if self.timeout_override is not None:
            config["timeout"] = self.timeout_override

        return config

    def to_provider_config(
        self, provider_id: str, template_manager: ConfigTemplateManager
    ) -> ProviderConfig:
        """Convert to ProviderConfig for AgentConfig."""
        template = self.get_template(template_manager)
        provider_config = self.get_primary_provider_config(template_manager)

        # Map provider name to ProviderType
        provider_type_mapping = {
            "claude": ProviderType.ANTHROPIC,
            "openai": ProviderType.OPENAI,
            "gemini": ProviderType.GOOGLE,
            "perplexity": ProviderType.PERPLEXITY,
            "ollama": ProviderType.OPENROUTER,  # Map to OpenRouter as fallback
            "xai": ProviderType.XAI,
            "mistral": ProviderType.MISTRAL,
            "cohere": ProviderType.OPENROUTER,  # Map to OpenRouter as fallback
            "claude-code": ProviderType.CLAUDE_CODE,
        }

        provider_type = provider_type_mapping.get(
            self.primary_provider, ProviderType.ANTHROPIC
        )
        model_id = template.recommended_models.get(self.primary_provider, "default")

        return ProviderConfig(
            provider_type=provider_type,
            enabled=True,
            model=model_id,
            temperature=provider_config.get("temperature", 0.7),
            max_tokens=provider_config.get("max_tokens", 4096),
            timeout_seconds=provider_config.get("timeout", 30),
            custom_headers={},
            provider_specific=provider_config.get("provider_specific", {}),
        )


class AgentPersona(ABC):
    """
    Abstract base class for agent personas.

    Defines the interface and common functionality for all agent personas.
    """

    def __init__(
        self,
        persona_id: str,
        name: str,
        description: str,
        category: PersonaCategory = PersonaCategory.GENERAL,
        template_manager: Optional[ConfigTemplateManager] = None,
    ):
        """
        Initialize agent persona.

        Args:
            persona_id: Unique identifier for the persona
            name: Human-readable name
            description: Description of the persona's purpose
            category: Category for organization
            template_manager: Configuration template manager
        """
        self.id = persona_id
        self.name = name
        self.description = description
        self.category = category
        self.status = PersonaStatus.AVAILABLE
        self.icon = self._get_default_icon()
        self.template_manager = template_manager or ConfigTemplateManager()

        # Core properties
        self.capabilities: Set[PersonaCapability] = set()
        self.configuration: Optional[PersonaConfiguration] = None
        self.metadata = PersonaMetadata()

        # State management
        self._is_active = False
        self._session_data: Dict[str, Any] = {}
        self._event_listeners: List[Callable] = []

    @abstractmethod
    def _get_default_icon(self) -> str:
        """Get default icon for this persona type."""
        pass

    @abstractmethod
    def _setup_capabilities(self) -> Set[PersonaCapability]:
        """Set up persona-specific capabilities."""
        pass

    @abstractmethod
    def _setup_configuration(self) -> PersonaConfiguration:
        """Set up persona-specific configuration."""
        pass

    def get_template(self) -> Optional[ConfigTemplate]:
        """Get the underlying configuration template."""
        if not self.configuration:
            return None
        return self.configuration.get_template(self.template_manager)

    def get_primary_provider_config(self) -> Optional[Dict[str, Any]]:
        """Get provider configuration for the primary provider."""
        if not self.configuration:
            return None
        return self.configuration.get_primary_provider_config(self.template_manager)

    def initialize(self) -> None:
        """Initialize the persona with capabilities and configuration."""
        self.capabilities = self._setup_capabilities()
        self.configuration = self._setup_configuration()
        self.status = PersonaStatus.AVAILABLE

    def activate(self) -> bool:
        """
        Activate the persona.

        Returns:
            True if activation was successful
        """
        try:
            self.status = PersonaStatus.LOADING
            self._is_active = True
            self.metadata.update_usage()
            self.status = PersonaStatus.ACTIVE

            # Notify listeners
            for listener in self._event_listeners:
                try:
                    listener("persona_activated", {"persona_id": self.id})
                except Exception as e:
                    logger.error(f"Error in persona event listener: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to activate persona {self.id}: {e}")
            self.status = PersonaStatus.ERROR
            return False

    def deactivate(self) -> bool:
        """
        Deactivate the persona.

        Returns:
            True if deactivation was successful
        """
        try:
            self._is_active = False
            self.status = PersonaStatus.AVAILABLE
            self._session_data.clear()

            # Notify listeners
            for listener in self._event_listeners:
                try:
                    listener("persona_deactivated", {"persona_id": self.id})
                except Exception as e:
                    logger.error(f"Error in persona event listener: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to deactivate persona {self.id}: {e}")
            return False

    @property
    def is_active(self) -> bool:
        """Check if persona is currently active."""
        return self._is_active and self.status == PersonaStatus.ACTIVE

    def get_session_data(self, key: str, default: Any = None) -> Any:
        """Get session-specific data."""
        return self._session_data.get(key, default)

    def set_session_data(self, key: str, value: Any) -> None:
        """Set session-specific data."""
        self._session_data[key] = value

    def add_event_listener(self, listener: Callable) -> None:
        """Add event listener for persona events."""
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable) -> None:
        """Remove event listener."""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "status": self.status.value,
            "icon": self.icon,
            "capabilities": [cap.value for cap in self.capabilities],
            "configuration": asdict(self.configuration),
            "metadata": asdict(self.metadata),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPersona":
        """Create persona from dictionary representation."""
        # This would be implemented by specific persona types
        raise NotImplementedError("Subclasses must implement from_dict")


class CodingPersona(AgentPersona):
    """Agent persona optimized for software development."""

    def __init__(self, template_manager: Optional[ConfigTemplateManager] = None):
        super().__init__(
            persona_id="coding",
            name="Coding Agent",
            description="Optimized for software development with code-capable models and dev tools",
            category=PersonaCategory.DEVELOPMENT,
            template_manager=template_manager,
        )
        self.initialize()

    def _get_default_icon(self) -> str:
        return "💻"

    def _setup_capabilities(self) -> Set[PersonaCapability]:
        return {
            PersonaCapability.TOOL_CALLING,
            PersonaCapability.CODE_GENERATION,
            PersonaCapability.FILE_OPERATIONS,
            PersonaCapability.FUNCTION_CALLING,
            PersonaCapability.REASONING,
            PersonaCapability.ANALYSIS,
        }

    def _setup_configuration(self) -> PersonaConfiguration:
        return PersonaConfiguration(
            template_id="coding",
            primary_provider="claude",
            fallback_providers=[
                "openai",
                "perplexity",
                "claude-code",
                "ollama",
            ],
            tools_enabled=True,
            file_operations_enabled=True,
            approval_required=True,
            auto_approve_safe_operations=True,
        )


class ResearchPersona(AgentPersona):
    """Agent persona configured for research with web search capabilities."""

    def __init__(self, template_manager: Optional[ConfigTemplateManager] = None):
        super().__init__(
            persona_id="research",
            name="Research Agent",
            description="Configured for research with web search and large context models",
            category=PersonaCategory.RESEARCH,
            template_manager=template_manager,
        )
        self.initialize()

    def _get_default_icon(self) -> str:
        return "🔍"

    def _setup_capabilities(self) -> Set[PersonaCapability]:
        return {
            PersonaCapability.WEB_SEARCH,
            PersonaCapability.LARGE_CONTEXT,
            PersonaCapability.RESEARCH,
            PersonaCapability.ANALYSIS,
            PersonaCapability.REASONING,
        }

    def _setup_configuration(self) -> PersonaConfiguration:
        return PersonaConfiguration(
            template_id="research",
            primary_provider="perplexity",
            fallback_providers=["claude", "gemini", "openai", "xai"],
            web_search_enabled=True,
            tools_enabled=True,
            file_operations_enabled=True,
            approval_required=True,
        )


class CreativePersona(AgentPersona):
    """Agent persona set up for creative writing."""

    def __init__(self, template_manager: Optional[ConfigTemplateManager] = None):
        super().__init__(
            persona_id="creative",
            name="Creative Agent",
            description="Set up for creative writing with high-temperature creative models",
            category=PersonaCategory.CREATIVE,
            template_manager=template_manager,
        )
        self.initialize()

    def _get_default_icon(self) -> str:
        return "🎨"

    def _setup_capabilities(self) -> Set[PersonaCapability]:
        return {
            PersonaCapability.CREATIVE_WRITING,
            PersonaCapability.HIGH_TEMPERATURE,
            PersonaCapability.LARGE_CONTEXT,
        }

    def _setup_configuration(self) -> PersonaConfiguration:
        return PersonaConfiguration(
            template_id="creative",
            primary_provider="claude",
            fallback_providers=["openai", "mistral", "cohere", "gemini"],
            tools_enabled=False,
            web_search_enabled=False,
            file_operations_enabled=True,
            approval_required=False,
        )


class PerformancePersona(AgentPersona):
    """Agent persona optimized for speed and cost efficiency."""

    def __init__(self, template_manager: Optional[ConfigTemplateManager] = None):
        super().__init__(
            persona_id="performance",
            name="Performance Agent",
            description="Optimized for speed and cost efficiency",
            category=PersonaCategory.PRODUCTIVITY,
            template_manager=template_manager,
        )
        self.initialize()

    def _get_default_icon(self) -> str:
        return "⚡"

    def _setup_capabilities(self) -> Set[PersonaCapability]:
        return {
            PersonaCapability.FAST_RESPONSE,
            PersonaCapability.COST_EFFICIENT,
            PersonaCapability.TOOL_CALLING,
        }

    def _setup_configuration(self) -> PersonaConfiguration:
        return PersonaConfiguration(
            template_id="performance",
            primary_provider="openai",
            fallback_providers=["claude", "gemini", "ollama", "claude-code"],
            timeout_override=15,
            tools_enabled=True,
            web_search_enabled=False,
            file_operations_enabled=True,
            approval_required=True,
            auto_approve_safe_operations=True,
        )


class GeneralPersona(AgentPersona):
    """Agent persona with balanced configuration for general-purpose use."""

    def __init__(self, template_manager: Optional[ConfigTemplateManager] = None):
        super().__init__(
            persona_id="general",
            name="General Agent",
            description="Balanced configuration for general-purpose use",
            category=PersonaCategory.GENERAL,
            template_manager=template_manager,
        )
        self.initialize()

    def _get_default_icon(self) -> str:
        return "🤖"

    def _setup_capabilities(self) -> Set[PersonaCapability]:
        return {
            PersonaCapability.BALANCED,
            PersonaCapability.GENERAL_PURPOSE,
            PersonaCapability.TOOL_CALLING,
            PersonaCapability.REASONING,
        }

    def _setup_configuration(self) -> PersonaConfiguration:
        return PersonaConfiguration(
            template_id="general",
            primary_provider="claude",
            fallback_providers=["openai", "gemini", "perplexity", "ollama"],
            tools_enabled=True,
            web_search_enabled=False,
            file_operations_enabled=True,
            approval_required=True,
        )


class PersonaManager:
    """
    Manages agent personas and their lifecycle.

    Handles persona registration, activation, deactivation, and integration
    with the existing configuration system.
    """

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        status_manager: Optional[AgentStatusManager] = None,
        template_manager: Optional[ConfigTemplateManager] = None,
    ):
        """
        Initialize persona manager.

        Args:
            agent_config: Agent configuration instance
            status_manager: Agent status manager for tracking
            template_manager: Configuration template manager
        """
        self.agent_config = agent_config
        self.status_manager = status_manager or get_status_manager()
        self.template_manager = template_manager or ConfigTemplateManager()

        # Persona storage
        self.personas: Dict[str, AgentPersona] = {}
        self.active_persona: Optional[AgentPersona] = None

        # Built-in personas
        self._builtin_personas = {
            "coding": CodingPersona,
            "research": ResearchPersona,
            "creative": CreativePersona,
            "performance": PerformancePersona,
            "general": GeneralPersona,
        }

        # Custom personas (loaded from configuration)
        self.custom_personas: Dict[str, Dict[str, Any]] = {}

        # Event listeners
        self._event_listeners: List[Callable] = []

        # Initialize built-in personas
        self._initialize_builtin_personas()

    def _initialize_builtin_personas(self) -> None:
        """Initialize all built-in personas."""
        for persona_id, persona_class in self._builtin_personas.items():
            try:
                persona = persona_class(template_manager=self.template_manager)
                self.personas[persona_id] = persona
                persona.add_event_listener(self._handle_persona_event)
                logger.debug(f"Initialized built-in persona: {persona_id}")
            except Exception as e:
                logger.error(f"Failed to initialize persona {persona_id}: {e}")

    def _handle_persona_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle persona events."""
        for listener in self._event_listeners:
            try:
                listener(event_type, data)
            except Exception as e:
                logger.error(f"Error in persona manager event listener: {e}")

    def get_persona(self, persona_id: str) -> Optional[AgentPersona]:
        """Get persona by ID."""
        return self.personas.get(persona_id)

    def get_all_personas(self) -> Dict[str, AgentPersona]:
        """Get all registered personas."""
        return self.personas.copy()

    def get_personas_by_category(self, category: PersonaCategory) -> List[AgentPersona]:
        """Get personas filtered by category."""
        return [p for p in self.personas.values() if p.category == category]

    def get_available_personas(self) -> List[AgentPersona]:
        """Get personas that are available for activation."""
        return [
            p
            for p in self.personas.values()
            if p.status in [PersonaStatus.AVAILABLE, PersonaStatus.ACTIVE]
        ]

    def activate_persona(self, persona_id: str) -> bool:
        """
        Activate a persona.

        Args:
            persona_id: ID of persona to activate

        Returns:
            True if activation was successful
        """
        persona = self.personas.get(persona_id)
        if not persona:
            logger.error(f"Persona not found: {persona_id}")
            return False

        try:
            # Deactivate current persona if any
            if self.active_persona and self.active_persona != persona:
                self.deactivate_persona()

            # Activate new persona
            if persona.activate():
                self.active_persona = persona

                # Update agent configuration if available
                if self.agent_config:
                    self._apply_persona_configuration(persona)

                # Update status manager
                if self.status_manager:
                    try:
                        import asyncio

                        # Try to get running loop, if not available, just skip status update
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            self.status_manager.set_agent_status(
                                agent_id=persona_id,
                                status=AgentStatus.ENABLED,
                                metadata={"persona_name": persona.name},
                            )
                        )
                    except RuntimeError:
                        # No event loop running, skip async status update
                        logger.debug(
                            f"No event loop available for status update of persona {persona_id}"
                        )
                        pass

                logger.info(f"Activated persona: {persona_id} ({persona.name})")
                return True

            return False
        except Exception as e:
            logger.error(f"Failed to activate persona {persona_id}: {e}")
            return False

    def deactivate_persona(self) -> bool:
        """
        Deactivate the currently active persona.

        Returns:
            True if deactivation was successful
        """
        if not self.active_persona:
            return True

        try:
            persona = self.active_persona
            if persona.deactivate():
                self.active_persona = None

                # Update status manager
                if self.status_manager:
                    try:
                        import asyncio

                        # Try to get running loop, if not available, just skip status update
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            self.status_manager.set_agent_status(
                                agent_id=persona.id,
                                status=AgentStatus.DISABLED,
                            )
                        )
                    except RuntimeError:
                        # No event loop running, skip async status update
                        logger.debug(
                            f"No event loop available for status update of persona {persona.id}"
                        )
                        pass

                logger.info(f"Deactivated persona: {persona.id}")
                return True

            return False
        except Exception as e:
            logger.error(f"Failed to deactivate persona: {e}")
            return False

    def _apply_persona_configuration(self, persona: AgentPersona) -> None:
        """Apply persona configuration to agent config."""
        if not self.agent_config or not persona.configuration:
            return

        try:
            # Convert persona configuration to provider config
            provider_config = persona.configuration.to_provider_config(
                persona.id, self.template_manager
            )

            # Add to agent configuration
            self.agent_config.add_provider_config(
                f"persona_{persona.id}", provider_config
            )

            logger.debug(f"Applied configuration for persona: {persona.id}")
        except Exception as e:
            logger.error(f"Failed to apply persona configuration: {e}")

    def register_custom_persona(self, persona_data: Dict[str, Any]) -> bool:
        """
        Register a custom persona from configuration data.

        Args:
            persona_data: Persona configuration dictionary

        Returns:
            True if registration was successful
        """
        try:
            # Validate required fields
            required_fields = ["id", "name", "description", "configuration"]
            for field in required_fields:
                if field not in persona_data:
                    raise ValueError(f"Missing required field: {field}")

            # Create custom persona (this would need a CustomPersona class)
            # For now, we'll store the data for later implementation
            persona_id = persona_data["id"]
            self.custom_personas[persona_id] = persona_data

            logger.info(f"Registered custom persona: {persona_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register custom persona: {e}")
            return False

    def get_persona_recommendations(self, context: str) -> List[AgentPersona]:
        """
        Get persona recommendations based on context.

        Args:
            context: Context description for recommendations

        Returns:
            List of recommended personas
        """
        # Simple keyword-based recommendations
        # In a real implementation, this could use ML or more sophisticated matching
        context_lower = context.lower()
        recommendations = []

        # Coding-related keywords
        if any(
            word in context_lower
            for word in [
                "code",
                "programming",
                "development",
                "bug",
                "function",
            ]
        ):
            coding_persona = self.get_persona("coding")
            if coding_persona:
                recommendations.append(coding_persona)

        # Research-related keywords
        if any(
            word in context_lower
            for word in ["research", "search", "find", "information", "study"]
        ):
            research_persona = self.get_persona("research")
            if research_persona:
                recommendations.append(research_persona)

        # Creative-related keywords
        if any(
            word in context_lower
            for word in ["write", "story", "creative", "poem", "novel"]
        ):
            creative_persona = self.get_persona("creative")
            if creative_persona:
                recommendations.append(creative_persona)

        # Performance-related keywords
        if any(
            word in context_lower for word in ["fast", "quick", "speed", "efficient"]
        ):
            performance_persona = self.get_persona("performance")
            if performance_persona:
                recommendations.append(performance_persona)

        # Default to general if no specific matches
        if not recommendations:
            general_persona = self.get_persona("general")
            if general_persona:
                recommendations.append(general_persona)

        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """Get persona manager statistics."""
        return {
            "total_personas": len(self.personas),
            "builtin_personas": len(self._builtin_personas),
            "custom_personas": len(self.custom_personas),
            "active_persona": (self.active_persona.id if self.active_persona else None),
            "available_personas": len(self.get_available_personas()),
            "categories": list(set(p.category.value for p in self.personas.values())),
        }

    def add_event_listener(self, listener: Callable) -> None:
        """Add event listener for persona manager events."""
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable) -> None:
        """Remove event listener."""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)


# Global instance for easy access
_global_persona_manager: Optional[PersonaManager] = None


def get_persona_manager() -> PersonaManager:
    """Get the global persona manager instance."""
    global _global_persona_manager
    if _global_persona_manager is None:
        _global_persona_manager = PersonaManager()
    return _global_persona_manager


def set_persona_manager(manager: PersonaManager) -> None:
    """Set the global persona manager instance."""
    global _global_persona_manager
    _global_persona_manager = manager
