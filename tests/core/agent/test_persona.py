"""
Tests for Agent Persona system.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from omnimancer.core.agent.persona import (
    AgentPersona,
    PersonaCapability,
    PersonaCategory,
    PersonaStatus,
    PersonaConfiguration,
    PersonaMetadata,
    PersonaManager,
    CodingPersona,
    ResearchPersona,
    CreativePersona,
    PerformancePersona,
    GeneralPersona,
    get_persona_manager,
    set_persona_manager
)
from omnimancer.core.agent.config import ProviderType, ProviderConfig
from omnimancer.core.models import ConfigTemplateManager


class TestPersonaConfiguration:
    """Test PersonaConfiguration class."""

    def test_persona_configuration_creation(self):
        """Test creating a PersonaConfiguration."""
        config = PersonaConfiguration(
            template_id="coding",
            primary_provider="claude",
            fallback_providers=["openai", "perplexity"]
        )
        
        assert config.template_id == "coding"
        assert config.primary_provider == "claude"
        assert config.fallback_providers == ["openai", "perplexity"]
        assert config.tools_enabled is True
        assert config.approval_required is True

    def test_get_template(self):
        """Test getting template from configuration."""
        template_manager = ConfigTemplateManager()
        config = PersonaConfiguration(
            template_id="coding",
            primary_provider="claude"
        )
        
        template = config.get_template(template_manager)
        assert template is not None
        assert template.name == "coding"

    def test_get_primary_provider_config(self):
        """Test getting primary provider configuration."""
        template_manager = ConfigTemplateManager()
        config = PersonaConfiguration(
            template_id="coding",
            primary_provider="claude",
            temperature_override=0.2
        )
        
        provider_config = config.get_primary_provider_config(template_manager)
        assert provider_config is not None
        assert provider_config['temperature'] == 0.2  # Override applied

    def test_get_primary_provider_config_invalid_provider(self):
        """Test getting config for invalid provider."""
        template_manager = ConfigTemplateManager()
        config = PersonaConfiguration(
            template_id="coding",
            primary_provider="invalid_provider"
        )
        
        with pytest.raises(ValueError, match="Provider invalid_provider not found"):
            config.get_primary_provider_config(template_manager)

    def test_to_provider_config(self):
        """Test converting to ProviderConfig."""
        template_manager = ConfigTemplateManager()
        config = PersonaConfiguration(
            template_id="coding",
            primary_provider="claude"
        )
        
        provider_config = config.to_provider_config("test_persona", template_manager)
        assert isinstance(provider_config, ProviderConfig)
        assert provider_config.provider_type == ProviderType.ANTHROPIC
        assert provider_config.enabled is True


class TestPersonaMetadata:
    """Test PersonaMetadata class."""

    def test_persona_metadata_creation(self):
        """Test creating PersonaMetadata."""
        metadata = PersonaMetadata()
        assert metadata.version == "1.0.0"
        assert metadata.author == "Omnimancer"
        assert metadata.usage_count == 0
        assert metadata.is_builtin is True

    def test_update_usage(self):
        """Test updating usage statistics."""
        metadata = PersonaMetadata()
        initial_updated_at = metadata.updated_at
        
        metadata.update_usage()
        
        assert metadata.usage_count == 1
        assert metadata.last_used is not None
        assert metadata.updated_at > initial_updated_at


class TestCodingPersona:
    """Test CodingPersona class."""

    def test_coding_persona_creation(self):
        """Test creating a CodingPersona."""
        template_manager = ConfigTemplateManager()
        persona = CodingPersona(template_manager)
        
        assert persona.id == "coding"
        assert persona.name == "Coding Agent"
        assert persona.category == PersonaCategory.DEVELOPMENT
        assert persona.icon == "💻"
        assert PersonaCapability.CODE_GENERATION in persona.capabilities
        assert PersonaCapability.TOOL_CALLING in persona.capabilities

    def test_coding_persona_configuration(self):
        """Test CodingPersona configuration."""
        template_manager = ConfigTemplateManager()
        persona = CodingPersona(template_manager)
        
        assert persona.configuration is not None
        assert persona.configuration.template_id == "coding"
        assert persona.configuration.primary_provider == "claude"
        assert persona.configuration.tools_enabled is True

    def test_get_template(self):
        """Test getting template from persona."""
        template_manager = ConfigTemplateManager()
        persona = CodingPersona(template_manager)
        
        template = persona.get_template()
        assert template is not None
        assert template.name == "coding"


class TestResearchPersona:
    """Test ResearchPersona class."""

    def test_research_persona_creation(self):
        """Test creating a ResearchPersona."""
        template_manager = ConfigTemplateManager()
        persona = ResearchPersona(template_manager)
        
        assert persona.id == "research"
        assert persona.name == "Research Agent"
        assert persona.category == PersonaCategory.RESEARCH
        assert persona.icon == "🔍"
        assert PersonaCapability.WEB_SEARCH in persona.capabilities
        assert PersonaCapability.RESEARCH in persona.capabilities

    def test_research_persona_configuration(self):
        """Test ResearchPersona configuration."""
        template_manager = ConfigTemplateManager()
        persona = ResearchPersona(template_manager)
        
        assert persona.configuration is not None
        assert persona.configuration.template_id == "research"
        assert persona.configuration.primary_provider == "perplexity"
        assert persona.configuration.web_search_enabled is True


class TestCreativePersona:
    """Test CreativePersona class."""

    def test_creative_persona_creation(self):
        """Test creating a CreativePersona."""
        template_manager = ConfigTemplateManager()
        persona = CreativePersona(template_manager)
        
        assert persona.id == "creative"
        assert persona.name == "Creative Agent"
        assert persona.category == PersonaCategory.CREATIVE
        assert persona.icon == "🎨"
        assert PersonaCapability.CREATIVE_WRITING in persona.capabilities
        assert PersonaCapability.HIGH_TEMPERATURE in persona.capabilities

    def test_creative_persona_configuration(self):
        """Test CreativePersona configuration."""
        template_manager = ConfigTemplateManager()
        persona = CreativePersona(template_manager)
        
        assert persona.configuration is not None
        assert persona.configuration.template_id == "creative"
        assert persona.configuration.primary_provider == "claude"
        assert persona.configuration.tools_enabled is False
        assert persona.configuration.approval_required is False


class TestPerformancePersona:
    """Test PerformancePersona class."""

    def test_performance_persona_creation(self):
        """Test creating a PerformancePersona."""
        template_manager = ConfigTemplateManager()
        persona = PerformancePersona(template_manager)
        
        assert persona.id == "performance"
        assert persona.name == "Performance Agent"
        assert persona.category == PersonaCategory.PRODUCTIVITY
        assert persona.icon == "⚡"
        assert PersonaCapability.FAST_RESPONSE in persona.capabilities
        assert PersonaCapability.COST_EFFICIENT in persona.capabilities

    def test_performance_persona_configuration(self):
        """Test PerformancePersona configuration."""
        template_manager = ConfigTemplateManager()
        persona = PerformancePersona(template_manager)
        
        assert persona.configuration is not None
        assert persona.configuration.template_id == "performance"
        assert persona.configuration.primary_provider == "openai"
        assert persona.configuration.timeout_override == 15


class TestGeneralPersona:
    """Test GeneralPersona class."""

    def test_general_persona_creation(self):
        """Test creating a GeneralPersona."""
        template_manager = ConfigTemplateManager()
        persona = GeneralPersona(template_manager)
        
        assert persona.id == "general"
        assert persona.name == "General Agent"
        assert persona.category == PersonaCategory.GENERAL
        assert persona.icon == "🤖"
        assert PersonaCapability.BALANCED in persona.capabilities
        assert PersonaCapability.GENERAL_PURPOSE in persona.capabilities

    def test_general_persona_configuration(self):
        """Test GeneralPersona configuration."""
        template_manager = ConfigTemplateManager()
        persona = GeneralPersona(template_manager)
        
        assert persona.configuration is not None
        assert persona.configuration.template_id == "general"
        assert persona.configuration.primary_provider == "claude"


class TestPersonaManager:
    """Test PersonaManager class."""

    def test_persona_manager_creation(self):
        """Test creating a PersonaManager."""
        manager = PersonaManager()
        
        assert len(manager.personas) == 5  # Five built-in personas
        assert "coding" in manager.personas
        assert "research" in manager.personas
        assert "creative" in manager.personas
        assert "performance" in manager.personas
        assert "general" in manager.personas

    def test_get_persona(self):
        """Test getting a persona by ID."""
        manager = PersonaManager()
        
        persona = manager.get_persona("coding")
        assert persona is not None
        assert persona.id == "coding"
        
        invalid_persona = manager.get_persona("invalid")
        assert invalid_persona is None

    def test_get_all_personas(self):
        """Test getting all personas."""
        manager = PersonaManager()
        
        personas = manager.get_all_personas()
        assert len(personas) == 5
        assert isinstance(personas, dict)

    def test_get_personas_by_category(self):
        """Test filtering personas by category."""
        manager = PersonaManager()
        
        dev_personas = manager.get_personas_by_category(PersonaCategory.DEVELOPMENT)
        assert len(dev_personas) == 1
        assert dev_personas[0].id == "coding"
        
        research_personas = manager.get_personas_by_category(PersonaCategory.RESEARCH)
        assert len(research_personas) == 1
        assert research_personas[0].id == "research"

    def test_get_available_personas(self):
        """Test getting available personas."""
        manager = PersonaManager()
        
        available = manager.get_available_personas()
        assert len(available) == 5  # All should be available initially
        
        # All personas should be available or active by default
        for persona in available:
            assert persona.status in [PersonaStatus.AVAILABLE, PersonaStatus.ACTIVE]

    @patch('omnimancer.core.agent.persona.logger')
    def test_activate_persona(self, mock_logger):
        """Test activating a persona."""
        manager = PersonaManager()
        
        # Test successful activation
        result = manager.activate_persona("coding")
        assert result is True
        assert manager.active_persona is not None
        assert manager.active_persona.id == "coding"
        assert manager.active_persona.is_active is True

    @patch('omnimancer.core.agent.persona.logger')
    def test_activate_invalid_persona(self, mock_logger):
        """Test activating an invalid persona."""
        manager = PersonaManager()
        
        result = manager.activate_persona("invalid")
        assert result is False
        assert manager.active_persona is None

    @patch('omnimancer.core.agent.persona.logger')
    def test_deactivate_persona(self, mock_logger):
        """Test deactivating a persona."""
        manager = PersonaManager()
        
        # First activate a persona
        manager.activate_persona("coding")
        assert manager.active_persona is not None
        
        # Then deactivate it
        result = manager.deactivate_persona()
        assert result is True
        assert manager.active_persona is None

    def test_deactivate_no_active_persona(self):
        """Test deactivating when no persona is active."""
        manager = PersonaManager()
        
        # Should return True when no persona is active
        result = manager.deactivate_persona()
        assert result is True

    def test_get_persona_recommendations(self):
        """Test getting persona recommendations based on context."""
        manager = PersonaManager()
        
        # Test coding context
        coding_recs = manager.get_persona_recommendations("I need help with Python code")
        assert len(coding_recs) >= 1
        assert any(p.id == "coding" for p in coding_recs)
        
        # Test research context
        research_recs = manager.get_persona_recommendations("I need to research this topic")
        assert len(research_recs) >= 1
        assert any(p.id == "research" for p in research_recs)
        
        # Test creative context
        creative_recs = manager.get_persona_recommendations("Help me write a story")
        assert len(creative_recs) >= 1
        assert any(p.id == "creative" for p in creative_recs)
        
        # Test performance context
        performance_recs = manager.get_persona_recommendations("I need fast responses")
        assert len(performance_recs) >= 1
        assert any(p.id == "performance" for p in performance_recs)
        
        # Test default context
        default_recs = manager.get_persona_recommendations("random text")
        assert len(default_recs) >= 1
        assert any(p.id == "general" for p in default_recs)

    def test_get_stats(self):
        """Test getting persona manager statistics."""
        manager = PersonaManager()
        
        stats = manager.get_stats()
        assert stats['total_personas'] == 5
        assert stats['builtin_personas'] == 5
        assert stats['custom_personas'] == 0
        assert stats['active_persona'] is None
        assert stats['available_personas'] == 5

    def test_register_custom_persona(self):
        """Test registering a custom persona."""
        manager = PersonaManager()
        
        persona_data = {
            'id': 'custom_test',
            'name': 'Custom Test Agent',
            'description': 'A test custom persona',
            'configuration': {
                'template_id': 'general',
                'primary_provider': 'claude'
            }
        }
        
        result = manager.register_custom_persona(persona_data)
        assert result is True
        assert 'custom_test' in manager.custom_personas

    def test_register_custom_persona_invalid(self):
        """Test registering invalid custom persona."""
        manager = PersonaManager()
        
        # Missing required fields
        invalid_persona_data = {
            'name': 'Invalid Persona'
        }
        
        result = manager.register_custom_persona(invalid_persona_data)
        assert result is False

    def test_event_listeners(self):
        """Test event listener functionality."""
        manager = PersonaManager()
        events_received = []
        
        def test_listener(event_type, data):
            events_received.append((event_type, data))
        
        manager.add_event_listener(test_listener)
        
        # Activate a persona to trigger events
        manager.activate_persona("coding")
        
        # Should have received persona_activated event
        assert len(events_received) > 0
        assert any(event[0] == 'persona_activated' for event in events_received)
        
        # Remove listener
        manager.remove_event_listener(test_listener)


class TestGlobalPersonaManager:
    """Test global persona manager functions."""

    def test_get_persona_manager(self):
        """Test getting global persona manager."""
        manager = get_persona_manager()
        assert isinstance(manager, PersonaManager)
        
        # Should return same instance on subsequent calls
        manager2 = get_persona_manager()
        assert manager is manager2

    def test_set_persona_manager(self):
        """Test setting global persona manager."""
        custom_manager = PersonaManager()
        set_persona_manager(custom_manager)
        
        retrieved_manager = get_persona_manager()
        assert retrieved_manager is custom_manager


class TestPersonaActivationDeactivation:
    """Test persona activation and deactivation flows."""

    @patch('omnimancer.core.agent.persona.logger')
    def test_persona_activation_flow(self, mock_logger):
        """Test complete persona activation flow."""
        template_manager = ConfigTemplateManager()
        persona = CodingPersona(template_manager)
        
        # Initial state
        assert persona.status == PersonaStatus.AVAILABLE
        assert not persona.is_active
        
        # Activate
        result = persona.activate()
        assert result is True
        assert persona.status == PersonaStatus.ACTIVE
        assert persona.is_active
        assert persona.metadata.usage_count == 1

    @patch('omnimancer.core.agent.persona.logger')
    def test_persona_deactivation_flow(self, mock_logger):
        """Test complete persona deactivation flow."""
        template_manager = ConfigTemplateManager()
        persona = CodingPersona(template_manager)
        
        # Activate first
        persona.activate()
        assert persona.is_active
        
        # Deactivate
        result = persona.deactivate()
        assert result is True
        assert persona.status == PersonaStatus.AVAILABLE
        assert not persona.is_active

    def test_persona_session_data(self):
        """Test persona session data management."""
        template_manager = ConfigTemplateManager()
        persona = CodingPersona(template_manager)
        
        # Set session data
        persona.set_session_data("test_key", "test_value")
        assert persona.get_session_data("test_key") == "test_value"
        
        # Get non-existent key with default
        assert persona.get_session_data("missing", "default") == "default"
        
        # Deactivation should clear session data
        persona.activate()
        persona.set_session_data("temp_key", "temp_value")
        persona.deactivate()
        assert persona.get_session_data("temp_key") is None

    def test_persona_dict_conversion(self):
        """Test converting persona to/from dictionary."""
        template_manager = ConfigTemplateManager()
        persona = CodingPersona(template_manager)
        
        persona_dict = persona.to_dict()
        
        # Check required fields
        assert persona_dict['id'] == "coding"
        assert persona_dict['name'] == "Coding Agent"
        assert persona_dict['category'] == PersonaCategory.DEVELOPMENT.value
        assert persona_dict['icon'] == "💻"
        assert isinstance(persona_dict['capabilities'], list)
        assert isinstance(persona_dict['configuration'], dict)
        assert isinstance(persona_dict['metadata'], dict)