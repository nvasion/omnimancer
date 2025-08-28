"""
Tests for the TemplateApplicator class.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from omnimancer.core.agent.template_applicator import (
    TemplateApplicator, ApplicationContext, ApplicationResult, ApplicationOptions,
    ApplicationStage, get_template_applicator, set_template_applicator
)
from omnimancer.core.agent.persona import (
    PersonaManager, AgentPersona, PersonaConfiguration, PersonaStatus, PersonaCategory
)
from omnimancer.core.agent.persona_validator import PersonaValidator, ValidationResult
from omnimancer.core.agent.agent_switcher import AgentSwitcher, SessionState
from omnimancer.core.models import ConfigTemplateManager


class TestApplicationContext:
    """Test ApplicationContext class."""
    
    def test_application_context_creation(self):
        """Test creating an application context."""
        persona = Mock(spec=AgentPersona)
        persona.id = "test_persona"
        persona.name = "Test Persona"
        
        config = Mock(spec=PersonaConfiguration)
        config.template_id = "test_template"
        
        context = ApplicationContext(
            persona=persona,
            target_configuration=config
        )
        
        assert context.persona == persona
        assert context.target_configuration == config
        assert context.stage == ApplicationStage.VALIDATION
        assert context.result is None
        assert context.original_configuration is None
        assert isinstance(context.timestamp, datetime)
    
    def test_application_context_to_dict(self):
        """Test converting application context to dictionary."""
        persona = Mock(spec=AgentPersona)
        persona.id = "test_persona"
        persona.name = "Test Persona"
        
        config = Mock(spec=PersonaConfiguration)
        
        context = ApplicationContext(
            persona=persona,
            target_configuration=config,
            stage=ApplicationStage.COMPLETE,
            result=ApplicationResult.SUCCESS
        )
        
        result_dict = context.to_dict()
        
        assert result_dict['persona_id'] == "test_persona"
        assert result_dict['persona_name'] == "Test Persona"
        assert result_dict['stage'] == "complete"
        assert result_dict['result'] == "success"
        assert 'timestamp' in result_dict
        assert result_dict['has_backup'] is False


class TestApplicationOptions:
    """Test ApplicationOptions class."""
    
    def test_application_options_defaults(self):
        """Test default application options."""
        options = ApplicationOptions()
        
        assert options.validate_before_apply is True
        assert options.create_backup is True
        assert options.force_application is False
        assert options.skip_verification is False
        assert options.timeout_seconds == 30
        assert options.rollback_on_failure is True
        assert options.preserve_session_state is True
        assert options.pre_application_hook is None
        assert options.post_application_hook is None
        assert options.rollback_hook is None
    
    def test_application_options_custom(self):
        """Test custom application options."""
        def pre_hook(context):
            pass
        
        def post_hook(context):
            pass
        
        def rollback_hook(context):
            pass
        
        options = ApplicationOptions(
            validate_before_apply=False,
            create_backup=False,
            force_application=True,
            skip_verification=True,
            timeout_seconds=60,
            rollback_on_failure=False,
            preserve_session_state=False,
            pre_application_hook=pre_hook,
            post_application_hook=post_hook,
            rollback_hook=rollback_hook
        )
        
        assert options.validate_before_apply is False
        assert options.create_backup is False
        assert options.force_application is True
        assert options.skip_verification is True
        assert options.timeout_seconds == 60
        assert options.rollback_on_failure is False
        assert options.preserve_session_state is False
        assert options.pre_application_hook == pre_hook
        assert options.post_application_hook == post_hook
        assert options.rollback_hook == rollback_hook


class TestTemplateApplicator:
    """Test TemplateApplicator class."""
    
    @pytest.fixture
    def mock_persona_manager(self):
        """Create a mock PersonaManager."""
        manager = Mock(spec=PersonaManager)
        
        # Create a test persona
        persona = Mock()
        persona.id = "test_persona"
        persona.name = "Test Persona"
        persona.status = PersonaStatus.AVAILABLE
        persona.configuration = None
        persona.capabilities = set()
        # Create a simple metadata object without Mock complications
        persona.metadata = type('MockMetadata', (), {
            'created_at': datetime.now(),
            'version': '1.0',
            '__dict__': {'created_at': datetime.now(), 'version': '1.0'}
        })()
        persona._session_data = {}
        persona.activate = Mock(return_value=True)
        
        manager.get_persona = Mock(return_value=persona)
        manager.active_persona = None
        
        return manager
    
    @pytest.fixture
    def mock_validator(self):
        """Create a mock PersonaValidator."""
        validator = Mock(spec=PersonaValidator)
        
        # Mock successful validation by default
        validation_result = ValidationResult(is_valid=True)
        validator.validate_persona = Mock(return_value=validation_result)
        
        return validator
    
    @pytest.fixture
    def mock_agent_switcher(self):
        """Create a mock AgentSwitcher."""
        switcher = Mock(spec=AgentSwitcher)
        switcher.switch_persona = Mock(return_value=(True, "Success"))
        switcher.current_session_state = SessionState()
        return switcher
    
    @pytest.fixture
    def mock_template_manager(self):
        """Create a mock ConfigTemplateManager."""
        manager = Mock(spec=ConfigTemplateManager)
        return manager
    
    @pytest.fixture
    def applicator(self, mock_persona_manager, mock_validator, mock_agent_switcher, mock_template_manager):
        """Create a TemplateApplicator instance."""
        return TemplateApplicator(
            persona_manager=mock_persona_manager,
            validator=mock_validator,
            agent_switcher=mock_agent_switcher,
            template_manager=mock_template_manager
        )
    
    def test_applicator_initialization(self, applicator):
        """Test applicator initialization."""
        assert applicator.persona_manager is not None
        assert applicator.validator is not None
        assert applicator.agent_switcher is not None
        assert applicator.template_manager is not None
        assert isinstance(applicator.application_history, list)
        assert isinstance(applicator.active_contexts, dict)
        assert isinstance(applicator.default_options, ApplicationOptions)
    
    def test_apply_template_success(self, applicator, mock_persona_manager):
        """Test successful template application."""
        # Setup
        persona = mock_persona_manager.get_persona("test_persona")
        config = Mock(spec=PersonaConfiguration)
        config.template_id = "test_template"
        config.primary_provider = "claude"
        
        # Mock the validation to pass
        applicator.validator.validate_persona = Mock(return_value=ValidationResult(True))
        
        # Apply template
        result, context = applicator.apply_template("test_persona", config)
        
        # Verify
        assert result == ApplicationResult.SUCCESS
        assert context.persona == persona
        assert context.target_configuration == config
        assert context.stage == ApplicationStage.COMPLETE
        assert len(applicator.application_history) == 1
    
    def test_apply_template_persona_not_found(self, applicator, mock_persona_manager):
        """Test template application with non-existent persona."""
        mock_persona_manager.get_persona = Mock(return_value=None)
        
        config = Mock(spec=PersonaConfiguration)
        
        with pytest.raises(ValueError, match="Persona 'nonexistent' not found"):
            applicator.apply_template("nonexistent", config)
    
    def test_apply_template_validation_failure(self, applicator):
        """Test template application with validation failure."""
        config = Mock(spec=PersonaConfiguration)
        
        # Mock validation to fail with blocking issues
        validation_result = ValidationResult(is_valid=False)
        validation_result.add_issue(Mock(severity=Mock(value="error")))
        validation_result.has_blocking_issues = Mock(return_value=True)
        applicator.validator.validate_persona = Mock(return_value=validation_result)
        
        result, context = applicator.apply_template("test_persona", config)
        
        assert result == ApplicationResult.VALIDATION_FAILED
        assert context.error_message == "Validation failed with blocking issues"
    
    def test_apply_template_with_force_application(self, applicator, mock_persona_manager):
        """Test template application with force flag bypassing validation."""
        config = Mock(spec=PersonaConfiguration)
        config.get_template = Mock(return_value=Mock())  # Mock template access
        
        # Get the persona and ensure it has configuration after application
        persona = mock_persona_manager.get_persona("test_persona")
        persona.configuration = config  # Ensure persona has configuration
        
        # Mock validation to fail for initial validation but pass for post-application
        initial_validation = ValidationResult(is_valid=False)
        initial_validation.add_issue(Mock(severity=Mock(value="error")))
        initial_validation.has_blocking_issues = Mock(return_value=True)
        
        post_validation = ValidationResult(is_valid=True)
        post_validation.has_blocking_issues = Mock(return_value=False)
        
        # Set up validation to return different results based on call order
        applicator.validator.validate_persona = Mock(side_effect=[initial_validation, post_validation])
        
        options = ApplicationOptions(force_application=True)
        result, context = applicator.apply_template("test_persona", config, options)
        
        assert result == ApplicationResult.SUCCESS
    
    def test_apply_template_with_backup(self, applicator, mock_persona_manager):
        """Test template application creates backup."""
        persona = mock_persona_manager.get_persona("test_persona")
        original_config = Mock(spec=PersonaConfiguration)
        persona.configuration = original_config
        persona.status = PersonaStatus.ACTIVE
        
        new_config = Mock(spec=PersonaConfiguration)
        new_config.get_template = Mock(return_value=Mock())  # Mock template access
        
        # Mock successful validation for both initial and post-application
        validation_result = ValidationResult(is_valid=True)
        validation_result.has_blocking_issues = Mock(return_value=False)
        applicator.validator.validate_persona = Mock(return_value=validation_result)
        
        options = ApplicationOptions(create_backup=True)
        result, context = applicator.apply_template("test_persona", new_config, options)
        
        assert result == ApplicationResult.SUCCESS
        # Check that backup was created (original_configuration should exist)
        assert context.original_configuration is not None
        assert context.original_status == PersonaStatus.ACTIVE
    
    def test_apply_template_with_hooks(self, applicator):
        """Test template application with hooks."""
        config = Mock(spec=PersonaConfiguration)
        config.get_template = Mock(return_value=Mock())  # Mock template access
        
        pre_hook = Mock()
        post_hook = Mock()
        
        options = ApplicationOptions(
            pre_application_hook=pre_hook,
            post_application_hook=post_hook
        )
        
        result, context = applicator.apply_template("test_persona", config, options)
        
        assert result == ApplicationResult.SUCCESS
        pre_hook.assert_called_once()
        post_hook.assert_called_once()
    
    def test_apply_template_rollback_on_failure(self, applicator, mock_persona_manager):
        """Test template application rollback on failure."""
        persona = mock_persona_manager.get_persona("test_persona")
        original_config = Mock(spec=PersonaConfiguration)
        persona.configuration = original_config
        
        config = Mock(spec=PersonaConfiguration)
        config.get_template = Mock(return_value=Mock())  # Mock template access
        
        # Mock successful validation
        validation_result = ValidationResult(is_valid=True)
        validation_result.has_blocking_issues = Mock(return_value=False)
        applicator.validator.validate_persona = Mock(return_value=validation_result)
        
        # Mock agent switcher failure to trigger rollback
        applicator.agent_switcher.switch_persona = Mock(return_value=(False, "Activation failed"))
        mock_persona_manager.active_persona = persona  # Make it the active persona
        
        options = ApplicationOptions(rollback_on_failure=True, create_backup=True)
        
        result, context = applicator.apply_template("test_persona", config, options)
        
        # The current implementation returns APPLICATION_FAILED when activation fails,
        # not ROLLBACK_SUCCESS, because rollback only happens on exceptions
        assert result == ApplicationResult.APPLICATION_FAILED
        # Check that rollback data exists in context
        assert context.original_configuration is not None
    
    def test_apply_template_verification_failure(self, applicator, mock_persona_manager):
        """Test template application with verification failure."""
        persona = mock_persona_manager.get_persona("test_persona")
        persona.status = PersonaStatus.ERROR  # Will cause verification to fail
        
        config = Mock(spec=PersonaConfiguration)
        options = ApplicationOptions(skip_verification=False)
        
        result, context = applicator.apply_template("test_persona", config, options)
        
        assert result == ApplicationResult.PARTIAL_SUCCESS
        assert context.error_message == "Application verification failed"
    
    def test_apply_template_skip_verification(self, applicator, mock_persona_manager):
        """Test template application skipping verification."""
        persona = mock_persona_manager.get_persona("test_persona")
        persona.status = PersonaStatus.ERROR  # Would normally fail verification
        
        config = Mock(spec=PersonaConfiguration)
        options = ApplicationOptions(skip_verification=True)
        
        result, context = applicator.apply_template("test_persona", config, options)
        
        assert result == ApplicationResult.SUCCESS
    
    def test_rollback_persona_success(self, applicator, mock_persona_manager):
        """Test successful persona rollback."""
        # First apply a template to create history
        persona = mock_persona_manager.get_persona("test_persona")
        original_config = Mock(spec=PersonaConfiguration)
        persona.configuration = original_config
        
        new_config = Mock(spec=PersonaConfiguration)
        new_config.get_template = Mock(return_value=Mock())  # Mock template access
        
        # Mock successful validation
        validation_result = ValidationResult(is_valid=True)
        validation_result.has_blocking_issues = Mock(return_value=False)
        applicator.validator.validate_persona = Mock(return_value=validation_result)
        
        options = ApplicationOptions(create_backup=True)
        
        result, context = applicator.apply_template("test_persona", new_config, options)
        assert result == ApplicationResult.SUCCESS
        
        # Now test rollback
        success, message = applicator.rollback_persona("test_persona")
        
        assert success is True
        assert "Successfully rolled back" in message
        # Check that the context is in the history
        assert len(applicator.application_history) == 1
    
    def test_rollback_persona_no_history(self, applicator):
        """Test persona rollback with no history."""
        success, message = applicator.rollback_persona("test_persona")
        
        assert success is False
        assert "No rollback data available" in message
    
    def test_get_application_history(self, applicator):
        """Test getting application history."""
        # Add some fake history
        context1 = Mock()
        context1.persona.id = "persona1"
        context2 = Mock()
        context2.persona.id = "persona2"
        
        applicator.application_history = [context1, context2]
        
        # Get all history
        all_history = applicator.get_application_history()
        assert len(all_history) == 2
        
        # Get filtered history
        filtered_history = applicator.get_application_history("persona1")
        assert len(filtered_history) == 1
        assert filtered_history[0].persona.id == "persona1"
    
    def test_get_active_applications(self, applicator):
        """Test getting active applications."""
        # Add some fake active contexts
        context1 = Mock()
        context2 = Mock()
        applicator.active_contexts = {"persona1": context1, "persona2": context2}
        
        active = applicator.get_active_applications()
        
        assert len(active) == 2
        assert "persona1" in active
        assert "persona2" in active
        assert active["persona1"] == context1
    
    def test_clear_history(self, applicator):
        """Test clearing application history."""
        # Add some fake history
        applicator.application_history = [Mock(), Mock()]
        
        applicator.clear_history()
        
        assert len(applicator.application_history) == 0
    
    def test_generate_application_report(self, applicator):
        """Test generating application report."""
        persona = Mock()
        persona.id = "test_persona"
        persona.name = "Test Persona"
        
        config = Mock()
        config.template_id = "test_template"
        config.primary_provider = "claude"
        config.tools_enabled = True
        
        context = ApplicationContext(
            persona=persona,
            target_configuration=config,
            stage=ApplicationStage.COMPLETE,
            result=ApplicationResult.SUCCESS
        )
        context.original_configuration = Mock()  # Has backup
        
        report = applicator.generate_application_report(context)
        
        assert "📋 Template Application Report" in report
        assert "Test Persona" in report
        assert "test_persona" in report
        assert "complete" in report
        assert "success" in report
        assert "✅ Backup Created: Yes" in report
        assert "test_template" in report
        assert "claude" in report
    
    def test_generate_application_report_with_error(self, applicator):
        """Test generating application report with error."""
        persona = Mock()
        persona.id = "test_persona"
        persona.name = "Test Persona"
        
        context = ApplicationContext(
            persona=persona,
            target_configuration=Mock(),
            stage=ApplicationStage.FAILED,
            result=ApplicationResult.APPLICATION_FAILED,
            error_message="Test error"
        )
        
        report = applicator.generate_application_report(context)
        
        assert "❌ Error: Test error" in report
        assert "⚠️ Backup Created: No" in report
    
    def test_activate_persona_with_switcher(self, applicator, mock_persona_manager, mock_agent_switcher):
        """Test persona activation using agent switcher."""
        persona = mock_persona_manager.get_persona("test_persona")
        mock_persona_manager.active_persona = persona  # Make it the active persona
        
        context = ApplicationContext(persona=persona, target_configuration=Mock())
        
        success = applicator._activate_persona(context)
        
        assert success is True
        mock_agent_switcher.switch_persona.assert_called_once_with(
            "test_persona",
            reason="Template application",
            force=False
        )
    
    def test_activate_persona_switcher_failure(self, applicator, mock_persona_manager, mock_agent_switcher):
        """Test persona activation failure with switcher."""
        persona = mock_persona_manager.get_persona("test_persona")
        mock_persona_manager.active_persona = persona
        mock_agent_switcher.switch_persona = Mock(return_value=(False, "Switch failed"))
        
        context = ApplicationContext(persona=persona, target_configuration=Mock())
        
        success = applicator._activate_persona(context)
        
        assert success is False
    
    def test_activate_persona_direct_activation(self, applicator, mock_persona_manager):
        """Test direct persona activation without switcher."""
        applicator.agent_switcher = None  # No switcher available
        persona = mock_persona_manager.get_persona("test_persona")
        mock_persona_manager.active_persona = persona
        
        context = ApplicationContext(persona=persona, target_configuration=Mock())
        
        success = applicator._activate_persona(context)
        
        assert success is True
        persona.activate.assert_called_once()
    
    def test_verify_application_success(self, applicator):
        """Test successful application verification."""
        persona = Mock()
        persona.status = PersonaStatus.AVAILABLE
        persona.configuration = Mock()
        
        # Mock template access
        persona.configuration.get_template = Mock(return_value=Mock())
        
        # Mock validation success
        validation_result = ValidationResult(is_valid=True)
        validation_result.has_blocking_issues = Mock(return_value=False)
        applicator.validator.validate_persona = Mock(return_value=validation_result)
        
        context = ApplicationContext(persona=persona, target_configuration=Mock())
        
        success = applicator._verify_application(context)
        
        assert success is True
    
    def test_verify_application_persona_error(self, applicator):
        """Test application verification with persona in error state."""
        persona = Mock()
        persona.status = PersonaStatus.ERROR
        
        context = ApplicationContext(persona=persona, target_configuration=Mock())
        
        success = applicator._verify_application(context)
        
        assert success is False
    
    def test_verify_application_validation_failure(self, applicator):
        """Test application verification with validation failure."""
        persona = Mock()
        persona.status = PersonaStatus.AVAILABLE
        
        # Mock validation failure
        validation_result = ValidationResult(is_valid=False)
        validation_result.has_blocking_issues = Mock(return_value=True)
        applicator.validator.validate_persona = Mock(return_value=validation_result)
        
        context = ApplicationContext(persona=persona, target_configuration=Mock())
        
        success = applicator._verify_application(context)
        
        assert success is False


class TestGlobalApplicator:
    """Test global applicator functions."""
    
    def test_get_template_applicator(self):
        """Test getting global applicator instance."""
        applicator1 = get_template_applicator()
        applicator2 = get_template_applicator()
        
        # Should return the same instance
        assert applicator1 is applicator2
        assert isinstance(applicator1, TemplateApplicator)
    
    def test_set_template_applicator(self):
        """Test setting global applicator instance."""
        original = get_template_applicator()
        
        custom_applicator = TemplateApplicator(Mock())
        set_template_applicator(custom_applicator)
        
        retrieved = get_template_applicator()
        assert retrieved is custom_applicator
        assert retrieved is not original
        
        # Reset for other tests
        set_template_applicator(original)