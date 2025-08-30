"""
Tests for the PersonaValidator class.
"""

from unittest.mock import Mock

import pytest

from omnimancer.core.agent.persona import (
    AgentPersona,
    PersonaCapability,
    PersonaCategory,
    PersonaConfiguration,
)
from omnimancer.core.agent.persona_validator import (
    PersonaValidator,
    ValidationCategory,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    get_persona_validator,
    set_persona_validator,
)
from omnimancer.core.models import ConfigTemplate, ConfigTemplateManager
from omnimancer.core.provider_registry import ProviderRegistry


class TestValidationIssue:
    """Test ValidationIssue class."""

    def test_validation_issue_creation(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category=ValidationCategory.REQUIRED_FIELDS,
            message="Test error message",
            field_path="test.field",
            suggestion="Fix this field",
        )

        assert issue.severity == ValidationSeverity.ERROR
        assert issue.category == ValidationCategory.REQUIRED_FIELDS
        assert issue.message == "Test error message"
        assert issue.field_path == "test.field"
        assert issue.suggestion == "Fix this field"

    def test_validation_issue_to_dict(self):
        """Test converting validation issue to dictionary."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category=ValidationCategory.MODEL_AVAILABILITY,
            message="Test warning",
            details={"model": "test-model"},
        )

        result = issue.to_dict()

        assert result["severity"] == "warning"
        assert result["category"] == "model_availability"
        assert result["message"] == "Test warning"
        assert result["details"] == {"model": "test-model"}


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert len(result.warnings) == 0
        assert len(result.errors) == 0
        assert len(result.critical_issues) == 0

    def test_add_warning_issue(self):
        """Test adding a warning issue."""
        result = ValidationResult(is_valid=True)
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category=ValidationCategory.TEMPLATE_STRUCTURE,
            message="Warning message",
        )

        result.add_issue(issue)

        assert result.is_valid is True  # Warnings don't invalidate
        assert len(result.issues) == 1
        assert len(result.warnings) == 1
        assert len(result.errors) == 0

    def test_add_error_issue(self):
        """Test adding an error issue."""
        result = ValidationResult(is_valid=True)
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category=ValidationCategory.REQUIRED_FIELDS,
            message="Error message",
        )

        result.add_issue(issue)

        assert result.is_valid is False  # Errors invalidate
        assert len(result.issues) == 1
        assert len(result.errors) == 1
        assert len(result.warnings) == 0

    def test_add_critical_issue(self):
        """Test adding a critical issue."""
        result = ValidationResult(is_valid=True)
        issue = ValidationIssue(
            severity=ValidationSeverity.CRITICAL,
            category=ValidationCategory.TEMPLATE_STRUCTURE,
            message="Critical message",
        )

        result.add_issue(issue)

        assert result.is_valid is False  # Critical issues invalidate
        assert len(result.issues) == 1
        assert len(result.critical_issues) == 1

    def test_has_blocking_issues(self):
        """Test checking for blocking issues."""
        result = ValidationResult(is_valid=True)

        # No issues - not blocking
        assert result.has_blocking_issues() is False

        # Warning only - not blocking
        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.TEMPLATE_STRUCTURE,
                message="Warning",
            )
        )
        assert result.has_blocking_issues() is False

        # Error - blocking
        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.REQUIRED_FIELDS,
                message="Error",
            )
        )
        assert result.has_blocking_issues() is True

    def test_get_summary(self):
        """Test getting validation summary."""
        result = ValidationResult(is_valid=True)

        # No issues
        assert "no issues" in result.get_summary()

        # Add different types of issues
        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.TEMPLATE_STRUCTURE,
                message="Warning",
            )
        )
        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.REQUIRED_FIELDS,
                message="Error",
            )
        )

        summary = result.get_summary()
        assert "1 errors" in summary
        assert "1 warnings" in summary


class TestPersonaValidator:
    """Test PersonaValidator class."""

    @pytest.fixture
    def mock_template_manager(self):
        """Create a mock template manager."""
        manager = Mock(spec=ConfigTemplateManager)

        # Create a mock template
        template = Mock(spec=ConfigTemplate)
        template.provider_configs = {
            "claude": {"temperature": 0.7, "max_tokens": 4096},
            "openai": {"temperature": 0.8, "max_tokens": 2048},
        }
        template.recommended_models = {
            "claude": "claude-3-sonnet-20240229",
            "openai": "gpt-4",
        }

        manager.get_template = Mock(return_value=template)
        return manager

    @pytest.fixture
    def mock_provider_registry(self):
        """Create a mock provider registry."""
        registry = Mock(spec=ProviderRegistry)
        registry.get_provider_info = Mock(return_value=Mock())
        return registry

    @pytest.fixture
    def validator(self, mock_template_manager, mock_provider_registry):
        """Create a PersonaValidator instance."""
        return PersonaValidator(
            template_manager=mock_template_manager,
            provider_registry=mock_provider_registry,
        )

    @pytest.fixture
    def valid_persona(self):
        """Create a valid persona for testing."""
        persona = Mock(spec=AgentPersona)
        persona.id = "test_persona"
        persona.name = "Test Persona"
        persona.description = "A test persona"
        persona.category = PersonaCategory.GENERAL
        persona.capabilities = {PersonaCapability.TOOL_CALLING}

        # Valid configuration
        config = Mock(spec=PersonaConfiguration)
        config.template_id = "test_template"
        config.primary_provider = "claude"
        config.fallback_providers = ["openai"]
        config.tools_enabled = True
        config.temperature_override = 0.7
        config.max_tokens_override = 2048
        config.timeout_override = 30

        persona.configuration = config
        return persona

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.template_manager is not None
        assert validator.provider_registry is not None
        assert validator.required_persona_fields == {
            "id",
            "name",
            "description",
            "category",
        }
        assert validator.required_configuration_fields == {
            "template_id",
            "primary_provider",
        }

    def test_validate_valid_persona(self, validator, valid_persona):
        """Test validating a valid persona."""
        # Mock provider availability
        validator._is_provider_available = Mock(return_value=True)
        validator._is_model_available = Mock(return_value=True)

        result = validator.validate_persona(valid_persona)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.critical_issues) == 0

    def test_validate_persona_missing_required_fields(self, validator):
        """Test validating persona with missing required fields."""
        persona = Mock(spec=AgentPersona)
        persona.id = ""  # Missing required field
        persona.name = "Test"
        persona.description = ""  # Missing required field
        persona.category = PersonaCategory.GENERAL
        persona.capabilities = set()
        persona.configuration = None

        result = validator.validate_persona(persona)

        assert result.is_valid is False
        assert len(result.errors) >= 2  # Missing id, description, and configuration

        # Check specific error messages
        error_messages = [issue.message for issue in result.errors]
        assert any(
            "'id'" in msg and "missing or empty" in msg for msg in error_messages
        )
        assert any(
            "'description'" in msg and "missing or empty" in msg
            for msg in error_messages
        )

    def test_validate_persona_invalid_id(self, validator):
        """Test validating persona with invalid ID."""
        persona = Mock(spec=AgentPersona)
        persona.id = "invalid-id-with-dashes"  # Invalid identifier
        persona.name = "Test"
        persona.description = "Test description"
        persona.category = PersonaCategory.GENERAL
        persona.capabilities = set()
        persona.configuration = None

        result = validator.validate_persona(persona)

        assert result.is_valid is False
        error_messages = [issue.message for issue in result.errors]
        assert any("not a valid identifier" in msg for msg in error_messages)

    def test_validate_configuration_missing_fields(self, validator):
        """Test validating configuration with missing fields."""
        config = Mock(spec=PersonaConfiguration)
        config.template_id = ""  # Missing
        config.primary_provider = ""  # Missing
        config.temperature_override = (
            None  # Add missing attributes that validator checks
        )
        config.max_tokens_override = None
        config.timeout_override = None

        result = validator.validate_configuration(config)

        assert result.is_valid is False
        assert len(result.errors) >= 2

        error_messages = [issue.message for issue in result.errors]
        assert any("'template_id'" in msg for msg in error_messages)
        assert any("'primary_provider'" in msg for msg in error_messages)

    def test_validate_configuration_invalid_constraints(self, validator):
        """Test validating configuration with invalid constraints."""
        config = Mock(spec=PersonaConfiguration)
        config.template_id = "test"
        config.primary_provider = "claude"
        config.temperature_override = 3.0  # Invalid - too high
        config.max_tokens_override = -100  # Invalid - negative
        config.timeout_override = -5  # Invalid - negative

        result = validator.validate_configuration(config)

        assert result.is_valid is False
        assert len(result.errors) >= 3

        error_messages = [issue.message for issue in result.errors]
        assert any(
            "Temperature override" in msg and "outside valid range" in msg
            for msg in error_messages
        )
        assert any(
            "Max tokens override" in msg and "must be positive" in msg
            for msg in error_messages
        )
        assert any(
            "Timeout override" in msg and "must be positive" in msg
            for msg in error_messages
        )

    def test_validate_template_compatibility(self, validator, mock_template_manager):
        """Test validating template compatibility."""
        config = Mock(spec=PersonaConfiguration)
        config.template_id = "test_template"
        config.primary_provider = "nonexistent_provider"  # Not in template
        config.fallback_providers = ["also_nonexistent"]

        result = ValidationResult(True)
        validator._validate_template_compatibility("test_template", config, result)

        # Should add errors for incompatible providers
        error_messages = [issue.message for issue in result.errors]
        assert any("not available in template" in msg for msg in error_messages)

    def test_validate_conflicting_capabilities(self, validator):
        """Test validating conflicting capabilities."""
        capabilities = {
            PersonaCapability.FAST_RESPONSE,
            PersonaCapability.LARGE_CONTEXT,  # Conflicting with fast response
        }

        result = ValidationResult(is_valid=True)
        validator._validate_capabilities(capabilities, result)

        # Should add warning for conflicting capabilities
        warning_messages = [issue.message for issue in result.warnings]
        assert any("Conflicting capabilities" in msg for msg in warning_messages)

    def test_validate_template_not_found(self, validator):
        """Test validating non-existent template."""
        validator.template_manager.get_template = Mock(return_value=None)

        result = validator.validate_template("nonexistent_template")

        assert result.is_valid is False
        assert len(result.critical_issues) >= 1

        critical_messages = [issue.message for issue in result.critical_issues]
        assert any("not found" in msg for msg in critical_messages)

    def test_provider_availability_caching(self, validator):
        """Test provider availability caching."""
        # Mock provider registry
        validator.provider_registry.get_provider_info = Mock(return_value=Mock())

        # First call should hit the registry
        result1 = validator._is_provider_available("test_provider")
        assert validator.provider_registry.get_provider_info.call_count == 1

        # Second call should use cache
        result2 = validator._is_provider_available("test_provider")
        assert (
            validator.provider_registry.get_provider_info.call_count == 1
        )  # No additional calls

        assert result1 == result2

    def test_model_availability_basic_models(self, validator):
        """Test model availability for basic models."""
        # Known models should be available
        assert (
            validator._is_model_available("claude", "claude-3-sonnet-20240229") is True
        )
        assert validator._is_model_available("openai", "gpt-4") is True
        assert validator._is_model_available("gemini", "gemini-pro") is True

        # Unknown models should not be available
        assert validator._is_model_available("unknown", "unknown-model") is False
        assert validator._is_model_available("claude", "nonexistent-model") is False

    def test_clear_cache(self, validator):
        """Test clearing validation caches."""
        # Populate caches
        validator._provider_cache["test"] = True
        validator._model_cache["test:model"] = True

        assert len(validator._provider_cache) == 1
        assert len(validator._model_cache) == 1

        # Clear caches
        validator.clear_cache()

        assert len(validator._provider_cache) == 0
        assert len(validator._model_cache) == 0

    def test_get_validation_report_no_issues(self, validator):
        """Test generating validation report with no issues."""
        result = ValidationResult(is_valid=True)

        report = validator.get_validation_report(result)

        assert "✅" in report
        assert "no issues" in report

    def test_get_validation_report_with_issues(self, validator):
        """Test generating validation report with various issues."""
        result = ValidationResult(is_valid=False)

        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.TEMPLATE_STRUCTURE,
                message="Critical issue",
                suggestion="Fix critical issue",
            )
        )

        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.REQUIRED_FIELDS,
                message="Error issue",
                suggestion="Fix error",
            )
        )

        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.MODEL_AVAILABILITY,
                message="Warning issue",
                suggestion="Fix warning",
            )
        )

        report = validator.get_validation_report(result)

        # Check that all issue types are present
        assert "🚨 Critical Issues:" in report
        assert "❌ Errors:" in report
        assert "⚠️ Warnings:" in report

        # Check that messages and suggestions are included
        assert "Critical issue" in report
        assert "Error issue" in report
        assert "Warning issue" in report
        assert "Fix critical issue" in report
        assert "Fix error" in report
        assert "Fix warning" in report


class TestGlobalValidator:
    """Test global validator functions."""

    def test_get_persona_validator(self):
        """Test getting global validator instance."""
        validator1 = get_persona_validator()
        validator2 = get_persona_validator()

        # Should return the same instance
        assert validator1 is validator2
        assert isinstance(validator1, PersonaValidator)

    def test_set_persona_validator(self):
        """Test setting global validator instance."""
        original = get_persona_validator()

        custom_validator = PersonaValidator()
        set_persona_validator(custom_validator)

        retrieved = get_persona_validator()
        assert retrieved is custom_validator
        assert retrieved is not original

        # Reset for other tests
        set_persona_validator(original)
