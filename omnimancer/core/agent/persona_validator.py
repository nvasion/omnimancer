"""
Persona Template Validation and Application system for Omnimancer.

This module provides robust validation and application logic for agent persona
templates with comprehensive error handling and rollback capabilities.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from omnimancer.core.models import ConfigTemplate, ConfigTemplateManager

from ..provider_registry import ProviderRegistry
from .config import AgentConfig
from .persona import (
    AgentPersona,
    PersonaCapability,
    PersonaCategory,
    PersonaConfiguration,
)

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation issues."""

    TEMPLATE_STRUCTURE = "template_structure"
    REQUIRED_FIELDS = "required_fields"
    MODEL_AVAILABILITY = "model_availability"
    TOOL_COMPATIBILITY = "tool_compatibility"
    PROVIDER_COMPATIBILITY = "provider_compatibility"
    CONFIGURATION_CONSTRAINTS = "configuration_constraints"
    RESOURCE_REQUIREMENTS = "resource_requirements"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during template validation."""

    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    field_path: str = ""
    suggestion: str = ""
    fix_action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "field_path": self.field_path,
            "suggestion": self.suggestion,
            "fix_action": self.fix_action,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of template validation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    critical_issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)

        if issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        elif issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.CRITICAL:
            self.critical_issues.append(issue)
            self.is_valid = False

    def has_blocking_issues(self) -> bool:
        """Check if there are blocking issues that prevent template use."""
        return len(self.errors) > 0 or len(self.critical_issues) > 0

    def get_summary(self) -> str:
        """Get a summary of validation results."""
        if self.is_valid and not self.warnings:
            return "Template validation passed with no issues."

        summary_parts = []
        if self.critical_issues:
            summary_parts.append(f"{len(self.critical_issues)} critical issues")
        if self.errors:
            summary_parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            summary_parts.append(f"{len(self.warnings)} warnings")

        return f"Template validation found: {', '.join(summary_parts)}"


class PersonaValidator:
    """
    Validates persona templates for integrity, compatibility, and correctness.

    This class provides comprehensive validation of agent persona templates,
    checking everything from basic structure to provider compatibility.
    """

    def __init__(
        self,
        template_manager: Optional[ConfigTemplateManager] = None,
        provider_registry: Optional[ProviderRegistry] = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the validator.

        Args:
            template_manager: Configuration template manager
            provider_registry: Provider registry for compatibility checks
            agent_config: Agent configuration for validation context
        """
        self.template_manager = template_manager or ConfigTemplateManager()
        self.provider_registry = provider_registry or ProviderRegistry()
        self.agent_config = agent_config

        # Required fields for different validation levels
        self.required_persona_fields = {
            "id",
            "name",
            "description",
            "category",
        }

        self.required_configuration_fields = {
            "template_id",
            "primary_provider",
        }

        # Cache for expensive validation operations
        self._provider_cache: Dict[str, bool] = {}
        self._model_cache: Dict[str, bool] = {}

    def validate_persona(self, persona: AgentPersona) -> ValidationResult:
        """
        Validate a complete persona instance.

        Args:
            persona: The persona to validate

        Returns:
            ValidationResult with all found issues
        """
        result = ValidationResult(is_valid=True)

        # Validate basic persona structure
        self._validate_persona_structure(persona, result)

        # Validate persona configuration
        if persona.configuration:
            self._validate_persona_configuration(persona.configuration, result)

            # Validate template compatibility
            if persona.configuration.template_id:
                self._validate_template_compatibility(
                    persona.configuration.template_id,
                    persona.configuration,
                    result,
                )
        else:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.REQUIRED_FIELDS,
                    message="Persona configuration is missing",
                    field_path="configuration",
                    suggestion="Ensure persona has a valid configuration object",
                )
            )

        # Validate capabilities
        self._validate_capabilities(persona.capabilities, result)

        # Validate provider compatibility
        if persona.configuration:
            self._validate_provider_compatibility(persona.configuration, result)

        return result

    def validate_template(self, template_id: str) -> ValidationResult:
        """
        Validate a configuration template.

        Args:
            template_id: ID of the template to validate

        Returns:
            ValidationResult with all found issues
        """
        result = ValidationResult(is_valid=True)

        try:
            template = self.template_manager.get_template(template_id)
            if not template:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category=ValidationCategory.TEMPLATE_STRUCTURE,
                        message=f"Template '{template_id}' not found",
                        suggestion="Check that the template ID is correct and the template exists",
                    )
                )
                return result

            # Validate template structure
            self._validate_template_structure(template, result)

            # Validate provider configurations
            self._validate_template_providers(template, result)

            # Validate recommended models
            self._validate_template_models(template, result)

        except Exception as e:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.TEMPLATE_STRUCTURE,
                    message=f"Failed to load template '{template_id}': {str(e)}",
                    suggestion="Check template file for syntax errors or corruption",
                )
            )

        return result

    def validate_configuration(self, config: PersonaConfiguration) -> ValidationResult:
        """
        Validate a persona configuration.

        Args:
            config: The configuration to validate

        Returns:
            ValidationResult with all found issues
        """
        result = ValidationResult(is_valid=True)
        self._validate_persona_configuration(config, result)
        return result

    def _validate_persona_structure(
        self, persona: AgentPersona, result: ValidationResult
    ) -> None:
        """Validate basic persona structure."""
        for field in self.required_persona_fields:
            if not hasattr(persona, field) or not getattr(persona, field):
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.REQUIRED_FIELDS,
                        message=f"Required persona field '{field}' is missing or empty",
                        field_path=field,
                        suggestion=f"Provide a valid value for the '{field}' field",
                    )
                )

        # Validate ID format
        if hasattr(persona, "id") and persona.id:
            if not persona.id.isidentifier():
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.TEMPLATE_STRUCTURE,
                        message=f"Persona ID '{persona.id}' is not a valid identifier",
                        field_path="id",
                        suggestion="Use only letters, numbers, and underscores for persona IDs",
                    )
                )

        # Validate category
        if hasattr(persona, "category") and persona.category:
            if not isinstance(persona.category, PersonaCategory):
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.TEMPLATE_STRUCTURE,
                        message=f"Persona category should be a PersonaCategory enum",
                        field_path="category",
                        suggestion="Use PersonaCategory enum values",
                    )
                )

    def _validate_persona_configuration(
        self, config: PersonaConfiguration, result: ValidationResult
    ) -> None:
        """Validate persona configuration."""
        for field in self.required_configuration_fields:
            if not hasattr(config, field) or not getattr(config, field):
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.REQUIRED_FIELDS,
                        message=f"Required configuration field '{field}' is missing or empty",
                        field_path=f"configuration.{field}",
                        suggestion=f"Provide a valid value for the '{field}' field",
                    )
                )

        # Validate numeric constraints
        if (
            hasattr(config, "temperature_override")
            and config.temperature_override is not None
        ):
            if not 0.0 <= config.temperature_override <= 2.0:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.CONFIGURATION_CONSTRAINTS,
                        message=f"Temperature override {config.temperature_override} is outside valid range (0.0-2.0)",
                        field_path="configuration.temperature_override",
                        suggestion="Set temperature between 0.0 and 2.0",
                    )
                )

        if (
            hasattr(config, "max_tokens_override")
            and config.max_tokens_override is not None
        ):
            if config.max_tokens_override <= 0:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.CONFIGURATION_CONSTRAINTS,
                        message=f"Max tokens override {config.max_tokens_override} must be positive",
                        field_path="configuration.max_tokens_override",
                        suggestion="Set max tokens to a positive integer",
                    )
                )

        if hasattr(config, "timeout_override") and config.timeout_override is not None:
            if config.timeout_override <= 0:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.CONFIGURATION_CONSTRAINTS,
                        message=f"Timeout override {config.timeout_override} must be positive",
                        field_path="configuration.timeout_override",
                        suggestion="Set timeout to a positive number of seconds",
                    )
                )

    def _validate_template_compatibility(
        self,
        template_id: str,
        config: PersonaConfiguration,
        result: ValidationResult,
    ) -> None:
        """Validate that configuration is compatible with its template."""
        try:
            template = self.template_manager.get_template(template_id)
            if not template:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.TEMPLATE_STRUCTURE,
                        message=f"Referenced template '{template_id}' not found",
                        field_path="configuration.template_id",
                        suggestion="Check that the template ID is correct",
                    )
                )
                return

            # Check if primary provider exists in template
            if config.primary_provider not in template.provider_configs:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.PROVIDER_COMPATIBILITY,
                        message=f"Primary provider '{config.primary_provider}' not available in template '{template_id}'",
                        field_path="configuration.primary_provider",
                        suggestion=f"Use one of: {', '.join(template.provider_configs.keys())}",
                    )
                )

            # Check fallback providers
            if hasattr(config, "fallback_providers") and config.fallback_providers:
                for provider in config.fallback_providers:
                    if provider not in template.provider_configs:
                        result.add_issue(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category=ValidationCategory.PROVIDER_COMPATIBILITY,
                                message=f"Fallback provider '{provider}' not available in template '{template_id}'",
                                field_path="configuration.fallback_providers",
                                suggestion=f"Remove '{provider}' or use one of: {', '.join(template.provider_configs.keys())}",
                            )
                        )

        except Exception as e:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.TEMPLATE_STRUCTURE,
                    message=f"Could not validate template compatibility: {str(e)}",
                    field_path="configuration.template_id",
                )
            )

    def _validate_capabilities(
        self, capabilities: Set[PersonaCapability], result: ValidationResult
    ) -> None:
        """Validate persona capabilities."""
        if not capabilities:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.TEMPLATE_STRUCTURE,
                    message="Persona has no defined capabilities",
                    field_path="capabilities",
                    suggestion="Define at least one capability for the persona",
                )
            )

        # Check for conflicting capabilities
        conflicting_pairs = [
            (PersonaCapability.FAST_RESPONSE, PersonaCapability.LARGE_CONTEXT),
            (
                PersonaCapability.COST_EFFICIENT,
                PersonaCapability.HIGH_TEMPERATURE,
            ),
        ]

        for cap1, cap2 in conflicting_pairs:
            if cap1 in capabilities and cap2 in capabilities:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.CONFIGURATION_CONSTRAINTS,
                        message=f"Conflicting capabilities: {cap1.value} and {cap2.value}",
                        field_path="capabilities",
                        suggestion=f"Consider removing one of the conflicting capabilities",
                    )
                )

    def _validate_provider_compatibility(
        self, config: PersonaConfiguration, result: ValidationResult
    ) -> None:
        """Validate provider compatibility."""
        # Check if primary provider is available
        if not self._is_provider_available(config.primary_provider):
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.PROVIDER_COMPATIBILITY,
                    message=f"Primary provider '{config.primary_provider}' is not available",
                    field_path="configuration.primary_provider",
                    suggestion="Check provider installation and API key configuration",
                    fix_action=f"install_{config.primary_provider}_provider",
                )
            )

        # Check fallback providers
        if hasattr(config, "fallback_providers") and config.fallback_providers:
            unavailable_providers = []
            for provider in config.fallback_providers:
                if not self._is_provider_available(provider):
                    unavailable_providers.append(provider)

            if unavailable_providers:
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.PROVIDER_COMPATIBILITY,
                        message=f"Fallback providers not available: {', '.join(unavailable_providers)}",
                        field_path="configuration.fallback_providers",
                        suggestion="Remove unavailable providers or install/configure them",
                        details={"unavailable_providers": unavailable_providers},
                    )
                )

    def _validate_template_structure(
        self, template: ConfigTemplate, result: ValidationResult
    ) -> None:
        """Validate template structure."""
        if not template.provider_configs:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.TEMPLATE_STRUCTURE,
                    message="Template has no provider configurations",
                    suggestion="Add at least one provider configuration to the template",
                )
            )

    def _validate_template_providers(
        self, template: ConfigTemplate, result: ValidationResult
    ) -> None:
        """Validate template provider configurations."""
        for provider_id, provider_config in template.provider_configs.items():
            if not isinstance(provider_config, dict):
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.TEMPLATE_STRUCTURE,
                        message=f"Provider config for '{provider_id}' is not a dictionary",
                        field_path=f"provider_configs.{provider_id}",
                        suggestion="Ensure provider configuration is a valid dictionary",
                    )
                )
                continue

            # Check for required provider config fields
            required_fields = ["temperature", "max_tokens"]
            for field in required_fields:
                if field not in provider_config:
                    result.add_issue(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category=ValidationCategory.TEMPLATE_STRUCTURE,
                            message=f"Provider '{provider_id}' missing recommended field '{field}'",
                            field_path=f"provider_configs.{provider_id}.{field}",
                            suggestion=f"Add '{field}' to provider configuration",
                        )
                    )

    def _validate_template_models(
        self, template: ConfigTemplate, result: ValidationResult
    ) -> None:
        """Validate template recommended models."""
        if (
            not hasattr(template, "recommended_models")
            or not template.recommended_models
        ):
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.TEMPLATE_STRUCTURE,
                    message="Template has no recommended models",
                    field_path="recommended_models",
                    suggestion="Add recommended models for better user experience",
                )
            )
            return

        for provider, model in template.recommended_models.items():
            if not self._is_model_available(provider, model):
                result.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.MODEL_AVAILABILITY,
                        message=f"Recommended model '{model}' for provider '{provider}' may not be available",
                        field_path=f"recommended_models.{provider}",
                        suggestion="Verify model availability or update recommendation",
                    )
                )

    def _is_provider_available(self, provider_id: str) -> bool:
        """Check if a provider is available."""
        if provider_id in self._provider_cache:
            return self._provider_cache[provider_id]

        try:
            # Check if provider is registered
            provider_info = self.provider_registry.get_provider_info(provider_id)
            available = provider_info is not None

            # Additional checks could be added here (API keys, installation, etc.)

            self._provider_cache[provider_id] = available
            return available
        except Exception:
            self._provider_cache[provider_id] = False
            return False

    def _is_model_available(self, provider_id: str, model_id: str) -> bool:
        """Check if a model is available for a provider."""
        cache_key = f"{provider_id}:{model_id}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        try:
            # This would need integration with actual model availability checking
            # For now, assume basic models are available
            basic_models = {
                "claude": [
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                ],
                "openai": ["gpt-4", "gpt-3.5-turbo"],
                "gemini": ["gemini-pro"],
                "perplexity": ["pplx-7b-online", "pplx-70b-online"],
            }

            available = (
                provider_id in basic_models and model_id in basic_models[provider_id]
            )

            self._model_cache[cache_key] = available
            return available
        except Exception:
            self._model_cache[cache_key] = False
            return False

    def clear_cache(self) -> None:
        """Clear validation caches."""
        self._provider_cache.clear()
        self._model_cache.clear()

    def get_validation_report(self, result: ValidationResult) -> str:
        """Generate a formatted validation report."""
        if not result.issues:
            return "✅ Validation passed with no issues."

        report_lines = [f"📋 Validation Report: {result.get_summary()}", ""]

        if result.critical_issues:
            report_lines.append("🚨 Critical Issues:")
            for issue in result.critical_issues:
                report_lines.append(f"  • {issue.message}")
                if issue.suggestion:
                    report_lines.append(f"    💡 {issue.suggestion}")
            report_lines.append("")

        if result.errors:
            report_lines.append("❌ Errors:")
            for issue in result.errors:
                report_lines.append(f"  • {issue.message}")
                if issue.suggestion:
                    report_lines.append(f"    💡 {issue.suggestion}")
            report_lines.append("")

        if result.warnings:
            report_lines.append("⚠️ Warnings:")
            for issue in result.warnings:
                report_lines.append(f"  • {issue.message}")
                if issue.suggestion:
                    report_lines.append(f"    💡 {issue.suggestion}")
            report_lines.append("")

        return "\n".join(report_lines)


# Global validator instance
_global_validator: Optional[PersonaValidator] = None


def get_persona_validator() -> PersonaValidator:
    """Get the global persona validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = PersonaValidator()
    return _global_validator


def set_persona_validator(validator: PersonaValidator) -> None:
    """Set the global persona validator instance."""
    global _global_validator
    _global_validator = validator
