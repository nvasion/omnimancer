"""
Template Applicator for safely applying persona configurations.

This module provides safe application of persona templates with rollback
capabilities and comprehensive error handling.
"""

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from omnimancer.core.models import ConfigTemplateManager

from .agent_switcher import AgentSwitcher, SessionState
from .config import AgentConfig
from .persona import (
    AgentPersona,
    PersonaConfiguration,
    PersonaManager,
    PersonaStatus,
)
from .persona_validator import (
    PersonaValidator,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ApplicationStage(Enum):
    """Stages of template application."""

    VALIDATION = "validation"
    BACKUP = "backup"
    CONFIGURATION = "configuration"
    ACTIVATION = "activation"
    VERIFICATION = "verification"
    COMPLETE = "complete"
    ROLLBACK = "rollback"
    FAILED = "failed"


class ApplicationResult(Enum):
    """Result of template application."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    VALIDATION_FAILED = "validation_failed"
    APPLICATION_FAILED = "application_failed"
    ROLLBACK_SUCCESS = "rollback_success"
    ROLLBACK_FAILED = "rollback_failed"


@dataclass
class ApplicationContext:
    """Context for template application operation."""

    persona: AgentPersona
    target_configuration: PersonaConfiguration
    original_configuration: Optional[PersonaConfiguration] = None
    original_status: Optional[PersonaStatus] = None
    session_backup: Optional[SessionState] = None
    stage: ApplicationStage = ApplicationStage.VALIDATION
    result: Optional[ApplicationResult] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    rollback_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "persona_id": self.persona.id,
            "persona_name": self.persona.name,
            "stage": self.stage.value,
            "result": self.result.value if self.result else None,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "has_backup": self.original_configuration is not None,
        }


@dataclass
class ApplicationOptions:
    """Options for template application."""

    validate_before_apply: bool = True
    create_backup: bool = True
    force_application: bool = False
    skip_verification: bool = False
    timeout_seconds: int = 30
    rollback_on_failure: bool = True
    preserve_session_state: bool = True

    # Hooks for custom behavior
    pre_application_hook: Optional[Callable] = None
    post_application_hook: Optional[Callable] = None
    rollback_hook: Optional[Callable] = None


class TemplateApplicator:
    """
    Safely applies persona template configurations with rollback capabilities.

    This class provides a robust system for applying persona configurations
    with comprehensive error handling, validation, and rollback mechanisms.
    """

    def __init__(
        self,
        persona_manager: PersonaManager,
        validator: Optional[PersonaValidator] = None,
        agent_switcher: Optional[AgentSwitcher] = None,
        template_manager: Optional[ConfigTemplateManager] = None,
        agent_config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the template applicator.

        Args:
            persona_manager: PersonaManager for managing personas
            validator: PersonaValidator for validation
            agent_switcher: AgentSwitcher for safe switching
            template_manager: ConfigTemplateManager for templates
            agent_config: AgentConfig for system integration
        """
        self.persona_manager = persona_manager
        self.validator = validator or PersonaValidator()
        self.agent_switcher = agent_switcher
        self.template_manager = template_manager or ConfigTemplateManager()
        self.agent_config = agent_config

        # Application history
        self.application_history: List[ApplicationContext] = []

        # Active application contexts (for monitoring)
        self.active_contexts: Dict[str, ApplicationContext] = {}

        # Default options
        self.default_options = ApplicationOptions()

    def apply_template(
        self,
        persona_id: str,
        configuration: PersonaConfiguration,
        options: Optional[ApplicationOptions] = None,
    ) -> Tuple[ApplicationResult, ApplicationContext]:
        """
        Apply a template configuration to a persona.

        Args:
            persona_id: ID of the persona to update
            configuration: New configuration to apply
            options: Application options

        Returns:
            Tuple of (result, context)
        """
        options = options or self.default_options

        # Get the persona
        persona = self.persona_manager.get_persona(persona_id)
        if not persona:
            raise ValueError(f"Persona '{persona_id}' not found")

        # Create application context
        context = ApplicationContext(
            persona=persona, target_configuration=configuration
        )

        # Add to active contexts
        self.active_contexts[persona_id] = context

        try:
            # Execute application stages
            context.result = self._execute_application_pipeline(context, options)

            # Add to history
            self.application_history.append(context)

            logger.info(
                f"Template application completed for persona {persona_id}: {context.result.value}"
            )
            return context.result, context

        except Exception as e:
            logger.error(f"Template application failed for persona {persona_id}: {e}")
            context.error_message = str(e)
            context.result = ApplicationResult.APPLICATION_FAILED

            # Attempt rollback if enabled
            if options.rollback_on_failure:
                try:
                    self._rollback_application(context, options)
                    context.result = ApplicationResult.ROLLBACK_SUCCESS
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
                    context.result = ApplicationResult.ROLLBACK_FAILED

            self.application_history.append(context)
            return context.result, context

        finally:
            # Remove from active contexts
            if persona_id in self.active_contexts:
                del self.active_contexts[persona_id]

    def _execute_application_pipeline(
        self, context: ApplicationContext, options: ApplicationOptions
    ) -> ApplicationResult:
        """Execute the application pipeline stages."""

        # Stage 1: Validation
        if options.validate_before_apply:
            context.stage = ApplicationStage.VALIDATION
            validation_result = self._validate_application(context)

            if (
                validation_result.has_blocking_issues()
                and not options.force_application
            ):
                context.error_message = "Validation failed with blocking issues"
                return ApplicationResult.VALIDATION_FAILED

        # Stage 2: Backup
        if options.create_backup:
            context.stage = ApplicationStage.BACKUP
            self._create_backup(context, options)

        # Stage 3: Pre-application hook
        if options.pre_application_hook:
            try:
                options.pre_application_hook(context)
            except Exception as e:
                logger.warning(f"Pre-application hook failed: {e}")

        # Stage 4: Configuration
        context.stage = ApplicationStage.CONFIGURATION
        self._apply_configuration(context)

        # Stage 5: Activation
        context.stage = ApplicationStage.ACTIVATION
        activation_success = self._activate_persona(context)

        if not activation_success:
            context.error_message = "Failed to activate persona with new configuration"
            return ApplicationResult.APPLICATION_FAILED

        # Stage 6: Verification
        if not options.skip_verification:
            context.stage = ApplicationStage.VERIFICATION
            verification_success = self._verify_application(context)

            if not verification_success:
                context.error_message = "Application verification failed"
                return ApplicationResult.PARTIAL_SUCCESS

        # Stage 7: Post-application hook
        if options.post_application_hook:
            try:
                options.post_application_hook(context)
            except Exception as e:
                logger.warning(f"Post-application hook failed: {e}")

        # Stage 8: Complete
        context.stage = ApplicationStage.COMPLETE
        return ApplicationResult.SUCCESS

    def _validate_application(self, context: ApplicationContext) -> ValidationResult:
        """Validate the application before proceeding."""
        # Create a temporary persona with the new configuration for validation
        temp_persona = copy.deepcopy(context.persona)
        temp_persona.configuration = context.target_configuration

        return self.validator.validate_persona(temp_persona)

    def _create_backup(
        self, context: ApplicationContext, options: ApplicationOptions
    ) -> None:
        """Create backup of current state."""
        # Backup original configuration
        if context.persona.configuration:
            context.original_configuration = copy.deepcopy(
                context.persona.configuration
            )

        # Backup original status
        context.original_status = context.persona.status

        # Backup session state if requested and agent switcher is available
        if options.preserve_session_state and self.agent_switcher:
            context.session_backup = copy.deepcopy(
                self.agent_switcher.current_session_state
            )

        # Store additional rollback data
        context.rollback_data = {
            "persona_metadata": (
                copy.deepcopy(context.persona.metadata.__dict__)
                if context.persona.metadata
                else {}
            ),
            "capabilities": copy.deepcopy(context.persona.capabilities),
            "session_data": copy.deepcopy(context.persona._session_data),
        }

    def _apply_configuration(self, context: ApplicationContext) -> None:
        """Apply the new configuration to the persona."""
        # Update persona configuration
        context.persona.configuration = context.target_configuration

        # Update capabilities based on new configuration if needed
        if hasattr(context.persona, "_setup_capabilities"):
            try:
                new_capabilities = context.persona._setup_capabilities()
                context.persona.capabilities = new_capabilities
            except Exception as e:
                logger.warning(f"Failed to update capabilities: {e}")

        # Update metadata
        if context.persona.metadata:
            context.persona.metadata.updated_at = datetime.now()

        logger.info(f"Applied new configuration to persona {context.persona.id}")

    def _activate_persona(self, context: ApplicationContext) -> bool:
        """Activate the persona with new configuration."""
        try:
            # If this persona is currently active, we need to handle the switch carefully
            if self.persona_manager.active_persona == context.persona:
                if self.agent_switcher:
                    # Use agent switcher for safe switching
                    success, message = self.agent_switcher.switch_persona(
                        context.persona.id,
                        reason="Template application",
                        force=False,
                    )
                    if not success:
                        logger.error(f"Failed to reactivate persona: {message}")
                        return False
                else:
                    # Fallback to direct activation
                    success = context.persona.activate()
                    if not success:
                        return False

            return True

        except Exception as e:
            logger.error(f"Error activating persona: {e}")
            return False

    def _verify_application(self, context: ApplicationContext) -> bool:
        """Verify the application was successful."""
        try:
            # Check persona status
            if context.persona.status == PersonaStatus.ERROR:
                return False

            # Validate the applied configuration
            validation_result = self.validator.validate_persona(context.persona)
            if validation_result.has_blocking_issues():
                logger.error("Post-application validation failed")
                return False

            # Check if persona can be used (basic functionality test)
            if (
                hasattr(context.persona, "configuration")
                and context.persona.configuration
            ):
                try:
                    # Try to get template to ensure it's accessible
                    template = context.persona.configuration.get_template(
                        self.template_manager
                    )
                    if not template:
                        logger.error("Cannot access persona template after application")
                        return False
                except Exception as e:
                    logger.error(f"Template access failed after application: {e}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Application verification error: {e}")
            return False

    def _rollback_application(
        self, context: ApplicationContext, options: ApplicationOptions
    ) -> None:
        """Rollback the application to previous state."""
        context.stage = ApplicationStage.ROLLBACK

        try:
            # Execute rollback hook if provided
            if options.rollback_hook:
                options.rollback_hook(context)

            # Restore original configuration
            if context.original_configuration:
                context.persona.configuration = context.original_configuration
            else:
                context.persona.configuration = None

            # Restore original status
            if context.original_status:
                context.persona.status = context.original_status

            # Restore capabilities
            if "capabilities" in context.rollback_data:
                context.persona.capabilities = context.rollback_data["capabilities"]

            # Restore session data
            if "session_data" in context.rollback_data:
                context.persona._session_data = context.rollback_data["session_data"]

            # Restore persona metadata
            if "persona_metadata" in context.rollback_data and context.persona.metadata:
                for key, value in context.rollback_data["persona_metadata"].items():
                    if hasattr(context.persona.metadata, key):
                        setattr(context.persona.metadata, key, value)

            # Restore session state if available
            if context.session_backup and self.agent_switcher:
                self.agent_switcher.current_session_state = context.session_backup

            # Reactivate with original configuration if needed
            if self.persona_manager.active_persona == context.persona:
                context.persona.activate()

            logger.info(f"Successfully rolled back persona {context.persona.id}")

        except Exception as e:
            logger.error(f"Rollback operation failed: {e}")
            raise

    def rollback_persona(self, persona_id: str) -> Tuple[bool, str]:
        """
        Rollback a persona to its previous configuration.

        Args:
            persona_id: ID of the persona to rollback

        Returns:
            Tuple of (success, message)
        """
        # Find the most recent application context for this persona
        recent_context = None
        for context in reversed(self.application_history):
            if context.persona.id == persona_id and context.original_configuration:
                recent_context = context
                break

        if not recent_context:
            return (
                False,
                f"No rollback data available for persona {persona_id}",
            )

        try:
            self._rollback_application(recent_context, self.default_options)
            return True, f"Successfully rolled back persona {persona_id}"
        except Exception as e:
            return False, f"Rollback failed: {str(e)}"

    def get_application_history(
        self, persona_id: Optional[str] = None
    ) -> List[ApplicationContext]:
        """
        Get application history.

        Args:
            persona_id: Optional persona ID to filter by

        Returns:
            List of application contexts
        """
        if persona_id:
            return [
                ctx for ctx in self.application_history if ctx.persona.id == persona_id
            ]
        return self.application_history.copy()

    def get_active_applications(self) -> Dict[str, ApplicationContext]:
        """Get currently active application contexts."""
        return self.active_contexts.copy()

    def clear_history(self) -> None:
        """Clear application history."""
        self.application_history.clear()

    def generate_application_report(self, context: ApplicationContext) -> str:
        """Generate a formatted application report."""
        lines = [
            f"📋 Template Application Report",
            f"Persona: {context.persona.name} ({context.persona.id})",
            f"Timestamp: {context.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Stage: {context.stage.value}",
            f"Result: {context.result.value if context.result else 'In Progress'}",
            "",
        ]

        if context.error_message:
            lines.extend([f"❌ Error: {context.error_message}", ""])

        if context.original_configuration:
            lines.extend(["✅ Backup Created: Yes", f"📁 Rollback Available: Yes", ""])
        else:
            lines.extend(["⚠️ Backup Created: No", "📁 Rollback Available: No", ""])

        if context.target_configuration:
            lines.extend(
                [
                    "🎯 Target Configuration:",
                    f"  Template: {context.target_configuration.template_id}",
                    f"  Primary Provider: {context.target_configuration.primary_provider}",
                    f"  Tools Enabled: {context.target_configuration.tools_enabled}",
                    "",
                ]
            )

        return "\n".join(lines)


# Global applicator instance
_global_applicator: Optional[TemplateApplicator] = None


def get_template_applicator(
    persona_manager: Optional[PersonaManager] = None,
) -> TemplateApplicator:
    """Get the global template applicator instance."""
    global _global_applicator
    if _global_applicator is None:
        from .persona import get_persona_manager

        pm = persona_manager or get_persona_manager()
        _global_applicator = TemplateApplicator(pm)
    return _global_applicator


def set_template_applicator(applicator: TemplateApplicator) -> None:
    """Set the global template applicator instance."""
    global _global_applicator
    _global_applicator = applicator
