"""
Custom Agent Integration with Omnimancer Security and Approval Systems.

This module provides the integration layer that allows custom agents to work
seamlessly with the existing security framework, approval workflows, and
permission systems.
"""

import logging
from typing import Any, Dict, List, Optional

from omnimancer.core.models import ConfigTemplateManager

from ..security.audit_logger import AuditEventType, AuditLevel, AuditLogger
from ..security.permission_controller import PermissionController
from .agent_config import (
    AgentRepository,
    AgentTool,
    CustomAgentConfig,
)
from .approval_manager import EnhancedApprovalManager
from .config import (
    AgentConfig,
    AgentSettings,
    FileSystemSettings,
    ProviderConfig,
    SecuritySettings,
    WebClientSettings,
)
from .persona import PersonaCapability

logger = logging.getLogger(__name__)


class CustomAgentIntegration:
    """Integration layer for custom agents with Omnimancer systems."""

    def __init__(
        self,
        repository: Optional[AgentRepository] = None,
        template_manager: Optional[ConfigTemplateManager] = None,
        permission_controller: Optional[PermissionController] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        """
        Initialize the integration layer.

        Args:
            repository: Agent repository for storage
            template_manager: Configuration template manager
            permission_controller: Permission controller
            audit_logger: Audit logger for security events
        """
        self.repository = repository or AgentRepository()
        self.template_manager = template_manager or ConfigTemplateManager()
        self.permission_controller = permission_controller or PermissionController()
        self.audit_logger = audit_logger or AuditLogger()

        # Cache for converted configurations
        self._config_cache: Dict[str, AgentConfig] = {}

        # Track active custom agent
        self._active_custom_agent: Optional[CustomAgentConfig] = None

    def convert_custom_to_standard_config(
        self, custom_config: CustomAgentConfig
    ) -> AgentConfig:
        """
        Convert custom agent configuration to standard AgentConfig.

        Args:
            custom_config: Custom agent configuration

        Returns:
            Standard AgentConfig compatible with existing systems
        """
        try:
            # Check cache first
            if custom_config.id in self._config_cache:
                cached_config = self._config_cache[custom_config.id]
                # Verify cache is still valid (check timestamps)
                if cached_config.updated_at >= custom_config.updated_at:
                    return cached_config

            # Create base AgentConfig
            agent_config = AgentConfig(auto_save=False)

            # Apply security settings with overrides
            security_settings = self._build_security_settings(custom_config)
            agent_config.security = security_settings

            # Apply agent settings with overrides
            agent_settings = self._build_agent_settings(custom_config)
            agent_config.agent = agent_settings

            # Build provider configuration from model settings
            provider_config = self._build_provider_config(custom_config)
            provider_id = f"custom_{custom_config.id[:8]}"
            agent_config.add_provider_config(provider_id, provider_config)

            # Apply tool-specific settings
            file_system_settings = self._build_filesystem_settings(custom_config)
            agent_config.filesystem = file_system_settings

            web_client_settings = self._build_webclient_settings(custom_config)
            agent_config.web_client = web_client_settings

            # Cache the converted configuration
            self._config_cache[custom_config.id] = agent_config

            # Log the conversion
            self.audit_logger.log_event(
                AuditEventType.SYSTEM_EVENT,
                AuditLevel.INFO,
                f"Converted custom agent '{custom_config.name}' to standard config",
                metadata={
                    "agent_id": custom_config.id,
                    "agent_name": custom_config.name,
                    "provider": custom_config.model_settings.provider_type.value,
                    "tools_count": len(custom_config.enabled_tools),
                    "capabilities_count": len(custom_config.capabilities),
                },
            )

            return agent_config

        except Exception as e:
            logger.error(f"Failed to convert custom agent config: {e}", exc_info=True)
            self.audit_logger.log_event(
                AuditEventType.SYSTEM_EVENT,
                AuditLevel.ERROR,
                f"Failed to convert custom agent '{custom_config.name}': {e}",
                metadata={"agent_id": custom_config.id, "error": str(e)},
            )
            raise

    def _build_security_settings(
        self, custom_config: CustomAgentConfig
    ) -> SecuritySettings:
        """Build security settings from custom agent configuration."""
        # Start with default security settings
        security = SecuritySettings()

        # Apply security level-based defaults
        safety_level = custom_config.behavior_rules.safety_level

        if safety_level == "strict":
            security.auto_approve_safe_operations = False
            security.backup_before_changes = True
            security.enable_sandboxing = True
            security.enable_audit_logging = True
            security.audit_log_level = "INFO"
        elif safety_level == "standard":
            security.auto_approve_safe_operations = False
            security.backup_before_changes = True
            security.enable_sandboxing = True
            security.enable_audit_logging = True
            security.audit_log_level = "INFO"
        elif safety_level == "permissive":
            security.auto_approve_safe_operations = True
            security.backup_before_changes = True
            security.enable_sandboxing = False
            security.enable_audit_logging = True
            security.audit_log_level = "WARNING"

        # Apply any security overrides from custom config
        if custom_config.security_overrides:
            for key, value in custom_config.security_overrides.items():
                if hasattr(security, key):
                    setattr(security, key, value)

        # Adjust based on enabled tools
        if AgentTool.PROGRAM_EXECUTOR not in custom_config.enabled_tools:
            # Remove commands from allowed list if tool is disabled
            security.allowed_commands = []

        # Apply confirmation requirements
        confirmation_patterns = custom_config.behavior_rules.require_confirmation_for
        if "file_delete" in confirmation_patterns:
            # This would be handled by approval workflow
            pass

        return security

    def _build_agent_settings(self, custom_config: CustomAgentConfig) -> AgentSettings:
        """Build agent settings from custom agent configuration."""
        agent_settings = AgentSettings()

        # Configure based on context parameters
        context = custom_config.context_parameters

        agent_settings.enable_learning = context.conversation_memory
        agent_settings.remember_user_preferences = context.conversation_memory
        agent_settings.verbose_logging = False  # Default to quiet
        agent_settings.enable_progress_reporting = True

        # Apply reasoning style settings
        reasoning = custom_config.behavior_rules.reasoning_style
        if reasoning == "verbose":
            agent_settings.verbose_logging = True
        elif reasoning == "concise":
            agent_settings.verbose_logging = False

        # Apply any agent overrides
        if custom_config.agent_overrides:
            for key, value in custom_config.agent_overrides.items():
                if hasattr(agent_settings, key):
                    setattr(agent_settings, key, value)

        return agent_settings

    def _build_provider_config(
        self, custom_config: CustomAgentConfig
    ) -> ProviderConfig:
        """Build provider configuration from model settings."""
        model_settings = custom_config.model_settings

        return ProviderConfig(
            provider_type=model_settings.provider_type,
            enabled=True,
            model=model_settings.model_name,
            temperature=model_settings.temperature,
            max_tokens=model_settings.max_tokens,
            timeout_seconds=30,  # Default timeout
            retry_attempts=3,
            provider_specific={
                "custom_agent_id": custom_config.id,
                "custom_agent_name": custom_config.name,
                "context_window": custom_config.context_parameters.context_window_size,
                "system_prompt": custom_config.context_parameters.system_prompt,
                "response_format": custom_config.context_parameters.response_format,
            },
        )

    def _build_filesystem_settings(
        self, custom_config: CustomAgentConfig
    ) -> FileSystemSettings:
        """Build filesystem settings based on enabled tools."""
        fs_settings = FileSystemSettings()

        if AgentTool.FILE_OPERATIONS not in custom_config.enabled_tools:
            # Disable file operations
            fs_settings.enable_atomic_operations = False
            fs_settings.enable_version_control = False
        else:
            # Enhanced settings for file operations
            safety_level = custom_config.behavior_rules.safety_level

            if safety_level == "strict":
                fs_settings.enable_atomic_operations = True
                fs_settings.backup_directory = (
                    f"/tmp/omnimancer_backups_{custom_config.id[:8]}"
                )
                fs_settings.enable_version_control = True
                fs_settings.git_auto_add = False
            elif safety_level == "permissive":
                fs_settings.enable_atomic_operations = True
                fs_settings.enable_version_control = False
                fs_settings.git_auto_add = False

        return fs_settings

    def _build_webclient_settings(
        self, custom_config: CustomAgentConfig
    ) -> WebClientSettings:
        """Build web client settings based on enabled tools."""
        web_settings = WebClientSettings()

        if AgentTool.WEB_CLIENT not in custom_config.enabled_tools:
            # Disable web client
            web_settings.enabled = False
        else:
            web_settings.enabled = True

            # Configure based on capabilities
            if PersonaCapability.WEB_SEARCH in custom_config.capabilities:
                web_settings.enable_caching = True
                web_settings.cache_ttl_seconds = 3600

            # Safety-based rate limiting
            safety_level = custom_config.behavior_rules.safety_level
            if safety_level == "strict":
                web_settings.requests_per_second = 2.0
                web_settings.requests_per_minute = 30.0
                web_settings.requests_per_hour = 100.0
            elif safety_level == "permissive":
                web_settings.requests_per_second = 10.0
                web_settings.requests_per_minute = 200.0
                web_settings.requests_per_hour = 2000.0

        return web_settings

    def apply_permission_overrides(self, custom_config: CustomAgentConfig) -> None:
        """
        Apply custom agent permission rules to the permission controller.

        Args:
            custom_config: Custom agent configuration
        """
        try:
            # Apply tool-based restrictions
            if AgentTool.FILE_OPERATIONS not in custom_config.enabled_tools:
                # Remove file operation permissions
                restricted_patterns = [
                    "file_read",
                    "file_write",
                    "file_delete",
                    "directory_create",
                    "directory_delete",
                ]
                for pattern in restricted_patterns:
                    # This would be implemented based on PermissionController API
                    pass

            if AgentTool.PROGRAM_EXECUTOR not in custom_config.enabled_tools:
                # Clear allowed commands
                self.permission_controller.allowed_commands.clear()

            # Apply custom restrictions
            for restriction in custom_config.behavior_rules.custom_restrictions:
                # Parse and apply restriction
                self._apply_custom_restriction(restriction)

            # Apply auto-approval patterns
            for pattern in custom_config.behavior_rules.auto_approve_patterns:
                # This would store approval patterns
                self._apply_auto_approval_pattern(pattern, custom_config)

            self.audit_logger.log_event(
                AuditEventType.PERMISSION_CHECK,
                AuditLevel.INFO,
                f"Applied permission overrides for custom agent '{custom_config.name}'",
                metadata={
                    "agent_id": custom_config.id,
                    "enabled_tools": [
                        tool.value for tool in custom_config.enabled_tools
                    ],
                    "restrictions_count": len(
                        custom_config.behavior_rules.custom_restrictions
                    ),
                    "auto_approve_count": len(
                        custom_config.behavior_rules.auto_approve_patterns
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Failed to apply permission overrides: {e}", exc_info=True)
            self.audit_logger.log_event(
                AuditEventType.SECURITY_ALERT,
                AuditLevel.ERROR,
                f"Failed to apply permissions for custom agent: {e}",
                metadata={"agent_id": custom_config.id, "error": str(e)},
            )

    def _apply_custom_restriction(self, restriction: str) -> None:
        """Apply a custom restriction rule."""
        # Parse restriction format (e.g., "block:*.env", "require_approval:system_commands")
        if ":" in restriction:
            action, target = restriction.split(":", 1)

            if action == "block":
                self.permission_controller.add_restricted_path(target)
            elif action == "require_approval":
                # This would be handled by the approval system
                pass
            elif action == "deny_command":
                self.permission_controller.remove_allowed_command(target)

    def _apply_auto_approval_pattern(
        self, pattern: str, custom_config: CustomAgentConfig
    ) -> None:
        """Apply an auto-approval pattern."""
        # Store pattern for future automatic approval
        # This would integrate with the permission controller's approval memory
        pass

    def setup_approval_integration(
        self,
        custom_config: CustomAgentConfig,
        approval_manager: EnhancedApprovalManager,
    ) -> None:
        """
        Set up approval workflow integration for custom agent.

        Args:
            custom_config: Custom agent configuration
            approval_manager: Enhanced approval manager
        """
        try:
            # Configure approval requirements based on behavior rules
            confirmation_requirements = (
                custom_config.behavior_rules.require_confirmation_for
            )

            # Set up approval callbacks with custom logic
            def custom_approval_filter(
                operation_type: str, operation_data: Dict[str, Any]
            ) -> bool:
                """Filter function for determining if approval is needed."""

                # Check if operation type requires confirmation
                if operation_type in confirmation_requirements:
                    return True

                # Check safety level
                safety_level = custom_config.behavior_rules.safety_level

                if safety_level == "strict":
                    # Require approval for most operations
                    risky_operations = [
                        "file_write",
                        "file_delete",
                        "command_execute",
                        "web_request",
                        "system_command",
                    ]
                    return operation_type in risky_operations

                elif safety_level == "permissive":
                    # Only require approval for very dangerous operations
                    dangerous_operations = ["file_delete", "system_command"]
                    return operation_type in dangerous_operations

                else:  # standard
                    # Default approval requirements
                    moderate_operations = [
                        "file_delete",
                        "command_execute",
                        "system_command",
                    ]
                    return operation_type in moderate_operations

            # Apply the filter to the approval manager
            # This would be implemented based on the approval manager's API

            self.audit_logger.log_event(
                AuditEventType.SYSTEM_EVENT,
                AuditLevel.INFO,
                f"Configured approval integration for custom agent '{custom_config.name}'",
                metadata={
                    "agent_id": custom_config.id,
                    "safety_level": custom_config.behavior_rules.safety_level,
                    "confirmation_types": confirmation_requirements,
                },
            )

        except Exception as e:
            logger.error(f"Failed to setup approval integration: {e}", exc_info=True)
            self.audit_logger.log_event(
                AuditEventType.SECURITY_ALERT,
                AuditLevel.ERROR,
                f"Failed to setup approval integration: {e}",
                metadata={"agent_id": custom_config.id, "error": str(e)},
            )

    def activate_custom_agent(self, custom_config: CustomAgentConfig) -> AgentConfig:
        """
        Activate a custom agent by converting and applying its configuration.

        Args:
            custom_config: Custom agent configuration to activate

        Returns:
            Converted AgentConfig ready for use
        """
        try:
            # Convert to standard configuration
            agent_config = self.convert_custom_to_standard_config(custom_config)

            # Apply permission overrides
            self.apply_permission_overrides(custom_config)

            # Update usage statistics
            custom_config.increment_usage()
            self.repository.update(custom_config)

            # Track as active
            self._active_custom_agent = custom_config

            self.audit_logger.log_event(
                AuditEventType.SYSTEM_EVENT,
                AuditLevel.INFO,
                f"Activated custom agent '{custom_config.name}'",
                metadata={
                    "agent_id": custom_config.id,
                    "agent_name": custom_config.name,
                    "provider": custom_config.model_settings.provider_type.value,
                    "model": custom_config.model_settings.model_name,
                    "safety_level": custom_config.behavior_rules.safety_level,
                },
            )

            logger.info(
                f"Successfully activated custom agent: {custom_config.name} ({custom_config.id})"
            )

            return agent_config

        except Exception as e:
            logger.error(f"Failed to activate custom agent: {e}", exc_info=True)
            self.audit_logger.log_event(
                AuditEventType.SECURITY_ALERT,
                AuditLevel.ERROR,
                f"Failed to activate custom agent '{custom_config.name}': {e}",
                metadata={"agent_id": custom_config.id, "error": str(e)},
            )
            raise

    def deactivate_custom_agent(self) -> None:
        """Deactivate the currently active custom agent."""
        if self._active_custom_agent:
            agent_name = self._active_custom_agent.name
            agent_id = self._active_custom_agent.id

            # Clear cache
            self._config_cache.pop(agent_id, None)

            # Clear active agent
            self._active_custom_agent = None

            self.audit_logger.log_event(
                AuditEventType.SYSTEM_EVENT,
                AuditLevel.INFO,
                f"Deactivated custom agent '{agent_name}'",
                metadata={"agent_id": agent_id},
            )

            logger.info(f"Deactivated custom agent: {agent_name} ({agent_id})")

    def get_active_custom_agent(self) -> Optional[CustomAgentConfig]:
        """Get the currently active custom agent."""
        return self._active_custom_agent

    def validate_custom_agent_security(
        self, custom_config: CustomAgentConfig
    ) -> List[str]:
        """
        Validate that custom agent configuration meets security requirements.

        Args:
            custom_config: Custom agent configuration to validate

        Returns:
            List of security warnings/issues (empty if valid)
        """
        warnings = []

        try:
            # Check for dangerous tool combinations
            if (
                AgentTool.PROGRAM_EXECUTOR in custom_config.enabled_tools
                and custom_config.behavior_rules.safety_level == "permissive"
            ):
                warnings.append(
                    "Program executor with permissive safety level may be risky"
                )

            # Check for missing confirmation requirements
            if (
                AgentTool.FILE_OPERATIONS in custom_config.enabled_tools
                and "file_delete"
                not in custom_config.behavior_rules.require_confirmation_for
            ):
                warnings.append(
                    "File operations enabled but file deletion doesn't require confirmation"
                )

            # Check system prompt for potentially dangerous instructions
            system_prompt = custom_config.context_parameters.system_prompt.lower()
            dangerous_phrases = [
                "ignore safety",
                "bypass security",
                "disable approval",
                "execute without asking",
                "override permissions",
            ]

            for phrase in dangerous_phrases:
                if phrase in system_prompt:
                    warnings.append(
                        f"System prompt contains potentially dangerous phrase: '{phrase}'"
                    )

            # Check model temperature for potentially unpredictable behavior
            if custom_config.model_settings.temperature > 1.5:
                warnings.append(
                    f"High temperature ({custom_config.model_settings.temperature}) may cause unpredictable behavior"
                )

            # Log validation results
            if warnings:
                self.audit_logger.log_event(
                    AuditEventType.SECURITY_ALERT,
                    AuditLevel.WARNING,
                    f"Security validation warnings for custom agent '{custom_config.name}'",
                    metadata={
                        "agent_id": custom_config.id,
                        "warnings": warnings,
                        "warning_count": len(warnings),
                    },
                )
            else:
                self.audit_logger.log_event(
                    AuditEventType.SECURITY_ALERT,
                    AuditLevel.INFO,
                    f"Custom agent '{custom_config.name}' passed security validation",
                    metadata={"agent_id": custom_config.id},
                )

            return warnings

        except Exception as e:
            logger.error(f"Security validation error: {e}", exc_info=True)
            return [f"Security validation error: {e}"]


# Utility functions for easy integration


def create_custom_agent_integration(
    repository: Optional[AgentRepository] = None,
    permission_controller: Optional[PermissionController] = None,
) -> CustomAgentIntegration:
    """Create and initialize custom agent integration."""
    return CustomAgentIntegration(
        repository=repository, permission_controller=permission_controller
    )


def activate_custom_agent_by_id(
    agent_id: str, integration: Optional[CustomAgentIntegration] = None
) -> Optional[AgentConfig]:
    """Activate a custom agent by ID."""
    integration = integration or create_custom_agent_integration()

    custom_config = integration.repository.get(agent_id)
    if not custom_config:
        return None

    return integration.activate_custom_agent(custom_config)


def validate_agent_security(
    agent_id: str, integration: Optional[CustomAgentIntegration] = None
) -> List[str]:
    """Validate security for a custom agent."""
    integration = integration or create_custom_agent_integration()

    custom_config = integration.repository.get(agent_id)
    if not custom_config:
        return ["Agent not found"]

    return integration.validate_custom_agent_security(custom_config)
