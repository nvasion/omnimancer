"""
Comprehensive Configuration Provider for Omnimancer.

This module consolidates configuration management, validation, compatibility,
and user interfaces into a single cohesive system that replaces the functionality
of multiple overlapping configuration modules.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_manager import ConfigManager
from .config_validator import ConfigValidator

# from .config_migration_helpers import ConfigMigrationHelper  # Removed - over-engineered
from .models import (
    Config,
    ConfigTemplate,
    ConfigTemplateManager,
    ProviderConfig,
)

logger = logging.getLogger(__name__)


class ConfigurationMode(Enum):
    """Configuration modes for different user experience levels."""

    SIMPLE = "simple"
    ADVANCED = "advanced"
    GUIDED = "guided"


class ConfigurationComplexity(Enum):
    """Complexity levels for configuration options."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ConfigurationContext:
    """Context information for configuration operations."""

    user_level: str = "beginner"
    use_case: str = "general"
    primary_providers: List[str] = None
    preferred_mode: ConfigurationMode = ConfigurationMode.SIMPLE
    show_experimental: bool = False

    def __post_init__(self):
        if self.primary_providers is None:
            self.primary_providers = []


@dataclass
class ValidationReport:
    """Comprehensive validation and compatibility report."""

    is_valid: bool
    is_compatible: bool
    compatibility_level: str
    total_issues: int
    critical_errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    health_score: float  # 0-100
    format_info: Dict[str, Any]
    provider_status: Dict[str, Dict[str, Any]]
    performance_issues: List[str]
    security_issues: List[str]
    upgrade_recommendations: List[str]
    migration_required: bool
    backup_available: bool
    timestamp: str

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class SimplifiedProviderConfig:
    """Simplified provider configuration for simple mode."""

    name: str
    display_name: str
    description: str
    enabled: bool = True
    api_key: Optional[str] = None
    model: Optional[str] = None
    use_case_fit: List[str] = None

    def __post_init__(self):
        if self.use_case_fit is None:
            self.use_case_fit = ["general"]


@dataclass
class ProviderHealthCheck:
    """Provider health check result."""

    name: str
    status: str  # "healthy", "warning", "error", "unknown"
    api_key_present: bool
    model_valid: bool
    connectivity_ok: bool
    issues: List[str]
    recommendations: List[str]
    last_tested: str


@dataclass
class ConfigurationRecommendation:
    """Recommendation for configuration based on user context."""

    template_name: str
    confidence: float
    reasons: List[str]
    providers: List[str]
    estimated_cost: str
    complexity: ConfigurationComplexity


@dataclass
class SimpleConfigOption:
    """Represents a simple configuration option for users."""

    key: str
    display_name: str
    description: str
    option_type: str  # "choice", "boolean", "text", "password"
    default_value: Any = None
    choices: Optional[List[Dict[str, str]]] = None
    required: bool = True
    complexity: ConfigurationComplexity = ConfigurationComplexity.BASIC
    category: str = "General"

    def __post_init__(self):
        if self.choices is None:
            self.choices = []


@dataclass
class SimpleConfigSection:
    """A section of related configuration options."""

    name: str
    description: str
    icon: str
    options: List[SimpleConfigOption]
    required: bool = True
    expanded: bool = True


class ConfigurationProvider:
    """
    Comprehensive configuration provider that consolidates all configuration functionality.

    This class provides:
    - Configuration management and validation
    - Compatibility checking and migration
    - Simple and advanced user interfaces
    - Provider health monitoring
    - Template-based configuration
    """

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the configuration provider.

        Args:
            config_manager: Optional ConfigManager instance
        """
        self.config_manager = config_manager or ConfigManager()
        self.template_manager = ConfigTemplateManager()
        self.base_validator = ConfigValidator()
        # self.migration_helper = ConfigMigrationHelper(self.config_manager)  # Removed - over-engineered

        # Configuration state
        self.current_mode = ConfigurationMode.SIMPLE
        self.context = ConfigurationContext()
        self.current_version = "2.0"
        self.legacy_version = "1.0"

        # Load mode preferences
        self._load_mode_preferences()

        # Initialize simple options
        self.simple_sections = self._create_simple_sections()

    # =============================================================================
    # Mode Management
    # =============================================================================

    def set_configuration_mode(self, mode: ConfigurationMode) -> None:
        """Set the configuration mode."""
        self.current_mode = mode
        self._save_mode_preferences()
        logger.info(f"Configuration mode set to: {mode.value}")

    def set_configuration_context(self, context: ConfigurationContext) -> None:
        """Set the configuration context for personalized experience."""
        self.context = context
        self._save_mode_preferences()
        logger.info(
            f"Configuration context updated: use_case={context.use_case}, level={context.user_level}"
        )

    def get_available_modes(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available configuration modes."""
        return {
            ConfigurationMode.SIMPLE.value: {
                "name": "Simple",
                "description": "Template-based configuration with minimal technical details",
                "target_audience": "Beginners and users who want quick setup",
                "features": [
                    "Pre-configured templates for common use cases",
                    "Automatic provider selection and optimization",
                    "Simplified settings with explanations",
                    "One-click setup for popular configurations",
                ],
            },
            ConfigurationMode.GUIDED.value: {
                "name": "Guided",
                "description": "Step-by-step configuration with recommendations",
                "target_audience": "Users who want some control with guidance",
                "features": [
                    "Interactive configuration wizard",
                    "Smart recommendations based on usage patterns",
                    "Explanations for configuration choices",
                    "Ability to customize templates",
                ],
            },
            ConfigurationMode.ADVANCED.value: {
                "name": "Advanced",
                "description": "Full access to all configuration options",
                "target_audience": "Power users and developers",
                "features": [
                    "Complete configuration control",
                    "Access to all provider settings",
                    "Custom MCP server configuration",
                    "Advanced debugging options",
                ],
            },
        }

    # =============================================================================
    # Validation & Compatibility
    # =============================================================================

    async def comprehensive_validation(
        self, config_path: Optional[Path] = None
    ) -> ValidationReport:
        """Perform comprehensive configuration validation and compatibility checking."""
        try:
            # Detect configuration format and compatibility
            format_info = self._detect_configuration_format(config_path)
            is_compatible, comp_errors, comp_warnings = self._validate_compatibility(
                config_path
            )

            # Load configuration safely
            config = await self._load_config_safely(config_path)

            # Perform base validation
            base_errors = []
            if config:
                base_errors = self.base_validator.validate_config(config)

            # Combine all errors and warnings
            all_errors = comp_errors + base_errors
            critical_errors = [e for e in all_errors if self._is_critical_error(e)]
            all_warnings = comp_warnings

            # Perform provider health checks
            provider_status = (
                await self._check_provider_health(config) if config else {}
            )

            # Check for performance and security issues
            performance_issues = (
                self._check_performance_issues(config, format_info) if config else []
            )
            security_issues = self._check_security_issues(config) if config else []

            # Generate suggestions and upgrade recommendations
            suggestions = self._generate_suggestions(
                config, format_info, all_errors, all_warnings
            )
            upgrade_recommendations = self._generate_upgrade_recommendations(
                format_info
            )

            # Calculate health score
            health_score = self._calculate_health_score(
                critical_errors,
                all_warnings,
                provider_status,
                performance_issues,
                security_issues,
            )

            # Check for migration and backup
            migration_required = (
                len(all_errors) > 3
                or format_info.get("version", self.current_version)
                < self.current_version
            )
            backup_available = self._check_backup_availability()

            return ValidationReport(
                is_valid=len(critical_errors) == 0,
                is_compatible=is_compatible,
                compatibility_level=format_info.get("compatibility_level", "unknown"),
                total_issues=len(all_errors) + len(all_warnings),
                critical_errors=critical_errors,
                warnings=all_warnings,
                suggestions=suggestions,
                health_score=health_score,
                format_info=format_info,
                provider_status=provider_status,
                performance_issues=performance_issues,
                security_issues=security_issues,
                upgrade_recommendations=upgrade_recommendations,
                migration_required=migration_required,
                backup_available=backup_available,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            return ValidationReport(
                is_valid=False,
                is_compatible=False,
                compatibility_level="unknown",
                total_issues=1,
                critical_errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                suggestions=["Check configuration file format and permissions"],
                health_score=0.0,
                format_info={},
                provider_status={},
                performance_issues=[],
                security_issues=[],
                upgrade_recommendations=[],
                migration_required=True,
                backup_available=False,
                timestamp=datetime.now().isoformat(),
            )

    def _detect_configuration_format(
        self, config_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Detect the format and version of an existing configuration."""
        if config_path is None:
            config_path = self.config_manager.get_storage_path() / "config.json"

        try:
            if not config_path.exists():
                return {
                    "version": "none",
                    "format_type": "none",
                    "detected_features": [],
                    "compatibility_level": "full",
                    "upgrade_available": False,
                    "issues": ["No configuration file found"],
                }

            with open(config_path, "r") as f:
                config_data = json.load(f)

            return self._analyze_config_format(config_data)

        except Exception as e:
            logger.error(f"Failed to detect configuration format: {e}")
            return {
                "version": "unknown",
                "format_type": "unknown",
                "detected_features": [],
                "compatibility_level": "incompatible",
                "upgrade_available": False,
                "issues": [f"Failed to read configuration: {str(e)}"],
            }

    def _analyze_config_format(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze configuration data to determine format."""
        detected_features = []
        format_type = "legacy"
        version = "1.0"
        compatibility_level = "full"
        upgrade_available = True
        issues = []

        # Check for version information
        if "config_version" in config_data:
            version = config_data["config_version"]
        elif "_schema_version" in config_data:
            version = config_data["_schema_version"]

        # Detect features based on configuration structure
        feature_checks = {
            "providers": "providers",
            "default_provider": "default_provider",
            "mcp": "mcp",
            "chat_settings": "chat_settings",
            "configuration_mode": "configuration_mode",
            "user_preferences": "user_preferences",
            "template_source": "template_source",
        }

        for feature, key in feature_checks.items():
            if key in config_data:
                detected_features.append(feature)

        # Determine format type
        if "configuration_mode" in detected_features:
            format_type = "advanced"
        elif (
            "user_preferences" in detected_features
            or "template_source" in detected_features
        ):
            format_type = "simple"

        # Check for issues
        if "providers" in config_data and len(config_data["providers"]) > 5:
            issues.append(
                f"Large number of providers ({len(config_data['providers'])}) might benefit from simplification"
            )

        # Check for deprecated fields
        deprecated_fields = [
            "legacy_auth",
            "old_mcp_format",
            "deprecated_settings",
        ]
        for field in deprecated_fields:
            if field in config_data:
                issues.append(f"Deprecated field '{field}' found")
                compatibility_level = "partial"

        return {
            "version": version,
            "format_type": format_type,
            "detected_features": detected_features,
            "compatibility_level": compatibility_level,
            "upgrade_available": upgrade_available and version < self.current_version,
            "issues": issues,
        }

    def _validate_compatibility(
        self, config_path: Optional[Path] = None
    ) -> Tuple[bool, List[str], List[str]]:
        """Validate configuration compatibility."""
        errors = []
        warnings = []

        try:
            format_info = self._detect_configuration_format(config_path)

            if format_info["compatibility_level"] == "incompatible":
                errors.append(
                    "Configuration format is incompatible and requires manual migration"
                )
                errors.extend(format_info["issues"])
                return False, errors, warnings

            elif format_info["compatibility_level"] == "partial":
                warnings.append(
                    "Configuration has compatibility issues that may need attention"
                )
                warnings.extend(format_info["issues"])

            # Load and validate configuration
            if config_path is None:
                config_path = self.config_manager.get_storage_path() / "config.json"

            if not config_path.exists():
                errors.append("Configuration file not found")
                return False, errors, warnings

            try:
                config = self.config_manager.load_config()
                validation_errors = self.base_validator.validate_config(config)

                if validation_errors:
                    for error in validation_errors:
                        if any(
                            critical in error.lower()
                            for critical in ["missing", "required", "invalid"]
                        ):
                            errors.append(error)
                        else:
                            warnings.append(error)

            except Exception as e:
                errors.append(f"Configuration loading failed: {str(e)}")
                return False, errors, warnings

            if format_info["upgrade_available"]:
                warnings.append(
                    f"Configuration upgrade available: v{format_info['version']} → v{self.current_version}"
                )

            is_compatible = len(errors) == 0
            return is_compatible, errors, warnings

        except Exception as e:
            logger.error(f"Compatibility validation failed: {e}")
            errors.append(f"Validation failed: {str(e)}")
            return False, errors, warnings

    async def _load_config_safely(
        self, config_path: Optional[Path] = None
    ) -> Optional[Config]:
        """Safely load configuration without failing validation."""
        try:
            if config_path:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                return self.config_manager.load_config_from_dict(config_data)
            else:
                return self.config_manager.load_config()
        except Exception as e:
            logger.debug(f"Failed to load configuration: {e}")
            return None

    def _is_critical_error(self, error: str) -> bool:
        """Determine if an error is critical."""
        critical_keywords = [
            "missing api key",
            "no providers configured",
            "default provider not configured",
            "invalid storage path",
            "configuration loading failed",
            "incompatible",
        ]
        return any(keyword in error.lower() for keyword in critical_keywords)

    async def _check_provider_health(self, config: Config) -> Dict[str, Dict[str, Any]]:
        """Check health of all configured providers."""
        provider_status = {}

        for provider_name, provider_config in config.providers.items():
            health_check = await self._check_single_provider_health(
                provider_name, provider_config
            )
            provider_status[provider_name] = asdict(health_check)

        return provider_status

    async def _check_single_provider_health(
        self, provider_name: str, provider_config: ProviderConfig
    ) -> ProviderHealthCheck:
        """Check health of a single provider."""
        issues = []
        recommendations = []
        status = "healthy"

        # Check API key
        api_key_present = bool(provider_config.api_key)
        if not api_key_present and provider_name not in [
            "claude-code",
            "ollama",
        ]:
            issues.append("API key missing")
            recommendations.append("Add API key for this provider")
            status = "error"

        # Check model validity
        model_valid = bool(provider_config.model)
        if not model_valid:
            issues.append("No model specified")
            recommendations.append("Specify a valid model for this provider")
            status = "error"

        # Check model name validity
        if model_valid and provider_config.model:
            model_issues = self._validate_model_name(
                provider_name, provider_config.model
            )
            if model_issues:
                issues.extend(model_issues)
                recommendations.append("Update to a supported model name")
                if status == "healthy":
                    status = "warning"

        # Check connectivity
        connectivity_ok = api_key_present and model_valid
        if not connectivity_ok and status == "healthy":
            status = "warning"

        # Check provider-specific issues
        provider_specific_issues = self._check_provider_specific_issues(
            provider_name, provider_config
        )
        if provider_specific_issues:
            issues.extend(provider_specific_issues)
            if status == "healthy":
                status = "warning"

        return ProviderHealthCheck(
            name=provider_name,
            status=status,
            api_key_present=api_key_present,
            model_valid=model_valid,
            connectivity_ok=connectivity_ok,
            issues=issues,
            recommendations=recommendations,
            last_tested=datetime.now().isoformat(),
        )

    def _validate_model_name(self, provider_name: str, model_name: str) -> List[str]:
        """Validate model name for a provider."""
        issues = []

        known_models = {
            "claude": [
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229",
                "claude-3-5-sonnet-20241022",
            ],
            "openai": [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
            ],
            "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
            "perplexity": ["sonar-pro", "sonar", "deep-research"],
            "mistral": ["mistral-small", "mistral-medium", "mistral-large"],
        }

        if provider_name in known_models:
            valid_models = known_models[provider_name]
            if not any(model in model_name for model in valid_models):
                issues.append(
                    f"Model '{model_name}' may not be valid for {provider_name}"
                )

        return issues

    def _check_provider_specific_issues(
        self, provider_name: str, provider_config: ProviderConfig
    ) -> List[str]:
        """Check provider-specific configuration issues."""
        issues = []

        if provider_name.lower() == "azure":
            if (
                not hasattr(provider_config, "azure_endpoint")
                or not provider_config.azure_endpoint
            ):
                issues.append("Azure endpoint not specified")
            if (
                not hasattr(provider_config, "azure_deployment")
                or not provider_config.azure_deployment
            ):
                issues.append("Azure deployment not specified")
        elif provider_name.lower() == "bedrock":
            if (
                not hasattr(provider_config, "aws_region")
                or not provider_config.aws_region
            ):
                issues.append("AWS region not specified")
        elif provider_name.lower() == "ollama":
            if hasattr(provider_config, "base_url") and provider_config.base_url:
                if not provider_config.base_url.startswith(("http://", "https://")):
                    issues.append("Invalid Ollama base URL format")

        return issues

    def _check_performance_issues(
        self, config: Config, format_info: Dict[str, Any]
    ) -> List[str]:
        """Check for performance-related configuration issues."""
        issues = []

        if len(config.providers) > 8:
            issues.append(
                f"Large number of providers ({len(config.providers)}) may impact performance"
            )

        timeout_issues = sum(
            1
            for provider_config in config.providers.values()
            if not hasattr(provider_config, "timeout") or not provider_config.timeout
        )

        if timeout_issues > len(config.providers) / 2:
            issues.append("Many providers missing timeout configurations")

        if hasattr(config, "mcp") and config.mcp and hasattr(config.mcp, "servers"):
            if len(config.mcp.servers) > 10:
                issues.append(
                    f"Large number of MCP servers ({len(config.mcp.servers)}) may impact startup time"
                )

        if (
            hasattr(config, "configuration_mode")
            and config.configuration_mode == "simple"
        ):
            if len(config.providers) > 3:
                issues.append("Simple mode with many providers may cause confusion")

        return issues

    def _check_security_issues(self, config: Config) -> List[str]:
        """Check for security-related configuration issues."""
        issues = []

        for provider_name, provider_config in config.providers.items():
            if provider_config.api_key:
                if len(provider_config.api_key) < 10:
                    issues.append(f"Suspiciously short API key for {provider_name}")
                elif provider_config.api_key.startswith(("demo", "test", "example")):
                    issues.append(f"Possible demo/test API key for {provider_name}")

        if hasattr(config, "mcp") and config.mcp:
            if hasattr(config.mcp, "auto_approve") and config.mcp.auto_approve:
                issues.append("MCP auto-approve enabled - may pose security risk")

        return issues

    def _generate_suggestions(
        self,
        config: Optional[Config],
        format_info: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> List[str]:
        """Generate helpful suggestions for configuration improvement."""
        suggestions = []

        if any("no providers" in error.lower() for error in errors):
            suggestions.append(
                "Run 'omnimancer config setup' to configure AI providers"
            )

        if any("api key" in error.lower() for error in errors):
            suggestions.append("Add missing API keys for enabled providers")

        if any("model" in error.lower() for error in errors):
            suggestions.append("Specify valid models for all providers")

        if format_info.get("upgrade_available", False):
            suggestions.append(
                f"Upgrade configuration from v{format_info.get('version')} to v{self.current_version}"
            )

        if (
            format_info.get("format_type") == "legacy"
            and len(format_info.get("detected_features", [])) > 5
        ):
            suggestions.append(
                "Consider using simplified configuration mode for easier management"
            )

        if config and len(config.providers) == 1:
            suggestions.append("Consider adding a backup provider for redundancy")
        elif config and len(config.providers) > 5:
            suggestions.append(
                "Consider simplifying provider configuration for better performance"
            )

        if len(warnings) > 3:
            suggestions.append("Review and address configuration warnings")

        return suggestions

    def _generate_upgrade_recommendations(
        self, format_info: Dict[str, Any]
    ) -> List[str]:
        """Generate upgrade recommendations."""
        recommendations = []

        if format_info.get("upgrade_available", False):
            compatibility_level = format_info.get("compatibility_level", "unknown")
            if compatibility_level == "full":
                recommendations.append(
                    "Automatic upgrade available - run 'omnimancer config migrate'"
                )
            elif compatibility_level == "partial":
                recommendations.append(
                    "Assisted upgrade recommended - review configuration issues first"
                )
            else:
                recommendations.append(
                    "Manual upgrade required - backup configuration before changes"
                )

        if (
            "mcp" in format_info.get("detected_features", [])
            and format_info.get("version", "2.0") < "2.0"
        ):
            recommendations.append(
                "MCP configuration can be simplified in newer format"
            )

        if len(format_info.get("detected_features", [])) > 6:
            recommendations.append(
                "Configuration complexity can be reduced with template-based setup"
            )

        return recommendations

    def _calculate_health_score(
        self,
        critical_errors: List[str],
        warnings: List[str],
        provider_status: Dict[str, Dict[str, Any]],
        performance_issues: List[str],
        security_issues: List[str],
    ) -> float:
        """Calculate overall configuration health score (0-100)."""
        score = 100.0

        score -= len(critical_errors) * 25
        score -= len(warnings) * 5

        for provider_name, status in provider_status.items():
            if status["status"] == "error":
                score -= 15
            elif status["status"] == "warning":
                score -= 5

        score -= len(performance_issues) * 3
        score -= len(security_issues) * 10

        return max(0.0, min(100.0, score))

    def generate_compatibility_report(self) -> str:
        """Generate a comprehensive compatibility report."""
        try:
            # Get comprehensive validation report
            import asyncio

            validation_report = asyncio.run(self.comprehensive_validation())

            report_lines = []
            report_lines.append("# Configuration Compatibility Report")
            report_lines.append("")

            # Basic info
            report_lines.append("## Configuration Status")
            report_lines.append(
                f"- Version: {validation_report.format_info.get('version', 'unknown')}"
            )
            report_lines.append(
                f"- Format Type: {validation_report.format_info.get('format_type', 'unknown').title()}"
            )
            report_lines.append(
                f"- Compatible: {'Yes' if validation_report.is_compatible else 'No'}"
            )
            report_lines.append(
                f"- Valid: {'Yes' if validation_report.is_valid else 'No'}"
            )
            report_lines.append(
                f"- Health Score: {validation_report.health_score:.1f}/100"
            )
            report_lines.append("")

            # Issues
            if validation_report.critical_errors:
                report_lines.append("## Critical Errors")
                for error in validation_report.critical_errors:
                    report_lines.append(f"- {error}")
                report_lines.append("")

            if validation_report.warnings:
                report_lines.append("## Warnings")
                for warning in validation_report.warnings:
                    report_lines.append(f"- {warning}")
                report_lines.append("")

            # Recommendations
            if (
                validation_report.suggestions
                or validation_report.upgrade_recommendations
            ):
                report_lines.append("## Recommendations")
                for suggestion in validation_report.suggestions:
                    report_lines.append(f"- {suggestion}")
                for upgrade in validation_report.upgrade_recommendations:
                    report_lines.append(f"- {upgrade}")
                report_lines.append("")

            # Provider details
            if validation_report.provider_status:
                report_lines.append("## Provider Status")
                for (
                    provider,
                    status,
                ) in validation_report.provider_status.items():
                    status_icon = (
                        "✓"
                        if status["status"] == "healthy"
                        else "⚠" if status["status"] == "warning" else "✗"
                    )
                    report_lines.append(
                        f"- **{provider}**: {status_icon} {status['status'].title()}"
                    )

                    if status.get("issues"):
                        for issue in status["issues"]:
                            report_lines.append(f"  - Issue: {issue}")

                    if status.get("recommendations"):
                        for rec in status["recommendations"]:
                            report_lines.append(f"  - Recommendation: {rec}")

                report_lines.append("")

            # Migration info
            if validation_report.migration_required:
                report_lines.append("## Migration Recommended")
                report_lines.append(
                    "Consider migrating to the latest configuration format for better features."
                )
                if validation_report.backup_available:
                    report_lines.append(
                        "Configuration backups are available for safe migration."
                    )
                else:
                    report_lines.append(
                        "**Warning**: No configuration backups found. Create backup before migration."
                    )

            return "\n".join(report_lines)

        except Exception as e:
            return f"Failed to generate compatibility report: {str(e)}"

    def ensure_backward_compatibility(self) -> Tuple[bool, List[str]]:
        """Ensure backward compatibility with existing configurations."""
        messages = []

        try:
            # Get validation report
            import asyncio

            validation_report = asyncio.run(self.comprehensive_validation())

            if validation_report.is_valid and validation_report.is_compatible:
                messages.append("✓ Configuration is backward compatible")
                return True, messages

            # Attempt automatic fixes for compatibility issues
            messages.append("Applying backward compatibility fixes...")

            try:
                config = self.config_manager.get_config()
                fixed_issues = 0

                # Fix deprecated model references
                for name, provider_config in config.providers.items():
                    if name == "claude" and provider_config.model == "claude-v1":
                        provider_config.model = "claude-3-haiku-20240307"
                        messages.append(f"✓ Updated {name} model to current version")
                        fixed_issues += 1
                    elif name == "openai" and provider_config.model in [
                        "text-davinci-003",
                        "code-davinci-002",
                    ]:
                        provider_config.model = "gpt-4o-mini"
                        messages.append(f"✓ Updated {name} model to current version")
                        fixed_issues += 1

                # Ensure configuration version is set
                if not hasattr(config, "version"):
                    config.version = self.current_version
                    messages.append("✓ Added version information to configuration")
                    fixed_issues += 1

                # Save fixes
                if fixed_issues > 0:
                    self.config_manager.save_config(config)
                    messages.append(f"✓ Applied {fixed_issues} compatibility fixes")

                # Final validation
                final_report = asyncio.run(self.comprehensive_validation())
                if final_report.is_valid and final_report.is_compatible:
                    messages.append("✅ Backward compatibility ensured")
                    return True, messages
                else:
                    remaining_issues = final_report.total_issues
                    messages.append(f"⚠ {remaining_issues} issues remain")
                    messages.extend(
                        final_report.suggestions[:3]
                    )  # Show top 3 suggestions
                    return False, messages

            except Exception as e:
                messages.append(f"✗ Failed to apply fixes: {str(e)}")
                return False, messages

        except Exception as e:
            logger.error(f"Backward compatibility check failed: {e}")
            messages.append(f"✗ Compatibility check failed: {str(e)}")
            return False, messages

    def _check_backup_availability(self) -> bool:
        """Check if configuration backups are available."""
        try:
            storage_path = self.config_manager.get_storage_path()
            backup_dir = storage_path / "backups"

            if backup_dir.exists():
                backup_files = list(backup_dir.glob("*.json"))
                return len(backup_files) > 0

            return False
        except:
            return False

    # =============================================================================
    # Simple Configuration Interface
    # =============================================================================

    def _create_simple_sections(self) -> List[SimpleConfigSection]:
        """Create simplified configuration sections."""
        sections = []

        # Quick Setup Section
        quick_setup_options = [
            SimpleConfigOption(
                key="use_case",
                display_name="What will you primarily use Omnimancer for?",
                description="Choose your main use case to get optimized settings",
                option_type="choice",
                choices=[
                    {
                        "value": "coding",
                        "label": "Coding & Development",
                        "description": "Writing code, debugging, technical documentation",
                    },
                    {
                        "value": "research",
                        "label": "Research & Analysis",
                        "description": "Research papers, data analysis, fact-checking",
                    },
                    {
                        "value": "creative",
                        "label": "Creative Writing",
                        "description": "Stories, articles, creative content",
                    },
                    {
                        "value": "general",
                        "label": "General Use",
                        "description": "Mixed tasks, general assistance",
                    },
                    {
                        "value": "performance",
                        "label": "Fast & Economical",
                        "description": "Quick responses, minimal costs",
                    },
                ],
                default_value="general",
                category="Quick Setup",
                complexity=ConfigurationComplexity.BASIC,
            ),
            SimpleConfigOption(
                key="experience_level",
                display_name="How familiar are you with AI tools?",
                description="This helps us show the right level of detail",
                option_type="choice",
                choices=[
                    {
                        "value": "beginner",
                        "label": "Beginner",
                        "description": "New to AI tools, prefer simple options",
                    },
                    {
                        "value": "intermediate",
                        "label": "Some Experience",
                        "description": "Used AI tools before, comfortable with basic settings",
                    },
                    {
                        "value": "advanced",
                        "label": "Experienced",
                        "description": "Familiar with AI, want more control",
                    },
                    {
                        "value": "expert",
                        "label": "Expert",
                        "description": "Power user, need full customization",
                    },
                ],
                default_value="intermediate",
                category="Quick Setup",
                complexity=ConfigurationComplexity.BASIC,
            ),
        ]

        sections.append(
            SimpleConfigSection(
                name="Quick Setup",
                description="Get started quickly with recommended settings",
                icon="⚡",
                options=quick_setup_options,
                required=True,
                expanded=True,
            )
        )

        # AI Providers Section
        provider_options = [
            SimpleConfigOption(
                key="enable_claude",
                display_name="Anthropic Claude",
                description="Excellent reasoning and coding. Requires API key ($).",
                option_type="boolean",
                default_value=True,
                category="AI Providers",
                complexity=ConfigurationComplexity.BASIC,
            ),
            SimpleConfigOption(
                key="claude_api_key",
                display_name="Claude API Key",
                description="Get from https://console.anthropic.com",
                option_type="password",
                required=False,
                category="AI Providers",
                complexity=ConfigurationComplexity.BASIC,
            ),
            SimpleConfigOption(
                key="enable_openai",
                display_name="OpenAI GPT",
                description="Strong creative abilities and problem solving. Requires API key ($).",
                option_type="boolean",
                default_value=True,
                category="AI Providers",
                complexity=ConfigurationComplexity.BASIC,
            ),
            SimpleConfigOption(
                key="openai_api_key",
                display_name="OpenAI API Key",
                description="Get from https://platform.openai.com",
                option_type="password",
                required=False,
                category="AI Providers",
                complexity=ConfigurationComplexity.BASIC,
            ),
        ]

        sections.append(
            SimpleConfigSection(
                name="AI Providers",
                description="Choose which AI services to use",
                icon="🤖",
                options=provider_options,
                required=True,
                expanded=False,
            )
        )

        return sections

    def get_simple_mode_providers(self) -> List[SimplifiedProviderConfig]:
        """Get simplified provider configurations for simple mode."""
        simplified_providers = []

        provider_info = {
            "claude": {
                "display_name": "Anthropic Claude",
                "description": "Excellent for reasoning, coding, and analysis. Very capable and safe.",
                "use_case_fit": ["coding", "research", "creative", "general"],
            },
            "openai": {
                "display_name": "OpenAI GPT",
                "description": "Strong general capabilities, good for creative tasks and problem solving.",
                "use_case_fit": ["creative", "coding", "general"],
            },
            "gemini": {
                "display_name": "Google Gemini",
                "description": "Large context window, good for document analysis and multimodal tasks.",
                "use_case_fit": ["research", "general"],
            },
            "claude-code": {
                "display_name": "Claude Code (Free)",
                "description": "Free local access to Claude via Claude Code. No API costs.",
                "use_case_fit": ["coding", "general"],
            },
        }

        try:
            current_config = self.config_manager.get_config()
            current_providers = current_config.providers
        except:
            current_providers = {}

        for provider_name, info in provider_info.items():
            enabled = provider_name in current_providers
            api_key = None
            model = None

            if enabled and provider_name in current_providers:
                provider_config = current_providers[provider_name]
                api_key = "***" if provider_config.api_key else None
                model = provider_config.model

            simplified_providers.append(
                SimplifiedProviderConfig(
                    name=provider_name,
                    display_name=info["display_name"],
                    description=info["description"],
                    enabled=enabled,
                    api_key=api_key,
                    model=model,
                    use_case_fit=info["use_case_fit"],
                )
            )

        return simplified_providers

    def get_quick_setup_templates(self) -> List[Dict[str, Any]]:
        """Get quick setup templates with simplified descriptions."""
        templates = []

        # Get all templates and simplify them
        for (
            name,
            template,
        ) in self.template_manager.get_all_templates().items():
            simplified = {
                "name": name,
                "display_name": template.name.title(),
                "description": template.description,
                "icon": self._get_template_icon(name),
                "estimated_setup_time": self._get_estimated_setup_time(template),
                "cost_estimate": self._get_cost_estimate(template),
                "complexity": (
                    "Simple" if name in ["general", "performance"] else "Moderate"
                ),
                "providers_needed": len(template.recommended_providers[:3]),
                "main_providers": template.recommended_providers[:2],
                "use_case": template.use_case,
                "recommended_for": self._get_recommended_for(template),
            }
            templates.append(simplified)

        # Sort by complexity (simple first) and popularity
        templates.sort(
            key=lambda x: (
                0 if x["complexity"] == "Simple" else 1,
                -x["providers_needed"],
            )
        )

        return templates

    def _get_template_icon(self, template_name: str) -> str:
        """Get an appropriate icon for a template."""
        icons = {
            "coding": "💻",
            "research": "🔬",
            "creative": "✏️",
            "general": "🌟",
            "performance": "⚡",
        }
        return icons.get(template_name, "🤖")

    def _get_estimated_setup_time(self, template: ConfigTemplate) -> str:
        """Estimate setup time for a template."""
        provider_count = len(template.recommended_providers[:3])

        if provider_count <= 1:
            return "2-3 minutes"
        elif provider_count == 2:
            return "5-7 minutes"
        else:
            return "8-12 minutes"

    def _get_cost_estimate(self, template: ConfigTemplate) -> str:
        """Get cost estimate for a template."""
        providers = template.recommended_providers[:3]

        # Check for free options
        free_providers = ["claude-code", "ollama"]
        has_free = any(p in free_providers for p in providers)

        if has_free and len(providers) <= 2:
            return "Free option available"
        elif "performance" in template.name.lower():
            return "Low cost (~$5-15/month)"
        else:
            return "Moderate cost (~$15-40/month)"

    def _get_recommended_for(self, template: ConfigTemplate) -> List[str]:
        """Get who this template is recommended for."""
        recommendations = {
            "coding": [
                "Software developers",
                "Students learning programming",
                "Technical writers",
            ],
            "research": ["Researchers", "Students", "Analysts", "Journalists"],
            "creative": [
                "Writers",
                "Content creators",
                "Marketing professionals",
            ],
            "general": [
                "General users",
                "Business professionals",
                "Personal use",
            ],
            "performance": [
                "Budget-conscious users",
                "High-volume usage",
                "Teams",
            ],
        }
        return recommendations.get(template.use_case, ["General users"])

    def get_setup_wizard_steps(self, use_case: str = "general") -> List[Dict[str, Any]]:
        """Get step-by-step wizard steps for configuration."""
        steps = []

        # Step 1: Welcome
        steps.append(
            {
                "step": 1,
                "title": "Welcome to Omnimancer",
                "description": "Let's set up Omnimancer for your needs",
                "type": "welcome",
                "content": {
                    "message": "This quick setup will configure Omnimancer based on how you plan to use it.",
                    "estimated_time": "5-10 minutes",
                    "features": [
                        "Choose your use case for optimized settings",
                        "Select AI providers that fit your needs",
                        "Set up API keys for your chosen services",
                        "Enable helpful tools and features",
                    ],
                },
            }
        )

        # Step 2: Use Case Selection
        steps.append(
            {
                "step": 2,
                "title": "Choose Your Primary Use Case",
                "description": "This helps us recommend the best settings for you",
                "type": "single_choice",
                "option": {
                    "key": "use_case",
                    "display_name": "What will you primarily use Omnimancer for?",
                    "description": "Choose your main use case to get optimized settings",
                    "option_type": "choice",
                    "choices": [
                        {
                            "value": "coding",
                            "label": "Coding & Development",
                            "description": "Writing code, debugging, technical documentation",
                        },
                        {
                            "value": "research",
                            "label": "Research & Analysis",
                            "description": "Research papers, data analysis, fact-checking",
                        },
                        {
                            "value": "creative",
                            "label": "Creative Writing",
                            "description": "Stories, articles, creative content",
                        },
                        {
                            "value": "general",
                            "label": "General Use",
                            "description": "Mixed tasks, general assistance",
                        },
                        {
                            "value": "performance",
                            "label": "Fast & Economical",
                            "description": "Quick responses, minimal costs",
                        },
                    ],
                    "default_value": "general",
                },
                "help_text": "Don't worry - you can change this later or use Omnimancer for multiple purposes.",
            }
        )

        # Step 3: Experience Level
        steps.append(
            {
                "step": 3,
                "title": "Your Experience Level",
                "description": "This helps us show the right amount of detail",
                "type": "single_choice",
                "option": {
                    "key": "experience_level",
                    "display_name": "How familiar are you with AI tools?",
                    "description": "This helps us show the right level of detail",
                    "option_type": "choice",
                    "choices": [
                        {
                            "value": "beginner",
                            "label": "Beginner",
                            "description": "New to AI tools, prefer simple options",
                        },
                        {
                            "value": "intermediate",
                            "label": "Some Experience",
                            "description": "Used AI tools before, comfortable with basic settings",
                        },
                        {
                            "value": "advanced",
                            "label": "Experienced",
                            "description": "Familiar with AI, want more control",
                        },
                        {
                            "value": "expert",
                            "label": "Expert",
                            "description": "Power user, need full customization",
                        },
                    ],
                    "default_value": "intermediate",
                },
            }
        )

        # Step 4: AI Providers
        steps.append(
            {
                "step": 4,
                "title": "Choose AI Providers",
                "description": "Select which AI services you'd like to use",
                "type": "provider_selection",
                "content": {
                    "message": "We recommend starting with 1-2 providers. You can add more later.",
                    "providers": self._get_simplified_provider_options(use_case),
                },
            }
        )

        # Step 5: API Keys
        steps.append(
            {
                "step": 5,
                "title": "Add API Keys",
                "description": "Enter API keys for your selected providers",
                "type": "api_keys",
                "content": {
                    "message": "API keys are encrypted and stored securely on your device.",
                    "help_link": "Need help getting API keys?",
                    "providers_with_keys": [
                        "claude",
                        "openai",
                        "gemini",
                        "perplexity",
                    ],
                },
            }
        )

        # Step 6: Tools and Features
        steps.append(
            {
                "step": 6,
                "title": "Enable Tools",
                "description": "Choose which capabilities to enable",
                "type": "feature_selection",
                "content": {
                    "message": "These tools enhance Omnimancer's capabilities. All require your approval before use.",
                    "features": [
                        {
                            "key": "web_search",
                            "name": "Web Search",
                            "description": "Search the internet for current information",
                            "recommended": True,
                        },
                        {
                            "key": "file_operations",
                            "name": "File Operations",
                            "description": "Read and write files (with your approval)",
                            "recommended": True,
                        },
                        {
                            "key": "code_tools",
                            "name": "Coding Tools",
                            "description": "Git, code analysis, and development tools",
                            "recommended": use_case == "coding",
                        },
                    ],
                },
            }
        )

        # Step 7: Review and Finish
        steps.append(
            {
                "step": 7,
                "title": "Review Your Configuration",
                "description": "Confirm your settings and complete setup",
                "type": "review",
                "content": {
                    "message": "Review your configuration below. You can change any of these settings later.",
                    "final_steps": [
                        "Save your configuration",
                        "Test your first AI conversation",
                        "Explore additional features",
                    ],
                },
            }
        )

        return steps

    def _get_simplified_provider_options(self, use_case: str) -> List[Dict[str, Any]]:
        """Get simplified provider options based on use case."""
        all_providers = self.get_simple_mode_providers()

        # Filter and rank providers based on use case
        relevant_providers = []
        for provider in all_providers:
            if use_case in provider.use_case_fit or "general" in provider.use_case_fit:
                # Calculate relevance score
                relevance = 1.0
                if use_case in provider.use_case_fit:
                    relevance += 0.5
                if provider.name in ["claude", "openai"]:  # Popular choices
                    relevance += 0.3
                if provider.name in ["claude-code", "ollama"]:  # Free options
                    relevance += 0.2

                relevant_providers.append(
                    {
                        "provider": asdict(provider),
                        "relevance": relevance,
                        "setup_difficulty": self._get_provider_setup_difficulty(
                            provider.name
                        ),
                        "recommended": relevance > 1.3,
                    }
                )

        # Sort by relevance
        relevant_providers.sort(key=lambda x: x["relevance"], reverse=True)

        return relevant_providers[:5]  # Return top 5 most relevant

    def _get_provider_setup_difficulty(self, provider_name: str) -> str:
        """Get setup difficulty level for a provider."""
        difficulty_map = {
            "claude-code": "Easy (No API key needed)",
            "ollama": "Moderate (Local installation required)",
            "claude": "Easy (API key required)",
            "openai": "Easy (API key required)",
            "gemini": "Easy (Free tier available)",
            "perplexity": "Moderate (Subscription required)",
        }
        return difficulty_map.get(provider_name, "Easy")

    def get_provider_setup_instructions(self, provider_name: str) -> Dict[str, Any]:
        """Get setup instructions for a specific provider."""
        instructions = {
            "claude": {
                "title": "Anthropic Claude Setup",
                "steps": [
                    "Visit https://console.anthropic.com",
                    "Sign up or log in to your account",
                    "Go to the API Keys section",
                    "Create a new API key",
                    "Copy the key (starts with 'sk-ant-')",
                    "Paste it in the API key field",
                ],
                "api_key_format": "sk-ant-...",
                "cost_info": "Pay-per-use pricing, free tier available",
                "recommended_models": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-haiku-20240307",
                ],
            },
            "openai": {
                "title": "OpenAI GPT Setup",
                "steps": [
                    "Visit https://platform.openai.com",
                    "Sign up or log in to your account",
                    "Go to the API Keys section",
                    "Create a new API key",
                    "Copy the key (starts with 'sk-')",
                    "Paste it in the API key field",
                ],
                "api_key_format": "sk-...",
                "cost_info": "Pay-per-use pricing, free tier available for new users",
                "recommended_models": ["gpt-4o", "gpt-4o-mini"],
            },
            "gemini": {
                "title": "Google Gemini Setup",
                "steps": [
                    "Visit https://makersuite.google.com/app/apikey",
                    "Sign in with your Google account",
                    "Create a new API key",
                    "Copy the key (starts with 'AIza')",
                    "Paste it in the API key field",
                ],
                "api_key_format": "AIza...",
                "cost_info": "Free tier with generous limits, pay-per-use beyond limits",
                "recommended_models": ["gemini-1.5-pro", "gemini-1.5-flash"],
            },
        }

        return instructions.get(
            provider_name,
            {
                "title": f"{provider_name} Setup",
                "steps": ["Refer to provider documentation for setup instructions"],
                "api_key_format": "Check provider documentation",
                "cost_info": "Varies by provider",
            },
        )

    def get_configuration_preview(self, answers: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a preview of what the configuration will look like."""
        preview = {
            "use_case": answers.get("use_case", "general"),
            "enabled_providers": [],
            "estimated_monthly_cost": "$0",
            "features_enabled": [],
            "setup_complete": False,
        }

        # Calculate enabled providers and costs
        cost_estimates = {
            "claude": 15,
            "openai": 15,
            "gemini": 5,
            "perplexity": 20,
        }
        total_cost = 0

        if answers.get("enable_claude") and answers.get("claude_api_key"):
            preview["enabled_providers"].append("Anthropic Claude")
            total_cost += cost_estimates.get("claude", 0)

        if answers.get("enable_openai") and answers.get("openai_api_key"):
            preview["enabled_providers"].append("OpenAI GPT")
            total_cost += cost_estimates.get("openai", 0)

        if answers.get("enable_gemini") and answers.get("gemini_api_key"):
            preview["enabled_providers"].append("Google Gemini")
            total_cost += cost_estimates.get("gemini", 0)

        if answers.get("enable_claude_code"):
            preview["enabled_providers"].append("Claude Code (Free)")

        # Update cost estimate
        if total_cost == 0:
            preview["estimated_monthly_cost"] = "Free"
        elif total_cost <= 10:
            preview["estimated_monthly_cost"] = f"~${total_cost}/month"
        else:
            preview["estimated_monthly_cost"] = (
                f"~${total_cost}-{total_cost + 20}/month"
            )

        # Features
        if answers.get("enable_web_search"):
            preview["features_enabled"].append("Web Search")
        if answers.get("enable_file_operations"):
            preview["features_enabled"].append("File Operations")
        if answers.get("enable_code_tools"):
            preview["features_enabled"].append("Coding Tools")

        # Check if setup would be complete
        is_valid, errors = self.validate_simple_answers(answers)
        preview["setup_complete"] = is_valid
        preview["missing_items"] = errors

        return preview

    def get_recommended_configuration(
        self, context: Optional[ConfigurationContext] = None
    ) -> ConfigurationRecommendation:
        """Get configuration recommendation based on user context."""
        ctx = context or self.context

        if ctx.use_case == "coding":
            return ConfigurationRecommendation(
                template_name="coding",
                confidence=0.9,
                reasons=[
                    "Optimized for software development",
                    "Includes code-capable models",
                    "Has essential development tools",
                ],
                providers=["claude", "openai", "claude-code"],
                estimated_cost="low",
                complexity=ConfigurationComplexity.BASIC,
            )
        elif ctx.use_case == "research":
            return ConfigurationRecommendation(
                template_name="research",
                confidence=0.9,
                reasons=[
                    "Includes web search capabilities",
                    "Large context models for document analysis",
                    "Citation and fact-checking support",
                ],
                providers=["perplexity", "claude", "gemini"],
                estimated_cost="medium",
                complexity=ConfigurationComplexity.BASIC,
            )
        else:
            return ConfigurationRecommendation(
                template_name="general",
                confidence=0.7,
                reasons=[
                    "Balanced configuration for general use",
                    "Good mix of capabilities",
                    "Suitable for various tasks",
                ],
                providers=["claude", "openai", "gemini"],
                estimated_cost="medium",
                complexity=ConfigurationComplexity.BASIC,
            )

    def apply_template_configuration(
        self, template_name: str, api_keys: Dict[str, str] = None
    ) -> Tuple[bool, List[str]]:
        """Apply a template configuration in simple mode."""
        messages = []

        try:
            template = self.template_manager.get_template(template_name)
            config = self.config_manager.get_config()

            # Apply provider configurations
            for provider_name in template.recommended_providers[:3]:
                provider_config = (
                    self.template_manager.create_provider_config_from_template(
                        template, provider_name
                    )
                )

                if provider_config:
                    if api_keys and provider_name in api_keys:
                        provider_config.api_key = api_keys[provider_name]
                    elif provider_name in ["claude-code", "ollama"]:
                        provider_config.api_key = None

                    config.providers[provider_name] = provider_config
                    messages.append(
                        f"Configured {provider_name} with {provider_config.model}"
                    )

            if template.recommended_providers:
                config.default_provider = template.recommended_providers[0]
                messages.append(f"Set default provider to {config.default_provider}")

            # Apply MCP configuration (simplified)
            mcp_config = self.template_manager.create_mcp_config_from_template(template)
            enabled_count = 0
            for server_name, server_config in mcp_config.servers.items():
                if enabled_count < 2:
                    server_config.enabled = True
                    enabled_count += 1
                else:
                    server_config.enabled = False

            config.mcp = mcp_config
            messages.append(f"Enabled {enabled_count} MCP tools")

            self.config_manager.save_config(config)
            messages.append("Configuration saved successfully")

            return True, messages

        except Exception as e:
            logger.error(f"Failed to apply template configuration: {e}")
            return False, [f"Failed to apply template: {str(e)}"]

    def validate_simple_answers(
        self, answers: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate answers from the simple configuration interface."""
        errors = []

        required_fields = ["use_case", "experience_level"]
        for field in required_fields:
            if field not in answers or not answers[field]:
                errors.append(f"{field.replace('_', ' ').title()} is required")

        enabled_providers = []
        if answers.get("enable_claude") and answers.get("claude_api_key"):
            enabled_providers.append("claude")
        if answers.get("enable_openai") and answers.get("openai_api_key"):
            enabled_providers.append("openai")
        if answers.get("enable_claude_code"):
            enabled_providers.append("claude-code")

        if not enabled_providers:
            errors.append("At least one AI provider must be enabled and configured")

        # Validate API key formats
        if answers.get("claude_api_key"):
            key = answers["claude_api_key"]
            if not key.startswith("sk-ant-"):
                errors.append("Claude API key should start with 'sk-ant-'")

        if answers.get("openai_api_key"):
            key = answers["openai_api_key"]
            if not key.startswith("sk-"):
                errors.append("OpenAI API key should start with 'sk-'")

        return len(errors) == 0, errors

    def generate_simple_configuration(
        self, answers: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Generate a complete configuration based on simple answers."""
        messages = []

        try:
            use_case = answers.get("use_case", "general")
            experience_level = answers.get("experience_level", "intermediate")

            context = ConfigurationContext(
                user_level=experience_level,
                use_case=use_case,
                preferred_mode=self.current_mode,
            )
            self.set_configuration_context(context)

            recommendation = self.get_recommended_configuration(context)
            messages.append(
                f"Recommended template: {recommendation.template_name} (confidence: {recommendation.confidence:.0%})"
            )

            api_keys = {}
            if answers.get("enable_claude") and answers.get("claude_api_key"):
                api_keys["claude"] = answers["claude_api_key"]
                messages.append("Claude API key provided")

            if answers.get("enable_openai") and answers.get("openai_api_key"):
                api_keys["openai"] = answers["openai_api_key"]
                messages.append("OpenAI API key provided")

            success, template_messages = self.apply_template_configuration(
                recommendation.template_name, api_keys
            )

            if success:
                messages.extend(template_messages)
                config_dict = self.get_configuration_summary()

                config_dict["user_preferences"] = {
                    "use_case": use_case,
                    "experience_level": experience_level,
                    "tools_enabled": {
                        "web_search": answers.get("enable_web_search", True),
                        "file_operations": answers.get("enable_file_operations", True),
                        "code_tools": answers.get("enable_code_tools", True),
                    },
                }

                messages.append("Configuration generated successfully")
                return True, config_dict, messages
            else:
                return False, {}, messages

        except Exception as e:
            logger.error(f"Failed to generate simple configuration: {e}")
            return False, {}, [f"Configuration generation failed: {str(e)}"]

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration appropriate for the current mode."""
        try:
            config = self.config_manager.get_config()

            summary = {
                "mode": self.current_mode.value,
                "context": asdict(self.context),
                "default_provider": config.default_provider,
                "enabled_providers": len(config.providers),
                "mcp_enabled": config.mcp.enabled,
                "enabled_mcp_servers": len(config.mcp.get_enabled_servers()),
            }

            if self.current_mode == ConfigurationMode.SIMPLE:
                summary["providers"] = []
                for name, provider_config in list(config.providers.items())[:3]:
                    summary["providers"].append(
                        {
                            "name": name,
                            "model": provider_config.model,
                            "enabled": provider_config.enabled,
                        }
                    )

                summary["tools"] = list(config.mcp.get_enabled_servers().keys())[:3]

            elif self.current_mode == ConfigurationMode.ADVANCED:
                summary.update(self.config_manager.get_config_summary())

            return summary

        except Exception as e:
            logger.error(f"Failed to get configuration summary: {e}")
            return {"error": str(e)}

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def _load_mode_preferences(self) -> None:
        """Load mode preferences from storage."""
        try:
            storage_path = self.config_manager.get_storage_path()
            preferences_file = storage_path / "mode_preferences.json"

            if preferences_file.exists():
                with open(preferences_file, "r") as f:
                    data = json.load(f)

                self.current_mode = ConfigurationMode(data.get("mode", "simple"))

                context_data = data.get("context", {})
                self.context = ConfigurationContext(
                    user_level=context_data.get("user_level", "beginner"),
                    use_case=context_data.get("use_case", "general"),
                    primary_providers=context_data.get("primary_providers", []),
                    preferred_mode=ConfigurationMode(
                        context_data.get("preferred_mode", "simple")
                    ),
                    show_experimental=context_data.get("show_experimental", False),
                )

        except Exception as e:
            logger.debug(f"Could not load mode preferences: {e}")

    def _save_mode_preferences(self) -> None:
        """Save mode preferences to storage."""
        try:
            storage_path = self.config_manager.get_storage_path()
            preferences_file = storage_path / "mode_preferences.json"

            data = {
                "mode": self.current_mode.value,
                "context": asdict(self.context),
                "updated_at": datetime.now().isoformat(),
            }

            with open(preferences_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.debug(f"Could not save mode preferences: {e}")


def create_configuration_provider(
    config_manager: Optional[ConfigManager] = None,
) -> ConfigurationProvider:
    """
    Create a configuration provider with default settings.

    Args:
        config_manager: Optional ConfigManager instance

    Returns:
        Configured ConfigurationProvider instance
    """
    return ConfigurationProvider(config_manager)
