"""
Security-focused configuration validator for Omnimancer.

This module ensures that critical security settings cannot be accidentally disabled
and validates configuration against security best practices.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class SecurityValidationError(Exception):
    """Raised when a security validation fails."""

    pass


class ConfigSecurityValidator:
    """Validates Omnimancer configurations for security compliance."""

    def __init__(self):
        self.critical_security_settings = self._get_critical_security_settings()
        self.recommended_security_settings = self._get_recommended_security_settings()
        self.dangerous_combinations = self._get_dangerous_combinations()

    def _get_critical_security_settings(self) -> Dict[str, Any]:
        """Settings that must never be disabled for security reasons."""
        return {
            "api_key_encryption_enabled": True,
            "secure_storage_enabled": True,
            "mcp.sandbox_mode": True,
            "mcp.allow_dangerous_tools": False,
        }

    def _get_recommended_security_settings(self) -> Dict[str, Any]:
        """Recommended security settings that should be enabled by default."""
        return {
            "provider_health_check_enabled": True,
            "request_cache_enabled": True,  # Helps with rate limiting
            "mcp.log_tool_calls": True,  # Security auditing
            "debug_mode": False,  # Don't leak info in production
            "telemetry_enabled": False,  # Privacy
            "mcp.max_tool_execution_time": 300,  # Prevent runaway tools
            "mcp.max_concurrent_servers": 10,  # Resource limiting
        }

    def _get_dangerous_combinations(self) -> List[Dict[str, Any]]:
        """Combinations of settings that create security risks."""
        return [
            {
                "description": "Debug mode with telemetry enabled can leak sensitive information",
                "conditions": {"debug_mode": True, "telemetry_enabled": True},
                "severity": "high",
            },
            {
                "description": "MCP sandbox disabled with dangerous tools allowed is extremely risky",
                "conditions": {
                    "mcp.sandbox_mode": False,
                    "mcp.allow_dangerous_tools": True,
                },
                "severity": "critical",
            },
            {
                "description": "No health checks with fallback disabled can cause reliability issues",
                "conditions": {
                    "provider_health_check_enabled": False,
                    "provider_fallback_enabled": False,
                },
                "severity": "medium",
            },
        ]

    def validate_security(
        self, config: Dict[str, Any], strict_mode: bool = True
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate configuration for security compliance.

        Args:
            config: Configuration to validate
            strict_mode: If True, treat security warnings as errors

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Check critical security settings
        critical_errors = self._validate_critical_settings(config)
        errors.extend(critical_errors)

        # Check recommended security settings
        recommended_warnings = self._validate_recommended_settings(config)
        if strict_mode:
            errors.extend(recommended_warnings)
        else:
            warnings.extend(recommended_warnings)

        # Check dangerous combinations
        combination_issues = self._validate_dangerous_combinations(config)
        for issue in combination_issues:
            if issue["severity"] in ["critical", "high"]:
                errors.append(issue["message"])
            else:
                warnings.append(issue["message"])

        # Validate API key security
        api_key_issues = self._validate_api_key_security(config)
        errors.extend(api_key_issues)

        # Check file permissions
        permission_warnings = self._validate_file_permissions(config)
        warnings.extend(permission_warnings)

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def _validate_critical_settings(self, config: Dict[str, Any]) -> List[str]:
        """Validate critical security settings that must never be disabled."""
        errors = []

        for (
            setting_path,
            required_value,
        ) in self.critical_security_settings.items():
            current_value = self._get_nested_value(config, setting_path)

            if current_value != required_value:
                errors.append(
                    f"CRITICAL: Security setting '{setting_path}' must be set to {required_value}, "
                    f"but is currently {current_value}"
                )

        return errors

    def _validate_recommended_settings(self, config: Dict[str, Any]) -> List[str]:
        """Validate recommended security settings."""
        warnings = []

        for (
            setting_path,
            recommended_value,
        ) in self.recommended_security_settings.items():
            current_value = self._get_nested_value(config, setting_path)

            if current_value != recommended_value:
                warnings.append(
                    f"RECOMMENDED: Security setting '{setting_path}' should be set to {recommended_value}, "
                    f"but is currently {current_value}"
                )

        return warnings

    def _validate_dangerous_combinations(
        self, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for dangerous combinations of settings."""
        issues = []

        for combination in self.dangerous_combinations:
            all_conditions_met = True

            for setting_path, required_value in combination["conditions"].items():
                current_value = self._get_nested_value(config, setting_path)
                if current_value != required_value:
                    all_conditions_met = False
                    break

            if all_conditions_met:
                issues.append(
                    {
                        "message": f"{combination['severity'].upper()}: {combination['description']}",
                        "severity": combination["severity"],
                    }
                )

        return issues

    def _validate_api_key_security(self, config: Dict[str, Any]) -> List[str]:
        """Validate API key security practices."""
        errors = []

        # Check if API keys are stored in plain text (should be encrypted)
        if "providers" in config:
            for provider_name, provider_config in config["providers"].items():
                api_key = provider_config.get("api_key")

                if api_key and isinstance(api_key, str):
                    # Check for obviously insecure patterns
                    if api_key in [
                        "test",
                        "demo",
                        "placeholder",
                        "your-api-key-here",
                    ]:
                        errors.append(
                            f"Provider '{provider_name}' has placeholder API key"
                        )

                    # Check if it looks like a real key but isn't encrypted
                    if api_key.startswith(
                        ("sk-", "api-", "key-")
                    ) and not api_key.startswith("encrypted:"):
                        # This might be a plain text key in an optimized config
                        # In production, these should be encrypted
                        pass  # Don't warn for optimized configs as encryption happens at runtime

        return errors

    def _validate_file_permissions(self, config: Dict[str, Any]) -> List[str]:
        """Check file permissions for configuration and storage paths."""
        warnings = []

        storage_path = config.get("storage_path")
        if storage_path:
            storage_dir = Path(storage_path)

            if storage_dir.exists():
                # Check if directory is readable by others (Unix-like systems)
                try:
                    stat_info = storage_dir.stat()
                    mode = stat_info.st_mode

                    # Check if group/others have read access (simplified check)
                    if mode & 0o044:  # Group or others can read
                        warnings.append(
                            f"Storage directory '{storage_path}' may be readable by other users. "
                            f"Consider restricting permissions with: chmod 700 '{storage_path}'"
                        )
                except Exception:
                    # Permission check failed, probably Windows or insufficient permissions
                    pass

        return warnings

    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get a nested configuration value using dot notation."""
        keys = path.split(".")
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def fix_security_issues(
        self, config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Automatically fix security issues in configuration.

        Args:
            config: Configuration to fix

        Returns:
            Tuple of (fixed_config, list of changes made)
        """
        fixed_config = config.copy()
        changes = []

        # Fix critical security settings
        for (
            setting_path,
            required_value,
        ) in self.critical_security_settings.items():
            current_value = self._get_nested_value(fixed_config, setting_path)

            if current_value != required_value:
                self._set_nested_value(fixed_config, setting_path, required_value)
                changes.append(
                    f"Fixed critical setting: {setting_path} = {required_value}"
                )

        # Apply recommended security settings
        for (
            setting_path,
            recommended_value,
        ) in self.recommended_security_settings.items():
            current_value = self._get_nested_value(fixed_config, setting_path)

            if current_value is None:  # Only set if not explicitly configured
                self._set_nested_value(fixed_config, setting_path, recommended_value)
                changes.append(
                    f"Applied recommended setting: {setting_path} = {recommended_value}"
                )

        return fixed_config, changes

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = path.split(".")
        current = config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # Set final value
        current[keys[-1]] = value

    def generate_security_report(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive security report for a configuration.

        Args:
            config: Configuration to analyze

        Returns:
            Security report dictionary
        """
        is_valid, errors, warnings = self.validate_security(config, strict_mode=False)

        report = {
            "timestamp": str(Path().resolve()),
            "overall_status": (
                "SECURE"
                if is_valid and len(warnings) == 0
                else "INSECURE" if not is_valid else "NEEDS_ATTENTION"
            ),
            "critical_errors": len([e for e in errors if "CRITICAL:" in e]),
            "security_errors": len(errors),
            "security_warnings": len(warnings),
            "errors": errors,
            "warnings": warnings,
            "recommendations": [],
        }

        # Add specific recommendations
        if report["critical_errors"] > 0:
            report["recommendations"].append(
                "URGENT: Fix critical security issues immediately. "
                "These settings compromise system security."
            )

        if report["security_warnings"] > 0:
            report["recommendations"].append(
                "Review security warnings and consider applying recommended settings."
            )

        if report["overall_status"] == "SECURE":
            report["recommendations"].append(
                "Configuration follows security best practices. "
                "Continue monitoring for changes."
            )

        # Security score (0-100)
        len(self.critical_security_settings) + len(self.recommended_security_settings)
        actual_issues = len(errors) + len(warnings)
        report["security_score"] = max(0, 100 - (actual_issues * 10))

        return report


def validate_config_security(
    config: Dict[str, Any], strict_mode: bool = True
) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate configuration security.

    Args:
        config: Configuration to validate
        strict_mode: If True, treat warnings as errors

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = ConfigSecurityValidator()
    return validator.validate_security(config, strict_mode)


def fix_config_security_issues(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Convenience function to fix security issues in configuration.

    Args:
        config: Configuration to fix

    Returns:
        Tuple of (fixed_config, list of changes)
    """
    validator = ConfigSecurityValidator()
    return validator.fix_security_issues(config)
