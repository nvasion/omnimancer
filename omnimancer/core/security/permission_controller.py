"""Permission controller for managing agent access permissions."""

import os
import re
import tempfile
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class PermissionLevel(Enum):
    """Permission levels for different operations."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class PermissionOperation:
    """Represents an operation that needs permission validation."""

    def __init__(
        self,
        operation_type: str,
        path: Optional[str] = None,
        command: Optional[str] = None,
        **kwargs,
    ):
        self.operation_type = operation_type
        self.path = path
        self.command = command
        self.metadata = kwargs


class PermissionController:
    """Controls and validates permissions for agent operations."""

    def __init__(self):
        self.restricted_paths = self._get_default_restricted_paths()
        self.allowed_commands = self._get_default_allowed_commands()
        self.permission_rules = self._get_default_permission_rules()

        # Initialize approval memory storage
        self._approval_memory: Dict[str, Dict[str, Any]] = {}

    def _get_default_restricted_paths(self) -> Set[str]:
        """Get default set of restricted file paths."""
        return {
            # SSH and security keys
            ".ssh",
            "~/.ssh",
            "/home/*/.ssh",
            "/Users/*/.ssh",
            # Environment and config files
            ".env",
            ".env.local",
            ".env.production",
            # System directories
            "/etc",
            "/System",
            "/sys",
            "/proc",
            "/boot",
            "/root",
            # Package manager directories
            "/usr/bin",
            "/usr/sbin",
            "/sbin",
            # Database files
            "*.db",
            "*.sqlite",
            "*.sqlite3",
            # Credential files
            "*credentials*",
            "*password*",
            "*secret*",
            "*token*",
            "*key*",
            ".aws/credentials",
            ".config/gcloud",
        }

    def _get_default_allowed_commands(self) -> Set[str]:
        """Get default set of allowed shell commands."""
        return {
            # File operations
            "ls",
            "cat",
            "head",
            "tail",
            "grep",
            "find",
            "locate",
            "cp",
            "mv",
            "mkdir",
            "rmdir",
            "touch",
            "chmod",
            "chown",
            # Text processing
            "sed",
            "awk",
            "sort",
            "uniq",
            "wc",
            "tr",
            "cut",
            # Basic commands
            "echo",
            "printf",
            "true",
            "false",
            "sleep",
            # Development tools
            "git",
            "npm",
            "pip",
            "python",
            "node",
            "go",
            "cargo",
            "make",
            "cmake",
            "gcc",
            "clang",
            # System info (read-only)
            "ps",
            "top",
            "df",
            "du",
            "free",
            "uptime",
            "whoami",
            "pwd",
            "which",
            "whereis",
            "uname",
            # Network (limited)
            "curl",
            "wget",
            "ping",
        }

    def _get_default_permission_rules(self) -> Dict[str, PermissionLevel]:
        """Get default permission rules for operation types."""
        return {
            "file_read": PermissionLevel.READ,
            "file_write": PermissionLevel.WRITE,
            "file_delete": PermissionLevel.WRITE,
            "file_execute": PermissionLevel.EXECUTE,
            "command_execute": PermissionLevel.EXECUTE,
            "network_request": PermissionLevel.READ,
            "system_info": PermissionLevel.READ,
            "package_install": PermissionLevel.ADMIN,
            "service_control": PermissionLevel.ADMIN,
        }

    def validate_path_access(self, path: str, operation: str = "read") -> bool:
        """Validate if path access is allowed."""
        try:
            # Normalize path
            normalized_path = str(Path(path).resolve())

            # Check against restricted paths
            for restricted in self.restricted_paths:
                if self._path_matches_pattern(normalized_path, restricted):
                    return False

            # Additional checks for write operations
            if operation in ["write", "delete", "execute"]:
                # Don't allow writes to system directories
                system_prefixes = ["/usr", "/etc", "/System", "/sys", "/proc"]
                if any(
                    normalized_path.startswith(prefix) for prefix in system_prefixes
                ):
                    return False

                # Allow writes to project directory and safe temporary directories
                project_root = os.getcwd()
                safe_temp_prefixes = [
                    "/tmp",
                    "/var/tmp",
                    tempfile.gettempdir(),
                ]

                # Check if path is in project directory or safe temp directory
                is_in_project = normalized_path.startswith(project_root)
                is_in_safe_temp = any(
                    normalized_path.startswith(prefix) for prefix in safe_temp_prefixes
                )

                if not (is_in_project or is_in_safe_temp):
                    return False

            return True

        except (OSError, ValueError):
            # If path resolution fails, deny access
            return False

    def validate_command(self, command: str) -> bool:
        """Validate if command execution is allowed."""
        # Extract base command (first word)
        base_command = command.strip().split()[0]

        # Remove path prefixes
        base_command = os.path.basename(base_command)

        # Check if command is in allowed list
        if base_command not in self.allowed_commands:
            return False

        # Additional security checks
        if self._contains_dangerous_patterns(command):
            return False

        return True

    def validate_operation(self, operation: PermissionOperation) -> bool:
        """Validate if an operation is allowed."""
        op_type = operation.operation_type

        # Check if operation type is allowed
        if op_type not in self.permission_rules:
            return False

        # Validate path if provided
        if operation.path:
            required_level = self.permission_rules[op_type]
            access_type = self._permission_level_to_access_type(required_level)
            if not self.validate_path_access(operation.path, access_type):
                return False

        # Validate command if provided
        if operation.command:
            if not self.validate_command(operation.command):
                return False

        return True

    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches a pattern (supports wildcards)."""
        # Expand user home directory
        pattern = os.path.expanduser(pattern)

        # Handle relative patterns (like .ssh, .env) by checking if they appear in the path
        if not pattern.startswith("/") and not pattern.startswith("~"):
            # For relative patterns, check if the pattern matches any part of the path
            path_parts = path.split("/")
            for i, part in enumerate(path_parts):
                if "*" not in pattern:
                    # Exact match for directory/file name
                    if part == pattern:
                        return True
                    # Check if any part of the path ends with the pattern (for files in directories)
                    remaining_path = "/".join(path_parts[i:])
                    if (
                        remaining_path.startswith(pattern + "/")
                        or remaining_path == pattern
                    ):
                        return True
                else:
                    # Wildcard pattern matching
                    regex_pattern = pattern.replace("*", ".*")
                    if re.match(f"^{regex_pattern}$", part, re.IGNORECASE):
                        return True
            return False

        # For absolute patterns, use the original logic
        regex_pattern = pattern.replace("*", ".*")

        # For exact matches (no wildcards), check if path starts with pattern
        if "*" not in pattern:
            # Check if path starts with the pattern (directory matching)
            if path.startswith(pattern):
                # Ensure it's a proper directory match (either exact or followed by /)
                if len(path) == len(pattern) or path[len(pattern)] == "/":
                    return True
            # Also check exact match
            regex_pattern = f"^{re.escape(pattern)}$"
        else:
            # For wildcard patterns, use full regex matching
            regex_pattern = f"^{regex_pattern}$"

        return bool(re.match(regex_pattern, path, re.IGNORECASE))

    def _contains_dangerous_patterns(self, command: str) -> bool:
        """Check if command contains dangerous patterns."""
        dangerous_patterns = [
            r"[;&|]",  # Command chaining
            r"`",  # Command substitution
            r"\$\(",  # Command substitution
            r">\s*/dev/",  # Device access
            r"rm\s+-rf",  # Recursive deletion
            r"sudo",  # Privilege escalation
            r"su\s",  # User switching
            r"chmod\s+[0-7]{3,4}",  # Permission changes
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                return True

        return False

    def _permission_level_to_access_type(self, level: PermissionLevel) -> str:
        """Convert permission level to access type string."""
        if level == PermissionLevel.READ:
            return "read"
        elif level == PermissionLevel.WRITE:
            return "write"
        elif level == PermissionLevel.EXECUTE:
            return "execute"
        else:
            return "admin"

    def add_restricted_path(self, path: str) -> None:
        """Add a path to the restricted list."""
        self.restricted_paths.add(path)

    def remove_restricted_path(self, path: str) -> None:
        """Remove a path from the restricted list."""
        self.restricted_paths.discard(path)

    def add_allowed_command(self, command: str) -> None:
        """Add a command to the allowed list."""
        self.allowed_commands.add(command)

    def remove_allowed_command(self, command: str) -> None:
        """Remove a command from the allowed list."""
        self.allowed_commands.discard(command)

    def get_restricted_paths(self) -> List[str]:
        """Get list of restricted paths."""
        return list(self.restricted_paths)

    def get_allowed_commands(self) -> List[str]:
        """Get list of allowed commands."""
        return list(self.allowed_commands)

    async def check_operation_permission(
        self,
        operation_type: str,
        operation_signature: str,
        operation_data: Dict[str, Any],
    ) -> bool:
        """
        Check if operation has stored permission (remembered approval).

        Args:
            operation_type: Type of operation
            operation_signature: Operation signature for matching
            operation_data: Operation data for validation

        Returns:
            True if operation has stored permission, False otherwise
        """
        try:
            # Check if we have stored approval for this signature
            if operation_signature in self._approval_memory:
                approval_data = self._approval_memory[operation_signature]

                # Check if approval has expired
                if "expires_at" in approval_data:
                    expires_at = datetime.fromisoformat(approval_data["expires_at"])
                    if datetime.now() > expires_at:
                        # Remove expired approval
                        del self._approval_memory[operation_signature]
                        return False

                # Verify operation type matches
                if approval_data.get("operation_type") == operation_type:
                    return True

            return False

        except Exception:
            # If there's any error in checking, don't auto-approve
            return False

    async def grant_operation_permission(
        self,
        operation_type: str,
        operation_signature: str,
        metadata: Dict[str, Any],
        expires_hours: int = 24,
    ) -> None:
        """
        Store operation permission for future auto-approval.

        Args:
            operation_type: Type of operation
            operation_signature: Operation signature for matching
            metadata: Additional metadata about the approval
            expires_hours: Hours until approval expires (default 24)
        """
        try:
            expires_at = datetime.now() + timedelta(hours=expires_hours)

            self._approval_memory[operation_signature] = {
                "operation_type": operation_type,
                "stored_at": datetime.now().isoformat(),
                "expires_at": expires_at.isoformat(),
                "metadata": metadata,
            }

        except Exception:
            # Log error but don't fail the operation
            pass

    def get_stored_approvals(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored approvals with their metadata."""
        return self._approval_memory.copy()

    def get_permission_rules(self) -> List[Dict[str, Any]]:
        """Get current permission rules in dictionary format."""
        rules = []
        rule_id = 1

        # Convert permission rules to display format
        for op_type, level in self.permission_rules.items():
            rule = {
                "id": str(rule_id),
                "pattern": op_type.replace("_", " ").title(),
                "type": "operation",
                "level": level.value,
            }
            rules.append(rule)
            rule_id += 1

        # Add path-based rules
        for path in self.restricted_paths:
            rule = {
                "id": str(rule_id),
                "pattern": path,
                "type": "file",
                "level": "blocked",
            }
            rules.append(rule)
            rule_id += 1

        # Add command rules
        for command in list(self.allowed_commands)[:10]:  # Limit for display
            rule = {
                "id": str(rule_id),
                "pattern": command,
                "type": "command",
                "level": "allowed",
            }
            rules.append(rule)
            rule_id += 1

        return rules

    def get_learned_permissions(self) -> List[Dict[str, Any]]:
        """Get learned permissions in display format."""
        learned = []
        for signature, approval_data in self._approval_memory.items():
            learned.append(
                {
                    "pattern": signature[:50] + ("..." if len(signature) > 50 else ""),
                    "decision": "approved",
                    "count": 1,  # Could track actual usage count in the future
                    "last_used": approval_data.get("stored_at", ""),
                }
            )
        return learned

    def revoke_approval(self, operation_signature: str) -> bool:
        """
        Revoke a stored approval.

        Args:
            operation_signature: Signature of approval to revoke

        Returns:
            True if approval was found and revoked, False otherwise
        """
        if operation_signature in self._approval_memory:
            del self._approval_memory[operation_signature]
            return True
        return False

    def cleanup_expired_approvals(self) -> int:
        """
        Remove expired approvals from memory.

        Returns:
            Number of approvals that were removed
        """
        expired_signatures = []
        now = datetime.now()

        for signature, approval_data in self._approval_memory.items():
            if "expires_at" in approval_data:
                try:
                    expires_at = datetime.fromisoformat(approval_data["expires_at"])
                    if now > expires_at:
                        expired_signatures.append(signature)
                except Exception:
                    # If we can't parse the date, consider it expired
                    expired_signatures.append(signature)

        # Remove expired approvals
        for signature in expired_signatures:
            del self._approval_memory[signature]

        return len(expired_signatures)
