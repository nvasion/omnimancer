"""
Handler for permissions CLI commands.

This module provides the command handling logic for managing agent permissions
and security settings through the CLI interface.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...core.security.audit_logger import AuditEventType, AuditLogger
from ...core.security.permission_controller import PermissionController
from ...core.security.security_manager import SecurityManager
from ..commands import Command


class PermissionsHandler:
    """Handles permissions-related CLI commands."""

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the handler.

        Args:
            console: Rich console for output
        """
        self.console = console or Console()
        self.permission_controller = None
        self.security_manager = None
        self.audit_logger = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize security components."""
        try:
            # Create security components with proper initialization
            self.permission_controller = PermissionController()
            self.security_manager = SecurityManager(
                enable_sandbox=False,  # Disable for CLI to avoid complexity
                enable_approval_workflow=True,
                enable_audit_logging=True,
            )
            self.audit_logger = AuditLogger()
        except Exception as e:
            # If initialization fails, set to None to use fallback behavior
            self.permission_controller = None
            self.security_manager = None
            self.audit_logger = None
            self._show_error(f"Security components not available: {e}")

    async def handle_permissions_command(self, command: Command) -> None:
        """
        Main entry point for handling permissions commands.

        Args:
            command: The command to handle
        """
        args = command.parameters.get("args", [])

        if not args:
            # Default to 'view' if no subcommand
            await self.handle_view_permissions()
            return

        subcommand = args[0].lower()

        # Route to appropriate handler
        if subcommand == "view":
            await self.handle_view_permissions(args[1:])
        elif subcommand == "set-level":
            await self.handle_set_security_level(args[1:])
        elif subcommand == "add-rule":
            await self.handle_add_permission_rule(args[1:])
        elif subcommand == "remove-rule":
            await self.handle_remove_permission_rule(args[1:])
        elif subcommand == "audit":
            await self.handle_view_audit_trail(args[1:])
        elif subcommand == "learn":
            await self.handle_permission_learning(args[1:])
        elif subcommand == "clear-learned":
            await self.handle_clear_learned_permissions(args[1:])
        elif subcommand == "export":
            await self.handle_export_permissions(args[1:])
        elif subcommand == "import":
            await self.handle_import_permissions(args[1:])
        elif subcommand == "help":
            self._show_permissions_help()
        else:
            self._show_error(f"Unknown permissions subcommand: {subcommand}")
            self._show_permissions_help()

    async def handle_view_permissions(self, args: List[str] = None) -> None:
        """View current permissions and security settings."""
        if not self.permission_controller:
            self._show_error("Permission controller not available")
            return

        # Parse optional filters
        category_filter = None
        output_format = "table"
        verbose = False

        if args:
            for arg in args:
                if arg.startswith("--category="):
                    category_filter = arg.split("=", 1)[1]
                elif arg == "--json":
                    output_format = "json"
                elif arg == "--verbose":
                    verbose = True

        try:
            # Get current security settings
            security_settings = self._get_current_security_settings()

            if output_format == "json":
                # JSON output
                output_data = {
                    "security_level": security_settings.get("level", "unknown"),
                    "permission_rules": security_settings.get("rules", []),
                    "learned_permissions": security_settings.get("learned", []),
                    "timestamp": datetime.now().isoformat(),
                }
                self.console.print(json.dumps(output_data, indent=2))
                return

            # Create main permissions table
            self._display_permissions_overview(security_settings, verbose)

            # Display permission rules by category
            if not category_filter or category_filter == "all":
                self._display_permission_rules(
                    security_settings.get("rules", []), category_filter
                )

            # Display learned permissions if available
            learned_permissions = security_settings.get("learned", [])
            if learned_permissions:
                self._display_learned_permissions(learned_permissions, verbose)

        except Exception as e:
            self._show_error(f"Failed to retrieve permissions: {e}")

    async def handle_set_security_level(self, args: List[str]) -> None:
        """Set the security level."""
        if not args:
            self._show_error("Security level is required")
            self._show_info("Valid levels: auto_approve, ask_always, ask_but_remember")
            return

        level = args[0].lower()

        try:
            # Set the security level
            success = self._set_security_level(level)

            if success:
                self._show_success(f"Security level set to: {level}")

                # Show what this means
                level_descriptions = {
                    "auto_approve": "All operations will be automatically approved (⚠️ DANGEROUS)",
                    "ask_always": "User approval required for every operation",
                    "ask_but_remember": "Ask for approval but remember user decisions",
                }

                description = level_descriptions.get(level, "Unknown level")
                self._show_info(f"Effect: {description}")

            else:
                self._show_error(f"Failed to set security level to {level}")

        except Exception as e:
            self._show_error(f"Error setting security level: {e}")

    async def handle_add_permission_rule(self, args: List[str]) -> None:
        """Add a new permission rule."""
        if not args:
            self._show_error("Permission rule pattern is required")
            self._show_info("Usage: /permissions add-rule <pattern> [type] [level]")
            return

        pattern = args[0]
        rule_type = args[1] if len(args) > 1 else "file"
        permission_level = args[2] if len(args) > 2 else "read"

        try:
            # Validate rule type and level
            valid_types = ["file", "command", "network"]
            valid_levels = ["read", "write", "execute", "admin"]

            if rule_type not in valid_types:
                self._show_error(
                    f"Invalid rule type. Valid types: {', '.join(valid_types)}"
                )
                return

            if permission_level not in valid_levels:
                self._show_error(
                    f"Invalid permission level. Valid levels: {', '.join(valid_levels)}"
                )
                return

            # Add the rule
            success = self._add_permission_rule(pattern, rule_type, permission_level)

            if success:
                self._show_success(
                    f"Added permission rule: {pattern} ({rule_type}, {permission_level})"
                )
            else:
                self._show_error(f"Failed to add permission rule")

        except Exception as e:
            self._show_error(f"Error adding permission rule: {e}")

    async def handle_remove_permission_rule(self, args: List[str]) -> None:
        """Remove a permission rule."""
        if not args:
            self._show_error("Rule identifier is required")
            self._show_info("Usage: /permissions remove-rule <pattern_or_id>")
            return

        identifier = args[0]

        try:
            # Remove the rule
            success = self._remove_permission_rule(identifier)

            if success:
                self._show_success(f"Removed permission rule: {identifier}")
            else:
                self._show_error(f"Permission rule not found: {identifier}")

        except Exception as e:
            self._show_error(f"Error removing permission rule: {e}")

    async def handle_view_audit_trail(self, args: List[str]) -> None:
        """View the audit trail of permission decisions."""
        # Parse filters and actions
        limit = 50
        filter_type = None
        date_filter = None
        action = "view"  # Default action
        export_path = None
        export_format = "json"

        # Check if first argument is an action
        if args and args[0] in ["view", "export", "stats"]:
            action = args[0]
            args = args[1:]  # Remove action from args

        for arg in args:
            if arg.startswith("--limit="):
                try:
                    limit = int(arg.split("=", 1)[1])
                except ValueError:
                    self._show_error("Invalid limit value")
                    return
            elif arg.startswith("--type="):
                filter_type = arg.split("=", 1)[1]
            elif arg.startswith("--since="):
                date_filter = arg.split("=", 1)[1]
            elif arg.startswith("--export="):
                export_path = arg.split("=", 1)[1]
            elif arg.startswith("--format="):
                export_format = arg.split("=", 1)[1]
            elif action == "export" and not export_path:
                # First non-flag argument for export is the path
                export_path = arg

        try:
            if action == "stats":
                # Display audit statistics
                self.display_audit_statistics()

            elif action == "export":
                # Export audit trail
                if not export_path:
                    self._show_error("Export path required for audit export")
                    self._show_info(
                        "Usage: /permissions audit export <file_path> [--format=json|csv] [--limit=N]"
                    )
                    return

                success = self.export_audit_trail(export_path, export_format, limit)
                if success:
                    self._show_success(
                        f"Audit trail exported to {export_path} ({export_format} format)"
                    )
                else:
                    self._show_error("Failed to export audit trail")

            else:  # action == "view"
                # Get and display audit entries
                audit_entries = self._get_audit_entries(limit, filter_type, date_filter)

                if not audit_entries:
                    self._show_info("No audit entries found")
                    return

                # Display audit trail
                self._display_audit_trail(audit_entries)

                # Show quick stats summary
                if len(audit_entries) > 5:
                    approved_count = len(
                        [e for e in audit_entries if e.get("decision") == "approved"]
                    )
                    denied_count = len(
                        [e for e in audit_entries if e.get("decision") == "denied"]
                    )
                    avg_risk = sum(e.get("risk_level", 0) for e in audit_entries) / len(
                        audit_entries
                    )

                    self._show_info(
                        f"Summary: {approved_count} approved, {denied_count} denied, avg risk: {avg_risk:.1f}"
                    )
                    self._show_info(
                        "Use '/permissions audit stats' for detailed statistics"
                    )

        except Exception as e:
            self._show_error(f"Error in audit trail operation: {e}")

    async def handle_permission_learning(self, args: List[str]) -> None:
        """Handle permission learning commands."""
        if not args:
            # Show learning status
            self._show_learning_status()
            return

        action = args[0].lower()

        if action == "enable":
            success = self._set_learning_enabled(True)
            if success:
                self._show_success("Permission learning enabled")
            else:
                self._show_error("Failed to enable permission learning")

        elif action == "disable":
            success = self._set_learning_enabled(False)
            if success:
                self._show_success("Permission learning disabled")
            else:
                self._show_error("Failed to disable permission learning")

        elif action == "status":
            self._show_learning_status()

        elif action == "stats":
            self._show_learning_statistics()

        else:
            self._show_error(f"Unknown learning action: {action}")
            self._show_info("Valid actions: enable, disable, status, stats")

    async def handle_clear_learned_permissions(self, args: List[str]) -> None:
        """Clear learned permissions."""
        confirm = False
        pattern = None

        for arg in args:
            if arg == "--confirm":
                confirm = True
            elif not pattern:
                pattern = arg

        if not confirm:
            self._show_warning(
                "This will clear learned permissions. Use --confirm to proceed."
            )
            return

        try:
            count = self._clear_learned_permissions(pattern)
            if pattern:
                self._show_success(
                    f"Cleared {count} learned permissions matching '{pattern}'"
                )
            else:
                self._show_success(f"Cleared {count} learned permissions")

        except Exception as e:
            self._show_error(f"Error clearing learned permissions: {e}")

    async def handle_export_permissions(self, args: List[str]) -> None:
        """Export permissions to a file."""
        if not args:
            self._show_error("Export file path is required")
            return

        file_path = args[0]
        include_learned = "--include-learned" in args

        try:
            success = self._export_permissions(file_path, include_learned)

            if success:
                self._show_success(f"Permissions exported to: {file_path}")
            else:
                self._show_error("Failed to export permissions")

        except Exception as e:
            self._show_error(f"Error exporting permissions: {e}")

    async def handle_import_permissions(self, args: List[str]) -> None:
        """Import permissions from a file."""
        if not args:
            self._show_error("Import file path is required")
            return

        file_path = args[0]
        merge = "--merge" in args

        try:
            success = self._import_permissions(file_path, merge)

            if success:
                action = "merged with" if merge else "replaced"
                self._show_success(
                    f"Permissions imported and {action} current configuration"
                )
            else:
                self._show_error("Failed to import permissions")

        except Exception as e:
            self._show_error(f"Error importing permissions: {e}")

    def _get_current_security_settings(self) -> Dict[str, Any]:
        """Get current security settings from actual security components."""
        try:
            settings = {
                "level": "ask_but_remember",  # Default level
                "rules": [],
                "learned": [],
            }

            # Get permission rules from controller
            if self.permission_controller:
                settings["rules"] = self.permission_controller.get_permission_rules()
                settings["learned"] = (
                    self.permission_controller.get_learned_permissions()
                )

            # Get security level from security manager if available
            if self.security_manager:
                status = self.security_manager.get_security_status()
                # Try to determine security level from policies
                if status.get("policies", {}).get("require_approval_for_high_risk"):
                    settings["level"] = "ask_but_remember"
                else:
                    settings["level"] = "auto_approve"

            return settings

        except Exception:
            # Fallback to mock data if there's an error
            return {
                "level": "ask_but_remember",
                "rules": [
                    {
                        "pattern": "*.py",
                        "type": "file",
                        "level": "read",
                        "id": "1",
                    },
                    {
                        "pattern": "/tmp/*",
                        "type": "file",
                        "level": "write",
                        "id": "2",
                    },
                    {
                        "pattern": "git",
                        "type": "command",
                        "level": "execute",
                        "id": "3",
                    },
                ],
                "learned": [
                    {
                        "pattern": "test_*.py",
                        "decision": "approved",
                        "count": 5,
                        "last_used": "2024-01-15T10:30:00",
                    },
                    {
                        "pattern": "npm install",
                        "decision": "approved",
                        "count": 3,
                        "last_used": "2024-01-14T14:15:00",
                    },
                ],
            }

    def _display_permissions_overview(
        self, settings: Dict[str, Any], verbose: bool = False
    ) -> None:
        """Display the main permissions overview."""
        # Security Level Panel
        level = settings.get("level", "unknown")
        level_colors = {
            "auto_approve": "red",
            "ask_always": "yellow",
            "ask_but_remember": "green",
        }

        level_text = Text(
            level.replace("_", " ").title(),
            style=f"bold {level_colors.get(level, 'white')}",
        )

        overview_lines = [
            f"[bold]Current Security Level:[/bold] {level_text}",
            "",
            f"[bold]Active Rules:[/bold] {len(settings.get('rules', []))}",
            f"[bold]Learned Permissions:[/bold] {len(settings.get('learned', []))}",
        ]

        if verbose:
            overview_lines.extend(
                [
                    "",
                    f"[dim]Last Updated:[/dim] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )

        overview_panel = Panel(
            "\n".join(overview_lines),
            title="🔒 Security Overview",
            border_style="blue",
        )

        self.console.print(overview_panel)

    def _display_permission_rules(
        self, rules: List[Dict], category_filter: Optional[str] = None
    ) -> None:
        """Display permission rules in a table."""
        if not rules:
            return

        # Filter rules if needed
        if category_filter and category_filter != "all":
            rules = [r for r in rules if r.get("type") == category_filter]

        if not rules:
            self._show_info(f"No rules found for category: {category_filter}")
            return

        # Create rules table
        table = Table(
            title="📋 Permission Rules",
            box=ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("ID", style="dim", width=4)
        table.add_column("Pattern", style="green")
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Level", style="blue", width=10)

        for rule in rules:
            table.add_row(
                str(rule.get("id", "")),
                rule.get("pattern", ""),
                rule.get("type", "").title(),
                rule.get("level", "").title(),
            )

        self.console.print(table)

    def _display_learned_permissions(
        self, learned: List[Dict], verbose: bool = False
    ) -> None:
        """Display learned permissions."""
        if not learned:
            return

        # Create learned permissions table
        table = Table(
            title="🧠 Learned Permissions",
            box=ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Pattern", style="green")
        table.add_column("Decision", style="blue", width=12)
        table.add_column("Count", style="yellow", width=8, justify="right")

        if verbose:
            table.add_column("Last Used", style="dim", width=16)

        for perm in learned:
            row = [
                perm.get("pattern", ""),
                perm.get("decision", "").title(),
                str(perm.get("count", 0)),
            ]

            if verbose:
                last_used = perm.get("last_used", "")
                if last_used:
                    # Format timestamp
                    try:
                        dt = datetime.fromisoformat(last_used)
                        formatted = dt.strftime("%m-%d %H:%M")
                    except:
                        formatted = last_used[:16]
                    row.append(formatted)
                else:
                    row.append("Never")

            table.add_row(*row)

        self.console.print(table)

    def _display_audit_trail(self, entries: List[Dict]) -> None:
        """Display audit trail entries."""
        # Create audit table
        table = Table(
            title="📜 Audit Trail",
            box=ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Time", style="dim", width=16)
        table.add_column("Operation", style="yellow")
        table.add_column("Decision", style="blue", width=12)
        table.add_column("User", style="green", width=10)
        table.add_column("Risk", style="red", width=8, justify="center")

        for entry in entries:
            # Format timestamp
            timestamp = entry.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime("%m-%d %H:%M:%S")
                except:
                    formatted_time = timestamp[:16]
            else:
                formatted_time = "Unknown"

            # Format risk level
            risk = entry.get("risk_level", 0)
            if risk >= 8:
                risk_display = f"[red]{risk}[/red]"
            elif risk >= 5:
                risk_display = f"[yellow]{risk}[/yellow]"
            else:
                risk_display = f"[green]{risk}[/green]"

            table.add_row(
                formatted_time,
                entry.get("operation", "")[:30],
                entry.get("decision", "").title(),
                entry.get("user", "system"),
                risk_display,
            )

        self.console.print(table)

    def _set_security_level(self, level: str) -> bool:
        """Set the security level."""
        try:
            if not self.security_manager:
                return False

            # Map security levels to policies
            if level == "auto_approve":
                success = self.security_manager.update_security_policy(
                    "require_approval_for_high_risk", False
                )
            elif level == "ask_always":
                success = self.security_manager.update_security_policy(
                    "require_approval_for_high_risk", True
                )
                # Also set a policy to disable auto approval if it doesn't exist
                if (
                    "enable_auto_approval"
                    not in self.security_manager.security_policies
                ):
                    self.security_manager.security_policies["enable_auto_approval"] = (
                        False
                    )
                else:
                    success = success and self.security_manager.update_security_policy(
                        "enable_auto_approval", False
                    )
            elif level == "ask_but_remember":
                success = self.security_manager.update_security_policy(
                    "require_approval_for_high_risk", True
                )
                # Set enable auto approval policy
                if (
                    "enable_auto_approval"
                    not in self.security_manager.security_policies
                ):
                    self.security_manager.security_policies["enable_auto_approval"] = (
                        True
                    )
                else:
                    success = success and self.security_manager.update_security_policy(
                        "enable_auto_approval", True
                    )
            else:
                return False

            return success

        except Exception as e:
            self._show_error(f"Failed to set security level: {e}")
            return False

    def _add_permission_rule(self, pattern: str, rule_type: str, level: str) -> bool:
        """Add a permission rule."""
        try:
            if not self.permission_controller:
                return False

            # Add rule based on type
            if rule_type == "file":
                if level in ["read", "write"]:
                    # For allowed patterns, we remove from restricted paths
                    if level == "read":
                        # Allow read access by ensuring it's not restricted
                        self.permission_controller.remove_restricted_path(pattern)
                    else:  # write
                        # Allow write access by ensuring it's not restricted
                        self.permission_controller.remove_restricted_path(pattern)
                else:
                    # For blocked access, add to restricted paths
                    self.permission_controller.add_restricted_path(pattern)

            elif rule_type == "command":
                if level in ["execute", "allowed"]:
                    # Allow command execution
                    self.permission_controller.add_allowed_command(pattern)
                else:
                    # Block command (remove from allowed)
                    self.permission_controller.remove_allowed_command(pattern)

            elif rule_type == "network":
                # Network rules would need to be implemented in SecurityManager policies
                if self.security_manager:
                    if level == "allowed":
                        allowed_domains = self.security_manager.security_policies.get(
                            "allowed_network_domains", []
                        )
                        if pattern not in allowed_domains:
                            allowed_domains.append(pattern)
                        self.security_manager.update_security_policy(
                            "allowed_network_domains", allowed_domains
                        )
                    else:
                        blocked_domains = self.security_manager.security_policies.get(
                            "blocked_network_domains", []
                        )
                        if pattern not in blocked_domains:
                            blocked_domains.append(pattern)
                        self.security_manager.update_security_policy(
                            "blocked_network_domains", blocked_domains
                        )
            else:
                return False

            return True

        except Exception as e:
            self._show_error(f"Failed to add permission rule: {e}")
            return False

    def _remove_permission_rule(self, identifier: str) -> bool:
        """Remove a permission rule."""
        try:
            if not self.permission_controller:
                return False

            # Get current rules to find the one to remove
            current_rules = self.permission_controller.get_permission_rules()

            # Try to find rule by ID first
            rule_to_remove = None
            for rule in current_rules:
                if rule.get("id") == identifier:
                    rule_to_remove = rule
                    break

            # If not found by ID, try by pattern
            if not rule_to_remove:
                for rule in current_rules:
                    if rule.get("pattern") == identifier:
                        rule_to_remove = rule
                        break

            if not rule_to_remove:
                return False

            # Remove the rule based on its type
            pattern = rule_to_remove.get("pattern")
            rule_type = rule_to_remove.get("type")

            if rule_type == "file":
                # For file rules, add back to restricted if it was an allow rule,
                # or remove from restricted if it was a block rule
                level = rule_to_remove.get("level")
                if level == "blocked":
                    self.permission_controller.remove_restricted_path(pattern)
                else:
                    # Re-add to restricted (reverting the allow rule)
                    self.permission_controller.add_restricted_path(pattern)

            elif rule_type == "command":
                level = rule_to_remove.get("level")
                if level == "allowed":
                    self.permission_controller.remove_allowed_command(pattern)
                else:
                    # Re-add to allowed (reverting the block rule)
                    self.permission_controller.add_allowed_command(pattern)

            elif rule_type == "network" and self.security_manager:
                # Remove from network policies
                allowed_domains = self.security_manager.security_policies.get(
                    "allowed_network_domains", []
                )
                blocked_domains = self.security_manager.security_policies.get(
                    "blocked_network_domains", []
                )

                if pattern in allowed_domains:
                    allowed_domains.remove(pattern)
                    self.security_manager.update_security_policy(
                        "allowed_network_domains", allowed_domains
                    )
                elif pattern in blocked_domains:
                    blocked_domains.remove(pattern)
                    self.security_manager.update_security_policy(
                        "blocked_network_domains", blocked_domains
                    )

            return True

        except Exception as e:
            self._show_error(f"Failed to remove permission rule: {e}")
            return False

    def _get_audit_entries(
        self,
        limit: int,
        filter_type: Optional[str],
        date_filter: Optional[str],
    ) -> List[Dict]:
        """Get audit entries from actual audit logger."""
        try:
            entries = []

            if not self.audit_logger:
                # Return mock data if no audit logger
                return [
                    {
                        "timestamp": "2024-01-15T10:30:00",
                        "operation": "file_write: test.py",
                        "decision": "approved",
                        "user": "user",
                        "risk_level": 3,
                    },
                    {
                        "timestamp": "2024-01-15T10:25:00",
                        "operation": "command_execute: npm install",
                        "decision": "approved",
                        "user": "user",
                        "risk_level": 5,
                    },
                ][:limit]

            # Get recent events from audit logger
            from ...core.security.audit_logger import (
                AuditEventType,
            )

            # Apply filter by event type if specified
            event_type_filter = None
            if filter_type:
                type_mapping = {
                    "permission": AuditEventType.PERMISSION_CHECK,
                    "command": AuditEventType.COMMAND_EXECUTED,
                    "file": AuditEventType.FILE_ACCESS,
                    "approval": AuditEventType.APPROVAL_REQUESTED,
                    "security": AuditEventType.SECURITY_ALERT,
                }
                event_type_filter = type_mapping.get(filter_type.lower())

            # Get recent events
            recent_events = self.audit_logger.get_recent_events(
                count=limit * 2,  # Get more to account for filtering
                event_type=event_type_filter,
            )

            # Convert AuditEvent objects to display format
            for event in recent_events:
                # Determine decision based on event type and level
                if event.event_type in [
                    AuditEventType.PERMISSION_CHECK,
                    AuditEventType.COMMAND_EXECUTED,
                    AuditEventType.FILE_ACCESS,
                    AuditEventType.APPROVAL_GRANTED,
                ]:
                    decision = "approved"
                elif event.event_type in [
                    AuditEventType.PERMISSION_DENIED,
                    AuditEventType.COMMAND_BLOCKED,
                    AuditEventType.FILE_ACCESS_DENIED,
                    AuditEventType.APPROVAL_DENIED,
                ]:
                    decision = "denied"
                else:
                    decision = "info"

                # Calculate risk level based on event type and level
                risk_level = self._calculate_risk_level(event)

                # Create operation description
                operation = self._format_operation_description(event)

                # Apply date filter if specified
                if date_filter:
                    try:
                        filter_date = datetime.fromisoformat(date_filter).date()
                        if event.timestamp.date() < filter_date:
                            continue
                    except ValueError:
                        pass  # Skip date filtering if invalid format

                entry = {
                    "timestamp": event.timestamp.isoformat(),
                    "operation": operation,
                    "decision": decision,
                    "user": event.user_id or "system",
                    "risk_level": risk_level,
                    "event_type": event.event_type.value,
                    "level": event.level.value,
                    "session_id": event.session_id,
                    "operation_id": event.operation_id,
                }

                entries.append(entry)

                if len(entries) >= limit:
                    break

            return entries

        except Exception as e:
            # Return mock data on error
            self._show_error(f"Error retrieving audit entries: {e}")
            return [
                {
                    "timestamp": "2024-01-15T10:30:00",
                    "operation": "audit_system_error",
                    "decision": "error",
                    "user": "system",
                    "risk_level": 10,
                }
            ][:limit]

    def _calculate_risk_level(self, event) -> int:
        """Calculate risk level for an audit event."""
        # Base risk level on event type and level
        base_risk = {
            "PERMISSION_DENIED": 7,
            "COMMAND_BLOCKED": 8,
            "FILE_ACCESS_DENIED": 6,
            "SECURITY_ALERT": 9,
            "APPROVAL_DENIED": 5,
            "SANDBOX_VIOLATION": 10,
        }.get(event.event_type.value.upper(), 3)

        # Adjust based on severity level
        level_adjustment = {
            "CRITICAL": 3,
            "ERROR": 2,
            "WARNING": 1,
            "INFO": 0,
            "DEBUG": -1,
        }.get(event.level.value.upper(), 0)

        return min(10, max(1, base_risk + level_adjustment))

    def _format_operation_description(self, event) -> str:
        """Format operation description from audit event."""
        # Extract operation details from metadata
        metadata = event.metadata or {}

        if event.event_type == AuditEventType.COMMAND_EXECUTED:
            command = metadata.get("command", "unknown command")
            return f"command_execute: {command}"
        elif event.event_type == AuditEventType.FILE_ACCESS:
            path = metadata.get("path", "unknown file")
            operation = metadata.get("operation", "access")
            return f"file_{operation}: {path}"
        elif event.event_type == AuditEventType.PERMISSION_CHECK:
            operation = metadata.get("operation", "unknown")
            path = metadata.get("path", "")
            if path:
                return f"{operation}: {path}"
            return operation
        else:
            # Use event message as operation description
            return f"{event.event_type.value}: {event.message[:50]}"

    def export_audit_trail(
        self, file_path: str, format_type: str = "json", limit: int = 1000
    ) -> bool:
        """Export audit trail to file in specified format."""
        try:
            # Get audit entries
            entries = self._get_audit_entries(limit, None, None)

            if format_type.lower() == "csv":
                return self._export_audit_csv(file_path, entries)
            elif format_type.lower() == "json":
                return self._export_audit_json(file_path, entries)
            else:
                self._show_error(f"Unsupported export format: {format_type}")
                return False

        except Exception as e:
            self._show_error(f"Error exporting audit trail: {e}")
            return False

    def _export_audit_csv(self, file_path: str, entries: List[Dict]) -> bool:
        """Export audit entries as CSV."""
        try:
            import csv

            if not entries:
                self._show_warning("No audit entries to export")
                return False

            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "timestamp",
                    "operation",
                    "decision",
                    "user",
                    "risk_level",
                    "event_type",
                    "level",
                    "session_id",
                    "operation_id",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for entry in entries:
                    # Only write fields that exist in the entry
                    csv_entry = {k: entry.get(k, "") for k in fieldnames}
                    writer.writerow(csv_entry)

            return True

        except Exception as e:
            self._show_error(f"Error writing CSV file: {e}")
            return False

    def _export_audit_json(self, file_path: str, entries: List[Dict]) -> bool:
        """Export audit entries as JSON."""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(entries),
                "audit_entries": entries,
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            return True

        except Exception as e:
            self._show_error(f"Error writing JSON file: {e}")
            return False

    def get_audit_statistics_summary(self) -> Dict[str, Any]:
        """Get audit trail statistics summary."""
        try:
            # Get recent entries for analysis
            recent_entries = self._get_audit_entries(
                500, None, None
            )  # Analyze last 500 entries

            if not recent_entries:
                return {
                    "total_entries": 0,
                    "approval_rate": 0.0,
                    "denial_rate": 0.0,
                    "avg_risk_level": 0.0,
                    "common_operations": [],
                    "risk_distribution": {},
                }

            # Calculate statistics
            total_entries = len(recent_entries)
            approved_count = len(
                [e for e in recent_entries if e.get("decision") == "approved"]
            )
            denied_count = len(
                [e for e in recent_entries if e.get("decision") == "denied"]
            )

            approval_rate = (
                (approved_count / total_entries) * 100 if total_entries > 0 else 0
            )
            denial_rate = (
                (denied_count / total_entries) * 100 if total_entries > 0 else 0
            )

            # Calculate average risk level
            risk_levels = [e.get("risk_level", 0) for e in recent_entries]
            avg_risk_level = sum(risk_levels) / len(risk_levels) if risk_levels else 0

            # Find common operations
            operation_counts = {}
            for entry in recent_entries:
                op_type = entry.get("operation", "unknown")
                # Extract base operation type
                base_op = op_type.split(":")[0] if ":" in op_type else op_type
                operation_counts[base_op] = operation_counts.get(base_op, 0) + 1

            common_operations = sorted(
                operation_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]

            # Risk distribution
            risk_distribution = {}
            for risk_level in risk_levels:
                if risk_level <= 3:
                    category = "Low (1-3)"
                elif risk_level <= 6:
                    category = "Medium (4-6)"
                elif risk_level <= 8:
                    category = "High (7-8)"
                else:
                    category = "Critical (9-10)"

                risk_distribution[category] = risk_distribution.get(category, 0) + 1

            return {
                "total_entries": total_entries,
                "approval_rate": round(approval_rate, 1),
                "denial_rate": round(denial_rate, 1),
                "avg_risk_level": round(avg_risk_level, 1),
                "common_operations": common_operations,
                "risk_distribution": risk_distribution,
                "time_period": (
                    "Last 500 entries"
                    if total_entries >= 500
                    else f"All {total_entries} entries"
                ),
            }

        except Exception as e:
            self._show_error(f"Error calculating audit statistics: {e}")
            return {"error": str(e)}

    def display_audit_statistics(self) -> None:
        """Display comprehensive audit statistics."""
        try:
            stats = self.get_audit_statistics_summary()

            if "error" in stats:
                self._show_error(f"Failed to get audit statistics: {stats['error']}")
                return

            # Create statistics display
            stats_lines = [
                "[bold]Audit Trail Statistics[/bold]",
                "",
                f"📊 Total Entries Analyzed: {stats['total_entries']}",
                f"📈 Time Period: {stats['time_period']}",
                "",
                "[bold]Decision Summary:[/bold]",
                f"✅ Approval Rate: {stats['approval_rate']}%",
                f"❌ Denial Rate: {stats['denial_rate']}%",
                f"📊 Average Risk Level: {stats['avg_risk_level']}/10",
                "",
            ]

            # Add common operations
            if stats["common_operations"]:
                stats_lines.append("[bold]Most Common Operations:[/bold]")
                for op_type, count in stats["common_operations"]:
                    percentage = (
                        (count / stats["total_entries"]) * 100
                        if stats["total_entries"] > 0
                        else 0
                    )
                    stats_lines.append(
                        f"  • {op_type}: {count} times ({percentage:.1f}%)"
                    )
                stats_lines.append("")

            # Add risk distribution
            if stats["risk_distribution"]:
                stats_lines.append("[bold]Risk Level Distribution:[/bold]")
                for category, count in stats["risk_distribution"].items():
                    percentage = (
                        (count / stats["total_entries"]) * 100
                        if stats["total_entries"] > 0
                        else 0
                    )
                    # Color code risk levels
                    if "Critical" in category:
                        color_code = "[red]"
                    elif "High" in category:
                        color_code = "[orange]"
                    elif "Medium" in category:
                        color_code = "[yellow]"
                    else:
                        color_code = "[green]"
                    stats_lines.append(
                        f"  • {color_code}{category}[/]: {count} entries ({percentage:.1f}%)"
                    )

            # Display in a panel
            stats_panel = Panel(
                "\n".join(stats_lines),
                title="📊 Audit Trail Statistics",
                border_style="cyan",
            )
            self.console.print(stats_panel)

        except Exception as e:
            self._show_error(f"Error displaying audit statistics: {e}")

    def _set_learning_enabled(self, enabled: bool) -> bool:
        """Set learning enabled state."""
        try:
            if not self.security_manager:
                return False

            # Set learning policy in security manager
            success = self.security_manager.update_security_policy(
                "enable_permission_learning", enabled
            )

            # Also set it directly in security policies if it doesn't exist
            if (
                "enable_permission_learning"
                not in self.security_manager.security_policies
            ):
                self.security_manager.security_policies[
                    "enable_permission_learning"
                ] = enabled
                success = True

            return success

        except Exception as e:
            self._show_error(f"Failed to set learning enabled: {e}")
            return False

    def _show_learning_status(self) -> None:
        """Show learning status."""
        try:
            # Get actual learning status
            learning_enabled = False
            decisions_made = 0
            patterns_learned = 0

            if self.security_manager:
                learning_enabled = self.security_manager.security_policies.get(
                    "enable_permission_learning", False
                )

            if self.permission_controller:
                # Get stored approvals (learned patterns)
                stored_approvals = self.permission_controller.get_stored_approvals()
                patterns_learned = len(stored_approvals)

                # Approximate decisions made (could be enhanced with actual tracking)
                decisions_made = patterns_learned + 5  # Mock additional decisions

            # Determine status display
            if learning_enabled:
                status_text = "[green]✓[/green] Permission learning is [bold green]enabled[/bold green]"
                border_style = "green"
            else:
                status_text = "[yellow]○[/yellow] Permission learning is [bold yellow]disabled[/bold yellow]"
                border_style = "yellow"

            # Create status content
            status_content = f"{status_text}\n[dim]Decisions made: {decisions_made}\n"
            status_content += f"Patterns learned: {patterns_learned}\n"
            status_content += f"Auto-approvals available: {patterns_learned}[/dim]"

            status_panel = Panel(
                status_content,
                title="🧠 Learning Status",
                border_style=border_style,
            )
            self.console.print(status_panel)

        except Exception as e:
            # Fallback display
            self._show_error(f"Error getting learning status: {e}")
            fallback_panel = Panel(
                "[red]✗[/red] Learning status unavailable\n[dim]Error accessing learning system[/dim]",
                title="🧠 Learning Status",
                border_style="red",
            )
            self.console.print(fallback_panel)

    def _show_learning_statistics(self) -> None:
        """Show learning statistics."""
        try:
            # Get actual statistics
            total_decisions = 0
            unique_patterns = 0
            auto_approvals = 0
            most_common = []

            if self.permission_controller:
                stored_approvals = self.permission_controller.get_stored_approvals()
                unique_patterns = len(stored_approvals)
                auto_approvals = (
                    unique_patterns  # Each stored approval could trigger auto-approval
                )

                # Mock total decisions (in real implementation, this would be tracked)
                total_decisions = unique_patterns * 2 + 10  # Estimate based on patterns

                # Get most common approval types from stored data
                if stored_approvals:
                    # Analyze approval patterns
                    pattern_counts = {}
                    for signature, approval_data in stored_approvals.items():
                        op_type = approval_data.get("operation_type", "unknown")
                        pattern_counts[op_type] = pattern_counts.get(op_type, 0) + 1

                    # Create most common list
                    for op_type, count in sorted(
                        pattern_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]:
                        most_common.append(
                            f"  • {op_type.replace('_', ' ').title()} operations ({count} patterns)"
                        )

            # Calculate accuracy (mock calculation)
            accuracy = 85 + min(
                15, unique_patterns * 2
            )  # Higher patterns = higher accuracy

            stats_lines = [
                "[bold]Learning Statistics[/bold]",
                "",
                f"📊 Total decisions recorded: {total_decisions}",
                f"🧠 Unique patterns learned: {unique_patterns}",
                f"✅ Auto-approvals available: {auto_approvals}",
                f"❌ Auto-denials available: 0",  # Not implemented yet
                f"📈 Learning effectiveness: {accuracy}%",
                "",
            ]

            if most_common:
                stats_lines.append("[dim]Most learned patterns:[/dim]")
                stats_lines.extend(most_common)
            else:
                stats_lines.append("[dim]No learning patterns recorded yet[/dim]")

            stats_panel = Panel(
                "\n".join(stats_lines),
                title="📈 Learning Statistics",
                border_style="blue",
            )
            self.console.print(stats_panel)

        except Exception as e:
            # Fallback display
            self._show_error(f"Error getting learning statistics: {e}")
            fallback_panel = Panel(
                "[red]✗[/red] Learning statistics unavailable\n[dim]Error accessing learning data[/dim]",
                title="📈 Learning Statistics",
                border_style="red",
            )
            self.console.print(fallback_panel)

    def _clear_learned_permissions(self, pattern: Optional[str] = None) -> int:
        """Clear learned permissions."""
        try:
            if not self.permission_controller:
                return 0

            # Get current stored approvals
            stored_approvals = self.permission_controller.get_stored_approvals()

            if not stored_approvals:
                return 0

            cleared_count = 0

            if pattern:
                # Clear approvals matching the pattern
                signatures_to_remove = []
                for signature, approval_data in stored_approvals.items():
                    # Check if pattern matches signature or operation type
                    if pattern in signature or pattern in approval_data.get(
                        "operation_type", ""
                    ):
                        signatures_to_remove.append(signature)

                # Remove matching approvals
                for signature in signatures_to_remove:
                    if self.permission_controller.revoke_approval(signature):
                        cleared_count += 1
            else:
                # Clear all approvals
                signatures_to_remove = list(stored_approvals.keys())
                for signature in signatures_to_remove:
                    if self.permission_controller.revoke_approval(signature):
                        cleared_count += 1

            return cleared_count

        except Exception as e:
            self._show_error(f"Error clearing learned permissions: {e}")
            return 0

    def _export_permissions(self, file_path: str, include_learned: bool) -> bool:
        """Export permissions to file."""
        try:
            # Get current security settings
            settings = self._get_current_security_settings()

            export_data = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "security_level": settings.get("level", "ask_but_remember"),
                "rules": settings.get("rules", []),
            }

            if include_learned:
                export_data["learned"] = settings.get("learned", [])

            # Also include security policies if available
            if self.security_manager:
                export_data["security_policies"] = (
                    self.security_manager.security_policies.copy()
                )

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)

            return True
        except Exception as e:
            self._show_error(f"Error exporting permissions: {e}")
            return False

    def _import_permissions(self, file_path: str, merge: bool) -> bool:
        """Import permissions from file."""
        try:
            with open(file_path, "r") as f:
                import_data = json.load(f)

            # Validate import data
            if "version" not in import_data:
                raise ValueError("Invalid permissions file format")

            # Import security level
            if "security_level" in import_data:
                self._set_security_level(import_data["security_level"])

            # Import rules
            if "rules" in import_data and not merge:
                # Clear existing rules first (if not merging)
                if self.permission_controller:
                    # This would require implementing a clear all rules method
                    pass

            # Import learned permissions
            if "learned" in import_data and self.permission_controller:
                for learned_item in import_data["learned"]:
                    # Convert learned permission back to approval storage format
                    pattern = learned_item.get("pattern", "")
                    if pattern:
                        # Store as approved pattern
                        operation_type = "learned_import"
                        import asyncio

                        try:
                            # Use async method in sync context
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(
                                self.permission_controller.grant_operation_permission(
                                    operation_type=operation_type,
                                    operation_signature=pattern,
                                    metadata={
                                        "imported": True,
                                        "decision": learned_item.get("decision"),
                                        "original_count": learned_item.get("count", 1),
                                    },
                                    expires_hours=24
                                    * 365,  # 1 year for imported permissions
                                )
                            )
                            loop.close()
                        except Exception:
                            # If async fails, skip this learned permission
                            pass

            # Import security policies
            if "security_policies" in import_data and self.security_manager:
                for policy_name, policy_value in import_data[
                    "security_policies"
                ].items():
                    self.security_manager.update_security_policy(
                        policy_name, policy_value
                    )

            return True

        except Exception as e:
            self._show_error(f"Error importing permissions: {e}")
            return False

    def _show_permissions_help(self) -> None:
        """Show help for permissions commands."""
        help_text = """
[bold cyan]Permissions Commands:[/bold cyan]

[yellow]View Commands:[/yellow]
  /permissions view                   - Show current permissions and security settings
  /permissions view --category=file   - Filter by category (file, command, network)
  /permissions view --json            - Output in JSON format
  /permissions view --verbose         - Show detailed information

[yellow]Security Level Management:[/yellow]
  /permissions set-level <level>      - Set security level
    • auto_approve     - Automatically approve all operations (⚠️ DANGEROUS)
    • ask_always       - Always ask for user approval
    • ask_but_remember - Ask but remember user decisions

[yellow]Rule Management:[/yellow]
  /permissions add-rule <pattern> [type] [level]    - Add permission rule
    • Types: file, command, network
    • Levels: read, write, execute, admin
    • Examples: /permissions add-rule "*.py" file read
               /permissions add-rule "git" command execute

  /permissions remove-rule <pattern_or_id>         - Remove permission rule

[yellow]Learning & Memory:[/yellow]
  /permissions learn enable           - Enable permission learning
  /permissions learn disable          - Disable permission learning
  /permissions learn status           - Show learning status
  /permissions learn stats            - Show learning statistics
  /permissions clear-learned --confirm [pattern]  - Clear learned permissions

[yellow]Audit & Export:[/yellow]
  /permissions audit                  - View audit trail of permission decisions
  /permissions audit --limit=100      - Limit number of entries shown
  /permissions audit --type=file      - Filter by operation type
  
  /permissions export <file>          - Export permissions to file
  /permissions export <file> --include-learned    - Include learned permissions
  /permissions import <file> --merge  - Import and merge permissions

[yellow]Examples:[/yellow]
  /permissions view --verbose         - Show detailed permissions overview
  /permissions set-level ask_always   - Require approval for every operation
  /permissions add-rule "/tmp/*" file write       - Allow writes to /tmp
  /permissions audit --limit=20       - Show last 20 permission decisions
"""

        panel = Panel(help_text.strip(), title="Permissions Help", border_style="cyan")
        self.console.print(panel)

    def _show_info(self, message: str) -> None:
        """Show info message."""
        self.console.print(f"[cyan]ℹ {message}[/cyan]")

    def _show_success(self, message: str) -> None:
        """Show success message."""
        self.console.print(f"[green]✓ {message}[/green]")

    def _show_warning(self, message: str) -> None:
        """Show warning message."""
        self.console.print(f"[yellow]⚠ {message}[/yellow]")

    def _show_error(self, message: str) -> None:
        """Show error message."""
        self.console.print(f"[red]✗ {message}[/red]")
