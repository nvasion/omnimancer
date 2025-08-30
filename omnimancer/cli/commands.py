"""
Command models and types for the Omnimancer CLI.

This module defines the command structure, types, and parsing logic
for handling user input in the command-line interface.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .dynamic_commands import CommandRegistry, DynamicCommand

# Global command registry instance
_command_registry: Optional[CommandRegistry] = None


def get_command_registry() -> CommandRegistry:
    """Get or create the global command registry."""
    global _command_registry
    if _command_registry is None:
        _command_registry = CommandRegistry()
        # Set built-in commands to prevent overriding
        _command_registry.set_builtin_commands([cmd.value for cmd in SlashCommand])
        # Load dynamic commands from default directory
        _command_registry.load_commands_from_directory()
    return _command_registry


class CommandType(Enum):
    """Types of commands that can be processed."""

    CHAT_MESSAGE = "chat"
    SLASH_COMMAND = "slash"
    SYSTEM_COMMAND = "system"
    DYNAMIC_COMMAND = "dynamic"


class SlashCommand(Enum):
    """Available slash commands."""

    HELP = "/help"
    MODELS = "/models"
    MODEL = "/model"  # Alias for /models
    SWITCH = "/switch"
    CLEAR = "/clear"
    SAVE = "/save"
    LOAD = "/load"
    QUIT = "/quit"
    STATUS = "/status"
    LIST = "/list"
    PROVIDERS = "/providers"
    TOOLS = "/tools"
    MCP = "/mcp"
    SETUP = "/setup"
    VALIDATE = "/validate"
    HEALTH = "/health"
    REPAIR = "/repair"
    DIAGNOSE = "/diagnose"
    HISTORY = "/history"
    ADD_MODEL = "/add-model"
    REMOVE_MODEL = "/remove-model"
    LIST_CUSTOM_MODELS = "/list-custom-models"
    AGENT = "/agent"
    AGENTS = "/agents"
    AGENTS_CUSTOM = "/agents-custom"
    AGENTSTATUS = "/agentstatus"
    APPROVALS = "/approvals"
    PERMISSIONS = "/permissions"

    @classmethod
    def from_string(cls, command_str: str) -> Optional["SlashCommand"]:
        """Get SlashCommand from string, case-insensitive."""
        command_str = command_str.lower().strip()
        for cmd in cls:
            if cmd.value == command_str:
                return cmd
        return None

    @classmethod
    def get_all_commands(cls) -> List[str]:
        """Get list of all available slash commands."""
        return [cmd.value for cmd in cls]


@dataclass
class Command:
    """Represents a parsed command from user input."""

    type: CommandType
    content: str
    parameters: Dict[str, Any]
    raw_input: str

    @classmethod
    def create_chat_message(cls, content: str) -> "Command":
        """Create a chat message command."""
        return cls(
            type=CommandType.CHAT_MESSAGE,
            content=content.strip(),
            parameters={},
            raw_input=content,
        )

    @classmethod
    def create_slash_command(
        cls, command: SlashCommand, args: List[str], raw_input: str
    ) -> "Command":
        """Create a slash command."""
        return cls(
            type=CommandType.SLASH_COMMAND,
            content=command.value,
            parameters={"args": args, "command": command},
            raw_input=raw_input,
        )

    @classmethod
    def create_system_command(cls, content: str) -> "Command":
        """Create a system command."""
        return cls(
            type=CommandType.SYSTEM_COMMAND,
            content=content,
            parameters={},
            raw_input=content,
        )

    @classmethod
    def create_dynamic_command(
        cls, command: DynamicCommand, args: List[str], raw_input: str
    ) -> "Command":
        """Create a dynamic command."""
        return cls(
            type=CommandType.DYNAMIC_COMMAND,
            content=command.slash_name,
            parameters={"args": args, "command": command},
            raw_input=raw_input,
        )

    @property
    def is_chat_message(self) -> bool:
        """Check if this is a chat message."""
        return self.type == CommandType.CHAT_MESSAGE

    @property
    def is_slash_command(self) -> bool:
        """Check if this is a slash command."""
        return self.type == CommandType.SLASH_COMMAND

    @property
    def is_system_command(self) -> bool:
        """Check if this is a system command."""
        return self.type == CommandType.SYSTEM_COMMAND

    @property
    def is_dynamic_command(self) -> bool:
        """Check if this is a dynamic command."""
        return self.type == CommandType.DYNAMIC_COMMAND

    @property
    def slash_command(self) -> Optional[SlashCommand]:
        """Get the slash command if this is a slash command."""
        if self.is_slash_command:
            return self.parameters.get("command")
        return None

    @property
    def dynamic_command(self) -> Optional[DynamicCommand]:
        """Get the dynamic command if this is a dynamic command."""
        if self.is_dynamic_command:
            return self.parameters.get("command")
        return None

    @property
    def args(self) -> List[str]:
        """Get command arguments."""
        return self.parameters.get("args", [])


def parse_command(user_input: str) -> Command:
    """
    Parse user input into a Command object with validation and sanitization.

    Args:
        user_input: Raw input from the user

    Returns:
        Parsed Command object

    Raises:
        ValueError: If input contains invalid characters or is malformed
    """
    if user_input is None:
        return Command.create_chat_message("")

    # Sanitize input - remove control characters except newlines and tabs
    sanitized_input = sanitize_input(user_input)

    if not sanitized_input.strip():
        return Command.create_chat_message("")

    # Check for slash commands
    if sanitized_input.startswith("/"):
        return _parse_slash_command(sanitized_input)

    # Check for system commands (exit, quit, etc.)
    if sanitized_input.lower().strip() in ["exit", "quit", "q"]:
        return Command.create_system_command("quit")

    # Validate chat message length
    if len(sanitized_input) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"Message too long. Maximum length is {MAX_MESSAGE_LENGTH} characters."
        )

    # Default to chat message
    return Command.create_chat_message(sanitized_input)


def _parse_slash_command(user_input: str) -> Command:
    """
    Parse a slash command with argument validation.

    Args:
        user_input: Input starting with '/'

    Returns:
        Parsed Command object
    """
    parts = user_input.split()
    command_str = parts[0]
    args = parts[1:] if len(parts) > 1 else []

    # Handle /exit as an alias for /quit
    if command_str.lower() == "/exit":
        command_str = "/quit"

    # Check for built-in slash commands first
    slash_cmd = SlashCommand.from_string(command_str)
    if slash_cmd:
        # Validate command-specific arguments
        validated_args = _validate_command_args(slash_cmd, args)
        return Command.create_slash_command(slash_cmd, validated_args, user_input)

    # Check for dynamic commands
    registry = get_command_registry()
    dynamic_cmd = registry.get_command(command_str)
    if dynamic_cmd:
        # No validation for dynamic commands yet - let handler manage it
        return Command.create_dynamic_command(dynamic_cmd, args, user_input)

    # Unknown slash command, treat as chat message
    return Command.create_chat_message(user_input)


def _validate_command_args(command: SlashCommand, args: List[str]) -> List[str]:
    """
    Validate arguments for specific slash commands.

    Args:
        command: The slash command
        args: List of arguments

    Returns:
        Validated arguments list

    Raises:
        ValueError: If arguments are invalid
    """
    if command == SlashCommand.SWITCH:
        if len(args) < 1:
            raise ValueError(
                "Switch command requires at least one argument: provider name"
            )
        if len(args) > 2:
            raise ValueError(
                "Switch command accepts at most two arguments: provider and model"
            )

        # Validate provider name (alphanumeric and underscores only)
        provider = args[0]
        if not provider.replace("_", "").isalnum():
            raise ValueError(
                "Provider name must contain only letters, numbers, and underscores"
            )

        # Validate model name if provided
        if len(args) == 2:
            model = args[1]
            # Allow model IDs with common formats: org/model:variant, model-name, model.version, etc.
            # Only restrict dangerous characters that could cause issues
            invalid_model_chars = [
                "\0",
                "\n",
                "\r",
                "\t",
                "\\",
                '"',
                "'",
                "`",
                "|",
                "&",
                ";",
                "(",
                ")",
                "{",
                "}",
                "[",
                "]",
                "<",
                ">",
                "$",
            ]
            if any(char in model for char in invalid_model_chars):
                raise ValueError("Model name contains invalid characters")

    elif command == SlashCommand.SAVE:
        if len(args) > 1:
            raise ValueError("Save command accepts at most one argument: filename")

        if args and args[0]:
            filename = args[0]
            # Basic filename validation
            if any(
                char in filename
                for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
            ):
                raise ValueError("Filename contains invalid characters")

    elif command == SlashCommand.LOAD:
        if len(args) != 1:
            raise ValueError("Load command requires exactly one argument: filename")

        filename = args[0]
        if any(
            char in filename for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        ):
            raise ValueError("Filename contains invalid characters")

    elif command == SlashCommand.MCP:
        if len(args) > 2:
            raise ValueError(
                "MCP command accepts at most two arguments: action and server name"
            )

        if args:
            action = args[0].lower()
            valid_actions = [
                "status",
                "reload",
                "connect",
                "disconnect",
                "health",
            ]
            if action not in valid_actions:
                raise ValueError(
                    f"Invalid MCP action. Valid actions: {', '.join(valid_actions)}"
                )

    elif command in [
        SlashCommand.HELP,
        SlashCommand.CLEAR,
        SlashCommand.QUIT,
        SlashCommand.STATUS,
    ]:
        if args:
            raise ValueError(f"{command.value} command does not accept arguments")

    elif command in [SlashCommand.MODELS, SlashCommand.MODEL]:
        # /models [filter_type] [filter_value]
        if len(args) > 2:
            raise ValueError(
                "Models command accepts at most two arguments: filter type and filter value"
            )

        if args:
            filter_type = args[0].lower()
            valid_filters = [
                "provider",
                "capability",
                "price",
                "performance",
                "free",
                "latest",
            ]
            if filter_type not in valid_filters:
                raise ValueError(
                    f"Invalid filter type. Valid filters: {', '.join(valid_filters)}"
                )

            # Validate second argument based on filter type
            if len(args) == 2:
                if filter_type == "capability":
                    valid_capabilities = ["tools", "multimodal"]
                    if args[1].lower() not in valid_capabilities:
                        raise ValueError(
                            f"Invalid capability. Valid capabilities: {', '.join(valid_capabilities)}"
                        )
                elif filter_type == "price":
                    try:
                        float(args[1])
                    except ValueError:
                        raise ValueError("Price filter value must be a number")
                elif filter_type == "performance":
                    try:
                        float(args[1])
                    except ValueError:
                        raise ValueError("Performance filter value must be a number")
                elif filter_type in ["free", "latest"]:
                    raise ValueError(f"{filter_type} filter does not accept a value")
            elif filter_type in [
                "provider",
                "capability",
                "price",
                "performance",
            ]:
                raise ValueError(f"{filter_type} filter requires a value")

    elif command == SlashCommand.VALIDATE:
        # /validate [provider] [--fix]
        if len(args) > 2:
            raise ValueError(
                "Validate command accepts at most two arguments: provider and --fix flag"
            )

        if args:
            # First arg should be provider name or --fix
            if (
                args[0] not in ["--fix", "--auto-fix"]
                and not args[0].replace("_", "").replace("-", "").isalnum()
            ):
                raise ValueError("Invalid provider name format")

    elif command == SlashCommand.HEALTH:
        # /health [provider] [--monitor] [--interval]
        if len(args) > 3:
            raise ValueError("Health command accepts at most three arguments")

        if args:
            # Validate provider name if not a flag
            if (
                not args[0].startswith("--")
                and not args[0].replace("_", "").replace("-", "").isalnum()
            ):
                raise ValueError("Invalid provider name format")

    elif command == SlashCommand.REPAIR:
        # /repair [provider] [--auto] [--backup]
        if len(args) > 3:
            raise ValueError("Repair command accepts at most three arguments")

        if args:
            # Validate provider name if not a flag
            if (
                not args[0].startswith("--")
                and not args[0].replace("_", "").replace("-", "").isalnum()
            ):
                raise ValueError("Invalid provider name format")

    elif command == SlashCommand.DIAGNOSE:
        # /diagnose [provider] [--detailed]
        if len(args) > 2:
            raise ValueError(
                "Diagnose command accepts at most two arguments: provider and --detailed flag"
            )

        if args:
            # Validate provider name if not a flag
            if (
                not args[0].startswith("--")
                and not args[0].replace("_", "").replace("-", "").isalnum()
            ):
                raise ValueError("Invalid provider name format")

    elif command == SlashCommand.HISTORY:
        # /history [recent|search|clear|export] [args...]
        if len(args) > 3:
            raise ValueError("History command accepts at most three arguments")

        if args:
            action = args[0].lower()
            valid_actions = ["recent", "search", "clear", "export", "stats"]
            if action not in valid_actions:
                raise ValueError(
                    f"Invalid history action. Valid actions: {', '.join(valid_actions)}"
                )

    elif command == SlashCommand.ADD_MODEL:
        # /add-model <name> <provider> [description] [other_params...]
        if len(args) < 2:
            raise ValueError(
                "Add-model command requires at least two arguments: model name and provider"
            )
        if len(args) > 10:
            raise ValueError("Add-model command accepts at most 10 arguments total")

        # Validate model name and provider
        model_name = args[0]
        provider = args[1]

        # Allow model IDs with common formats: org/model:variant, model-name, model.version, etc.
        # Only restrict dangerous characters that could cause issues
        invalid_model_chars = [
            "\0",
            "\n",
            "\r",
            "\t",
            "\\",
            '"',
            "'",
            "`",
            "|",
            "&",
            ";",
            "(",
            ")",
            "{",
            "}",
            "[",
            "]",
            "<",
            ">",
            "$",
        ]
        if any(char in model_name for char in invalid_model_chars):
            raise ValueError("Model name contains invalid characters")

        # Validate provider name (more restrictive since it's used internally)
        if not provider.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Provider name contains invalid characters")

    elif command == SlashCommand.REMOVE_MODEL:
        # /remove-model <name> <provider>
        if len(args) != 2:
            raise ValueError(
                "Remove-model command requires exactly two arguments: model name and provider"
            )

        # Validate model name and provider
        model_name = args[0]
        provider = args[1]

        # Allow model IDs with common formats: org/model:variant, model-name, model.version, etc.
        # Only restrict dangerous characters that could cause issues
        invalid_model_chars = [
            "\0",
            "\n",
            "\r",
            "\t",
            "\\",
            '"',
            "'",
            "`",
            "|",
            "&",
            ";",
            "(",
            ")",
            "{",
            "}",
            "[",
            "]",
            "<",
            ">",
            "$",
        ]
        if any(char in model_name for char in invalid_model_chars):
            raise ValueError("Model name contains invalid characters")

        # Validate provider name (more restrictive since it's used internally)
        if not provider.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Provider name contains invalid characters")

    elif command == SlashCommand.LIST_CUSTOM_MODELS:
        # /list-custom-models
        if args:
            raise ValueError("List-custom-models command does not accept arguments")

    elif command == SlashCommand.AGENTS:
        # /agents [action] [agent_name] [options]
        if len(args) > 3:
            raise ValueError(
                "Agents command accepts at most three arguments: action, agent_name, and options"
            )

        if args:
            action = args[0].lower()
            valid_actions = [
                "list",
                "enable",
                "disable",
                "status",
                "config",
                "switch",
                "current",
                "info",
            ]
            if action not in valid_actions:
                raise ValueError(
                    f"Invalid agents action. Valid actions: {', '.join(valid_actions)}"
                )

            # Some actions require an agent name
            if (
                action in ["enable", "disable", "config", "switch", "info"]
                and len(args) < 2
            ):
                raise ValueError(f"Action '{action}' requires an agent name")

    elif command == SlashCommand.APPROVALS:
        # /approvals [action] [signature]
        if len(args) > 2:
            raise ValueError(
                "Approvals command accepts at most two arguments: action and signature"
            )

        if args:
            action = args[0].lower()
            valid_actions = ["list", "revoke", "clear", "stats", "cleanup"]
            if action not in valid_actions:
                raise ValueError(
                    f"Invalid approvals action. Valid actions: {', '.join(valid_actions)}"
                )

            # Revoke action requires a signature
            if action == "revoke" and len(args) < 2:
                raise ValueError("Revoke action requires a signature to revoke")

    elif command == SlashCommand.PERMISSIONS:
        # /permissions [action] [args...]
        if len(args) > 4:
            raise ValueError("Permissions command accepts at most four arguments")

        if args:
            action = args[0].lower()
            valid_actions = [
                "view",
                "set-level",
                "add-rule",
                "remove-rule",
                "audit",
                "learn",
                "clear-learned",
                "export",
                "import",
            ]
            if action not in valid_actions:
                raise ValueError(
                    f"Invalid permissions action. Valid actions: {', '.join(valid_actions)}"
                )

            # Actions that require additional arguments
            if action == "set-level" and len(args) < 2:
                raise ValueError(
                    "Set-level action requires a security level (auto_approve, ask_always, ask_but_remember)"
                )
            elif action in ["add-rule", "remove-rule"] and len(args) < 2:
                raise ValueError(
                    f"{action.capitalize()} action requires at least one additional argument"
                )
            elif action in ["export", "import"] and len(args) < 2:
                raise ValueError(f"{action.capitalize()} action requires a file path")

            # Validate security levels for set-level
            if action == "set-level" and len(args) >= 2:
                level = args[1].lower()
                valid_levels = [
                    "auto_approve",
                    "ask_always",
                    "ask_but_remember",
                ]
                if level not in valid_levels:
                    raise ValueError(
                        f"Invalid security level. Valid levels: {', '.join(valid_levels)}"
                    )

    # Commands that don't accept arguments
    elif command in [
        SlashCommand.CLEAR,
        SlashCommand.QUIT,
        SlashCommand.STATUS,
        SlashCommand.PROVIDERS,
        SlashCommand.TOOLS,
        SlashCommand.SETUP,
    ]:
        if args:
            raise ValueError(f"Command {command.value} does not accept arguments")

    return args


def sanitize_input(user_input: str) -> str:
    """
    Sanitize user input by removing dangerous characters.

    Args:
        user_input: Raw user input

    Returns:
        Sanitized input string
    """
    if not user_input:
        return ""

    # Remove null bytes and other control characters except newlines and tabs
    sanitized = "".join(
        char for char in user_input if ord(char) >= 32 or char in ["\n", "\t"]
    )

    # Limit length to prevent memory issues
    if len(sanitized) > MAX_INPUT_LENGTH:
        sanitized = sanitized[:MAX_INPUT_LENGTH]

    return sanitized


def validate_command_syntax(command: Command) -> bool:
    """
    Validate that a command has proper syntax and structure.

    Args:
        command: Command to validate

    Returns:
        True if command is valid

    Raises:
        ValueError: If command is invalid
    """
    if not isinstance(command, Command):
        raise ValueError("Invalid command object")

    if not isinstance(command.type, CommandType):
        raise ValueError("Invalid command type")

    if command.content is None:
        raise ValueError("Command content cannot be None")

    if not isinstance(command.parameters, dict):
        raise ValueError("Command parameters must be a dictionary")

    # Validate slash command structure
    if command.is_slash_command:
        if not command.content.startswith("/"):
            raise ValueError("Slash command must start with '/'")

        if "command" not in command.parameters:
            raise ValueError("Slash command missing command parameter")

        if not isinstance(command.parameters["command"], SlashCommand):
            raise ValueError("Invalid slash command type")

    return True


# Constants for input validation
MAX_MESSAGE_LENGTH = 10000  # Maximum length for chat messages
MAX_INPUT_LENGTH = 50000  # Maximum length for any input
