"""
Dynamic command registration and management system for Omnimancer.

This module provides a system for dynamically loading and registering
custom slash commands from external files, similar to Claude Code's
command system.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DynamicCommand:
    """Represents a dynamically loaded command."""

    name: str  # Command name without slash (e.g., "custom-test")
    description: str
    handler: Optional[Callable] = None  # Python handler if applicable
    script_path: Optional[Path] = None  # Path to script file
    metadata: Dict[str, Any] = field(default_factory=dict)
    arguments: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Argument definitions

    @property
    def slash_name(self) -> str:
        """Get the full slash command name."""
        return f"/{self.name}"

    def get_argument_names(self) -> List[str]:
        """Get list of argument names for autocomplete."""
        return [arg.get("name", "") for arg in self.arguments if arg.get("name")]

    def get_argument_at_position(self, position: int) -> Optional[Dict[str, Any]]:
        """Get argument definition at a specific position."""
        if 0 <= position < len(self.arguments):
            return self.arguments[position]
        return None


class CommandRegistry:
    """Registry for managing dynamic commands."""

    def __init__(self, commands_dir: Optional[Path] = None):
        """
        Initialize the command registry.

        Args:
            commands_dir: Directory to load commands from
        """
        self.commands: Dict[str, DynamicCommand] = {}
        self.commands_dir = commands_dir or self._get_default_commands_dir()
        self._builtin_commands: List[str] = []  # Track built-in commands

    def _get_default_commands_dir(self) -> Path:
        """Get the default commands directory."""
        # Use ~/.omnimancer/commands as requested by user
        return Path.home() / ".omnimancer" / "commands"

    def register_command(self, command: DynamicCommand) -> bool:
        """
        Register a dynamic command.

        Args:
            command: The command to register

        Returns:
            True if registered successfully, False otherwise
        """
        try:
            # Check for conflicts with built-in commands
            if command.name in self._builtin_commands:
                logger.warning(f"Cannot override built-in command: {command.name}")
                return False

            self.commands[command.name] = command
            logger.debug(f"Registered dynamic command: {command.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register command {command.name}: {e}")
            return False

    def unregister_command(self, name: str) -> bool:
        """
        Unregister a dynamic command.

        Args:
            name: Command name (without slash)

        Returns:
            True if unregistered successfully, False otherwise
        """
        if name in self.commands:
            del self.commands[name]
            logger.debug(f"Unregistered command: {name}")
            return True
        return False

    def get_command(self, name: str) -> Optional[DynamicCommand]:
        """
        Get a registered command by name.

        Args:
            name: Command name (with or without slash)

        Returns:
            The command if found, None otherwise
        """
        # Remove leading slash if present
        if name.startswith("/"):
            name = name[1:]
        return self.commands.get(name)

    def list_commands(self) -> List[str]:
        """
        List all registered dynamic command names.

        Returns:
            List of command names (with slash prefix)
        """
        return [f"/{name}" for name in self.commands.keys()]

    def load_commands_from_directory(self, directory: Optional[Path] = None) -> int:
        """
        Load commands from a directory.

        Args:
            directory: Directory to load from (uses default if None)

        Returns:
            Number of commands loaded
        """
        load_dir = directory or self.commands_dir
        if not load_dir.exists():
            logger.debug(f"Commands directory does not exist: {load_dir}")
            return 0

        loaded_count = 0

        # Load JSON command definitions
        for json_file in load_dir.glob("*.json"):
            try:
                command = self._load_json_command(json_file)
                if command and self.register_command(command):
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load command from {json_file}: {e}")

        # Load Python command modules
        for py_file in load_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # Skip private modules
            try:
                command = self._load_python_command(py_file)
                if command and self.register_command(command):
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load command from {py_file}: {e}")

        # Load shell script commands
        for script_file in load_dir.glob("*.sh"):
            try:
                command = self._load_script_command(script_file)
                if command and self.register_command(command):
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load command from {script_file}: {e}")

        logger.info(f"Loaded {loaded_count} dynamic commands from {load_dir}")
        return loaded_count

    def _load_json_command(self, file_path: Path) -> Optional[DynamicCommand]:
        """Load a command definition from a JSON file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract command info
            name = data.get("name") or file_path.stem
            description = data.get("description", "")
            arguments = data.get("arguments", [])
            metadata = data.get("metadata", {})

            # Check for associated script
            script_path = None
            if "script" in data:
                script_file = file_path.parent / data["script"]
                if script_file.exists():
                    script_path = script_file

            return DynamicCommand(
                name=name,
                description=description,
                script_path=script_path,
                metadata=metadata,
                arguments=arguments,
            )

        except Exception as e:
            logger.error(f"Error loading JSON command from {file_path}: {e}")
            return None

    def _load_python_command(self, file_path: Path) -> Optional[DynamicCommand]:
        """Load a command from a Python module."""
        try:
            import importlib.util

            # Load the module
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for command metadata
            if not hasattr(module, "COMMAND_INFO"):
                logger.warning(f"No COMMAND_INFO found in {file_path}")
                return None

            info = module.COMMAND_INFO
            name = info.get("name", file_path.stem)
            description = info.get("description", "")
            arguments = info.get("arguments", [])

            # Look for handler function
            handler = None
            if hasattr(module, "handle_command"):
                handler = module.handle_command

            return DynamicCommand(
                name=name,
                description=description,
                handler=handler,
                arguments=arguments,
                metadata={"source": str(file_path)},
            )

        except Exception as e:
            logger.error(f"Error loading Python command from {file_path}: {e}")
            return None

    def _load_script_command(self, file_path: Path) -> Optional[DynamicCommand]:
        """Load a command from a shell script."""
        try:
            # Look for metadata in script comments
            name = file_path.stem
            description = ""
            arguments = []

            with open(file_path, "r") as f:
                lines = f.readlines()

            # Parse header comments for metadata
            for line in lines[:20]:  # Check first 20 lines
                if line.startswith("# NAME:"):
                    name = line.replace("# NAME:", "").strip()
                elif line.startswith("# DESCRIPTION:"):
                    description = line.replace("# DESCRIPTION:", "").strip()
                elif line.startswith("# ARG:"):
                    # Format: # ARG: name:type:description
                    arg_info = line.replace("# ARG:", "").strip()
                    parts = arg_info.split(":", 2)
                    if len(parts) >= 1:
                        arg = {"name": parts[0]}
                        if len(parts) >= 2:
                            arg["type"] = parts[1]
                        if len(parts) >= 3:
                            arg["description"] = parts[2]
                        arguments.append(arg)

            return DynamicCommand(
                name=name,
                description=description or f"Script command: {name}",
                script_path=file_path,
                arguments=arguments,
                metadata={"type": "shell_script"},
            )

        except Exception as e:
            logger.error(f"Error loading script command from {file_path}: {e}")
            return None

    def set_builtin_commands(self, commands: List[str]) -> None:
        """
        Set the list of built-in commands to prevent overriding.

        Args:
            commands: List of built-in command names (without slash)
        """
        self._builtin_commands = [cmd.lstrip("/") for cmd in commands]

    def get_completions_for_command(
        self, command_name: str, arg_position: int, text: str
    ) -> List[str]:
        """
        Get autocomplete suggestions for a command's arguments.

        Args:
            command_name: The command name (with or without slash)
            arg_position: The argument position being completed
            text: The current text being typed

        Returns:
            List of completion suggestions
        """
        command = self.get_command(command_name)
        if not command:
            return []

        # Get argument definition for this position
        arg_def = command.get_argument_at_position(arg_position)
        if not arg_def:
            return []

        # Check if argument has predefined choices
        if "choices" in arg_def:
            choices = arg_def["choices"]
            if callable(choices):
                # Dynamic choices function
                try:
                    choices = choices()
                except Exception:
                    return []
            return [c for c in choices if str(c).startswith(text)]

        # Check for type-based completions
        arg_type = arg_def.get("type", "string")
        if arg_type == "file":
            return self._complete_file_path(text)
        elif arg_type == "directory":
            return self._complete_directory_path(text)

        return []

    def _complete_file_path(self, text: str) -> List[str]:
        """Complete file paths."""
        try:
            path = Path(text)
            if text.endswith("/"):
                # List directory contents
                if path.exists() and path.is_dir():
                    return [str(p) for p in path.iterdir()]
            else:
                # Complete partial path
                parent = path.parent
                prefix = path.name
                if parent.exists():
                    return [
                        str(p) for p in parent.iterdir() if p.name.startswith(prefix)
                    ]
        except Exception:
            pass
        return []

    def _complete_directory_path(self, text: str) -> List[str]:
        """Complete directory paths."""
        try:
            paths = self._complete_file_path(text)
            # Filter to only directories
            return [p for p in paths if Path(p).is_dir()]
        except Exception:
            pass
        return []
