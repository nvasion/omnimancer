"""
Auto-completion handler for Omnimancer CLI commands.

This module provides readline-based auto-completion for slash commands,
including dynamic commands loaded from ~/.omnimancer/commands directory.
"""

import logging
import readline
from pathlib import Path
from typing import Dict, List, Optional, Set

from .commands import SlashCommand
from .dynamic_commands import CommandRegistry, DynamicCommand

logger = logging.getLogger(__name__)


class CompletionHandler:
    """Handles auto-completion for Omnimancer CLI commands."""

    def __init__(self):
        """Initialize the completion handler."""
        self.command_registry = CommandRegistry()
        self.builtin_commands: Set[str] = set()
        self.dynamic_commands: Dict[str, DynamicCommand] = {}
        self._load_commands()

        # Set up readline completion
        self._setup_readline()

    def _load_commands(self) -> None:
        """Load both built-in and dynamic commands."""
        # Load built-in commands
        self._load_builtin_commands()

        # Load dynamic commands from ~/.omnimancer/commands
        self._load_dynamic_commands()

    def _load_builtin_commands(self) -> None:
        """Load built-in slash commands."""
        try:
            # Get all built-in slash commands
            for cmd in SlashCommand:
                # cmd.value already includes the slash, don't add another one
                self.builtin_commands.add(cmd.value)

            logger.debug(f"Loaded {len(self.builtin_commands)} built-in commands")
        except Exception as e:
            logger.error(f"Failed to load built-in commands: {e}")

    def _load_dynamic_commands(self) -> None:
        """Load dynamic commands from ~/.omnimancer/commands directory."""
        try:
            # Use the default commands directory (~/.omnimancer/commands)
            commands_dir = Path.home() / ".omnimancer" / "commands"

            if not commands_dir.exists():
                logger.debug(f"Commands directory does not exist: {commands_dir}")
                return

            # Set built-in commands to prevent overriding
            builtin_names = [cmd.lstrip("/") for cmd in self.builtin_commands]
            self.command_registry.set_builtin_commands(builtin_names)

            # Load commands from directory
            loaded_count = self.command_registry.load_commands_from_directory(
                commands_dir
            )

            # Update our dynamic commands cache
            self.dynamic_commands = self.command_registry.commands.copy()

            logger.info(f"Loaded {loaded_count} dynamic commands from {commands_dir}")

        except Exception as e:
            logger.error(f"Failed to load dynamic commands: {e}")

    def _setup_readline(self) -> None:
        """Set up readline completion."""
        try:
            # Set the completer function
            readline.set_completer(self.complete)

            # Set completion display matches function for better formatting
            readline.set_completion_display_matches_hook(self._display_matches)

            # Use tab for completion
            readline.parse_and_bind("tab: complete")

            # Enable history and completion
            readline.parse_and_bind("set show-all-if-ambiguous on")
            readline.parse_and_bind("set completion-ignore-case on")

            logger.debug("Readline completion set up successfully")

        except Exception as e:
            logger.error(f"Failed to set up readline completion: {e}")

    def complete(self, text: str, state: int) -> Optional[str]:
        """
        Readline completion function.

        Args:
            text: The text being completed
            state: The completion state (0 for first call, incrementing)

        Returns:
            The next completion option, or None when done
        """
        try:
            if state == 0:
                # First call - generate all possible completions
                line = readline.get_line_buffer()
                self._completions = self._get_completions(line, text)

            # Return the next completion
            if state < len(self._completions):
                return self._completions[state]
            else:
                return None

        except Exception as e:
            logger.error(f"Completion error: {e}")
            return None

    def _get_completions(self, line: str, text: str) -> List[str]:
        """
        Get all possible completions for the given text.

        Args:
            line: The full input line
            text: The text being completed

        Returns:
            List of completion options
        """
        completions = []

        # Check if we're completing a slash command
        if line.strip().startswith("/") or text.startswith("/"):
            completions.extend(self._get_command_completions(text))

        # If no slash commands match and text doesn't start with '/',
        # still show commands if user types '/'
        elif text == "/" or (len(text) == 0 and line.strip() == ""):
            # User typed just '/' - show all commands
            completions.extend(self._get_all_commands())

        return completions

    def _get_command_completions(self, text: str) -> List[str]:
        """
        Get command completions for the given text.

        Args:
            text: The text being completed (may or may not start with '/')

        Returns:
            List of matching commands
        """
        # Ensure text starts with '/' for matching
        if not text.startswith("/"):
            text = "/" + text

        completions = []

        # Match built-in commands
        for cmd in self.builtin_commands:
            if cmd.startswith(text):
                completions.append(cmd)

        # Match dynamic commands
        for cmd_name, cmd in self.dynamic_commands.items():
            slash_name = f"/{cmd_name}"
            if slash_name.startswith(text):
                completions.append(slash_name)

        return sorted(completions)

    def _get_all_commands(self) -> List[str]:
        """Get all available commands."""
        all_commands = []

        # Add built-in commands
        all_commands.extend(self.builtin_commands)

        # Add dynamic commands
        for cmd_name in self.dynamic_commands:
            all_commands.append(f"/{cmd_name}")

        return sorted(all_commands)

    def _display_matches(
        self, substitution: str, matches: List[str], longest_match_length: int
    ) -> None:
        """
        Custom display function for completion matches.

        Args:
            substitution: The substitution text
            matches: List of matches
            longest_match_length: Length of longest match
        """
        try:
            print("\nAvailable commands:")

            # Group matches by type
            builtin_matches = []
            dynamic_matches = []

            for match in matches:
                if match in self.builtin_commands:
                    builtin_matches.append(match)
                else:
                    dynamic_matches.append(match)

            # Display built-in commands
            if builtin_matches:
                print("  Built-in commands:")
                for match in builtin_matches:
                    print(f"    {match}")

            # Display dynamic commands with descriptions
            if dynamic_matches:
                print("  Custom commands:")
                for match in dynamic_matches:
                    cmd_name = match.lstrip("/")
                    cmd = self.dynamic_commands.get(cmd_name)
                    if cmd and cmd.description:
                        print(f"    {match} - {cmd.description}")
                    else:
                        print(f"    {match}")

            # Redisplay the prompt
            print(">>> ", end="", flush=True)

        except Exception as e:
            logger.error(f"Error displaying matches: {e}")

    def reload_commands(self) -> None:
        """Reload all commands (useful for development/testing)."""
        logger.info("Reloading commands...")
        self.dynamic_commands.clear()
        self._load_commands()

    def get_command_help(self, command_name: str) -> Optional[str]:
        """
        Get help text for a specific command.

        Args:
            command_name: The command name (with or without slash)

        Returns:
            Help text or None if command not found
        """
        # Remove leading slash if present
        if command_name.startswith("/"):
            command_name = command_name[1:]

        # Check dynamic commands first
        if command_name in self.dynamic_commands:
            cmd = self.dynamic_commands[command_name]
            help_text = f"/{command_name}"
            if cmd.description:
                help_text += f" - {cmd.description}"

            # Add argument information
            if cmd.arguments:
                help_text += "\n  Arguments:"
                for arg in cmd.arguments:
                    arg_name = arg.get("name", "arg")
                    arg_type = arg.get("type", "string")
                    arg_desc = arg.get("description", "")
                    help_text += f"\n    {arg_name} ({arg_type})"
                    if arg_desc:
                        help_text += f" - {arg_desc}"

            return help_text

        return None
