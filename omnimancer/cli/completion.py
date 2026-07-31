"""Readline tab completion for the CLI interface."""

import atexit
import logging
import readline
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from .commands import SlashCommand, get_command_registry

if TYPE_CHECKING:
    from ..core.engine import CoreEngine

logger = logging.getLogger(__name__)


class CompletionManager:
    """Unified completion manager for command completion.

    With an engine reference it also serves live provider/model names, so
    ``/switch <tab>`` completes real configured providers and their models
    (shared by both the readline fallback and the prompt_toolkit completer).
    """

    def __init__(self, engine: Optional["CoreEngine"] = None) -> None:
        self.engine = engine

    def provider_names(self, prefix: str) -> List[str]:
        """Configured provider names (aliases included) matching prefix."""
        try:
            if self.engine is None or not getattr(self.engine, "providers", None):
                return []
            return sorted(
                name for name in self.engine.providers if name.startswith(prefix)
            )
        except Exception:
            return []

    def model_names(self, provider_name: str, prefix: str) -> List[str]:
        """Model names for a provider: its catalog plus custom models."""
        try:
            if self.engine is None:
                return []
            names: List[str] = []
            providers = getattr(self.engine, "providers", None) or {}
            provider = providers.get(provider_name)
            if provider is not None and hasattr(provider, "get_available_models"):
                models = provider.get_available_models()
                if isinstance(models, dict):
                    names.extend(models.keys())
                elif isinstance(models, list):
                    for model in models:
                        names.append(getattr(model, "name", str(model)))
            config_manager = getattr(self.engine, "config_manager", None)
            if config_manager is not None:
                for model in config_manager.get_custom_models():
                    if getattr(model, "provider", None) == provider_name:
                        names.append(model.name)
            seen = set()
            unique = []
            for name in names:
                if name not in seen:
                    seen.add(name)
                    unique.append(name)
            return [name for name in unique if name.startswith(prefix)]
        except Exception:
            return []

    def custom_model_names(self, prefix: str) -> List[str]:
        """Names of user-registered custom models."""
        try:
            if self.engine is None:
                return []
            config_manager = getattr(self.engine, "config_manager", None)
            if config_manager is None:
                return []
            return [
                model.name
                for model in config_manager.get_custom_models()
                if model.name.startswith(prefix)
            ]
        except Exception:
            return []

    def get_completions(
        self, command: str, arg_index: int, text: str, args: List[str]
    ) -> List[str]:
        if command.startswith("/"):
            command = command[1:]

        # Dynamic, engine-backed argument completion.
        if command == "switch":
            if arg_index == 0:
                return self.provider_names(text)
            if arg_index == 1 and args:
                return self.model_names(args[0], text)
        if command == "remove-model" and arg_index == 0:
            return self.custom_model_names(text)

        static_completions = {
            "mcp": {
                0: [
                    "status",
                    "reload",
                    "connect",
                    "disconnect",
                    "health",
                    "servers",
                    "tools",
                    "add",
                    "remove",
                ]
            },
            "history": {0: ["list", "clear", "export", "import"]},
            "config": {
                0: [
                    "show",
                    "set",
                    "get",
                    "generate",
                    "validate",
                    "setup",
                    "mode",
                    "migrate",
                    "templates",
                    "reset",
                ]
            },
            "hooks": {
                0: ["list", "add", "remove", "on", "off"],
                1: [
                    "pre_send_message",
                    "post_send_message",
                    "tool_use_request",
                    "post_tool",
                ],
            },
            "permissions": {
                0: ["list", "allow", "deny", "ask", "remove", "on", "off"],
            },
            "models": {0: ["refresh"]},
            "prompts": {0: ["list"]},
            "subagents": {0: ["list", "run"]},
            "validate": {0: ["--fix", "--auto-fix"]},
            "health": {0: ["--monitor", "--interval"]},
            "repair": {0: ["--backup", "--force"]},
            "diagnose": {0: ["--verbose", "--deep"]},
        }

        command_completions = static_completions.get(command, {})
        completions = command_completions.get(arg_index, [])

        return [c for c in completions if c.startswith(text)]


class CompletionMixin:
    """Mixin providing readline completion methods for CommandLineInterface.

    Expects the host class to have:
        self.history_manager: HistoryManager
        self.completion_manager: CompletionManager
        self.engine: CoreEngine
    """

    history_manager: Any
    completion_manager: CompletionManager
    engine: "CoreEngine"

    def _setup_readline_history(self) -> None:
        try:
            history_file = self.history_manager.storage_path / "readline_history"
            history_file.parent.mkdir(parents=True, exist_ok=True)

            readline.set_startup_hook(None)
            readline.set_completer(self._complete_command)
            readline.set_completer_delims(" \t\n")
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind(r'"\e[A": history-search-backward')
            readline.parse_and_bind(r'"\e[B": history-search-forward')

            try:
                readline.read_history_file(str(history_file))
            except FileNotFoundError:
                pass

            atexit.register(readline.write_history_file, str(history_file))
            readline.set_history_length(1000)

        except Exception:
            pass

    def _create_completion_callback(
        self,
    ) -> Optional[Callable[[str], List[str]]]:
        def completion_callback(text: str) -> List[str]:
            try:
                completions = self._get_completions(text, text)
                return completions
            except Exception as e:
                logger.debug(f"Completion callback error: {e}")
                return []

        return completion_callback

    def _complete_command(self, text: str, state: int) -> Optional[str]:
        try:
            line_buffer = readline.get_line_buffer()
            completions = self._get_completions(line_buffer, text)

            if state < len(completions):
                return completions[state]
            return None

        except Exception:
            return None

    def _get_completions(self, line_buffer: str, text: str) -> List[str]:
        try:
            parts = line_buffer.split()

            if not parts or (
                len(parts) == 1
                and line_buffer.endswith(" ") is False
                and text.startswith("/")
            ):
                return self._complete_slash_commands(text)

            if parts:
                command = parts[0]
                if command.startswith("/"):
                    return self._complete_command_arguments(
                        command, parts[1:], text, line_buffer.endswith(" ")
                    )

            return []

        except Exception:
            return []

    def _complete_slash_commands(self, text: str) -> List[str]:
        all_commands = SlashCommand.get_all_commands()
        registry = get_command_registry()
        all_commands.extend(registry.list_commands())
        return [cmd for cmd in all_commands if cmd.startswith(text)]

    def _complete_command_arguments(
        self, command: str, args: List[str], text: str, at_end: bool
    ) -> List[str]:
        try:
            if at_end:
                arg_index = len(args)
            else:
                arg_index = len(args) - 1 if args else 0

            return list(
                self.completion_manager.get_completions(command, arg_index, text, args)
            )
        except Exception:
            return []

    # Thin shims: the shared CompletionManager owns the engine-backed
    # sources so the prompt_toolkit completer and this readline path
    # serve identical candidates.
    def _get_provider_names(self, text: str) -> List[str]:
        return self.completion_manager.provider_names(text)

    def _get_model_names(self, provider_name: str, text: str) -> List[str]:
        return self.completion_manager.model_names(provider_name, text)

    def _get_custom_model_names(self, text: str) -> List[str]:
        return self.completion_manager.custom_model_names(text)
