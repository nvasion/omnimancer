"""Readline tab completion for the CLI interface."""

import atexit
import logging
import readline
from typing import Callable, List, Optional

from .commands import SlashCommand, get_command_registry

logger = logging.getLogger(__name__)


class CompletionManager:
    """Unified completion manager for command completion."""

    def __init__(self):
        pass

    def get_completions(
        self, command: str, arg_index: int, text: str, args: List[str]
    ) -> List[str]:
        if command.startswith("/"):
            command = command[1:]

        static_completions = {
            "mcp": {0: ["status", "reload", "connect", "disconnect", "health"]},
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

            return self.completion_manager.get_completions(
                command, arg_index, text, args
            )
        except Exception:
            return []

    def _get_provider_names(self, text: str) -> List[str]:
        try:
            if hasattr(self.engine, "providers") and self.engine.providers:
                provider_names = list(self.engine.providers.keys())
            else:
                provider_names = [
                    "openai", "claude", "gemini", "openrouter", "azure",
                    "bedrock", "mistral", "perplexity", "cohere", "xai",
                    "ollama", "vertex",
                ]

            return [name for name in provider_names if name.startswith(text)]
        except Exception:
            return []

    def _get_model_names(self, provider_name: str, text: str) -> List[str]:
        try:
            model_names = []

            if (
                hasattr(self.engine, "providers")
                and provider_name in self.engine.providers
            ):
                provider = self.engine.providers[provider_name]
                if hasattr(provider, "get_available_models"):
                    models = provider.get_available_models()
                    if isinstance(models, dict):
                        model_names.extend(list(models.keys()))
                    elif isinstance(models, list):
                        if models and hasattr(models[0], "name"):
                            model_names.extend([m.name for m in models])
                        else:
                            model_names.extend(models)

            if hasattr(self.engine, "config_manager"):
                custom_models = self.engine.config_manager.get_custom_models()
                custom_model_names = [
                    m.name for m in custom_models if m.provider == provider_name
                ]
                model_names.extend(custom_model_names)

            return [name for name in model_names if name.startswith(text)]

        except Exception:
            return []

    def _get_custom_model_names(self, text: str) -> List[str]:
        try:
            if hasattr(self.engine, "config_manager"):
                custom_models = self.engine.config_manager.get_custom_models()
                if isinstance(custom_models, dict):
                    model_names = []
                    for provider_models in custom_models.values():
                        if isinstance(provider_models, dict):
                            model_names.extend(provider_models.keys())
                    return [name for name in model_names if name.startswith(text)]
            return []
        except Exception:
            return []
