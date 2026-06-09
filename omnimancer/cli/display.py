"""Display methods for the CLI interface.

This module contains all display/output methods extracted from interface.py,
including help text, status display, and message formatting.
"""

import re
from enum import Enum
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class MessageType(Enum):
    """Message type enumeration for unified display."""

    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DisplayManager:
    """Unified display manager for consistent message formatting."""

    def __init__(self, console: Console):
        self.console = console

        self._formats: Dict[MessageType, Dict[str, Any]] = {
            MessageType.SUCCESS: {
                "text": "[green]✓ {message}[/green]",
                "panel": None,
            },
            MessageType.ERROR: {
                "text": None,
                "panel": {"title": "Error", "style": "red"},
            },
            MessageType.WARNING: {
                "text": "[yellow]⚠ {message}[/yellow]",
                "panel": None,
            },
            MessageType.INFO: {
                "text": "[cyan]ℹ {message}[/cyan]",
                "panel": None,
            },
        }

    def show_message(self, message: str, msg_type: MessageType) -> None:
        format_config = self._formats[msg_type]

        if format_config["text"]:
            self.console.print(format_config["text"].format(message=message))
        elif format_config["panel"]:
            panel_config = format_config["panel"]
            panel = Panel(
                message,
                title=panel_config["title"],
                border_style=panel_config["style"],
            )
            self.console.print(panel)

    def show_panel(
        self,
        content: str,
        title: str,
        style: str = "blue",
        icon: Optional[str] = None,
    ) -> None:
        display_title = f"{icon} {title}" if icon else title
        panel = Panel(content, title=display_title, border_style=style)
        self.console.print(panel)


class DisplayMixin:
    """Mixin providing all display/output methods for CommandLineInterface.

    Expects the host class to have:
        self.console: Console
        self.display_manager: DisplayManager
        self.engine: CoreEngine
    """

    console: Console
    display_manager: Any
    engine: Any

    def _show_welcome(self) -> None:
        welcome_text = Text("Welcome to Omnimancer!", style="bold blue")
        welcome_panel = Panel(
            welcome_text,
            title="Omnimancer CLI",
            border_style="blue",
        )
        self.console.print(welcome_panel)
        self.console.print("Type /help for available commands" " or start chatting!")
        self.console.print()

    def _show_goodbye(self) -> None:
        self.console.print("\n[blue]Goodbye! Thanks for using" " Omnimancer.[/blue]")

    def _show_help(self) -> None:
        help_text = """Available Commands:

Core Commands:
/help      - Show this help message
/quit      - Exit Omnimancer (/exit also works)
/clear     - Clear screen
/status    - Show current status

Provider & Model Management:
/providers - List all AI providers with status
/models    - List available models
/switch    - Switch provider/model: /switch <provider> [model]

Agent Management:
/agent     - Manage agent mode (on/off/status)

MCP Tool Integration:
/tools     - List available MCP tools
/mcp       - MCP server management

Conversation Management:
/save      - Save conversation
/load      - Load conversation
/list      - List saved conversations

Configuration:
/config    - Configuration management

Tips:
- Use Tab completion for commands
- Just type your message to chat with AI!"""

        help_panel = Panel(
            help_text,
            title="Omnimancer Help",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(help_panel)

    def _show_command_help(self, command_name: str) -> None:
        command_name = command_name.lower().strip()
        if command_name.startswith("/"):
            command_name = command_name[1:]

        help_content = self._get_command_help_content(command_name)

        if help_content:
            help_panel = Panel(
                help_content,
                title=f"Help: /{command_name}",
                border_style="green",
            )
            self.console.print(help_panel)
        else:
            self._show_error(f"No help available for command: /{command_name}")
            self._show_info(
                "Available commands: /help, /quit,"
                " /clear, /status, /models,"
                " /providers, /switch, /tools,"
                " /mcp, /save, /load, /list, /config"
            )

    def _get_command_help_content(self, command_name: str) -> str:
        help_content = {
            "help": (
                "Show help information.\n\nUsage:\n"
                "  /help           - Show all commands\n"
                "  /help <command> - Show help for a"
                " specific command"
            ),
            "quit": ("Exit Omnimancer.\n\nUsage: /quit\n" "Aliases: Ctrl+D, /exit"),
            "clear": ("Clear the terminal screen.\n\n" "Usage: /clear"),
            "status": (
                "Show current session status.\n\n"
                "Usage: /status\n\nShows: message count,"
                " provider, model, session ID"
            ),
            "providers": ("List all AI providers with status.\n\n" "Usage: /providers"),
            "models": (
                "List available models.\n\nUsage:"
                " /models [filter_type] [filter_value]"
                "\n\nExamples:\n  /models\n"
                "  /models provider claude"
            ),
            "switch": (
                "Switch AI provider/model.\n\nUsage:\n"
                "  /switch <provider>\n"
                "  /switch <provider> <model>\n\n"
                "Examples:\n  /switch claude\n"
                "  /switch openai gpt-4o"
            ),
            "tools": ("List available MCP tools.\n\n" "Usage: /tools"),
            "mcp": (
                "Manage MCP servers.\n\n"
                "Usage: /mcp <action>\n"
                "Actions: status, health, reload, list"
            ),
            "save": ("Save conversation.\n\n" "Usage: /save [filename]"),
            "load": ("Load conversation.\n\n" "Usage: /load <filename>"),
            "list": ("List saved conversations.\n\n" "Usage: /list"),
            "config": (
                "Configuration management.\n\nUsage:\n"
                "  /config\n"
                "  /config set <key> <value>\n"
                "  /config get <key>\n"
                "  /config validate [provider]"
            ),
            "hooks": (
                "Manage lifecycle hooks (shell commands run on events).\n\n"
                "Usage:\n"
                "  /hooks                       list configured hooks\n"
                "  /hooks add <event> <name> [--matcher RE] [--blocking] "
                "[--timeout N] <command>\n"
                "  /hooks remove <event> <name>\n"
                "  /hooks on | off\n\n"
                "Events: pre_send_message, post_send_message, "
                "tool_use_request, post_tool"
            ),
            "permissions": (
                "Manage permission rules (auto allow/deny/ask for tools).\n\n"
                "Usage:\n"
                "  /permissions                       list rules\n"
                "  /permissions <allow|deny|ask> <tool> [matcher]\n"
                "  /permissions remove <allow|deny|ask> <index>\n"
                "  /permissions on | off\n\n"
                "tool is an operation type (file_write, command_execute, …) "
                "or '*'. Precedence: deny > ask > allow."
            ),
            "prompts": (
                "List and render prompts exposed by connected MCP servers.\n\n"
                "Usage:\n"
                "  /prompts                       list available MCP prompts\n"
                "  /prompts <name> [key=value ...]   render a prompt"
            ),
            "subagents": (
                "Run scoped child agents defined in config.\n\n"
                "Usage:\n"
                "  /subagents                     list configured subagents\n"
                "  /subagents run <name> <task>   run a subagent on a task"
            ),
            "validate": (
                "Validate provider configurations.\n\n"
                "Usage: /validate [provider] [--fix]"
            ),
            "health": ("Check provider health.\n\n" "Usage: /health [provider]"),
            "agent": (
                "Manage agent mode.\n\nUsage:\n"
                "  /agent on    - Enable agent mode\n"
                "  /agent off   - Disable agent mode\n"
                "  /agent status - Show agent status"
            ),
        }

        return help_content.get(command_name, "")

    def _show_provider_help(self, provider_name: str) -> None:
        provider_name = provider_name.lower()

        provider_help = {
            "claude": (
                "Claude (Anthropic)\n\nSetup: Get API"
                " key from"
                " https://console.anthropic.com/\n"
                "Models: claude-sonnet-4, claude-opus-4,"
                " claude-3-5-sonnet\nFeatures: Tool"
                " calling, multimodal, 200K context"
            ),
            "openai": (
                "OpenAI\n\nSetup: Get API key from"
                " https://platform.openai.com/\n"
                "Models: gpt-4o, gpt-4-turbo,"
                " gpt-3.5-turbo\nFeatures: Tool"
                " calling, multimodal, function calling"
            ),
            "gemini": (
                "Google Gemini\n\nSetup: Get API key"
                " from https://aistudio.google.com/\n"
                "Models: gemini-1.5-pro,"
                " gemini-1.5-flash\nFeatures: Tool"
                " calling, multimodal, up to 2M context"
            ),
            "ollama": (
                "Ollama (Local)\n\nSetup: Install from"
                " https://ollama.ai/, run"
                " 'ollama serve'\nModels: llama3.1,"
                " codellama, mistral\nFeatures: Local,"
                " private, no API costs"
            ),
            "bedrock": (
                "AWS Bedrock\n\nSetup: Enable in AWS"
                " console, configure credentials\n"
                "Models: anthropic.claude-3-5-sonnet,"
                " amazon.titan\nFeatures: AWS native,"
                " enterprise security"
            ),
        }

        help_content = provider_help.get(provider_name)
        if help_content:
            help_panel = Panel(
                help_content,
                title=f"{provider_name.title()} Provider Help",
                border_style="cyan",
            )
            self.console.print(help_panel)
        else:
            self._show_error(f"No help available for provider: {provider_name}")
            self._show_info(
                "Available providers: claude, openai, gemini, ollama, bedrock"
            )

    def _show_status(self) -> None:
        summary = self.engine.get_conversation_summary()
        model_info = self.engine.get_current_model_info()

        status_text = f"""Current Status:

Messages in conversation: {summary['message_count']}
Current provider: {summary.get('current_provider') or 'None'}
Current model: {summary.get('current_model') or 'None'}
Session ID: {summary.get('session_id')}
Model available: {'Yes' if model_info else 'No'}"""

        status_panel = Panel(status_text, title="Status", border_style="cyan")
        self.console.print(status_panel)

    def _show_user_message(self, message: str) -> None:
        user_panel = Panel(message, title="You", border_style="green")
        self.console.print(user_panel)

    def _show_assistant_message(self, message: str, model: str) -> None:
        escaped_message = re.sub(
            r"<!--(?:read-only|modifies-system)-->\s*", "", message
        )

        operation_patterns = [
            r"\[FILE_WRITE:[^\]]+\]",
            r"\[FILE_READ:[^\]]+\]",
            r"\[COMMAND_EXEC\]",
            r"\[/COMMAND_EXEC\]",
            r"\[/FILE_WRITE\]",
            r"\[WEB_REQUEST:[^\]]+\\?\]",
            r"\[SAFE_EXEC\]",
            r"\[/SAFE_EXEC\]",
        ]

        for pattern in operation_patterns:
            escaped_message = re.sub(
                pattern,
                lambda m: m.group(0).replace("[", "\\[").replace("]", "\\]"),
                escaped_message,
            )

        assistant_panel = Panel(
            escaped_message,
            title=f"Assistant ({model})",
            border_style="blue",
        )
        self.console.print(assistant_panel)

    def _show_info(self, message: str) -> None:
        self.display_manager.show_message(message, MessageType.INFO)

    def _show_error(self, message: str) -> None:
        self.display_manager.show_message(message, MessageType.ERROR)

    def _show_success(self, message: str) -> None:
        self.display_manager.show_message(message, MessageType.SUCCESS)

    def _show_warning(self, message: str) -> None:
        self.display_manager.show_message(message, MessageType.WARNING)

    def _show_token_status(self, response: Any) -> None:
        input_t = response.input_tokens or 0
        output_t = response.output_tokens or 0
        cost = response.cost_estimate or 0.0
        if input_t or output_t:
            self.console.print(
                f"[dim]  tokens: {input_t} in /"
                f" {output_t} out"
                f" | ~${cost:.4f}[/dim]"
            )

    def _clear_screen(self) -> None:
        self.console.clear()
        self._show_welcome()
