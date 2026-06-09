"""
Command-line interface for Omnimancer.

This module provides the interactive command-line interface
for the Omnimancer application.
"""

# Standard library imports
import asyncio
import json
import logging
import os
import re
import readline  # noqa: F401
import select
import sys
from typing import Any, Optional

import click

# Third-party imports
from rich.console import Console
from rich.prompt import Confirm

# Internal imports - Core
from ..core.agent_mode_manager import AgentModeManager
from ..core.config_manager import ConfigManager
from ..core.engine import CoreEngine
from ..core.history_manager import HistoryManager
from ..core.models import ChatResponse, ToolCall
from ..core.signal_handler import SignalHandler

# UI imports
from ..ui.cancellation_handler import CancellationHandler, EnhancedStatusDisplay
from ..ui.progress_indicator import ProgressIndicator, set_progress_indicator
from .agent_loop import AgentLoopMixin

# Internal imports - CLI
from .approval_integration import (
    create_cli_approval_integration,
    inject_approval_integration_into_agent_engine,
)
from .command_dispatch import CommandDispatchMixin
from .commands import Command, parse_command
from .completion import CompletionManager, CompletionMixin
from .display import DisplayManager, DisplayMixin
from .system_prompts import build_agent_prompt, get_agent_capabilities_prompt
from .tool_handler import MAX_TOOL_ITERATIONS, ToolHandler

logger = logging.getLogger(__name__)

# Version import
try:
    from omnimancer import __version__
except ImportError:
    __version__ = "unknown"


def apply_session_overrides(
    config_manager: "ConfigManager",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    """Apply --provider/--model/--base-url overrides to the in-memory config.

    These are ephemeral (never written to disk). ``initialize_providers()``
    reads the in-memory config, so mutating it here is enough for the overrides
    to take effect for the session.
    """
    if not (provider or model or base_url):
        return

    from ..core.models import ProviderConfig

    cfg = config_manager.get_config()
    if provider:
        cfg.default_provider = provider

    target = provider or cfg.default_provider
    if not target:
        return

    provider_cfg = cfg.providers.get(target)
    if provider_cfg is None:
        provider_cfg = ProviderConfig(model=model or "")
        cfg.providers[target] = provider_cfg
    if model:
        provider_cfg.model = model
    if base_url:
        provider_cfg.base_url = base_url.rstrip("/")


class CommandLineInterface(
    DisplayMixin,
    CompletionMixin,
    AgentLoopMixin,
    CommandDispatchMixin,
):
    """
    Interactive command-line interface for Omnimancer.

    This class handles user input, command processing, and output
    formatting for the terminal-based interface.
    """

    def __init__(self, engine: CoreEngine, no_approval: bool = False):
        """
        Initialize the CLI interface.

        Args:
            engine: Core engine instance
            no_approval: Whether to skip approval prompts (DANGEROUS)
        """
        self.engine = engine
        self.no_approval = no_approval
        # Initialize console with robust terminal handling and fallback mechanism
        self.console = self._initialize_console_with_fallback()
        self.running = False

        # Initialize new unified managers
        self.display_manager = DisplayManager(self.console)
        self.completion_manager = CompletionManager()

        # Initialize signal handler for graceful shutdown
        self.signal_handler = SignalHandler(getattr(engine, "agent_engine", None))

        # Initialize cancellation handler for ESC key support
        self.cancellation_handler = CancellationHandler(self.console)
        self.status_display = EnhancedStatusDisplay(self.console)

        # Initialize progress indicator
        self.progress_indicator = ProgressIndicator(self.console)
        set_progress_indicator(self.progress_indicator)

        # Initialize agent mode manager
        self.agent_manager = None
        self.agent_progress_ui = None

        # Initialize CLI approval integration
        self.approval_integration = None
        self._setup_approval_integration()

        # Initialize file interaction integration
        self._setup_file_interaction_integration()

        # Initialize command history
        self.history_manager = HistoryManager()

        # Setup readline for arrow key history navigation
        self._setup_readline_history()

        # Completion is handled by existing _complete_command method

        # Enhanced input disabled to avoid arrow key display issues
        self.enhanced_input = None

    def start(self) -> None:
        """Start the interactive CLI session."""
        asyncio.run(self._async_start())

    async def _async_start(self) -> None:
        """Async version of the start method."""
        # Setup signal handlers for graceful shutdown
        self.signal_handler.setup_signal_handlers()

        # Ensure terminal is in normal mode (fix for arrow key display issue)
        self._reset_terminal()

        try:
            self.running = True
            self._show_welcome()

            # Initialize providers
            try:
                await self.engine.initialize_providers()
            except Exception as e:
                self._show_error(f"Failed to initialize providers: {e}")

                # Check if we have no providers configured - if so, start setup wizard
                config = self.engine.config_manager.get_config()
                if not hasattr(config, "providers") or not config.providers:
                    self.console.print(
                        "\n[yellow]No providers configured."
                        " Set API keys in your .env file"
                        " or environment variables."
                        "[/yellow]"
                    )
                else:
                    self.console.print(
                        "[yellow]Check your .env file or"
                        " environment variables for"
                        " provider API keys.[/yellow]"
                    )

                # Continue running even if provider initialization failed

            # Initialize MCP servers
            try:
                await self.engine.initialize_mcp()
            except Exception as e:
                self._show_error(f"Failed to initialize MCP servers: {e}")
                # Continue even if MCP initialization fails

            # Initialize agent mode manager
            try:

                self.agent_manager = AgentModeManager(self.engine.config_manager)

                # Start agent mode if it's enabled by default
                if self.agent_manager.mode.value == "on":
                    await self.agent_manager._start_execution_loop_when_ready()
            except Exception as e:
                self._show_error(f"Failed to initialize agent mode: {e}")
                # Continue without agent mode

            # Enhanced input already initialized in constructor

            # Main interaction loop
            while self.running and not self.signal_handler.shutdown_event.is_set():
                try:
                    # Create a task for user input handling
                    input_task = asyncio.create_task(self._handle_user_input())
                    self.signal_handler.register_operation(input_task)

                    try:
                        await input_task
                    except asyncio.CancelledError:
                        # Operation was cancelled by signal handler
                        break

                except KeyboardInterrupt:
                    # This should be handled by signal handler, but keep as fallback
                    break
                except EOFError:
                    break
                except Exception as e:
                    self._show_error(f"Unexpected error: {e}")

        except KeyboardInterrupt:
            # Fallback in case signal handler doesn't work
            pass
        finally:
            # Shutdown agent mode
            try:
                if self.agent_manager:
                    await self.agent_manager.disable_agent_mode(
                        wait_for_completion=False
                    )
            except Exception:
                pass

            # Cleanup approval integration
            try:
                if hasattr(self, "approval_integration") and self.approval_integration:
                    await self.approval_integration.cleanup()
            except Exception as e:
                logger.debug(f"Error cleaning up approval integration: {e}")

            # Shutdown MCP servers
            try:
                await self.engine.shutdown_mcp()
            except Exception:
                pass

            # Wait for graceful shutdown if needed
            if self.signal_handler.shutdown_in_progress:
                await self.signal_handler.wait_for_shutdown()

            # Ensure terminal is reset on exit
            self._reset_terminal()

            self._show_goodbye()

    async def _handle_user_input(self) -> None:
        """Handle a single user input cycle."""
        user_input = self._get_user_input()
        if user_input is None:  # EOF
            self.running = False
            return

        command = parse_command(user_input)
        await self._process_command(command)

    def stop(self) -> None:
        """Stop the CLI session."""
        self.running = False

    def _initialize_console_with_fallback(self) -> Console:
        """
        Initialize Rich Console with comprehensive fallback handling.

        This method provides a robust console initialization with multiple
        fallback levels to handle various terminal environments and issues.

        Returns:
            Console: A working Rich Console instance
        """
        # Terminal capability detection
        is_tty = (
            hasattr(sys, "stdout")
            and hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
        )

        # Level 1: Full-featured console (best case)
        try:
            if is_tty:
                console = Console(
                    force_terminal=True,
                    legacy_windows=False,
                    width=None,  # Auto-detect width
                    height=None,  # Auto-detect height
                    no_color=False,
                    stderr=False,
                )
                # Test console functionality
                console.size  # This will fail if terminal detection is broken
                return console
        except (
            OSError,
            ImportError,
            AttributeError,
            ValueError,
            RuntimeError,
        ) as e:
            logger.debug(f"Full-featured console initialization failed: {e}")

        # Level 2: Basic console with minimal features
        try:
            console = Console(
                force_terminal=is_tty,
                legacy_windows=True,  # Better compatibility
                no_color=False,
                stderr=False,
            )
            # Test basic functionality
            console.size
            return console
        except (
            OSError,
            ImportError,
            AttributeError,
            ValueError,
            RuntimeError,
        ) as e:
            logger.debug(f"Basic console initialization failed: {e}")

        # Level 3: Safe console (no advanced features)
        try:
            console = Console(
                force_terminal=False,
                no_color=True,  # Disable colors for safety
                stderr=False,
                highlight=False,  # Disable syntax highlighting
                markup=True,  # Keep basic markup
                emoji=False,  # Disable emojis for compatibility
                width=80,  # Fixed width
                legacy_windows=True,
            )
            return console
        except (
            OSError,
            ImportError,
            AttributeError,
            ValueError,
            RuntimeError,
        ) as e:
            logger.debug(f"Safe console initialization failed: {e}")

        # Level 4: Minimal console to stderr (last resort)
        try:
            console = Console(
                file=sys.stderr,
                force_terminal=False,
                no_color=True,
                stderr=True,
                highlight=False,
                markup=False,  # Disable all markup
                emoji=False,
                width=80,
                legacy_windows=True,
            )
            return console
        except Exception as e:
            logger.debug(f"Minimal console initialization failed: {e}")

        # Level 5: Mock console (absolute last resort)
        # This should never fail, but provides a working interface
        class MockConsole:
            """Minimal console implementation for broken environments."""

            def __init__(self) -> None:
                self.file = sys.stdout
                self.stderr = False
                self.quiet = False
                self.size = (80, 24)  # Default size
                self.width = 80
                self.height = 24
                self.encoding = getattr(sys.stdout, "encoding", "utf-8")

            def print(self, *args: Any, **kwargs: Any) -> None:
                """Print to stdout/stderr with basic formatting."""
                # Strip Rich markup for plain text output

                if args:
                    text = str(args[0])
                    # Remove Rich markup tags
                    text = re.sub(r"\[/?[^\]]*\]", "", text)
                    print(text, *args[1:], **kwargs)
                else:
                    print(*args, **kwargs)

            def log(self, *args: Any, **kwargs: Any) -> None:
                """Log method (same as print for mock)."""
                self.print(*args, **kwargs)

            def status(self, *args: Any, **kwargs: Any) -> Any:
                """Mock status context manager."""

                class MockStatus:
                    def __enter__(self) -> None:
                        return self  # type: ignore[return-value]

                    def __exit__(self, *args: Any) -> None:
                        pass

                    def update(self, *args: Any, **kwargs: Any) -> None:
                        pass

                return MockStatus()

            def rule(self, title: Any = None, *args: Any, **kwargs: Any) -> None:
                """Print a simple rule."""
                if title:
                    print(f"--- {title} ---")
                else:
                    print("---")

        logger.warning("All console initialization methods failed, using mock console")
        return MockConsole()  # type: ignore[return-value]

    def _setup_approval_integration(self) -> None:
        """Set up CLI approval integration for agent operations."""
        try:

            # Check if we have an agent engine with approval manager
            agent_engine = getattr(self.engine, "agent_engine", None)
            approval_manager = getattr(self.engine, "enhanced_approval", None)

            if agent_engine and approval_manager:
                # Store setup parameters for deferred async initialization
                self._approval_setup_params = {
                    "approval_manager": approval_manager,
                    "agent_engine": agent_engine,
                }
                logger.debug("CLI approval integration deferred for async setup")
            else:
                logger.debug(
                    "No agent engine or approval manager available for integration"
                )

        except Exception as e:
            logger.error(f"Failed to set up approval integration: {e}")
            # Don't fail CLI startup for approval integration issues

    async def _complete_approval_integration_setup(self) -> None:
        """Complete the async setup of approval integration if deferred."""
        # Skip if already set up
        if self.approval_integration:
            return

        try:

            # Get agent engine - check both direct attribute and via engine
            agent_engine = getattr(self.engine, "agent_engine", None)

            # If no agent_engine on CoreEngine, the engine
            # itself might be an AgentEngine
            if not agent_engine and hasattr(self.engine, "approval"):
                agent_engine = self.engine

            # Get approval manager - try multiple locations
            approval_manager = (
                getattr(agent_engine, "enhanced_approval", None)
                if agent_engine
                else None
            ) or getattr(self.engine, "enhanced_approval", None)

            if not agent_engine:
                logger.debug(
                    "Agent engine not available - approval integration skipped"
                )
                return

            if not approval_manager:
                logger.debug(
                    "Approval manager not available - approval integration skipped"
                )
                return

            # Create approval integration
            self.approval_integration = await create_cli_approval_integration(
                approval_manager=approval_manager,
                console=self.console,
                config={
                    "enable_auto_approval": True,
                    "approval_timeout_seconds": 300,  # 5 minutes
                },
            )

            # Configure no-approval flag if set
            if self.no_approval:
                self.approval_integration.add_no_approval_flag_support(True)

            # Inject into agent engine
            inject_approval_integration_into_agent_engine(
                agent_engine, self.approval_integration
            )

            logger.info("✅ CLI approval integration set up successfully")

            # Clean up deferred setup params if they exist
            if hasattr(self, "_approval_setup_params"):
                delattr(self, "_approval_setup_params")

        except Exception as e:
            logger.error(
                f"Failed to complete async approval integration setup: {e}",
                exc_info=True,
            )

    def _setup_file_interaction_integration(self) -> Any:
        """Set up file interaction integration."""
        try:
            agent_engine = getattr(self.engine, "agent_engine", None)

            if agent_engine and hasattr(
                agent_engine,
                "set_read_before_write_callback",
            ):

                async def simple_review_callback(
                    review_data: Any,
                ) -> Any:
                    """Show path and ask for confirm."""
                    file_path = review_data.get("file_path", "unknown")
                    operation = review_data.get("operation", "modify")
                    self.console.print(
                        f"\n[yellow]Agent wants to"
                        f" {operation}: {file_path}"
                        f"[/yellow]"
                    )
                    approved = Confirm.ask("Approve?", default=True)
                    return {
                        "approved": approved,
                        "cancelled": not approved,
                    }

                agent_engine.set_read_before_write_callback(simple_review_callback)
                logger.debug("File interaction integration" " set up successfully")
            else:
                logger.debug(
                    "No agent engine available for" " file interaction integration"
                )
        except Exception as e:
            logger.error("Failed to set up file interaction" f" integration: {e}")

    def _reset_terminal(self) -> None:
        """Reset terminal to ensure it's in normal mode."""
        try:

            # Force terminal reset even if isatty() returns False
            # This handles cases where terminal detection fails
            if os.name == "posix":
                try:
                    # Reset terminal to sane state
                    os.system("stty sane 2>/dev/null")
                    # Ensure we're in canonical mode with echo
                    os.system("stty icanon echo 2>/dev/null")
                except Exception:
                    pass
        except Exception:
            # Ignore any errors during terminal reset
            pass

    def _get_agent_capabilities_prompt(self) -> str:
        """Get system prompt for agent capabilities."""
        return get_agent_capabilities_prompt()

    def _get_user_input(self) -> Optional[str]:
        """
        Get input from the user with multi-line paste support.

        Returns:
            User input string or None if EOF
        """
        try:
            # Check if there's immediate input available (paste detection)
            # This detects if user is pasting multiple lines
            lines = []

            # Get first line
            first_line = input(">>> ")
            lines.append(first_line)

            # Check for additional pasted lines (non-blocking)
            # On Unix-like systems, check if more input is immediately available
            if sys.stdin.isatty():
                # For interactive terminals, check if more lines are waiting
                import termios
                import tty

                # Save terminal settings
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    # Set non-blocking mode briefly to check for paste
                    tty.setcbreak(sys.stdin.fileno())

                    # Check if data is available (indicates paste)
                    while select.select([sys.stdin], [], [], 0.05)[0]:
                        try:
                            line = sys.stdin.readline()
                            if line:
                                lines.append(line.rstrip("\n"))
                            else:
                                break
                        except Exception:
                            break
                finally:
                    # Restore terminal settings
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

            # Join all lines
            user_input = "\n".join(lines) if len(lines) > 1 else first_line

            # Add to history if we got valid input
            if user_input and user_input.strip():
                self.history_manager.add_command(user_input.strip())

            return user_input

        except (EOFError, KeyboardInterrupt):
            return None
        except Exception:
            # Fallback to simple input on any error
            return first_line if "first_line" in locals() else None

    async def _process_command(self, command: Command) -> None:
        """
        Process a parsed command.

        Args:
            command: Parsed command object
        """
        if command.is_chat_message:
            await self._handle_chat_message(command)
        elif command.is_slash_command:
            await self._handle_slash_command(command)
        elif command.is_dynamic_command:
            await self._handle_dynamic_command(command)
        elif command.is_system_command:
            self._handle_system_command(command)

    async def _handle_chat_message(self, command: Command) -> None:
        """
        Handle a chat message command.

        Args:
            command: Chat message command
        """
        if not command.content.strip():
            return

        # Show user message
        self._show_user_message(command.content)

        # Create AI processing task that can be cancelled
        async def ai_processing_task() -> None:
            self.progress_indicator.disable()

            try:
                agent_mode = (
                    self.agent_manager and self.agent_manager.mode.value == "on"
                )
                use_native_tools = agent_mode and self.engine.provider_supports_tools()

                if agent_mode:
                    await self._complete_approval_integration_setup()

                if use_native_tools:
                    await self._handle_tool_calling_flow(command.content)
                else:
                    final_message = command.content
                    if agent_mode:
                        agent_prompt = self._get_agent_capabilities_prompt()
                        final_message = f"{agent_prompt}\n\nUser: {command.content}"

                    provider = getattr(self.engine, "current_provider", None)
                    can_stream = (
                        not agent_mode and provider and provider.supports_streaming()
                    )

                    if can_stream:
                        response = await self._stream_chat_response(final_message)
                    else:
                        response = await self.engine.send_message(final_message)

                    if response.is_success:
                        if agent_mode:
                            await self._execute_continuous_workflow(
                                command.content, response
                            )
                        elif not can_stream:
                            self._show_assistant_message(
                                response.content, response.model_used
                            )
                        self._show_token_status(response)
                    else:
                        self._show_error(f"Failed to get response: {response.error}")

            except Exception:
                raise
            finally:
                self.progress_indicator.enable()

        # Use enhanced cancellation handler for AI processing with progress display
        try:
            # Set processing flag so Ctrl+C cancels operation instead of exiting
            self.signal_handler.is_processing = True

            await self.cancellation_handler.start_cancellable_operation(
                operation=ai_processing_task,
                status_message="Processing",
                cancellation_message="AI processing cancelled by user",
                signal_handler=self.signal_handler,
            )
        except asyncio.CancelledError:
            # Cancellation message already shown by cancellation handler
            pass
        except Exception as e:
            self._show_error(f"Error sending message: {e}")
        finally:
            # Clear processing flag - back to idle at prompt
            self.signal_handler.is_processing = False
            # Always clear operations on completion or error
            self.progress_indicator.clear_all_operations()

    async def _handle_tool_calling_flow(self, user_message: str) -> None:
        """Execute the native tool calling loop.

        Sends message with tool definitions, executes any tool calls from the
        response, sends results back, and repeats until the model stops
        requesting tools or we hit the iteration limit.
        """
        agent_engine = getattr(self.engine, "agent_engine", None)
        if not agent_engine:
            self._show_error("Agent engine not available for tool calling.")
            return

        tool_handler = ToolHandler(agent_engine)
        tools = tool_handler.get_tool_definitions()

        agent_prompt = build_agent_prompt(supports_tools=True)
        current_message = f"{agent_prompt}\n\nUser: {user_message}"

        provider = getattr(self.engine, "current_provider", None)
        use_streaming = provider and provider.supports_streaming()

        # Detect runaway loops: a weak model may repeat the same tool call
        # forever. Stop if an identical call is issued too many times.
        seen_calls: dict = {}

        for iteration in range(MAX_TOOL_ITERATIONS):
            if use_streaming:
                response = await self._stream_tool_response(current_message, tools)
            else:
                response = await self.engine.send_message_with_tools(
                    current_message, tools
                )
                if response.content:
                    self._show_assistant_message(response.content, response.model_used)

            if not response.is_success:
                self._show_error(f"Failed to get response: {response.error}")
                return

            self._show_token_status(response)

            # No tool calls → the model gave its final answer; we're done.
            if not response.tool_calls:
                return

            repeated = False
            for tc in response.tool_calls:
                sig = (
                    f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, default=str)}"
                )
                seen_calls[sig] = seen_calls.get(sig, 0) + 1
                if seen_calls[sig] >= 3:
                    repeated = True
            if repeated:
                self._show_warning(
                    "Stopping: the model kept repeating the same tool call. "
                    "Try rephrasing your request or using a more capable model."
                )
                return

            self.console.print(
                f"[dim]Executing {len(response.tool_calls)} tool call(s)...[/dim]"
            )
            for tc in response.tool_calls:
                summary = self._summarize_tool_call(tc)
                label = f"{tc.name}: {summary}" if summary else tc.name
                self.console.print(f"  [dim]→ {label}[/dim]")

            results = await tool_handler.execute_tool_calls(response.tool_calls)

            result_parts = []
            for tc, result in zip(response.tool_calls, results):
                summary = self._summarize_tool_call(tc)
                label = f"{tc.name} ({summary})" if summary else tc.name
                if result.error:
                    status = f"[{tc.name}] Error: {result.error}"
                    self.console.print(f"  [red]✗ {label}: {result.error}[/red]")
                else:
                    status = f"[{tc.name}] Result: {result.content}"
                    self.console.print(f"  [green]✓ {label}[/green]")
                result_parts.append(status)

            # Feed results back without forcing the model to "continue" — let it
            # decide whether more work is needed or it's time to answer.
            current_message = "Tool results:\n\n" + "\n\n".join(result_parts)

        self._show_warning(
            "Reached maximum tool call iterations" f" ({MAX_TOOL_ITERATIONS})."
        )

    async def _stream_tool_response(self, message: str, tools: Any) -> "ChatResponse":
        from ..core.models import ChatResponse, StreamEventType
        from ..ui.streaming_display import StreamingDisplay

        display = StreamingDisplay(self.console)
        display.start()
        final_response: Optional[ChatResponse] = None

        try:
            async for event in self.engine.send_message_with_tools_stream(
                message, tools
            ):
                display.handle_event(event)
                if event.type == StreamEventType.MESSAGE_COMPLETE:
                    final_response = event.response
        finally:
            display.stop()

        if final_response:
            final_response.tool_calls = (
                display.tool_calls if display.tool_calls else None
            )
            final_response.content = display.accumulated_text
            self.engine.chat_manager.add_user_message(message)
            if final_response.is_success:
                self.engine.chat_manager.add_assistant_message(
                    final_response.content, final_response.model_used
                )
            return final_response

        return ChatResponse(
            content="",
            model_used="",
            tokens_used=0,
            error="Stream failed",
        )

    async def _stream_chat_response(self, message: str) -> "ChatResponse":
        from ..core.models import ChatResponse, StreamEventType
        from ..ui.streaming_display import StreamingDisplay

        display = StreamingDisplay(self.console)
        display.start()
        final_response: Optional[ChatResponse] = None

        try:
            async for event in self.engine.send_message_stream(message):
                display.handle_event(event)
                if event.type == StreamEventType.MESSAGE_COMPLETE:
                    final_response = event.response
        finally:
            display.stop()

        if final_response:
            final_response.content = display.accumulated_text
            self.engine.chat_manager.add_user_message(message)
            if final_response.is_success:
                self.engine.chat_manager.add_assistant_message(
                    final_response.content, final_response.model_used
                )
            return final_response

        return ChatResponse(
            content="",
            model_used="",
            tokens_used=0,
            error="Stream failed",
        )

    def _summarize_tool_call(self, tool_call: ToolCall) -> str:
        """Return a short, human-readable summary of what a tool call does.

        Picks the most descriptive argument (file path, command, url, etc.) so
        the user can see *what* each call is acting on rather than just the
        tool name. Falls back to compact JSON of the arguments.
        """
        args = getattr(tool_call, "arguments", None) or {}
        if not isinstance(args, dict):
            return ""

        # Keys ordered by how well they describe the action being taken.
        descriptive_keys = (
            "file_path",
            "path",
            "filename",
            "command",
            "cmd",
            "url",
            "pattern",
            "query",
            "search",
            "text",
            "name",
        )
        summary = ""
        for key in descriptive_keys:
            value = args.get(key)
            if value:
                summary = str(value)
                break
        else:
            if args:
                try:
                    summary = json.dumps(args, default=str)
                except (TypeError, ValueError):
                    summary = str(args)

        summary = " ".join(summary.split())
        if len(summary) > 80:
            summary = summary[:77] + "..."
        return summary

    def _handle_keyboard_interrupt(self) -> None:
        """Handle Ctrl+C interrupt."""
        self.console.print("\n[yellow]Use /quit or Ctrl+D to exit[/yellow]")

    # Display methods are in cli/display.py (DisplayMixin)


def main() -> None:
    """
    Main entry point for the Omnimancer CLI application.

    This function initializes the application and starts the interactive CLI.
    """

    @click.command()
    @click.option(
        "--help",
        "-h",
        is_flag=True,
        help="Show this help message and exit",
    )
    @click.option(
        "--version",
        "-v",
        is_flag=True,
        help="Show version information",
    )
    @click.option("--config", "-c", help="Path to configuration file")
    @click.option(
        "--no-approval",
        is_flag=True,
        help="Skip approval prompts and auto-approve all operations",
    )
    @click.option(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Run in headless mode with the given prompt",
    )
    @click.option(
        "--output-format",
        "output_format",
        type=click.Choice(["text", "json", "stream-json"]),
        default="text",
        help="Output format for headless mode",
    )
    @click.option(
        "--verbose",
        is_flag=True,
        default=False,
        help="Verbose output in headless mode",
    )
    @click.option(
        "--dangerously-skip-permissions",
        is_flag=True,
        default=False,
        help="Auto-approve all tool operations (interactive and headless)",
    )
    @click.option(
        "--provider",
        type=str,
        default=None,
        help="AI provider to use",
    )
    @click.option(
        "--model",
        type=str,
        default=None,
        help="Model to use",
    )
    @click.option(
        "--base-url",
        "base_url",
        type=str,
        default=None,
        help="Override the provider API endpoint (headless mode)",
    )
    def cli_main(
        help: Any,
        version: Any,
        config: Any,
        no_approval: Any,
        prompt: Any,
        output_format: Any,
        verbose: Any,
        dangerously_skip_permissions: Any,
        provider: Any,
        model: Any,
        base_url: Any,
    ) -> None:
        """Omnimancer - A multi-model coding agent for the terminal."""

        if help:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            return

        if version:
            click.echo(f"Omnimancer CLI v{__version__}")
            return

        # Headless pipe mode
        if prompt:
            full_prompt = prompt

            if not sys.stdin.isatty():
                stdin_content = sys.stdin.read()
                if stdin_content.strip():
                    full_prompt = f"Context:\n{stdin_content}\n\nRequest: {prompt}"

            from .headless import run_headless

            exit_code = asyncio.run(
                run_headless(
                    prompt=full_prompt,
                    output_format=output_format,
                    config_path=config,
                    no_approval=no_approval or dangerously_skip_permissions,
                    verbose=verbose,
                    provider=provider,
                    model=model,
                    base_url=base_url,
                )
            )
            sys.exit(exit_code)

        # Interactive REPL mode
        try:
            config_manager = ConfigManager(config)

            # Apply CLI overrides for this session (ephemeral — not persisted).
            apply_session_overrides(config_manager, provider, model, base_url)

            engine = CoreEngine(config_manager)
            cli = CommandLineInterface(
                engine,
                no_approval=no_approval or dangerously_skip_permissions,
            )
            cli.start()

        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            click.echo(f"Error starting Omnimancer: {e}", err=True)
            sys.exit(1)

    cli_main()


if __name__ == "__main__":
    main()


# Additional methods that should be part of CommandLineInterface class
# These were accidentally placed outside the class definition
