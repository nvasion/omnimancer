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
import sys
from typing import Any, Optional

import click

# Third-party imports
from rich.console import Console
from rich.prompt import Confirm

# Internal imports - Core
from ..core.agent.status_core import EventType
from ..core.agent_mode_manager import AgentModeManager
from ..core.config_manager import ConfigManager
from ..core.engine import CoreEngine
from ..core.history_manager import HistoryManager
from ..core.models import (
    ChatResponse,
    ToolCall,
    ToolResult,
    ToolResultRecord,
    parse_described_tool_calls,
)
from ..core.signal_handler import SignalHandler
from ..events import emitter as fleet_events
from ..events.schema import MESSAGE_PREVIEW_CHARS, PREVIEW_CHARS, truncate

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
from .session import apply_session_overrides
from .system_prompts import build_agent_prompt, get_agent_capabilities_prompt
from .tool_handler import DUPLICATE_CALL_NUDGE, RepeatedCallTracker, ToolHandler
from .turn_notify import TurnNotifier, fire_turn_complete

logger = logging.getLogger(__name__)

# Version import
try:
    from omnimancer import __version__
except ImportError:
    __version__ = "unknown"


def validate_prompt_options(
    prompt: Optional[str], initial_prompt: Optional[str]
) -> None:
    """Reject simultaneous headless and interactive initial prompts.

    Args:
        prompt: Headless prompt value.
        initial_prompt: Interactive first-message value.

    Raises:
        click.UsageError: If both prompt modes were requested.
    """
    if prompt is not None and initial_prompt is not None:
        raise click.UsageError("--initial-prompt cannot be used with -p/--prompt")


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

    def __init__(
        self,
        engine: CoreEngine,
        no_approval: bool = False,
        full_trust: bool = False,
        notify_cmd: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        read_only: bool = False,
    ) -> None:
        """
        Initialize the CLI interface.

        Args:
            engine: Core engine instance
            no_approval: Whether to skip approval prompts (DANGEROUS)
            full_trust: Whether to relax project-local security restrictions.
            notify_cmd: Optional external turn-completion command.
            initial_prompt: Optional first message submitted before the input loop.
            read_only: Whether to deny write and command operations for the session.
        """
        self.engine = engine
        self.no_approval = no_approval
        self.full_trust = full_trust
        self.initial_prompt = initial_prompt
        self.read_only = read_only
        self.turn_notifier = TurnNotifier(notify_cmd=notify_cmd, cwd=os.getcwd())
        self._turn_seq = 0
        # Initialize console with robust terminal handling and fallback mechanism
        self.console = self._initialize_console_with_fallback()
        self.running = False

        # Initialize new unified managers
        self.display_manager = DisplayManager(self.console)
        self.completion_manager = CompletionManager(engine=engine)

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

        # Session token/cost totals (shown by /status)
        from .usage import TokenAccumulator

        self.usage = TokenAccumulator()

        # Input layer: prompt_toolkit on interactive terminals (multiline
        # editing, bracketed paste, Ctrl-R search — and it un-blocks the
        # event loop that builtin input() silently blocked). Non-TTY
        # sessions (pipes, tests) keep the historical input()+readline
        # path. Only the fallback registers readline's atexit history
        # writer, so prompt_toolkit mode never touches readline_history.
        self.prompt_input = None
        if (
            os.environ.get("OMNIMANCER_PLAIN_INPUT") != "1"
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        ):
            try:
                from prompt_toolkit.completion import ThreadedCompleter

                from .prompt_input import PromptInput
                from .pt_completion import OmnimancerCompleter

                self.prompt_input = PromptInput(
                    history_dir=self.history_manager.storage_path,
                    # Threaded: the @-file source may shell out to git.
                    completer=ThreadedCompleter(
                        OmnimancerCompleter(self.completion_manager)
                    ),
                    mode_toggle=self._cycle_session_approval_mode,
                    mode_provider=self._session_approval_mode_name,
                )
            except Exception as e:
                logger.warning(
                    "prompt_toolkit input unavailable (%s); "
                    "falling back to readline",
                    e,
                )
        if self.prompt_input is None:
            # Setup readline for arrow key history navigation
            self._setup_readline_history()

        # Completion is handled by existing _complete_command method

        # Enhanced input disabled to avoid arrow key display issues
        self.enhanced_input = None

        # Register the interactive rate-limit fallback approval callback.
        self._register_fallback_callback()

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

            self._configure_agent_engine_session()

            # Fleet event feed: one JSONL per session (default-on; see
            # EventsConfig). init/emit never raise and no-op when disabled.
            session_config = self.engine.config_manager.get_config()
            await fleet_events.init_events(
                self.turn_notifier.session_id, "interactive", session_config.events
            )
            provider_entry = session_config.providers.get(
                session_config.default_provider
            )
            await fleet_events.emit_event(
                EventType.SESSION_START,
                {
                    "provider": session_config.default_provider,
                    "model": getattr(provider_entry, "model", None),
                    "read_only": bool(getattr(self, "read_only", False)),
                },
            )

            if self.initial_prompt is not None:
                initial_prompt = self.initial_prompt
                self.initial_prompt = None
                await self._process_command(Command.create_chat_message(initial_prompt))

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

            # Close the fleet event feed.
            await fleet_events.emit_event(EventType.SESSION_END, {"reason": "exit"})
            await fleet_events.shutdown_events()

            # Ensure terminal is reset on exit
            self._reset_terminal()

            self._show_goodbye()

    async def _handle_user_input(self) -> None:
        """Handle a single user input cycle."""
        user_input = await self._get_user_input_async()
        if user_input is None:  # EOF
            self.running = False
            return

        command = parse_command(user_input)
        await self._process_command(command)

    def stop(self) -> None:
        """Stop the CLI session."""
        self.running = False

    # ------------------------------------------------------------------
    # Rate-limit fallback
    # ------------------------------------------------------------------

    def _register_fallback_callback(self) -> None:
        """Wire up the interactive approval callback on the engine."""
        try:
            self.engine.set_fallback_approval_callback(self._fallback_approval_callback)
        except Exception as exc:
            logger.debug("Could not register fallback callback: %s", exc)

    async def _fallback_approval_callback(
        self,
        current_provider: str,
        next_provider: str,
        error: str,
    ) -> bool:
        """Show a prompt asking whether to fall back to *next_provider*.

        Called by the engine when a rate-limit error is detected and
        ``auto_fallback`` is False.  Returns True to proceed with the switch.
        """
        # Keep the display clean — print to a fresh line.
        self.console.print()
        self.console.print(
            f"[bold yellow]⚠  Rate limit hit on "
            f"[cyan]{current_provider}[/cyan].[/bold yellow]"
        )
        # Show a brief excerpt of the error for context.
        brief = (error[:120] + "…") if len(error) > 120 else error
        self.console.print(f"   [dim]{brief}[/dim]")
        self.console.print(
            f"   [bold]Fall back to [cyan]{next_provider}[/cyan]?[/bold]"
        )

        try:
            answer = self.console.input("   [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            self.console.print("[dim]Fallback declined.[/dim]")
            return False

        approved = answer in ("", "y", "yes")
        if approved:
            self.console.print(
                f"[green]✓ Switching to [cyan]{next_provider}[/cyan]…[/green]"
            )
        else:
            self.console.print("[dim]Fallback declined.[/dim]")
        return approved

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

            if self.full_trust:
                from .headless import HeadlessRunner

                HeadlessRunner._enable_auto_approval(agent_engine)
                HeadlessRunner._enable_full_trust(agent_engine)

            logger.info("✅ CLI approval integration set up successfully")

            # Clean up deferred setup params if they exist
            if hasattr(self, "_approval_setup_params"):
                delattr(self, "_approval_setup_params")

        except Exception as e:
            logger.error(
                f"Failed to complete async approval integration setup: {e}",
                exc_info=True,
            )

    def _configure_agent_engine_session(self) -> None:
        """Apply process-only trust and read-only settings to the agent engine."""
        agent_engine = getattr(self.engine, "agent_engine", None)
        if agent_engine is None:
            return

        if self.read_only and hasattr(agent_engine, "set_read_only"):
            agent_engine.set_read_only(True)

        if self.full_trust:
            from .headless import HeadlessRunner

            HeadlessRunner._enable_auto_approval(agent_engine)
            HeadlessRunner._enable_full_trust(agent_engine)

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

    def _cycle_session_approval_mode(self) -> None:
        """Shift+Tab handler: advance the /accept session approval mode."""
        integration = getattr(self, "approval_integration", None)
        if integration is not None and hasattr(integration, "cycle_approval_mode"):
            integration.cycle_approval_mode()

    def _session_approval_mode_name(self) -> str:
        """Current approval mode name for the prompt toolbar indicator."""
        integration = getattr(self, "approval_integration", None)
        mode = getattr(integration, "session_approval_mode", None)
        return getattr(mode, "value", "normal")

    async def _get_user_input_async(self) -> Optional[str]:
        """
        Get one user submission without blocking the event loop.

        prompt_toolkit path on interactive terminals; plain-input fallback
        otherwise.

        Returns:
            User input string, or None to end the session (EOF / exit).
        """
        if self.prompt_input is not None:
            try:
                user_input = await self.prompt_input.prompt_async()
            except (EOFError, KeyboardInterrupt):
                return None
            if user_input and user_input.strip():
                self.history_manager.add_command(user_input.strip())
            return user_input
        return self._get_user_input()

    def _get_user_input(self) -> Optional[str]:
        """
        Plain-input fallback for non-TTY sessions (pipes, tests).

        Multi-line paste handling lives in the prompt_toolkit path
        (bracketed paste); this fallback reads single lines.

        Returns:
            User input string or None if EOF
        """
        try:
            user_input = input(">>> ")

            # Add to history if we got valid input
            if user_input and user_input.strip():
                self.history_manager.add_command(user_input.strip())

            return user_input

        except (EOFError, KeyboardInterrupt):
            return None

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

        self.turn_notifier.reset_turn()
        self._turn_final_response = None
        self._turn_seq += 1
        await fleet_events.emit_event(
            EventType.TURN_START,
            {
                "turn": self._turn_seq,
                "prompt_preview": truncate(command.content, PREVIEW_CHARS),
                "prompt_chars": len(command.content),
            },
        )

        try:
            await self._handle_chat_message_turn(command)
        finally:
            await fire_turn_complete(self.engine, self.turn_notifier)
            # Guarded: a diagnostics emission inside a finally must never
            # mask the turn's real exception.
            try:
                turn_payload = self.turn_notifier.build_payload()
                await fleet_events.emit_event(
                    EventType.TURN_END,
                    {
                        "turn": self._turn_seq,
                        "usage": turn_payload.get("usage"),
                        "last_message_preview": truncate(
                            str(turn_payload.get("last-assistant-message") or ""),
                            MESSAGE_PREVIEW_CHARS,
                        ),
                    },
                )
            except Exception as exc:
                logger.debug(f"turn_end event emission failed: {exc}")

    async def _handle_chat_message_turn(self, command: Command) -> None:
        """Process one non-empty chat turn without finalization.

        Args:
            command: Chat message command whose completion is owned by the caller.
        """

        # Show user message
        self._show_user_message(command.content)

        # e: prefix — enhance the draft first (PromptFoundry port). Parity
        # with the user's Claude Code hook: the enhanced prompt is sent
        # automatically; on failure the original draft goes through.
        effective_content = command.content
        from ..core.prompt_enhancer import enhance as enhance_prompt
        from ..core.prompt_enhancer import enhancement_enabled, split_enhance_prefix

        draft = split_enhance_prefix(command.content)
        if draft is not None and not enhancement_enabled(
            self.engine.config_manager.get_config()
        ):
            # Feature off: "e:" is not intercepted — the message goes
            # through verbatim as ordinary chat.
            draft = None
        if draft is not None:
            profile = self._default_enhance_profile()
            with self.console.status(
                f"[dim]Enhancing prompt ({profile})...[/dim]", spinner="dots"
            ):
                enhanced, enhance_ok = await enhance_prompt(
                    draft,
                    profile,
                    self.engine.config_manager,
                    fallback_model=self._enhance_fallback_model(),
                )
            if enhance_ok:
                from rich.panel import Panel

                self.console.print(
                    Panel(
                        enhanced,
                        title=f"Enhanced prompt ({profile})",
                        border_style="magenta",
                    )
                )
                effective_content = enhanced
            else:
                self._show_warning(
                    "Enhancement unavailable — sending the original draft."
                )
                effective_content = draft

        # Expand @file mentions into injected content before sending, so
        # both the native-tool and marker paths receive the same expanded
        # message. The panel above shows the original text.
        from pathlib import Path

        from .file_mentions import expand_file_mentions

        message_content, mentions = expand_file_mentions(effective_content, Path.cwd())
        for mention in mentions:
            if mention.injected:
                self.console.print(f"[dim]  @{mention.path} injected[/dim]")
            else:
                self.console.print(
                    f"[yellow]  @{mention.path} skipped: " f"{mention.reason}[/yellow]"
                )

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
                    await self._handle_tool_calling_flow(message_content)
                else:
                    final_message = message_content
                    if agent_mode:
                        agent_prompt = self._get_agent_capabilities_prompt()
                        final_message = f"{agent_prompt}\n\nUser: {message_content}"

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
                                message_content, response
                            )
                        elif not can_stream:
                            self._show_assistant_message(
                                response.content, response.model_used
                            )
                        self._show_token_status(response)
                        final_response = self._turn_final_response
                        if agent_mode and final_response is not None:
                            self.turn_notifier.record_assistant(
                                final_response.content, final_response
                            )
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

        # Identical repeated calls get skipped with a corrective nudge; the
        # turn is aborted only if the model keeps repeating despite nudges.
        repeat_tracker = RepeatedCallTracker()

        # No iteration cap in interactive mode — the user can interrupt at any
        # time, and the repeat tracker catches actual runaway loops. Headless
        # mode keeps its own hard cap.
        while True:
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
            # Unless it mimicked the "[Called tools: ...]" history notation
            # as plain text — recover those so the turn doesn't end silently.
            if not response.tool_calls:
                recovered = parse_described_tool_calls(response.content)
                if not recovered:
                    return
                response.tool_calls = recovered

            repeat_tracker.record(response.tool_calls)
            offender = repeat_tracker.abort_offender(response.tool_calls)
            if offender is not None:
                summary = self._summarize_tool_call(offender)
                label = f"{offender.name} ({summary})" if summary else offender.name
                self._show_warning(
                    f"Stopping: the model repeated the same tool call "
                    f"{repeat_tracker.count(offender)} times ({label}) despite "
                    "duplicate warnings. Try rephrasing your request or using "
                    "a more capable model."
                )
                return

            self.console.print(
                f"[dim]Executing {len(response.tool_calls)} tool call(s)...[/dim]"
            )
            for tc in response.tool_calls:
                summary = self._summarize_tool_call(tc)
                label = f"{tc.name}: {summary}" if summary else tc.name
                self.console.print(f"  [dim]→ {label}[/dim]")

            # Execute each call, skipping exact repeats with a nudge so the
            # model can recover instead of the turn being killed.
            results = []
            skipped = set()
            for i, tc in enumerate(response.tool_calls):
                if repeat_tracker.is_duplicate(tc):
                    skipped.add(i)
                    results.append(ToolResult(content=DUPLICATE_CALL_NUDGE))
                else:
                    results.append(await tool_handler.execute_tool_call(tc))

            result_parts = []
            records = []
            for i, (tc, result) in enumerate(zip(response.tool_calls, results)):
                summary = self._summarize_tool_call(tc)
                label = f"{tc.name} ({summary})" if summary else tc.name
                # Label results with the call's target so the model can tell
                # which call each result belongs to.
                if result.error:
                    status = f"[{label}] Error: {result.error}"
                    self.console.print(f"  [red]✗ {label}: {result.error}[/red]")
                elif i in skipped:
                    status = f"[{label}] {result.content}"
                    self.console.print(
                        f"  [yellow]↻ {label} — duplicate call, skipped[/yellow]"
                    )
                else:
                    status = f"[{label}] Result: {result.content}"
                    self.console.print(f"  [green]✓ {label}[/green]")
                result_parts.append(status)
                records.append(
                    ToolResultRecord(
                        tool_call_id=tc.id or f"call_{i}",
                        content=result.error or result.content,
                    )
                )

            # "q" at the approval prompt cancels the whole turn — drop back
            # to the prompt instead of letting the model try something else.
            if any(r.cancelled for r in results):
                self.console.print(
                    "[yellow]🚫 Agent turn cancelled — back to prompt.[/yellow]"
                )
                return

            # Feed results back without forcing the model to "continue" — let it
            # decide whether more work is needed or it's time to answer.
            results_text = "Tool results:\n\n" + "\n\n".join(result_parts)
            if self.engine.provider_supports_native_tool_history() is True:
                # Native protocol: record results as structured history and
                # continue with an empty message — flattened "Tool results:"
                # text violates the chat template and makes models leak
                # template tokens as plain text.
                self.engine.record_tool_results(results_text, records)
                current_message = ""
            else:
                current_message = results_text

    async def _stream_tool_response(self, message: str, tools: Any) -> "ChatResponse":
        from ..core.models import ChatResponse, StreamEventType, describe_tool_calls
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
            # Prefer the response's own tool calls (they carry provider ids);
            # the display's reconstruction is the fallback for providers whose
            # final stream event lacks them.
            final_response.tool_calls = final_response.tool_calls or (
                display.tool_calls if display.tool_calls else None
            )
            final_response.content = display.accumulated_text
            # "" is a continuation request — its tool results are already
            # recorded as structured history.
            if message:
                self.engine.chat_manager.add_user_message(message)
            if final_response.is_success:
                # Record tool calls with the text — tool results return as
                # plain text, so history must show which calls were made or
                # the model re-issues them. The structured form rides along
                # for providers that replay history natively.
                recorded = final_response.content or ""
                calls_note = describe_tool_calls(final_response.tool_calls)
                if calls_note:
                    recorded = f"{recorded}\n{calls_note}".strip()
                self.engine.chat_manager.add_assistant_message(
                    recorded,
                    final_response.model_used,
                    tool_calls=final_response.tool_calls,
                    raw_content=final_response.content,
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

    # Subcommand pre-dispatch. cli_main stays a single @click.command (not a
    # group) because converting it would change --help and flag parsing for
    # `omn -p`, which codex-orchestrator and the fleet wrapper parse.
    if len(sys.argv) > 1 and sys.argv[1] == "fleet":
        from omnimancer.tui.fleet.cli import fleet_main

        fleet_main.main(args=sys.argv[2:], prog_name="omn fleet")
        return

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
        "--initial-prompt",
        type=str,
        default=None,
        help="Submit an initial message, then continue in interactive mode",
    )
    @click.option(
        "--notify-cmd",
        type=str,
        default=None,
        help="Command invoked with a turn-completion JSON payload",
    )
    @click.option(
        "--read-only",
        is_flag=True,
        default=False,
        help="Deny file writes and command execution for this session",
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
        "--max-iterations",
        "max_iterations",
        type=int,
        default=None,
        help="Tool iteration cap for headless mode (default: 25)",
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
        initial_prompt: Any,
        notify_cmd: Any,
        read_only: Any,
        output_format: Any,
        verbose: Any,
        max_iterations: Any,
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

        validate_prompt_options(prompt, initial_prompt)

        # Headless pipe mode
        if prompt is not None:
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
                    max_iterations=max_iterations,
                    notify_cmd=notify_cmd,
                    read_only=read_only,
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
                full_trust=dangerously_skip_permissions,
                notify_cmd=notify_cmd,
                initial_prompt=initial_prompt,
                read_only=read_only,
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
