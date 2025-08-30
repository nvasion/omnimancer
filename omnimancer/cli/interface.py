"""
Command-line interface for Omnimancer.

This module provides the interactive command-line interface
for the Omnimancer application.
"""

import asyncio
import atexit
import logging
import readline
import sys
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..core.engine import CoreEngine
from ..core.history_manager import HistoryManager
from ..core.signal_handler import SignalHandler

# Enhanced input temporarily disabled to fix arrow key display issues
# from ..core.enhanced_input import EnhancedInput, create_completion_callback
from ..ui.cancellation_handler import (
    CancellationHandler,
    EnhancedStatusDisplay,
)
from ..ui.progress_indicator import (
    ProgressIndicator,
    set_progress_indicator,
)
from .commands import (
    Command,
    SlashCommand,
    get_command_registry,
    parse_command,
)
from .handlers import AgentCLIHandler, AgentPersonaHandler, PermissionsHandler


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

        # Message format configurations
        self._formats = {
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
        """Show a message with consistent formatting."""
        format_config = self._formats[msg_type]

        if format_config["text"]:
            # Simple text format
            self.console.print(format_config["text"].format(message=message))
        elif format_config["panel"]:
            # Panel format
            panel_config = format_config["panel"]
            panel = Panel(
                message,
                title=panel_config["title"],
                border_style=panel_config["style"],
            )
            self.console.print(panel)

    def show_panel(
        self, content: str, title: str, style: str = "blue", icon: str = None
    ) -> None:
        """Show content in a panel with consistent formatting."""
        display_title = f"{icon} {title}" if icon else title
        panel = Panel(content, title=display_title, border_style=style)
        self.console.print(panel)


class CompletionManager:
    """Unified completion manager for command completion."""

    def __init__(self):
        """Initialize the completion manager."""
        pass

    def get_completions(
        self, command: str, arg_index: int, text: str, args: List[str]
    ) -> List[str]:
        """Get completions for a command."""
        # Remove leading slash if present
        if command.startswith("/"):
            command = command[1:]

        # Static completion mappings
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


class CommandLineInterface:
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
        self.agent_persona_handler = None

        # Initialize agent CLI handler with verbose support
        verbose_mode = (
            getattr(engine, "verbose", False) or sys.argv.count("--verbose") > 0
        )
        self.agent_cli_handler = AgentCLIHandler(
            agent_engine=getattr(engine, "agent_engine", None),
            approval_manager=getattr(engine, "enhanced_approval", None),
            console=self.console,
            verbose=verbose_mode,
        )

        # Initialize permissions handler
        self.permissions_handler = PermissionsHandler(console=self.console)

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
                        "\n[yellow]No providers configured. Starting setup wizard...[/yellow]"
                    )
                    try:
                        from ..core.provider_registry import ProviderRegistry
                        from ..core.setup_wizard import SetupWizard

                        # Initialize components
                        provider_registry = ProviderRegistry()
                        setup_wizard = SetupWizard(
                            self.engine.config_manager, provider_registry
                        )

                        # Run the setup wizard
                        success = await setup_wizard.start_wizard()

                        if success:
                            self.console.print(
                                "\n[bold green]✅ Setup completed successfully![/bold green]"
                            )
                            # Try to initialize providers again after setup
                            try:
                                await self.engine.initialize_providers()
                            except Exception as retry_e:
                                self.console.print(
                                    f"[yellow]Warning: Provider initialization still failed: {retry_e}[/yellow]"
                                )
                                self.console.print(
                                    "[yellow]You can continue using Omnimancer and configure providers later with /setup[/yellow]"
                                )
                        else:
                            self.console.print(
                                "\n[yellow]Setup was cancelled. You can run /setup anytime to configure providers.[/yellow]"
                            )
                    except Exception as setup_e:
                        self._show_error(f"Setup wizard failed: {setup_e}")
                        self.console.print(
                            "[yellow]You can try running /setup manually later.[/yellow]"
                        )
                else:
                    self.console.print(
                        "[yellow]You can try running /setup to reconfigure providers or /validate to check your configuration.[/yellow]"
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
                from ..core.agent_mode_manager import AgentModeManager
                from ..core.agent_progress_ui import AgentProgressUI

                self.agent_manager = AgentModeManager(self.engine.config_manager)
                self.agent_progress_ui = AgentProgressUI(
                    self.agent_manager, self.console
                )
                self.agent_persona_handler = AgentPersonaHandler(self.console)

                # Start agent mode if it's enabled by default
                if self.agent_manager.mode.value == "on":
                    await self.agent_manager._start_execution_loop_when_ready()
                    if self.agent_progress_ui:
                        self.agent_progress_ui.start_monitoring()
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
                if self.agent_progress_ui:
                    self.agent_progress_ui.stop_monitoring()
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

            def __init__(self):
                self.file = sys.stdout
                self.stderr = False
                self.quiet = False
                self.size = (80, 24)  # Default size
                self.width = 80
                self.height = 24
                self.encoding = getattr(sys.stdout, "encoding", "utf-8")

            def print(self, *args, **kwargs):
                """Print to stdout/stderr with basic formatting."""
                # Strip Rich markup for plain text output
                import re

                if args:
                    text = str(args[0])
                    # Remove Rich markup tags
                    text = re.sub(r"\[/?[^\]]*\]", "", text)
                    print(text, *args[1:], **kwargs)
                else:
                    print(*args, **kwargs)

            def log(self, *args, **kwargs):
                """Log method (same as print for mock)."""
                self.print(*args, **kwargs)

            def status(self, *args, **kwargs):
                """Mock status context manager."""

                class MockStatus:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def update(self, *args, **kwargs):
                        pass

                return MockStatus()

            def rule(self, title=None, *args, **kwargs):
                """Print a simple rule."""
                if title:
                    print(f"--- {title} ---")
                else:
                    print("---")

        logger.warning("All console initialization methods failed, using mock console")
        return MockConsole()

    def _setup_approval_integration(self):
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

    async def _complete_approval_integration_setup(self):
        """Complete the async setup of approval integration if deferred."""
        if hasattr(self, "_approval_setup_params") and not self.approval_integration:
            try:
                from .approval_integration import (
                    create_cli_approval_integration,
                    inject_approval_integration_into_agent_engine,
                )

                params = self._approval_setup_params

                # Create approval integration
                self.approval_integration = await create_cli_approval_integration(
                    approval_manager=params["approval_manager"],
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
                    params["agent_engine"], self.approval_integration
                )

                logger.debug("CLI approval integration set up successfully (async)")

                # Clean up setup params
                delattr(self, "_approval_setup_params")

            except Exception as e:
                logger.error(
                    f"Failed to complete async approval integration setup: {e}"
                )

    def _setup_file_interaction_integration(self):
        """Set up CLI file interaction integration for read-before-write operations."""
        try:
            from ..core.agent.read_before_write_ui import (
                create_confirmation_callback,
                create_review_callback,
            )

            # Check if we have an agent engine
            agent_engine = getattr(self.engine, "agent_engine", None)

            if agent_engine and hasattr(agent_engine, "set_read_before_write_callback"):
                # Create review callback with our console
                review_callback = create_review_callback(console=self.console)
                confirmation_callback = create_confirmation_callback(
                    console=self.console
                )

                # Set up the default review callback for read-before-write operations
                agent_engine.set_read_before_write_callback(review_callback)

                # Also set up file existence confirmation if available
                if hasattr(agent_engine.file_system, "set_confirmation_callback"):
                    agent_engine.file_system.set_confirmation_callback(
                        confirmation_callback
                    )

                logger.debug("File interaction UI integration set up successfully")
            else:
                logger.debug(
                    "No agent engine available for file interaction integration"
                )

        except Exception as e:
            logger.error(f"Failed to set up file interaction integration: {e}")
            # Don't fail CLI startup for file interaction integration issues

    def _reset_terminal(self) -> None:
        """Reset terminal to ensure it's in normal mode."""
        try:
            import os

            # Force terminal reset even if isatty() returns False
            # This handles cases where terminal detection fails
            if os.name == "posix":
                try:
                    # Reset terminal to sane state
                    os.system("stty sane 2>/dev/null")
                    # Ensure we're in canonical mode with echo
                    os.system("stty icanon echo 2>/dev/null")
                except:
                    pass
        except Exception:
            # Ignore any errors during terminal reset
            pass

    def _get_agent_capabilities_prompt(self) -> str:
        """Get system prompt describing agent capabilities with safety and directory awareness."""
        from pathlib import Path

        # Get current directory context
        current_dir = Path.cwd()

        # Check if we're in a git repository
        git_repo_root = None
        is_git_repo = False
        relative_path = ""

        try:
            # Walk up the directory tree to find .git folder
            check_dir = current_dir
            while check_dir != check_dir.parent:
                if (check_dir / ".git").exists():
                    git_repo_root = check_dir
                    is_git_repo = True
                    relative_path = str(current_dir.relative_to(git_repo_root))
                    break
                check_dir = check_dir.parent
        except Exception:
            # If any error occurs, assume not in git repo
            pass

        # Build directory context string
        directory_info = f"""
📍 CURRENT ENVIRONMENT:
- Working Directory: {current_dir}
- Git Repository: {'Yes' if is_git_repo else 'No'}"""

        if is_git_repo:
            directory_info += f"""
- Repository Root: {git_repo_root}
- Relative Path: {relative_path if relative_path else '/'}"""

        return f"""SYSTEM: You are an autonomous AI agent with the ability to perform actions on the local system. You have the following capabilities:
{directory_info}

🔧 FILE OPERATIONS:
- Autonomous file operations with rich approval interface
- Create, read, write, and delete files with user consent
- Interactive preview and modification before writing
- Backup and atomic file operations
- File existence checking and safe overwrite protection
- Automatic backup creation for file modifications

💻 COMMAND EXECUTION:
- Execute shell commands and scripts
- Run development tools (git, npm, pip, etc.)
- Compile and run programs
- System administration tasks

🌐 WEB OPERATIONS:
- Make HTTP requests (GET, POST, PUT, DELETE)
- Scrape web content and extract data
- Download files from URLs
- API integrations

⚙️ SYSTEM INTEGRATION:
- MCP (Model Context Protocol) tool integration
- Configuration management
- Environment variable access
- Process monitoring

🔒 SECURITY FEATURES:
- All operations go through security validation
- Read-before-write logic ensures you see existing file content before modifications
- File existence checking prevents accidental overwrites
- User approval required for high-risk operations including file overwrites
- Automatic backup creation when modifying existing files
- Sandboxed execution environment
- Directory awareness prevents operations outside intended scope

SAFETY PROTOCOLS (CANNOT BE OVERRIDDEN):
- Always check file existence before creation or modification
- Show existing file content to user before overwriting
- Request explicit user confirmation for file overwrites
- Create backups automatically when modifying existing files
- Maintain awareness of current working directory and git context
- Validate all file paths are within expected project boundaries

When a user asks you to do something that requires these capabilities, you should:
1. Explain what you're going to do
2. Check existing files and show content if modifying
3. Perform the actual operations using your capabilities
4. Show the results and any backups created

For example, if asked to "create a hello world script", you should:
- Check if the file already exists
- If it exists, show the current content and ask for confirmation
- Create backup if overwriting
- Actually create the file, not just provide instructions

You can perform these operations directly - don't just provide instructions. Take action!

To perform operations, use these markers in your response:
- [FILE_WRITE:filename] content [/FILE_WRITE] - Write content to a file
- [FILE_READ:filename] - Read content from a file  
- [COMMAND_EXEC] command [/COMMAND_EXEC] - Execute a shell command
- [WEB_REQUEST:url] - Make a web request

IMPORTANT: Always use these exact markers when you want to perform operations. Do not just describe what you would do - actually use the markers to do it!"""

    async def _execute_continuous_workflow(
        self, original_message: str, initial_response
    ) -> None:
        """
        Execute a continuous workflow, sending AI responses back for more actions until complete.

        Args:
            original_message: The original user message
            initial_response: The first AI response
        """
        current_response = initial_response
        iteration_count = 0
        max_iterations = 10  # Prevent infinite loops

        while iteration_count < max_iterations:
            iteration_count += 1

            # Parse and execute operations in the current response
            executed_response = await self._parse_and_execute_operations(
                current_response.content
            )

            # Show the response to the user
            self._show_assistant_message(executed_response, current_response.model_used)

            # Check if there were any operation markers that got executed
            import re

            operation_patterns = [
                r"\[FILE_WRITE:[^\]]+\].*?\[/FILE_WRITE\]",
                r"\[FILE_READ:[^\]]+\]",
                r"\[COMMAND_EXEC\].*?\[/COMMAND_EXEC\]",
                r"\[WEB_REQUEST:[^\]]+\]",
            ]

            had_operations = any(
                re.search(pattern, current_response.content, re.DOTALL)
                for pattern in operation_patterns
            )

            # If no operations were executed, we're done
            if not had_operations:
                break

            # Send the executed response back to AI to continue workflow
            continue_message = f"I executed the operations. Here are the results:\n\n{executed_response}\n\nWhat should I do next to complete the task: {original_message}"

            # Get next response from AI
            try:
                next_response = await self.engine.send_message(continue_message)
            except (asyncio.TimeoutError, ConnectionError, OSError):
                # Handle network-related errors gracefully
                break

            if not next_response.is_success:
                self._show_error(f"Workflow continuation failed: {next_response.error}")
                break

            # Check if AI indicates it's done
            done_indicators = [
                "task is complete",
                "analysis complete",
                "finished",
                "done",
                "complete",
                "workflow finished",
                "i'm done",
                "task completed",
                "everything looks good",
            ]
            content_lower = next_response.content.lower()
            if any(indicator in content_lower for indicator in done_indicators):
                # Show final response and break
                self._show_assistant_message(
                    next_response.content, next_response.model_used
                )
                break

            current_response = next_response

        if iteration_count >= max_iterations:
            self._show_warning(
                "Workflow stopped after maximum iterations to prevent infinite loop"
            )

    def _is_action_request(self, message: str) -> bool:
        """
        Determine if a message is asking the AI to perform an action.

        Args:
            message: The user's message

        Returns:
            bool: True if this appears to be an action request
        """
        # Normalize message
        normalized = message.lower().strip()

        # Skip very short messages (likely not action requests)
        if len(normalized.split()) < 2:
            return False

        # Action verbs that indicate the user wants something done
        action_verbs = [
            "analyze",
            "check",
            "review",
            "examine",
            "look at",
            "inspect",
            "fix",
            "repair",
            "solve",
            "resolve",
            "debug",
            "troubleshoot",
            "create",
            "make",
            "build",
            "generate",
            "write",
            "add",
            "update",
            "modify",
            "change",
            "edit",
            "improve",
            "optimize",
            "delete",
            "remove",
            "clean",
            "cleanup",
            "refactor",
            "install",
            "setup",
            "configure",
            "deploy",
            "run",
            "execute",
            "test",
            "validate",
            "verify",
            "scan",
            "find",
            "search",
            "help me",
            "can you",
            "could you",
            "would you",
            "please",
            "implement",
            "develop",
            "code",
            "program",
            "script",
        ]

        # Check for imperative patterns
        imperative_patterns = [normalized.startswith(verb) for verb in action_verbs]

        # Check for action verbs anywhere in the message
        contains_action_verb = any(verb in normalized for verb in action_verbs)

        # Check for question patterns that imply action
        question_patterns = [
            "how do i",
            "how can i",
            "what should i",
            "can you help",
            "could you help",
            "would you help",
            "please help",
        ]

        contains_action_question = any(
            pattern in normalized for pattern in question_patterns
        )

        # Determine if this is an action request
        is_action = (
            any(imperative_patterns) or contains_action_verb or contains_action_question
        )

        # Exclude pure questions without action intent
        pure_question_starters = [
            "what is",
            "what are",
            "who is",
            "who are",
            "when is",
            "when was",
            "where is",
            "why",
        ]
        is_pure_question = any(normalized.startswith(q) for q in pure_question_starters)

        return is_action and not is_pure_question

    async def _parse_and_execute_operations(self, response_content: str) -> str:
        """Parse model response for operation markers and execute them."""
        import re

        from ..core.agent.types import Operation, OperationType

        # Make a copy to modify
        updated_response = response_content

        try:
            # Parse FILE_WRITE operations
            file_write_pattern = r"\[FILE_WRITE:([^\]]+)\](.*?)\[/FILE_WRITE\]"
            for match in re.finditer(file_write_pattern, response_content, re.DOTALL):
                filename = match.group(1).strip()
                content = match.group(2).strip()

                # Create operation
                operation = Operation(
                    type=OperationType.FILE_WRITE,
                    description=f"Write file: {filename}",
                    data={"path": filename, "content": content},
                    requires_approval=False,  # Auto-approve for now
                )

                # Execute using agent engine (if available)
                if hasattr(self.engine, "agent_engine"):
                    result = await self.engine.agent_engine.execute_with_approval(
                        operation
                    )
                    if result.success:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"✅ Successfully created file '{filename}' ({len(content)} characters)",
                        )
                    else:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"❌ Failed to create file '{filename}': {result.error}",
                        )
                else:
                    # Fallback: use simple file write
                    try:
                        with open(filename, "w") as f:
                            f.write(content)
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"✅ Successfully created file '{filename}' ({len(content)} characters)",
                        )
                    except Exception as e:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"❌ Failed to create file '{filename}': {str(e)}",
                        )

            # Parse FILE_READ operations
            file_read_pattern = r"\[FILE_READ:([^\]]+)\]"
            for match in re.finditer(file_read_pattern, response_content):
                filename = match.group(1).strip()

                # Create operation
                operation = Operation(
                    type=OperationType.FILE_READ,
                    description=f"Read file: {filename}",
                    data={"path": filename},
                    requires_approval=False,  # Auto-approve for now
                )

                # Execute using agent engine (if available)
                if hasattr(self.engine, "agent_engine"):
                    result = await self.engine.agent_engine.execute_with_approval(
                        operation
                    )
                    if result.success:
                        file_content = result.data
                        # Truncate if too long
                        if len(file_content) > 5000:
                            file_content = (
                                file_content[:5000] + "\n\n[... content truncated ...]"
                            )
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"📄 Contents of '{filename}':\n```\n{file_content}\n```",
                        )
                    else:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"❌ Failed to read file '{filename}': {result.error}",
                        )
                else:
                    # Fallback: use simple file read
                    try:
                        with open(filename, "r") as f:
                            file_content = f.read()
                        # Truncate if too long
                        if len(file_content) > 5000:
                            file_content = (
                                file_content[:5000] + "\n\n[... content truncated ...]"
                            )
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"📄 Contents of '{filename}':\n```\n{file_content}\n```",
                        )
                    except Exception as e:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"❌ Failed to read file '{filename}': {str(e)}",
                        )

            # Parse COMMAND_EXEC operations
            command_pattern = r"\[COMMAND_EXEC\](.*?)\[/COMMAND_EXEC\]"
            for match in re.finditer(command_pattern, response_content, re.DOTALL):
                command = match.group(1).strip()

                # Execute command
                try:
                    import subprocess

                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        output = result.stdout.strip()
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"✅ Command executed successfully: `{command}`\nOutput: {output}",
                        )
                    else:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"❌ Command failed: `{command}`\nError: {result.stderr}",
                        )
                except Exception as e:
                    updated_response = updated_response.replace(
                        match.group(0),
                        f"❌ Failed to execute command: `{command}`: {str(e)}",
                    )

        except Exception as e:
            self._show_error(f"Error parsing operations: {e}")
            return response_content

        return updated_response

    def _get_user_input(self) -> Optional[str]:
        """
        Get input from the user with basic readline support.

        Returns:
            User input string or None if EOF
        """
        try:
            # Simple input with readline for arrow key history navigation
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

        # Show user message
        self._show_user_message(command.content)

        # Create AI processing task that can be cancelled
        async def ai_processing_task():
            # Disable progress indicator panels
            self.progress_indicator.disable()

            try:
                # Add agent capabilities to message if agent mode is enabled
                final_message = command.content
                if self.agent_manager and self.agent_manager.mode.value == "on":
                    agent_prompt = self._get_agent_capabilities_prompt()
                    final_message = f"{agent_prompt}\n\nUser: {command.content}"

                # Send message to AI provider
                response = await self.engine.send_message(final_message)

                if response.is_success:
                    # Handle agent workflows with continuous execution
                    if self.agent_manager and self.agent_manager.mode.value == "on":
                        await self._execute_continuous_workflow(
                            command.content, response
                        )
                    else:
                        self._show_assistant_message(
                            response.content, response.model_used
                        )
                else:
                    self._show_error(f"Failed to get response: {response.error}")

            except Exception:
                raise
            finally:
                # Re-enable progress indicator for other operations
                self.progress_indicator.enable()

        # Use enhanced cancellation handler for AI processing with progress display
        try:
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
            # Always clear operations on completion or error
            self.progress_indicator.clear_all_operations()

    async def _handle_slash_command(self, command: Command) -> None:
        """
        Handle a slash command.

        Args:
            command: Slash command
        """
        slash_cmd = command.slash_command

        if slash_cmd == SlashCommand.HELP:
            # Check if specific command help is requested
            if command.args:
                self._show_command_help(command.args[0])
            else:
                self._show_help()
        elif slash_cmd == SlashCommand.QUIT:
            self.stop()
        elif slash_cmd == SlashCommand.CLEAR:
            self._clear_screen()
        elif slash_cmd == SlashCommand.STATUS:
            self._show_status()
        elif slash_cmd == SlashCommand.MODELS or slash_cmd == SlashCommand.MODEL:
            await self._show_models(command)
        elif slash_cmd == SlashCommand.SWITCH:
            await self._handle_switch_command(command)
        elif slash_cmd == SlashCommand.SAVE:
            await self._handle_save_command(command)
        elif slash_cmd == SlashCommand.LOAD:
            await self._handle_load_command(command)
        elif slash_cmd == SlashCommand.LIST:
            await self._show_conversations()
        elif slash_cmd == SlashCommand.PROVIDERS:
            await self._show_providers(command)
        elif slash_cmd == SlashCommand.TOOLS:
            await self._show_tools()
        elif slash_cmd == SlashCommand.MCP:
            await self._handle_mcp_command(command)
        elif slash_cmd == SlashCommand.SETUP:
            await self._handle_setup_command(command)
        elif slash_cmd == SlashCommand.VALIDATE:
            await self._handle_validate_command(command)
        elif slash_cmd == SlashCommand.HEALTH:
            await self._handle_health_command(command)
        elif slash_cmd == SlashCommand.REPAIR:
            await self._handle_repair_command(command)
        elif slash_cmd == SlashCommand.DIAGNOSE:
            await self._handle_diagnose_command(command)
        elif slash_cmd == SlashCommand.HISTORY:
            await self._handle_history_command(command)
        elif slash_cmd == SlashCommand.ADD_MODEL:
            await self._handle_add_model_command(command)
        elif slash_cmd == SlashCommand.REMOVE_MODEL:
            await self._handle_remove_model_command(command)
        elif slash_cmd == SlashCommand.LIST_CUSTOM_MODELS:
            await self._handle_list_custom_models_command(command)
        elif slash_cmd == SlashCommand.AGENT:
            await self._handle_agent_command(command)
        elif slash_cmd == SlashCommand.AGENTS:
            await self._handle_agents_command(command)
        elif slash_cmd == SlashCommand.AGENTSTATUS:
            await self._handle_agent_status_command(command)
        elif slash_cmd == SlashCommand.APPROVALS:
            await self._handle_approvals_command(command)
        elif slash_cmd == SlashCommand.PERMISSIONS:
            await self._handle_permissions_command(command)
        else:
            self._show_info(f"Command {slash_cmd.value} is not yet implemented")

    async def _handle_dynamic_command(self, command: Command) -> None:
        """
        Handle a dynamic command.

        Args:
            command: Dynamic command
        """
        dynamic_cmd = command.dynamic_command
        if not dynamic_cmd:
            self._show_error("Invalid dynamic command")
            return

        try:
            # Check if command has a Python handler
            if dynamic_cmd.handler:
                # Call the Python handler
                result = await self._execute_dynamic_handler(
                    dynamic_cmd.handler, command.args
                )
                if result:
                    self.console.print(result)

            # Check if command has a script to execute
            elif dynamic_cmd.script_path:
                result = await self._execute_dynamic_script(
                    dynamic_cmd.script_path, command.args
                )
                if result:
                    self.console.print(result)

            else:
                self._show_error(
                    f"Dynamic command '{dynamic_cmd.name}' has no handler or script"
                )

        except Exception as e:
            self._show_error(
                f"Error executing dynamic command '{dynamic_cmd.name}': {e}"
            )

    async def _execute_dynamic_handler(
        self, handler: Callable, args: List[str]
    ) -> Optional[str]:
        """Execute a Python handler for a dynamic command."""
        import inspect

        # Check if handler is async
        if inspect.iscoroutinefunction(handler):
            return await handler(args, engine=self.engine, console=self.console)
        else:
            return handler(args, engine=self.engine, console=self.console)

    async def _execute_dynamic_script(
        self, script_path: Path, args: List[str]
    ) -> Optional[str]:
        """Execute a script for a dynamic command."""
        import asyncio

        try:
            # Make script executable if it isn't already
            script_path.chmod(script_path.stat().st_mode | 0o111)

            # Execute the script
            process = await asyncio.create_subprocess_exec(
                str(script_path),
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Script execution failed"
                self._show_error(f"Script error: {error_msg}")
                return None

            return stdout.decode() if stdout else None

        except Exception as e:
            self._show_error(f"Failed to execute script: {e}")
            return None

    def _handle_system_command(self, command: Command) -> None:
        """
        Handle a system command.

        Args:
            command: System command
        """
        if command.content == "quit":
            self.stop()

    def _handle_keyboard_interrupt(self) -> None:
        """Handle Ctrl+C interrupt."""
        self.console.print("\n[yellow]Use /quit or Ctrl+D to exit[/yellow]")

    def _show_welcome(self) -> None:
        """Show welcome message."""
        welcome_text = Text("Welcome to Omnimancer!", style="bold blue")
        welcome_panel = Panel(welcome_text, title="Omnimancer CLI", border_style="blue")
        self.console.print(welcome_panel)
        self.console.print("Type /help for available commands or start chatting!")
        self.console.print()

    def _show_goodbye(self) -> None:
        """Show goodbye message."""
        self.console.print("\n[blue]Goodbye! Thanks for using Omnimancer.[/blue]")

    def _show_help(self) -> None:
        """Show comprehensive help information with enhanced command documentation."""
        help_text = """Available Commands:

🔧 Core Commands:
/help      - Show this comprehensive help message
/quit      - Exit Omnimancer gracefully (/exit also works)
/clear     - Clear the screen and show welcome message
/status    - Show current status, conversation info, and model details

🤖 Provider & Model Management:
/providers - List all AI providers with status and capabilities
           Shows: Provider name, status, model count, features (🔧 tools, 🖼️ multimodal)
           Supports 11+ providers: Claude, OpenAI, Gemini, Perplexity, xAI, Mistral, 
           Azure OpenAI, Vertex AI, Bedrock, OpenRouter, Claude-code, Cohere, Ollama
           
/models    - List available AI models grouped by provider with enhanced information
           Shows: Model names, SWE scores, pricing per million tokens, capabilities
           Indicators: 🔧 Tool calling, 🖼️ Multimodal, ⭐ Latest, 💰 Cost info, ★★★ SWE ratings
           Features: Filter by provider, capability, price range, performance scores
           Usage: /models [filter_type] [filter_value]
           Filters: provider <name>, capability <tools|multimodal>, price <max_cost>, 
                   performance <min_score>, free, latest
           Examples: /models provider openrouter
                    /models capability tools
                    /models price 5.0
                    /models performance 70
           
/switch    - Switch AI provider and/or model with validation
           Usage: /switch <provider> [model]
           Examples: /switch claude
                    /switch openai gpt-4o
                    /switch perplexity sonar-pro
                    /switch xai grok-3
           Features: Auto-completion suggestions, capability display, model recommendations
           • Persistent state across sessions

📝 Custom Model Management:
/add-model - Add custom model ID to configuration for any provider
           Usage: /add-model <model_id> <provider> [description] [key=value...]
           Parameters: max_tokens=N, cost_input=N, cost_output=N, swe_score=N,
                      supports_tools=true/false, supports_multimodal=true/false, is_free=true/false
           Examples: /add-model z-ai/glm-4.5-air:free openrouter "GLM 4.5 Air Free" is_free=true
                    /add-model anthropic/claude-3-opus custom "Custom Claude Opus" cost_input=15
           
/remove-model - Remove custom model from configuration
           Usage: /remove-model <model_id> <provider>
           Example: /remove-model z-ai/glm-4.5-air:free openrouter
           
/list-custom-models - Display all configured custom models
           Shows: Model name, provider, description, capabilities, pricing

🤖 Agent Management:
/agent     - Manage agent personas and mode
           Subcommands:
           • list                    - Show all available agent personas
           • use <persona>           - Switch to a specific persona
           • current                 - Show currently active persona
           • info <persona>          - Show detailed persona information
           • history                 - Show persona switch history
           • on/enable               - Enable agent mode
           • off/disable             - Disable agent mode
           • status                  - Show agent mode status
           
           Available Personas:
           • coding      - Development-focused with Claude Sonnet + dev tools
           • research    - Research-oriented with Perplexity + web search
           • creative    - Creative writing with high-temperature models
           • performance - Fast and cost-efficient with optimized models
           • general     - Balanced general-purpose configuration
           
           Examples: /agent use coding
                    /agent list
                    /agent info research
                    /agent on --auto-approve

/agents    - Alias for /agent (deprecated)

/agentstatus - Enhanced agent status display with real-time monitoring
           Subcommands:
           • show                    - Show current status snapshot
           • show --detailed         - Show detailed status information
           • start                   - Start live status display with real-time updates
           • stop                    - Stop live status display
           • quick                   - Get quick one-line status summary
           • help                    - Show status command help
           
           Features:
           • Real-time operation tracking with progress indicators
           • Event streaming for instant status updates
           • Rich terminal UI with colored status displays
           • Operation history and performance metrics
           
           Examples: /agentstatus show -d
                    /agentstatus start
                    /agentstatus quick

🔐 Approval Management:
/approvals - Manage approval decisions and batch operations
           Individual Approvals:
           • list                        - List all stored approval decisions with metadata
           • revoke <signature>          - Revoke a specific stored approval by signature
           • clear                       - Clear all stored approval decisions
           • stats                       - Show approval statistics and breakdowns
           • cleanup                     - Remove expired approval decisions
           
           Batch Operations:
           • batch-list                  - List pending batch approval requests
           • batch-show <batch_id>       - Show detailed information for a batch
           • batch-approve <batch_id>    - Approve entire batch
           • batch-deny <batch_id>       - Deny entire batch
           • batch-filter <batch_id>     - Apply filters to batch display
           • batch-interactive <batch_id> - Interactive batch approval mode
           
           Features:
           • View approval history with timestamps and expiration dates
           • Monitor approval patterns by operation type and risk level
           • Manage auto-approval storage for repeated operations
           • Interactive batch approval with filtering and sorting
           • Security through signature-based approval matching
           
           Examples: /approvals list
                    /approvals batch-list
                    /approvals batch-show abc123
                    /approvals batch-interactive abc123
                    /approvals batch-filter abc123 --risk high --type file_write

🛠️ MCP Tool Integration:
/tools     - List all available MCP tools from connected servers
           Shows: Tool name, description, server source, auto-approval status
           
/mcp       - Comprehensive MCP server management
           Actions:
           • status  - Show all server statuses and connection info
           • health  - Check health of all MCP servers
           • reload  - Hot-reload server configurations
           • list    - List configured servers
           Usage: /mcp <action> [server_name]
           Examples: /mcp status
                    /mcp reload filesystem

💬 Conversation Management:
/save      - Save current conversation with optional filename
           Usage: /save [filename]
           Auto-generates filename if not provided
           
/load      - Load a previously saved conversation
           Usage: /load <filename>
           Shows conversation metadata after loading
           
/list      - List all saved conversations with details
           Shows: Filename, creation date, message count, model used

🚀 Setup & Initialization:
/setup     - Interactive setup wizard for first-time configuration
           Features:
           • Guided provider selection with detailed descriptions
           • Interactive API key setup with real-time validation
           • Model selection with capability information
           • Configuration testing and troubleshooting guidance
           • Support for all 11+ providers with specific setup instructions


🔍 Validation & Health Monitoring:
/validate  - Validate provider configurations and credentials
           Usage: /validate [provider] [--fix]
           Examples: /validate                  (check all providers)
                    /validate claude           (check specific provider)
                    /validate --fix            (auto-fix issues)
           
/health    - Check provider health and connectivity
           Usage: /health [provider] [--monitor] [--interval N]
           Examples: /health                    (check all providers)
                    /health openai             (check specific provider)
                    /health --monitor          (continuous monitoring)
           
/repair    - Repair configuration issues automatically
           Usage: /repair [provider] [--auto] [--backup]
           Examples: /repair --auto             (auto-repair all issues)
                    /repair claude --auto      (repair specific provider)
           
/diagnose  - Comprehensive configuration diagnosis
           Usage: /diagnose [provider] [--detailed]
           Examples: /diagnose                  (full system diagnosis)
                    /diagnose --detailed       (detailed analysis)
                    /diagnose claude           (provider-specific diagnosis)

/history   - Command history management and navigation
           Usage: /history [action] [args...]
           Actions: recent          - Show recent commands (default: 20)
                   search <query>   - Search command history
                   stats           - Display history statistics
                   clear           - Clear all history (requires confirmation)
                   export <path>   - Export history to file (json/txt)
           Examples: /history                   (show recent commands)
                    /history recent 50         (show last 50 commands)
                    /history search "model"    (find commands with "model")
                    /history export ~/history.json

🚀 Enhanced Features:

Multi-Provider Support:
• Claude (Anthropic) - Advanced reasoning, tool calling, multimodal
• OpenAI - GPT models, function calling, vision capabilities  
• Gemini (Google) - Massive context windows, multimodal AI
• Cohere - Conversational AI, multilingual support
• Ollama - Local AI models, complete privacy, no API costs

MCP Integration:
• File system operations (read, write, search files)
• Web search with real-time information
• Database queries (SQLite, PostgreSQL)
• Git repository management
• Cloud service integrations (AWS, Docker, Kubernetes)
• Development tools and API integrations

Tool Calling Support:
Compatible providers (Claude, OpenAI, Gemini) can automatically use MCP tools:
• Automatic tool discovery and execution
• User approval system with auto-approve options
• Secure sandboxed execution environment
• Rich tool result formatting and error handling

Provider Health Monitoring:
• Real-time status checking for all providers
• Automatic retry logic with exponential backoff
• Graceful degradation when providers are unavailable
• Connection health indicators and diagnostics

💡 Tips:
• Use Tab completion for commands and arguments
• Provider switching keeps message history, but different models may interpret context differently
• MCP tools enhance AI capabilities significantly
• Check /status regularly for system health
• Save important conversations with /save

🆘 Getting Help:
• Type /help [command] for specific command help
• Visit: https://github.com/omnimancer-cli/omnimancer
• Documentation: docs/provider-setup.md, docs/mcp-setup.md

Just type your message to start chatting with AI!"""

        help_panel = Panel(
            help_text,
            title="Omnimancer Help - Enhanced AI Integration v0.4.0",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(help_panel)

    def _show_command_help(self, command_name: str) -> None:
        """Show detailed help for a specific command."""
        command_name = command_name.lower().strip()

        # Remove leading slash if present
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
                "Available commands: /help, /quit, /clear, /status, /models, /providers, /switch, /tools, /mcp, /save, /load, /list, /config"
            )

    def _get_command_help_content(self, command_name: str) -> str:
        """Get detailed help content for a specific command."""
        help_content = {
            "help": """Show help information for Omnimancer commands.

Usage:
  /help           - Show comprehensive help for all commands
  /help <command> - Show detailed help for a specific command

Examples:
  /help
  /help switch
  /help mcp

The help system provides detailed information about command usage,
parameters, examples, and related features.""",
            "quit": """Exit Omnimancer gracefully.

Usage:
  /quit

Aliases:
  Ctrl+D (EOF)
  /exit (system command)

This command will:
• Save any pending conversation state
• Disconnect from MCP servers cleanly
• Close all provider connections
• Exit the application

No confirmation is required.""",
            "clear": """Clear the terminal screen and redisplay the welcome message.

Usage:
  /clear

This command will:
• Clear all text from the terminal
• Redisplay the Omnimancer welcome banner
• Preserve conversation history (not cleared from memory)
• Keep all provider connections active

Use this to clean up your terminal view without losing your session.""",
            "status": """Show current Omnimancer status and session information.

Usage:
  /status

Information displayed:
• Current conversation message count
• Active AI model and provider
• Session ID for debugging
• Model availability status
• Provider connection health
• MCP server status (if configured)

This is useful for debugging connection issues or checking
your current session state.""",
            "providers": """List all available AI providers with their status and capabilities.

Usage:
  /providers

Information shown for each provider:
• Provider name and status (✅ Available, ❌ Unavailable, ⚙️ Configured)
• Number of available models
• Capabilities: 🔧 Tool calling, 🖼️ Multimodal support
• Connection health and last check time
• Configuration status

Supported providers:
• Claude (Anthropic) - Advanced reasoning, tool calling
• OpenAI - GPT models, function calling, vision
• Gemini (Google) - Large context, multimodal AI
• Cohere - Conversational AI, multilingual
• Ollama - Local AI models, privacy-focused""",
            "models": """List all available AI models grouped by provider with detailed information.

Usage:
  /models

Information shown for each model:
• Model name and version
• Provider and availability status
• Capabilities: 🔧 Tool calling, 🖼️ Multimodal, ⭐ Latest version
• Context window size (token limits)
• Cost information (where available)
• Performance characteristics

Model indicators:
🔧 - Supports tool calling (MCP integration)
🖼️ - Supports multimodal input (images, etc.)
⭐ - Latest/recommended version
💰 - Cost per token information available

Use this to choose the best model for your specific task.""",
            "switch": """Switch between AI providers and models with validation.

Usage:
  /switch <provider>        - Switch to provider with default model
  /switch <provider> <model> - Switch to specific provider and model

Examples:
  /switch claude                    - Use Claude with default model
  /switch openai gpt-4o            - Use OpenAI GPT-4o specifically
  /switch ollama llama3.1          - Use local Ollama Llama 3.1
  /switch gemini                   - Use Gemini with default model

Features:
• Maintains conversation history when switching
• Validates provider and model availability
• Shows model capabilities after switching
• Provides suggestions for typos
• Displays MCP tool availability for compatible models

The switch is immediate and message history is preserved, though different models may interpret context differently.""",
            "tools": """List all available MCP (Model Context Protocol) tools from connected servers.

Usage:
  /tools

Information shown for each tool:
• Tool name and description
• Source MCP server
• Auto-approval status
• Required parameters
• Usage examples

Tool categories typically include:
• File operations (read, write, search)
• Web search and information retrieval
• Database queries and management
• Git repository operations
• Cloud service integrations
• Development and deployment tools

Only providers with tool calling support (Claude, OpenAI, Gemini)
can use these tools automatically in conversations.""",
            "mcp": """Manage MCP (Model Context Protocol) servers and connections.

Usage:
  /mcp <action> [server_name]

Actions:
  status   - Show status of all MCP servers
  health   - Check health of all servers
  reload   - Hot-reload server configurations
  list     - List all configured servers

Examples:
  /mcp status                 - Show all server statuses
  /mcp health                 - Check server health
  /mcp reload                 - Reload all server configs
  /mcp reload filesystem      - Reload specific server

Server management:
• Servers can be started/stopped without restarting Omnimancer
• Configuration changes are applied immediately with reload
• Health checks verify server connectivity and tool availability
• Status shows connection state, tool count, and last activity

See docs/mcp-setup.md for server configuration details.""",
            "save": """Save the current conversation to a file.

Usage:
  /save [filename]

Examples:
  /save                    - Auto-generate filename
  /save important-chat     - Save with specific name
  /save project-discussion - Save with descriptive name

Features:
• Auto-generates timestamp-based filename if not provided
• Saves complete conversation history with metadata
• Includes model information and session details
• Files stored in JSON format for easy processing
• Preserves message timestamps and provider information

Saved conversations can be loaded later with /load command.""",
            "load": """Load a previously saved conversation.

Usage:
  /load <filename>

Examples:
  /load important-chat
  /load project-discussion-2024-01-15

Features:
• Restores complete conversation history
• Preserves original model and provider information
• Shows conversation metadata after loading
• Replaces current conversation (use /save first if needed)
• Maintains message timestamps and context

Use /list to see all available saved conversations.""",
            "list": """List all saved conversations with details.

Usage:
  /list

Information shown for each conversation:
• Filename and creation date
• Number of messages in conversation
• AI model used
• File size and last modified time

This helps you:
• Find specific conversations to load
• See conversation history overview
• Manage saved conversation files
• Choose which conversation to restore

Files are stored in your Omnimancer data directory and can be
managed through the filesystem as well.""",
            "config": """Enhanced configuration management with generation capabilities.

Usage:
  /config                           - Show all configuration
  /config generate [template] [path] - Generate configuration files from templates
  /config validate [provider]       - Validate provider configurations
  /config set <key> <value>         - Set configuration value
  /config get <key>                 - Get configuration value

Subcommands:
  generate - Create comprehensive configuration files from templates
    • /config generate                    - Full configuration with all 11+ providers
    • /config generate coding             - Coding-optimized template with dev tools
    • /config generate research           - Research-focused with large context models
    • /config generate creative           - Creative writing optimized settings
    • /config generate general            - Balanced all-purpose configuration
    • /config generate performance        - Speed and cost optimized
    • /config generate claude             - Claude-only configuration
    • /config generate coding ~/my-config.json - Save to specific path

  validate - Test and validate configuration with real-time checks
    • /config validate                    - Check all configured providers
    • /config validate claude             - Check specific provider only
    • /config validate --fix              - Auto-fix detected issues

Template Features:
• Smart Provider Selection: Each template selects optimal providers for its use case
• Optimized Model Choices: Automatically configures the most suitable models
• MCP Tool Integration: Includes relevant MCP servers and development tools
• Tuned Settings: Temperature, token limits, and parameters optimized per use case
• Comprehensive Documentation: Generated configs include detailed explanatory comments

Available Templates:
• coding - Software development (Claude, OpenAI with dev MCP tools)
• research - Academic research (Gemini, Perplexity with web search)
• creative - Content creation (Claude, OpenAI with creative settings)
• general - Balanced usage (multiple providers, versatile settings)
• performance - Cost/speed optimized (efficient models, reduced costs)

Supported Providers (11+):
• Claude, OpenAI, Gemini, Perplexity, xAI (Grok), Mistral
• Azure OpenAI, Vertex AI, AWS Bedrock, OpenRouter
• Claude-code (local), Cohere, Ollama (local)

Examples:
  /config generate coding               - Generate development-optimized config
  /config generate research ~/research.json - Save research config to file
  /config validate                      - Validate all provider configurations
  /config set default_provider claude   - Set default provider

Generated configurations include:
• All provider configurations with placeholder API keys
• Recommended model selections per provider
• MCP server configurations for enhanced capabilities
• Optimized settings for specific use cases
• Detailed comments explaining configuration choices
• Backup functionality for existing configurations""",
            "setup": """Interactive setup wizard for first-time configuration.

Usage:
  /setup

Features:
• Guided provider selection with descriptions
• Interactive API key setup with validation
• Model selection with capability information
• Configuration testing and validation
• Troubleshooting guidance for common issues

The setup wizard will:
1. Show available AI providers with their strengths
2. Guide you through API key setup for your chosen provider
3. Help you select the best model for your needs
4. Test your configuration to ensure it works
5. Save your settings for immediate use

This is the recommended way to configure Omnimancer for first-time users.
You can run the setup wizard multiple times to add more providers.""",
            "validate": """Validate provider configurations and API credentials.

Usage:
  /validate [provider] [--fix]

Examples:
  /validate                       - Check all configured providers
  /validate claude                - Check specific provider only
  /validate --fix                 - Auto-fix detected issues
  /validate claude --fix          - Fix specific provider issues

Validation checks:
• API key format and validity
• Provider endpoint connectivity
• Model availability and access
• Configuration completeness
• Authentication status

Features:
• Real-time credential testing
• Detailed error reporting with solutions
• Auto-fix capability for common issues
• Provider-specific troubleshooting guidance
• Configuration repair suggestions

The validation system helps ensure:
• All providers are properly configured
• API keys are valid and have necessary permissions
• Models are accessible and available
• Network connectivity is working
• Configuration files are properly formatted""",
            "health": """Monitor provider health and connectivity status.

Usage:
  /health [provider] [--monitor] [--interval N]

Examples:
  /health                         - Check all providers once
  /health openai                  - Check specific provider
  /health --monitor               - Continuous monitoring mode
  /health --monitor --interval 30 - Monitor every 30 seconds

Health checks include:
• Connection status and response times
• API endpoint availability
• Rate limit status and quotas
• Model accessibility
• Error rate monitoring

Features:
• Real-time health monitoring
• Historical health data tracking
• Automatic retry logic testing
• Performance metrics collection
• Alert system for degraded services

Monitoring mode provides:
• Continuous health status updates
• Real-time performance metrics
• Automatic issue detection
• Health trend analysis
• Service availability reporting""",
            "repair": """Automatically repair configuration issues and problems.

Usage:
  /repair [provider] [--auto] [--backup]

Examples:
  /repair                         - Interactive repair for all providers
  /repair claude                  - Repair specific provider issues
  /repair --auto                  - Automatic repair without prompts
  /repair --auto --backup         - Auto-repair with configuration backup

Repair capabilities:
• Fix malformed configuration files
• Repair API key format issues
• Update deprecated model references
• Fix provider endpoint URLs
• Restore missing configuration sections

Features:
• Interactive repair with user confirmation
• Automatic backup creation before repairs
• Detailed repair logs and summaries
• Rollback capability if repairs fail
• Provider-specific repair procedures

The repair system can fix:
• Configuration syntax errors
• Missing required settings
• Deprecated configuration formats
• Invalid API key formats
• Broken provider connections""",
            "diagnose": """Comprehensive system diagnosis and troubleshooting.

Usage:
  /diagnose [provider] [--detailed]

Examples:
  /diagnose                       - Full system diagnosis
  /diagnose claude                - Provider-specific diagnosis
  /diagnose --detailed            - Detailed analysis with debug info

Diagnostic checks:
• Complete configuration analysis
• Provider connectivity testing
• Provider connectivity
• MCP server status
• System resource usage

Features:
• Comprehensive system health report
• Detailed error analysis and solutions
• Performance bottleneck identification
• Configuration optimization suggestions
• Troubleshooting guidance with step-by-step fixes

Diagnosis includes:
• Configuration file validation
• Network connectivity tests
• API endpoint accessibility
• Authentication verification
• Resource usage analysis
• Error log examination
• Performance metrics review""",
        }

        return help_content.get(command_name, "")

    def _show_provider_help(self, provider_name: str) -> None:
        """Show provider-specific help and troubleshooting information."""
        provider_name = provider_name.lower()

        provider_help = {
            "claude": """Claude (Anthropic) Provider Help

Setup:
1. Get API key from https://console.anthropic.com/
2. Configure: /config claude api_key YOUR_KEY

Models:
• claude-3-5-sonnet-20241022 - Latest, most capable (200K context)
• claude-3-5-haiku-20241022 - Fast and efficient (200K context)
• claude-3-opus-20240229 - Most powerful for complex tasks

Features:
✅ Tool calling (MCP integration)
✅ Multimodal (text + images)
✅ Large context windows (200K tokens)
✅ Advanced reasoning and analysis

Troubleshooting:
• "Invalid API key" - Check key format (sk-ant-api03-...)
• "Rate limit exceeded" - Upgrade account tier or wait
• "Model not found" - Use exact model names listed above
• "Insufficient credits" - Add payment method in console

Best for: Complex reasoning, code analysis, research tasks""",
            "openai": """OpenAI Provider Help

Setup:
1. Get API key from https://platform.openai.com/
2. Configure: /config openai api_key YOUR_KEY

Models:
• gpt-4o - Latest multimodal model (128K context)
• gpt-4o-mini - Fast, cost-effective (128K context)
• gpt-4-turbo - Enhanced GPT-4 (128K context)
• gpt-3.5-turbo - Affordable option (16K context)

Features:
✅ Tool calling (function calling)
✅ Multimodal (text, images, audio)
✅ Code generation and debugging
✅ Structured outputs

Troubleshooting:
• "Insufficient quota" - Add payment method
• "Model not found" - Check spelling and availability
• "Rate limit" - Upgrade tier or implement delays
• "Invalid request" - Check message format

Best for: General purpose, coding, creative writing""",
            "gemini": """Google Gemini Provider Help

Setup:
1. Get API key from https://aistudio.google.com/
2. Configure: /config gemini api_key YOUR_KEY

Models:
• gemini-1.5-pro-latest - Most capable (2M context)
• gemini-1.5-flash-latest - Fast and efficient (1M context)
• gemini-1.5-flash-8b-latest - Lightweight (1M context)

Features:
✅ Tool calling (function calling)
✅ Massive context windows (up to 2M tokens)
✅ Multimodal (text, images, audio, video)
✅ Advanced reasoning and math

Troubleshooting:
• "API key not valid" - Check key format (AIza...)
• "Quota exceeded" - Enable billing or wait
• "Safety filter" - Rephrase potentially sensitive content
• "Model overloaded" - Try again or use different model

Best for: Long document analysis, multimodal tasks, research""",
            "cohere": """Cohere Provider Help

Setup:
1. Get API key from https://cohere.com/
2. Configure: /config cohere api_key YOUR_KEY

Models:
• command-r-plus - Most advanced (128K context)
• command-r - Balanced performance (128K context)
• command-light - Fast and affordable (4K context)

Features:
✅ Excellent conversational AI
✅ Strong reasoning capabilities
✅ Multilingual support
✅ Cost-effective pricing
❌ Tool calling (coming soon)

Troubleshooting:
• "Invalid API key" - Check key format (co-...)
• "Model not available" - Some models need special access
• "Rate limit" - Check usage limits in dashboard
• "Generation failed" - Try rephrasing or shorter input

Best for: Conversations, content generation, multilingual tasks""",
            "ollama": """Ollama (Local AI) Provider Help

Setup:
1. Install Ollama from https://ollama.ai/
2. Start server: ollama serve
3. Download models: ollama pull llama3.1
4. Configure: /config ollama base_url http://localhost:11434

Popular Models:
• llama3.1 - Latest Meta model (8B, 70B variants)
• codellama - Code-specialized (7B-34B)
• mistral - Efficient general-purpose (7B)
• llava - Vision-language model (7B)

Features:
✅ Complete privacy (local processing)
✅ No API costs or usage limits
✅ Offline usage
✅ Custom model support
❌ Tool calling (limited, model-dependent)

System Requirements:
• RAM: 8GB minimum (16GB+ recommended)
• Storage: 4-40GB per model
• GPU: Optional NVIDIA GPU for acceleration

Troubleshooting:
• "Connection refused" - Start ollama serve
• "Model not found" - Download with ollama pull
• "Out of memory" - Use smaller models or add RAM
• "Slow performance" - Enable GPU or use smaller models
• "Server not responding" - Check port 11434 availability

Commands:
• ollama list - Show installed models
• ollama pull <model> - Download model
• ollama rm <model> - Remove model
• ollama serve - Start server

Best for: Privacy-focused usage, offline work, experimentation""",
            "perplexity": """Perplexity AI Provider Help

Setup:
1. Get API key from https://www.perplexity.ai/settings/api
2. Configure: /config perplexity api_key YOUR_KEY

Models:
• sonar-pro - Advanced reasoning with web search (127K context)
• sonar - Balanced performance with search (127K context)
• llama-3.1-sonar-small-128k-online - Cost-effective with search
• llama-3.1-sonar-large-128k-online - High performance with search

Features:
✅ Real-time web search integration
✅ Tool calling support
✅ Large context windows (up to 127K tokens)
✅ Recency filtering for search results
✅ Citation support

Troubleshooting:
• "Invalid API key" - Check key format (pplx-...)
• "Rate limit exceeded" - Upgrade account tier
• "Search failed" - Check internet connectivity
• "Model not found" - Verify model name spelling

Best for: Research, real-time information, web-enhanced conversations""",
            "xai": """xAI (Grok) Provider Help

Setup:
1. Get API key from https://console.x.ai/
2. Configure: /config xai api_key YOUR_KEY

Models:
• grok-3 - Latest Grok model with enhanced reasoning (131K context)
• grok-3-fast - Faster version for quick responses
• grok-2 - Previous generation model
• grok-beta - Beta access to newest features

Features:
✅ Tool calling support
✅ Multimodal capabilities (text + images)
✅ Large context windows (up to 131K tokens)
✅ Real-time information access
✅ Conversational and creative modes

Troubleshooting:
• "Invalid API key" - Check key format (xai-...)
• "Model not available" - Some models require special access
• "Rate limit" - Check usage limits in console
• "Quota exceeded" - Add payment method

Best for: Creative tasks, conversational AI, multimodal interactions""",
            "mistral": """Mistral AI Provider Help

Setup:
1. Get API key from https://console.mistral.ai/
2. Configure: /config mistral api_key YOUR_KEY

Models:
• mistral-large-latest - Most capable model (128K context)
• mistral-small-latest - Fast and efficient (128K context)
• codestral-latest - Code-specialized model
• mistral-nemo - Lightweight option

Features:
✅ Tool calling support
✅ Strong reasoning capabilities
✅ Code generation and analysis
✅ Multilingual support
✅ Safety-focused design

Troubleshooting:
• "Invalid API key" - Check key format (mistral-...)
• "Model not found" - Verify model availability
• "Rate limit" - Check account limits
• "Safety filter" - Rephrase potentially sensitive content

Best for: Code generation, multilingual tasks, safe AI interactions""",
            "azure": """Azure OpenAI Provider Help

Setup:
1. Create Azure OpenAI resource in Azure Portal
2. Deploy models in Azure OpenAI Studio
3. Get endpoint URL and API key
4. Configure: /config azure api_key YOUR_KEY endpoint YOUR_ENDPOINT

Models (deployed in Azure):
• gpt-4o - Latest GPT-4 Omni model
• gpt-4-turbo - Enhanced GPT-4
• gpt-35-turbo - Cost-effective option
• dall-e-3 - Image generation

Features:
✅ Enterprise security and compliance
✅ Tool calling support
✅ Multimodal capabilities
✅ Custom deployment names
✅ Regional data residency

Configuration:
• api_key - Azure OpenAI API key
• endpoint - Azure OpenAI endpoint URL
• api_version - API version (default: 2024-02-01)
• deployment_name - Custom deployment name

Troubleshooting:
• "Deployment not found" - Check deployment name in Azure
• "Invalid endpoint" - Verify endpoint URL format
• "Authentication failed" - Check API key and permissions
• "Model not available" - Deploy model in Azure OpenAI Studio

Best for: Enterprise deployments, compliance requirements, regional data needs""",
            "vertex": """Google Vertex AI Provider Help

Setup:
1. Create Google Cloud project
2. Enable Vertex AI API
3. Set up authentication (service account or gcloud)
4. Configure: /config vertex project_id YOUR_PROJECT location YOUR_LOCATION

Models:
• gemini-1.5-pro - Most capable Gemini model (2M context)
• gemini-1.5-flash - Fast and efficient (1M context)
• gemini-1.0-pro - Original Gemini model
• text-bison - PaLM-based text model

Features:
✅ Enterprise-grade security
✅ Tool calling support
✅ Massive context windows
✅ Multimodal capabilities
✅ Custom model tuning

Authentication Options:
• Service Account JSON key file
• Application Default Credentials (gcloud)
• Workload Identity (GKE)

Troubleshooting:
• "Authentication failed" - Check service account permissions
• "Project not found" - Verify project ID and access
• "API not enabled" - Enable Vertex AI API in console
• "Location not supported" - Use supported regions

Best for: Google Cloud integration, enterprise AI, large-scale deployments""",
            "bedrock": """AWS Bedrock Provider Help

Setup:
1. Enable AWS Bedrock in your AWS account
2. Request model access in Bedrock console
3. Configure AWS credentials
4. Configure: /config bedrock region YOUR_REGION

Models:
• anthropic.claude-3-5-sonnet - Latest Claude via AWS
• anthropic.claude-3-haiku - Fast Claude model
• amazon.titan-text-express - Amazon's text model
• ai21.j2-ultra - AI21 Labs model

Features:
✅ AWS native integration
✅ Tool calling support (Claude models)
✅ Enterprise security
✅ Pay-per-use pricing
✅ Multiple model providers

Authentication:
• AWS Access Key ID and Secret
• IAM roles and policies
• AWS CLI configuration
• Environment variables

Troubleshooting:
• "Model access denied" - Request access in Bedrock console
• "Authentication failed" - Check AWS credentials
• "Region not supported" - Use Bedrock-enabled regions
• "Throttling" - Check service quotas and limits

Best for: AWS integration, enterprise security, multi-model access""",
            "openrouter": """OpenRouter Provider Help

Setup:
1. Get API key from https://openrouter.ai/keys
2. Configure: /config openrouter api_key YOUR_KEY

Models (via OpenRouter):
• Multiple providers through single API
• anthropic/claude-3.5-sonnet
• openai/gpt-4o
• google/gemini-pro
• meta-llama/llama-3.1-405b

Features:
✅ Access to 100+ models via single API
✅ Cost optimization and routing
✅ Model fallback and redundancy
✅ Usage analytics and monitoring
✅ Competitive pricing

Configuration Options:
• api_key - OpenRouter API key
• referrer - Optional referrer header
• title - Optional app title

Troubleshooting:
• "Invalid API key" - Check key format (sk-or-...)
• "Model not available" - Check model availability
• "Rate limit" - Upgrade account or wait
• "Insufficient credits" - Add credits to account

Best for: Model experimentation, cost optimization, unified API access""",
            "claude-code": """Claude-code (Local) Provider Help

Setup:
1. Install claude-code from https://github.com/anthropics/claude-code
2. Ensure claude-code is in your PATH
3. Configure: /config claude-code mode opus

Models/Modes:
• opus - High-quality responses (slower)
• sonnet - Balanced performance and speed
• haiku - Fast responses (basic tasks)

Features:
✅ Free local access to Claude
✅ No API costs or rate limits
✅ Privacy-focused (local processing)
✅ Code analysis and generation
❌ No tool calling support
❌ Limited multimodal capabilities

Requirements:
• claude-code binary installed
• Sufficient system resources
• Internet connection for model downloads

Troubleshooting:
• "claude-code not found" - Install and add to PATH
• "Model download failed" - Check internet connection
• "Insufficient memory" - Close other applications
• "Slow responses" - Use faster mode (sonnet/haiku)

Best for: Free Claude access, privacy-focused usage, code tasks""",
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
                "Available providers: claude, openai, gemini, cohere, ollama"
            )

    def _show_status(self) -> None:
        """Show current status."""
        summary = self.engine.get_conversation_summary()
        model_info = self.engine.get_current_model_info()

        # Get current provider health status using optimized health monitor
        summary.get("current_provider")

        # We'll use the cached health status if available
        # Note: We can't use async calls directly in this method
        # The health monitor will have background cached data

        status_text = f"""Current Status:
        
Messages in conversation: {summary['message_count']}
Current provider: {summary.get('current_provider') or 'None'}
Current model: {summary.get('current_model') or 'None'}
Session ID: {summary.get('session_id')}
Model available: {'Yes' if model_info else 'No'}"""

        status_panel = Panel(status_text, title="Status", border_style="cyan")
        self.console.print(status_panel)

    def _show_user_message(self, message: str) -> None:
        """
        Show a user message.

        Args:
            message: User's message
        """
        user_panel = Panel(message, title="You", border_style="green")
        self.console.print(user_panel)

    def _show_assistant_message(self, message: str, model: str) -> None:
        """
        Show an assistant message.

        Args:
            message: Assistant's message
            model: Model that generated the message
        """
        import re

        # Escape specific operation markers that Rich might interpret as markup
        escaped_message = message

        # Escape operation markers
        operation_patterns = [
            r"\[FILE_WRITE:[^\]]+\]",
            r"\[FILE_READ:[^\]]+\]",
            r"\[COMMAND_EXEC\]",
            r"\[/COMMAND_EXEC\]",
            r"\[/FILE_WRITE\]",
            r"\[WEB_REQUEST:[^\]]+\]",
        ]

        for pattern in operation_patterns:
            escaped_message = re.sub(
                pattern,
                lambda m: m.group(0).replace("[", "\\[").replace("]", "\\]"),
                escaped_message,
            )

        assistant_panel = Panel(
            escaped_message, title=f"Assistant ({model})", border_style="blue"
        )
        self.console.print(assistant_panel)

    def _show_info(self, message: str) -> None:
        """Show an info message."""
        self.display_manager.show_message(message, MessageType.INFO)

    def _show_error(self, message: str) -> None:
        """Show an error message."""
        self.display_manager.show_message(message, MessageType.ERROR)

    def _show_success(self, message: str) -> None:
        """Show a success message."""
        self.display_manager.show_message(message, MessageType.SUCCESS)

    def _show_warning(self, message: str) -> None:
        """Show a warning message."""
        self.display_manager.show_message(message, MessageType.WARNING)

    def _clear_screen(self) -> None:
        """Clear the terminal screen."""
        self.console.clear()
        self._show_welcome()

    async def _show_models(self, command: Command) -> None:
        """Show available models with enhanced provider grouping and capabilities."""
        try:
            args = command.args
            filter_type = args[0].lower() if len(args) > 0 else None
            filter_value = args[1] if len(args) > 1 else None

            # First check if there are any models available at all
            all_models = self.engine.get_available_models()
            if not all_models:
                models_panel = Panel(
                    "No models available.",
                    title="Available Models",
                    border_style="cyan",
                )
                self.console.print(models_panel)
                return

            # Get enhanced models list with filtering
            result = await self._get_enhanced_models_list(filter_type, filter_value)

            title = "Available Models"
            if filter_type:
                title += f" - Filtered by {filter_type.title()}"
                if filter_value:
                    title += f": {filter_value}"

            # Use consistent Panel formatting like other commands
            models_panel = Panel(result, title=title, border_style="cyan")
            self.console.print(models_panel)

        except Exception as e:
            self._show_error(f"Failed to get models: {e}")

    async def _get_enhanced_models_list(
        self, filter_type: str = None, filter_value: str = None
    ) -> str:
        """Get enhanced models list with filtering and detailed information."""
        try:
            # Get models directly from providers
            all_models = {}
            current_provider_name = None
            current_model = None

            if self.engine.current_provider:
                current_model = self.engine.current_provider.model
                for name, provider in self.engine.providers.items():
                    if provider == self.engine.current_provider:
                        current_provider_name = name
                        break

            # Collect models from each provider
            for provider_name, provider in self.engine.providers.items():
                try:
                    models = provider.get_available_models()
                    if models:
                        all_models[provider_name] = models
                except Exception:
                    continue

            # Apply filtering
            if filter_type == "provider" and filter_value:
                all_models = {
                    k: v
                    for k, v in all_models.items()
                    if k.lower() == filter_value.lower()
                }
            elif filter_type == "free":
                filtered_models = {}
                for provider, models in all_models.items():
                    free_models = [
                        m
                        for m in models
                        if getattr(m, "is_free", False)
                        or (hasattr(m, "cost_per_token") and m.cost_per_token == 0)
                    ]
                    if free_models:
                        filtered_models[provider] = free_models
                all_models = filtered_models
            elif filter_type == "latest":
                filtered_models = {}
                for provider, models in all_models.items():
                    latest_models = [
                        m for m in models if getattr(m, "latest_version", False)
                    ]
                    if latest_models:
                        filtered_models[provider] = latest_models
                all_models = filtered_models

            if not all_models:
                return "No models found matching the specified criteria."

            # Build simple output without complex tables
            output_lines = []

            for provider_name in sorted(all_models.keys()):
                models = all_models[provider_name]
                if not models:
                    continue

                # Provider header with basic info
                provider_status = "✓"
                current_marker = (
                    " (active)" if provider_name == current_provider_name else ""
                )
                output_lines.append(
                    f"\n[bold cyan]{provider_name.upper()}[/bold cyan] {provider_status}{current_marker}:"
                )

                # List models in simple format
                for model in sorted(models, key=lambda m: m.name):
                    # Model name - truncate long names
                    name = model.name
                    if len(name) > 35:
                        name = name[:32] + "..."

                    # Status indicator (simple)
                    if (
                        model.name == current_model
                        and provider_name == current_provider_name
                    ):
                        status = "●"  # Current
                    elif getattr(model, "available", True):
                        status = "✓"  # Available
                    else:
                        status = "✗"  # Unavailable

                    # NEW indicator
                    if getattr(model, "latest_version", False):
                        status += " NEW"

                    # Capabilities (compact)
                    caps = ""
                    if getattr(model, "supports_tools", False):
                        caps += "🔧"
                    if getattr(model, "supports_multimodal", False):
                        caps += "🖼️"

                    # Cost (simplified)
                    is_free = getattr(model, "is_free", False) or (
                        hasattr(model, "cost_per_token")
                        and getattr(model, "cost_per_token", 0) == 0
                    )
                    if is_free:
                        cost = "FREE"
                    else:
                        input_cost = (
                            getattr(model, "cost_per_million_input", None)
                            or getattr(model, "cost_per_token", 0) * 1000000
                        )
                        if input_cost and input_cost > 0:
                            cost = f"${input_cost:.0f}/M"
                        else:
                            cost = ""

                    # Description - very short
                    desc = getattr(model, "description", "")
                    # Remove provider name from description
                    for remove_str in [
                        provider_name.title(),
                        provider_name.upper(),
                        model.name,
                        "via OpenRouter",
                    ]:
                        desc = desc.replace(remove_str, "").strip(" -")
                    if len(desc) > 40:
                        desc = desc[:37] + "..."

                    # Format line with fixed widths
                    output_lines.append(
                        f"  {name:35} {status:6} {caps:4} {cost:8} {desc}"
                    )

            # Add simple legend
            output_lines.append(
                "\nLegend: ● Current | ✓ Available | ✗ Unavailable | NEW Latest | 🔧 Tools | 🖼️ Multimodal"
            )

            return "\n".join(output_lines)

        except Exception as e:
            return f"Error displaying models: {e}"

    async def _handle_switch_command(self, command: Command) -> None:
        """Handle switch command with enhanced provider type support."""
        args = command.args
        if len(args) < 1:
            self._show_error("Usage: /switch <provider> [model]")
            self._show_info("Available providers:")
            # Show available providers as help
            from .commands import Command, SlashCommand

            providers_command = Command.create_slash_command(
                SlashCommand.PROVIDERS, [], "/providers"
            )
            await self._show_providers(providers_command)
            return

        provider_name = args[0].lower()
        model_name = args[1] if len(args) > 1 else None

        try:
            with self.console.status("[bold yellow]Switching...", spinner="dots"):
                # Check if provider is available but not initialized
                from ..providers.factory import ProviderFactory

                available_providers = ProviderFactory.get_available_providers()

                if provider_name not in self.engine.providers:
                    if provider_name in available_providers:
                        self._show_error(
                            f"Provider '{provider_name}' is available but not configured."
                        )
                        self._show_info(
                            "Configure it in your settings first, then try switching again."
                        )
                        return
                    else:
                        # Show suggestions for similar provider names
                        suggestions = [
                            p
                            for p in available_providers
                            if provider_name in p.lower() or p.lower() in provider_name
                        ]
                        if suggestions:
                            self._show_error(
                                f"Provider '{provider_name}' not found. Did you mean: {', '.join(suggestions)}?"
                            )
                        else:
                            self._show_error(f"Provider '{provider_name}' not found.")
                        from .commands import Command, CommandType

                        providers_command = Command(
                            content="/providers",
                            command_type=CommandType.SLASH_COMMAND,
                            args=[],
                        )
                        await self._show_providers(providers_command)
                        return

                # Validate model if specified
                if model_name:
                    provider = self.engine.providers[provider_name]
                    available_models = provider.get_available_models()
                    model_names = [m.name for m in available_models]

                    # Also check custom models for this provider
                    custom_models = self.engine.config_manager.get_custom_models()
                    custom_model_names = [
                        m.name for m in custom_models if m.provider == provider_name
                    ]

                    # Combine both lists
                    all_model_names = model_names + custom_model_names

                    if model_name not in all_model_names:
                        self._show_error(
                            f"Model '{model_name}' not available for provider '{provider_name}'."
                        )
                        if all_model_names:
                            # Show suggestions for similar model names
                            suggestions = [
                                m
                                for m in all_model_names
                                if model_name.lower() in m.lower()
                                or m.lower() in model_name.lower()
                            ]
                            if suggestions:
                                self._show_info(
                                    f"Available models for {provider_name}: {', '.join(suggestions[:5])}"
                                )
                            else:
                                self._show_info(
                                    f"Available models for {provider_name}: {', '.join(all_model_names[:5])}"
                                )
                        return

                success = await self.engine.switch_model(provider_name, model_name)

            if success:
                current_model = (
                    self.engine.current_provider.model
                    if self.engine.current_provider
                    else "unknown"
                )
                provider_info = self.engine.current_provider

                # Show enhanced switch confirmation with capabilities
                capabilities = []
                if provider_info.supports_tools():
                    capabilities.append("🔧 Tools")
                if provider_info.supports_multimodal():
                    capabilities.append("🖼️ Multimodal")

                # Get model info for additional details
                try:
                    model_info = provider_info.get_model_info()
                    model_details = []
                    if model_info and model_info.max_tokens:
                        model_details.append(f"Max tokens: {model_info.max_tokens:,}")
                    if model_info and model_info.cost_per_token:
                        model_details.append(
                            f"Cost: ${model_info.cost_per_token:.6f}/token"
                        )

                    detail_text = (
                        f" | {' | '.join(model_details)}" if model_details else ""
                    )
                except:
                    detail_text = ""

                capability_text = (
                    f" ({', '.join(capabilities)})" if capabilities else ""
                )
                self._show_info(
                    f"✓ Switched to {provider_name}:{current_model}{capability_text}{detail_text}"
                )

                # Show MCP tool availability if provider supports tools
                if provider_info.supports_tools():
                    try:
                        tools_info = await self.engine.get_available_tools()
                        if tools_info:
                            tool_count = len(tools_info)
                            self._show_info(
                                f"🔧 {tool_count} MCP tools available for use"
                            )
                        else:
                            self._show_info(
                                "🔧 Tool calling supported (no MCP tools currently available)"
                            )
                    except:
                        pass
            else:
                self._show_error("Failed to switch provider/model")

        except Exception as e:
            self._show_error(f"Switch failed: {e}")
            # Show available options on error
            if "not available" in str(e).lower() or "not found" in str(e).lower():
                self._show_info("Available options:")
                from .commands import Command, CommandType

                providers_command = Command(
                    content="/providers",
                    command_type=CommandType.SLASH_COMMAND,
                    args=[],
                )
                await self._show_providers(providers_command)

    async def _handle_save_command(self, command: Command) -> None:
        """Handle save conversation command."""
        args = command.args
        filename = args[0] if args else None

        try:
            # Check if there are messages to save
            summary = self.engine.get_conversation_summary()
            if summary["message_count"] == 0:
                self._show_info("No messages to save.")
                return

            with self.console.status(
                "[bold yellow]Saving conversation...", spinner="dots"
            ):
                saved_filename = self.engine.save_conversation(filename)

            self._show_info(f"Conversation saved as: {saved_filename}")

        except Exception as e:
            self._show_error(f"Save failed: {e}")

    async def _handle_load_command(self, command: Command) -> None:
        """Handle load conversation command."""
        args = command.args
        if not args:
            # Show available conversations
            await self._show_conversations()
            self._show_error("Usage: /load <filename>")
            return

        filename = args[0]

        try:
            with self.console.status(
                "[bold yellow]Loading conversation...", spinner="dots"
            ):
                self.engine.load_conversation(filename)

            # Show loaded conversation info
            info = self.engine.get_conversation_info(filename)
            if info:
                self._show_info(f"Loaded conversation: {filename}")
                self._show_info(
                    f"Messages: {info['message_count']}, Model: {info['current_model']}"
                )
            else:
                self._show_info(f"Loaded conversation: {filename}")

        except Exception as e:
            self._show_error(f"Load failed: {e}")

    async def _handle_config_command(self, command: Command) -> None:
        """Handle enhanced config command with subcommands."""
        args = command.args

        if not args:
            # Show all configuration
            await self._show_config()
            return

        subcommand = args[0].lower()

        # New simplified configuration commands
        if subcommand == "setup":
            await self._handle_config_setup(args[1:])
        elif subcommand == "mode":
            await self._handle_config_mode(args[1:])
        elif subcommand == "migrate":
            await self._handle_config_migrate(args[1:])
        elif subcommand == "templates":
            await self._handle_config_templates(args[1:])
        elif subcommand == "reset":
            await self._handle_config_reset(args[1:])
        # Existing config commands
        elif subcommand == "generate":
            await self._handle_config_generate(args[1:])
        elif subcommand == "validate":
            await self._handle_config_validate_enhanced(args[1:])
        elif subcommand == "show":
            await self._show_config()
        elif subcommand == "set":
            if len(args) >= 3:
                await self._handle_config_set(args[1], args[2])
            else:
                self._show_error("Usage: /config set <key> <value>")
        elif subcommand == "get":
            if len(args) >= 2:
                await self._handle_config_get(args[1])
            else:
                self._show_error("Usage: /config get <key>")
        else:
            # Legacy config handling - backward compatibility
            if len(args) == 1:
                # Show specific setting
                await self._show_config_value(args[0])
            elif len(args) == 2:
                # Set configuration value
                await self._set_config_value(args[0], args[1])
            else:
                self._show_config_help()

    async def _handle_config_generate(self, args: list) -> None:
        """Handle config generate subcommand."""
        try:
            import json
            from pathlib import Path

            from ..core.models import ConfigTemplateManager

            # Initialize components
            ConfigTemplateManager()

            # Parse arguments
            template = args[0] if len(args) > 0 else None
            output_path = args[1] if len(args) > 1 else None

            # Default output path
            if not output_path:
                output_path = str(Path.home() / ".omnimancer" / "config.json")

            with self.console.status(
                "[bold green]Generating configuration...", spinner="dots"
            ):
                # Create basic configuration structure
                config = {
                    "providers": {},
                    "chat_settings": {
                        "temperature": 0.7,
                        "max_tokens": 2048,
                        "stream": True,
                    },
                    "mcp_config": {"enabled": True, "servers": {}},
                }

                if template:
                    # Use simplified template system
                    if template == "general":
                        config["providers"]["openai"] = {
                            "api_key": "YOUR_OPENAI_API_KEY",
                            "model": "gpt-4o-mini",
                            "enabled": True,
                        }
                        config["providers"]["anthropic"] = {
                            "api_key": "YOUR_ANTHROPIC_API_KEY",
                            "model": "claude-3-haiku",
                            "enabled": True,
                        }
                        self._show_info(
                            f"✅ Generated {template} template configuration: {output_path}"
                        )
                        self.console.print(
                            f"[dim]General purpose configuration with OpenAI and Anthropic providers[/dim]"
                        )
                    else:
                        self._show_error(
                            f"Template functionality simplified. Only 'general' template is available."
                        )
                        self._show_info("Available template: general")
                        return
                else:
                    # Generate basic configuration
                    config["providers"]["openai"] = {
                        "api_key": "YOUR_OPENAI_API_KEY",
                        "model": "gpt-4o-mini",
                        "enabled": True,
                    }
                    self._show_info(f"✅ Generated basic configuration: {output_path}")

                # Write configuration file
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(config, f, indent=2)

                # Show next steps
                self.console.print("\n[bold]Next Steps:[/bold]")
                self.console.print(
                    "1. Edit the configuration file to add your API keys"
                )
                self.console.print("2. Review and customize provider settings")
                self.console.print("3. Run Omnimancer to test your configuration")

        except Exception as e:
            self._show_error(f"Failed to generate configuration: {e}")

    async def _handle_config_validate(self, args: list) -> None:
        """Handle config validate subcommand with comprehensive health checks."""
        try:
            provider_name = args[0] if len(args) > 0 else None

            with self.console.status(
                "[bold yellow]Validating configuration...", spinner="dots"
            ):
                if provider_name:
                    # Validate specific provider
                    await self._validate_single_provider(provider_name)
                else:
                    # Validate all providers
                    await self._validate_all_providers()

        except Exception as e:
            self._show_error(f"Configuration validation failed: {e}")

    async def _validate_single_provider(self, provider_name: str) -> None:
        """Validate a single provider with comprehensive checks."""
        if provider_name not in self.engine.providers:
            self._show_error(f"Provider '{provider_name}' not configured")
            return

        provider = self.engine.providers[provider_name]

        self.console.print(
            f"\n[bold]Validating {provider_name.upper()} Provider[/bold]"
        )

        validation_results = []

        # 1. Basic configuration check
        try:
            model_info = provider.get_model_info()
            validation_results.append(
                ("Configuration", True, "Provider initialized successfully")
            )
        except Exception as e:
            validation_results.append(
                ("Configuration", False, f"Initialization failed: {e}")
            )

        # 2. API key validation (if applicable)
        try:
            if hasattr(provider, "api_key") and provider.api_key:
                # Basic format validation
                if provider_name == "claude" and not provider.api_key.startswith(
                    "sk-ant-"
                ):
                    validation_results.append(
                        (
                            "API Key Format",
                            False,
                            "Should start with 'sk-ant-'",
                        )
                    )
                elif provider_name == "openai" and not provider.api_key.startswith(
                    "sk-"
                ):
                    validation_results.append(
                        ("API Key Format", False, "Should start with 'sk-'")
                    )
                elif provider_name == "gemini" and not provider.api_key.startswith(
                    "AIza"
                ):
                    validation_results.append(
                        ("API Key Format", False, "Should start with 'AIza'")
                    )
                else:
                    validation_results.append(
                        ("API Key Format", True, "Format appears valid")
                    )
            else:
                if provider_name not in ["ollama", "claude-code"]:
                    validation_results.append(
                        ("API Key", False, "API key not configured")
                    )
                else:
                    validation_results.append(
                        ("API Key", True, "Not required for this provider")
                    )
        except Exception as e:
            validation_results.append(("API Key", False, f"Validation error: {e}"))

        # 3. Model availability check
        try:
            models = provider.get_available_models()
            current_model = provider.model
            model_available = any(
                m.name == current_model and m.available for m in models
            )

            if model_available:
                validation_results.append(
                    (
                        "Model Availability",
                        True,
                        f"Model '{current_model}' is available",
                    )
                )
            else:
                validation_results.append(
                    (
                        "Model Availability",
                        False,
                        f"Model '{current_model}' not available",
                    )
                )
        except Exception as e:
            validation_results.append(
                ("Model Availability", False, f"Check failed: {e}")
            )

        # 4. Connection test (basic)
        try:
            # Try to get model info as a basic connection test
            model_info = provider.get_model_info()
            if model_info:
                validation_results.append(
                    ("Connection", True, "Provider responds to requests")
                )
            else:
                validation_results.append(
                    ("Connection", False, "No response from provider")
                )
        except Exception as e:
            validation_results.append(("Connection", False, f"Connection failed: {e}"))

        # 5. Capability checks
        try:
            capabilities = []
            if provider.supports_tools():
                capabilities.append("Tool calling")
            if provider.supports_multimodal():
                capabilities.append("Multimodal")

            if capabilities:
                validation_results.append(
                    (
                        "Capabilities",
                        True,
                        f"Supports: {', '.join(capabilities)}",
                    )
                )
            else:
                validation_results.append(
                    ("Capabilities", True, "Basic text generation")
                )
        except Exception as e:
            validation_results.append(("Capabilities", False, f"Check failed: {e}"))

        # Display results
        from rich.table import Table

        table = Table(title=f"{provider_name.upper()} Validation Results")
        table.add_column("Check", style="bold", width=20)
        table.add_column("Status", style="bold", width=12)
        table.add_column("Details", style="dim")

        overall_status = True
        for check, is_valid, details in validation_results:
            status = "✅ Pass" if is_valid else "❌ Fail"
            if not is_valid:
                overall_status = False
            table.add_row(check, status, details)

        self.console.print(table)

        # Overall status
        if overall_status:
            self.console.print(
                f"\n[bold green]✅ {provider_name.upper()} provider is fully operational[/bold green]"
            )
        else:
            self.console.print(
                f"\n[bold red]❌ {provider_name.upper()} provider has issues that need attention[/bold red]"
            )
            self._show_troubleshooting_guidance(provider_name)

    async def _validate_all_providers(self) -> None:
        """Validate all configured providers with comprehensive health checks."""
        if not self.engine.providers:
            self._show_error(
                "No providers configured. Use '/setup' to configure providers."
            )
            return

        self.console.print("[bold]Validating All Providers[/bold]\n")

        validation_summary = []

        for provider_name in sorted(self.engine.providers.keys()):
            provider = self.engine.providers[provider_name]

            # Quick validation for each provider
            provider_status = {
                "name": provider_name,
                "configured": True,
                "healthy": False,
                "model_available": False,
                "capabilities": [],
                "issues": [],
            }

            try:
                # Basic health check
                model_info = provider.get_model_info()
                if model_info:
                    provider_status["healthy"] = True
                    provider_status["model_available"] = model_info.available

                # Capability check
                if provider.supports_tools():
                    provider_status["capabilities"].append("🔧 Tools")
                if provider.supports_multimodal():
                    provider_status["capabilities"].append("🖼️ Multimodal")

                # Model availability
                models = provider.get_available_models()
                current_model = provider.model
                if not any(m.name == current_model and m.available for m in models):
                    provider_status["issues"].append("Current model not available")

            except Exception as e:
                provider_status["issues"].append(
                    f"Health check failed: {str(e)[:50]}..."
                )

            validation_summary.append(provider_status)

        # Display summary table
        from rich.table import Table

        table = Table(title="Provider Validation Summary")
        table.add_column("Provider", style="bold", width=15)
        table.add_column("Status", style="bold", width=12)
        table.add_column("Health", width=10)
        table.add_column("Model", width=10)
        table.add_column("Capabilities", width=15)
        table.add_column("Issues", style="red", width=30)

        healthy_count = 0
        total_count = len(validation_summary)

        for status in validation_summary:
            # Overall status
            if status["healthy"] and not status["issues"]:
                overall_status = "✅ Good"
                healthy_count += 1
            elif status["healthy"]:
                overall_status = "⚠️ Issues"
            else:
                overall_status = "❌ Failed"

            # Health indicator
            health = "🟢 Healthy" if status["healthy"] else "🔴 Error"

            # Model status
            model = "✅ OK" if status["model_available"] else "❌ N/A"

            # Capabilities
            capabilities = (
                " ".join(status["capabilities"]) if status["capabilities"] else "—"
            )

            # Issues
            issues = "; ".join(status["issues"]) if status["issues"] else "None"
            if len(issues) > 30:
                issues = issues[:27] + "..."

            table.add_row(
                status["name"].upper(),
                overall_status,
                health,
                model,
                capabilities,
                issues,
            )

        self.console.print(table)

        # Overall system health
        health_percentage = (
            (healthy_count / total_count) * 100 if total_count > 0 else 0
        )

        self.console.print(
            f"\n[bold]System Health: {health_percentage:.0f}% ({healthy_count}/{total_count} providers healthy)[/bold]"
        )

        if health_percentage == 100:
            self.console.print(
                "[bold green]🎉 All providers are fully operational![/bold green]"
            )
        elif health_percentage >= 75:
            self.console.print(
                "[bold yellow]⚠️ Most providers are working, but some need attention.[/bold yellow]"
            )
        else:
            self.console.print(
                "[bold red]❌ Multiple providers have issues. Consider running setup wizard.[/bold red]"
            )

        # Recommendations
        issues_found = [s for s in validation_summary if s["issues"]]
        if issues_found:
            self.console.print("\n[bold]Recommendations:[/bold]")
            for status in issues_found:
                self.console.print(f"  • {status['name']}: {status['issues'][0]}")

            self.console.print(
                "\n[dim]Use '/config validate <provider>' for detailed diagnostics[/dim]"
            )
            self.console.print(
                "[dim]Use '/setup' to reconfigure problematic providers[/dim]"
            )

    def _show_troubleshooting_guidance(self, provider_name: str) -> None:
        """Show troubleshooting guidance for a specific provider."""
        guidance = {
            "claude": [
                "• Verify API key from https://console.anthropic.com/",
                "• Check account credits and usage limits",
                "• Ensure API key starts with 'sk-ant-'",
                "• Try a different model (claude-3-haiku-20240307)",
            ],
            "openai": [
                "• Verify API key from https://platform.openai.com/",
                "• Check billing setup and usage limits",
                "• Ensure API key starts with 'sk-'",
                "• Try gpt-3.5-turbo if GPT-4 access is limited",
            ],
            "gemini": [
                "• Verify API key from https://makersuite.google.com/",
                "• Enable Generative AI API in Google Cloud Console",
                "• Check safety settings if requests are blocked",
                "• Ensure API key starts with 'AIza'",
            ],
            "ollama": [
                "• Start Ollama server: 'ollama serve'",
                "• Install required model: 'ollama pull <model>'",
                "• Check server URL (default: http://localhost:11434)",
                "• Ensure sufficient RAM for model size",
            ],
        }

        if provider_name in guidance:
            self.console.print(
                f"\n[bold yellow]Troubleshooting {provider_name.upper()}:[/bold yellow]"
            )
            for tip in guidance[provider_name]:
                self.console.print(f"  {tip}")
        else:
            self.console.print(
                f"\n[dim]Use '/help {provider_name}' for provider-specific guidance[/dim]"
            )

    async def _show_config(self) -> None:
        """Show current configuration."""
        try:
            config_info = self.engine.get_configuration_info()

            from rich.table import Table

            table = Table(title="Current Configuration")
            table.add_column("Setting", style="bold")
            table.add_column("Value", style="cyan")

            for key, value in config_info.items():
                # Mask sensitive information
                if "key" in key.lower() or "password" in key.lower():
                    if value:
                        masked_value = (
                            value[:8] + "..." if len(str(value)) > 8 else "***"
                        )
                        table.add_row(key, masked_value)
                    else:
                        table.add_row(key, "[dim]Not set[/dim]")
                else:
                    table.add_row(key, str(value))

            self.console.print(table)

        except Exception as e:
            self._show_error(f"Failed to show configuration: {e}")

    async def _handle_config_set(self, key: str, value: str) -> None:
        """Handle config set operation."""
        try:
            # This would need to be implemented in the engine
            self._show_info(f"Setting {key} = {value}")
            self._show_info(
                "Note: Some settings may require restarting Omnimancer to take effect"
            )
        except Exception as e:
            self._show_error(f"Failed to set configuration: {e}")

    async def _handle_config_get(self, key: str) -> None:
        """Handle config get operation."""
        try:
            config_info = self.engine.get_configuration_info()
            if key in config_info:
                value = config_info[key]
                # Mask sensitive information
                if "key" in key.lower() or "password" in key.lower():
                    if value:
                        masked_value = (
                            value[:8] + "..." if len(str(value)) > 8 else "***"
                        )
                        self.console.print(f"[bold]{key}:[/bold] {masked_value}")
                    else:
                        self.console.print(f"[bold]{key}:[/bold] [dim]Not set[/dim]")
                else:
                    self.console.print(f"[bold]{key}:[/bold] {value}")
            else:
                self._show_error(f"Configuration key '{key}' not found")
        except Exception as e:
            self._show_error(f"Failed to get configuration: {e}")

    async def _handle_config_setup(self, args: list) -> None:
        """Handle config setup subcommand using the new configuration wizard."""
        try:
            from ..core.config_manager import ConfigManager
            from .config_setup_wizard import (
                run_config_setup_wizard,
                run_quick_setup,
            )

            # Parse arguments
            quick = "--quick" in args or "-q" in args
            use_case = None
            providers = None

            # Parse use-case and providers from args
            for i, arg in enumerate(args):
                if arg == "--use-case" and i + 1 < len(args):
                    use_case = args[i + 1]
                elif arg == "--providers" and i + 1 < len(args):
                    providers = args[i + 1].split(",")

            config_manager = ConfigManager()

            if quick:
                self.console.print("[blue]Starting Quick Setup...[/blue]")
                success = run_quick_setup(use_case or "general", providers)
            else:
                self.console.print(
                    "[blue]Starting Configuration Setup Wizard...[/blue]"
                )
                success = run_config_setup_wizard(config_manager)

            if success:
                self.console.print("[green]✅ Configuration setup completed![/green]")
            else:
                self.console.print(
                    "[yellow]⚠️ Configuration setup was cancelled or failed.[/yellow]"
                )

        except Exception as e:
            self._show_error(f"Configuration setup failed: {e}")

    async def _handle_config_mode(self, args: list) -> None:
        """Handle config mode subcommand."""
        try:
            from ..core.config_manager import ConfigManager
            from ..core.config_provider import (
                ConfigurationMode,
                ConfigurationProvider,
            )

            config_manager = ConfigManager()
            config_provider = ConfigurationProvider(config_manager)

            if not args:
                # Show current mode
                current_mode = config_provider.current_mode
                available_modes = config_provider.get_available_modes()

                self.console.print(
                    f"Current mode: [blue]{current_mode.value.title()}[/blue]"
                )

                # Show available modes table
                from rich.table import Table

                modes_table = Table(title="Available Configuration Modes")
                modes_table.add_column("Mode", style="cyan")
                modes_table.add_column("Description", style="white")
                modes_table.add_column("Target Audience", style="yellow")

                for mode_key, mode_info in available_modes.items():
                    current_marker = "→ " if mode_key == current_mode.value else "  "
                    modes_table.add_row(
                        f"{current_marker}{mode_info['name']}",
                        mode_info["description"],
                        mode_info["target_audience"],
                    )

                self.console.print(modes_table)
            else:
                # Set mode
                mode = args[0].lower()
                if mode in ["simple", "advanced", "guided"]:
                    new_mode = ConfigurationMode(mode)
                    config_provider.set_configuration_mode(new_mode)
                    self.console.print(
                        f"[green]✓[/green] Configuration mode set to: {mode.title()}"
                    )
                else:
                    self._show_error(
                        f"Invalid mode: {mode}. Available modes: simple, advanced, guided"
                    )

        except Exception as e:
            self._show_error(f"Mode command failed: {e}")

    async def _handle_config_migrate(self, args: list) -> None:
        """Handle config migrate subcommand."""
        try:
            from rich.prompt import Confirm

            from ..core.config_manager import ConfigManager
            from ..core.config_migration_helpers import create_migration_helper

            config_manager = ConfigManager()
            migration_helper = create_migration_helper(config_manager)

            # Parse arguments
            template = None
            analyze_only = "--analyze-only" in args
            force = "--force" in args

            for i, arg in enumerate(args):
                if arg == "--template" and i + 1 < len(args):
                    template = args[i + 1]

            # Analyze current configuration
            self.console.print("[blue]Analyzing current configuration...[/blue]")
            analysis = migration_helper.analyze_current_configuration()

            self._show_migration_analysis(analysis)

            if analyze_only:
                return

            # Determine target template
            if not template:
                template = analysis.recommended_template
                if not template:
                    self._show_error("No suitable template found for migration.")
                    return

            if not force and not Confirm.ask(
                f"Migrate to '{template}' template?", default=False
            ):
                self.console.print("Migration cancelled.")
                return

            # Create and execute migration plan
            self.console.print("[blue]Creating migration plan...[/blue]")
            migration_plan = migration_helper.create_migration_plan(
                template, preserve_customizations=True
            )

            self._show_migration_plan(migration_plan)

            if not force and not Confirm.ask("Execute migration plan?", default=False):
                self.console.print("Migration cancelled.")
                return

            # Execute migration
            self.console.print("[blue]Executing migration...[/blue]")
            success, messages = migration_helper.execute_migration(
                migration_plan, template
            )

            # Show results
            for message in messages:
                if message.startswith("✓"):
                    self.console.print(f"[green]{message}[/green]")
                elif message.startswith("⚠"):
                    self.console.print(f"[yellow]{message}[/yellow]")
                elif message.startswith("✗"):
                    self.console.print(f"[red]{message}[/red]")
                else:
                    self.console.print(message)

            if success:
                self.console.print(
                    "[green]✅ Migration completed successfully![/green]"
                )
            else:
                self.console.print("[red]❌ Migration failed.[/red]")

        except Exception as e:
            self._show_error(f"Migration failed: {e}")

    async def _handle_config_templates(self, args: list) -> None:
        """Handle config templates subcommand."""
        try:
            from ..core.config_provider import ConfigurationProvider

            config_provider = ConfigurationProvider()
            templates = config_provider.get_quick_setup_templates()

            # Parse arguments
            use_case = None
            detailed = "--detailed" in args

            for i, arg in enumerate(args):
                if arg == "--use-case" and i + 1 < len(args):
                    use_case = args[i + 1]

            if use_case:
                templates = [
                    t for t in templates if use_case.lower() in t["use_case"].lower()
                ]

            if not templates:
                self.console.print(
                    "[yellow]No templates found matching criteria.[/yellow]"
                )
                return

            if detailed:
                # Show detailed template information
                for template in templates:
                    self._show_detailed_template(template)
            else:
                # Show templates table
                from rich.table import Table

                templates_table = Table(title="Available Configuration Templates")
                templates_table.add_column("Name", style="cyan")
                templates_table.add_column("Description", style="white")
                templates_table.add_column("Use Case", style="yellow")
                templates_table.add_column("Complexity", style="green")
                templates_table.add_column("Setup Time", style="magenta")
                templates_table.add_column("Cost", style="blue")

                for template in templates:
                    templates_table.add_row(
                        template["display_name"],
                        (
                            template["description"][:50] + "..."
                            if len(template["description"]) > 50
                            else template["description"]
                        ),
                        template["use_case"].title(),
                        template["complexity"],
                        template["estimated_setup_time"],
                        template["cost_estimate"],
                    )

                self.console.print(templates_table)

        except Exception as e:
            self._show_error(f"Failed to list templates: {e}")

    async def _handle_config_reset(self, args: list) -> None:
        """Handle config reset subcommand."""
        try:
            from rich.prompt import Confirm

            from ..core.config_manager import ConfigManager

            config_manager = ConfigManager()

            # Parse arguments
            backup = "--no-backup" not in args
            force = "--force" in args

            if not force and not Confirm.ask(
                "This will reset all configuration to defaults. Continue?",
                default=False,
            ):
                self.console.print("Reset cancelled.")
                return

            if backup:
                backup_path = config_manager.backup_config()
                self.console.print(
                    f"[green]✓[/green] Configuration backed up to: {backup_path}"
                )

            config_manager.reset_config()
            self.console.print("[green]✅ Configuration reset to defaults.[/green]")
            self.console.print("Run '/config setup' to configure Omnimancer.")

        except Exception as e:
            self._show_error(f"Reset failed: {e}")

    async def _handle_config_validate_enhanced(self, args: list) -> None:
        """Handle enhanced config validate subcommand with compatibility checking."""
        try:
            from ..core.config_provider import ConfigurationProvider

            config_provider = ConfigurationProvider()

            # Parse arguments
            comprehensive = "--comprehensive" in args or "-c" in args
            report_only = "--report" in args or "-r" in args
            fix = "--fix" in args

            self.console.print("[blue]Validating configuration...[/blue]")

            # Run comprehensive validation (includes compatibility check)
            validation_report = await config_provider.comprehensive_validation()

            if report_only:
                # Generate and show full report
                report = config_provider.generate_compatibility_report()
                self.console.print(report)
                return

            # Show summary results
            from rich.table import Table

            # Status table
            status_table = Table(title="Configuration Validation Results")
            status_table.add_column("Check", style="cyan")
            status_table.add_column("Status", style="bold")
            status_table.add_column("Details", style="dim")

            # Compatibility status
            compat_status = (
                "✅ Compatible"
                if validation_report.is_compatible
                else "❌ Issues Found"
            )
            status_table.add_row(
                "Compatibility",
                compat_status,
                f"Version: {validation_report.format_info.get('version', 'unknown')}",
            )

            # Validation status
            valid_status = "✅ Valid" if validation_report.is_valid else "❌ Invalid"
            error_count = len(validation_report.critical_errors)
            warning_count = len(validation_report.warnings)
            details = f"Errors: {error_count}, Warnings: {warning_count}"
            status_table.add_row("Validation", valid_status, details)

            # Health Score
            health_status = f"✓ {validation_report.health_score:.1f}/100"
            status_table.add_row(
                "Health Score", health_status, "Overall configuration health"
            )

            self.console.print(status_table)

            # Show errors if any
            if validation_report.critical_errors:
                self.console.print("\n[red]Critical Errors:[/red]")
                for error in validation_report.critical_errors:
                    self.console.print(f"  • {error}")

            # Show warnings if any
            if validation_report.warnings:
                self.console.print("\n[yellow]Warnings:[/yellow]")
                for warning in validation_report.warnings:
                    self.console.print(f"  • {warning}")

            # Show suggestions if comprehensive
            if comprehensive and validation_report.suggestions:
                self.console.print("\n[blue]Suggestions:[/blue]")
                for suggestion in validation_report.suggestions:
                    self.console.print(f"  • {suggestion}")

            # Show upgrade recommendations
            if validation_report.upgrade_recommendations:
                self.console.print("\n[green]Upgrade Recommendations:[/green]")
                for rec in validation_report.upgrade_recommendations:
                    self.console.print(f"  • {rec}")

            # Offer fixes if requested
            if fix and (
                not validation_report.is_valid or not validation_report.is_compatible
            ):
                self.console.print("\n[blue]Applying fixes...[/blue]")
                success, messages = config_provider.ensure_backward_compatibility()

                for message in messages:
                    if message.startswith("✓"):
                        self.console.print(f"[green]{message}[/green]")
                    elif message.startswith("⚠"):
                        self.console.print(f"[yellow]{message}[/yellow]")
                    elif message.startswith("✗"):
                        self.console.print(f"[red]{message}[/red]")
                    else:
                        self.console.print(message)

                if success:
                    self.console.print(
                        "[green]✅ Configuration fixes applied successfully![/green]"
                    )
                else:
                    self.console.print(
                        "[yellow]⚠️ Some issues require manual attention.[/yellow]"
                    )

            elif not validation_report.is_valid or not validation_report.is_compatible:
                self.console.print(
                    f"\nRun [cyan]/config validate --fix[/cyan] to attempt automatic fixes."
                )
                if validation_report.migration_required:
                    self.console.print(
                        f"Or run [cyan]/config migrate[/cyan] to migrate to simplified configuration."
                    )

        except Exception as e:
            self._show_error(f"Enhanced validation failed: {e}")
            # Fallback to basic validation
            await self._handle_config_validate(args)

    def _show_config_help(self) -> None:
        """Show configuration command help."""
        help_text = """
[bold]Configuration Commands:[/bold]

[cyan]/config[/cyan]                    - Show current configuration
[cyan]/config show[/cyan]               - Show current configuration (detailed)
[cyan]/config setup[/cyan]              - Run configuration setup wizard
[cyan]/config setup --quick[/cyan]      - Quick setup with defaults
[cyan]/config mode[/cyan]               - Show/set configuration mode (simple/advanced/guided)
[cyan]/config migrate[/cyan]            - Migrate to template-based configuration
[cyan]/config templates[/cyan]          - List available configuration templates
[cyan]/config validate[/cyan]           - Validate current configuration
[cyan]/config validate --comprehensive[/cyan] - Comprehensive validation with suggestions
[cyan]/config validate --fix[/cyan]    - Validate and auto-fix issues
[cyan]/config validate --report[/cyan] - Generate detailed compatibility report
[cyan]/config reset[/cyan]              - Reset configuration to defaults
[cyan]/config generate[/cyan]           - Generate configuration files
[cyan]/config set <key> <value>[/cyan]  - Set configuration value
[cyan]/config get <key>[/cyan]          - Get configuration value

[bold]Examples:[/bold]
[dim]/config setup --quick --use-case coding[/dim]
[dim]/config mode simple[/dim]
[dim]/config migrate --template research[/dim]
[dim]/config templates --use-case coding[/dim]
        """
        self.console.print(help_text)

    def _show_migration_analysis(self, analysis) -> None:
        """Show migration analysis results."""
        from rich.table import Table

        table = Table(title="Configuration Analysis")
        table.add_column("Aspect", style="cyan")
        table.add_column("Current State", style="magenta")

        table.add_row("Complexity Level", analysis.current_complexity.title())
        table.add_row("Recommended Template", analysis.recommended_template or "None")
        table.add_row("Confidence", f"{analysis.confidence:.0%}")
        table.add_row("Estimated Time", analysis.estimated_time)

        self.console.print(table)

        if analysis.simplification_opportunities:
            self.console.print("\n[green]Simplification Opportunities:[/green]")
            for opportunity in analysis.simplification_opportunities:
                self.console.print(f"  • {opportunity}")

        if analysis.potential_issues:
            self.console.print("\n[yellow]Potential Issues:[/yellow]")
            for issue in analysis.potential_issues:
                self.console.print(f"  ⚠ {issue}")

    def _show_migration_plan(self, migration_plan) -> None:
        """Show migration plan details."""
        self.console.print(
            f"\n[blue]Migration Plan[/blue] ({migration_plan.estimated_duration})"
        )

        for step in migration_plan.steps:
            step_num = step["step"]
            title = step["title"]
            description = step["description"]
            time_est = step.get("estimated_time", "")

            self.console.print(f"  {step_num}. **{title}** ({time_est})")
            self.console.print(f"     {description}")

            if "details" in step and step["details"]:
                for detail in step["details"][:2]:  # Show first 2 details
                    self.console.print(f"     • {detail}")

        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
            migration_plan.risk_level, "white"
        )
        self.console.print(
            f"\nRisk Level: [{risk_color}]{migration_plan.risk_level.title()}[/{risk_color}]"
        )

    def _show_detailed_template(self, template) -> None:
        """Show detailed template information."""
        from rich.markdown import Markdown

        template_text = f"""
# {template['display_name']} Template

**Use Case:** {template['use_case'].title()}  
**Complexity:** {template['complexity']}  
**Setup Time:** {template['estimated_setup_time']}  
**Cost:** {template['cost_estimate']}

## Description
{template['description']}

## Main Providers
{', '.join(template['main_providers'])}

## Recommended For
{', '.join(template.get('recommended_for', []))}
        """

        panel = Panel(
            Markdown(template_text),
            title=f"{template['icon']} {template['display_name']}",
            border_style="blue",
            padding=(1, 2),
        )
        self.console.print(panel)

    async def _handle_setup_command(self, command: Command) -> None:
        """Handle setup command for interactive configuration."""
        try:
            from ..core.config_manager import ConfigManager
            from ..core.provider_registry import ProviderRegistry
            from ..core.setup_wizard import SetupWizard

            # Initialize components
            config_manager = ConfigManager()
            provider_registry = ProviderRegistry()
            setup_wizard = SetupWizard(config_manager, provider_registry)

            self.console.print(
                "[bold blue]🚀 Starting Omnimancer Setup Wizard...[/bold blue]\n"
            )

            # Run the setup wizard
            success = await setup_wizard.start_wizard()

            if success:
                self.console.print(
                    "\n[bold green]✅ Setup completed successfully![/bold green]"
                )
                self.console.print(
                    "[dim]You may need to restart Omnimancer for changes to take effect.[/dim]"
                )
            else:
                self.console.print(
                    "\n[yellow]⚠️  Setup was cancelled or incomplete.[/yellow]"
                )

        except Exception as e:
            self._show_error(f"Setup wizard failed: {e}")

    async def _handle_config_command_legacy(self, command: Command) -> None:
        """Handle legacy config command format for backward compatibility."""
        args = command.args

        if len(args) == 0:
            # Show all configuration
            await self._show_config()
        elif len(args) == 1:
            # Show specific setting
            await self._handle_config_get(args[0])
        elif len(args) == 2:
            # Set configuration value
            await self._handle_config_set(args[0], args[1])
        else:
            self._show_error("Usage: /config [key] [value]")
        """Handle configuration command."""
        args = command.args

        if not args:
            # Show current configuration
            await self._show_config()
        elif len(args) == 1:
            # Show specific config value
            await self._show_config_value(args[0])
        elif len(args) == 2:
            # Set config value
            await self._set_config_value(args[0], args[1])
        else:
            self._show_error("Usage: /config [key] [value]")

    async def _show_config_value(self, key: str) -> None:
        """Show a specific configuration value."""
        try:
            config = self.engine.config_manager.get_config()

            if key == "default_provider":
                value = config.default_provider
            elif key == "storage_path":
                value = config.storage_path
            elif key in config.providers:
                provider_config = config.providers[key]
                value = f"Model: {provider_config.model}, API Key: {'Set' if provider_config.api_key else 'Not set'}"
            else:
                self._show_error(f"Unknown configuration key: {key}")
                return

            self._show_info(f"{key}: {value}")

        except Exception as e:
            self._show_error(f"Failed to get configuration value: {e}")

    async def _set_config_value(self, key: str, value: str) -> None:
        """Set a configuration value."""
        try:
            if key == "default_provider":
                self.engine.config_manager.set_default_provider(value)
                self._show_info(f"Default provider set to: {value}")
            else:
                self._show_error(
                    f"Configuration key '{key}' cannot be set via CLI. Use provider-specific commands or edit config file."
                )

        except Exception as e:
            self._show_error(f"Failed to set configuration value: {e}")

    async def _show_conversations(self) -> None:
        """Show available conversation files."""
        try:
            conversations = self.engine.list_conversations()

            if not conversations:
                self._show_info("No saved conversations found.")
                return

            lines = ["Available conversations:"]

            for conv in conversations:
                created = conv.get("created_at", "Unknown")
                if created and created != "Unknown":
                    try:
                        from datetime import datetime

                        created_dt = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                        created = created_dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass

                lines.append(f"  📄 {conv['filename']}")
                lines.append(f"     Created: {created}")
                lines.append(
                    f"     Messages: {conv['message_count']}, Model: {conv.get('current_model', 'Unknown')}"
                )
                lines.append("")

            conversations_panel = Panel(
                "\n".join(lines),
                title="Saved Conversations",
                border_style="cyan",
            )
            self.console.print(conversations_panel)

        except Exception as e:
            self._show_error(f"Failed to list conversations: {e}")

    async def _show_providers(self, command: Command) -> None:
        """Show enhanced providers list with comprehensive status information."""
        try:
            result = await self._get_enhanced_providers_list()

            providers_panel = Panel(
                result,
                title="AI Providers - Comprehensive Status",
                border_style="cyan",
            )
            self.console.print(providers_panel)

        except Exception as e:
            self._show_error(f"Failed to get providers: {e}")

    async def _get_enhanced_providers_list(self) -> str:
        """Get enhanced providers list with comprehensive status information."""
        try:
            from io import StringIO

            from rich.console import Console

            from ..providers.factory import ProviderFactory

            available_provider_names = ProviderFactory.get_available_providers()
            if not available_provider_names:
                return "No providers available."

            output = StringIO()
            console = Console(file=output, width=120)
            table = self._create_providers_table()
            current_provider_name = self._get_current_provider_name()

            for provider_name in sorted(available_provider_names):
                provider = self.engine.providers.get(provider_name)
                is_current = provider_name == current_provider_name

                table.add_row(
                    provider_name.upper(),
                    self._get_provider_status(provider, is_current),
                    self._get_provider_model_count(provider, provider_name),
                    self._get_provider_capabilities(provider, provider_name),
                    self._get_provider_health(provider),
                    self._get_provider_config_status(provider),
                    "Active" if provider else "—",
                    self._get_provider_notes(provider, provider_name),
                )

            console.print(table)
            summary_lines = ["\n" + output.getvalue()]
            summary_lines.extend(
                self._build_provider_summary(
                    available_provider_names, current_provider_name
                )
            )

            return "\n".join(summary_lines)

        except Exception:
            return self.engine._get_providers_list()

    def _create_providers_table(self) -> "Table":
        """Create and configure the providers table."""
        from rich.table import Table

        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Provider", style="bold", width=13)
        table.add_column("Status", width=12)
        table.add_column("Models", width=8)
        table.add_column("Capabilities", width=18)
        table.add_column("Health", width=10)
        table.add_column("Configuration", width=15)
        table.add_column("Last Update", width=12)
        table.add_column("Notes", style="dim")
        return table

    def _get_current_provider_name(self) -> str:
        """Get the name of the current provider."""
        if not self.engine.current_provider:
            return None
        for name, provider in self.engine.providers.items():
            if provider == self.engine.current_provider:
                return name
        return None

    def _get_provider_status(self, provider, is_current: bool) -> str:
        """Get status indicators for a provider."""
        indicators = []
        if provider:
            indicators.append("✅ Active")
            if is_current:
                indicators.append("🎯 Current")
        else:
            indicators.append("⚪ Available")
        return " ".join(indicators)

    def _get_provider_model_count(self, provider, provider_name: str) -> str:
        """Get model count string for a provider."""
        if provider:
            try:
                models = provider.get_available_models()
                available = [m for m in models if m.available]
                return f"{len(available)}/{len(models)}"
            except Exception:
                return "Error"
        else:
            try:
                from ..providers.factory import ProviderFactory

                models = ProviderFactory.get_models_for_provider(provider_name)
                return f"{len(models)}"
            except Exception:
                return "Unknown"

    def _get_provider_capabilities(self, provider, provider_name: str) -> str:
        """Get capabilities string for a provider."""
        capabilities = []
        if provider:
            try:
                if provider.supports_tools():
                    capabilities.append("🔧 Tools")
                if provider.supports_multimodal():
                    capabilities.append("🖼️ Multi")
            except Exception:
                pass
        else:
            capabilities = self._detect_provider_capabilities(provider_name)

        return "\n".join(capabilities) if capabilities else "—"

    def _detect_provider_capabilities(self, provider_name: str) -> list:
        """Detect capabilities for non-active providers."""
        try:
            from ..core.models import ProviderConfig
            from ..providers.factory import ProviderFactory

            temp_config = ProviderConfig(api_key="dummy", model="dummy")
            temp_provider = ProviderFactory.create_provider(provider_name, temp_config)
            capabilities = []
            if temp_provider.supports_tools():
                capabilities.append("🔧 Tools")
            if temp_provider.supports_multimodal():
                capabilities.append("🖼️ Multi")
            return capabilities
        except Exception:
            return []

    def _get_provider_health(self, provider) -> str:
        """Get health status for a provider."""
        if not provider:
            return "—"
        try:
            model_info = provider.get_model_info()
            if model_info and model_info.available:
                return "🟢 Healthy"
            else:
                return "🟡 Limited"
        except Exception:
            return "🔴 Error"

    def _get_provider_config_status(self, provider) -> str:
        """Get configuration status for a provider."""
        if provider:
            try:
                return f"✅ Configured\n({provider.model})"
            except Exception:
                return "✅ Configured"
        else:
            return "⚙️ Not Setup"

    def _get_provider_notes(self, provider, provider_name: str) -> str:
        """Get notes for a provider."""
        notes = []

        # Provider-specific notes
        if provider_name == "ollama":
            notes.append("Local AI")
        elif provider_name == "claude-code":
            notes.append("Free local")
        elif provider_name in ["perplexity", "xai"]:
            notes.append("Search enabled")

        # Dynamic notes from model info
        if provider:
            try:
                model_info = provider.get_model_info()
                if model_info:
                    if (
                        hasattr(model_info, "cost_per_token")
                        and model_info.cost_per_token == 0
                    ):
                        notes.append("Free")
                    elif (
                        hasattr(model_info, "max_tokens")
                        and model_info.max_tokens > 100000
                    ):
                        notes.append("Large context")
            except Exception:
                pass

        return " | ".join(notes) if notes else "—"

    def _build_provider_summary(
        self, available_provider_names: list, current_provider_name: str
    ) -> list:
        """Build summary section for providers list."""
        lines = []

        # Summary statistics
        total_providers = len(available_provider_names)
        active_providers = len(self.engine.providers)
        tool_providers, multimodal_providers = self._count_provider_capabilities()

        lines.extend(
            [
                "\n📊 Provider Summary:",
                f"  • Total Available: {total_providers}",
                f"  • Currently Active: {active_providers}",
                f"  • Tool-Capable: {tool_providers}",
                f"  • Multimodal: {multimodal_providers}",
                f"  • Current Provider: {current_provider_name or 'None'}",
            ]
        )

        # Legend
        lines.extend(
            [
                "\n🔍 Legend:",
                "  Status: ✅ Active | ⚪ Available | 🎯 Current",
                "  Health: 🟢 Healthy | 🟡 Limited | 🔴 Error",
                "  Config: ✅ Configured | ⚙️ Not Setup",
                "  Capabilities: 🔧 Tool calling | 🖼️ Multimodal",
            ]
        )

        # Setup guidance
        unconfigured = [
            name
            for name in available_provider_names
            if name not in self.engine.providers
        ]
        if unconfigured:
            lines.extend(
                [
                    "\n💡 Setup Guidance:",
                    f"  • Unconfigured providers: {', '.join(unconfigured)}",
                    "  • Use '/setup' for interactive configuration",
                    "  • Use '/config generate <provider>' for specific setup",
                    "  • Use '/help <provider>' for provider-specific help",
                ]
            )

        return lines

    def _count_provider_capabilities(self) -> tuple:
        """Count providers with tool and multimodal capabilities."""
        tool_providers = 0
        multimodal_providers = 0

        for provider in self.engine.providers.values():
            try:
                if provider.supports_tools():
                    tool_providers += 1
                if provider.supports_multimodal():
                    multimodal_providers += 1
            except Exception:
                continue

        return tool_providers, multimodal_providers

    async def _show_providers_legacy(self) -> None:
        """Show available providers and their status."""
        try:
            result = self.engine._get_providers_list()

            providers_panel = Panel(
                result, title="Available Providers", border_style="cyan"
            )
            self.console.print(providers_panel)

        except Exception as e:
            self._show_error(f"Failed to get providers: {e}")

    async def _show_tools(self) -> None:
        """Show available MCP tools."""
        try:
            result = await self.engine._get_tools_list()

            tools_panel = Panel(
                result, title="Available MCP Tools", border_style="cyan"
            )
            self.console.print(tools_panel)

        except Exception as e:
            self._show_error(f"Failed to get tools: {e}")

    async def _handle_mcp_command(self, command: Command) -> None:
        """Handle MCP management commands."""
        try:
            with self.console.status(
                "[bold yellow]Processing MCP command...", spinner="dots"
            ):
                result = await self.engine._handle_mcp_command(command)

            mcp_panel = Panel(result, title="MCP Command Result", border_style="cyan")
            self.console.print(mcp_panel)

        except Exception as e:
            self._show_error(f"MCP command failed: {e}")

    async def _handle_health_command(self, command: Command) -> None:
        """Handle the /health command for provider health checks."""
        args = command.args
        provider_name = None
        monitor_mode = "--monitor" in args
        interval = 300  # Default 5 minutes

        # Extract provider name and interval
        for i, arg in enumerate(args):
            if not arg.startswith("--"):
                provider_name = arg
            elif arg == "--interval" and i + 1 < len(args):
                try:
                    interval = int(args[i + 1])
                except ValueError:
                    self._show_error(
                        "Invalid interval value. Using default 300 seconds."
                    )

        try:
            if monitor_mode:
                # Start continuous monitoring
                self._show_info(
                    f"Starting health monitoring (interval: {interval}s). Press Ctrl+C to stop."
                )

                try:
                    while True:
                        # Perform health check
                        health_status = await self.engine.check_provider_health(
                            provider_name
                        )

                        # Display results
                        lines = ["Provider Health Status:"]
                        for provider, status in health_status.items():
                            status_icon = (
                                "✅" if status["status"] == "healthy" else "❌"
                            )
                            lines.append(
                                f"  {status_icon} {provider}: {status['message']}"
                            )

                        self.console.print("\n".join(lines))
                        await asyncio.sleep(interval)

                except KeyboardInterrupt:
                    self._show_info("Health monitoring stopped.")
            else:
                # One-time health check
                with self.console.status(
                    "[bold yellow]Checking provider health...", spinner="dots"
                ):
                    health_status = await self.engine.check_provider_health(
                        provider_name
                    )

                    if not health_status:
                        self._show_error("No providers configured or available.")
                        return

                    # Format and display results
                    lines = ["Provider Health Status:"]
                    lines.append("=" * 40)

                    for provider, status in health_status.items():
                        if status["status"] == "healthy":
                            status_indicator = "✅ Healthy"
                            status_color = "green"
                        elif status["status"] == "warning":
                            status_indicator = "⚠️ Warning"
                            status_color = "yellow"
                        elif status["status"] == "error":
                            status_indicator = "❌ Error"
                            status_color = "red"
                        else:
                            status_indicator = "❓ Unknown"
                            status_color = "yellow"

                        lines.append(
                            f"\n[bold]{provider}[/bold]: [{status_color}]{status_indicator}[/{status_color}]"
                        )

                        if status.get("message"):
                            lines.append(f"  Message: {status['message']}")

                        if status.get("credentials_valid") is not None:
                            cred_status = (
                                "✅ Valid"
                                if status["credentials_valid"]
                                else "❌ Invalid"
                            )
                            lines.append(f"  Credentials: {cred_status}")

                        if status.get("model_available") is not None:
                            model_status = (
                                "✅ Available"
                                if status["model_available"]
                                else "❌ Unavailable"
                            )
                            lines.append(f"  Model: {model_status}")

                    health_panel = Panel(
                        "\n".join(lines),
                        title="Provider Health Check",
                        border_style="cyan",
                    )
                    self.console.print(health_panel)

        except Exception as e:
            self._show_error(f"Health check failed: {str(e)}")

    async def _handle_repair_command(self, command: Command) -> None:
        """Handle the /repair command for configuration repair."""
        args = command.args
        auto_fix = "--auto-fix" in args or "--fix" in args
        provider_name = None

        # Extract provider name
        for arg in args:
            if not arg.startswith("--"):
                provider_name = arg
                break

        try:
            with self.console.status(
                "[bold yellow]Analyzing configuration...", spinner="dots"
            ):
                config = self.engine.config_manager.get_config()

                # Simple repair logic - check for common issues
                issues = []

                if provider_name:
                    # Check specific provider
                    if provider_name not in config.providers:
                        self._show_error(
                            f"Provider '{provider_name}' is not configured."
                        )
                        return

                    provider_config = config.providers[provider_name]

                    # Check for missing API key
                    if not provider_config.api_key:
                        issues.append(f"{provider_name}: Missing API key")

                    # Check for missing model
                    if not provider_config.model:
                        issues.append(f"{provider_name}: Missing model specification")

                else:
                    # Check all providers
                    for prov_name, prov_config in config.providers.items():
                        if not prov_config.api_key:
                            issues.append(f"{prov_name}: Missing API key")
                        if not prov_config.model:
                            issues.append(f"{prov_name}: Missing model specification")

                if not issues:
                    self._show_info("✅ No configuration issues found.")
                    return

                # Display issues
                issue_lines = ["Configuration Issues Found:"]
                issue_lines.append("=" * 40)
                for issue in issues:
                    issue_lines.append(f"❌ {issue}")

                if auto_fix:
                    issue_lines.append(
                        f"\n🔧 Auto-fix mode enabled - attempting repairs..."
                    )
                    # Basic auto-fix logic would go here
                    issue_lines.append("⚠️ Auto-fix functionality not yet implemented.")
                    issue_lines.append("Please manually fix the issues listed above.")
                else:
                    issue_lines.append(
                        f"\nUse '/repair --fix' to attempt automatic repairs."
                    )

                repair_panel = Panel(
                    "\n".join(issue_lines),
                    title="Configuration Repair",
                    border_style="red" if issues else "green",
                )
                self.console.print(repair_panel)

        except Exception as e:
            self._show_error(f"Configuration repair failed: {e}")

    async def _handle_diagnose_command(self, command: Command) -> None:
        """Handle the /diagnose command for comprehensive system diagnostics."""
        args = command.args
        provider_name = None
        detailed = "--detailed" in args

        # Extract provider name
        for arg in args:
            if not arg.startswith("--"):
                provider_name = arg
                break

        try:
            with self.console.status(
                "[bold green]Running system diagnostics...", spinner="dots"
            ):
                diagnostics = {}

                # Check configuration
                try:
                    config = self.engine.config_manager.get_config()
                    diagnostics["config"] = {
                        "providers_configured": len(config.providers),
                        "default_provider": config.default_provider,
                        "valid": True,
                    }
                except Exception as e:
                    diagnostics["config"] = {
                        "valid": False,
                        "error": str(e),
                        "providers_configured": 0,
                        "default_provider": None,
                    }

                # Check provider health
                try:
                    if provider_name:
                        health_status = await self.engine.check_provider_health(
                            provider_name
                        )
                    else:
                        health_status = await self.engine.check_provider_health()
                    diagnostics["providers"] = health_status
                except Exception as e:
                    diagnostics["providers"] = {"error": str(e)}

                # Check system resources
                diagnostics["system"] = {
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "platform": sys.platform,
                }

                # Format results
                diag_lines = ["System Diagnostics Report"]
                diag_lines.append("=" * 50)

                # Configuration status
                diag_lines.append("\n📋 Configuration:")
                config_diag = diagnostics["config"]
                if config_diag["valid"]:
                    diag_lines.append(f"  ✅ Configuration loaded successfully")
                    diag_lines.append(
                        f"  📊 Providers configured: {config_diag['providers_configured']}"
                    )
                    diag_lines.append(
                        f"  🎯 Default provider: {config_diag['default_provider'] or 'None'}"
                    )
                else:
                    diag_lines.append(
                        f"  ❌ Configuration error: {config_diag.get('error', 'Unknown')}"
                    )

                # Provider health
                diag_lines.append("\n🏥 Provider Health:")
                providers_diag = diagnostics["providers"]
                if isinstance(providers_diag, dict) and "error" not in providers_diag:
                    healthy_count = sum(
                        1
                        for status in providers_diag.values()
                        if status.get("status") == "healthy"
                    )
                    total_count = len(providers_diag)
                    diag_lines.append(
                        f"  📊 {healthy_count}/{total_count} providers healthy"
                    )

                    if detailed:
                        for prov_name, status in providers_diag.items():
                            status_icon = (
                                "✅" if status.get("status") == "healthy" else "❌"
                            )
                            diag_lines.append(
                                f"    {status_icon} {prov_name}: {status.get('message', 'Unknown status')}"
                            )
                else:
                    diag_lines.append(
                        f"  ❌ Health check failed: {providers_diag.get('error', 'Unknown error')}"
                    )

                # System info
                diag_lines.append("\n💻 System Information:")
                system_diag = diagnostics["system"]
                diag_lines.append(
                    f"  🐍 Python version: {system_diag['python_version']}"
                )
                diag_lines.append(f"  🖥️ Platform: {system_diag['platform']}")

                # Recommendations
                diag_lines.append("\n💡 Recommendations:")
                if diagnostics["config"]["providers_configured"] == 0:
                    diag_lines.append("  • Run '/setup' to configure providers")
                elif diagnostics["providers"] and isinstance(
                    diagnostics["providers"], dict
                ):
                    unhealthy_providers = [
                        name
                        for name, status in diagnostics["providers"].items()
                        if status.get("status") != "healthy"
                    ]
                    if unhealthy_providers:
                        diag_lines.append(
                            f"  • Check credentials for: {', '.join(unhealthy_providers)}"
                        )
                        diag_lines.append("  • Run '/validate' to check configuration")
                        diag_lines.append("  • Run '/repair --fix' to attempt fixes")

                diagnose_panel = Panel(
                    "\n".join(diag_lines),
                    title="System Diagnostics",
                    border_style="cyan",
                )
                self.console.print(diagnose_panel)

        except Exception as e:
            self._show_error(f"Diagnostics failed: {e}")

    async def _handle_validate_command(self, command: Command) -> None:
        """Handle the /validate command for provider configuration validation."""
        args = command.args
        provider_name = None
        fix_mode = "--fix" in args

        # Extract provider name
        for arg in args:
            if not arg.startswith("--"):
                provider_name = arg
                break

        try:
            with self.console.status(
                "[bold yellow]Validating configurations...", spinner="dots"
            ):
                if provider_name:
                    # Validate specific provider
                    config = self.engine.config_manager.get_config()
                    if provider_name not in config.providers:
                        self._show_error(
                            f"Provider '{provider_name}' is not configured."
                        )
                        return

                    provider_config = config.providers[provider_name]
                    validation_errors = []

                    # Basic validation
                    if not provider_config.api_key:
                        validation_errors.append("API key is missing")
                    if not provider_config.model:
                        validation_errors.append("Model is not specified")

                    # Test connection if API key exists
                    if provider_config.api_key:
                        try:
                            health_status = await self.engine.check_provider_health(
                                provider_name
                            )
                            if health_status[provider_name]["status"] != "healthy":
                                validation_errors.append(
                                    f"Provider health check failed: {health_status[provider_name]['message']}"
                                )
                        except Exception as e:
                            validation_errors.append(f"Health check failed: {str(e)}")

                    if validation_errors:
                        validation_panel = Panel(
                            f"❌ Validation failed for {provider_name}:\n"
                            + "\n".join(f"  • {error}" for error in validation_errors),
                            title=f"Validation Results - {provider_name}",
                            border_style="red",
                        )
                        self.console.print(validation_panel)

                        if fix_mode:
                            self._show_info(
                                "Auto-fix mode not yet implemented for provider validation."
                            )
                    else:
                        validation_panel = Panel(
                            f"✅ {provider_name} configuration is valid and working correctly.",
                            title=f"Validation Results - {provider_name}",
                            border_style="green",
                        )
                        self.console.print(validation_panel)
                else:
                    # Validate all providers
                    config = self.engine.config_manager.get_config()
                    validation_results = {}

                    for prov_name, prov_config in config.providers.items():
                        errors = []

                        # Basic validation
                        if not prov_config.api_key:
                            errors.append("API key missing")
                        if not prov_config.model:
                            errors.append("Model not specified")

                        # Health check
                        if prov_config.api_key:
                            try:
                                health_status = await self.engine.check_provider_health(
                                    prov_name
                                )
                                if health_status[prov_name]["status"] != "healthy":
                                    errors.append(
                                        f"Health check failed: {health_status[prov_name]['message']}"
                                    )
                            except Exception as e:
                                errors.append(f"Health check error: {str(e)}")

                        validation_results[prov_name] = errors

                    # Display results
                    validation_lines = []
                    validation_lines.append("Provider Validation Results:")
                    validation_lines.append("=" * 50)

                    healthy_count = 0
                    for prov_name, errors in validation_results.items():
                        if errors:
                            validation_lines.append(f"\n❌ {prov_name}:")
                            for error in errors:
                                validation_lines.append(f"   • {error}")
                        else:
                            validation_lines.append(f"\n✅ {prov_name}: Valid")
                            healthy_count += 1

                    validation_lines.append(
                        f"\nSummary: {healthy_count}/{len(validation_results)} providers are healthy"
                    )

                    validation_panel = Panel(
                        "\n".join(validation_lines),
                        title="Validation Results",
                        border_style="cyan",
                    )
                    self.console.print(validation_panel)

                    if fix_mode:
                        self._show_info(
                            "Use '/repair --fix' to automatically fix configuration issues."
                        )

        except Exception as e:
            self._show_error(f"Validation failed: {e}")

    async def _handle_agent_command(self, command: Command) -> None:
        """
        Handle agent mode commands.

        Commands:
        - /agent on [--auto-approve] - Enable agent mode
        - /agent off - Disable agent mode
        - /agent status - Show agent status
        - /agent enable [--auto-approve] - Alias for 'on'
        - /agent disable - Alias for 'off'
        - /agent list - List available personas
        - /agent use <persona> - Switch to a persona
        - /agent current - Show current persona
        - /agent info <persona> - Show persona info
        """
        args = command.args
        if not args:
            # Default to list personas if no subcommand provided
            if self.agent_persona_handler:
                await self.agent_persona_handler.handle_agent_command(command)
                return
            else:
                args = ["status"]

        subcommand = args[0].lower()

        # Route persona-related commands to persona handler
        if (
            subcommand
            in [
                "list",
                "use",
                "current",
                "info",
                "history",
                "recommend",
                "compare",
                "preview",
                "discover",
                "help",
            ]
            and self.agent_persona_handler
        ):
            await self.agent_persona_handler.handle_agent_command(command)
            return

        # Handle agent mode commands
        if not self.agent_manager:
            self._show_error(
                "Agent mode is not available. Failed to initialize agent manager."
            )
            return

        try:
            if subcommand in ["on", "enable"]:
                # Check for auto-approve flag
                auto_approve = "--auto-approve" in args

                if self.agent_manager.mode.value == "on":
                    self._show_info("Agent mode is already enabled.")
                    return

                self._show_info("Enabling agent mode...")
                success = await self.agent_manager.enable_agent_mode(
                    auto_approve=auto_approve
                )

                if success:
                    mode_text = "Agent mode enabled"
                    if auto_approve:
                        mode_text += " with auto-approval for low-risk operations"

                    self.console.print(
                        Panel(
                            mode_text
                            + "\n\nAgent will now process operations automatically.\n"
                            "Use '/agent status' to monitor progress.\n"
                            "Use '/agent off' to disable agent mode.",
                            title="Agent Mode Enabled",
                            border_style="green",
                        )
                    )

                    # Start progress monitoring
                    if self.agent_progress_ui:
                        self.agent_progress_ui.start_monitoring()
                else:
                    self._show_error("Failed to enable agent mode.")

            elif subcommand in ["off", "disable"]:
                if self.agent_manager.mode.value == "off":
                    self._show_info("Agent mode is already disabled.")
                    return

                # Check if there are active operations
                status = self.agent_manager.get_status()
                active_count = status["operations"]["in_progress"]

                if active_count > 0:
                    from rich.prompt import Confirm

                    if not Confirm.ask(
                        f"There are {active_count} active operations. Disable anyway?"
                    ):
                        self._show_info("Agent mode remains enabled.")
                        return

                self._show_info("Disabling agent mode...")
                success = await self.agent_manager.disable_agent_mode(
                    wait_for_completion=True
                )

                if success:
                    self.console.print(
                        Panel(
                            "Agent mode disabled successfully.\n\n"
                            "All operations have been stopped or completed.",
                            title="Agent Mode Disabled",
                            border_style="red",
                        )
                    )

                    # Stop progress monitoring
                    if self.agent_progress_ui:
                        self.agent_progress_ui.stop_monitoring()
                else:
                    self._show_error("Failed to disable agent mode.")

            elif subcommand == "status":
                await self._show_agent_status()

            elif subcommand == "dashboard":
                await self._show_agent_dashboard()

            elif subcommand == "pause":
                success = self.agent_manager.pause_agent_mode()
                if success:
                    self._show_success(
                        "Agent mode paused. Use '/agent resume' to continue."
                    )
                else:
                    self._show_error(
                        "Failed to pause agent mode. Agent may not be running."
                    )

            elif subcommand == "resume":
                success = self.agent_manager.resume_agent_mode()
                if success:
                    self._show_success("Agent mode resumed.")
                else:
                    self._show_error(
                        "Failed to resume agent mode. Use '/agent on' to enable."
                    )

            else:
                self._show_error(f"Unknown agent subcommand: {subcommand}")
                self._show_info(
                    "Available commands: on, off, status, dashboard, pause, resume"
                )

        except Exception as e:
            self._show_error(f"Agent command failed: {e}")

    async def _show_agent_status(self):
        """Show current agent mode status."""
        if not self.agent_progress_ui:
            self._show_error("Agent progress UI not available.")
            return

        # Show status panel
        status_panel = self.agent_progress_ui.show_status_panel()
        self.console.print(status_panel)

        # Show operations table
        operations_table = self.agent_progress_ui.show_operations_table(limit=10)
        self.console.print(operations_table)

        # Show approval queue if any
        approval_panel = self.agent_progress_ui.show_approval_queue_panel()
        self.console.print(approval_panel)

    async def _show_agent_dashboard(self):
        """Show interactive agent dashboard."""
        if not self.agent_progress_ui:
            self._show_error("Agent progress UI not available.")
            return

        try:
            self.console.print("\n[bold cyan]Agent Dashboard[/bold cyan]")
            self.console.print("Press Ctrl+C to exit dashboard view\n")

            async with self.agent_progress_ui.live_status_display():
                # Keep dashboard running until user interrupts
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    pass

            self.console.print("\n[dim]Exited dashboard view[/dim]")

        except Exception as e:
            self._show_error(f"Dashboard error: {e}")

    async def _handle_agents_command(self, command: Command) -> None:
        """
        Handle /agents command by delegating to the agent CLI handler.

        Commands:
        - /agents list - Show available agent personas
        - /agents enable <agent> - Enable/activate an agent persona
        - /agents disable [agent] - Disable active agent
        - /agents switch <agent> - Switch to different agent persona
        - /agents status [agent] - Show agent status (general or specific)
        - /agents current - Show currently active agent
        - /agents config <agent> - View agent configuration
        - /agents info <agent> - Show detailed agent information
        """
        try:
            args = command.args
            await self.agent_cli_handler.handle_command(args)
        except Exception as e:
            self._show_error(f"Agents command failed: {e}")

    async def _handle_agent_status_command(self, command: Command) -> None:
        """
        Handle /agentstatus command for enhanced agent status display.

        Commands:
        - /agentstatus show - Show current status snapshot
        - /agentstatus show --detailed - Show detailed status information
        - /agentstatus start - Start live status display
        - /agentstatus stop - Stop live status display
        - /agentstatus quick - Get quick one-line status
        - /agentstatus help - Show help message
        """
        try:
            # Check if we have status integration available
            if not hasattr(self, "status_integration") or not self.status_integration:
                try:
                    # Try to create status integration
                    from ..core.agent.cli_integration import (
                        create_cli_status_integration,
                    )

                    self.status_integration = await create_cli_status_integration(
                        self.config_manager, self.console
                    )
                except Exception as setup_error:
                    self._show_error(f"Status display not available: {setup_error}")
                    self._show_info(
                        "To enable status display, ensure you're using a status-integrated agent engine"
                    )
                    return

            # Create status command handler
            from ..core.agent.cli_integration import CLIStatusCommand

            status_command = CLIStatusCommand(self.status_integration)

            # Handle the command
            args = command.args if command.args else ["show"]
            await status_command.handle_command(args)

        except Exception as e:
            self._show_error(f"Agent status command failed: {e}")

    async def _handle_approvals_command(self, command: Command) -> None:
        """
        Handle /approvals command for managing remembered approval decisions.

        Commands:
        - /approvals list - List all stored approvals with metadata
        - /approvals revoke <signature> - Revoke a specific stored approval
        - /approvals clear - Clear all stored approvals
        - /approvals stats - Show approval statistics
        - /approvals cleanup - Remove expired approvals
        """
        try:
            # Get permission controller instance
            if (
                not hasattr(self, "approval_integration")
                or not self.approval_integration
            ):
                # Try to initialize approval integration
                from .approval_integration import (
                    create_cli_approval_integration,
                )

                try:
                    self.approval_integration = await create_cli_approval_integration(
                        console=self.console
                    )
                except Exception as setup_error:
                    self._show_error(f"Approval system not available: {setup_error}")
                    return

            permission_controller = self.approval_integration.permission_controller
            args = command.args if command.args else ["list"]
            action = args[0].lower()

            if action == "list":
                await self._list_stored_approvals(permission_controller)
            elif action == "revoke":
                if len(args) < 2:
                    self._show_error("Revoke action requires a signature to revoke")
                    return
                signature = args[1]
                await self._revoke_stored_approval(permission_controller, signature)
            elif action == "clear":
                await self._clear_stored_approvals(permission_controller)
            elif action == "stats":
                await self._show_approval_stats(permission_controller)
            elif action == "cleanup":
                await self._cleanup_expired_approvals(permission_controller)
            elif action == "batch-list":
                await self._list_batch_requests()
            elif action == "batch-show":
                if len(args) < 2:
                    self._show_error("batch-show requires a batch ID")
                    return
                batch_id = args[1]
                await self._show_batch_details(batch_id)
            elif action == "batch-approve":
                if len(args) < 2:
                    self._show_error("batch-approve requires a batch ID")
                    return
                batch_id = args[1]
                await self._approve_batch(batch_id)
            elif action == "batch-deny":
                if len(args) < 2:
                    self._show_error("batch-deny requires a batch ID")
                    return
                batch_id = args[1]
                await self._deny_batch(batch_id)
            elif action == "batch-filter":
                if len(args) < 2:
                    self._show_error("batch-filter requires a batch ID")
                    return
                batch_id = args[1]
                filter_args = args[2:] if len(args) > 2 else []
                await self._filter_batch(batch_id, filter_args)
            elif action == "batch-interactive":
                if len(args) < 2:
                    self._show_error("batch-interactive requires a batch ID")
                    return
                batch_id = args[1]
                await self._interactive_batch_approval(batch_id)
            else:
                self._show_error(f"Unknown approvals action: {action}")
                self._show_info(
                    "Valid actions: list, revoke, clear, stats, cleanup, batch-list, batch-show, batch-approve, batch-deny, batch-filter, batch-interactive"
                )

        except Exception as e:
            self._show_error(f"Approvals command failed: {e}")

    async def _list_stored_approvals(self, permission_controller) -> None:
        """List all stored approval decisions."""
        from datetime import datetime

        from rich.table import Table

        approvals = permission_controller.get_stored_approvals()

        if not approvals:
            self._show_info("No stored approvals found.")
            return

        # Create table for display
        table = Table(title="Stored Approval Decisions")
        table.add_column("Signature", style="cyan", no_wrap=True)
        table.add_column("Operation Type", style="green")
        table.add_column("Stored At", style="yellow")
        table.add_column("Expires At", style="red")
        table.add_column("Risk Level", style="magenta")

        for signature, approval_data in approvals.items():
            # Parse dates for display
            stored_at = "Unknown"
            if "stored_at" in approval_data:
                try:
                    stored_dt = datetime.fromisoformat(approval_data["stored_at"])
                    stored_at = stored_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            expires_at = "Never"
            if "expires_at" in approval_data:
                try:
                    expires_dt = datetime.fromisoformat(approval_data["expires_at"])
                    expires_at = expires_dt.strftime("%Y-%m-%d %H:%M")
                    # Check if expired
                    if datetime.now() > expires_dt:
                        expires_at = f"{expires_at} (EXPIRED)"
                except:
                    pass

            operation_type = approval_data.get("operation_type", "Unknown")

            # Get risk level from metadata
            risk_level = "Unknown"
            if "metadata" in approval_data:
                risk_level = approval_data["metadata"].get("risk_level", "Unknown")

            # Truncate signature for display
            display_signature = (
                signature[:50] + "..." if len(signature) > 50 else signature
            )

            table.add_row(
                display_signature,
                operation_type,
                stored_at,
                expires_at,
                risk_level,
            )

        self.console.print(table)
        self.console.print(f"\n[blue]Total stored approvals: {len(approvals)}[/blue]")

    async def _revoke_stored_approval(
        self, permission_controller, signature: str
    ) -> None:
        """Revoke a specific stored approval."""
        success = permission_controller.revoke_approval(signature)

        if success:
            self.console.print(
                f"[green]✅ Revoked approval for signature: {signature[:50]}...[/green]"
            )
        else:
            self.console.print(
                f"[red]❌ No approval found with signature: {signature[:50]}...[/red]"
            )

    async def _clear_stored_approvals(self, permission_controller) -> None:
        """Clear all stored approvals."""
        approvals = permission_controller.get_stored_approvals()
        count = len(approvals)

        if count == 0:
            self._show_info("No stored approvals to clear.")
            return

        # Clear all approvals
        permission_controller._approval_memory.clear()

        self.console.print(f"[green]✅ Cleared {count} stored approval(s)[/green]")

    async def _show_approval_stats(self, permission_controller) -> None:
        """Show approval statistics."""
        from datetime import datetime

        from rich.table import Table

        approvals = permission_controller.get_stored_approvals()

        if not approvals:
            self._show_info("No stored approvals found.")
            return

        # Calculate statistics
        total_approvals = len(approvals)
        expired_count = 0
        operation_types = {}
        risk_levels = {}

        for signature, approval_data in approvals.items():
            # Check expiration
            if "expires_at" in approval_data:
                try:
                    expires_dt = datetime.fromisoformat(approval_data["expires_at"])
                    if datetime.now() > expires_dt:
                        expired_count += 1
                except:
                    pass

            # Count operation types
            op_type = approval_data.get("operation_type", "Unknown")
            operation_types[op_type] = operation_types.get(op_type, 0) + 1

            # Count risk levels
            if "metadata" in approval_data:
                risk = approval_data["metadata"].get("risk_level", "Unknown")
                risk_levels[risk] = risk_levels.get(risk, 0) + 1

        # Create summary table
        stats_table = Table(title="Approval Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Stored Approvals", str(total_approvals))
        stats_table.add_row("Active Approvals", str(total_approvals - expired_count))
        stats_table.add_row("Expired Approvals", str(expired_count))

        self.console.print(stats_table)

        # Show operation type breakdown
        if operation_types:
            op_table = Table(title="Operation Types")
            op_table.add_column("Operation Type", style="cyan")
            op_table.add_column("Count", style="green")

            for op_type, count in sorted(operation_types.items()):
                op_table.add_row(op_type, str(count))

            self.console.print(op_table)

        # Show risk level breakdown
        if risk_levels:
            risk_table = Table(title="Risk Levels")
            risk_table.add_column("Risk Level", style="cyan")
            risk_table.add_column("Count", style="green")

            for risk, count in sorted(risk_levels.items()):
                risk_table.add_row(risk, str(count))

            self.console.print(risk_table)

    async def _cleanup_expired_approvals(self, permission_controller) -> None:
        """Remove expired approvals from memory."""
        cleaned_count = permission_controller.cleanup_expired_approvals()

        if cleaned_count > 0:
            self.console.print(
                f"[green]✅ Cleaned up {cleaned_count} expired approval(s)[/green]"
            )
        else:
            self.console.print("[blue]ℹ️  No expired approvals found to clean up[/blue]")

    async def _list_batch_requests(self) -> None:
        """List all pending batch approval requests."""
        try:
            # Get approval manager instance
            approval_manager = self.approval_integration.approval_manager

            if not approval_manager.pending_batches:
                self._show_info("No pending batch approval requests found.")
                return

            from rich.table import Table

            # Create overview table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Batch ID", style="cyan", width=12)
            table.add_column("Created", style="dim", width=16)
            table.add_column("Operations", style="green", justify="right", width=10)
            table.add_column("Approved", style="bright_green", justify="right", width=8)
            table.add_column("Status", style="yellow", width=10)
            table.add_column("Risk", style="red", width=8)
            table.add_column("Expires", style="red", width=12)

            for (
                batch_id,
                batch_request,
            ) in approval_manager.pending_batches.items():
                # Calculate basic stats
                total_ops = len(batch_request.operations)
                approved_ops = len(batch_request.approved_operations)

                # Assess overall risk
                risk_counts = self._calculate_batch_risk_distribution(
                    batch_request.previews
                )
                overall_risk = self._assess_batch_overall_risk(risk_counts)

                # Format expiration
                expires_str = "None"
                if batch_request.expires_at:
                    from datetime import datetime

                    time_remaining = batch_request.expires_at - datetime.now()
                    if time_remaining.total_seconds() > 0:
                        minutes_left = int(time_remaining.total_seconds() // 60)
                        expires_str = f"{minutes_left}m"
                    else:
                        expires_str = "[red]EXPIRED[/red]"

                table.add_row(
                    batch_id[:8] + "...",
                    batch_request.created_at.strftime("%m-%d %H:%M"),
                    str(total_ops),
                    str(approved_ops),
                    batch_request.status.value.title(),
                    overall_risk,
                    expires_str,
                )

            self.console.print("\n[bold]Pending Batch Approval Requests[/bold]\n")
            self.console.print(table)

            self.console.print(
                f"\n[blue]ℹ️  Use `/approvals batch-show <batch_id>` to view details[/blue]"
            )
            self.console.print(
                f"[blue]ℹ️  Use `/approvals batch-interactive <batch_id>` for interactive approval[/blue]"
            )

        except Exception as e:
            self._show_error(f"Failed to list batch requests: {e}")

    async def _show_batch_details(self, batch_id: str) -> None:
        """Show detailed information for a specific batch."""
        try:
            approval_manager = self.approval_integration.approval_manager

            # Find batch by ID (support partial matching)
            batch_request = None
            for bid, request in approval_manager.pending_batches.items():
                if bid.startswith(batch_id) or bid == batch_id:
                    batch_request = request
                    batch_id = bid
                    break

            if not batch_request:
                # Check completed batches too
                for bid, request in approval_manager.completed_batches.items():
                    if bid.startswith(batch_id) or bid == batch_id:
                        batch_request = request
                        batch_id = bid
                        break

            if not batch_request:
                self._show_error(f"Batch not found: {batch_id}")
                return

            from .batch_approval_display import create_batch_approval_panel

            panel = create_batch_approval_panel(console=self.console)

            # Display batch overview
            overview = panel.render_batch_overview(batch_request)
            self.console.print(overview)

            # Display operations table
            operations_table = panel.render_operations_table(batch_request, page=0)
            self.console.print(operations_table)

            # Display summary
            summary = panel.render_batch_summary(batch_request)
            self.console.print(summary)

        except Exception as e:
            self._show_error(f"Failed to show batch details: {e}")

    async def _approve_batch(self, batch_id: str) -> None:
        """Approve an entire batch."""
        try:
            approval_manager = self.approval_integration.approval_manager

            # Find batch
            batch_request = None
            for bid, request in approval_manager.pending_batches.items():
                if bid.startswith(batch_id) or bid == batch_id:
                    batch_request = request
                    batch_id = bid
                    break

            if not batch_request:
                self._show_error(f"Pending batch not found: {batch_id}")
                return

            # Confirm approval
            total_ops = len(batch_request.operations)
            risk_counts = self._calculate_batch_risk_distribution(
                batch_request.previews
            )
            high_risk_ops = risk_counts.get("high", 0) + risk_counts.get("critical", 0)

            warning_msg = ""
            if high_risk_ops > 0:
                warning_msg = (
                    f" [red](Warning: {high_risk_ops} high-risk operations)[/red]"
                )

            self.console.print(
                f"\n[yellow]⚠️  About to approve {total_ops} operations{warning_msg}[/yellow]"
            )

            confirm = input(
                "Are you sure you want to approve this entire batch? (yes/no): "
            )
            if confirm.lower() not in ["yes", "y"]:
                self._show_info("Batch approval cancelled.")
                return

            # Process approval
            approval_result = {"approve_all": True}
            approval_manager._process_batch_approval_result(
                batch_request, approval_result
            )

            self.console.print(
                f"[green]✅ Approved batch {batch_id[:8]}... with {total_ops} operations[/green]"
            )

        except Exception as e:
            self._show_error(f"Failed to approve batch: {e}")

    async def _deny_batch(self, batch_id: str) -> None:
        """Deny an entire batch."""
        try:
            approval_manager = self.approval_integration.approval_manager

            # Find batch
            batch_request = None
            for bid, request in approval_manager.pending_batches.items():
                if bid.startswith(batch_id) or bid == batch_id:
                    batch_request = request
                    batch_id = bid
                    break

            if not batch_request:
                self._show_error(f"Pending batch not found: {batch_id}")
                return

            # Get reason for denial
            reason = input("Reason for denial (optional): ").strip()
            if not reason:
                reason = "User denied entire batch"

            # Process denial
            approval_result = {"deny_all": True, "reason": reason}
            approval_manager._process_batch_approval_result(
                batch_request, approval_result
            )

            total_ops = len(batch_request.operations)
            self.console.print(
                f"[red]❌ Denied batch {batch_id[:8]}... with {total_ops} operations[/red]"
            )
            self.console.print(f"[dim]Reason: {reason}[/dim]")

        except Exception as e:
            self._show_error(f"Failed to deny batch: {e}")

    async def _filter_batch(self, batch_id: str, filter_args: list) -> None:
        """Apply filters to batch display."""
        try:
            approval_manager = self.approval_integration.approval_manager

            # Find batch
            batch_request = None
            for bid, request in approval_manager.pending_batches.items():
                if bid.startswith(batch_id) or bid == batch_id:
                    batch_request = request
                    batch_id = bid
                    break

            if not batch_request:
                self._show_error(f"Pending batch not found: {batch_id}")
                return

            # Parse filter arguments
            from .batch_approval_filters import (
                ActionTypeFilter,
                BatchFilterManager,
                OperationType,
                RiskLevelFilter,
                StatusFilter,
                TargetPatternFilter,
            )

            filter_manager = BatchFilterManager()

            # Simple argument parsing
            i = 0
            while i < len(filter_args):
                if filter_args[i] == "--risk" and i + 1 < len(filter_args):
                    risk_levels = filter_args[i + 1].split(",")
                    filter_manager.add_filter(RiskLevelFilter(risk_levels))
                    i += 2
                elif filter_args[i] == "--type" and i + 1 < len(filter_args):
                    type_names = filter_args[i + 1].split(",")
                    # Map type names to OperationType
                    type_map = {
                        "file_read": OperationType.FILE_READ,
                        "file_write": OperationType.FILE_WRITE,
                        "file_delete": OperationType.FILE_DELETE,
                        "command": OperationType.COMMAND_EXECUTE,
                        "web": OperationType.WEB_REQUEST,
                        "mcp": OperationType.MCP_TOOL_CALL,
                    }
                    op_types = [
                        type_map[name] for name in type_names if name in type_map
                    ]
                    if op_types:
                        filter_manager.add_filter(ActionTypeFilter(op_types))
                    i += 2
                elif filter_args[i] == "--target" and i + 1 < len(filter_args):
                    patterns = filter_args[i + 1].split(",")
                    filter_manager.add_filter(TargetPatternFilter(patterns))
                    i += 2
                elif filter_args[i] == "--status" and i + 1 < len(filter_args):
                    statuses = filter_args[i + 1].split(",")
                    filter_manager.add_filter(StatusFilter(statuses, batch_request))
                    i += 2
                else:
                    i += 1

            # Apply filters
            filtered_ops, filtered_previews = filter_manager.apply_filters(
                batch_request
            )

            # Display filtered results
            from .batch_approval_display import create_batch_approval_panel

            panel = create_batch_approval_panel(console=self.console)

            # Create temporary batch request for display
            from ..core.agent.approval_manager import BatchApprovalRequest

            filtered_batch = BatchApprovalRequest(
                id=batch_request.id,
                operations=filtered_ops,
                previews=filtered_previews,
                status=batch_request.status,
                created_at=batch_request.created_at,
                expires_at=batch_request.expires_at,
            )

            self.console.print(
                f"\n[bold]Filtered Results for Batch {batch_id[:8]}...[/bold]"
            )

            if filter_manager.get_filter_count() > 0:
                filter_descriptions = filter_manager.get_filter_descriptions()
                self.console.print(
                    f"[dim]Active filters: {', '.join(filter_descriptions)}[/dim]\n"
                )

            if filtered_ops:
                operations_table = panel.render_operations_table(filtered_batch, page=0)
                self.console.print(operations_table)

                summary = panel.render_batch_summary(filtered_batch)
                self.console.print(summary)
            else:
                self._show_info("No operations match the current filters.")

        except Exception as e:
            self._show_error(f"Failed to filter batch: {e}")

    async def _interactive_batch_approval(self, batch_id: str) -> None:
        """Interactive batch approval mode with keyboard shortcuts."""
        try:
            approval_manager = self.approval_integration.approval_manager

            # Find batch
            batch_request = None
            for bid, request in approval_manager.pending_batches.items():
                if bid.startswith(batch_id) or bid == batch_id:
                    batch_request = request
                    batch_id = bid
                    break

            if not batch_request:
                self._show_error(f"Pending batch not found: {batch_id}")
                return

            from rich.layout import Layout
            from rich.live import Live

            from .batch_approval_display import create_batch_approval_panel
            from .batch_approval_filters import (
                BatchFilterManager,
                BatchSorter,
                SortBy,
                SortCriteria,
            )

            panel = create_batch_approval_panel(console=self.console)
            filter_manager = BatchFilterManager()
            sorter = BatchSorter()

            current_page = 0
            show_approved = True
            sort_criteria = SortCriteria(SortBy.RISK_LEVEL)

            def create_display():
                # Apply filters and sorting
                filtered_ops, filtered_previews = filter_manager.apply_filters(
                    batch_request
                )
                sorted_ops, sorted_previews = sorter.sort_batch(
                    filtered_ops,
                    filtered_previews,
                    sort_criteria,
                    batch_request,
                )

                # Create layout
                layout = Layout()
                layout.split_column(
                    Layout(name="header", size=8),
                    Layout(name="main"),
                    Layout(name="footer", size=3),
                )

                # Header with overview
                temp_batch = BatchApprovalRequest(
                    id=batch_request.id,
                    operations=sorted_ops,
                    previews=sorted_previews,
                    status=batch_request.status,
                    created_at=batch_request.created_at,
                    expires_at=batch_request.expires_at,
                    approved_operations=batch_request.approved_operations,
                )

                layout["header"].update(panel.render_batch_overview(temp_batch))

                # Main operations table
                layout["main"].update(
                    panel.render_operations_table(
                        temp_batch,
                        page=current_page,
                        show_approved=show_approved,
                    )
                )

                # Footer with controls
                controls_text = (
                    "[bold cyan]Controls:[/bold cyan] "
                    "[green]A[/green]=Approve All  "
                    "[red]D[/red]=Deny All  "
                    "[yellow]F[/yellow]=Filter  "
                    "[magenta]S[/magenta]=Sort  "
                    "[blue]H[/blue]=Toggle Approved  "
                    "[dim]Q[/dim]=Quit"
                )
                layout["footer"].update(controls_text)

                return layout

            # Interactive loop
            self.console.print(
                f"\n[bold]Interactive Batch Approval - {batch_id[:8]}...[/bold]"
            )
            self.console.print(
                "[dim]Use keyboard shortcuts to interact with the batch[/dim]\n"
            )

            with Live(
                create_display(), console=self.console, refresh_per_second=2
            ) as live:
                while True:
                    try:
                        key = (
                            input("\nPress key (or 'help' for commands): ")
                            .lower()
                            .strip()
                        )

                        if key in ["q", "quit", "exit"]:
                            break
                        elif key in ["a", "approve"]:
                            # Approve all
                            approval_result = {"approve_all": True}
                            approval_manager._process_batch_approval_result(
                                batch_request, approval_result
                            )
                            self.console.print(
                                "[green]✅ Approved entire batch[/green]"
                            )
                            break
                        elif key in ["d", "deny"]:
                            # Deny all
                            reason = (
                                input("Reason for denial: ").strip()
                                or "User denied in interactive mode"
                            )
                            approval_result = {
                                "deny_all": True,
                                "reason": reason,
                            }
                            approval_manager._process_batch_approval_result(
                                batch_request, approval_result
                            )
                            self.console.print("[red]❌ Denied entire batch[/red]")
                            break
                        elif key in ["h", "toggle"]:
                            # Toggle showing approved operations
                            show_approved = not show_approved
                            live.update(create_display())
                        elif key in ["help", "?"]:
                            self.console.print("\n[bold]Interactive Controls:[/bold]")
                            self.console.print(
                                "  [green]a/approve[/green] - Approve entire batch"
                            )
                            self.console.print(
                                "  [red]d/deny[/red] - Deny entire batch"
                            )
                            self.console.print(
                                "  [yellow]h/toggle[/yellow] - Toggle showing approved operations"
                            )
                            self.console.print(
                                "  [blue]q/quit[/blue] - Exit interactive mode"
                            )
                        else:
                            self.console.print(f"[dim]Unknown command: {key}[/dim]")

                    except KeyboardInterrupt:
                        break
                    except EOFError:
                        break

            self._show_info("Interactive batch approval session ended.")

        except Exception as e:
            self._show_error(f"Interactive batch approval failed: {e}")

    def _calculate_batch_risk_distribution(self, previews) -> dict:
        """Calculate risk distribution for batch previews."""
        risk_counts = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
            "unknown": 0,
        }

        for preview in previews:
            if preview and preview.risk_assessment:
                risk_level = self._extract_batch_risk_level(preview.risk_assessment)
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            else:
                risk_counts["unknown"] += 1

        return risk_counts

    def _extract_batch_risk_level(self, risk_assessment: str) -> str:
        """Extract risk level from risk assessment string."""
        risk_lower = risk_assessment.lower()
        if "critical" in risk_lower:
            return "critical"
        elif "high" in risk_lower:
            return "high"
        elif "medium" in risk_lower:
            return "medium"
        elif "low" in risk_lower:
            return "low"
        else:
            return "unknown"

    def _assess_batch_overall_risk(self, risk_counts: dict) -> str:
        """Assess overall risk level based on distribution."""
        if risk_counts.get("critical", 0) > 0:
            return "[red]CRIT[/red]"
        elif risk_counts.get("high", 0) > 0:
            return "[red]HIGH[/red]"
        elif risk_counts.get("medium", 0) > 0:
            return "[yellow]MED[/yellow]"
        elif risk_counts.get("low", 0) > 0:
            return "[green]LOW[/green]"
        else:
            return "[dim]UNK[/dim]"

    async def _handle_permissions_command(self, command: Command) -> None:
        """Handle /permissions command for managing security settings."""
        await self.permissions_handler.handle_permissions_command(command)

    async def _handle_history_command(self, command: Command) -> None:
        """
        Handle history commands.

        Commands:
        - /history - Show recent commands
        - /history recent [count] - Show recent commands (default: 20)
        - /history search <query> - Search command history
        - /history stats - Show history statistics
        - /history clear --confirm - Clear all history
        - /history export <file> [format] - Export history to file
        """
        args = command.args

        if not args:
            # Default to showing recent commands
            args = ["recent"]

        action = args[0].lower()

        try:
            if action == "recent":
                # Show recent commands
                count = 20
                if len(args) > 1:
                    try:
                        count = int(args[1])
                        count = max(1, min(count, 100))  # Limit between 1-100
                    except ValueError:
                        self._show_error(
                            "Invalid count for recent commands. Using default (20)."
                        )

                recent_commands = self.history_manager.get_recent_commands(count)

                if not recent_commands:
                    self._show_info("No command history available.")
                    return

                from rich.table import Table

                table = Table(title=f"Recent Commands (Last {len(recent_commands)})")
                table.add_column("Index", style="dim", width=6)
                table.add_column("Time", style="cyan", width=16)
                table.add_column("Command", style="white")

                for i, entry in enumerate(reversed(recent_commands), 1):
                    time_str = entry.datetime.strftime("%m-%d %H:%M:%S")
                    # Truncate long commands
                    cmd_display = (
                        entry.command[:80] + "..."
                        if len(entry.command) > 80
                        else entry.command
                    )
                    table.add_row(str(i), time_str, cmd_display)

                self.console.print(table)

            elif action == "search":
                if len(args) < 2:
                    self._show_error(
                        "Search requires a query. Usage: /history search <query>"
                    )
                    return

                query = " ".join(args[1:])
                results = self.history_manager.search_history(query, limit=30)

                if not results:
                    self._show_info(f"No commands found matching '{query}'.")
                    return

                from rich.table import Table

                table = Table(
                    title=f"Search Results for '{query}' ({len(results)} found)"
                )
                table.add_column("Index", style="dim", width=6)
                table.add_column("Time", style="cyan", width=16)
                table.add_column("Command", style="white")

                for i, entry in enumerate(results, 1):
                    time_str = entry.datetime.strftime("%m-%d %H:%M:%S")
                    # Highlight matching text
                    cmd_display = entry.command
                    if len(cmd_display) > 80:
                        cmd_display = cmd_display[:80] + "..."

                    table.add_row(str(i), time_str, cmd_display)

                self.console.print(table)

            elif action == "stats":
                # Show history statistics
                stats = self.history_manager.get_statistics()

                from rich.panel import Panel

                stats_text = f"""Total Commands: {stats['total_commands']}
Current Session: {stats['current_session_commands']}
Unique Commands: {stats['unique_commands']}
Sessions Tracked: {stats.get('sessions', 1)}

Oldest Entry: {stats['oldest_entry'] or 'None'}
Newest Entry: {stats['newest_entry'] or 'None'}"""

                self.console.print(
                    Panel(
                        stats_text,
                        title="Command History Statistics",
                        border_style="blue",
                    )
                )

            elif action == "clear":
                # Clear history with confirmation
                if "--confirm" not in args:
                    self._show_info(
                        "To clear command history, use: /history clear --confirm"
                    )
                    self._show_info("This action cannot be undone!")
                    return

                success = self.history_manager.clear_history(confirm=True)
                if success:
                    self._show_success("Command history cleared successfully.")
                else:
                    self._show_error("Failed to clear command history.")

            elif action == "export":
                if len(args) < 2:
                    self._show_error(
                        "Export requires a filename. Usage: /history export <file> [format]"
                    )
                    return

                filename = args[1]
                format_type = args[2] if len(args) > 2 else "json"

                if format_type not in ["json", "txt"]:
                    self._show_error(
                        "Format must be 'json' or 'txt'. Defaulting to 'json'."
                    )
                    format_type = "json"

                success = self.history_manager.export_history(filename, format_type)
                if success:
                    self._show_success(
                        f"History exported to {filename} in {format_type} format."
                    )
                else:
                    self._show_error(f"Failed to export history to {filename}.")

            else:
                self._show_error(f"Unknown history action: {action}")
                self._show_info(
                    "Available actions: recent, search, stats, clear, export"
                )

        except Exception as e:
            self._show_error(f"History command error: {e}")

    def _setup_readline_history(self) -> None:
        """Setup readline for command history and arrow key navigation."""
        try:
            # Setup history file
            history_file = self.history_manager.storage_path / "readline_history"
            history_file.parent.mkdir(parents=True, exist_ok=True)

            # Configure readline with tab completion
            readline.set_startup_hook(None)
            readline.set_completer(self._complete_command)
            readline.set_completer_delims(" \t\n")
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind(r'"\e[A": history-search-backward')
            readline.parse_and_bind(r'"\e[B": history-search-forward')

            # Load existing history
            try:
                readline.read_history_file(str(history_file))
            except FileNotFoundError:
                pass  # No history file yet

            # Setup auto-save on exit
            atexit.register(readline.write_history_file, str(history_file))

            # Limit history size
            readline.set_history_length(1000)

        except Exception:
            # If readline fails, just continue without arrow key support
            pass

    def _create_completion_callback(
        self,
    ) -> Optional[Callable[[str], List[str]]]:
        """Create a completion callback for the improved terminal input."""

        def completion_callback(text: str) -> List[str]:
            """Generate completions for the given text."""
            try:
                # Use existing completion logic
                line_buffer = text  # For improved input, we get the full text
                completions = self._get_completions(line_buffer, text)
                return completions
            except Exception as e:
                logger.debug(f"Completion callback error: {e}")
                return []

        return completion_callback

    def _complete_command(self, text: str, state: int) -> Optional[str]:
        """
        Tab completion function for commands and arguments.

        Args:
            text: Current text being completed
            state: Completion state (0 for first match, 1 for second, etc.)

        Returns:
            Next completion match or None if no more matches
        """
        try:
            # Get the current line buffer
            line_buffer = readline.get_line_buffer()

            # Get all potential completions
            completions = self._get_completions(line_buffer, text)

            # Return the completion for the current state
            if state < len(completions):
                return completions[state]
            return None

        except Exception:
            # If completion fails, return None to avoid interrupting user input
            return None

    def _get_completions(self, line_buffer: str, text: str) -> List[str]:
        """
        Get all possible completions for the current input.

        Args:
            line_buffer: Full line being edited
            text: Current text being completed

        Returns:
            List of possible completions
        """
        try:
            parts = line_buffer.split()

            # If we're at the beginning or completing a command that starts with /
            if not parts or (
                len(parts) == 1
                and line_buffer.endswith(" ") is False
                and text.startswith("/")
            ):
                return self._complete_slash_commands(text)

            # If we have parts, determine what we're completing
            if parts:
                command = parts[0]

                # Handle slash commands
                if command.startswith("/"):
                    return self._complete_command_arguments(
                        command, parts[1:], text, line_buffer.endswith(" ")
                    )

            return []

        except Exception:
            return []

    def _complete_slash_commands(self, text: str) -> List[str]:
        """Complete slash commands including dynamic ones."""
        from .commands import SlashCommand

        # Get built-in commands
        all_commands = SlashCommand.get_all_commands()

        # Add dynamic commands
        registry = get_command_registry()
        all_commands.extend(registry.list_commands())

        return [cmd for cmd in all_commands if cmd.startswith(text)]

    def _complete_command_arguments(
        self, command: str, args: List[str], text: str, at_end: bool
    ) -> List[str]:
        """Complete arguments for specific slash commands."""
        try:
            # Determine which argument position we're completing
            if at_end:
                arg_index = len(args)
            else:
                arg_index = len(args) - 1 if args else 0

            # Use unified completion manager
            return self.completion_manager.get_completions(
                command, arg_index, text, args
            )
        except Exception:
            return []

    def _get_provider_names(self, text: str) -> List[str]:
        """Get provider names that match the text."""
        try:
            if hasattr(self.engine, "providers") and self.engine.providers:
                # Get provider names from engine
                provider_names = list(self.engine.providers.keys())
            else:
                # Fallback to common provider names
                provider_names = [
                    "openai",
                    "claude",
                    "gemini",
                    "openrouter",
                    "azure",
                    "bedrock",
                    "mistral",
                    "perplexity",
                    "cohere",
                    "xai",
                    "ollama",
                    "vertex",
                ]

            return [name for name in provider_names if name.startswith(text)]
        except Exception:
            return []

    def _get_model_names(self, provider_name: str, text: str) -> List[str]:
        """Get model names for a specific provider that match the text."""
        try:
            model_names = []

            # Get provider's built-in models
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

            # Also get custom models for this provider
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
        """Get custom model names that match the text."""
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

    async def _handle_add_model_command(self, command: Command) -> None:
        """Handle the add-model command."""
        try:
            from datetime import datetime

            from ..core.models import EnhancedModelInfo

            args = command.args
            if len(args) < 2:
                self._show_error(
                    "Add-model command requires at least model ID and provider"
                )
                return

            model_name = args[0]
            provider = args[1]

            # Find where key=value parameters start
            param_start = 2
            description_parts = []

            for i, arg in enumerate(args[2:], 2):
                if "=" in arg:
                    param_start = i
                    break
                description_parts.append(arg)

            if description_parts:
                description = " ".join(description_parts).strip(
                    "\"'"
                )  # Remove quotes if present
            else:
                description = f"Custom {model_name} model"

            # Parse optional parameters
            max_tokens = 4096
            cost_input = 1.0
            cost_output = 3.0
            swe_score = 50.0
            supports_tools = False
            supports_multimodal = False
            is_free = False

            # Parse additional arguments as key=value pairs
            for arg in args[param_start:]:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = key.lower()

                    try:
                        if key == "max_tokens":
                            max_tokens = int(value)
                        elif key == "cost_input":
                            cost_input = float(value)
                        elif key == "cost_output":
                            cost_output = float(value)
                        elif key == "swe_score":
                            swe_score = float(value)
                        elif key == "supports_tools":
                            supports_tools = value.lower() in [
                                "true",
                                "yes",
                                "1",
                            ]
                        elif key == "supports_multimodal":
                            supports_multimodal = value.lower() in [
                                "true",
                                "yes",
                                "1",
                            ]
                        elif key == "is_free":
                            is_free = value.lower() in ["true", "yes", "1"]
                    except ValueError:
                        self._show_warning(f"Invalid value for {key}: {value}")

            # Create enhanced model info
            model_info = EnhancedModelInfo(
                name=model_name,
                provider=provider,
                description=description,
                max_tokens=max_tokens,
                cost_per_million_input=cost_input,
                cost_per_million_output=cost_output,
                swe_score=swe_score,
                available=True,
                supports_tools=supports_tools,
                supports_multimodal=supports_multimodal,
                latest_version=False,
                deprecated=False,
                release_date=datetime.now(),
                context_window=max_tokens,
                is_free=is_free,
            )

            # Update SWE rating
            model_info.update_swe_rating()

            # Add to configuration
            self.engine.config_manager.add_custom_model(model_info)

            self._show_success(
                f"Added custom model '{model_name}' for provider '{provider}'"
            )

        except Exception as e:
            self._show_error(f"Failed to add custom model: {e}")

    async def _handle_remove_model_command(self, command: Command) -> None:
        """Handle the remove-model command."""
        try:
            args = command.args
            if len(args) != 2:
                self._show_error("Remove-model command requires model ID and provider")
                return

            model_name = args[0]
            provider = args[1]

            # Remove from configuration
            success = self.engine.config_manager.remove_custom_model(
                model_name, provider
            )

            if success:
                self._show_success(
                    f"Removed custom model '{model_name}' from provider '{provider}'"
                )
            else:
                self._show_error(
                    f"Custom model '{model_name}' for provider '{provider}' not found"
                )

        except Exception as e:
            self._show_error(f"Failed to remove custom model: {e}")

    async def _handle_list_custom_models_command(self, command: Command) -> None:
        """Handle the list-custom-models command."""
        try:
            from rich.panel import Panel
            from rich.table import Table

            custom_models = self.engine.config_manager.get_custom_models()

            if not custom_models:
                self._show_info("No custom models configured")
                return

            # Create table for custom models
            table = Table(
                title="Custom Models",
                show_header=True,
                header_style="bold blue",
            )
            table.add_column("Model", style="cyan", width=25)
            table.add_column("Provider", style="green", width=15)
            table.add_column("Description", style="white", width=35)
            table.add_column("Tokens", justify="right", style="yellow", width=8)
            table.add_column("Cost/1M", justify="right", style="red", width=12)
            table.add_column("SWE", justify="center", style="magenta", width=6)
            table.add_column("Tools", justify="center", style="blue", width=6)
            table.add_column("MM", justify="center", style="purple", width=4)

            for model in custom_models:
                cost_display = f"${model.cost_per_million_input:.1f}/${model.cost_per_million_output:.1f}"
                swe_display = f"{model.swe_score:.1f}" if model.swe_score else "N/A"
                tools_display = "✓" if model.supports_tools else "✗"
                mm_display = "✓" if model.supports_multimodal else "✗"

                table.add_row(
                    model.name,
                    model.provider,
                    (
                        model.description[:35] + "..."
                        if len(model.description) > 35
                        else model.description
                    ),
                    f"{model.max_tokens:,}",
                    cost_display,
                    swe_display,
                    tools_display,
                    mm_display,
                )

            # Display in a panel
            models_panel = Panel(
                table,
                title="🎯 Custom Models Configuration",
                subtitle=f"Total: {len(custom_models)} custom models",
                border_style="blue",
            )

            self.console.print(models_panel)

        except Exception as e:
            self._show_error(f"Failed to list custom models: {e}")


def main() -> None:
    """
    Main entry point for the Omnimancer CLI application.

    This function initializes the application and starts the interactive CLI.
    """
    import sys

    import click

    from ..core.config_manager import ConfigManager
    from ..core.engine import CoreEngine

    @click.command()
    @click.option("--help", "-h", is_flag=True, help="Show this help message and exit")
    @click.option("--version", "-v", is_flag=True, help="Show version information")
    @click.option("--config", "-c", help="Path to configuration file")
    @click.option(
        "--no-approval",
        is_flag=True,
        help="Skip approval prompts and auto-approve all operations (DANGEROUS)",
    )
    def cli_main(help, version, config, no_approval):
        """Omnimancer - A unified CLI for multiple AI language models."""

        if help:
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            return

        if version:
            from omnimancer import __version__

            click.echo(f"Omnimancer CLI v{__version__}")
            return

        try:
            # Initialize configuration manager
            config_manager = ConfigManager(config)

            # Initialize core engine
            engine = CoreEngine(config_manager)

            # Initialize CLI interface
            cli = CommandLineInterface(engine, no_approval=no_approval)

            # Start the interactive session
            cli.start()

        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            click.echo(f"Error starting Omnimancer: {e}", err=True)
            sys.exit(1)

    # Handle the case where this is called directly
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        click.echo(
            """Omnimancer - A unified CLI for multiple AI language models.

Usage: omnimancer [OPTIONS]

Options:
  -h, --help     Show this help message and exit
  -v, --version  Show version information
  -c, --config   Path to configuration file

Omnimancer provides a unified interface to interact with multiple AI providers
including Claude, OpenAI, and others through a single command-line tool.

Commands available in interactive mode:
  /help     - Show available commands
  /quit     - Exit the application
  /clear    - Clear the screen
  /status   - Show current status
  /models   - List available models (coming soon)
  /switch   - Switch AI provider/model (coming soon)
  /config   - Manage configuration (coming soon)
  /save     - Save conversation (coming soon)
  /load     - Load conversation (coming soon)

Just type your message to start chatting with AI!
"""
        )
        return

    cli_main()


if __name__ == "__main__":
    main()


# Additional methods that should be part of CommandLineInterface class
# These were accidentally placed outside the class definition
