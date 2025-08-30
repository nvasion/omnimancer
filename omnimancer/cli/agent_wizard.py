"""
Agent Creation Wizard for Omnimancer.

Interactive CLI wizard that guides users through creating custom agent
configurations step-by-step with validation and real-time feedback.
"""

import logging
from typing import List, Optional, Set

from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from ..core.agent.agent_config import (
    AgentRepository,
    AgentTool,
    BehaviorRules,
    ContextParameters,
    CustomAgentConfig,
    CustomAgentStatus,
    CustomAgentValidator,
    ModelSettings,
)
from ..core.agent.config import ProviderType
from ..core.agent.persona import PersonaCapability, PersonaCategory
from ..core.models import ConfigTemplateManager

logger = logging.getLogger(__name__)


class AgentCreationWizard:
    """Interactive wizard for creating custom agents."""

    def __init__(
        self,
        console: Optional[Console] = None,
        template_manager: Optional[ConfigTemplateManager] = None,
        repository: Optional[AgentRepository] = None,
    ):
        """
        Initialize the agent creation wizard.

        Args:
            console: Rich console for output
            template_manager: Configuration template manager
            repository: Agent repository for storage
        """
        self.console = console or Console()
        self.template_manager = template_manager or ConfigTemplateManager()
        self.repository = repository or AgentRepository()
        self.validator = CustomAgentValidator()

        # Wizard state
        self.current_config: Optional[CustomAgentConfig] = None
        self.step_history: List[str] = []

    async def run_wizard(
        self, base_config: Optional[CustomAgentConfig] = None
    ) -> Optional[CustomAgentConfig]:
        """
        Run the complete agent creation wizard.

        Args:
            base_config: Optional base configuration to start with

        Returns:
            Created agent configuration or None if cancelled
        """
        try:
            self._show_welcome()

            # Initialize with base config or create new
            self.current_config = base_config or CustomAgentConfig()

            # Run wizard steps
            if not await self._step_basic_info():
                return None

            if not await self._step_select_template():
                return None

            if not await self._step_model_settings():
                return None

            if not await self._step_tools_capabilities():
                return None

            if not await self._step_context_parameters():
                return None

            if not await self._step_behavior_rules():
                return None

            if not await self._step_review_and_save():
                return None

            return self.current_config

        except KeyboardInterrupt:
            self.console.print("\\n[yellow]Wizard cancelled by user.[/yellow]")
            return None
        except Exception as e:
            self.console.print(f"\\n[red]Wizard error: {e}[/red]")
            logger.error(f"Agent creation wizard error: {e}", exc_info=True)
            return None

    def _show_welcome(self):
        """Show welcome message and instructions."""
        welcome_text = Text.assemble(
            "Welcome to the ",
            ("Agent Creation Wizard", "bold cyan"),
            "!\\n\\n",
            "This wizard will guide you through creating a custom agent configuration.\\n",
            "You can press ",
            ("Ctrl+C", "bold"),
            " at any time to cancel.\\n\\n",
            "Let's get started! 🚀",
        )

        panel = Panel(
            Align.center(welcome_text),
            title="🤖 Custom Agent Creator",
            box=ROUNDED,
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    async def _step_basic_info(self) -> bool:
        """Step 1: Collect basic agent information."""
        self.console.rule("📝 Step 1: Basic Information")

        # Agent name
        while True:
            name = Prompt.ask(
                "Enter a name for your agent",
                default=(
                    self.current_config.name
                    if self.current_config.name != "Untitled Agent"
                    else ""
                ),
            ).strip()

            if not name:
                self.console.print("[red]Agent name cannot be empty.[/red]")
                continue

            if len(name) > 100:
                self.console.print(
                    "[red]Agent name too long (max 100 characters).[/red]"
                )
                continue

            # Check for name conflicts
            existing = self.repository.get_by_name(name)
            if existing:
                self.console.print(
                    f"[yellow]An agent named '{name}' already exists.[/yellow]"
                )
                if not Confirm.ask(
                    "Do you want to use a different name?", default=True
                ):
                    continue

            self.current_config.name = name
            break

        # Description
        description = Prompt.ask(
            "Enter a description for your agent (optional)",
            default=self.current_config.description,
        ).strip()

        if len(description) > 500:
            self.console.print(
                "[yellow]Description truncated to 500 characters.[/yellow]"
            )
            description = description[:500]

        self.current_config.description = description

        # Category
        self.console.print("\\nSelect a category for your agent:")
        categories = list(PersonaCategory)
        for i, category in enumerate(categories, 1):
            self.console.print(f"  {i}. {category.value.title()}")

        while True:
            try:
                choice = IntPrompt.ask(
                    "Category",
                    default=categories.index(self.current_config.category) + 1,
                    show_default=True,
                )
                if 1 <= choice <= len(categories):
                    self.current_config.category = categories[choice - 1]
                    break
                else:
                    self.console.print(
                        f"[red]Please enter a number between 1 and {len(categories)}.[/red]"
                    )
            except (ValueError, EOFError):
                self.console.print("[red]Invalid input. Please enter a number.[/red]")

        self.step_history.append("basic_info")
        return True

    async def _step_select_template(self) -> bool:
        """Step 2: Select base template (optional)."""
        self.console.rule("🎯 Step 2: Base Template")

        self.console.print(
            "You can base your agent on an existing template for easier setup."
        )

        use_template = Confirm.ask(
            "Would you like to use a base template?", default=False
        )

        if not use_template:
            self.current_config.base_template_id = None
            self.step_history.append("template_skipped")
            return True

        # Get available templates
        try:
            templates = self.template_manager.list_templates()

            if not templates:
                self.console.print(
                    "[yellow]No templates available. Continuing without template.[/yellow]"
                )
                self.current_config.base_template_id = None
                self.step_history.append("no_templates")
                return True

            self.console.print("\\nAvailable templates:")
            for i, template in enumerate(templates, 1):
                description = template.description[:60] + (
                    "..." if len(template.description) > 60 else ""
                )
                self.console.print(f"  {i}. {template.name} - {description}")

            while True:
                try:
                    choice = IntPrompt.ask(
                        "Select template (0 for none)",
                        default=0,
                        show_default=True,
                    )

                    if choice == 0:
                        self.current_config.base_template_id = None
                        break
                    elif 1 <= choice <= len(templates):
                        selected_template = templates[choice - 1]
                        self.current_config.base_template_id = selected_template.id

                        # Inherit from template
                        self.current_config.inherit_from_template(self.template_manager)

                        self.console.print(
                            f"[green]Selected template: {selected_template.name}[/green]"
                        )
                        break
                    else:
                        self.console.print(
                            f"[red]Please enter a number between 0 and {len(templates)}.[/red]"
                        )
                except (ValueError, EOFError):
                    self.console.print(
                        "[red]Invalid input. Please enter a number.[/red]"
                    )

        except Exception as e:
            self.console.print(f"[red]Error loading templates: {e}[/red]")
            self.current_config.base_template_id = None

        self.step_history.append("template_selected")
        return True

    async def _step_model_settings(self) -> bool:
        """Step 3: Configure model settings."""
        self.console.rule("🧠 Step 3: Model Configuration")

        current_settings = self.current_config.model_settings

        # Provider selection
        self.console.print("\\nSelect AI provider:")
        providers = list(ProviderType)
        for i, provider in enumerate(providers, 1):
            self.console.print(f"  {i}. {provider.value}")

        while True:
            try:
                choice = IntPrompt.ask(
                    "Provider",
                    default=providers.index(current_settings.provider_type) + 1,
                    show_default=True,
                )
                if 1 <= choice <= len(providers):
                    provider_type = providers[choice - 1]
                    break
                else:
                    self.console.print(
                        f"[red]Please enter a number between 1 and {len(providers)}.[/red]"
                    )
            except (ValueError, EOFError):
                self.console.print("[red]Invalid input. Please enter a number.[/red]")

        # Model name
        model_suggestions = self._get_model_suggestions(provider_type)
        if model_suggestions:
            self.console.print(f"\\nSuggested models for {provider_type.value}:")
            for suggestion in model_suggestions[:5]:  # Show top 5
                self.console.print(f"  • {suggestion}")

        model_name = Prompt.ask(
            "Model name", default=current_settings.model_name
        ).strip()

        if not model_name:
            model_name = self._get_default_model(provider_type)
            self.console.print(f"[yellow]Using default model: {model_name}[/yellow]")

        # Temperature
        while True:
            try:
                temperature = FloatPrompt.ask(
                    "Temperature (0.0 = deterministic, 2.0 = very creative)",
                    default=current_settings.temperature,
                    show_default=True,
                )
                if 0.0 <= temperature <= 2.0:
                    break
                else:
                    self.console.print(
                        "[red]Temperature must be between 0.0 and 2.0.[/red]"
                    )
            except (ValueError, EOFError):
                self.console.print("[red]Invalid input. Please enter a number.[/red]")

        # Max tokens (optional)
        max_tokens = None
        if Confirm.ask("Set maximum response length?", default=False):
            while True:
                try:
                    max_tokens = IntPrompt.ask(
                        "Maximum tokens",
                        default=current_settings.max_tokens or 4096,
                        show_default=True,
                    )
                    if max_tokens > 0:
                        break
                    else:
                        self.console.print(
                            "[red]Maximum tokens must be positive.[/red]"
                        )
                except (ValueError, EOFError):
                    self.console.print(
                        "[red]Invalid input. Please enter a number.[/red]"
                    )

        # Update model settings
        self.current_config.model_settings = ModelSettings(
            provider_type=provider_type,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.step_history.append("model_configured")
        return True

    async def _step_tools_capabilities(self) -> bool:
        """Step 4: Select tools and capabilities."""
        self.console.rule("🔧 Step 4: Tools and Capabilities")

        # Tools selection
        self.console.print("\\nSelect tools to enable for your agent:")
        all_tools = list(AgentTool)
        current_tools = self.current_config.enabled_tools

        tool_table = Table(show_header=True, header_style="bold magenta")
        tool_table.add_column("Tool", style="cyan")
        tool_table.add_column("Description", style="white")
        tool_table.add_column("Enabled", style="green")

        tool_descriptions = {
            AgentTool.FILE_OPERATIONS: "Read, write, and manage files",
            AgentTool.PROGRAM_EXECUTOR: "Execute system commands and programs",
            AgentTool.WEB_CLIENT: "Make web requests and browse content",
            AgentTool.MCP_TOOLS: "Model Context Protocol integration",
            AgentTool.SEARCH: "Search capabilities and information retrieval",
            AgentTool.CODE_ANALYSIS: "Analyze and understand code structure",
            AgentTool.SYSTEM_INFO: "Access system information and diagnostics",
        }

        for tool in all_tools:
            description = tool_descriptions.get(tool, "Advanced tool functionality")
            enabled = "✅" if tool in current_tools else "❌"
            tool_table.add_row(
                tool.value.replace("_", " ").title(), description, enabled
            )

        self.console.print(tool_table)

        # Interactive tool selection
        selected_tools = set(current_tools)

        while True:
            self.console.print("\\nTool selection:")
            self.console.print("1. Toggle individual tools")
            self.console.print("2. Select preset configuration")
            self.console.print("3. Continue with current selection")

            choice = Prompt.ask("Choose option", choices=["1", "2", "3"], default="3")

            if choice == "1":
                selected_tools = await self._interactive_tool_selection(
                    all_tools, selected_tools
                )
            elif choice == "2":
                selected_tools = await self._preset_tool_selection()
            else:
                break

        self.current_config.enabled_tools = selected_tools

        # Capabilities selection
        self.console.print("\\nSelect capabilities for your agent:")
        all_capabilities = list(PersonaCapability)
        current_capabilities = self.current_config.capabilities

        # Smart capability suggestions based on selected tools
        suggested_capabilities = self._suggest_capabilities_for_tools(selected_tools)

        if suggested_capabilities:
            self.console.print(
                "\\n[cyan]Suggested capabilities based on selected tools:[/cyan]"
            )
            for cap in suggested_capabilities:
                self.console.print(f"  • {cap.value.replace('_', ' ').title()}")

            if Confirm.ask("Use suggested capabilities?", default=True):
                self.current_config.capabilities = suggested_capabilities
            else:
                self.current_config.capabilities = (
                    await self._interactive_capability_selection(
                        all_capabilities, current_capabilities
                    )
                )
        else:
            self.current_config.capabilities = (
                await self._interactive_capability_selection(
                    all_capabilities, current_capabilities
                )
            )

        self.step_history.append("tools_capabilities_configured")
        return True

    async def _step_context_parameters(self) -> bool:
        """Step 5: Configure context and behavior parameters."""
        self.console.rule("💭 Step 5: Context Configuration")

        current_params = self.current_config.context_parameters

        # System prompt
        self.console.print("\\nConfigure the system prompt for your agent:")
        self.console.print("This defines the agent's personality and behavior.")

        current_prompt = current_params.system_prompt
        if current_prompt:
            self.console.print(f"\\nCurrent prompt preview:")
            preview = current_prompt[:200] + (
                "..." if len(current_prompt) > 200 else ""
            )
            self.console.print(Panel(preview, box=ROUNDED))

        if Confirm.ask("Edit system prompt?", default=bool(current_prompt)):
            # For simplicity, use basic input. In a full implementation,
            # this could open an editor or provide multi-line input
            system_prompt = Prompt.ask("System prompt", default=current_prompt).strip()
        else:
            system_prompt = current_prompt

        # Context window size
        context_size = IntPrompt.ask(
            "Context window size (tokens)",
            default=current_params.context_window_size,
            show_default=True,
        )

        if context_size < 1000:
            self.console.print(
                "[yellow]Warning: Very small context window may limit functionality.[/yellow]"
            )

        # Memory settings
        conversation_memory = Confirm.ask(
            "Enable conversation memory?",
            default=current_params.conversation_memory,
        )

        memory_limit = current_params.memory_limit
        if conversation_memory:
            memory_limit = IntPrompt.ask(
                "Memory limit (number of messages to remember)",
                default=current_params.memory_limit,
                show_default=True,
            )

        # Response format
        self.console.print("\\nResponse format options:")
        self.console.print("1. Default - Natural conversational responses")
        self.console.print("2. JSON - Structured JSON responses")
        self.console.print("3. Markdown - Markdown-formatted responses")

        format_choices = ["default", "json", "markdown"]
        format_choice = Prompt.ask(
            "Response format", choices=["1", "2", "3"], default="1"
        )

        response_format = format_choices[int(format_choice) - 1]

        # Update context parameters
        self.current_config.context_parameters = ContextParameters(
            system_prompt=system_prompt,
            context_window_size=context_size,
            conversation_memory=conversation_memory,
            memory_limit=memory_limit,
            response_format=response_format,
        )

        self.step_history.append("context_configured")
        return True

    async def _step_behavior_rules(self) -> bool:
        """Step 6: Configure behavior rules and constraints."""
        self.console.rule("⚙️ Step 6: Behavior Rules")

        current_rules = self.current_config.behavior_rules

        # Safety level
        self.console.print("\\nSelect safety level:")
        self.console.print("1. Strict - Maximum safety, conservative behavior")
        self.console.print("2. Standard - Balanced safety and functionality")
        self.console.print("3. Permissive - Minimal restrictions, maximum freedom")

        safety_levels = ["strict", "standard", "permissive"]
        safety_choice = Prompt.ask("Safety level", choices=["1", "2", "3"], default="2")
        safety_level = safety_levels[int(safety_choice) - 1]

        # Reasoning style
        self.console.print("\\nSelect reasoning style:")
        self.console.print("1. Concise - Brief, to-the-point responses")
        self.console.print("2. Balanced - Moderate detail level")
        self.console.print("3. Verbose - Detailed explanations")
        self.console.print("4. Step-by-step - Explicit reasoning process")

        reasoning_styles = ["concise", "balanced", "verbose", "step_by_step"]
        reasoning_choice = Prompt.ask(
            "Reasoning style", choices=["1", "2", "3", "4"], default="2"
        )
        reasoning_style = reasoning_styles[int(reasoning_choice) - 1]

        # Creativity level
        self.console.print("\\nSelect creativity level:")
        self.console.print("1. Low - Conservative, predictable responses")
        self.console.print("2. Medium - Balanced creativity")
        self.console.print("3. High - Creative, exploratory responses")

        creativity_levels = ["low", "medium", "high"]
        creativity_choice = Prompt.ask(
            "Creativity level", choices=["1", "2", "3"], default="2"
        )
        creativity_level = creativity_levels[int(creativity_choice) - 1]

        # Confirmation requirements
        require_confirmation = []

        self.console.print("\\nConfigure operations that require user confirmation:")
        confirmation_options = [
            ("file_delete", "File deletion operations"),
            ("system_commands", "System command execution"),
            ("web_requests", "External web requests"),
            ("code_execution", "Code execution"),
            ("data_modification", "Data modification operations"),
        ]

        for option, description in confirmation_options:
            if Confirm.ask(
                f"Require confirmation for {description}?",
                default=option in current_rules.require_confirmation_for,
            ):
                require_confirmation.append(option)

        # Update behavior rules
        self.current_config.behavior_rules = BehaviorRules(
            require_confirmation_for=require_confirmation,
            reasoning_style=reasoning_style,
            creativity_level=creativity_level,
            safety_level=safety_level,
            custom_restrictions=current_rules.custom_restrictions,  # Keep existing custom restrictions
        )

        self.step_history.append("behavior_configured")
        return True

    async def _step_review_and_save(self) -> bool:
        """Step 7: Review configuration and save."""
        self.console.rule("📋 Step 7: Review and Save")

        # Show configuration summary
        self._show_configuration_summary()

        # Validation
        try:
            self.validator.validate_config(self.current_config)
            self.console.print("[green]✅ Configuration is valid![/green]")
        except ValueError as e:
            self.console.print(f"[red]❌ Configuration validation failed: {e}[/red]")
            if not Confirm.ask("Continue anyway?", default=False):
                return False

        # Final confirmation
        if not Confirm.ask("\\nSave this agent configuration?", default=True):
            return False

        # Save the configuration
        try:
            self.current_config.status = CustomAgentStatus.ACTIVE
            saved_config = self.repository.create(self.current_config)

            self.console.print(
                f"\\n[green]✅ Agent '{saved_config.name}' created successfully![/green]"
            )
            self.console.print(f"[cyan]Agent ID: {saved_config.id}[/cyan]")

            return True

        except Exception as e:
            self.console.print(f"[red]❌ Failed to save agent: {e}[/red]")
            return False

    def _show_configuration_summary(self):
        """Display a summary of the current configuration."""
        config = self.current_config

        # Basic info
        basic_table = Table(show_header=False, box=None)
        basic_table.add_column("Field", style="cyan")
        basic_table.add_column("Value", style="white")

        basic_table.add_row("Name", config.name)
        basic_table.add_row("Description", config.description or "[dim]None[/dim]")
        basic_table.add_row("Category", config.category.value.title())
        basic_table.add_row(
            "Base Template", config.base_template_id or "[dim]None[/dim]"
        )

        self.console.print(Panel(basic_table, title="Basic Information", box=ROUNDED))

        # Model settings
        model_table = Table(show_header=False, box=None)
        model_table.add_column("Field", style="cyan")
        model_table.add_column("Value", style="white")

        model_table.add_row("Provider", config.model_settings.provider_type.value)
        model_table.add_row("Model", config.model_settings.model_name)
        model_table.add_row("Temperature", str(config.model_settings.temperature))
        model_table.add_row(
            "Max Tokens",
            (
                str(config.model_settings.max_tokens)
                if config.model_settings.max_tokens
                else "[dim]Unlimited[/dim]"
            ),
        )

        self.console.print(Panel(model_table, title="Model Configuration", box=ROUNDED))

        # Tools and capabilities
        tools_text = ", ".join(
            [tool.value.replace("_", " ").title() for tool in config.enabled_tools]
        )
        caps_text = ", ".join(
            [cap.value.replace("_", " ").title() for cap in config.capabilities]
        )

        tools_table = Table(show_header=False, box=None)
        tools_table.add_column("Field", style="cyan")
        tools_table.add_column("Value", style="white")

        tools_table.add_row("Tools", tools_text or "[dim]None[/dim]")
        tools_table.add_row("Capabilities", caps_text or "[dim]None[/dim]")

        self.console.print(
            Panel(tools_table, title="Tools and Capabilities", box=ROUNDED)
        )

    async def _interactive_tool_selection(
        self, all_tools: List[AgentTool], current_selection: Set[AgentTool]
    ) -> Set[AgentTool]:
        """Interactive tool selection interface."""
        selected = set(current_selection)

        while True:
            self.console.print("\\nCurrent tool selection:")
            for i, tool in enumerate(all_tools, 1):
                status = "✅" if tool in selected else "❌"
                self.console.print(
                    f"  {i}. {status} {tool.value.replace('_', ' ').title()}"
                )

            self.console.print(f"\\n  {len(all_tools) + 1}. Done")

            try:
                choice = IntPrompt.ask(
                    "Toggle tool (enter number)",
                    default=len(all_tools) + 1,
                    show_default=False,
                )

                if choice == len(all_tools) + 1:
                    break
                elif 1 <= choice <= len(all_tools):
                    tool = all_tools[choice - 1]
                    if tool in selected:
                        selected.remove(tool)
                    else:
                        selected.add(tool)
                else:
                    self.console.print(
                        f"[red]Please enter a number between 1 and {len(all_tools) + 1}.[/red]"
                    )
            except (ValueError, EOFError):
                self.console.print("[red]Invalid input. Please enter a number.[/red]")

        return selected

    async def _preset_tool_selection(self) -> Set[AgentTool]:
        """Tool selection from presets."""
        presets = {
            "minimal": {AgentTool.FILE_OPERATIONS},
            "standard": {
                AgentTool.FILE_OPERATIONS,
                AgentTool.PROGRAM_EXECUTOR,
                AgentTool.SYSTEM_INFO,
            },
            "web_enabled": {
                AgentTool.FILE_OPERATIONS,
                AgentTool.PROGRAM_EXECUTOR,
                AgentTool.WEB_CLIENT,
                AgentTool.SEARCH,
            },
            "full_featured": set(AgentTool),
        }

        self.console.print("\\nTool presets:")
        preset_names = list(presets.keys())
        for i, name in enumerate(preset_names, 1):
            tool_list = ", ".join(
                [t.value.replace("_", " ").title() for t in presets[name]]
            )
            self.console.print(f"  {i}. {name.title()}: {tool_list}")

        while True:
            try:
                choice = IntPrompt.ask(
                    "Select preset", default=2, show_default=True  # Standard
                )
                if 1 <= choice <= len(preset_names):
                    return presets[preset_names[choice - 1]]
                else:
                    self.console.print(
                        f"[red]Please enter a number between 1 and {len(preset_names)}.[/red]"
                    )
            except (ValueError, EOFError):
                self.console.print("[red]Invalid input. Please enter a number.[/red]")

    async def _interactive_capability_selection(
        self,
        all_capabilities: List[PersonaCapability],
        current_selection: Set[PersonaCapability],
    ) -> Set[PersonaCapability]:
        """Interactive capability selection interface."""
        # Group capabilities for easier selection
        capability_groups = {
            "Core": [
                PersonaCapability.GENERAL_PURPOSE,
                PersonaCapability.REASONING,
                PersonaCapability.ANALYSIS,
            ],
            "Development": [
                PersonaCapability.CODE_GENERATION,
                PersonaCapability.FILE_OPERATIONS,
                PersonaCapability.TOOL_CALLING,
            ],
            "Research": [
                PersonaCapability.RESEARCH,
                PersonaCapability.WEB_SEARCH,
                PersonaCapability.LARGE_CONTEXT,
            ],
            "Creative": [
                PersonaCapability.CREATIVE_WRITING,
                PersonaCapability.HIGH_TEMPERATURE,
            ],
            "Performance": [
                PersonaCapability.FAST_RESPONSE,
                PersonaCapability.COST_EFFICIENT,
                PersonaCapability.BALANCED,
            ],
        }

        selected = set(current_selection)

        # Show grouped capabilities
        for group_name, group_caps in capability_groups.items():
            self.console.print(f"\\n[bold]{group_name} Capabilities:[/bold]")
            for cap in group_caps:
                if cap in all_capabilities:
                    status = "✅" if cap in selected else "❌"
                    self.console.print(
                        f"  {status} {cap.value.replace('_', ' ').title()}"
                    )

        # Simple yes/no for each group
        for group_name, group_caps in capability_groups.items():
            available_caps = [cap for cap in group_caps if cap in all_capabilities]
            if not available_caps:
                continue

            current_count = len([cap for cap in available_caps if cap in selected])

            if Confirm.ask(
                f"\\nEnable {group_name.lower()} capabilities? (currently {current_count}/{len(available_caps)})",
                default=current_count > 0,
            ):
                selected.update(available_caps)
            else:
                selected.difference_update(available_caps)

        return selected

    def _suggest_capabilities_for_tools(
        self, tools: Set[AgentTool]
    ) -> Set[PersonaCapability]:
        """Suggest capabilities based on selected tools."""
        suggestions = {PersonaCapability.GENERAL_PURPOSE}

        if AgentTool.FILE_OPERATIONS in tools:
            suggestions.add(PersonaCapability.FILE_OPERATIONS)
            suggestions.add(PersonaCapability.CODE_GENERATION)

        if AgentTool.WEB_CLIENT in tools or AgentTool.SEARCH in tools:
            suggestions.add(PersonaCapability.WEB_SEARCH)
            suggestions.add(PersonaCapability.RESEARCH)

        if AgentTool.PROGRAM_EXECUTOR in tools:
            suggestions.add(PersonaCapability.TOOL_CALLING)

        if AgentTool.CODE_ANALYSIS in tools:
            suggestions.add(PersonaCapability.ANALYSIS)
            suggestions.add(PersonaCapability.REASONING)

        return suggestions

    def _get_model_suggestions(self, provider_type: ProviderType) -> List[str]:
        """Get model suggestions for a provider."""
        suggestions = {
            ProviderType.ANTHROPIC: [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            ProviderType.OPENAI: [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "gpt-4o",
            ],
            ProviderType.GOOGLE: [
                "gemini-pro",
                "gemini-pro-vision",
                "gemini-1.5-pro",
            ],
            ProviderType.MISTRAL: [
                "mistral-large",
                "mistral-medium",
                "mistral-small",
            ],
        }

        return suggestions.get(provider_type, [])

    def _get_default_model(self, provider_type: ProviderType) -> str:
        """Get default model for a provider."""
        defaults = {
            ProviderType.ANTHROPIC: "claude-3-haiku-20240307",
            ProviderType.OPENAI: "gpt-3.5-turbo",
            ProviderType.GOOGLE: "gemini-pro",
            ProviderType.MISTRAL: "mistral-medium",
        }

        return defaults.get(provider_type, "default-model")


# Utility functions for easy integration


async def create_agent_wizard(
    console: Optional[Console] = None,
) -> AgentCreationWizard:
    """Create and initialize an agent creation wizard."""
    return AgentCreationWizard(console=console)


async def quick_agent_creation(
    name: str,
    provider: str = "anthropic",
    model: str = "claude-3-haiku-20240307",
) -> Optional[CustomAgentConfig]:
    """Quick agent creation with minimal configuration."""
    try:
        provider_type = ProviderType(provider.lower())
    except ValueError:
        provider_type = ProviderType.ANTHROPIC

    config = CustomAgentConfig(
        name=name,
        model_settings=ModelSettings(provider_type=provider_type, model_name=model),
        status=CustomAgentStatus.ACTIVE,
    )

    repository = AgentRepository()
    return repository.create(config)
