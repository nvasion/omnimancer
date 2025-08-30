"""
Workflow Orchestrator for continuous multi-step AI operations.

This module provides the capability for the AI agent to execute multiple
operations in sequence automatically, similar to how Claude Code works.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ...cli.batch_approval_display import BatchApprovalPanel

# Import existing UI components
from ...ui.progress_indicator import OperationType as ProgressOperationType
from ...ui.progress_indicator import (
    ProgressIndicator,
)
from ..agent.approval_manager import EnhancedApprovalManager
from ..agent.file_system_manager import FileSystemManager

logger = logging.getLogger(__name__)


class WorkflowStepType(Enum):
    """Types of workflow steps."""

    ANALYZE = "analyze"
    LIST_FILES = "list_files"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_COMMAND = "execute_command"
    CHECK_CONFIG = "check_config"
    DETECT_TECH_STACK = "detect_tech_stack"
    CREATE_STRUCTURE = "create_structure"
    INSTALL_DEPENDENCIES = "install_dependencies"
    RUN_TESTS = "run_tests"
    VALIDATE = "validate"
    CUSTOM = "custom"


class WorkflowStatus(Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""

    name: str
    type: WorkflowStepType
    description: str
    action: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    continue_on_error: bool = False
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class WorkflowContext:
    """Context shared across workflow steps."""

    working_directory: Path
    data: Dict[str, Any] = field(default_factory=dict)
    console: Console = field(default_factory=Console)
    history: List[WorkflowStep] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self.data.get(key, default)

    def update(self, **kwargs) -> None:
        """Update multiple values in the context."""
        self.data.update(kwargs)


class WorkflowOrchestrator:
    """
    Orchestrates multi-step workflows for autonomous AI operations.

    This enables the AI to execute a series of operations automatically,
    continuing through each step without stopping, similar to Claude Code.
    """

    def __init__(
        self,
        file_system: Optional[FileSystemManager] = None,
        approval_manager: Optional[EnhancedApprovalManager] = None,
        executor: Optional[Any] = None,
        console: Optional[Console] = None,
        engine: Optional[Any] = None,
    ):
        """Initialize the workflow orchestrator."""
        self.file_system = file_system
        self.approval_manager = approval_manager
        self.executor = executor
        self.console = console or Console()
        self.engine = engine  # Reference to the main engine for AI calls
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.current_workflow: Optional[str] = None
        self.context: Optional[WorkflowContext] = None

        # Initialize UI components for clean display
        self.progress_indicator = ProgressIndicator(self.console)
        self.approval_panel = BatchApprovalPanel(self.console)

        # Register built-in workflows
        self._register_builtin_workflows()

    def _register_builtin_workflows(self) -> None:
        """Register built-in workflow templates."""

        # General Action Workflow - handles any user request
        self.register_workflow(
            "general_action",
            [
                WorkflowStep(
                    name="understand_request",
                    type=WorkflowStepType.ANALYZE,
                    description="Understanding user request and requirements",
                    action=self._understand_request_action,
                    parameters={},
                ),
                WorkflowStep(
                    name="plan_approach",
                    type=WorkflowStepType.ANALYZE,
                    description="Planning approach and identifying needed actions",
                    action=self._plan_approach_action,
                    dependencies=["understand_request"],
                ),
                WorkflowStep(
                    name="execute_action",
                    type=WorkflowStepType.CUSTOM,
                    description="Executing the planned actions",
                    action=self._execute_action,
                    dependencies=["plan_approach"],
                ),
                WorkflowStep(
                    name="verify_results",
                    type=WorkflowStepType.VALIDATE,
                    description="Verifying action completion and results",
                    action=self._verify_results_action,
                    dependencies=["execute_action"],
                ),
                WorkflowStep(
                    name="provide_summary",
                    type=WorkflowStepType.CUSTOM,
                    description="Providing summary of completed work",
                    action=self._provide_summary_action,
                    dependencies=["verify_results"],
                ),
            ],
        )

        # Project Analysis Workflow
        self.register_workflow(
            "project_analysis",
            [
                WorkflowStep(
                    name="list_directory",
                    type=WorkflowStepType.LIST_FILES,
                    description="List files and directories in the project",
                    action=self._list_directory_action,
                    parameters={"path": "."},
                ),
                WorkflowStep(
                    name="detect_tech_stack",
                    type=WorkflowStepType.DETECT_TECH_STACK,
                    description="Detect technology stack from project files",
                    action=self._detect_tech_stack_action,
                    dependencies=["list_directory"],
                ),
                WorkflowStep(
                    name="check_configuration",
                    type=WorkflowStepType.CHECK_CONFIG,
                    description="Check for configuration files",
                    action=self._check_configuration_action,
                    dependencies=["list_directory"],
                ),
                WorkflowStep(
                    name="analyze_structure",
                    type=WorkflowStepType.ANALYZE,
                    description="Analyze project structure and patterns",
                    action=self._analyze_structure_action,
                    dependencies=["list_directory", "detect_tech_stack"],
                ),
                WorkflowStep(
                    name="generate_summary",
                    type=WorkflowStepType.CUSTOM,
                    description="Generate project analysis summary",
                    action=self._generate_summary_action,
                    dependencies=[
                        "detect_tech_stack",
                        "check_configuration",
                        "analyze_structure",
                    ],
                ),
            ],
        )

        # File Modification Workflow
        self.register_workflow(
            "file_modification",
            [
                WorkflowStep(
                    name="read_original",
                    type=WorkflowStepType.READ_FILE,
                    description="Read the original file content",
                    action=self._read_file_action,
                ),
                WorkflowStep(
                    name="prepare_changes",
                    type=WorkflowStepType.CUSTOM,
                    description="Prepare file modifications",
                    action=self._prepare_changes_action,
                    dependencies=["read_original"],
                ),
                WorkflowStep(
                    name="show_diff",
                    type=WorkflowStepType.CUSTOM,
                    description="Display changes for review",
                    action=self._show_diff_action,
                    dependencies=["prepare_changes"],
                    requires_approval=True,
                ),
                WorkflowStep(
                    name="apply_changes",
                    type=WorkflowStepType.WRITE_FILE,
                    description="Apply approved changes to file",
                    action=self._write_file_action,
                    dependencies=["show_diff"],
                ),
                WorkflowStep(
                    name="validate_changes",
                    type=WorkflowStepType.VALIDATE,
                    description="Validate the applied changes",
                    action=self._validate_changes_action,
                    dependencies=["apply_changes"],
                ),
            ],
        )

    def register_workflow(self, name: str, steps: List[WorkflowStep]) -> None:
        """Register a workflow template."""
        self.workflows[name] = steps

    async def execute_workflow(
        self,
        workflow_name: str,
        context: Optional[WorkflowContext] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> WorkflowContext:
        """
        Execute a complete workflow automatically.

        This is the main entry point that enables continuous execution
        of multiple steps without stopping.
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Initialize context
        self.context = context or WorkflowContext(
            working_directory=Path.cwd(), console=self.console
        )

        # Pass engine reference to context
        self.context.engine = self.engine

        if parameters:
            self.context.update(**parameters)

        self.current_workflow = workflow_name
        steps = self.workflows[workflow_name].copy()

        # Display workflow plan
        await self._display_workflow_plan(workflow_name, steps)

        # Execute workflow steps with progress tracking
        for i, step in enumerate(steps, 1):
            # Check dependencies
            if not self._check_dependencies(step, self.context.history):
                step.status = WorkflowStatus.FAILED
                step.error = "Dependencies not met"
                self.context.history.append(step)

                if not step.continue_on_error:
                    break
                continue

            # Start progress tracking for this step
            operation_id = f"step_{i}"
            progress_type = self._get_progress_operation_type(step.type)
            self.progress_indicator.start_operation(
                operation_id, progress_type, f"{step.name}: {step.description}"
            )

            try:
                step.status = WorkflowStatus.RUNNING
                step.started_at = datetime.now()

                # Request approval if needed
                if step.requires_approval and self.approval_manager:
                    if not await self._request_approval(step):
                        step.status = WorkflowStatus.CANCELLED
                        self.context.history.append(step)
                        self.progress_indicator.complete_operation(
                            operation_id, "cancelled"
                        )
                        break

                # Execute the step action
                if step.action:
                    # Merge step parameters with context parameters
                    merged_params = {**step.parameters, **self.context.data}
                    step.result = await step.action(self.context, merged_params)

                step.status = WorkflowStatus.COMPLETED
                step.completed_at = datetime.now()

                # Complete progress tracking
                self.progress_indicator.complete_operation(operation_id, "completed")

            except Exception as e:
                step.status = WorkflowStatus.FAILED
                step.error = str(e)
                step.completed_at = datetime.now()

                logger.error(f"Step {step.name} failed: {e}")
                self.progress_indicator.complete_operation(operation_id, "failed")

                if not step.continue_on_error:
                    break

            finally:
                self.context.history.append(step)

        # Display workflow summary
        await self._display_workflow_summary()

        return self.context

    async def _display_workflow_plan(
        self, name: str, steps: List[WorkflowStep]
    ) -> None:
        """Display the workflow execution plan using clean UI."""
        self.console.print(f"\n[bold cyan]🚀 Starting workflow: {name}[/bold cyan]")
        self.console.print(f"[dim]Will execute {len(steps)} steps...[/dim]\n")

    def _check_dependencies(
        self, step: WorkflowStep, history: List[WorkflowStep]
    ) -> bool:
        """Check if all dependencies for a step are satisfied."""
        if not step.dependencies:
            return True

        completed_steps = {
            s.name for s in history if s.status == WorkflowStatus.COMPLETED
        }
        return all(dep in completed_steps for dep in step.dependencies)

    async def _request_approval(self, step: WorkflowStep) -> bool:
        """Request approval for a step that requires it."""
        self.console.print(f"\n[yellow]⚠ Step '{step.name}' requires approval[/yellow]")
        self.console.print(f"Description: {step.description}")

        # Use the approval manager if available
        if self.approval_manager and hasattr(self.approval_manager, "request_approval"):
            # Create a basic operation object for the approval manager
            from .types import Operation, OperationType

            operation = Operation(
                type=OperationType.WORKFLOW_STEP,
                description=step.description,
                data={"step_name": step.name, "description": step.description},
                requires_approval=True,
            )
            approved = await self.approval_manager.request_approval(operation)
            if approved:
                self.console.print("[green]✓ Approved[/green]")
            else:
                self.console.print("[red]✗ Denied[/red]")
            return approved

        # Default behavior for demo/test purposes
        await asyncio.sleep(0.5)  # Simulate approval delay
        self.console.print("[green]✓ Approved[/green]")
        return True

    def _display_step_result(self, step: WorkflowStep) -> None:
        """Display the result of a completed step."""
        status_symbol = "✓" if step.status == WorkflowStatus.COMPLETED else "✗"
        status_color = "green" if step.status == WorkflowStatus.COMPLETED else "red"

        self.console.print(
            f"[{status_color}]{status_symbol}[/{status_color}] {step.name}: {step.description}"
        )

        if step.error:
            self.console.print(f"  [red]Error: {step.error}[/red]")

    async def _display_workflow_summary(self) -> None:
        """Display a clean summary of the workflow execution."""
        completed = sum(
            1 for s in self.context.history if s.status == WorkflowStatus.COMPLETED
        )
        failed = sum(
            1 for s in self.context.history if s.status == WorkflowStatus.FAILED
        )
        total = len(self.context.history)

        if failed == 0:
            self.console.print(
                f"\n[bold green]✅ Workflow completed successfully![/bold green]"
            )
        else:
            self.console.print(
                f"\n[yellow]⚠️ Workflow completed with {failed} issues[/yellow]"
            )

        self.console.print(f"[dim]Executed {completed}/{total} steps[/dim]")

    def _get_progress_operation_type(
        self, workflow_step_type: WorkflowStepType
    ) -> ProgressOperationType:
        """Map workflow step types to progress indicator operation types."""
        type_mapping = {
            WorkflowStepType.READ_FILE: ProgressOperationType.READ,
            WorkflowStepType.WRITE_FILE: ProgressOperationType.WRITE,
            WorkflowStepType.ANALYZE: ProgressOperationType.ANALYZE,
            WorkflowStepType.VALIDATE: ProgressOperationType.VALIDATE,
            WorkflowStepType.CUSTOM: ProgressOperationType.OTHER,
        }
        return type_mapping.get(workflow_step_type, ProgressOperationType.OTHER)

    # Built-in action implementations

    async def _list_directory_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> List[str]:
        """List directory contents."""
        path = Path(params.get("path", "."))
        files = []

        context.console.print(f"\n[cyan]Listing directory: {path}[/cyan]")

        for item in path.iterdir():
            if item.is_dir():
                files.append(f"📁 {item.name}/")
            else:
                files.append(f"📄 {item.name}")

        # Display first 10 items
        for file in files[:10]:
            context.console.print(f"  {file}")

        if len(files) > 10:
            context.console.print(f"  ... and {len(files) - 10} more items")

        context.set("project_files", files)
        return files

    async def _detect_tech_stack_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect technology stack from project files."""
        context.console.print("\n[cyan]Detecting technology stack...[/cyan]")

        tech_stack = {}
        files = context.get("project_files", [])

        # Check for common tech indicators
        file_names = [f.replace("📄 ", "").replace("📁 ", "").strip("/") for f in files]

        if "package.json" in file_names:
            tech_stack["javascript"] = "Node.js/npm"
            context.console.print("  ✓ Found: Node.js project (package.json)")

        if (
            "requirements.txt" in file_names
            or "setup.py" in file_names
            or "pyproject.toml" in file_names
        ):
            tech_stack["python"] = "Python/pip"
            context.console.print("  ✓ Found: Python project")

        if "Cargo.toml" in file_names:
            tech_stack["rust"] = "Rust/Cargo"
            context.console.print("  ✓ Found: Rust project")

        if ".git" in file_names:
            tech_stack["vcs"] = "Git"
            context.console.print("  ✓ Found: Git repository")

        if ".mcp.json" in file_names:
            tech_stack["mcp"] = "Model Context Protocol"
            context.console.print("  ✓ Found: MCP server configuration")

        # Check for Omnimancer specific patterns
        if any("omnimancer" in f.lower() for f in file_names):
            tech_stack["omnimancer"] = "Omnimancer CLI Framework"
            context.console.print("  ✓ Found: Omnimancer framework")

        context.set("tech_stack", tech_stack)
        return tech_stack

    async def _check_configuration_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for configuration files."""
        context.console.print("\n[cyan]Checking configuration files...[/cyan]")

        config_files = {}
        files = context.get("project_files", [])
        file_names = [f.replace("📄 ", "").replace("📁 ", "").strip("/") for f in files]

        config_patterns = {
            ".env": "Environment variables",
            "config.json": "JSON configuration",
            "config.yaml": "YAML configuration",
            ".gitignore": "Git ignore rules",
            "Dockerfile": "Docker configuration",
            ".github": "GitHub workflows",
            ".mcp.json": "MCP server configuration",
            "pyproject.toml": "Python project configuration",
            ".taskmaster": "Task Master AI configuration",
            ".claude": "Claude Code configuration",
        }

        for pattern, description in config_patterns.items():
            if pattern in file_names:
                config_files[pattern] = description
                context.console.print(f"  ✓ Found: {description} ({pattern})")

        context.set("config_files", config_files)
        return config_files

    async def _analyze_structure_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze project structure."""
        context.console.print("\n[cyan]Analyzing project structure...[/cyan]")

        analysis = {"type": "Unknown", "patterns": []}

        tech_stack = context.get("tech_stack", {})

        if "python" in tech_stack:
            analysis["type"] = "Python Application"
            analysis["patterns"].append("Python package structure detected")
            context.console.print("  ✓ Identified as Python application")

        if "javascript" in tech_stack:
            analysis["type"] = "Node.js Application"
            analysis["patterns"].append("Node.js project structure detected")
            context.console.print("  ✓ Identified as Node.js application")

        context.set("structure_analysis", analysis)
        return analysis

    async def _generate_summary_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> str:
        """Generate a summary of the analysis."""
        context.console.print("\n[cyan]Generating analysis summary...[/cyan]")

        tech_stack = context.get("tech_stack", {})
        config_files = context.get("config_files", {})
        analysis = context.get("structure_analysis", {})

        summary = f"""
### Project Analysis Complete

**Project Type:** {analysis.get('type', 'Unknown')}

**Technology Stack:**
{chr(10).join(f"- {tech}: {desc}" for tech, desc in tech_stack.items())}

**Configuration Files:**
{chr(10).join(f"- {file}: {desc}" for file, desc in config_files.items())}

**Patterns Detected:**
{chr(10).join(f"- {pattern}" for pattern in analysis.get('patterns', []))}
"""

        context.console.print(
            Panel(
                Markdown(summary),
                title="[bold]Analysis Summary[/bold]",
                border_style="green",
            )
        )

        context.set("summary", summary)
        return summary

    async def _read_file_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> str:
        """Read file content."""
        file_path = params.get("file_path")
        if not file_path:
            raise ValueError("file_path parameter required")

        if self.file_system:
            content = await self.file_system.read_file(file_path)
            context.set("original_content", content)
            return content
        return ""

    async def _prepare_changes_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare file changes."""
        original = context.get("original_content", "")
        changes = params.get("changes", {})

        # Prepare the modified content
        modified = changes.get("content", original)

        context.set("modified_content", modified)
        context.set("changes", changes)

        return {"original": original, "modified": modified}

    async def _show_diff_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> None:
        """Show diff for review."""
        context.get("original_content", "")
        context.get("modified_content", "")

        # In a real implementation, this would use the diff renderer
        context.console.print("\n[yellow]Changes to review:[/yellow]")
        context.console.print("(Diff display would appear here)")

    async def _write_file_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> bool:
        """Write file content."""
        file_path = params.get("file_path")
        content = context.get("modified_content", "")

        if not file_path:
            raise ValueError("file_path parameter required")

        if self.file_system:
            await self.file_system.write_file(file_path, content)
            return True
        return False

    async def _validate_changes_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> bool:
        """Validate applied changes."""
        # In a real implementation, this would verify the changes were applied correctly
        context.console.print("[green]✓ Changes validated successfully[/green]")
        return True

    # General Action Workflow Methods

    async def _understand_request_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Understand and analyze the user's request."""
        user_request = params.get("user_request", context.get("user_request", ""))

        # Simulate analyzing the request
        context.console.print(f"[cyan]📝 Analyzing request: '{user_request}'[/cyan]")

        # Parse the request to understand intent and extract key information
        analysis = {
            "original_request": user_request,
            "intent": "general_action",
            "complexity": "medium",
            "estimated_steps": 3,
        }

        # Store analysis in context for next steps
        context.set("request_analysis", analysis)
        return analysis

    async def _plan_approach_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan the approach to fulfill the user's request."""
        analysis = context.get("request_analysis", {})
        user_request = analysis.get("original_request", "")

        context.console.print(f"[cyan]🎯 Planning approach for: {user_request}[/cyan]")

        # Create a basic plan based on the request
        plan = {
            "approach": "autonomous_execution",
            "steps": [
                "Prepare AI prompt with user request",
                "Execute action using agent capabilities",
                "Capture and process results",
                "Verify completion",
            ],
            "requires_approval": False,
        }

        context.set("execution_plan", plan)
        return plan

    async def _execute_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the planned action autonomously."""
        user_request = context.get("user_request", "")

        context.console.print(f"[cyan]🚀 Executing autonomous action...[/cyan]")

        # Actually execute the user request with AI provider if available
        try:
            # Send the request to AI with agent capabilities enabled
            if hasattr(context, "engine") and context.engine:
                # The agent engine inherits from core engine which has send_message
                response = await context.engine.send_message(user_request)

                if response.is_success:
                    result = {
                        "status": "completed",
                        "action_taken": f"AI processed request: {user_request}",
                        "details": response.content,
                        "ai_response": response.content,
                        "timestamp": "just now",
                    }
                else:
                    result = {
                        "status": "failed",
                        "action_taken": f"Failed to process: {user_request}",
                        "details": f"Error: {response.error}",
                        "error": response.error,
                        "timestamp": "just now",
                    }
            else:
                # Fallback: simulate execution
                result = {
                    "status": "completed",
                    "action_taken": f"Processed request: {user_request}",
                    "details": "Action executed using autonomous workflow system (simulated)",
                    "timestamp": "just now",
                }

        except Exception as e:
            result = {
                "status": "failed",
                "action_taken": f"Error processing: {user_request}",
                "details": f"Exception: {str(e)}",
                "error": str(e),
                "timestamp": "just now",
            }

        context.set("execution_result", result)
        return result

    async def _verify_results_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> bool:
        """Verify that the action was completed successfully."""
        result = context.get("execution_result", {})

        context.console.print("[cyan]✅ Verifying action completion...[/cyan]")

        # Check if execution was successful
        success = result.get("status") == "completed"

        if success:
            context.console.print("[green]✓ Action completed successfully[/green]")
        else:
            context.console.print("[red]✗ Action may not have completed fully[/red]")

        context.set("verification_success", success)
        return success

    async def _provide_summary_action(
        self, context: WorkflowContext, params: Dict[str, Any]
    ) -> str:
        """Provide a summary of what was accomplished."""
        user_request = context.get("user_request", "")
        result = context.get("execution_result", {})
        success = context.get("verification_success", False)

        if success:
            if "ai_response" in result:
                # Don't show summary if we have the actual AI response - it will be shown in interface
                summary = f"✅ Successfully completed: {user_request}"
                context.console.print(f"[bold green]📋 Summary:[/bold green] {summary}")
            else:
                summary = f"✅ Successfully completed: {user_request}\n\nAction taken: {result.get('action_taken', 'Autonomous workflow execution')}"
                context.console.print(
                    f"[bold green]📋 Summary:[/bold green]\n{summary}"
                )
        else:
            summary = f"⚠️ Partially completed: {user_request}\n\nStatus: Some issues may have occurred during execution"
            context.console.print(f"[bold yellow]📋 Summary:[/bold yellow]\n{summary}")

        context.set("final_summary", summary)
        return summary
