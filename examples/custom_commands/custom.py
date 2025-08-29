#!/usr/bin/env python3
"""
Custom command manager - manage dynamic commands
"""

COMMAND_INFO = {
    "name": "custom",
    "description": "Manage custom commands (list, reload, info)",
    "arguments": [
        {
            "name": "action",
            "type": "string",
            "description": "Action to perform",
            "choices": ["list", "reload", "info", "path"],
            "required": False,
            "default": "list",
        },
        {
            "name": "command",
            "type": "string",
            "description": "Command name for info action",
            "required": False,
        },
    ],
}


def handle_command(args, **kwargs):
    """
    Handle the custom command manager.

    Args:
        args: List of command arguments
        kwargs: Additional context

    Returns:
        String message to display
    """
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from pathlib import Path
    import sys
    import os

    # Add parent directory to path to import our modules
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    try:
        from omnimancer.cli.commands import get_command_registry
    except ImportError:
        return "Error: Could not import command registry"

    console = kwargs.get("console")
    action = args[0] if len(args) > 0 else "list"

    registry = get_command_registry()

    if action == "list":
        # List all dynamic commands
        commands = registry.commands

        if not commands:
            if console:
                console.print("[yellow]No custom commands loaded[/yellow]")
                console.print(f"[dim]Commands directory: {registry.commands_dir}[/dim]")
            return "No custom commands loaded"

        if console:
            table = Table(
                title="Custom Commands", show_header=True, header_style="bold magenta"
            )
            table.add_column("Command", style="cyan", no_wrap=True)
            table.add_column("Description", style="green")
            table.add_column("Type", style="yellow")

            for name, cmd in commands.items():
                cmd_type = (
                    "Python" if cmd.handler else "Script" if cmd.script_path else "JSON"
                )
                table.add_row(f"/{name}", cmd.description, cmd_type)

            console.print(table)
            console.print(f"\n[dim]Commands loaded from: {registry.commands_dir}[/dim]")
            return None
        else:
            lines = ["Custom Commands:"]
            for name, cmd in commands.items():
                lines.append(f"  /{name} - {cmd.description}")
            return "\n".join(lines)

    elif action == "reload":
        # Reload commands from directory
        count = registry.load_commands_from_directory()
        msg = f"Reloaded {count} custom commands from {registry.commands_dir}"

        if console:
            console.print(f"[green]✓[/green] {msg}")
            return None
        return msg

    elif action == "info":
        # Show info about a specific command
        if len(args) < 2:
            return "Error: Please specify a command name"

        cmd_name = args[1].lstrip("/")
        cmd = registry.get_command(cmd_name)

        if not cmd:
            return f"Command '/{cmd_name}' not found"

        if console:
            info_text = Text()
            info_text.append(f"Command: ", style="bold")
            info_text.append(f"/{cmd.name}\n")
            info_text.append(f"Description: ", style="bold")
            info_text.append(f"{cmd.description}\n")

            if cmd.arguments:
                info_text.append(f"Arguments:\n", style="bold")
                for arg in cmd.arguments:
                    info_text.append(f"  • {arg.get('name', 'unnamed')}", style="cyan")
                    if arg.get("type"):
                        info_text.append(f" ({arg['type']})", style="dim")
                    if arg.get("description"):
                        info_text.append(f" - {arg['description']}", style="green")
                    if arg.get("choices"):
                        info_text.append(
                            f"\n    Choices: {', '.join(arg['choices'])}",
                            style="yellow",
                        )
                    info_text.append("\n")

            if cmd.script_path:
                info_text.append(f"Script: ", style="bold")
                info_text.append(f"{cmd.script_path}\n", style="blue")

            panel = Panel(
                info_text,
                title=f"[bold blue]Command Info[/bold blue]",
                border_style="blue",
            )
            console.print(panel)
            return None
        else:
            lines = [f"Command: /{cmd.name}", f"Description: {cmd.description}"]
            if cmd.arguments:
                lines.append("Arguments:")
                for arg in cmd.arguments:
                    lines.append(
                        f"  - {arg.get('name', 'unnamed')}: {arg.get('description', '')}"
                    )
            return "\n".join(lines)

    elif action == "path":
        # Show the commands directory path
        path = str(registry.commands_dir)
        if console:
            console.print(f"[cyan]Custom commands directory:[/cyan] {path}")
            if registry.commands_dir.exists():
                console.print(f"[green]✓ Directory exists[/green]")
            else:
                console.print(f"[yellow]⚠ Directory does not exist[/yellow]")
                console.print(f"[dim]Create it with: mkdir -p {path}[/dim]")
            return None
        return f"Commands directory: {path}"

    else:
        return f"Unknown action: {action}. Available: list, reload, info, path"
