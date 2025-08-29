#!/usr/bin/env python3
"""
Example custom command: Greet
A friendly greeting command that demonstrates how to create custom commands.
"""

COMMAND_INFO = {
    "name": "greet",
    "description": "A friendly greeting command",
    "arguments": [
        {
            "name": "name",
            "type": "string",
            "description": "Name to greet",
            "required": False,
            "default": "friend",
        },
        {
            "name": "style",
            "type": "string",
            "description": "Greeting style",
            "choices": ["formal", "casual", "enthusiastic"],
            "required": False,
            "default": "casual",
        },
    ],
}


def handle_command(args, **kwargs):
    """
    Handle the greet command.

    Args:
        args: List of command arguments
        kwargs: Additional context (engine, console, etc.)

    Returns:
        String message to display
    """
    from rich.panel import Panel
    from rich.text import Text

    console = kwargs.get("console")

    # Parse arguments
    name = args[0] if len(args) > 0 else "friend"
    style = args[1] if len(args) > 1 else "casual"

    # Generate greeting based on style
    greetings = {
        "formal": f"Good day, {name}. How may I assist you today?",
        "casual": f"Hey there, {name}! What's up?",
        "enthusiastic": f"🎉 HELLO {name.upper()}! SO GREAT TO SEE YOU! 🎉",
    }

    greeting = greetings.get(style, greetings["casual"])

    # Display using rich formatting if console is available
    if console:
        greeting_text = Text(greeting, style="bold cyan")
        panel = Panel(
            greeting_text, title="[bold blue]Greeting[/bold blue]", border_style="blue"
        )
        console.print(panel)
        return None  # Console already printed
    else:
        return greeting
