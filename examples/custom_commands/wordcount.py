#!/usr/bin/env python3
"""
Example custom command: Word Count
Analyzes the conversation history and provides statistics.
"""

COMMAND_INFO = {
    "name": "wordcount",
    "description": "Count words in conversation history",
    "arguments": [
        {
            "name": "type",
            "type": "string",
            "description": "What to count",
            "choices": ["words", "messages", "tokens", "all"],
            "required": False,
            "default": "all",
        }
    ],
}


async def handle_command(args, **kwargs):
    """
    Handle the wordcount command.

    Args:
        args: List of command arguments
        kwargs: Additional context (engine, console, etc.)

    Returns:
        String message to display
    """
    from rich.table import Table

    engine = kwargs.get("engine")
    console = kwargs.get("console")

    if not engine:
        return "Error: Engine not available"

    count_type = args[0] if len(args) > 0 else "all"

    # Get conversation history
    try:
        history = (
            engine.conversation_manager.get_current_messages()
            if hasattr(engine, "conversation_manager")
            else []
        )
    except:
        history = []

    # Calculate statistics
    total_messages = len(history)
    user_messages = sum(1 for msg in history if msg.get("role") == "user")
    assistant_messages = sum(1 for msg in history if msg.get("role") == "assistant")

    total_words = 0
    user_words = 0
    assistant_words = 0

    for msg in history:
        content = msg.get("content", "")
        word_count = len(content.split())
        total_words += word_count

        if msg.get("role") == "user":
            user_words += word_count
        elif msg.get("role") == "assistant":
            assistant_words += word_count

    # Estimate tokens (rough approximation: 1 token ≈ 0.75 words)
    estimated_tokens = int(total_words / 0.75)

    if console and count_type == "all":
        # Create a rich table
        table = Table(
            title="Conversation Statistics",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("User", style="green", justify="right")
        table.add_column("Assistant", style="blue", justify="right")
        table.add_column("Total", style="yellow", justify="right")

        table.add_row(
            "Messages",
            str(user_messages),
            str(assistant_messages),
            str(total_messages),
        )
        table.add_row("Words", str(user_words), str(assistant_words), str(total_words))
        table.add_row(
            "Avg Words/Msg",
            str(round(user_words / max(user_messages, 1), 1)),
            str(round(assistant_words / max(assistant_messages, 1), 1)),
            str(round(total_words / max(total_messages, 1), 1)),
        )

        console.print(table)
        console.print(f"\n[dim]Estimated tokens: ~{estimated_tokens}[/dim]")
        return None

    # Return specific count based on type
    if count_type == "words":
        return f"Total words: {total_words}"
    elif count_type == "messages":
        return f"Total messages: {total_messages}"
    elif count_type == "tokens":
        return f"Estimated tokens: ~{estimated_tokens}"
    else:
        return f"Messages: {total_messages}, Words: {total_words}, Tokens: ~{estimated_tokens}"
