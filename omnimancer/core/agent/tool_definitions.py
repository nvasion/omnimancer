"""Coding agent tool definitions for native tool calling."""

from ..models import ToolDefinition

CODING_AGENT_TOOLS = [
    ToolDefinition(
        name="file_read",
        description="Read the contents of a file at the given path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path to read",
                }
            },
            "required": ["path"],
        },
        auto_approved=True,
    ),
    ToolDefinition(
        name="file_write",
        description=(
            "Write content to a file. Creates the file "
            "if it doesn't exist, overwrites if it does."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to write to",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    ),
    ToolDefinition(
        name="file_delete",
        description="Delete a file at the given path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to delete",
                }
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="command_exec",
        description=(
            "Execute a shell command and return its "
            "stdout, stderr, and exit code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command (optional)",
                },
            },
            "required": ["command"],
        },
    ),
    ToolDefinition(
        name="find_files",
        description="Find files matching a glob pattern recursively.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern to match "
                        "(e.g. '**/*.py', 'src/**/*.ts')"
                    ),
                },
                "directory": {
                    "type": "string",
                    "description": (
                        "Root directory to search from "
                        "(optional, defaults to current "
                        "directory)"
                    ),
                },
            },
            "required": ["pattern"],
        },
        auto_approved=True,
    ),
    ToolDefinition(
        name="search_text",
        description=(
            "Search for text or regex pattern in files. "
            "Returns matching lines with file paths "
            "and line numbers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text or regex pattern to search for",
                },
                "directory": {
                    "type": "string",
                    "description": (
                        "Directory to search in "
                        "(optional, defaults to current "
                        "directory)"
                    ),
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py')",
                },
            },
            "required": ["pattern"],
        },
        auto_approved=True,
    ),
    ToolDefinition(
        name="web_request",
        description="Make an HTTP request and return the response.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to request",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, POST, etc.)",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                },
                "body": {
                    "type": "string",
                    "description": "Request body (for POST/PUT/PATCH)",
                },
            },
            "required": ["url"],
        },
    ),
]
