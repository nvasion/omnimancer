"""Coding agent tool definitions for native tool calling.

Tool names and input schemas mirror Claude Code's conventions (Read, Write,
Edit, Bash, Glob, Grep, WebFetch) so models call them the way they expect.
"""

from ..models import ToolDefinition

CODING_AGENT_TOOLS = [
    ToolDefinition(
        name="Read",
        description=(
            "Read a file from the local filesystem. Returns the file contents. "
            "Use offset/limit to read a slice of a large file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from (1-based)",
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read",
                },
            },
            "required": ["file_path"],
        },
        auto_approved=True,
    ),
    ToolDefinition(
        name="Write",
        description=(
            "Write a file to the local filesystem. Creates the file if it does "
            "not exist and overwrites it if it does."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        },
    ),
    ToolDefinition(
        name="Edit",
        description=(
            "Perform an exact string replacement in a file. The old_string must "
            "match the file contents exactly and be unique unless replace_all is "
            "set."
        ),
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace",
                },
                "new_string": {
                    "type": "string",
                    "description": (
                        "The text to replace it with (must differ from old_string)"
                    ),
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default false)",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    ),
    ToolDefinition(
        name="Bash",
        description=(
            "Execute a bash command and return its stdout, stderr, and exit code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Clear, concise description of what this command does in "
                        "5-10 words, in active voice"
                    ),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Set to true to run this command in the background",
                },
            },
            "required": ["command"],
        },
    ),
    ToolDefinition(
        name="Glob",
        description=(
            "Fast file pattern matching. Returns paths of files matching a glob "
            "pattern (e.g. '**/*.py', 'src/**/*.ts')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match files against",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The directory to search in (defaults to the current "
                        "working directory)"
                    ),
                },
            },
            "required": ["pattern"],
        },
        auto_approved=True,
    ),
    ToolDefinition(
        name="Grep",
        description=(
            "Search file contents using a regular expression. Returns matching "
            "file paths by default; use output_mode='content' for matching lines."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "The regular expression to search for (POSIX extended "
                        "regex: alternation, +, ? and groups work; use [0-9] "
                        "rather than \\d)"
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (defaults to cwd)",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob to filter files (e.g. '*.py', '*.{ts,tsx}')",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "content shows matching lines, files_with_matches shows "
                        "file paths (default), count shows match counts"
                    ),
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case-insensitive search",
                },
                "-n": {
                    "type": "boolean",
                    "description": (
                        "Show line numbers (output_mode='content' only, default true)"
                    ),
                },
                "-C": {
                    "type": "integer",
                    "description": "Lines of context before and after each match",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Limit output to the first N lines/entries",
                },
            },
            "required": ["pattern"],
        },
        auto_approved=True,
    ),
    ToolDefinition(
        name="WebFetch",
        description="Fetch content from a URL and return the response body.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "What to extract or look for in the fetched content"
                    ),
                },
            },
            "required": ["url"],
        },
        auto_approved=True,
    ),
]
