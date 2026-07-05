"""
Agent loop mixin for marker-based workflow execution.

Handles parsing operation markers from AI responses and executing them
via the agent engine, with a continuous workflow loop for multi-step tasks.
"""

import asyncio
import ipaddress
import logging
import re
import shlex
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from ..core.agent.types import Operation, OperationType

# ---------------------------------------------------------------------------
# Web request constants and helpers
# ---------------------------------------------------------------------------

#: Maximum characters returned from a single web response.
_MAX_WEB_RESPONSE_CHARS: int = 10_000

#: Hostnames that must never be contacted – cloud/VM metadata endpoints.
_BLOCKED_METADATA_HOSTNAMES: frozenset = frozenset(
    {
        "169.254.169.254",  # AWS IMDSv1/v2, GCP, Azure shared endpoint
        "metadata.google.internal",  # GCP metadata server
        "fd00:ec2::254",  # AWS IPv6 IMDS
        "metadata.internal",  # Azure internal alias
    }
)

#: Compiled patterns for web operation markers.
_WEB_GET_PATTERN = re.compile(r"\[WEB_GET:([^\]]+)\]")
_WEB_REQUEST_PATTERN = re.compile(r"\[WEB_REQUEST:([^\]]+)\]")
_WEB_POST_PATTERN = re.compile(r"\[WEB_POST:([^\]]+)\]")


def _check_url_safety(url: str) -> Optional[str]:
    """Validate whether *url* is safe to request.

    Enforces:
    - HTTP or HTTPS schemes only.
    - No embedded credentials (``user:pass@host``).
    - No loopback, private, link-local, or reserved addresses.
    - No known cloud/VM metadata endpoints.

    Returns an error-message string when the URL is unsafe, ``None`` if OK.
    """
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return "Invalid URL format"

    if parsed.scheme not in ("http", "https"):
        return f"Unsupported scheme '{parsed.scheme}':" " only http/https allowed"

    if parsed.username or parsed.password:
        return "URLs with embedded credentials are not allowed"

    hostname = parsed.hostname
    if not hostname:
        return "URL has no hostname"

    hostname_lower = hostname.lower()

    # Block well-known localhost names before IP parsing.
    if hostname_lower in ("localhost", "::1", "0.0.0.0"):
        return f"Requests to '{hostname}' are not allowed"

    # Block cloud metadata hostnames.
    if hostname_lower in _BLOCKED_METADATA_HOSTNAMES:
        return f"Requests to cloud metadata endpoint '{hostname}' are blocked"

    # Reject private/reserved IP addresses.
    try:
        addr = ipaddress.ip_address(hostname_lower)
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
        ):
            return f"Requests to private/reserved IP '{hostname}'" " are not allowed"
    except ValueError:
        pass  # Hostname is a domain name – that's fine.

    return None  # URL passed all safety checks.


def _sanitize_url_for_log(url: str) -> str:
    """Return *url* with credentials and query-string stripped for safe logging."""
    try:
        parsed = urlparse(url)
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        safe = parsed._replace(netloc=netloc, query="", fragment="")
        return urlunparse(safe)
    except Exception:
        return "<invalid-url>"


logger = logging.getLogger(__name__)


class AgentLoopMixin:
    """Mixin providing marker-based agent workflow execution.

    Expects the host class to have: engine, console, _show_error,
    _show_assistant_message (from DisplayMixin).
    """

    engine: Any
    console: Any

    def _show_assistant_message(self, content: str, model: str) -> None: ...
    def _show_error(self, message: str) -> None: ...

    async def _execute_web_request(
        self,
        marker_text: str,
        url: str,
        http_method: str,
        current_response: str,
    ) -> tuple[str, bool]:
        """Execute a single web-request marker and return the updated response.

        Validates the *url* for safety, requests user approval, issues the
        request via the agent engine, and inlines the result.

        Args:
            marker_text: The raw marker string to replace (e.g. ``[WEB_GET:…]``).
            url: The target URL extracted from the marker.
            http_method: HTTP verb – ``"GET"`` or ``"POST"``.
            current_response: The response text being built up.

        Returns:
            A ``(updated_response, was_cancelled)`` tuple.  When ``was_cancelled``
            is ``True`` the returned string already contains the
            ``__WORKFLOW_CANCELLED__`` sentinel and the caller must return
            immediately.
        """
        safe_url = _sanitize_url_for_log(url)

        url_error = _check_url_safety(url)
        if url_error:
            return (
                current_response.replace(
                    marker_text,
                    f"❌ Web request blocked: {url_error}",
                ),
                False,
            )

        agent_engine = getattr(self.engine, "agent_engine", None)
        if not agent_engine:
            return (
                current_response.replace(
                    marker_text,
                    "❌ Agent engine not available"
                    " - web requests disabled for security",
                ),
                False,
            )

        operation = Operation(
            type=OperationType.WEB_REQUEST,
            description=f"{http_method} request to: {safe_url}",
            data={"url": url, "method": http_method},
            requires_approval=True,
        )

        try:
            result = await agent_engine.execute_with_approval(operation)

            if result.success:
                if isinstance(result.data, dict):
                    content = result.data.get("content", "")
                    if not isinstance(content, str):
                        content = str(content)
                else:
                    content = str(result.data or "")

                if len(content) > _MAX_WEB_RESPONSE_CHARS:
                    content = (
                        content[:_MAX_WEB_RESPONSE_CHARS]
                        + "\n\n[... response truncated ...]"
                    )

                return (
                    current_response.replace(
                        marker_text,
                        f"🌐 {http_method} {safe_url}\n{content}",
                    ),
                    False,
                )
            else:
                if result.was_cancelled:
                    updated = current_response.replace(
                        marker_text,
                        "🚫 Agent workflow cancelled by user",
                    )
                    return updated + "\n\n__WORKFLOW_CANCELLED__", True
                else:
                    return (
                        current_response.replace(
                            marker_text,
                            f"❌ Web request failed: {result.error}",
                        ),
                        False,
                    )

        except Exception as e:
            logger.error(
                f"Web request execution failed: {e}",
                exc_info=True,
            )
            return (
                current_response.replace(
                    marker_text,
                    f"❌ Web request error: {str(e)}",
                ),
                False,
            )

    async def _execute_continuous_workflow(
        self, original_message: str, initial_response: Any
    ) -> None:
        """Execute a continuous workflow.

        Sends AI responses back for more actions
        until complete.
        """
        current_response = initial_response

        while True:
            self._show_assistant_message(
                current_response.content, current_response.model_used
            )

            executed_response = await self._parse_and_execute_operations(
                current_response.content
            )

            if executed_response != current_response.content:
                self.console.print()

                original_lines = set(current_response.content.splitlines())
                executed_lines = executed_response.splitlines()

                for line in executed_lines:
                    if (
                        line.strip().startswith(("✅", "❌"))
                        and line not in original_lines
                    ):
                        self.console.print(line)

            operation_patterns = [
                r"\[FILE_WRITE:[^\]]+\\?\].*?\[/FILE_WRITE\]",
                r"\[FILE_READ:[^\]]+\\?\]",
                r"\[COMMAND_EXEC\].*?\[/COMMAND_EXEC\]",
                r"\[WEB_REQUEST:[^\]]+\]",
                r"\[WEB_GET:[^\]]+\]",
                r"\[WEB_POST:[^\]]+\]",
                r"\[SAFE_EXEC\].*?\[/SAFE_EXEC\]",
            ]

            had_operations = any(
                re.search(pattern, current_response.content, re.DOTALL)
                for pattern in operation_patterns
            )

            if not had_operations:
                break

            if "__WORKFLOW_CANCELLED__" in executed_response:
                executed_response = executed_response.replace(
                    "\n\n__WORKFLOW_CANCELLED__", ""
                )
                self.console.print(
                    "\n[yellow]⚠️  Agent workflow stopped by user[/yellow]"
                )
                break

            continue_message = (
                "I executed the operations."
                " Here are the results:\n\n"
                f"{executed_response}\n\n"
                "What should I do next to complete"
                f" the task: {original_message}"
            )

            try:
                next_response = await self.engine.send_message(continue_message)
            except (asyncio.TimeoutError, ConnectionError, OSError):
                break

            if not next_response.is_success:
                self._show_error(
                    "Workflow continuation failed:" f" {next_response.error}"
                )
                break

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
                self._show_assistant_message(
                    next_response.content, next_response.model_used
                )
                break

            current_response = next_response

    def _is_action_request(self, message: str) -> bool:
        """Determine if a message is asking the AI to perform an action."""
        normalized = message.lower().strip()

        if len(normalized.split()) < 2:
            return False

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

        imperative_patterns = [normalized.startswith(verb) for verb in action_verbs]
        contains_action_verb = any(verb in normalized for verb in action_verbs)

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

        is_action = (
            any(imperative_patterns) or contains_action_verb or contains_action_question
        )

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

    def _fuzzy_find_file(self, filename: str, search_dir: str = ".") -> Optional[str]:
        """Find file with fuzzy matching for typos."""

        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        target = Path(search_dir) / filename
        if target.exists():
            return str(target)

        for item in Path(search_dir).rglob("*"):
            if item.name.lower() == filename.lower():
                return str(item)

        best_match = None
        best_score = 0.7

        for item in Path(search_dir).rglob("*"):
            score = similarity(filename, item.name)
            if score > best_score:
                best_score = score
                best_match = str(item)

        return best_match

    async def _parse_and_execute_operations(self, response_content: str) -> str:
        """Parse model response for operation markers and execute them."""
        updated_response = response_content

        try:
            # FILE_WRITE
            file_write_pattern = r"\[FILE_WRITE:([^\]]+)\\?\](.*?)\[/FILE_WRITE\]"
            for match in re.finditer(file_write_pattern, response_content, re.DOTALL):
                filename = match.group(1).strip()
                content = match.group(2).strip()

                operation = Operation(
                    type=OperationType.FILE_WRITE,
                    description=f"Write file: {filename}",
                    data={
                        "path": filename,
                        "content": content,
                        "autonomous_mode": True,
                    },
                    requires_approval=True,
                )

                if hasattr(self.engine, "agent_engine"):
                    result = await self.engine.agent_engine.execute_with_approval(
                        operation
                    )
                    if result.success:
                        msg = (
                            "✅ Successfully created file"
                            f" '{filename}'"
                            f" ({len(content)} characters)"
                        )
                        updated_response = updated_response.replace(
                            match.group(0),
                            msg,
                        )
                    else:
                        if result.was_cancelled:
                            updated_response = updated_response.replace(
                                match.group(0),
                                "🚫 Agent workflow cancelled by user",
                            )
                            return updated_response + "\n\n__WORKFLOW_CANCELLED__"
                        else:
                            updated_response = updated_response.replace(
                                match.group(0),
                                f"❌ Failed to create file '{filename}': {result.error}",
                            )
                else:
                    try:
                        with open(filename, "w") as f:
                            f.write(content)
                        msg = (
                            "✅ Successfully created file"
                            f" '{filename}'"
                            f" ({len(content)} characters)"
                        )
                        updated_response = updated_response.replace(
                            match.group(0),
                            msg,
                        )
                    except Exception as e:
                        msg = "❌ Failed to create file" f" '{filename}': {str(e)}"
                        updated_response = updated_response.replace(
                            match.group(0),
                            msg,
                        )

            # FILE_READ
            file_read_pattern = r"\[FILE_READ:([^\]]+)\\?\]"
            for match in re.finditer(file_read_pattern, response_content):
                filename = match.group(1).strip()

                actual_file = self._fuzzy_find_file(filename)
                if actual_file:
                    filename = actual_file

                operation = Operation(
                    type=OperationType.FILE_READ,
                    description=f"Read file: {filename}",
                    data={"path": filename},
                    requires_approval=False,
                )

                if hasattr(self.engine, "agent_engine"):
                    result = await self.engine.agent_engine.execute_with_approval(
                        operation
                    )
                    if result.success:
                        file_content = result.data
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
                    try:
                        with open(filename, "r") as f:
                            file_content = f.read()
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

            # FILE_DELETE
            file_delete_pattern = r"\[FILE_DELETE:([^\]]+)\]"
            for match in re.finditer(file_delete_pattern, response_content):
                filename = match.group(1).strip()

                actual_file = self._fuzzy_find_file(filename)
                if actual_file:
                    filename = actual_file

                operation = Operation(
                    type=OperationType.FILE_DELETE,
                    description=f"Delete file: {filename}",
                    data={"path": filename},
                    requires_approval=True,
                )

                if hasattr(self.engine, "agent_engine"):
                    result = await self.engine.agent_engine.execute_with_approval(
                        operation
                    )
                    if result.success:
                        status = "✅ Deleted"
                    elif result.was_cancelled:
                        updated_response = updated_response.replace(
                            match.group(0), "🚫 Agent workflow cancelled by user"
                        )
                        return updated_response + "\n\n__WORKFLOW_CANCELLED__"
                    else:
                        status = f"❌ Failed: {result.error}"

                    updated_response = updated_response.replace(
                        match.group(0), f"{status} '{filename}'"
                    )

            # FIND
            find_pattern = r"\[FIND:([^\]]+)\]"
            for match in re.finditer(find_pattern, response_content):
                pattern = match.group(1).strip()

                try:
                    result = subprocess.run(
                        f"find . -name '{pattern}' -type f 2>/dev/null | head -20",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    files = result.stdout.strip()
                    if files:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"🔍 Found files matching '{pattern}':\n{files}",
                        )
                    else:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"🔍 No files found matching '{pattern}'",
                        )
                except Exception as e:
                    updated_response = updated_response.replace(
                        match.group(0),
                        f"❌ Find failed: {str(e)}",
                    )

            # SEARCH
            search_pattern = r"\[SEARCH:([^\]]+)\]"
            for match in re.finditer(search_pattern, response_content):
                search_text = match.group(1).strip()

                try:
                    result = subprocess.run(
                        f"grep -r -n '{search_text}' . 2>/dev/null | head -20",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    matches = result.stdout.strip()
                    if matches:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"🔍 Found '{search_text}' in:\n{matches}",
                        )
                    else:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"🔍 No matches found for '{search_text}'",
                        )
                except Exception as e:
                    updated_response = updated_response.replace(
                        match.group(0),
                        f"❌ Search failed: {str(e)}",
                    )

            # LOCATE
            locate_pattern = r"\[LOCATE:([^\]]+)\]"
            for match in re.finditer(locate_pattern, response_content):
                filename = match.group(1).strip()

                found_file = self._fuzzy_find_file(filename)
                if found_file:
                    updated_response = updated_response.replace(
                        match.group(0),
                        f"📍 Located: {found_file}",
                    )
                else:
                    updated_response = updated_response.replace(
                        match.group(0),
                        f"❌ Could not locate file similar to '{filename}'",
                    )

            # SAFE_EXEC
            safe_exec_pattern = r"\[SAFE_EXEC\](.*?)\[/SAFE_EXEC\]"
            for match in re.finditer(safe_exec_pattern, response_content, re.DOTALL):
                command = match.group(1).strip()

                safe_commands = [
                    "ls",
                    "cat",
                    "head",
                    "tail",
                    "grep",
                    "find",
                    "pwd",
                    "whoami",
                    "date",
                    "echo",
                    "wc",
                    "sort",
                    "uniq",
                    "which",
                    "type",
                    "file",
                    "stat",
                ]
                cmd_base = command.split()[0] if command.split() else ""

                if cmd_base in safe_commands:
                    try:
                        result = subprocess.run(
                            command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=10,
                            executable="/bin/bash",
                        )
                        output = (
                            result.stdout.strip()
                            if result.returncode == 0
                            else f"Error: {result.stderr}"
                        )
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"✅ `{command}`\n{output}",
                        )
                    except Exception as e:
                        updated_response = updated_response.replace(
                            match.group(0),
                            f"❌ Failed: {str(e)}",
                        )
                else:
                    updated_response = updated_response.replace(
                        match.group(0),
                        f"❌ '{cmd_base}' is not a safe"
                        " command. Use [COMMAND_EXEC]"
                        " for approval workflow.",
                    )

            # COMMAND_EXEC
            command_pattern = (
                r"(?:<!--(read-only|modifies-system)-->"
                r"\s*)?\[COMMAND_EXEC\](.*?)"
                r"\[/COMMAND_EXEC\]"
            )
            for match in re.finditer(command_pattern, response_content, re.DOTALL):
                metadata = match.group(1)
                command = match.group(2).strip()

                requires_approval = metadata != "read-only"

                agent_engine = getattr(self.engine, "agent_engine", None)

                if not agent_engine:
                    updated_response = updated_response.replace(
                        match.group(0),
                        "❌ Agent engine not available"
                        " - command execution disabled"
                        " for security",
                    )
                    continue

                try:
                    try:
                        parts = shlex.split(command)
                        cmd = parts[0] if parts else command
                        args = parts[1:] if len(parts) > 1 else []
                    except ValueError:
                        parts = command.split()
                        cmd = parts[0] if parts else command
                        args = parts[1:] if len(parts) > 1 else []

                    operation = Operation(
                        type=OperationType.COMMAND_EXECUTE,
                        description=f"Execute command: {command[:50]}...",
                        data={"command": cmd, "args": args},
                        requires_approval=requires_approval,
                    )
                    result = await agent_engine.execute_with_approval(operation)

                    if result.success:
                        if result.error and (
                            "not approved" in result.error.lower()
                            or "denied" in result.error.lower()
                        ):
                            deny_msg = (
                                "❌ Command denied:"
                                f" `{command}`\nReason:"
                                f" {result.error}"
                            )
                            updated_response = updated_response.replace(
                                match.group(0),
                                deny_msg,
                            )
                        else:
                            output = (
                                result.data.get("stdout", "")
                                if isinstance(result.data, dict)
                                else str(result.data)
                            )
                            updated_response = updated_response.replace(
                                match.group(0),
                                f"✅ Command executed: `{command}`\nOutput: {output}",
                            )
                    else:
                        if result.was_cancelled:
                            updated_response = updated_response.replace(
                                match.group(0),
                                "🚫 Agent workflow cancelled by user",
                            )
                            return updated_response + "\n\n__WORKFLOW_CANCELLED__"
                        else:
                            updated_response = updated_response.replace(
                                match.group(0),
                                f"❌ Command failed: `{command}`\nError: {result.error}",
                            )
                except Exception as e:
                    logger.error(f"Command execution failed: {e}", exc_info=True)
                    updated_response = updated_response.replace(
                        match.group(0),
                        f"❌ Command execution error: {str(e)}",
                    )

            # WEB_GET, WEB_REQUEST, and WEB_POST markers – each handled via the
            # shared _execute_web_request helper which validates safety, requests
            # user approval, and inlines the response content.
            for marker_pattern, http_method in (
                (_WEB_GET_PATTERN, "GET"),
                (_WEB_REQUEST_PATTERN, "GET"),
                (_WEB_POST_PATTERN, "POST"),
            ):
                for match in marker_pattern.finditer(response_content):
                    url = match.group(1).strip()
                    updated_response, cancelled = await self._execute_web_request(
                        match.group(0), url, http_method, updated_response
                    )
                    if cancelled:
                        return updated_response

        except Exception as e:
            self._show_error(f"Error parsing operations: {e}")
            return response_content

        return updated_response
