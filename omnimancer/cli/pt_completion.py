"""prompt_toolkit completer for the REPL: slash commands, their arguments,
and @-file mentions.

One completer, routed by the token under the cursor:

* line starts with ``/`` and the cursor is in the first token → slash
  command names (built-in enum + dynamic command registry);
* line starts with ``/`` and the cursor is in an argument → the shared
  :class:`~omnimancer.cli.completion.CompletionManager` (static subcommand
  dict plus live provider/model names);
* the current token starts with ``@`` (anywhere in a chat message) →
  project file paths, fuzzy-matched (subsequence), gitignore-aware via
  ``git ls-files`` with an ``os.walk`` fallback.
"""

import logging
import os
import subprocess
import time
from typing import Iterable, List, Optional

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from .commands import SlashCommand, get_command_registry
from .completion import CompletionManager

logger = logging.getLogger(__name__)

#: Maximum @-file completions shown.
MAX_FILE_RESULTS = 20
#: File-candidate cache lifetime (seconds).
FILE_CACHE_TTL = 5.0
#: Directories skipped by the os.walk fallback (git ls-files handles its
#: own exclusions via .gitignore).
WALK_SKIP_DIRS = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}
)


def _subsequence_match(needle: str, haystack: str) -> bool:
    """True if needle's characters appear in order within haystack."""
    it = iter(haystack)
    return all(ch in it for ch in needle)


class OmnimancerCompleter(Completer):
    """Completer for the interactive prompt."""

    def __init__(self, completion_manager: CompletionManager) -> None:
        self.completion_manager = completion_manager
        self._file_cache: Optional[List[str]] = None
        self._file_cache_time = 0.0
        self._file_cache_cwd: Optional[str] = None

    # ------------------------------------------------------------------
    # File candidates
    # ------------------------------------------------------------------

    def invalidate_file_cache(self) -> None:
        self._file_cache = None

    def _list_files(self) -> List[str]:
        cwd = os.getcwd()
        now = time.monotonic()
        if (
            self._file_cache is not None
            and self._file_cache_cwd == cwd
            and now - self._file_cache_time < FILE_CACHE_TTL
        ):
            return self._file_cache

        files = self._git_files(cwd)
        if files is None:
            files = self._walk_files(cwd)

        self._file_cache = files
        self._file_cache_time = now
        self._file_cache_cwd = cwd
        return files

    @staticmethod
    def _git_files(cwd: str) -> Optional[List[str]]:
        """Tracked + untracked-but-not-ignored files, or None outside git."""
        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return [line for line in result.stdout.splitlines() if line]

    @staticmethod
    def _walk_files(cwd: str) -> List[str]:
        files: List[str] = []
        for root, dirs, names in os.walk(cwd):
            dirs[:] = [d for d in dirs if d not in WALK_SKIP_DIRS]
            for name in names:
                rel = os.path.relpath(os.path.join(root, name), cwd)
                files.append(rel)
            if len(files) > 5000:  # runaway-tree guard
                break
        return sorted(files)

    # ------------------------------------------------------------------
    # Completion routing
    # ------------------------------------------------------------------

    def get_completions(
        self, document: Document, complete_event: object
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        # Token under the cursor (whitespace-delimited).
        token = text.rsplit(None, 1)[-1] if text and not text[-1].isspace() else ""

        if token.startswith("@"):
            yield from self._complete_files(token)
            return

        if not text.startswith("/"):
            return

        parts = text.split()
        in_first_token = len(parts) <= 1 and not text[-1].isspace()

        if in_first_token:
            prefix = parts[0] if parts else "/"
            commands = SlashCommand.get_all_commands()
            try:
                commands = commands + get_command_registry().list_commands()
            except Exception:
                pass
            for command in sorted(set(commands)):
                if command.startswith(prefix):
                    yield Completion(command, start_position=-len(prefix))
            return

        command = parts[0]
        if text[-1].isspace():
            args = parts[1:]
            arg_index = len(args)
            current = ""
        else:
            args = parts[1:-1]
            arg_index = len(args)
            current = parts[-1]

        try:
            candidates = self.completion_manager.get_completions(
                command, arg_index, current, args
            )
        except Exception:
            candidates = []
        for candidate in candidates:
            yield Completion(candidate, start_position=-len(current))

    def _complete_files(self, token: str) -> Iterable[Completion]:
        needle = token[1:].lower()
        shown = 0
        for path in self._list_files():
            if needle and not _subsequence_match(needle, path.lower()):
                continue
            yield Completion(f"@{path}", start_position=-len(token))
            shown += 1
            if shown >= MAX_FILE_RESULTS:
                return
