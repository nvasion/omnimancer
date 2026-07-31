"""@-file mentions: expand ``@path`` tokens into injected file content.

Pure functions, called from the chat pipeline after the user's message is
displayed and before it is sent — both the native tool path and the marker
fallback receive the expanded message.

Safety and false-positive rules:

* Injection happens only when the path exists inside the project directory
  — ``user@example.com`` or a casual ``@mention`` never triggers.
* Files above ``max_bytes`` and binary files (NUL byte in the first 8 KB)
  are skipped with a reason the CLI can surface.
* Directories inject a one-level listing.
* The original ``@path`` token stays in the message body for referential
  clarity; content is appended after it in fenced sections.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

#: ``@`` at start or after whitespace, capturing a quoted or bare path.
_MENTION_RE = re.compile(r'(?:(?<=\s)|^)@(?:"([^"]+)"|(\S+))')

#: Default cap on injected file content.
DEFAULT_MAX_BYTES = 65_536

#: Extension → fence language for syntax highlighting on re-render.
_FENCE_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".tf": "hcl",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".java": "java",
    ".rb": "ruby",
}


@dataclass
class MentionResult:
    """Outcome of one @-mention in a message."""

    path: str
    injected: bool
    reason: Optional[str] = None


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(8192)
    except OSError:
        return True


def _inside(project_dir: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(project_dir.resolve())
        return True
    except ValueError:
        return False


def expand_file_mentions(
    message: str,
    project_dir: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Tuple[str, List[MentionResult]]:
    """Expand @path mentions in ``message`` against ``project_dir``.

    Returns:
        (expanded_message, mention_results). The message is returned
        unchanged when nothing injectable is found.
    """
    project_dir = Path(project_dir)
    sections: List[str] = []
    results: List[MentionResult] = []
    seen: set = set()

    for match in _MENTION_RE.finditer(message):
        raw_path = match.group(1) or match.group(2)
        # Strip common trailing punctuation from prose ("see @main.py.")
        raw_path = raw_path.rstrip(".,;:!?")
        if not raw_path or raw_path in seen:
            continue

        target = (project_dir / raw_path).expanduser()
        if not target.exists():
            continue
        seen.add(raw_path)

        if not _inside(project_dir, target):
            results.append(
                MentionResult(
                    path=raw_path,
                    injected=False,
                    reason="outside the project directory",
                )
            )
            continue

        if target.is_dir():
            try:
                entries = sorted(p.name for p in target.iterdir())
            except OSError as e:
                results.append(
                    MentionResult(path=raw_path, injected=False, reason=str(e))
                )
                continue
            listing = "\n".join(entries)
            sections.append(f"--- @{raw_path} (directory listing) ---\n{listing}")
            results.append(MentionResult(path=raw_path, injected=True))
            continue

        try:
            size = target.stat().st_size
        except OSError as e:
            results.append(MentionResult(path=raw_path, injected=False, reason=str(e)))
            continue
        if size > max_bytes:
            results.append(
                MentionResult(
                    path=raw_path,
                    injected=False,
                    reason=f"too large ({size} bytes > {max_bytes})",
                )
            )
            continue
        if _is_binary(target):
            results.append(
                MentionResult(path=raw_path, injected=False, reason="binary file")
            )
            continue

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            results.append(MentionResult(path=raw_path, injected=False, reason=str(e)))
            continue

        language = _FENCE_LANGUAGES.get(target.suffix.lower(), "")
        sections.append(
            f"--- @{raw_path} (injected file content) ---\n"
            f"```{language}\n{content}\n```"
        )
        results.append(MentionResult(path=raw_path, injected=True))

    if not sections:
        return message, results

    return message + "\n\n" + "\n\n".join(sections), results
