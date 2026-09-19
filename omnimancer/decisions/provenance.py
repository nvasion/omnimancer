"""Hash the loaded package and report its own Git provenance, when available."""

import hashlib
import json
import subprocess
from pathlib import Path

from .routing import RoutingPolicy


def _package() -> Path:
    return Path(__file__).resolve().parents[1]


def implementation_digest(package: Path | None = None) -> str:
    """Commit to all package Python sources, including validators and CLI code.

    This hashes installed source bytes, not a function's introspected body or
    the caller's working directory. Third-party versions are separate metadata.
    """
    package = package or _package()
    sources = {
        path.relative_to(package)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    if not sources:
        raise ValueError("No installed Python source available for provenance")
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def policy_digest(policy: RoutingPolicy) -> str:
    value = {
        "policy": policy.model_dump(),
        "implementation_sha256": implementation_digest(),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def source_provenance(package: Path | None = None) -> dict:
    """Return commit/dirty state only when the loaded package is Git-tracked."""
    package = (package or _package()).resolve()
    unknown = {"source_commit": None, "source_dirty": None}

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(package), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        ).stdout.strip()

    try:
        # An installed wheel may be inside an unrelated project's .venv. Its
        # package initializer must itself be tracked before claiming that HEAD.
        git("ls-files", "--error-unmatch", "--", "__init__.py")
        commit = git("rev-parse", "HEAD")
        root = git("rev-parse", "--show-toplevel")
        status = git("status", "--porcelain", "--untracked-files=all", "--", root)
        return {"source_commit": commit, "source_dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        return unknown
