"""CLI entry for the omn fleet dashboard.

Reached via argv pre-dispatch in :func:`omnimancer.cli.interface.main` —
``omn fleet ...`` never touches the main click command, so the existing
``omn``/``omn -p`` flag surface (which codex-orchestrator parses) stays
byte-identical. Textual is imported lazily behind :func:`_require_textual`
so this module stays importable on installs without the ``tui`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

INSTALL_HINT = (
    "The 'textual' package is required for 'omn fleet'. "
    "Install with: pip install 'omnimancer-cli[tui]'"
)


def _require_textual() -> None:
    """Fail with an install hint when the tui extra is absent."""
    try:
        import textual  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(INSTALL_HINT) from exc


def default_jobs_dir() -> Path:
    """Return the codex-agent jobs directory."""
    return Path.home() / ".codex-agent" / "jobs"


def default_events_dir() -> Path:
    """Return the omnimancer fleet-events directory."""
    return Path.home() / ".omnimancer" / "events"


@click.command(name="fleet")
@click.option(
    "--jobs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="codex-agent jobs directory (default: ~/.codex-agent/jobs)",
)
@click.option(
    "--events-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="omnimancer events directory (default: ~/.omnimancer/events)",
)
@click.option(
    "--project",
    type=click.Path(path_type=Path),
    default=None,
    help="Project root whose agents.log feeds the comms panel (default: cwd)",
)
@click.option(
    "--refresh",
    type=float,
    default=1.0,
    show_default=True,
    help="Jobs rescan interval in seconds (event tails poll at half this)",
)
@click.option(
    "--once",
    is_flag=True,
    help="Render one snapshot and exit (scripting / smoke tests)",
)
def fleet_main(
    jobs_dir: Optional[Path],
    events_dir: Optional[Path],
    project: Optional[Path],
    refresh: float,
    once: bool,
) -> None:
    """Full-screen live dashboard of fleet agents, activity, and comms."""
    _require_textual()
    from omnimancer.tui.fleet.app import FleetApp

    app = FleetApp(
        jobs_dir=jobs_dir or default_jobs_dir(),
        events_dir=events_dir or default_events_dir(),
        project_dir=project or Path.cwd(),
        refresh=refresh,
        once=once,
    )
    app.run()
