"""
JSONL Writer for event logging.

This module provides a thread-safe JSONL (JSON Lines) writer that can
handle high-volume event logging with file size limits and queue management.
"""

from __future__ import annotations

import logging
import os
import pathlib
import queue
import re
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Event files are namespaced "omn-<uuid4>.jsonl"; retention and budget
# sweeps only ever touch names matching this. Owner of the naming scheme;
# omnimancer/events/emitter.py and the fleet dashboard import it from here.
SESSION_FILE_RE = (
    r"^omn-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jsonl$"
)


class JsonlWriter:
    """A thread-safe JSONL writer with queue management and file size limits."""

    def __init__(
        self,
        path: pathlib.Path,
        max_file_mb: int = 20,
        queue_size: int = 1000,
        cap_notice: Optional[Callable[[], str]] = None,
        on_start: Optional[Callable[[], None]] = None,
        autostart: bool = True,
    ) -> None:
        """Initialize the JSONL writer.

        No filesystem I/O happens here: directory creation, the initial
        size stat, and the optional on_start hook (e.g. retention cleanup)
        all run on the writer thread, so constructing a writer can never
        block the caller on a slow filesystem.

        Args:
            path: The file path to write JSONL events to.
            max_file_mb: Maximum file size in MB before stopping writes.
            queue_size: Maximum number of events to queue.
            cap_notice: Called at cap time to build the final line, so its
                timestamp reflects when the cap was actually reached.
            on_start: Ran once on the writer thread before draining.
            autostart: Whether to start the writer thread automatically.
        """
        self._path = path
        self._max_file_mb = max_file_mb
        self._queue: queue.Queue[str] = queue.Queue(maxsize=queue_size)
        self._cap_notice = cap_notice
        self._on_start = on_start
        # _stopped: no new enqueues, drain loop exits once the queue empties.
        # _capped: file size limit reached — drain and discard, never write.
        # Distinct flags: shutdown must flush in-flight lines, the cap must not.
        self._stopped = False
        self._capped = False
        self._dropped = 0
        self._write_errors = 0
        self._written_bytes = 0

        self._thread: Optional[threading.Thread] = None
        if autostart:
            self._start_thread()

    def _start_thread(self) -> None:
        """Start the writer thread."""
        self._thread = threading.Thread(
            target=self._drain_loop, name="omn-events-writer", daemon=True
        )
        self._thread.start()

    def enqueue(self, line: str) -> bool:
        """Enqueue a JSONL line for writing.

        Args:
            line: The JSONL line to write.

        Returns:
            True if successfully queued, False if failed to queue or writer is stopped.
        """
        if self._stopped or self._capped:
            return False

        try:
            self._queue.put_nowait(line)
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def _prepare(self) -> None:
        """Thread-side startup: directory first, then hook, then size stat.

        The directory exists before on_start runs so the hook can operate
        on it (e.g. permission repair + retention sweep).
        """
        try:
            # 0o700: event files carry command targets and output previews.
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        except Exception as e:
            logger.debug(f"JSONL writer mkdir failed: {e}")
        if self._on_start is not None:
            try:
                self._on_start()
            except Exception as e:
                logger.debug(f"JSONL writer on_start hook failed: {e}")
        try:
            self._written_bytes = self._path.stat().st_size
        except FileNotFoundError:
            self._written_bytes = 0
        except Exception as e:
            logger.debug(f"JSONL writer startup failed: {e}")

    def _open_append(self):  # type: ignore[no-untyped-def]
        """Open the event file for appending with owner-only permissions."""
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        return os.fdopen(fd, "a", encoding="utf-8")

    def _drain_loop(self) -> None:
        """Drain the queue and write lines to the file.

        Every dequeued item is acknowledged with task_done() exactly once
        (written, cap-discarded, or failed) — flush() relies on the queue's
        unfinished-task count, which has no empty-but-in-flight race.
        """
        self._prepare()
        while not self._stopped or not self._queue.empty():
            batch_size = 0
            try:
                # Get first item with timeout
                line = self._queue.get(timeout=0.2)
                batch_size = 1
                lines = [line]

                # Get any additional items that are ready immediately
                while True:
                    try:
                        lines.append(self._queue.get_nowait())
                        batch_size += 1
                    except queue.Empty:
                        break

                # Past the cap: keep draining so shutdown terminates, write
                # nothing. Shutdown (_stopped) still flushes this batch.
                if self._capped:
                    continue

                # Write all lines
                with self._open_append() as f:
                    for line in lines:
                        # Check if adding this line would exceed the limit
                        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
                        if (
                            self._written_bytes + line_bytes
                            > self._max_file_mb * 1024 * 1024
                        ):
                            if self._cap_notice is not None:
                                f.write(self._cap_notice() + "\n")
                            self._capped = True
                            break

                        f.write(line + "\n")
                        self._written_bytes += line_bytes

            except queue.Empty:
                # Timeout occurred, continue loop if not stopped
                if self._stopped:
                    break
                continue
            except Exception as e:
                self._write_errors += 1
                logger.debug(f"Error in JSONL writer: {e}")
                continue
            finally:
                for _ in range(batch_size):
                    self._queue.task_done()

    def flush(self, timeout: float = 2.0) -> bool:
        """Flush all queued items to disk.

        Args:
            timeout: Maximum time to wait for flush completion.

        Returns:
            True if flushed successfully, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def shutdown(self, timeout: float = 2.0) -> None:
        """Shutdown the writer gracefully.

        Args:
            timeout: Maximum time to wait for shutdown.
        """
        self._stopped = True
        # Give some time for the thread to process any remaining items
        if self._thread and self._thread.is_alive():
            # Wait a bit more to let the thread process the queue
            time.sleep(0.01)
            self._thread.join(timeout=timeout)

    @property
    def dropped(self) -> int:
        """Number of dropped events due to queue overflow."""
        return self._dropped

    @property
    def write_errors(self) -> int:
        """Number of write errors encountered."""
        return self._write_errors


def cleanup_old_files(
    directory: pathlib.Path, retention_days: int, name_re: Optional[str] = None
) -> int:
    """Clean up old .jsonl files in the directory.

    Args:
        directory: Directory to scan for old files.
        retention_days: Number of days to retain files.
        name_re: When set, only filenames matching this regex are eligible.
            The emitter passes a session-uuid pattern so a user-configured
            shared directory never loses unrelated .jsonl data.

    Returns:
        Number of files deleted.
    """
    deleted_count = 0
    cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
    pattern = re.compile(name_re) if name_re else None

    try:
        for item in directory.iterdir():
            if not (item.is_file() and item.suffix == ".jsonl"):
                continue
            if pattern is not None and not pattern.fullmatch(item.name):
                continue
            try:
                if item.stat().st_mtime < cutoff_time:
                    item.unlink()
                    deleted_count += 1
            except Exception as e:
                logger.debug(f"Error deleting file {item}: {e}")
    except FileNotFoundError:
        pass

    return deleted_count


def enforce_size_budget(
    directory: pathlib.Path,
    max_total_bytes: int,
    name_re: Optional[str] = None,
    min_age_s: float = 600.0,
) -> int:
    """Enforce a total size budget by deleting the oldest eligible .jsonl files.

    Scans *directory* for regular ``.jsonl`` files. When *name_re* is provided,
    only files whose name matches the compiled pattern are considered.  The
    function sums their sizes and, if the total exceeds *max_total_bytes*,
    deletes files one at a time starting from the oldest (by ``st_mtime``)
    until the remaining considered files fall within the budget.

    The *min_age_s* parameter protects recently-active session files: a file
    whose age (``time.time() - st_mtime``) is strictly less than *min_age_s*
    is never deleted, even when over budget.  This prevents the retention
    sweep from removing files that are still actively written to.

    Args:
        directory: Directory to scan for eligible .jsonl files.
        max_total_bytes: Maximum allowed total size of remaining considered
            files in bytes.
        name_re: Optional regex string. When set, only files whose name
            matches ``re.compile(name_re).fullmatch(...)`` are considered.
        min_age_s: Minimum file age in seconds. Files younger than this are
            protected from deletion. Defaults to 600.0 (10 minutes).

    Returns:
        Number of bytes freed by deleting files.
    """
    if not directory.exists():
        return 0

    # Destructive-input guard: a negative budget must behave like zero
    # (prune all age-eligible files), never like "delete everything".
    max_total_bytes = max(0, max_total_bytes)

    pattern = re.compile(name_re) if name_re else None
    now = time.time()

    # Gather eligible files with their sizes and mtimes
    candidates: list[tuple[pathlib.Path, int, float]] = []
    for item in directory.iterdir():
        if not (item.is_file() and item.suffix == ".jsonl"):
            continue
        if pattern is not None and not pattern.fullmatch(item.name):
            continue
        try:
            st = item.stat()
            candidates.append((item, st.st_size, st.st_mtime))
        except Exception:
            logger.debug(f"Error stat-ing file {item}")
            continue

    total_size = sum(sz for _, sz, _ in candidates)
    if total_size <= max_total_bytes:
        return 0

    # Sort by mtime ascending (oldest first)
    candidates.sort(key=lambda t: t[2])

    freed = 0
    deletable: list[pathlib.Path] = [
        path for path, _sz, mtime in candidates if now - mtime >= min_age_s
    ]

    for path in deletable:
        # Revalidate against the LIVE file, not the scan snapshot: a
        # concurrent append can rejuvenate a candidate (skip it), and the
        # accounting must use the size actually freed — a failed unlink
        # or a vanished file must not shrink the running total.
        try:
            st = path.stat()
        except OSError:
            continue
        if time.time() - st.st_mtime < min_age_s:
            continue
        try:
            path.unlink()
        except Exception:
            logger.debug(f"Error deleting file {path}")
            continue
        freed += st.st_size
        total_size -= st.st_size
        if total_size <= max_total_bytes:
            break

    return freed
