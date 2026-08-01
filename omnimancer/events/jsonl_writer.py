"""
JSONL Writer for event logging.

This module provides a thread-safe JSONL (JSON Lines) writer that can
handle high-volume event logging with file size limits and queue management.
"""

from __future__ import annotations

import logging
import pathlib
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class JsonlWriter:
    """A thread-safe JSONL writer with queue management and file size limits."""

    def __init__(
        self,
        path: pathlib.Path,
        max_file_mb: int = 20,
        queue_size: int = 1000,
        cap_notice_line: Optional[str] = None,
        autostart: bool = True,
    ) -> None:
        """Initialize the JSONL writer.

        Args:
            path: The file path to write JSONL events to.
            max_file_mb: Maximum file size in MB before stopping writes.
            queue_size: Maximum number of events to queue.
            cap_notice_line: Line to write when file size limit is reached.
            autostart: Whether to start the writer thread automatically.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._max_file_mb = max_file_mb
        self._queue: queue.Queue[str] = queue.Queue(maxsize=queue_size)
        self._cap_notice_line = cap_notice_line
        # _stopped: no new enqueues, drain loop exits once the queue empties.
        # _capped: file size limit reached — drain and discard, never write.
        # Distinct flags: shutdown must flush in-flight lines, the cap must not.
        self._stopped = False
        self._capped = False
        self._dropped = 0
        self._write_errors = 0
        self._written_bytes = 0

        # Record existing file size
        try:
            self._written_bytes = path.stat().st_size
        except FileNotFoundError:
            self._written_bytes = 0

        self._thread: Optional[threading.Thread] = None
        if autostart:
            self._start_thread()
            # Small delay to ensure thread is ready
            time.sleep(0.001)

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

    def _drain_loop(self) -> None:
        """Drain the queue and write lines to the file."""
        while not self._stopped or not self._queue.empty():
            try:
                # Get first item with timeout
                line = self._queue.get(timeout=0.2)
                lines = [line]

                # Get any additional items that are ready immediately
                while True:
                    try:
                        lines.append(self._queue.get_nowait())
                    except queue.Empty:
                        break

                # Past the cap: keep draining so shutdown terminates, write
                # nothing. Shutdown (_stopped) still flushes this batch.
                if self._capped:
                    continue

                # Write all lines
                with open(self._path, "a", encoding="utf-8") as f:
                    for line in lines:
                        # Check if adding this line would exceed the limit
                        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
                        if (
                            self._written_bytes + line_bytes
                            > self._max_file_mb * 1024 * 1024
                        ):
                            if self._cap_notice_line:
                                f.write(self._cap_notice_line + "\n")
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

    def flush(self, timeout: float = 2.0) -> bool:
        """Flush all queued items to disk.

        Args:
            timeout: Maximum time to wait for flush completion.

        Returns:
            True if flushed successfully, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        return self._queue.empty()

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


def cleanup_old_files(directory: pathlib.Path, retention_days: int) -> int:
    """Clean up old .jsonl files in the directory.

    Args:
        directory: Directory to scan for old files.
        retention_days: Number of days to retain files.

    Returns:
        Number of files deleted.
    """
    deleted_count = 0
    cutoff_time = time.time() - (retention_days * 24 * 60 * 60)

    try:
        for item in directory.iterdir():
            if item.is_file() and item.suffix == ".jsonl":
                try:
                    if item.stat().st_mtime < cutoff_time:
                        item.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.debug(f"Error deleting file {item}: {e}")
    except FileNotFoundError:
        pass

    return deleted_count
