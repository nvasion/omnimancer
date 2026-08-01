from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class JobsSnapshot:
    jobs: Dict[str, dict]
    turn_complete_ids: Set[str]
    json_mtimes: Dict[str, float]
    log_mtimes: Dict[str, float]

    @classmethod
    def empty(cls) -> JobsSnapshot:
        return cls(jobs={}, turn_complete_ids=set(), json_mtimes={}, log_mtimes={})


class JobsSource:
    """
    Source for job data from a directory of JSON files.

    Reads job state files (.json), turn completion signals (.turn-complete),
    and log files (.log) to build a snapshot of current jobs.
    """

    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.reads_performed: int = 0
        self._json_cache: Dict[str, Tuple[float, float, dict]] = {}

    def scan(self) -> JobsSnapshot:
        """
        Scan the jobs directory for job data.

        Returns:
            JobsSnapshot containing job data, turn complete IDs, and mtimes.
        """
        if not self.jobs_dir.exists():
            return JobsSnapshot.empty()

        jobs = {}
        turn_complete_ids = set()
        json_mtimes = {}
        log_mtimes = {}

        # Process JSON files
        for json_file in self.jobs_dir.glob("*.json"):
            if json_file.name == "index.json":
                continue

            try:
                stat = json_file.stat()
                mtime = stat.st_mtime
                size = stat.st_size

                # Check if we've cached this file and it hasn't changed
                if (
                    json_file.name in self._json_cache
                    and self._json_cache[json_file.name][0] == mtime
                    and self._json_cache[json_file.name][1] == size
                ):
                    # Use cached data
                    jobs[json_file.stem] = self._json_cache[json_file.name][2]
                else:
                    # Read and parse the file
                    self.reads_performed += 1
                    with open(json_file, "r") as f:
                        data = json.load(f)
                    jobs[json_file.stem] = data
                    # Cache the data
                    self._json_cache[json_file.name] = (mtime, size, data)

                json_mtimes[json_file.stem] = mtime

            except (json.JSONDecodeError, OSError) as e:
                logger.debug(f"Failed to read {json_file}: {e}")
                # Remove from cache if it was there
                if json_file.name in self._json_cache:
                    del self._json_cache[json_file.name]

        # Process turn-complete files
        for turn_file in self.jobs_dir.glob("*.turn-complete"):
            turn_complete_ids.add(turn_file.stem)

        # Process log files
        for log_file in self.jobs_dir.glob("*.log"):
            try:
                stat = log_file.stat()
                log_mtimes[log_file.stem] = stat.st_mtime
            except OSError as e:
                logger.debug(f"Failed to stat {log_file}: {e}")

        return JobsSnapshot(
            jobs=jobs,
            turn_complete_ids=turn_complete_ids,
            json_mtimes=json_mtimes,
            log_mtimes=log_mtimes,
        )


class EventsTailer:
    """
    Tailer for events from JSONL files.

    Maintains state for each file to process new lines incrementally.
    """

    def __init__(self, events_dir: Path):
        self.events_dir = events_dir
        self._file_states: Dict[str, Tuple[int, bytes]] = {}  # (offset, buffer)

    def poll(self) -> List[dict]:
        """
        Poll for new events from JSONL files.

        Returns:
            List of parsed JSON objects from new lines.
        """
        if not self.events_dir.exists():
            return []

        events = []

        # Get all JSONL files sorted for deterministic processing
        jsonl_files = sorted(self.events_dir.glob("*.jsonl"))

        for jsonl_file in jsonl_files:
            try:
                # Get current file state or initialize it
                if jsonl_file.name not in self._file_states:
                    self._file_states[jsonl_file.name] = (0, b"")

                offset, buffer = self._file_states[jsonl_file.name]

                # Check if file was truncated
                stat = jsonl_file.stat()
                if stat.st_size < offset:
                    # File was truncated, reset state
                    offset = 0
                    buffer = b""

                # Open file and read from offset
                with open(jsonl_file, "rb") as f:
                    f.seek(offset)
                    content = f.read()

                # Update offset
                new_offset = offset + len(content)
                self._file_states[jsonl_file.name] = (new_offset, buffer)

                # Combine buffer with new content
                full = buffer + content

                # Split on newlines to get complete lines
                parts = full.split(b"\n")

                # The last part might be incomplete (if it doesn't end with \n)
                # so we save it as our new buffer
                buffer = parts[-1]
                self._file_states[jsonl_file.name] = (new_offset, buffer)

                # Process complete lines (everything except the last part)
                for part in parts[:-1]:
                    if not part.strip():
                        continue
                    try:
                        # Decode each complete line individually to preserve UTF-8
                        line = part.decode("utf-8", errors="replace")
                        event = json.loads(line)
                        events.append(event)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.debug(
                            f"Bad JSON line in {jsonl_file}: "
                            f"{part.decode('utf-8', errors='replace')[:100]}... ({e})"
                        )

            except OSError as e:
                logger.debug(f"Failed to read {jsonl_file}: {e}")

        return events


class AgentsLogParser:
    """
    Parser for agents log file.

    Parses structured entries from a markdown-style log file.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._offset: int = 0
        self._buffer: bytes = b""

    def poll(self) -> List[dict]:
        """
        Parse new entries from the agents log.

        Returns:
            List of parsed log entries.
        """
        if not self.log_path.exists():
            return []

        try:
            stat = self.log_path.stat()
            if stat.st_size < self._offset:
                # File was truncated, reset state
                self._offset = 0
                self._buffer = b""

            # Binary read: the offset tracks bytes, and len(str) is not a
            # byte count for multibyte content.
            with open(self.log_path, "rb") as f:
                f.seek(self._offset)
                raw = f.read()

            self._offset += len(raw)

            # Combine buffer with new content
            full = self._buffer + raw

            # Split on newlines to get complete lines
            parts = full.split(b"\n")

            # The last part might be incomplete (if it doesn't end with \n)
            # so we save it as our new buffer
            self._buffer = parts[-1]

            entries = []
            for part in parts[:-1]:
                if not part.strip():
                    continue

                # Decode each complete line individually to preserve UTF-8
                line = part.decode("utf-8", errors="replace")
                entry = self._parse_line(line)
                if entry:
                    entries.append(entry)

            return entries

        except OSError as e:
            logger.debug(f"Failed to read {self.log_path}: {e}")
            return []

    def _parse_line(self, line: str) -> Optional[dict]:
        """
        Parse a single line into a log entry.

        Args:
            line: The line to parse

        Returns:
            Dictionary representing the parsed entry, or None if no match.
        """
        # Session line - exact match
        if line.startswith("## Session:"):
            return {"kind": "session"}

        # Spawned line - extract job ID
        if line.startswith("### Spawned:"):
            # Split by colon and extract the first 8-character hex string
            parts = line.split(":", 1)
            if len(parts) > 1:
                job_part = parts[1].strip()
                # Find the first 8-character hex string
                match = re.search(r"([0-9a-f]{8})", job_part)
                if match:
                    return {"kind": "spawned", "job_id": match.group(1)}

        # Complete line - extract job ID
        if line.startswith("### Complete:"):
            # Split by colon and extract the first 8-character hex string
            parts = line.split(":", 1)
            if len(parts) > 1:
                job_part = parts[1].strip()
                # Find the first 8-character hex string
                match = re.search(r"([0-9a-f]{8})", job_part)
                if match:
                    return {"kind": "complete", "job_id": match.group(1)}

        # Died line - extract job ID
        if line.startswith("### Died:"):
            # Split by colon and extract the first 8-character hex string
            parts = line.split(":", 1)
            if len(parts) > 1:
                job_part = parts[1].strip()
                # Find the first 8-character hex string
                match = re.search(r"([0-9a-f]{8})", job_part)
                if match:
                    return {"kind": "died", "job_id": match.group(1)}

        # Synthesis line
        if line.startswith("### Synthesis"):
            return {"kind": "synthesis"}

        # Verdict line - handle both "VERDICT: PASS" and "- VERDICT: PASS"
        match = re.search(r"VERDICT:?\s*(PASS|FAIL)", line, re.IGNORECASE)
        if match:
            return {"kind": "verdict", "verdict": match.group(1)}

        return None
