from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DisplayState(str, Enum):
    """
    Enumeration of possible display states for a job.

    The display state represents how a job should be visually presented
    in the TUI based on its current condition and progress.
    """

    PENDING = "pending"
    STARTING = "starting"
    WORKING = "working"
    WAITING = "waiting"
    BLOCKED = "blocked"
    STALE = "stale"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


STALE_AFTER_S: float = 120.0


@dataclass
class JobRecord:
    """
    Data structure representing a job record from the codex-agent.

    This class maps the JSON structure of job records into a Python dataclass
    with appropriate field names and default values.
    """

    job_id: str = "unknown"
    backend: str = ""
    status: str = ""
    process_state: str = ""
    turn_state: str = ""
    blocker_kind: Optional[str] = None
    model: str = ""
    provider: Optional[str] = None
    sandbox: str = ""
    cwd: str = ""
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    turns_completed: int = 0
    last_agent_message: Optional[str] = None
    usage: Optional[dict] = None
    malformed: bool = False


def normalize_usage(raw: dict) -> dict:
    """
    Normalize usage dictionary from different formats to standard format.

    Converts short-form keys ("input", "output") to long-form keys
    ("input_tokens", "output_tokens") and ensures token counts are integers.

    Args:
        raw: Raw usage dictionary that may have short or long keys

    Returns:
        Normalized usage dictionary with standardized keys and integer values
    """
    if not isinstance(raw, dict):
        return None

    normalized = {}

    # Handle input tokens
    input_value = raw.get("input_tokens", raw.get("input", 0))
    try:
        normalized["input_tokens"] = int(input_value)
    except (ValueError, TypeError):
        normalized["input_tokens"] = 0

    # Handle output tokens
    output_value = raw.get("output_tokens", raw.get("output", 0))
    try:
        normalized["output_tokens"] = int(output_value)
    except (ValueError, TypeError):
        normalized["output_tokens"] = 0

    # Pass through any other keys
    for key, value in raw.items():
        if key not in ("input_tokens", "output_tokens", "input", "output"):
            normalized[key] = value

    return normalized


def parse_job(data: object) -> JobRecord:
    """
    Parse job data into a JobRecord.

    Args:
        data: Raw job data, typically a dictionary from JSON

    Returns:
        JobRecord with parsed data, or placeholder for malformed input
    """
    # Handle non-dict inputs
    if not isinstance(data, dict):
        return JobRecord(malformed=True)

    # Map camelCase keys to field names
    mapping = {
        "id": "job_id",
        "backend": "backend",
        "status": "status",
        "processState": "process_state",
        "turnState": "turn_state",
        "blockerKind": "blocker_kind",
        "model": "model",
        "provider": "provider",
        "sandbox": "sandbox",
        "cwd": "cwd",
        "createdAt": "created_at",
        "startedAt": "started_at",
        "completedAt": "completed_at",
        "turnsCompleted": "turns_completed",
        "lastAgentMessage": "last_agent_message",
        "reasoningEffort": "reasoning_effort",  # Ignored as per requirements
    }

    # Initialize job record with defaults
    job = JobRecord()

    # Populate fields from data
    for json_key, field_name in mapping.items():
        if json_key in data:
            value = data[json_key]
            # Handle None values specially
            if value is None and field_name == "blocker_kind":
                job.__dict__[field_name] = None
            elif value is None:
                # For other fields, we want to skip setting None values
                continue
            else:
                try:
                    # Try to convert to appropriate type
                    if field_name == "turns_completed":
                        job.__dict__[field_name] = int(value)
                    elif field_name == "malformed":
                        job.__dict__[field_name] = bool(value)
                    else:
                        job.__dict__[field_name] = str(value)
                except (ValueError, TypeError):
                    # Fall back to default for invalid values
                    pass

    # Handle usage mapping
    raw_usage = data.get("usage")
    if isinstance(raw_usage, dict):
        job.usage = normalize_usage(raw_usage)
    else:
        job.usage = None

    return job


def derive_state(
    job: JobRecord,
    has_turn_complete: bool = False,
    activity_age_s: Optional[float] = None,
) -> DisplayState:
    """
    Derive the display state for a job based on its properties.

    Implements the precedence rules for determining job state:
    1. job.malformed -> FAILED
    2. status "cancelled" or process_state "cancelled" -> CANCELLED
    3. process_state "exited_failure" or status "failed" -> FAILED
    4. process_state "exited_success" or status "completed" -> COMPLETED
    5. blocker_kind not None -> BLOCKED
    6. turn_state "idle" and has_turn_complete -> WAITING (any backend)
    7. backend "omn", activity_age_s beyond STALE_AFTER_S, and no
       turn-complete signal -> STALE
    8. process_state "starting" -> STARTING
    9. status "pending" or process_state "created" -> PENDING
    10. otherwise -> WORKING

    Args:
        job: The job record to analyze
        has_turn_complete: Whether the turn completion signal is present
        activity_age_s: Age of last activity in seconds (None if unknown)

    Returns:
        The appropriate DisplayState for the job
    """
    # Rule 1: malformed jobs are always failed
    if job.malformed:
        return DisplayState.FAILED

    # Rule 2: cancelled status or process state takes precedence
    if job.status == "cancelled" or job.process_state == "cancelled":
        return DisplayState.CANCELLED

    # Rule 3: failure states
    if job.process_state == "exited_failure" or job.status == "failed":
        return DisplayState.FAILED

    # Rule 4: completion states
    if job.process_state == "exited_success" or job.status == "completed":
        return DisplayState.COMPLETED

    # Rule 5: blockers take precedence over others
    if job.blocker_kind is not None:
        return DisplayState.BLOCKED

    # Rule 6: waiting state for codex jobs when idle and turn complete
    if job.backend == "codex" and job.turn_state == "idle" and has_turn_complete:
        return DisplayState.WAITING

    # Rule 7: waiting state for omn jobs when idle and turn complete
    if job.backend == "omn" and job.turn_state == "idle" and has_turn_complete:
        return DisplayState.WAITING

    # Rule 8: stale state for omn jobs with age exceeding threshold
    if (
        job.backend == "omn"
        and activity_age_s is not None
        and activity_age_s > STALE_AFTER_S
        and not has_turn_complete
    ):
        return DisplayState.STALE

    # Rule 9: starting state
    if job.process_state == "starting":
        return DisplayState.STARTING

    # Rule 10: pending state
    if job.status == "pending" or job.process_state == "created":
        return DisplayState.PENDING

    # Rule 11: default to working
    return DisplayState.WORKING
