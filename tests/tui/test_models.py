from __future__ import annotations

from omnimancer.tui.fleet.models import (
    STALE_AFTER_S,
    DisplayState,
    derive_state,
    parse_job,
)


def test_parse_real_shape():
    """Test parsing a real job record."""
    data = {
        "id": "4553891c",
        "backend": "codex",
        "status": "running",
        "model": "gpt-5.6-sol",
        "reasoningEffort": "xhigh",
        "sandbox": "read-only",
        "cwd": "/home/x/repo",
        "createdAt": "2026-08-01T03:14:34.686Z",
        "startedAt": "2026-08-01T03:14:34.690Z",
        "turnState": "working",
        "processState": "running",
        "blockerKind": None,
        "turnsCompleted": 0,
        "lastAgentMessage": None,
        "usage": None,
        "provider": None,
        "completedAt": None,
    }
    job = parse_job(data)
    assert job.job_id == "4553891c"
    assert job.backend == "codex"
    assert job.status == "running"
    assert job.turn_state == "working"
    assert job.process_state == "running"
    assert job.blocker_kind is None
    assert job.model == "gpt-5.6-sol"
    assert job.sandbox == "read-only"
    assert job.turns_completed == 0


def test_parse_missing_fields_never_raises():
    """Test parsing a job with missing fields."""
    data = {"id": "aabbccdd"}
    job = parse_job(data)
    assert job.job_id == "aabbccdd"
    assert job.backend == ""
    assert job.status == ""
    assert job.turn_state == ""
    assert job.process_state == ""
    assert job.blocker_kind is None
    assert job.usage is None
    assert job.turns_completed == 0


def test_parse_garbage_returns_placeholder():
    """Test parsing garbage data returns placeholder."""
    job1 = parse_job("not a dict")
    assert job1.job_id == "unknown"
    assert job1.malformed is True

    job2 = parse_job(None)
    assert job2.job_id == "unknown"
    assert job2.malformed is True


def test_usage_mapped_long_keys():
    """Test that usage with long keys is mapped correctly."""
    job = parse_job(
        {"id": "aabbccdd", "usage": {"input_tokens": 100, "output_tokens": 50}}
    )
    assert job.usage == {"input_tokens": 100, "output_tokens": 50}


def test_usage_mapped_short_codex_keys():
    """Test that usage with short codex keys is mapped correctly."""
    job = parse_job({"id": "aabbccdd", "usage": {"input": 7, "output": 3}})
    assert job.usage["input_tokens"] == 7
    assert job.usage["output_tokens"] == 3


def test_usage_non_dict_is_none():
    """Test that non-dict usage is handled as None."""
    job = parse_job({"id": "aabbccdd", "usage": "garbage"})
    assert job.usage is None


def test_display_state_enum():
    """Test DisplayState enum has correct values."""
    assert DisplayState.PENDING.value == "pending"
    assert DisplayState.STARTING.value == "starting"
    assert DisplayState.WORKING.value == "working"
    assert DisplayState.WAITING.value == "waiting"
    assert DisplayState.BLOCKED.value == "blocked"
    assert DisplayState.STALE.value == "stale"
    assert DisplayState.COMPLETED.value == "completed"
    assert DisplayState.FAILED.value == "failed"
    assert DisplayState.CANCELLED.value == "cancelled"


def test_derive_state_table():
    """Test derive_state with various combinations."""
    # Test case a: status "pending", processState "created" -> PENDING
    job_a = parse_job({"id": "test", "status": "pending", "processState": "created"})
    assert derive_state(job_a) == DisplayState.PENDING

    # Test case b: processState "starting" -> STARTING
    job_b = parse_job({"id": "test", "processState": "starting"})
    assert derive_state(job_b) == DisplayState.STARTING

    # Test case c: status "cancelled" -> CANCELLED
    job_c = parse_job({"id": "test", "status": "cancelled"})
    assert derive_state(job_c) == DisplayState.CANCELLED

    # Test case d: processState "cancelled" -> CANCELLED
    job_d = parse_job({"id": "test", "processState": "cancelled"})
    assert derive_state(job_d) == DisplayState.CANCELLED

    # Test case e: processState "exited_failure" -> FAILED
    job_e = parse_job({"id": "test", "processState": "exited_failure"})
    assert derive_state(job_e) == DisplayState.FAILED

    # Test case f: status "failed" -> FAILED
    job_f = parse_job({"id": "test", "status": "failed"})
    assert derive_state(job_f) == DisplayState.FAILED

    # Test case g: processState "exited_success" -> COMPLETED
    job_g = parse_job({"id": "test", "processState": "exited_success"})
    assert derive_state(job_g) == DisplayState.COMPLETED

    # Test case h: status "completed" -> COMPLETED
    job_h = parse_job({"id": "test", "status": "completed"})
    assert derive_state(job_h) == DisplayState.COMPLETED

    # Test case i: blockerKind "context_limit", status "running" -> BLOCKED
    job_i = parse_job(
        {"id": "test", "blockerKind": "context_limit", "status": "running"}
    )
    assert derive_state(job_i) == DisplayState.BLOCKED

    # # Test case j: -> WAITING
    job_j = parse_job(
        {"id": "test", "backend": "codex", "status": "running", "turnState": "idle"}
    )
    assert derive_state(job_j, has_turn_complete=True) == DisplayState.WAITING

    # # Test case k: -> WORKING
    job_k = parse_job(
        {"id": "test", "backend": "codex", "status": "running", "turnState": "working"}
    )
    assert derive_state(job_k, has_turn_complete=False) == DisplayState.WORKING

    # # Test case l: -> STALE
    job_l = parse_job(
        {"id": "test", "backend": "omn", "status": "running", "turnState": "working"}
    )
    assert (
        derive_state(job_l, has_turn_complete=False, activity_age_s=300.0)
        == DisplayState.STALE
    )

    # # Test case m: -> WORKING
    job_m = parse_job(
        {"id": "test", "backend": "omn", "status": "running", "turnState": "working"}
    )
    assert (
        derive_state(job_m, has_turn_complete=False, activity_age_s=30.0)
        == DisplayState.WORKING
    )

    # # Test case n: -> WAITING
    job_n = parse_job(
        {"id": "test", "backend": "omn", "status": "running", "turnState": "idle"}
    )
    assert derive_state(job_n, has_turn_complete=True) == DisplayState.WAITING

    # # Test case o: -> WORKING
    job_o = parse_job({"id": "test", "backend": "omn", "status": "running"})
    assert (
        derive_state(job_o, has_turn_complete=False, activity_age_s=None)
        == DisplayState.WORKING
    )

    # # Test case p: -> CANCELLED (terminal beats blocked)
    job_p = parse_job({"id": "test", "status": "cancelled", "blockerKind": "auth"})
    assert derive_state(job_p) == DisplayState.CANCELLED


def test_stale_threshold_constant():
    """Test STALE_AFTER_S constant."""
    assert STALE_AFTER_S == 120.0
