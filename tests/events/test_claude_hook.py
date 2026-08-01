"""Claude Code hook adapter tests (WU-D1).

Drives omn_fleet_hook.py exactly as Claude Code does: one subprocess per
event with the payload on stdin. The adapter must map every supported
event to a valid omn.event.v1 line, create owner-only files, honor the
kill switch, and exit 0 no matter what.
"""

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "omn_fleet_hook.py"
SESSION = "aaaabbbb-cccc-dddd-eeee-ffff00001111"


def _run(payload: object, events_dir: Path, **env_extra: str):
    env = dict(os.environ)
    env["OMNIMANCER_EVENTS_DIR"] = str(events_dir)
    env.update(env_extra)
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=raw,
        env=env,
        capture_output=True,
        timeout=30,
    )


def _lines(events_dir: Path) -> list:
    path = events_dir / f"omn-{SESSION}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _payload(event_name: str, **extra) -> dict:
    return {
        "session_id": SESSION,
        "transcript_path": "/tmp/t.jsonl",
        "cwd": "/tmp/proj",
        "hook_event_name": event_name,
        **extra,
    }


class TestEventMapping:
    def test_full_session_lifecycle(self, tmp_path):
        events = [
            _payload("SessionStart", model="claude-fable-5", source="startup"),
            _payload("UserPromptSubmit", prompt="fix the bug"),
            _payload(
                "PreToolUse",
                tool_name="Bash",
                tool_input={"command": "pytest -q"},
                tool_use_id="toolu_01",
            ),
            _payload(
                "PostToolUse",
                tool_name="Bash",
                tool_input={"command": "pytest -q"},
                tool_output="ok",
                tool_use_id="toolu_01",
            ),
            _payload("Stop", last_assistant_message="done, tests pass"),
            _payload("SessionEnd", reason="logout"),
        ]
        for event in events:
            result = _run(event, tmp_path)
            assert result.returncode == 0
            assert result.stdout == b"" and result.stderr == b""

        lines = _lines(tmp_path)
        assert [line["event"] for line in lines] == [
            "session_start",
            "turn_start",
            "tool_start",
            "tool_end",
            "turn_end",
            "session_end",
        ]
        for line in lines:
            assert line["v"] == 1
            assert line["mode"] == "claude"
            assert line["session_id"] == SESSION
            assert line["agent_id"] == "main"
            assert line["seq"] == -1
            assert line["ts"].endswith("+00:00")
            assert line["cwd"] == "/tmp/proj"
        start = lines[0]["data"]
        assert start["harness"] == "claude"
        assert start["model"] == "claude-fable-5"
        tool_start = lines[2]["data"]
        assert tool_start["tool"] == "Bash"
        assert tool_start["target"] == "pytest -q"
        assert tool_start["op_id"] == "toolu_01"
        assert tool_start["invocation"] == "claude"
        assert lines[3]["data"]["success"] is True
        assert lines[4]["data"]["last_message_preview"] == "done, tests pass"
        assert lines[5]["data"]["reason"] == "logout"

    def test_failure_event_carries_error(self, tmp_path):
        result = _run(
            _payload(
                "PostToolUseFailure",
                tool_name="Bash",
                error="x" * 900,
                is_interrupt=False,
                tool_use_id="toolu_02",
            ),
            tmp_path,
        )
        assert result.returncode == 0
        (line,) = _lines(tmp_path)
        assert line["event"] == "tool_end"
        assert line["data"]["success"] is False
        assert len(line["data"]["error"]) == 500

    def test_subagent_identity(self, tmp_path):
        _run(
            _payload(
                "Stop",
                last_assistant_message="sub done",
                agent_id="abc123def456",
                agent_type="Explore",
            ),
            tmp_path,
        )
        (line,) = _lines(tmp_path)
        assert line["agent_id"] == "subagent-Explore-abc123de"
        assert line["parent_id"] == "main"

    def test_interrupt_marks_cancelled(self, tmp_path):
        _run(
            _payload(
                "PostToolUseFailure",
                tool_name="Bash",
                error="interrupted by user",
                is_interrupt=True,
                tool_use_id="toolu_04",
            ),
            tmp_path,
        )
        (line,) = _lines(tmp_path)
        assert line["data"]["was_cancelled"] is True

    def test_legacy_field_names_still_accepted(self, tmp_path):
        # Older documented shapes: prompt_text / tool_output.
        _run(_payload("UserPromptSubmit", prompt_text="legacy prompt"), tmp_path)
        _run(
            _payload(
                "PostToolUseFailure",
                tool_name="Bash",
                tool_output="legacy error",
                tool_use_id="toolu_05",
            ),
            tmp_path,
        )
        lines = _lines(tmp_path)
        assert lines[0]["data"]["prompt_preview"] == "legacy prompt"
        assert lines[1]["data"]["error"] == "legacy error"

    def test_long_target_clipped(self, tmp_path):
        _run(
            _payload(
                "PreToolUse",
                tool_name="Bash",
                tool_input={"command": "y" * 900},
                tool_use_id="toolu_03",
            ),
            tmp_path,
        )
        (line,) = _lines(tmp_path)
        assert len(line["data"]["target"]) == 200


class TestSafety:
    def test_kill_switch(self, tmp_path):
        result = _run(
            _payload("Stop", last_assistant_message="x"),
            tmp_path,
            OMNIMANCER_EVENTS="0",
        )
        assert result.returncode == 0
        assert not tmp_path.exists() or not list(tmp_path.iterdir())

    def test_unknown_event_ignored(self, tmp_path):
        result = _run(_payload("PreCompact"), tmp_path)
        assert result.returncode == 0
        assert _lines(tmp_path) == []

    def test_malformed_stdin_exits_zero(self, tmp_path):
        result = _run(b"{not json", tmp_path)
        assert result.returncode == 0
        assert _lines(tmp_path) == []

    def test_evil_session_id_rejected(self, tmp_path):
        result = _run(
            _payload("Stop", last_assistant_message="x") | {"session_id": "../../etc"},
            tmp_path,
        )
        assert result.returncode == 0
        assert not tmp_path.exists() or not list(tmp_path.iterdir())

    def test_non_uuid_session_id_rejected(self, tmp_path):
        # Loose hex ids would mint filenames outside SESSION_FILE_RE and
        # escape the retention/budget sweeps entirely.
        for bad in ("deadbeef", "a" * 36, "aaaabbbb-cccc-dddd-eeee"):
            result = _run(
                _payload("Stop", last_assistant_message="x") | {"session_id": bad},
                tmp_path,
            )
            assert result.returncode == 0
        assert not tmp_path.exists() or not list(tmp_path.iterdir())

    def test_file_permissions(self, tmp_path):
        events_dir = tmp_path / "evdir"
        _run(_payload("Stop", last_assistant_message="x"), events_dir)
        path = events_dir / f"omn-{SESSION}.jsonl"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(events_dir.stat().st_mode) == 0o700

    def test_preexisting_loose_permissions_repaired(self, tmp_path):
        """mode= on makedirs/open only applies at creation: a pre-existing
        world-readable dir or session file must be repaired, not left
        exposing prompt/command previews."""
        events_dir = tmp_path / "evdir"
        events_dir.mkdir(mode=0o755)
        loose_file = events_dir / f"omn-{SESSION}.jsonl"
        loose_file.write_text("{}\n")
        loose_file.chmod(0o644)

        _run(_payload("Stop", last_assistant_message="x"), events_dir)

        assert stat.S_IMODE(events_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(loose_file.stat().st_mode) == 0o600

    def test_startup_speed(self, tmp_path):
        start = time.monotonic()
        _run(_payload("Stop", last_assistant_message="x"), tmp_path)
        elapsed = time.monotonic() - start
        # Bare-python startup + one append; generous CI margin.
        assert elapsed < 1.0
