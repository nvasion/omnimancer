from __future__ import annotations

import json
import os
from pathlib import Path

from omnimancer.tui.fleet.sources import AgentsLogParser, EventsTailer, JobsSource


def test_scan_reads_jobs(tmp_path: Path) -> None:
    # Write test job files
    (tmp_path / "aabbccdd.json").write_text('{"id":"aabbccdd","status":"running"}')
    (tmp_path / "11223344.json").write_text('{"id":"11223344","status":"completed"}')
    (tmp_path / "index.json").write_text('{"x":1}')

    s = JobsSource(tmp_path)
    snap = s.scan()

    assert set(snap.jobs.keys()) == {"aabbccdd", "11223344"}
    assert snap.jobs["aabbccdd"]["status"] == "running"
    assert "index" not in snap.jobs


def test_scan_detects_turn_complete(tmp_path: Path) -> None:
    (tmp_path / "aabbccdd.json").write_text('{"id":"aabbccdd","status":"running"}')
    (tmp_path / "aabbccdd.turn-complete").touch()

    s = JobsSource(tmp_path)
    snap = s.scan()

    assert snap.turn_complete_ids == {"aabbccdd"}


def test_scan_is_incremental(tmp_path: Path) -> None:
    # First scan
    (tmp_path / "aabbccdd.json").write_text('{"id":"aabbccdd","status":"running"}')
    (tmp_path / "11223344.json").write_text('{"id":"11223344","status":"completed"}')

    s = JobsSource(tmp_path)
    s.reads_performed = 0  # Reset counter

    s.scan()
    assert s.reads_performed == 2

    # Second scan - modify one file
    (tmp_path / "aabbccdd.json").write_text('{"id":"aabbccdd","status":"completed"}')

    # Force a different mtime to ensure file is re-read
    aabbccdd_path = tmp_path / "aabbccdd.json"
    stat = aabbccdd_path.stat()
    os.utime(aabbccdd_path, (stat.st_mtime + 5, stat.st_mtime + 5))

    s.reads_performed = 0  # Reset counter
    snap2 = s.scan()

    # Should only have read 1 file (the modified one)
    assert s.reads_performed == 1
    assert snap2.jobs["aabbccdd"]["status"] == "completed"


def test_scan_skips_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "bad0bad0.json").write_text("{not json")

    s = JobsSource(tmp_path)
    snap = s.scan()

    assert "bad0bad0" not in snap.jobs


def test_scan_missing_dir(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nope"

    s = JobsSource(nonexistent)
    snap = s.scan()

    assert snap.jobs == {}
    assert snap.turn_complete_ids == set()


def test_poll_reads_new_lines(tmp_path: Path) -> None:
    (tmp_path / "s1.jsonl").write_text('{"a":1}\n{"b":2}\n')

    t = EventsTailer(tmp_path)
    result = t.poll()

    assert result == [{"a": 1}, {"b": 2}]

    # Second poll should return empty
    result2 = t.poll()
    assert result2 == []


def test_poll_incremental_append(tmp_path: Path) -> None:
    (tmp_path / "s1.jsonl").write_text('{"a":1}\n{"b":2}\n')

    t = EventsTailer(tmp_path)
    t.poll()  # First poll consumes the lines

    # Append a new line
    with open(tmp_path / "s1.jsonl", "a") as f:
        f.write('{"c":3}\n')

    result = t.poll()
    assert result == [{"c": 3}]


def test_poll_partial_line_buffered(tmp_path: Path) -> None:
    (tmp_path / "s1.jsonl").write_text('{"a":1}\n')

    t = EventsTailer(tmp_path)
    t.poll()  # Consume the first line

    # Append partial line
    with open(tmp_path / "s1.jsonl", "a") as f:
        f.write('{"a":')  # No newline yet

    result = t.poll()
    assert result == []

    # Now finish the line
    with open(tmp_path / "s1.jsonl", "a") as f:
        f.write("1}\n")

    result = t.poll()
    assert result == [{"a": 1}]


def test_poll_skips_bad_lines(tmp_path: Path) -> None:
    (tmp_path / "s1.jsonl").write_text('{"a":1}\nnotjson\n{"b":2}\n')

    t = EventsTailer(tmp_path)
    result = t.poll()

    assert result == [{"a": 1}, {"b": 2}]


def test_poll_truncation_resets(tmp_path: Path) -> None:
    (tmp_path / "s1.jsonl").write_text('{"a":1}\n{"b":2}\n')

    t = EventsTailer(tmp_path)
    t.poll()  # Consume the lines

    # Truncate file to zero bytes
    (tmp_path / "s1.jsonl").write_text("")

    # Write a new line
    (tmp_path / "s1.jsonl").write_text('{"c":3}\n')

    result = t.poll()
    assert result == [{"c": 3}]


def test_poll_discovers_new_files(tmp_path: Path) -> None:
    (tmp_path / "s1.jsonl").write_text('{"a":1}\n')

    t = EventsTailer(tmp_path)
    t.poll()  # Consume the first line

    # Create a new file
    (tmp_path / "s2.jsonl").write_text('{"b":2}\n')

    result = t.poll()
    assert result == [{"b": 2}]


def test_poll_missing_dir(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nope"

    t = EventsTailer(nonexistent)
    result = t.poll()

    assert result == []


def test_tailer_multibyte_split_across_polls(tmp_path: Path) -> None:
    # Create a line with a 4-byte UTF-8 character (𝄞)
    line = json.dumps({"v": "a𝄞b"}, ensure_ascii=False).encode("utf-8")

    # Cut two bytes INTO the 4-byte sequence — computed from its actual
    # position so the test genuinely exercises a mid-code-point split.
    glyph = "𝄞".encode("utf-8")
    assert len(glyph) == 4
    split_index = line.index(glyph) + 2
    part1 = line[:split_index]
    part2 = line[split_index:]

    # Write first part
    with open(tmp_path / "test.jsonl", "ab") as f:
        f.write(part1)

    t = EventsTailer(tmp_path)
    result1 = t.poll()  # Should return empty since line is incomplete

    assert result1 == []

    # Write second part and newline
    with open(tmp_path / "test.jsonl", "ab") as f:
        f.write(part2 + b"\n")

    result2 = t.poll()  # Should return the complete line

    assert len(result2) == 1
    assert result2[0]["v"] == "a𝄞b"  # Should contain the intact character


def test_parse_entries(tmp_path: Path) -> None:
    log_content = """## Session: 2026-08-01 03:10
### Spawned: aabbccdd - 03:11
### Complete: aabbccdd
### Died: 11223344 - crash
### Synthesis - 03:30
- VERDICT: PASS tail line\n"""

    (tmp_path / "agents.log").write_text(log_content)

    p = AgentsLogParser(tmp_path / "agents.log")
    entries = p.poll()

    assert len(entries) == 6

    kinds = [entry["kind"] for entry in entries]
    assert kinds == ["session", "spawned", "complete", "died", "synthesis", "verdict"]

    assert entries[1]["job_id"] == "aabbccdd"  # spawned
    assert entries[2]["job_id"] == "aabbccdd"  # complete
    assert entries[3]["job_id"] == "11223344"  # died
    assert entries[5]["verdict"] == "PASS"  # verdict


def test_parse_incremental(tmp_path: Path) -> None:
    log_content = """## Session: 2026-08-01 03:10
### Spawned: aabbccdd - 03:11\n"""

    (tmp_path / "agents.log").write_text(log_content)

    p = AgentsLogParser(tmp_path / "agents.log")
    entries1 = p.poll()

    # Append new entry
    with open(tmp_path / "agents.log", "a") as f:
        f.write("### Spawned: 55667788 - 03:40\n")

    entries2 = p.poll()

    assert len(entries1) == 2
    assert len(entries2) == 1
    assert entries2[0]["job_id"] == "55667788"


def test_parse_verdict_fail(tmp_path: Path) -> None:
    log_content = "VERDICT: FAIL\n"

    (tmp_path / "agents.log").write_text(log_content)

    p = AgentsLogParser(tmp_path / "agents.log")
    entries = p.poll()

    assert len(entries) == 1
    assert entries[0]["kind"] == "verdict"
    assert entries[0]["verdict"] == "FAIL"


def test_parse_missing_file(tmp_path: Path) -> None:
    nonexistent = tmp_path / "none.log"

    p = AgentsLogParser(nonexistent)
    entries = p.poll()

    assert entries == []


def test_logparser_multibyte_split_across_polls(tmp_path: Path) -> None:
    # Create a line with a 4-byte UTF-8 character (𝄞)
    line_bytes = b"### Spawned: aabbccdd - \xf0\x9d\x84\x9e\n"

    # Cut two bytes INTO the 4-byte sequence — computed, not hardcoded.
    split_index = line_bytes.index(b"\xf0\x9d\x84\x9e") + 2
    part1 = line_bytes[:split_index]
    part2 = line_bytes[split_index:]

    # Write first part
    with open(tmp_path / "agents.log", "ab") as f:
        f.write(part1)

    p = AgentsLogParser(tmp_path / "agents.log")
    result1 = p.poll()  # Should return empty since line is incomplete

    assert result1 == []

    # Write second part
    with open(tmp_path / "agents.log", "ab") as f:
        f.write(part2)

    result2 = p.poll()  # Should return the complete line

    assert len(result2) == 1
    assert result2[0]["kind"] == "spawned"
    assert result2[0]["job_id"] == "aabbccdd"  # Should contain the intact character
