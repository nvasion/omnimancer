import os
import pathlib
import time

from omnimancer.events.jsonl_writer import JsonlWriter, cleanup_old_files


def test_lines_written_in_order(tmp_path: pathlib.Path) -> None:
    """Test that lines are written in order to the file."""
    w = JsonlWriter(tmp_path / "s.jsonl")
    assert w.enqueue('{"a":1}') is True
    assert w.enqueue('{"b":2}') is True
    w.flush()

    # Check file content
    content = (tmp_path / "s.jsonl").read_text()
    assert content == '{"a":1}\n{"b":2}\n'

    w.shutdown()


def test_appends_to_existing_file(tmp_path: pathlib.Path) -> None:
    """Test that new lines are appended to existing files."""
    # Pre-create the file with initial content
    file_path = tmp_path / "s.jsonl"
    file_path.write_text('{"old":1}\n')

    w = JsonlWriter(file_path)
    assert w.enqueue('{"new":2}') is True
    w.flush()

    # Check file content - should have old line first, then new line
    content = file_path.read_text()
    assert content == '{"old":1}\n{"new":2}\n'

    w.shutdown()


def test_drop_on_full_never_blocks_never_raises(tmp_path: pathlib.Path) -> None:
    """Test that queue drops items when full and never blocks or raises."""
    w = JsonlWriter(tmp_path / "s.jsonl", queue_size=2, autostart=False)

    # First two should succeed
    assert w.enqueue('{"a":1}') is True
    assert w.enqueue('{"b":2}') is True

    # Third should fail (dropped)
    assert w.enqueue('{"c":3}') is False
    assert w.dropped == 1

    w.shutdown()


def test_size_cap_stops_writing(tmp_path: pathlib.Path) -> None:
    """Test that file size cap stops writing and writes notice."""
    cap_line = '{"event":"error","data":{"message":"event file size cap reached"}}'
    w = JsonlWriter(
        tmp_path / "s.jsonl",
        max_file_mb=1,
        cap_notice_line=cap_line,
    )

    # Write 12 lines of 100000 "x" characters each
    for i in range(12):
        line = '{"data":"' + "x" * 100000 + '"}'
        w.enqueue(line)

    w.flush()

    # Check that writing stopped after cap was reached
    # The last line should be the cap notice
    content = (tmp_path / "s.jsonl").read_text()
    lines = content.strip().split("\n")

    # Should have the cap notice as the last line
    assert (
        lines[-1]
        == '{"event":"error","data":{"message":"event file size cap reached"}}'
    )

    # Check file size is reasonable (under 1.5MB)
    file_size = (tmp_path / "s.jsonl").stat().st_size
    assert file_size < 1.5 * 1024 * 1024

    # Further enqueues should fail
    assert w.enqueue('{"should": "fail"}') is False

    w.shutdown()


def test_cleanup_old_files(tmp_path: pathlib.Path) -> None:
    """Test cleanup of old files based on modification time."""
    # Create test files
    a_file = tmp_path / "a.jsonl"
    b_file = tmp_path / "b.jsonl"
    c_file = tmp_path / "c.txt"

    a_file.write_text("content_a")
    b_file.write_text("content_b")
    c_file.write_text("content_c")

    # Set modification time for a.jsonl to 10 days ago
    ten_days_ago = time.time() - (10 * 24 * 60 * 60)
    os.utime(a_file, (ten_days_ago, ten_days_ago))

    # Cleanup files older than 7 days
    deleted_count = cleanup_old_files(tmp_path, retention_days=7)

    # Should have deleted only a.jsonl
    assert deleted_count == 1
    assert not a_file.exists()
    assert b_file.exists()
    assert c_file.exists()


def test_shutdown_flushes_and_stops(tmp_path: pathlib.Path) -> None:
    """Test that shutdown flushes pending items and stops the writer."""
    w = JsonlWriter(tmp_path / "s.jsonl")

    # Enqueue one line
    assert w.enqueue('{"test": "line"}') is True

    # Shutdown should flush and stop the writer
    w.shutdown()

    # Check that the line was written
    content = (tmp_path / "s.jsonl").read_text()
    assert content == '{"test": "line"}\n'

    # Check that the thread is not alive
    if w._thread:
        assert not w._thread.is_alive()

    # Subsequent enqueue should fail
    assert w.enqueue('{"should": "fail"}') is False
