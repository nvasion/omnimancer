import os
import pathlib
import time

from omnimancer.events.jsonl_writer import (
    JsonlWriter,
    cleanup_old_files,
    enforce_size_budget,
)


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
        cap_notice=lambda: cap_line,
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


class TestEnforceSizeBudget:
    """Tests for the enforce_size_budget function."""

    def test_under_budget_deletes_nothing(self, tmp_path: pathlib.Path) -> None:
        """Under budget: returns 0, both files remain."""
        a = tmp_path / "omn-aaaaaaaa-bbbbccccdddd-eeee-ffff00001111.jsonl"
        b = tmp_path / "omn-aaaaaaaa-bbbbccccdddd-eeee-ffff00002222.jsonl"
        a.write_bytes(b"x" * 100)
        b.write_bytes(b"y" * 100)
        result = enforce_size_budget(tmp_path, max_total_bytes=1000)
        assert result == 0
        assert a.exists()
        assert b.exists()

    def test_prunes_oldest_first_until_under_budget(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Three files with ordered mtimes; delete oldest until under 700 bytes."""
        now = time.time()
        # a: 2 days ago, b: 1 day ago, c: 25 hours ago
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        c = tmp_path / "c.jsonl"
        a.write_bytes(b"x" * 400)
        b.write_bytes(b"y" * 400)
        c.write_bytes(b"z" * 400)
        os.utime(a, (now - 3 * 24 * 3600, now - 3 * 24 * 3600))
        os.utime(b, (now - 2 * 24 * 3600, now - 2 * 24 * 3600))
        os.utime(c, (now - 1 * 24 * 3600, now - 1 * 24 * 3600))
        result = enforce_size_budget(tmp_path, max_total_bytes=700)
        assert result == 800
        assert not a.exists()
        assert not b.exists()
        assert c.exists()

    def test_young_files_are_protected(self, tmp_path: pathlib.Path) -> None:
        """Over budget but young: returns 0, nothing deleted."""
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_bytes(b"x" * 600)
        b.write_bytes(b"y" * 600)
        # No utime call — files keep current mtime, younger than min_age_s
        result = enforce_size_budget(tmp_path, max_total_bytes=500)
        assert result == 0
        assert a.exists()
        assert b.exists()

    def test_name_re_scopes_deletion(self, tmp_path: pathlib.Path) -> None:
        """name_re matches only omn- pattern; keep-me.jsonl stays."""
        now = time.time()
        omn = tmp_path / "omn-00000000-1111-2222-3333-444444444444.jsonl"
        keep = tmp_path / "keep-me.jsonl"
        omn.write_bytes(b"x" * 600)
        keep.write_bytes(b"y" * 600)
        os.utime(omn, (now - 86400, now - 86400))
        os.utime(keep, (now - 86400, now - 86400))
        name_re = (
            r"^omn-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jsonl$"
        )
        result = enforce_size_budget(tmp_path, max_total_bytes=0, name_re=name_re)
        assert result == 600
        assert not omn.exists()
        assert keep.exists()

    def test_missing_directory_returns_zero(self, tmp_path: pathlib.Path) -> None:
        """Non-existent directory -> returns 0."""
        result = enforce_size_budget(tmp_path / "nope", 100)
        assert result == 0

    def test_mixed_young_and_old_over_budget(self, tmp_path: pathlib.Path) -> None:
        """Old deletable + young protected; old deleted, young remains."""
        now = time.time()
        old = tmp_path / "old.jsonl"
        young = tmp_path / "young.jsonl"
        old.write_bytes(b"x" * 500)
        young.write_bytes(b"y" * 500)
        os.utime(old, (now - 2 * 24 * 3600, now - 2 * 24 * 3600))
        # young keeps current mtime
        result = enforce_size_budget(tmp_path, max_total_bytes=600)
        assert result == 500
        assert not old.exists()
        assert young.exists()
