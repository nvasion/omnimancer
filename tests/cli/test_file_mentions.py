"""expand_file_mentions — @path tokens inject file content into the message.

Injection only happens for paths that actually exist (which naturally
guards against email addresses and casual @mentions), with a size cap,
binary detection, and directory listings.
"""

import pytest

from omnimancer.cli.file_mentions import expand_file_mentions


@pytest.fixture
def project(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')\n")
    (tmp_path / "notes.md").write_text("# Notes\n")
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("app = 1\n")
    return tmp_path


class TestInjection:
    def test_existing_file_content_is_appended(self, project):
        expanded, mentions = expand_file_mentions("look at @main.py", project)
        assert "look at @main.py" in expanded
        assert "print('hello')" in expanded
        assert "main.py" in expanded
        assert len(mentions) == 1
        assert mentions[0].path == "main.py"
        assert mentions[0].injected is True

    def test_language_fence_from_extension(self, project):
        expanded, _ = expand_file_mentions("@main.py", project)
        assert "```python" in expanded

    def test_nested_path(self, project):
        expanded, _ = expand_file_mentions("check @src/app.py", project)
        assert "app = 1" in expanded

    def test_multiple_mentions(self, project):
        expanded, mentions = expand_file_mentions("@main.py and @notes.md", project)
        assert "print('hello')" in expanded
        assert "# Notes" in expanded
        assert len(mentions) == 2

    def test_directory_injects_listing(self, project):
        expanded, mentions = expand_file_mentions("what is in @src", project)
        assert "app.py" in expanded
        assert mentions[0].injected is True


class TestGuards:
    def test_missing_path_is_untouched(self, project):
        message = "ping @nonexistent.py please"
        expanded, mentions = expand_file_mentions(message, project)
        assert expanded == message
        assert mentions == []

    def test_email_address_is_untouched(self, project):
        message = "mail brandonp@l337.co about it"
        expanded, mentions = expand_file_mentions(message, project)
        assert expanded == message
        assert mentions == []

    def test_oversize_file_skipped(self, project):
        big = project / "big.txt"
        big.write_text("x" * 100_000)
        expanded, mentions = expand_file_mentions("@big.txt", project, max_bytes=65_536)
        assert "x" * 1000 not in expanded
        assert mentions[0].injected is False
        assert "too large" in (mentions[0].reason or "")

    def test_binary_file_skipped(self, project):
        binary = project / "blob.bin"
        binary.write_bytes(b"\x00\x01\x02data")
        expanded, mentions = expand_file_mentions("@blob.bin", project)
        assert mentions[0].injected is False
        assert "binary" in (mentions[0].reason or "")

    def test_path_outside_project_skipped(self, project):
        expanded, mentions = expand_file_mentions("@/etc/passwd", project)
        assert "root:" not in expanded
        assert mentions == [] or mentions[0].injected is False

    def test_no_mentions_returns_message_unchanged(self, project):
        message = "just a normal question"
        expanded, mentions = expand_file_mentions(message, project)
        assert expanded == message
        assert mentions == []
