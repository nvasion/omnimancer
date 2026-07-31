"""Assistant output renders as Markdown (headings, syntax-highlighted code)
while operation markers and marker-stripping behavior survive."""

from rich.console import Console

from omnimancer.cli.display import DisplayMixin


class _Harness(DisplayMixin):
    def __init__(self):
        self.console = Console(record=True, width=100, force_terminal=False)

    @property
    def text(self):
        return self.console.export_text(clear=False)


class TestAssistantMarkdown:
    def test_heading_renders(self):
        harness = _Harness()
        harness._show_assistant_message("# Big Title\n\nbody text", "m")
        assert "Big Title" in harness.text
        assert "body text" in harness.text
        # Markdown headings are decorated, not shown with the raw hash
        assert "# Big Title" not in harness.text

    def test_code_block_content_preserved(self):
        harness = _Harness()
        harness._show_assistant_message("```python\nprint('hi')\n```", "m")
        assert "print" in harness.text

    def test_operation_marker_survives_literally(self):
        harness = _Harness()
        harness._show_assistant_message("Writing now [FILE_WRITE:notes.txt] done", "m")
        assert "[FILE_WRITE:notes.txt]" in harness.text

    def test_html_comment_markers_stripped(self):
        harness = _Harness()
        harness._show_assistant_message("<!--read-only-->\nJust text", "m")
        assert "read-only" not in harness.text
        assert "Just text" in harness.text

    def test_weird_input_does_not_raise(self):
        harness = _Harness()
        harness._show_assistant_message(
            "```\nunclosed fence\n\x00odd bytes [[]]((", "m"
        )
        assert harness.text  # rendered something, didn't crash
