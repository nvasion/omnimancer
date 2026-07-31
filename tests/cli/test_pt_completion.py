"""OmnimancerCompleter — one completer for slash commands, their arguments
(including live provider/model names), and @-file mentions."""

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from prompt_toolkit.document import Document

from omnimancer.cli.completion import CompletionManager
from omnimancer.cli.pt_completion import OmnimancerCompleter


def _model(name):
    return SimpleNamespace(name=name)


@pytest.fixture
def engine():
    engine = MagicMock()
    gateway = MagicMock()
    gateway.get_available_models.return_value = [
        _model("qwen3-coder-30b"),
        _model("gpt-oss-120b"),
    ]
    local = MagicMock()
    local.get_available_models.return_value = [_model("qwen3-coder-30b")]
    engine.providers = {"gateway": gateway, "local": local}
    engine.config_manager.get_custom_models.return_value = [
        SimpleNamespace(name="qwen3-8b", provider="gateway"),
    ]
    return engine


@pytest.fixture
def completer(engine):
    manager = CompletionManager(engine=engine)
    return OmnimancerCompleter(manager)


def _complete(completer, text):
    document = Document(text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(document, None)]


class TestSlashCommands:
    def test_command_name_prefix(self, completer):
        results = _complete(completer, "/sw")
        assert "/switch" in results

    def test_switch_completes_providers(self, completer):
        results = _complete(completer, "/switch ")
        assert set(results) >= {"gateway", "local"}

    def test_switch_provider_prefix(self, completer):
        assert _complete(completer, "/switch ga") == ["gateway"]

    def test_switch_completes_models_for_provider(self, completer):
        results = _complete(completer, "/switch gateway ")
        assert "qwen3-coder-30b" in results
        assert "gpt-oss-120b" in results
        assert "qwen3-8b" in results  # custom model for that provider

    def test_static_subcommands_still_work(self, completer):
        results = _complete(completer, "/permissions ")
        assert "allow" in results
        assert "deny" in results

    def test_plain_chat_text_has_no_completions(self, completer):
        assert _complete(completer, "explain this repo") == []


class TestFileMentions:
    @pytest.fixture
    def project(self, tmp_path, monkeypatch):
        (tmp_path / "main.py").write_text("x")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "manager.py").write_text("x")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "junk.pyc").write_text("x")
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_at_token_completes_files(self, completer, project):
        results = _complete(completer, "@ma")
        assert any("main.py" in r for r in results)

    def test_fuzzy_subsequence_match(self, completer, project):
        results = _complete(completer, "@srcman")
        assert any("src/manager.py" in r for r in results)

    def test_walk_skips_cache_dirs(self, completer, project):
        results = _complete(completer, "@junk")
        assert results == []

    def test_at_mid_message(self, completer, project):
        results = _complete(completer, "please read @ma")
        assert any("main.py" in r for r in results)

    def test_gitignored_files_excluded_in_repo(self, completer, project):
        subprocess.run(
            ["git", "init", "-q"], cwd=project, check=True, capture_output=True
        )
        (project / ".gitignore").write_text("secret.txt\n")
        (project / "secret.txt").write_text("x")
        completer.invalidate_file_cache()

        results = _complete(completer, "@secret")
        assert results == []

    def test_result_cap(self, completer, tmp_path, monkeypatch):
        for i in range(40):
            (tmp_path / f"file{i:02d}.txt").write_text("x")
        monkeypatch.chdir(tmp_path)
        completer.invalidate_file_cache()

        results = _complete(completer, "@file")
        assert len(results) == 20


class TestCompletionManagerDynamic:
    def test_provider_names(self, engine):
        manager = CompletionManager(engine=engine)
        assert manager.provider_names("ga") == ["gateway"]

    def test_model_names_include_custom(self, engine):
        manager = CompletionManager(engine=engine)
        names = manager.model_names("gateway", "")
        assert "qwen3-coder-30b" in names
        assert "qwen3-8b" in names

    def test_no_engine_is_safe(self):
        manager = CompletionManager()
        assert manager.provider_names("") == []
        assert manager.model_names("gateway", "") == []
